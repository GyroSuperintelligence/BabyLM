# To run this adapter:
# uvicorn toys.communication.external_adapter:app --host 0.0.0.0 --port 8000 --reload
# toys/communication/external_adapter.py
"""
FastAPI adapter that makes GyroSI visible through two well-known REST
interfaces:

• OpenAI-compatible (models list + chat completions)
• HuggingFace text-generation endpoint

Design notes
------------
* No change to baby/*.py.  We only import and call the existing API.
* Three GyroSI agents back every conversation:
    1. a system agent  (id="gyro-system")
    2. a user agent    (id derived from X-User-ID header or anon-<ip>)
    3. an assistant    (shared id="gyro-assistant")
  System messages are ingested once (bootstrap); thereafter turns are
  brokered with baby.intelligence.orchestrate_turn().
* One AgentPool is shared across requests; it lives for the process
  and handles eviction automatically.
"""

from __future__ import annotations

import os
import uuid
import time
from typing import List

from fastapi import FastAPI, Header, Request
from pydantic import BaseModel, Field

from baby.intelligence import AgentPool, orchestrate_turn

# ---------------------------------------------------------------------
# Configuration helpers – override with env-vars if you like
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
DEFAULT_ONT_PATH = os.getenv(
    "GYROSI_ONTOLOGY_PATH",
    os.path.join(BASE_DIR, "memories/public/meta/ontology_map.json"),
)
DEFAULT_KNOWLEDGE_PATH = os.getenv(
    "GYROSI_PUBLIC_KNOWLEDGE",
    os.path.join(BASE_DIR, "memories/public/meta/knowledge.pkl.gz"),
)

os.makedirs(os.path.dirname(DEFAULT_KNOWLEDGE_PATH), exist_ok=True)

# ---------------------------------------------------------------------
# One shared AgentPool for the whole process
# ---------------------------------------------------------------------
agent_pool = AgentPool(DEFAULT_ONT_PATH, DEFAULT_KNOWLEDGE_PATH)

# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------
app = FastAPI(
    title="GyroSI Baby External Adapter",
    version="0.9.6",
    summary="OpenAI & HuggingFace compatible REST facade for GyroSI-Baby",
)

# ---------------------------------------------------------------------
# 1. OpenAI-compatible schema models
# ---------------------------------------------------------------------


class OAChatMessage(BaseModel):
    role: str
    content: str


class OAChatRequest(BaseModel):
    model: str
    messages: List[OAChatMessage]


class OAChatChoice(BaseModel):
    index: int = 0
    message: OAChatMessage
    finish_reason: str = "stop"


class OAChatResponse(BaseModel):
    id: str
    object: str = Field("chat.completion")
    created: int
    model: str
    choices: List[OAChatChoice]


# ---------------------------------------------------------------------
# 2. HuggingFace text-generation schema
# ---------------------------------------------------------------------


class HFGenerateRequest(BaseModel):
    inputs: str


class HFGenerateResponse(BaseModel):
    generated_text: str


# ---------------------------------------------------------------------
# Routes – OpenAI style
# ---------------------------------------------------------------------


@app.get("/v1/models")
def list_models() -> dict:
    """Return a single ‘model’ so OpenAI clients are satisfied."""
    return {
        "data": [
            {
                "id": "gyrosi-baby-0.9.6",
                "object": "model",
                "created": 0,
                "owned_by": "gyro",
            }
        ]
    }


@app.post("/v1/chat/completions", response_model=OAChatResponse)
async def chat_completions(
    payload: OAChatRequest,
    request: Request,
    x_user_id: str | None = Header(default=None, convert_underscores=False),
) -> OAChatResponse:
    """
    Minimal implementation of the OpenAI /v1/chat/completions endpoint.

    Strategy
    --------
    • system messages           → bootstrap: ingest into assistant once
    • accumulating assistant    → ingest into assistant (memory)
    • last user message         → orchestrate turn and return reply
    """
    # Derive stable user-id                                 ──────────
    remote = request.client.host if request.client else "anon"
    user_id = x_user_id or f"anon-{hash(remote)}"
    assistant_id = "gyro-assistant"
    system_id = "gyro-system"

    # Get or create the three agents                        ──────────
    system_agent = agent_pool.get_or_create_agent(system_id)
    assistant_agent = agent_pool.get_or_create_agent(assistant_id)

    # --------------------------------------------------------------
    # 1. Handle system messages (bootstrap once per assistant reset)
    # --------------------------------------------------------------
    if assistant_agent.engine.cycle_count == 0:
        system_msgs = [m.content for m in payload.messages if m.role == "system"]
        if system_msgs:
            system_text = "\n".join(system_msgs)
            # Let system agent think & emit stimulus
            stimulus = system_agent.respond(system_text.encode("utf-8"))
            # Feed that to assistant so its first cycles = system prompt
            assistant_agent.ingest(stimulus)

    # --------------------------------------------------------------
    # 2. Feed prior assistant utterances back into assistant memory
    # --------------------------------------------------------------
    assistant_memories = [m.content for m in payload.messages if m.role == "assistant"]
    if assistant_memories:
        assistant_agent.ingest("\n".join(assistant_memories).encode("utf-8"))

    # --------------------------------------------------------------
    # 3. Find last user message and run a turn
    # --------------------------------------------------------------
    last_user = next((m for m in reversed(payload.messages) if m.role == "user"), None)
    user_text = last_user.content if last_user else ""
    reply = orchestrate_turn(agent_pool, user_id, assistant_id, user_text)

    # Build OpenAI-style response                             ───────
    resp = OAChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=int(time.time()),
        model="gyrosi-baby-0.9.6",
        choices=[
            OAChatChoice(
                index=0,
                message=OAChatMessage(role="assistant", content=reply),
            )
        ],
    )
    return resp


# ---------------------------------------------------------------------
# Routes – HuggingFace style
# ---------------------------------------------------------------------


@app.post("/generate", response_model=HFGenerateResponse)
async def hf_generate(payload: HFGenerateRequest, request: Request) -> HFGenerateResponse:
    user_id = f"hf-{hash(request.client.host) if request.client else 'anon'}"
    assistant_id = "gyro-assistant"
    reply = orchestrate_turn(agent_pool, user_id, assistant_id, payload.inputs)
    return HFGenerateResponse(generated_text=reply)
