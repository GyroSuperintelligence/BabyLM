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
import time
import uuid
import signal
from typing import List, Any, Iterator
from pathlib import Path
import atexit
import json

from fastapi import FastAPI, Header, Request
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from baby.intelligence import AgentPool, orchestrate_turn

# Import the tokenizer bridge
from baby.information import encode_text, decode_text, bytes_to_token_ids

# ---------------------------------------------------------------------
# Load preferences from canonical JSON
# ---------------------------------------------------------------------
PREFERENCES_PATH = os.getenv("GYROSI_PREFERENCES_PATH", "memories/memory_preferences.json")
with open(PREFERENCES_PATH) as f:
    PREFERENCES = json.load(f)

BASE_PATH = Path(PREFERENCES.get("base_path", Path(PREFERENCES_PATH).parent)).resolve()


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# One shared AgentPool for the whole process
# ---------------------------------------------------------------------
# Resolve knowledge path relative to project root, not memories directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
base_knowledge_path = str(PROJECT_ROOT / PREFERENCES["public_knowledge"]["path"])


agent_pool = AgentPool(
    ontology_path=str(PROJECT_ROOT / PREFERENCES["ontology"]["ontology_map_path"]),
    base_knowledge_path=base_knowledge_path,
    preferences=PREFERENCES,
    allowed_ids={"user", "system", "assistant"},
    allow_auto_create=False,
    private_agents_base_path=str(BASE_PATH / PREFERENCES["private_knowledge"]["base_path"]),
    base_path=BASE_PATH,
)

# Signal handling for graceful shutdown
# ---------------------------------------------------------------------
def signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM for graceful shutdown."""
    print(f"Received signal {signum}, shutting down gracefully...")
    try:
        agent_pool.close_all()
    except:
        pass
    exit(0)


# Register signal handler for SIGTERM (only if not in a critical section)
try:
    signal.signal(signal.SIGTERM, signal_handler)
except:
    pass  # Ignore if already registered
agent_pool.ensure_triad()

# Sanity probe to confirm sidecars are loaded
try:
    # unwrap ReadOnlyView → CanonicalView → OrbitStore
    cv = agent_pool._public_store.base_store     # CanonicalView
    store = cv.base_store                        # OrbitStore
    sp = store.store_path
    print("[public-store] path:", sp)
    print("[public-store] bloom exists:", os.path.exists(sp + ".bloom"))
    print("[public-store] idx exists:",   os.path.exists(sp + ".idx"))
    print("[public-store] index entries:", len(store.index))
except Exception as e:
    print("sanity probe failed:", e)

atexit.register(agent_pool.close_all)

# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------
app = FastAPI(
    title="GyroSI Baby External Adapter",
    version="0.9.6.7",
    summary="OpenAI & HuggingFace compatible REST facade for GyroSI-Baby (Token-Aware)",
)

@app.on_event("startup")
async def warm():
    """Pre-warm the system on startup to avoid first-turn penalty."""
    agent_pool.ensure_triad()
    try:
        cv = agent_pool._public_store.base_store   # CanonicalView
        store = cv.base_store                      # OrbitStore
        sp = store.store_path
        print(f"[warmup] public store: {sp}")
        print(f"[warmup] bloom: {os.path.exists(sp + '.bloom')}, idx: {os.path.exists(sp + '.idx')}")
        print(f"[warmup] indexed keys: {len(store.index)}")

        # Touch epistemology to page-in a tiny slice (avoids first-turn page faults)
        a = agent_pool.get("assistant").engine
        _ = int(a.epistemology[0,0])  # tiny read is enough to map a page
        print(f"[warmup] epistemology touched: {a.epistemology.shape}")
    except Exception as e:
        print("[warmup] failed:", e)

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
def list_models() -> dict[str, Any]:
    """Return a single ‘model’ so OpenAI clients are satisfied."""
    return {
        "data": [
            {
                "id": "gyrosi-baby",
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
) -> OAChatResponse | StreamingResponse:
    """
    Minimal implementation of the OpenAI /v1/chat/completions endpoint.
    Now supports HTTP keep-alive and streaming if client sets stream=true.
    """
    # Derive stable user-id                                 ──────────
    remote = request.client.host if request.client else "anon"
    user_id = x_user_id or f"anon-{hash(remote)}"
    # map all external users → internal "user"
    user_id = "user"
    assistant_id = "assistant"
    system_id = "system"

    # Get or create the three agents                        ──────────
    system_agent = agent_pool.get(system_id)
    assistant_agent = agent_pool.get(assistant_id)

    # --------------------------------------------------------------
    # 1. Handle system messages (bootstrap once per assistant reset)
    # --------------------------------------------------------------
    if assistant_agent.engine.cycle_count == 0:
        system_msgs = [m.content for m in payload.messages if m.role == "system"]
        if system_msgs:
            system_text = "\n".join(system_msgs)
            # Encode with tokenizer
            stimulus = system_agent.respond(encode_text(system_text, name=PREFERENCES["tokenizer"]["name"]))
            # Feed that to assistant so its first cycles = system prompt
            assistant_agent.ingest(stimulus)

    # --------------------------------------------------------------
    # 2. Feed prior assistant utterances back into assistant memory
    # --------------------------------------------------------------
    assistant_memories = [m.content for m in payload.messages if m.role == "assistant"]
    if assistant_memories:
        # Encode with tokenizer
        memory_bytes = encode_text("\n".join(assistant_memories), name=PREFERENCES["tokenizer"]["name"])
        assistant_agent.ingest(memory_bytes)

    # --------------------------------------------------------------
    # 3. Find last user message and run a turn
    # --------------------------------------------------------------
    last_user = next((m for m in reversed(payload.messages) if m.role == "user"), None)
    user_text = last_user.content if last_user else ""
    # Call orchestrate_turn with the tokenizer name
    reply = orchestrate_turn(
        agent_pool, user_id, assistant_id, user_text, tokenizer_name=PREFERENCES["tokenizer"]["name"]
    )

    # Streaming support: if client sets stream=true, yield tokens as SSE
    if request.query_params.get("stream", "false").lower() == "true":

        def token_stream() -> Iterator[str]:
            # Encode the reply to bytes, then decode token by token
            reply_bytes = encode_text(reply, name=PREFERENCES["tokenizer"]["name"])
            # Get individual token IDs using public API
            ids = bytes_to_token_ids(reply_bytes)

            # Load tokenizer once, reuse
            from baby.information import _load_tokenizer
            tokenizer = _load_tokenizer(PREFERENCES["tokenizer"]["name"])

            for i, token_id in enumerate(ids):
                # FIXED: Use direct tokenizer decode to avoid double-mask issue
                # Instead of: token_bytes = gyrotok.id_to_bytes(token_id)
                #           token_text = gyrotok.decode(token_bytes, name=PREFERENCES["tokenizer"]["name"])
                # Use direct tokenizer method to get token text
                try:
                    # Get token text directly from tokenizer
                    token_text = tokenizer.decode([token_id])
                except (AttributeError, Exception):
                    # Fallback: use direct token ID to text conversion
                    token_text = f"[TOKEN_{token_id}]"

                # OpenAI-compatible SSE chunk
                chunk = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "gyrosi-baby",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token_text},
                            "finish_reason": None if i < len(ids) - 1 else "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # Build OpenAI-style response                             ───────
    resp = OAChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        object="chat.completion",
        created=int(time.time()),
        model="gyrosi-baby",
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
    # map all external users → internal "user"
    user_id = "user"
    assistant_id = "assistant"
    # system_id = "system"  # (stale, do not reinstate)
    # Call orchestrate_turn with the tokenizer name
    reply = orchestrate_turn(
        agent_pool, user_id, assistant_id, payload.inputs, tokenizer_name=PREFERENCES["tokenizer"]["name"]
    )
    # Ensure output is in the tokenizer's alphabet (lowercase for bert-base-uncased)
    reply = reply.lower()
    return HFGenerateResponse(generated_text=reply)
