# baby/responses_api/inference/gyro.py

from typing import Callable, Dict, Any
import collections, json, threading
from pathlib import Path
from openai_harmony import StreamableParser, Role
from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import (
    START, CHANNEL, MESSAGE, END, RETURN, CALL,
    assistant_role_id, token_id
)

_engine_lock = threading.RLock()

def setup_model(encoding, config_path: str) -> Callable[[list[int], float], int]:
    """
    Returns infer_next_token(tokens, temperature=..., request_id=..., new_request=...)
    Engine and sessions are closed over; no module-level globals required.
    """
    cfg_path = Path(config_path).resolve()
    cfg = json.loads(cfg_path.read_text())
    base = cfg_path.parent

    def _abs(p: str) -> str:
        pth = Path(p)
        cand = (base / pth).resolve()
        if cand.exists() or cand.parent.exists():
            return str(cand)
        alt = (base.parent / pth).resolve()
        return str(alt)

    atlas_paths = {k: _abs(v) for k, v in cfg["atlas"].items()}
    store_paths = {k: _abs(v) for k, v in cfg["stores"].items()}
    runtime     = cfg.get("runtime", {})
    version     = cfg.get("version", {})

    with _engine_lock:
        engine = GyroEngine(
            atlas_paths=atlas_paths,
            store_paths=store_paths,
            runtime=runtime,
            version_info=version,
            vocab_size=201_088  # o200k_harmony upper bound; safe cap
        )

    # Per-request session state
    sessions: Dict[str, Dict[str, Any]] = {}
    sessions_lock = threading.RLock()

    # Resolve dynamic control tokens once
    ASSISTANT_TOK = assistant_role_id(encoding)
    # Let Harmony assume the default "final" channel; don't emit it explicitly.
    ASSISTANT_HEADER = [START, ASSISTANT_TOK, MESSAGE]
    CONTROL = {START, CHANNEL, MESSAGE, END, RETURN, CALL}

    def _bad_surface_form(encoding, out_text: str, last_tokens: collections.deque[int], candidate: int) -> bool:
        """Minimal surface-form guard to stop gibberish without blocking flow."""
        s = encoding.decode([candidate])
        
        # No leading whitespace at the very start
        if not out_text and s.strip() == "":
            return True
            
        # Avoid triple-repeat of the same token
        if len(last_tokens) >= 2:
            if last_tokens[-1] == candidate and last_tokens[-2] == candidate:
                return True
                
        # Cap consecutive identical character run at 4
        if len(s) == 1 and len(out_text) > 0:
            char = s[0]
            run_length = 0
            for i in range(len(out_text) - 1, -1, -1):
                if out_text[i] == char:
                    run_length += 1
                else:
                    break
            if run_length >= 4:
                return True
                
        # Prefer a space after .?! but don't hard-block other valid tokens.
        if out_text and out_text[-1] in ".?!" and s not in {" ", "\n"}:
            return False
                
        # Prevent malformed punctuation sequences
        if s in ".,!?;:" and len(out_text) > 0:
            if out_text[-1] in ".,!?;:":
                return True
                
        return False

    def _apply_new_tokens(sess: Dict[str, Any], new_tokens: list[int]) -> None:
        """
        Feed only the delta tokens to the streaming parser and update engine state.
        Egress for user content; ingress for assistant content.
        """
        parser = sess["parser"]
        for tok in new_tokens:
            parser.process(tok)

            # Feed boundary-aware tokens into engine
            if parser.current_channel == "final":
                if parser.current_role == "user":
                    # Fold on PRE-state (as per your spec): evolve_on_user
                    sess["state"] = engine.evolve_on_user(sess["state"], tok)
                elif parser.current_role == "assistant":
                    # No fold unless self-reinforcement is on
                    sess["state"] = engine.evolve_on_assistant(sess["state"], tok)

    def infer_next_token(tokens: list[int], temperature: float = 0.0, **kwargs) -> int:
        request_id: str = kwargs.get("request_id", "__singleton__")
        new_request: bool = kwargs.get("new_request", False)

        with sessions_lock:
            sess = sessions.get(request_id)
            if sess is None or new_request:
                # Fresh session: start state and a fresh parser
                start_state = engine.start_state()
                sess = {
                    "state": start_state,
                    "parser": StreamableParser(encoding, role=Role.ASSISTANT),
                    "fed_len": 0,
                    "header_i": 0,
                    "out_last_tokens": collections.deque(maxlen=8),
                    "sentence_end": True,  # allow initial capitalisation/space logic
                    "out_text": ""
                }
                sessions[request_id] = sess

            # 1) Emit assistant header if not fully sent
            if sess["header_i"] < len(ASSISTANT_HEADER):
                nxt = ASSISTANT_HEADER[sess["header_i"]]
                sess["header_i"] += 1
                return nxt

            # 2) Ingest only the new tokens since last call
            delta = tokens[sess["fed_len"]:]
            if delta:
                _apply_new_tokens(sess, delta)
                sess["fed_len"] = len(tokens)

            # 3) Ask the engine for the next admissible token
            state = sess["state"]
            next_token = engine.next_token_deterministic(state)

            # 4) Hard exclude control tokens as content
            if next_token is None or next_token in CONTROL:
                # fall back to recovery directly
                rec = engine.recover_candidates(state)
                next_token = rec[0] if rec else token_id(encoding, ".")  # deterministic dot as last resort

            # 5) Surface-form guard (lightweight, not overfitted)
            if _bad_surface_form(encoding, sess["out_text"], sess["out_last_tokens"], next_token):
                # Try a space or full stop if admissible
                for fallback in (token_id(encoding, " "), token_id(encoding, ".")):
                    if engine.is_admissible(state, fallback):
                        next_token = fallback
                        break

            # 6) Update session with emitted token and evolve state (ingress, no fold by default)
            sess["out_last_tokens"].append(next_token)
            sess["state"] = engine.evolve_on_assistant(sess["state"], next_token)
            
            # Update output text tracking
            text_piece = encoding.decode([next_token])
            sess["out_text"] += text_piece

            return next_token

    return infer_next_token


# This removes the global engine hazard and guarantees user text actually reaches passive memory.
