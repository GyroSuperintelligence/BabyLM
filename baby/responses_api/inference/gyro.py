# baby/responses_api/inference/gyro.py
# Streaming wrapper wired to the five-map GyroEngine.
# No scores / greedy paths; learning only on user content; deterministic BU-In.

from typing import Callable, Dict, Any, Optional
import json, threading, collections
from pathlib import Path
from openai_harmony import StreamableParser, Role
from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import MESSAGE, ROLE_USER, ROLE_ASSISTANT, ALL_CONTROL_TOKENS

_engine_lock = threading.RLock()


def setup_model(encoding, config_path: str) -> Callable[..., Optional[int]]:
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
    store_paths = {k: _abs(v) for k, v in cfg.get("stores", {}).items()}
    runtime = cfg.get("runtime", {})
    version = cfg.get("version", {})

    with _engine_lock:
        engine = GyroEngine(
            atlas_paths=atlas_paths,
            store_paths=store_paths,
            runtime=runtime,
            version_info=version,
            vocab_size=201_088,  # o200k_harmony upper bound; safe cap
        )

    # Per-request session state
    sessions: Dict[str, Dict[str, Any]] = {}
    sessions_lock = threading.RLock()

    # Harmony tokenization — pure pass-through; no surface forcing

    def _is_user_role(current_role) -> bool:
        try:
            if current_role == Role.USER:
                return True
        except Exception:
            pass
        if current_role == "user" or current_role == ROLE_USER:
            return True
        val = getattr(current_role, "value", None)
        return val == "user"

    def _is_assistant_role(current_role) -> bool:
        try:
            if current_role == Role.ASSISTANT:
                return True
        except Exception:
            pass
        if current_role == "assistant" or current_role == ROLE_ASSISTANT:
            return True
        val = getattr(current_role, "value", None)
        return val == "assistant"

    def _apply_new_tokens(sess: Dict[str, Any], new_tokens: list[int]) -> None:
        """
        Feed only the delta tokens; learn only from user content.
        """
        parser = sess["parser"]
        for tok in new_tokens:
            parser.process(tok)

            if _is_user_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                # Egress: fold & step on user tokens
                new_state = engine.evolve_on_user(sess["state"], tok)
                sess["state"] = new_state
                sess["user_token_count"] = sess.get("user_token_count", 0) + 1

                # Capture anchor after K user tokens (optional deterministic hook)
                target_k = sess.get("anchor_target_k", int(engine.runtime.get("anchor_prefix_tokens", 12)))
                if sess["user_token_count"] == target_k:
                    sess["user_anchor_state"] = new_state
                    sess["anchor_last_seen_k"] = sess["user_token_count"]
                elif sess["user_token_count"] > target_k:
                    # Update anchor to latest user token state if more tokens arrived
                    sess["user_anchor_state"] = new_state
                    sess["anchor_last_seen_k"] = sess["user_token_count"]
            else:
                # Ingress transit for assistant content (no learning)
                sess["state"] = engine.evolve_on_assistant(sess["state"], tok)

    def infer_next_token(tokens: list[int], temperature: float = 0.0, **kwargs) -> Optional[int]:
        request_id: str = kwargs.get("request_id", "__singleton__")
        new_request: bool = kwargs.get("new_request", False)
        ingest_only: bool = kwargs.get("ingest_only", False)

        with sessions_lock:
            sess = sessions.get(request_id)
            if new_request or sess is None:
                sess = {
                    "parser": StreamableParser(encoding, role=Role.SYSTEM),
                    "fed_len": 0,
                    "state": engine.start_state(),
                    "bootstrap_step": 0,
                    "user_token_count": 0,
                    "user_anchor_state": None,
                    "anchor_target_k": int(engine.runtime.get("anchor_prefix_tokens", 12)),
                    "anchor_last_seen_k": 0,
                    "anchor_applied": False,
                    # PPE state scoped per session
                    "omega": {},
                    "bucket_key": {},
                    "bucket_pos": {},
                }
                sessions[request_id] = sess

                if tokens:
                    parser = StreamableParser(encoding, role=Role.SYSTEM)
                    for tok in tokens:
                        parser.process(tok)
                        if _is_user_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                            new_state = engine.learn_on_user(sess["state"], tok)
                            sess["state"] = new_state
                            sess["user_token_count"] = sess.get("user_token_count", 0) + 1
                            target_k = sess.get("anchor_target_k", int(engine.runtime.get("anchor_prefix_tokens", 12)))
                            if sess["user_token_count"] == target_k:
                                sess["user_anchor_state"] = new_state
                                sess["anchor_last_seen_k"] = sess["user_token_count"]
                            elif sess["user_token_count"] > target_k:
                                # Update anchor to latest user token state if more tokens arrived
                                sess["user_anchor_state"] = new_state
                                sess["anchor_last_seen_k"] = sess["user_token_count"]
                        elif _is_assistant_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                            sess["state"] = engine.transit_on_assistant(sess["state"], tok)
                        else:
                            pass
                    sess["parser"] = parser
                    sess["fed_len"] = len(tokens)
                else:
                    sess["parser"] = StreamableParser(encoding, role=Role.SYSTEM)
            else:
                delta = tokens[sess["fed_len"] :]
                if delta:
                    _apply_new_tokens(sess, delta)
                    sess["fed_len"] = len(tokens)

        if ingest_only:
            return None

        # Apply one-time anchor (if captured) just before generation
        anchor_state = sess.get("user_anchor_state")
        if anchor_state is not None and not sess.get("anchor_applied", False):
            # Use the latest anchor if more user tokens arrived after target_k
            target_k = sess.get("anchor_target_k", 12)
            if sess["user_token_count"] > target_k:
                # Use latest user token state as anchor
                sess["state"] = anchor_state
            else:
                # Use original K-token anchor
                sess["state"] = anchor_state
            sess["anchor_applied"] = True

        # --- Channel bootstrap: open final channel/message deterministically ---
        if sess.get("parser") is None:
            sess["parser"] = StreamableParser(encoding, role=Role.SYSTEM)

        if sess["parser"].current_channel != "final":
            bootstrap_step = sess.get("bootstrap_step", 0)

            if bootstrap_step == 0:
                from baby.constants.harmony_tokens import CHANNEL

                sess["bootstrap_step"] = 1
                return CHANNEL
            elif bootstrap_step == 1:
                from baby.constants.harmony_tokens import final_channel_id

                sess["bootstrap_step"] = 2
                return final_channel_id(encoding)
            elif bootstrap_step == 2:
                opener = MESSAGE  # <|message|>
                sess["last_tokens"] = collections.deque(maxlen=8)
                sess["out_text"] = ""
                sess["sentence_end"] = True
                sess["fed_len"] = len(tokens)
                sess["bootstrap_step"] = 3
                return opener

        # Pure deterministic emission from the five-map engine
        res = engine.emit_next_from_state(
            sess["state"], 
            sess["omega"], 
            sess["bucket_key"], 
            sess["bucket_pos"]
        )
        if res is None:
            return None

        next_token, new_state, omega, bucket_key, bucket_pos = res
        # Advance session state and update PPE state
        sess["state"] = new_state
        sess["omega"] = omega
        sess["bucket_key"] = bucket_key
        sess["bucket_pos"] = bucket_pos
        return next_token

    return infer_next_token
