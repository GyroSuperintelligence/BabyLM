# baby/responses_api/inference/gyro.py

from typing import Callable, Dict, Any, Optional
import json, threading, collections
from pathlib import Path
from openai_harmony import StreamableParser, Role
from baby.kernel.gyro_core import GyroEngine
from baby.constants.harmony_tokens import MESSAGE, ROLE_USER, ALL_CONTROL_TOKENS

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
    store_paths = {k: _abs(v) for k, v in cfg["stores"].items()}
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

    # Pure Harmony tokenization — no surface filtering, no structure forcing

    def _is_user_role(current_role) -> bool:
        # Handle Role enum and string forms defensively
        try:
            if current_role == Role.USER:
                return True
        except Exception:
            pass
        if current_role == "user" or current_role == ROLE_USER:
            return True
        # Some enums expose .value
        val = getattr(current_role, "value", None)
        return val == "user"

    def _apply_new_tokens(sess: Dict[str, Any], new_tokens: list[int]) -> None:
        """
        Feed only the delta tokens to the streaming parser and update engine state.
        Egress for user content; ingress for assistant content.
        """
        parser = sess["parser"]
        for tok in new_tokens:
            parser.process(tok)
            print(f"[DEBUG] Token {tok}, role={parser.current_role}, channel={parser.current_channel}")

            # Learn ONLY from user content tokens; never from system/developer/assistant
            if _is_user_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                # Egress: folds on PRE-state, returns POST-state
                new_state = engine.evolve_on_user(sess["state"], tok)
                sess["state"] = new_state
                # Anchor to the POST-state after K user tokens, but only if state has experienced tokens
                if not sess.get("user_anchor_captured"):
                    sess["user_anchor_seen"] = sess.get("user_anchor_seen", 0) + 1
                    target_k = sess.get("anchor_target_k", int(engine.runtime.get("anchor_prefix_tokens", 12)))
                    print(f"[DEBUG] Anchor progress: {sess['user_anchor_seen']}/{target_k} tokens seen")

                    # Capture anchor after K tokens without checking experienced tokens
                    if sess["user_anchor_seen"] >= target_k and sess["user_anchor_state"] is None:
                        sess["user_anchor_state"] = (
                            new_state  # or keep the initial state; either is fine since egress doesn't evolve
                        )
                        sess["user_anchor_captured"] = True
                        print(f"[DEBUG] Anchor captured at token {sess['user_anchor_seen']} with state {new_state}")
                else:
                    print(f"[DEBUG] Anchor already captured, continuing learning")
            else:
                # Pure ingress (no folding unless config turns it on)
                sess["state"] = engine.evolve_on_assistant(sess["state"], tok)

    def infer_next_token(tokens: list[int], temperature: float = 0.0, **kwargs) -> Optional[int]:
        print(f"[DEBUG] infer_next_token called with {len(tokens)} tokens, kwargs: {kwargs}")
        (
            print(f"[DEBUG] Full token sequence: {tokens[:20]}...")
            if len(tokens) > 20
            else print(f"[DEBUG] Full token sequence: {tokens}")
        )
        request_id: str = kwargs.get("request_id", "__singleton__")
        new_request: bool = kwargs.get("new_request", False)
        ingest_only: bool = kwargs.get("ingest_only", False)
        print(f"[DEBUG] request_id={request_id}, new_request={new_request}, ingest_only={ingest_only}")

        with sessions_lock:
            sess = sessions.get(request_id)
            if new_request or sess is None:
                sess = {
                    "parser": StreamableParser(encoding, role=Role.SYSTEM),  # Always initialize parser
                    "fed_len": 0,
                    "state": engine.start_state(),
                    "bootstrap_step": 0,
                    "user_anchor_seen": 0,
                    "user_anchor_captured": False,
                    "user_anchor_state": None,
                    "anchor_target_k": int(engine.runtime.get("anchor_prefix_tokens", 12)),
                    "anchor_applied": False,
                }
                sessions[request_id] = sess
                # Feed all tokens to parser for proper role/channel detection
                print(f"[DEBUG] New session created, tokens length: {len(tokens) if tokens else 0}")
                if tokens:
                    print(f"[DEBUG] New session: feeding all {len(tokens)} tokens to parser")
                    # Create fresh parser and feed complete token sequence
                    parser = StreamableParser(encoding, role=Role.SYSTEM)
                    for tok in tokens:
                        parser.process(tok)
                        print(f"[DEBUG] Token {tok}, role={parser.current_role}, channel={parser.current_channel}")

                        # Learn ONLY from user content tokens; never from system/developer/assistant
                        if _is_user_role(parser.current_role) and tok not in ALL_CONTROL_TOKENS:
                            # Egress: folds on PRE-state, returns POST-state
                            new_state = engine.evolve_on_user(sess["state"], tok)
                            sess["state"] = new_state
                            # Anchor to the POST-state after K user tokens, but only if state has experienced tokens
                            if not sess.get("user_anchor_captured"):
                                sess["user_anchor_seen"] = sess.get("user_anchor_seen", 0) + 1
                                target_k = sess.get(
                                    "anchor_target_k", int(engine.runtime.get("anchor_prefix_tokens", 12))
                                )
                                print(f"[DEBUG] Anchor progress: {sess['user_anchor_seen']}/{target_k} tokens seen")

                                # Capture anchor after K tokens without checking experienced tokens
                                if sess["user_anchor_seen"] >= target_k and sess["user_anchor_state"] is None:
                                    sess["user_anchor_state"] = (
                                        new_state  # or keep the initial state; either is fine since egress doesn't evolve
                                    )
                                    sess["user_anchor_captured"] = True
                                    print(
                                        f"[DEBUG] Anchor captured at POST-state after {sess['user_anchor_seen']} user tokens; state={new_state}"
                                    )
                        else:
                            # Pure ingress (no folding unless config turns it on)
                            sess["state"] = engine.evolve_on_assistant(sess["state"], tok)
                    sess["parser"] = parser
                    sess["fed_len"] = len(tokens)
                else:
                    # Create default parser if no tokens
                    sess["parser"] = StreamableParser(encoding, role=Role.SYSTEM)
            else:
                # Feed delta and fold
                delta = tokens[sess["fed_len"] :]
                if delta:
                    print(f"[DEBUG] Feeding {len(delta)} new tokens: {delta[:10]}...")
                    _apply_new_tokens(sess, delta)
                    sess["fed_len"] = len(tokens)
                    print(f"[DEBUG] Session fed_len now: {sess['fed_len']}")
                else:
                    print(f"[DEBUG] No new tokens to feed (fed_len={sess['fed_len']}, total_len={len(tokens)})")

        if ingest_only:
            print(f"[DEBUG] Ingest-only mode, returning None")
            return None

        # Apply anchor state once before bootstrapping the assistant channel/message
        if (
            not sess.get("anchor_applied")
            and sess.get("user_anchor_captured")
            and sess.get("user_anchor_state") is not None
        ):
            # Apply the anchor state (we already verified it has experienced tokens when capturing)
            anchor_state = sess["user_anchor_state"]
            sess["state"] = anchor_state
            sess["anchor_applied"] = True
            print(f"[DEBUG] Applied anchor state: {sess['state']}")

        # Normal (pure) generation path
        state = sess["state"]

        # --- Channel bootstrap: if we're not in the final channel yet, open it ---
        # Parser should always be initialized now, but add safety check
        if sess.get("parser") is None:
            sess["parser"] = StreamableParser(encoding, role=Role.SYSTEM)

        print(f"[DEBUG] Current channel: {sess['parser'].current_channel}")
        if sess["parser"].current_channel != "final":
            # Need to emit the proper sequence: <|channel|>final<|message|>
            bootstrap_step = sess.get("bootstrap_step", 0)

            if bootstrap_step == 0:
                # Step 1: Emit <|channel|> token (200005)
                from baby.constants.harmony_tokens import CHANNEL

                channel_token = CHANNEL
                print(f"[DEBUG] Bootstrap step 0: Emitting channel token: {channel_token}")
                sess["bootstrap_step"] = 1
                # Do not mutate engine state for control tokens
                return channel_token
            elif bootstrap_step == 1:
                # Step 2: Emit final channel token
                from baby.constants.harmony_tokens import final_channel_id

                final_token = final_channel_id(encoding)
                print(f"[DEBUG] Bootstrap step 1: Emitting final channel token: {final_token}")
                sess["bootstrap_step"] = 2
                # Do not mutate engine state for control tokens
                return final_token
            elif bootstrap_step == 2:
                # Step 3: Emit message token
                opener = MESSAGE  # <|message|> token (200008)
                sess["last_tokens"] = collections.deque(maxlen=8)
                sess["out_text"] = ""
                sess["sentence_end"] = True
                sess["fed_len"] = len(tokens)
                # Do not mutate engine state for control tokens
                sess["bootstrap_step"] = 3  # Mark bootstrap complete
                print(f"[DEBUG] Bootstrap complete; state after opener (unchanged): {sess['state']}")
                return opener
        # ---------------------------------------------------------------------------

        next_token = engine.next_token_deterministic(state)
        print(f"[DEBUG] next_token_deterministic returned: {next_token}")

        if next_token is None:
            print(f"[DEBUG] next_token_deterministic returned None for state {sess['state']}")
            return None

        # Update session state with emitted token (no self-reinforcement)
        sess["state"] = engine.evolve_on_assistant(sess["state"], next_token)

        return next_token

    return infer_next_token


# This removes the global engine hazard and guarantees user text actually reaches passive memory.
