#!/usr/bin/env python3
"""
GyroSI Wikipedia Training Pipeline - Token-Aware Stream Compiler

This script converts Wikipedia text dumps into compact "gyro-tapes" (raw intron streams)
and optionally updates token-aware knowledge stores. It's optimized for the 0.9.6.7
token-aware refactored architecture.

Key Features:
- Streams articles one-by-one to avoid memory issues
- Uses HF FastTokenizer for efficient tokenization
- Converts tokens to LEB128 bytes then to introns via XOR 0xAA
- Writes compact .gyro tape files (~1.5 bytes/token)
- Optionally updates token-aware knowledge store automatically via process_egress
- Real-time progress tracking with performance metrics
- PEP8 compliant with proper typing for static analysis tools

Usage:
    # Simple Wikipedia to tape only (fastest)
    python gyro_tape_compiler.py --simple -o simple_wiki.gyro

    # Full Wikipedia to tape with learning
    python gyro_tape_compiler.py --full -o full_wiki.gyro --learn

    # Limit articles for testing
    python gyro_tape_compiler.py --simple -o test.gyro --limit 1000
"""

import argparse
import gzip
import json
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Iterator, Iterable, Optional, List, Union, cast, Dict

from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from toys.communication.tokenizer import id_to_bytes  # noqa: E402
from baby.intelligence import GyroSI  # noqa: E402
from baby.contracts import AgentConfig, PreferencesConfig  # noqa: E402

# Default constants
DEFAULT_LOG_INTERVAL = 50_000  # Default logging interval
GENE_MIC_S = 0xAA  # ψ(b) = b ^ 0xAA - the holographic boundary transcription
DEFAULT_BLANK_LINES = 3  # Default number of blank lines that separate articles


def iter_wiki_articles(files: Iterable[Path], blank_line_threshold: int = DEFAULT_BLANK_LINES) -> Iterator[str]:
    """
    Yield one article (str) at a time from Wikipedia dataset files.

    Articles are separated by blank_line_threshold or more blank lines.

    Args:
        files: Iterable of file paths to process
        blank_line_threshold: Number of consecutive blank lines to define article boundary

    Yields:
        str: Complete article text as a single string
    """
    for file_path in files:
        # Handle both .txt and .gz files
        open_func = gzip.open if file_path.suffix == ".gz" else open

        with open_func(file_path, "rt", encoding="utf-8", errors="ignore") as f:
            buffer: List[str] = []
            blank_line_count = 0

            for line in f:
                if line.strip():
                    buffer.append(line)
                    blank_line_count = 0
                else:
                    blank_line_count += 1
                    # Configurable blank line threshold for article boundary
                    if blank_line_count >= blank_line_threshold and buffer:
                        yield "".join(buffer)
                        buffer.clear()

            # Don't forget the last article if file doesn't end with blank lines
            if buffer:
                yield "".join(buffer)


def build_agent(private_knowledge_path: Path) -> GyroSI:
    """
    Create a private GyroSI agent for training.

    Args:
        private_knowledge_path: Path to private knowledge store file

    Returns:
        GyroSI: Configured agent instance
    """
    # Create dummy public knowledge file if it doesn't exist
    dummy_public = PROJECT_ROOT / "toys/training/dummy_public_knowledge.bin"
    if not dummy_public.exists():
        dummy_public.parent.mkdir(parents=True, exist_ok=True)
        dummy_public.write_bytes(b"")

    # Load preferences for auto-pruning
    prefs_path = PROJECT_ROOT / "memories/memory_preferences.json"
    if prefs_path.exists():
        with open(prefs_path) as f:
            preferences = json.load(f)
    else:
        # Default pruning preferences
        preferences = {
            "pruning": {
                "confidence_threshold": 0.05,
                "enable_auto_decay": True,
                "decay_factor": 0.995,
            }
        }

    # Configure agent with proper paths
    config: AgentConfig = {
        "ontology_path": str(PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"),
        "phenomenology_map_path": str(PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"),
        "public_knowledge_path": str(dummy_public),
        "private_knowledge_path": str(private_knowledge_path),
        "preferences": cast(PreferencesConfig, preferences.get("pruning", {})),
    }

    return GyroSI(config, agent_id="wiki_trainer")


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def format_time(seconds: float) -> str:
    """
    Format seconds into human readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted time string (HH:MM:SS or MM:SS)
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


def compile_stream(
    articles: Iterable[str],
    output_tape_path: Path,
    agent: Optional[GyroSI] = None,
    limit: Optional[int] = None,
    log_interval: int = DEFAULT_LOG_INTERVAL,
) -> Dict[str, Union[int, float, str]]:
    """
    Compile Wikipedia articles into a gyro-tape and optionally update knowledge store.

    Args:
        articles: Iterable of article texts
        output_tape_path: Path to output .gyro file
        agent: Optional GyroSI agent for learning (via process_egress)
        limit: Optional limit on number of articles to process
        log_interval: How often to log progress (in number of articles)

    Returns:
        Dictionary of statistics about the compilation
    """
    # Initialize fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)  # type: ignore

    # Create output directory
    output_tape_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize counters
    articles_processed = 0
    tokens_processed = 0
    bytes_written = 0
    start_time = time.time()
    last_log_time = start_time

    print("🚀 Starting compilation...")
    print(f"   Output: {output_tape_path}")
    print(f"   Learning: {'Yes' if agent else 'No'}")
    if limit:
        print(f"   Limit: {limit:,} articles")
    print("-" * 60)

    # Open output file for writing
    with output_tape_path.open("wb") as tape_file:
        # Apply limit if specified
        article_iterator = islice(articles, limit) if limit else articles

        try:
            for article_text in article_iterator:
                articles_processed += 1

                # Tokenize article
                token_ids = tokenizer.encode(article_text, add_special_tokens=False)
                tokens_processed += len(token_ids)

                # Process each token
                for token_id in token_ids:
                    # Convert token ID to LEB128 bytes
                    leb_bytes = id_to_bytes(token_id)

                    # Ensure leb_bytes is treated as bytes for iteration
                    # This handles any potential future API changes
                    if not isinstance(leb_bytes, bytes):
                        leb_bytes = bytes(cast(List[int], leb_bytes))

                    # Convert all bytes to introns and write as batch
                    intron_bytes = bytes(b ^ GENE_MIC_S for b in leb_bytes)
                    tape_file.write(intron_bytes)
                    bytes_written += len(intron_bytes)

                    # If learning, feed each intron to engine
                    # process_egress handles token boundary detection and learning
                    if agent:
                        for intron in intron_bytes:
                            agent.engine.process_egress(intron)

                # Log progress periodically based on articles or time
                current_time = time.time()
                if (
                    articles_processed % log_interval == 0 or current_time - last_log_time >= 60
                ):  # At least every minute

                    elapsed = current_time - start_time
                    rate_articles = articles_processed / elapsed if elapsed > 0 else 0
                    rate_tokens = tokens_processed / elapsed if elapsed > 0 else 0
                    rate_bytes = bytes_written / elapsed / (1024 * 1024) if elapsed > 0 else 0  # MB/s

                    # Calculate ETA if we know the limit
                    eta_str = ""
                    if limit:
                        articles_remaining = limit - articles_processed
                        if rate_articles > 0:
                            eta_seconds = articles_remaining / rate_articles
                            eta_str = f" | ETA: {format_time(eta_seconds)}"

                    progress_msg = (
                        f"📊 Progress: {articles_processed:,} articles"
                        f"{f'/{limit:,}' if limit else ''} | "
                        f"{tokens_processed:,} tokens | "
                        f"{format_size(bytes_written)} written | "
                        f"{rate_articles:.0f} arts/s | "
                        f"{rate_tokens:.0f} tokens/s | "
                        f"{rate_bytes:.1f} MB/s{eta_str}"
                    )
                    print(progress_msg)
                    last_log_time = current_time

                    # Flush output to show progress in logs
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n⚠️  Compilation interrupted! Saving progress...")

        finally:
            # Always flush the tape file before closing
            tape_file.flush()

    # Commit any pending changes to knowledge store
    if agent and hasattr(agent.engine.operator.store, "commit"):
        agent.engine.operator.store.commit()

    # Final statistics
    total_elapsed = time.time() - start_time
    final_rate_articles = articles_processed / total_elapsed if total_elapsed > 0 else 0
    final_rate_tokens = tokens_processed / total_elapsed if total_elapsed > 0 else 0
    final_rate_bytes = bytes_written / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0

    # Prepare stats dictionary
    stats: Dict[str, Union[int, float, str]] = {
        "articles_processed": articles_processed,
        "tokens_processed": tokens_processed,
        "bytes_written": bytes_written,
        "processing_time": total_elapsed,
        "rate_articles": final_rate_articles,
        "rate_tokens": final_rate_tokens,
        "rate_bytes": final_rate_bytes,
    }

    # Add knowledge stats if learning was enabled
    if agent:
        knowledge_path_str = agent.config.get("private_knowledge_path")
        if knowledge_path_str is not None:
            knowledge_path = Path(knowledge_path_str)
            if knowledge_path.exists():
                knowledge_size = knowledge_path.stat().st_size
                stats["knowledge_size"] = knowledge_size

    print("-" * 60)
    print("✅ Compilation completed!")
    print(f"   Articles processed: {articles_processed:,}")
    print(f"   Tokens processed: {tokens_processed:,}")
    print(f"   Tape size: {format_size(bytes_written)}")
    print(f"   Processing time: {format_time(total_elapsed)}")
    final_perf_msg = (
        f"   Performance: {final_rate_articles:.0f} arts/s | "
        f"{final_rate_tokens:.0f} tokens/s | "
        f"{final_rate_bytes:.1f} MB/s"
    )
    print(final_perf_msg)
    if agent and "knowledge_size" in stats:
        print(f"   Knowledge store: {format_size(int(stats['knowledge_size']))}")

    # Write stats to JSON file alongside the tape
    stats_path = output_tape_path.with_suffix(".stats.json")
    with stats_path.open("w") as f:
        # Convert stats to JSON-serializable format
        json_stats = {k: v if not isinstance(v, Path) else str(v) for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)

    print(f"   Stats saved to: {stats_path}")

    return stats


def replay_tape(
    tape_path: Path,
    agent: GyroSI,
    log_interval: int = 1_000_000,  # Log every 1M bytes by default
) -> Dict[str, Union[int, float, str]]:
    """
    Replay a gyro-tape through an agent for learning.

    Args:
        tape_path: Path to .gyro file
        agent: GyroSI agent to feed introns to
        log_interval: How often to log progress (in bytes)

    Returns:
        Dictionary of statistics about the replay
    """
    print(f"🎬 Replaying tape: {tape_path}")

    # Get tape file size for progress tracking
    tape_size = tape_path.stat().st_size

    bytes_processed = 0
    start_time = time.time()
    last_log_time = start_time

    # Optimize reading with larger buffer
    buffer_size = 1024 * 1024  # 1MB buffer for better throughput

    # Get initial state for tracking
    initial_state = agent.engine.get_state_info()["tensor_index"]

    with tape_path.open("rb") as f:
        try:
            while True:
                # Read in chunks for better performance
                chunk = f.read(buffer_size)
                if not chunk:
                    break

                # Process each byte in the chunk
                for intron in chunk:
                    agent.engine.process_egress(intron)

                bytes_processed += len(chunk)

                # Log progress periodically based on bytes or time
                current_time = time.time()
                if bytes_processed % log_interval == 0 or current_time - last_log_time >= 60:  # At least every minute

                    elapsed = current_time - start_time
                    rate = bytes_processed / elapsed / (1024 * 1024) if elapsed > 0 else 0

                    # Calculate progress percentage and ETA
                    progress_pct = bytes_processed / tape_size * 100
                    eta_seconds = (
                        (tape_size - bytes_processed) / (bytes_processed / elapsed)
                        if elapsed > 0 and bytes_processed > 0
                        else 0
                    )

                    # Current state for tracking evolution
                    current_state = agent.engine.get_state_info()["tensor_index"]
                    state_delta = "same" if current_state == initial_state else "changed"

                    progress_msg = (
                        f"   {progress_pct:.1f}% | "
                        f"Processed {format_size(bytes_processed)}/{format_size(tape_size)} | "
                        f"Rate: {rate:.1f} MB/s | "
                        f"ETA: {format_time(eta_seconds)} | "
                        f"State: {state_delta}"
                    )
                    print(progress_msg)
                    last_log_time = current_time

                    # Flush output to show progress in logs
                    sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n⚠️  Replay interrupted!")

    # Commit changes
    if hasattr(agent.engine.operator.store, "commit"):
        agent.engine.operator.store.commit()

    # Final state check
    final_state = agent.engine.get_state_info()["tensor_index"]
    state_changed = final_state != initial_state

    total_elapsed = time.time() - start_time
    final_rate = bytes_processed / total_elapsed / (1024 * 1024) if total_elapsed > 0 else 0

    # Prepare stats dictionary
    stats: Dict[str, Union[int, float, str]] = {
        "tape_size": tape_size,
        "bytes_processed": bytes_processed,
        "processing_time": total_elapsed,
        "rate_bytes": final_rate,
        "state_changed": state_changed,
        "initial_state": initial_state,
        "final_state": final_state,
    }

    # Add knowledge stats
    knowledge_path_str = agent.config.get("private_knowledge_path")
    if knowledge_path_str is not None:
        knowledge_path = Path(knowledge_path_str)
        if knowledge_path.exists():
            knowledge_size = knowledge_path.stat().st_size
            stats["knowledge_size"] = knowledge_size

    print(f"✅ Replay completed: {format_size(bytes_processed)} in {format_time(total_elapsed)}")
    print(f"   Final rate: {final_rate:.1f} MB/s")
    print(f"   State evolution: {'Changed' if state_changed else 'Unchanged'}")
    if "knowledge_size" in stats:
        print(f"   Knowledge store: {format_size(int(stats['knowledge_size']))}")

    # Write stats to JSON file
    stats_path = tape_path.with_suffix(".replay.json")
    with stats_path.open("w") as f:
        # Convert stats to JSON-serializable format
        json_stats = {k: v if not isinstance(v, Path) else str(v) for k, v in stats.items()}
        json.dump(json_stats, f, indent=2)

    print(f"   Stats saved to: {stats_path}")

    return stats


def main() -> int:
    """
    Main entry point for the gyro-tape compiler.

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="GyroSI Wikipedia Training Pipeline - Token-Aware Stream Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple Wikipedia to tape only (fastest)
  python gyro_tape_compiler.py --simple -o simple_wiki.gyro

  # Full Wikipedia to tape with learning
  python gyro_tape_compiler.py --full -o full_wiki.gyro --learn

  # Limit articles for testing
  python gyro_tape_compiler.py --simple -o test.gyro --limit 1000

  # Replay existing tape for learning
  python gyro_tape_compiler.py --replay tape.gyro --learn

  # Customize blank line threshold for article boundary
  python gyro_tape_compiler.py --simple -o simple_wiki.gyro --blank-lines 2
        """,
    )

    # Input source group
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--simple", action="store_true", help="Use Simple Wikipedia dataset")

    source_group.add_argument("--full", action="store_true", help="Use Full English Wikipedia dataset")

    source_group.add_argument("--replay", type=str, help="Replay existing .gyro tape file")

    # Output specification
    parser.add_argument("-o", "--output", help="Output .gyro tape file path (required unless using --replay)")

    # Learning option
    parser.add_argument("--learn", action="store_true", help="Also update private knowledge store")

    # Processing options
    parser.add_argument("--limit", type=int, help="Stop after processing N articles (useful for testing)")

    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help=f"Log progress every N articles (default: {DEFAULT_LOG_INTERVAL})",
    )

    parser.add_argument(
        "--blank-lines",
        type=int,
        default=DEFAULT_BLANK_LINES,
        help=f"Number of blank lines that separate articles (default: {DEFAULT_BLANK_LINES})",
    )

    args = parser.parse_args()

    # Handle replay mode
    if args.replay:
        if not args.learn:
            print("❌ Error: --replay requires --learn", file=sys.stderr)
            return 1

        tape_path = Path(args.replay)
        if not tape_path.exists():
            print(f"❌ Error: Tape file not found: {tape_path}", file=sys.stderr)
            return 1

        private_knowledge_path = tape_path.with_suffix(".bin")
        print(f"🧠 Creating agent with knowledge store: {private_knowledge_path}")
        replay_agent = build_agent(private_knowledge_path)

        try:
            replay_tape(tape_path, replay_agent, log_interval=args.log_interval * 1000)
            return 0
        except KeyboardInterrupt:
            print("\n⚠️  Replay interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error during replay: {e}", file=sys.stderr)
            return 1
        finally:
            replay_agent.close()

    # Validate compilation mode arguments
    if not args.output:
        print("❌ Error: --output is required for compilation mode", file=sys.stderr)
        return 1

    # Determine dataset directory
    dataset_dir = (
        PROJECT_ROOT / "toys/training/wikipedia_simple_data"
        if args.simple
        else PROJECT_ROOT / "toys/training/wikipedia_full_data"
    )

    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 1

    # Find dataset files
    files = sorted(dataset_dir.rglob("*"))
    files = [f for f in files if f.is_file() and (f.suffix == ".txt" or f.suffix == ".gz")]

    if not files:
        print(f"❌ Error: No .txt or .gz files found in {dataset_dir}", file=sys.stderr)
        return 1

    print(f"📚 Found {len(files)} files in {dataset_dir}")

    # Create agent if learning is enabled
    agent: Optional[GyroSI] = None
    if args.learn:
        private_knowledge_path = Path(args.output).with_suffix(".bin")
        knowledge_msg = f"🧠 Creating private agent with knowledge store: " f"{private_knowledge_path}"
        print(knowledge_msg)
        agent = build_agent(private_knowledge_path)

    # Create article iterator with configurable blank line threshold
    articles = iter_wiki_articles(files, blank_line_threshold=args.blank_lines)

    # Compile the stream
    try:
        compile_stream(
            articles=articles,
            output_tape_path=Path(args.output),
            agent=agent,
            limit=args.limit,
            log_interval=args.log_interval,
        )
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during compilation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Clean up agent if created
        if agent:
            try:
                agent.close()
                print("🧹 Agent closed successfully")
            except Exception as e:
                print(f"⚠️  Error closing agent: {e}")


if __name__ == "__main__":
    sys.exit(main())
