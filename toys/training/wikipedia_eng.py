#!/usr/bin/env python3
"""
Wikipedia Training - High-Performance Implementation

Key optimizations for 1000+ articles/second:
- Cached tokenizer for maximum speed
- Sentence buffering with 100 sentences per batch_learn
- Less frequent commits (every 100 articles)
- Streaming processing to avoid memory issues
- Optimized for 16-core server
- Reduced logging/checkpoint frequency for speed
- Fixed checkpoint mutation that caused progress loops
- Fixed knowledge file growth issue (batch_learn always learns)
- Proper tokenizer byte encoding
- Better progress tracking
"""

from toys.communication import tokenizer as gyrotok
from baby.intelligence import GyroSI
from baby.contracts import AgentConfig
import sys
from pathlib import Path
import argparse
import json
import logging
import os
import re
import signal
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Any

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Cache the tokenizer for high performance
TOKENIZER = gyrotok._load("bert-base-uncased")

# High-performance constants
SENTENCES_PER_LEARN = 100  # Bigger batches
ARTICLES_PER_COMMIT = 100  # Less frequent commits

# Environment setup for parallel tokenization
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_NUM_THREADS"] = "16"
os.environ["NUMBA_NUM_THREADS"] = "16"  # For JIT optimization

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

# Constants - Optimized for 16-core server and maximum speed
SENT_RE = re.compile(r"""(?<=[.!?])['")]*\s+(?=[A-Z])""", re.VERBOSE)
MIN_ARTICLE_TOKENS = 20
MIN_SENTENCE_TOKENS = 3
LEARNING_BATCH_SIZE = 50  # Increased for better throughput
PREPROCESSING_BATCH_SIZE = 16000  # Increased for maximum parallel processing
CHECKPOINT_INTERVAL = 5000  # Less frequent checkpoints for speed
LOG_INTERVAL = 500  # Less frequent logging for speed
MONITOR_INTERVAL = 2000  # Less frequent monitoring for speed

# Global state
shutdown_requested = False


def signal_handler(signum: int, frame: Any) -> None:
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print("\n⚠️  Shutdown requested. Finishing current batch...")


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup rotating file logger."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("wikipedia_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Rotating file handler
    log_file = log_dir / f"training_{int(time.time())}.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f}GB"


def robust_sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using robust regex approach.
    Handles abbreviations and common edge cases.
    """
    # Handle common abbreviations
    text = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.\s*", r"\1<PERIOD>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Inc|Ltd|Corp|Co|LLC)\.\s*", r"\1<PERIOD>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s*", r"\1<PERIOD>", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(BC|AD)\.\s*", r"\1<PERIOD>", text, flags=re.IGNORECASE)

    # Split using regex
    sentences = SENT_RE.split(text)

    # Clean up and restore periods
    result = []
    for sentence in sentences:
        sentence = sentence.replace("<PERIOD>", ".").strip()
        if sentence and len(sentence) > 10:
            result.append(sentence)

    return result


def preprocess_article(article_text: str) -> Optional[List[str]]:
    """
    Preprocess article into clean sentences.
    Returns list of sentence strings or None if article should be skipped.
    """
    # Much more lenient
    if len(article_text.strip()) < 50:  # Was 100
        return None
    if "#REDIRECT" in article_text.upper()[:20]:  # Only check start
        return None
    # Remove disambiguation check entirely - these are valid content

    sentences = robust_sentence_split(article_text)
    return sentences if sentences else None


def preprocess_articles_parallel(
    articles: List[Tuple[int, str]], max_workers: Optional[int] = None
) -> List[Tuple[int, List[str]]]:
    """
    Preprocess articles in parallel, return (article_idx, sentences) pairs.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 16

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(preprocess_article, text): idx for idx, text in articles}

        for future in as_completed(future_to_idx):
            if shutdown_requested:
                executor.shutdown(wait=False)
                break

            idx = future_to_idx[future]
            try:
                sentences = future.result()
                if sentences:
                    results.append((idx, sentences))
            except Exception as e:
                print(f"⚠️  Error preprocessing article {idx}: {e}")

    results.sort(key=lambda x: x[0])
    return results


def create_agent(knowledge_path: Path) -> GyroSI:
    """Create a private agent with proper configuration."""
    # Create dummy public knowledge file if it doesn't exist
    dummy_public = PROJECT_ROOT / "toys/training/dummy_public_knowledge.bin"
    if not dummy_public.exists():
        dummy_public.parent.mkdir(parents=True, exist_ok=True)
        dummy_public.write_bytes(b"")

    # Load preferences
    prefs_path = PROJECT_ROOT / "memories/memory_preferences.json"
    if prefs_path.exists():
        with open(prefs_path) as f:
            preferences = json.load(f)
    else:
        preferences = {"pruning": {"confidence_threshold": 0.05, "enable_auto_decay": True, "decay_factor": 0.995}}

    # Use explicit paths that match your project structure
    config: AgentConfig = {
        "ontology_path": str(PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"),
        "phenomenology_map_path": str(PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"),
        "public_knowledge_path": str(dummy_public),
        "private_knowledge_path": str(knowledge_path),
        "learn_batch_size": 1000,
        "preferences": preferences.get("pruning", {}),
    }

    return GyroSI(config, agent_id="wikipedia_trainer")


def save_checkpoint_atomic(checkpoint_data: Dict[str, Any], checkpoint_path: Path) -> None:
    """Save checkpoint atomically with CRC validation. FIXED: no longer mutates input dict."""
    tmp_path = checkpoint_path.with_suffix(".tmp")

    try:
        # Create a deep copy to avoid mutating the original stats dict
        data_to_write = json.loads(json.dumps(checkpoint_data))

        # Add CRC to the copy (not the original)
        json_str = json.dumps(data_to_write, indent=2, sort_keys=True)
        crc32 = format(zlib.crc32(json_str.encode()) & 0xFFFFFFFF, "08x")
        data_to_write["crc32"] = crc32

        # Write to temp file
        with open(tmp_path, "w") as f:
            json.dump(data_to_write, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        tmp_path.replace(checkpoint_path)

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Load and validate checkpoint."""
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path) as f:
            data = json.load(f)

        # Validate CRC if present
        if "crc32" in data:
            crc32 = data.pop("crc32")
            json_str = json.dumps(data, indent=2, sort_keys=True)
            expected_crc32 = format(zlib.crc32(json_str.encode()) & 0xFFFFFFFF, "08x")
            if crc32 != expected_crc32:
                print("⚠️  Checkpoint CRC mismatch, may be corrupted")

        return data
    except Exception as e:
        print(f"⚠️  Error loading checkpoint: {e}")
        return None


def get_knowledge_entry_count(knowledge_path: Path) -> int:
    """Get number of entries in knowledge file."""
    if not knowledge_path.exists():
        return 0

    try:
        # Read file and count entries (expensive operation)
        with open(knowledge_path, "rb") as f:
            data = f.read()
        return len(data) // 1024  # Rough estimate
    except Exception:
        return 0


def stream_articles(file_path: Path):
    """Stream articles one at a time from disk."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        article_lines = []
        for line in f:
            if line.strip():
                article_lines.append(line)
            elif article_lines:  # Empty line = article boundary
                yield "".join(article_lines)
                article_lines = []
        if article_lines:  # Don't forget the last one
            yield "".join(article_lines)


def train_with_batch_learn(
    agent: GyroSI,
    all_articles: List[str],
    checkpoint_path: Path,
    knowledge_path: Path,
    logger: logging.Logger,
    max_workers: Optional[int] = None,
    max_articles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train using batch_learn with proper state evolution.
    Process ALL articles, not just the first batch.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 16

    stats: Dict[str, Any] = {
        "articles_processed": 0,
        "articles_skipped": 0,
        "sentences_processed": 0,
        "batches_processed": 0,
        "start_time": time.time(),
        "unique_states_count": 0,  # Just count, don't store
        "last_seen_states": set(),  # Keep only last 1000 for sampling
    }

    # Process and system monitoring
    process = psutil.Process() if HAVE_PSUTIL else None

    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        start_article = checkpoint.get("last_article_index", 0)
        checkpoint_stats = checkpoint.get("stats", {})
        # Merge checkpoint stats but preserve the set type for last_seen_states
        for key, value in checkpoint_stats.items():
            if key == "last_seen_states" and isinstance(value, list):
                stats[key] = set(value)
            else:
                stats[key] = value
        logger.info(f"Resuming from checkpoint at article {start_article}")
    else:
        start_article = 0

    # Limit articles if requested
    if max_articles and len(all_articles) > max_articles:
        all_articles = all_articles[:max_articles]

    total_articles = len(all_articles)
    logger.info(f"Processing {total_articles:,} total articles")

    # SEP token bytes
    sep_bytes = gyrotok.sep_bytes(1)

    # Track state evolution
    initial_state = agent.engine.get_state_info()["tensor_index"]
    logger.info(f"Starting from state index: {initial_state}")

    # Process articles in chunks
    articles_to_process = all_articles[start_article:]
    last_checkpoint_time = time.time()

    # Setup progress tracking
    if HAVE_TQDM:
        pbar = tqdm(total=len(articles_to_process), desc="Training", unit="articles")
        pbar.update(0)

    # High-performance sentence buffering
    sentence_buffer = bytearray()
    sentence_count = 0
    articles_since_commit = 0

    # Process ALL articles, not just first batch
    for chunk_start in range(0, len(articles_to_process), PREPROCESSING_BATCH_SIZE):
        if shutdown_requested:
            break

        # Get chunk of articles
        chunk_end = min(chunk_start + PREPROCESSING_BATCH_SIZE, len(articles_to_process))
        article_chunk = articles_to_process[chunk_start:chunk_end]

        # Create indexed articles for preprocessing
        indexed_articles = [(start_article + chunk_start + i, article) for i, article in enumerate(article_chunk)]

        logger.info(
            f"Preprocessing articles {chunk_start + start_article} to "
            f"{chunk_end + start_article} ({len(indexed_articles)} articles)"
        )

        # Preprocess in parallel
        preprocessed = preprocess_articles_parallel(indexed_articles, max_workers)

        if not preprocessed:
            logger.warning(f"No valid articles in chunk {chunk_start}-{chunk_end}")
            stats["articles_skipped"] += len(article_chunk)
            continue

        logger.info(f"Found {len(preprocessed)} valid articles in chunk")

        # Process preprocessed articles in small learning batches
        for batch_start in range(0, len(preprocessed), LEARNING_BATCH_SIZE):
            if shutdown_requested:
                break

            batch_end = min(batch_start + LEARNING_BATCH_SIZE, len(preprocessed))
            learning_batch = preprocessed[batch_start:batch_end]

            for article_idx, sentences in learning_batch:
                for sentence in sentences:
                    try:
                        ids = TOKENIZER.encode(sentence).ids  # Use cached tokenizer
                        sentence_buffer.extend(gyrotok.bytes_from_ids(ids))
                        sentence_buffer.extend(sep_bytes)
                        sentence_count += 1
                        stats["sentences_processed"] += 1

                        if sentence_count >= SENTENCES_PER_LEARN:
                            agent.ingest(bytes(sentence_buffer))
                            sentence_buffer = bytearray()
                            sentence_count = 0

                    except Exception as e:
                        logger.warning(f"Failed to encode sentence: {e}")

                stats["articles_processed"] += 1
                articles_since_commit += 1

                # Update progress bar
                if HAVE_TQDM:
                    pbar.update(1)

                # Track state evolution
                if stats["articles_processed"] % MONITOR_INTERVAL == 0:
                    current_state = agent.engine.get_state_info()["tensor_index"]
                    if current_state not in stats["last_seen_states"]:
                        stats["unique_states_count"] += 1
                    stats["last_seen_states"].add(current_state)
                    if len(stats["last_seen_states"]) > 1000:
                        # Remove oldest item (sets don't have pop(), so we'll just keep the size bounded)
                        # Convert to list, remove first item, convert back to set
                        temp_list = list(stats["last_seen_states"])
                        temp_list.pop(0)
                        stats["last_seen_states"] = set(temp_list)
                    logger.info(
                        f"State evolution check - Current state: {current_state}, "
                        f"Unique states seen: {stats['unique_states_count']}"
                    )

                # Commit less frequently
                if articles_since_commit >= ARTICLES_PER_COMMIT:
                    if hasattr(agent.engine.operator.store, "commit"):
                        agent.engine.operator.store.commit()
                    articles_since_commit = 0

            # Don't forget the final buffer
            if sentence_buffer:
                agent.ingest(bytes(sentence_buffer))

            # Commit once per batch (20 articles) instead of per article
            if hasattr(agent.engine.operator.store, "commit"):
                agent.engine.operator.store.commit()

            stats["batches_processed"] += 1

            # Progress updates
            if stats["articles_processed"] % LOG_INTERVAL == 0:
                elapsed = time.time() - stats["start_time"]
                rate = stats["articles_processed"] / elapsed if elapsed > 0 else 0

                # System stats
                mem_str = ""
                cpu_str = ""
                if process:
                    try:
                        mem_mb = process.memory_info().rss / (1024 * 1024)
                        cpu_percent = process.cpu_percent(interval=0.1)
                        mem_str = f" | Mem: {mem_mb:.0f}MB"
                        cpu_str = f" | CPU: {cpu_percent:.0f}%"
                    except Exception:
                        pass

                # File stats
                file_size = knowledge_path.stat().st_size if knowledge_path.exists() else 0
                # entry_count = get_knowledge_entry_count(knowledge_path)  # REMOVE THIS - too expensive

                # Progress message
                progress_pct = (stats["articles_processed"] / total_articles * 100) if total_articles > 0 else 0
                message = (
                    f"Progress: {stats['articles_processed']:,}/{total_articles:,} "
                    f"({progress_pct:.1f}%) | "
                    f"{stats['sentences_processed']:,} sentences | "
                    f"{rate:.1f} art/s{mem_str}{cpu_str} | "
                    f"Knowledge: {format_size(file_size)}"  # No entry count
                )

                print(f"📊 {message}")
                logger.info(message)

            # Checkpoint periodically
            if time.time() - last_checkpoint_time > 300 or stats["articles_processed"] % CHECKPOINT_INTERVAL == 0:

                # Save checkpoint (stats won't be mutated anymore)
                checkpoint_data = {
                    "last_article_index": start_article + chunk_start + batch_end,
                    "stats": {
                        "articles_processed": stats["articles_processed"],
                        "articles_skipped": stats["articles_skipped"],
                        "sentences_processed": stats["sentences_processed"],
                        "batches_processed": stats["batches_processed"],
                        "start_time": stats["start_time"],
                        "last_seen_states": list(stats["last_seen_states"]),
                    },
                    "timestamp": time.time(),
                }

                save_checkpoint_atomic(checkpoint_data, checkpoint_path)

                # Log checkpoint
                file_size = knowledge_path.stat().st_size if knowledge_path.exists() else 0
                # entry_count = get_knowledge_entry_count(knowledge_path)  # REMOVE THIS - too expensive
                message = (
                    f"Checkpoint saved at article {stats['articles_processed']}, "
                    f"knowledge: {format_size(file_size)}"
                )  # No entry count
                print(f"💾 {message}")
                logger.info(message)

                last_checkpoint_time = time.time()

    if HAVE_TQDM:
        pbar.close()

    # Final state check
    final_state = agent.engine.get_state_info()["tensor_index"]
    if final_state not in stats["last_seen_states"]:
        stats["unique_states_count"] += 1
    stats["last_seen_states"].add(final_state)
    if len(stats["last_seen_states"]) > 1000:
        # Remove oldest item (sets don't have pop(), so we'll just keep the size bounded)
        # Convert to list, remove first item, convert back to set
        temp_list = list(stats["last_seen_states"])
        temp_list.pop(0)
        stats["last_seen_states"] = set(temp_list)
    logger.info(f"Final state: {final_state}, Total unique states: {stats['unique_states_count']}")

    return stats


def train_with_streaming(
    agent: GyroSI,
    files: List[Path],
    checkpoint_path: Path,
    knowledge_path: Path,
    logger: logging.Logger,
    max_workers: Optional[int] = None,
    max_articles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    High-performance streaming training that processes articles one at a time.
    """
    # Initialize stats
    stats = {
        "articles_processed": 0,
        "articles_skipped": 0,
        "sentences_processed": 0,
        "batches_processed": 0,
        "start_time": time.time(),
        "unique_states_count": 0,
        "last_seen_states": set(),
    }

    # Process and system monitoring
    process = psutil.Process() if HAVE_PSUTIL else None

    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        start_article = checkpoint.get("last_article_index", 0)
        checkpoint_stats = checkpoint.get("stats", {})
        # Merge checkpoint stats but preserve the set type for last_seen_states
        for key, value in checkpoint_stats.items():
            if key == "last_seen_states" and isinstance(value, list):
                stats[key] = set(value)
            else:
                stats[key] = value
        logger.info(f"Resuming from checkpoint at article {start_article}")
    else:
        start_article = 0

    # SEP token bytes
    sep_bytes = gyrotok.sep_bytes(1)

    # Track state evolution
    initial_state = agent.engine.get_state_info()["tensor_index"]
    logger.info(f"Starting from state index: {initial_state}")

    # Setup progress tracking
    total_articles = 0
    if max_articles:
        total_articles = max_articles
    else:
        # Count total articles for progress
        for file_path in files:
            total_articles += sum(1 for _ in stream_articles(file_path))

    if HAVE_TQDM:
        pbar = tqdm(total=total_articles, desc="Training", unit="articles")
        pbar.update(0)

    # High-performance stream processing
    sentence_buffer = bytearray()
    sentence_count = 0
    articles_since_commit = 0
    last_checkpoint_time = time.time()

    for file_idx, file_path in enumerate(files):
        if shutdown_requested:
            break

        logger.info(f"Processing file {file_idx + 1}/{len(files)}: {file_path.name}")

        for article_idx, article in enumerate(stream_articles(file_path)):
            if shutdown_requested:
                break

            if max_articles and stats["articles_processed"] >= max_articles:
                break

            # Preprocess article
            sentences = preprocess_article(article)
            if not sentences:
                stats["articles_skipped"] += 1
                continue

            # Process sentences
            for sentence in sentences:
                try:
                    ids = TOKENIZER.encode(sentence).ids  # Use cached tokenizer
                    sentence_buffer.extend(gyrotok.bytes_from_ids(ids))
                    sentence_buffer.extend(sep_bytes)
                    sentence_count += 1
                    stats["sentences_processed"] += 1

                    if sentence_count >= SENTENCES_PER_LEARN:
                        agent.ingest(bytes(sentence_buffer))
                        sentence_buffer = bytearray()
                        sentence_count = 0

                except Exception as e:
                    logger.warning(f"Failed to encode sentence: {e}")

            stats["articles_processed"] += 1
            articles_since_commit += 1

            # Update progress bar
            if HAVE_TQDM:
                pbar.update(1)

            # Track state evolution
            if stats["articles_processed"] % MONITOR_INTERVAL == 0:
                current_state = agent.engine.get_state_info()["tensor_index"]
                if current_state not in stats["last_seen_states"]:
                    stats["unique_states_count"] += 1
                stats["last_seen_states"].add(current_state)
                if len(stats["last_seen_states"]) > 1000:
                    temp_list = list(stats["last_seen_states"])
                    temp_list.pop(0)
                    stats["last_seen_states"] = set(temp_list)
                logger.info(
                    f"State evolution check - Current state: {current_state}, "
                    f"Unique states seen: {stats['unique_states_count']}"
                )

            # Commit less frequently
            if articles_since_commit >= ARTICLES_PER_COMMIT:
                if hasattr(agent.engine.operator.store, "commit"):
                    agent.engine.operator.store.commit()
                articles_since_commit = 0

            # Progress updates
            if stats["articles_processed"] % LOG_INTERVAL == 0:
                elapsed = time.time() - stats["start_time"]
                rate = stats["articles_processed"] / elapsed if elapsed > 0 else 0

                # System stats
                mem_str = ""
                cpu_str = ""
                if process:
                    try:
                        mem_mb = process.memory_info().rss / (1024 * 1024)
                        cpu_percent = process.cpu_percent(interval=0.1)
                        mem_str = f" | Mem: {mem_mb:.0f}MB"
                        cpu_str = f" | CPU: {cpu_percent:.0f}%"
                    except Exception:
                        pass

                # File stats
                file_size = knowledge_path.stat().st_size if knowledge_path.exists() else 0

                # Progress message
                progress_pct = (stats["articles_processed"] / total_articles * 100) if total_articles > 0 else 0
                message = (
                    f"Progress: {stats['articles_processed']:,}/{total_articles:,} "
                    f"({progress_pct:.1f}%) | "
                    f"{stats['sentences_processed']:,} sentences | "
                    f"{rate:.1f} art/s{mem_str}{cpu_str} | "
                    f"Knowledge: {format_size(file_size)}"
                )

                print(f"📊 {message}")
                logger.info(message)

            # Checkpoint periodically
            if time.time() - last_checkpoint_time > 300 or stats["articles_processed"] % CHECKPOINT_INTERVAL == 0:

                checkpoint_data = {
                    "last_article_index": stats["articles_processed"],
                    "stats": {
                        "articles_processed": stats["articles_processed"],
                        "articles_skipped": stats["articles_skipped"],
                        "sentences_processed": stats["sentences_processed"],
                        "batches_processed": stats["batches_processed"],
                        "start_time": stats["start_time"],
                        "last_seen_states": list(stats["last_seen_states"]),
                    },
                    "timestamp": time.time(),
                }

                save_checkpoint_atomic(checkpoint_data, checkpoint_path)

                # Log checkpoint
                file_size = knowledge_path.stat().st_size if knowledge_path.exists() else 0
                message = (
                    f"Checkpoint saved at article {stats['articles_processed']}, "
                    f"knowledge: {format_size(file_size)}"
                )
                print(f"💾 {message}")
                logger.info(message)

                last_checkpoint_time = time.time()

    # Don't forget the final buffer
    if sentence_buffer:
        agent.ingest(bytes(sentence_buffer))

    # Final commit
    if hasattr(agent.engine.operator.store, "commit"):
        agent.engine.operator.store.commit()

    stats["batches_processed"] += 1

    if HAVE_TQDM:
        pbar.close()

    # Final state check
    final_state = agent.engine.get_state_info()["tensor_index"]
    if final_state not in stats["last_seen_states"]:
        stats["unique_states_count"] += 1
    stats["last_seen_states"].add(final_state)
    if len(stats["last_seen_states"]) > 1000:
        temp_list = list(stats["last_seen_states"])
        temp_list.pop(0)
        stats["last_seen_states"] = set(temp_list)
    logger.info(f"Final state: {final_state}, Total unique states: {stats['unique_states_count']}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Wikipedia Training - Fixed Implementation")
    parser.add_argument("--simple-data", action="store_true", help="Use Simple Wikipedia")
    parser.add_argument("--full-data", action="store_true", help="Use Full Wikipedia")
    parser.add_argument("--max-articles", type=int, help="Limit articles to process")
    parser.add_argument("--workers", type=int, default=None, help="Parallel preprocessing workers")
    parser.add_argument("--batch-size", type=int, help="Learning batch size (default: 20)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    if not (args.simple_data or args.full_data):
        parser.error("Must specify either --simple-data or --full-data")

    # Setup
    signal.signal(signal.SIGINT, signal_handler)

    # Update batch size if specified
    global LEARNING_BATCH_SIZE
    if args.batch_size:
        LEARNING_BATCH_SIZE = args.batch_size

    # Default workers to CPU count
    workers = args.workers or os.cpu_count() or 16

    # Paths
    dataset_type = "simple" if args.simple_data else "full"
    data_dir = PROJECT_ROOT / f"toys/training/wikipedia_{dataset_type}_data"
    knowledge_dir = PROJECT_ROOT / "toys/training/knowledge"
    log_dir = PROJECT_ROOT / "toys/training/logs"

    # Create directories
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting Wikipedia {dataset_type} training")
    logger.info(f"Learning batch size: {LEARNING_BATCH_SIZE} articles")
    logger.info(f"Preprocessing workers: {workers}")
    logger.info(f"Preprocessing batch size: {PREPROCESSING_BATCH_SIZE}")

    # Find or create knowledge file
    if args.resume:
        knowledge_files = list(knowledge_dir.glob(f"{dataset_type}_wikipedia_*.bin"))
        if knowledge_files:
            # Sort by creation time (newest first)
            knowledge_path = max(
                knowledge_files, key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0
            )
            logger.info(f"Resuming with knowledge file: {knowledge_path}")
        else:
            logger.error("No previous run found for resume")
            return 1
    else:
        existing = list(knowledge_dir.glob(f"{dataset_type}_wikipedia_*.bin"))
        run_num = 1
        if existing:
            # Find highest run number
            for path in existing:
                try:
                    num = int(path.stem.split("_")[-1])
                    run_num = max(run_num, num + 1)
                except (ValueError, IndexError):
                    pass

        knowledge_path = knowledge_dir / f"{dataset_type}_wikipedia_{run_num}.bin"
        logger.info(f"Starting new run: {knowledge_path}")

    checkpoint_path = knowledge_path.with_suffix(".checkpoint.json")

    # Create agent
    agent = None
    try:
        logger.info("Creating GyroSI agent...")
        agent = create_agent(knowledge_path)
        logger.info("Agent created successfully")

        # Find files
        if args.simple_data:
            files = [data_dir / "AllCombined.txt"]
        else:
            files = []
            for subdir in data_dir.iterdir():
                if subdir.is_dir() and len(subdir.name) == 2 and subdir.name.isupper():
                    files.extend(sorted(subdir.glob("wiki_*")))

        if not files:
            logger.error(f"No files found in {data_dir}")
            return 1

        logger.info(f"Found {len(files)} files to process")

        # Use streaming training for maximum performance (1000+ articles/second)
        stats = train_with_streaming(
            agent, files, checkpoint_path, knowledge_path, logger, max_workers=workers, max_articles=args.max_articles
        )

        # Final stats
        logger.info("Training completed")
        logger.info(f"Articles processed: {stats['articles_processed']:,}")
        logger.info(f"Articles skipped: {stats['articles_skipped']:,}")
        logger.info(f"Sentences processed: {stats['sentences_processed']:,}")
        logger.info(f"Batches processed: {stats['batches_processed']:,}")
        logger.info(f"Unique states visited: {stats['unique_states_count']}")

        # Verify knowledge file
        if knowledge_path.exists():
            size = knowledge_path.stat().st_size
            # entry_count = get_knowledge_entry_count(knowledge_path)  # REMOVE THIS - too expensive

            logger.info(f"Final knowledge file: {format_size(size)}")
            print("\n📊 Final Results:")
            print(f"  • Knowledge file: {format_size(size)}")
            print(f"  • Articles processed: {stats['articles_processed']:,}")
            print(f"  • Unique states: {stats['unique_states_count']}")

            # Success indicator
            if size > 1024 * 1024:  # 1MB threshold instead of entry count
                print("  ✅ Knowledge file grew successfully!")
            else:
                print(f"  ⚠️  Warning: Only {format_size(size)} for {stats['articles_processed']:,} articles!")
        else:
            logger.error("Knowledge file was not created!")
            print("\n❌ Knowledge file was not created!")

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1

    finally:
        # Graceful shutdown
        if agent:
            try:
                logger.info("Closing agent...")
                agent.close()
                logger.info("Agent closed successfully")
            except Exception as e:
                logger.error(f"Error closing agent: {e}")

    if shutdown_requested:
        logger.info("Training interrupted by user")
        print("\n🛑 Training interrupted by user")
    else:
        logger.info("Training completed successfully")
        print("\n✅ Training completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
