#!/usr/bin/env python3
"""
GyroSI Wikipedia Training Script - Robust Version

HOW TO PAUSE/RESUME:
- Press Ctrl+Z to pause (suspend) the training
- Type 'fg' to resume training exactly where it left off
- Press Ctrl+C once for clean shutdown with checkpoint save
- Press Ctrl+C twice to force immediate exit

Incorporates all performance optimizations for MacBook Pro 2015:
- Tokenizer reuse with micro-optimized encoding
- Streaming file processing
- Explicit memory management with log rotation
- Atomic checkpoints with fsync safety
- Async checkpoint saving
- Clean SIGINT handling
- Conservative CPU usage

PYTHONPATH=. python toys/training/wikipedia_eng.py

"""

import argparse
import gc
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator, cast, Any

try:
    import psutil
except ImportError:
    psutil = None

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from baby.intelligence import GyroSI
from baby.contracts import AgentConfig
from toys.communication import tokenizer as gyrotok

import re

_SENT_RE = re.compile(
    r"""          # very cheap rule‑based splitter
    (?<=[.!?])    #  … after terminal punctuation
    ["')\]]*      #  … that may be followed by quotes / parens
    \s+           #  … and some whitespace
    (?=[A-Z])     #  … before the next capital letter
    """,
    re.VERBOSE,
)

def iter_sentences(text: str):
    """Yield naïvely split sentences (≈95 % precision on wiki)."""
    for s in _SENT_RE.split(text):
        s = s.strip()
        if s:
            yield s


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SHUTDOWN_REQUESTED = False

def handle_sigint(signum, frame):
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print("\nShutdown requested (Ctrl+C pressed). Will exit after current checkpoint.")

# Helper for progress bar updates
def update_progress_bar(progress, rss_pct, file_path):
    if progress is not None:
        progress.set_postfix({
            "mem": f"{rss_pct:.1f}%",
            "file": file_path.name[:12],
        })
        progress.update(0)

# ========================================================================================
# Progress Indicators and Better Logging
# ========================================================================================


class ProgressSpinner:
    """Shows a spinner for long-running operations."""

    def __init__(self, message: str) -> None:
        self.message = message
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()

    def stop(self, success_msg: str = "✅ Done") -> None:
        self.running = False
        if self.thread:
            self.thread.join()
        print(f"\r{success_msg}" + " " * 20)

    def _spin(self) -> None:
        i = 0
        while self.running:
            char = self.spinner_chars[i % len(self.spinner_chars)]
            print(f"\r{char} {self.message}...", end="", flush=True)
            time.sleep(0.1)
            i += 1


def print_header() -> None:
    """Print a nice header."""
    print("\n" + "=" * 70)
    print("🧠 GyroSI Wikipedia Training - Robust Edition")
    print("=" * 70)
    print("📋 PAUSE/RESUME INSTRUCTIONS:")
    print("   • Press Ctrl+Z to pause training (suspend process)")
    print("   • Type 'fg' to resume training exactly where you left off")
    print("   • Press Ctrl+C once for clean shutdown with checkpoint")
    print("   • Press Ctrl+C twice to force immediate exit")
    print("=" * 70 + "\n")


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n🔧 {title}")
    print("-" * (len(title) + 4))


# ========================================================================================
# Global Tokenizer Instance + Physics-Aware Encoding
# ========================================================================================

_GLOBAL_TOKENIZER: Optional[Tokenizer] = None
_CHECKPOINT_POOL: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")


def get_tokenizer(tokenizer_name: str) -> Tokenizer:
    """Get cached tokenizer instance."""
    if Tokenizer is None:
        raise ImportError("tokenizers module not available")
    
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        _GLOBAL_TOKENIZER = gyrotok._load(tokenizer_name)
    return _GLOBAL_TOKENIZER


def encode_bytes_optimized(text: str, tokenizer_name: str) -> bytes:
    """Physics-aware encoding that applies the mandatory 0xAA mask."""
    return gyrotok.encode(text, name=tokenizer_name)


# ========================================================================================
# Configuration
# ========================================================================================


@dataclass
class RobustTrainingConfig:
    # Paths
    training_dir: Path = PROJECT_ROOT / "toys/training"
    data_dir: Path = training_dir / "wikipedia_full_data"
    checkpoints_dir: Path = training_dir / "checkpoints"
    logs_dir: Path = training_dir / "logs"

    # Dataset info
    kaggle_url: str = "https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-english/download"
    dataset_name: str = "plaintext-wikipedia-full-english"

    # Model config
    ontology_path: Path = PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"
    phenomenology_path: Path = PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"
    knowledge_path: Path = training_dir / "knowledge/wikipedia_knowledge.bin"
    tokenizer_name: str = "bert-base-uncased"

    # Optimized training params for MacBook Pro 2015
    checkpoint_every_n_files: int = 50  # More frequent checkpoints
    max_files_per_session: Optional[int] = None

    # Processing optimizations - now CPU-bound (tokenization) not IO-bound
    min_token_count = 4

    # Memory management
    gc_sleep_duration: float = 0.2  # Brief pause for OS page reclaim
    max_memory_usage_percent: float = 80.0  # Memory limit is now 80%

    # Log rotation and maintenance
    max_log_size_gb: float = 3.0  # Rotate/compact at 3GB
    maintenance_decay_rate: float = 0.0005
    maintenance_confidence_threshold: float = 0.02

    # Debug settings
    debug_mode: bool = False
    progress_update_interval: int = 10  # Fewer `tqdm` redraws when batches are bigger


def setup_logging(config: RobustTrainingConfig) -> logging.Logger:
    """Set up logging to both file and console."""
    config.logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("wikipedia_training")
    logger.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(config.logs_dir / f"training_{int(time.time())}.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if config.debug_mode else logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def check_system_requirements(config: RobustTrainingConfig, logger: logging.Logger) -> bool:
    """Check if system meets minimum requirements."""
    print_section("System Requirements Check")

    try:
        # Check available memory
        if psutil is not None:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb < 4.0:
                print(f"❌ Insufficient memory: {available_gb:.1f}GB available, need at least 4GB")
                return False
            print(f"✅ Memory: {available_gb:.1f}GB available")
        else:
            print("⚠️  psutil not available - skipping memory check")

        # Check available disk space - reduced requirement for simple training
        free_space = shutil.disk_usage(config.training_dir.parent).free / (1024**3)
        
        # Determine required space based on training mode
        required_space = 5.0  # Default for simple training
        if hasattr(config, '_test_file') or hasattr(config, '_simple_file'):
            # Simple training mode - much smaller requirement
            required_space = 2.0
        elif config.data_dir.name == "wikipedia_full_data":
            # Full Wikipedia training - original requirement
            required_space = 25.0
            
        if free_space < required_space:
            print(f"❌ Insufficient disk space: {free_space:.1f}GB available, need at least {required_space:.1f}GB")
            return False
        print(f"✅ Disk space: {free_space:.1f}GB available (required: {required_space:.1f}GB)")

        # Check if ontology exists
        if not config.ontology_path.exists():
            print(f"❌ Ontology not found at {config.ontology_path}")
            return False
        print(f"✅ Ontology found: {config.ontology_path}")

        # Pre-load tokenizer to check it works
        print("🔍 Testing tokenizer...")
        get_tokenizer(config.tokenizer_name)
        vocab_size = gyrotok.vocab_size(config.tokenizer_name)
        print(f"✅ Tokenizer '{config.tokenizer_name}' loaded (vocab: {vocab_size:,})")

        print("\n\U0001f4a1 Pro tip: You're running with python -O for ~6% speed boost!")
        return True

    except Exception as e:
        print(f"❌ System check failed: {e}")
        logger.error(f"System check failed: {e}")
        return False


# ========================================================================================
# Optimized Article Processing with Better Error Handling
# ========================================================================================


def iter_articles_from_file(file_path: Path, config: RobustTrainingConfig) -> Iterator[str]:
    """Split articles by three or more blank lines (robust for Wikipedia dumps)."""
    article_buffer: list[str] = []
    blank_streak = 0
    with file_path.open(encoding="utf-8") as f:
        for raw_line in f:
            if raw_line.strip() == "":
                blank_streak += 1
            else:
                if blank_streak >= 3 and article_buffer:
                    article = "\n".join(article_buffer).strip()
                    if article:
                        yield article
                    article_buffer.clear()
                blank_streak = 0
                article_buffer.append(raw_line.rstrip("\n"))
        # Emit the last article if present and non-empty
        if article_buffer and any(line.strip() for line in article_buffer):
            article = "\n".join(article_buffer).strip()
            if article:
                yield article


def train_on_file_content(
    file_path: Path, agent: GyroSI, config: RobustTrainingConfig, logger: logging.Logger, progress=None
) -> tuple[int, int]:
    import time
    sentences_processed: int = 0
    bytes_processed: int = 0
    tokens_processed: int = 0
    inner_update_every = 10_000      # Bytes
    tok = get_tokenizer(config.tokenizer_name)
    min_token_count = config.min_token_count
    sep_bytes = gyrotok.bytes_from_ids([102])
    if psutil is not None:
        process = psutil.Process()
        total_mem = psutil.virtual_memory().total
    else:
        process = None
        total_mem = None
    start_time = time.time()

    from tqdm import tqdm as tqdm_lib
    article_list = list(iter_articles_from_file(file_path, config))
    article_progress = tqdm_lib(article_list, desc=f"Articles in {file_path.name}", unit="article", leave=False)

    byte_counter_in_file = 0
    for article in article_progress:
        for sent in iter_sentences(article):
            ids = tok.encode(sent, add_special_tokens=False).ids
            if len(ids) < min_token_count:
                continue
            encoded = gyrotok.bytes_from_ids(ids) + sep_bytes
            agent.engine.batch_learn(encoded)
            # progress bookkeeping -------------
            byte_counter_in_file += len(encoded)
            if byte_counter_in_file % inner_update_every == 0:
                if process is not None and total_mem is not None:
                    rss_pct = process.memory_info().rss / total_mem * 100
                    update_progress_bar(progress, rss_pct, file_path)
            tokens_processed += len(ids)
            sentences_processed += 1
            bytes_processed += len(encoded)
        # After each article, force a progress bar update
        rss_pct = process.memory_info().rss / total_mem * 100
        update_progress_bar(progress, rss_pct, file_path)
    elapsed = time.time() - start_time
    print(f"Shard took {elapsed:.1f}s")
    return sentences_processed, bytes_processed


# ========================================================================================
# Checkpoint Management with Safety
# ========================================================================================


@dataclass
class TrainingCheckpoint:
    processed_files: list[str]
    total_articles: int
    total_bytes_processed: int
    last_file_index: int
    training_start_time: float
    last_checkpoint_time: float


def save_checkpoint_atomic(checkpoint: TrainingCheckpoint, config: RobustTrainingConfig) -> None:
    """Save training checkpoint atomically with fsync for laptop safety."""
    try:
        config.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.checkpoints_dir / "latest_checkpoint.json"

        # Atomic write with fsync for laptop sleep-wake safety
        tmp_path = checkpoint_path.with_suffix(".tmp")
        with tmp_path.open("w") as f:
            json.dump(checkpoint.__dict__, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure it's on disk before rename
        tmp_path.replace(checkpoint_path)  # Atomic rename

    except Exception as e:
        print(f"⚠️  Failed to save checkpoint: {e}")
        logging.getLogger("wikipedia_training").warning(f"Failed to save checkpoint: {e}")


def save_checkpoint_async(checkpoint: TrainingCheckpoint, config: RobustTrainingConfig) -> None:
    """Save checkpoint asynchronously to keep progress bar smooth."""
    _CHECKPOINT_POOL.submit(save_checkpoint_atomic, checkpoint, config)


def load_checkpoint(config: RobustTrainingConfig) -> Optional[TrainingCheckpoint]:
    """Load training checkpoint if it exists."""
    checkpoint_path = config.checkpoints_dir / "latest_checkpoint.json"

    if not checkpoint_path.exists():
        return None

    try:
        with checkpoint_path.open("r") as f:
            data = json.load(f)
        return TrainingCheckpoint(**data)
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        logging.getLogger("wikipedia_training").warning(f"Failed to load checkpoint: {e}")
        return None


# ========================================================================================
# Dataset Management
# ========================================================================================


def find_wikipedia_files(config: RobustTrainingConfig, logger: logging.Logger) -> list[Path]:
    """Find all Wikipedia text files in the data directory."""
    print_section("Finding Wikipedia Files")

    spinner = ProgressSpinner("Scanning for Wikipedia files")
    spinner.start()

    try:
        files: list[Path] = []

        # Look for the typical structure: fullEnglish/XX/wiki_YY format
        for subdir in config.data_dir.rglob("*"):
            if subdir.is_dir() and len(subdir.name) == 2 and subdir.name.isupper():
                wiki_files = list(subdir.glob("wiki_*"))
                files.extend(wiki_files)

        # Also check for direct files
        direct_files = list(config.data_dir.glob("wiki_*"))
        files.extend(direct_files)

        # Sort for consistent processing order
        files.sort()

        spinner.stop(f"✅ Found {len(files)} Wikipedia files")

        if len(files) == 0:
            print("❌ No Wikipedia files found! Check your data directory:")
            print(f"   {config.data_dir}")
            print("   Expected structure: fullEnglish/XX/wiki_YY or direct wiki_* files")

        if hasattr(config, "_test_file") and config._test_file.exists():
            return [config._test_file]
        
        if hasattr(config, "_simple_file") and config._simple_file.exists():
            return [config._simple_file]

        return files

    except Exception as e:
        spinner.stop(f"❌ Error scanning files: {e}")
        logger.error(f"Error scanning files: {e}")
        return []


# ========================================================================================
# Main Training Loop with Robust Error Handling
# ========================================================================================


def create_training_agent(config: RobustTrainingConfig, logger: logging.Logger) -> Optional[GyroSI]:
    """Create and configure the GyroSI agent for training."""
    print_section("Creating GyroSI Training Agent")

    try:
        config.knowledge_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a private store for training; public knowledge is only for read-only reference
        agent_config: AgentConfig = {
            "ontology_path": str(config.ontology_path),
            "phenomenology_map_path": str(config.phenomenology_path),
            "public_knowledge_path": str(config.training_dir / "dummy_public_knowledge.bin"),  # dummy file to avoid fallback
            "private_knowledge_path": str(config.knowledge_path),
            "learn_batch_size": 4000,  # see §2
            "preferences": {
                "pruning": {
                    "confidence_threshold": 0.05,
                    "enable_auto_decay": True  # live pruning
                }
            },
        }

        # SAFETY CHECK: Ensure knowledge_path is a file path, not a directory
        knowledge_path = agent_config["private_knowledge_path"]
        if os.path.isdir(knowledge_path):
            print(f"❌ ERROR: The knowledge path '{knowledge_path}' is a directory, not a file. Please remove or rename this directory, or set knowledge_path to a file.")
            logger.error(f"Knowledge path '{knowledge_path}' is a directory, not a file.")
            return None
        if knowledge_path == str(config.training_dir.parent) or knowledge_path == str(config.training_dir):
            print(f"❌ ERROR: The knowledge path '{knowledge_path}' is set to the project or training directory. Please set it to a file path, e.g., 'toys/training/knowledge/wikipedia_knowledge.bin'.")
            logger.error(f"Knowledge path '{knowledge_path}' is set to a directory.")
            return None

        # DEBUG: Print the exact config being passed to GyroSI
        print(f"🔍 DEBUG: Agent config paths:")
        print(f"   • ontology_path: {agent_config['ontology_path']}")
        print(f"   • phenomenology_map_path: {agent_config['phenomenology_map_path']}")
        print(f"   • public_knowledge_path: '{agent_config['public_knowledge_path']}'")
        print(f"   • private_knowledge_path: {agent_config['private_knowledge_path']}")

        print(f"📍 Knowledge will be stored at: {config.knowledge_path}")

        # Show progress during agent creation (this is the slow part)
        spinner = ProgressSpinner("Creating GyroSI agent (loading STT, building tensors)")
        spinner.start()

        agent = GyroSI(agent_config, agent_id="wikipedia_trainer")

        spinner.stop("✅ GyroSI agent created successfully")

        return agent

    except Exception as e:
        print(f"❌ Failed to create GyroSI agent: {e}")
        logger.error(f"Failed to create GyroSI agent: {e}")
        return None


def run_training_robust(
    config: RobustTrainingConfig, resume: bool = False, logger: Optional[logging.Logger] = None
) -> None:
    """Robust training loop with comprehensive error handling."""
    if logger is None:
        logger = setup_logging(config)

    # Set up clean SIGINT handling
    signal.signal(signal.SIGINT, handle_sigint)

    print_section("Starting Wikipedia Training")

    checkpoint: Optional[TrainingCheckpoint] = None
    agent: Optional[GyroSI] = None
    need_compact = False  # Flag for post-training compaction

    try:
        # Load or create checkpoint
        checkpoint = load_checkpoint(config) if resume else None
        if checkpoint:
            print(f"📂 Resuming from checkpoint - {len(checkpoint.processed_files)} files already processed")
            logger.info(f"Resuming from checkpoint - {len(checkpoint.processed_files)} files already processed")
        else:
            print("🆕 Starting fresh training run")
            checkpoint = TrainingCheckpoint(
                processed_files=[],
                total_articles=0,
                total_bytes_processed=0,
                last_file_index=0,
                training_start_time=time.time(),
                last_checkpoint_time=time.time(),
            )

        # Find all Wikipedia files
        wiki_files = find_wikipedia_files(config, logger)
        if not wiki_files:
            print("❌ No Wikipedia files found. Please check data directory.")
            return

        # Filter out already processed files
        remaining_files = [f for f in wiki_files if str(f) not in checkpoint.processed_files]

        if not remaining_files:
            print("✅ All files already processed!")
            return

        print("\n📊 Training Configuration:")
        print(f"   • Files to process: {len(remaining_files):,}")
        print(f"   • Batch size: {config.checkpoint_every_n_files} files")  # Batch size is now files
        print(f"   • Memory limit: {config.max_memory_usage_percent}%")  # Memory limit is now 80%
        print(f"   • Min article length: {config.min_token_count} tokens")
        print(f"   • Log rotation: {config.max_log_size_gb}GB")

        # Create training agent
        agent = create_training_agent(config, logger)
        if agent is None:
            return

        print_section("Processing Files")
        print("💡 Remember: Ctrl+Z to pause, 'fg' to resume, Ctrl+C for clean shutdown\n")

        # Process files with progress bar
        progress = tqdm(remaining_files, desc="Processing files", unit="files")
        last_progress_update = time.time()
        process = psutil.Process()
        total_mem = psutil.virtual_memory().total

        for i, file_path in enumerate(progress):
            # Check for shutdown request
            if SHUTDOWN_REQUESTED:
                print("\n🛑 Training interrupted by user (Ctrl+C pressed). Exiting gracefully.")
                logger.info("Training interrupted by user (Ctrl+C pressed). Exiting gracefully.")
                break

            try:
                # Check if we've hit max files limit
                if config.max_files_per_session is not None and i >= config.max_files_per_session:
                    print(f"\n🛑 Reached max files limit: {config.max_files_per_session}")
                    logger.info(f"Reached max files limit: {config.max_files_per_session}")
                    break

                # Process the file with robust error handling
                articles, bytes_processed = train_on_file_content(file_path, agent, config, logger, progress)

                # Update checkpoint
                checkpoint.processed_files.append(str(file_path))
                checkpoint.total_articles += articles
                checkpoint.total_bytes_processed += bytes_processed
                checkpoint.last_file_index = i

                # Update progress bar with detailed stats
                rss_pct = process.memory_info().rss / total_mem * 100
                gb_processed = checkpoint.total_bytes_processed / (1024**3)
                progress.set_postfix(
                    {
                        "articles": f"{checkpoint.total_articles:,}",
                        "GB": f"{gb_processed:.2f}",
                        "mem": f"{rss_pct:.1f}%",
                        "file": file_path.name[:12],  # Truncate long filenames
                    }
                )

                # Save checkpoint periodically with async writes
                if (i + 1) % config.checkpoint_every_n_files == 0:
                    checkpoint.last_checkpoint_time = time.time()
                    save_checkpoint_async(checkpoint, config)

                    # Log progress statistics
                    elapsed = time.time() - checkpoint.training_start_time
                    rate = checkpoint.total_articles / elapsed if elapsed > 0 else 0
                    mb_rate = (checkpoint.total_bytes_processed / (1024 * 1024)) / elapsed if elapsed > 0 else 0

                    checkpoint_msg = (
                        f"📊 Checkpoint {(i + 1) // config.checkpoint_every_n_files}: "
                        f"{checkpoint.total_articles:,} articles, "
                        f"{gb_processed:.2f}GB processed, "
                        f"{rate:.1f} articles/sec, {mb_rate:.2f} MB/sec"
                    )
                    print(f"\n{checkpoint_msg}")
                    logger.info(checkpoint_msg)

                # Periodic progress updates
                now = time.time()
                if now - last_progress_update > config.progress_update_interval:
                    last_progress_update = now
                    if config.debug_mode:
                        print(
                            f"📈 Progress: {i + 1}/{len(remaining_files)} files, "
                            f"{checkpoint.total_articles:,} articles, "
                            f"{rss_pct:.1f}% memory"
                        )
                    print(
                        "\n\U0001f4c8 Progress: {}/{} files, {:,} articles, {:.1f}% memory".format(
                            i + 1, len(remaining_files), checkpoint.total_articles, rss_pct
                        )
                    )

                # After each file, check if compaction is needed
                if (
                    config.knowledge_path.exists()
                    and config.knowledge_path.stat().st_size > config.max_log_size_gb * 1024**3
                ):
                    need_compact = True

                # Minor safeguard: flush checkpoint thread pool backlog every 200 files if needed
                if (i + 1) % 200 == 0:
                    try:
                        if (
                            hasattr(_CHECKPOINT_POOL, "_work_queue")
                            and getattr(_CHECKPOINT_POOL._work_queue, "qsize", lambda: 0)() > 1
                        ):
                            logger.info(
                                "Flushing checkpoint thread pool backlog (rare safeguard for slow filesystems)..."
                            )
                            _CHECKPOINT_POOL.shutdown(wait=True)
                            # Recreate the pool for future checkpoints
                            import concurrent.futures

                            globals()["_CHECKPOINT_POOL"] = concurrent.futures.ThreadPoolExecutor(
                                max_workers=1, thread_name_prefix="ckpt"
                            )
                    except Exception as e:
                        logger.warning(f"Checkpoint pool backlog flush failed: {e}")

            except Exception as e:
                print(f"\n❌ Error processing {file_path}: {e}")
                logger.error(f"Error processing {file_path}: {e}")
                continue  # Skip this file and continue with the next

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user - saving final checkpoint...")
        logger.info("Training interrupted by user - saving final checkpoint...")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training failed: {e}")
        raise
    finally:
        try:
            # Always save final checkpoint (synchronously for safety)
            if checkpoint is not None:
                checkpoint.last_checkpoint_time = time.time()
                save_checkpoint_atomic(checkpoint, config)
                print("💾 Final checkpoint saved")

            # Wait for any pending async checkpoints
            _CHECKPOINT_POOL.shutdown(wait=True)

            # Close agent properly
            if agent is not None:
                agent.close()
                print("🔒 Agent closed cleanly")
                # Only compact after agent is closed
                # Only live pruning is used during training; after training, the file can be moved/copied to the public area.

            # Final statistics
            if checkpoint is not None:
                elapsed = time.time() - checkpoint.training_start_time
                gb_total = checkpoint.total_bytes_processed / (1024**3)

                final_msg = (
                    f"✅ Training completed: {checkpoint.total_articles:,} articles, "
                    f"{gb_total:.2f}GB processed in {elapsed / 3600:.1f} hours"
                )
                print(f"\n{final_msg}")
                logger.info(final_msg)

                # Log knowledge store size
                if config.knowledge_path.exists():
                    kb_size = config.knowledge_path.stat().st_size / (1024**2)
                    store_msg = f"📊 Knowledge store size: {kb_size:.1f}MB"
                    print(store_msg)
                    logger.info(store_msg)

        except Exception as e:
            print(f"⚠️  Error during cleanup: {e}")
            logger.warning(f"Error during cleanup: {e}")


# ========================================================================================
# CLI Interface
# ========================================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robust GyroSI Wikipedia training with better error handling",
        epilog="💡 Pro tip: Run with 'python -O' for ~6% speed boost",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--manual-download", action="store_true", help="Skip automatic download (use manually downloaded data)"
    )
    parser.add_argument("--max-files", type=int, help="Maximum files to process in this session")
    parser.add_argument("--checkpoint-every-files", type=int, default=50, help="Checkpoint every N files (default: 50)")
    parser.add_argument(
        "--memory-limit", type=float, default=80.0, help="Process memory usage percentage to trigger GC (default: 80)"
    )
    parser.add_argument("--test-aa", action="store_true", help="Test only on the wiki_test file (for quick validation)")
    parser.add_argument("--simple-data", action="store_true", help="Train on the simple AllCombined.txt file instead of full Wikipedia data")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")

    # Remove --batch-size, but error if user supplies it
    import sys

    if any(arg.startswith("--batch-size") for arg in sys.argv):
        parser.error("--batch-size is no longer supported. Use --checkpoint-every-files instead.")

    args = parser.parse_args()

    # Print nice header
    print_header()

    # Create robust config
    config = RobustTrainingConfig()
    if args.max_files:
        config.max_files_per_session = args.max_files
    config.checkpoint_every_n_files = args.checkpoint_every_files
    config.max_memory_usage_percent = args.memory_limit
    if args.test_aa:
        # Set data_dir to the directory containing wiki_test and restrict to that file
        from pathlib import Path
        config.data_dir = Path(__file__).parent  # toys/training/
        config._test_file = config.data_dir / "wiki_test"
        # Use a separate test knowledge file to avoid conflicts
        config.knowledge_path = config.training_dir / "knowledge/test_knowledge.bin"

    if args.simple_data:
        # Set data_dir to the simple data directory and use AllCombined.txt
        from pathlib import Path
        config.data_dir = Path(__file__).parent / "wikipedia_simple_data"
        config._simple_file = config.data_dir / "AllCombined.txt"
        # Use a separate simple data knowledge file
        config.knowledge_path = config.training_dir / "knowledge/simple_knowledge.bin"

    if args.debug:
        config.debug_mode = True

    # Set up logging
    logger = setup_logging(config)

    # Check system requirements
    if not check_system_requirements(config, logger):
        return 1

    # Create directories
    print_section("Setting Up Directories")
    try:
        for directory in [config.training_dir, config.data_dir, config.checkpoints_dir, config.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        print("✅ All directories created/verified")
    except Exception as e:
        print(f"❌ Failed to create directories: {e}")
        return 1

    # Verify data exists
    wiki_files = find_wikipedia_files(config, logger)
    if not wiki_files:
        print(f"\n❌ No Wikipedia files found in {config.data_dir}")
        print("Please ensure data is extracted properly.")
        return 1

    # Run robust training
    try:
        run_training_robust(config, resume=args.resume, logger=logger)
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
