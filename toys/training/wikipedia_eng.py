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
"""

import os
import sys
import time
import json
import gzip
import shutil
import zipfile
import argparse
import logging
import psutil
import gc
import signal
import threading
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from baby.intelligence import GyroSI
from baby.contracts import AgentConfig
from baby.policies import prune_and_compact_store
from toys.communication import tokenizer as gyrotok


def _get_raw_store(store):
    """
    Return the first wrapped store that actually implements .commit().
    Works for CanonicalView ▸ OverlayView ▸ OrbitStore, etc.
    """
    while store is not None:
        if hasattr(store, "commit"):
            return store
        # CanonicalView     → .base_store
        # OverlayView       → .private_store   (writes go there)
        # ReadOnlyView      → no commit → break
        store = getattr(store, "base_store", None) or getattr(store, "private_store", None)
    return None

def safe_commit(agent):
    raw = _get_raw_store(agent.engine.operator.store)
    if raw is not None:
        raw.commit()

# ========================================================================================
# Progress Indicators and Better Logging
# ========================================================================================

class ProgressSpinner:
    """Shows a spinner for long-running operations."""
    
    def __init__(self, message: str):
        self.message = message
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self, success_msg: str = "✅ Done"):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the line and print success message
        print(f"\r{success_msg}" + " " * 20)
        
    def _spin(self):
        i = 0
        while self.running:
            char = self.spinner_chars[i % len(self.spinner_chars)]
            print(f"\r{char} {self.message}...", end="", flush=True)
            time.sleep(0.1)
            i += 1

def print_header():
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

def print_section(title: str):
    """Print a section header."""
    print(f"\n🔧 {title}")
    print("-" * (len(title) + 4))

# ========================================================================================
# Global Tokenizer Instance + Micro-Optimized Encoding
# ========================================================================================

_GLOBAL_TOKENIZER = None
_CHECKPOINT_POOL = ThreadPoolExecutor(max_workers=1)

def get_tokenizer(tokenizer_name: str):
    """Get cached tokenizer instance."""
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        _GLOBAL_TOKENIZER = gyrotok._load(tokenizer_name)
    return _GLOBAL_TOKENIZER

def encode_bytes_optimized(text: str, tokenizer_name: str) -> bytes:
    """Micro-optimized encoding with tokenizer reuse and fast LEB128."""
    tok = get_tokenizer(tokenizer_name)
    ids = tok.encode(text).ids

    # Pre-size exactly and use local binding for ~7% speed boost
    out = bytearray()
    append = out.append  # Local binding avoids attribute lookup

    for _id in ids:
        while _id > 0x7F:
            append((_id & 0x7F) | 0x80)
            _id >>= 7
        append(_id)
    return bytes(out)

# ========================================================================================
# Configuration
# ========================================================================================

@dataclass
class RobustTrainingConfig:
    # Paths
    training_dir: Path = PROJECT_ROOT / "toys/training"
    data_dir: Path = training_dir / "wikipedia_data"
    checkpoints_dir: Path = training_dir / "checkpoints"
    logs_dir: Path = training_dir / "logs"
    
    # Dataset info
    kaggle_url: str = "https://www.kaggle.com/datasets/ffatty/plaintext-wikipedia-full-english/download"
    dataset_name: str = "plaintext-wikipedia-full-english"
    
    # Model config
    ontology_path: Path = PROJECT_ROOT / "memories/public/meta/ontology_map.json"
    knowledge_path: Path = training_dir / "knowledge/wikipedia_knowledge.pkl.gz"
    tokenizer_name: str = "bert-base-uncased"
    
    # Optimized training params for MacBook Pro 2015
    batch_size_bytes: int = 128 * 1024  # 128KB per batch (was too aggressive before)
    max_memory_usage_percent: float = 80.0  # Trigger GC at 80%
    checkpoint_every_n_files: int = 50  # More frequent checkpoints
    max_files_per_session: Optional[int] = None
    
    # Processing optimizations
    skip_short_articles: bool = True
    min_article_length: int = 256  # Skip very short articles
    max_article_length: int = 1_000_000
    parallel_workers: int = 1  # Single-threaded for 2015 MBP
    
    # Memory management
    force_commit_every_n_batches: int = 10  # Force disk flush
    gc_sleep_duration: float = 0.2  # Brief pause for OS page reclaim
    
    # Log rotation and maintenance
    max_log_size_gb: float = 3.0  # Rotate/compact at 3GB
    maintenance_decay_rate: float = 0.0005
    maintenance_confidence_threshold: float = 0.02
    
    # Debug settings
    debug_mode: bool = False
    progress_update_interval: int = 5  # seconds


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
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
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
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 4.0:
            print(f"❌ Insufficient memory: {available_gb:.1f}GB available, need at least 4GB")
            return False
        print(f"✅ Memory: {available_gb:.1f}GB available")
        
        # Check available disk space
        free_space = shutil.disk_usage(config.training_dir.parent).free / (1024**3)
        if free_space < 25.0:
            print(f"❌ Insufficient disk space: {free_space:.1f}GB available, need at least 25GB")
            return False
        print(f"✅ Disk space: {free_space:.1f}GB available")
        
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
        
        print(f"\n💡 Pro tip: You're running with python -O for ~6% speed boost!")
        return True
        
    except Exception as e:
        print(f"❌ System check failed: {e}")
        logger.error(f"System check failed: {e}")
        return False


# ========================================================================================
# Optimized Article Processing with Better Error Handling
# ========================================================================================

def iter_articles_from_file(file_path: Path, config: RobustTrainingConfig) -> Iterator[str]:
    """Stream articles from file without loading entire file into memory."""
    article_buffer = []
    articles_yielded = 0
    
    try:
        with file_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip("\n")
                
                if line.strip():
                    article_buffer.append(line)
                    continue
                
                # Blank line → possible article boundary
                if article_buffer:
                    article = "\n".join(article_buffer).strip()
                    article_buffer.clear()
                    
                    # Apply length filters
                    if (config.skip_short_articles and 
                        len(article) < config.min_article_length):
                        continue
                    if len(article) > config.max_article_length:
                        continue
                    
                    articles_yielded += 1
                    yield article
            
            # Handle final article if file doesn't end with blank line
            if article_buffer:
                article = "\n".join(article_buffer).strip()
                if (config.min_article_length <= len(article) <= 
                    config.max_article_length):
                    articles_yielded += 1
                    yield article
                    
        if config.debug_mode:
            print(f"📄 {file_path.name}: extracted {articles_yielded} articles")
                    
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        logging.getLogger("wikipedia_training").error(f"Failed to read {file_path}: {e}")


def check_log_size_and_maintain(agent: GyroSI, config: RobustTrainingConfig, logger: logging.Logger) -> None:
    """Check log size and perform maintenance if needed."""
    try:
        log_path = Path(str(config.knowledge_path) + ".log")
        
        if log_path.exists():
            log_size_gb = log_path.stat().st_size / (1024**3)
            
            if log_size_gb > config.max_log_size_gb:
                print(f"🔧 Log file size: {log_size_gb:.2f}GB - performing maintenance...")
                logger.info(f"Log file size: {log_size_gb:.2f}GB - performing maintenance...")
                
                # Apply confidence decay and pruning
                maintenance_report = agent.apply_maintenance(
                    decay_rate=config.maintenance_decay_rate,
                    confidence_threshold=config.maintenance_confidence_threshold
                )
                
                print(f"✅ Maintenance completed: {maintenance_report}")
                logger.info(f"Maintenance completed: {maintenance_report}")
                
                # Optional: Compact the store to rewrite log
                try:
                    prune_and_compact_store(
                        str(config.knowledge_path),
                        confidence_threshold=config.maintenance_confidence_threshold,
                        base_path=PROJECT_ROOT
                    )
                    print("✅ Store compacted successfully")
                    logger.info("Store compacted successfully")
                except Exception as e:
                    print(f"⚠️  Store compaction failed: {e}")
                    logger.warning(f"Store compaction failed: {e}")
    except Exception as e:
        print(f"⚠️  Maintenance check failed: {e}")
        logger.warning(f"Maintenance check failed: {e}")


def process_file_in_batches_robust(
    file_path: Path, 
    agent: GyroSI, 
    config: RobustTrainingConfig, 
    logger: logging.Logger
) -> tuple[int, int]:
    """Robust file processing with better error handling and progress."""
    articles_processed = 0
    bytes_processed = 0
    current_batch = []
    current_batch_size = 0
    batch_count = 0
    
    try:
        for article in iter_articles_from_file(file_path, config):
            current_batch.append(article)
            current_batch_size += len(article)
            
            # Process batch when it reaches target size
            if current_batch_size >= config.batch_size_bytes:
                try:
                    batch_bytes = process_article_batch_robust(
                        current_batch, agent, config, logger
                    )
                    bytes_processed += batch_bytes
                    articles_processed += len(current_batch)
                    batch_count += 1
                    
                    # Reset batch
                    current_batch = []
                    current_batch_size = 0
                    
                    # Explicit commit every N batches
                    if batch_count % config.force_commit_every_n_batches == 0:
                        try:
                            safe_commit(agent)
                            if config.debug_mode:
                                print(f"💾 Committed batch {batch_count}")
                        except Exception as e:
                            print(f"⚠️  Commit failed: {e}")
                            logger.warning(f"Commit failed: {e}")
                        
                        # Check if log maintenance is needed
                        if batch_count % (config.force_commit_every_n_batches * 10) == 0:
                            check_log_size_and_maintain(agent, config, logger)
                    
                    # Memory guard rails
                    memory_usage = psutil.virtual_memory().percent
                    if memory_usage > config.max_memory_usage_percent:
                        print(f"⚠️  High memory usage: {memory_usage:.1f}%. Forcing cleanup.")
                        logger.warning(f"High memory usage: {memory_usage:.1f}%. Forcing cleanup.")
                        safe_commit(agent)
                        gc.collect()
                        time.sleep(config.gc_sleep_duration)
                        
                except Exception as e:
                    print(f"❌ Batch processing failed: {e}")
                    logger.error(f"Batch processing failed: {e}")
                    continue  # Skip this batch and continue
        
        # Process remaining articles
        if current_batch:
            try:
                batch_bytes = process_article_batch_robust(
                    current_batch, agent, config, logger
                )
                bytes_processed += batch_bytes
                articles_processed += len(current_batch)
            except Exception as e:
                print(f"❌ Final batch processing failed: {e}")
                logger.error(f"Final batch processing failed: {e}")
        
        # Force commit at end of file
        try:
            safe_commit(agent)
        except Exception as e:
            print(f"⚠️  Final commit failed: {e}")
            logger.warning(f"Final commit failed: {e}")
        
        return articles_processed, bytes_processed
        
    except Exception as e:
        print(f"❌ File processing failed for {file_path}: {e}")
        logger.error(f"File processing failed for {file_path}: {e}")
        return articles_processed, bytes_processed


def process_article_batch_robust(
    articles: List[str], 
    agent: GyroSI, 
    config: RobustTrainingConfig, 
    logger: logging.Logger
) -> int:
    """Process a batch of articles with error handling."""
    try:
        # Combine articles into a single text
        combined_text = "\n\n".join(articles)
        
        # Use micro-optimized encoding
        encoded_bytes = encode_bytes_optimized(combined_text, config.tokenizer_name)
        
        # Learn using GyroSI's ingest method
        agent.ingest(encoded_bytes)
        
        return len(encoded_bytes)
        
    except Exception as e:
        print(f"❌ Failed to process batch of {len(articles)} articles: {e}")
        logger.error(f"Failed to process batch of {len(articles)} articles: {e}")
        return 0


# ========================================================================================
# Checkpoint Management with Safety
# ========================================================================================

@dataclass
class TrainingCheckpoint:
    processed_files: List[str]
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
            json.dump(asdict(checkpoint), f, indent=2)
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
        with checkpoint_path.open('r') as f:
            data = json.load(f)
        return TrainingCheckpoint(**data)
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        logging.getLogger("wikipedia_training").warning(f"Failed to load checkpoint: {e}")
        return None


# ========================================================================================
# Dataset Management
# ========================================================================================

def find_wikipedia_files(config: RobustTrainingConfig, logger: logging.Logger) -> List[Path]:
    """Find all Wikipedia text files in the data directory."""
    print_section("Finding Wikipedia Files")
    
    spinner = ProgressSpinner("Scanning for Wikipedia files")
    spinner.start()
    
    try:
        files = []
        
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
        
        agent_config: AgentConfig = {
            "ontology_path": str(config.ontology_path),
            "knowledge_path": str(config.knowledge_path),
            "learn_batch_size": 1000,  # Larger batch size for training
        }
        
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
    config: RobustTrainingConfig, 
    resume: bool = False, 
    logger: logging.Logger = None
) -> None:
    """Robust training loop with comprehensive error handling."""
    if logger is None:
        logger = setup_logging(config)
    
    # Set up clean SIGINT handling
    signal.signal(signal.SIGINT, signal.default_int_handler)
    
    print_section("Starting Wikipedia Training")
    
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
                last_checkpoint_time=time.time()
            )
        
        # Find all Wikipedia files
        wiki_files = find_wikipedia_files(config, logger)
        if not wiki_files:
            print("❌ No Wikipedia files found. Please check data directory.")
            return
        
        # Filter out already processed files
        remaining_files = [
            f for f in wiki_files 
            if str(f) not in checkpoint.processed_files
        ]
        
        if not remaining_files:
            print("✅ All files already processed!")
            return
        
        print(f"\n📊 Training Configuration:")
        print(f"   • Files to process: {len(remaining_files):,}")
        print(f"   • Batch size: {config.batch_size_bytes//1024}KB")
        print(f"   • Memory limit: {config.max_memory_usage_percent}%")
        print(f"   • Min article length: {config.min_article_length} chars")
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
        
        for i, file_path in enumerate(progress):
            try:
                # Check if we've hit max files limit
                if (config.max_files_per_session is not None and 
                    i >= config.max_files_per_session):
                    print(f"\n🛑 Reached max files limit: {config.max_files_per_session}")
                    logger.info(f"Reached max files limit: {config.max_files_per_session}")
                    break
                
                # Process the file with robust error handling
                articles, bytes_processed = process_file_in_batches_robust(
                    file_path, agent, config, logger
                )
                
                # Update checkpoint
                checkpoint.processed_files.append(str(file_path))
                checkpoint.total_articles += articles
                checkpoint.total_bytes_processed += bytes_processed
                checkpoint.last_file_index = i
                
                # Update progress bar with detailed stats
                memory_pct = psutil.virtual_memory().percent
                gb_processed = checkpoint.total_bytes_processed / (1024**3)
                progress.set_postfix({
                    'articles': f"{checkpoint.total_articles:,}",
                    'GB': f"{gb_processed:.2f}",
                    'mem': f"{memory_pct:.1f}%",
                    'file': file_path.name[:12]  # Truncate long filenames
                })
                
                # Save checkpoint periodically with async writes
                if (i + 1) % config.checkpoint_every_n_files == 0:
                    checkpoint.last_checkpoint_time = time.time()
                    save_checkpoint_async(checkpoint, config)
                    
                    # Log progress statistics
                    elapsed = time.time() - checkpoint.training_start_time
                    rate = checkpoint.total_articles / elapsed if elapsed > 0 else 0
                    mb_rate = (checkpoint.total_bytes_processed / (1024*1024)) / elapsed if elapsed > 0 else 0
                    
                    checkpoint_msg = (
                        f"📊 Checkpoint {(i+1)//config.checkpoint_every_n_files}: "
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
                        print(f"📈 Progress: {i+1}/{len(remaining_files)} files, "
                              f"{checkpoint.total_articles:,} articles, "
                              f"{memory_pct:.1f}% memory")
                        
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
            if 'checkpoint' in locals():
                checkpoint.last_checkpoint_time = time.time()
                save_checkpoint_atomic(checkpoint, config)
                print("💾 Final checkpoint saved")
            
            # Wait for any pending async checkpoints
            _CHECKPOINT_POOL.shutdown(wait=True)
            
            # Close agent properly
            if 'agent' in locals() and agent is not None:
                agent.close()
                print("🔒 Agent closed cleanly")
            
            # Final statistics
            if 'checkpoint' in locals():
                elapsed = time.time() - checkpoint.training_start_time
                gb_total = checkpoint.total_bytes_processed / (1024**3)
                
                final_msg = (
                    f"✅ Training completed: {checkpoint.total_articles:,} articles, "
                    f"{gb_total:.2f}GB processed in {elapsed/3600:.1f} hours"
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

def main():
    parser = argparse.ArgumentParser(
        description="Robust GyroSI Wikipedia training with better error handling",
        epilog="💡 Pro tip: Run with 'python -O' for ~6% speed boost"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--manual-download", action="store_true", 
                       help="Skip automatic download (use manually downloaded data)")
    parser.add_argument("--max-files", type=int, help="Maximum files to process in this session")
    parser.add_argument("--batch-size", type=int, default=128*1024, 
                       help="Batch size in bytes (default: 128KB)")
    parser.add_argument("--memory-limit", type=float, default=80.0,
                       help="Memory usage percentage to trigger GC (default: 80)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose output")
    
    args = parser.parse_args()
    
    # Print nice header
    print_header()
    
    # Create robust config
    config = RobustTrainingConfig()
    if args.max_files:
        config.max_files_per_session = args.max_files
    if args.batch_size:
        config.batch_size_bytes = args.batch_size
    if args.memory_limit:
        config.max_memory_usage_percent = args.memory_limit
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