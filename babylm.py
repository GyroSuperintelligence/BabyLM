#!/usr/bin/env python3
"""
baby_lm.py - Main script for GyroSI Baby LM

This script provides a command-line interface to interact with the GyroSI Baby LM,
supporting input processing, response generation, and thread management.
"""

import argparse
import os
import sys
import json
import logging
from typing import List, Optional
from baby import initialize_intelligence_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("baby_lm.log")],
)
logger = logging.getLogger("baby_lm")


def main():
    """Main entry point for GyroSI Baby LM"""
    parser = argparse.ArgumentParser(description="GyroSI Baby LM")

    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--input", type=str, help="Input text")
    input_group.add_argument("--input-file", type=str, help="Input file path")

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-file", type=str, help="Output file path")

    # Generation options
    generation_group = parser.add_argument_group("Generation")
    generation_group.add_argument("--generate", action="store_true", help="Generate text")
    generation_group.add_argument("--length", type=int, default=100, help="Generation length")

    # Thread management
    thread_group = parser.add_argument_group("Thread Management")
    thread_group.add_argument("--list-threads", action="store_true", help="List all threads")
    thread_group.add_argument("--load-thread", type=str, help="Load thread by UUID")

    # Format management
    format_group = parser.add_argument_group("Format Management")
    format_group.add_argument("--discover-format", type=str, metavar="DOMAIN", help="Discover format for domain")
    format_group.add_argument("--compose-formats", type=str, nargs="+", metavar="UUID", help="Compose multiple formats")

    # Debugging
    debug_group = parser.add_argument_group("Debugging")
    debug_group.add_argument("--info", action="store_true", help="Show system information")
    debug_group.add_argument("--version", action="store_true", help="Show version information")

    args = parser.parse_args()

    # Show version information
    if args.version:
        from baby import __version__

        print(f"GyroSI Baby LM version {__version__}")
        return

    # Initialize the intelligence engine
    try:
        engine = initialize_intelligence_engine()
        logger.info("Intelligence engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize intelligence engine: {e}")
        return

    # Show system information
    if args.info:
        show_system_info(engine)
        return

    # List threads
    if args.list_threads:
        list_threads(engine)
        return

    # Load thread
    if args.load_thread:
        load_thread(engine, args.load_thread, args.output_file)
        return

    # Discover format
    if args.discover_format:
        discover_format(engine, args.discover_format)
        return

    # Compose formats
    if args.compose_formats:
        compose_formats(engine, args.compose_formats)
        return

    # Process input
    if args.input or args.input_file:
        process_input(engine, args.input, args.input_file, args.output_file)
        return

    # Generate response
    if args.generate:
        generate_response(engine, args.length, args.output_file)
        return

    # If no arguments provided, show help
    parser.print_help()


def show_system_info(engine):
    """Display system information"""
    print("GyroSI Baby LM System Information")
    print("---------------------------------")
    print(f"Agent UUID: {engine.agent_uuid}")
    print(f"Format UUID: {engine.format_uuid}")
    print(f"Thread count: {len(engine.memory_prefs['uuid_registry']['thread_uuids'])}")
    print(f"Epigenome tensor shape: {engine.inference_engine.T.shape}")
    print(f"Cycle counter: {engine.inference_engine.cycle_counter}")
    print(f"Pattern count: {len(engine.M['patterns'])}")

    # Count patterns with semantic labels
    labeled_patterns = sum(1 for p in engine.M["patterns"] if p.get("semantic") is not None)
    print(f"Labeled patterns: {labeled_patterns}")

    # Format metadata
    print("\nFormat Information:")
    print(f"  Name: {engine.M['format_name']}")
    print(f"  Version: {engine.M['format_version']}")
    print(f"  Stability: {engine.M['stability']}")
    print(f"  Usage count: {engine.M['metadata']['usage_count']}")
    print(f"  Last updated: {engine.M['metadata']['last_updated']}")

    # Storage information
    print("\nStorage Information:")
    memory_dir = "memories"
    if os.path.exists(memory_dir):
        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(memory_dir)
            for filename in filenames
        )
        print(f"  Total storage used: {total_size / 1024:.2f} KB")


def list_threads(engine):
    """List all threads"""
    print("Thread List")
    print("-----------")

    thread_uuids = engine.memory_prefs["uuid_registry"]["thread_uuids"]
    if not thread_uuids:
        print("No threads found.")
        return

    for i, thread_uuid in enumerate(thread_uuids):
        # Try to load thread data to get size
        shard = str(thread_uuid)[: engine.memory_prefs["storage_config"]["shard_prefix_length"]]
        thread_path = f"memories/private/{engine.agent_uuid}/threads/{shard}/thread-{thread_uuid}.enc"

        try:
            size = os.path.getsize(thread_path)
            size_str = f"{size} bytes"

            # Try to get creation time
            created = os.path.getctime(thread_path)
            import datetime

            created_str = datetime.datetime.fromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")
        except FileNotFoundError:
            size_str = "Not found"
            created_str = "Unknown"

        print(f"{i+1}. {thread_uuid} - {size_str} - Created: {created_str}")


def load_thread(engine, thread_uuid, output_file):
    """Load and display a thread"""
    logger.info(f"Loading thread {thread_uuid}")
    content = engine.load_thread(thread_uuid)

    if content is None:
        print(f"Error: Thread {thread_uuid} not found or could not be loaded")
        logger.error(f"Failed to load thread {thread_uuid}")
        return

    logger.info(f"Thread {thread_uuid} loaded successfully ({len(content)} bytes)")

    # Try to decode as UTF-8 if possible
    try:
        text_content = content.decode("utf-8")
        print(f"Thread {thread_uuid} content ({len(content)} bytes):")
        print("-----")
        print(text_content)
        print("-----")
    except UnicodeDecodeError:
        print(f"Thread {thread_uuid} contains binary data ({len(content)} bytes)")

    # Save to output file if specified
    if output_file:
        with open(output_file, "wb") as f:
            f.write(content)
        print(f"Saved thread content to {output_file}")
        logger.info(f"Thread content saved to {output_file}")


def discover_format(engine, domain):
    """Discover format for a specific domain"""
    print(f"Discovering format for domain: {domain}")

    # Try all stability levels
    for stability in ["stable", "beta", "experimental"]:
        format_uuid = engine.select_stable_format(domain, stability)
        if format_uuid:
            print(f"Found {stability} format: {format_uuid}")
            return

    print(f"No format found for domain: {domain}")


def compose_formats(engine, format_uuids):
    """Compose multiple formats"""
    if len(format_uuids) < 2:
        print("Error: At least two format UUIDs are required for composition")
        return

    primary = format_uuids[0]
    secondary = format_uuids[1:]

    print(f"Composing formats: {primary} (primary) with {', '.join(secondary)} (secondary)")

    composed_uuid = engine.compose_formats(primary, secondary)

    if composed_uuid:
        print(f"Successfully created composed format: {composed_uuid}")
    else:
        print("Error: Failed to compose formats")


def process_input(engine, input_text, input_file, output_file):
    """Process input text or file"""
    # Get input data
    if input_text:
        data = input_text.encode("utf-8")
        source = "command line"
    elif input_file:
        try:
            with open(input_file, "rb") as f:
                data = f.read()
            source = input_file
        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found")
            logger.error(f"Input file not found: {input_file}")
            return
    else:
        return

    print(f"Processing {len(data)} bytes of input from {source}...")
    logger.info(f"Processing {len(data)} bytes from {source}")

    # Process the input
    plaintext, encrypted = engine.process_input_stream(data)

    print(f"Input processed and saved to thread {engine.thread_uuid}")
    logger.info(f"Input processed and saved to thread {engine.thread_uuid}")

    # Save to output file if specified
    if output_file:
        with open(output_file, "wb") as f:
            f.write(encrypted)
        print(f"Saved encrypted output to {output_file}")
        logger.info(f"Encrypted output saved to {output_file}")


def generate_response(engine, length, output_file):
    """Generate a response"""
    print(f"Generating {length} bytes of response...")
    logger.info(f"Generating response of length {length}")

    # Generate the response
    response = engine.generate_and_save_response(length)

    print(f"Response generated and saved to thread {engine.thread_uuid}")
    logger.info(f"Response generated and saved to thread {engine.thread_uuid}")

    # Try to decode as UTF-8 if possible
    try:
        text_response = response.decode("utf-8")
        print("Generated response:")
        print("-----")
        print(text_response)
        print("-----")
    except UnicodeDecodeError:
        print(f"Generated binary response ({len(response)} bytes)")

    # Save to output file if specified
    if output_file:
        with open(output_file, "wb") as f:
            f.write(response)
        print(f"Saved generated response to {output_file}")
        logger.info(f"Generated response saved to {output_file}")


if __name__ == "__main__":
    main()
