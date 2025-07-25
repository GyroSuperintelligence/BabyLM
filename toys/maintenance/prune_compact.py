"""
Prune + compact an OrbitStore.

Usage examples:
---------------
# In-place: remove entries older than 30 days or confidence < 0.05
python toys/maintenance/prune_compact.py \
    --store memories/public/meta/knowledge.mpk \
    --max-age-days 30 --min-confidence 0.05

# Write compacted copy to a new file (safety) and keep archive summary
python toys/maintenance/prune_compact.py \
    --store memories/public/meta/knowledge.mpk \
    --output memories/public/meta/knowledge_compacted.mpk \
    --max-age-days 60 \
    --archive memories/public/meta/pruned_summary.json
"""

import argparse
import json
from baby.policies import prune_and_compact_store


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune and compact a GyroSI OrbitStore.")
    p.add_argument("--store", required=True, help="Path passed originally to OrbitStore (without .log/.idx).")
    p.add_argument("--output", help="Optional destination path; if omitted, compacts in-place.")
    p.add_argument("--max-age-days", type=float, help="Prune entries last updated more than this many days ago.")
    p.add_argument("--min-confidence", type=float, help="Prune entries with confidence below this value.")
    p.add_argument("--dry-run", action="store_true", help="Report what would be pruned without modifying data.")
    p.add_argument("--archive", help="Optional JSON file to record summary of pruned entries.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Refuse to run on pickle blobs
    for path in (args.store, args.output):
        if path and (path.endswith(".pkl") or path.endswith(".pkl.gz")):
            print(f"ERROR: This script only supports msgpack-based stores (.mpk). Refusing to run on: {path}")
            exit(1)
    report = prune_and_compact_store(
        store_path=args.store,
        output_path=args.output,
        max_age_days=args.max_age_days,
        min_confidence=args.min_confidence,
        dry_run=args.dry_run,
        archive_summary_path=args.archive,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
