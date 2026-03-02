"""
Data loader script for test parking data.

This script loads static content into Weaviate and dynamic data into PostgreSQL
for testing and development purposes.

⚠️ MVP only!

Usage:
    # Load all test data (idempotent)
    python -m src.data.loader

    # Reset and reload all data
    python -m src.data.loader --reset

    # Load specific parking facility
    python -m src.data.loader --parking-id downtown_plaza

    # Verbose logging
    python -m src.data.loader --verbose
"""

import argparse
import csv
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5
from typing import Dict, List

from .chunker import chunk_text_smart, prepare_chunk_for_insertion
from .sql_store import SQLStore
from .vector_store import WeaviateStore
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from pydantic import SecretStr
from src.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Main data loader for parking test data.

    Handles loading static markdown content into Weaviate and dynamic CSV data
    into PostgreSQL with idempotent operations.

    ⚠️ MVP only
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize data loader.

        Args:
            verbose: Enable verbose logging
        """
        if verbose:
            logger.setLevel(logging.DEBUG)

        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.static_dir = self.data_dir / "static"
        self.dynamic_dir = self.data_dir / "dynamic"

    def load_static_data(self, parking_ids: List[str], reset: bool = False) -> int:
        """
        Load static markdown content into Weaviate.

        Args:
            parking_ids: List of parking facility IDs to load
            reset: If True, delete and recreate collection

        Returns:
            Total number of chunks inserted

        Raises:
            FileNotFoundError: If markdown files are missing
            Exception: If Weaviate operations fail
        """
        logger.info("=" * 70)
        logger.info("LOADING STATIC DATA INTO WEAVIATE")
        logger.info("=" * 70)

        with WeaviateStore() as store:
            # Reset if requested
            if reset:
                logger.info("Resetting Weaviate collection...")
                store.delete_collection()
                logger.info("Collection deleted")

            # Create collection if not exists
            if not store.collection_exists():
                logger.info("Creating ParkingContent collection...")
                store.create_collection()
                logger.info("Collection created successfully")
            else:
                logger.info("ParkingContent collection already exists")

            vector_store = WeaviateVectorStore(
                client=store.client,
                index_name=store.collection_name,
                text_key="content",
                embedding=OpenAIEmbeddings(api_key=SecretStr(settings.openai_api_key)),
                attributes=[
                    "parking_id",
                    "content_type",
                    "source_file",
                    "chunk_index",
                    "metadata",
                ],
            )

            total_chunks = 0

            for parking_id in parking_ids:
                logger.info(f"\nProcessing parking facility: {parking_id}")

                # Delete existing data for this parking_id (for idempotency)
                if not reset:
                    deleted = store.delete_by_parking_id(parking_id)
                    if deleted > 0:
                        logger.info(f"  Deleted {deleted} existing chunks for {parking_id}")

                # Find all markdown files for this parking facility
                parking_static_dir = self.static_dir / parking_id
                if not parking_static_dir.exists():
                    logger.warning(f"  Directory not found: {parking_static_dir}")
                    continue

                markdown_files = list(parking_static_dir.glob("*.md"))
                if not markdown_files:
                    logger.warning(f"  No markdown files found in {parking_static_dir}")
                    continue

                logger.info(f"  Found {len(markdown_files)} markdown files")

                # Process each markdown file
                all_chunks = []

                for md_file in sorted(markdown_files):
                    logger.debug(f"  Reading {md_file.name}...")

                    # Read file content
                    content = md_file.read_text(encoding="utf-8")

                    # Chunk the content
                    chunks = chunk_text_smart(content, md_file.name, max_tokens=500, prefer_headings=True)

                    logger.debug(f"    Created {len(chunks)} chunks from {md_file.name}")

                    # Prepare chunks for insertion
                    for chunk_text, chunk_idx in chunks:
                        chunk_dict = prepare_chunk_for_insertion(
                            parking_id=parking_id,
                            source_file=md_file.name,
                            chunk_text=chunk_text,
                            chunk_index=chunk_idx,
                        )
                        all_chunks.append(chunk_dict)

                if not all_chunks:
                    logger.warning(f"  No chunks created for {parking_id}")
                    continue

                # Insert into Weaviate using deterministic IDs for idempotent upserts.
                logger.info(f"  Inserting {len(all_chunks)} chunks into Weaviate...")
                chunk_texts = [chunk["content"] for chunk in all_chunks]
                metadatas = [
                    {
                        "parking_id": chunk["parking_id"],
                        "content_type": chunk["content_type"],
                        "source_file": chunk["source_file"],
                        "chunk_index": chunk["chunk_index"],
                        "metadata": chunk.get("metadata", {}),
                    }
                    for chunk in all_chunks
                ]
                ids = [
                    self._build_chunk_id(
                        parking_id=chunk["parking_id"],
                        source_file=chunk["source_file"],
                        chunk_index=chunk["chunk_index"],
                    )
                    for chunk in all_chunks
                ]
                inserted_ids = vector_store.add_texts(
                    texts=chunk_texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                inserted = len(inserted_ids)
                logger.info(f"  ✓ Inserted {inserted} chunks for {parking_id}")

                total_chunks += inserted

            # Verify final count
            final_count = store.count_objects()
            logger.info(f"\n{'=' * 70}")
            logger.info("STATIC DATA LOAD COMPLETE")
            logger.info(f"Total chunks in Weaviate: {final_count}")
            logger.info(f"{'=' * 70}\n")

            return total_chunks

    @staticmethod
    def _build_chunk_id(parking_id: str, source_file: str, chunk_index: int) -> str:
        """Build deterministic UUID from chunk identity fields."""
        stable_key = f"{parking_id}::{source_file}::{chunk_index}"
        return str(uuid5(NAMESPACE_URL, stable_key))

    def load_dynamic_data(self, parking_ids: List[str], reset: bool = False) -> Dict[str, int]:
        """
        Load dynamic CSV data into PostgreSQL.

        Args:
            parking_ids: List of parking facility IDs to load
            reset: If True, drop and recreate all tables

        Returns:
            Dictionary with counts of inserted records per table

        Raises:
            FileNotFoundError: If CSV files are missing
            Exception: If database operations fail
        """
        logger.info("=" * 70)
        logger.info("LOADING DYNAMIC DATA INTO POSTGRESQL")
        logger.info("=" * 70)

        store = SQLStore()

        # Reset if requested
        if reset:
            logger.info("Resetting PostgreSQL tables...")
            store.drop_tables()
            logger.info("Tables dropped")

        # Create tables if not exist
        logger.info("Creating tables (if not exist)...")
        store.create_tables()
        logger.info("Tables ready")

        counts = {"facilities": 0, "working_hours": 0, "special_hours": 0, "pricing_rules": 0, "availability": 0}

        for parking_id in parking_ids:
            logger.info(f"\nProcessing parking facility: {parking_id}")

            parking_dynamic_dir = self.dynamic_dir / parking_id
            if not parking_dynamic_dir.exists():
                logger.warning(f"  Directory not found: {parking_dynamic_dir}")
                continue

            try:
                # Load facilities
                facilities_file = parking_dynamic_dir / "facilities.csv"
                if facilities_file.exists():
                    count = self._load_facilities(store, facilities_file)
                    counts["facilities"] += count
                    logger.info(f"  ✓ Loaded {count} facility record(s)")
                else:
                    logger.warning(f"  Missing: {facilities_file.name}")

                # Load working hours
                working_hours_file = parking_dynamic_dir / "working_hours.csv"
                if working_hours_file.exists():
                    count = self._load_working_hours(store, parking_id, working_hours_file)
                    counts["working_hours"] += count
                    logger.info(f"  ✓ Loaded {count} working hours record(s)")
                else:
                    logger.warning(f"  Missing: {working_hours_file.name}")

                # Load special hours
                special_hours_file = parking_dynamic_dir / "special_hours.csv"
                if special_hours_file.exists():
                    count = self._load_special_hours(store, parking_id, special_hours_file)
                    counts["special_hours"] += count
                    logger.info(f"  ✓ Loaded {count} special hours record(s)")
                else:
                    logger.warning(f"  Missing: {special_hours_file.name}")

                # Load pricing rules
                pricing_rules_file = parking_dynamic_dir / "pricing_rules.csv"
                if pricing_rules_file.exists():
                    count = self._load_pricing_rules(store, parking_id, pricing_rules_file)
                    counts["pricing_rules"] += count
                    logger.info(f"  ✓ Loaded {count} pricing rule(s)")
                else:
                    logger.warning(f"  Missing: {pricing_rules_file.name}")

                # Load availability
                availability_file = parking_dynamic_dir / "availability.csv"
                if availability_file.exists():
                    count = self._load_availability(store, parking_id, availability_file)
                    counts["availability"] += count
                    logger.info(f"  ✓ Loaded {count} availability record(s)")
                else:
                    logger.warning(f"  Missing: {availability_file.name}")

            except Exception as e:
                logger.error(f"  ✗ Error loading data for {parking_id}: {e}")
                raise

        logger.info(f"\n{'=' * 70}")
        logger.info("DYNAMIC DATA LOAD COMPLETE")
        for table, count in counts.items():
            logger.info(f"  {table}: {count} records")
        logger.info(f"{'=' * 70}\n")

        return counts

    def _load_facilities(self, store: SQLStore, csv_file: Path) -> int:
        """Load facilities from CSV."""
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                store.upsert_facility(
                    parking_id=row["parking_id"],
                    name=row["name"],
                    address=row["address"],
                    city=row["city"],
                    total_spaces=int(row["total_spaces"]),
                    latitude=float(row["latitude"]) if row.get("latitude") else None,
                    longitude=float(row["longitude"]) if row.get("longitude") else None,
                )
                count += 1
        return count

    def _load_working_hours(self, store: SQLStore, parking_id: str, csv_file: Path) -> int:
        """Load working hours from CSV."""
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                store.upsert_working_hours(
                    parking_id=parking_id,
                    day_of_week=int(row["day_of_week"]),
                    open_time=datetime.strptime(row["open_time"], "%H:%M:%S").time(),
                    close_time=datetime.strptime(row["close_time"], "%H:%M:%S").time(),
                    is_closed=row["is_closed"].lower() == "true",
                )
                count += 1
        return count

    def _load_special_hours(self, store: SQLStore, parking_id: str, csv_file: Path) -> int:
        """Load special hours from CSV."""
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                open_time = None
                close_time = None

                if row.get("open_time"):
                    open_time = datetime.strptime(row["open_time"], "%H:%M:%S").time()
                if row.get("close_time"):
                    close_time = datetime.strptime(row["close_time"], "%H:%M:%S").time()

                store.upsert_special_hours(
                    parking_id=parking_id,
                    date_val=datetime.strptime(row["date"], "%Y-%m-%d").date(),
                    open_time=open_time,
                    close_time=close_time,
                    is_closed=row["is_closed"].lower() == "true",
                    reason=row.get("reason"),
                )
                count += 1
        return count

    def _load_pricing_rules(self, store: SQLStore, parking_id: str, csv_file: Path) -> int:
        """Load pricing rules from CSV."""
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                # Helper to safely parse optional fields (empty string becomes None)
                def safe_int(val):
                    return int(val) if val and val.strip() else None

                def safe_time(val):
                    return datetime.strptime(val, "%H:%M:%S").time() if val and val.strip() else None

                # Parse fields
                min_duration = safe_int(row.get("min_duration_minutes"))
                max_duration = safe_int(row.get("max_duration_minutes"))
                dow_start = safe_int(row.get("day_of_week_start"))
                dow_end = safe_int(row.get("day_of_week_end"))
                time_start = safe_time(row.get("time_start"))
                time_end = safe_time(row.get("time_end"))
                priority = safe_int(row.get("priority", "0")) or 0  # Default to 0 if empty
                is_active = (row.get("is_active", "true").strip().lower() == "true")

                store.upsert_pricing_rule(
                    parking_id=parking_id,
                    rule_name=row["rule_name"],
                    time_unit=row["time_unit"],
                    price_per_unit=Decimal(row["price_per_unit"]),
                    priority=priority,
                    is_active=is_active,
                    min_duration_minutes=min_duration,
                    max_duration_minutes=max_duration,
                    day_of_week_start=dow_start,
                    day_of_week_end=dow_end,
                    time_start=time_start,
                    time_end=time_end,
                )
                count += 1
        return count

    def _load_availability(self, store: SQLStore, parking_id: str, csv_file: Path) -> int:
        """Load availability from CSV."""
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                store.upsert_availability(
                    parking_id=parking_id,
                    total_spaces=int(row["total_spaces"]),
                    occupied_spaces=int(row["occupied_spaces"]),
                )
                count += 1
        return count


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Load test parking data into Weaviate and PostgreSQL.\n\n" "⚠️  MVP only - Not production-ready.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--parking-id",
        action="append",
        dest="parking_ids",
        help="Parking facility ID to load (can be specified multiple times). " "Default: downtown_plaza and airport_parking",
    )

    parser.add_argument("--reset", action="store_true", help="Reset and recreate all collections/tables before loading")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Load only static data (Weaviate)",
    )

    parser.add_argument(
        "--dynamic-only",
        action="store_true",
        help="Load only dynamic data (PostgreSQL)",
    )

    args = parser.parse_args()

    # Default parking IDs if none specified
    parking_ids = args.parking_ids or ["downtown_plaza", "airport_parking"]

    # Print warning banner
    print("\n" + "=" * 70)
    print("⚠️  TEST DATA LOADER - FOR DEVELOPMENT/TESTING ONLY")
    print("=" * 70)
    print(f"Parking facilities: {', '.join(parking_ids)}")
    print(f"Reset mode: {'YES' if args.reset else 'NO'}")
    print("=" * 70 + "\n")

    if args.reset:
        confirm = input("⚠️  Reset will DELETE ALL EXISTING DATA. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    # Initialize loader
    loader = DataLoader(verbose=args.verbose)

    try:
        # Load static data
        if not args.dynamic_only:
            loader.load_static_data(parking_ids, reset=args.reset)

        # Load dynamic data
        if not args.static_only:
            loader.load_dynamic_data(parking_ids, reset=args.reset)

        print("✓ Data loading completed successfully!\n")

    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n✗ Data loading failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
