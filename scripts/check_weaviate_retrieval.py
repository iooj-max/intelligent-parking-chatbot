#!/usr/bin/env python3
"""
Debug script to verify Weaviate retrieval behavior.

Run from project root:
    python scripts/check_weaviate_retrieval.py

Requires: Docker with Weaviate running, .env with OPENAI_API_KEY.
"""

import os
import sys

# Add project root to path (run from project root: python scripts/check_weaviate_retrieval.py)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from parking_agent.tools import retrieve_static_parking_info


def main():
    print("=" * 70)
    print("Weaviate retrieval check (query only, no parking_id)")
    print("=" * 70)

    for query in ["How to Book Airport Long-Term Parking", "How to Book Parking"]:
        print(f"\nQuery: {query!r}")
        result = retrieve_static_parking_info.invoke({"query": query})
        print(f"  status: {result.get('status')}, count: {result.get('count')}")
        if result.get("results"):
            for i, r in enumerate(result["results"][:3]):
                print(f"  result[{i}]: {r.get('source_file')} (parking_id={r.get('parking_id')})")
        else:
            print("  results: []")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
