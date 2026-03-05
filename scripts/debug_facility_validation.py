#!/usr/bin/env python3
"""Debug script for facility validation. Run from project root."""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main() -> None:
    from parking_agent.facility_validation import (
        _fetch_parking_facilities,
        _exact_match,
        validate_facility,
    )
    from parking_agent.clients import build_postgres_uri

    print("1. Postgres DSN (host redacted):")
    dsn = build_postgres_uri()
    # Redact password for display
    if "@" in dsn:
        parts = dsn.split("@")
        if ":" in parts[0]:
            user_part = parts[0].split(":")[0]
            print(f"   {user_part}:****@{parts[1]}")
    else:
        print(f"   {dsn[:50]}...")

    print("\n2. Fetching parking facilities from DB...")
    try:
        facilities = _fetch_parking_facilities()
        print(f"   OK: fetched {len(facilities)} facilities")
        for pid, name, addr, city in facilities:
            print(f"   - parking_id={pid!r} name={name!r}")
    except Exception as e:
        print(f"   FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n3. Exact match for 'airport_parking':")
    match = _exact_match("airport_parking", facilities)
    if match:
        print(f"   OK: matched {match}")
    else:
        print("   FAILED: no match")

    print("\n4. Full validate_facility(['airport_parking']):")
    result = validate_facility(["airport_parking"])
    print(f"   status={result.get('status')}")
    print(f"   is_valid={result.get('is_valid')}")
    print(f"   reason={result.get('reason', '')!r}")
    for r in result.get("results", []):
        print(f"   result: original={r.get('original')!r} matched_parking_id={r.get('matched_parking_id')!r}")

if __name__ == "__main__":
    main()
