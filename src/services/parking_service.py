"""Service for dynamic parking facility management."""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from src.rag.sql_store import SQLStore, ParkingFacility

logger = logging.getLogger(__name__)


class ParkingFacilityService:
    """
    Centralized service for parking facility management.
    Loads parking data dynamically from PostgreSQL.
    """

    def __init__(self, sql_store: SQLStore):
        self._sql_store = sql_store
        self._cache: Dict[str, ParkingFacility] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_timestamp is None:
            return False
        return datetime.now() - self._cache_timestamp < self._cache_ttl

    def _refresh_cache(self) -> None:
        """Refresh facility cache from database."""
        with self._sql_store.get_session_context() as session:
            facilities = session.query(ParkingFacility).all()
            self._cache = {f.parking_id: f for f in facilities}
            self._cache_timestamp = datetime.now()
            logger.info(f"Loaded {len(self._cache)} parking facilities from database")

    def get_all_facilities(self) -> List[ParkingFacility]:
        """Load all parking facilities from DB."""
        if not self._is_cache_valid():
            self._refresh_cache()
        return list(self._cache.values())

    def get_facility_by_id(self, parking_id: str) -> Optional[ParkingFacility]:
        """Get specific facility by ID."""
        if not self._is_cache_valid():
            self._refresh_cache()
        return self._cache.get(parking_id)

    def get_facility_id_mapping(self) -> Dict[str, str]:
        """Get mapping of parking_id -> human readable name."""
        facilities = self.get_all_facilities()
        return {f.parking_id: f.name for f in facilities}

    def get_parking_ids(self) -> List[str]:
        """Get list of all parking IDs."""
        return list(self.get_facility_id_mapping().keys())


# Global singleton instance (single-threaded MVP)
_parking_service: Optional[ParkingFacilityService] = None


def get_parking_service() -> ParkingFacilityService:
    """Get or create ParkingFacilityService singleton."""
    global _parking_service

    if _parking_service is None:
        from src.rag.sql_store import SQLStore
        sql_store = SQLStore()
        _parking_service = ParkingFacilityService(sql_store)

    return _parking_service
