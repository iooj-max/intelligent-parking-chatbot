"""
PostgreSQL store for dynamic parking data.

This module provides SQLAlchemy ORM models and a store class for managing
dynamic parking data including facilities, working hours, pricing rules,
special hours, and real-time space availability.

MVP only - Not production-ready implementation.
"""

from contextlib import contextmanager
from datetime import date, datetime, time
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    Time,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.sql import func

from ..config import settings

Base = declarative_base()


class ParkingFacility(Base):
    """Parking facility master data."""

    __tablename__ = "parking_facilities"

    parking_id = Column(String(50), primary_key=True)
    name = Column(String(255), nullable=False)
    address = Column(Text, nullable=False)
    city = Column(String(100), nullable=False)
    latitude = Column(Numeric(10, 8))
    longitude = Column(Numeric(11, 8))
    total_spaces = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationships
    working_hours = relationship("WorkingHours", back_populates="facility", cascade="all, delete-orphan")
    special_hours = relationship("SpecialHours", back_populates="facility", cascade="all, delete-orphan")
    pricing_rules = relationship("PricingRule", back_populates="facility", cascade="all, delete-orphan")
    availability = relationship("SpaceAvailability", back_populates="facility", cascade="all, delete-orphan", uselist=False)

    __table_args__ = (
        CheckConstraint("total_spaces > 0", name="check_total_spaces_positive"),
        Index("idx_parking_facilities_city", "city"),
    )


class WorkingHours(Base):
    """Regular weekly operating hours for parking facilities."""

    __tablename__ = "working_hours"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parking_id = Column(String(50), ForeignKey("parking_facilities.parking_id", ondelete="CASCADE"), nullable=False)
    day_of_week = Column(SmallInteger, nullable=False)  # 0=Monday, 6=Sunday
    open_time = Column(Time, nullable=False)
    close_time = Column(Time, nullable=False)
    is_closed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationships
    facility = relationship("ParkingFacility", back_populates="working_hours")

    __table_args__ = (
        CheckConstraint("day_of_week BETWEEN 0 AND 6", name="check_day_of_week_range"),
        UniqueConstraint("parking_id", "day_of_week", name="unique_parking_day"),
        Index("idx_working_hours_parking_id", "parking_id"),
    )


class SpecialHours(Base):
    """Special operating hours for holidays, maintenance, events."""

    __tablename__ = "special_hours"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parking_id = Column(String(50), ForeignKey("parking_facilities.parking_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    open_time = Column(Time)
    close_time = Column(Time)
    is_closed = Column(Boolean, nullable=False, default=False)
    reason = Column(String(255))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    facility = relationship("ParkingFacility", back_populates="special_hours")

    __table_args__ = (
        UniqueConstraint("parking_id", "date", name="unique_parking_date"),
        Index("idx_special_hours_parking_date", "parking_id", "date"),
    )


class PricingRule(Base):
    """Pricing rules with time-based and dynamic pricing support."""

    __tablename__ = "pricing_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parking_id = Column(String(50), ForeignKey("parking_facilities.parking_id", ondelete="CASCADE"), nullable=False)
    rule_name = Column(String(100), nullable=False)
    time_unit = Column(String(20), nullable=False)  # hour, day, week, month
    price_per_unit = Column(Numeric(10, 2), nullable=False)
    min_duration_minutes = Column(Integer)
    max_duration_minutes = Column(Integer)
    day_of_week_start = Column(SmallInteger)  # 0=Monday, 6=Sunday
    day_of_week_end = Column(SmallInteger)
    time_start = Column(Time)
    time_end = Column(Time)
    priority = Column(Integer, nullable=False, default=0)  # Higher priority applies first
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationships
    facility = relationship("ParkingFacility", back_populates="pricing_rules")

    __table_args__ = (
        CheckConstraint("time_unit IN ('hour', 'day', 'week', 'month')", name="check_time_unit_valid"),
        CheckConstraint("price_per_unit >= 0", name="check_price_non_negative"),
        CheckConstraint("day_of_week_start IS NULL OR (day_of_week_start BETWEEN 0 AND 6)", name="check_dow_start_range"),
        CheckConstraint("day_of_week_end IS NULL OR (day_of_week_end BETWEEN 0 AND 6)", name="check_dow_end_range"),
        Index("idx_pricing_rules_parking_id", "parking_id"),
        Index("idx_pricing_rules_active", "parking_id", "is_active", "priority"),
    )


class SpaceAvailability(Base):
    """Real-time parking space availability with denormalized counts."""

    __tablename__ = "space_availability"

    parking_id = Column(String(50), ForeignKey("parking_facilities.parking_id", ondelete="CASCADE"), primary_key=True)
    total_spaces = Column(Integer, nullable=False)
    occupied_spaces = Column(Integer, nullable=False, default=0)
    # available_spaces is computed column: total_spaces - occupied_spaces
    # Using Computed to properly mark as generated column
    available_spaces = Column(Integer, Computed("total_spaces - occupied_spaces", persisted=True))
    last_updated = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    facility = relationship("ParkingFacility", back_populates="availability")

    __table_args__ = (
        CheckConstraint("total_spaces > 0", name="check_total_spaces_positive_avail"),
        CheckConstraint("occupied_spaces >= 0", name="check_occupied_non_negative"),
    )


class SQLStore:
    """
    PostgreSQL store for dynamic parking data.

    Provides connection management, table creation, and CRUD operations
    for parking facilities, hours, pricing, and availability.

    MVP only - Not production-ready implementation.
    """

    def __init__(self, dsn: Optional[str] = None):
        """
        Initialize SQLStore with database connection.

        Args:
            dsn: PostgreSQL connection string. If None, uses settings.postgres_dsn
        """
        self.dsn = dsn or settings.postgres_dsn
        self._dsn_redacted = self._redact_password(self.dsn)
        self.engine = create_engine(self.dsn, pool_pre_ping=True, pool_size=5, max_overflow=10)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def _redact_password(self, dsn: str) -> str:
        """
        Redact password from DSN for safe logging.

        Args:
            dsn: Database connection string

        Returns:
            DSN with password replaced by ****
        """
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', dsn)

    def create_tables(self) -> None:
        """
        Create all tables with constraints and triggers.

        Idempotent - safe to call multiple times.
        """
        # Create tables
        Base.metadata.create_all(bind=self.engine)

        # Create generated column for available_spaces (not supported by SQLAlchemy ORM)
        # and trigger for last_updated
        with self.engine.connect() as conn:
            # Check if available_spaces column needs to be recreated as generated
            result = conn.execute(
                text(
                    """
                    SELECT column_name, is_generated
                    FROM information_schema.columns
                    WHERE table_name = 'space_availability'
                    AND column_name = 'available_spaces'
                """
                )
            )
            row = result.fetchone()

            if row and row[1] != "ALWAYS":
                # Drop and recreate as generated column
                conn.execute(text("ALTER TABLE space_availability DROP COLUMN IF EXISTS available_spaces"))
                conn.execute(
                    text(
                        """
                        ALTER TABLE space_availability
                        ADD COLUMN available_spaces INTEGER
                        GENERATED ALWAYS AS (total_spaces - occupied_spaces) STORED
                    """
                    )
                )
                conn.commit()

            # Create trigger function for space availability validation
            conn.execute(
                text(
                    """
                CREATE OR REPLACE FUNCTION check_space_availability()
                RETURNS TRIGGER AS $$
                BEGIN
                    IF NEW.occupied_spaces > NEW.total_spaces THEN
                        RAISE EXCEPTION 'Occupied spaces (%) cannot exceed total spaces (%)',
                            NEW.occupied_spaces, NEW.total_spaces;
                    END IF;
                    NEW.last_updated = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """
                )
            )

            # Create trigger (DROP IF EXISTS for idempotency)
            conn.execute(text("DROP TRIGGER IF EXISTS space_availability_check ON space_availability"))
            conn.execute(
                text(
                    """
                CREATE TRIGGER space_availability_check
                    BEFORE INSERT OR UPDATE ON space_availability
                    FOR EACH ROW
                    EXECUTE FUNCTION check_space_availability()
            """
                )
            )

            conn.commit()

    def drop_tables(self) -> None:
        """
        Drop all tables and functions.

        FOR TESTING ONLY - Destroys all data.
        """
        with self.engine.connect() as conn:
            # Drop trigger and function
            conn.execute(text("DROP TRIGGER IF EXISTS space_availability_check ON space_availability"))
            conn.execute(text("DROP FUNCTION IF EXISTS check_space_availability()"))
            conn.commit()

        # Drop all tables
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    @contextmanager
    def get_session_context(self):
        """
        Context manager for automatic session cleanup.

        Ensures sessions are properly closed even on exceptions,
        preventing memory leaks.

        Usage:
            with sql_store.get_session_context() as session:
                # ... database operations ...
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # Facility CRUD operations

    def upsert_facility(
        self,
        parking_id: str,
        name: str,
        address: str,
        city: str,
        total_spaces: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> ParkingFacility:
        """
        Insert or update parking facility.

        Args:
            parking_id: Unique facility identifier
            name: Facility name
            address: Street address
            city: City name
            total_spaces: Total parking spaces
            latitude: GPS latitude
            longitude: GPS longitude

        Returns:
            ParkingFacility object
        """
        session = self.get_session()
        try:
            facility = session.query(ParkingFacility).filter_by(parking_id=parking_id).first()

            if facility:
                # Update existing
                facility.name = name
                facility.address = address
                facility.city = city
                facility.total_spaces = total_spaces
                facility.latitude = latitude
                facility.longitude = longitude
                facility.updated_at = datetime.utcnow()
            else:
                # Insert new
                facility = ParkingFacility(
                    parking_id=parking_id,
                    name=name,
                    address=address,
                    city=city,
                    total_spaces=total_spaces,
                    latitude=latitude,
                    longitude=longitude,
                )
                session.add(facility)

            session.commit()
            session.refresh(facility)
            return facility
        finally:
            session.close()

    def delete_facility(self, parking_id: str) -> bool:
        """
        Delete parking facility and all related data (cascading).

        Args:
            parking_id: Facility identifier

        Returns:
            True if deleted, False if not found
        """
        session = self.get_session()
        try:
            facility = session.query(ParkingFacility).filter_by(parking_id=parking_id).first()
            if facility:
                session.delete(facility)
                session.commit()
                return True
            return False
        finally:
            session.close()

    # Working Hours operations

    def upsert_working_hours(self, parking_id: str, day_of_week: int, open_time: time, close_time: time, is_closed: bool = False) -> WorkingHours:
        """
        Insert or update working hours for a specific day.

        Args:
            parking_id: Facility identifier
            day_of_week: 0=Monday, 6=Sunday
            open_time: Opening time
            close_time: Closing time
            is_closed: Whether facility is closed this day

        Returns:
            WorkingHours object
        """
        session = self.get_session()
        try:
            hours = session.query(WorkingHours).filter_by(parking_id=parking_id, day_of_week=day_of_week).first()

            if hours:
                hours.open_time = open_time
                hours.close_time = close_time
                hours.is_closed = is_closed
                hours.updated_at = datetime.utcnow()
            else:
                hours = WorkingHours(
                    parking_id=parking_id, day_of_week=day_of_week, open_time=open_time, close_time=close_time, is_closed=is_closed
                )
                session.add(hours)

            session.commit()
            session.refresh(hours)
            return hours
        finally:
            session.close()

    def get_working_hours(self, parking_id: str) -> List[WorkingHours]:
        """
        Get all working hours for a parking facility.

        Args:
            parking_id: Facility identifier

        Returns:
            List of WorkingHours objects ordered by day_of_week
        """
        session = self.get_session()
        try:
            return session.query(WorkingHours).filter_by(parking_id=parking_id).order_by(WorkingHours.day_of_week).all()
        finally:
            session.close()

    # Special Hours operations

    def upsert_special_hours(
        self,
        parking_id: str,
        date_val: date,
        open_time: Optional[time] = None,
        close_time: Optional[time] = None,
        is_closed: bool = False,
        reason: Optional[str] = None,
    ) -> SpecialHours:
        """
        Insert or update special hours for a specific date.

        Args:
            parking_id: Facility identifier
            date_val: Date for special hours
            open_time: Opening time (None if closed)
            close_time: Closing time (None if closed)
            is_closed: Whether facility is closed this date
            reason: Reason for special hours

        Returns:
            SpecialHours object
        """
        session = self.get_session()
        try:
            special = session.query(SpecialHours).filter_by(parking_id=parking_id, date=date_val).first()

            if special:
                special.open_time = open_time
                special.close_time = close_time
                special.is_closed = is_closed
                special.reason = reason
            else:
                special = SpecialHours(
                    parking_id=parking_id, date=date_val, open_time=open_time, close_time=close_time, is_closed=is_closed, reason=reason
                )
                session.add(special)

            session.commit()
            session.refresh(special)
            return special
        finally:
            session.close()

    def get_special_hours(self, parking_id: str, date_val: Optional[date] = None) -> List[SpecialHours]:
        """
        Get special hours for a parking facility.

        Args:
            parking_id: Facility identifier
            date_val: Specific date (if None, returns all future special hours)

        Returns:
            List of SpecialHours objects
        """
        session = self.get_session()
        try:
            query = session.query(SpecialHours).filter_by(parking_id=parking_id)
            if date_val:
                query = query.filter_by(date=date_val)
            else:
                # Return future special hours
                query = query.filter(SpecialHours.date >= func.current_date())
            return query.order_by(SpecialHours.date).all()
        finally:
            session.close()

    # Pricing Rules operations

    def upsert_pricing_rule(
        self,
        parking_id: str,
        rule_name: str,
        time_unit: str,
        price_per_unit: Decimal,
        priority: int = 0,
        is_active: bool = True,
        min_duration_minutes: Optional[int] = None,
        max_duration_minutes: Optional[int] = None,
        day_of_week_start: Optional[int] = None,
        day_of_week_end: Optional[int] = None,
        time_start: Optional[time] = None,
        time_end: Optional[time] = None,
    ) -> PricingRule:
        """
        Insert or update pricing rule.

        Args:
            parking_id: Facility identifier
            rule_name: Name of pricing rule
            time_unit: hour, day, week, month
            price_per_unit: Price per time unit
            priority: Rule priority (higher applies first)
            is_active: Whether rule is active
            min_duration_minutes: Minimum duration in minutes
            max_duration_minutes: Maximum duration in minutes
            day_of_week_start: Start day (0=Monday)
            day_of_week_end: End day (6=Sunday)
            time_start: Start time of day
            time_end: End time of day

        Returns:
            PricingRule object
        """
        session = self.get_session()
        try:
            # Find existing rule by parking_id and rule_name
            rule = session.query(PricingRule).filter_by(parking_id=parking_id, rule_name=rule_name).first()

            if rule:
                # Update existing
                rule.time_unit = time_unit
                rule.price_per_unit = price_per_unit
                rule.priority = priority
                rule.is_active = is_active
                rule.min_duration_minutes = min_duration_minutes
                rule.max_duration_minutes = max_duration_minutes
                rule.day_of_week_start = day_of_week_start
                rule.day_of_week_end = day_of_week_end
                rule.time_start = time_start
                rule.time_end = time_end
                rule.updated_at = datetime.utcnow()
            else:
                # Insert new
                rule = PricingRule(
                    parking_id=parking_id,
                    rule_name=rule_name,
                    time_unit=time_unit,
                    price_per_unit=price_per_unit,
                    priority=priority,
                    is_active=is_active,
                    min_duration_minutes=min_duration_minutes,
                    max_duration_minutes=max_duration_minutes,
                    day_of_week_start=day_of_week_start,
                    day_of_week_end=day_of_week_end,
                    time_start=time_start,
                    time_end=time_end,
                )
                session.add(rule)

            session.commit()
            session.refresh(rule)
            return rule
        finally:
            session.close()

    def get_pricing_rules(self, parking_id: str, active_only: bool = True) -> List[PricingRule]:
        """
        Get pricing rules for a parking facility.

        Args:
            parking_id: Facility identifier
            active_only: If True, return only active rules

        Returns:
            List of PricingRule objects ordered by priority (descending)
        """
        session = self.get_session()
        try:
            query = session.query(PricingRule).filter_by(parking_id=parking_id)
            if active_only:
                query = query.filter_by(is_active=True)
            return query.order_by(PricingRule.priority.desc()).all()
        finally:
            session.close()

    # Space Availability operations

    def upsert_availability(self, parking_id: str, total_spaces: int, occupied_spaces: int) -> SpaceAvailability:
        """
        Insert or update space availability.

        Args:
            parking_id: Facility identifier
            total_spaces: Total parking spaces
            occupied_spaces: Currently occupied spaces

        Returns:
            SpaceAvailability object with computed available_spaces

        Raises:
            ValueError: If occupied_spaces > total_spaces
        """
        if occupied_spaces > total_spaces:
            raise ValueError(f"Occupied spaces ({occupied_spaces}) cannot exceed total spaces ({total_spaces})")

        session = self.get_session()
        try:
            avail = session.query(SpaceAvailability).filter_by(parking_id=parking_id).first()

            if avail:
                avail.total_spaces = total_spaces
                avail.occupied_spaces = occupied_spaces
                # last_updated set by trigger
            else:
                avail = SpaceAvailability(parking_id=parking_id, total_spaces=total_spaces, occupied_spaces=occupied_spaces)
                session.add(avail)

            session.commit()
            session.refresh(avail)
            return avail
        finally:
            session.close()

    def get_availability(self, parking_id: str) -> Optional[SpaceAvailability]:
        """
        Get current space availability for a parking facility.

        Args:
            parking_id: Facility identifier

        Returns:
            SpaceAvailability object or None if not found
        """
        session = self.get_session()
        try:
            return session.query(SpaceAvailability).filter_by(parking_id=parking_id).first()
        finally:
            session.close()
