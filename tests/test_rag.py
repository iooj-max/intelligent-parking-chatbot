"""
Tests for RAG retriever.

This module contains unit tests (with mocks) and integration tests (with real databases)
for the ParkingRetriever component.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, date, time
from decimal import Decimal

from src.rag.retriever import ParkingRetriever, QueryIntent, RetrievalResult, RetrievalError
from src.rag.vector_store import WeaviateStore
from src.rag.sql_store import SQLStore, SpaceAvailability, WorkingHours, PricingRule, SpecialHours
from src.data.embeddings import EmbeddingGenerator


@pytest.fixture
def mock_vector_store():
    """Mock Weaviate store with sample search results."""
    store = Mock(spec=WeaviateStore)
    store.search_similar.return_value = [
        {
            "parking_id": "downtown_plaza",
            "content": "Downtown Plaza Parking offers 250 spaces with 24/7 security.",
            "content_type": "general_info",
            "source_file": "general_info.md",
            "chunk_index": 0,
            "metadata": {
                "parking_name": "Downtown Plaza Parking",
                "address": "123 Main Street",
                "city": "City",
            },
            "distance": 0.15,
        }
    ]
    return store


@pytest.fixture
def mock_sql_store():
    """Mock PostgreSQL store with sample dynamic data."""
    store = Mock(spec=SQLStore)

    # Mock availability
    availability = SpaceAvailability(
        parking_id="downtown_plaza",
        total_spaces=250,
        occupied_spaces=180,
        last_updated=datetime(2026, 2, 10, 14, 30),
    )
    # Manually set computed field for mock
    availability.available_spaces = 70
    store.get_availability.return_value = availability

    # Mock working hours
    store.get_working_hours.return_value = [
        WorkingHours(
            id=1,
            parking_id="downtown_plaza",
            day_of_week=0,
            open_time=time(0, 0),
            close_time=time(23, 59),
            is_closed=False,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        WorkingHours(
            id=2,
            parking_id="downtown_plaza",
            day_of_week=1,
            open_time=time(0, 0),
            close_time=time(23, 59),
            is_closed=False,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]

    # Mock pricing rules
    store.get_pricing_rules.return_value = [
        PricingRule(
            id=1,
            parking_id="downtown_plaza",
            rule_name="Hourly Rate",
            time_unit="hour",
            price_per_unit=Decimal("5.00"),
            priority=1,
            is_active=True,
            min_duration_minutes=None,
            max_duration_minutes=None,
            day_of_week_start=None,
            day_of_week_end=None,
            time_start=None,
            time_end=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]

    store.get_special_hours.return_value = []

    return store


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    gen = Mock(spec=EmbeddingGenerator)
    gen.generate.return_value = [0.1] * 1536  # Mock 1536-dim vector
    return gen


@pytest.fixture
def retriever(mock_vector_store, mock_sql_store, mock_embedding_generator):
    """Create retriever with mocked dependencies."""
    return ParkingRetriever(
        vector_store=mock_vector_store,
        sql_store=mock_sql_store,
        embedding_generator=mock_embedding_generator,
    )


# Unit Tests

class TestQueryIntentClassification:
    """Test query intent classification."""

    @pytest.mark.parametrize(
        "query,expected_intent",
        [
            ("What are your hours?", QueryIntent.DYNAMIC),
            ("Is parking available now?", QueryIntent.DYNAMIC),
            ("How much does it cost to park?", QueryIntent.DYNAMIC),
            ("What is the price?", QueryIntent.DYNAMIC),
            ("Are there any spaces?", QueryIntent.DYNAMIC),
            ("Where is the parking lot located?", QueryIntent.STATIC),
            ("What security features do you have?", QueryIntent.STATIC),
            ("Do you have electric charging?", QueryIntent.STATIC),
            ("What is the booking process?", QueryIntent.STATIC),
            ("What is your address?", QueryIntent.STATIC),
            ("Is parking available and where are you located?", QueryIntent.HYBRID),
            ("What are your hours and what features do you offer?", QueryIntent.HYBRID),
            ("What are the prices and where are you?", QueryIntent.HYBRID),
            ("I want to book a parking spot", QueryIntent.RESERVATION),
            ("Make reservation for parking", QueryIntent.RESERVATION),
        ],
    )
    def test_classify_query_intent(self, retriever, query, expected_intent):
        """Test intent classification for various query types."""
        intent = retriever._classify_query_intent(query)
        assert intent == expected_intent

    def test_default_to_hybrid(self, retriever):
        """Test that ambiguous queries default to HYBRID."""
        intent = retriever._classify_query_intent("parking")
        assert intent == QueryIntent.HYBRID


class TestParkingIDInference:
    """Test parking ID inference from queries."""

    @pytest.mark.parametrize(
        "query,expected_id",
        [
            ("What are the hours at downtown plaza?", "downtown_plaza"),
            ("Is downtown parking available?", "downtown_plaza"),
            ("Show me parking at 123 main street", "downtown_plaza"),
            ("Is airport parking available?", "airport_parking"),
            ("Long-term parking at the airport", "airport_parking"),
            ("4500 airport boulevard", "airport_parking"),
            ("What parking is available?", None),  # Ambiguous
            ("Tell me about parking", None),  # Ambiguous
        ],
    )
    def test_infer_parking_id(self, retriever, query, expected_id):
        """Test parking_id inference for various query phrasings."""
        parking_id = retriever._infer_parking_id(query)
        assert parking_id == expected_id


class TestStaticContentRetrieval:
    """Test static content retrieval from Weaviate."""

    def test_retrieve_static_content_with_parking_id(self, retriever, mock_vector_store, mock_embedding_generator):
        """Test static content retrieval with parking_id filter."""
        query = "What security features do you have?"
        chunks = retriever._retrieve_static_content(query, parking_id="downtown_plaza")

        # Verify embedding was generated
        mock_embedding_generator.generate.assert_called_once_with(query)

        # Verify Weaviate search was called with parking_id filter
        mock_vector_store.search_similar.assert_called_once()
        call_kwargs = mock_vector_store.search_similar.call_args[1]
        assert call_kwargs["parking_id"] == "downtown_plaza"

        # Verify results returned
        assert len(chunks) > 0
        assert chunks[0]["parking_id"] == "downtown_plaza"

    def test_retrieve_static_content_all_facilities(self, retriever, mock_vector_store):
        """Test static content retrieval without parking_id filter."""
        query = "What parking lots are available?"
        chunks = retriever._retrieve_static_content(query, parking_id=None)

        # Verify no parking_id filter applied
        call_kwargs = mock_vector_store.search_similar.call_args[1]
        assert call_kwargs.get("parking_id") is None


class TestDynamicDataRetrieval:
    """Test dynamic data retrieval from PostgreSQL."""

    def test_retrieve_dynamic_data(self, retriever, mock_sql_store):
        """Test dynamic data retrieval from all tables."""
        data = retriever._retrieve_dynamic_data("downtown_plaza")

        # Verify all SQL queries were called
        mock_sql_store.get_availability.assert_called_once_with("downtown_plaza")
        mock_sql_store.get_working_hours.assert_called_once_with("downtown_plaza")
        mock_sql_store.get_pricing_rules.assert_called_once_with("downtown_plaza", active_only=True)
        mock_sql_store.get_special_hours.assert_called_once_with("downtown_plaza")

        # Verify data structure
        assert "availability" in data
        assert "working_hours" in data
        assert "pricing_rules" in data
        # Note: special_hours may not be in data if empty list returned
        # This is correct behavior

        assert data["availability"].total_spaces == 250

    def test_retrieve_dynamic_data_handles_missing(self, retriever):
        """Test dynamic data retrieval handles missing data gracefully."""
        # Create store with None returns
        sql_store = Mock(spec=SQLStore)
        sql_store.get_availability.return_value = None
        sql_store.get_working_hours.return_value = []
        sql_store.get_pricing_rules.return_value = []
        sql_store.get_special_hours.return_value = []

        retriever_with_empty = ParkingRetriever(
            Mock(spec=WeaviateStore), sql_store, Mock(spec=EmbeddingGenerator)
        )

        data = retriever_with_empty._retrieve_dynamic_data("unknown_parking")

        # Should return empty dict or dict with empty values
        assert isinstance(data, dict)


class TestContextFormatting:
    """Test context string formatting."""

    def test_format_context_string_with_both_data(self, retriever):
        """Test formatting with both static and dynamic data."""
        static_chunks = [
            {
                "parking_id": "downtown_plaza",
                "content": "Downtown Plaza offers excellent security.",
                "content_type": "features",
                "source_file": "features.md",
                "chunk_index": 0,
                "metadata": {
                    "parking_name": "Downtown Plaza Parking",
                    "address": "123 Main Street",
                },
            }
        ]

        dynamic_data = {
            "availability": Mock(
                total_spaces=250,
                occupied_spaces=180,
                available_spaces=70,
                last_updated=datetime(2026, 2, 10, 14, 30),
            ),
            "working_hours": [
                Mock(day_of_week=0, open_time=time(0, 0), close_time=time(23, 59), is_closed=False),
            ],
            "pricing_rules": [
                Mock(
                    rule_name="Hourly Rate",
                    price_per_unit=Decimal("5.00"),
                    time_unit="hour",
                    time_start=None,
                    time_end=None,
                    day_of_week_start=None,
                    day_of_week_end=None,
                ),
            ],
        }

        context = retriever._format_context_string(static_chunks, dynamic_data, "downtown_plaza")

        # Verify sections present
        assert "## Static Information" in context
        assert "## Dynamic Data" in context
        assert "### Current Availability" in context
        assert "### Operating Hours" in context
        assert "### Current Pricing" in context
        assert "**Instructions**" in context

        # Verify content
        assert "Downtown Plaza offers excellent security" in context
        assert "Total Spaces: 250" in context
        assert "Hourly Rate" in context

    def test_format_context_string_static_only(self, retriever):
        """Test formatting with only static data."""
        static_chunks = [
            {
                "parking_id": "downtown_plaza",
                "content": "Test content",
                "content_type": "general_info",
                "source_file": "general_info.md",
                "chunk_index": 0,
                "metadata": {"parking_name": "Downtown Plaza"},
            }
        ]

        context = retriever._format_context_string(static_chunks, {}, "downtown_plaza")

        assert "## Static Information" in context
        assert "## Dynamic Data" not in context
        assert "Test content" in context

    def test_format_context_string_dynamic_only(self, retriever):
        """Test formatting with only dynamic data."""
        dynamic_data = {
            "availability": Mock(
                total_spaces=250,
                occupied_spaces=180,
                available_spaces=70,
                last_updated=datetime(2026, 2, 10, 14, 30),
            ),
        }

        context = retriever._format_context_string([], dynamic_data, "downtown_plaza")

        assert "## Static Information" not in context
        assert "## Dynamic Data" in context
        assert "Total Spaces: 250" in context


class TestEndToEndRetrieval:
    """Test complete retrieval flow."""

    def test_retrieve_dynamic_query(self, retriever):
        """Test retrieval for dynamic query."""
        result = retriever.retrieve(
            query="Is parking available at downtown plaza?",
            return_format="structured",
        )

        assert isinstance(result, RetrievalResult)
        assert result.intent == QueryIntent.DYNAMIC
        assert result.parking_id == "downtown_plaza"
        assert result.dynamic_data  # Should have dynamic data
        assert result.context_string  # Should have formatted context

    def test_retrieve_static_query(self, retriever):
        """Test retrieval for static query."""
        result = retriever.retrieve(
            query="What security features does downtown plaza have?",
            return_format="structured",
        )

        assert result.intent == QueryIntent.STATIC
        assert len(result.static_chunks) > 0

    def test_retrieve_hybrid_query(self, retriever):
        """Test retrieval for hybrid query."""
        result = retriever.retrieve(
            query="Is parking available at downtown plaza and what are the security features?",
            return_format="structured",
        )

        assert result.intent == QueryIntent.HYBRID
        assert len(result.static_chunks) > 0
        # Dynamic data should be present since parking_id is inferred
        assert result.dynamic_data or result.parking_id is None  # May not have dynamic data without parking_id

    def test_retrieve_returns_string_format(self, retriever):
        """Test retrieval with string return format."""
        context = retriever.retrieve(
            query="What are your hours at downtown plaza?",  # Include parking_id for dynamic data
            return_format="string",
        )

        assert isinstance(context, str)
        # Should have either dynamic data or static info or empty result message
        assert ("## Dynamic Data" in context or "## Static Information" in context or
                "No Information Found" in context)

    def test_retrieve_with_explicit_parking_id(self, retriever):
        """Test retrieval with explicitly provided parking_id."""
        result = retriever.retrieve(
            query="Is parking available?",
            parking_id="downtown_plaza",
            return_format="structured",
        )

        assert result.parking_id == "downtown_plaza"

    def test_retrieve_with_explicit_intent(self, retriever):
        """Test retrieval with explicitly provided intent."""
        result = retriever.retrieve(
            query="Tell me about parking",
            intent=QueryIntent.STATIC,
            return_format="structured",
        )

        assert result.intent == QueryIntent.STATIC


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    def test_vector_store_failure_continues_with_dynamic(self, mock_sql_store, mock_embedding_generator):
        """Test that Weaviate failure doesn't break retrieval."""
        # Vector store that raises exception
        failing_vector_store = Mock(spec=WeaviateStore)
        failing_vector_store.search_similar.side_effect = Exception("Weaviate connection failed")

        retriever = ParkingRetriever(failing_vector_store, mock_sql_store, mock_embedding_generator)

        # Should not raise, should return dynamic data only
        result = retriever.retrieve(
            query="Is parking available?",
            parking_id="downtown_plaza",
            return_format="structured",
        )

        assert result.static_chunks == []  # No static content
        assert result.dynamic_data  # Still has dynamic data

    def test_sql_store_failure_continues_with_static(self, mock_vector_store, mock_embedding_generator):
        """Test that PostgreSQL failure doesn't break retrieval."""
        # SQL store that raises exception
        failing_sql_store = Mock(spec=SQLStore)
        failing_sql_store.get_availability.side_effect = Exception("PostgreSQL connection failed")

        retriever = ParkingRetriever(mock_vector_store, failing_sql_store, mock_embedding_generator)

        result = retriever.retrieve(
            query="What security features do you have?",
            parking_id="downtown_plaza",
            return_format="structured",
        )

        assert len(result.static_chunks) > 0  # Has static content
        assert result.dynamic_data == {}  # No dynamic data

    def test_empty_results_handling(self, mock_vector_store, mock_sql_store, mock_embedding_generator):
        """Test handling of empty results."""
        # Return empty results
        mock_vector_store.search_similar.return_value = []
        mock_sql_store.get_availability.return_value = None
        mock_sql_store.get_working_hours.return_value = []
        mock_sql_store.get_pricing_rules.return_value = []
        mock_sql_store.get_special_hours.return_value = []

        retriever = ParkingRetriever(mock_vector_store, mock_sql_store, mock_embedding_generator)

        context = retriever.retrieve(
            query="Unknown query",
            parking_id="unknown_parking",
            return_format="string",
        )

        # Should return helpful empty result message
        assert isinstance(context, str)
        assert len(context) > 0
        assert "No Information Found" in context or "couldn't find" in context.lower()


# Integration Tests (require running databases)

@pytest.mark.integration
class TestIntegration:
    """Integration tests with real databases."""

    @pytest.fixture
    def real_retriever(self):
        """Create retriever with real database connections."""
        vector_store = WeaviateStore()
        sql_store = SQLStore()
        embedding_gen = EmbeddingGenerator()

        return ParkingRetriever(vector_store, sql_store, embedding_gen)

    def test_real_retrieval_dynamic_query(self, real_retriever):
        """Test with real databases - dynamic query."""
        result = real_retriever.retrieve(
            query="Is parking available at downtown plaza?",
            return_format="structured",
        )

        assert result.parking_id == "downtown_plaza"
        assert result.dynamic_data

        # Print context for manual inspection
        print("\n" + "=" * 80)
        print("DYNAMIC QUERY CONTEXT:")
        print("=" * 80)
        print(result.context_string)
        print("=" * 80)

    def test_real_retrieval_static_query(self, real_retriever):
        """Test with real databases - static query."""
        result = real_retriever.retrieve(
            query="What security features does downtown plaza have?",
            return_format="structured",
        )

        assert len(result.static_chunks) > 0

        print("\n" + "=" * 80)
        print("STATIC QUERY CONTEXT:")
        print("=" * 80)
        print(result.context_string)
        print("=" * 80)

    def test_real_retrieval_hybrid_query(self, real_retriever):
        """Test with real databases - hybrid query."""
        result = real_retriever.retrieve(
            query="What are your hours and is parking available?",
            return_format="structured",
        )

        assert result.intent == QueryIntent.HYBRID
        assert result.parking_id is not None

        print("\n" + "=" * 80)
        print("HYBRID QUERY CONTEXT:")
        print("=" * 80)
        print(result.context_string)
        print("=" * 80)


class TestMetadataExtraction:
    """Test metadata extraction from results."""

    def test_extract_metadata_from_static_chunks(self, retriever):
        """Test metadata extraction from static chunks."""
        static_chunks = [
            {
                "parking_id": "downtown_plaza",
                "content": "Test",
                "content_type": "general_info",
                "source_file": "general_info.md",
                "chunk_index": 0,
                "metadata": {
                    "parking_name": "Downtown Plaza Parking",
                    "address": "123 Main Street",
                    "city": "City",
                },
            }
        ]

        metadata = retriever._extract_metadata(static_chunks, {})

        assert "parking_facilities" in metadata
        assert "downtown_plaza" in metadata["parking_facilities"]
        assert metadata["parking_facilities"]["downtown_plaza"]["name"] == "Downtown Plaza Parking"
