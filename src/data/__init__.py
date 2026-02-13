"""
Data loading and processing utilities.

This module provides tools for loading test parking data into Weaviate
and PostgreSQL databases.

MVP only - Not production-ready implementation.
"""

from .loader import DataLoader, main as load_data

__all__ = ["DataLoader", "load_data"]
