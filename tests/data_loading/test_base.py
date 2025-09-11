"""Tests for fm_app_toolkit.data_loading.base module."""

import pandas as pd
import pytest

from fm_app_toolkit.data_loading.base import BaseRepository


def test__base_repository__is_abstract():
    """BaseRepository cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseRepository"):
        BaseRepository()


def test__base_repository__requires_load_data_implementation():
    """Concrete implementations must implement load_data method."""

    class IncompleteRepository(BaseRepository):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteRepository"):
        IncompleteRepository()


def test__base_repository__abstract_method_signature():
    """load_data method must have correct signature."""

    class ConcreteRepository(BaseRepository):
        def load_data(self, path: str) -> pd.DataFrame:
            return pd.DataFrame({"test": [1, 2, 3]})

    # Should instantiate without error
    repo = ConcreteRepository()
    result = repo.load_data("dummy_path")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["test"]
