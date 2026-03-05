import pytest

from valediction.demo import create_demo_dataset
from valediction.exceptions import DataIntegrityError
from valediction.integrity import (
    get_config,
    reset_default_config,
    reset_injected_config_variables,
)


# Parameters
@pytest.fixture(autouse=True)
def _isolate_global_default():  # noqa
    """Ensure each test starts clean and leaves it clean."""
    reset_injected_config_variables()
    reset_default_config()
    yield
    reset_injected_config_variables()
    reset_default_config()


# Tests
def test_pk_col_length_config():
    with get_config() as config:
        config.pk_col_max_length = 64
        assert config.pk_col_max_length == 64

    assert get_config().pk_col_max_length == 0


def test_pk_col_length_raises():
    dataset = create_demo_dataset()
    dataset.validate()
    dataset.check()
    assert dataset.validated

    with get_config() as config:
        config.pk_col_max_length = 64
        dataset.validate()
        dataset.check()
        assert dataset.validated

    with get_config() as config:
        config.pk_col_max_length = 10
        dataset.validate()
        with pytest.raises(DataIntegrityError):
            dataset.check()


if __name__ == "__main__":
    pytest.main([__file__])
