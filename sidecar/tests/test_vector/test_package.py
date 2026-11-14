"""The vector package advertises only available public exports."""
from colonymind import vector


def test_public_exports_are_defined():
    for name in vector.__all__:
        assert hasattr(vector, name), name
