import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def taichi_init():
    """Initialise Taichi once for the entire test session (CPU backend)."""
    ti.init(arch=ti.cpu)
