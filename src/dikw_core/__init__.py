"""dikw-core: AI-native knowledge engine across the DIKW pyramid."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dikw-core")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
