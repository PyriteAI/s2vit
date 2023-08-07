from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("s2vit")
except PackageNotFoundError:
    # package is not installed
    __version__ = "0.0.0"
