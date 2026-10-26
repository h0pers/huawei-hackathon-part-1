# lightgbm must load its OpenMP runtime before torch loads its own, or the process segfaults on
# macOS once both are in use (pytorch#191933, LightGBM#6595); see techarena/choice.py.
import lightgbm  # noqa: F401

from techarena import config  # noqa: F401
