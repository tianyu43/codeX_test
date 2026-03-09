import os
import random

import numpy as np
import sympy.core.numbers as sympy_numbers
import typing_extensions

os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

if not hasattr(sympy_numbers, "equal_valued"):
    def _equal_valued(a, b):
        return a == b

    sympy_numbers.equal_valued = _equal_valued

if not hasattr(typing_extensions, "deprecated"):
    def _deprecated(*_args, **_kwargs):
        def _decorator(obj):
            return obj

        return _decorator

    typing_extensions.deprecated = _deprecated

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
