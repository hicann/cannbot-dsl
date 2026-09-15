# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Load repository samples without colliding with installed modules."""

import importlib.util
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys

_SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "samples"


def load_sample(relative_path):
    relative_path = Path(relative_path)
    path = _SAMPLE_ROOT / relative_path
    # Give each sample a private package so sibling imports stay within the sample directory.
    # Generic names such as flash_kda may already belong to external examples.
    parts = relative_path.with_suffix("").parts
    prefix = "_cannbot_samples"
    for depth in range(len(parts)):
        package_name = ".".join((prefix, *parts[:depth]))
        if package_name not in sys.modules:
            package_spec = ModuleSpec(package_name, loader=None, is_package=True)
            package_spec.submodule_search_locations = [str(_SAMPLE_ROOT.joinpath(*parts[:depth]))]
            sys.modules[package_name] = importlib.util.module_from_spec(package_spec)
    name = ".".join((prefix, *parts))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sample {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module
