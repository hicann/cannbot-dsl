# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Set the Torch import environment shared by the FlashKDA sample modules."""

import importlib
import os


def configure():
    """Disable Torch backend auto-loading before framework modules are imported."""
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def load_module(module_name):
    """Import a framework module after the Torch environment is configured."""
    configure()
    return importlib.import_module(module_name)


def load_symbols(module_name, *symbol_names):
    """Load selected symbols without introducing order-sensitive import statements."""
    module = load_module(module_name)
    return tuple(getattr(module, name) for name in symbol_names)


configure()
