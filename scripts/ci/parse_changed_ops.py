#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Map PR changed files to the operators that need verification.

The CI writes the changed-file list (from ``git diff``) into a file, then calls
this script to decide which operators must be compiled/tested.

Usage:
    python3 scripts/ci/parse_changed_ops.py <changed_list> \
        [--config test/test_config.yaml] [--op-list scripts/ci/operator_list.yaml]

Input format of <changed_list> (one file per line):
    samples/matmul/matmul.py            (git diff --name-only)
    M\tsamples/matmul/matmul.py         (git diff --name-status)

Output (stdout):
    op1;op2;op3   — operators to verify
    all           — full-suite verification (a shared file changed)
    ""            — no operator affected
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 结果输出专用 logger：写 stdout 且无前缀，供 CI 读取 ops 结果
result_logger = logging.getLogger("result")
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(logging.Formatter("%(message)s"))
result_logger.addHandler(_stdout_handler)
result_logger.setLevel(logging.INFO)
result_logger.propagate = False

_SKIP_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".svg", ".gif")
_STATUS_PREFIXES = ("M\t", "A\t", "D\t", "R\t", "C\t")


class Module:
    """A rule node in test_config.yaml: guarded sources -> related options."""

    def __init__(self, name, desc):
        self.name = name
        src = desc.get("src", [])
        self.src_files = [Path(p) for p in (src if isinstance(src, list) else [src])]
        options = desc.get("options", [])
        self.options = list(options) if isinstance(options, list) else [options]

    def matches(self, file_path):
        changed = Path(file_path)
        for src in self.src_files:
            try:
                changed.relative_to(src)
                return True
            except ValueError:
                continue
        return False


def load_modules(config_path):
    """Recursively walk the yaml tree and collect all module nodes."""
    with open(config_path, encoding="utf-8") as f:
        desc = yaml.safe_load(f)

    modules = []

    def walk(obj, prefix=""):
        if not isinstance(obj, dict):
            return
        if "module" in obj:
            modules.append(Module(prefix, obj))
            return
        for key, value in obj.items():
            walk(value, f"{prefix}/{key}" if prefix else key)

    walk(desc)
    return modules


def load_operator_list(op_list_path):
    with open(op_list_path, encoding="utf-8") as f:
        desc = yaml.safe_load(f)
    return desc.get("operators", [])


def load_ignored_paths(config_path):
    """Load exact paths that do not affect the operator test scope."""
    with open(config_path, encoding="utf-8") as f:
        desc = yaml.safe_load(f)

    ignored = desc.get("ignore", [])
    if not isinstance(ignored, list):
        ignored = [ignored]
    return {Path(path) for path in ignored}


def is_skippable(file_path):
    return file_path.endswith(_SKIP_SUFFIXES)


def normalize_line(line):
    """Strip git diff --name-status prefixes such as 'M\\tfile'."""
    line = line.strip()
    for prefix in _STATUS_PREFIXES:
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return line


def matching_ops(modules, line):
    """Return options of modules matching the given line, or None."""
    for mod in modules:
        if mod.matches(line):
            return mod.options
    return None


def collect_related_ops(modules, changed_list_path, ignored_paths=()):
    """Collect related operator options for the changed-file list."""
    related = []
    with open(changed_list_path, encoding="utf-8") as f:
        for line in f:
            line = normalize_line(line)
            if not line or line.startswith("#") or is_skippable(line):
                continue
            if Path(line) in ignored_paths:
                logger.info("ignored metadata file: %s", line)
                continue
            options = matching_ops(modules, line)
            if options:
                related.extend(options)
    return list(dict.fromkeys(related))


def resolve_ops_output(related, op_list_path):
    """Convert collected options to the final ops string for stdout."""
    if not related:
        logger.info("no operators affected by this change")
        return ""
    if "all" in related:
        operators = load_operator_list(op_list_path)
        logger.info("shared file changed, triggering full suite: %s", operators)
        return "all"
    logger.info("related operators: %s", related)
    return ";".join(related)


def main():
    parser = argparse.ArgumentParser(
        description="Parse changed files and print related operators."
    )
    parser.add_argument("changed_list", type=Path, help="changed-file list from CI")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("test/test_config.yaml"),
        help="rule file (test/test_config.yaml)",
    )
    parser.add_argument(
        "--op-list",
        type=Path,
        default=Path("scripts/ci/operator_list.yaml"),
        help="full operator list file",
    )
    args = parser.parse_args()

    if not args.changed_list.exists():
        logger.error("changed-list file not found: %s", args.changed_list)
        return 1, ""
    if not args.config.exists():
        logger.error("config file not found: %s", args.config)
        return 1, ""

    modules = load_modules(args.config)
    ignored_paths = load_ignored_paths(args.config)
    related = collect_related_ops(modules, args.changed_list, ignored_paths)
    return 0, resolve_ops_output(related, args.op_list)


if __name__ == "__main__":
    ret_code, ops_output = main()
    result_logger.info("%s", ops_output)
    sys.exit(ret_code)
