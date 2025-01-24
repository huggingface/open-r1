# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re

import packaging.version


REPLACE_PATTERNS = {
    "init": (
        re.compile(r'^__version__\s+=\s+"([^"]+)"\s*$', re.MULTILINE),
        '__version__ = "VERSION"\n',
    ),
    "setup": (
        re.compile(r'^(\s*)version\s*=\s*"[^"]+",', re.MULTILINE),
        r'\1version="VERSION",',
    ),
    "citation": (re.compile(r"^version:\s+[^ ]+", re.MULTILINE), "version: VERSION"),
    "readme": (
        re.compile(r"version\s+=\s+\{[^}]+\}", re.MULTILINE),
        "version = {VERSION}",
    ),
}

README_FILE = "README.md"

REPLACE_FILES = {
    "init": "src/open_r1/__init__.py",
    "setup": "setup.py",
    "citation": "CITATION.cff",
    "readme": README_FILE,
}


def update_version_in_file(fname, version, pattern):
    """Update the version in one file using a specific pattern."""
    with open(fname, "r", encoding="utf-8", newline="\n") as f:
        code = f.read()
    re_pattern, replace = REPLACE_PATTERNS[pattern]
    replace = replace.replace("VERSION", version)
    code = re_pattern.sub(replace, code)
    with open(fname, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)


def global_version_update(version, patch=False):
    """Update the version in all needed files."""
    for pattern, fname in REPLACE_FILES.items():
        update_version_in_file(fname, version, pattern)


def get_version():
    """Reads the current version in the __init__."""
    with open(REPLACE_FILES["init"], "r") as f:
        code = f.read()
    default_version = REPLACE_PATTERNS["init"][0].search(code).groups()[0]
    return packaging.version.parse(default_version)


def pre_release_work(patch=False):
    """Do all the necessary pre-release steps."""
    # First let's get the default version: base version if we are in dev, bump minor otherwise.
    default_version = get_version()
    if patch and default_version.is_devrelease:
        raise ValueError("Can't create a patch version from the dev branch, checkout a released version!")
    if default_version.is_devrelease:
        default_version = default_version.base_version
    elif patch:
        default_version = f"{default_version.major}.{default_version.minor}.{default_version.micro + 1}"
    else:
        default_version = f"{default_version.major}.{default_version.minor + 1}.0"

    # Now let's ask nicely if that's the right one.
    version = input(f"Which version are you releasing? [{default_version}]")
    if len(version) == 0:
        version = default_version

    print(f"Updating version to {version}.")
    global_version_update(version, patch=patch)


def post_release_work():
    """Do all the necessary post-release steps."""
    # First let's get the current version
    current_version = get_version()
    dev_version = f"{current_version.major}.{current_version.minor + 1}.0.dev0"
    current_version = current_version.base_version

    # Check with the user we got that right.
    version = input(f"Which version are we developing now? [{dev_version}]")
    if len(version) == 0:
        version = dev_version

    print(f"Updating version to {version}.")
    global_version_update(version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--post_release",
        action="store_true",
        help="Whether this is pre or post release.",
    )
    parser.add_argument("--patch", action="store_true", help="Whether or not this is a patch release.")
    args = parser.parse_args()
    if not args.post_release:
        pre_release_work(patch=args.patch)
    elif args.patch:
        print("Nothing to do after a patch :-)")
    else:
        post_release_work()
