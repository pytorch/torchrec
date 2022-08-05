#!/bin/bash
cd $(dirname $0)/../
export CIBW_BEFORE_BUILD="tools/before_linux_build.sh"

# Use env CIBW_BUILD="cp*-manylinux_x86_64" tools/build_wheels.sh to build
# all kinds of CPython.
export CIBW_BUILD=${CIBW_BUILD:-"cp39-manylinux_x86_64"}

# Do not auditwheels since tde uses torch's shared libraries.
export CIBW_REPAIR_WHEEL_COMMAND="mv {wheel} {dest_dir}"

cibuildwheel --platform linux --archs x86_64
