#!/usr/bin/env bash

set -e -x

rm -rf hexagon_Release_dynamic_toolv83_v60
rm -rf hexagon_Release_dynamic_toolv83_v65
rm -rf hexagon_Release_dynamic_toolv83_v66
rm -rf android_Release
rm -rf android_Release_aarch64

make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv83_v60
make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv83_v65 V65=1
make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv83_v66 V66=1
make tree VERBOSE=1 V=android_Release CDSP_FLAG=1
make tree VERBOSE=1 V=android_Release_aarch64 CDSP_FLAG=1

if [ -n "${MACE_PATH}" ]; then
    cp hexagon_Release_dynamic_toolv83_v60/ship/libhexagon_nn_skel.so ${MACE_PATH}/third_party/nnlib/v60/
    cp hexagon_Release_dynamic_toolv83_v65/ship/libhexagon_nn_skel.so ${MACE_PATH}/third_party/nnlib/v65/
    cp hexagon_Release_dynamic_toolv83_v66/ship/libhexagon_nn_skel.so ${MACE_PATH}/third_party/nnlib/v66/
    cp interface/ops.def ${MACE_PATH}/third_party/nnlib/ops.h
    cp android_Release/ship/libhexagon_controller.so ${MACE_PATH}/third_party/nnlib/armeabi-v7a/
    cp android_Release_aarch64/ship/libhexagon_controller.so ${MACE_PATH}/third_party/nnlib/arm64-v8a/
fi
