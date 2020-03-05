#!/bin/bash
DEVDIR=/data/local/tmp/udo

adb shell mkdir -p $DEVDIR
adb shell rm -rf $DEVDIR

BINDIR=$DEVDIR/bin
adb shell mkdir -p $BINDIR
adb push android_Release_aarch64/udo_sample_exe $BINDIR/

DSPDIR=$DEVDIR/dsp
adb shell mkdir -p $DSPDIR
adb push hexagon_Release_dynamic_toolv82_v65/ship/libhexagon_nn_skel.so $DSPDIR/
# adb push hexagon_Release_dynamic_toolv82_v66/ship/libhexagon_nn_skel.so $DSPDIR/

UDODIR=$DSPDIR/udoLibs
adb shell mkdir -p $UDODIR
adb push udo/hexagon_Release_dynamic_toolv82_v65/udoExampleImplLib.so $UDODIR/


echo "---------------------------------------------"
echo "Running udo sample exe"
echo "---------------------------------------------"
adb shell "export ADSP_LIBRARY_PATH=\"$DSPDIR/;/system/lib/rfsa/adsp/\" && $BINDIR/udo_sample_exe"

adb shell rm -rf $DEVDIR
