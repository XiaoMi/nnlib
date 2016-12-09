# builds the static/dynamic skel
ifeq (1,$(V_dynamic))
BUILD_DLLS = libhexagon_nn_skel
else
$(error Static library not supported)
endif

include $(QURT_IMAGE_DIR)/qurt_libs.min

include hexagon/files.mak

CC_FLAGS += -DUSE_OS_QURT $(MHVX_DOUBLE_FLAG) -Ihexagon/include

# Use -O0 temporarily when trying to debug C code.
#_OPT = -O0

INCDIRS += \
  interface

# Needs dspCV lib for worker pool
libhexagon_nn_skel_DLLS+=libdspCV_skel

libhexagon_nn_skel_QAICIDLS = interface/hexagon_nn
libhexagon_nn_skel_C_SRCS += $V/hexagon_nn_skel \
$(HEXAGON_NN_C_SRCS:.c=)

libhexagon_nn_skel.ASM_SRCS +=  \
$(HEXAGON_NN_ASM_SRCS)

# copy final build products to the ship directory
BUILD_COPIES = \
   $(DLLS) \
   $(EXES) \
   $(LIBS) \
   $(SHIP_DIR)/ ;
