# builds the static/dynamic skel
ifeq (1,$(V_dynamic))
BUILD_DLLS = libhexagon_nn_skel
else
$(error Static library not supported)
endif

include $(QURT_IMAGE_DIR)/qurt_libs.min

include hexagon/files.mak

CC_FLAGS += -DNN_LOG_MAXLEV=9 -DUSE_OS_QURT $(MHVX_DOUBLE_FLAG) -Ihexagon/include
ASM_FLAGS += $(MHVX_DOUBLE_FLAG)
CXX_FLAGS += $(MHVX_DOUBLE_FLAG)

ifeq ($(V65), 1)
CC_FLAGS += -DV65=1 -DHEXAGON_V65=1 -mv65
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbb_circ_d64_v65_h.S \
	hexagon/asm_src/gvconv2dbbb_circ_d32_v65_h.S \
	hexagon/asm_src/repstream2_h.S
endif


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
