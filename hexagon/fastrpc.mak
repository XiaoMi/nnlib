# builds the static/dynamic skel
BUILD_LIBS = libhexagon_nn_skel

ifeq (1,$(V_dynamic))
BUILD_DLLS = libhexagon_nn_skel
endif

include $(QURT_IMAGE_DIR)/qurt_libs.min

include hexagon/files.mak

CC_FLAGS += -DNN_LOG_MAXLEV=9 -DUSE_OS_QURT $(MHVX_DOUBLE_FLAG) -Ihexagon/include
ASM_FLAGS += $(MHVX_DOUBLE_FLAG)
CXX_FLAGS += $(MHVX_DOUBLE_FLAG)

ifdef $(NUM_HVX)
CC_FLAGS += -DNUM_VECTOR_THREADS=$(NUM_HVX)
endif

ifeq ($(DO_NOT_CONFIG_PMU), 1)
CC_FLAGS += -DNO_PMU_CONFIG
endif

ifeq ($(USE_FPIC), 1)
CC_FLAGS += -fPIC
endif

ifeq (1,$(USE_SIMULATOR))
CC_FLAGS += -DNN_GRAPH_ON_SIMULATOR
endif

ifeq ($(V65), 1)
CC_FLAGS += -DV65=1 -DHEXAGON_V65=1 -mv65
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbb_circ_d64_v65_h.S \
	hexagon/asm_src/gvconv2dbbb_circ_d32_v65_h.S \
	hexagon/asm_src/repstream2_h.S \
	hexagon/asm_src/gvconv2dbbb_circ6_d32_v65_h.S \
	hexagon/asm_src/gvconv2dbbb_circ6_d64_v65_h.S \
	hexagon/asm_src/repstreamN_h.S \
	hexagon/asm_src/gvconv2db2b2b2_d32_h_v65.S \
	hexagon/asm_src/gvconv2db2b2b2u_d32_h_v65.S
endif

ifeq ($(V66), 1)
CC_FLAGS += -DV66=1 -DHEXAGON_V66=1 -mv66
ASM_FLAGS += -DFAST_16B_CONV
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbbs1x4_d32_h_v66.S \
	hexagon/asm_src/gvconv2dbbb_d32_s1f_h_v66.S \
	hexagon/asm_src/gvconv2dbbb_d16_s1f_h_v66.S \
	hexagon/asm_src/gvconv2dbbb_d32_h_v66.S \
	hexagon/asm_src/gvconv2db2b2b2_d32_h_v66.S
endif


# Use -O0 temporarily when trying to debug C code.
#_OPT = -O0

INCDIRS += \
  interface

libhexagon_nn_skel_QAICIDLS = interface/hexagon_nn
libhexagon_nn_skel_C_SRCS += $V/hexagon_nn_skel $V/hexagon_nn_domains_skel \
$(HEXAGON_NN_C_SRCS:.c=)

libhexagon_nn_skel.ASM_SRCS +=  \
$(HEXAGON_NN_ASM_SRCS)

# copy final build products to the ship directory
BUILD_COPIES = \
    interface/hexagon_nn.idl \
    interface/hexagon_nn_ops.h \
    interface/ops.def \
   $(DLLS) \
   $(EXES) \
   $(LIBS) \
   $(SHIP_DIR)/ ;

-include $(wildcard $(V)/*.d)
