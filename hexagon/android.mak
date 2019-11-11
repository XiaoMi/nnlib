$(info ------------------------------------------)
$(info --- V = $(V))
$(info --- GLUE_DIR = $(GLUE_DIR))
$(info --- HEXAGON_SDK_ROOT = $(HEXAGON_SDK_ROOT))
$(info ------------------------------------------)

ANDROID_GLIBSTDC_DIR = $(ANDROID_NDK_HOME)/sources/cxx-stl/llvm-libc++
ifeq ($(GRAPH_WRAPPER), 1)
    INCDIRS += interface
endif
BUILD_DLLS=libhexagon_controller

hexagon_controller_lib_QAICIDLS += interface/hexagon_nn

hexagon_controller_lib_C_SRCS += hexagon/host/hexnn_dsp_controller

ifeq ($(GRAPH_WRAPPER), 1)
    hexagon_controller_lib_C_SRCS += hexagon/host/hexnn_dsp_api \
                                     hexagon/host/hexnn_dsp_api hexagon/host/hexnn_dsp_api_impl
    hexagon_controller_lib_CPP_SRCS += hexagon/host/hexnn_graph_wrapper
else
    hexagon_controller_lib_C_SRCS += $V/hexagon_nn_stub
endif

ifeq ($(GRAPH_WRAPPER), 1)
ifeq ($(V_aarch64), 1)
    hexagon_controller_lib_DLLS += $(ANDROID_GLIBSTDC_DIR)/libs/arm64-v8a/libc++_shared.so
else
    hexagon_controller_lib_DLLS += $(ANDROID_GLIBSTDC_DIR)/libs/armeabi-v7a/libc++_shared.so
endif
endif


hexagon_controller_lib_LIBS += rpcmem
hexagon_controller_lib_DLLS += libcdsprpc
hexagon_controller_lib_LD_FLAGS += -llog

libhexagon_controller_QAICIDLS += $(hexagon_controller_lib_QAICIDLS)
libhexagon_controller_C_SRCS += $(hexagon_controller_lib_C_SRCS)
libhexagon_controller_CPP_SRCS += $(hexagon_controller_lib_CPP_SRCS)
libhexagon_controller_DLLS += $(hexagon_controller_lib_DLLS)
libhexagon_controller_LIBS += $(hexagon_controller_lib_LIBS)
libhexagon_controller_LD_FLAGS += $(hexagon_controller_lib_LD_FLAGS)

$(info --- libhexagon_controller_DLLS = $(libhexagon_controller_DLLS))

BUILD_EXES += controller_test

controller_test_QAICIDLS += interface/hexagon_nn
controller_test_C_SRCS += $V/hexagon_nn_stub hexagon/host/test
controller_test_DLLS += libhexagon_controller libcdsprpc
controller_test_LD_FLAGS += -llog

BUILD_COPIES = \
   interface/hexnn_graph_wrapper_interface.h \
   hexagon/host/hexnn_dsp_controller.h \
   $(DLLS) \
   $(EXES) \
   $(LIBS) \
   $(SHIP_DIR)/ ;
