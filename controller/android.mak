$(info ------------------------------------------)
$(info --- V = $(V))
$(info --- GLUE_DIR = $(GLUE_DIR))
$(info --- HEXAGON_SDK_ROOT = $(HEXAGON_SDK_ROOT))
$(info ------------------------------------------)

INCDIRS += controller
BUILD_DLLS=libhexagon_controller

hexagon_controller_lib_QAICIDLS += interface/hexagon_nn
hexagon_controller_lib_C_SRCS += $V/hexagon_nn_stub

hexagon_controller_lib_LIBS += rpcmem
hexagon_controller_lib_DLLS += libcdsprpc
hexagon_controller_lib_LD_FLAGS += -llog

libhexagon_controller_QAICIDLS += $(hexagon_controller_lib_QAICIDLS)
libhexagon_controller_C_SRCS += $(hexagon_controller_lib_C_SRCS) controller/config
libhexagon_controller_DLLS += $(hexagon_controller_lib_DLLS)
libhexagon_controller_LIBS += $(hexagon_controller_lib_LIBS)
libhexagon_controller_LD_FLAGS += $(hexagon_controller_lib_LD_FLAGS)

BUILD_COPIES = \
   $(DLLS) \
   $(EXES) \
   $(LIBS) \
   $(SHIP_DIR)/ ;
