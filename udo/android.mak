# stand-alone gemm executable
BUILD_EXES+=udo_sample_exe
udo_sample_exe_QAICIDLS += interface/hexagon_nn \

udo_sample_exe_C_SRCS += \
    udo/udoSampleCallFlow \

udo_sample_exe_C_SRCS += $V/hexagon_nn_stub \

udo_sample_exe_C_SRCS += hexagon/host/UdoFlatten

udo_sample_exe_LIBS += rpcmem


ifeq ($(CDSP_FLAG), 1)
	udo_sample_exe_DLLS += libcdsprpc
	CC_FLAGS += -DCDSP_FLAG
else
	udo_sample_exe_DLLS += libadsprpc
endif

udo_sample_exe_LD_FLAGS += -llog

CC_FLAGS += -Iinterface

udo_sample_exe_DEFINES += VERIFY_PRINT_ERROR

INCDIRS += \
  interface

# copy final build products to the ship directory
BUILD_COPIES = \
    interface/hexagon_nn.idl \
    interface/hexagon_nn_ops.h \
    interface/ops.def \
   $(DLLS) \
   $(EXES) \
   $(LIBS) \
   $(SHIP_DIR)/ ;
