# stand-alone executable
BUILD_EXES+=graph_app
graph_app_QAICIDLS += interface/hexagon_nn \
                        $(MAKE_D_DSPCV_INCDIR)/dspCV

#GRAPHINIT=test/graphinit_small
COMPILE_GRAPHINIT := $(GRAPHINIT:.c=)


graph_app_C_SRCS += \
    test/graph_app \
    test/graphmain \
    test/graphinfo \
    test/options \
    test/imagenet_info \
    test/append_const_node_large_array \
    $(COMPILE_GRAPHINIT) \
    $(V)/hexagon_nn_stub \
    $(V)/dspCV_stub

graph_app_C_SRCS += $(TESTDATA:.c=)

graph_app_LIBS += rpcmem
ifeq ($(CDSP_FLAG), 1)
	graph_app_DLLS += libcdsprpc
	CC_FLAGS += -DCDSP_FLAG
else
	graph_app_DLLS += libadsprpc
endif
CC_FLAGS += -Iinterface
graph_app_DEFINES += VERIFY_PRINT_ERROR

# copy final build products to the ship directory
BUILD_COPIES = \
    interface/hexagon_nn.idl \
    interface/hexagon_nn_ops.h \
    interface/ops.def \
   $(DLLS) \
   $(EXES) \
   $(LIBS) \
   $(SHIP_DIR)/ 
