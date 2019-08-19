# SMART WRAPPER FLAG to turn on GRAPH_WRAPPER
ifeq ($(SMART_WRAPPER), 1)
        GRAPH_WRAPPER = 1
endif

# stand-alone executable
BUILD_EXES+=graph_app
graph_app_QAICIDLS += interface/hexagon_nn \

#GRAPHINIT=test/graphinit_small
COMPILE_GRAPHINIT := $(GRAPHINIT:.c=)

ifeq ($(GRAPH_WRAPPER), 1)
    INCDIRS += interface
endif

graph_app_C_SRCS += \
    test/graph_app \
    test/graphmain \
    test/graphinfo \
    test/options \
    test/imagenet_info \
    test/append_const_node_large_array \
    $(COMPILE_GRAPHINIT) \

ifeq ($(GRAPH_WRAPPER), 1)
    graph_app_C_SRCS += hexagon/host/hexnn_dsp_api_impl
    graph_app_CPP_SRCS += hexagon/host/hexnn_graph_wrapper

    ifeq ($(SMART_WRAPPER), 1)
        graph_app_C_SRCS += hexagon/host/hexnn_dsp_smart_wrapper_api hexagon/host/hexnn_dsp_domains_api_impl
    else
        graph_app_C_SRCS += hexagon/host/hexnn_dsp_api
    endif
else
    graph_app_C_SRCS += $V/hexagon_nn_stub
endif

graph_app_C_SRCS += $(TESTDATA:.c=)

ifeq ($(GRAPH_WRAPPER), 1)
ifeq ($(V_aarch64), 1)
    graph_app_DLLS += $(ANDROID_GLIBSTDC_DIR)/libs/arm64-v8a/libgnustl_shared
else
    graph_app_DLLS += $(ANDROID_GLIBSTDC_DIR)/libs/armeabi-v7a/libgnustl_shared
endif
endif

graph_app_LIBS += rpcmem
ifeq ($(CDSP_FLAG), 1)
	graph_app_DLLS += libcdsprpc
	CC_FLAGS += -DCDSP_FLAG
else
	graph_app_DLLS += libadsprpc
endif
graph_app_LD_FLAGS += -llog
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
