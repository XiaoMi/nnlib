# stand-alone gemm executable
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
$(COMPILE_GRAPHINIT) \
                       $V/hexagon_nn_stub \
                       $V/dspCV_stub

graph_app_C_SRCS += $(TESTDATA:.c=)

graph_app_DLLS += libadsprpc 
graph_app_LIBS += rpcmem adspmsgd
graph_app_LD_FLAGS += -llog
CC_FLAGS += -Iinterface
graph_app_DEFINES += VERIFY_PRINT_ERROR

# copy final build products to the ship directory
BUILD_COPIES = \
   $(DLLS) \
   $(EXES) \
   $(LIBS) \
   $(SHIP_DIR)/ ;
