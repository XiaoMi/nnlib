ifneq (1,$(V_dynamic))
ifneq ($(V_ARCH),$(filter $(V_ARCH),v4 v5 v55 v56 v60 v62))

# stand-alone executable
BUILD_EXES+=graph_app_q

include $(QURT_IMAGE_DIR)/qurt_libs.min

graph_app_q_QAICIDLS += interface/hexagon_nn \
                        $(MAKE_D_DSPCV_INCDIR)/dspCV

#GRAPHINIT=test/graphinit_small
COMPILE_GRAPHINIT := $(GRAPHINIT:.c=)


graph_app_q_C_SRCS += \
test/graph_app \
test/graphmain \
test/graphinfo \
test/options \
test/imagenet_info \
test/append_const_node_large_array \
$(COMPILE_GRAPHINIT) \

graph_app_q_C_SRCS += $(TESTDATA:.c=)

graph_app_q_LIBS += $(QURT_INIT_LIBS) $(QURT_LINK_LIBS)
graph_app_q_LIBS += apps_mem_heap_stub rpcmem test_util atomic libdspCV_skel libhexagon_nn_skel
graph_app_q_LIBS += $(QURT_FINI_LIBS)

CC_FLAGS += -Iinterface
graph_app_q_DEFINES += VERIFY_PRINT_ERROR

# defining ahb address is a temporary workaround for 8.1.04 tools, to be fixed in 8.1.05. See HEXSUPPORT 1854.
QEXE_EXEC_SIM_OPTIONS +=--dsp_clock 1000 --ahb:lowaddr 0xc0000000 --ahb:highaddr 0xc0ffffff
QEXE_EXEC_CMD_OPTIONS +=--height 299 --width 299 --depth 3 --elementsize 1 test/panda_299x299.dat

endif
endif
