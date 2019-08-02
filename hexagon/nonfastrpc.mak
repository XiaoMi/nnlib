# This is a -*- Makefile -*-


CFLAGS += -g -MMD -MP

ifeq ($(V65), 1)
CFLAGS += -DV65
endif

ifeq ($(V66), 1)
CFLAGS += -DV66
endif

ifeq ($(SCRATCH), 1)
CFLAGS += -DSCRATCH
endif
#DEBUG_MEM=1


# Test whether this hexagon-clang version has -mhvx-double option
HVX_OPTION_NO_DOUBLE_TEST = hexagon-clang -mhvx -mhvx-double -Wall -Werror -c -o test/empty.o test/empty.c 2> /dev/null; echo $$?
HVX_OPTION_NO_DOUBLE := $(shell $(HVX_OPTION_NO_DOUBLE_TEST))
HVX_OPTION_NO_128B_TEST = hexagon-clang -mhvx -mhvx-length=128B -Wall -Werror -c -o test/empty.o test/empty.c 2> /dev/null; echo $$?
HVX_OPTION_NO_128B := $(shell $(HVX_OPTION_NO_128B_TEST))
ifeq ($(HVX_OPTION_NO_DOUBLE), 1)
  ifeq ($(HVX_OPTION_NO_128B), 1)
  else
    HVX_OPTION = -mhvx -mhvx-length=128B
  endif
else
HVX_OPTION = -mhvx -mhvx-double
endif


ifdef V66
Q6VERSION ?= v66
NUM_THREADS ?= 4
CFLAGS += $(HVX_OPTION)
ASFLAGS += $(HVX_OPTION)
AXI_BUS_PENALTY := 310
else
ifdef V65
Q6VERSION ?= v65
NUM_THREADS ?= 2
CFLAGS += $(HVX_OPTION)
ASFLAGS += $(HVX_OPTION)
AXI_BUS_PENALTY := 242
else
ifdef V62
Q6VERSION ?= v62
NUM_THREADS ?= 2
CFLAGS += $(HVX_OPTION)
ASFLAGS += $(HVX_OPTION)
AXI_BUS_PENALTY := 178
else
Q6VERSION ?= v60
NUM_THREADS ?= 2
# for v60, default is -mhvx (single) - explicitly specifying hvx-double for compiler
CFLAGS += $(HVX_OPTION)
ASFLAGS += $(HVX_OPTION)
AXI_BUS_PENALTY := 167
endif
endif
endif

DEVHEXARCH_LOC ?= /prj/qct/coredev/hexagon/sitelinks/arch


include hexagon/files.mak
ifeq (linux,$(OS))
CC := hexagon-linux-clang -m$(Q6VERSION)
HEXAGON_NN_C_SRCS += hexagon/src/pmu_control_linux.c
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbb_circ_d64_v65_h.S hexagon/asm_src/gvconv2dbbb_circ_d32_v65_h.S hexagon/asm_src/repstream2_h.S
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbb_circ6_d64_v65_h.S hexagon/asm_src/gvconv2dbbb_circ6_d32_v65_h.S hexagon/asm_src/repstreamN_h.S
BOOTER := 
LDFLAGS := -g -mv60 -Wl,--section-start=.init=0x01000000 $(LDOPTS)
export PATH:=${PATH}:/prj/qct/coredev/hexagon/austin/teams/hexagon-linux/builds/latest/Tools/bin
export PKW_VERSIONS?=hexagon-tools=8.2.02
export PKW_arch?=v60_stable
else
CC := hexagon-clang
ifdef LOCAL_RTOS_INSTALL_DIR
Q6_RTOS_INSTALL := $(LOCAL_RTOS_INSTALL_DIR)
else
Q6_RTOS_INSTALL := $(DEVHEXARCH_LOC)/pkg/h2/x86_64/$(Q6VERSION)_stable
#Q6_RTOS_INSTALL := ../copied/h2/$(Q6VERSION)_stable
endif
INCPATH := -I$(Q6_RTOS_INSTALL)/include
LDPATH := -L$(Q6_RTOS_INSTALL)/lib
BOOTER := $(Q6_RTOS_INSTALL)/bin/booter
LDFLAGS := -m$(Q6VERSION) -Wl,--defsym=HEAP_SIZE=0x40000000 -Wl,--section-start=.start=0x01000000 $(LDPATH) -moslib=h2  $(LDOPTS)
endif


ifdef V66
CFLAGS += -DHEXAGON_V66=1
ASFLAGS += -DFAST_16B_CONV
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbb_d32_h_v66.S hexagon/asm_src/gvconv2dbbb_d32_s1f_h_v66.S hexagon/asm_src/gvconv2dbbb_d16_s1f_h_v66.S
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbbs1x4_d32_h_v66.S
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2db2b2b2_d32_h_v66.S
SIM_OPTIONS += --core V66G_1024
else
ifdef V65
CFLAGS += -DHEXAGON_V65=1
HEXAGON_NN_ASM_SRCS += hexagon/asm_src/gvconv2dbbb_circ_d64_v65_h.S \
                       hexagon/asm_src/gvconv2dbbb_circ_d32_v65_h.S \
                       hexagon/asm_src/repstream2_h.S \
                       hexagon/asm_src/gvconv2dbbb_circ6_d64_v65_h.S \
                       hexagon/asm_src/gvconv2dbbb_circ6_d32_v65_h.S \
                       hexagon/asm_src/repstreamN_h.S \
                       hexagon/asm_src/gvconv2db2b2b2_d32_h_v65.S \
                       hexagon/asm_src/gvconv2db2b2b2u_d32_h_v65.S
SIM_OPTIONS += --core V65A_512
endif
endif

TEST_C_SRCS = test/graph_app.c \
test/graphmain.c \
test/graphinfo.c \
test/options.c \
test/imagenet_info.c

#CFLAGS = -DSNPE_TEST

APP_LOOPS := 2

RUN_OPTIONS := --iters $(APP_LOOPS) --node_perf 1

C_OBJS = $(HEXAGON_NN_C_SRCS:.c=.o) $(TEST_C_SRCS:.c=.o)
ASM_OBJS = $(HEXAGON_NN_ASM_SRCS:.S=.o)
HEXAGON_NN_OBJS = $(HEXAGON_NN_C_SRCS:.c=.o) $(HEXAGON_NN_ASM_SRCS:.S=.o)
DEPS = $(C_OBJS:.o=.d)

ifeq (linux,$(OS))
CFLAGS += -DAPP_LOOPS=$(APP_LOOPS) -Wall -DUSE_OS_LINUX -Ihexagon/include -Iinterface -O2 $(INCPATH)
else
CFLAGS += -m$(Q6VERSION) -Wall -Werror -DAPP_LOOPS=$(APP_LOOPS) -DUSE_OS_H2 -Ihexagon/include -Iinterface -O3 $(INCPATH)
ASFLAGS += -m$(Q6VERSION)
endif

ifdef NO_VERBOSE
CFLAGS += -DNO_VERBOSE
endif

ifdef BAIL_EARLY
CFLAGS += -DBAIL_EARLY
endif

ifdef SHOWY_DEBUG
CFLAGS += -DSHOWY_DEBUG
endif
ifdef LINUX_DEBUG
CFLAGS += -DLINUX_DEBUG
endif
ifdef TIMING_MODE
CFLAGS += -DNN_LOG_MAXLEV=-1
endif

ifdef CANARIES
#CFLAGS += -DDEBUG_MEMORY_CANARIES=1 -DCHECK_CANARIES=1 -DDEBUG_MEM=1
CFLAGS += -DDEBUG_MEMORY_CANARIES=1 
ARCHSTRING := ARCHSTRING="--poison_check 1 --poison_check_value 0xcafebeef"
# HACK: This overrides all SIM_OPTIONS in favor of just setting our core correctly! 
SIM_OPTIONS := --core V66G_1024
endif

ifdef DEBUG_MEM
CFLAGS += -DDEBUG_MEM=1
endif

DEBUG_OUTPUT = $(patsubst %.dat,%.dat.png,$(wildcard debug/*.dat))
DEBUG_OUTPUT += $(patsubst %.dot,%.dot.png,$(wildcard debug/*.dot))

LDLIBS = -lm

DEFAULT_BASE := inceptionv3
DEFAULT_BIN := $(DEFAULT_BASE).elf

#PERF_SUITE_BASE := cnns_googlenet_v1 cnns_lenet cnns_resnet_50 cnns_squeezenet custom_facebook_conv_relus custom_sensetime_attribute_conv custom_sensetime_face_detect_conv custom_zongmu_convs inception_resnet_v2 inceptionv3 inceptionv3_rank8 mobilenet lens_lstm_20 lens_lstm_100 lens_lstm_300
PERF_SUITE_BASE := cnns_googlenet_v1 cnns_lenet cnns_resnet_50 cnns_squeezenet custom_facebook_conv_relus custom_sensetime_attribute_conv custom_sensetime_face_detect_conv custom_zongmu_convs inception_resnet_v2 inceptionv3 inceptionv3_rank8 mobilenet
PERF_SUITE_ELF := $(patsubst %,%.elf,$(PERF_SUITE_BASE))
PERF_SUITE_SIM := $(patsubst %,%.sim,$(PERF_SUITE_BASE))
PERF_SUITE_TIMING := $(patsubst %,%.timing,$(PERF_SUITE_BASE))


default: inceptionv3.elf
default: $(DEFAULT_BIN)
perf_suite: perf_suite_sim perf_suite_timing
perf_suite_elf: $(PERF_SUITE_ELF)
perf_suite_sim: $(PERF_SUITE_SIM)
perf_suite_timing: $(PERF_SUITE_TIMING)


ALL_OBJS = $(C_OBJS) $(ASM_OBJS)


$(ALL_OBJS): interface/ops.def interface/hexagon_nn_ops.h

$(C_OBJS): $(addprefix hexagon/include/, nn_graph.h nn_graph_ops.h nn_graph_types.h nn_graph_im2col.h nn_graph_if.h nn_asm_ops.h nn_graph_os.h nn_atomic.h)

objs/%.o: apps/%.c
	$(CC) -c $(CPPFLAGS) $(CFLAGS) -o $@ $^

%.elf: $(ALL_OBJS) objs/%.o
	$(CC) $(LDFLAGS) -o $@ $^

.S.o:
	$(CC) $(ASFLAGS) -c -o $@ $<

clean:
	rm -f $(ALL_OBJS) $(DEPS) *.elf objs/* pa_dump.core.* stats_dump.iss.* pmu_stats.txt gmon-*.out *.html *.timing *.sim

TESTFILE = test/panda_299x299.dat

#%.dat: %.jpg
#	-python scripts/imagedump.py $< $@

%.sim: %.elf $(TESTFILE)
	WIDTH=299; \
	HEIGHT=299; \
	TESTFILE=$(TESTFILE); \
	if [ "$*" == "mobilenet" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_googlenet_v1" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_resnet_50" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_vgg_16" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_vgg_19" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	echo $(ARCHSTRING) archsim --ahb_base 0xe0000000 --axi2_base 0xe8000000 --axi2_size 0x01000 --magic_angel --quiet --profile $(SIM_OPTIONS) $(BOOTER) $< $(RUN_OPTIONS) --elementsize 1 --width $$WIDTH --height $$HEIGHT --depth 3 $$TESTFILE | tee $@ ; \
	$(ARCHSTRING) archsim --ahb_base 0xe0000000 --axi2_base 0xe8000000 --axi2_size 0x01000 --magic_angel --quiet --profile $(SIM_OPTIONS) $(BOOTER) $< $(RUN_OPTIONS) --elementsize 1 --width $$WIDTH --height $$HEIGHT --depth 3 $$TESTFILE | tee $@

%.timing: %.elf $(TESTFILE)
	mkdir -p $*
	WIDTH=299; \
	HEIGHT=299; \
	TESTFILE=../$(TESTFILE); \
	if [ "$*" == "mobilenet" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=../test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_googlenet_v1" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_resnet_50" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_vgg_16" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	if [ "$*" == "cnns_vgg_19" ] ; then WIDTH=224; HEIGHT=224; TESTFILE=test/panda_224x224.dat ; fi; \
	cd $* && archsim --timing --fastforward 3 --packet_analyze packets.json --simulated_returnval --magic_angel --axibuspenalty $(AXI_BUS_PENALTY) --quiet --profile $(SIM_OPTIONS) $(BOOTER) ../$< --iters 2 --elementsize 1 --width $$WIDTH --height $$HEIGHT --depth 3 --benchmark 1 $$TESTFILE | tee ../$@
	PATH=/pkg/qct/software/python/2.7.9/bin:${PATH} /prj/qct/coredev/hexagon/sitelinks/arch/pkg/hexagon-tools/x86_64/v66_stable/Tools/lib/profiler/proftool.py --packet_analyze $*/packets.json $< $*.html

%.quick_sim: %.elf $(TESTFILE)
	hexagon-sim $(BOOTER) -- $< $(RUN_OPTIONS) $(TESTFILE) | tee $@

%.bus_bw_sim: %.elf $(TESTFILE)
	archsim --magic_angel --quiet --profile --core V66G_1024 $(DEVHEXARCH_LOC)/pkg/h2/x86_64/v66_stable/bin/booter $< --iters 1 --elementsize 1 --width 299 --height 299 --depth 3 --bus_bw 1 $(TESTFILE) | tee $@

debug: $(DEBUG_OUTPUT)

%.dat.png: %.dat
	./scripts/dat2img.py $<

%.dot.png: %.dot
	dot -Tpng < $< > $@

slurm_sanity:
	echo "Checking for correct OS selected"
	[ $(OS) = "linux" ]

%.slurm: slurm_sanity %.elf $(TESTFILE)
	# Run the testcase on the Slurm swarm (a bunch of dragonboards)
	source /prj/dsp/austin/hexagon_farm/rootfs/users/rkuo/slurm/setenv.bash slurm && export PATH=/pkg/qct/software/llvm/build_tools/bin/:${PATH} && srun $(word 2,$^) --iters 1 --elementsize 1 --width 299 --height 299 --depth 3 $(TESTFILE) | tee $@


%.profile: %.elf
	hexagon-gprof $< gmon* | tee $@

include unittest/unittest.mak

include cpptest/cpptest.mak
include cannedtest/cannedtest.mak
include canphone/canphone.mak

checkin_test: unit_test_sim

all_clean: clean unit_test_clean cpp_test_clean canphone_clean canned_test_clean snpetest_clean goog_test_clean

# snpetest targets - build and clean

include snpetest/snpetest.mak

snpetest: snpetest_sim

snpetest_clean: snpetest_sim_clean

SNPE_MINI_TARGETS := $(shell cpp -P -D "TEST(name)=name.snpelog" snpetest/tests.def)
snpetest_minis: $(SNPE_MINI_TARGETS)
	echo SUCCESS

.PRECIOUS: %.elf objs/%.o

sim: $(DEFAULT_BASE).sim
slurm: $(DEFAULT_BASE).slurm
timing: $(DEFAULT_BASE).timing


-include googtest/googtest.mak

-include $(DEPS)
