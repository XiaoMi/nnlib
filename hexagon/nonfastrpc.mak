# This is a -*- Makefile -*-

ifdef V62
Q6VERSION ?= v62
else
Q6VERSION ?= v60
endif

include hexagon/files.mak
ifeq (linux,$(OS))
CC := hexagon-linux-clang -m$(Q6VERSION)
BOOTER := 
else
CC := hexagon-clang
Q6_RTOS_INSTALL := /prj/qct/coredev/hexagon/sitelinks/arch/pkg/h2/x86_64/$(Q6VERSION)_stable
INCPATH := -I$(Q6_RTOS_INSTALL)/include
LDPATH := -L$(Q6_RTOS_INSTALL)/lib
BOOTER := $(Q6_RTOS_INSTALL)/bin/booter
LDFLAGS := -m$(Q6VERSION) -Wl,--defsym=HEAP_SIZE=0x40000000 -Wl,--section-start=.start=0x01000000 $(LDPATH) -moslib=h2 
endif


TEST_C_SRCS = test/graph_app.c \
test/graphmain.c \
test/graphinfo.c \
test/options.c \
test/imagenet_info.c \
$(GRAPHINIT)

#CFLAGS = -DSNPE_TEST

APP_LOOPS := 1

RUN_OPTIONS := --iters $(APP_LOOPS)

C_OBJS = $(HEXAGON_NN_C_SRCS:.c=.o) $(TEST_C_SRCS:.c=.o)
ASM_OBJS = $(HEXAGON_NN_ASM_SRCS:.S=.o)
HEXAGON_NN_OBJS = $(HEXAGON_NN_C_SRCS:.c=.o) $(HEXAGON_NN_ASM_SRCS:.S=.o)

ifeq (linux,$(OS))
CFLAGS += -DAPP_LOOPS=$(APP_LOOPS) -Wall -Werror -DUSE_OS_LINUX -Ihexagon/include -Iinterface -O3 $(INCPATH)
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

LDLIBS = -lm

TESTNAME = nn_test

$(TESTNAME): $(C_OBJS) $(ASM_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

.S.o:
	$(CC) $(ASFLAGS) -c -o $@ $<

clean:
	rm -f $(C_OBJS) $(ASM_OBJS) $(TESTNAME) pa_dump.core.* stats_dump.iss.* pmu_stats.txt gmon-*.out test/keyboard_299.dat

TESTFILE = test/keyboard_299.dat

test/keyboard_299.dat: test/keyboard_299x299.jpg
	python scripts/imagedump.py test/keyboard_299x299.jpg test/keyboard_299.dat

sim: $(TESTNAME) $(TESTFILE)
	archsim --magic_angel --quiet --profile $(SIM_OPTIONS) $(BOOTER) $(TESTNAME) $(RUN_OPTIONS) $(TESTFILE)

profile:
	hexagon-gprof $(TESTNAME) gmon* | tee gprof.txt

#include unittest/unittest.mak

checkin_test: unit_test_sim

all_clean: clean unit_test_clean

