include hexagon/files.mak
ifeq (linux,$(OS))
CC := hexagon-linux-clang -mv60
BOOTER := 
else
CC := hexagon-clang -moslib=h2 -mv60
Q6_RTOS_INSTALL := /prj/qct/coredev/hexagon/sitelinks/arch/pkg/h2/x86_64/v60_latest
INCPATH := -I$(Q6_RTOS_INSTALL)/include
LDPATH := -L$(Q6_RTOS_INSTALL)/lib
BOOTER := $(Q6_RTOS_INSTALL)/bin/booter
LDFLAGS := -Wl,--defsym=HEAP_SIZE=0x40000000 -Wl,--section-start=.start=0x01000000 $(LDPATH)
endif


TEST_C_SRCS = test/graph_app.c \
test/graphmain.c \
test/graphinfo.c \
$(TESTDATA) \
$(GRAPHINIT)

#CFLAGS = -DSNPE_TEST

C_OBJS = $(HEXAGON_NN_C_SRCS:.c=.o) $(TEST_C_SRCS:.c=.o)
ASM_OBJS = $(HEXAGON_NN_ASM_SRCS:.S=.o)

ifeq (linux,$(OS))
CFLAGS += -DUSE_OS_LINUX -Ihexagon/include -Iinterface -O3 $(INCPATH)
else
CFLAGS += -DUSE_OS_H2 -Ihexagon/include -Iinterface -O3 $(INCPATH)
endif
 
LDLIBS = -lm

TESTNAME = nn_test

$(TESTNAME): $(C_OBJS) $(ASM_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

clean:
	rm -f $(C_OBJS) $(ASM_OBJS) $(TESTNAME)

sim:
	archsim --magic_angel --quiet $(BOOTER) $(TESTNAME)

