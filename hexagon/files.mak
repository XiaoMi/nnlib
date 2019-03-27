C_SRCS := hexagon/hexagon_nn_c_srcs.txt
ASM_SRCS := hexagon/hexagon_nn_asm_srcs.txt

HEXAGON_NN_C_SRCS := $(shell cat ${C_SRCS})
HEXAGON_NN_ASM_SRCS := $(shell cat ${ASM_SRCS})

