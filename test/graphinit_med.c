
/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
/*
 */
/*
 * This is a non-trivial but non-complicated example
 * of setting up a graph 
 */

#include <hexagon_nn.h>
#include <stdio.h>
#include "../interface/hexagon_nn_ops.h"
typedef int32_t qint32;
typedef int32_t int32;
typedef float float32;
typedef uint8_t quint8;
#define ALIGNED __attribute__((aligned(128)))
void info_for_debug(unsigned int id, const char *name, const char *opname);

#define NN_PAD_ANY NN_PAD_NA

#define OUTPUT_4D(B,H,W,D,ES) \
	{ .rank = 4, .max_sizes = {B,H,W,D}, .elementsize = ES, \
	.zero_offset = 0, .stepsize = 0.0f, }
	



static quint8 data_for_op_1000b[] ALIGNED = {
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
};

static float32 data_for_op_1000c[] ALIGNED = {
  -4.0,
};
static float32 data_for_op_1000d[] ALIGNED = {
  4.0,
};
static int32 data_for_op_1000e[] ALIGNED = {
  -1,
};
static int32 data_for_op_1000f[] ALIGNED = {
  0,
};

static quint8 data_for_op_10010[] ALIGNED = {
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
  128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,
};

static float32 data_for_op_10011[] ALIGNED = {
  -1.5,
};
static float32 data_for_op_10012[] ALIGNED = {
  4.0,
};

static hexagon_nn_output outputs_for_1024a[] = {
 OUTPUT_4D(1,299,299,3,4),
};

static hexagon_nn_input inputs_for_1024b[] = {
 { .src_id = 0x1024a, .output_idx = 0, },
 { .src_id = 0x1000e, .output_idx = 0, },
};
static hexagon_nn_output outputs_for_1024b[] = {
 OUTPUT_4D(1,1,1,268203,4), 
};

static hexagon_nn_input inputs_for_1024c[] = {
 { .src_id = 0x1024b, .output_idx = 0, },
 { .src_id = 0x1000f, .output_idx = 0, },
};
static hexagon_nn_output outputs_for_1024c[] = {
 OUTPUT_4D(1,1,1,1,4), 
};
static hexagon_nn_input inputs_for_1024d[] = {
 { .src_id = 0x1024b, .output_idx = 0, },
 { .src_id = 0x1000f, .output_idx = 0, },
};
static hexagon_nn_output outputs_for_1024d[] = {
 OUTPUT_4D(1,1,1,1,4), 
};

static hexagon_nn_input inputs_for_1024e[] = {
 { .src_id = 0x1024a, .output_idx = 0, },
 { .src_id = 0x1024c, .output_idx = 0, },
 { .src_id = 0x1024d, .output_idx = 0, },
};
static hexagon_nn_output outputs_for_1024e[] = {
 OUTPUT_4D(1,299,299,3,1), //{ .max_size = 268203, }, // 1x299x299x3 x1
 OUTPUT_4D(1,1,1,1,4),
 OUTPUT_4D(1,1,1,1,4),
};

static hexagon_nn_input inputs_for_1024f[] = {
 { .src_id = 0x1024e, .output_idx = 0, },
 { .src_id = 0x1000b, .output_idx = 0, },
 { .src_id = 0x1024e, .output_idx = 1, },
 { .src_id = 0x1024e, .output_idx = 2, },
 { .src_id = 0x1000c, .output_idx = 0, },
 { .src_id = 0x1000d, .output_idx = 0, },
 { .src_id = 0x10250, .output_idx = 0, },
};
static hexagon_nn_output outputs_for_1024f[] = {
 OUTPUT_4D(1,149,149,32,1), //{ .max_size = 2841728, }, // 1x149x149x32 x4
 OUTPUT_4D(1,1,1,1,4),
 OUTPUT_4D(1,1,1,1,4),
};

static hexagon_nn_input inputs_for_10251[] = {
 { .src_id = 0x1024f, .output_idx = 0, },
 { .src_id = 0x1024f, .output_idx = 1, },
 { .src_id = 0x1024f, .output_idx = 2, },
};
static hexagon_nn_output outputs_for_10251[] = {
 OUTPUT_4D(1,149,149,32,1), // { .max_size = 710432, }, // 1x149x149x32 x1
 OUTPUT_4D(1,1,1,1,4),
 OUTPUT_4D(1,1,1,1,4),
};

static hexagon_nn_input inputs_for_10252[] = {
 { .src_id = 0x10251, .output_idx = 0, },
 { .src_id = 0x10010, .output_idx = 0, },
 { .src_id = 0x10251, .output_idx = 1, },
 { .src_id = 0x10251, .output_idx = 2, },
 { .src_id = 0x10011, .output_idx = 0, },
 { .src_id = 0x10012, .output_idx = 0, },
};
static hexagon_nn_output outputs_for_10252[] = {
 OUTPUT_4D(1,149,149,32,4), // { .max_size = 2841728, }, // 1x149x149x32 x4
 OUTPUT_4D(1,1,1,1,4),
 OUTPUT_4D(1,1,1,1,4),
};

static hexagon_nn_input inputs_for_10253[] = {
 { .src_id = 0x10252, .output_idx = 0, },
 { .src_id = 0x10252, .output_idx = 1, },
 { .src_id = 0x10252, .output_idx = 2, },
};
static hexagon_nn_output outputs_for_10253[] = {
 OUTPUT_4D(1,149,149,32,1), //{ .max_size = 710432, }, // 1x149x149x32 x1
 OUTPUT_4D(1,1,1,1,4),
 OUTPUT_4D(1,1,1,1,4),
};
static hexagon_nn_input inputs_for_10254[] = {
 { .src_id = 0x10253, .output_idx = 0, },
 { .src_id = 0x10253, .output_idx = 1, },
 { .src_id = 0x10253, .output_idx = 2, },
};
static hexagon_nn_output outputs_for_10254[] = {
 OUTPUT_4D(1,149,149,32,1), //{ .max_size = 710432, }, // 1x149x149x32 x1
 OUTPUT_4D(1,1,1,1,4),
 OUTPUT_4D(1,1,1,1,4),
};


static hexagon_nn_input inputs_for_10442[] = {
 { .src_id = 0x10254, .output_idx = 0, },
 { .src_id = 0x10254, .output_idx = 1, },
 { .src_id = 0x10254, .output_idx = 2, },
};
static hexagon_nn_output outputs_for_10442[] = {
 OUTPUT_4D(1,149,149,32,4), // { .max_size = (710432*4), }, // 1x149x149x32x4
};

static hexagon_nn_input inputs_for_1044d[] = {
 { .src_id = 0x10442, .output_idx = 0, },
};

#define APPEND_CONST_NODE(ID,...) if (hexagon_nn_append_const_node(nn_id,ID,__VA_ARGS__) != 0) \
	printf("node %x returned nonzero\n",ID); else printf("const node %x success\n",ID)
#define APPEND_NODE(NAME,ID,OP,...) info_for_debug(ID,NAME,#OP); \
	if (hexagon_nn_append_node(nn_id,ID,OP,__VA_ARGS__) != 0) \
	printf("node %x <%s/%s> returned nonzero\n",ID,NAME,#OP)
void init_graph(int nn_id) {
  APPEND_CONST_NODE(0x1000b,3,3,3,32,(const uint8_t *)data_for_op_1000b,864);
  APPEND_CONST_NODE(0x1000c,1,1,1,1,(const uint8_t *)data_for_op_1000c,4);
  APPEND_CONST_NODE(0x1000d,1,1,1,1,(const uint8_t *)data_for_op_1000d,4);
  APPEND_CONST_NODE(0x1000e,1,1,1,1,(const uint8_t *)data_for_op_1000e,4);
  APPEND_CONST_NODE(0x1000f,1,1,1,1,(const uint8_t *)data_for_op_1000f,4);
  APPEND_CONST_NODE(0x10010,1,1,1,32,(const uint8_t *)data_for_op_10010,32);
  APPEND_CONST_NODE(0x10011,1,1,1,1,(const uint8_t *)data_for_op_10011,4);
  APPEND_CONST_NODE(0x10012,1,1,1,1,(const uint8_t *)data_for_op_10012,4);
  APPEND_CONST_NODE(0x10250,1,2,2,1,(const uint8_t *)NULL,0);

  APPEND_NODE("INPUT",0x1024a,OP_INPUT,NN_PAD_ANY,NULL,0,outputs_for_1024a,1);
  APPEND_NODE("Flatten_for_minmax_for_Quantize",0x1024b,OP_Flatten,NN_PAD_ANY,inputs_for_1024b,2,outputs_for_1024b,1);
  APPEND_NODE("min_for_Quantize",0x1024c,OP_Min_f,NN_PAD_ANY,inputs_for_1024c,2,outputs_for_1024c,1);
  APPEND_NODE("max_for_Quantize",0x1024d,OP_Max_f,NN_PAD_ANY,inputs_for_1024d,2,outputs_for_1024d,1);
  APPEND_NODE("Quantize",0x1024e,OP_Quantize,NN_PAD_ANY,inputs_for_1024e,3,outputs_for_1024e,3);
  APPEND_NODE("ConvLayer_conv2d",0x1024f,OP_QuantizedConv2d_8x8to32,NN_PAD_VALID,inputs_for_1024f,7,outputs_for_1024f,3);
  APPEND_NODE("ConvLayer_requant0",0x10251,OP_QuantizeDownAndShrinkRange_32to8,NN_PAD_ANY,inputs_for_10251,3,outputs_for_10251,3);
  APPEND_NODE("ConvLayer_biasadd",0x10252,OP_QuantizedBiasAdd_8p8to32,NN_PAD_ANY,inputs_for_10252,6,outputs_for_10252,3);
  APPEND_NODE("ConvLayer_requant1",0x10253,OP_QuantizeDownAndShrinkRange_32to8,NN_PAD_ANY,inputs_for_10253,3,outputs_for_10253,3);
  APPEND_NODE("ConvLayer_Relu",0x10254,OP_QuantizedRelu_8,NN_PAD_ANY,inputs_for_10254,3,outputs_for_10254,3);
  APPEND_NODE("Dequantize",0x10442,OP_Dequantize,NN_PAD_ANY,inputs_for_10442,3,outputs_for_10442,1);
  APPEND_NODE("OUTPUT",0x1044d,OP_OUTPUT,NN_PAD_ANY,inputs_for_1044d,1,NULL,0);

}
