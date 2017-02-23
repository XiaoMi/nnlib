
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
 * This is an example of how to set up a trivial graph.
 */

#include <hexagon_nn.h>
#include <stdio.h>
#include "../interface/hexagon_nn_ops.h"
typedef int32_t qint32;
typedef int32_t int32;
typedef float float32;
typedef uint8_t quint8;
#define ALIGNED __attribute__((aligned(128)))

#define NN_PAD_ANY NN_PAD_NA

static hexagon_nn_output outputs_for_1024a[] = {
 { .max_size = 1072812, .unused = 0, }, // 1x299x299x3 x4
};
static hexagon_nn_input inputs_for_10300[] = {
 { .src_id = 0x1024a, .output_idx = 0, },
};
static hexagon_nn_output outputs_for_10300[] = {
 { .max_size = 1072812, .unused = 0, }, // 1x299x299x3 x4
};
static hexagon_nn_input inputs_for_1044d[] = {
 { .src_id = 0x10300, .output_idx = 0, },
};

#define APPEND_CONST_NODE(ID,...) if (hexagon_nn_append_const_node(nn_id,ID,__VA_ARGS__) != 0) \
	printf("node %d returned nonzero\n",ID); else printf("const node %d success\n",ID)
#define APPEND_NODE(ID,...) if (hexagon_nn_append_node(nn_id,ID,__VA_ARGS__) != 0) \
	printf("node %d returned nonzero\n",ID); else printf("node %d success\n",ID)
void init_graph(int nn_id) {
  APPEND_NODE(0x1024a,OP_INPUT,NN_PAD_ANY,NULL,0,outputs_for_1024a,1);
  APPEND_NODE(0x10300,OP_Nop,NN_PAD_ANY,inputs_for_10300,1,outputs_for_10300,1);
  APPEND_NODE(0x1044d,OP_OUTPUT,NN_PAD_ANY,inputs_for_1044d,1,NULL,0);
}

