/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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


#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#include <nn_broadcast.h>


// subtract vector from vector
#define OPERATOR_SUB(X,Y) ((X)-(Y))
BROADCAST_STRIDE_11_FUNC(sub_f_stride_11, float, float, OPERATOR_SUB)
// subtract scalar from vector
BROADCAST_STRIDE_10_FUNC(sub_f_stride_10, float, float, OPERATOR_SUB)
// subtract vector from scalar
BROADCAST_REV_STRIDE_01_FUNC(sub_f_rev_stride_01, float, float, OPERATOR_SUB)


static const struct elementwise_funcs Sub_f_funcs = {
	.op_stride_11 = sub_f_stride_11,
	.op_stride_10 = sub_f_stride_10,
	.op_rev_stride_01 = sub_f_rev_stride_01,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_FLOAT
};

static int sub_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast( self, nn, &Sub_f_funcs,NULL,NULL, NULL );
}



BROADCAST_STRIDE_11_FUNC(sub_int32_stride_11, int32_t, int32_t, OPERATOR_SUB)
// subtract scalar from vector
BROADCAST_STRIDE_10_FUNC(sub_int32_stride_10, int32_t, int32_t, OPERATOR_SUB)
// subtract vector from scalar
BROADCAST_REV_STRIDE_01_FUNC(sub_int32_rev_stride_01, int32_t, int32_t, OPERATOR_SUB)

static const struct elementwise_funcs Sub_int32_funcs = {
	.op_stride_11 = sub_int32_stride_11,
	.op_stride_10 = sub_int32_stride_10,
	.op_rev_stride_01 = sub_int32_rev_stride_01,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};

static int sub_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast( self, nn, &Sub_int32_funcs,NULL,NULL, NULL );
}


struct nn_node_ops nn_ops_for_Sub_f = {
	.execute = sub_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};


struct nn_node_ops nn_ops_for_Sub_int32 = {
	.execute = sub_int32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};
