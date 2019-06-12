/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (mulject to the limitations in the
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
#include <stdbool.h>

#define OPERATOR_AND(X,Y) ((X)&(Y))
BROADCAST_STRIDE_11_FUNC( and_int32_stride_11, int32_t, int32_t, OPERATOR_AND)

static void  and_int32_stride_10( void *out, void const *in1, void const *in2, int n, void *opaque)
{
	int32_t * op = (int32_t*)out;
	int32_t const * inp1 = (int32_t const *)in1;
	int32_t xin2 = *(int32_t const *)in2;
	if( n >0){
		if (xin2 == -1) memcpy( out, in1, n*sizeof(int32_t));
		else if (xin2 == 0) memset( out, 0, n*sizeof(int32_t));
		else
			for( int i =0; i < n; i++) op[i] = inp1[i]&xin2;
	}
}


static const struct elementwise_funcs And_int32_funcs = {
	.op_stride_11 = and_int32_stride_11,
	.op_stride_10 = and_int32_stride_10,
	.op_rev_stride_01 = and_int32_stride_10,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};

static int and_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast( self, nn, &And_int32_funcs,NULL, NULL, NULL );
}

#define OPERATOR_IOR(X,Y) ((X)|(Y))
BROADCAST_STRIDE_11_FUNC( ior_int32_stride_11, int32_t, int32_t, OPERATOR_IOR)

static void  ior_int32_stride_10( void *out, void const *in1, void const *in2, int n, void *opaque)
{
	int32_t * op = (int32_t*)out;
	int32_t const * inp1 = (int32_t const *)in1;
	int32_t xin2 = *(int32_t const *)in2;
	if( n >0){
		if (xin2 == 0) memcpy( out, in1, n*sizeof(int32_t));
		else if( xin2 == -1) memset( out, -1, n*sizeof(int32_t));
		else
			for( int i =0; i < n; i++) op[i] = inp1[i]|xin2;
	}
}


static const struct elementwise_funcs Ior_int32_funcs = {
	.op_stride_11 = ior_int32_stride_11,
	.op_stride_10 = ior_int32_stride_10,
	.op_rev_stride_01 = ior_int32_stride_10,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};

static int ior_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast( self, nn, &Ior_int32_funcs,NULL, NULL, NULL );
}
#define OPERATOR_XOR(X,Y) ((X)^(Y))
BROADCAST_STRIDE_11_FUNC( xor_int32_stride_11, int32_t, int32_t, OPERATOR_XOR)

static void  xor_int32_stride_10( void *out, void const *in1, void const *in2, int n, void *opaque)
{
	int32_t * op = (int32_t*)out;
	int32_t const * inp1 = (int32_t const *)in1;
	int32_t xin2 = *(int32_t const *)in2;
	if( n >0){
		if (xin2 == 0) memcpy( out, in1, n*sizeof(int32_t));
		else
			for( int i =0; i < n; i++) op[i] = inp1[i]^xin2;
	}
}


static const struct elementwise_funcs Xor_int32_funcs = {
	.op_stride_11 = xor_int32_stride_11,
	.op_stride_10 = xor_int32_stride_10,
	.op_rev_stride_01 = xor_int32_stride_10,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};

static int xor_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast( self, nn, &Xor_int32_funcs,NULL,NULL, NULL );
}


static int not_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    struct tensor *out_tensor = self->outputs[0];

    out_tensor->shape.batches = in_tensor->shape.batches;
    out_tensor->shape.height = in_tensor->shape.height;
    out_tensor->shape.width = in_tensor->shape.width;
    out_tensor->shape.depth = in_tensor->shape.depth;
    out_tensor->data_size = out_tensor->shape.batches *
		out_tensor->shape.height *
		out_tensor->shape.width *
		out_tensor->shape.depth *
        sizeof(int32_t);
    out_tensor->format.layout = NN_LAYOUT_PLAIN;
    out_tensor->format.type = NN_TYPE_INT32;

    int i, count;
    int32_t *in  = (int32_t *) in_tensor->data;
    int32_t *out = (int32_t *) out_tensor->data;
    count = in_tensor->data_size/tensor_type_size(in_tensor->format.type);
    for (i=0; i<count; i++) {
        *(out++) = ~*(in++);
    }

    return 0;
}


struct nn_node_ops nn_ops_for_BitwiseAnd_int32 = {
	.execute = and_int32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_BitwiseOr_int32 = {
	.execute = ior_int32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_BitwiseXor_int32 = {
	.execute = xor_int32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_BitwiseNot_int32 = {
	.execute = not_int32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};
