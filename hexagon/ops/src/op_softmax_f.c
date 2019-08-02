
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
#include <math.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains min and max (floating) ops
 */


static inline int softmax_execute(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	int j;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int depth = in_tensor->shape.depth;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	float beta = (self->n_inputs < 2) ? 1.0f : tensor_get_float(self->inputs[1],0);
	const float *data = in_tensor->data;
	float *out = out_tensor->data;
	float maxval;
	float sum;
	float sum_recip;

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_FLOAT)!= 0){
		return errlog(nn,"out too small");
	}
	for (j = 0; j < batches*height*width; j++) {
		sum = 0.0f;
		maxval = data[0];
		for (i = 0; i < depth; i++) {
			maxval = fmaxf(data[i],maxval);
		}
		for (i = 0; i < depth; i++) {
			sum += (out[i] = expf(beta*(data[i] - maxval)));
		}
		sum_recip = 1.0f/sum;
		for (i = 0; i < depth; i++) {
			out[i] *= sum_recip;
		}
		out += depth;
		data += depth;
	}
	return 0;
}


static inline int softmax_execute_uint8(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	int j;
	const struct tensor *in_tensor = self->inputs[0];
	float min = tensor_get_float(self->inputs[1],0);
	float max = tensor_get_float(self->inputs[2],0);
	float beta = (self->n_inputs < 4) ? 1.0f : tensor_get_float(self->inputs[3],0);
	struct tensor *out_tensor = self->outputs[0];
	int depth = in_tensor->shape.depth;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	const unsigned char *data = in_tensor->data;
	float *out = out_tensor->data;
	float sum;
	float sum_recip;
	float *precomputed = self->opaque;

	float scalex = beta * (max-min);
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_FLOAT)!= 0){
		return errlog(nn,"out too small");
	}

	if (precomputed[255] == 0.0f || precomputed[256]!= scalex ) {
		// The precomputed-array needs initialization
		float scale = scalex/255.0;
		for (i=0; i<256; i++) {
			precomputed[i] = expf(scale*((float)(i-255)));
		}
		precomputed[256] = scalex;
	}

	for (j = 0; j < batches*height*width; j++) {
		sum = 0.0f;
		for (i = 0; i < depth; i++) {
			sum += (out[i] = precomputed[data[i]]);
		}
		sum_recip = 1.0f/sum;
		for (i = 0; i < depth; i++) {
			out[i] *= sum_recip;
		}
		out += depth;
		data += depth;
	}
	return 0;
}

struct nn_node *node_alloc_softmax_uint8(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	struct nn_node *newnode = node_alloc_common(nn,node_id, operation, padding, num_inputs, num_outputs, inputs, outputs);

	if (newnode == NULL) {
		return NULL;
	}

	if ((newnode->opaque = nn_calloc(256+1,sizeof(float))) == NULL) {
		errlog(nn,"softmax cache alloc failed");
		goto err_free;
	}
	return newnode;

err_free:
	node_free_common(newnode, nn);
	return NULL;
}

int node_free_softmax_uint8(struct nn_node *node, struct nn_graph *nn)
{
	if (node->opaque) nn_free(node->opaque);
	return node_free_common(node, nn);
}


struct nn_node_ops nn_ops_for_Softmax_f = {
	.execute = softmax_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,2),
	.n_outputs = NN_IOCOUNT(1),
};


struct nn_node_ops nn_ops_for_Softmax_uint8 = {
	.execute = softmax_execute_uint8,
	.check = NULL,
	.ctor = node_alloc_softmax_uint8,
	.dtor = node_free_softmax_uint8,
	.n_inputs = NN_IOCOUNT_RANGE(3,4),
	.n_outputs = NN_IOCOUNT(1),
};

