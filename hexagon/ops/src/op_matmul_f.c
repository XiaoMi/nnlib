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


#include <nn_graph.h>
#include <string.h>
#include <stdlib.h>
/* transpose the weights matrix and shufle blocks of 32 together */


static int matmul_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	uint32_t a_batches = a_tensor->shape.batches;
	uint32_t a_width = a_tensor->shape.width;
	uint32_t a_height = a_tensor->shape.height;
	uint32_t a_depth = a_tensor->shape.depth;

	uint32_t b_batches = b_tensor->shape.batches;
	uint32_t b_width = b_tensor->shape.width;
	uint32_t b_height = b_tensor->shape.height;
	uint32_t b_depth = b_tensor->shape.depth;

	uint32_t out_batches = a_batches;
	uint32_t out_height = a_height;
	uint32_t out_width = a_width;
	uint32_t out_depth = b_depth;

	int32_t x;
	int32_t y;
	int32_t i;

	const float *a = (const float *)a_tensor->data;
	const float *b = (const float *)b_tensor->data;
	float *out = (float *)out_tensor->data;

	float adata;
	float bdata;
	float sum;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements*sizeof(float);

	logmsg(nn,2,"matmul execute. self=%p",self);
	logmsg(nn,2,"matmul in dims: %dx%dx%dx%d * %dx%dx%dx%d",
		a_batches,a_height,a_width,a_depth,
		b_batches,b_height,b_width,b_depth);
	if (a_height != 1) return errlog(nn,"oops, height != 1");
	if (b_height != 1) return errlog(nn,"oops, height != 1");
	if (a_batches != 1) return errlog(nn,"fixme: support batches");
	if (b_batches != 1) return errlog(nn,"fixme: support batches");
	if (out_size > (out_tensor->max_size)) return errlog(nn,"output too small");

	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_size;

	for (y = 0; y < a_width; y++) {
		for (x = 0; x < b_depth; x++) {
			sum = 0.0f;
			for (i = 0; i < a_depth; i++) {
				adata = a[i+y*a_depth];
				bdata = b[x+i*b_depth];
				sum += adata * bdata;
				//printf("y=%ld, x=%ld, i=%ld, adata=%f, bdata=%f, sum=%f\n",y,x,i,adata,bdata,sum);
			}
			out[x+y*out_depth] = sum;
		}
	}
	logmsg(nn,2,"matmul execute (ref) done!");
	return 0;
}


static int matmul_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking matmul node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"matmul wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"matmul wrong # outputs");
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	logmsg(nn,2,"matmul node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_MatMul_f = {
	matmul_execute_ref,
	matmul_check_ref,
	node_alloc_common,
	node_free_common,
};

