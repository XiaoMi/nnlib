/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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


static int depthspace_s2d_execute(struct nn_node *self, struct nn_graph *nn, int elementsize, int dtype)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *block_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	uint32_t block_size = tensor_get_int32(block_tensor,0);
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_batches = in_tensor->shape.batches;
	int32_t copy_size = in_depth * elementsize * block_size;
	const char *in_base = in_tensor->data;
	char *out = out_tensor->data;
	int b,w,h,hi;

	if ((in_width % block_size) != 0) return errlog(nn,"width must be multiple of block size");
	if ((in_height % block_size) != 0) return errlog(nn,"height must be multiple of block size");

	int32_t out_batches = in_batches;
	int32_t out_height = in_height / block_size;
	int32_t out_width = in_width / block_size;
	int32_t out_depth = in_depth * block_size * block_size;

	if (tensor_out_prepare_normal(out_tensor,out_batches,out_height,out_width,out_depth,dtype)!=0){
		return errlog(nn,"failed to prepare output");
	}

	for (b = 0; b < in_batches; b++) {
		for (h = 0; h < out_height; h++) {
			for (w = 0; w < out_width; w++) {
				for (hi = 0; hi < block_size; hi++) {
					const char *in = in_base
						+ elementsize * (b*in_depth*in_width*in_height
						+ (h*block_size+hi)*in_width*in_depth
						+ w*block_size*in_depth);
					memcpy(out,in,copy_size);
					out += copy_size;
				}
			}
		}
	}
	return 0;
}



static int depthspace_d2s_execute(struct nn_node *self, struct nn_graph *nn, int elementsize, int dtype)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *block_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	uint32_t block_size = tensor_get_int32(block_tensor,0);
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_batches = in_tensor->shape.batches;
	int32_t copy_size = in_depth * elementsize / block_size;
	const char *in = in_tensor->data;
	char *out_base = out_tensor->data;
	int b,w,h,hi;

	if ((in_depth % (block_size*block_size)) != 0) return errlog(nn,"depth must be multiple of square of block size");

	int32_t out_batches = in_batches;
	int32_t out_height = in_height * block_size;
	int32_t out_width = in_width * block_size;
	int32_t out_depth = in_depth / (block_size * block_size);

	if (tensor_out_prepare_normal(out_tensor,out_batches,out_height,out_width,out_depth,dtype)!=0){
		return errlog(nn,"failed to prepare output");
	}

	for (b = 0; b < in_batches; b++) {
		for (h = 0; h < in_height; h++) {
			for (w = 0; w < in_width; w++) {
				for (hi = 0; hi < block_size; hi++) {
					char *out = out_base
						+ elementsize * (b*out_depth*out_width*out_height
						  + (h*block_size+hi)*out_width*out_depth
						  + w*block_size*out_depth);
					memcpy(out,in,copy_size);
					in += copy_size;
				}
			}
		}
	}
	return 0;
}

static int depthspace_d2s_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return depthspace_d2s_execute(self,nn,sizeof(float),NN_TYPE_FLOAT);
}

static int depthspace_s2d_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return depthspace_s2d_execute(self,nn,sizeof(float),NN_TYPE_FLOAT);
}

static int depthspace_d2s_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[2]);
	tensor_copy(self->outputs[2],self->inputs[3]);
	return depthspace_d2s_execute(self,nn,sizeof(uint8_t),NN_TYPE_QUINT8);
}

static int depthspace_s2d_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[2]);
	tensor_copy(self->outputs[2],self->inputs[3]);
	return depthspace_s2d_execute(self,nn,sizeof(uint8_t),NN_TYPE_QUINT8);
}


static int depthspace_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking depthspace node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"depthspace %p check OK",self);
	return 0;
}

static int depthspace_check_q(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking depthspace node %p",self);
	if (self->n_inputs != 4) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"depthspace %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_DepthToSpace_f = {
	.execute = depthspace_d2s_execute_f,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_SpaceToDepth_f = {
	.execute = depthspace_s2d_execute_f,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_DepthToSpace_8 = {
	.execute = depthspace_d2s_execute_8,
	.check = depthspace_check_q,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_SpaceToDepth_8 = {
	.execute = depthspace_s2d_execute_8,
	.check = depthspace_check_q,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

