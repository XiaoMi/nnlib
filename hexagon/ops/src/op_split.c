
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
 * Input 0: dimension to split on
 * Input 1: data
 * Evenly divide Data across all Outputs
 */

#include <nn_graph.h>
#include <string.h>

static int split_impl(
	struct nn_node *self,
	struct nn_graph *nn,
	int element_size,
	int n_outs)
{
	//const struct tensor *dimdef_tensor = self->inputs[0];
	const struct tensor *data_tensor = self->inputs[1];
	struct tensor **outs = self->outputs;
	//int32_t dimdef = tensor_get_int32(dimdef_tensor,0);
	int32_t depth = data_tensor->shape.depth;
	int32_t width = data_tensor->shape.width;
	int32_t height = data_tensor->shape.height;
	int32_t batches = data_tensor->shape.batches;
	int32_t out_depth = depth / n_outs;
	uint32_t out_bytes = data_tensor->data_size / n_outs;
	const char *in_data = (const char *)data_tensor->data;
	const char *inptr;
	char *outptr;
	int i;
	int j;
	int bytestride;
	

	logmsg(nn,2,"split node %p execute",self);

	if (n_outs == 1) {
		return tensor_copy(self->outputs[0],self->inputs[1]);
	}

	for (i = 0; i < n_outs; i++) {
		if (outs[i]->max_size < out_bytes) return errlog(nn,"out %d too small",i);
		outs[i]->data_size = out_bytes;
		tensor_set_shape(outs[i],batches,height,width,out_depth);
	}

	/* Guess depth */

	if ((depth % n_outs) == 0) {
		bytestride = out_depth * element_size;
	} else {
		return errlog(nn,"FIXME: split assumes depth, need compatible dimension value");
	}

	for (j = 0; j < n_outs; j++) {
		inptr = in_data + j*bytestride;
		outptr = (char *)outs[j]->data;
		for (i = 0; i < batches*height*width*out_depth; i++) {
			memcpy(outptr,inptr,bytestride);
			outptr += bytestride;
			inptr += (bytestride * n_outs);
		}
	}
	return 0;
}

static int split_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return split_impl(self,nn,sizeof(float),self->n_outputs);
}

static int qsplit_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	int n_outputs = self->n_outputs;
	tensor_copy(self->outputs[n_outputs-2],self->inputs[2]);
	tensor_copy(self->outputs[n_outputs-1],self->inputs[3]);
	return split_impl(self,nn,1,n_outputs-2);
}

static int split_check_f(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking split node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"num inputs");
	if (self->n_outputs < 1) return errlog(nn,"num outputs");
	return 0;
}

static int qsplit_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking split node %p",self);
	if (self->n_inputs != 4) return errlog(nn,"num inputs");
	if (self->n_outputs < 3) return errlog(nn,"num outputs");
	return 0;
}



struct nn_node_ops nn_ops_for_Split_f = {
	SFINIT(.execute, split_execute_f),
	SFINIT(  .check, split_check_f),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Split_int32 = {
	SFINIT(.execute, split_execute_f),
	SFINIT(  .check, split_check_f),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedSplit_8 = {
	SFINIT(.execute, qsplit_execute_8),
	SFINIT(  .check, qsplit_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

