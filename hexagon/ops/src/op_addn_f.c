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

static int addn_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor **ins = self->inputs;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	uint32_t b = in_tensor->shape.batches;
	uint32_t h = in_tensor->shape.height;
	uint32_t w = in_tensor->shape.width;
	uint32_t d = in_tensor->shape.depth;
	size_t elements = b*h*w*d;
	float *out_data = out_tensor->data;
	float sum;
	uint32_t i,j;

	logmsg(nn,2,"addN execute. self=%p ",self);
	for (i = 1; i < self->n_inputs; i++) {
		if (ins[i]->shape.batches != b) return errlog(nn,"shape mismatch");
		if (ins[i]->shape.height != h) return errlog(nn,"shape mismatch");
		if (ins[i]->shape.width != w) return errlog(nn,"shape mismatch");
		if (ins[i]->shape.depth != d) return errlog(nn,"shape mismatch");
	}
	if(  tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_FLOAT)!= 0){
		 return errlog(nn,"out too small");
	}

	for (i = 0; i < elements; i++) {
		sum = 0.0f;
		for (j = 0; j < self->n_inputs; j++) {
			sum += tensor_get_float(ins[j],i);
		}
		*out_data++ = sum;
	}

	logmsg(nn,2,"addn %p done",self);
	return 0;
}


struct nn_node_ops nn_ops_for_AddN_f = {
	.execute = addn_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(1),
	.n_outputs = NN_IOCOUNT(1),
};

