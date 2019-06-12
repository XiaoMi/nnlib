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
#include <quantize.h>


static int biasadd_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *bias_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	uint32_t batches = in_tensor->shape.batches;
	uint32_t width = in_tensor->shape.width;
	uint32_t height = in_tensor->shape.height;
	uint32_t depth = in_tensor->shape.depth;

	int32_t stripe;

	const float *in = in_tensor->data;
	const float *bias = bias_tensor->data;
	float *out = out_tensor->data;

	int32_t i;

	/* Assert min and max are size 1,1,1,1 ? */

	if (bias_tensor->shape.height != 1) return errlog(nn,"bias shape (height!=1)");
	if (bias_tensor->shape.batches != 1) return errlog(nn,"bias shape (batches!=1)");
	if (bias_tensor->shape.width != 1) return errlog(nn,"bias shape (width!=1)");
	if (bias_tensor->shape.depth != depth) {
		return errlog(nn,"depth mismatch %d vs %d",bias_tensor->shape.depth,depth);
	}

	logmsg(nn,2,"biasadd execute. self=%p ",self);
	if( tensor_out_prepare_normal( out_tensor, batches,height,width,depth, NN_TYPE_FLOAT)!= 0 )
		return errlog(nn,"out too small");

	for (stripe = 0; stripe < width*height*batches; stripe++) {
		for (i = 0; i < depth; i++) {
			out[i] = in[i] + bias[i];
		}
		in += depth;
		out += depth;
	}
	logmsg(nn,2,"biasadd %p done",self);
	return 0;
}


struct nn_node_ops nn_ops_for_BiasAdd_f = {
	.execute = biasadd_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),

};


