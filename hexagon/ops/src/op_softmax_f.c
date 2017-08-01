
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
	const float *data = (const float *)in_tensor->data;
	float *out = (float *)out_tensor->data;
	float maxval;
	float sum;
	float sum_recip;
	if (out_tensor->max_size < in_tensor->data_size) {
		return errlog(nn,"out too small");
	}
	for (j = 0; j < batches*height*width; j++) {
		sum = 0.0f;
		maxval = data[0];
		for (i = 0; i < depth; i++) {
			maxval = fmaxf(data[i],maxval);
		}
		for (i = 0; i < depth; i++) {
			sum += (out[i] = expf(data[i] - maxval));
		}
		sum_recip = 1.0f/sum;
		for (i = 0; i < depth; i++) {
			out[i] *= sum_recip;
		}
		out += depth;
		data += depth;
	}
	tensor_set_shape(out_tensor,batches,height,width,depth);
	out_tensor->data_size = in_tensor->data_size;
	return 0;
}


static int softmax_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking softmax node %p",self);
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	if (self->inputs[0] == NULL) return errlog(nn,"NULL input 0");
	if (self->outputs[0] == NULL) return errlog(nn,"NULL output 0");
	if (self->n_inputs != 1) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # inputs");
	logmsg(nn,2,"softmax node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Softmax_f = {
	softmax_execute,
	softmax_check,
	node_alloc_common,
	node_free_common,
};

