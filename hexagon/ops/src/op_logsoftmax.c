
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


static inline int logsoftmax_execute(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	int j;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int depth = in_tensor->shape.depth;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	const float *data = in_tensor->data;
	float *out = out_tensor->data;
	float maxval;
	float sum_of_exps;
	float log_sum_of_exps;
	if( tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, NN_TYPE_FLOAT)!= 0){
		return errlog(nn,"out too small");
	}
	for (j = 0; j < batches*height*width; j++) {
		sum_of_exps = 0.0f;
		maxval = data[0];
		for (i = 0; i < depth; i++) {
			if (data[i] > maxval) maxval = data[i];
		}
		for (i = 0; i < depth; i++) {
			sum_of_exps += expf(data[i] - maxval);
		}
		log_sum_of_exps = logf(sum_of_exps);
		for (i = 0; i < depth; i++) {
			out[i] = data[i] - maxval - log_sum_of_exps;
		}
		out += depth;
		data += depth;
	}
	return 0;
}

struct nn_node_ops nn_ops_for_LogSoftmax_f = {
	.execute = logsoftmax_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};

