
/*
 * Copyright (c) 2016, The Linux Foundation. All rights reserved.
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
 * This contains the code for an output node.
 */

static int output_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"output execute. self=%p ",self);
	if (nn->output_data->max_size < self->inputs[0]->data_size) {
		return errlog(nn,"output too small: %d < %d",
			nn->output_data->max_size,
			self->inputs[0]->data_size);
	}
	nn->output_data->shape = self->inputs[0]->shape;
	nn->output_data->data_size = self->inputs[0]->data_size; // FIXME: check vs. max size
#if 0
	/* Do softmax so we don't have to monkey with the graph */
	const struct tensor *in_tensor = self->inputs[0];
	//struct tensor *out_tensor = self->outputs[0];
	const float *data = in_tensor->data;
	float *out = nn->output_data->data;
	float maxval = data[0];
	float sum = 0.0f;
	float sum_recip;
	int n_elements = in_tensor->shape.depth;
	int i;
	for (i = 0; i < n_elements; i++) {
		if (data[i] > maxval) maxval = data[i];
	}
	for (i = 0; i < n_elements; i++) {
		sum += (out[i] = expf(data[i] - maxval));
	}
	sum_recip = 1.0f/sum;
	for (i = 0; i < n_elements; i++) {
		out[i] *= sum_recip;
	}
#else
	memcpy(nn->output_data->data,self->inputs[0]->data,self->inputs[0]->data_size);
#endif
	/* Copy input tensor to output */
	logmsg(nn,2,"copied tensor %d bytes of data",self->inputs[0]->data_size);
	return 0;
}

static int output_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking output node %p",self);
	if (self->inputs == NULL) {
		return errlog(nn,0,"output: fatal: NULL inputs");
	}
	if (self->inputs[0] == NULL) {
		return errlog(nn,0,"output: fatal: NULL input 0");
	}
	logmsg(nn,2,"output node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_OUTPUT = {
	.execute = output_execute,
	.check = output_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

