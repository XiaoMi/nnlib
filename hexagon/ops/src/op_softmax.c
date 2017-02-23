
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
#include <stdio.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains min and max (floating) ops
 */


static int qsoftmax_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	int depth = in_tensor->shape.depth;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	size_t elements = depth * batches * height * width;
	size_t bytes = elements * sizeof(uint8_t);

	const uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	int i;
	int j;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float stepsize = (in_max - in_min)/255.0f;
	float maxval, inval, outval, outmax=-INFINITY, outmin=INFINITY;
	float sum, sum_recip, outmax_recip;

	logmsg(nn,2,"qsoftmax execute. self=%p ",self);
	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = bytes;

	for (j = 0; j < batches*height*width; j++) {

		/* Convert quantized uint8 to float and find maxval */
		maxval = in_min + stepsize * in_data[0];
		for (i = 0; i < depth; i++) {
			inval = in_min + stepsize * in_data[i];
			maxval = fmaxf(inval,maxval);
			//printf("inval=%f\n",inval);
		}

		/* Quantized softmax reference code */
		sum = 0.0f;
		for (i = 0; i < depth; i++) {
			inval = in_min + stepsize * in_data[i];
			sum += expf(inval - maxval);
		}
		sum_recip = 1.0f/sum;
		for (i = 0; i < depth; i++) {

			// convert quint8 to float
			inval = in_min + stepsize * in_data[i];

			// output in float
			outval = expf(inval - maxval) * sum_recip;
			//printf("outval=%f\n",outval);
			outmax = fmaxf(outmax,outval);
			outmin = fminf(outmin,outval);

			// convert float to quint8
			outval = (outval) * 255.0f;
			if (outval > 255.0f) outval = 255.0f;
			out_data[i] = outval;
		}
		out_data += depth;
		in_data += depth;
	}

	out_data = out_tensor->data;
	for (j = 0; j < batches*height*width; j++) {
		/* Requantize based on outmax, outmin */
		outmax_recip = 1.0f/outmax;
		for (i = 0; i < depth; i++) {
			out_data[i] = out_data[i] * outmax_recip + outmin + 0.5f;
		}
		out_data += depth;
	}

	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,outmin);
	out_min_tensor->data_size = sizeof(float);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_max_tensor,0,outmax);
	out_max_tensor->data_size = sizeof(float);

	logmsg(nn,2,"qsoftmax %p done",self);
	return 0;
}

static int qsoftmax_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking softmax node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"softmax node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8_ref = {
	.execute = qsoftmax_execute_ref,
	.check = qsoftmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8 = {
	.execute = qsoftmax_execute_ref,
	.check = qsoftmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

