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
#include <quantize.h>


/*
 * LRN:
 * * Input tensor
 * * window_shape
 * * Bias
 * * Alpha
 * * Beta
 * out = in / (bias + alpha * (sum(foreach (input element determined by window_shape)**2)))**beta
 */

static inline int fourd_index(
	int32_t b,
	int32_t y,
	int32_t x,
	int32_t z,
	int32_t batches __attribute__((unused)),
	int32_t height,
	int32_t width,
	int32_t depth)
{
	return	(b * height * width * depth)
		+ (y * width * depth)
		+ (x * depth)
		+ z;
}
	
static inline float compute_lrn_at(
	unsigned char *data,
	float min,
	float max,
	int32_t b,
	int32_t y_start,
	int32_t x_start,
	int32_t z_start,
	int32_t batches,
	int32_t height,
	int32_t width,
	int32_t depth,
	const struct tensor *shape_tensor,
	const float bias,
	const float alpha,
	const float beta)
{

	int32_t x, y, z;
	int32_t window_y = shape_tensor->shape.height;
	int32_t window_x = shape_tensor->shape.width;
	int32_t window_z = shape_tensor->shape.depth;

	int32_t window_eachside_y = (window_y-1)/2;
	int32_t window_eachside_x = (window_x-1)/2;
	int32_t window_eachside_z = (window_z-1)/2;

	float stepsize = (max-min)/255.0f;
	float input;
	float sum = 0;

	for (y = y_start - window_eachside_y; y < y_start + window_eachside_y + 1; y++) {
	  if (y < 0) continue;
	  if (y >= height) continue;
	  for (x = x_start - window_eachside_x; x < x_start + window_eachside_x + 1; x++) {
	    if (x < 0) continue;
	    if (x >= height) continue;
	    for (z = z_start - window_eachside_z; z < z_start + window_eachside_z + 1; z++) {
	      if (z < 0) continue;
	      if (z >= depth) continue;
	      input = min + stepsize * data[fourd_index(b,y,x,z,batches,height,width,depth)];
	      sum += input * input;
	    }
	  }
	}
	/* We have the sum of squares of values in the window */
	/* Multiply by alpha, add bias */
	sum *= alpha;
	sum += bias;
	/* Then we pow by -beta... that's the same as exp(ln(x)*-beta) */
	sum = expf(logf(x) * -beta);
	/* Then we multiply by input value */
	input = data[fourd_index(b,y_start,x_start,z_start,batches,height,width,depth)];
	sum *= input;
	/* Then we are done! */
	return quantize_uint8(sum,min,max);
}

static int lrn_8_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *shape_tensor = self->inputs[3];
	const struct tensor *bias_tensor = self->inputs[4];
	const struct tensor *alpha_tensor = self->inputs[5];
	const struct tensor *beta_tensor = self->inputs[6];
	const float bias = tensor_get_float(bias_tensor,0);
	const float alpha = tensor_get_float(alpha_tensor,0);
	const float beta = tensor_get_float(beta_tensor,0);

	struct tensor *out_tensor = self->outputs[0];
	float *out = out_tensor->data;

	float min = tensor_get_float(in_min_tensor,0);
	float max = tensor_get_float(in_max_tensor,0);

	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth = in_tensor->shape.depth;

	int32_t b;
	int32_t x;
	int32_t y;
	int32_t z;

	if (in_tensor->data_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",
			out_tensor->max_size,
			in_tensor->data_size);
	}
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = in_tensor->data_size;

	if (shape_tensor->shape.batches != 1) return errlog(nn,"LRN by batches?");
	for (b = 0; b < batches; b++) {
	  for (y = 0; y < height; y++) {
	    for (x = 0; x < width; x++) {
	      for (z = 0; z < height; z++) {
	        uint32_t out_data = compute_lrn_at(
		        in_tensor->data,
			min,
			max,
			b,
			y,
			x,
			z,
			batches,
			height,
			width,
			depth,
			shape_tensor,
			bias,
			alpha,
			beta);
	        int out_idx = fourd_index(b,y,x,z,batches,height,width,depth);
	        out[out_idx] = out_data;
	      }
	    }
	  }
	}
	return 0;
}


static int lrn_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking q lrn node %p",self);
	if (self->n_inputs != 7) return errlog(nn,"LRN wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"LRN wrong # outs");
	logmsg(nn,2,"q lrn %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedLRN_8 = {
	.execute = lrn_8_execute,
	.check = lrn_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};


