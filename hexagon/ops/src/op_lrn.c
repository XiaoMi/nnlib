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
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

//#define DEBUG_PRINT_LRN_CYCLECOUNT
#define DEBUG_PRINT_LRN_PERFORMANCE
//#define DEBUG_USE_VEXTRACT
//#define DEBUG_USE_LRN_SCALAR_INNERLOOP

/*
 * LRN:
 * * Input tensor
 * * window_shape
 * * Bias
 * * Alpha
 * * Beta
 * out = in / (bias + ((alpha/window_size) * (sum(foreach (input element determined by window_shape)**2))))**beta
 * implementation equivalent: out = in*expf[logf{scaling*(bias/scaling+sum_squared_inputs_over_win)}*-beta]
*/
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((ALIGN_SIZE)-1)))

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
	return (b * height * width * depth)
		 + (y * width * depth)
		 + (x * depth)
		 + z;
}

static inline float compute_ref_lrn_at(
	int16_t *scratch,
	const uint8_t *in,
	const float min,
	const float max,
	const int32_t b,
	const int32_t y_start,
	const int32_t x_start,
	const int32_t z_start,
	const int32_t batches,
	const int32_t height,
	const int32_t width,
	const int32_t depth,
	const struct tensor *shape_tensor,
	const float in_step,
	const float bias,
	const float scaling,
	const float beta)
{
	int32_t x, y, z;
	int32_t window_y = shape_tensor->shape.height;
	int32_t window_x = shape_tensor->shape.width;
	int32_t window_z = shape_tensor->shape.depth;
	
	int32_t window_eachside_y = (window_y-1)/2;
	int32_t window_eachside_x = (window_x-1)/2;
	int32_t window_eachside_z = (window_z-1)/2;
	
	/* calc sum-squared-elemidx in the window */
	float dqelem = 0;
	float sum = 0;
	for (y = y_start - window_eachside_y; y < y_start + window_eachside_y + 1; y++) {
	  if (y < 0) continue;
	  if (y >= height) continue;
	  for (x = x_start - window_eachside_x; x < x_start + window_eachside_x + 1; x++) {
	    if (x < 0) continue;
	    if (x >= width) continue;
	    for (z = z_start - window_eachside_z; z < z_start + window_eachside_z + 1; z++) {
	      if (z < 0) continue;
	      if (z >= depth) continue;
	      /* save floats */
	      dqelem = min + in_step * in[fourd_index(b,y,x,z,batches,height,width,depth)];
	      sum += dqelem * dqelem;
	    }
	  }
	}
	
	/* Multiply by alpha-scaling, add bias */
	/* pow by -beta... that's the same as exp(ln(x)*-beta) */
	sum *= scaling;
	sum += bias;
	sum = expf(logf(sum) * -beta);
	return sum;
}

static int lrn_8_execute_ref(struct nn_node *self, struct nn_graph *nn)
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
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	const float in_step = (in_max-in_min)/255.0f;
	float in_data;
	uint8_t *in = in_tensor->data;
	
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float out_min = 0.0f;
	float out_max = 0.0f;
	float out_data;

	uint8_t *out = out_tensor->data;
	
	const int32_t window_size = (int32_t) tensor_get_float(shape_tensor, 0);
	const float scaling = alpha / (float)window_size;
	float lrn_multiplier;
	
	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth = in_tensor->shape.depth;
	int32_t elemcount = batches*height*width*depth;
	if (nn_scratch_grow(nn, (elemcount * sizeof(float)))){
		return errlog(nn,"failed to get scratch");
	}
	float *tmpdata = nn->scratch;
	
	int32_t b;
	int32_t x_start;
	int32_t y_start;
	int32_t z_start;
	int32_t i;
	
	/* check parameters and report errors to skip calc */
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!=0){
		return errlog(nn,"output too small, %d < %d",
			out_tensor->max_size,
			in_tensor->data_size);
	}
	if (shape_tensor->shape.batches != 1) return errlog(nn,"LRN by batches?");

	/* calc LRN at each idx */
	/* LRN math (normal formula): out = in * [{scaling *(bias/scaling + sum_squared_inputs_over_win)} ** -beta] */
	/* implementation equivalent: out = in*expf[logf{scaling*(bias/scaling+sum_squared_inputs_over_win)}*-beta] */
#ifdef DEBUG_PRINT_LRN_REF_PERFORMANCE
	int start_time =  nn_os_get_cycles(nn);
#endif
	
	
	/* Use elementwise calc */
	for (b = 0; b < batches; b++) {
	  tmpdata = nn->scratch;
	  for (y_start = 0; y_start < height; y_start++) {
	    for (x_start = 0; x_start < width; x_start++) {
	      for (z_start = 0; z_start < depth; z_start++) {
			/* REMOVED: reasonable optimization to skip 0 inputs since they will be 0 outputs. */
			/* read input value non-0 */
			/* get output sum-squared-elemidx */
			/* multiplied by alpha-scaling, add bias */
			/* then powd by -beta... that's the same as exp(ln(x)*-beta) */
			/* multiply by input value */
			/* then quantize and write output value */
			in_data = in_min + in_step * *in++;
			lrn_multiplier = compute_ref_lrn_at(
							nn->scratch,
							in_tensor->data,
							in_min,
							in_max,
							b,
							y_start,
							x_start,
							z_start,
							batches,
							height,
							width,
							depth,
							shape_tensor,
							in_step,
							bias,
							scaling,
							beta);
			out_data = lrn_multiplier * in_data;
			out_max = fmaxf(out_max,out_data);
			out_min = fminf(out_min,out_data);
			*tmpdata++ = out_data;
	      }
	    }
	  }
	  tmpdata = nn->scratch;
	  for (i = 0; i < height*width*depth; i++) {
	    *out++ = quantize_uint8(*tmpdata++,out_min,out_max);
	  }
	}
	
	/* report output min/max */
	tensor_set_single_float( out_min_tensor, out_min);
	tensor_set_single_float( out_max_tensor, out_max);

#ifdef DEBUG_PRINT_LRN_REF_PERFORMANCE
	int end_time =  nn_os_get_cycles(nn);
	int elem_size = elemcount;
	logmsg(nn, 2, "qlrn ref cycles = %d (elements = %d)\n", (end_time-start_time), elem_size);
#endif
	
	return 0;
}


static int lrn_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking q lrn node %p",self);
	for (uint32_t i = 3; i < 7; i++) {
		if (self->inputs[i]->data == NULL) {
			return errlog(nn,"input %d not const",i);
		}
	}

	const int32_t window_size = (int32_t) tensor_get_float(self->inputs[3], 0);
	if (window_size < 1) {
		return errlog(nn, "LRN invalid window size (< 1)"); // int(window_size)>=1 check
	}
	const float bias = tensor_get_float(self->inputs[4], 0);
	if (bias < 1.0) {
		return errlog(nn, "LRN unsupported bias-value (< 1.0)"); // bias>=1 check
	}
	const float alpha = tensor_get_float(self->inputs[5], 0);
	if (alpha < 0.0) {
		return errlog(nn, "LRN unsupported alpha-value (< 0.0)"); // alpha>0 check
	}
	const float beta = tensor_get_float(self->inputs[6], 0);
	if (beta < 0.0) {
		return errlog(nn, "LRN unsupported beta-value (< 0.0)"); // beta>0 check
	}
	logmsg(nn,2,"q lrn %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedLRN_8_ref = {
	.execute = lrn_8_execute_ref,
	.check = lrn_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedLRN_8 = {
	.execute = lrn_8_execute_ref,
	.check = lrn_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};


