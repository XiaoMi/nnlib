
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
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains implementations for instance normalization
 * 
 * Instance normalization is like batch normalization, but we don't average over all the images.
 * Find per-channel mean and variance
 * out = (in - mean) / sqrt(variance + variance_epsilon)
 * 
 */

/*
 * EJP: THE ONLY THING I'VE TESTED IS THAT THIS COMPILES!!! IT PROBABLY DOESN'T WORK!!!
 * EJP: THE ONLY THING I'VE TESTED IS THAT THIS COMPILES!!! IT PROBABLY DOESN'T WORK!!!
 * EJP: THE ONLY THING I'VE TESTED IS THAT THIS COMPILES!!! IT PROBABLY DOESN'T WORK!!!
 * EJP: THE ONLY THING I'VE TESTED IS THAT THIS COMPILES!!! IT PROBABLY DOESN'T WORK!!!
 * EJP: THE ONLY THING I'VE TESTED IS THAT THIS COMPILES!!! IT PROBABLY DOESN'T WORK!!!
 * EJP: THE ONLY THING I'VE TESTED IS THAT THIS COMPILES!!! IT PROBABLY DOESN'T WORK!!!
 */


#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>

/*
 * I think good goals here:
 * * Keep things in quantized if possible (I think it is)
 * * Since variance(X-k) == variance(X) (according to Wikipedia), maybe find variance of raw uint8 values
 * * Also, variance(N*X) = N**2 * variance(X), so we can find variance of data and factor out the float conversion constants.
 * * Since 0 is a uint8 value, we shouldn't have accuracy loss if we're careful before the invsqrt/divides
 * (sum_of_squares - (sum*sum/(w*h))) / (w*h)
 * Eventually we take the invsqrt of the variance, so N**2 * variance(X) becomes N*sqrt(variance(X)...
 */

static int execute_qinstancenorm_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	//const struct tensor *in_min_tensor = self->inputs[1];
	//const struct tensor *in_max_tensor = self->inputs[2];
	//const struct tensor *epsilon_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth = in_tensor->shape.depth;

	uint32_t tmp;
	uint32_t *sum = (uint32_t *)nn->scratch;
	uint32_t *sum_of_squares = sum + depth;
	float *mean = (float *)(sum_of_squares + depth);
	float *variance = mean + depth;
	float *invsqrt_variance = mean + depth;
	float *tmp_data = invsqrt_variance + depth;
	float out_min = 0.0f;
	float out_max = 0.0f;
	float ftmp;

	const uint8_t *in_data = (const uint8_t *)in_tensor->data;
	const uint8_t *data;
	uint8_t *out_data = (uint8_t *)out_tensor->data;
	//const float epsilon = tensor_get_float(epsilon_tensor,0);

	int32_t b,h,w,d;
	int32_t wh = width*height;
	int i;
	size_t bytes = batches * width * height * depth;
	//float in_min = tensor_get_float(in_min_tensor,0);
	//float in_max = tensor_get_float(in_max_tensor,0);
	//float in_level_size = (in_max-in_min)/255.0f;
	memset(nn->scratch,0,sizeof(float)*depth*5);

	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	if (batches != 1) return errlog(nn,"Currently batches not supported");
	tensor_set_shape(out_tensor,batches,width,height,depth);
	logmsg(nn,2,"set out tensor shape %dx%dx%dx%d",batches,width,height,depth);
	out_tensor->data_size = bytes;


	//for (b = 0; b < batches; b++) {
	b = 0;
		data = in_data + (b * width * height * depth);
		/* Try to keep data in fixed point as long as possible */
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				for (d = 0; d < depth; d++) {
					tmp = *data++;
					sum[d] += tmp;
					sum_of_squares[d] += tmp*tmp;
				}
			}
		}
		for (d = 0; d < depth; d++) {
			/* Compute variance.  This would be better as fractional? */
			/* Or maybe... if we add a constant to every input, is the variance the same? */
			/* We need mean in addition to variance */
			/* When we do fixed point here, we need to be careful about division. */
			mean[d] = (float)sum[d] / (wh);
			variance[d] = (sum_of_squares[d] - ((mean[d]*sum[d])))/(wh);
			// alternatively: variance[d] (sum_of_squares[d] - (sum[d]*sum[d]/(w*h))) / (w*h)
			// alternatively: variance[d] sum_of_squares[d]/(w*h) - mean[d]*mean[d]
			/* 
			 * Looking at the graph, epsilon is pretty small. Maybe
			 * quantizes to zero. Can probably handle with clipped
			 * reciprocal estimate
			 */
			/* replace with optimized invsqrt */
			invsqrt_variance[d] = 1.0f / sqrtf((float)(variance[d]) + 0.00001f);
		}
		data = in_data + (b * width * height * depth);
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				for (d = 0; d < depth; d++) {
					/* EJP: I think subtracting the mean cancels out the offset */
					ftmp = (*data++ - mean[d]) * invsqrt_variance[d]; // * in_level_size;
					*tmp_data++ = ftmp;
					out_min = fminf(ftmp,out_min);
					out_max = fmaxf(ftmp,out_max);
				}
			}
		}
		tmp_data -= height*width*depth;
		for (i = 0; i < height*width*depth; i++) {
			*out_data++ = quantize_uint8(*tmp_data++,out_min,out_max);
		}
	//}
        //out_min = out_min * in_level_size;
        //out_max = out_max * in_level_size;

	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,out_min);
	tensor_set_float(out_max_tensor,0,out_max);
	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);
	return 0;
}


/*
 * Wikipedia says this method for calculating variance isn't very good in FP if square of mean
 * and mean of squares are similar, due to massive cancellation of the subtract.
 */

static int execute_finstancenorm(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	//const struct tensor *epsilon_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth = in_tensor->shape.depth;

	float tmp;
	float *sum = (float *)nn->scratch;
	float *sum_of_squares = sum + depth;
	float *mean = sum_of_squares + depth;
	float *variance = mean + depth;
	float *invsqrt_variance = mean + depth;

	const float *in_data = (const float *)in_tensor->data;
	const float *data;
	float *out_data;
	//const float epsilon = tensor_get_float(epsilon_tensor,0);
	const float epsilon = 1.0e-5f;

	int32_t b,h,w,d;
        int32_t wh = width * height;

	size_t bytes = batches * width * height * depth * sizeof(float);

	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");

	tensor_set_shape(out_tensor,batches,width,height,depth);
	out_tensor->data_size = bytes;

	for (b = 0; b < batches; b++) {
		memset(nn->scratch,0,sizeof(float)*depth*5);
		data = in_data + (b * width * height * depth);
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				for (d = 0; d < depth; d++) {
					tmp = *data++;
					sum[d] += tmp;
					sum_of_squares[d] += tmp*tmp;
				}
			}
		}
		for (d = 0; d < depth; d++) {
			mean[d] = sum[d] / (wh);
			variance[d] = (sum_of_squares[d]/(wh)) - (mean[d]*mean[d]);
			invsqrt_variance[d] = 1.0f / sqrtf(variance[d] + epsilon);
		}
		data = in_data + (b * width * height * depth);
		out_data = (float *)out_tensor->data;
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				for (d = 0; d < depth; d++) {
					*out_data++ = (*data++ - mean[d]) * invsqrt_variance[d];
				}
			}
		}
	}
	return 0;
}


static int check_qinstancenorm(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 3) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	return 0;
}

static int check_finstancenorm(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 1) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	return 0;
}



struct nn_node_ops nn_ops_for_QuantizedInstanceNorm_8_ref = {
	SFINIT(.execute, execute_qinstancenorm_ref),
	SFINIT(  .check, check_qinstancenorm),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedInstanceNorm_8 = {
	SFINIT(.execute, execute_qinstancenorm_ref),
	SFINIT(  .check, check_qinstancenorm),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_InstanceNorm_f = {
	SFINIT(.execute, execute_finstancenorm),
	SFINIT(  .check, check_finstancenorm),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};
