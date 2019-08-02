
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
#include <math.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif
#include "hvx_hexagon_protos.h"
#include "quantize.h"
/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains min and max (floating) ops
 */
//#define SOFTMAX_PERFORMANCE

// expf approx, same domain as 'proper' expf, and
// accurate enough for softmax with 8-bit output.
//
static inline float expf_approx(float x){
	int xfrac = roundf_i32(x*(float)(16384.0/0.69314718056) );
	// now we want 2^xfrac (which has 14 fractional bits).
	float xf  = (xfrac & 0x3FFF) * (float)(1./16384.);
	// approx for 2^x
	// good to about 11 bits
	// 4th order would be:
	// 1.          0.69304423  0.24130777  0.0521633   0.0134847

	float exp0 = 1.0f + xf*(0.69051585f + xf*(0.23793659f + xf*0.07154756f));
	return flt_power2(xfrac>>14)*exp0;
}

// softmax on qu8 inputs
// defined (over a set of inputs x[0..n-1]) as
//
//       t[i] = exp( beta * x[i])
//       y[i] = t[i]/sum(t[i])
//
// All results are in range 0..1, and have the same ordering as the input values.
// - sum of all outputs is 1.0 (ideally, mathematically..)
// - if one value is distinctly greater than all the others, it will tend to be close to 1.0
//    and the others close to 0.0
//   The amount of margin needed for 'distinctly' depends on beta.
//
// Note that adding a constant amount to all inputs has no effect; and
// scaling the inputs by some factor is equivalent to changing beta; so we  can
// assume for the discussion that x[i] are 0..255  and beta has been thus scaled.
//
// To keep t[i] values in a manageable range this is typically done as
//       t[i] = exp( beta * (x[i]  - K))
//       y[i] = t[i]/sum(t[i])
//
// .. where K is max(x[i]), so that all the exp() parameters are <= 0, and
//     at least one is zero.
//
// When using floats, expf(x) is good up to x = 79.0  (even allowing up to 16k
// accumulations of the largest value); and is fully precise down to x = -87.
// (but can only go down to x= -83 for reasons given below).
// So, whenever beta *255 < 162, which is pretty much always, we can use K = 83/beta, and avoid the need to
// find the actual maximum ('beta' here refers to the scaled beta, which is 'stepsize' in code below).
// We need to ensure that 255.0/sum(expf) doesn't overflow; that gives the lower limit of -83.
//
static int qsoftmax_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	int i,j;
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
	float beta = (self->n_inputs < 4) ? 1.0f : tensor_get_float(self->inputs[3],0);
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float stepsize = beta*(in_max-in_min)/255.0f;
	const uint8_t *in = in_tensor->data;
	uint8_t *out = out_tensor->data;
	int maxval;
	int inval;
	float *temp_slice;
	float sum;
	float recip;
	logmsg(nn,2,"qsoftmax ref in=%dx%dx%dx%d beta=%f stepsize=%f",batches,height,width,depth,beta,stepsize);

#ifdef SOFTMAX_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	if ((temp_slice = nn_scratch_alloc(nn,depth*sizeof(float))) == NULL) {
		return errlog(nn,"can't alloc temp buffer");
	}
	if (tensor_out_prepare_normal(out_tensor,batches,height,width,depth,NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out too small");
	}
	if( tensor_set_single_float(out_min_tensor,0.0f) != 0
	    || tensor_set_single_float(out_max_tensor,1.0f) != 0){
		return errlog(nn,"can't prep min or max");
	}

	if( stepsize < 0.63529f){
		// stepsize * 255 < 162
		// so stepsize*255 - 83  <= 79, and we can do expf of the
		// whole range of (stepsize * uint8 -  83). Avoids the need for
		// a 'max' pass.
		// Note that 255./expf(-83) does not overflow (as when depth =1
		// and the only input is zero).
		//
		for (j = 0; j < batches*height*width; j++) {
			float sum = 0.0f;
			for (i = 0; i < depth; i++) {
				inval = in[j*depth+i];
				sum += (temp_slice[i] = expf_approx(stepsize*inval-83.0f));
			}
			recip = 255.0f/sum;
			for (i = 0; i < depth; i++) {
				out[j*depth+i] = saturate_u8(fast_roundf(recip*temp_slice[i]));
			}
		}
	}else{
		for (j = 0; j < batches*height*width; j++) {
			maxval = 0;
			sum = 0.0f;
			for (i = 0; i < depth; i++) {
				inval = in[j*depth+i];
				if (maxval < inval) maxval = inval;
			}
			for (i = 0; i < depth; i++) {
				inval = in[j*depth+i];
				sum += (temp_slice[i] = expf_approx(stepsize*(inval-maxval)));
			}
			/* best temp_slice is 1.0. */
			recip = 255.0f/sum;
			for (i = 0; i < depth; i++) {
				out[j*depth+i] = saturate_u8(fast_roundf(recip*temp_slice[i]));
			}
		}
	}
#ifdef SOFTMAX_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsoftmax ref cycles = %d\n",end_time-start_time);
#endif

	logmsg(nn,2,"qsoftmax ref done");
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8_ref = {
	.execute = qsoftmax_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(3,4),
	.n_outputs = NN_IOCOUNT(3),
};
// The hvx version of QuantizedSoftmax_8
// is in op_softmax_d32.c now
//

