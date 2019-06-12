
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

/*
 *  method for finding mean & variance of a bunch of u8 values:
 *   - find their sum as u32
 *   - find the sums of square,in u64
 *   - variance is ( pop* ssq -   sum*sum) / (pop^2)
 *   - mean is sum/pop
 *     (all of which can  be done in integer, except for the divide).
 *   - so 1/sqrt(var) can be found as
 *         pop/ sqrt( pop* ssq -   sum*sum)
 *
 */
struct integer_acc {
	uint32_t sm;
	uint64_t ssq;
	uint8_t xmin,xmax;
};

static int execute_qinstancenorm_d32_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth = in_tensor->shape.depth;

	if (tensor_out_prepare_d32(out_tensor,batches,height,width,depth,NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out too small");
	}
	tensor_set_single_float( out_min_tensor, -6.0f );
	tensor_set_single_float( out_max_tensor, 6.0f );
	return errlog(nn,"FIXME: do d32 ref instance norm");
};

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
	size_t bytes = batches * width * height * depth;

	uint32_t tmp;
	if(nn_scratch_grow(nn,(sizeof(float) * depth * 5) + bytes )){
		return errlog(nn,"failed to get scratch");
	}

	// work area:
	//   sum:    struct integer_acc * [depth]
	//    mean, invsqrt_variance:   each float * [batches*depth]
	//   out_scale,out_offs: each int32_t[depth]

	int scratch_needed = ( sizeof(struct integer_acc)
				 + 2*batches* sizeof(float) + 2*sizeof(int32_t)) * depth;

	if( nn->scratch_size < scratch_needed ){
		return errlog(nn, "needed %d bytes of scratch", scratch_needed);
	}

	struct integer_acc *sum = nn->scratch;
	float *mean = (float *)(sum + depth);
	float *invsqrt_variance = mean + depth*batches;
	int32_t *out_scale = (int32_t*)(invsqrt_variance + depth*batches);
	int32_t *out_offs = out_scale + depth;
	float out_min = 0.0f;
	float out_max = 0.0f;

	const uint8_t *in_data = in_tensor->data;
	const uint8_t *data;
	uint8_t *out_data = out_tensor->data;
	//const float epsilon = tensor_get_float(epsilon_tensor,0);

	int32_t b,ihw,d;
	int32_t wh = width*height;
	//float in_min = tensor_get_float(in_min_tensor,0);
	//float in_max = tensor_get_float(in_max_tensor,0);
	//float in_level_size = (in_max-in_min)/255.0f;


    if( tensor_out_prepare_normal( out_tensor,batches,height,width,depth, NN_TYPE_QUINT8 )!=0) {
    	return errlog(nn,"out too small");
    }
	logmsg(nn,2,"set out tensor shape %dx%dx%dx%d",batches,height,width,depth);


	//
	// first pass finds scaling for each depth index, each batch;
	// these are kept in 'mean' and  'invsqrt_variance'
	// in the process, finds full range of output.
	//

	for (b = 0; b < batches; b++) {
		// find sums at each d index
		// also, the range of inputs present.
		memset(nn->scratch,0,sizeof(struct integer_acc)*depth);
		for(d = 0; d < depth; d++) sum[d].xmin = 0xFF;
		data = in_data + (b * wh * depth);

		for (ihw = 0; ihw <wh; ihw++) {
			for (d = 0; d < depth; d++) {
				tmp = *data++;
				sum[d].xmin = tmp < sum[d].xmin? tmp: sum[d].xmin;
				sum[d].xmax = tmp > sum[d].xmax? tmp: sum[d].xmax;
				sum[d].sm += tmp;
				sum[d].ssq += tmp*tmp;
			}
		}

		// in cases where xmin = xmax in a partition, 'del' value
		//  (ssq*pop- sm*sm) is always 0, and the output there will be 0.0.
		// an extreme case is e.g.
		//    999 of '100' and 1 of 101
		//   in which case
		//      xmean = 100.001
		//      d = 999 ->  stdev = 0.03160  = 1/31.638
		//      'normalization' gives
		//       999 of -0.0316
		//         1 of  31.607
		// (you'd get the same result for 999 of any one value, and 1 of a larger value).
		// so, stdev never gets really small with integer inputs, it's really just
		// zero, or non-zero.
		//

		for (d = 0; d < depth; d++) {
			int xmin = sum[d].xmin;
			int xmax = sum[d].xmax;
			float xmean, inv_stdev;
			if( xmax > xmin){
				/* Compute mean, variance */
				uint32_t sm = sum[d].sm;		// sum of all inputs.
				uint64_t del = sum[d].ssq * (unsigned)wh - (uint64_t)sm * (uint64_t)sm;
				// if we divide del by wh^2, we get the variance.
				// to find 1/sqrt(variance), divide wh by the sqrt(del).
				xmean = (float)sm / (float)(wh);
				inv_stdev = (float)wh/ ( sqrtf((float)del) + 0.00001f*(float)wh);
				invsqrt_variance[d] = inv_stdev;
				// find the range of results at this d index
				float ymin = ((float)xmin-xmean) * inv_stdev;
				float ymax = ((float)xmax-xmean) * inv_stdev;
				out_min = fminf( out_min,ymin);
				out_max = fmaxf( out_max, ymax);
			}else{
				// all are the same... all output will be zero for this d.
				xmean = 0.0f;
				inv_stdev= 0.0f;
			}
			mean[b*batches + d] = xmean;
			invsqrt_variance[b*batches + d] = inv_stdev;

		}
	} // b

	// adjust the output range so we have a proper zero
	out_max = fmaxf( out_max, out_min + 1e-5f);
	adjust_minmax_for_zero( &out_min, &out_max);

	float scl = 255.0f /( out_max-out_min);

	// second pass - for each [b,d], find the scaling needed for
	// that, and apply it with integer op.

	for (b = 0; b < batches; b++) {
		// go back through and find out_scale, out_offset to do the mapping in one step
		// i.e.
		//     y =  (x-xmean)*normfac
		//   out =    round( [ y  - out_min] * scl )
		//
		// where scl = 255/(outmax-outmin)
		//
		//  ->  out = round(  x*(normfac*scl)   - [xmean*normfac+out_min]*scl )
		//
		// and expressed with 10 fractional bits...
		//  -> out = (x * outk - out_offs)/1024
		//
		// note: all scale factors are >=0.5 or so here; it's unlikely for the normalization
		// to reduce the range of quint8 codes within a depth slot, should only happen
		// if some slots are skewed one way, and some another, so that aligning the means
		// requires a range expansion.
		// (exception: if a [b,d] slot
		// was found to contain all identical values, we'll have out_scale = 0,
		// and out_offs will be 1024*out_min*scl which results in the 'zero code' output).
		//
		//

		for (d = 0; d < depth; d++) {
			float xmean = mean[b*batches+d];
			float normfac = invsqrt_variance[b*batches+d];
			out_scale[d] = roundf_i32(1024.0f*normfac * scl);
 			out_offs[d] = roundf_i32(1024.0f * (xmean * normfac + out_min) *scl);
		}

		data = in_data + (b * width * height * depth);
		for (ihw = 0; ihw <wh; ihw++) {
			for (d = 0; d < depth; d++) {
				int val = *data * out_scale[d] - out_offs[d];
				*out_data ++ = saturate_u8( (val+512)>>10);
				data++;
			}
		}
	}
	tensor_set_single_float( out_min_tensor, out_min );
	tensor_set_single_float( out_max_tensor, out_max );

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
	size_t bytes = batches * width * height * depth * sizeof(float);

	float tmp;
	if(nn_scratch_grow(nn,(batches * sizeof(float) * depth * 5) + bytes )){
		return errlog(nn,"failed to get scratch");
	}
	float *sum = nn->scratch;
	float *sum_of_squares = sum + depth;
	// followed by 2nd sum, 2nd sum of squares..
	// (cascaded acc to reduce loss of precision)
	float *mean = sum_of_squares + depth*3;
	float *variance = mean + depth;
	float *invsqrt_variance = variance + depth;

	const float *in_data = (const float *)in_tensor->data;
	const float *data;
	float *out_data;
	//const float epsilon = tensor_get_float(epsilon_tensor,0);
	const float epsilon = 1.0e-5f;

	int32_t b,d;
	int32_t wh = width * height;
	int32_t ihw;



    if( tensor_out_prepare_normal( out_tensor,batches,height,width,depth, NN_TYPE_FLOAT )!=0) {
    	return errlog(nn,"out too small");
    }

	out_data = (float*) out_tensor->data;

	for (b = 0; b < batches; b++) {
		memset(nn->scratch,0,sizeof(float)*depth*4);	// clear sums
		data = in_data + (b * width * height * depth);
		for (ihw = 0; ihw <wh; ihw++) {
			for (d = 0; d < depth; d++) {
				tmp = *data++;
				sum[d] += tmp;
				sum_of_squares[d] += tmp*tmp;
			}
			if( ((ihw+1)&127)==0){
				// unload accums (slots 0,1) to slots 2,3
				for( int i = 0; i < depth*2;i++){
					float x = sum[i];
					sum[depth*2+i] += x;
					sum[i] = 0.0f;
				}
			}
		}
		// find mean & variance - adding the two parts of the
		// accs together.
		for (d = 0; d < depth; d++) {
			float sm = sum[d] + sum[d + depth*2];
			float ssq = sum_of_squares[d] + sum_of_squares[d + depth*2];
			float mn = sm/ (wh);
			mean[d] = mn ;
			variance[d] = (ssq/(wh)) - (mn*mn);
			invsqrt_variance[d] = 1.0f / sqrtf(variance[d] + epsilon);
		}
		data = in_data + (b * width * height * depth);
		for (ihw = 0; ihw <wh;ihw++) {
			for (d = 0; d < depth; d++) {
				*out_data++ = (*data++ - mean[d]) * invsqrt_variance[d];
			}
		}
	}
	return 0;
}



struct nn_node_ops nn_ops_for_QuantizedInstanceNorm_8_ref = {
	.execute = execute_qinstancenorm_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedInstanceNorm_8 = {
	.execute = execute_qinstancenorm_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedInstanceNorm_8_d32_ref = {
	.execute = execute_qinstancenorm_d32_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_InstanceNorm_f = {
	.execute = execute_finstancenorm,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};
