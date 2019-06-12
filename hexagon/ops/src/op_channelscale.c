
/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
 * This contains implementations for quantized channel scale (qi32 * float -> qi32)
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>

static int channelscale_32xf_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *scale_tensor = self->inputs[1];
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	uint32_t bhw = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width;
	uint32_t depth = in_tensor->shape.depth;

	int32_t const *in = in_tensor->data;
	float const *scales = scale_tensor->data;
	int32_t *out = out_tensor->data;


	float in_min_float = tensor_get_float(in_min_tensor,0);
	float in_max_float = tensor_get_float(in_max_tensor,0);

	int scale_depth = scale_tensor->shape.depth;

	if (scale_tensor->shape.height != 1
			|| scale_tensor->shape.batches != 1
			|| scale_tensor->shape.width != 1) return errlog(nn,"bad scale shape");
	if (scale_depth != depth && scale_depth != 1) {
		return errlog(nn,"depth mismatch %d vs %d",scale_tensor->shape.depth,depth);
	}
	tensor_set_single_float(out_min_tensor, in_min_float);
	tensor_set_single_float(out_max_tensor, in_max_float);


	if( scale_depth == 1 && depth !=1){	// broadcast case
		if( scales[0] ==1.0f){	// 'bypass' case'
			if( tensor_copy( out_tensor, in_tensor)!=0) return errlog(nn,"out too small");
			return 0;
		}
		logmsg(nn,0, "** ChannelScale broadcasting from single value which is not 1.0");
		// force broadcast from 1.
		bhw *= depth;
		depth=1;
	}
	if(  tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_INT32)!= 0 ){
		return errlog(nn,"out too small");
	}
	nn_scratch_reset( nn);
	int32_t * scbuf = nn_scratch_alloc(nn, depth * sizeof(int32_t));
	if( scbuf==NULL) return errlog(nn,"scratch alloc failed");

	// convert all the scales to ints with 28 fractional bits.
	int any_bad_ones = 0;
	for( int i =0; i < depth; i++){
		float sc = scales[i];
		int x = roundf_i32( sc * (float)(1<<28));
		if( x <= 0 || x > (1<<28) || sc > 1.0001f || sc <= 0.0f){
			logmsg(nn,0, "bad  value scale[%d] = %f", i, sc);
			any_bad_ones ++;
		}
		scbuf[i] = x;
	}

	for(int i = 0; i < bhw; i++){
		for(int j = 0; j < depth; j++ ){
			int64_t prod = Q6_P_mpy_RR(scbuf[j], in[i*depth+j]);
			out[i*depth+j] = Q6_R_sat_P( Q6_P_asrrnd_PI( prod,28));
		}
	}
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedChannelScale_32xf = {
	.execute = channelscale_32xf_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};
