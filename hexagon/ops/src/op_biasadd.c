
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
 * This contains implementations for quantized bias add node
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>

static int biasadd_32p32to32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *bias_tensor = self->inputs[1];
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	const struct tensor *bias_min_tensor = self->inputs[4];
	const struct tensor *bias_max_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	uint32_t batches = in_tensor->shape.batches;
	uint32_t width = in_tensor->shape.width;
	uint32_t height = in_tensor->shape.height;
	uint32_t depth = in_tensor->shape.depth;

	int32_t *in = in_tensor->data;
	int32_t *bias = bias_tensor->data;
	int32_t *out = out_tensor->data;

	float in_max_float = tensor_get_float(in_max_tensor,0);
	float in_min_float = tensor_get_float(in_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);
	float bias_min_float = tensor_get_float(bias_min_tensor,0);

	float in_level_size = (in_max_float - in_min_float)/0x1.0p32f;
	float bias_level_size = (bias_max_float - bias_min_float)/0x1.0p32f;


	if (bias_tensor->shape.height != 1) return errlog(nn,"bias shape");
	if (bias_tensor->shape.batches != 1) return errlog(nn,"bias shape");
	if (bias_tensor->shape.width != 1) return errlog(nn,"bias shape");
	if (bias_tensor->shape.depth != depth) {
		return errlog(nn,"depth mismatch %d vs %d",bias_tensor->shape.depth,depth);
	}

	logmsg(nn,2,"biasadd32 execute. self=%p bhwd=%d,%d,%d,%d inmax=%f inmin=%f biasmax=%f biasmin=%f",self,batches,height,width,depth,in_max_float,in_min_float,bias_max_float,bias_min_float);

	if(  tensor_out_prepare_normal( out_tensor, batches,height,width,depth, NN_TYPE_INT32)!= 0 ){
		return errlog(nn,"out too small: %d > %d (id=%x)",sizeof(int32_t)*tensor_element_count(in_tensor),
				out_tensor->max_size,self->node_id);
	}
	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	float bias_to_in = bias_level_size/in_level_size;

	// get buffer for rescaled bias
	nn_scratch_reset(nn);
	int rescaled_bias_num = depth;
	if( depth& 1) rescaled_bias_num *= 2;	// double up so we can use 64-bit

	int32_t *rescaled_bias = nn_scratch_alloc( nn, sizeof(int32_t)* rescaled_bias_num );

	// scale the bias (we should do this once only...

	for( int i =0; i < depth; i++ ){
		rescaled_bias[i] = fast_roundf(bias[i]*bias_to_in);
	}
	// if depth is odd, make a second copy so we can use 64-bit adds.
	// e.g. if depth =9, we will make the rescaled_bias 18 long,
	// and work in units of 18).

	int nloops = height*width*batches;
	int perloop = depth;
	int remain = 0;
	if( depth & 1){
		memcpy( &rescaled_bias[depth], &rescaled_bias[0], depth*sizeof(int32_t));
		remain = nloops & 1;	//  need an extra loop at end?
		nloops >>= 1;			// half as many outer loops
	}else{
		perloop >>=1;		// depth/2 per loop if depth is even
	}


	int64_t const * __restrict inp = (int64_t const *)in;
	int64_t  *  __restrict outp = (int64_t *)out;

	int perloop_quads = perloop>>2;

	for( int outer = 0; outer < nloops; outer++){
		int64_t const *biasp = (int64_t const*)rescaled_bias;
		if (perloop&1){
			*outp++ = Q6_P_vaddw_PP(*inp++,*biasp++);
		}
		if (perloop&2){
			int64_t sum0 = Q6_P_vaddw_PP(inp[0],biasp[0]);
			int64_t sum1 = Q6_P_vaddw_PP(inp[1],biasp[1]);
			outp[0] = sum0;
			outp[1] = sum1;
			outp += 2;
			inp += 2;
			biasp += 2;
		}
		for( int i = 0; i < perloop_quads;i++){
			int64_t sum0 = Q6_P_vaddw_PP(inp[0],biasp[0]);
			int64_t sum1 = Q6_P_vaddw_PP(inp[1],biasp[1]);
			int64_t sum2 = Q6_P_vaddw_PP(inp[2],biasp[2]);
			int64_t sum3 = Q6_P_vaddw_PP(inp[3],biasp[3]);
			outp[0] = sum0;
			outp[1] = sum1;
			outp[2] = sum2;
			outp[3] = sum3;
			outp += 4;
			inp += 4;
			biasp += 4;
		}
	}

	if( remain ){	// odd depth, odd batches
		int32_t * op = (int32_t*)outp;
		int32_t const * ip = (int32_t const*)inp;
		for(int i = 0; i < depth; i++)
			op[i] = ip[i]+rescaled_bias[i];
	}
	return 0;
}

static int biasadd_8p8to32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *bias_tensor = self->inputs[1];
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	const struct tensor *bias_min_tensor = self->inputs[4];
	const struct tensor *bias_max_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	uint32_t batches = in_tensor->shape.batches;
	uint32_t width = in_tensor->shape.width;
	uint32_t height = in_tensor->shape.height;
	uint32_t depth = in_tensor->shape.depth;

	int32_t stripe;

	uint8_t *in = in_tensor->data;
	uint8_t *bias = bias_tensor->data;
	int32_t *out = out_tensor->data;

	int32_t i;

	float in_mpy_amt;
	int32_t in_sub_amt;
	float bias_mpy_amt;
	int32_t bias_sub_amt;

	//int32_t indata;
	//int32_t biasdata;
	//int32_t outdata;

	float in_max_float = tensor_get_float(in_max_tensor,0);
	float in_min_float = tensor_get_float(in_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);
	float bias_min_float = tensor_get_float(bias_min_tensor,0);

	float out_min;
	float out_max;
	float out_level_size;

	float in_level_size = (in_max_float - in_min_float)/255;
	float bias_level_size = (bias_max_float - bias_min_float)/255;

	/* Assert min and max are size 1,1,1,1 ? */

	if (bias_tensor->shape.height != 1) return errlog(nn,"bias shape");
	if (bias_tensor->shape.batches != 1) return errlog(nn,"bias shape");
	if (bias_tensor->shape.width != 1) return errlog(nn,"bias shape");
	if (bias_tensor->shape.depth != depth) {
		return errlog(nn,"depth mismatch %d vs %d",bias_tensor->shape.depth,depth);
	}

	logmsg(nn,2,"biasadd execute. self=%p bhwd=%d,%d,%d,%d",self,batches,height,width,depth);
	if(  tensor_out_prepare_normal( out_tensor, batches,height,width,depth, NN_TYPE_INT32)!= 0 ){
		return errlog(nn,"out too small: %d > %d (id=%x)",sizeof(int32_t)*tensor_element_count(in_tensor),
				out_tensor->max_size,self->node_id);
	}

	/*
	 * We have two input sets, X and B.
	 * X is represented by three values: X[i], Xmin, Xmax
	 * B is represented by three values: B[i], Bmin, Bmax
	 * The true value for an X element is X[i]*(Xmax-Xmin)/255 + Xmin
	 * In order to add values, we need to have common min/max
	 * 
	 * Find maximum magnitude of Xmin, Xmax, Bmin, Bmax.  
	 * New range is +/- max_mag * 2**N
	 * We increase precision to 32 bits, and need to convert both X and B
	 * to the new space.
	 * 
	 */

	out_max = fmaxf(fabsf(in_min_float),in_max_float);
	out_max = fmaxf(out_max,fmaxf(fabsf(bias_min_float),bias_max_float));
	out_max *= (1 << 17);
	out_min = -out_max;
	out_level_size = (out_max - out_min) / 0x1.0p32f;

	in_sub_amt = quantize_uint8(0.0f,in_min_float,in_max_float);
	bias_sub_amt = quantize_uint(0.0f,bias_min_float,bias_max_float);

	in_mpy_amt = (in_level_size / out_level_size);
	bias_mpy_amt = (bias_level_size / out_level_size);

	logmsg(nn,9,"biasadd in min/max: %f/%f bias min/max: %f/%f",in_min_float,in_max_float,bias_min_float,bias_max_float);
	logmsg(nn,9,"biasadd: in/bias/out level size: %f/%f/%f",in_level_size,bias_level_size,out_level_size);

	tensor_set_single_float(out_min_tensor, out_min);
	tensor_set_single_float(out_max_tensor, out_max);

	logmsg(nn,9,"biasadd: in sub/mpy: %x/%f bias sub/mpy: %x/%f",
		in_sub_amt,in_mpy_amt,bias_sub_amt,bias_mpy_amt);

	for (stripe = 0; stripe < width*height*batches; stripe++) {
		for (i = 0; i < depth; i++) {
			out[i] = (int)( ((in[i] - in_sub_amt) * in_mpy_amt) + (((int32_t)bias[i] - bias_sub_amt) * bias_mpy_amt)+0.5f);
			//indata = round((in[i] - in_sub_amt) * in_mpy_amt);
			//biasdata = round((bias[i] - bias_sub_amt) * bias_mpy_amt);
			//outdata = indata + biasdata;
			/* out is signed int32, and min=-max, so 0 is 0 */
			//out[i] = outdata;
/*
			logmsg(nn,9,"biasadd [%d,%d] %x = %x+%x\n (~= %f)",
				stripe,depth,outdata,indata,biasdata,outdata*out_level_size);
*/
		}
		in += depth;
		out += depth;
	}
	logmsg(nn,2,"biasadd %p done",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedBiasAdd_8p8to32 = {
	.execute = biasadd_8p8to32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedBiasAdd_8p8to32_ref = {
	.execute = biasadd_8p8to32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedBiasAdd_32p32to32 = {
	.execute = biasadd_32p32to32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedBiasAdd_32p32to32_ref = {
	.execute = biasadd_32p32to32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};


