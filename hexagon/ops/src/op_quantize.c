
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
#include <hvx_hexagon_protos.h>
#include <hexagon_types.h>
#include <stdio.h>
#include "hvx_inlines.h"

/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains quantize / dequantize ops
 */

static int quantize32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *min_tensor = self->inputs[1];
	const struct tensor *max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float min_in = tensor_get_float(min_tensor,0);
	float max_in = tensor_get_float(max_tensor,0);
	float recip_stepsize;
	float min_out;
	float max_out;
	float batches = in_tensor->shape.batches;
	float height = in_tensor->shape.height;
	float width = in_tensor->shape.width;
	float depth = in_tensor->shape.depth;
	const float *in = in_tensor->data;
	int32_t *out = out_tensor->data;
	int out_bytes = batches*height*width*depth*4;
	float inval;
	int i;
	int ival;
	logmsg(nn,2,"quantize32 execute. self=%p ",self);
	if( tensor_out_prepare_normal( out_tensor, batches,height,width,depth, NN_TYPE_INT32)!= 0 ){
		return errlog(nn,"out too small for node %d (%p) %d < %d",
									    out_tensor, out_tensor->max_size,out_bytes);
	}

	max_out = fmaxf(-min_in,max_in);
	min_out = -max_out;
	recip_stepsize = 0x1.0p24/(max_out-min_out);
	logmsg(nn,2,"min_out=%f max_out=%f recip_stepsize=%f",min_out,max_out,recip_stepsize);

	for (i = 0; i < batches*height*width*depth; i++) {
		inval = in[i];
		ival = roundf_i32(inval*recip_stepsize);
		out[i] = ival;
		logmsg(nn,2,"i=%d inval=%f ival=%d",i,inval,ival);
	}
	max_out *= 256.0f;
	min_out *= 256.0f;

	tensor_set_single_float(out_min_tensor, min_out);
	tensor_set_single_float(out_max_tensor, max_out);
	return 0;
}


static int quantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *min_tensor = self->inputs[1];
	const struct tensor *max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float min_in = tensor_get_float(min_tensor,0);
	float max_in = tensor_get_float(max_tensor,0);
	float recip_stepsize;
	float min_out;
	float max_out;
	float stepsize;
	float batches = in_tensor->shape.batches;
	float height = in_tensor->shape.height;
	float width = in_tensor->shape.width;
	float depth = in_tensor->shape.depth;
	const float *in = in_tensor->data;
	uint8_t *out = out_tensor->data;
	int out_bytes = batches*height*width*depth;
	float inval;
	int i;
	int ival;
	logmsg(nn,2,"quantize execute. self=%p ",self);

	if( tensor_out_prepare_normal( out_tensor, batches,height,width,depth, NN_TYPE_QUINT8)!= 0 ){
		return errlog(nn,"out too small for node %d (%p) %d < %d",
			      self->node_id, out_tensor, out_tensor->max_size,out_bytes);
	}
	quantize_adjust_range(
		&min_out,&max_out,
		&stepsize,&recip_stepsize,
		min_in,max_in);
	for (i = 0; i < batches*height*width*depth; i++) {
		inval = in[i];
		ival = roundf_i32((inval - min_out)*recip_stepsize);
		out[i] = saturate_u8(ival);
	}

	tensor_set_single_float(out_min_tensor, min_out);
	tensor_set_single_float(out_max_tensor, max_out);
	return 0;
}


#define MANTBITS 23
static inline HVX_Vector requantize_words(HVX_Vector vals, int scaling, int min_offset_in, int maxexp_in)
{
	const HVX_Vector const_zero = Q6_V_vsplat_R(0);
	const HVX_Vector const_007f_ffff = Q6_V_vsplat_R(0x007fffff);
	const HVX_Vector const_0000_00ff = Q6_V_vsplat_R(0x000000ff);
	const HVX_Vector const_0080_0000 = Q6_V_vsplat_R(0x00800000);
	const HVX_Vector const_31 = Q6_V_vsplat_R(31);
	const HVX_Vector min_offset = Q6_V_vsplat_R(min_offset_in);
	const HVX_Vector maxexp = Q6_V_vsplat_R(maxexp_in);
	const uint32_t r_scaling = Q6_R_combine_RlRl(scaling,scaling);
	HVX_VectorPred p_neg;
	HVX_Vector mant,exp;
	HVX_Vector tmp;
	HVX_Vector shift;
	/* Check for negative values */
	p_neg = Q6_Q_vcmp_gt_VwVw(const_zero,vals);
	/* Extract exponent and mantissa, add back hidden 1 */
	exp = Q6_Vuw_vlsr_VuwR(vals,MANTBITS);
	mant = Q6_V_vand_VV(vals,const_007f_ffff);
	exp = Q6_V_vand_VV(exp,const_0000_00ff);
	mant = Q6_V_vor_VV(mant,const_0080_0000);
	/* We need to shift right small exponents */
	shift = Q6_Vw_vsub_VwVw(maxexp,exp);
	shift = Q6_Vw_vmin_VwVw(shift,const_31);
	mant = Q6_Vw_vlsr_VwVw(mant,shift);
	tmp = Q6_Vw_vsub_VwVw(const_zero,mant);
	/* Turn negative values into two's complement negative values */
	mant = Q6_V_vmux_QVV(p_neg,tmp,mant);
	/* Add min_offset to turn to unsigned */
	mant = Q6_Vw_vadd_VwVw(mant,min_offset);
	/* Note the range here should be 0 .. 0x00ff_ffff */
	/* Multiply up to 0xFFFF_FFFF */
	mant = Q6_Vw_vmpyi_VwRh(mant,r_scaling);
	return mant;
}

#if 0
static inline HVX_Vector quantize_4vec(
	const HVX_Vector in0,
	const HVX_Vector in1,
	const HVX_Vector in2,
	const HVX_Vector in3,
	int scaling,
	int min_offset,
	int maxexp)
{
	HVX_Vector words0,words1,words2,words3;
	HVX_Vector halves0,halves1;
	HVX_Vector bytes;
	/* We want to end up with a vector of bytes, so repeat 4x */
	words0 = requantize_words(in0,scaling,min_offset,maxexp);
	words1 = requantize_words(in1,scaling,min_offset,maxexp);
	words2 = requantize_words(in2,scaling,min_offset,maxexp);
	words3 = requantize_words(in3,scaling,min_offset,maxexp);
	/* Should have values in MSB of words, pack together */
	halves0 = Q6_Vh_vpacko_VwVw(words1,words0);
	halves1 = Q6_Vh_vpacko_VwVw(words3,words2);
	bytes = Q6_Vb_vpacko_VhVh(halves1,halves0);
	return bytes;
}

#define BLOCK_SIZE     (16*1024/128)  //# of vectors per chunk
// find range of floats in n values starting
// at *ptr (which must be vector aligned).
// The range is always assumed to include 0.0, even
// if that's not in the data.
//
// The values are stored to a *vector* at *out;
// note that 'min' is generated without sign.
//
// (*out).w[0] = - min( 0.0,  min_of_all )
// (*out).w[1] = max( 0.0,  max_of_all )
//
static void
find_minmax_of_floats_hvx( float const * ptr, uint32_t nvals, HVX_Vector *out )
{
	/* Find min and max, always inclusive of zero */
	int i, n;
	HVX_Vector const *in = (HVX_Vector const *)ptr;
	HVX_Vector const_zero = Q6_V_vzero();
	HVX_Vector const_80000000 = Q6_V_vsplat_R(0x80000000u);
	HVX_Vector vmin = const_zero;
	HVX_Vector vmax = const_zero;

	int vectors_in_rounddown = (nvals)/32;
	int leftover_precise_elements = nvals & 31;
	/*
	 * Strategy: since we have signed-magnitude floats, max will discard all neg values,
	 * For min, toggle sign bit and take max
	 */
	int block = Q6_R_min_RR(vectors_in_rounddown, BLOCK_SIZE);
	l2fetch(in, 128, 128, block);
	for (n = 0; n < vectors_in_rounddown; n += BLOCK_SIZE) {
		int next_block = Q6_R_min_RR(vectors_in_rounddown-n-block, BLOCK_SIZE);
		wait_for_l2fetch();
		if (next_block > 0) l2fetch(&in[n+BLOCK_SIZE], 128, 128, next_block);
		for (i = 0; i < block; i++) {
			HVX_Vector new = in[n + i];
			vmax = Q6_Vw_vmax_VwVw(vmax,new);
			vmin = Q6_Vw_vmax_VwVw(vmin,Q6_V_vxor_VV(new,const_80000000));
		}
		block = next_block;
	}
	if (leftover_precise_elements) {
		HVX_VectorPred partial_mask = Q6_Q_vsetq_R(leftover_precise_elements*4);
		HVX_Vector partial = in[vectors_in_rounddown];
		partial = Q6_V_vmux_QVV(partial_mask,partial,const_zero);
		vmax = Q6_Vw_vmax_VwVw(vmax,partial);
		vmin = Q6_Vw_vmax_VwVw(vmin,Q6_V_vxor_VV(partial,const_80000000));
	}

	/* Reduce max and min_mag */
	HVX_VectorPair shuf =  Q6_W_vshuff_VVR( vmax,vmin, 4 );
	// 'minmax' has 'min' in even lanes, 'max' in odd.
	// 4 more reductions...
	HVX_Vector minmax = Q6_Vw_vmax_VwVw( Q6_V_lo_W(shuf), Q6_V_hi_W(shuf));

	minmax = Q6_Vw_vmax_VwVw(minmax,Q6_V_vror_VR(minmax,8));
	minmax = Q6_Vw_vmax_VwVw(minmax,Q6_V_vror_VR(minmax,16));
	minmax = Q6_Vw_vmax_VwVw(minmax,Q6_V_vror_VR(minmax,32));
	minmax = Q6_Vw_vmax_VwVw(minmax,Q6_V_vror_VR(minmax,64));

	*out = minmax;
	Q6_dcfetch_A(out);
}
#endif

//
// parms for converting from float.
// Procedure for input 'x', in the hvx code, is:
//    (1) extract mantissa, restoring 'hidden bit', to get 24 bit #
//    (2) >> (common_exp- exp), where 'exp' is the exponent field from input.
//       (if >> more than 24, result is 0)
//    (3) negate, if x input has sign bit
//    (4) add 'min_offs' to that
//    (5) multiply the result (considered unsigned) by uint16 value 'scaling'.
//        Result is guaranteed (by contrivance of the numbers,vs known min,max)
//        to fit in u32
//    (6) upper 8 bits of the 32-bit product are the quantized result.
//
struct hvx_quant_parms {
	int scaling;
	int min_offset;
	int common_exp;
	float minval, maxval;
};
// work out the 'scale', 'min_offs' and 'common_exp'
// which are used in the hvx code to do quantizing.
// Note: the input is actually {-min and max} (both >= 0)
//
// The output "struct hvx_quant_parms" has the scaling parms
// and also the 'adjusted' min and max.
//
static inline void
find_scaling_for_hvx_quant ( float const minmax[2], struct hvx_quant_parms *out)
{
	union {
		float f;
		uint32_t i;
		struct {
			uint32_t mant:23;
			uint32_t exp:8;
			uint32_t sign:1;
		};
	} minval,range;
	float range_f;
	uint32_t min_offset;
	uint32_t common_exp;
	uint32_t minval_exp;
	uint32_t rangemant;
	uint32_t rangeexp;
	uint32_t scaling;
	float minv = -minmax[0];
	float maxv = fmaxf( minv+1e-5f, minmax[1]);

	if( minv < 0.0f)	// make sure the 'zero' is an integer
		adjust_minmax_for_zero( & minv , & maxv);

	minval.f = minv;
	range_f = maxv-minv;
	range.f = range_f * 1.003921569f;	// * 256/255 to get scale ratio
	rangeexp = range.exp;
	common_exp = rangeexp +4;		// use as 'baseline' exponent
									// all mantissas will need >> by at least 3 or 4 bits
	minval_exp = minval.exp;
	min_offset = minval.mant | (1<<23);
	if ((common_exp-minval_exp) >= 24) min_offset = 0;
	else min_offset >>= common_exp-minval_exp;

	rangemant = range.mant | (1<<23);
	rangemant >>= common_exp - rangeexp;

	// since 'rangemant' will be about 2^19, 'scaling' will be about 2^13.
	// The back end will denorm mantissa, add min_offset, mul by 'scaling', and then
	// wind up with a value ranging up to about 0xFF000000 (for input = maxval).
	// we want that to be 2^23 larger, for a rounding offset; so add (2^23/scaling) to min_offset,
	// since scaling = 2^32/rangemant, that value is about rangemant/2^9.
	scaling = 0xFFFFFFFFU/rangemant;
	min_offset += rangemant >> 9;			// for rounding bias at the back end.

	out->min_offset = min_offset;
	out->scaling = scaling;
	out->common_exp = common_exp;

	out->minval = minv;
	out->maxval = maxv;
}

#if 0
static void autoquantize_hvx(struct nn_graph *nn, void *info)
{
	struct nn_node *self = info;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	const HVX_Vector const_zero = Q6_V_vsplat_R(0);
	HVX_Vector *in = in_tensor->data;
	HVX_Vector *out = out_tensor->data;
	HVX_Vector in0,in1,in2,in3, out_leftovers;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;
	unsigned int elements = batches*height*width*depth;
	unsigned int vectors_in_rounddown = elements/32;
	unsigned int out_leftovers_elements = elements % 128;
	int i, n;

	union {
		HVX_Vector v;
		float fbuf[32];
	} *minmaxbuf = nn->scratch;
	struct hvx_quant_parms qparms;

	uint32_t min_offset;
	uint32_t common_exp;
	uint32_t scaling;

	// find the range
	find_minmax_of_floats_hvx( (float const *)in, elements, &minmaxbuf->v );
	// compute the scaling parms (and corrected range)

	find_scaling_for_hvx_quant( minmaxbuf[0].fbuf, &qparms);

	scaling = qparms.scaling;
	min_offset = qparms.min_offset;
	common_exp = qparms.common_exp;

	tensor_set_float(out_min_tensor,0,qparms.minval);
	tensor_set_float(out_max_tensor,0,qparms.maxval);
	logmsg(nn,2,"minval=%f maxval=%f range=%f scaling = %d, common_exp = %d min_offset=%x",
			qparms.minval,qparms.maxval, qparms.maxval-qparms.minval, scaling, common_exp, min_offset);

	/* Quantize! */
	int block = Q6_R_min_RR(4*(vectors_in_rounddown>>2), BLOCK_SIZE);
	l2fetch(in, 128, 128, block);
	for (n = 0; n < 4*(vectors_in_rounddown>>2); n += BLOCK_SIZE) {
		int next_block = Q6_R_min_RR(vectors_in_rounddown-n-block, BLOCK_SIZE);
		wait_for_l2fetch();
		if (next_block > 0) l2fetch(&in[block], 128, 128, next_block);
		for (i = 0; i < (block>>2); i++) {
			*out++ = quantize_4vec(in[0],in[1],in[2],in[3],scaling,min_offset,common_exp);
			in += 4;
		}
		block = next_block;
	}

	if(out_leftovers_elements){
		in0 = in1 = in2 = in3 = const_zero;
		in0 = in[0];
		if (out_leftovers_elements > 32) in1 = in[1];
		if (out_leftovers_elements > 64) in2 = in[2];
		if (out_leftovers_elements > 96) in3 = in[3];
		out_leftovers = quantize_4vec(in0,in1,in2,in3,scaling,min_offset,common_exp);
		memcpy(out, &out_leftovers, out_leftovers_elements);
	}
	nn_sem_post(self->opaque);
}
#else
static void autoquantize_hvx(struct nn_graph *nn, void *info)
{
	struct nn_node *self = info;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	const float *in = in_tensor->data;
	uint8_t *out = out_tensor->data;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;
	int elements = batches*height*width*depth;

	union {
		HVX_Vector v;
		float fbuf[32];
	} *minmaxbuf = nn->scratch;

	struct hvx_quant_parms qparms;

	// find the range
	find_minmax_of_floats_asm((float const *)in, elements, minmaxbuf->fbuf );

	// compute the scaling parms (and corrected range)
	find_scaling_for_hvx_quant(minmaxbuf->fbuf, &qparms);

	tensor_set_float(out_min_tensor,0,qparms.minval);
	tensor_set_float(out_max_tensor,0,qparms.maxval);
	logmsg(nn,2,"minval=%f maxval=%f range=%f scaling = %d, common_exp = %d min_offset=%x",
			qparms.minval,qparms.maxval, qparms.maxval-qparms.minval, qparms.scaling, qparms.common_exp, qparms.min_offset);

	/* Quantize! */
	quantize_floats_to_8b_asm(in,out, elements, qparms.min_offset, qparms.common_exp, qparms.scaling);
    
	nn_sem_post(self->opaque);
}
#endif

//
// Multi-thread version of autoquantization
//
#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif
struct tdata_autoq {
	struct nn_node  *self;
	nn_sem_t        *donesem;
	const float     *in;
	uint8_t         *out;
	float           *minmax;
	int32_t         elements;
	uint32_t        min_offset;
	uint32_t        common_exp;
	uint32_t        scaling;
};

static void autoq_execute_find_minmax_work(struct nn_graph *nn, void *info)
{
	struct tdata_autoq *work = info;
	find_minmax_of_floats_asm(work->in, work->elements, work->minmax);
	nn_sem_post(work->donesem);
}

static void autoq_execute_quantization_work(struct nn_graph *nn, void *info)
{
	struct tdata_autoq *work = info;
	quantize_floats_to_8b_asm(work->in, work->out, work->elements, work->min_offset,work->common_exp,work->scaling);
	nn_sem_post(work->donesem);
}

static void autoquantize_work(
	struct nn_graph *nn,
	const struct tensor *in_tensor,
	struct tensor *out_tensor,
	struct tensor *out_min_tensor,
	struct tensor *out_max_tensor,
	int elements )
{
	const float *in = (const float *)in_tensor->data;
	uint8_t *out = out_tensor->data;

	union {
		HVX_Vector v;
		float f[32];
	} *minmaxbuf = nn->scratch;

	struct tdata_autoq work[NUM_THREADS];
	nn_sem_t donesem;
	nn_sem_init(&donesem,0);

	int t_elements = ((elements + NUM_THREADS-1)/NUM_THREADS + 127)&~127;
	int n_threads = (elements + t_elements -1)/t_elements;
	int i, start;

	// setup parameters for multi-thread 
	for (i=0; i < n_threads; i++) {
		start = i * t_elements;
		work[i].in  = in  + start;
		work[i].out = out + start;
		work[i].elements = Q6_R_min_RR(elements-start, t_elements);
		work[i].minmax = minmaxbuf[i].f;
		work[i].donesem = &donesem;
	}

	// find the range
	for (i=0; i < n_threads; i++) {
		nn_os_work_for_vector(nn, autoq_execute_find_minmax_work, &work[i]);
	}

	for (i=0; i < n_threads; i++) nn_sem_wait(&donesem);
		
	for (i=1; i < n_threads; i++) {
		minmaxbuf[0].f[0] = Q6_R_sfmax_RR(minmaxbuf[0].f[0],minmaxbuf[i].f[0]);
		minmaxbuf[0].f[1] = Q6_R_sfmax_RR(minmaxbuf[0].f[1],minmaxbuf[i].f[1]);
	}

	// compute the scaling parms (and corrected range)
	struct hvx_quant_parms qparms;

	find_scaling_for_hvx_quant(minmaxbuf[0].f, &qparms);

	tensor_set_float(out_min_tensor,0,qparms.minval);
	tensor_set_float(out_max_tensor,0,qparms.maxval);

	logmsg(nn,2,"minval=%f maxval=%f range=%f scaling = %d, common_exp = %d min_offset=%x",
			qparms.minval,qparms.maxval, qparms.maxval-qparms.minval, qparms.scaling, qparms.common_exp, qparms.min_offset);

	/* Quantize! */
	for (i=0; i < n_threads; i++) {
		work[i].min_offset = qparms.min_offset;
		work[i].common_exp = qparms.common_exp;
		work[i].scaling = qparms.scaling;
		nn_os_work_for_vector(nn, autoq_execute_quantization_work, &work[i]);
	}

	for (i=0; i < n_threads; i++) nn_sem_wait(&donesem);
		
	return;
}

static int autoquantize_execute_opt(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;
	int elements = batches*height*width*depth;

	if( tensor_out_prepare_normal(out_tensor,batches,height,width,depth, NN_TYPE_QUINT8)!=0 ){
		return errlog(nn,"out too small for node %d to be %dx%dx%dx%d (%p) %d < %d",
			      self->node_id, batches, height, width, depth,
			      out_tensor, out_tensor->max_size, elements);
	}

	if( tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT )!=0
	  || tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT )!=0 ){
		return errlog(nn,"out min or max too small");
	}

	nn_sem_t sem;
	nn_sem_init(&sem,0);
	self->opaque = &sem;
	nn_os_work_for_vector(nn,autoquantize_hvx,self);
	nn_sem_wait(&sem);
	self->opaque = NULL;
	return 0;
}
static int autoquantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float min_in = 0.0f;
	float max_in = 0.0f;
	float recip_stepsize;
	float min_out;
	float max_out;
	float stepsize;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;
	const float *in = (const float *)in_tensor->data;
	uint8_t *out = (uint8_t *)out_tensor->data;
	int out_bytes = batches*height*width*depth;
	float inval;
	int i;
	int ival;
	logmsg(nn,2,"autoquantize execute. self=%p ",self);

	if( tensor_out_prepare_normal( out_tensor,batches,height,width,depth, NN_TYPE_QUINT8)!=0 ){
		return errlog(nn,"out too small for node %d %d < %d",self->node_id,out_tensor->max_size,out_bytes);
	}

	for (i = 0; i < batches*height*width*depth; i++) {
		min_in = fminf(min_in,in[i]);
		max_in = fmaxf(max_in,in[i]);
	}
	logmsg(nn,2,"min=%f max=%f bhwd=%d,%d,%d,%d",min_in,max_in,batches,height,width,depth);
	quantize_adjust_range(
		&min_out,&max_out,
		&stepsize,&recip_stepsize,
		min_in,max_in);
	for (i = 0; i < batches*height*width*depth; i++) {
		inval = in[i];
		ival = roundf_i32((inval - min_out)*recip_stepsize);
		out[i] = saturate_u8(ival);
	}
	tensor_set_single_float(out_min_tensor,min_out);
	tensor_set_single_float(out_max_tensor,max_out);
	return 0;
}
//////////////// AutoQuantize d32 format ///////////////////

struct autoquant_d32_work {
	float const *flt_arr;			// input data
	struct shape shape;				// shape of data
	uint32_t n_elements_per_batch;	// width*height*depth
	uint8_t * out_arr;				// pointer to 1st output element (after padding)

	int32_t out_batch_stride, out_row_stride, out_d32_stride;


	nn_sem_t done_sem;
	union {
		float as_flt[32];
		HVX_Vector as_v;
	} minmax_union;
	struct hvx_quant_parms qparms;
};
static void quant_to_d32_operation(struct nn_graph *nn, void * wrkpv );
static void quant_do_d32_core_op(
		struct autoquant_d32_work *wrkp,
		float const *inptr,		// pointer to the 1st input float
		uint8_t * optr,			// first output (32 aligned)
		int depth,				// depth to process (1..32)
		int d0);			// depth padding on left (dnow+d0 <= 32)

// find the min & max (in a thread)
static void autoq_d32_find_minmax( struct nn_graph *nn, void *wrkpv )
{
	struct autoquant_d32_work *wrkp = (struct autoquant_d32_work*) wrkpv;
//	find_minmax_of_floats_hvx( wrkp->flt_arr,
//			wrkp->shape.batches*wrkp->n_elements_per_batch,
//			&wrkp->minmax_union.as_v );
	find_minmax_of_floats_asm( wrkp->flt_arr,
			wrkp->shape.batches*wrkp->n_elements_per_batch,
			(float *)&wrkp->minmax_union.as_v );
	nn_sem_post( & wrkp->done_sem);

}
static int get_option( struct nn_graph * nn, const struct tensor * tens_in, int default_val, char const *option_desc, int maxval)
{
	if( tens_in == NULL) return default_val;
	int newval = tensor_get_int32( tens_in,  0 );
	if( newval < 0 || newval > maxval){
		logmsg(nn,0,"autoquant_d32: value %d out of range (0..%d) for %s; using default of %d",
				newval, maxval, option_desc, default_val );
		return default_val;
	}else{
		return newval;
	}
}

static int autoquantize_d32_execute_opt(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;

	struct autoquant_d32_work wrk;

	// do the min/max on a thread
	wrk.shape = in_tensor->shape;
	wrk.flt_arr = (float const*) in_tensor->data;
	wrk.n_elements_per_batch = height*width*depth;

	nn_sem_init(&wrk.done_sem,0);
	nn_os_work_for_vector(nn,autoq_d32_find_minmax,&wrk);
	nn_sem_wait( &wrk.done_sem );

	//----
	//
	// now allocate the output array
	//
	// process optional padding
	int d_pad_before = 0;		// defaults
	int w_pad_left = 4;
	int w_pad_right_min = 0;
	int h_pad_top = 4;

	if( self->n_inputs >=2){
		d_pad_before = get_option( nn, self->inputs[1], d_pad_before, "depth padding", MAX_PADDING_DEPTH );
		if( self->n_inputs >=3 )
			w_pad_left = get_option( nn, self->inputs[2], w_pad_left, "width padding(left)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=4 )
			w_pad_right_min = get_option( nn, self->inputs[3], w_pad_right_min, "width padding (min right)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=5 )
			h_pad_top = get_option( nn, self->inputs[4], h_pad_top, "height padding", MAX_PADDING_HEIGHT );
	}
	int wtotal = (w_pad_left + width + w_pad_right_min + 3)&~3;
	int w_pad_right = wtotal - (w_pad_left + width);
	if( w_pad_right > MAX_PADDING_WIDTH) w_pad_right -= 4;

	int d_pad_after = (-(depth+d_pad_before))&31;
	int h_pad_bottom = h_pad_top;

	if (tensor_out_prepare_padded_d32(
		out_tensor,
		batches,
		height, h_pad_top, h_pad_bottom,
		width,  w_pad_left, w_pad_right,
		depth, d_pad_before, d_pad_after,
		NN_TYPE_QUINT8) != 0) {

		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,batches,height,width,depth);
	}
	if (out_min_tensor->max_size < sizeof(float)) return errlog(nn,"out min too small");
	if (out_max_tensor->max_size < sizeof(float)) return errlog(nn,"out max too small");

	wrk.out_batch_stride = tensor_batch_stride_d32( out_tensor);
	wrk.out_row_stride = tensor_row_stride_d32( out_tensor);
	wrk.out_d32_stride = tensor_d32_stride_d32( out_tensor);
	wrk.out_arr = tensor_location_d32(out_tensor, 0,0,0,0);

	// now find the scaling
	find_scaling_for_hvx_quant ( wrk.minmax_union.as_flt, &wrk.qparms );

	// launch thread to do the scaling...
	nn_sem_init(&wrk.done_sem,0);
	nn_os_work_for_vector(nn,quant_to_d32_operation,&wrk);


	tensor_set_single_float(out_min_tensor,wrk.qparms.minval);
	tensor_set_single_float(out_max_tensor,wrk.qparms.maxval);

	// wait for done;
	//
	nn_sem_wait( &wrk.done_sem );
	return 0;

}

//
// function, run in thread, for quantizing to d32 format
//
// This function loops over batches & depth slices
//
//
static void quant_to_d32_operation(struct nn_graph *nn, void * wrkpv )
{
	struct autoquant_d32_work *wrkp = (struct autoquant_d32_work*) wrkpv;
	uint8_t *outp00 = wrkp->out_arr;

	int batch_index,dn_index;
	int batches = wrkp->shape.batches;
	int depth = wrkp->shape.depth;
	int d32_stride = wrkp->out_d32_stride;
	int batch_stride = wrkp->out_batch_stride;
	int el_per_batch = wrkp->n_elements_per_batch;

	int doffs= (size_t)outp00 & 31;				// depth offset

	int dn = (unsigned)(doffs + depth+31)/32;		// # of depth slices

	uint8_t * outp = outp00 - doffs;			// remove depth padding (align to 32)
	float const* inptr0 = wrkp->flt_arr;

	for( batch_index = 0; batch_index < batches; batch_index ++){
		int d0= doffs;
		int dremain = depth;
		float const * inptr = inptr0 + batch_index * el_per_batch;
		for( dn_index = 0; dn_index < dn; dn_index++){
			int dnow = min_i32( 32-d0, dremain);
			quant_do_d32_core_op (wrkp,
					inptr,
					outp + batch_index * batch_stride + dn_index * d32_stride,
					dnow,
					d0 );
			inptr += dnow;
			dremain -= dnow;
			d0 = 0;
		}
	}

	nn_sem_post( & wrkp->done_sem);

}
//
// this processes a 'depth slice' of height x width x depth,
// where
//   - depth <=32
//   - addressing input needs to respect in_width_stride (which is the 'full depth')
//              (and  row_stride = in_width_stride * width per row).
//   - 'optr' is 32 aligned; out_row_stride is vector aligned
//
//  d0 is the offset, in the output array, to start writing depth.  d0+depth <=32.
//
// The operation is done as follows:
//
//
//  (1) 'left operation' handes 1..4 columns, and takes us to a a vector boundary in the
//    output. This is necessary if there is width padding on the left which is not a multiple
//    of 4. It will also be done in cases where there is no left padding, but width < 8, so the
//    'middle' op can be eliminated by doing this.
//  (2) 'middle' op works on height x middle-cols, full vectors, fully aligned output.
//
//  (3) 'right operation' if needed to handle 1..3 columns at the end (partial width fill).
//
// Since the left & right loops are rather messy, they are both actually handled with the same
// code (there is a 'pseudo-loop' which wraps around to handle this).
//
// - all loads of 'float' values are done with unaligned vector loads.
// - in order to avoid 'out of range vector reads, we adjust the float read pointer
//   so that the last value (of 0..depth-1) goes into the last slot of the vector,
//   and then rotate as needed after conversion. The except is the first vector
//   loaded in the partial row loop; this is done with one or two reads, and a valign
//   (to avoid reading below the initial address).
//
//
//
static void
quant_do_d32_core_op(
		struct autoquant_d32_work *wrkp,
		float const *inptr,		// pointer to the 1st input float
		uint8_t * optr,			// first output (32 aligned)
		int depth,				// depth to process (1..32)
		int d0)			// depth padding on left (dnow+d0 <= 32)
{
	int out_row_stride = wrkp->out_row_stride;
	int height = wrkp->shape.height;
	int width = wrkp->shape.width;
	int in_width_stride = wrkp->shape.depth;
	int in_row_stride = in_width_stride * width;

	int scaling = wrkp->qparms.scaling;
	int min_offset = wrkp->qparms.min_offset;
	int common_exp = wrkp->qparms.common_exp;

	int w_off_bytes = (size_t)optr & (128-32);	// 0,32,64,96
	int w_left_pad = w_off_bytes >> 5;				// 0,1,2,3

	optr -= w_off_bytes;					// align output pointer
	inptr -= (32-depth);					// so that unaligned reads load to *end* of vector
	int n_vror = (32-depth-d0);			// the 'vror' used to align result.

	// break width up into left,middle, right parts...
	// any of these may be empty (but not all!)
	//
	int pw_middle_cols = 0;
	int pw_right_cols = 0;
	int pw_left_cols = 4-w_left_pad;		// 1,2,3,4:  partial-width cols on left
	if( pw_left_cols > width){		// 'left' is the whole thing.
		pw_left_cols = width;
		pw_middle_cols = -1;			// no more after partial left
	}else{
		// eliminate the 'left' pass if it's doing all 4,  and depth = 32 (so no risk of
		// reading 'too low' with unaligned reads)
		if( w_left_pad == 0 && depth == 32){
			pw_left_cols = 0;
		}
		pw_right_cols = (width-pw_left_cols)& 3;	// 0,1,2,3:  partial_width cols at right
		pw_middle_cols = (width-pw_left_cols)& ~3;	// the remainder in the middle.
	}

	//
	// find the col count (1..4) and offset (0..3) for the left partial op
	//
	int pw_cols = pw_left_cols;
	int pw_offs = w_left_pad;

	for(int dummy = 0; dummy < 2; dummy++ ){
		if( pw_cols > 0 ){		// perform partial width operation
			HVX_Vector * wrp = (HVX_Vector *)optr;
			float const * rdp =  inptr;
			int pw_vror = n_vror - 32*pw_offs;		// adjust for width offset

			HVX_Vector vin0,vin1,vin2,vin3;
			HVX_Vector words0, words1, words2, words3;
			HVX_Vector halves01, halves23,result;

			// for reading vin0, we want to do
			//   vin0 = q6op_V_vldu_A((HVX_Vector const*) rdp)
			// but there's a risk of reading from data we aren't allowed to read, at too
			// low an address, when depth is < 32). The first element we really need
			// is at rdp[32-depth]
			// and the last is at  rdp[31]
			// So the method is to read both of those with aligned reads; if the desired
			// data doesn't span a vector boundary, both will read the same vector.
			// we then combine them with a valign.

			// we have one loop for pw_cols = 1 or 2, and
			// another for pw_cols = 3 or 4.

			int v0_offs = 32-depth;


			if( pw_cols <=2){			// 1 & 2 cases
				for( int i = 0; i < height ; i++){
					vin0 = Q6_V_valign_VVR( *(HVX_Vector const*)(rdp+31), *(HVX_Vector const*)(rdp+v0_offs),
							(size_t)rdp );
					vin1 = Q6_V_vzero();
					if( pw_cols > 1)
						vin1 = q6op_V_vldu_A((HVX_Vector const*) (rdp + in_width_stride));
					words0 = requantize_words(vin0,scaling,min_offset,common_exp);
					words1 = requantize_words(vin1,scaling,min_offset,common_exp);
					halves01 = Q6_Vh_vpacko_VwVw(words1,words0);
					result = Q6_Vb_vpacko_VhVh(halves01,halves01);
					*wrp = Q6_V_vror_VR( result, pw_vror);
					wrp = (HVX_Vector *)( (uint8_t*) wrp + out_row_stride );
					rdp += in_row_stride;
				}
			}else{			// 3 & 4 cases
				for( int i = 0; i < height ; i++){
					vin0 = Q6_V_valign_VVR( *(HVX_Vector const*)(rdp+31), *(HVX_Vector const*)(rdp+v0_offs),
							(size_t)rdp );
					vin3 = Q6_V_vzero();
					if( pw_cols >3){
						vin3 = q6op_V_vldu_A((HVX_Vector const*) (rdp + 3*in_width_stride));
					}
					vin1 = q6op_V_vldu_A((HVX_Vector const*) (rdp + in_width_stride));
					vin2 = q6op_V_vldu_A((HVX_Vector const*) (rdp + 2*in_width_stride));
					words2 = requantize_words(vin2,scaling,min_offset,common_exp);
					words3 = requantize_words(vin3,scaling,min_offset,common_exp);
					halves23 = Q6_Vh_vpacko_VwVw(words3,words2);
					words0 = requantize_words(vin0,scaling,min_offset,common_exp);
					words1 = requantize_words(vin1,scaling,min_offset,common_exp);
					halves01 = Q6_Vh_vpacko_VwVw(words1,words0);
					result = Q6_Vb_vpacko_VhVh(halves23,halves01);
					*wrp = Q6_V_vror_VR( result, pw_vror);
					wrp = (HVX_Vector *)( (uint8_t*) wrp + out_row_stride );
					rdp += in_row_stride;
				}
			}
			// at end of partial width op, advance the pointers.
			optr += 4*32;			// move to next out vector
			inptr += pw_cols*in_width_stride;
		}


		if( pw_middle_cols < 0)
			break;		// done! no middle or right.
		if( pw_middle_cols > 0){	// do 'middle' op
			// pointers for start of row
			int nvec_middle_m1 = (pw_middle_cols>>2)-1;		// number of 'vector ops' in middle loop,-1.
			HVX_Vector *wrp0 = (HVX_Vector *)optr;
			float const *rdp0 = inptr;
			for( int i = 0; i < height ; i++){
				HVX_Vector * __restrict wrp = wrp0;
				float const * __restrict  rdp = rdp0;
				HVX_Vector vin0,vin1,vin2,vin3;
				HVX_Vector words0, words1, words2, words3;
				HVX_Vector halves01, halves23,result;

				vin0 = q6op_V_vldu_A((HVX_Vector const*) rdp);
				vin1 = q6op_V_vldu_A((HVX_Vector const*) (rdp + in_width_stride));
				vin2 = q6op_V_vldu_A((HVX_Vector const*) (rdp + 2*in_width_stride));
				vin3 = q6op_V_vldu_A((HVX_Vector const*) (rdp + 3*in_width_stride));
				words0 = requantize_words(vin0,scaling,min_offset,common_exp);
				words1 = requantize_words(vin1,scaling,min_offset,common_exp);

				for(int j = 0 ; j < nvec_middle_m1 ; j++){
					words2 = requantize_words(vin2,scaling,min_offset,common_exp);
					words3 = requantize_words(vin3,scaling,min_offset,common_exp);
					halves01 = Q6_Vh_vpacko_VwVw(words1,words0);
					halves23 = Q6_Vh_vpacko_VwVw(words3,words2);
					result = Q6_Vb_vpacko_VhVh(halves23,halves01);
					rdp += 4*in_width_stride;

					vin0 = q6op_V_vldu_A((HVX_Vector const*) rdp);
					vin1 = q6op_V_vldu_A((HVX_Vector const*) (rdp + in_width_stride));
					vin2 = q6op_V_vldu_A((HVX_Vector const*) (rdp + 2*in_width_stride));
					vin3 = q6op_V_vldu_A((HVX_Vector const*) (rdp + 3*in_width_stride));
					words0 = requantize_words(vin0,scaling,min_offset,common_exp);
					words1 = requantize_words(vin1,scaling,min_offset,common_exp);

					*wrp = Q6_V_vror_VR( result, n_vror);
					wrp ++;

				}
				words2 = requantize_words(vin2,scaling,min_offset,common_exp);
				words3 = requantize_words(vin3,scaling,min_offset,common_exp);
				halves01 = Q6_Vh_vpacko_VwVw(words1,words0);
				halves23 = Q6_Vh_vpacko_VwVw(words3,words2);
				result = Q6_Vb_vpacko_VhVh(halves23,halves01);
				*wrp = Q6_V_vror_VR( result, n_vror);


				// bump pointers down to next row.
				wrp0 = (HVX_Vector *)( (uint8_t*) wrp0 + out_row_stride );
				rdp0 += in_row_stride;
			}


			// advance pointers over the middle op
			optr += pw_middle_cols*32;
			inptr += pw_middle_cols*in_width_stride;
		} // if middle
		// done left and middle .. set up for right
		if( pw_right_cols <= 0)
			break;
		pw_cols = pw_right_cols;
		pw_offs = 0;
		pw_middle_cols = -1;		// so we don't do the middle again.
	}
}

///////////////////////////////////////////////////////////



static int dequantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *min_tensor = self->inputs[1];
	const struct tensor *max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	float minval = tensor_get_float(min_tensor,0);
	float maxval = tensor_get_float(max_tensor,0);
	float range = fmaxf(0.0001f,maxval-minval);
	float stepsize = flt_div_255(range);
	float batches = in_tensor->shape.batches;
	float height = in_tensor->shape.height;
	float width = in_tensor->shape.width;
	float depth = in_tensor->shape.depth;
	const uint8_t *in = in_tensor->data;
	float *out = out_tensor->data;
	int out_bytes = batches*height*width*depth*sizeof(float);
	int i;
	logmsg(nn,2,"dequantize execute. self=%p ",self);
	if( tensor_out_prepare_normal(out_tensor,batches,height,width,depth, NN_TYPE_FLOAT)!= 0 ){
		return errlog(nn,"out too small for node %d %d < %d",self->node_id,out_tensor->max_size,out_bytes);
	}
	for (i = 0; i < batches*height*width*depth; i++) {
		out[i] = (in[i] * stepsize) + minval;
	}
	return 0;
}

static int dequantize_i32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *min_tensor = self->inputs[1];
	const struct tensor *max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	float minval = tensor_get_float(min_tensor,0);
	float maxval = tensor_get_float(max_tensor,0);
	float range = fmaxf(0.0001f,maxval-minval);
	float stepsize = range/4294967296.0f/*0x1.0p32f*/;
	float batches = in_tensor->shape.batches;
	float height = in_tensor->shape.height;
	float width = in_tensor->shape.width;
	float depth = in_tensor->shape.depth;
	const int32_t *in = in_tensor->data;
	float *out = out_tensor->data;
	int out_bytes = batches*height*width*depth*sizeof(float);
	int i;
	logmsg(nn,2,"dequantize 32 execute.");
	if( tensor_out_prepare_normal(out_tensor,batches,height,width,depth, NN_TYPE_FLOAT)!= 0 ){
		return errlog(nn,"out too small for node %d %d < %d",self->node_id,out_tensor->max_size,out_bytes);
	}
	for (i = 0; i < batches*height*width*depth; i++) {
		out[i] = (in[i] * stepsize);
	}
	return 0;
}


static int quantize_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking quantize node %p",self);
	int k = node_check_inputs_outputs_n( self, nn, "quantize", 3, 3 );
	if(k!=0) return k;
	logmsg(nn,2,"quantize node %p OK",self);
	return 0;
}


static int autoquantize_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking quantize node %p",self);
	int k = node_check_inputs_outputs_n( self, nn, "autoquant", 1, 3 );
	if(k!=0) return k;
	logmsg(nn,2,"quantize node %p OK",self);
	return 0;
}

//
// autoquantize_d32 has 1..5 inputs; 4 are optional
//
static int autoquantize_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking autoquant_d32 node %p",self);
	int k = node_check_inputs_range( self, nn, "autoquant_d32", 1, -5 );
	if(k==0) k = node_check_outputs_n( self, nn, "autoquant_d32", 3);
	if(k!=0) return k;
	logmsg(nn,2,"autoquant_d32 node %p OK",self);
	return 0;

}

static int dequantize_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking dequantize node %p",self);
	int k = node_check_inputs_outputs_n( self, nn, "dequantize", 3, 1 );
	if(k!=0) return k;
	logmsg(nn,2,"dequantize node %p OK",self);
	return 0;
}

//
// Quantize:
// convert float to 8-bit quantized, using the supplied
// min and max output range.
// The output range may be expanded slightly to produce a
// a properly aligned zero code.

//  input 0:   float tensor
//  input 1,2:  scalar float, given min & max
//
//  output 0:   qu8 tensor
//  output 1,2:  scalar float, output min & max


struct nn_node_ops nn_ops_for_Quantize = {
	.execute = quantize_execute,
	.check = quantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Quantize_ref = {
	.execute = quantize_execute,
	.check = quantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};


// AutoQuantize:
// convert float to 8-bit quantized, using the actual
// min and max values.

//  input 0:   float tensor
//
//  output 0:   qu8 tensor
//  output 1,2:  scalar float, output min & max

struct nn_node_ops nn_ops_for_AutoQuantize = {
	.execute = autoquantize_execute_opt,
	.check = autoquantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_AutoQuantize_ref = {
	.execute = autoquantize_execute,
	.check = autoquantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
// AutoQuantize for d32
// convert float to 8-bit quantized, using the actual
// min and max values found in the code.

//  input 0:   float tensor
//  input 1:  (optional) scalar int: depth padding start - default 0  (0..31)
//  input 2:  (optional) scalar int: width padding start - default 4  (0..MAX_PADDING_WIDTH)
//  input 3:  (optional) scalar int: width padding end (min) - default 0  (0..MAX_PADDING_WIDTH)
//    The 'end' padding will be adjusted up so that the width total is a multiple of 4. If it exceeds
//    MAX_PADDING_WIDTH as a result, it will then be adjusted down by 4.
//  input 4:  (optional): scalar int: top/bottom padding for height  default 4 (0..MAX_PADDING_HEIGHT)
//
//
//  output 0:   qu8 tensor (d32 format)
//  output 1,2:  scalar float, output min & max
// Note: AutoQuantize_d32_ref is in ops_quantfortest_d32.c
//
struct nn_node_ops nn_ops_for_AutoQuantize_d32 = {
	.execute = autoquantize_d32_execute_opt,
	.check = autoquantize_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};


// Dequantize
//  Convert quantized to float
//  input 0:   qu8 tensor
//  input 1,2:  scalar float, input min & max
//
//  output 0:   float tensor

struct nn_node_ops nn_ops_for_Dequantize = {
	.execute = dequantize_execute,
	.check = dequantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};


struct nn_node_ops nn_ops_for_Dequantize_ref = {
	.execute = dequantize_execute,
	.check = dequantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

// Dequantize_qint32_f
//  Convert quantized to float
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//
//  output 0:   float tensor

struct nn_node_ops nn_ops_for_Dequantize_qint32_f = {
	.execute = dequantize_i32_execute,
	.check = dequantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Quantize_int32 = {
	.execute = quantize32_execute,
	.check = quantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
struct nn_node_ops nn_ops_for_Quantize_int32_ref = {
	.execute = quantize32_execute,
	.check = quantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};



struct tdata_copytensor {
	nn_sem_t        *donesem;
	const struct tensor *in_tensor;
	const struct tensor *in_min_tensor;
	const struct tensor *in_max_tensor;
	struct tensor *out_tensor;
	struct tensor *out_min_tensor;
	struct tensor *out_max_tensor;
};

static int autoq_execute_copy_tensor_work(struct nn_graph *nn, void *info)
{
	struct tdata_copytensor *work = info;

	const struct tensor *in_tensor = work->in_tensor;
	const struct tensor *in_min_tensor = work->in_min_tensor;
	const struct tensor *in_max_tensor = work->in_max_tensor;
	struct tensor *out_tensor = work->out_tensor;
	struct tensor *out_min_tensor = work->out_min_tensor;
	struct tensor *out_max_tensor = work->out_max_tensor;

	vmemcpy_asm(out_tensor->data,in_tensor->data,in_tensor->max_size);
	vmemcpy_asm(out_min_tensor->data,in_min_tensor->data,in_min_tensor->max_size);
	vmemcpy_asm(out_max_tensor->data,in_max_tensor->data,in_max_tensor->max_size);
    return 0;
}

static void input_copy_work(
	struct nn_graph *nn,
	const struct tensor *in_tensor,
	const struct tensor *in_min_tensor,
	const struct tensor *in_max_tensor,
	struct tensor *out_tensor,
	struct tensor *out_min_tensor,
	struct tensor *out_max_tensor)
{
	struct tdata_copytensor work;
	work.in_tensor = in_tensor;
	work.in_min_tensor = in_min_tensor;
	work.in_max_tensor = in_max_tensor;
	work.out_tensor = out_tensor;
	work.out_min_tensor = out_min_tensor;
	work.out_max_tensor = out_max_tensor;
	nn_os_vector_call(nn, autoq_execute_copy_tensor_work, &work);
    return;
}

#define NUM_METADATA_PER_INPUT_TENSOR 4
#define NUM_METADATA_PER_QUANTIZED_TENSOR 3
static int input_quantize_execute_opt(struct nn_node *self, struct nn_graph *nn){
	if (nn->n_inputs % NUM_METADATA_PER_INPUT_TENSOR) {
		return errlog(nn,"oops, input # does not have enough metadata %d",nn->n_inputs);
	}

	int num_data_tensors_input = nn->n_inputs / NUM_METADATA_PER_INPUT_TENSOR;
	int num_data_tensors_output = self->n_outputs / NUM_METADATA_PER_QUANTIZED_TENSOR;
	if (num_data_tensors_input != num_data_tensors_output) {
		return errlog(nn,"oops, number of inputs to network %d(%d) does not equal number of outputs from input op %d(%d) ",
			num_data_tensors_input,nn->n_inputs,num_data_tensors_output,self->n_outputs);
	}

	int i,batches,height,width,depth;
	struct tensor *out_tensor,*out_min_tensor,*out_max_tensor;
	const struct tensor *in_tensor, *in_min_tensor, *in_max_tensor;
	const struct tensor *needs_quantization_tensor;

	for(i =0; i < num_data_tensors_input; i++){
		out_tensor = self->outputs[i*NUM_METADATA_PER_QUANTIZED_TENSOR + 0];
		out_min_tensor = self->outputs[i*NUM_METADATA_PER_QUANTIZED_TENSOR + 1];
		out_max_tensor = self->outputs[i*NUM_METADATA_PER_QUANTIZED_TENSOR + 2];
		in_tensor=&nn->inputs[i*NUM_METADATA_PER_INPUT_TENSOR];
		batches = in_tensor->shape.batches;
		height = in_tensor->shape.height;
		width = in_tensor->shape.width;
		depth = in_tensor->shape.depth;
		if( tensor_out_prepare_normal(out_tensor,batches,height,width,depth, NN_TYPE_QUINT8)!=0 ){
			return errlog(nn,"out too small (%p) %d < %d",
								    out_tensor, out_tensor->max_size, batches*height*width*depth);
		}
	}

	for (i =0; i < num_data_tensors_input; i++){
		int input_index  = i * NUM_METADATA_PER_INPUT_TENSOR;
		int output_index = i * NUM_METADATA_PER_QUANTIZED_TENSOR;
		in_tensor = &nn->inputs[input_index];
		out_tensor = self->outputs[output_index];
		out_min_tensor = self->outputs[output_index + 1];
		out_max_tensor = self->outputs[output_index + 2];
		needs_quantization_tensor = &nn->inputs[input_index + 3];

		int needs_quantization =  tensor_get_int32(needs_quantization_tensor,0);

		if (!needs_quantization){
			in_min_tensor = &nn->inputs[input_index + 1];
			in_max_tensor = &nn->inputs[input_index + 2];
			input_copy_work(nn, in_tensor, in_min_tensor, in_max_tensor, out_tensor, out_min_tensor, out_max_tensor);

		} else {
			logmsg(nn,9,"input tensor data %x scratch %x",(uint32_t)in_tensor->data,(uint32_t)nn->scratch);

			int elements = in_tensor->max_size/sizeof(float);
			autoquantize_work(nn, in_tensor, out_tensor, out_min_tensor, out_max_tensor, elements);
		}
	}
	return 0;
}

static int input_quantize_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	size_t current_size, max_size=0;

	if (self->n_outputs % NUM_METADATA_PER_QUANTIZED_TENSOR) {
		return errlog(nn,"oops, quantize # does not have enough space for metadata (inputs %d outputs %d)",nn->n_inputs,self->n_outputs);
	}
	for (i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) {
			return errlog(nn,"input: fatal: NULL output");
		}
		if (i%NUM_METADATA_PER_QUANTIZED_TENSOR && tensor_out_prepare_normal(self->outputs[i],1,1,1,1,NN_TYPE_FLOAT )!=0){
			return errlog(nn,"output tensor %d is not a single float",i);
		}
		if ((i%NUM_METADATA_PER_QUANTIZED_TENSOR)==0){
			current_size = self->outputs[i]->max_size;
			max_size = (current_size > max_size) ? current_size : max_size;
		}
	}
		logmsg(nn,2,"input quantize node %p OK",self);
	return 0;
}
//  nn->inputs[4 * i + 0] 0:  input, float tensor or quantized tensor
//  nn->inputs[4 * i + (1,2)] :  scalar float, input min & max if quantized tensor
//  nn->inputs[4 * i + 3]:  int 1 if it needs quantization, 0 otherwise
//  self->outputs[3 * 0]:   qu8 tensor (d32 format)
//  self->outputs[3 * i + (1,2)]:  scalar float, output min & max
//
struct nn_node_ops nn_ops_for_QuantizeINPUT_f_to_8 = {
	.execute = input_quantize_execute_opt,
	.check = input_quantize_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

