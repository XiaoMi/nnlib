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
 * This contains implementations for quantized Prelu node
 */
#include <nn_graph.h>
#include <quantize.h>
#include <string.h>
#if defined(USE_OS_LINUX)
#include <malloc.h>
#endif


#define NEW_PRELU_D32

#ifdef NEW_PRELU_D32		// new implementation
#include "hvx_inlines.h"
#define MAX_THREADS 2

struct nodeinfo {
	int16_t *alphabuf;
	int alpha_depth;
	int shift;				// 0..7
	float alpha_mult;		// 2**-shift, used to quantize alphas to -128..127
	float alpha_mag;		// range as a power of 2 (= 2**shift)
	float alpha_min, alpha_max;	// the actual range
};

//
// new data path for prelu:
// (1) read u8, and subtract input offset, to obtain a value in range -255..255;
//     shift it left 7 bits to get full range of i16
// (2) pick a scale factor and perform fractional mul. If the result from (1) is >=0,
//     a common scale factor corresponding to '1.0' is used; if <0, another value (which varies according to
//     depth) is used.
// (3)  add an offset, and >>n saturating to u8.
//
// The 'common' scale factor in (2), combined with the shift in (3), do unity scaling and thus
//   adapt to the input vs. output ranges.
// The 'negative' scale factor in (2) are variable per-depth; these scales need to be cooked from
//  the alpha inputs and from the in:out ratio (i.e. each one of these is the 'common' ratio multiplied
//   by the particular alpha; and 'shift' is reduced as needed to keep all of those value in the i16 range.
//
// Initally, we scan the alpha floats from the input to find their range, and store them as i16 fractions.
// There is a 'alphashift' which is used to normalize these (alphashift = 0 means all have abs val <1.0).
// So, the 'cooking' process for adapting the depth scales to the range, consists of doing a full multiply
// of the unscaled alpha with the common factor, and then >>alphashift; if the scaling has been done correctly
// then (ipso facto) the results will fit in i16. In fact, since the common scale factor can be small, it's better to
// use a version of it with n more fractional bits and then >> the product by n+alphashift.
// The pre-cooked alpha (before and after scaling) are stored in interleaved order for convenient vector use,
// with each value stored 4 times:
//   0,2,4,..28,30, 0,2,4,..28,30, 0,2,4,..28,30,  0,2,4,..28,30  <- first vector
//   1,3,5 ..29,31, 1,3,5 ..29,31, 1,3,5 .. 29,31, 1,3,5 ..29,31, <- second vector
//  then each group of 32 stored in 2 vectors in the same way.
//
struct scale_parms {
	int16_t in_qzero;
	int16_t common_scale;		// common scale for vals >= 0
	int16_t out_offset;			// amount to add in step (3)
	int16_t final_shift;
	// parms for scaling the alphas
	int16_t alpha_prescale;		// mul by this,
	int16_t alpha_pre_shift;	// and >> this.
};
static int
set_scaling( struct nn_graph * nn, struct scale_parms *sp, struct nodeinfo const *info,
		float in_min, float in_max,
		float out_min, float out_max)
{
	float in_range = in_max-in_min;
	float out_range = out_max-out_min;
	int in_qzero = saturate_u8( roundf_i32(-255.0f*in_min/in_range));
	int out_qzero = saturate_u8( roundf_i32(-255.0f*out_min/out_range));

	sp->in_qzero = in_qzero;
	// find the 'through' gain.
	float pos_gain = in_range/out_range;
	// find the max.abs gain
	float max_alpha_mag = fmaxf(info->alpha_max, -info->alpha_min );
	float max_gain = pos_gain * fmaxf(1.0f, max_alpha_mag);
	// so, max_gain is the biggest in->out gain we need. It will
	// normally be > 1 and not too large; you can construct scenarios where it's
	// really large, but the ranging should be set up to avoid these
	// (e.g by padding the output range).
	int gain_expo = max_i32(0,flt_getexp(max_gain));		// floor(log2(max_gain))+1
	int final_rsh = 7-gain_expo;
	if(final_rsh <0) return errlog(nn,"infeasible scaling");
	// find the scaled through-gain
	sp->common_scale = saturate_i16( roundf_i32(flt_ldexp( pos_gain, 15-gain_expo)));
	sp->final_shift = final_rsh;
	sp->out_offset= out_qzero << final_rsh;

	// work out how to prescale the alphas. This is done
	// as (alpha_prescale* alpha) >> alpha_pre_shift
	// alpha_prescale is the same as common_scale, but maybe with more fractional bits.
	// alpha_pre_shift takes into account this extra gain...
	//  - alpha values have 15-alpha_shift frac bits
	//  - alpha_prescale has 15-g0exp frac bits
	//  - => prod has 30-alpha_shift-g0exp
	//   we want 15-gain_expo so >> by (30-alpha_shift-g0exp)-(15_gain_expo)
	//    = 15 + (gain_expo-g0_exp)- alpha_shift
	// if we've done everything right, none of these will overflow when scaled
	// to 15-gain_expo fractional bits.
	//
	int g0exp = flt_getexp(pos_gain);	// this is <= gain_expo...
	int extra_shift= gain_expo-g0exp;	// this is>=0
	int ashift = 15 + extra_shift - info->shift;
	sp->alpha_pre_shift = ashift;
	sp->alpha_prescale = saturate_i16( roundf_i32(flt_ldexp( pos_gain, 15-g0exp)));

	return 0;
}


static inline HVX_Vector
hvx_process_relu( HVX_Vector vin, HVX_Vector valpha0, HVX_Vector valpha1,
		int in_qzero,
		int common_gain,
		int outoff,
		int final_shift
		)
{
	HVX_Vector vinqz = q6op_Vh_vsplat_R(in_qzero*128);
	HVX_Vector vcommon = q6op_Vh_vsplat_R(common_gain);
	HVX_Vector voutoff = q6op_Vh_vsplat_R(outoff);
	// widen to h while <<7
	HVX_VectorPair in_x128 = Q6_Wuh_vmpy_VubRub( vin, 0x80808080);
	HVX_Vector inx128_0 = Q6_V_lo_W(in_x128);
	HVX_Vector inx128_1 = Q6_V_hi_W(in_x128);

	// in each path
	// compare to 128*in_qzero to see if the result is going to be
	// negative or positive
	HVX_VectorPred qneg0 = Q6_Q_vcmp_gt_VuhVuh( vinqz, inx128_0);
	HVX_Vector x_0 = Q6_Vh_vsub_VhVh( inx128_0, vinqz);
	HVX_Vector gain_0 = Q6_V_vmux_QVV( qneg0, valpha0, vcommon);
	// frac mul...
	HVX_Vector prod_0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat( x_0, gain_0 );
	// add outoffs
	prod_0 = Q6_Vh_vadd_VhVh_sat( prod_0, voutoff);

	HVX_VectorPred qneg1 = Q6_Q_vcmp_gt_VuhVuh( vinqz, inx128_1);
	HVX_Vector x_1 = Q6_Vh_vsub_VhVh( inx128_1, vinqz);
	HVX_Vector gain_1 = Q6_V_vmux_QVV( qneg1, valpha1, vcommon);
	HVX_Vector prod_1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat( x_1, gain_1 );
	prod_1 = Q6_Vh_vadd_VhVh_sat( prod_1, voutoff);

	return Q6_Vub_vasr_VhVhR_rnd_sat(prod_1, prod_0, final_shift);
}

// this is used to scale the alpha values according to the
// current input & output ranges.
static void
scale_alphas( struct scale_parms const *sp,
		int16_t const *inptr,
		int16_t * optr,
		int nvec )		// total # vecs (always even)
{
	HVX_Vector const * vinp = (HVX_Vector const*)inptr;
	HVX_Vector  * voutp = (HVX_Vector *)optr;
	int rscale = Q6_R_combine_RlRl( sp->alpha_prescale, sp->alpha_prescale);
	int rsh = sp->alpha_pre_shift;

	for(int i= 0; i < nvec; i+=2){
		HVX_VectorPair prod0 = Q6_Ww_vmpy_VhRh( vinp[i], rscale);
		HVX_VectorPair prod1 = Q6_Ww_vmpy_VhRh( vinp[i+1], rscale);
		HVX_Vector result0 = Q6_Vh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(prod0),Q6_V_lo_W(prod0),rsh);
		HVX_Vector result1 = Q6_Vh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(prod1),Q6_V_lo_W(prod1),rsh);
		voutp[i] = result0;
		voutp[i+1] = result1;
	}
}


struct tdata {
	struct nn_node *self;
	struct shape opshape;
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	struct scale_parms scaling;
	// main code launches one thread, and waits for init_done;
	// it then launches additional threads.
	// whichever thread gets work unit 0 does the alpha conversion and
	// posts init_done.
	nn_sem_t init_done;
	int work_units;
	volatile int next_work;
	// each batch is sliced into work units of size h_per_slice
	int h_per_slice;			// # of height units per slice.
	int slice_per_batch;		// # of slices per batch.
	int16_t *alphabuf;
	int16_t *scratch;
	int32_t d32_iters;
	nn_sem_t donesem;
};


static void prelu_hvx_work_func(struct nn_graph *nn, void *vinfo)
{
	// scale the alphas...
	struct tdata *td = vinfo;
	int nd32 = td->tin.nd32;
	int work_unit_index = __sync_fetch_and_add( &td->next_work, 1);
	if( work_unit_index ==  0){	// we are first thread
		// two alpha vectors per d32 slice - scale them.
		scale_alphas( &td->scaling, td->alphabuf, td->scratch,nd32*2);
		nn_sem_post( &td->init_done);	// ok to start other threads now
	}

	HVX_Vector const * alphas = (HVX_Vector const *)td->scratch;
	uint8_t const * inp0 = td->tin.data;
	uint8_t * outp0 = td->tout.data;
	int batches = td->opshape.batches;
	int height = td->opshape.height;
	int widvecs = td->d32_iters;

	uint32_t in_d32_stride = td->tin.d32_stride;

	uint32_t in_batch_stride = td->tin.batch_stride;
	uint32_t in_height_stride = td->tin.height_stride;
	uint32_t out_batch_stride = td->tout.batch_stride;
	uint32_t out_height_stride = td->tout.height_stride;
	uint32_t out_d32_stride = td->tout.d32_stride;

	int in_qzero = td->scaling.in_qzero;
	int common_gain= td->scaling.common_scale;
	int outoff = td->scaling.out_offset;
	int final_shift = td->scaling.final_shift;
	int slice_per_batch = td->slice_per_batch;
	int h_per_slice = td->h_per_slice;

	int b = 0;
	int batch_workindex = 0;		// always = b*slice_per_batch
	int work_units = td->work_units;
	// work unit loop.
	while( work_unit_index < work_units){
		// figure out what batch and height range we are at.
		while( (work_unit_index-batch_workindex) >= slice_per_batch ){
			// need to be in the next batch
			b++;
			if( b >= batches) goto done;
			batch_workindex += slice_per_batch;
		}
		int h_base = (work_unit_index-batch_workindex)*h_per_slice;
		int h_count = min_i32( height-h_base, h_per_slice);
		uint8_t const * inp_b = inp0 + b*in_batch_stride + h_base*in_height_stride;
		l2fetch(inp_b, in_d32_stride, 128*widvecs, h_count*nd32);
		uint8_t  * outp_b = outp0 + b*out_batch_stride + h_base*out_height_stride;

		for( int h = 0; h < h_count; h++){
			uint8_t const * inp = inp_b + h*in_height_stride;
			uint8_t * outp = outp_b +  h*out_height_stride;
			HVX_Vector const * alpha_p = alphas;
			for(int id32 = 0; id32 < nd32; id32++){
				HVX_Vector alph0 = *alpha_p++;
				HVX_Vector alph1 = *alpha_p++;
				HVX_Vector const *vinp = (HVX_Vector const*)inp;
				HVX_Vector *voutp = (HVX_Vector *)outp;
				for( int i = 0; i < widvecs; i++){
					voutp[i]= hvx_process_relu( vinp[i],alph0,alph1, in_qzero, common_gain, outoff, final_shift);
				}
				inp += in_d32_stride;
				outp += out_d32_stride;
			}
		}
		// get next work unit...
		work_unit_index = __sync_fetch_and_add( &td->next_work, 1);
	}
  done:
	nn_sem_post( &td->donesem);
}

static int prelu_opt_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	struct nodeinfo *nodeinfo = self->opaque;


	int b = in_tensor->shape.batches;
	int h = in_tensor->shape.height;
	int h_pad_before = in_tensor->format.height_pad[0];
	int h_pad_after = in_tensor->format.height_pad[1];
	int w = in_tensor->shape.width;
	int w_pad_before = in_tensor->format.width_pad[0];
	int w_pad_after = in_tensor->format.width_pad[1];
	int d = in_tensor->shape.depth;
	int d_pad_before = in_tensor->format.depth_pad[0];
	int d_pad_after = in_tensor->format.depth_pad[1];
	//int h_total = h+h_pad_before+h_pad_after;
	int w_total = w+w_pad_before+w_pad_after;
	int d_total = d+d_pad_before+d_pad_after;

	if( d != nodeinfo->alpha_depth ) return errlog(nn,"depth mismatch : alphas=%d, input = %d", nodeinfo->alpha_depth, d);
	// find output range.
	// out_max must allow for most negative input * most negative alpha
	// and for most positive input.
	//
	float out_max = fmaxf(in_max,in_min*nodeinfo->alpha_min);
	// out_min just allows for most -ve input and most +ve alpha.
	float out_min = in_min * nodeinfo->alpha_max;

	// make sure there is a clean zero.
	adjust_minmax_for_zero( &out_min, &out_max);


	struct tdata td;
	if(set_scaling(nn,&td.scaling, nodeinfo, in_min,in_max, out_min, out_max )){
		return -1;
	}
	logmsg(nn,2,"d: %d<%d>%d ==> %d\n",d_pad_before,d,d_pad_after,d_total);
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		b,
		h,h_pad_before,h_pad_after,
		w,w_pad_before,w_pad_after,
		d,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out prepare fail");
	}

	logmsg(nn,2,"Prelu execute. self=%p ",self);

	int wskip = w_pad_before & ~3;	// skip 4 if wpad = 4

	td.self = self;
	td.opshape = in_tensor->shape;
	td.tin = tensor_addressing_d32( in_tensor);
	td.tout = tensor_addressing_d32( out_tensor);
	td.tin.data -= -32*(w_pad_before-wskip),
	td.tout.data -= -32*(w_pad_before-wskip),
	td.d32_iters = (w_total-wskip) / 4u,
	td.alphabuf = nodeinfo->alphabuf;
	td.scratch = nn->scratch;
	td.next_work = 0;

	nn_sem_init(&td.donesem,0);
	nn_sem_init(&td.init_done,0);

	// work out how many parts to slice each batch into vertically.
	//
	unsigned row_work = td.d32_iters * td.tin.nd32;	// vectors per row
	// this is the chunk size on the h dimension
	int hchunk = (row_work >=1024) ? 1: (1024 >> floor_log2( row_work));
	if(hchunk>= h){
		if( b== 1){
			hchunk = (h+1)>>1;
		}else{
			hchunk = h;
		}
	}
	unsigned slice_per_batch = 1;
	if( h  > hchunk){
		slice_per_batch = (h+(hchunk-1))/(unsigned)hchunk;		// >=2
		// redistribute ?
		if( slice_per_batch < 8)
			hchunk = (h + (slice_per_batch-1))/slice_per_batch;
	}
	td.h_per_slice = hchunk;
	td.slice_per_batch = slice_per_batch;
	td.work_units = slice_per_batch * b;

	int nthreads = min_i32(td.work_units, MAX_THREADS);

	//printf("%d threads; %d work units; h=%d divided into %d parts of %d\n",
	//		nthreads, td.work_units, h, slice_per_batch, hchunk);

	// prefetch the alpha table
	l2fetch( td.alphabuf, 128,128,2*td.tin.nd32);
	//
	// first thread prepares the alpha, and posts init_done, so we don't start
	// any more until that is seen.
	for( int i =0; i < nthreads; i++){
		if( i==1)  nn_sem_wait(&td.init_done);
		nn_os_work_for_vector(nn,prelu_hvx_work_func,&td);
	}

	tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT);
	//tensor_set_float(out_min_tensor,0,nodeinfo->alpha_mag*tensor_get_float(in_min_tensor,0));
	//tensor_set_float(out_max_tensor,0,nodeinfo->alpha_mag*tensor_get_float(in_max_tensor,0));
	tensor_set_float(out_min_tensor,0,out_min);
	tensor_set_float(out_max_tensor,0,out_max);

	nn_sem_wait_n_times(&td.donesem,nthreads);
	logmsg(nn,2,"Prelu %p done",self);
	return 0;
}

static int prelu_check_opt(struct nn_node *self, struct nn_graph *nn)
{
	//if (self->n_inputs != 4) return errlog(nn,"maxpool wrong # inputs");
	//if (self->n_outputs != 3) return errlog(nn,"maxpool wrong # outs");
	const struct tensor *in_alpha_tensor = self->inputs[3];
	int32_t alpha_depth = in_alpha_tensor->shape.depth;
	int32_t alpha_depth_roundup = (alpha_depth + 31) & ~31;
	const float *alphas = in_alpha_tensor->data;
	struct nodeinfo *nodeinfo = self->opaque;
	int16_t *alpha_frac_buf = NULL;

	int i,j,w;
	float maxmag;
	float magrecip;
	int shift;
	if (nodeinfo == NULL) {
		if ((nodeinfo = nn_malloc(sizeof(*nodeinfo))) == NULL) {
			return errlog(nn,"can't alloc nodeinfo");
		}
		if ((alpha_frac_buf = nn_memalign(128,alpha_depth_roundup*4*2)) == NULL) {
			nn_free(nodeinfo);
			return errlog(nn,"can't allocate alpha buf");
		}
		nodeinfo->alphabuf = alpha_frac_buf;
		self->opaque = nodeinfo;
	} else {
		alpha_frac_buf = nodeinfo->alphabuf;
	}
	if( nn_scratch_grow(nn,alpha_depth_roundup*4*2)){
		return errlog(nn,"scratch alloc");
	}
	/*
	 * Find the maximum magnitude of the alphas 
	 * We will use this to scale the alpha values down and the output min/max up
	 * We also need to calculate the shift down of the positive input values
	 * All this is typically unnecessary, alpha input is likely small.
	 * Note, the hvx code requires shift <=7 since it does a right-shift by 7-shift.
	 */
	if( find_range_of_floats( alphas, alpha_depth,&nodeinfo->alpha_min, &nodeinfo->alpha_max)!= 0){
		return errlog(nn,"prelu: inf or nan in alphas");
	}
	float alpha_maxmag = fmaxf(-nodeinfo->alpha_min, nodeinfo->alpha_max);	// mag of range
	shift = 0;
	magrecip = maxmag = 1.0f;
	if( alpha_maxmag > 1.0f){
		shift = flt_getexp(alpha_maxmag);	// floor(log2(x))+1
		maxmag = flt_power2(shift);
		magrecip = flt_power2(-shift);
		if( shift >7 ) return errlog(nn,"magnitude of alphas is too large");
	}

	logmsg(nn,3,"alpha range= %f ... %f maxmag=%f shift=%d",nodeinfo->alpha_min,nodeinfo->alpha_max, maxmag,shift);

	nodeinfo->alpha_depth = alpha_depth;
	nodeinfo->alpha_mult = magrecip;
	nodeinfo->alpha_mag = maxmag;
	nodeinfo->shift = shift;

	/*
	 * Create our buffer of alpha values, scaled with 15-shift fractional bits
	 * so that they all are all in -32678 .. 32767.
	 * Ordering of first 32 16 bit values is as follows (each repeated 4 times):
	 *  0,2,4,6 ..  30, 0,2,4,6 ..30, 0,2,4,6 ..  30, 0,2,4,6 ..30  (one vector)
	 *  1,3,5,7 ..  31, 1,3,5,7 ..31, 1,3,5,7 ..  31, 1,3,5,7 ..31  (second vector)
	 *  Each group of 32 is in 2 vectors following the same order.
	 */
	float alpha0 = alphas[0];
	for (i = 0; i < alpha_depth_roundup; i += 32) {
		for (j = 0; j < 16; j++) {
			int ij = i+2*j;
			float alphaval_0 = alpha0,alphaval_1 = alpha0;
			if (ij < alpha_depth) alphaval_0 = alphas[ij];
			if (ij+1 < alpha_depth) alphaval_1 = alphas[ij+1];
			int alphafxp_0 = roundf_i32( 32768.0f * magrecip * alphaval_0);
			int alphafxp_1 = roundf_i32( 32768.0f * magrecip * alphaval_1);
			alphafxp_0 = min_i32( max_i32( alphafxp_0,-32768),32767);
			alphafxp_1 = min_i32( max_i32( alphafxp_1,-32768),32767);

			int16_t * dstp = & alpha_frac_buf[4*i + j];
			logmsg(nn,4,"alphas[%d+%d] = %f --> %x",i,2*j,alphaval_0,alphafxp_0&0xFFFF);
			logmsg(nn,4,"alphas[%d+%d] = %f --> %x",i,2*j+1,alphaval_1,alphafxp_1&0xFFFF);
			for( w = 0; w < 4; w++){
				dstp[w*16] = alphafxp_0;		// 4 slots in first vector
				dstp[64+w*16] = alphafxp_1;		// 4 slots in second vector.
			}
		}
	}
	return 0;
}
#else // previous implementation
struct tdata {
	struct nn_node *self;
	const uint8_t *indata;
	uint8_t *outdata;
	uint8_t *alphabuf;
	int32_t in_next_row;
	int32_t in_next_d32;
	int32_t out_next_row;
	int32_t out_next_d32;
	int32_t in_qzero;
	int32_t out_qzero;
	int32_t d32_iters;
	int32_t d_iters;
	int32_t h_iters;
	int32_t shift;
	uint32_t shrink;
	nn_sem_t donesem;
};

struct nodeinfo {
	uint8_t *alphabuf;
	int shift;				// 0..7
	float alpha_mult;		// 2**-shift, used to quantize alphas to -128..127
	float alpha_mag;		// range as a power of 2 (= 2**shift)
	float alpha_min, alpha_max;	// the actual range
};

void prelu_hvx_d32(
	uint8_t *outdata,
	const uint8_t *indata,
	int32_t in_next_row,
	int32_t in_next_d32,
	const uint8_t *alphabuf,
	int32_t in_qzero,
	int32_t d32iters,
	int32_t d_iters,
	int32_t h_iters,
	uint32_t shrink,
	int32_t out_qzero,
	int32_t alpha_shift);

static void prelu_execute_slice(struct nn_graph *nn, void *vinfo)
{
	struct tdata *td = vinfo;
	prelu_hvx_d32(
		td->outdata,
		td->indata,
		td->in_next_row,
		td->in_next_d32,
		td->alphabuf,
		td->in_qzero,
		td->d32_iters,
		td->d_iters,
		td->h_iters,
		td->shrink,
		td->out_qzero,
		td->shift);
	nn_sem_post(&td->donesem);
}




static int prelu_opt_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	uint32_t i;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	uint8_t in_qzero = quantize_uint8(0.0f,in_min,in_max);
	struct nodeinfo *nodeinfo = self->opaque;
	uint8_t *alphabuf = nodeinfo->alphabuf;
	int shift = nodeinfo->shift;

	int b = in_tensor->shape.batches;
	int h = in_tensor->shape.height;
	int h_pad_before = in_tensor->format.height_pad[0];
	int h_pad_after = in_tensor->format.height_pad[1];
	int w = in_tensor->shape.width;
	int w_pad_before = in_tensor->format.width_pad[0];
	int w_pad_after = in_tensor->format.width_pad[1];
	int d = in_tensor->shape.depth;
	int d_pad_before = in_tensor->format.depth_pad[0];
	int d_pad_after = in_tensor->format.depth_pad[1];
	//int h_total = h+h_pad_before+h_pad_after;
	int w_total = w+w_pad_before+w_pad_after;
	int d_total = d+d_pad_before+d_pad_after;

	float in_range = (in_max-in_min);
	// find output range.
	// out_max must allow for most negative input * most negative alpha
	// and for most positive input.
	//
#if 1
	float out_max = fmaxf(in_max, in_min*-nodeinfo->alpha_mag);
	float out_min = fminf(in_min, in_min*nodeinfo->alpha_mag);
#else
	// @@@ this stuff produces a better range,
	// but causes errors due to limitations in the implementation.

	float out_max = fmaxf(in_max,in_min*nodeinfo->alpha_min);
	// out_min just allows for most -ve input and most +ve alpha.
	// however : don't allow output range to be less than 128/255 input range;
	// reduce min if needed to ensure that.
	//
	float out_min = in_min * nodeinfo->alpha_max;
	out_min = fminf( out_min, out_max-(float)(128./255.)*in_range);
#endif
	// make sure there is a clean zero.
	adjust_minmax_for_zero( &out_min, &out_max);

	uint8_t out_qzero = quantize_uint8(0.0f,out_min,out_max);

	float out_range = (out_max-out_min);
	float shrink_factor = in_range/out_range;
	uint32_t fixed_shrink_factor = Q6_R_satub_R(fast_roundf(128*shrink_factor));
	logmsg(nn,2,"in_range=%f out_range=%f shrink factor: %f fixed: %x shift: %d",in_range,out_range,shrink_factor,fixed_shrink_factor,shift);

	if (tensor_get_float(in_min_tensor,0) < -tensor_get_float(in_max_tensor,0)) {
		logmsg(nn,1,"Caution: min < -max");
	}

	logmsg(nn,2,"d: %d<%d>%d ==> %d\n",d_pad_before,d,d_pad_after,d_total);
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		b,
		h,h_pad_before,h_pad_after,
		w,w_pad_before,w_pad_after,
		d,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out prepare fail");
	}

	struct tensor_addressing tin = tensor_addressing_d32( in_tensor);
	struct tensor_addressing tout = tensor_addressing_d32( out_tensor);

	int wskip = w_pad_before & ~3;	// skip 4 if wpad = 4

	struct tdata td = {
		.self = self,
		.indata = tin.data-32*(w_pad_before-wskip),
		.outdata = tout.data-32*(w_pad_before-wskip),
		.in_next_row = tin.height_stride,
		.in_next_d32 = tin.d32_stride,
		.out_next_row = tout.height_stride,
		.out_next_d32 = tout.d32_stride,
		.in_qzero = in_qzero,
		.out_qzero = out_qzero,
		.d32_iters = (w_total-wskip) / 4u,
		.d_iters = tin.nd32,
		.h_iters = h,
		.alphabuf = alphabuf,
		.shrink = fixed_shrink_factor,
		.shift = shift,
	};

	nn_sem_init(&td.donesem,0);

	logmsg(nn,2,"Prelu execute. self=%p ",self);

	tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT);
	//tensor_set_float(out_min_tensor,0,nodeinfo->alpha_mag*tensor_get_float(in_min_tensor,0));
	//tensor_set_float(out_max_tensor,0,nodeinfo->alpha_mag*tensor_get_float(in_max_tensor,0));
	tensor_set_float(out_min_tensor,0,out_min);
	tensor_set_float(out_max_tensor,0,out_max);

	for (i = 0; i < b; i++) {
		nn_os_work_for_vector(nn,prelu_execute_slice,&td);
		nn_sem_wait(&td.donesem);
		td.indata += tin.batch_stride;
		td.outdata += tout.batch_stride;
	}

	logmsg(nn,2,"Prelu %p done",self);
	return 0;
}

static int prelu_check_opt(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 4) return errlog(nn,"maxpool wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"maxpool wrong # outs");
	const struct tensor *in_alpha_tensor = self->inputs[3];
	int32_t alpha_depth = in_alpha_tensor->shape.depth;
	int32_t alpha_depth_roundup = (alpha_depth + 31) & ~31;
	const float *alphas = in_alpha_tensor->data;
	struct nodeinfo *nodeinfo = self->opaque;
	uint8_t *alpha_frac_buf = NULL;
	int i,j,w;
	float maxmag;
	float magrecip;
	int shift;
	if (nodeinfo == NULL) {
		if ((nodeinfo = nn_malloc(sizeof(*nodeinfo))) == NULL) {
			return errlog(nn,"can't alloc nodeinfo");
		}
		if ((alpha_frac_buf = nn_memalign(128,alpha_depth_roundup*4)) == NULL) {
			nn_free(nodeinfo);
			return errlog(nn,"can't allocate alpha buf");
		}
		nodeinfo->alphabuf = alpha_frac_buf;
		self->opaque = nodeinfo;
	} else {
		alpha_frac_buf = nodeinfo->alphabuf;
	}
	/*
	 * Find the maximum magnitude of the alphas
	 * We will use this to scale the alpha values down and the output min/max up
	 * We also need to calculate the shift down of the positive input values
	 * All this is typically unnecessary, alpha input is likely small.
	 * Note, the hvx code requires shift <=7 since it does a right-shift by 7-shift.
	 */
	if( find_range_of_floats( alphas, alpha_depth,&nodeinfo->alpha_min, &nodeinfo->alpha_max)!= 0){
		return errlog(nn,"prelu: inf or nan in alphas");
	}
	float alpha_maxmag = fmaxf(-nodeinfo->alpha_min, nodeinfo->alpha_max);	// mag of range
	shift = 0;
	magrecip = maxmag = 1.0f;
	if( alpha_maxmag > 1.0f){
		shift = flt_getexp(alpha_maxmag);	// floor(log2(x))+1
		maxmag = flt_power2(shift);
		magrecip = flt_power2(-shift);
		if( shift >7 ) return errlog(nn,"magnitude of alphas is too large");
	}

	logmsg(nn,3,"alpha range= %f ... %f maxmag=%f shift=%d",nodeinfo->alpha_min,nodeinfo->alpha_max, maxmag,shift);

	nodeinfo->alpha_mult = magrecip;
	nodeinfo->alpha_mag = maxmag;
	nodeinfo->shift = shift;
	/*
	 * Create our buffer of alpha values.  We duplicate 4x so that it lines up with the D32 format.
	 * Alpha values are also turned into 8 bit signed (q7) format.
	 */
	for (i = 0; i < alpha_depth_roundup; i += 32) {
		for (j = 0; j < 32; j++) {
			float alphaval;
			if ((i+j) >= alpha_depth) alphaval = alphas[0];
			else alphaval = alphas[i+j];
			// round alpha * 128/maxmag to -128..127
			// quantize over 0..255 rather than -128..127, so that round-to-nearest works properly.
			int qval = Q6_R_satub_R((int)(128.0f* magrecip * alphaval + 128.5f))-128;	// b / ub / shift amt
			logmsg(nn,4,"alphas[%d+%d] = %f --> %x",i,j,alphaval,qval&0xFF);
			for (w = 0; w < 4; w++) {
				alpha_frac_buf[i*4+w*32+j] = qval;
			}
		}
	}
	return 0;
}

#endif

static int prelu_ref_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *in_alpha_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	const float *alphas = in_alpha_tensor->data;
	uint8_t quantized_zero = quantize_uint8(0.0f,in_min,in_max);
	int32_t val;
	int32_t b,h,w,d;
	const uint8_t *in_data;
	uint8_t *out_data;

	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int h_pad_before = in_tensor->format.height_pad[0];
	int h_pad_after = in_tensor->format.height_pad[1];
	int width = in_tensor->shape.width;
	int w_pad_before = in_tensor->format.width_pad[0];
	int w_pad_after = in_tensor->format.width_pad[1];
	int depth = in_tensor->shape.depth;
	int d_pad_before = in_tensor->format.depth_pad[0];
	int d_pad_after = in_tensor->format.depth_pad[1];

	if (tensor_out_prepare_padded_d32(
		out_tensor,
		batches,
		height,h_pad_before,h_pad_after,
		width,w_pad_before,w_pad_after,
		depth,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out prepare fail");
	}

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	for (b = 0; b < batches; b++) {
		for (h = 0; h < height; h++) {
			for (w = 0; w < width; w++) {
				for (d = 0; d < depth; d++) {
					in_data = tensor_location_d32(in_tensor,b,h,w,d);
					out_data = tensor_location_d32(out_tensor,b,h,w,d);
					val = *in_data;
					val = val - quantized_zero;
					if (val < 0) val = (val * alphas[d]) - 0.5f;
					val = val + quantized_zero;
					if (val < 0) val = 0;
					if (val > 255) val = 255;
					*out_data = val;
				}
			}
		}
	}
	logmsg(nn,2,"Prelu id %x ref done",self->node_id);
	return 0;
}

static int prelu_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct nodeinfo *nodeinfo = self->opaque;
	if (nodeinfo) {
		if (nodeinfo->alphabuf) nn_free(nodeinfo->alphabuf);
		nn_free(nodeinfo);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}
	

struct nn_node_ops nn_ops_for_QuantizedPRelu_8_d32 = {
	.execute = prelu_opt_execute,
	.check = prelu_check_opt,
	.ctor = node_alloc_common,
	.dtor = prelu_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedPRelu_8_d32_ref = {
	.execute = prelu_ref_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

