
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
 * This contains implementations for requantizing
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include "hvx_inlines.h"

#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif

#define VELEM(elem_size)  (sizeof(HVX_Vector) / elem_size)


// find the min/max of all values in arr[0..n]
// Result is at minmaxp (which must be vector-aligned and have room for 32*i32:
//   minmaxp[0] = minval
//   minmaxp[1[ = ~maxval

void
find_min_max_int32( int32_t const *arr, int n, int32_t * minmaxp )
{
	int nvec = n/ (unsigned)VELEM(sizeof(int32_t));
	int residue_elements = n % (unsigned)VELEM(sizeof(int32_t));

	HVX_Vector vec_in_max_val = Q6_V_vsplat_R(INT32_MIN);
	HVX_Vector vec_in_min_val = Q6_V_vsplat_R(INT32_MAX);
	HVX_Vector* pVec_inval = (HVX_Vector*) arr;
	HVX_Vector tmp_val;
	for( int i = 0; i < nvec; i++)
	{
		tmp_val = *pVec_inval++;
		vec_in_max_val = Q6_Vw_vmax_VwVw(vec_in_max_val, tmp_val);
		vec_in_min_val = Q6_Vw_vmin_VwVw(vec_in_min_val, tmp_val);
	}
	if(residue_elements > 0)
	{
		HVX_VectorPred residue_elements_mask = Q6_Q_vsetq_R(residue_elements * sizeof(int32_t));
		tmp_val = *pVec_inval;
		vec_in_max_val = Q6_Vw_vmax_VwVw(vec_in_max_val, Q6_V_vmux_QVV(residue_elements_mask, tmp_val, vec_in_max_val));
		vec_in_min_val = Q6_Vw_vmin_VwVw(vec_in_min_val, Q6_V_vmux_QVV(residue_elements_mask, tmp_val, vec_in_min_val));
	}
	// combine min and ~max
	HVX_VectorPair rshuff = Q6_W_vshuff_VVR( Q6_V_vnot_V(vec_in_max_val), vec_in_min_val, 4);
	vec_in_min_val = Q6_Vw_vmin_VwVw( Q6_V_hi_W(rshuff),Q6_V_lo_W(rshuff) );
	// now we have 16 of { min, ~max }
	int txn = 4;
	for( int  i = 0; i < 4; i++){
		txn *= 2;	// 8,16,32,64
		rshuff = Q6_W_vshuff_VVR( vec_in_min_val, vec_in_min_val, txn);
		vec_in_min_val = Q6_Vw_vmin_VwVw( Q6_V_hi_W(rshuff),Q6_V_lo_W(rshuff) );
	}
	*(HVX_Vector *)minmaxp = vec_in_min_val;
}



static int do_autorequantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;

	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min;
	float out_max;
	float stepsize;
	float recip_stepsize;
	int32_t in_max_val;
	int32_t in_min_val;
	int32_t inval;
	float in_level_size = (in_max - in_min) * (1.0f/ 4294967296.0f)/*0x1.0p-32f*/;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"autorequantize execute. self=%p ",self);
	logmsg(nn,2,"autorequantize in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));

	if( tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!=0){
		return errlog(nn,"out too small");
	}

	/* Find min and max quantized 32 bit val */

	//following hvx version code used to optimize the "for loop" for performance benifits
	// for (i = 0; i < elements; i++) {
	// 	inval = in_data[i];
	// 	if (inval > in_max_val) in_max_val = inval;
	// 	if (inval < in_min_val) in_min_val = inval;
	// }

	const uint32_t  log2VLEN = 5;
	uint32_t elements_vector_iterations = elements >> log2VLEN;
	inval = 0;
	l2fetch(in_data, 128, 128, elements_vector_iterations + 1);
	union { HVX_Vector as_v; int32_t as_i32[32];} minmax_union;
	find_min_max_int32( in_data, elements, & minmax_union.as_i32[0] );
	Q6_dcfetch_A(&minmax_union);

	in_max_val = max_i32(~minmax_union.as_i32[1], 0);
	in_min_val = min_i32(minmax_union.as_i32[0], 0);

	/* Make sure min val <= 0.0 in floaty land */
	out_min = in_level_size * (float)in_min_val;
	out_max = in_level_size * (float)in_max_val;

	quantize_adjust_range(
		&out_min,&out_max,
		&stepsize,&recip_stepsize,
		out_min,out_max);

	/* Requantize with new range */
	nn_requantize_i32_to_qu8_hvx( out_data, in_data, elements, in_level_size, out_min, out_max);

	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"autorequantize out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"autorequantize %p done",self);
	return 0;
}

static int do_requantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *min_val_tensor = self->inputs[3];
	const struct tensor *max_val_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min = tensor_get_float(min_val_tensor,0);
	float out_max = tensor_get_float(max_val_tensor,0);
	float in_level_size = (in_max - in_min) * (0x1.0p-32f);

	float recip_stepsize;
	float stepsize;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"autorequantize execute. self=%p ",self);
	logmsg(nn,2,"autorequantize in min/max=%f/%f ", in_min, in_max);

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!= 0 ){
		return errlog(nn,"out too small");
	}

	quantize_adjust_range(
		&out_min,&out_max,
		&stepsize,&recip_stepsize,
		out_min,out_max);

	/* Requantize with new range */

	nn_requantize_i32_to_qu8_hvx( out_data, in_data, elements, in_level_size,  out_min,  out_max);

	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"requantize out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"requantize %p done",self);
	return 0;
}

static void requantize_hvx_work_func(struct nn_graph *nn, void *rstpv)
{
    struct requant_runstate *rstp= (struct requant_runstate *)rstpv;
    uint8_t const* inp0 = rstp->inp;
    uint8_t *outp0 = rstp->outp;
    unsigned n_elem = rstp->n_elem;
    unsigned chunk = rstp->chunk;
    unsigned pos;
    float gain = rstp->gain;
    int32_t in_offset = rstp->in_offset;
    int32_t out_offset = rstp->out_offset;

    while(pos = __sync_fetch_and_add(&rstp->current_pos, chunk), pos < n_elem)
    {
        uint8_t const* inp = inp0 + pos;
        uint8_t *outp = outp0 + pos;
        unsigned numel = min_u32(chunk, n_elem - pos);
        l2fetch(inp, 128, 128, (numel + 127) / 128u);
        nn_requantize_qu8_to_qu8_hvx(outp, inp, numel, gain, in_offset, out_offset);
    }
    nn_sem_post(&rstp->done_sem);
}

static int do_requantize_8to8_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    const struct tensor *out_min_tensor = self->inputs[3];
    const struct tensor *out_max_tensor = self->inputs[4];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *new_out_min_tensor = self->outputs[1];
    struct tensor *new_out_max_tensor = self->outputs[2];

    if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!= 0 ){
        return errlog(nn,"out too small");
    }

    size_t n_elements = tensor_element_count(in_tensor);
    uint8_t *in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;
    const float in_min = tensor_get_float(in_min_tensor, 0);
    const float in_max = tensor_get_float(in_max_tensor, 0);
    float out_min = tensor_get_float(out_min_tensor, 0);
    float out_max = tensor_get_float(out_max_tensor, 0);
    float stepsize, recip_stepsize;

    float old_out_min = out_min, old_out_max = out_max;
    quantize_adjust_range(&out_min, &out_max, &stepsize, &recip_stepsize, out_min, out_max);
    if(old_out_min != out_min || old_out_max != out_max) {
        return errlog(nn, "Requant_8to8: Supplied output range (%f, %f) does not yield an integer zero-point. Try using (%f, %f)!", old_out_min, old_out_max, out_min, out_max);
    }
    tensor_set_single_float(new_out_min_tensor, out_min);
    tensor_set_single_float(new_out_max_tensor, out_max);

    const float in_range = in_max - in_min;
    const float out_range = out_max - out_min;
    const float gain = in_range / out_range;
    const uint8_t in_offset = saturate_u8(roundf_i32(-255 * in_min / in_range));
    const uint8_t out_offset = saturate_u8(roundf_i32(-255 * out_min / out_range));

    if(flt_getexp(gain) > 15) {
        return errlog(nn, "Requantize_8to8: Out range too small compared to in range!");
    }

    struct requant_runstate rstate;
    rstate.inp = in_data;
    rstate.outp = out_data;
    rstate.n_elem = n_elements;
    rstate.gain = gain;
    rstate.in_offset = in_offset;
    rstate.out_offset = out_offset;
    nn_sem_init(&rstate.done_sem, 0);

    unsigned n_vectors = (n_elements + 127) / 128u;
    unsigned chunk = 256;
    if(n_vectors < 512) {
        chunk = (n_vectors < 32) ? n_vectors : ((n_vectors + 1) >> 1);
    }
    rstate.chunk = 128 * chunk;
    rstate.current_pos = 0;
    int n_threads = (n_vectors > (NUM_THREADS - 1) * chunk) ? NUM_THREADS : (n_vectors + (chunk - 1)) / chunk;

    for(int i = 0; i < n_threads; i++) {
        nn_os_work_for_vector(nn, requantize_hvx_work_func, &rstate);
    }

    nn_sem_wait_n_times(&rstate.done_sem, n_threads);

   // nn_requantize_qu8_to_qu8_hvx(out_data, in_data, n_elements, gain, in_offset, out_offset);

    return 0;
}

static void scale_32_to_16( int16_t * outp, int32_t const * inp, int num, int32_t gain, int rsh );

//
// requant i32->i16 based on actual range of data.
// output range is always symmetrical.
//
static int do_autorequantize_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	int16_t *out_data = out_tensor->data;

	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min;
	float out_max;
	int32_t in_max_val;
	int32_t in_min_val;

	float in_level_size = (in_max - in_min) * (1.0f/ 4294967296.0f)/*0x1.0p-32f*/;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"autorequantize_16 execute. self=%p ",self);

	if( tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, NN_TYPE_QINT16)!=0){
		return errlog(nn,"out too small");
	}

	/* Find min and max quantized 32 bit val */

	const uint32_t  log2VLEN = 5;
	uint32_t elements_vector_iterations = elements >> log2VLEN;
	l2fetch(in_data, 128, 128, elements_vector_iterations + 1);
	union { HVX_Vector as_v; int32_t as_i32[32];} minmax_union;
	find_min_max_int32( in_data, elements, & minmax_union.as_i32[0] );
	Q6_dcfetch_A(&minmax_union);

	in_max_val = max_i32(~minmax_union.as_i32[1], 2);	// ensure range isn't 0:0
	in_min_val = min_i32(minmax_union.as_i32[0], 0);

	// make symmetric range
	out_max = in_level_size * (float)max_i32(in_max_val, -in_min_val);
	out_min = -out_max;

	// determine scaling...
	float out_level_size = out_max * (float)(1.0/32768.0);
	int32_t scalefac;
	float scale_by = in_level_size/out_level_size;
	int rsh;
	// since we are deciding the output range, we can enlarge it if
	// the actual data range is really small.
	// The output range will not be smaller than the input range/64k;
	// i.e. the out_level_size must be >= in_level_size
	if( scale_by >= 1.0f ){		// force to 1.0f
		out_max = in_level_size * 32768.0f;
		out_min = -out_max;
		scalefac = 0x7fffffff;		// select 'unity scale'
		rsh = 0;
	}else{

		// determine scaling, as xout = xin*scalefac/(2^(31+rsh))
		// (eqv. to xout = xin*scale_by)
		// where
		//     rsh = -24 ..15 (but normally >= 0)
		//	    0 < scalefac < 0x7FFFFFFF
		//       (but normally scalefac >= 0x40000000, except when needed to keep rsh <= 15)
		//
		int expo = flt_getexp(scale_by);	 // <= 0, >=-16 since scale_by < 1.0f , >= 1/64k
		rsh = min_i32(-expo, 15);
		scalefac = roundf_i32( flt_ldexp( scale_by, rsh+31));	// should be <= 0x7fffff80
		// so e.g, 0.5 maps to rsh = 0,  scalefac = 0x40000000
		//         0.375 maps to rsh = 1, scalefac =0x60000000
	}

	scale_32_to_16( out_data, in_data, elements, scalefac, rsh);


	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"autorequantize_16 out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"autorequantize_16 %p done",self);
	return 0;
}

//
// requantize i32->i16 based on indicated output range
//  inputs 1,2 represent the range of the i32
//  inputs 3,4 represent the desired range of the i16 output;
//   this is expanded as needed to make it symmetric.

static int do_requantize_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *min_val_tensor = self->inputs[3];
	const struct tensor *max_val_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	int16_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min = tensor_get_float(min_val_tensor,0);
	float out_max = tensor_get_float(max_val_tensor,0);
	float in_level_size = (in_max - in_min) * (1.0f/ 4294967296.0f)/*0x1.0p-32f*/;
	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"requantize_16 execute. self=%p ",self);
	logmsg(nn,2,"requantize_16 in min/max=%f/%f ",in_min, in_max );

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QINT16)!= 0 ){
		return errlog(nn,"out too small");
	}
	out_max = fmaxf( (float)(1./2048.),fmaxf( out_max, -out_min)) ;
	out_min = -out_max;
	float out_level_size = out_max * (float)(1.0/32768.0);
	float scale_by = in_level_size/out_level_size;

	// scaling is done, usually, by:
	// (1) multiply using fractional 32-bit mul (scale value <2^31, implied /2^31)
	// (2) right-shift by rsh= 0..15, round, truncate to i16 range.
	//
	// Ideally the scale factor in (1) is normalized; to avoid a shift > 15 in (2), we allow
	// it to be denormalized.
	// If the overall scale factor is >= 1, we can't use this approach; instead we have rsh < 0 and
	// (1) clip the 32-bit value to a 'safe' range
	// (2) << the value by (4-rsh) bits
	// (3) as (1) above
	// (4) right-shift by 2, round, truncate to i16 range
	//
	if( ! flt_isfinite(scale_by) || scale_by < 0x1.0p-29f  || scale_by >= 0x1.0p24f) {
		return errlog(nn,"scale ratio for 32->16 is not reasonable");
	}
	int expo = flt_getexp(scale_by);
	int rsh = min_i32(-expo, 15);
	int32_t scalefac = roundf_i32( flt_ldexp( scale_by, rsh+31));	// should be <= 0x7fffff80
	// so e.g, 0.5 maps to rsh = 0,  scalefac = 0x40000000
	//         0.375 maps to rsh = 1, scalefac =0x60000000

	scale_32_to_16( out_data, in_data, elements, scalefac, rsh);
	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"requantize_16 out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"requantize_16 %p done",self);
	return 0;
}

//
// scale n 'int32' to 'int16' using vector ops.  The scaling
// done by  'gain' * 2^-(31+rsh); gain must be >1, <= 0x7fffffff
// and rsh must be in range -24..15
//
// -- input, output pointers must be vec-aligned
//
// When rsh is >= 0, this is done by:
//    (1) frac mul by gain/2^31
//    (2) >>rsh ,round, truncate to i16 range
// When rsh < 0, done by:
//    (1) clip value to range to avoid overflow at (2)
//    (2)  << by (4-rsh)
//    (3) frac mul by gain/2^31
//    (4) >>4, round, truncate to i16 range.
//
static void
scale_32_to_16( int16_t * outp, int32_t const * inp, int num, int32_t gain, int rsh )
{
	HVX_Vector vgain = Q6_V_vsplat_R( gain );
	int nv32 = (num+31)/32u;		// # of vecs to read
	int nv64 = nv32>>1;		// # of full 2->1 operations

	HVX_Vector const * vinp = (HVX_Vector const*)inp;
	HVX_Vector *voutp = (HVX_Vector *)outp;

	if( rsh >= 0 ){
		for( int i = 0; i < nv64; i++){
			HVX_Vector v0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[0], vgain);
			HVX_Vector v1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[1], vgain);
			*voutp =Q6_Vh_vdeal_Vh( Q6_Vh_vasr_VwVwR_rnd_sat( v1,v0, rsh));
			vinp += 2;
			voutp += 1;
		}
		if( nv32&1){
			HVX_Vector v0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[0], vgain);
			*voutp = Q6_Vh_vdeal_Vh(Q6_Vh_vasr_VwVwR_rnd_sat( v0,v0, rsh));

		}
	}else{
		int lshamt = 4-rsh;		// this is 5..28
		int maxin = ((unsigned)0x80000000 >> lshamt) - 1;	// upper limit for pre-clip
		HVX_Vector v_max = Q6_V_vsplat_R( maxin);
		HVX_Vector v_min = Q6_V_vnot_V(v_max);		// lower-limit for pre-clip
		for( int i = 0; i < nv64; i++){
			HVX_Vector v0 = Q6_Vw_vmax_VwVw( Q6_Vw_vmin_VwVw( vinp[0], v_max), v_min );
			HVX_Vector v1 = Q6_Vw_vmax_VwVw( Q6_Vw_vmin_VwVw( vinp[1], v_max), v_min );
			v0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_Vw_vasl_VwR(v0,lshamt), vgain);
			v1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_Vw_vasl_VwR(v1,lshamt), vgain);
			*voutp =Q6_Vh_vdeal_Vh( Q6_Vh_vasr_VwVwR_rnd_sat( v1,v0, 4));
			vinp += 2;
			voutp += 1;
		}
		if( nv32&1){
			HVX_Vector v0 = Q6_Vw_vmax_VwVw( Q6_Vw_vmin_VwVw( vinp[0], v_max), v_min );
			v0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_Vw_vasl_VwR(v0,lshamt), vgain);
			*voutp =Q6_Vh_vdeal_Vh( Q6_Vh_vasr_VwVwR_rnd_sat( v0,v0, 4));
		}
	}
}




static int do_requantrange_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_min_tensor = self->outputs[0];
	struct tensor *out_max_tensor = self->outputs[1];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	uint32_t i;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float out_min;
	float out_max;
	int32_t in_max_val;
	int32_t in_min_val;
	int32_t inval;
	float in_level_size = (in_max - in_min) * (1.0f/ 4294967296.0f)/*0x1.0p-32f*/;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"requantrange execute. self=%p ",self);
	logmsg(nn,2,"requantrange in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));

	/* Find min and max quantized 32 bit val */
	/* start from 0 */
	in_max_val = 0;
	in_min_val = 0;
	for (i = 0; i < elements; i++) {
		inval = in_data[i];
		if (inval > in_max_val) in_max_val = inval;
		if (inval < in_min_val) in_min_val = inval;
	}

	out_min = in_level_size * (float)in_min_val;
	out_max = in_level_size * (float)in_max_val;

	tensor_set_single_float(out_min_tensor,out_min);
	tensor_set_single_float(out_max_tensor,out_max);

	logmsg(nn,2,"requantrange out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"requantrange %p done",self);
	return 0;
}

struct tdata {
	int (*f)(struct nn_node *self, struct nn_graph *nn);
	int retval;


	struct nn_node *self;
	nn_sem_t donesem;
};

void worker(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	td->retval = td->f(td->self,nn);
	nn_sem_post(&td->donesem);
}

int launcher(struct nn_node *self, struct nn_graph *nn, 
	int (*f)(struct nn_node *self, struct nn_graph *nn))
{
	struct tdata td = {
		.f = f,
		.self = self,
		.retval = 0,
	};
	nn_sem_init(&td.donesem,0);
	nn_os_work_for_vector(nn,worker,&td);
	nn_sem_wait(&td.donesem);
	return td.retval;
}

static int requantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_requantize_execute);
}

static int requantize_8to8_execute(struct nn_node *self, struct nn_graph *nn)
{
    return launcher(self, nn, do_requantize_8to8_execute);
}

static int requantrange_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_requantrange_execute);
}
static int autorequantize_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_autorequantize_execute);
}

static int autorequantize_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_autorequantize_16_execute);
}
static int requantize_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self,nn,do_requantize_16_execute);
}

static int do_requantize_u16_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *min_val_tensor = self->inputs[3];
	const struct tensor *max_val_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t elements = tensor_element_count(in_tensor);
	const int32_t *in_data = in_tensor->data;
	uint16_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor, 0);
	float in_max = tensor_get_float(in_max_tensor, 0);
	float out_min = tensor_get_float(min_val_tensor, 0);
	float out_max = tensor_get_float(max_val_tensor, 0);
	float in_level_size = (in_max - in_min) * 0x1.0p-32f;

	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn, 2, "autorequantize execute. self=%p ", self);
	logmsg(nn, 2, "autorequantize in min/max=%f/%f ", in_min, in_max);

	if (tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, NN_TYPE_QUINT8) != 0) {
		return errlog(nn, "out too small");
	}

	adjust_minmax_for_zero_16b(&out_min, &out_max);

	/* Requantize with new range */
	uint32_t i;
	for (i = 0; i < elements; i++) {
		out_data[i] = quantize_uint16(
			in_level_size* (float)in_data[i], out_min, out_max);
	}

	tensor_set_single_float(out_min_tensor, out_min);
	tensor_set_single_float(out_max_tensor, out_max);

	logmsg(nn, 2, "requantize out min/max=%f/%f ",
		tensor_get_float(out_min_tensor, 0),
		tensor_get_float(out_max_tensor, 0));
	logmsg(nn, 2, "requantize %p done", self);
	return 0;
}

static int requantize_u16_execute(struct nn_node *self, struct nn_graph *nn)
{
	return launcher(self, nn, do_requantize_u16_execute);
}

//
// QuantizeDownAndShrinkRange_32to8:
// convert 32-bit quantized to 8-bit quantized, checking the actual
// range of values to find the output range.
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//
//  output 0:   qu8 tensor
//  output 1,2:  scalar float, output min & max
//
struct nn_node_ops nn_ops_for_QuantizeDownAndShrinkRange_32to8 = {
	.execute = autorequantize_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizeDownAndShrinkRange_32to8_ref = {
	.execute = autorequantize_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};


//
// QuantizeDownAndShrinkRange_32to16:
// convert 32-bit quantized to 16-bit quantized, checking the actual
// range of values to find the output range.
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//
//  output 0:   qi16 tensor
//  output 1,2:  scalar float, output min & max
//
struct nn_node_ops nn_ops_for_QuantizeDownAndShrinkRange_32to16 = {
	.execute = autorequantize_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizeDownAndShrinkRange_32to16_ref = {
	.execute = autorequantize_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
//
// Requantize_32to8:
// convert 32-bit quantized to 8-bit quantized, relying on
// supplied range (inputs 3 & 4) to determine range.
// The output range may be expanded slightly to produce a
// a properly aligned zero code.
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//  input 3,4:  scalar float, output min & max
//
//  output 0:   qu8 tensor
//  output 1,2:  scalar float, output min & max
//
struct nn_node_ops nn_ops_for_Requantize_32to8 = {
	.execute = requantize_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Requantize_32to8_ref = {
	.execute = requantize_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

// Requantize_8to8:
// Change the quantization range of 8-bit quantized values.
//
//  input 0: quint8 tensor
//  input 1, 2: float32 input min, max
//  input 3, 4: float32 target output min, max
//
//  output 0: quint8 requantized tensor
//  output 1, 2: float32 new output min, max
//
struct nn_node_ops nn_ops_for_Requantize_8to8 = {
    .execute = requantize_8to8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(5),
    .n_outputs = NN_IOCOUNT(3)
};

// Requantize_32to16:
// convert 32-bit quantized to 16-bit quantized, relying on
// supplied range (inputs 3 & 4) to determine range.
// If necessary, the indicated range is expanded to make it
// symmetric
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//  input 3,4:  scalar float, output min & max
//
//  output 0:   qi16 tensor
//  output 1,2:  scalar float, output min & max
struct nn_node_ops nn_ops_for_Requantize_32to16 = {
	.execute = requantize_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Requantize_32to16_ref = {
	.execute = requantize_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Requantize_32tou16 = {
	.execute = requantize_u16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

//
// RequantizationRange_32
// find the range of values in qi32 data, and produce this
// range as min,max floats
// Note that the range is expanded if needed so that min <=0,
// but is not adjusted to have an integer zero point.
//
//  input 0:   qi32 tensor
//  input 1,2:  scalar float, input min & max
//
//  output 0,1  scalar float, input actual min & max
//

struct nn_node_ops nn_ops_for_RequantizationRange_32 = {
	.execute = requantrange_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(2),
	.flags = NN_NODE_FLAG_CLS_REQUANTRANGE,
};

struct nn_node_ops nn_ops_for_RequantizationRange_32_ref = {
	.execute = requantrange_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(2)
};


