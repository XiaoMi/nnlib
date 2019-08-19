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


#include <nn_graph.h>
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#include "nn_broadcast.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_inlines.h"
#include "nn_oper16.h"

//
// '16-bit' implemenations of Add/Sub/Mul with broadcast
// All require a specific output range (inputs 6,7).
// for the 'signed' 16 ops, it is assumed that max = -min;
//  the function may only look at the differene between for scaling.
//
//   input 0:  'a' input
//   input 1:  'b' input
//   input 2,3:  'a' min, max
//   input 4,5:  'b' min, max
//   input 6,7   output min,max
//
//   output 0: result
//   output 1,2: output range (copied from inputs 6,7)
//
// In some *very* unusual cases, the node will replace the specified output range
// with a larger one (generally this is when the specified range is far too small
// for the given input ranges).

//
// QuantizedAdd_16		 - signed 16-bit ops (symmetric range)
// QuantizedSub_16
// QuantizedMul_16
//
// QuantizedAdd_u16		 - signed 16-bit ops (asymmetric range)
// QuantizedSub_u16
// QuantizedMul_u16
//

/////////////////////////////////// Add/Sub //////////////////////////////////////////
//
// 16 bit signed add/subtract are done as below:
//
//     int32_t sop = a * scalea + b * scaleb
//     result = saturate_16(  (sop >> rsh ))     // (>> done with rounding
//
//  here:
//    scalea = a_qstep/out_qstep * (1<<rsh)
//    scaleb = b_qstep/out_qstep * (1<<rsh)    // (and change sign if subtracting)

//  .. with rsh chosen as large as possible so that scalea, |scaleb| are both < 32768.
//  (and not to exceed 15. so, if the output range is larger than both input ranges,
//  the rsh stays at 15 and the scale values get small
//  We need rsh to be at least 0; to enforce this, we require the output range to be
//  at least 1/8192 of the largest of the input ranges, and if it is not, we use that
//  larger value instead of the recommended range.
//
//---------------
// 16 bit unsigned (asymmetric) add is done as below:
//     uint32_t sop0 = a * scalea + b * scaleb
//     int32_t sop = (sop>>1)	+ delta			// saturating add
//     result = saturate_u16(  (sop >> rsh ))     // (>> done with rounding)
//
//  here:
//    scalea = a_qstep/out_qstep * (1<<(rsh+1))
//    scaleb = b_qstep/out_qstep * (1<<(rsh+1))
//    delta = (zero_out<<rsh) -  (zero_a*scalea - zero_b * scaleb)/2
//
// The multiplies are done by u16*u16; since scalea, scaleb are both <= 32767, no unsigned overflow
// can occur when adding them, but the '+delta' should be done with saturation.
// for subtraction, we take the 1's complement of b before doing the above, so we are really finding
//     sop =  a* scalea - b *scaleb + 65535*scaleb
//  .. and therefore
//    	delta = (zero_out<<rsh) -  zero_a*scale - (65535-zero_b) * scaleb
//
//

struct addsub_16_scaling_parms {
	int16_t scalea;				// 'a' scale (always >=0, <= 32767
	int16_t scaleb;				// 'b' scale ( 0..32767; or negative in s16 subtract)
	int16_t rsh;				//  rsh amount 0..15
	int16_t notb;				// [u16] only in u16 op: -1 if sub, 0 if add
	int32_t delta;				// [u16] .
	int32_t s32a,s32b;			// [u16] 32-bit a,b scales used in vector+scalar
	int32_t delta_revscalar;	// [u16] delta for (scalar)a +/- (vector)b
	float out_min;
	float out_max;
};

static int
set_addsub_scaling( struct nn_graph *nn, struct nn_node * self, struct addsub_16_scaling_parms * parms, int is_sub, int is_u16)
{
	float tmp[6];
	for( int i = 0; i < 6; i++){
		tmp[i] = tensor_get_float( self->inputs[2+i],0);
	}
	float a_qstep = (tmp[1]-tmp[0]) * (float)(1./65536.);
	float b_qstep = (tmp[3]-tmp[2]) * (float)(1./65536.);
	float out_qstep = (tmp[5]-tmp[4]) * (float)(1./65536.);
	int a_zero, b_zero, out_zero = 0x8000;
	if (is_u16 ){
		a_zero = saturate_u16( (int)(-tmp[0]/a_qstep + 0.5f));
		b_zero = saturate_u16( (int)(-tmp[2]/b_qstep + 0.5f));
		out_zero = saturate_u16( (int)(-tmp[4]/out_qstep + 0.5f));
	}
	parms->out_min = tmp[4];
	parms->out_max = tmp[5];

	float min_outstep = fmaxf( a_qstep, b_qstep) * (float)(1./8192.);
	if( out_qstep < min_outstep){
		out_qstep = min_outstep;
		parms->out_min = (float)(-out_zero)* out_qstep;
		parms->out_max = (float)(65536-out_zero)*out_qstep;
		logmsg(nn,0,"NOTE: expanding range to %f .. %f so scaling is feasible\n", parms->out_min, parms->out_max );
	}
	float a_thru_scale = a_qstep / out_qstep;
	float b_thru_scale = b_qstep / out_qstep;

	int k = is_u16? -1:0;

	// the 1.0004 is to prevent e.g. thru_scale = 0.99999 from giving scale = 32768 instead of 16384
	int rsh = 15-max_i32(k,flt_getexp( 1.00004f * fmaxf(a_thru_scale, b_thru_scale)));
	// rsh is 1..15 ( min = 1 due to 'min_outstep' ) or  1..6 for u16
	int scale_a = min_i32(32767, roundf_u32( flt_ldexp( a_thru_scale, rsh )));
	int scale_b = min_i32(32767, roundf_u32( flt_ldexp( b_thru_scale, rsh )));

	parms->scalea = scale_a;
	parms->scaleb = scale_b;
	rsh+= k;
	parms->rsh = rsh;

	logmsg(nn,3,"ina = (%f...%f)[%.6g @ %d]  inb = (%f...%f)[ %.6g @ %d]  out= (%f...%f)[ %.6g @ %d]",
		tmp[0],tmp[1], a_qstep,a_zero, tmp[2], tmp[3], b_qstep,b_zero,
		parms->out_min, parms->out_max, out_qstep, out_zero);
	logmsg(nn,3,"a * %f;  b* %f; scale_a = %d scale_b = %d  rsh = %d", a_thru_scale, b_thru_scale,
		scale_a, scale_b, rsh );


	if( !is_u16){ // almost done
		if( is_sub ) parms->scaleb = -scale_b;
	}else{
		// 'through gains' with 18 fractional bits, used by vec+scalar cases
		parms->s32a = roundf_u32( a_thru_scale * (float)(1<<18) );
		int sb = roundf_u32( b_thru_scale * (float)(1<<18) );
		parms->s32b = is_sub? -sb:sb;
		// is a u16 operation...
		parms->notb = is_sub?-1:0;
		int bzero_adj = is_sub? (65535-b_zero): b_zero;
		int32_t delta= (out_zero << rsh) - ( (uint32_t)(a_zero*scale_a + bzero_adj *scale_b)>>1);
		parms->delta = delta;
		if( is_sub){
			// special delta for (scalar)a - (vector)b case.
			delta= (out_zero << rsh) -( (int32_t)(a_zero*scale_a - b_zero*scale_b)>>1);
		}
		parms->delta_revscalar = delta;
		logmsg(nn,3,"notb = %d; delta = %d\n", parms->notb, (int)parms->delta );
	}
	return 0;
}
////////////////////////////////////////////////////////
// general s16 add/sub function (vector+vector)
//

static void addsub_s16_stride_11( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	int16_t * op = (int16_t*)out;
	int16_t const * inpa = (int16_t const *)in1;
	int16_t const * inpb = (int16_t const *)in2;
	int ka = info->scalea;
	int kb = info->scaleb;
	int rsh = info->rsh;
	int rnd = (1<<rsh)>>1;

	if( n > 0){
		int32_t s = ka* (*inpa++) + kb * (*inpb++);

		for(int i = 0; i < n-1; i++){
			*op++ = saturate_i16( (s + rnd)>>rsh);
			s = ka* (*inpa++) + kb * (*inpb++);
		}
		*op = saturate_i16( (s + rnd)>>rsh);
	}
}
//
// hvx version of addsub_s16_stride_11
//
static
void addsub_s16_stride_11_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	HVX_Vector * op = (HVX_Vector*)out;
	HVX_Vector const * inpa = (HVX_Vector const *)in1;
	HVX_Vector const * inpb = (HVX_Vector const *)in2;
	int ka = Q6_R_combine_RlRl(info->scalea,info->scalea);
	int kb = Q6_R_combine_RlRl(info->scaleb,info->scaleb);
	int rsh = info->rsh;
	int nvec1 = (n-1)/64u;
	if( n > 0){
		HVX_Vector vina = q6op_V_vldu_A(inpa++);
		HVX_Vector vinb = q6op_V_vldu_A(inpb++);
		HVX_VectorPair sop = Q6_Ww_vmpyacc_WwVhRh_sat( Q6_Ww_vmpy_VhRh(vina,ka),vinb,kb);
		for(int i = 0; i < nvec1; i++){
			HVX_Vector vout = q6op_Vh_vasr_WwR_rnd_sat(sop,rsh);
			vina = q6op_V_vldu_A(inpa++);
			vinb = q6op_V_vldu_A(inpb++);
			q6op_vstu_AV( op, vout); op++;
			sop = Q6_Ww_vmpyacc_WwVhRh_sat( Q6_Ww_vmpy_VhRh(vina,ka),vinb,kb);
		}
		HVX_Vector vout = q6op_Vh_vasr_WwR_rnd_sat(sop,rsh);
		hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
	}
}
///////////////////////////////////////////
// 'inner' op for the add/sub s16 vector+/- scalar case:
//         out[i] = (  in[i]*scale + offs) >>rsh
static void addsub_s16_withscalar( int16_t * op, int16_t const *inp, int scale, int offs, int rsh, int n)
{
	for( int i =0 ; i < n; i++){
		int32_t s = Q6_R_add_RR_sat(inp[i]*scale, offs);
		op[i] = saturate_i16( s >> rsh);
	}
}
// 'in2' pointer (b) does not move here.
static void addsub_s16_stride_10( void *out, void const *in1, void const *in2, int n, void *infov)
{
 	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
 	int16_t * op = (int16_t*)out;
 	int16_t const * inpa = (int16_t const *)in1;
 	int16_t const * inpb = (int16_t const *)in2;
 	int ka = info->scalea;
 	int kb = info->scaleb;
 	int rsh = info->rsh;
 	int rnd = (1<<rsh)>>1;
 	int offs = kb*inpb[0] + rnd;

 	addsub_s16_withscalar( op, inpa, ka, offs, rsh, n );
}
// 'in2' pointer (a) does not move here.
static void addsub_s16_rev_stride_01( void *out, void const *in1, void const *in2, int n, void *infov)
{
 	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
 	int16_t * op = (int16_t*)out;
 	int16_t const * inpa = (int16_t const *)in2;
	int16_t const * inpb = (int16_t const *)in1;
 	int ka = info->scalea;
 	int kb = info->scaleb;
 	int rsh = info->rsh;
 	int rnd = (1<<rsh)>>1;
 	int offs = ka*inpa[0] + rnd;

 	addsub_s16_withscalar( op, inpb, kb, offs, rsh, n );
}
// hvx version of that...
// 'inner' op for the add/sub s16 vector+/- scalar case:
//         out[i] = (  in[i]*scale + offs) >>rsh
// (this is different from the scalar version, in that 'offs' does not
// have the rounding bias baked in)
static void __attribute__((noinline))
addsub_s16_withscalar_hvx( int16_t * op0, int16_t const *inp0, int rsh_scale, int n,  int offs)
{
	int rsh = rsh_scale >>16;
	int scale = Q6_R_combine_RlRl( rsh_scale,rsh_scale);
	HVX_Vector * op = (HVX_Vector*)op0;
	HVX_Vector const * inp = (HVX_Vector const *)inp0;
	HVX_VectorPair voffs = Q6_W_vcombine_VV( Q6_V_vsplat_R(offs),Q6_V_vsplat_R(offs));
	int nvec1 = (n-1)/64u;

	HVX_Vector vout;
	HVX_Vector vin = q6op_V_vldu_A(inp); inp++;
	HVX_VectorPair prod = Q6_Ww_vmpyacc_WwVhRh_sat(voffs, vin,scale);
	for( int i =0 ; i < nvec1; i++){
		vout = q6op_Vh_vasr_WwR_rnd_sat( prod, rsh);
		vin = q6op_V_vldu_A(inp); inp++;
		q6op_vstu_AV( op, vout);		op ++;
		prod = Q6_Ww_vmpyacc_WwVhRh_sat(voffs, vin,scale);
	}
	vout = q6op_Vh_vasr_WwR_rnd_sat( prod, rsh);
	hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
}
// 'in2' pointer (b) does not move here.
static void addsub_s16_stride_10_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	int16_t * op = (int16_t*)out;
	int16_t const * inpa = (int16_t const *)in1;
	int16_t const * inpb = (int16_t const *)in2;
	int ka = info->scalea;
	int kb = info->scaleb;
	int rsh_scale = Q6_R_combine_RlRl( info->rsh,ka);
	int offs = kb*inpb[0];

	addsub_s16_withscalar_hvx( op, inpa, rsh_scale, n, offs );
}
// 'in2' pointer (a) does not move here.
static void addsub_s16_rev_stride_01_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	int16_t * op = (int16_t*)out;
	int16_t const * inpa = (int16_t const *)in2;
	int16_t const * inpb = (int16_t const *)in1;
	int ka = info->scalea;
	int kb = info->scaleb;
	int rsh_scale = Q6_R_combine_RlRl( info->rsh, kb );
	int offs = ka*inpa[0];

	addsub_s16_withscalar_hvx( op, inpb, rsh_scale, n, offs );
}


static const struct elementwise_funcs AddSub_s16_funcs = {
	.op_stride_11 = addsub_s16_stride_11,
	.op_stride_10 = addsub_s16_stride_10,
	.op_rev_stride_01 = addsub_s16_rev_stride_01,

	.op_hvx_stride_11 = addsub_s16_stride_11_hvx,
	.op_hvx_stride_10 = addsub_s16_stride_10_hvx,
	.op_hvx_rev_stride_01 = addsub_s16_rev_stride_01_hvx,
	.minlen_hvx = 8,		// use scalar if inner-loop n < 8

	.in_elbytes = 2,
	.out_elbytes = 2,
	.out_typecode =  NN_TYPE_QINT16
};
//--------------------------------------
//
// general u16 add/sub function (vector+vector)
//

static void addsub_u16_stride_11( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inp1 = (uint16_t const *)in1;
	uint16_t const * inp2 = (uint16_t const *)in2;
	int ka = info->scalea;
	int kb = info->scaleb;
	int invb = info->notb & 0xFFFF;
	int rsh = info->rsh;
	int delta = info->delta;
	int rnd = (1<<rsh)>>1;
	delta += rnd;			// roll rounding bias into delta...

	if( n > 0){
		int32_t s = (unsigned)( ka* (*inp1++) + kb * (invb^ *inp2++)) >>1;
		for(int i = 0; i < n-1; i++){
			*op++ = saturate_u16(  Q6_R_add_RR_sat( s, delta) >>rsh);
			s = (unsigned)(ka* (*inp1++) + kb * (invb^ *inp2++))>>1;
		}
		*op = saturate_u16(  Q6_R_add_RR_sat( s, delta) >>rsh);
	}
}

//
// HVX version of that
//
// for v60 we can't >> sat to u16, so we >> sat to i16 (with an offset) and then xor 0x8000.

static void addsub_u16_stride_11_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	HVX_Vector * op = (HVX_Vector*)out;
	HVX_Vector const * inp1 = (HVX_Vector const *)in1;
	HVX_Vector const * inp2 = (HVX_Vector const *)in2;
	HVX_Vector b_inv = Q6_V_vsplat_R( info->notb);
	int rsh = info->rsh;
	int delta = info->delta;
#if __HEXAGON_ARCH__ < 62
	delta -= 0x8000<<rsh;		// shift to 'int16' range
	HVX_Vector vk8000= Q6_V_vsplat_R(0x80008000);
#endif
	HVX_Vector vdelta = Q6_V_vsplat_R(delta);
	int ka = info->scalea;
	int kb = info->scaleb;
	ka = Q6_R_combine_RlRl( ka, ka);
	kb = Q6_R_combine_RlRl( kb, kb);
	if ( n <= 0) return;
	int nvec1= (n-1)/64u;

	HVX_Vector vout;
	HVX_Vector vina = q6op_V_vldu_A( inp1);	inp1++;
	HVX_Vector vinb = q6op_V_vldu_A( inp2);	inp2++;
	vinb = Q6_V_vxor_VV( vinb, b_inv);
	HVX_VectorPair sum = Q6_Wuw_vmpyacc_WuwVuhRuh( Q6_Wuw_vmpy_VuhRuh(vina,ka),vinb,kb);
	for( int i =0; i < nvec1; i++){
		HVX_Vector sum0 = Q6_Vw_vadd_VwVw_sat( Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(sum),1), vdelta );
		HVX_Vector sum1 = Q6_Vw_vadd_VwVw_sat( Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(sum),1), vdelta );
#if __HEXAGON_ARCH__ >= 62
		vout = Q6_Vuh_vasr_VwVwR_rnd_sat( sum1, sum0, rsh );
#else
		vout = Q6_V_vxor_VV(Q6_Vh_vasr_VwVwR_rnd_sat( sum1, sum0, rsh ),vk8000);
#endif
		vina = q6op_V_vldu_A( inp1);	inp1++;
		vinb = q6op_V_vldu_A( inp2);	inp2++;
		q6op_vstu_AV( op, vout); op++;
		vinb = Q6_V_vxor_VV( vinb, b_inv);
		sum = Q6_Wuw_vmpyacc_WuwVuhRuh( Q6_Wuw_vmpy_VuhRuh(vina,ka),vinb,kb);
	}
	HVX_Vector sum0 = Q6_Vw_vadd_VwVw_sat( Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(sum),1), vdelta );
	HVX_Vector sum1 = Q6_Vw_vadd_VwVw_sat( Q6_Vuw_vlsr_VuwR(Q6_V_hi_W(sum),1), vdelta );
#if __HEXAGON_ARCH__ >= 62
	vout = Q6_Vuh_vasr_VwVwR_rnd_sat( sum1, sum0, rsh );
#else
	vout = Q6_V_vxor_VV(Q6_Vh_vasr_VwVwR_rnd_sat( sum1, sum0, rsh ),vk8000);
#endif
	hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
}


// 'in2' pointer (b) does not move here.
static void addsub_u16_stride_10( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in1;
	uint16_t const * inpb = (uint16_t const *)in2;
	int gain = info->s32a;
	int kb = info->scaleb;
	int rsh = info->rsh;
	int delta = info->delta;
	int bval =  kb*( (info->notb&0xFFFF) ^ inpb[0]);
	delta = Q6_R_add_RR_sat( (bval>>1), delta );
	// ok, we need a result of (1/8)( gain * in/2^15 +   [delta/2^(rsh-3)] )
	// nn_do_scale_s16_hvx  will do that, if
	//    (offset-gain)+ 32768*8 =   [delta/2^(rsh-3)]. So, solve for offset...
	if( rsh>3 ){
		delta = ((delta >> (rsh-4))+1) >> 1;
	}else{
		delta <<= (3-rsh);
	}
	int offset = delta - 32768*8 + gain;
	nn_do_scaleoff_16to16( op, inpa, 0x80008000, gain, offset,n );
}

// 'in2' pointer (a) does not move here.
// This is trickier; if 'notb' we need to change the sign of kb and adjust the offset.
static void addsub_u16_rev_stride_01( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in2;
	uint16_t const * inpb = (uint16_t const *)in1;
	int gain = info->s32b;
	int ka = info->scalea;
	int rsh = info->rsh;
	int delta = info->delta_revscalar;
	int aval =  ka*inpa[0];
	delta = Q6_R_add_RR_sat( (aval>>1), delta );
	// ok, we need a result of (1/8)( gain * in/2^15 +   [delta/2^(rsh-3)] )
	// nn_do_scale_s16_hvx  will do that, if
	//    (offset-gain)+ 32768*8 =   [delta/2^(rsh-3)]. So, solve for offset...
	// when subtracting, we need to increase delta by (65535*scaleb)/2 to compensate
	// for the ~b we are not doing (we have gain<0 instead -- but this is already cooked
	// into delta_revscalar.
	if( rsh>3 ){
		delta = ((delta >> (rsh-4))+1) >> 1;
	}else{
		delta <<= (3-rsh);
	}
	int offset = delta - 32768*8 + gain;
	nn_do_scaleoff_16to16( op, inpb, 0x80008000, gain, offset,n );
}


// hvx version...
// 'in2' pointer (b) does not move here.
static void addsub_u16_stride_10_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in1;
	uint16_t const * inpb = (uint16_t const *)in2;
	int gain = info->s32a;
	int kb = info->scaleb;
	int rsh = info->rsh;
	int delta = info->delta;
	int bval =  kb*( (info->notb&0xFFFF) ^ inpb[0]);
	delta = Q6_R_add_RR_sat( (bval>>1), delta );
	// ok, we need a result of (1/8)( gain * in/2^15 +   [delta/2^(rsh-3)] )
	// nn_do_scale_s16_hvx  will do that, if
	//    (offset-gain)+ 32768*8 =   [delta/2^(rsh-3)]. So, solve for offset...
	if( rsh>3 ){
		delta = ((delta >> (rsh-4))+1) >> 1;
	}else{
		delta <<= (3-rsh);
	}
	int offset = delta - 32768*8 + gain;
	nn_do_scaleoff_16to16_hvx( op, inpa, 0x80008000, gain, offset,n );
}
// hvx versionb
// 'in2' pointer (a) does not move here.
static void addsub_u16_rev_stride_01_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_16_scaling_parms const * info = (struct addsub_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in2;
	uint16_t const * inpb = (uint16_t const *)in1;
	int gain = info->s32b;
	int ka = info->scalea;
	int rsh = info->rsh;
	int delta = info->delta_revscalar;
	int aval =  ka*inpa[0];
	delta = Q6_R_add_RR_sat( (aval>>1), delta );
	// ok, we need a result of (1/8)( gain * in/2^15 +   [delta/2^(rsh-3)] )
	// nn_do_scale_s16_hvx  will do that, if
	//    (offset-gain)+ 32768*8 =   [delta/2^(rsh-3)]. So, solve for offset...
	// when subtracting, we need to increase delta by (65535*scaleb)/2 to compensate
	// for the ~b we are not doing (we have gain<0 instead -- but this is already cooked
	// into delta_revscalar.
	if( rsh>3 ){
		delta = ((delta >> (rsh-4))+1) >> 1;
	}else{
		delta <<= (3-rsh);
	}
	int offset = delta - 32768*8 + gain;
	nn_do_scaleoff_16to16_hvx( op, inpb, 0x80008000, gain, offset,n );
}



static const struct elementwise_funcs AddSub_u16_funcs = {
	.op_stride_11 = addsub_u16_stride_11,
	.op_stride_10 = addsub_u16_stride_10,
	.op_rev_stride_01 = addsub_u16_rev_stride_01,

	.op_hvx_stride_11 = addsub_u16_stride_11_hvx,
	.op_hvx_stride_10 = addsub_u16_stride_10_hvx,
	.op_hvx_rev_stride_01 = addsub_u16_rev_stride_01_hvx,
	.minlen_hvx = 8,		// use scalar if inner-loop n < 8

	.in_elbytes = 2,
	.out_elbytes = 2,
	.out_typecode =  NN_TYPE_QUINT16
};
//--------------------------------------


static int
addsub_16_execute(struct nn_node *self,  struct nn_graph *nn )
{
	int node_type = self->node_type;
	int is_u16 = ( node_type == OP_QuantizedAdd_u16 || node_type == OP_QuantizedSub_u16 );
	int is_sub = ( node_type == OP_QuantizedSub_16 || node_type == OP_QuantizedSub_u16 );

	struct addsub_16_scaling_parms sc;

	if( set_addsub_scaling( nn,self, &sc,is_sub,is_u16)!= 0){
		return errlog(nn,"scaling failed");
	}
	tensor_set_single_float( self->outputs[1], sc.out_min);
	tensor_set_single_float( self->outputs[2], sc.out_max);

	struct elementwise_funcs const *ew_funcs = is_u16 ? &AddSub_u16_funcs : &AddSub_s16_funcs;
	return nn_elementwise_with_broadcast( self, nn, ew_funcs,NULL, NULL, &sc );
}
//////////////////////////////////// MULTIPLY //////////////////////////////////////
//
// signed multiply is done as:
//    (1) multiply s16*s16 -> s32
//    (2) scale by a 32-bit scale factor (31 fractional bits)
//    (3) saturate to i16.
//    The 32-bit quantity is found as
//         (a_qstep*b_qstep)/out_qstep * (1<<31)
//  We don't allow the scale to be > (1<<31); if it is, we enlarge the output range as needed.
//  In the case where the scale works out to  exactly 65536 (meaning 1/32768), we can use a 'fast path'
//  which uses the 16-bit 'fractional mul'.
//
///////////////////////
// unsigned 16 mul. Much messier.
// This could be done as
//     (a[i]-azero)*(b[i]-bzero) * (some scale) + out_zero
//   ... but this is problematic since the product needs 33 bits signed.
//
//   so we start with
//     tsum = (a[i]-azero)*(b[i]-bzero)+ range_bias
//
//   (using partial products, and modulo 2^32, without saturation; and range_bias is just
//    a precalculated value that ensures the whole range fits in i32)
//
//    and then
//        scaled = (tsum * mulfac )>>31
//        result = ( scaled  + postadd )>> 3   (saturated to u16)
//
//        so postadd =  outzero*8 - (range_bias*mulfac >> 31)
//
//    We need to ensure that postadd fits in i32, which means mulfac
//    must be less than (4095/4096); so the overall gain can't exceed
//    4095/32k. 'normal' overall gain is around 1/32k.
//
//    The 'tsum' is done as
//       tsum =  a[i]*b[i] - (a[i]*bzero + b[i]*azero)  + (azero*bzero+range_bias)
//    .. where all of the multiplies are u16*u16->u32, and all the adds are done without saturation.
//
//
// a method of finding range_bias:
//  range_bias =   - ( 2*azero-65535)*(2*bzero-65535)/2
//  (done as signed; this requires some attention, to avoid signed overflow).
//  Or use  - ((2*azero-65535)*(bzero-32768) + (azero-32768))
//   ... which doesn't overflow.
//
//
struct mul_16_scaling_parms {
	int32_t mulscale;			// the 32 bit scale
	// for unsigned
	uint16_t zeroa,zerob;
	uint16_t zero_out;
	uint32_t preadd;			// add this in the modulo add
	uint32_t postadd;			// add this in the 'average' step.

	float out_min;
	float out_max;
};

static int
set_mul_scaling( struct nn_graph *nn, struct nn_node * self, struct mul_16_scaling_parms * parms, int is_u16)
{
	 float tmp[6];
	 for( int i = 0; i < 6; i++){
		 tmp[i] = tensor_get_float( self->inputs[2+i],0);
	 }
	 float a_qstep = (tmp[1]-tmp[0]) * (float)(1./65536.);
	 float b_qstep = (tmp[3]-tmp[2]) * (float)(1./65536.);
	 float out_qstep = (tmp[5]-tmp[4]) * (float)(1./65536.);
	 int a_zero, b_zero, out_zero = 0x8000;
	 if (is_u16 ){
		 a_zero = saturate_u16( (int)(-tmp[0]/a_qstep + 0.5f));
		 b_zero = saturate_u16( (int)(-tmp[2]/b_qstep + 0.5f));
		 out_zero = saturate_u16( (int)(-tmp[4]/out_qstep + 0.5f));
	 }
	 parms->out_min = tmp[4];
	 parms->out_max = tmp[5];

	 float ab_step = a_qstep * b_qstep;
	 // compensate for shifts...
	 if( is_u16) ab_step *= 8.0f;
	 float max_scale = is_u16? (float)(4095./4096.): 1.0f;

	 float scale;
	 if( ab_step > out_qstep*max_scale){
		 out_qstep = ab_step/max_scale;
		 float tmp= out_zero;
		 parms->out_min = -out_qstep* tmp;
		 parms->out_max = out_qstep*(65536.0f-tmp);
		 logmsg(nn,0,"NOTE: expanding output range to %f .. %f so scaling is feasible\n",
				 parms->out_min, parms->out_max );
		 scale = max_scale;
	 }else{
		 scale = ab_step/out_qstep;
	 }
	 parms->mulscale = min_i32(0x7fffffff, roundf_u32( scale * (float)(1u<<31)));

	 if( is_u16){
		 parms->zeroa = a_zero;
		 parms->zerob = b_zero;
		 parms->zero_out = out_zero;
		 // u16 mode.
		 int range_bias = (32768-a_zero) - (2*a_zero-65535)*(b_zero-32768);
		 parms->preadd = (uint32_t)range_bias + (uint32_t)a_zero*(uint32_t)b_zero;

		 // scale range-bias using mul_scale
		 int rbscaled = ((int64_t)range_bias * parms->mulscale + (1<<30))>>31;
		 parms->postadd = 8*out_zero - rbscaled;
	 }
	 return 0;
}
// this is a i32*i32 -> i32 fractional mul with >>31, round
// and sat (to emulate the hvx ops).
// unfortunately no scalar equivalent; and this one is
// impaired by  QTOOL-39544 issue.
static inline int32_t
mul_frac_i32( int32_t a, int32_t b)
{
	int64_t p = Q6_P_mpy_RR(a,b);
	return Q6_R_sat_P( Q6_P_asrrnd_PI( p,31));
}
// i32*i16->i32 wirh >>16 and round. overflow not possible
static inline int32_t
mul_i32xi16_rsh16_rnd( int32_t a, int16_t b)
{
	int64_t p = Q6_P_mpy_RR(a,b);
	return Q6_P_asrrnd_PI( p,16);
}
// i32*i16->i32 wirh >>15 and round, saturate
static inline int32_t __attribute__((unused))
mul_i32xi16_rsh15_rnd_sat( int32_t a, int16_t b)
{
	int64_t p = Q6_P_mpy_RR(a,b);
	return Q6_R_sat_P(Q6_P_asrrnd_PI( p,15));
}


// s16 multiply vector * vector.

static inline int16_t muls16_single( int16_t a, int16_t b, int32_t mulscale)
{
	return Q6_R_sath_R( mul_frac_i32(mulscale,Q6_R_mpy_RlRl_sat(a,b)) );
}
static void mul_s16_stride_11( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	int16_t * op = (int16_t*)out;
	int16_t const * inpa = (int16_t const *)in1;
	int16_t const * inpb = (int16_t const *)in2;
	int mulscale = info->mulscale;

	if( n > 0){
		int16_t out = muls16_single( *inpa++, *inpb++, mulscale );
		for(int i = 0; i < n-1; i++){
			int16_t out1 = muls16_single( *inpa++, *inpb++, mulscale );
			*op++ = out;
			out= out1;
		}
		*op = out;
	}
}
//
// hvx version of mul_s16_stride_11
//
static
void mul_s16_stride_11_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	HVX_Vector * op = (HVX_Vector*)out;
	HVX_Vector const * inpa = (HVX_Vector const *)in1;
	HVX_Vector const * inpb = (HVX_Vector const *)in2;
	int mulscale = info->mulscale;
	HVX_Vector vmulscale = Q6_V_vsplat_R(mulscale);
	int nvec1 = (n-1)/64u;
	if( n > 0){
		HVX_Vector vina = q6op_V_vldu_A(inpa++);
		HVX_Vector vinb = q6op_V_vldu_A(inpb++);
		HVX_VectorPair prod =  Q6_Ww_vmpy_VhVh(vina,vinb);
		HVX_Vector prod0,prod1,vout;
		if( mulscale != 0x10000){
			for(int i = 0; i < nvec1; i++){
				prod0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_lo_W(prod),vmulscale);
				prod1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_hi_W(prod),vmulscale);
				vout = Q6_Vh_vsat_VwVw(prod1,prod0);
				vina = q6op_V_vldu_A(inpa++);
				vinb = q6op_V_vldu_A(inpb++);
				q6op_vstu_AV( op, vout); op++;
				prod =  Q6_Ww_vmpy_VhVh(vina,vinb);
			}
			prod0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_lo_W(prod),vmulscale);
			prod1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_hi_W(prod),vmulscale);
			vout = Q6_Vh_vsat_VwVw(prod1,prod0);
		}else{ // much easier when mulscale= 0x10000
			vout = Q6_Vh_vmpy_VhVh_s1_rnd_sat( vina,vinb);
			for(int i = 0; i < nvec1; i++){
				vina = q6op_V_vldu_A(inpa++);
				vinb = q6op_V_vldu_A(inpb++);
				q6op_vstu_AV( op, vout); op++;
				vout =  Q6_Vh_vmpy_VhVh_s1_rnd_sat(vina,vinb);
			}
		}
		hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
	}
}

//
// s16 vector * scalar.
// is also used for the 'reverse' case.
//
// the full expression is ( mulfac * (a*b)  )/2^31
// when b is fixed we can write it as
//             (  (mulfac*b)/2^16 * a )/2^31
// so find pfac = mulfac*b/2^16
// if
//   pfac=0  :   memset(0)
//   pfac = 32768:  memcpy
//   pfac fits in i16: use a 16-bit fractional mul
//   other cases, use a 32x16 mul with all the saturation.
//
static void mul_s16_stride_10( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	int16_t bval = *(int16_t const *)in2;
	int32_t pfac = mul_i32xi16_rsh16_rnd( info->mulscale, bval);
	if( n > 0 )nn_do_scale_s16( out, in1, pfac,n);
}


// HVX version.
// s16 vector * scalar.
// is also used for the 'reverse' case.
static void mul_s16_stride_10_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;

	int16_t bval = *(int16_t const *)in2;
	int32_t pfac = mul_i32xi16_rsh16_rnd( info->mulscale, bval);

	if( n > 0 )nn_do_scale_s16_hvx( out, in1, pfac,n);
}


static const struct elementwise_funcs Mul_s16_funcs = {
	.op_stride_11 = mul_s16_stride_11,
	.op_stride_10 = mul_s16_stride_10,
	.op_rev_stride_01 = mul_s16_stride_10,

	.op_hvx_stride_11 = mul_s16_stride_11_hvx,
	.op_hvx_stride_10 = mul_s16_stride_10_hvx,
	.op_hvx_rev_stride_01 = mul_s16_stride_10_hvx,
	.minlen_hvx = 8,		// use scalar if inner-loop n < 8

	.in_elbytes = 2,
	.out_elbytes = 2,
	.out_typecode =  NN_TYPE_QINT16
};

//////////// u16 multiply /////////////////////////

static inline int32_t mulu16_stage1( uint16_t a, uint16_t b, uint32_t zab, uint32_t preadd)
{
	uint32_t sum1 = Q6_R_mpyuacc_RlRl( preadd, a, b);  // preadd + a*b
	uint32_t sum2 = Q6_R_mpyu_RhRl( zab, a);
	sum2 = Q6_R_mpyuacc_RlRl(sum2, zab, b);				// a*zb + b*za
	return (int32_t)(sum1-sum2);
}
static inline uint16_t mulu16_stage2(int32_t stage1, int32_t postadd, int32_t mulscale)
{
	int32_t t = mul_frac_i32(mulscale, stage1);
	t = Q6_R_add_RR_sat( t, postadd);
	return Q6_R_satuh_R( Q6_R_asrrnd_RI(t,3));
}
/// vector * vector
static void mul_u16_stride_11( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	int16_t * op = (int16_t*)out;
	int16_t const * inpa = (int16_t const *)in1;
	int16_t const * inpb = (int16_t const *)in2;
	int mulscale = info->mulscale;
	uint32_t zab = Q6_R_combine_RlRl(info->zerob, info->zeroa);
	uint32_t preadd = info->preadd;
	int32_t postadd =info->postadd;

	if( n > 0){
		int32_t stage1 = mulu16_stage1( *inpa++, *inpb++, zab, preadd);
		for(int i = 0; i < n-1; i++){
			int16_t out = mulu16_stage2(stage1,postadd, mulscale);
			stage1 = mulu16_stage1( *inpa++, *inpb++, zab,preadd );
			*op++ = out;
		}
		*op = mulu16_stage2(stage1,postadd, mulscale);
	}
}
//
// HVX version of that
//
static inline HVX_VectorPair
mulu16_stage1_hvx( HVX_Vector va, HVX_Vector vb,
		uint32_t zaa, uint32_t zbb, HVX_Vector vpreadd)
{
	// all adds/sub done modulo u32!
	//    preadd + a*b - (a*zb + b*za)
	HVX_VectorPair sumn = Q6_Wuw_vmpy_VuhRuh( va,zbb);	// a * zb
	sumn = Q6_Wuw_vmpyacc_WuwVuhRuh(sumn, vb,zaa);	// + b * za
	HVX_VectorPair sump = Q6_Wuw_vmpy_VuhVuh( va,vb);
	sump = Q6_W_vcombine_VV(
			Q6_Vw_vadd_VwVw( Q6_V_hi_W(sump), vpreadd),
			Q6_Vw_vadd_VwVw( Q6_V_lo_W(sump), vpreadd));
	return Q6_Ww_vsub_WwWw( sump, sumn);
}
static inline HVX_Vector
mulu16_stage2_hvx(HVX_VectorPair stage1, HVX_Vector postadd, HVX_Vector mulscale)
{
	HVX_Vector t0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_lo_W(stage1), mulscale);
	HVX_Vector t1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( Q6_V_hi_W(stage1), mulscale);
	t0 = Q6_Vw_vadd_VwVw_sat( t0, postadd);
	t1 = Q6_Vw_vadd_VwVw_sat( t1, postadd);
#if __HEXAGON_ARCH__ < 62
	HVX_Vector t = Q6_Vh_vasr_VwVwR_rnd_sat( t1, t0, 3);
	return Q6_V_vxor_VV( t, Q6_V_vsplat_R(0x80008000));
#else
	return Q6_Vuh_vasr_VwVwR_rnd_sat( t1, t0, 3);
#endif
}

static void mul_u16_stride_11_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	HVX_Vector * op = (HVX_Vector*)out;
	HVX_Vector const * inp1 = (HVX_Vector const *)in1;
	HVX_Vector const * inp2 = (HVX_Vector const *)in2;
	int post_add = info->postadd;
#if __HEXAGON_ARCH__ < 62		// because we need to sat to int16 on v60
	post_add -= 0x8000 <<3;
#endif
	HVX_Vector preadd = Q6_V_vsplat_R( info->preadd);
	HVX_Vector postadd = Q6_V_vsplat_R( post_add);
	HVX_Vector mulscale = Q6_V_vsplat_R( info->mulscale);
	int zaa = Q6_R_combine_RlRl( info->zeroa, info->zeroa);
	int zbb = Q6_R_combine_RlRl( info->zerob, info->zerob);
	if ( n <= 0) return;
	int nvec1= (n-1)/64u;

	HVX_Vector vout;
	HVX_Vector vina = q6op_V_vldu_A( inp1);	inp1++;
	HVX_Vector vinb = q6op_V_vldu_A( inp2);	inp2++;
	HVX_VectorPair stg1 = mulu16_stage1_hvx( vina, vinb, zaa,zbb,preadd);
	for( int i =0; i < nvec1; i++){
		vout = mulu16_stage2_hvx( stg1, postadd, mulscale);
		vina = q6op_V_vldu_A( inp1);	inp1++;
		vinb = q6op_V_vldu_A( inp2);	inp2++;
		q6op_vstu_AV( op, vout); op++;
		stg1 = mulu16_stage1_hvx( vina, vinb, zaa,zbb,preadd);
	}
	vout = mulu16_stage2_hvx( stg1, postadd, mulscale);
	hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
}


/// vector * scalar core
//  The first mul is a*b+preadd - (a*bz+ b*az)
//  knowing that b is constant, we can change it to a*ka + preadd'
// (where ka doesn't fit in 16 bits).
//
static void
mul_u16_scalar( uint16_t *op, uint16_t const *inp, int ka,
		int n, struct mul_16_scaling_parms const * info,
		uint32_t preadd )
{
	int mulscale = info->mulscale;
	int32_t postadd = info->postadd;
	if( n > 0){
		if( ka == 0){
			uint16_t oz= info->zero_out;
			for(int i = 0; i<n; i++) op[i] = oz;
			return;
		}
		int32_t stage1 =  (uint32_t)(*inp++ * ka) + preadd;
		for(int i = 0; i < n-1; i++){
			int16_t out = mulu16_stage2(stage1,postadd, mulscale);
			stage1 =(uint32_t)(*inp++ * ka) + preadd;
			*op++ = out;
		}
		*op = mulu16_stage2(stage1,postadd, mulscale);
	}

}
/// vector + scalar
static void mul_u16_stride_10( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in1;
	uint16_t const * inpb = (uint16_t const *)in2;
	int za = info->zeroa;
	int zb = info->zerob;
	uint32_t preadd = info->preadd;
	uint16_t bcode = inpb[0];
	int bval = bcode-zb;
	preadd -= (unsigned)bcode * (unsigned)za;

	mul_u16_scalar( op, inpa, bval, n, info, preadd);
}
/// vector + scalar backwards
static void mul_u16_rev_stride_01( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in2;
	uint16_t const * inpb = (uint16_t const *)in1;
	int za = info->zeroa;
	int zb = info->zerob;
	uint32_t preadd = info->preadd;
	uint16_t acode = inpa[0];
	int aval = acode-za;
	preadd -= (unsigned)acode * (unsigned)zb;

	mul_u16_scalar( op, inpb, aval, n, info, preadd);
}

// HVX vector * scalar
// (Probably there's a better way to do this using nn_do_scaleoff_16to16_hvx)
static inline HVX_VectorPair
mulu16_stage1_scalar_hvx( HVX_Vector va,
		uint32_t kap, uint32_t kan, HVX_Vector vpreadd)
{
	// all adds/sub done modulo u32!
	//    preadd + a*kap - a*kan
	HVX_VectorPair sumn = Q6_Wuw_vmpy_VuhRuh( va,kan);	// a * zb
	HVX_VectorPair sump = Q6_Wuw_vmpy_VuhRuh( va,kap);
	sump = Q6_W_vcombine_VV(
			Q6_Vw_vadd_VwVw( Q6_V_hi_W(sump), vpreadd),
			Q6_Vw_vadd_VwVw( Q6_V_lo_W(sump), vpreadd));
	return Q6_Ww_vsub_WwWw( sump, sumn);
}
static void __attribute__((unused))
mul_u16_scalar_hvx( uint16_t *out, uint16_t const *in1, int ka,
		int n, struct mul_16_scaling_parms const * info,
		uint32_t preadd )
{
	if ( n <= 0) return;
	if( ka == 0 ){
		// 16 bit fill of 'zero_out'
		vmemset_16_2d_asm( out, info->zero_out, sizeof(uint16_t)*n, 1, 0);
		return;
	}
	HVX_Vector * op = (HVX_Vector*)out;
	HVX_Vector const * inp = (HVX_Vector const *)in1;
	int post_add = info->postadd;
#if __HEXAGON_ARCH__ < 62		// because we need to sat to int16 on v60
	post_add -= 0x8000 <<3;
#endif
	HVX_Vector vpreadd = Q6_V_vsplat_R( preadd);
	HVX_Vector postadd = Q6_V_vsplat_R( post_add);
	HVX_Vector mulscale = Q6_V_vsplat_R( info->mulscale);

	// ka is in range -65535 .. 65535
	// the first computation we do is in[i]*ka + preadd
	//  which is u16*i16 + i32 (modulo u32)
	// to handle ka < 0, we do it as
	//          in[i]*kap - (in[i]*kan) + preadd
	//  where kap, kan, are both u16 and kap-kan = ka.
	//

	int kap = max_i32(ka,0);
	int kan = max_i32(-ka,0);

	kap = Q6_R_combine_RlRl( kap, kap);
	kan = Q6_R_combine_RlRl( kan, kan);
	int nvec1= (n-1)/64u;

	HVX_Vector vout;
	HVX_Vector vina = q6op_V_vldu_A( inp);	inp++;
	HVX_VectorPair stg1 = mulu16_stage1_scalar_hvx( vina, kap,kan,vpreadd);
	for( int i =0; i < nvec1; i++){
		vout = mulu16_stage2_hvx( stg1, postadd, mulscale);
		vina = q6op_V_vldu_A(inp);	inp++;
		q6op_vstu_AV( op, vout); op++;
		stg1 = mulu16_stage1_scalar_hvx( vina, kap,kan,vpreadd);
	}
	vout = mulu16_stage2_hvx( stg1, postadd, mulscale);
	hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
}
/// vector + scalar
static void mul_u16_stride_10_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in1;
	uint16_t const * inpb = (uint16_t const *)in2;
	int za = info->zeroa;
	int zb = info->zerob;
	uint32_t preadd = info->preadd;
	uint16_t bcode = inpb[0];
	int bval = bcode-zb;
	preadd -= (unsigned)bcode * (unsigned)za;

	mul_u16_scalar_hvx( op, inpa, bval, n, info, preadd);
}
/// vector + scalar backwards
static void mul_u16_rev_stride_01_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_16_scaling_parms const * info = (struct mul_16_scaling_parms const*)infov;
	uint16_t * op = (uint16_t*)out;
	uint16_t const * inpa = (uint16_t const *)in2;
	uint16_t const * inpb = (uint16_t const *)in1;
	int za = info->zeroa;
	int zb = info->zerob;
	uint32_t preadd = info->preadd;
	uint16_t acode = inpa[0];
	int aval = acode-za;
	preadd -= (unsigned)acode * (unsigned)zb;

	mul_u16_scalar_hvx( op, inpb, aval, n, info, preadd);
}



static const struct elementwise_funcs Mul_u16_funcs = {
	.op_stride_11 = mul_u16_stride_11,
	.op_stride_10 = mul_u16_stride_10,
	.op_rev_stride_01 = mul_u16_rev_stride_01,

	.op_hvx_stride_11 = mul_u16_stride_11_hvx,
	.op_hvx_stride_10 = mul_u16_stride_10_hvx,
	.op_hvx_rev_stride_01 = mul_u16_rev_stride_01_hvx,
	.minlen_hvx = 8,		// use scalar if inner-loop n < 8


	.in_elbytes = 2,
	.out_elbytes = 2,
	.out_typecode =  NN_TYPE_QUINT16
};



static int
mul_16_execute(struct nn_node *self,  struct nn_graph *nn )
{
	int node_type = self->node_type;
	int is_u16 = ( node_type == OP_QuantizedMul_u16  );

	struct mul_16_scaling_parms sc;

	if( set_mul_scaling( nn,self, &sc, is_u16)!= 0){
		return errlog(nn,"scaling failed");
	}
	tensor_set_single_float( self->outputs[1], sc.out_min);
	tensor_set_single_float( self->outputs[2], sc.out_max);

	struct elementwise_funcs const *ew_funcs = is_u16 ? &Mul_u16_funcs : &Mul_s16_funcs;
	return nn_elementwise_with_broadcast( self, nn, ew_funcs,NULL, NULL, &sc );
}



struct nn_node_ops nn_ops_for_QuantizedAdd_16 = {
	.execute = addsub_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedSub_16 = {
	.execute = addsub_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_QuantizedMul_16 = {
	.execute = mul_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedAdd_u16 = {
	.execute = addsub_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedSub_u16 = {
	.execute = addsub_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_QuantizedMul_u16 = {
	.execute = mul_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(8),
	.n_outputs = NN_IOCOUNT(3),
};

