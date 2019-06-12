
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
 * This contains misc ops on 16-bit tensors which can be shared across nodes.
 */

#include "nn_graph.h"
#include "quantize.h"
#include "hvx_inlines.h"
#include "nn_oper16.h"

// workaround for  QTOOL-39544 issue: empty 'asm' to make the the compiler
// think that 'gain' changes per loop iteration.
#if defined(__hexagon__)
#define FAKEDEP_RR(var1,var2) 	   asm ( "/* %0 %2 */": "=r"(gain):"0"(var1), "r"(var2));
#else
#define FAKEDEP_RR(var1,var2)
#endif

// i32*i16->i32 wirh >>15 and round, saturate
static inline int32_t
mul_i32xi16_rsh15_rnd_sat( int32_t a, int16_t b)
{
	int64_t p = Q6_P_mpy_RR(a,b);
	return Q6_R_sat_P(Q6_P_asrrnd_PI( p,15));
}
//
// this applies a given 32-bit gain (with 16 fractional bits) to an s16 array;
// the operation is
//   out[i] =  saturate_i16(  round( inp[i] * gain /65536) )
//
// There are special cases for gain = 0x10000 (copy), gain = 0 (fill 0)
// and gain-fits-in-i16 (faster path).
// The output buffer may be identical to the input, but otherwise may not
// overlap. input, output pointers must be 16-bit aligned.
//
//
void
nn_do_scale_s16( int16_t * outp, int16_t const * inp, int32_t gain, int n)
{

	if ( n <= 0) return;
	if( likely( saturate_i16(gain)== gain)){ 	// fits in i16
		if( gain==0){		// b=0 or close enough
			memset(outp,0,n*sizeof(int16_t));
			return;
		}
		int npair1 = (n-1)/2u;
		int gain2 = Q6_R_combine_RlRl(gain,gain);
		uint32_t vals = Q6_R_combine_RlRl(inp[1],inp[0]);
		uint64_t prod = Q6_P_vmpyh_RR_sat(vals,gain2);
		inp += 2;
		for( int i =0; i < npair1; i++ ){
			vals = Q6_R_combine_RlRl(inp[1],inp[0]);
			inp += 2;
			outp[0] = Q6_R_sath_R(Q6_R_asrrnd_RI((int32_t)prod,15));
			outp[1] = Q6_R_sath_R(Q6_R_asrrnd_RI((int32_t)(prod>>32),15));
			outp += 2;
			prod = Q6_P_vmpyh_RR_sat(vals,gain2);
		}
		// 1 or 2 left at the end.
		outp[0] = Q6_R_sath_R(Q6_R_asrrnd_RI((int32_t)prod,15));
		if( !(n&1)){
			outp[1] = Q6_R_sath_R(Q6_R_asrrnd_RI((int32_t)(prod>>32),15));
		}
	}else{	// value doesn't fit in i16
		if( gain == 0x8000){
			if( inp != outp)
			memcpy( outp, inp, n*sizeof(int16_t));
			return;
		}
		int32_t outval = mul_i32xi16_rsh15_rnd_sat( gain,*inp++);
		for( int i =0; i < n-1; i++ ){
			FAKEDEP_RR( gain, inp);
			int32_t outval1 = mul_i32xi16_rsh15_rnd_sat( gain,*inp++);
			*outp ++ = Q6_R_sath_R( outval);
			outval = outval1;
		}
		*outp = Q6_R_sath_R( outval);
	}
}

// HVX version. inp, outp need not be vector aligned.
void
nn_do_scale_s16_hvx( int16_t * outp, int16_t const * inp, int32_t gain, int n)
{
	HVX_Vector * op = (HVX_Vector*)outp;
	HVX_Vector const * inpa = (HVX_Vector const *)inp;
	if ( n <= 0) return;
	int nvec1 = (n-1)/64u;
	HVX_Vector vout;
	if( likely( saturate_i16(gain)== gain)){ 	// fits in i16
		if( gain==0){		// b=0 or close enough
			vmemset_asm(op,0,n*sizeof(int16_t));
			return;
		}
		int gain2 = Q6_R_combine_RlRl(gain,gain);
		HVX_Vector vin = q6op_V_vldu_A( inpa);	inpa ++;
		vout = Q6_Vh_vmpy_VhRh_s1_rnd_sat( vin, gain2);
		for( int i =0; i < nvec1; i++ ){
			vin = q6op_V_vldu_A( inpa);	inpa ++;
			q6op_vstu_AV( op, vout); op ++;
			vout = Q6_Vh_vmpy_VhRh_s1_rnd_sat( vin, gain2);
		}
	}else{	// value doesn't fit in i16
		if( gain == 0x8000){
			if( inpa != op)
				vmemcpy_asm( op, inpa, n*sizeof(int16_t));
			return;
		}
		HVX_Vector vmul = Q6_V_vsplat_R(gain);
		HVX_Vector vin = q6op_V_vldu_A( inpa);	inpa ++;
		HVX_Vector vx1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vmul, vin);
		HVX_Vector vx0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vmul, Q6_Vh_vshuffe_VhVh(vin,vin));
		for( int i =0; i < nvec1; i++ ){
			vout = Q6_Vh_vsat_VwVw( vx1,vx0);
			vin = q6op_V_vldu_A( inpa);	inpa ++;
			q6op_vstu_AV( op, vout); op ++;
			vx1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vmul, vin);
			vx0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vmul, Q6_Vh_vshuffe_VhVh(vin,vin ));
		}
		vout = Q6_Vh_vsat_VwVw( vx1,vx0);
	}
	hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
}


//
// This does scale and/or offset of u16 tensors, and can also be used to convert between u16/s16 ranges.
//
//    parameters:
//         uint32    inout_offset:   LS word must be 0x8000 if input is u16,  0 if input is s16
//                                   MS word must be 0x8000 if output is u16,  0 if output is s16
//        int32_t gain	   :   a signed 32-bit gain value
//        int32_t offset   :   a signed 32-bit offset term
//
// The operation is equivalent to the below (assuming saturation does not occur):
// For  u16->u16 (inout_offset = 0x80008000):
//     out[i] = (gain * 2^-18) * in[i] + [  (offset-gain)*2^-3 + 32768]
// For u16->s16 (inout_offs = 0x8000):
//     out[i] = (gain * 2^-18) * in[i] + [  (offset-gain)*2^-3]
// For s16->u16 ( (inout_offs = 0x80000000:)
//     out[i] = (gain * 2^-18) * in[i] + [  offset*2^-3 + 32768]
// For s16->s16 ( inout_offs = 0:)
//     out[i] = (gain * 2^-18) * in[i] + [  offset*2^-3]

//
//  operation is done as:
// in_offset = (uint16_t)inout_offset
// out_offset = (uint16_t)(inout_offset>>16)
//
//       (1) int16 intmp = in[i]-in_offset			// input in -0x8000 .. 0x7fff range
//       (2) int32 prod = round_sati32( gain* intmp >> 15)
//       (3) prod =  add_sat( prod, offset);
//       (4) int16 out = round_sati16( prod >> 3 );
//       (5) out[i] = out + out_offset.
// Note that step 2 will not 'saturate' expect in the corner case gain = -0x80000000, intmp = -0x8000
//
void
nn_do_scaleoff_16to16( uint16_t * outp, uint16_t const * inp, uint32_t inout_offs, int32_t gain, int32_t offset, int n)
{
	uint32_t in_offset = Q6_R_combine_RlRl(inout_offs, inout_offs);
	uint32_t out_offset = Q6_R_combine_RhRh(inout_offs, inout_offs);
	if( n <= 0) return;
	uint64_t offsoffs = Q6_P_combine_RR(offset+4,offset+4);		// put in rounding bias for >>3

	int npair1 = (n-1)/2u;
	int32_t vals = Q6_R_combine_RlRl(inp[1],inp[0]) ^ in_offset;
	int32_t p0 = mul_i32xi16_rsh15_rnd_sat( gain, (int16_t)vals);
	int32_t p1 = mul_i32xi16_rsh15_rnd_sat( gain, (int16_t)(vals>>16));
	inp += 2;
	for( int i =0; i < npair1; i++ ){
		FAKEDEP_RR( gain, outp);
		uint64_t offs_sum = Q6_P_vaddw_PP_sat( Q6_P_combine_RR(p1,p0),offsoffs);
		offs_sum = Q6_P_vasrw_PI( offs_sum, 3);		// >> 3
		uint32_t out_hh = Q6_R_vsatwh_P(offs_sum)^out_offset;
		vals = Q6_R_combine_RlRl(inp[1],inp[0]) ^ in_offset;
		inp +=2;
		outp[0] = (uint16_t)out_hh;
		outp[1] = (uint16_t)(out_hh>>16);
		outp += 2;
		p0 = mul_i32xi16_rsh15_rnd_sat( gain, (int16_t)vals);
		p1 = mul_i32xi16_rsh15_rnd_sat( gain, (int16_t)(vals>>16));
	}
	// 1 or 2 more to do...
	uint64_t offs_sum = Q6_P_vaddw_PP_sat( Q6_P_combine_RR(p1,p0),offsoffs);
	offs_sum = Q6_P_vasrw_PI( offs_sum, 3);		// >> 3
	uint32_t out_hh = Q6_R_vsatwh_P(offs_sum)^out_offset;
	outp[0] = (uint16_t)out_hh;
	if( (n&1)==0)
		outp[1] = (uint16_t)(out_hh>>16);
}

//
// hvx version
// much less clumsy...

void nn_do_scaleoff_16to16_hvx( uint16_t * outp, uint16_t const * inp, uint32_t inout_offs, int32_t gain, int32_t offset, int n)
{

	HVX_Vector * op = (HVX_Vector*)outp;
	HVX_Vector const * inpa = (HVX_Vector const *)inp;
	if ( n <= 0) return;
	int nvec1 = (n-1)/64u;
	HVX_Vector vout;
	HVX_Vector voffset = Q6_V_vsplat_R(offset);
	HVX_Vector vgain = Q6_V_vsplat_R(gain);
	HVX_Vector in_xor = Q6_V_vsplat_R(inout_offs);
	HVX_Vector out_xor = Q6_Vh_vshuffo_VhVh(in_xor,in_xor);	// odd lanes -> out xor
	in_xor = Q6_Vh_vshuffe_VhVh(in_xor,in_xor);	// even lanes -> in xor

	HVX_Vector vin = q6op_V_vldu_A( inpa);	inpa ++;
	vin = Q6_V_vxor_VV( vin, in_xor);
	HVX_Vector prod1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vgain, vin);
	HVX_Vector prod0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vgain, Q6_Vh_vshuffe_VhVh(vin,vin));
	for( int  i=0; i < nvec1; i++){
		prod0 = Q6_Vw_vadd_VwVw_sat( prod0, voffset);
		prod1 = Q6_Vw_vadd_VwVw_sat( prod1, voffset);
		vout = Q6_V_vxor_VV( Q6_Vh_vasr_VwVwR_rnd_sat( prod1, prod0, 3), out_xor);
		vin = q6op_V_vldu_A( inpa);	inpa ++;
		vin = Q6_V_vxor_VV( vin, in_xor);
		q6op_vstu_AV( op, vout); op ++;
		prod1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vgain, vin);
		prod0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( vgain, Q6_Vh_vshuffe_VhVh(vin,vin));
	}
	prod0 = Q6_Vw_vadd_VwVw_sat( prod0, voffset);
	prod1 = Q6_Vw_vadd_VwVw_sat( prod1, voffset);
	vout = Q6_V_vxor_VV( Q6_Vh_vasr_VwVwR_rnd_sat( prod1, prod0, 3), out_xor);
	hvx_store_vec_x2_unaligned( op, vout, vout, n*sizeof(int16_t)-nvec1*sizeof(HVX_Vector));
}
