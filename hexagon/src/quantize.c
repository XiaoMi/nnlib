
/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
#include <quantize.h>
#include "hvx_inlines.h"

#define CLOSE_ENUF(X, Y) (fabsf(X - Y) < 6.1035156e-05f)
//
// This adjusts a min .. max range
// (by decreasing min, or increasing max) until
// the 'zero point' (the value which encodes zero)
// is an exact integer.
//  if 0 encodes 'mn', and 255 encodes 'mx', then zero is encoded by
//     z = -255*mn/(mx-mn)
//
//
// If this is not an integer, we want to either adjust mx up (to mx+b)
//  or mn down (to mn-a), in order to make z an integer, using
//  the smallest possible adjustment.
//
// (we require mn < mx, and discussion below assumes mn <= 0, mx > 0).
//
// Sensitivity of z to adjustments a,b:
//   dz/da = -dz/d(mn) = 255*mx/(mx-mn)^2        (this is > 0)
//   dz/db =  dz/d(mx) = 255*mn/(mx-mn)^2		 (this is <=0)
// (note that these are in the ratio mx:mn)
//
// if z = zi + zf (integer and fraction), we have the choice
//   of (a) increasing it by (1-zf), using a ~=~ (1-zf)/(dz/da)
//   or (b) decreasing it by zf,  by b ~=~ -zf / (dz/db)
//
//  to pick the one which chooses the smallest move:
//     - choose (b) if the mag of (1-zf)/mx  is more than mag of -zf/mn;   else (a)
//  ie - choose (b) if f (zf-1)*mn is more  than zf*mx;   else (a).
//
// in cases where zi <= 0, or zi >=254, we choose (a) or (b) respectively, thus
// avoiding division by 0 when calculating the new endpoint;
// this also means that cases (0 < mn < mx) and (mn < mx < 0)
// are handled properly (in these cases, zi will be <=0 or >=255 resp, and
// the adjustment is always made to the endpoint which is closest to 0).
//
//
// return value:
//  -1 : test failed for max > min (includes either being NaN)
//   0 : ok, no adjustment made; zero point already an integer +/- 2^-14 (8b) or 2^-6 (16b)
//   1 : adjusted min downward
//   2 : adjusted max upward
//
// note: if you want min <=0 and max >=0, enforce that before calling
// (and if you decrease min to 0.0, you don't need to call this since
// the condition is met by that).
//

int adjust_minmax_for_zero( float *min_p, float *max_p )
{
	float mn = *min_p;
	float mx = *max_p;
	float dif = mx-mn;
	if( !(dif >= 1e-6f)) return -1;	// check valid min,max_p
	if( mn == 0.0f) return 0;		// common case
	float z = (-255.0f)*mn / dif;		// current 'zero point'
	float zi = floorf(z);
	float zf = z - zi;
	// if within 2^-14 of an integer, call it close enough
	if( zf <= 6.1035156e-05f || zf >= 0.999938965f)
		return 0;
	// choose which end to move
	// if zi <= 0  or >= 254, the decision is based on that only (to
	// avoid divide by 0) otherwise choose based on zf.
	//
	if( zi > 0.0f && ( zi > 253.0f || (zf-1.0f)*mn >= zf*mx )) {
		// move max, change z to zi
		*max_p = mn - 255.0f*mn/zi;
		return 2;
	}else{
		// move min; change z to zi+1
		*min_p = mx*(zi+1.0f)/(zi-254.0f);
		return 1;
	}
}

int adjust_minmax_for_zero_16b(float *min_p, float *max_p)
{
	float mn = *min_p;
	float mx = *max_p;
	float dif = mx - mn;
	if (!(dif >= 1e-6f)) return -1;	// check valid min,max_p
	if (mn == 0.0f) return 0;		// common case
	float z = (-65536.0f)*mn / dif;		// current 'zero point'
	float zi = floorf(z);
	float zf = z - zi;
	// if within 2^-6 of an integer, call it close enough
	// (don't accept z outside 0.. 65535 though)
	if ((zf <= 0.015625f || zf >= 0.984375f ) && z >=-0.02f && z <= 65535.02f)
		return 0;
	// choose which end to move
	// if zi <= 0  or >= 65535, the decision is based on that only (to
	// avoid divide by 0) otherwise choose based on zf.
	//
	if (zi > 0.0f && (zi > 65534.0f || (zf - 1.0f)*mn >= zf * mx)) {
		// move max, change z to zi	(and zi->65535 if greater)
		zi = fminf( zi, 65535.0f);
		*max_p = mn - 65536.0f*mn / zi;
		return 2;
	}
	else {
		// move min; change z to zi+1
		zi = fmaxf(zi+1.0,1.0f);
		*min_p = mx * zi / (zi - 65536.0f);
		return 1;
	}
}

int adjust_minmax_for_zero_with_constraints_16b( float *min_p, float *max_p , int constraint )
{
        float mn = *min_p;
        float mx = *max_p;
        float dif = mx-mn;
        if( !(dif >= 1e-6f)) return -1; // check valid min,max_p
        if( mn == 0.0f) return 0;               // common case
        float z = (-65536.f)*mn / dif;          // current 'zero point'
        float zi = floorf(z);
        float zf = z - zi;
        // if within 2^-6 of an integer, call it close enough
        // (don't accept z outside 0.. 65535 though)
        if ((zf <= 0.015625f || zf >= 0.984375f ) && z >=-0.02f && z <= 65535.02f)
                return 0;
        // choose which end to move
        // Avoid divide by 0.
        //

        float mnk = (zf-1.0f) * mn;
        float mxk =  zf *mx;
        float zirnd = (zf >=0.5f)? (zi+1.0f): zi;

        switch( constraint &3 ){
         default:
         case 0:
                // move whichever requires least change.
                if( zi >= 1.0f && ( zi >= 65535.0f || mnk >= mxk ))
                        goto move_max_out;
                goto move_min_out;
         case 1:                // min is fixed; max is free
                // skew decision to 'move max out'
                if( zi >= 1.0f && (z >= 49152.25f ||  mnk * 8.0f > mxk) )
                        goto move_max_out;
                goto move_min_nearest;
         case 2:
                // skew decision to 'move min out'
                if( z >= 16383.75f && ( zi >= 65535.0f || mnk > mxk * 8.0f ))
                        goto move_max_nearest;
                goto move_min_out;
         case 3:
                // move single endpoint if range is skewed at least 3:1; otherwise both.
                if( z <= 16383.75f ) goto move_min_nearest;
                if( z >= 49152.25f) goto move_max_nearest;
                // shift zero to zirnd, by moving both ends.
                float adj = (z-zirnd)*dif * (float)(1./65536.);
                *min_p = mn + adj;
                *max_p = mx + adj;
                return 3;
        }
        /* NOTREACHED*/

   move_max_nearest:   // move max, change z to nearest
     zi = zirnd;
   move_max_out:                // move max, change z to zi
        *max_p = mn - 65536.0f*mn/zi;
        return 2;

   move_min_out:   // move min; change z to zi+1
         zirnd = zi+1.0f;
   move_min_nearest:    // move min, change z to nearest
        *min_p = mx*zirnd/(zirnd-65536.0f);
         printf(" djusted min max = %f %f \n",*min_p, *max_p);
        return 1;
}



//
// This is like adjust_minmax_for_zero except that it accepts a 'constraint'
// parameter which indicates which endpoints are (nominally) fixed:
//
//  constraint =
//       0          same as adjust_minmax_for_zero
//       1          'min' is fixed, max is free
//       2          'max' is fixed; min is free
//       3          both are fixed.
//
// If only one endpoint is fixed:
//     (1) decide which endpoint is to be moved - as per adjust_minmax_for_zero
//         but skew the decision towards moving the 'free' endpoint outward (increasing
//         the range).  So, this is what will usually happen 
//      (2)If the decision nonetheless falls to move the 'fixed' endpoint,
//         it will be moved, but it will be moved as little as possible (moving inward
//         if needed). 
// If both endpoints are 'fixed': We adjust both endpoints by the same amount;
// unless the range is skewed by 3:1 or more
// in which case we adjust the endpoint which is closest to 0.
//
// An example of a case where the skewed decision still goes to the 'fixed' end:
//   fixed min:
//    min = -1.3    max = 10.6    => z = 27.85714
// In this case, it will adjust min to -1.3074 (for z = 28)
//  rather than changing max to 10.9778 (to get z = 29)
//

int adjust_minmax_for_zero_with_constraints( float *min_p, float *max_p , int constraint )
{
	float mn = *min_p;
	float mx = *max_p;
	float dif = mx-mn;
	if(!(dif > 0.0f)) return -1;
	if( mn == 0.0f) return 0;		// common case
	float z = (-255.0f)*mn / dif;		// current 'zero point'
	float zi = floorf(z);
	float zf = z - zi;
	// if within 2^-14 of an integer, call it close enough
	if( zf <= 6.1035156e-05f || zf >= 0.999938965f)
		return 0;
	// choose which end to move
	// Avoid divide by 0.
	//
	
	float mnk = (zf-1.0f) * mn;
	float mxk =  zf *mx;
	float zirnd = (zf >=0.5f)? (zi+1.0f): zi;
	
	switch( constraint &3 ){
	 default:
	 case 0:
		// move whichever requires least change.
		if( zi >= 1.0f && ( zi >= 254.0f || mnk >= mxk ))
			goto move_max_out;
		goto move_min_out;
	 case 1:		// min is fixed; max is free
		// skew decision to 'move max out'
		if( zi >= 1.0f && (z >= 192.25f ||  mnk * 8.0f > mxk) ) 
			goto move_max_out;
		goto move_min_nearest;
	 case 2:
		// skew decision to 'move min out'
		if( z >= 63.75f && ( zi >= 254.0f || mnk > mxk * 8.0f )) 
			goto move_max_nearest;
		goto move_min_out;
	 case 3:
		// move single endpoint if range is skewed at least 3:1; otherwise both.
		if( z <= 63.75f ) goto move_min_nearest;
		if( z >= 192.25f) goto move_max_nearest; 
		// shift zero to zirnd, by moving both ends.
		float adj = (z-zirnd)*dif * (float)(1./255.);
		*min_p = mn + adj;
		*max_p = mx + adj;
		return 3;
	}
	/* NOTREACHED*/
	
   move_max_nearest:   // move max, change z to nearest
     zi = zirnd;
   move_max_out:  		// move max, change z to zi
	*max_p = mn - 255.0f*mn/zi;
	return 2;

   move_min_out:   // move min; change z to zi+1
	 zirnd = zi+1.0f;
   move_min_nearest:  	// move min, change z to nearest
	*min_p = mx*zirnd/(zirnd-255.0f);
	return 1;
}

int quantize_adjust_range_and_check( float *out_min, float *out_max, float *out_stepsize, float *out_recip_stepsize, float in_min, float in_max)
{
	int k = check_range_is_sane( in_min, in_max);
	if( k == 0)
		quantize_adjust_range( out_min,out_max, out_stepsize, out_recip_stepsize, in_min, in_max);
	return k;
}

///////////////////////////////////////
static void scalar_requantize_i32_to_qu8(int32_t const * inp, int offseto, int gaini, uint8_t *outp, int n);
static void fallback_requantize_i32_to_qu8(int32_t const * inp, float offset, float gain, uint8_t *outp, int n);
inline HVX_Vector __attribute__((always_inline))
hvx_requantize_vector_8to8(HVX_Vector const* vinp, int32_t gaini, HVX_VectorPair vvin_off_i16, HVX_Vector vout_off_i16);
inline HVX_Vector __attribute__((always_inline))
hvx_requantize_vector_8to8_hi_gain(HVX_Vector const* vinp, int32_t gaini, int32_t prod_rsh, HVX_VectorPair vvin_off_i16, HVX_Vector vout_off_i16);
inline HVX_Vector __attribute__((always_inline))
hvx_requantize_vector_8to8_uniform_gain(HVX_Vector const* vinp, HVX_VectorPair vvtot_off_i16);

//
// requantize n 32-bit numbers to u8; equiv to
//     outp[i] =  quantize_uint8( in_level_size* (float)inp[i],out_min,out_max);
//
// This uses HVX for larger n, and so it must be called from a vector thread.
// if n < 128, or if the gain is very small, or output unaligned, it uses hvx_scale_32_to_8 instead.
// Note: if n is not a multiple of 128, it may fill output to a vector boundary. FIXME
//    (this occurs when quantize_asm is called).
//
// if input pointer is not aligned, or if gain >= 1.0,  it will use scalar operations which are equivalent.
//
void
nn_requantize_i32_to_qu8_hvx( uint8_t *outp, int32_t const * inp, int n, float in_level_size, float out_min, float out_max)
{

	// we want to do:  ( x * in_levelsize- out_min)*255/(out_max-out_min)
	//  = (  x - offs) * alpha
	//  where alpha = in_level_size * 255/(out_max-out_min)
	//       offs = out_min/in_level_size
	float out_scale = 255.0f / (out_max-out_min);
	float gain = in_level_size * out_scale;
	float offset_f = 0.5f - out_min*out_scale;

	if( gain >= 1.0f){
		fallback_requantize_i32_to_qu8(inp, offset_f, gain, outp,  n);
		return;
	}
	int gaini  = roundf_i32( gain* (float)(1u<<31) );

	if( (((size_t)inp)&127)!= 0){
		// input not aligned!
		int offseto = (int)offset_f;
		scalar_requantize_i32_to_qu8(inp, offseto, gaini, outp, n);
	}else{
		if( n>=128 && gaini >= 1024 && ((size_t)outp&127)==0){			// ok to use quantize_asm
			float offset_f = out_min / in_level_size;
			int offseti = roundf_i32(offset_f);
			quantize_asm(inp, offseti, gaini, outp, n);
		}else{
			int offseto = (int)offset_f;
			hvx_quantize_32_to_8(inp, offseto, gaini,outp,n);
		}
	}
}

//
// non-hvx version of nn_requantize_i32_to_qu8
void
nn_requantize_i32_to_qu8( uint8_t *outp, int32_t const * inp, int n, float in_level_size, float out_min, float out_max)
{
	float out_scale = 255.0f / (out_max-out_min);
	float gain = in_level_size * out_scale;
	float offset_f = 0.5f - out_min*out_scale;

	if( gain >= 1.0f){
		fallback_requantize_i32_to_qu8(inp, offset_f, gain, outp,  n);
		return;
	}
	int gaini  = roundf_i32( gain * (float)(1u<<31) );
	int offseto = (int)offset_f;
	scalar_requantize_i32_to_qu8( inp, offseto, gaini, outp, n);
}
static void
scalar_requantize_i32_to_qu8(int32_t const * inp, int offseto, int gaini, uint8_t *outp, int n)
{
	for( int i =0; i < n; i++){
		int64_t p =(int64_t)inp[i]*gaini;
		int32_t scaled = Q6_P_asrrnd_PI(p,31);
		scaled = Q6_R_add_RR_sat(scaled,offseto);
		outp[i] = saturate_u8(scaled);
	}
}
// when the i32->u8 conversion requires a gain > 1.0, this is used.
// Note that 'offset' is presumed to contain a +0.5 rounding bias.
//
static void fallback_requantize_i32_to_qu8(int32_t const * inp, float offset, float gain, uint8_t *outp, int n)
{
	for( int i= 0; i < n; i++){
		float x = inp[i] * gain + offset;
		outp[i] = saturate_u8( (int)x);
	}
}



//
// This is like quantize_asm but with a different order of operations; the offset
// is added after, instead of being subtracted before, so we can avoid overflow problems
// for small gain.
// The output pointer can have any alignment, and only 'n' bytes are  stored.
//
// The operation is:
//    out[i] = saturate_u8(
//           add32_sat(  offset,
//                 (in[i] * gain)/(2^31)
//           )
//    )
//
void
hvx_quantize_32_to_8(
		int32_t const * inp,		// input [n], vector aligned
		int32_t offset,				// offset to add
		int32_t scale,				// scale with 31 fractional bits
		uint8_t * outp,				// output [n], no alignment constraint
		int n )						// # elements
{
	int nloop = n/128u;				// # of full output vectors

	HVX_Vector vscale = Q6_V_vsplat_R(scale);
	HVX_Vector voffs = q6op_Vh_vsplat_R(saturate_i16(offset));

	HVX_Vector const * vinp = (HVX_Vector const*)inp;

	for( int i = 0; i < nloop; i++ ){
		HVX_Vector x0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[0], vscale);
		HVX_Vector x1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[1], vscale);
		HVX_Vector x2 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[2], vscale);
		HVX_Vector x3 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[3], vscale);
		HVX_Vector p01 = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vpack_VwVw_sat( x1,x0), voffs );
		HVX_Vector p23 = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vpack_VwVw_sat( x3,x2), voffs );
		q6op_vstu_AV( (HVX_Vector*)outp,  Q6_Vub_vpack_VhVh_sat(p23,p01));
		vinp += 4;
		outp += 128;
	}
	unsigned nextra = n&127;
	if( nextra >0 ){
		unsigned nv = (nextra+31)/32u;	// 1..4
		HVX_Vector p01,p23;
		if( nv >= 3){		// get first 2
			HVX_Vector x0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[0], vscale);
			HVX_Vector x1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[1], vscale);
			p01 = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vpack_VwVw_sat( x1,x0), voffs );
			vinp += 2;
		}
		HVX_Vector x2  = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[0], vscale);
		HVX_Vector x3 = Q6_V_vzero();
		if( (nv&1)== 0 )
			x3  = q6op_Vw_vmpy_VwVw_s1_rnd_sat( vinp[1], vscale);
		p23 = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vpack_VwVw_sat( x3,x2), voffs );
		if( nv < 3) p01 = p23;
		HVX_Vector result = Q6_Vub_vpack_VhVh_sat(p23,p01);

		q6op_vstu_variable_ARV( outp, nextra, result );
	}
}

// Requantizes 8-bit input to a new quantization range. Supports any gain value 
// with magnitude less than 2^15. Order of operations is:
// output = (gain) * (input - in_offset) + out_offset
void nn_requantize_qu8_to_qu8_hvx(uint8_t *outp, uint8_t const* inp, unsigned n, float gain, int32_t in_offset, int32_t out_offset)
{
    int nloop = n / 128u;
    int nextra = n % 128;

    HVX_Vector vin_off_i16 = q6op_Vh_vsplat_R(saturate_i16(in_offset));
    HVX_VectorPair vvin_off_i16 = Q6_W_vcombine_VV(vin_off_i16, vin_off_i16);
    HVX_Vector vout_off_i16 = q6op_Vh_vsplat_R(saturate_i16(out_offset));

    HVX_Vector const *vinp = (HVX_Vector const*) inp;

    if(CLOSE_ENUF(gain, 1.0f))
    {
        HVX_Vector vtot_off_i16 = Q6_Vh_vsub_VhVh_sat(vout_off_i16, vin_off_i16);
        HVX_VectorPair vvtot_off_i16 = Q6_W_vcombine_VV(vtot_off_i16, vtot_off_i16);

        for(int i = 0; i < nloop; i++)
        {
            q6op_vstu_AV((HVX_Vector *)outp, hvx_requantize_vector_8to8_uniform_gain(vinp, vvtot_off_i16));
            vinp += 1;
            outp += 128;
        }

        if(nextra != 0)
        {
            q6op_vstu_variable_ARV(outp, nextra, hvx_requantize_vector_8to8_uniform_gain(vinp, vvtot_off_i16));
        }
    }
    else if(gain > 1.0f)
    {
        float gain_coef = flt_getfrac(gain);
        int32_t gaini = saturate_i16(roundf_i32(gain_coef * (float)(1u << 15)));
        int32_t prod_rsh = 15 - flt_getexp(gain);

        for(int i = 0; i < nloop; i++)
        {
            q6op_vstu_AV((HVX_Vector *)outp, hvx_requantize_vector_8to8_hi_gain(vinp, gaini, prod_rsh, vvin_off_i16, vout_off_i16));
            vinp += 1;
            outp += 128;
        }

        if(nextra != 0)
        {
            q6op_vstu_variable_ARV(outp, nextra, hvx_requantize_vector_8to8_hi_gain(vinp, gaini, prod_rsh, vvin_off_i16, vout_off_i16));
        }
    }
    else
    {
        int32_t gaini = saturate_i16(roundf_i32(gain * (float)(1u << 15)));

        for(int i = 0; i < nloop; i++)
        {
            q6op_vstu_AV((HVX_Vector *)outp, hvx_requantize_vector_8to8(vinp, gaini, vvin_off_i16, vout_off_i16));
            vinp += 1;
            outp += 128;
        }

        if(nextra != 0)
        {
            q6op_vstu_variable_ARV(outp, nextra, hvx_requantize_vector_8to8(vinp, gaini, vvin_off_i16, vout_off_i16));
        }
    } //if gain > 1.0f
}

void nn_requantize_qu8_to_qu8_hvx_d32(uint8_t *outp_b, uint8_t const *inp_b, int h_count, int nd32, int widvecs, int height_stride, int d32_stride, float gain, int32_t in_offset, int32_t out_offset)
{
    HVX_Vector vin_off_i16 = q6op_Vh_vsplat_R(saturate_i16(in_offset));
    HVX_VectorPair vvin_off_i16 = Q6_W_vcombine_VV(vin_off_i16, vin_off_i16);
    HVX_Vector vout_off_i16 = q6op_Vh_vsplat_R(saturate_i16(out_offset));

    uint8_t const *inp;
    uint8_t * outp;
    HVX_Vector const *vinp;
    HVX_Vector *voutp;

    if(CLOSE_ENUF(gain, 1.0f))
    {
        HVX_Vector vtot_off_i16 = Q6_Vh_vsub_VhVh_sat(vout_off_i16, vin_off_i16);
        HVX_VectorPair vvtot_off_i16 = Q6_W_vcombine_VV(vtot_off_i16, vtot_off_i16);

        for(int h = 0; h < h_count; h++)
        {
            inp = inp_b + h * height_stride;
            outp = outp_b + h * height_stride;

            for(int id32 = 0; id32 < nd32; id32++)
            {
                vinp = (HVX_Vector const *)inp;
                voutp = (HVX_Vector *)outp;

                for(int i = 0; i < widvecs; i++)
                {
                    voutp[i] = hvx_requantize_vector_8to8_uniform_gain(vinp+i, vvtot_off_i16);
                }
                inp += d32_stride;
                outp += d32_stride;
            }
        }
    }
    else if(gain > 1.0f)
    {
        float gain_coef = flt_getfrac(gain);
        int32_t gaini = saturate_i16(roundf_i32(gain_coef * (float)(1u << 15)));
        int32_t prod_rsh = 15 - flt_getexp(gain);

        for(int h = 0; h < h_count; h++)
        {
            inp = inp_b + h * height_stride;
            outp = outp_b + h * height_stride;

            for(int id32 = 0; id32 < nd32; id32++)
            {
                vinp = (HVX_Vector const *)inp;
                voutp = (HVX_Vector *)outp;

                for(int i = 0; i < widvecs; i++)
                {
                    voutp[i] = hvx_requantize_vector_8to8_hi_gain(vinp+i, gaini, prod_rsh, vvin_off_i16, vout_off_i16); 
                }
                inp += d32_stride;
                outp += d32_stride;
            }
        }
    }
    else
    {
        int32_t gaini = saturate_i16(roundf_i32(gain * (float)(1u << 15)));

        for(int h = 0; h < h_count; h++)
        {
            inp = inp_b + h * height_stride;
            outp = outp_b + h * height_stride;

            for(int id32 = 0; id32 < nd32; id32++)
            {
                vinp = (HVX_Vector const *)inp;
                voutp = (HVX_Vector *)outp;

                for(int i = 0; i < widvecs; i++)
                {
                    voutp[i] = hvx_requantize_vector_8to8(vinp+i, gaini, vvin_off_i16, vout_off_i16);
                }
                inp += d32_stride;
                outp += d32_stride;
            }
        }
    } // if gain > 1.0f ... 
}

// handles cases where gain <= 1.0f
inline HVX_Vector  __attribute__((always_inline))
hvx_requantize_vector_8to8(
    HVX_Vector const* vinp,
    int32_t gaini,
    HVX_VectorPair vvin_off_i16,
    HVX_Vector vout_off_i16 )
{
    // cast uint8 input to 16 bits
    HVX_VectorPair vvinput_i16 = Q6_Wuh_vunpack_Vub(*vinp);
    gaini |= gaini << 16;

    // subtract input offset
    HVX_VectorPair vvdiff_i16 = Q6_Wh_vsub_WhWh(vvinput_i16, vvin_off_i16);
    HVX_Vector vdiff_i16_0 = Q6_V_lo_W(vvdiff_i16);
    HVX_Vector vdiff_i16_1 = Q6_V_hi_W(vvdiff_i16);

    // multiply by gain
    HVX_Vector vprod_i16_0 = Q6_Vh_vmpy_VhRh_s1_rnd_sat(vdiff_i16_0, gaini);
    HVX_Vector vprod_i16_1 = Q6_Vh_vmpy_VhRh_s1_rnd_sat(vdiff_i16_1, gaini);

    // add output offset
    HVX_Vector vsum_i16_0 = Q6_Vh_vadd_VhVh_sat(vprod_i16_0, vout_off_i16);
    HVX_Vector vsum_i16_1 = Q6_Vh_vadd_VhVh_sat(vprod_i16_1, vout_off_i16);

    // saturate to 8 bits
    HVX_Vector vres_u8 = Q6_Vub_vpack_VhVh_sat(vsum_i16_1, vsum_i16_0);

    return vres_u8;
}

// handles cases where gain > 1.0f
inline HVX_Vector  __attribute__((always_inline))
hvx_requantize_vector_8to8_hi_gain(
    HVX_Vector const* vinp,
    int32_t gaini,
    int32_t prod_rsh,
    HVX_VectorPair vvin_off_i16,
    HVX_Vector vout_off_i16 )
{
    // cast uint8 input to 16 bits
    HVX_VectorPair vvinput_i16 = Q6_Wuh_vunpack_Vub(*vinp);
    gaini |= gaini << 16;

    // subtract input offset
    HVX_VectorPair vvdiff_i16 = Q6_Wh_vsub_WhWh(vvinput_i16, vvin_off_i16);
    HVX_Vector vdiff_i16_0 = Q6_V_lo_W(vvdiff_i16);
    HVX_Vector vdiff_i16_1 = Q6_V_hi_W(vvdiff_i16);

    // multiply by gain, 32-bit accumulators
    HVX_VectorPair vvprod_i32_0 = Q6_Ww_vmpy_VhRh(vdiff_i16_0, gaini);
    HVX_VectorPair vvprod_i32_1 = Q6_Ww_vmpy_VhRh(vdiff_i16_1, gaini);

    // shift to halfword by (15 - gain_lsh) 
    HVX_Vector vprod_i16_0 = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(vvprod_i32_0), Q6_V_lo_W(vvprod_i32_0), prod_rsh);
    HVX_Vector vprod_i16_1 = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(vvprod_i32_1), Q6_V_lo_W(vvprod_i32_1), prod_rsh);

    // add output offset
    HVX_Vector vsum_i16_0 = Q6_Vh_vadd_VhVh_sat(vprod_i16_0, vout_off_i16);
    HVX_Vector vsum_i16_1 = Q6_Vh_vadd_VhVh_sat(vprod_i16_1, vout_off_i16);

    // saturate to 8 bits
    HVX_Vector vres_u8 = Q6_Vub_vpack_VhVh_sat(vsum_i16_1, vsum_i16_0);

    return vres_u8;
}

inline HVX_Vector __attribute__((always_inline))
hvx_requantize_vector_8to8_uniform_gain(
    HVX_Vector const* vinp,
    HVX_VectorPair vvtot_off_i16)
{
    // cast uint8 input to 16 bits
    HVX_VectorPair vvinput_i16 = Q6_Wuh_vunpack_Vub(*vinp);
    // add total offset
    HVX_VectorPair vvsum_i16 = Q6_Wh_vadd_WhWh_sat(vvinput_i16, vvtot_off_i16);
    // saturate to 8 bits
    return Q6_Vub_vpack_VhVh_sat(Q6_V_hi_W(vvsum_i16), Q6_V_lo_W(vvsum_i16));
}
