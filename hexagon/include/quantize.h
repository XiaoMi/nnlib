
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
#ifndef NN_QUANTIZE_H
#define NN_QUANTIZE_H 1
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for common quantization routines
 */

#include <stdint.h>
#include <math.h>
#ifdef __hexagon__
#include "hexagon_protos.h"
#endif

///////////////////////////////////////////////////////////
//  min/max inline functions
//  min_XX and max_XX where XX is one of u32, i32, i64, u64;
////////////////////////////////////////////////////////////
#ifdef __hexagon__
#define MINMAX_FUNC_DECL( TYP, FNAME, OP, Q6OP)  static inline __attribute__((unused,always_inline)) TYP FNAME( TYP a, TYP b) { return Q6OP(a,b); }
#else
#define MINMAX_FUNC_DECL( TYP, FNAME, OP, Q6OP)  static inline __attribute__((unused,always_inline)) TYP FNAME( TYP a, TYP b) { return (a OP b)? a: b; }
#endif
MINMAX_FUNC_DECL( uint32_t, min_u32, < , Q6_R_minu_RR)
MINMAX_FUNC_DECL( uint32_t, max_u32, > , Q6_R_maxu_RR)
MINMAX_FUNC_DECL( int32_t, min_i32, < , Q6_R_min_RR)
MINMAX_FUNC_DECL( int32_t, max_i32, > , Q6_R_max_RR)
MINMAX_FUNC_DECL( uint64_t, min_u64, < , Q6_P_minu_PP)
MINMAX_FUNC_DECL( uint64_t, max_u64, > , Q6_P_maxu_PP)
MINMAX_FUNC_DECL( int64_t, min_i64, < , Q6_P_min_PP)
MINMAX_FUNC_DECL( int64_t, max_i64, > , Q6_P_max_PP)
#undef MINMAX_FUNC_DECL

///////////////////////////////////////////////////////////
// clip_XX( val, lo, hi) where XX is one of u32,i32,flt
// range-clip 'val' to lo..hi.
// result unspecified when lo > hi.
///////////////////////////////////////////////////////////

#define CLIP_FUNC_DECL( TYP,FNAME, FMIN,FMAX)\
static inline __attribute__((unused,always_inline)) TYP FNAME( TYP x, TYP mn, TYP mx) { return FMAX(FMIN(x,mx),mn); }
CLIP_FUNC_DECL( uint32_t, clip_u32, min_u32, max_u32 )
CLIP_FUNC_DECL( int32_t, clip_i32, min_i32, max_i32 )
CLIP_FUNC_DECL( int32_t, clip_flt, fminf, fmaxf)

#undef CLIP_FUNC_DECL

// saturate_u8
// this is the same as clip_i32( val, 0, 255) (but there's an opcode for it)

static inline __attribute__((unused,always_inline)) int32_t saturate_u8( int32_t val )
{
#ifdef __hexagon__
	return Q6_R_satub_R(val);
#else
	return (val < 0) ? 0 : ((val > 255) ? 255 : val);
#endif
}
// saturate_u16
// this is the same as clip_i32( val, 0, 65535) (but there's an opcode for it)

static inline __attribute__((unused,always_inline)) int32_t saturate_u16( int32_t val )
{
#ifdef __hexagon__
	return Q6_R_satuh_R(val);
#else
	return (val < 0) ? 0 : ((val > 65535) ? 65535 : val);
#endif
}
// saturate_i16
// this is the same as clip_i32( val, -32768, 32767) (but there's an opcode for it)

static inline __attribute__((unused,always_inline)) int32_t saturate_i16( int32_t val )
{
#ifdef __hexagon__
	return Q6_R_sath_R(val);
#else
	return (val < -32768) ? -32768 : ((val > 32767) ? 32767 : val);
#endif
}

// functions to round float to nearest integer.
// On hexagon, these work without using a function call
// We could also use Q6_R_convert_sf2w_R, but that depends on FP rounding mode.
// For finite x, roundf_i32(x) should give the same result as (int)roundf(x).
// (which is the same as (int)floorf(x+0.5f), except for x = -0.5, -1.5, -2.5 etc).
//
//
static inline __attribute__((unused,always_inline)) int32_t roundf_i32( float val )
{
	// add 0.5 (with same sign as val) and then conversion to int truncates toward 0.
	// values exactly halfway will round away from 0 (like roundf).

	return (int) ( val + copysignf(0.5f,val) );
}
// same thing for rounding to unsigned range; -ve inputs will give 0.
//
static inline __attribute__((unused,always_inline)) uint32_t roundf_u32( float val )
{
	// add 0.5f and then convert to uint (trunc towards 0; -ve values are clipped to 0).
#ifdef __hexagon__
	// use intrinsic since conv of -ve float to unsigned is 'undefined behaviour' in C.
	return Q6_R_convert_sf2uw_R_chop( val + 0.5f);
#else
	return (val < 0.5f)?0 : (uint32_t)( val + 0.5f);
#endif
}


///////////////////////
// Some utilities for replacing frexpf and ldexpf, on hexagon these
// will each compile to a few instructions, no function call.
// Note that they don't handle inputs which are zero, denormal, infinity or NaN
///////////////////////
//
// getexp for floats (compiles with no function call).
// This returns the same result as the exponent from frexp,
// i.e. it will be 0 if abs(x) is in range 0.5  .. 0.99999
//                 1 if abs(x) is in range 1.0  .. 1.9999 etc
// But
//   - for inf/nan you will get 129
//   - for zero, or denormal, you will get -126
//
//
static inline __attribute__((unused,always_inline))
int flt_getexp( float x)
{
	union {
		float f;
		int32_t u32;
	} uu = { x };
	return ((uu.u32>>23) & 0xFF) - 126;
}
//
// This 'normalizes' a float to 0.5 .. 0.9999  (sign is retained)
// Same result as the return value from frexpf, without using a function call
// Results are not valid if x is 0, denormal, or inf/nan
//
static inline __attribute__((unused,always_inline))
int flt_getfrac( float x)
{
	union {
		float f;
		int32_t u32;
	} uu = { x };
	uu.u32 = (uu.u32 & 0x807fffffu) | (126<<23);	// force exponent = 126
	return uu.f;
}


//
// This generates powf(2.0f, iexpo)  as a float, without using a function call
// constraints: the parameter 'expo' must be in range -126 .. 127
//
static inline __attribute__((unused,always_inline))
float flt_power2( int iexpo )
{
	union {
		int32_t u32;
		float f;
	} uu = { saturate_u8(iexpo+127) << 23 };
	return uu.f;
}
// this is the same as ldexpf( x, iexpo), without using a function call,
// and with the constraint that iexpo can't exceed -126 .. 127
//
static inline __attribute__((unused,always_inline))
float flt_ldexp( float x, int iexpo )
{
	return x * flt_power2(iexpo);
}

//
// for val >= 1, return floor(log2(val)); range is 0..31
// result is undefined when val = 0.
//
static inline __attribute__((unused,always_inline))
int floor_log2( unsigned val )
{
    return 31 - __builtin_clz(val);
}
// for val >= 1, return ceil(log2(val))
// result is -1 when val= 0; otherwise range is 0..32
//
static inline __attribute__((unused,always_inline))
int ceiling_log2( unsigned val )
{
    if( val < 4) return (int)val -1;
    return 32 - __builtin_clz(val-1);
}


static inline uint8_t quantize_uint8(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(0.0001f,maxval-minval);
	float resize_amt = 255.0f/range;
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	return saturate_u8( value_i);
}

static inline int32_t quantize_uint(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(0.0001f,maxval-minval);
	float resize_amt = 255.0f/range;
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	int32_t ret = (value_i < 0) ? 0 : value_i;
	return ret;
}

static inline int32_t quantize_int(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(0.0001f,maxval-minval);
	float resize_amt = 255.0f/(range);
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	return value_i;
}

//
// x * (float)(1./255) is faster than, but not as accurate as,
//  x /255.0f
// because the former has two sources of error (rounding 1/255 and rounding the multiply;
// and it happens that 1/255. falls almost halfway between two float values)
// This function gives a result using multiply/add which is as accurate as x/255.0f
// it's just k1*x + k2*x where k1,k2 add up accurately to 1/255.
// And we use force use of fma to prevent the larger term from being rounded before they are added.
//
static inline float flt_div_255( float x)
{
#ifdef __hexagon__
	return  Q6_R_sfmpyacc_RR(-0xF.EFEFFp-36f * x, x, 0x8.08081p-11f );
#else
	return (0x8.08081p-11f * x)  +  ( (-0xF.EFEFFp-36f) * x);
#endif
}

static inline void quantize_adjust_range(float *out_min, float *out_max, float *out_stepsize, float *out_recip_stepsize, float in_min, float in_max)
{
	float minval = fminf(0.0f,in_min);
	float maxval = fmaxf(0.0f,in_max);
	float range = fmaxf(0.0001f,maxval-minval);
	float recip_stepsize = 255.0f/range;

	// move either min, or max, as  little as possible, so that
	// the 'zero' point  -min *255/range  is an integer. if minval == 0
	// this is already true.
	if( minval < 0.0f ){
		float z = - minval *recip_stepsize;		// current 'zero point'
		float zi = floorf(z);					// integer part, >=0
		float zf = z - zi;
		// if within 2^-14 of an integer, call it close enough
		if( zf > 6.1035156e-05f && zf < 0.999938965f){
			// choose which end to move
			// if zi <= 0  or >= 254, the decision is based on that only (to
			// avoid divide by 0) otherwise choose based on which can be moved
			// the least.
			//
			if( zi > 0.0f && ( zi > 253.0f || (zf-1.0f)*minval>= zf*maxval )) {
				// increase max, change z to zi
				range = -255.0f*minval/zi;
				maxval = minval+ range;
			}else{
				// decrease min; change z to zi+1
				minval = maxval*(zi+1.0f)/(zi-254.0f);
				range = maxval-minval;
			}
			// recalc range
			recip_stepsize = 255.0f/range;
		}
	}
	*out_min = minval;
	*out_max = maxval;
	*out_stepsize = flt_div_255(range);
	*out_recip_stepsize = recip_stepsize;
}


//
// This adjusts a min .. max range
// (by decreasing min, or increasing max) until
// the 'zero point' (the value which encodes zero)
// is an exact integer.
//  if 0 encodes 'mn', and 255 encodes 'mx', then zero is encoded by
//     z = -255*mn/(mx-mn)
//
// if this is not an integer ( +/- 2^-14) then
// *min_p will be decreased, or *max_p increased,
// (whichever adjustment is smaller) to meet this condition
//
//
// return value:
//  -1 : test failed for max > min (includes either being NaN)
//   0 : ok, no adjustment made; zero point already an integer +/- 2^-14
//   1 : adjusted min downward
//   2 : adjusted max upward
//
// note: if you want min <=0 and max >=0, enforce that before calling
// (and if you decrease min to 0.0, you don't need to call this since
// z = 0 in that case. The function supports 'unbalanced'
// ranges (e.g. min = 2.0, max = 257.0 -> z = -2).
//

int adjust_minmax_for_zero( float *min_p, float *max_p );

//
// same thing but with constraints as to which endpoints should be considered 'fixed'
//   bit 0 - minimum is fixed
//   bit 1 - maximum is fixed

int adjust_minmax_for_zero_with_constraints( float *min_p, float *max_p, int constraint );

#endif
