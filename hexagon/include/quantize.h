
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
#include "hvx_inlines.h"
#endif
#include <nn_graph_builtin.h>

///////////////////////////////////////////////////////////
//  min/max inline functions
//  min_XX and max_XX where XX is one of u32, i32, i64, u64;
////////////////////////////////////////////////////////////
#ifdef __hexagon__
#define MINMAX_FUNC_DECL(TYP, FNAME, OP, Q6OP) \
	static inline __attribute__((unused, always_inline)) TYP FNAME(TYP a, TYP b) { return Q6OP(a, b); }
#else
#define MINMAX_FUNC_DECL(TYP, FNAME, OP, Q6OP) \
	static inline __attribute__((unused, always_inline)) TYP FNAME(TYP a, TYP b) { return (a OP b) ? a : b; }
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
struct hvx_quant_parms
{
	int scaling;
	int min_offset;
	int common_exp;
	float minval, maxval;
};

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
// equivalent to isfinite(float x), but no function call
// false if x is +/- inf, or NaN; true otherwise
static inline __attribute__((unused,always_inline)) int flt_isfinite( float x )
{
	union {
		float f;
		int32_t u32;
	} uu = { x };
	return ((uu.u32>>23) & 0xFF) != 0xFF;
}

// equivalent to isnan(float x), but no function call
// true if x is NaN; false otherwise
static inline __attribute__((unused,always_inline)) int flt_isnan( float x )
{
	union {
		float f;
		int32_t u32;
	} uu = { x };
	return (uu.u32 & 0x7FFFFFFF)> 0x7F800000;
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
static inline __attribute__((unused, always_inline)) int flt_getexp(float x)
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
static inline __attribute__((unused, always_inline)) float flt_getfrac(float x)
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
static inline __attribute__((unused, always_inline)) float flt_power2(int iexpo)
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
static inline __attribute__((unused, always_inline)) float flt_ldexp(float x, int iexpo)
{
	return x * flt_power2(iexpo);
}

//
// for val >= 1, return floor(log2(val)); range is 0..31
// result is undefined when val = 0.
//
static inline __attribute__((unused, always_inline)) int floor_log2(unsigned val)
{
    return 31 - __builtin_clz(val);
}
// for val >= 1, return ceil(log2(val))
// result is -1 when val= 0; otherwise range is 0..32
//
static inline __attribute__((unused, always_inline)) int ceiling_log2(unsigned val)
{
	if (val < 4)
		return (int)val - 1;
    return 32 - __builtin_clz(val-1);
}

// this would not be needed if hexagon-clang implemented __builtin_bswap32 properly
// instead of falling back to a generic expansion.
static inline uint32_t __attribute__((always_inline)) byteswap_u32(uint32_t x)
{
#ifndef __hexagon__
	return __builtin_bswap32( x );
#else
	return Q6_R_swiz_R(x);
#endif
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

#if 0 // disused - 16 bit scaling divides by 65536.
/* tailor series expansion (1-2^-16)^-1 = 1 + k + k^2 +k^3 */
static inline float flt_div_65535( float x)
{
#ifdef __hexagon__
        //return  Q6_R_sfmpyacc_RR(0x8.0008p-51f * x, x, 0x8.0008p-19f );
        return  (x / 65535.0f); //Q6_R_sfmpyacc_RR(0x8.0008p-51f * x, x, 0x8.0008p-19f );
#else
        return (0x8.0008p-19f * x)  +  ( (0x8.0008p-51f) * x);
#endif
}
#endif // disused

static inline uint8_t quantize_uint8(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(1e-18f, maxval-minval);
	float resize_amt = 255.0f/range;
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	return saturate_u8( value_i);
}

// this converts a min and max to level_size and 'zero'.
static inline int get_qu8_level_size_zero( float minval, float maxval, float * levsize_p)
{
    float level_size = flt_div_255( maxval-minval );
    int zeroval = roundf_i32( -minval/level_size);
    *levsize_p = level_size;
    return saturate_u8( zeroval );
}

// this converts a min and max to level_size and 'zero' for qu16
static inline int get_qu16_level_size_zero( float minval, float maxval, float * levsize_p)
{
    float level_size = ( maxval-minval )*(float)(1.0/65536.0);
    int zeroval = roundf_i32( -minval/level_size);
    *levsize_p = level_size;
    return saturate_u16( zeroval );
}
static inline uint16_t quantize_uint16(float val, float minval, float maxval)
{
	float range = fmaxf(1e-18f, maxval - minval);
	float resize_amt = 65536.0f / range;
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	return saturate_u16(value_i);
}


static inline int32_t quantize_uint(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(1e-18f, maxval - minval);
	float resize_amt = 255.0f/range;
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	int32_t ret = (value_i < 0) ? 0 : value_i;
	return ret;
}

static inline int32_t quantize_int(float val, float minval, float maxval)
{
	/* We want 0.0 -- 255.0 to resize to 0..255 */
	float range = fmaxf(1e-18f, maxval - minval);
	float resize_amt = 255.0f/(range);
	float value_f = (val - minval) * resize_amt;
	int32_t value_i = roundf(value_f);
	return value_i;
}

// find the range of an array of floats, in such a way that NaN values get noticed
// NOTE: the 'min_out' is always <=0:  min(0.0, .. all inputs.. )
//   and max_out is >=0:     max(0.0, .. all inputs ... )
//
// if there are any '+nan', then max_out will be a +nan;
//    otherwise if there are any +inf, max_out will be +inf
// if there are any '-nan', then min_out will be a +nan;
//    otherwise if there are any -inf, min_out will be +inf
//
// Method is
//  (1) map the image of a float to an int:
//  (2) find the max of 'signed' ints; this will discard the -ves and give the largest >0 value
//     and the max of 'unsigned' starting at 0x80000000; this will discard the +ves and give most -ve vlaue
//  (3) convert results back to float
//
// return is 1 if inf or max were found; 0 otherwise.
//
static inline int __attribute__((unused))
find_range_of_floats( float const * arr, int n, float * min_out, float *max_out)
{
	uint32_t all_min_code = 0x80000000;
	int32_t all_max_code = 0;
	for (int i = 0; i < n; i++)
	{
		union {
			float f;
			uint32_t u32;
		} uu = {arr[i]};
		all_min_code = max_u32( all_min_code, uu.u32);	// only <0 values will be included here
		all_max_code = max_i32( all_max_code, uu.u32);				// only > 0 values will be included here.
	}
	union {
		uint32_t u32;
		float f;
	} umin = {all_min_code}, umax = {(uint32_t)all_max_code};
	*min_out = umin.f;
	*max_out = umax.f;
	uint32_t exp_test = max_u32( all_max_code, all_min_code & 0x7fffffff);
	return exp_test >= 0x7f800000;
}


//
// operations for carefully rounding floats...
//
// flt_round_up_4eps rounds the value upwards (away from 0) if needed to reach
// the next higher value which has 2 zeros in the lsbs (i.e. a multiple of 4x its own epsilon).
// Has no effect unless the exponent field is <= 0xFE;
//  this is to avoid 'rounding' things from NaN to zero; it's still possible
//  to round from just-below-infinity to infinity;
//
static inline float
flt_round_up_4eps( float x)
{
	union {
		float f;
		uint32_t u32;
	} uu = { x };
	if ((uu.u32 << 1) < 0xFF000000u)
	{										  // is safe...
		uu.u32 = (uu.u32 + 3) & ~(uint32_t)3;	// round it up.
	}
	return uu.f;
}
//
// flt_round_near_4eps rounds the value to the nearest value which has 2 zeros in the lsbs
// (i.e. a multiple of 4x its own epsilon). If the 2 lsbs are 10, it will round 110 up and 010
// down.
// Has no effect unless the exponent field is <= 0xFE;
//  this is to avoid 'rounding' things from NaN to zero; it's still possible
//  to round from just-below-infinity to infinity;
//
static inline float
flt_round_near_4eps( float x)
{
	union {
		float f;
		uint32_t u32;
	} uu = { x };
	if ((uu.u32 << 1) < 0xFF000000u)
	{ // is safe...
		uint32_t rnd = (uu.u32 & 4)? 2:1;
		uu.u32 = (uu.u32 + rnd) & ~(uint32_t)3;
	}
	return uu.f;
}



static inline void quantize_adjust_range(float *out_min, float *out_max, float *out_stepsize, float *out_recip_stepsize, float in_min, float in_max)
{
	float minval = fminf(0.0f,in_min);
	float maxval = fmaxf(0.0f,in_max);
	float range = fmaxf(1e-18f, maxval - minval);
	float recip_stepsize = 255.0f/range;

	// move either min, or max, as  little as possible, so that
	// the 'zero' point  -min *255/range  is an integer. if minval == 0
	// this is already true.
	if (minval < 0.0f)
	{
		float z = - minval *recip_stepsize;		// current 'zero point'
		float zi = floorf(z);					// integer part, >=0
		float zf = z - zi;
		// if within 2^-14 of an integer, call it close enough
		if (zf > 6.1035156e-05f && zf < 0.999938965f)
		{
			// choose which end to move
			// if zi <= 0  or >= 254, the decision is based on that only (to
			// avoid divide by 0) otherwise choose based on which can be moved
			// the least.
			//
			if (zi > 0.0f && (zi > 253.0f || (zf - 1.0f) * minval >= zf * maxval))
			{
				// increase max, change z to zi
				range = -255.0f*minval/zi;
				maxval = minval+ range;
			}
			else
			{
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

// adust a 'nominal' 16-bit range to have a proper zero.
//
static inline void
quantize_adjust_range_u16(float *out_min, float *out_max, float *out_stepsize, float *out_recip_stepsize, float in_min, float in_max)
{
	float minval = fminf(0.0f,in_min);
	float maxval = fmaxf(0.0f,in_max);
	float range = fmaxf(1e-18f, maxval - minval);
	float recip_stepsize = 65536.0f/range;

	if (minval < 0.0f)
	{
		float z = - minval *recip_stepsize;		// current 'zero point'
		float zi = floorf(z);					// integer part, >=0
		float zf = z - zi;
		// if within 2^-6 of an integer, call it close enough.
		if (zf > 0.015625f && zf < 0.984375f)
		{
			// choose which end to move
			// if zi <= 0  or >= 65535, the decision is based on that only (to
			// avoid divide by 0) otherwise choose based on which can be moved
			// the least.
			//
			if (zi > 0.0f && (zi > 65534.0f || (zf - 1.0f) * minval >= zf * maxval))
			{
				// increase max, change z to zi
				zi = fminf(zi,65535.0f);		// disallow 65536
				range = -65536.0f*minval/zi;
				maxval = minval+ range;
			}
			else
			{
				// decrease min; change z to zi+1
				minval = maxval*(zi+1.0f)/(zi-65535.0f);
				range = maxval-minval;
			}
			// recalc range
			recip_stepsize = 65536.0f/range;
		}
	}
	*out_min = minval;
	*out_max = maxval;
	*out_stepsize = range *(float)(1./65536.0f);
	*out_recip_stepsize = recip_stepsize;
}



// check if range is basically sane  (no inf/nan; min <=0, max >= 0, max > min
// returns -1 if error, 0, if ok

static inline int
check_range_is_sane( float min_in, float max_in)
{

	if (!flt_isfinite(min_in) || !flt_isfinite(max_in) || min_in > 0.0f || max_in < 0.0f || max_in - min_in < 1e-8f)
		return -1;
	return 0;
}
// this is like quantize_adjust_range but it also checks if supplied min, max are sane
// if so: adjust range and return 0; if not, return -1 (and don't store any results).

int quantize_adjust_range_and_check( float *out_min, float *out_max, float *out_stepsize, float *out_recip_stepsize, float in_min, float in_max);

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
int adjust_minmax_for_zero_16b(float *min_p, float *max_p);

//
// same thing but with constraints as to which endpoints should be considered 'fixed'
//   bit 0 - minimum is fixed
//   bit 1 - maximum is fixed

int adjust_minmax_for_zero_with_constraints( float *min_p, float *max_p, int constraint );
int adjust_minmax_for_zero_with_constraints_16b( float *min_p, float *max_p, int constraint );
void hvx_do_dequantize( uint8_t const * inp, float * outp, int n, int qzero, float qstep );


//
// requantize n 32-bit numbers to u8; equiv to
//     outp[i] =  quantize_uint8( in_level_size* (float)inp[i],out_min,out_max);
//
// This uses HVX when possible, and so it must be called from a vector thread.
// In some cases (gain >1.0, input not aligned) it will use scalar code
//
void nn_requantize_i32_to_qu8_hvx( uint8_t *outp, int32_t const * inp, int n, float in_level_size, float out_min, float out_max);

// same with no hvx
void nn_requantize_i32_to_qu8( uint8_t *outp, int32_t const * inp, int n, float in_level_size, float out_min, float out_max);

void nn_requantize_qu8_to_qu8_hvx(uint8_t *outp, uint8_t const* inp, unsigned n, float gain, int32_t in_offset, int32_t out_offset);

void nn_requantize_qu8_to_qu8_hvx_d32(uint8_t *outp_b, uint8_t const *inp_b, int h_count, int nd32, int widvecs, int height_stride, int d32_stride, float gain, int32_t in_offset, int32_t out_offset);

//Find min/max of i32 tensor
void find_min_max_int32( int32_t const *arr, int n, int32_t * minmaxp );


//
// This is like quantize_asm but with a different order of operations; the offset
// is added after, instead of being subtracted before, so we can avoid overflow problems
// for small gain.
// Only n output bytes are stored (no alignment requirement)
//
// The operation is:
//    out[i] = saturate_u8(   add_sat(  offset, (in[i] * gain)/(2^31));
//
void hvx_quantize_32_to_8(
		int32_t const * inp,		// input [n], vector aligned
		int32_t offset,				// offset to add
		int32_t scale,				// scale with 31 fractional bits
		uint8_t * outp,				// output [n], any alignment ok
		int n );					// # elements

int find_scaling_for_hvx_quant ( float const minmax[2], struct hvx_quant_parms *out);
static inline int QuantizeMultiplierSmallerThanOne(float multiplier, int32_t *quantized_multiplier, int32_t *right_shift)
{
	if (multiplier <= 0 || multiplier >= 1)
	{
		return -1;
	}
	const float q = flt_getfrac(multiplier);
	*right_shift = -flt_getexp(multiplier);
	int64_t q_fixed = (int64_t)(floorf(q * (1LL << 31)));
	if (q_fixed > (1LL << 31))
	{
		return -1;
	}
	if (q_fixed == (1LL << 31))
	{
		q_fixed /= 2;
		--*right_shift;
	}
	if (*right_shift < 0)
	{
		return -1;
	}
	if (q_fixed > ((2LL << 31) - 1))
	{
		return -1;
	}
	*quantized_multiplier = (int32_t)(q_fixed);
	return 0;
}

//In cases where x is negative, this rounds up toward +inf
static inline HVX_Vector RoundingDivideByPOT(HVX_Vector x, int32_t exponent)
{
	unsigned rbias = (exponent > 31) ? 0 : ((1u << exponent) >> 1);
	HVX_Vector vrbias = Q6_V_vsplat_R(rbias);
	int exp_sat = min_i32(exponent, 31);
	HVX_Vector xbiased = Q6_Vw_vadd_VwVw_sat(x, vrbias);
	HVX_Vector result = Q6_Vw_vasr_VwR(xbiased, exp_sat);
	return result;
}

static inline HVX_Vector MultiplyByQuantizedMultiplier(HVX_Vector x, int32_t quantized_multiplier, int32_t shift)
{
	int32_t left_shift = shift > 0 ? shift : 0;
	int32_t right_shift = shift > 0 ? 0 : -shift;
	HVX_Vector qmulti = Q6_V_vsplat_R(quantized_multiplier);
	HVX_Vector interm_res1 = Q6_Vw_vasl_VwR(x, left_shift);
	HVX_Vector interm_res2 = q6op_Vw_vmpy_VwVw_s1_rnd_sat(interm_res1, qmulti);

	return RoundingDivideByPOT(interm_res2, right_shift);
}

#endif
