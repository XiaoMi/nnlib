#ifndef HVX_MATHOPS_H
#define HVX_MATHOPS_H 1
/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
#include "hvx_inlines.h"
#include <stdint.h>
//
//
// 'fractional division'
//  gives n/d  with 15 fractional bits,
//   subject to :
//      (1) d must be normalized in range 0x4000 .. 0x7fff
//      (2) if abs(n) >= d, the result will saturate to 0x7fff or -0x8000
//          (same sign as n)
//      (3) if n==0, result is always 0, even if d is not normalized (or 0).
// Steps are:
//  - find k1 by lookup table from bits [13:9] of d
//  -  d1 = d + mul_s1_round(d,k1)
//  -  n1 = n + mul_s1_round(n,k1)
//  -  d1 >= 0x7C00 now
//  - find k2 by lookup table from bits [9:5] of d1
//  -  d2 = d1 + mul_s1_round(d1,k2)
//  -  n2 = n1 + mul_s1_round(n1,k2)
//  -  d2 >= 0x7FDF now
//  - find e = 0x8000 - d2
//  - result is n2 + mul_s1_round(n2,e)
// The first table lookup is done using (d>>10) which is always in range 16..31
// (since d is 0x4000..0x7fff)
// The second is done using (d1>>7)&0xF which is in range 0..15
// So the two lookup tables can be held in odd/even 16-bit slots of a vector.
// Each table entry for k {or k1} is chosen to be as large as possible given that the largest
// d { or d1} value selecting the entry must not cause the calculation of d1 { or d2} to
// exceed 0x7fff.
// Exceptions:
//  (1) if n=0 the result is always 0, regardless of d.
//  (2) if d is not in the proper range, and n!=0, the result is undefined.
//
/*
 * const int16_t
	lut_fracdivide_k1k2[64] = {
            1023, 30784, 989, 28915,        955, 27153, 921, 25488,
            887, 23913, 854, 22422,         820, 21007, 786, 19662,
            753, 18383, 719, 17165,         686, 16004, 653, 14895,
            620, 13836, 586, 12823,         553, 11853, 520, 10923,

            487, 10032, 454, 9176,          421, 8353, 389, 7562,
            356, 6801, 323, 6068,            291, 5362, 258, 4681,
            226, 4024, 193, 3390,           161, 2777, 129, 2185,
             96, 1612, 64, 1057,             32, 520, 0, 0
       };
*/

static inline  HVX_Vector
hvx_fracdivide_Vh_VhVh( HVX_Vector n, HVX_Vector d )
{
	// We extract bits 13:9 from d; since 15:14 are assumed to be 01,
	// we end up with a value in range 0x20 .. 0x3f in the even bytes of dr).
	//
    HVX_Vector dx = Q6_Vuh_vlsr_VuhR( d, 9);
	HVX_Vector lkup = *(HVX_Vector const*)lut_fracdivide_k1k2;
	HVX_VectorPair k1a = Q6_Wh_vlut16_VbVhR( dx, lkup , 2);

	HVX_Vector k1= Q6_V_lo_W( Q6_Wh_vlut16or_WhVbVhR( k1a, dx, lkup ,3 ));

	// find d1 = d + d*k1  ( is >= 0x7C00, <= 0x7fff)
	// find n1 = n + n*k1  (saturating add)

    HVX_Vector d1 = Q6_Vh_vadd_VhVh_sat( d, Q6_Vh_vmpy_VhVh_s1_rnd_sat( d, k1 ));
    HVX_Vector n1 = Q6_Vh_vadd_VhVh_sat( n, Q6_Vh_vmpy_VhVh_s1_rnd_sat( n, k1 ));

	// get bits 9:5 of d1
	dx = Q6_V_vand_VV( Q6_Vuh_vlsr_VuhR( d1, 5), Q6_V_vsplat_R( 0x001F001F));

	HVX_VectorPair k2a = Q6_Wh_vlut16_VbVhR( dx, lkup , 0);
	HVX_Vector k2= Q6_V_lo_W( Q6_Wh_vlut16or_WhVbVhR( k2a, dx, lkup ,1 ));

    HVX_Vector d2 = Q6_Vh_vadd_VhVh_sat( d1, Q6_Vh_vmpy_VhVh_s1_rnd_sat( d1, k2 ));  // >= 0x7fdf
    HVX_Vector n2 = Q6_Vh_vadd_VhVh_sat( n1, Q6_Vh_vmpy_VhVh_s1_rnd_sat( n1, k2 ));

    HVX_Vector e = Q6_Vh_vsub_VhVh( Q6_V_vsplat_R( 0x80008000), d2 );		// 1 .. 0x21
    return Q6_Vh_vadd_VhVh_sat( n2, Q6_Vh_vmpy_VhVh_s1_rnd_sat( n2, e ));
}

//
// 'scalar' reference of the fractional divide
//
static inline int
ref_fracdivide_Rh_RhRh(int n, int d )
{
	int dx = (d >> 9)-0x20;	// should be 0 .. 0x1f
	int k1 = ((unsigned) dx < 32) ?  lut_fracdivide_k1k2[2*dx+1]: 0;
	int d1 = Q6_R_vaddh_RR_sat( d, Q6_R_vmpyh_RR_s1_rnd_sat( d, k1 ));
	int n1 = Q6_R_vaddh_RR_sat( n, Q6_R_vmpyh_RR_s1_rnd_sat( n, k1 ));
	// get bits 9:5 of d1
	dx =  (d1>>5) & 0x1f;
	int k2 = lut_fracdivide_k1k2[2*dx];
	int d2 = Q6_R_vaddh_RR_sat( d1, Q6_R_vmpyh_RR_s1_rnd_sat( d1, k2 ));
	int n2 = Q6_R_vaddh_RR_sat( n1, Q6_R_vmpyh_RR_s1_rnd_sat( n1, k2 ));
	int e = 0x8000 - d2;	// 0 ... 0x21

    return (int16_t) Q6_R_vaddh_RR_sat( n2, Q6_R_vmpyh_RR_s1_rnd_sat( n2, e ));
}



//
// A lower precision version of the fractional divide, fewer steps.
//
//  gives n/d  with 15 fractional bits (10 bits effective precision)
//   subject to :
//      (1) d must be normalized in range 0x4000 .. 0x7fff
//      (2) if abs(n) >= d, the result will saturate to 0x7fff or -0x8000
//          (same sign as n)
//      (3) if n==0, result is always 0, even if d is not normalized (or 0).
//
// The '10 bits relative precision' is relative to the magnitude of the result;
// e.g. return value in range +/- 1000 will be very accurate.
//
static inline  HVX_Vector
hvx_fracdivide_10bit_Vh_VhVh( HVX_Vector n, HVX_Vector d )
{
	// We extract bits 13:9 from d; since 15:14 are assumed to be 0,
	// we end up with a value in range 0x20 .. 0x3f in the even bytes of dr).
	//
    HVX_Vector dx = Q6_Vuh_vlsr_VuhR( d, 9);
	HVX_Vector lkup = *(HVX_Vector const*)lut_fracdivide_k1k2;
	HVX_VectorPair k1a = Q6_Wh_vlut16_VbVhR( dx, lkup , 2);
	HVX_Vector k1= Q6_V_lo_W( Q6_Wh_vlut16or_WhVbVhR( k1a, dx, lkup ,3 ));

	// find d1 = d + d*k1  ( is >= 0x7C00, <= 0x7fff)
	// find n1 = n + n*k1  (saturating add)

    HVX_Vector d1 = Q6_Vh_vadd_VhVh_sat( d, Q6_Vh_vmpy_VhVh_s1_rnd_sat( d, k1 ));
    HVX_Vector n1 = Q6_Vh_vadd_VhVh_sat( n, Q6_Vh_vmpy_VhVh_s1_rnd_sat( n, k1 ));

	//
	// n/d = n1/d1  but d1 = 1-e  (where e is small)
    // so use n1*(1+e)  - n1 + n1*e
    //

    HVX_Vector e = Q6_Vh_vsub_VhVh( Q6_V_vsplat_R( 0x80008000), d1 );
    return Q6_Vh_vadd_VhVh_sat( n1, Q6_Vh_vmpy_VhVh_s1_rnd_sat( n1, e ));
}

//
// 'scalar' reference of the fractional divide
//
static inline int
ref_fracdivide_10bit_Rh_RhRh(int n, int d )
{
	int dx = (d >> 9)-0x20;	// should be 0 .. 0x1f
	int k1 = ((unsigned) dx < 32) ?  lut_fracdivide_k1k2[2*dx+1]: 0;
	int d1 = Q6_R_vaddh_RR_sat( d, Q6_R_vmpyh_RR_s1_rnd_sat( d, k1 ));
	int n1 = Q6_R_vaddh_RR_sat( n, Q6_R_vmpyh_RR_s1_rnd_sat( n, k1 ));
	int e = 0x8000 - d1;	// 0 ... 0x21
    return (int16_t) Q6_R_vaddh_RR_sat( n1, Q6_R_vmpyh_RR_s1_rnd_sat( n1, e ));
}


// reciprocal square root, 10 bit  result
// Input is a uint16_t value in range 0x4000 ..0xFFFF (considered to have 16 fractional
// bits, and thus be in the range 0.25..0.999985 )
// find its root-reciprocal, with 9 fractional bits, in range 1.0 ... 2.0
//  (0x200 .. 0x400 is the actual range)
//
// The error (in result lsbs), across all input codes, vs ideal result:
//    range: -1.12 ..0.50
//    mean:   -0.0687
//    rms :   0.3073
//

//
// lut_root_recip
//  0, 0, 0, 0,    0, 0, 0, 0,
//  0, 0, 0, 0,    0, 0, 0, 0,
//  63, 61, 60, 58,   57, 55, 54, 53,
//  52, 51, 50, 49,   48, 47, 46, 46,
//  45, 44, 44, 43,   42, 42, 41, 41,
//  40, 40, 39, 39,   38, 38, 38, 37,
//  37, 36, 36, 36,   35, 35, 35, 34,
//  34, 34, 33, 33,   33, 33, 32, 32
//
static inline
HVX_Vector
hvx_Vuh_rsqrt10_Vuh( HVX_Vector vin )
{
    // get the 6 MSBs from each value
    HVX_Vector msbs = Q6_Vuh_vlsr_VuhR( vin, 10 );
    //
    // table lookup to map 16..63 to 63..32
    // Note that the odd bytes lanes will be replaced with tbl[0] = 0
    //
	HVX_Vector tbl = *(HVX_Vector const*)lut_root_recip;
    HVX_Vector r6 =  q6op_Vb_vlut32_VbVbI( msbs, tbl, 0 );    // look up 0..31
    r6 = q6op_Vb_vlut32or_VbVbVbI( r6, msbs, tbl, 1 );        // look up 32..63

    // square that...
    HVX_Vector r6sq = Q6_Vh_vmpyi_VhVh(r6,r6);					// 1024 .. 3969
    // multiply by input .. result is close to 2**26
    HVX_VectorPair prod = Q6_Wuw_vmpy_VuhVuh(vin,r6sq);			// 0x3dae800 .. 0x42cb1f0
    // now exract bits 23::8 of that ... *not* saturating
    HVX_Vector pd = q6op_Vh_vasr_WwR( prod, 8 );			// -0x2518 .. 0x2cb1
    // now find (r6 << 4) - (r6*pd>>15)     (allowing for signed pd)
    HVX_Vector px = Q6_Vh_vmpy_VhVh_s1_rnd_sat( r6,pd);  // -17 .. 21
    return Q6_Vh_vsub_VhVh(  Q6_Vh_vasl_VhR( r6,4), px );       // 0x200 .. 0x400
}
//
// 'scalar' reference for hvx_Vuh_rsqrt10_Vuh
//

static inline
int
ref_Ruh_rsqrt10_Ruh( int vin )
{
	int r6 = lut_root_recip[ (vin>>10)&63];
	int prod = (uint16_t)vin * r6 * r6;
	int pd = (int16_t)( prod >> 8);
	int px = (int16_t)Q6_R_vmpyh_RR_s1_rnd_sat( r6, pd);
	return r6 * 16 - px;
}
//
// a common part of hvx_Vuh_rsqrt_Vuh and hvx_Vh_rsqrt_Vuh
// 'vin' is the input, r10 is hvx_Vuh_rsqrt10_Vuh(vin),
// and the return value is a correction to be subtracted
// from 32*r10 to get a 15-bit result.
//
static inline HVX_Vector
correction_for_rsqrts_Vuh( HVX_Vector vin, HVX_Vector r10 )
{
    // we need to find r10*r10*vin; actually this
    // is done as (r10*r10>>4)*vin, with the intermediate product saturated to u16
    //
    HVX_VectorPair prr =  Q6_Wuw_vmpy_VuhVuh(r10,r10);		// 0x40000 .. 0x100000
    HVX_Vector rx = q6op_Vuh_vasr_WwR_sat(prr, 4 );			// 0x4000 .. 0xFFFF
    // now find the full product (close to 2**30)

    HVX_VectorPair prod =  Q6_Wuw_vmpy_VuhVuh(vin,rx);		// 0x3FD17809 .. 0x401F9FC0
    // extract bits 25:10 (w/o saturating)
    HVX_Vector pd = q6op_Vh_vasr_WwR( prod, 10 );			// -0xBA2 .. 0x7E7
    // average to 9 (empirical rounding offset )
    pd = Q6_Vh_vavg_VhVh( pd, Q6_V_vsplat_R( (9 << 16) + 9) );	// -0x5CD .. 0x3F8

    // multiply by r10, respecting sign.
    return  Q6_Vh_vmpy_VhVh_s1_rnd_sat( pd, r10);		// -36 .. 16
}

// Input is a uint16_t value in range 0x4000 ..0xFFFF (considered to have 16 fractional
// bits, and thus be in the range 0.25..0.999985 )
// find its root-reciprocal, with 14 fractional bits, in range 1.0 ... 2.0
//  (0x4000 .. 0x8000)
// result is accurate to +/- 0.72 LSBs; and rel error < 1 in 23000.
// The error (in result lsbs), across all input codes, vs ideal result:
//    range: -0.66 .. 0.71
//    mean:   -0.0159
//    rms :   0.3038


static inline
HVX_Vector
hvx_Vuh_rsqrt_Vuh( HVX_Vector vin )
{
    HVX_Vector r10 = hvx_Vuh_rsqrt10_Vuh(vin);
    HVX_Vector px = correction_for_rsqrts_Vuh(vin, r10);
    // subtract from 32 * r10
    return Q6_Vh_vsub_VhVh(  Q6_Vh_vasl_VhR( r10,5), px );            // (r10<<5)-px
}
//
// This is identical to hvx_Vh_rsqrt_Vuh
// except result is limited to  0x4000 .. 0x7fff
// (so the error will be -1 msbs when the input is 0x4000).
//
static inline
HVX_Vector
hvx_Vh_rsqrt_Vuh( HVX_Vector vin )
{
    HVX_Vector r10 = hvx_Vuh_rsqrt10_Vuh(vin);
    HVX_Vector px = correction_for_rsqrts_Vuh(vin, r10);

    // now find 32*r10 -px, while saturating to 0  .. 0x7fff.

    // find 16 * r10 (which is <= 0x4000)
    HVX_Vector r10_x16 = Q6_Vh_vasl_VhR( r10,4);
    // now we want r10_x16 + (r10_x16-px), with the add saturating.
    return Q6_Vh_vadd_VhVh_sat( r10_x16,
    		Q6_Vh_vsub_VhVh( r10_x16, px ));
}

//
// 'scalar' reference for hvx_Vuh_rsqrt_Vuh, hvx_Vh_rsqrt_Vuh
//

static inline
int
ref_Ruh_rsqrt_Ruh( int vin )
{
	int r10 = ref_Ruh_rsqrt10_Ruh( vin);
	int rx = saturate_u16(r10 * r10 >> 4);
	int prod = rx * (uint16_t)vin;
	int pd = (int16_t)( prod >> 10);
	pd = (pd + 9)>>1;
	int px = (int16_t)Q6_R_vmpyh_RR_s1_rnd_sat( r10, pd);
	return 32*r10-px;
}

static inline
int
ref_Rh_rsqrt_Ruh( int vin )
{
	return saturate_i16( ref_Ruh_rsqrt_Ruh(vin));
}


//
// find the square roots of 64 uint32s, as uint16.
//
// 'adj' is a value in range (-1 .. 13);
// this scales results larger by a factor of (2^adj)
// (but they will still be saturated to 0 .. 0xFFFF)
//
static inline
HVX_Vector
hvx_Vuh_sqrt_VuwVuwI( HVX_Vector vin_even, HVX_Vector vin_odd , int adj)
{
	HVX_Vector sh_0 = Q6_Vuw_vcl0_Vuw( vin_even);
	HVX_Vector sh_1 = Q6_Vuw_vcl0_Vuw( vin_odd);

	sh_0 = Q6_V_vand_VV( sh_0, q6op_Vh_vsplat_R(0xFFFE));
	sh_1 = Q6_V_vand_VV( sh_1, q6op_Vh_vsplat_R(0xFFFE));
	// << both by even amount and extract upper 16 bits
	//
	HVX_Vector u16 = Q6_Vh_vshuffo_VhVh(
		Q6_Vw_vlsr_VwVw( vin_odd, sh_1),
		Q6_Vw_vlsr_VwVw( vin_even, sh_0));
	// those are all in range (0x4000 .. 0x7fff); find recip square root
	//
	HVX_Vector rsqrt = hvx_Vuh_rsqrt_Vuh( u16);
	// now mul by the original
	HVX_VectorPair pr = Q6_Wuw_vmpy_VuhVuh( u16, rsqrt);
	// and then >> by 14 + sh/2.
	HVX_Vector pr_0 = Q6_Vw_vasr_VwVw( Q6_V_lo_W( pr),  Q6_Vh_vavg_VhVh( sh_0, Q6_V_vzero()));
	HVX_Vector pr_1 = Q6_Vw_vasr_VwVw( Q6_V_hi_W( pr),  Q6_Vh_vavg_VhVh( sh_1, Q6_V_vzero()));

	HVX_Vector rbias = Q6_V_vsplat_R(  1 << (13-adj));
	pr_0 = Q6_Vw_vadd_VwVw( pr_0, rbias);
	pr_1 = Q6_Vw_vadd_VwVw( pr_1, rbias);
	return  Q6_Vuh_vasr_VwVwR_sat( pr_1, pr_0, 14-adj);

}
//
// find square root of 64 'uint16', as uint16
// adj is in range -8 .. 7; results are scaled by 2^adj.
// e.g. where input = 40000, you should get a result of
// 200 (with adj = 0) and 800 (with adj = 2); and 25 (with adj = -3).
// Note that this uses hvx_Vuh_rsqrt10_Vuh, so results are accurate
// to 9 or 10 significant bits.
//
static inline
HVX_Vector
hvx_Vuh_sqrt_VuhI(HVX_Vector vin, int adj)
{
	// normalize : << by an even amount
	HVX_Vector sh = Q6_Vuh_vcl0_Vuh( vin );
	sh = Q6_V_vand_VV( sh, q6op_Vh_vsplat_R(0xFFFE));
	HVX_Vector normin = Q6_Vh_vasl_VhVh( vin, sh);
	HVX_Vector rsqrt = hvx_Vuh_rsqrt10_Vuh( normin);
	HVX_VectorPair prod = Q6_Wuw_vmpy_VuhVuh( normin, rsqrt);
	// need to >> by 17 + sh/2 which is in range 17..24
	// to implement 'adj', do 16-adj+sh/2  (in range (9..31) and then
	// >>1 with rounding
	HVX_Vector tmp = q6op_Vh_vsplat_R(32-2*adj);
	sh = Q6_Vh_vavg_VhVh( tmp, sh);
	return Q6_Vh_vasr_VwVwR_rnd_sat(
		Q6_Vw_vlsr_VwVw( Q6_V_hi_W(prod), Q6_Vh_vshuffo_VhVh(sh,sh)),
		Q6_Vw_vlsr_VwVw( Q6_V_lo_W(prod), sh), 1 );
}
// reference 'scalar' version of hvx_Vuh_sqrt_VuhI
static inline int
ref_ruh_sqrt_RuhI(int vin, int adj)
{
	vin = (uint16_t)vin;
	int sh = (Q6_R_cl0_R(vin)-16) & ~1;		// 0,2, 16
	int normin = vin << sh;
	int rsqrt = ref_Ruh_rsqrt10_Ruh(normin);	// 512..1024
	int prod = normin*rsqrt;
	sh = (sh>>1) + 16- adj;		// 16-adj .. 23-adj
	prod >>= sh;
	return saturate_i16((prod+1)>>1);
}



//
// find the sum all of the elements in a single d32 slice; giving
// separate sums for each of 32 depth positions.
//   ptr = a 32-byte aligned pointer (i.e. skip width padding).
//   height = any ht >= 1
//   width = any width >= 1
//   height_stride = amount to add for next row
//
// The return value is 32 sums, each 32 bit, in a vector.
//
static inline
HVX_Vector
hvx_sum_hxw_d32_slice( uint8_t const * ptr, int height, int width, int height_stride )
{
	HVX_VectorPair allsum = Q6_W_vcombine_VV( Q6_V_vzero(),Q6_V_vzero());
	if( height < 1) return Q6_V_lo_W(allsum);	// compiler now assumes height >= 1
	HVX_VectorPred q0 = Q6_Q_vsetq_R( (int)(size_t)ptr);	// left mask (1 = don't process)

	// vec align the pointer
	uint8_t const * ptr_align = (uint8_t const *)( (size_t) ptr & ~(size_t)127);
	uint8_t const * ptr_end = ptr + 32*width;
	const int kb_1 = 0x01010101;
	const int kh_1 = 0x00010001;

	if( width > 4 ){
		// determine # of full vectors (this is actually 1 less than the # of vecs, and is >= 1
		int vecs_across = (unsigned)(ptr_end-ptr_align-1)/128u;
		HVX_VectorPred q1 = q6op_Q_vsetq2_R( (int)(size_t)ptr_end);	// right mask ( 0 = don't process )

		HVX_Vector extadd;
		// there are at least 2 per row ( vecs_across >= 1)
		for( int i =0; i < height; i ++){
			HVX_Vector const *rowp = (HVX_Vector const *)( ptr_align + height_stride * i);
			HVX_Vector vin0 = q6op_V_vand_QnV( q0, *rowp++ );
			HVX_Vector vin = Q6_Vb_vshuff_Vb(vin0);
			for(int i = 0; i < vecs_across-1; i++){
				extadd = Q6_Vh_vdmpy_VubRb( vin, kb_1);
				allsum = Q6_Wuw_vmpyacc_WuwVuhRuh( allsum, extadd, kh_1);
				vin = Q6_Vb_vshuff_Vb( *rowp++);
			}
			extadd = Q6_Vh_vdmpy_VubRb( vin, kb_1);
			allsum = Q6_Wuw_vmpyacc_WuwVuhRuh( allsum, extadd, kh_1);
			vin0 = q6op_V_vand_QV( q1, *rowp );
			vin = Q6_Vb_vshuff_Vb(vin0);
			extadd = Q6_Vh_vdmpy_VubRb( vin, kb_1);
			allsum = Q6_Wuw_vmpyacc_WuwVuhRuh( allsum, extadd, kh_1);
		}
	}else if( (((size_t)(ptr_end-1) ^ (size_t)ptr_align )& 128) != 0 ){	//  <= 4 but crosses a vector boundary
		q0 = Q6_Q_vsetq_R( -32*width);	// mask (1 = don't process)
		uint8_t const  *ptr0 = ptr_end - 128;			// read things into the *end* of the vector
		for( int i =0; i < height; i ++ ){
			HVX_Vector vin =  q6op_V_vldu_A( (HVX_Vector const*)( ptr0 + i*height_stride ));
			vin = Q6_Vb_vshuff_Vb(q6op_V_vand_QnV( q0, vin ));
			HVX_Vector extadd = Q6_Vh_vdmpy_VubRb( vin, kb_1);
			allsum = Q6_Wuw_vmpyacc_WuwVuhRuh( allsum, extadd, kh_1);
		}
	}else{		// single vector column
		q0 = Q6_Q_or_QQn( q0, q6op_Q_vsetq2_R( (int)(size_t)ptr_end));
		for( int i =0; i < height; i ++ ){
			HVX_Vector vin =  * (HVX_Vector const*)( ptr_align + i*height_stride );
			vin = Q6_Vb_vshuff_Vb(q6op_V_vand_QnV( q0, vin ));
			HVX_Vector extadd = Q6_Vh_vdmpy_VubRb( vin, kb_1);
			allsum = Q6_Wuw_vmpyacc_WuwVuhRuh( allsum, extadd, kh_1);
		}
	}
	// now reduce to one vector.
	// Each input is A0 ..A31 B0 .. B31  C0 .. C31   D0 .. D31
	// After shuffle:  A0 C0 A1 ... A31 C31 B0 D0 ... B31 D31
	//  After Q6_Vh_vdmpy_VubRb:   AC0 AC1 ... AC31 BD0 BD1  .. BD31
	// So the low part of allsum contains sums for AC0 AC2 .. AC30 BD0 BD2 .. BD30
	// and the high part contains the odd sums...
	//
	HVX_VectorPair shuff = Q6_W_vshuff_VVR( Q6_V_hi_W(allsum),  Q6_V_lo_W(allsum), -4 );
	//  now have AC0 AC1 AC2 .. AC31
	//           BD0 BD1 BD2 .. BD31
	// So just add them together
	//
	return Q6_Vw_vadd_VwVw( Q6_V_lo_W(shuff),Q6_V_hi_W(shuff)) ;
}



#endif // HVX_MATHOPS_H

