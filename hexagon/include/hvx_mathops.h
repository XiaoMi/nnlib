#ifndef HVX_MATHOPS_H
#define HVX_MATHOPS_H 1
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
#include "hvx_inlines.h"
#include <stdint.h>

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


/////////////////////////////////////////////////////////////
//
// reciprocal operation
// each lane of d must be in range 0x4000 .. 0x7fff
// result is in the same range; 2^29/d.
//
// the 'prec' parameter *must* be constant, so the compiler
// expands all the if's depending on it.
// This parameter sets the desired worst-case precision, in units of 1e-5
// relative to the result. The best precision is about 4.5e-5,
// obtained by prec = 5 or less.
//
//
//
//  for V60, V62:
//     prec < 17:
//         2-order approx with 2 newton-raphson refinement;  5e-5
//     prec >=17
//         2-order approx with newton-raphson refinement;  17e-5
//
//  for >=V65:
//     prec < 13:
//         2-order 4seg poly with newton-raphson refinement;  5e-5
//     prec >=13, < 52:
//         1-order 4seg with newton-raphson refinement;  13e-5
//     prec >= 52
//         2-order 4seg poly with no refinement             52e-5
//
//
//
// At least one refinement is used when prec < 52. When prec <=5,
// the last one is of higher accuracy. So prec=6 will be a bit faster, and a 
// tiny bit less accurate than prec = 5
//
// IMPORTANT: you will get different results on V65 vs non-v65,
// since different algos will be used for the same value of prec.
// if this is a problem, use a negative value for prec; the v65
// algos will be disabled.

//----------------------------------------------------------
// Performance measured across all 16384 valid input codes:
//   ref_result = (2.0^29)/input_code
//    err = result-ref_result
//    relative_err = err/ref_result
//  excess error = abs(err)-abs(ref_result-rounded(ref_result))
//  average error = sum(err)/16384
//
// 
// V60,V62 performance:      prec <= 5   6...16       >= 17
// max excess err, codes:      1.00      1.00         4.05
// average error, codes:       0.03     -0.08        -1.18
// max relative error, 1e-5:   4.5       5.0          17.2
// rms relative error, 1e-5:   1.6       1.8           6.8
// 
// >=V65 performance:        prec <= 5   6...12   13...51   >=52
// max excess err, codes:      1.00      1.00    3.00      15.92
// average error, codes:       0.03      0.03   -0.34      0.48
// max relative error, 1e-5:   4.5       5.6     13.3      51.6
// rms relative error, 1e-5:   1.6       2.1      3.4      23.2
//
// for comparison, an 'ideal' rounded-to-nearest result
// has a max rel error of 3.05e-5 and rms of 1.34e-5.
//-------------------------------------------------------------
//
static inline HVX_Vector __attribute__((always_inline,unused))
hvx_recip16_inline( HVX_Vector d , int prec )
{
	int preca = (prec < 0)?-prec: prec;
	HVX_Vector x0;
	int use_v65 = 0;
#if __HEXAGON_ARCH__ >= 65
	if( prec >=0 ){
		use_v65 = 1;
		HVX_Vector d4 = Q6_Vh_vasl_VhR ( d, 2);	// shift up, it's now 0 .. 0xFFFC
		HVX_Vector p;
		if( prec < 13 || prec >= 52 ){
			// 4-segment 2nd order: after one correction result is +/-1 count of best answer
			p = Q6_Vh_vlut4_VuhPh( d4, 0x09E20F69199B2E12ull);
			p = Q6_Vh_vmps_VhVhVuhPuh_sat( p, d4, 0x472E57516AB67D56ull );
			p = Q6_Vh_vmpa_VhVhVuhPuh_sat( p, d4, 0x736C794E7DE88001ull );
		}else{
			// linear version - result +/- 3 units after one correction
			p = Q6_Vh_vlut4_VuhPh( d4, 0xEE58E790DF56CAC2ull);
			p = Q6_Vh_vmpa_VhVhVuhPuh_sat( p, d4, 0x63516D7D75B78001ull );
		}
		x0 = Q6_Vh_vadd_VhVh_sat(p,p);
	}
#endif	
	if( !use_v65 ){
		// 2nd order poly in d2= (2*d-2)
		HVX_Vector d2 = Q6_Vh_vadd_VhVh( d, d ); // intentional overflow to negative
		HVX_Vector p = Q6_Vh_vmpy_VhRh_s1_rnd_sat( d2, 10598* 0x10001);
		p = Q6_Vh_vadd_VhVh_sat( p, q6op_Vh_vsplat_R( -5306 ));
		p = Q6_Vh_vmpy_VhVh_s1_rnd_sat( d2, p );
		x0 = Q6_Vh_vadd_VhVh_sat( p, q6op_Vh_vsplat_R( 16569 ));
	}
	// now he have result x0 ... need to iterate it once or twice
	// depending on prec.
	if( preca < 17 && !use_v65 ){
		HVX_Vector pd = Q6_Vh_vmpy_VhVh_s1_rnd_sat( d, x0);
		pd = Q6_Vh_vsub_VhVh_sat( pd, q6op_Vh_vsplat_R(1<<14 ));
		HVX_Vector pd15 = Q6_Vh_vadd_VhVh_sat( pd, pd );
		HVX_Vector xpd15 = Q6_Vh_vmpy_VhVh_s1_rnd_sat( pd15, x0 );
		x0 = Q6_Vh_vsub_VhVh_sat( x0, xpd15 );
	}
	if( preca < 52 || !use_v65 ){
		HVX_Vector pd15;
		if(  preca > 5){
			HVX_Vector pd = Q6_Vh_vmpy_VhVh_s1_rnd_sat( d, x0);
			pd = Q6_Vh_vsub_VhVh_sat( pd, q6op_Vh_vsplat_R(1<<14 ));
			pd15 = Q6_Vh_vadd_VhVh_sat( pd, pd );
		}else{		// 'high accuracy' iteration - pd15 has 1 more bit.
			HVX_VectorPair pd = Q6_Ww_vmpy_VhVh( d, x0 );
			pd15 = Q6_Vh_vasr_VwVwR( Q6_V_hi_W(pd),Q6_V_lo_W(pd),13);
			pd15 = Q6_Vh_vavg_VhVh_rnd( pd15, Q6_V_vzero());
		}
		HVX_Vector xpd15 = Q6_Vh_vmpy_VhVh_s1_rnd_sat( pd15, x0 );
		x0 = Q6_Vh_vsub_VhVh_sat( x0, xpd15 );
	}
	return x0;
}	


//
// This is given a vector of 32 x u32 and a uint32 scalar;
// it finds the exact 64-bit products and returns the lower and upper 32 in a
// HVX_Vector_x2. If you only want the upper 32, this function is good to use,
// the operations which generate the lower will be dropped if you don't  use it.
//
// - Start by finding four u16xu16->32 partial products.
// - Lower result:
//       add the two middle pps, then add with <<16 to lower pp.
// - upper result is trickier:
//    (1) add lower pp to one of the middles, with >> 16 (it won't overflow)
//  >=v65:
//    (2) average the two middle PPs (using Vuw average)
//    (3) add with >>15 to upper PP, that's the result.
//  <= v65
//    (2) add the middle PPs together; check for overflow
//    (3) add with >> 16 to upper PP;
//    (4) add 64K where overflow occurred.
//
static inline
HVX_Vector_x2  __attribute__((always_inline,unused))
find_u64_prod_VuwRuw( HVX_Vector input, uint32_t factor)
{
	int32_t lo_lo = Q6_R_combine_RlRl( factor,factor);
	int32_t hi_hi = Q6_R_combine_RhRh( factor,factor);

	// find PPs
	HVX_VectorPair prodH = Q6_Wuw_vmpy_VuhRuh( input, hi_hi);
	HVX_VectorPair prodL = Q6_Wuw_vmpy_VuhRuh( input, lo_lo);
	HVX_Vector pp_HH = Q6_V_hi_W( prodH);
	HVX_Vector pp_M0 = Q6_V_lo_W( prodH);
	HVX_Vector pp_M1 = Q6_V_hi_W( prodL);
	HVX_Vector pp_LL = Q6_V_lo_W( prodL);

	HVX_Vector_x2 result;

	result.val[0] = Q6_Vw_vaslacc_VwVwR(
			pp_LL,								// low product
			Q6_Vw_vadd_VwVw( pp_M0, pp_M1), 16);	// sum of middles << 16

	// now do upper.
	// >> pp_LL by 16 with zero-extend (it will be <= 0xFFFE)
	pp_LL = Q6_Vh_vshuffo_VhVh( Q6_V_vzero(), pp_LL);
	// add that to pp_M0; max sum is 0xFFFE0001 + 0xFFFE = 0xFFFEFFFF
	pp_M0 = Q6_Vw_vadd_VwVw( pp_M0, pp_LL);


#if __HEXAGON_ARCH__ >= 65
	// average the two pp_M together (without rounding, of course)...
	HVX_Vector pp_M = Q6_Vuw_vavg_VuwVuw( pp_M0, pp_M1);
	result.val[1] = Q6_Vw_vadd_VwVw(		// add to ppHH with >>15 (zero extend)
			pp_HH,
			Q6_Vuw_vlsr_VuwR( pp_M, 15));

#else
	// add middle PP's together
	HVX_Vector pp_M = Q6_Vw_vadd_VwVw( pp_M0, pp_M1);
	// overflow iff result < either input (or both)
	HVX_VectorPred pp_oflo = Q6_Q_vcmp_gt_VuwVuw( pp_M0, pp_M);
	// >> the sum by 16 with zero extend
	pp_M = Q6_Vh_vshuffo_VhVh( Q6_V_vzero(), pp_M);
	// meanwhile add 64K to pp_H where overflow occurred
	pp_HH = Q6_Vw_vadd_VwVw( pp_HH, Q6_V_vand_QR( pp_oflo, 0x00010000));
	result.val[1] = Q6_Vw_vadd_VwVw(pp_HH, pp_M);
#endif
	return result;
}






/////////////////////////////////////////////////////////////
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

// return reciprocal sqrt with 2^31 scale factor, scale = 31
// vin is a signed word vector (leave the highest bit 0), which represents a 16-bit unsigned value (vin >> 15)
// the 16-bit value has to be in the range of [0x4000, 0x10000)
// select is a word vector that contains 0s and 1s.
// If vin is from a vin' that is shifted left, and the number of bits it's shifted is even, the corresponding word in select is 0
// the number of bits it's shifted is odd, the corresponding word in select is 0
// Using Newton-Raphson iteration:
// in each iteration, x = (3/2) * x - (input/2) * (x^3)
static inline
HVX_Vector rsqrt_newton_hvx(struct nn_graph *nn, HVX_Vector vin, HVX_Vector select) {

    HVX_Vector half_three = Q6_V_vsplat_R((uint32_t)1610612736);   //1.5 << 30; should << 31, but leave 1 bit for the sign. should be 3221225472
    HVX_Vector half_input = Q6_Vw_vasr_VwR(vin, 1);// the orginal 16-bit input << (scale - 16) before calling this function, then >> 1

    HVX_Vector x = Q6_V_vsplat_R(8375186);   //0.0039 << 31. 0.0039 is the initial x when input is 65535, which is the max possible input
                                                //initial x needs to be smaller than res. when an input < 65535, its res is bigger
    HVX_Vector x3 = Q6_V_vzero(); //x^3
    HVX_Vector tmp1 = Q6_V_vzero();
    HVX_Vector tmp2 = Q6_V_vzero();
    HVX_Vector v_16 = Q6_V_vsplat_R(16);
    HVX_Vector v_1 = Q6_V_vsplat_R(1);

    HVX_Vector v_2div_sqrt2 = Q6_V_vsplat_R((uint32_t) 759250124);// 2/sart(2) << 30

    for(int i = 0; i < 4; ++i) {
        x3 = q6op_Vw_vmpy_VwVw_s1_rnd_sat(x, x);// x * x >> 31
        x3 = q6op_Vw_vmpy_VwVw_s1_rnd_sat(x, x3);

        tmp1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat(half_three, x);
        tmp1 = Q6_Vw_vasl_VwVw(tmp1, v_1);//<<1 since half_three only has 30 bits scale factor

        tmp2 = q6op_Vw_vmpy_VwVw_s1_rnd_sat(half_input, x3);
        tmp2 = Q6_Vw_vasl_VwVw(tmp2, v_16);// << 16 since half_input is supposed to be << 31, but it has only been << 15
        x = Q6_Vw_vsub_VwVw(tmp1, tmp2);
    }
    HVX_Vector odd_x = q6op_Vw_vmpy_VwVw_s1_rnd_sat(v_2div_sqrt2, x);//* 2/sart(2) for rounding
    odd_x = Q6_Vw_vasl_VwVw(odd_x, v_1);//to get 31 bit scale factor
    x = Q6_V_vmux_QVV(select, odd_x, x);

    return x;
}
/* float ref Newton-Raphson iteration
static inline
float rsqrt_newton(float vin) {
    float x = 0.0039;
    float x3 = 0;
    for(int i = 0; i < 4; ++i) {
        x3 = x * x * x;
        x = 1.5 * x - 0.5 * vin * x3;
        printf(" x = %f \n", x);
    }
    return x;
}*/

#endif // HVX_MATHOPS_H

