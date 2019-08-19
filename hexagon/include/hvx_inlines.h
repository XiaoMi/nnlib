
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

#ifndef HVX_INLINES_H_
#define HVX_INLINES_H_

#if defined(__hexagon__)
#include "locale.h"
#ifdef LC_COLLATE_MASK		// this is how I'm detecting 8.0 compiler...
#define HEXAGON_COMPILER_GE_8_0
#endif
#include <stdlib.h>
#include "hexagon_types.h"
#include "hexagon_circ_brev_intrinsics.h"

// this is how I'm detecting 8.1 compiler
// it's not reliable since 8.0.10 has this (but 8.0.05 does not)
#if defined(Q6_vmaskedstorenq_QAV)
#define HEXAGON_COMPILER_GE_8_1
#else
#ifdef HEXAGON_COMPILER_GE_8_0
#define HEXAGON_COMPILER_IS_8_0
#endif
#endif
#endif //__hexagon__

#include "hvx_hexagon_protos.h"
#include "nn_graph_builtin.h"
#define HVX_INLINE_ALWAYS inline __attribute__((unused,always_inline))

typedef struct
{
	HVX_Vector val[2];
} HVX_Vector_x2;

typedef struct
{
	HVX_Vector val[3];
} HVX_Vector_x3;
typedef struct
{
	HVX_Vector val[4];
} HVX_Vector_x4;

typedef struct {
	HVX_VectorPair val[2];
} HVX_VectorPair_x2;
typedef struct {
	HVX_VectorPair val[3];
} HVX_VectorPair_x3;
typedef struct {
	HVX_VectorPair val[4];
} HVX_VectorPair_x4;

//
// Predicate shuffle - emulated for v60
//
static HVX_INLINE_ALWAYS
HVX_VectorPred q6op_Qb_vshuffe_QhQh( HVX_VectorPred Qs, HVX_VectorPred Qt )
{
#if	__HEXAGON_ARCH__ >= 62
	return Q6_Qb_vshuffe_QhQh(Qs,Qt);
#else
	HVX_VectorPred even_b_lanes = Q6_Q_vand_VR( Q6_V_vsplat_R(0x010001), 0x10001);

	return Q6_Q_or_QQ( Q6_Q_and_QQ( Qt,even_b_lanes), Q6_Q_and_QQn( Qs,even_b_lanes));
#endif
}
static HVX_INLINE_ALWAYS
HVX_VectorPred q6op_Qh_vshuffe_QwQw( HVX_VectorPred Qs, HVX_VectorPred Qt )
{
#if	__HEXAGON_ARCH__ >= 62
	return Q6_Qh_vshuffe_QwQw(Qs,Qt);
#else
	HVX_VectorPred even_h_lanes = Q6_Q_vand_VR( Q6_V_vsplat_R(0x0101), 0x101);

	return Q6_Q_or_QQ( Q6_Q_and_QQ( Qt,even_h_lanes), Q6_Q_and_QQn( Qs,even_h_lanes));
#endif
}
// vector & pred - emulated for v60
//

static HVX_INLINE_ALWAYS
HVX_Vector q6op_V_vand_QV(HVX_VectorPred Qv, HVX_Vector Vu)
{
#if	__HEXAGON_ARCH__ >= 62
    return Q6_V_vand_QV( Qv, Vu);
#else
    return Q6_V_vmux_QVV( Qv, Vu, Q6_V_vzero());
#endif
}

static HVX_INLINE_ALWAYS
HVX_Vector q6op_V_vand_QnV(HVX_VectorPred Qv, HVX_Vector Vu)
{
#if	__HEXAGON_ARCH__ >= 62
    return Q6_V_vand_QnV( Qv, Vu);
#else
    return Q6_V_vmux_QVV( Qv, Q6_V_vzero(),Vu);
#endif
}

// emulate Q6_V_vand_QnR for V60
static HVX_INLINE_ALWAYS
HVX_Vector q6op_V_vand_QnR(HVX_VectorPred Qv, int Rt)
{
#if	__HEXAGON_ARCH__ >= 62
    return Q6_V_vand_QnR( Qv, Rt);
#else
    return Q6_V_vand_QR( Q6_Q_not_Q(Qv), Rt);
#endif
}
//
// emulate Q6_Ww_vmpyacc_WwVhRh for < V65
// (previous arch can only do this with 'sat').
//
static HVX_INLINE_ALWAYS
HVX_VectorPair q6op_Ww_vmpyacc_WwVhRh(HVX_VectorPair Vxx, HVX_Vector Vu, int Rt)
{
#if	__HEXAGON_ARCH__ >= 65
    return Q6_Ww_vmpyacc_WwVhRh( Vxx, Vu,Rt);
#else
    return Q6_Ww_vadd_WwWw( Vxx, Q6_Ww_vmpy_VhRh(Vu,Rt));
#endif
}

//
// 32x32 fractional multiply - expands to two ops
//  equiv to :
//    p  = (a*b + (1<<30)) >> 31     [with rounding]
//    p  = a*b >> 31     			[without rounding]
// The 'sat' only takes effect when both inputs
// are -0x80000000 and causes the result to saturate to 0x7fffffff
//
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vw_vmpy_VwVw_s1_rnd_sat( HVX_Vector vu, HVX_Vector vv) {
	return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift( Q6_Vw_vmpye_VwVuh( vu, vv), vu, vv );
}
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vw_vmpy_VwVw_s1_sat( HVX_Vector vu, HVX_Vector vv) {
	return Q6_Vw_vmpyoacc_VwVwVh_s1_sat_shift( Q6_Vw_vmpye_VwVuh( vu, vv), vu, vv );
}

//
// Splat from 16 bits
// Note: you can use this for constants, e.g. q6op_Vh_vsplat_Rh(0x12), the compiler
// will just simplify it to Q6_V_vsplat_R(0x120012) for v60.
//
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vh_vsplat_R(int r)
{
#if __HEXAGON_ARCH__ >= 62
    return Q6_Vh_vsplat_R(r);
#else
    return Q6_V_vsplat_R( Q6_R_combine_RlRl(r,r));
#endif
}
// splat from 8 bits
static HVX_INLINE_ALWAYS
HVX_Vector q6op_Vb_vsplat_R( int n)
{
#if __HEXAGON_ARCH__ >= 62
	return Q6_Vb_vsplat_R(n);
#else
	return Q6_V_vsplat_R( Q6_R_vsplatb_R(n));
#endif
}


// v62+ have the 'I' variant of these vlut ops, in which the final
// 'R' parameter is a fixed constant in range 0..7; use these
// for portability. Make sure that the parameter is a compile-time constant in
// range 0..7, or the code will build on v60, and fail to build on v62+
// IMHO the compiler should map the R variant to these automatically when
// possible.
//
#if __HEXAGON_ARCH__ >= 62
#define q6op_Wh_vlut16_VbVhI     Q6_Wh_vlut16_VbVhI
#define q6op_Wh_vlut16or_WhVbVhI Q6_Wh_vlut16or_WhVbVhI
#define q6op_Vb_vlut32_VbVbI     Q6_Vb_vlut32_VbVbI
#define q6op_Vb_vlut32or_VbVbVbI Q6_Vb_vlut32or_VbVbVbI
#else
#define q6op_Wh_vlut16_VbVhI     Q6_Wh_vlut16_VbVhR
#define q6op_Wh_vlut16or_WhVbVhI Q6_Wh_vlut16or_WhVbVhR
#define q6op_Vb_vlut32_VbVbI     Q6_Vb_vlut32_VbVbR
#define q6op_Vb_vlut32or_VbVbVbI Q6_Vb_vlut32or_VbVbVbR
#endif

//////////////////////////////////////////////////////////////////////
// 'reducing shifts' which take a VectorPair instead of two vectors
//////////////////////////////////////////////////////////////////////
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vasr_WwR ( HVX_VectorPair a, int sh )
{
    return Q6_Vh_vasr_VwVwR( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vasr_WwR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vh_vasr_VwVwR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vasr_WwR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
#if __HEXAGON_ARCH__ >= 62
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vasr_WwR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vuh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
#endif
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vasr_WwR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vuh_vasr_VwVwR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vasr_WhR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vub_vasr_VhVhR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vasr_WhR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vub_vasr_VhVhR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

#if __HEXAGON_ARCH__ >= 62
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vasr_WhR_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vb_vasr_VhVhR_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}
#endif


static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vasr_WhR_rnd_sat ( HVX_VectorPair a, int sh )
{
    return Q6_Vb_vasr_VhVhR_rnd_sat( Q6_V_hi_W(a), Q6_V_lo_W(a), sh);
}

//////////////////////////////////////////////////////////////////////
// 'round' which take a VectorPair instead of two vectors
//////////////////////////////////////////////////////////////////////

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vround_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vh_vround_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vround_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vuh_vround_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#if __HEXAGON_ARCH__ >= 62

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vround_Wuw_sat ( HVX_VectorPair a )
{
    return Q6_Vuh_vround_VuwVuw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#endif

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vround_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vb_vround_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vround_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vub_vround_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#if __HEXAGON_ARCH__ >= 62
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vround_VuhVuh_sat ( HVX_VectorPair a )
{
    return Q6_Vub_vround_VuhVuh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
#endif

//////////////////////////////////////////////////////////////////
// pack operations (with VectorPair input instead of two vectors)
/////////////////////////////////////////////////////////////////
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vpacke_Wh ( HVX_VectorPair a )
{
    return Q6_Vb_vpacke_VhVh( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vpacke_Ww ( HVX_VectorPair a )
{
    return Q6_Vh_vpacke_VwVw( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}

static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vpacko_Wh ( HVX_VectorPair a )
{
    return Q6_Vb_vpacko_VhVh( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vpacko_Ww ( HVX_VectorPair a )
{
    return Q6_Vh_vpacko_VwVw( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vub_vpack_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vub_vpack_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vb_vpack_Wh_sat ( HVX_VectorPair a )
{
    return Q6_Vb_vpack_VhVh_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vuh_vpack_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vuh_vpack_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}
static HVX_INLINE_ALWAYS HVX_Vector q6op_Vh_vpack_Ww_sat ( HVX_VectorPair a )
{
    return Q6_Vh_vpack_VwVw_sat( Q6_V_hi_W(a), Q6_V_lo_W(a) );
}

// v60 doesn't have vsetq2
static HVX_INLINE_ALWAYS
HVX_VectorPred q6op_Q_vsetq2_R( int rval)
{
#if __HEXAGON_ARCH__ >= 62
	return Q6_Q_vsetq2_R(rval);
#else
    HVX_VectorPred result = Q6_Q_vsetq_R(rval);
    static const unsigned VECN = sizeof(HVX_Vector);
// Q6_vmaskedstorenq_QAV is being used to detect compiler >= 8.1.02
// earlier ones will ICE on the short version of the below (generating
// VecPred based on scalar condition)
//
#if defined(Q6_vmaskedstorenq_QAV)
    if ( ((unsigned)rval &(VECN-1)) == 0 ) result = Q6_Q_not_Q(result);
#else
    //
    // if rval is a multiple of VECN, force to all ones, otherwise
    // leave it alone.
    //
    unsigned ones = -1;
    unsigned tcond = ( ((unsigned)rval &(VECN-1)) == 0 )? ones: 0;
    result =  Q6_Q_vandor_QVR( result, Q6_V_vsplat_R(ones),tcond );
#endif
    return result;
#endif
}



//
//
// conditional vector store
//   void q6op_vstcc_[n]QAV[_nt] ( cond, addr, vec );
// These do *not* work in HVXDBL on 7.4.01 (compiler will abort with getRegForInlineAsmConstraint Unhandled data type)
// occurs if these are used (it's ok to just have them in the header though). Seem to be ok with 8.0.05
//
#if defined(__hexagon__)
static HVX_INLINE_ALWAYS void q6op_vstcc_QAV(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef Q6_vmaskedstoreq_QAV
    Q6_vmaskedstoreq_QAV( cond, addr, v );
#else
  __asm__ __volatile__( "if(%0) vmem(%1)=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
}
static HVX_INLINE_ALWAYS void q6op_vstcc_QnAV(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef Q6_vmaskedstorenq_QAV
    Q6_vmaskedstorenq_QAV( cond, addr, v );
#else
  __asm__ __volatile__( "if(!%0) vmem(%1)=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
}
static HVX_INLINE_ALWAYS void q6op_vstcc_QAV_nt(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef Q6_vmaskedstorentq_QAV
    Q6_vmaskedstorentq_QAV( cond, addr, v );
#else
  __asm__ __volatile__( "if(%0) vmem(%1):nt=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
}
static HVX_INLINE_ALWAYS void q6op_vstcc_QnAV_nt(  HVX_VectorPred cond, HVX_Vector *addr, HVX_Vector v )
{
#ifdef Q6_vmaskedstorentnq_QAV
    Q6_vmaskedstorentnq_QAV( cond, addr, v );
#else
  __asm__ __volatile__( "if(!%0) vmem(%1):nt=%2;" : : "q"(cond),"r"(addr),"v"(v) : "memory");
#endif
}
#endif

//
// unaligned vector load
// Done by stating that the compiler should read a vector within a packed
// data structure.
//
static HVX_INLINE_ALWAYS HVX_Vector q6op_V_vldu_A(  HVX_Vector const *addr )
{
    struct varr { HVX_Vector v;}__attribute__((packed)) const *pp;
    pp = (struct varr const *)addr;
    return pp->v;
}
// unaligned vector store.
static HVX_INLINE_ALWAYS void q6op_vstu_AV(  HVX_Vector *addr, HVX_Vector v )
{
    struct varr { HVX_Vector v;}__attribute__((packed)) *pp;
    pp = (struct varr *)addr;
    pp->v = v;
}
//////////////////////////////////////////////////////////

// this stores the first n bytes from vector vin to address 'addr'.
// n must be in range 1..128, addr may have any alignment; does one or
// two masked stores.
// Note: if the last byte falls in lane 127, the operation on V60
//  will do a byte-masked store to the next vector with all lanes
//  disabled. on >=V62 it will only do stores where needed.
//
static inline void q6op_vstu_variable_ARV( void * addr, int n, HVX_Vector vin)
{
	vin = Q6_V_vlalign_VVR( vin, vin, (size_t)addr); //rotate as needed.
	unsigned left_off = (size_t)addr & 127;
	unsigned right_off = left_off + n;
	HVX_VectorPred qL_not = Q6_Q_vsetq_R( (size_t)addr );
#if __HEXAGON_ARCH__ >= 62
	HVX_VectorPred qR = Q6_Q_vsetq2_R( right_off );
	if( right_off > 128 ){
#else
	HVX_VectorPred qR = Q6_Q_vsetq_R( right_off );
	if( right_off >= 128 ){		// v60 does a null store here when right_off = 128.
#endif
		q6op_vstcc_QAV( qR, (HVX_Vector*)addr + 1, vin);
		qR = Q6_Q_vcmp_eq_VbVb( vin,vin);	// all 1's
	}
	qL_not = Q6_Q_or_QQn( qL_not, qR );
	q6op_vstcc_QnAV( qL_not,(HVX_Vector*)addr, vin );
}

// this is called with a dest pointer, two vectors, and 'bytes' in range 1..256.
// The first 'bytes' bytes from the vectors (v0 followed by v1) will be stored at the address, using
// unaligned and masked stores as needed. If bytes <=0, nothing is stored; if bytes > 256
// the effect is the same as bytes == 256 (all stored).
// Done as a function; in many cases this is done just before a function returns, so it's not
// time critical and it can be done as a a tail call and save some code bloat.
//
void hvx_store_vec_x2_unaligned ( void * addr, HVX_Vector v0, HVX_Vector v1, int bytes);

// HVX_8_0_FAKEDEP_VM( vector_var, mem_location)
// .. creates a fake dependency between the vector and the memory location.
// It generates no code, but constrains instruction reordering. Some loops will crash 8.0 unless one of these
// is placed after the loop (generally, this occurs when the hvx operations after the loop  resemble those
// at the top of the loop).
// The macro can be eliminated when using >= 8.1
// (however 8.1 detection is not reliable)


#if defined ( __hexagon__ ) && defined( HEXAGON_COMPILER_GE_8_0)
#define HVX_8_0_FAKEDEP_VM( vec, mem)	asm ("/*%0 %1*/": "=v"(vec), "=m"(mem): "0"(vec))
#else
#define HVX_8_0_FAKEDEP_VM( vec, mem)
#endif

// Create a fake 'store' operation which will cause subsequent memory operation wait for its completion
// Normally used to sync up vscatter operation store on VTCM, which are delayed through a routing network,
// and there is no automatic coherence mechanism. The this fake operation to sync up is required.
static HVX_INLINE_ALWAYS void q6op_scatter_release_A( volatile void * addr )
{
#if defined(__hexagon__)
	__asm__ __volatile__( "vmem(%0+#0):scatter_release"::"r"(addr):"memory");
#endif
}

/////////////////////////////////////////////////////////////////////////////////
//
// 'd32' utilities
//

// these are in hvx_constants.c
extern const uint8_t const_Count128[128] __attribute__((aligned(128)));	/// {0..127}
extern const uint8_t const_Count64[128] __attribute__((aligned(128)));	/// {0..63, 0..63}
extern const uint8_t const_Count32[128] __attribute__((aligned(128)));	/// {0..31, 0..31, 0..31, 0..31}
extern const uint8_t const_InverseCount128[128] __attribute__((aligned(128)));	/// {127..0}
extern const int16_t lut_Log2_and_Pow2[6*64] __attribute__((aligned(128)));
extern const int16_t lut_fracdivide_k1k2[6*64] __attribute__((aligned(128)));
extern const int16_t lut_root_recip[64] __attribute__ ((aligned(128)));
extern const float lut_reciprocal[128] __attribute__ ((aligned(128)));  // 1.0/1.0   ... 1.0/128.
extern const int32_t lut_reciprocal_i32[128] __attribute__ ((aligned(128)));    // 1/1 .. 1/128 w/31 frac bits

extern const int16_t tanh_16_cubic_lookup[64] __attribute__( (aligned(128)));

extern const uint32_t const_Deal3_table[7*3][32] __attribute__( (aligned(128)));
//
// tables for 'fracdivide'
// two lookup tables of 32 x int16 (k1 in even lanes, k2 in odd)
//
// this generates a vector which can be applied using vdelta, to rotate  n within each group of 32
// lanes (rotate right, as in vror).
// The same value, applied with vrdelta, will rotate by n left within each group.
//
// formula: lane i (for i in 0..31) is ((i-n) ^ i ) &31; this is repeated across all four groups.
//

static inline HVX_Vector __attribute__((unused,always_inline))
hvx_make_rorcode_in_d32( int n){
	HVX_Vector vc32 = *(HVX_Vector const*)const_Count32;		// i
	HVX_Vector t = Q6_Vb_vsub_VbVb( vc32,  q6op_Vb_vsplat_R(n));
	return Q6_V_vand_VV( Q6_V_vxor_VV( t,vc32),q6op_Vb_vsplat_R(31));
}

// constructs a condition with lanes 0 .. n-1 'on', in each d32 section
// if n is -128...0, all lanes are off
// if n is 32..127, all lanes are on.
static inline HVX_VectorPred __attribute__((unused,always_inline))
hvx_make_d32_mask( int n )
{
	HVX_Vector vc32 = *(HVX_Vector const*)const_Count32;		// i
	return Q6_Q_vcmp_gt_VbVb( q6op_Vb_vsplat_R(n), vc32);	// n > i
}


// make a vectorpred which covers lanes lo ...hi-1 within each group of 32 lanes
// lo must be 0..31; hi is 1..32 and hi > lo.
// done as   (i >= lo) & (i < hi )
//   = ( i > lo-1) & (hi > i )
static inline HVX_VectorPred __attribute__((unused,always_inline))
hvx_make_d32_range_mask( int lo, int hi )
{
	HVX_Vector vc32 = *(HVX_Vector const*)const_Count32;		// i
	return Q6_Q_vcmp_gtand_QVbVb(
			Q6_Q_vcmp_gt_VbVb( q6op_Vb_vsplat_R(hi), vc32),	// hi > i
			vc32, q6op_Vb_vsplat_R(lo-1) );					// i > (lo-1)

}

//
// make a vector, to be used with vdelta, to splat one depth lane to others,
// and/or to splat one width unit (group of 32 byte lanes) to others.
// It will also work with a vrdelta.
//
// width_unit:  0..3  to splat that width unit to others; -1 to pass all 4
// depth_unit:  0..31 to splat that depth unit to others; -1 to pass all 32
//
// So, width_unit = depth_unit = -1 gives a zero result.
//
static inline HVX_Vector  __attribute__((unused,always_inline))
hvx_make_lateral_splat(int width_unit, int depth_unit )
{
	// each output lane i  is (i^k) & m, where
	//  k [4:0] is 5 lsbs of depth_unit
	//  k [6:5] is 2 lsbs of width_unit
	//  m [4:0] is 0x1F if depth_unit >= 0, and 0 otherwise
	//  m [6:5] is 0x3 if width_unit >=0, and 0 otherwise;
	// bits of k are don't-care where m is 0.
	//
	int kval = (depth_unit & 0x1f) | (width_unit <<5);
	int mval = ((depth_unit >= 0)? 0x1f:0) | ((width_unit >= 0)? 0x60:0);
	HVX_Vector vc128 = *(HVX_Vector const*)const_Count128;		// i
	return Q6_V_vand_VV(  Q6_V_vxor_VV( vc128, q6op_Vb_vsplat_R(kval)), q6op_Vb_vsplat_R(mval));
}


//
// given vmin0 and vmin1 (representing the i16 min's arising from 128 byte lanes,
//  split by even and odd);
// and vmax0, vmax1 representing the max values, reduce to a single
// vector containing 32 records of { ~minr, maxr }
// .. where maxr[2*i] was reduced from vmax0[i + k*16], (k = 0,1,2,3)
//          maxr[2*i+1] was reduced from vmax1[i + k*16], (k = 0,1,2,3)
//
// I.e. each of the 32 output records arises from a particular 'depth lane' within
//  the original 128, so it's possible to 'mask' the results in the depth dimension
//  before proceeding in the reduction.
//
//
static inline HVX_Vector  __attribute__((unused,always_inline))
hvx_reduce_minmax_h_over_width(
	HVX_Vector vmin0, HVX_Vector vmax0,
	HVX_Vector vmin1, HVX_Vector vmax1 )
{
	// first, shuffle the odd & even together (full shuffle)
	// then reduce across the halves of the result
	HVX_VectorPair vshuf = Q6_W_vshuff_VVR(  vmin1, vmin0, -2 );
	vmin0 = Q6_Vh_vmin_VhVh( Q6_V_hi_W( vshuf), Q6_V_lo_W( vshuf ));
	vshuf = Q6_W_vshuff_VVR(  vmax1, vmax0, -2 );
	vmax0 = Q6_Vh_vmax_VhVh( Q6_V_hi_W( vshuf), Q6_V_lo_W( vshuf ));
	// vmin0 has 32x{min}, and again 32x{min}, each group follows
	// the depth structure. Same for vmax0.
	//  shuffle ~vmin0 and vmax0 together, and reduce across..

	vshuf = Q6_W_vshuff_VVR(  vmax0, Q6_V_vnot_V(vmin0), -2 );
	return Q6_Vh_vmax_VhVh( Q6_V_hi_W( vshuf), Q6_V_lo_W( vshuf ));
}
//
//
// this is like hvx_reduce_minmax_h_over_width but it
// finishes the reduction with depth gating:
//  all 32-byte lanes in output will contain the same {~min,max} reduced
// across the whole input, but depth lanes which are not in range
//   d_lo <= d < d_hi
// are excluded.
//
// normally 0 <= d_lo < d_hi <=32
// for instance:
//   d_lo = 1, d_hi = 30:
//     => exclude even depth lanes 0,30 and odd lane 31, so
//
//     - h lanes 0,15,   16,31,   32,47,   48,63   are  disabled from vmin0,vmax0
//     - h lanes   15,      31,      47,      63   are disabled from vmin1, vmax1.
//
static inline HVX_Vector __attribute__((unused,always_inline))
hvx_reduce_minmax_h_depthrange(
	HVX_Vector vmin0, HVX_Vector vmax0,
	HVX_Vector vmin1, HVX_Vector vmax1,
	int d_lo, int d_hi )
{
	HVX_Vector vminmax = hvx_reduce_minmax_h_over_width( vmin0, vmax0, vmin1, vmax1);

	// now vminmax has 32 x { ~vmin, vmax} each of which correponds to a depth lane.
	// now, replace the 'disabled' lanes with {0x8000,0x8000}
#if	__HEXAGON_ARCH__ >= 62
	HVX_VectorPred qnotright = Q6_Q_vsetq2_R(4*d_hi);
#else
	HVX_Vector vc128 = *(HVX_Vector const*)const_Count128;	// 0 .. 127
	// compare  {0x03020100, 0x07060504 ... 0x7f7e7d7c } to 0x04000000 * d_hi
	HVX_VectorPred qnotright = Q6_Q_vcmp_gt_VuwVuw(Q6_V_vsplat_R( d_hi << 26), vc128);
#endif
	HVX_Vector qok = Q6_Q_and_QQn(  qnotright, Q6_Q_vsetq_R(4*d_lo));
	// replace all the not-ok lanes...
	vminmax = Q6_V_vmux_QVV( qok, vminmax, Q6_V_vsplat_R( 0x80008000));

	int k = 4;
	int i;
	//finish the operation across all lanes
	for( i = 0; i < 5 ; i++){
		HVX_VectorPair sh = Q6_W_vshuff_VVR( vminmax, vminmax, k);
		k <<= 1;
		vminmax = Q6_Vh_vmax_VhVh( Q6_V_hi_W(sh), Q6_V_lo_W(sh));
	}
	return vminmax;
}



// input: vector of 128 x u8, considered to be four groups of 32
// - find the max (ub) in each group; the output has this value
// replicated over all 32 in each group.
//
static inline HVX_Vector
reduce_in_quadrants_vmax_Vub( HVX_Vector vin )
{
	HVX_VectorPair tmp = Q6_Wb_vshuffoe_VbVb( vin, vin );
	HVX_Vector val = Q6_Vub_vmax_VubVub( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));   // 16 of 2 now
	tmp = Q6_Wh_vshuffoe_VhVh( val, val );
	val = Q6_Vub_vmax_VubVub( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));	  // 8 of 4 now
	tmp = Q6_W_vshuff_VVR( val, val ,32-4 );
	val = Q6_Vub_vmax_VubVub( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));	  // 4 of 8 now
	tmp = Q6_W_vshuff_VVR( val, val ,32-4 );
	val = Q6_Vub_vmax_VubVub( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));	  // 2 of 16 now
	tmp = Q6_W_vshuff_VVR( val, val ,32-4 );
	val = Q6_Vub_vmax_VubVub( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));	  // 1 of 32 now
	return val;
}
// input: vector of 32 xi32, considered to be four groups of 8
// - find the sum in each group; the output has this value
// replicated over all 8 in each group.
// e.g. { 0, 1, 2, 3, 4, 5, 6, 7,   10, 20, 30, 40, 50, 60, 70, 80, ...
// ->
// { 28, 28, 28, 28, 28, 28, 28, 28,   360,360,360,360,360,360,360,360,...
//
//
static inline HVX_Vector
reduce_in_quadrants_vadd_Vw( HVX_Vector vin )
{
	HVX_VectorPair tmp = Q6_W_vshuff_VVR( vin, vin ,32-4 );
	HVX_Vector msum = Q6_Vw_vadd_VwVw( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));	  // 4 of 2 now
	tmp = Q6_W_vshuff_VVR( msum, msum ,32-4 );
	msum = Q6_Vw_vadd_VwVw( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));	  // 2 of 4 now
	tmp = Q6_W_vshuff_VVR( msum, msum ,32-4 );
	msum = Q6_Vw_vadd_VwVw( Q6_V_hi_W(tmp), Q6_V_lo_W(tmp));	  // 1 of 8 now
	return msum;
}


/////////////////////////////////////////////////////////

// Facility to do 3-way shuffle/deal, with elementsizes
// that are power-of-two up to 64 bytes.
// This defines several functions that return structs
// containing vectors; all of these functions should
// be inlined for proper performance.
//
//
// how to do 3-way deal of n-byte elements, with n being a power of 2:
// First, sort across registers:
// ==> if the # of elements in a vector is an *even* power of 2:
//
//  v0 ABCAB..ABCA
//  v1 BCABC..BCAB
//  v2 CABCA..CABC
//    (a) swap v1/v2 in lanes :x::x..:x::
//  v0 ABCAB..ABCA
//  v1 BAABA..BAAB
//  v2 CCBCC..CCBC
//     (b) swap v0/v1 in lanes :xx:x..:xx:
//  v0 AAAAA..AAAA
//  v1 BBCBB..BBCB
//  v2 CCBCC..CCBC
//    (c) swap v1/v2 in lanes ::x::..::x:
//  v0 AAAAA..AAAA
//  v1 BBBBB..BBBB
//  v2 CCCCC..CCCC
//
// ==> if the # of elements in a vector is an *odd* power of 2:
//
//  v0 ABCAB..BCAB
//  v1 CABCA..ABCA
//  v2 BCABC..CABC
//    (a) swap v1/v2 in lanes x:xx:..x:xx
//  v0 ABCAB..ABCA
//  v1 BAABA..BAAB
//  v2 CCBCC..CCBC
// The rest is the same as before.
//
//
//  next step is to 'unroll' the elements.
//  Cases:
//   N = 4:   | A0 A3 A2 A1 | B1 B0 B3 B2 | C2 C1 C0 C3
//   N = 8:   | A0 A3 A6 A1 A4 A7 A2 A5 | B5 B0 B3 B6 B1 B4 B7 B2 | C2 C5 .. C7
//   N = 16:  | A0 A11 .. A5 | B5 B0 .. B10| C10 C5 .. C15 
//   N = 32:  | A0 A11 .. A21| B21 B0 .. B10|C10 C21 .. C31
//
// The mapping to unroll A can always be described as (lane 3*i mod N)-> lane i
//    To unroll B: (lane 3*i+1 mod N)-> lane i
//    To unroll C: (lane 3*i+2 mod N)-> lane i
//
// so in general we need three 'VectorPred' q0,q1,q2 and three 'rdelta'
// controls; const_Deal3_table consists of the 3 delta controls for all
// supported elementsize; the three q's are stored in bit 7.
// The 'shuffle3' is the reverse process (using delta).
//
// pack of values needed for the 3x shuffle/deal
// ops. these depend on elementsize. The elementsize
// is in here; it is not needed for the shuffle/deal
// but could be useful in containing ops (e.g. 6-way shuffle/deal)
struct hvx_shufdeal3_consts {
	HVX_VectorPred q0,q1,q2;		// swap controls
	HVX_Vector rdeltaA,rdeltaB,rdeltaC;	// vdelta
	int elementsize;
};

// set up the constants for a given elementsize, power of 2 in range 1..64
//
static inline struct hvx_shufdeal3_consts 
__attribute__((always_inline))
hvx_shufdeal3_get_consts( int elementsize )
{
	struct hvx_shufdeal3_consts result;

	result.elementsize = elementsize;
	int log2_elsize = Q6_R_ct0_R(elementsize);
	HVX_Vector const * ptr = (HVX_Vector const *)const_Deal3_table + 3*log2_elsize;
	result.rdeltaA = ptr[0];
	result.rdeltaB = ptr[1];
	result.rdeltaC = ptr[2];
	result.q0 = Q6_Q_vand_VR( result.rdeltaA, 0x80808080);
	result.q1 = Q6_Q_vand_VR( result.rdeltaB, 0x80808080);
	result.q2 = Q6_Q_vand_VR( result.rdeltaC, 0x80808080);
	return result;
}

// this is a 3-way deal operation.
// 3x(128/wid) elements in; 3 packs of 128/wid out.
static inline HVX_Vector_x3
__attribute__((always_inline))
hvx_deal3( struct hvx_shufdeal3_consts konst,
	HVX_Vector v0, HVX_Vector v1, HVX_Vector v2 )
{
	// given A0:B0:C0:A1:B1 ... 
	// start by getting all the A in v0, B in v1, C in v2.
	HVX_VectorPair v12 = Q6_W_vswap_QVV( konst.q0, v2, v1 );
	HVX_VectorPair v01 = Q6_W_vswap_QVV( konst.q1, Q6_V_lo_W(v12), v0 );
	v12 = Q6_W_vswap_QVV( konst.q2, Q6_V_hi_W(v12), Q6_V_hi_W(v01) );

	// reorder them
	HVX_Vector_x3 result;
	result.val[0] = Q6_V_vrdelta_VV( Q6_V_lo_W(v01), konst.rdeltaA );
	result.val[1] = Q6_V_vrdelta_VV( Q6_V_lo_W(v12), konst.rdeltaB );
	result.val[2] = Q6_V_vrdelta_VV( Q6_V_hi_W(v12), konst.rdeltaC );
	return result;
}

// this is a 3-way shuffle operation.
// 3 packs of (128/wid) elements in;   3*(128/wid) out.
//inverse of hvx_deal3.
static inline HVX_Vector_x3
__attribute__((always_inline))
hvx_shuffle3( struct hvx_shufdeal3_consts konst,
	HVX_Vector v0, HVX_Vector v1, HVX_Vector v2 )
{
	v0 = Q6_V_vdelta_VV( v0, konst.rdeltaA );		// get all the elements in proper lanes
	v1 = Q6_V_vdelta_VV( v1, konst.rdeltaB );
	v2 = Q6_V_vdelta_VV( v2, konst.rdeltaC );
	// then in the proper vectors.
	HVX_VectorPair v12 = Q6_W_vswap_QVV( konst.q2, v2, v1 );
	HVX_VectorPair v01 = Q6_W_vswap_QVV( konst.q1, Q6_V_lo_W(v12), v0 );
	v12 = Q6_W_vswap_QVV( konst.q0, Q6_V_hi_W(v12), Q6_V_hi_W(v01) );

	HVX_Vector_x3 result;
	result.val[0] = Q6_V_lo_W(v01);
	result.val[1] = Q6_V_lo_W(v12);
	result.val[2] = Q6_V_hi_W(v12);
	return result;
}
// Design delta (or rdelta) network
//   - mapping indicates the desired source lane (0.. NVEC-1) for each of the outputs.
//   - 'dontcare' = 0 where mapping is valid; 255 for lanes where the output is a don't care.
//   - 'reverse' =  0 for delta network, 1 for rdelta network
//   Actually, only the sign bit of the 'dontcare_lanes' is used; so you can use the same
//   vector for 'mapping' and dontcare_lanes, if bit 7 is set in the outputs which are don't-care.
//
// The return value is the control word for '[r]delta' operation to implement this mapping.
// Infeasible mappings are not detected, but you can 'test' a mapping as below:
//
//    // find the mapping...
//    HVX_Vector mapping = ...
//    HVX_Vector dontcare_lanes = ...
//    HVX_Vector ctlvec = design_for_delta( mapping, dontcare_lanes , 0 );
//
//    // "test" it... use delta with input = { 0, 1 ... NVEC-1 }
//    HVX_Vector testres = Q6_V_vdelta_VV( q6op_Vb_vindices(), ctlvec );
//
//    // compare equal to mapping?
//    HVX_VectorPred match = Q6_Q_cmp_eq_VbVb( testres, mapping );
//    // the next step is only needed if there are any dont-care lanes
//    match = Q6_Q_vcmp_gtor_QVbVb( match, Q6_V_vzero(), dontcare_lanes );
//    //  The resulting 'match' should have all '1'; otherwise the mapping was not feasible.
//
//
static inline HVX_Vector design_for_delta( HVX_Vector mapping, HVX_VectorPred dontcare_lanes, int reverse)
{

	HVX_Vector result = Q6_V_vzero();

	// work our way from output of delta to input (span from 1 to N/2)
	// for rdelta, go from N/2 to 1
	HVX_Vector span = Q6_V_vsplat_R( Q6_R_vsplatb_R( reverse? (int)sizeof(HVX_Vector)/2: 1));
	int rsh = reverse ? 2: 0;

	HVX_VectorPred dcare = Q6_Q_vcmp_gt_VbVb( Q6_V_vzero(), dontcare_lanes );	// dcare = 0 > lanes.
	HVX_Vector valid = q6op_V_vand_QnR(dcare,-1);  	// 0xFF in valid lanes, 0 in non-valid
	HVX_Vector indices = *(HVX_Vector const*)const_Count128;			// 0,1,2 ...

	// starting at the last set of switches...
    int nloops = (sizeof(HVX_Vector)==128)? 7:6;
	for( unsigned k = 0; k < nloops; k++) {
		// which lanes need to switch relative to previous row?
		// true where ((mapping ^i) & span) == span
		//
		HVX_VectorPred sw = Q6_Q_vcmp_eq_VbVb( Q6_V_vand_VV( Q6_V_vxor_VV( mapping, indices), span ), span );

		// - lanes which are valid and which are not switching
		HVX_Vector valid_here = q6op_V_vand_QnV( sw, valid);
		// - lanes which are valid and which are switching.
		HVX_Vector valid_from_switched = q6op_V_vand_QV( sw, valid);

		// switch the 'from_switched' to source lane
		HVX_Vector valid_to_switched = Q6_V_vdelta_VV( valid_from_switched, span);
		// record the switch for result.
		result = Q6_V_vor_VV( result,Q6_V_vand_VV(span,valid_from_switched ));

		// swap the mapping over switches...
		HVX_Vector mapping_sw = Q6_V_vdelta_VV( mapping, span);

		// merge to next mapping using valid_here and valid_to_switched;

		mapping = Q6_V_vor_VV(
				Q6_V_vand_VV( valid_here, mapping),
				Q6_V_vand_VV( valid_to_switched, mapping_sw));

		// this is the new 'valid'
		valid = Q6_V_vor_VV( valid_here, valid_to_switched);
		// << span 1 bit
		span = Q6_Vb_vadd_VbVb( span, span);
		// >>2 if designing reverse delta.
		span = Q6_Vuw_vlsr_VuwR( span, rsh);
	}
	return result;
}

/*
    We have two independently quantized arrays. We need to compute
    joint_a = (((a - a_offset) * a_mult) >> shift) + qzero
    joint_b = (((b - b_offset) * b_mult) >> shift) + qzero
    to yield both values quantized in the same range so we can compare values
*/
static inline HVX_Vector jointly_quantize(HVX_Vector ind_a, HVX_Vector va_offset, HVX_Vector va_mult, HVX_Vector vqzero, int shift)
{
	// (a - a_offset), (b - offset) ~ ub*ub->2h
	HVX_VectorPair a_sub = Q6_Wh_vsub_VubVub(ind_a, va_offset);

	// ((a - offset) * a_mult) ~ h*h->w
	HVX_VectorPair a_mult_res_lo = Q6_Ww_vmpy_VhVuh(Q6_V_lo_W(a_sub), va_mult);
	HVX_VectorPair a_mult_res_hi = Q6_Ww_vmpy_VhVuh(Q6_V_hi_W(a_sub), va_mult);

	// (((a - offset) * a_mult)) >> shift) ~ w*w->h
	HVX_Vector shift_a_lo = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(a_mult_res_lo), Q6_V_lo_W(a_mult_res_lo), shift);
	HVX_Vector shift_a_hi = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(a_mult_res_hi), Q6_V_lo_W(a_mult_res_hi), shift);

	// shuffle odd and even so that the elements are back to the original order
	HVX_VectorPair ashuff = Q6_W_vshuff_VVR(shift_a_hi, shift_a_lo, -2);

	// (((a - offset) * a_mult)) >> shift) + qzero ~ h*h->h
	HVX_Vector joint_a_lo = Q6_Vh_vadd_VhVh_sat(Q6_V_lo_W(ashuff), vqzero);
	HVX_Vector joint_a_hi = Q6_Vh_vadd_VhVh_sat(Q6_V_hi_W(ashuff), vqzero);

	HVX_Vector va = Q6_Vub_vpack_VhVh_sat(joint_a_hi,joint_a_lo);

	return va;
}

//
// These are intended for debug
//
static inline
int vextract_u8( HVX_Vector v, int pos){
	union { HVX_Vector as_v; uint8_t as_u8[128]; } uu  = { v };
	return uu.as_u8[pos & 127];
}
static inline
int vextract_i16( HVX_Vector v, int pos){
	union { HVX_Vector as_v; int16_t as_i16[128/2]; } uu  = { v };
	return uu.as_i16[pos & 63];
}
static inline
int vextract_u16( HVX_Vector v, int pos){ return (uint16_t)vextract_i16(v,pos);}
static inline
int vextract_i32( HVX_Vector v, int pos){
	union { HVX_Vector as_v; int32_t as_i32[128/4]; } uu  = { v };
	return uu.as_i32[pos & 31];
}



#endif /* HVX_INLINES_H_ */
