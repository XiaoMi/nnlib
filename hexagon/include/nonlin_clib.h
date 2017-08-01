/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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


//===============================================================
//  Q6Vect512 Intrinsic Code
//===============================================================
// NoteA: "nonlin_coef.h" header file used as a medium for passing macros computed by Python to intrinsic/ASM code.
// - macros consist of parameters like coef_scale, remez_order, sature_max, sature_min, num_bits_lut_integral_part, num_bits_lut_fractional_part, Q format use etc. 
// - macros also consist of defines to skip parts of code that are unused (for example skipping code based on cases of odd and even symmetry)
// NoteB: "nonlin_coef.h" header file has two different formats for polynomial-approximation coefficient-containing Look-Up Table (LUT):
// - lut_non_lin_cn (for use as reference under CNATURAL_NONLIN_DEFS) in "natural-C" format - row of increasing-poly-order-wise coeffs for 1st rng/interval then row for 2nd interval and so on.
// - lut_non_lin_asm[] (under CINTRINSIC_NONLIN_DEFS) in different format as table is put in "vectorized" format for lookup using HVX vectors in intrinsic/ASM code.
// NoteC: "nonlin_coef.h" header file has function prototypes (under CINTRINSIC_NONLIN_DEFS) to allow calls to intrinsic/ASM code functions by C test code
//#define CINTRINSIC_NONLIN_DEFS
//#include <nonlin_coef.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif


/*************************************************************************
 * MACROS for Halfword
 *************************************************************************/

#define LOG2_VECT_SIZE_HFW (6)		// VECT_SIZE_HFW = 64

// calc vrngidx ((vin >> rngidx_rshift) ^ vrngsignbits) byt vect
#define CALC_VRNGIDX_HFW(vrngidx, RNGIDX_RSHIFT, vin_a, vin_b, vtemp_a, vtemp_b) \
	vtemp_a = Q6_Vuh_vlsr_VuhR(vin_a, RNGIDX_RSHIFT); \
	vtemp_b = Q6_Vuh_vlsr_VuhR(vin_b, RNGIDX_RSHIFT); \
	vrngidx = Q6_Vb_vshuffe_VbVb(vtemp_b, vtemp_a);

// calc vdeltax ((vin & vdltmask) << deltax_lshift) hfw vect 
#define CALC_VDELTAX_HFW(vdeltax, vdltmask, DELTAX_LSHIFT, vin) \
	vdeltax = Q6_V_vand_VV(vin, vdltmask); \
	vdeltax = Q6_Vh_vasl_VhR(vdeltax, DELTAX_LSHIFT);

// get wcoeff hfw vectpair from vlut16 hfw vect using vrngidx byt vect
#define GET_WCOEF_HFW(wcoef, vrngidx, vlut, ORDER) \
	wcoef_##ORDER = Q6_Wh_vlut16_VbVhR(vrngidx, vlut_##ORDER, 0);
#define GET_WCOEF_1ORDER_HFW(wcoef, vrngidx, vlut, ORDER1) \
	GET_WCOEF(wcoef, vrngidx, vlut, ORDER1)
#define GET_WCOEF_2ORDER_HFW(wcoef, vrngidx, vlut, ORDER1, ORDER2) \
	GET_WCOEF_HFW(wcoef, vrngidx, vlut, ORDER1) \
	GET_WCOEF_HFW(wcoef, vrngidx, vlut, ORDER2)

// set vout using line-order multiply-accumulate w/rnd+sat of wcoeff hfw vectpair with vdeltax hfw vect
// init v =                    lut_non_lin[((ORDER+1)*rngidx)+(ORDER)];
// loop v = mpyrsat(v,delta) + lut_non_lin[((ORDER+1)*rngidx)+(ORDER-loopcount)];
#define CALC_VOUT_ORDERLINE_HFW(vout, vdeltax, wcoef, IDX, OPT) \
	vout_##IDX = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vdeltax_##IDX, Q6_V_##OPT##_W(wcoef_1)); \
	vout_##IDX = Q6_Vh_vadd_VhVh_sat(vout_##IDX, Q6_V_##OPT##_W(wcoef_0));

// set vout using quad-order multiply-accumulate w/rnd+sat of wcoeff hfw vectpair with vdeltax hfw vect
#define CALC_VOUT_ORDERQUAD_HFW(vout, vdeltax, wcoef, IDX, OPT) \
	vout_##IDX = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vdeltax_##IDX, Q6_V_##OPT##_W(wcoef_2)); \
	vout_##IDX = Q6_Vh_vadd_VhVh_sat(vout_##IDX, Q6_V_##OPT##_W(wcoef_1)); \
	vout_##IDX = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vdeltax_##IDX, vout_##IDX); \
	vout_##IDX = Q6_Vh_vadd_VhVh_sat(vout_##IDX, Q6_V_##OPT##_W(wcoef_0));

// set vout using cube-order multiply-accumulate w/rnd+sat of wcoeff hfw vectpair with vdeltax hfw vect
#define CALC_VOUT_ORDERCUBE_HFW(vout, vdeltax, wcoef, IDX, OPT) \
	vout_##IDX = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vdeltax_##IDX, Q6_V_##OPT##_W(wcoef_3)); \
	vout_##IDX = Q6_Vh_vadd_VhVh_sat(vout_##IDX, Q6_V_##OPT##_W(wcoef_2)); \
	vout_##IDX = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vdeltax_##IDX, vout_##IDX); \
	vout_##IDX = Q6_Vh_vadd_VhVh_sat(vout_##IDX, Q6_V_##OPT##_W(wcoef_1)); \
	vout_##IDX = Q6_Vh_vmpy_VhVh_s1_rnd_sat(vdeltax_##IDX, vout_##IDX); \
	vout_##IDX = Q6_Vh_vadd_VhVh_sat(vout_##IDX, Q6_V_##OPT##_W(wcoef_0));

// convert vout to q16 from q15 hfw vect
#define CONVERT_VOUT_TO_Q16_FROM_Q15_HFW(vout) \
	vout = Q6_Vh_vasl_VhR(vout, 1);

// load vin hfw vect
#define LOAD_VIN_1VEC_HFW(vin, pvmemin, IDX) \
    vin_##IDX = *pvmemin++;

#define CALC_VDELTAX_1VEC_HFW(vdeltax, vdltmask, DELTAX_LSHIFT, vin, IDX) \
    CALC_VDELTAX_HFW(vdeltax_##IDX, vdltmask, DELTAX_LSHIFT, vin_##IDX)

#define CALC_VOUT_1VEC_ORDERLINE_HFW(vout, vdeltax, wcoef, IDX) \
    CALC_VOUT_ORDERLINE_HFW(vout, vdeltax, wcoef, IDX, lo)

#define CALC_VOUT_1VEC_ORDERQUAD_HFW(vout, vdeltax, wcoef, IDX) \
    CALC_VOUT_ORDERQUAD_HFW(vout, vdeltax, wcoef, IDX, lo)

#define CALC_VOUT_1VEC_ORDERCUBE_HFW(vout, vdeltax, wcoef, IDX) \
    CALC_VOUT_ORDERCUBE_HFW(vout, vdeltax, wcoef, IDX, lo)

#define CONVERT_VOUT_TO_Q16_FROM_Q15_1VEC_HFW(vout, IDX) \
    CONVERT_VOUT_TO_Q16_FROM_Q15_HFW(vout_##IDX)

// store vout hfw vect
#define STORE_VOUT_1VEC_HFW(pvmemout, vout, IDX) \
	*pvmemout++ = vout_##IDX;

#define LOAD_VIN_2VEC_HFW(vin, pvmemin, IDX1, IDX2) \
	LOAD_VIN_1VEC_HFW(vin, pvmemin, IDX1) \
	LOAD_VIN_1VEC_HFW(vin, pvmemin, IDX2)

#define CALC_VRNGIDX_2VEC_HFW(vrngidx, RNGIDX_RSHIFT, vin, vtemp, IDX1, IDX2) \
    CALC_VRNGIDX_HFW(vrngidx, RNGIDX_RSHIFT, vin_##IDX1, vin_##IDX2, vtemp_##IDX1, vtemp_##IDX2)

#define CALC_VDELTAX_2VEC_HFW(vdeltax, vdltmask, DELTAX_LSHIFT, vin, IDX1, IDX2) \
	CALC_VDELTAX_1VEC_HFW(vdeltax, vdltmask, DELTAX_LSHIFT, vin, IDX1) \
	CALC_VDELTAX_1VEC_HFW(vdeltax, vdltmask, DELTAX_LSHIFT, vin, IDX2) \
    
#define CALC_VOUT_2VEC_ORDERLINE_HFW(vout, vdeltax, wcoef, IDX1, IDX2) \
	CALC_VOUT_ORDERLINE_HFW(vout, vdeltax, wcoef, IDX1, lo) \
	CALC_VOUT_ORDERLINE_HFW(vout, vdeltax, wcoef, IDX2, hi)

#define CALC_VOUT_2VEC_ORDERQUAD_HFW(vout, vdeltax, wcoef, IDX1, IDX2) \
    CALC_VOUT_ORDERQUAD_HFW(vout, vdeltax, wcoef, IDX1, lo) \
    CALC_VOUT_ORDERQUAD_HFW(vout, vdeltax, wcoef, IDX2, hi)
    
#define CALC_VOUT_2VEC_ORDERCUBE_HFW(vout, vdeltax, wcoef, IDX1, IDX2) \
	CALC_VOUT_ORDERCUBE_HFW(vout, vdeltax, wcoef, IDX1, lo) \
	CALC_VOUT_ORDERCUBE_HFW(vout, vdeltax, wcoef, IDX2, hi)

#define CONVERT_VOUT_TO_Q16_FROM_Q15_2VEC_HFW(vout, IDX1, IDX2) \
	CONVERT_VOUT_TO_Q16_FROM_Q15_1VEC_HFW(vout, IDX1) \
	CONVERT_VOUT_TO_Q16_FROM_Q15_1VEC_HFW(vout, IDX2)
    
#define STORE_VOUT_2VEC_HFW(pvmemout, vout, IDX1, IDX2) \
	STORE_VOUT_1VEC_HFW(pvmemout, vout, IDX1) \
	STORE_VOUT_1VEC_HFW(pvmemout, vout, IDX2)

#define NON_LIN_I_16(OUT, IN, LUT, NUMELEMENTS, N_ORDER, RNGIDX_MASK, RNGIDX_RSHIFT, DELTAX_MASK, DELTAX_LSHIFT) \
    /*NOTE: lut_non_lin_asm array is pre-shuffled to be in the correct order for the vlut16 instruction */ \
    HVX_Vector *pvlut = (HVX_Vector *)LUT;\
    HVX_Vector *pvmemin = (HVX_Vector *)IN;\
    HVX_Vector *pvmemout = (HVX_Vector *)OUT;\
    HVX_Vector vin_lo, vin_hi, vout_lo, vout_hi, vtemp_lo, vtemp_hi;\
    HVX_Vector vrngidx, vdeltax_lo, vdeltax_hi;\
    HVX_Vector vdltmask;\
    HVX_Vector vlut_0, vlut_1, vlut_2, vlut_3;\
    HVX_VectorPair wcoef_0, wcoef_1, wcoef_2, wcoef_3;\
    int elementcount;\
    int dltmask;\
    \
    /* vdltmask example  : 0x0fff0fff (frac bits b11-b0) or 0x07ff07ff (frac bits b10-b0) */\
    dltmask = Q6_R_combine_RlRl(DELTAX_MASK, DELTAX_MASK); \
    vdltmask = Q6_V_vsplat_R(dltmask);\
    \
    /* set lut pointers based on polynomial n_order */\
    vlut_0 = *pvlut++;\
    vlut_1 = *pvlut++;\
    vlut_2 = *pvlut++;\
    vlut_3 = *pvlut;\
    \
    /*PROLOG */\
    int odd_flag = 0;\
    odd_flag = (numelements >> LOG2_VECT_SIZE_HFW) & 1;		/* set flag if odd number of 128B vectors */\
    numelements = numelements >> (LOG2_VECT_SIZE_HFW+1);	/* loop counter multiple of 256 elements */\
    /* load input x <=== use for current iteration */\
    LOAD_VIN_2VEC_HFW(vin, pvmemin, lo, hi)\
    \
    /* get rngidx=((x>>rngidx_rshift)^rngsignbits) & deltax=((x&deltax_mask)<<deltax_lshift) <=== use for current iteration */\
    CALC_VRNGIDX_2VEC_HFW(vrngidx, RNGIDX_RSHIFT, vin, vtemp, lo, hi)\
    CALC_VDELTAX_2VEC_HFW(vdeltax, vdltmask, DELTAX_LSHIFT, vin, lo, hi)\
    \
    /* read lut_non_lin <=== use for current iteration */\
    GET_WCOEF_2ORDER_HFW(wcoef, vrngidx, vlut, 2, 3)\
    \
    /*LOOP */\
    for (elementcount = 0; elementcount < NUMELEMENTS; elementcount++) {\
        /* read lut_non_lin <=== use for current iteration */\
        GET_WCOEF_2ORDER_HFW(wcoef, vrngidx, vlut, 0, 1)\
        \
        /* load input x <=== setup for next iteration */\
        LOAD_VIN_2VEC_HFW(vin, pvmemin, lo, hi)\
        \
        /* compute output using lut_non_lin & rngidx & deltax  <=== use for current iteration */\
        /* Note: Should configure compiler optimization level that removes branches that are never reachable */\
        if (N_ORDER > 2) {\
        	CALC_VOUT_2VEC_ORDERCUBE_HFW(vout, vdeltax, wcoef, lo, hi)\
        }\
        else if (N_ORDER > 1) {\
        	CALC_VOUT_2VEC_ORDERQUAD_HFW(vout, vdeltax, wcoef, lo, hi)\
        }\
        else {\
        	CALC_VOUT_2VEC_ORDERLINE_HFW(vout, vdeltax, wcoef, lo, hi)\
        }\
        \
        /* get rngidx=((x>>rngidx_rshift)^rngsignbits) & deltax=((x&deltax_mask)<<deltax_lshift) <=== setup for next iteration */\
        CALC_VRNGIDX_2VEC_HFW(vrngidx, RNGIDX_RSHIFT, vin, vtemp, lo, hi)\
        CALC_VDELTAX_2VEC_HFW(vdeltax, vdltmask, DELTAX_LSHIFT, vin, lo, hi)\
        \
        /* store output y <=== use for current iteration */\
        CONVERT_VOUT_TO_Q16_FROM_Q15_2VEC_HFW(vout, lo, hi)\
        STORE_VOUT_2VEC_HFW(pvmemout, vout, lo, hi)\
        \
        /* read lut_non_lin <=== setup for next iteration */\
        GET_WCOEF_2ORDER_HFW(wcoef, vrngidx, vlut, 2, 3)\
    }\
    \
    /*EPILOG */\
    if(odd_flag) {\
        /* read lut_non_lin <=== use for current iteration */\
        GET_WCOEF_2ORDER_HFW(wcoef, vrngidx, vlut, 0, 1)\
        \
		/* compute output using lut_non_lin & rngidx & deltax  <=== use for current iteration */\
		/* Note: Should configure compiler optimization level that removes branches that are never reachable */\
        if (N_ORDER > 2) {\
        	CALC_VOUT_2VEC_ORDERCUBE_HFW(vout, vdeltax, wcoef, lo, hi)\
        }\
        else if (N_ORDER > 1) {\
        	CALC_VOUT_2VEC_ORDERQUAD_HFW(vout, vdeltax, wcoef, lo, hi)\
        }\
        else {\
        	CALC_VOUT_2VEC_ORDERLINE_HFW(vout, vdeltax, wcoef, lo, hi)\
        }\
		\
        /* store output y <=== use for current iteration */\
        CONVERT_VOUT_TO_Q16_FROM_Q15_1VEC_HFW(vout, lo)\
        STORE_VOUT_1VEC_HFW(pvmemout, vout, lo)\
    }

/*************************************************************************
 * MACROS for Bytes
 *************************************************************************/

#define LOG2_VECT_SIZE_BYT (7)		// VECT_SIZE_BYTES = 128

// rngidx = (((x & rngidx_mask) >> rngidx_rshift) ^ rngsignbits)
#define CALC_VRNGIDX_FROM_VIN_BYT(VRNGIDX, VIN, VRNGMASK, VRNGSIGNBITS, RNGIDX_RSHIFT_BYT) \
    VRNGIDX = Q6_V_vand_VV(VIN, VRNGMASK); \
    VRNGIDX = Q6_Vuh_vlsr_VuhR(VRNGIDX, RNGIDX_RSHIFT_BYT); \
    
// deltax = ((x & deltax_mask)
#define CALC_VDELTAX_FROM_VIN_BYT(VDELTAX,VIN,VDLTMASK,DELTAX_LSHIFT) \
    VDELTAX = Q6_V_vand_VV(VIN, VDLTMASK);\
	VDELTAX = Q6_Vh_vasl_VhR(VDELTAX, DELTAX_LSHIFT);
    
// Read coeff from LUT
#define GET_VCOEF_FROM_VLUT_VRNGIDX_BYT(VCOEF0,VCOEF1,VLUT0,VLUT1,VRNGIDX) \
    VCOEF1 = Q6_Vb_vlut32_VbVbR(VRNGIDX, VLUT1, 0); \
    VCOEF0 = Q6_Vb_vlut32_VbVbR(VRNGIDX, VLUT0, 0);
    
// compute output using lut_non_lin & rngidx & deltax
#define CALC_VOUT_INTERPOLATE_BYT(VOUT, WTEMP, VCOEF1, VDELTAX, VCOEF0, VDLTMULT) \
    WTEMP = Q6_Wh_vmpy_VbVb(VDELTAX,VCOEF1); \
    WTEMP = Q6_Wh_vmpyacc_WhVubVb(WTEMP, VDLTMULT, VCOEF0); \
    VOUT = Q6_Vb_vasr_VhVhR_rnd_sat(Q6_V_hi_W(WTEMP),Q6_V_lo_W(WTEMP),LOG2_VECT_SIZE_BYT);

#define LOAD_VIN_1VEC_BYT(IN) \
    vin##IN = *ptr_vx++; \

#define STORE_VOUT_1VEC_BYT(OUT) \
    *ptr_vy++ = vout##OUT ; \
    
#define CALC_VRNGIDX_1VEC_BYT(IN, RNGIDX_RSHIFT) \
    CALC_VRNGIDX_FROM_VIN_BYT(vrngidx##IN, vin##IN, vrngmask, vrngsignbits, RNGIDX_RSHIFT) \

#define CALC_VDELTAX_1VEC_BYT(IN,DELTAX_LSHIFT) \
    CALC_VDELTAX_FROM_VIN_BYT(vdeltax##IN, vin##IN, vdltmask,DELTAX_LSHIFT) \

#define GET_VCOEF_1VEC_BYT(IN) \
    GET_VCOEF_FROM_VLUT_VRNGIDX_BYT(vcoef0_##IN, vcoef1_##IN, vlut_order0, vlut_order1, vrngidx##IN) \

#define CALC_VOUT_1VEC_BYT(IN) \
    CALC_VOUT_INTERPOLATE_BYT(vout##IN, wtemp, vcoef1_##IN, vdeltax##IN, vcoef0_##IN, vdltmult) \

#define LOAD_VIN_2VEC_BYT(IN1, IN2) \
	LOAD_VIN_1VEC_BYT(IN1) \
	LOAD_VIN_1VEC_BYT(IN2)
    
#define STORE_VOUT_2VEC_BYT(OUT1, OUT2) \
	STORE_VOUT_1VEC_BYT(OUT1) \
	STORE_VOUT_1VEC_BYT(OUT2)
    
#define CALC_VRNGIDX_2VEC_BYT(IN1,IN2,RNGIDX_RSHIFT) \
	CALC_VRNGIDX_1VEC_BYT(IN1,RNGIDX_RSHIFT) \
	CALC_VRNGIDX_1VEC_BYT(IN2,RNGIDX_RSHIFT)
    
#define CALC_VDELTAX_2VEC_BYT(IN1,IN2,DELTAX_LSHIFT) \
	CALC_VDELTAX_1VEC_BYT(IN1,DELTAX_LSHIFT) \
	CALC_VDELTAX_1VEC_BYT(IN2,DELTAX_LSHIFT)
    
#define GET_VCOEF_2VEC_BYT(IN1,IN2) \
	GET_VCOEF_1VEC_BYT(IN1) \
	GET_VCOEF_1VEC_BYT(IN2)
    
#define CALC_VOUT_2VEC_BYT(IN1,IN2) \
	CALC_VOUT_1VEC_BYT(IN1) \
	CALC_VOUT_1VEC_BYT(IN2) \

#define NON_LIN_I_8(OUT, IN, LUT, NUMELEMENTS, RNGIDX_MASK, RNGIDX_RSHIFT, RNGIDX_NBITS, DELTAX_MASK, DELTAX_LSHIFT) \
	/*NOTE: lut_non_lin_asm array is pre-shuffled to be in the correct order for the vlut32 instruction */\
	HVX_Vector *vptr_non_lin = (HVX_Vector *)LUT;\
	HVX_Vector *ptr_vx = (HVX_Vector *)IN;\
	HVX_Vector *ptr_vy = (HVX_Vector *)OUT;\
	HVX_Vector vin1, vin2, vout1, vout2;\
	HVX_Vector vrngidx1, vdeltax1, vrngidx2, vdeltax2;\
	HVX_Vector vrngmask, vdltmask, vdltmult;\
	HVX_Vector vlut_order0, vlut_order1;\
	HVX_Vector vcoef0_1, vcoef1_1, vcoef0_2, vcoef1_2;\
	HVX_VectorPair wcoef, wtemp;\
	int elementcount;\
	int rngmask, dltmask, dltmult;\
	/* vrngmask example  : 0xf0f0f0f0 (int bits b7-b4) or 0x78787878 (int bits b6-b3) */\
	rngmask = (((int32_t)RNGIDX_MASK<<8) & 0x0000FF00) | ((int32_t)RNGIDX_MASK & 0x000000FF);\
	vrngmask = Q6_V_vsplat_R(Q6_R_combine_RlRl(rngmask,rngmask));\
	\
	/* vdltmask example  : 0x0f0f0f0f (frac bits b3-b0) or 0x07070707 (frac bits b2-b0) */\
	dltmask = (((int32_t)DELTAX_MASK<<8) & 0x0000FF00) | ((int32_t)DELTAX_MASK & 0x000000FF);\
	vdltmask = Q6_V_vsplat_R(Q6_R_combine_RlRl(dltmask,dltmask));\
	\
	/* vdltmult - Convert COEF0 to 16-bit, COEF0 << 7 */\
	dltmult = (1<<LOG2_VECT_SIZE_BYT);\
	dltmult = (((int32_t)(dltmult)<<8) & 0x0000FF00) | ((int32_t)(dltmult) & 0x000000FF);\
	vdltmult = Q6_V_vsplat_R(Q6_R_combine_RlRl(dltmult,dltmult));\
	\
	/* set lut pointers based on polynomial n_order */\
	vlut_order0 = *vptr_non_lin++;\
	vlut_order1 = *vptr_non_lin;\
	\
	/*PROLOG */\
	int odd_flag = 0;\
	odd_flag = (numelements >> LOG2_VECT_SIZE_BYT) & 1;	/* set flag if odd number of 128B vectors */\
	numelements = numelements >> (LOG2_VECT_SIZE_BYT+1);	/* loop counter multiple of 256 elements */\
	\
	/*LOOP */\
	for (elementcount = 0; elementcount < NUMELEMENTS; elementcount++) {\
	\
		/* load input x */\
		LOAD_VIN_2VEC_BYT(1,2)\
		\
		/* rngidx = ((x & rngidx_mask) >> rngidx_rshift) */\
		CALC_VRNGIDX_2VEC_BYT(1,2,RNGIDX_RSHIFT)\
		\
		/* deltax = (x & deltax_mask) */\
		CALC_VDELTAX_2VEC_BYT(1,2,DELTAX_LSHIFT)\
		\
		/* read lut_non_lin */\
		GET_VCOEF_2VEC_BYT(1,2)\
		\
		/* compute output using lut_non_lin & rngidx & deltax */\
		CALC_VOUT_2VEC_BYT(1,2)\
		\
		/* store output y */\
		STORE_VOUT_2VEC_BYT(1,2)\
	}\
	\
	/*EPILOG */\
	if(odd_flag) {\
		/* load input x */\
		LOAD_VIN_1VEC_BYT(1)\
		\
		/* rngidx = ((x & rngidx_mask) >> rngidx_rshift) */\
		CALC_VRNGIDX_1VEC_BYT(1,RNGIDX_RSHIFT)\
		\
		/* deltax = (x & deltax_mask) */\
		CALC_VDELTAX_1VEC_BYT(1,DELTAX_LSHIFT)\
		\
		/* read lut_non_lin */\
		GET_VCOEF_1VEC_BYT(1)\
		\
		/* compute output using lut_non_lin & rngidx & deltax */\
		CALC_VOUT_1VEC_BYT(1)\
		\
		/* store output y */\
		STORE_VOUT_1VEC_BYT(1)\
	}

