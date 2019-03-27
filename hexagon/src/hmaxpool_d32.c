
/*
 * Copyright (c) 2017,2018 The Linux Foundation. All rights reserved.
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
#include "nn_hmaxpool_d32.h"
#include "nn_asm_ops.h"

hmaxpool_funcp nn_hmaxpool_select_function( struct hmaxpool_func_info * info,
		int win, int stride, int out_len, int inoffs )
{
	return hmaxpool_select_function_inline( info, win, stride, out_len, inoffs);
}

// this is the function pointer returned when the parameters are insensible
void hmaxpool_w0s0o0(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	return;
}
#if 0 // unused now
// this is a completely generic hmaxpool
// which handles any condition, done with non-hvx vector ops.
// Here nloops is the width of output, not output vectors.
//
void hmaxpool_wXsXoX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output values

	int in_offs = info->inoffs;
	int win = info->win;
	int str = info->stride;
	inp0 += 32*in_offs;
	inp1 += 32*in_offs;
	// do one row at a time
	uint64_t const *pin = (uint64_t const *) inp0;
	uint64_t *pout = (uint64_t *) out0;

	for(;;){
		for( int i = 0; i < nlp; i++){
			uint64_t x0 = pin[0];
			uint64_t x1 = pin[1];
			uint64_t x2 = pin[2];
			uint64_t x3 = pin[3];
			for( int iw = 1; iw < win; iw++){
				x0  = Q6_P_vmaxub_PP( x0, pin[4*iw]);
				x1  = Q6_P_vmaxub_PP( x1, pin[4*iw+1]);
				x2  = Q6_P_vmaxub_PP( x2, pin[4*iw+2]);
				x3  = Q6_P_vmaxub_PP( x3, pin[4*iw+3]);
			}
			pin += 4*str;
			pout[4*i] = x0;
			pout[4*i+1] = x1;
			pout[4*i+2] = x2;
			pout[4*i+3] = x3;
		}
		if( pout == (uint64_t *) out1) break;	// already did this (or both the same)
		pin = (uint64_t const *) inp1;
		pout = (uint64_t *) out1;
	}
}
#endif

// handles any case win >= 5
// for each output:
// - do unaligned load * pin; add k1 to pin
// - repeat k2 times:  do unaligned load *pin; max to previous; add 128 to pin
// 'k1' is 32 * ((win-1)%4+1)
// 'k2' is (win-1)/4
//

void hmaxpool_wge5sXoX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nout = info->nloops;		// # of outputs.

	int in_offs = info->inoffs;
	int win = info->win;
	int str = info->stride;
	inp0 += 32*in_offs;
	inp1 += 32*in_offs;
	// do one row at a time
	uint8_t const *pin =  inp0;
	uint8_t *pout0 = out0;

	str *= 32;		// stride in bytes

	int k1 = 32*((win-1)&3)+ 32;
	int k2 = ((win-1)>>2);
	int k3 = str*2 - (128*k2+k1);	//amount to bump after

	int nlp  = (nout+1)>>1;  // output loops -- 2 outputs, (1/2 vector) per loop
	if( nlp < 1) return;
	if( k2 < 1) return;
	// process only one row at a time
	for(;;){
		HVX_Vector xprev = Q6_V_vzero();
		int store = 0;
		uint8_t *pout = pout0;
		uint8_t const * pin1 = pin + str;

		for( int i = 0; i < nlp; i++){
			HVX_Vector x0 = q6op_V_vldu_A((HVX_Vector const *)pin);
			pin += k1;
			HVX_Vector x1 = q6op_V_vldu_A((HVX_Vector const *)pin1);
			pin1 += k1;
			for( int j = 0; j < k2; j ++ ){
				HVX_Vector y0 = q6op_V_vldu_A((HVX_Vector const *)pin);
				pin += 128;
				x0 = Q6_Vub_vmax_VubVub(x0,y0);
				HVX_Vector y1 = q6op_V_vldu_A((HVX_Vector const *)pin1);
				pin1 += 128;
				x1 = Q6_Vub_vmax_VubVub(x1,y1);
			}
			pin += k3;		// set up for next loop
			pin1 += k3;
			// now we have to max across the 4 quadrants in x0 and x1
			HVX_VectorPair dealt = Q6_W_vdeal_VVR( x1, x0, 32);
			x0 = Q6_Vub_vmax_VubVub( Q6_V_hi_W(dealt), Q6_V_lo_W(dealt));
			x0 = Q6_Vub_vmax_VubVub( x0, Q6_V_vror_VR( x0, 64));
			// two results in first half of 'x0'.
			xprev = Q6_V_valign_VVR( x0, xprev, 64);
			if( store){
				*(HVX_Vector *)pout = xprev;
				pout += 128;
			}
			store = !store;
		}
		if( nlp&1 ) *(HVX_Vector *)pout = Q6_V_vror_VR( xprev, 64);	// last 1/2 vector?

		if( pout0 == out1) break;	// already did this (or both the same)
		pin = inp1;
		pout0 = out1;
	}

}


////////////// general case for w  = 1,2,3,4, any stride
// Here nloops is outputs, not vectors
void hmaxpool_w1234sXoX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nout = info->nloops;		// # of outputs

	int in_offs = info->inoffs;
	int win = info->win;
	int str = info->stride;
	inp0 += 32*in_offs;
	inp1 += 32*in_offs;
	// do one row at a time
	uint8_t const *pin =  inp0;
	uint8_t *pout0 = out0;

	str *= 32;		// stride in bytes

	// the 'max operation is done by
	//    x = max( x, vror(x, vror_a))
	//    x = max( x, vror(x, vror_b))
	// where
	//  win     vror_a  vror_b
	//	  1       0        0
	//    2       32       0
	//    3       32       32
	//    4       64       32

	int vror_b = (win >= 3)? 32:0;
	int vror_a = win*32 - (32+vror_b);
	int nlp  = (nout+1)>>1;  // output loops -- 2 outputs, (1/2 vector) per loop
	if( nlp < 1) return;
	// process only one row at a time
	for(;;){
		HVX_Vector xprev = Q6_V_vzero();
		int store = 0;
		uint8_t *pout = pout0;
		for( int i = 0; i < nlp; i++){
			HVX_Vector x0 = q6op_V_vldu_A((HVX_Vector const *)pin);
			pin += str;
			HVX_Vector x1 = q6op_V_vldu_A((HVX_Vector const *)pin);
			pin += str;
			x0 = Q6_Vub_vmax_VubVub( x0, Q6_V_vror_VR(x0,vror_a));
			x0 = Q6_Vub_vmax_VubVub( x0, Q6_V_vror_VR(x0,vror_b));
			x1 = Q6_Vub_vmax_VubVub( x1, Q6_V_vror_VR(x1,vror_a));
			x1 = Q6_Vub_vmax_VubVub( x1, Q6_V_vror_VR(x1,vror_b));
			xprev = Q6_V_valign_VVR( x0, xprev, 32);
			xprev = Q6_V_valign_VVR( x1, xprev, 32);
			if( store){
				*(HVX_Vector *)pout = xprev;
				pout += 128;
			}
			store = !store;
		}
		if( nlp&1 ) *(HVX_Vector *)pout = Q6_V_vror_VR( xprev, 64);	// last 1/2 vector?
		if( pout0 == out1) break;	// already did this (or both the same)
		pin = inp1;
		pout0 = out1;
	}
}
/////////////////////////////// W=1, s=1 ///////////////

void hmaxpool_w1s1oX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int in_offs = info->inoffs*32;
	int nbytes = info->nloops*128;
	if( out0 != out1){
		vmemcpy_asm( out0, inp0 + in_offs, nbytes);
	}
	vmemcpy_asm( out1, inp1+in_offs, nbytes);
}

/////////////////////////////// W=2 ////////////////////


static inline HVX_Vector core_w2s2( HVX_Vector v_abcd, HVX_Vector  v_efgh )
{
	HVX_VectorPair v_shuf = Q6_W_vdeal_VVR( v_efgh, v_abcd, -32);
	// AA CC  EE GG
	// BB DD  FF HH
	return Q6_Vub_vmax_VubVub( Q6_V_hi_W(v_shuf), Q6_V_lo_W(v_shuf));
}



void hmaxpool_w2s2o0(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors

	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;

	//   AA  BB  CC  DD     EE  FF  GG  HH
	//  (AB)(CD)(EF)(GH)
	//
	for(int i=0; i < nlp; i++){
		HVX_Vector v0 = core_w2s2(vin0[0], vin0[1] );
		HVX_Vector v1 = core_w2s2(vin1[0], vin1[1] );
		*vout0 ++ = v0;
		*vout1 ++ = v1;
		vin0 += 2;
		vin1 += 2;
	}
}

static inline HVX_Vector core_w2s1( HVX_Vector v_abcd, HVX_Vector  v_efgh, int x0, int x1 )
{
	HVX_Vector vx_abcd = Q6_V_valign_VVR( v_efgh, v_abcd, x0);		// x0 = 32*in_align
	HVX_Vector vx_bcde = Q6_V_vlalign_VVR( v_efgh, v_abcd, x1);		// x1 = 96-x0
	return Q6_Vub_vmax_VubVub( vx_abcd, vx_bcde);
}

void hmaxpool_w2s1oX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors
	int x0 = info->inoffs *32;	// left offset (bytes)
	int x1 = 96-x0;
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;

	//   AA  BB  CC  DD     EE  FF  GG  HH
	//  (AB)(BC)(CD)(DE)
	//
	HVX_Vector v0_abcd = *vin0++;
	HVX_Vector v1_abcd = *vin1++;

	for(int i=0; i < nlp; i++){
		HVX_Vector v0_efgh = *vin0++;
		HVX_Vector v1_efgh = *vin1++;
		*vout0 ++ = core_w2s1( v0_abcd, v0_efgh,x0,x1);
		*vout1 ++ = core_w2s1( v1_abcd, v1_efgh,x0,x1);
		v0_abcd = v0_efgh;
		v1_abcd = v1_efgh;
	}
}

/////////////////////////////// W=3 ////////////////////


static inline HVX_Vector core_w3s1( HVX_Vector v_abcd, HVX_Vector  v_efgh )
{
	HVX_Vector v_bcde = Q6_V_valign_VVR( v_efgh, v_abcd, 32);
	HVX_Vector v_cdef = Q6_V_valign_VVR( v_efgh, v_abcd, 64);
	return Q6_Vub_vmax_VubVub(Q6_Vub_vmax_VubVub( v_abcd, v_bcde), v_cdef );
}

void hmaxpool_w3s1o0(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;

	HVX_Vector v0_abcd = *vin0++;
	HVX_Vector v1_abcd = *vin1++;

	for(int i=0; i < nlp; i++){
		HVX_Vector v0_efgh = *vin0++;
		HVX_Vector v1_efgh = *vin1++;
		*vout0 ++ = core_w3s1( v0_abcd, v0_efgh);
		*vout1 ++ = core_w3s1( v1_abcd, v1_efgh);
		v0_abcd = v0_efgh;
		v1_abcd = v1_efgh;
	}
}
// This is like hmaxpool_w3s1o0
// but it discards 'oshift' bytes from the initial output vector
// (and all others shifted along).

void hmaxpool_w3s1oX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors
	int oshift = info->oshift;
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;

	//   AA  BB  CC  DD     EE  FF  GG  HH
	//  (AB)(BC)(CD)(DE)
	//
	HVX_Vector v0_first = *vin0++;
	HVX_Vector v1_first = *vin1++;
	HVX_Vector v0_abcd = *vin0++;
	HVX_Vector v1_abcd = *vin1++;
	HVX_Vector v0_prev = core_w3s1( v0_first, v0_abcd);
	HVX_Vector v1_prev = core_w3s1( v1_first, v1_abcd);

	for(int i=0; i < nlp; i++){
		HVX_Vector v0_efgh = *vin0++;
		HVX_Vector v1_efgh = *vin1++;
		HVX_Vector v0 = core_w3s1( v0_abcd, v0_efgh);
		HVX_Vector v1 = core_w3s1( v1_abcd, v1_efgh);
		*vout0 ++ = Q6_V_valign_VVR( v0, v0_prev, oshift); v0_prev = v0;
		*vout1 ++ = Q6_V_valign_VVR( v1, v1_prev, oshift); v1_prev = v1;
		v0_abcd = v0_efgh;
		v1_abcd = v1_efgh;
	}
}


//   AA  BB  CC  DD     EE  FF  GG  HH    II JJ KK LL
//  (ABC)(CDE)(EFG)(GHI)

static inline HVX_Vector core_w3s2( HVX_Vector v_abcd, HVX_Vector  v_efgh, HVX_Vector  v_ijkl )
{
	HVX_VectorPair v_shuf = Q6_W_vdeal_VVR( v_efgh, v_abcd, -32);
	// AA CC  EE GG
	// BB DD  FF HH
	HVX_Vector tmp = Q6_Vub_vmax_VubVub( Q6_V_hi_W(v_shuf), Q6_V_lo_W(v_shuf));
	// CC EE  GG II
	HVX_Vector v_cegi = Q6_V_valign_VVR( v_ijkl, Q6_V_lo_W(v_shuf), 32);
	return Q6_Vub_vmax_VubVub( tmp, v_cegi );
}
// wid 3 span 2, offs 0

void
hmaxpool_w3s2o0(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;


	HVX_Vector v0_abcd = *vin0++;
	HVX_Vector v1_abcd = *vin1++;

	for(int i=0; i < nlp; i++){
		HVX_Vector v0_efgh = *vin0++;
		HVX_Vector v1_efgh = *vin1++;
		HVX_Vector v0_ijkl = *vin0++;
		HVX_Vector v1_ijkl = *vin1++;
		*vout0 ++ = core_w3s2( v0_abcd, v0_efgh, v0_ijkl);
		*vout1 ++ = core_w3s2( v1_abcd, v1_efgh, v1_ijkl);
		v0_abcd = v0_ijkl;
		v1_abcd = v1_ijkl;
	}
}

// wid 3 span 2, offs x
// (x must be 1..3)
// this works by cutting 1..3 elements from the start of each row
// according to 'inoffs'; otherwise the same as hmaxpool_w3s2o0
//
void
hmaxpool_w3s2oX(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int inshift = info->inoffs*32;
	int nlp = info->nloops;		// # of output vectors
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;

	HVX_Vector vin0_LL = *vin0++;
	HVX_Vector vin0_L = *vin0++;
	HVX_Vector v0_abcd = Q6_V_valign_VVR( vin0_L, vin0_LL, inshift);

	HVX_Vector vin1_LL = *vin1++;
	HVX_Vector vin1_L = *vin1++;
	HVX_Vector v1_abcd = Q6_V_valign_VVR( vin1_L, vin1_LL, inshift);

	for(int i=0; i < nlp; i++){
		HVX_Vector vin0_R = *vin0++;
		HVX_Vector vin0_RR = *vin0++;
		HVX_Vector v0_efgh = Q6_V_valign_VVR( vin0_R, vin0_L, inshift);
		HVX_Vector v0_ijkl = Q6_V_valign_VVR( vin0_RR, vin0_R, inshift);
		vin0_L = vin0_RR;

		HVX_Vector vin1_R = *vin1++;
		HVX_Vector vin1_RR = *vin1++;
		HVX_Vector v1_efgh = Q6_V_valign_VVR( vin1_R, vin1_L, inshift);
		HVX_Vector v1_ijkl = Q6_V_valign_VVR( vin1_RR, vin1_R, inshift);
		vin1_L = vin1_RR;


		*vout0 ++ = core_w3s2( v0_abcd, v0_efgh, v0_ijkl);
		*vout1 ++ = core_w3s2( v1_abcd, v1_efgh, v1_ijkl);
		v0_abcd = v0_ijkl;
		v1_abcd = v1_ijkl;
	}
}

//  w3 s3
// we want 3->1 reduction:
//   AA  BB  CC  DD     EE  FF  GG  HH    II JJ KK LL
//  (ABC)(DEF)(GHI)(JKL)
//
// start with
//  A  B  C  D
//  E  F  G  H
//  I  J  K  L
//
//  A B  G  H 	<- vmux
//  E F  K  L	<- vmux
//  C D  I  J 	<- valign
//
// Shuffle the first 2:
//
//  A  E  G  K   (shuffle)
//  B  F  H  L	 (shuffle)
//  C  D  I  J
//
// now just max across those 3.

static inline HVX_Vector core_w3s3( HVX_Vector v_abcd, HVX_Vector  v_efgh, HVX_Vector  v_ijkl )
{
	HVX_Vector qhalf = Q6_Q_vsetq_R(64);
	HVX_Vector v_abgh  = Q6_V_vmux_QVV(qhalf, v_abcd, v_efgh );
	HVX_Vector v_efkl  = Q6_V_vmux_QVV(qhalf, v_efgh, v_ijkl );
	HVX_VectorPair vshuf = Q6_W_vshuff_VVR(v_efkl, v_abgh, 32 );	// aegk and bfhl
	HVX_Vector v_cdij = Q6_V_valign_VVR( v_ijkl, v_abcd, 64);

	return Q6_Vub_vmax_VubVub( Q6_Vub_vmax_VubVub( Q6_V_hi_W(vshuf), Q6_V_lo_W(vshuf)),v_cdij);
}


void hmaxpool_w3s3o0(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;

	for(int i=0; i < nlp; i++){
		HVX_Vector v0 = core_w3s3( vin0[0], vin0[1], vin0[2]);
		HVX_Vector v1 = core_w3s3( vin1[0], vin1[1], vin1[2]);
		*vout0 ++ = v0;
		*vout1 ++ = v1;
		vin0 += 3;
		vin1 += 3;
	}
}
// w3 s3 offs 3: same as w3 s3 o0 except we drop the first
// output and shift the rest along. (This is more efficient than
// shifting all the inputs by 3)
void hmaxpool_w3s3o3(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;


	HVX_Vector v0_prev = core_w3s3( vin0[0], vin0[1], vin0[2]);
	HVX_Vector v1_prev = core_w3s3( vin1[0], vin1[1], vin1[2]);
	vin0 += 3;
	vin1 += 3;

	for(int i=0; i < nlp; i++){
		HVX_Vector v0 = core_w3s3( vin0[0], vin0[1], vin0[2]);
		HVX_Vector v1 = core_w3s3( vin1[0], vin1[1], vin1[2]);
		*vout0 ++ = Q6_V_valign_VVR(v0,v0_prev, 32 ); 	v0_prev = v0;
		*vout1 ++ = Q6_V_valign_VVR(v1,v1_prev, 32 ); 	v1_prev = v1;
		vin0 += 3;
		vin1 += 3;
	}
}

// w3 s3 offs 1 or 2:
//  for off=1
//     .AAA|BBBC|CCDD|DEEE|...
//  for off=2:
//     ..AA|ABBB|CCCD|DDEE|EFFF|...
//
// So, for offset = 1 we handle the 'A' output specially, and then squish that into the
// rest of the outputs.
//     for offset = 2 we handle the 'A' and 'B' output specially, and then squish that
//     into the rest of the output.
//
void hmaxpool_w3s3o12(struct hmaxpool_func_info const * info ,
		uint8_t * out0, uint8_t *out1,  uint8_t const *inp0, uint8_t const *inp1)
{
	int nlp = info->nloops;		// # of output vectors
	int inoffs = info->inoffs;	// 1 or 2
	int outshift = inoffs*32;	// 32 or 64
	HVX_Vector const *vin0 = (HVX_Vector const *) inp0;
	HVX_Vector const *vin1 = (HVX_Vector const *) inp1;
	HVX_Vector *vout0 = (HVX_Vector *) out0;
	HVX_Vector *vout1 = (HVX_Vector *) out1;

	HVX_Vector z0 = *vin0++;
	HVX_Vector z1 = *vin1++;
	HVX_Vector y0 = z0;
	HVX_Vector y1 = z1;
	if( inoffs == 2){	// need to do two
		z0 = *vin0++;
		z1 = *vin1++;
	}
	HVX_Vector v0_prev = core_w3s3( Q6_V_vzero(), y0,z0);
	HVX_Vector v1_prev = core_w3s3( Q6_V_vzero(), y1,z1);

	for(int i=0; i < nlp; i++){
		HVX_Vector v0 = core_w3s3( vin0[0], vin0[1], vin0[2]);
		HVX_Vector v1 = core_w3s3( vin1[0], vin1[1], vin1[2]);
		*vout0 ++ = Q6_V_vlalign_VVR(v0,v0_prev, outshift ); 	v0_prev = v0;
		*vout1 ++ = Q6_V_vlalign_VVR(v1,v1_prev, outshift ); 	v1_prev = v1;
		vin0 += 3;
		vin1 += 3;
	}
}

