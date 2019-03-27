/*
 * Copyright (c) 2018, The Linux Foundation. All rights reserved.
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

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_hexagon_protos.h"
#include "hvx_inlines.h"

#ifndef HVX_INTRINSIC_REFFUNC
#define HVX_INTRINSIC_REFFUNC(f) f
#endif
//
// Four functions, N = 1,2,3 or 4:
// Reads width *N pixels, expands to width*4 pixels, with padding:
//   output is padded on the left by 4*left_pad pixels
//   output is padded on the right by 4*right_pad pixels
//  padding bytes are set to 'in_offset'
//
//   void incopy_expand_Nto4(
//           uint8_t * out,			// out, vector aligned
//           uint8_t const * in,	// in, no alignment assumed, width*N pixels
//           int width,				// width of input, >= 1
//           int in_offset,			// byte to fill padding (including within groups)
//           int left_pad,			// prepend this many pixels of zeros; 0..31
//			 int right_pad)			// append this many pixels of zeros; 0..31
//
// The extent written will be exactly 4*(left_pad + width + right_pad) bytes, rounded up to to a multiple of a vector
//
//////////////////////////////////////////////////
// copy3to4_cntrl: A vdelta control which does
//    0..2 -> 0..2
//    3..5 -> 4..6
//    6..8 -> 8..10
//
//  3*k .. 3*k+2  -> 4*k .. 4*k+2   (for k = 0..31)
//
// And the 4*k+3 outputs are "don't care"
//
extern const unsigned char copy3to4_cntrl[128] __attribute__ ((aligned (128)));
/////////////////////////////////////////////////

void incopy_expand_1to4 (
		uint8_t * out,			// out, vector aligned
		uint8_t const * in,	// in, no alignment assumed, width*N pixels
		int width,				// width of input, >= 1
		int in_offset,			// byte to fill padding (including within groups)
		int left_pad,			// prepend this many pixels of zeros; 0..31
		int right_pad)			// append this many pixels of zeros; 0..31
{
	int n_in = 1;
	HVX_Vector * voutp = (HVX_Vector*) out;
	HVX_Vector qlast = q6op_Q_vsetq2_R( (left_pad + width)*4);
	HVX_Vector vfill = q6op_Vb_vsplat_R( in_offset );
	// get the first vector; stuff it with pixels on the left
	HVX_Vector v0 = Q6_V_vlalign_VVR( q6op_V_vldu_A((HVX_Vector const *)in), vfill, left_pad*n_in );
	in += 32*n_in -left_pad*n_in;			// point to the next chunk of pixels
	// nvec is 1 less than the # of output vectors needed for left_pad and width.
	int nvec = (left_pad + width-1)/32u;
	int extra =(nvec+1)*32 < left_pad + width + right_pad;

#if 0
	for(int i =0; i < nvec; i++){
		// shuffle fill bytes in
		HVX_VectorPair vpair = Q6_W_vshuff_VVR( vfill, v0, -1);
		*voutp ++ = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill, Q6_V_lo_W(vpair), -2));
		v0 = q6op_V_vldu_A((HVX_Vector const *)in);
		in += n_in * 32;
	}

#else
	in += 32*n_in;
	if( nvec > 0){
		for(int i =0; i < nvec>>1; i++){
			// store two at once
			HVX_Vector vx = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill, v0, -1));
			HVX_VectorPair outpair = Q6_W_vshuff_VVR( vfill, vx, -2);
			*voutp ++ = Q6_V_lo_W(outpair);
			*voutp ++ = Q6_V_hi_W(outpair);
			v0 = q6op_V_vldu_A((HVX_Vector const *)in);
			in += n_in * 32*2;
		}
		if( nvec & 1){
			HVX_VectorPair vpair = Q6_W_vshuff_VVR( vfill, v0, -1);
			*voutp ++ = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill, Q6_V_lo_W(vpair), -2));
			v0 = Q6_V_vror_VR(v0,32);
		}
	}
#endif
	v0 = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill, v0, -1));
	v0 = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill, v0, -2));	// expand last
	v0 = Q6_V_vmux_QVV( qlast, v0, vfill);	// replace post-bytes
	*voutp ++  = v0;						// write last
	// need one more?
	if( extra) *voutp = vfill;

}

void incopy_expand_2to4 (
		uint8_t * out,			// out, vector aligned
		uint8_t const * in,	// in, no alignment assumed, width*N pixels
		int width,				// width of input, >= 1
		int in_offset,			// byte to fill padding (including within groups)
		int left_pad,			// prepend this many pixels of zeros; 0..31
		int right_pad)			// append this many pixels of zeros; 0..31
{
	int n_in = 2;
	HVX_Vector * voutp = (HVX_Vector*) out;
	HVX_Vector qlast = q6op_Q_vsetq2_R( (left_pad + width)*4);
	HVX_Vector vfill = q6op_Vb_vsplat_R( in_offset );
	// get the first vector; stuff it with pixels on the left
	HVX_Vector v0 = Q6_V_vlalign_VVR( q6op_V_vldu_A((HVX_Vector const *)in), vfill, left_pad*n_in );
	in += 32*n_in -left_pad*n_in;			// point to the next chunk of pixels
	// nvec is 1 less than the # of output vectors needed for left_pad and width.
	int nvec = (left_pad + width-1)/32u;
	int extra =(nvec+1)*32 < left_pad + width + right_pad;

#if 0
	for(int i =0; i < nvec; i++){
		// shuffle fill bytes in
		*voutp ++ = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill, v0, -2));
		v0 = q6op_V_vldu_A((HVX_Vector const *)in);
		in += n_in * 32;
	}
#else
	in += 32*n_in;
	if( nvec > 0){
		for(int i =0; i < nvec>>1; i++){
			// store two at once
			// shuffle fill bytes in
			HVX_VectorPair outpair = Q6_W_vshuff_VVR( vfill, v0, -2);
			*voutp ++ = Q6_V_lo_W(outpair);
			*voutp ++ = Q6_V_hi_W(outpair);
			v0 = q6op_V_vldu_A((HVX_Vector const *)in);
			in += n_in * 32*2;
		}
		if( nvec &1){
			HVX_VectorPair outpair = Q6_W_vshuff_VVR( vfill, v0, -2);
			*voutp ++ = Q6_V_lo_W(outpair);
			v0 = Q6_V_vror_VR(v0,64);	// just need the upper half of that
		}
	}
#endif
	v0 = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill, v0, -2));
	v0 = Q6_V_vmux_QVV( qlast, v0, vfill);	// replace post-bytes
	*voutp ++  = v0;						// write last
	// need one more?
	if( extra) *voutp = vfill;
}


void incopy_expand_3to4 (
		uint8_t * out,			// out, vector aligned
		uint8_t const * in,	// in, no alignment assumed, width*N pixels
		int width,				// width of input, >= 1
		int in_offset,			// byte to fill padding (including within groups)
		int left_pad,			// prepend this many pixels of zeros; 0..31
		int right_pad)			// append this many pixels of zeros; 0..31
{

	int n_in = 3;
	HVX_Vector * voutp = (HVX_Vector*) out;
	HVX_Vector qlast = q6op_Q_vsetq2_R( (left_pad + width)*4);
	HVX_Vector vfill = q6op_Vb_vsplat_R( in_offset );
	// get the first vector; stuff it with pixels on the left
	HVX_Vector v0 = Q6_V_vlalign_VVR( q6op_V_vldu_A((HVX_Vector const *)in), vfill, left_pad*n_in );
	in += 32*n_in -left_pad*n_in;			// point to the next chunk of pixels
	// nvec is 1 less than the # of output vectors needed for left_pad and width.
	int nvec = (left_pad + width-1)/32u;
	int extra =(nvec+1)*32 < left_pad + width + right_pad;
	HVX_Vector vctl = *(HVX_Vector const*)copy3to4_cntrl;
	HVX_VectorPred qkeep = Q6_Q_vand_VR( Q6_V_vnot_V(Q6_V_vzero()), 0x0001010101);

	for(int i =0; i < nvec; i++){
		// shuffle fill bytes in, move along
		*voutp ++ = Q6_V_vmux_QVV( qkeep, Q6_V_vdelta_VV( v0, vctl), vfill);
		v0 = q6op_V_vldu_A((HVX_Vector const *)in);
		in += n_in * 32;
	}
	v0 = Q6_V_vdelta_VV(v0, vctl);
	v0 = Q6_V_vmux_QVV( Q6_Q_and_QQ(qlast,qkeep), v0, vfill);	// replace post-bytes
	*voutp ++  = v0;						// write last
	// need one more?
	if( extra) *voutp = vfill;
}

void incopy_expand_4to4 (
		uint8_t * out,			// out, vector aligned
		uint8_t const * in,	// in, no alignment assumed, width*N pixels
		int width,				// width of input, >= 1
		int in_offset,			// byte to fill padding (including within groups)
		int left_pad,			// prepend this many pixels of zeros; 0..31
		int right_pad)			// append this many pixels of zeros; 0..31
{
	int n_in = 4;
	HVX_Vector * voutp = (HVX_Vector*) out;
	HVX_Vector qlast = q6op_Q_vsetq2_R( (left_pad + width)*4);
	HVX_Vector vfill = q6op_Vb_vsplat_R( in_offset );
	// get the first vector; stuff it with pixels on the left
	HVX_Vector v0 = Q6_V_vlalign_VVR( q6op_V_vldu_A((HVX_Vector const *)in), vfill, left_pad*n_in );
	in += 32*n_in -left_pad*n_in;			// point to the next chunk of pixels
	// nvec is 1 less than the # of output vectors needed for left_pad and width.
	int nvec = (left_pad + width-1)/32u;
	int extra =(nvec+1)*32 < left_pad + width + right_pad;

	for(int i =0; i < nvec; i++){
		*voutp++ = v0;
		v0 = q6op_V_vldu_A((HVX_Vector const *)in);
		in += n_in * 32;
	}
	v0 = Q6_V_vmux_QVV( qlast, v0, vfill);	// replace post-bytes
	*voutp ++ = v0;							// write last
	// need one more?
	if( extra) *voutp = vfill;
}
