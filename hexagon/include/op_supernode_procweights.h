
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

#ifndef OP_SUPERNODE_PROCWEIGHTS_H_
#define OP_SUPERNODE_PROCWEIGHTS_H_

#include <stdint.h>
#include "nn_graph_types.h"
#include "hvx_inlines.h"

// this enables 'full-scaling' e.g. if the coeffs
// zero point is 100, they will be scaled by 210/256 and offset so that
//   0->  -82,    100 ->0,  255 -> +127
//
// without it they will be scaled by 1/2
//  and 0->  -50,  100 -> 0,   255 -> +77
#define ENABLE_FULL_V65_COEFF_SCALING


//////////////////////////////////////////////////////////////////////////
// rearrange filter weights for the convolution op
// The input is a 'flat' tensor of u8, shaped like this:
//
//     [ filt_height ][ filt_width ][ filt_depth ][ filt_batches ]
//
// In the process of converting, we pad filt_depth and filt_batches up to a multiple of 32 each,
// so each [ filt_depth ][ filt_batches ] could then be seen as an array of 32x32 matrices;
// In the output format, each of these occupies 8 consecutive vectors, but is interleaved.
//
// The output format is as follows
//
//     [ batch_hi ][ filt_height ] [ depth_hi ] [ filt_width ] [ depth_mid=8] [ batch_lo=32] [depth_lo=4]
//
//  .. where batch_hi, depth_hi are padded_batches /32, padded_depth/32
// The lower 5 bits of padded_depth is split into depth_mid (which selects one of 8 vectors) and depth_lo
// ( each group of 4 adjacent bytes).
//
//
//  Each output vector is a 32x4 matrix, which can be collected from a slice of 32 in the 'batches'
//  dimension, and a slice of 4 in the 'depth' dimension, so we can gather the 4 with unaligned reads
//  and shuffle them. Where filt_batches is >32, we can do two of these at once.
//  Depth padding can be done at both depth_mid and depth_lo levels; for instance, if filter_depth = 79,
//   that is 2*32 + 3*4 + 3, padded out to 3*32; so
//        depth_hi = 0,1,2;
//        depth_hi = 2:
//              depth_mid index = 0..2 are full  (4 each)
//              depth_mid index = 3 contains 3 valid  and one padding (in depth_lo)
//              depth_mid index = 4..7 are all padding
//
//
//   The strategy for the whole operation is:
//   (1) assuming filt_batches >= 64, we first process batches in full groups of 2*32.
//       each operation extracts 4 in depth dimension, 64 in batch dimension, and stores
//       two vectors to the output (which differ by 1 in the 'batch_hi' dimension)
//       - if the filter_depth is not a multiple of 4, there are partial ops to be done at the end
//   (2) second operation is to do 1..63 'extra' in the batch dimension, as above, but with a less efficient
//       operation which handles the padding; one or two 'batch_hi' are written according to whether the extra
//       is > 32.
//   (3) if the filter_depth, padded up to a multiple of 4, is not a multiple of 32, then we have padding to
//       do in the 'depth_mid' dimension; this means filling n vectors at the end of each group of 8, in all locations
///      with the maximum depth_hi value.
//

// load 64 bytes at each of 4 offsets from pos (using unalinged vector loads)
// and interleave them into 2 vectors.
//  Load:
//    A0 ... A31   a0 .. a31
//    B0 ... B31   b0 .. b31
//    C0 ... C31   c0 .. c31
//    D0 ... D31   d0 .. d31
//  result is:
//    A0 B0 C0 D0 A1 B1 C1 D1 ... C31 D31   (lo vector)
//    a0 b0 c0 d0 a1 b1 c1 d1 ... c31 d31   (hi vector)
//
//
static inline HVX_VectorPair
collect_double_4slice( uint8_t const * pos, int32_t depth_memstride )
{
    HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector const*)pos );  pos += depth_memstride;
    HVX_Vector v1 = q6op_V_vldu_A( (HVX_Vector const*)pos );  pos += depth_memstride;
    HVX_Vector v2 = q6op_V_vldu_A( (HVX_Vector const*)pos );  pos += depth_memstride;
    HVX_Vector v3 = q6op_V_vldu_A( (HVX_Vector const*)pos );  pos += depth_memstride;
    HVX_Vector shuf01 = Q6_V_lo_W(Q6_W_vshuff_VVR( v1,v0,-1));      // shuffle these
    HVX_Vector shuf23 = Q6_V_lo_W(Q6_W_vshuff_VVR( v3,v2,-1));      // and these
    return  Q6_W_vshuff_VVR( shuf23, shuf01,-2);         // and shuff together.

}


//
// This is like collect_double_4slice but it only loads 'n' rows; 'vfill' is used
// in other rows.
//  n must be in range 1..3.
//

static inline HVX_VectorPair
collect_double_Nslice( uint8_t const * pos, int32_t depth_memstride , int n, HVX_Vector vfill)
{
    HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector const*)pos );  pos += depth_memstride;
    HVX_Vector v1 = vfill;
    HVX_Vector shuf23 = vfill;
    if( n >= 2 ){
    	v1 = q6op_V_vldu_A( (HVX_Vector const*)pos );  pos += depth_memstride;
        if( n > 2){
        	HVX_Vector v2 = q6op_V_vldu_A( (HVX_Vector const*)pos );
        	shuf23 = Q6_V_lo_W(Q6_W_vshuff_VVR( vfill,v2,-1));      // shuffle these
        }
    }
    HVX_Vector shuf01 = Q6_V_lo_W(Q6_W_vshuff_VVR( v1,v0,-1));      // shuffle these
    return  Q6_W_vshuff_VVR( shuf23, shuf01,-2);         // and shuff together.

}
// inline to convert d4 (depth index, missing lower 2 bits) to a byte offset into output array.
// dmid = lower 3  bits: mul by 128
// dhi =  bits above that; mul by 128* (8*filt_width)
//
static inline int
repack_find_depth_offset( int d4, int filt_width )
{
	int dhi = d4 >> 3;
	if(0){	// conceptually
		int dmid = d4 & 7;
		return 128 *( dmid +  8*filt_width*dhi);
	}
	// a bit quicker... relies on d4 = dhi*8+dmid
	return 128 *( d4 +  8*(filt_width-1)*dhi);
};

// This struct contains all the parms to repack the filter coeffs.
struct repack_filter_parms {
  uint8_t *out_data;            		// out area (aligned)
  struct tensor const * filt_tensor;  	// input tensor, data & shape
  int16_t zero_offset;      			// byte to fill when padding
  int16_t signed_mode_sel;				// process to 'signed' mode: 0 = no, 1= offset only, 2= offset & shift.
  int32_t * gemsumb;					// area to put the the coefficient sums (aligned; or NULL)
  float    coeffscale;					// output: parms were scaled by this much (if signed_mode != 0)
  nn_sem_t done_sem;
};



//
// This executes the repack operation defined in the struct.
//  if signed_mode_sel = 0:
//    - values are repacked to the output area, and
//      the gemsumb is found on the actual values; it is stored at gemsumb.
//  if signed_mode_sel = 1:
//    - values are packed to output area, with 'zero_offset' subtracted from all,
//    - if zero_offset is in range 121..136, we subtract the offset and the result is saturated to
//        -128 .. 127;
//    - otherwise, we subtract the zero offset and divide by 2, (rounding towards 0), result being -127..127.
//    - in either case the gemsumb is the sum of these signed values.
//
// 'coeffscale' is set to the value by which the coefficients were scaled (1.0 or 0.5)
//
// it is ok if gemsumb is NULL; it will do everything else and not store gemsumb.
//
// Note:
// for signed_mode_sel = 0:
//   - Any elements in the 'padding' added to depth & height dimensions will be = zero_offset;
//   - gemsumb reflects this (so, in 'padding' batches, it will be
//         filter_width * filter_height * filt_batches_padded * zero_offset).
// for signed_mode_sel= 1
//   - Any elements in the 'padding' added to depth & height dimensions will be 0;
//   - gemsumb reflects this (so, in 'padding' batches, it will be 0).
//
// when compiled for < V65, only signed_mode_sel = 0 is supported.
//
//


static void
repack_filter_for_d32( struct nn_graph *nn, void *vrpfp )
{

	struct repack_filter_parms *rpfp = (struct repack_filter_parms *)vrpfp;

	struct tensor const * filt_tensor = rpfp->filt_tensor;


    int filt_height = filt_tensor->shape.filt_height;
    int filt_width = filt_tensor->shape.filt_width;
    int filt_depth = filt_tensor->shape.filt_depth;
    int filt_batches = filt_tensor->shape.filt_batches;
    const uint8_t* inptr = (const uint8_t*) filt_tensor->data;
    uint8_t* outptr = rpfp->out_data;

    // this is to avoid all the zero tests in various loops
    if( filt_width <=0 || filt_height <= 0 ){
    	nn_sem_post( &rpfp->done_sem);
    	return;
    }

    // work out the 'padded' format
    int filt_hw  = filt_height * filt_width;
    int filt_depth_padded = (filt_depth+31)& ~31;
    int filt_batches_padded = (filt_batches+31)& ~31;

    // number of full units of 4 in filter depth,  >=0
    int filt_depth_4unit = filt_depth >>2;
    // number of 'extra' to pad to a multiple of 4 (0..3)
    int filt_depth_extra = filt_depth & 3;
    // number of padding groups for 4 (0..7)
    int filt_depth_pad4 = (filt_depth_padded - filt_depth)>>2;

    // strides in bytes
    int out_height_stride = filt_width * filt_depth_padded * (8*128/32);
    int out_batch_32_stride = filt_height * out_height_stride;
    int in_width_stride = filt_depth * filt_batches;

    HVX_Vector v_fill = Q6_V_vsplat_R( Q6_R_vsplatb_R(rpfp->zero_offset));


    // find depth_extra_offs (only needed if filt_depth_extra != 0
    // this is what the offset would be in the 'd4' loop, if d4 reached a value = filt_depth_4unit
    int filt_depth_extra_offs = repack_find_depth_offset( filt_depth_4unit, filt_width);

    //
    // ok, process the batch units in multiples of 64
    //
    int batch_64_units = filt_batches >> 6;     // # of units of 64 (may be 0)
  /*  printf("[ %d %d %d %d] -> [%d %d]\n",
    		filt_height, filt_width, filt_depth, filt_batches, filt_depth_padded, filt_batches_padded);
    printf("b64_units = %d, batch_extra = %d\n", batch_64_units, filt_batches & 63);
*/
    for( int ib64= 0; ib64 < batch_64_units; ib64 ++ ){
        for( int h = 0; h < filt_height; h ++ ){
            for( int w = 0; w < filt_width; w++ ){
                uint8_t const * in_bhw = inptr +  ib64*64 + in_width_stride *( w + filt_width*h);
                uint8_t * out_bhw = outptr + out_batch_32_stride *2*ib64 + 1024 *w + out_height_stride * h;

                // collect filt_depth_4unit full units of 4x64 now
                for( int d4 = 0; d4 < filt_depth_4unit; d4 ++ ){
                    HVX_VectorPair vals = collect_double_4slice( in_bhw + filt_batches*4*d4,  filt_batches );
                    uint8_t * optr = out_bhw  +  repack_find_depth_offset(d4, filt_width);
                    *(HVX_Vector *)optr = Q6_V_lo_W( vals );                            // 64*ib64 ... 64*ib64+31
                    *(HVX_Vector *)(optr + out_batch_32_stride) = Q6_V_hi_W( vals );    // 64*ib64+32 .. 64*ib64+63
                }
                if( filt_depth_extra != 0 ){
                    HVX_VectorPair vals = collect_double_Nslice( in_bhw + filt_batches*4*filt_depth_4unit,
                          filt_batches, filt_depth_extra, v_fill );
                    uint8_t * optr = out_bhw  + filt_depth_extra_offs;
                    *(HVX_Vector *)optr = Q6_V_lo_W( vals );                            // 64*ib64 ... 64*ib64+31
                    *(HVX_Vector *)(optr + out_batch_32_stride) = Q6_V_hi_W( vals );    // 64*ib64+32 .. 64*ib64+63
                }
            }
        }
    }
    // any 'extra' batch units, beyond a multiple of 64

	int batch_extra = filt_batches & 63;
    if( batch_extra ){
        int two_stores = batch_extra > 32;      // need two stores per inner loop
        // masking for last partial, in units of 4 bytes; for batch_extra=32 we need all 1's so use vsetq2.
        HVX_VectorPred qpartial_batch = q6op_Q_vsetq2_R(batch_extra*4);

        for( int h = 0; h < filt_height; h ++ ){
            for( int w = 0; w < filt_width; w++ ){
                uint8_t const * in_bhw = inptr +  batch_64_units*64 + in_width_stride *( w + filt_width*h);
                uint8_t * out_bhw = outptr + out_batch_32_stride *2*batch_64_units + 1024 *w + out_height_stride * h;
                // collect filt_depth_4unit full units of 4x64 now...
                for( int d4 = 0; d4 < filt_depth_4unit; d4 ++ ){
                    HVX_VectorPair vals = collect_double_4slice( in_bhw + filt_batches*4*d4,  filt_batches );
                    uint8_t * optr = out_bhw  +  repack_find_depth_offset(d4, filt_width);
                    HVX_Vector vb0 = Q6_V_lo_W(vals);
                    HVX_Vector vb1 = Q6_V_hi_W(vals);
                    if( two_stores ){
                        *(HVX_Vector *)optr = vb0;
                        vb0 =  vb1;
                        optr += out_batch_32_stride;
                    }
                    // now do a store of v0 to optr, masking partial batches.
                    *(HVX_Vector *)optr = Q6_V_vmux_QVV( qpartial_batch, vb0, v_fill);
                }
                if( filt_depth_extra != 0 ){
                    HVX_VectorPair vals = collect_double_Nslice( in_bhw + filt_batches*4*filt_depth_4unit,
                         filt_batches, filt_depth_extra , v_fill);
                    uint8_t * optr = out_bhw  + filt_depth_extra_offs;
                    HVX_Vector vb0 = Q6_V_lo_W(vals);
                    HVX_Vector vb1 = Q6_V_hi_W(vals);
                    if( two_stores ){
                        *(HVX_Vector *)optr = vb0;
                        vb0 =  vb1;
                        optr += out_batch_32_stride;
                    }
                    // now do a store of v0 to optr, masking of partial batches.
                    *(HVX_Vector *)optr = Q6_V_vmux_QVV( qpartial_batch, vb0, v_fill);
                }
            }
        }
    }

    //  [ batch_hi ][ filt_height ] [ depth_hi ] [ filt_width ] [ depth_mid=8] [ batch_lo=32] [depth_lo=4]
    //
    // "backfill" - need to fill in "filt_depth_pad4" vectors in each group of 8 (depth_mid dimension),
    // to round up depth dimension from a multiple of 4 to multiple of 32. Only done in the last dim of "depth_hi"
    //
    // This is a 3-nested loop:
    //  outer loop:   (fil_batches_padded/32)* filt_height   over outer two dimensions, full range;
    //       [depth_hi dimension, is fixed at max index]
    //      middle loop:   filt_width
    //         inner loop:  'filt_depth_pad4'  vectors at the end of each group of 8.
    //
    //
    if( filt_depth_pad4 > 0 ){
    	int outercount = filt_batches_padded * filt_height >> 5;	// # outer loops
    	// offset the initial pointer
    	//
    	uint8_t * fill_base = outptr+ 128*(
    			((filt_depth_padded-1)>>5)* filt_width * 8	// offset to last depth_hi index
    			 + (8-filt_depth_pad4));				// offset to proper pos in depth_mid index
    	for( int h = 0; h < outercount; h++){
    		HVX_Vector *wp = (HVX_Vector*)(fill_base + out_height_stride * h);
    		for( int w = 0; w < filt_width; w ++ ){
                for( int j = 0; j < filt_depth_pad4; j++ ){
                    wp[w*8+j] = v_fill;
                }
    		}
    	}
    }

    // remaining :
    //  - if applicable, convert to signed;
    //  - in the process (or, if not converting, as a separate process)
    //     find the gemsumb values.
    //
    // The 'signed' modes aren't compiled in for < v65
    // (since we don't need them, and also we don't have the Q6_Vub_vadd_VubVb_sat in v60)
    //
#ifdef V65
    int signed_mode_sel = rpfp->signed_mode_sel;
#else
    int signed_mode_sel = 0;
#endif
    rpfp->coeffscale = 1.0f;

	int32_t * gemsumb = rpfp->gemsumb;
    if( signed_mode_sel == 0){
    	if( gemsumb !=0){
    		// find the sum of each *outer* buffer section, using vector ops
    		// we want the 32-bit sum of the 4 elements in each lane of vecs (i.e.
    		// the sums are done on all dimensions except 'batch').
    		int nvecper = filt_hw * filt_depth_padded >> 2; //
			int nb32 = filt_batches_padded >> 5;
			HVX_Vector  const * rp = (HVX_Vector const *) outptr;
			for( int ib32 = 0; ib32 < nb32; ib32++){
				HVX_Vector sum = Q6_V_vzero();
				for( int i = 0; i < nvecper; i++){
					sum = Q6_Vuw_vrmpyacc_VuwVubRub( sum, rp[i],  0x01010101);
				}
				rp += nvecper;
				((HVX_Vector*) gemsumb)[ib32]= sum;
			}
    	}
    }

#if defined(V65) && defined(ENABLE_FULL_V65_COEFF_SCALING)
    else{
		// need to adapt the coefficients to -128 .. 127 range.
		//  old way: if zero is in range 121 .. 135. then just offset them, with possible clipping
		//           - otherwise divide the range by 2
		//
		// new way:
    	//     if zero in in range 127..129, then offset as before,
		//     Otherwise find a scale factor k,
		//       and scale all as round( k*(w[i]-zero)/256)
    	// 'k' is found as the largest value that won't clip (128..253; with a special case of 127
    	// when the zero =0).
		//
		//  This is actually done as:
		//       tmp = k*w[i] + 128*256 - k*zero + 128
		//  and then  tmp >> 8 is the result, as u8, excess 128
		//   We can calculate k such that the addend     128*256-k*zero+128
		//   is always >=0 and the whole sum is always < 65536, so that no overflow occurs,
		//
		int zval = rpfp->zero_offset;
		int scaleK=0;
		int scaling_offs=0;
		int need_scale= 1;
		//
		// pick the largest scale that won't overflow the calculation
		// (note that the overall scale calc is compensated for whatever scaleK
		// that we actually use here).
		if( zval < 127 ){	///
			scaleK = (256*127+127)/(255-zval);
		}else if( zval > 129){
			scaleK = (256*128+127)/zval;
		}else{
			need_scale = 0;
		}
		if( need_scale ){
			// for zval >= 128, scaling_offs ranges over 1..200 ish;
			// for zval = 0,1,2, it's 32896. 32768, 32638 ... gradually decreasing.
			//
			scaling_offs = 128*256 + 128 - scaleK* zval;
			rpfp->coeffscale = (float)scaleK * (float)(1./256);
		}
		//printf("need_scale = %d, zer = %d, scaleK = %d, offs = %d, 0->%d, 255->%d\n", need_scale,  zval, scaleK, scaling_offs,
		//		(scaling_offs>>8)-128, ((scaling_offs+scaleK*255)>>8)-128);
		HVX_Vector k80 = Q6_Vb_vsplat_R( 0x80);
		HVX_Vector vdelt = Q6_Vb_vsub_VbVb(k80, v_fill);
		HVX_Vector voffs = Q6_Vh_vsplat_R(scaling_offs);
		int nvecper = filt_hw * filt_depth_padded >> 2;
		int nb32 = filt_batches_padded >> 5;
		HVX_Vector   * vp = (HVX_Vector *) outptr;
		HVX_Vector suminit = Q6_V_vsplat_R( -4*128 * nvecper );	// correction offset
		int scaleKsplat = Q6_R_vsplatb_R( scaleK);

		if( nvecper > 0)	// avoid zero tests for inner loops
		 for( int ib32 = 0; ib32 < nb32; ib32++){
			HVX_Vector sum = suminit;

			if( !need_scale){
				for( int i = 0; i < nvecper; i++){
					HVX_Vector val = Q6_Vub_vadd_VubVb_sat( vp[i], vdelt );	// adjust the value
					sum = Q6_Vuw_vrmpyacc_VuwVubRub( sum, val,  0x01010101);	// sum it
					vp[i] = Q6_V_vxor_VV( val, k80);			// conv to signed and store
				}
			}else{
				for( int i = 0; i < nvecper; i++){
					HVX_Vector vx = vp[i];
					HVX_VectorPair prod = Q6_W_vcombine_VV( voffs, voffs);
					prod = Q6_Wuh_vmpyacc_WuhVubRub( prod, vx, scaleKsplat);	// k*x[i] + offs
					HVX_Vector vy = Q6_Vb_vshuffo_VbVb( Q6_V_hi_W(prod), Q6_V_lo_W(prod));	// >> 8
					sum = Q6_Vuw_vrmpyacc_VuwVubRub( sum,  vy,  0x01010101);	// sum it
					vp[i] = Q6_V_vxor_VV( vy, k80);			// conv to signed and store
				}
			}
			vp += nvecper;
			if( gemsumb != NULL){
				*(HVX_Vector *)gemsumb = sum;
				gemsumb += 32;
			}
		}
    }
#endif	// defined(V65) && defined(ENABLE_FULL_V65_COEFF_SCALING)
#if defined(V65) && !defined(ENABLE_FULL_V65_COEFF_SCALING)
    else {
   //printf("signed_mode_sel= %d; zoff= %d\n",signed_mode_sel, rpfp->zero_offset);
    	// ====
    	// signed_mode_sel = 1
    	// recode by subtracting 'vfill' from each value; find the sums of the results
    	// Done as follows:
    	//    (1) add signed 'tweak' to unsigned byte using saturating add; result is unsigned,
    	//        and +128 more than it should be for the output buffer;
    	//    (2) xor with 0x80 before storing to correct, making it signed.
    	//    (3) sums are found using the value in (1) (so we can use unsigned vrmpy) and then we
    	//      subtract 128*N from the result
    	// (it is assumed that vdelt is small, e.g. +/-12, or we wouldn't be doing this)
    	// ====
    	// signed_mode_sel = 2
    	// in this case we need to (conceptually):
    	//   - subtract the zero point, then, divide by 2, rounding towards 0;
    	//   - resulting value is always in range -127 ..127, stored in i8. Also find sum.
    	// We actually do this, for each input x:
    	//   (1) where x < zeroval, add 1 to x    (to make it round up)
    	//   (2)  find y = (x-zeroval)>>1		   ( vnavg op, ub->b)
    	//         This is the proper output.
    	//   (3) xor that with 0x80 to make u8; acc with vrmpy; subtract 128*N at the end
    	//
    	//
    	HVX_Vector k80 = Q6_Vb_vsplat_R( 0x80);
    	HVX_Vector vdelt = Q6_Vb_vsub_VbVb(k80, v_fill);
    	HVX_Vector kFF = Q6_V_vsplat_R( -1);
    	//
		int nvecper = filt_hw * filt_depth_padded >> 2;
		int nb32 = filt_batches_padded >> 5;
		HVX_Vector   * vp = (HVX_Vector *) outptr;
		HVX_Vector suminit = Q6_V_vsplat_R( -4*128 * nvecper );	// correction offset

		int zval = rpfp->zero_offset;
		int div_by_2 = (zval <= 120 || zval >= 136);
		if( div_by_2 )
		    rpfp->coeffscale = 0.5f;

		if( nvecper > 0)	// avoid zero tests for inner loops
		 for( int ib32 = 0; ib32 < nb32; ib32++){
			HVX_Vector sum = suminit;

			if( !div_by_2){
				for( int i = 0; i < nvecper; i++){
					HVX_Vector val = Q6_Vub_vadd_VubVb_sat( vp[i], vdelt );	// adjust the value
					sum = Q6_Vuw_vrmpyacc_VuwVubRub( sum, val,  0x01010101);	// sum it
					vp[i] = Q6_V_vxor_VV( val, k80);			// conv to signed and store
				}
			}else{
				for( int i = 0; i < nvecper; i++){
					HVX_Vector vx = vp[i];
					HVX_VectorPred qlt = Q6_Q_vcmp_gt_VubVub( v_fill, vx);	// vx < center
					vx = Q6_Vb_condnac_QVbVb( qlt, vx, kFF);	// add 1 where
					HVX_Vector vy = Q6_Vb_vnavg_VubVub( vx, v_fill);
					vp[i] = vy;
					sum = Q6_Vuw_vrmpyacc_VuwVubRub( sum,  Q6_V_vxor_VV( vy, k80),  0x01010101);	// sum it
				}

			}
			vp += nvecper;
			if( gemsumb != NULL){
				*(HVX_Vector *)gemsumb = sum;
				gemsumb += 32;
			}
		}
    }
#endif // defined(V65) && !defined(ENABLE_FULL_V65_COEFF_SCALING)

    nn_sem_post( &rpfp->done_sem);
}

#endif /* OP_SUPERNODE_PROCWEIGHTS_H_ */
