
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
#include <stdint.h>
#include "nn_graph.h"
#include "hvx_inlines.h"
#include "op_supernode_procweights.h"
#include "quantize.h"

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

/* >> in header file op_supernode_procweights.h >>>
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
*/
#if defined(V66) || defined(V65)
// This operates on a packed slice of 32 batches; contained in filterslice_len/128 vectors
// at srcp. Each vector contains 32 groups of 4 unsigned weights, which belong to 32 filter batches
// It finds the min and max values within each of the 32 batches
// The returned vector contains 32 groups of bytes {  min, max, garbage, garbage }
//
static  void __attribute__((noinline,unused))
find_slice_minmax( HVX_Vector *resultp, uint8_t const * srcp, int filterslice_len )
{
	HVX_Vector maxvals = Q6_V_vzero();
	HVX_Vector minvals = Q6_V_vnot_V(maxvals);

	HVX_Vector const *vp = (HVX_Vector const *)srcp;

	for(int i = 0; i  < filterslice_len/128u; i++ ){
		HVX_Vector x =  vp[i];
		minvals = Q6_Vub_vmin_VubVub( minvals, x);
		maxvals = Q6_Vub_vmax_VubVub( maxvals, x);
	}
	HVX_VectorPair shuf = Q6_Wb_vshuffoe_VbVb(Q6_V_vnot_V(maxvals), minvals );
	minvals = Q6_Vub_vmin_VubVub( Q6_V_lo_W(shuf), Q6_V_hi_W(shuf));
	// each group of 4 now has {min, ~max, min, ~max} but we need one more reduction
	minvals = Q6_Vub_vmin_VubVub( minvals, Q6_Vh_vshuffo_VhVh(minvals,minvals ));
	*resultp = Q6_V_vxor_VV( minvals, Q6_V_vsplat_R(0xFF00));	// fix the ~max ->max
	Q6_dcfetch_A(resultp);
}
//
// This operates 'slices' packed slices, each of 32 batches, contained in filterslice_len/128 vectors
// at srcp. Each vector contains 32 groups of 4 unsigned weights, which belong to 32 filter batches;
// the values are u8, but each w[i] represents a value w[i] - zval (where zval is 0..255).
// We need to convert them all (in-place) to signed 8-bit values with proper 0 (so any w[i] = zval
// will always convert to w[i] = 0 in the result).
//
// This is done by a scaling process; e.g if zval = 160, then the input values represent
//   -160..95, and we need to 'squeeze' this to -128..127. We find the largest scale factor K
//  which supports this and can be expressed as a fraction of 256; in this case it's K = 205
//  so that -160->-128, and 95->76
// However, we want to find the scaling per-lane, so we find the smalled value actually in use in
// each 'lane'; if the smallest value in a lane is 15, then that represents 15-160 = -145, so
// in that lane we can use a scale of 226/256   which gives -145 -> -128, 95->84.
//
// In this case with zval = 160, if the smallest x[i] seen is >= 32, we don't need to scale,
// we can just subtract 160 from all and wind up with -128.. 95 (in this case K=0 indicating
// no scaling; but we still need to offset). In fact, we allow some borderline cases to
// map to this (e.g. if min val is 31, we do it as unscaled, but saturate code 31 to -128; but
// but if min_val = 30, it will be scaled by K = 253.
//
// when zval > 128, we need to find the max val in each lane instead, to find the max K.
//
//
// Process is:
//  (1) find scaleK and offset in each lane.
//    if zval is in 127,128,129, then scaleK=0 for all lanes.
///   Otherwise, if zval >=130, find the 'min' in each lane; and scaleK based on that; or
//    if <=126, find the max in each lane, and scaleK based on that.
//  (2) Apply the scaling to the weight data. In the process, find the sum of each output's weights.
//
//
// This also sets up an output array scalefac[i] which contains the scale factor used in each for each lane, with
// 31 fractional bits; if the lane is unscaled, it will be 0x7FFFFFFF, otherwise it will be K<<23.
//
// more detail:
//
// For each output, it finds a value 'scaleK'  and offset.
// scaleK is in range 128..255, or has the value 0 indicating 'no scaling'.
//
//
// This is the logic per-channel:
//
// (1) find min or max of the u8 values across filter depth (as needed depending on zval)
//
// (2) if min(all) < zval - 129   (use formula [A])
//     elif max(all) > zval +128  (use formula [B])
//      else (formula [C])
//
//      Note that [A] is possible only when zval >= 130
//            and [B] is possible only when zval <= 126
//       ... so at most one is possible in a given situation
//
//      Formula [A]:
//            scaleK.ub = (128*256 + 127)/(zval-min(all))  (denom = 130 .. 255;result = 253..129)
//            offset.h = 128*256 +  RND - scaleK*zval;
//      Formula [B]:
//            scaleK.ub = (127*256 + 127)/(max(all)-zval)  (denom = 129 .. 255; result = 253..127)
//            offset.h = 128*256 +  RND - scaleK*zval;
//
//      Formula [C]:
//            scaleK = 0
//            offset = 0 (not actually used)
//
//  The 'divisions' are done using a 7-bit lookup, with one of two different tables.
//
//  if zval >= 128,  we find (zval-128)-min(all), saturating to u8; then index the formula 'A' table
//   which contains { 0, 0, 253, 251, 249, ... 129, 129 }
//  if zval < 128  we find max(all)-(zval+128), saturating to u8, then index the formula 'B' table
//   which contains { 0,  253, 251, 249, 247, ...  128, 127 }
//
//
// For the scaling step, it is done as follows:
//  (1) first convert each set of weights to an unsigned value (excess 128), using two paths:
//    for scaleK  =0, we use a saturating ub-b ->ub subract, subtracting 'zval-128' from each byte;
//    for scaleK != 0, we muliply by K, and add offset, then >>8. The offset has the rounding and bias
//    cooked into it already.
//  (2) the sum of these u8 value is found. The 'gemsumb' result is obtained by subtracting 128*count from this.
//  (3) the values are xor'd with 0x80 and stored as the signed results.


static  void __attribute__((unused, noinline))
do_perlane_weight_scaling(  uint8_t  * weights,
		int slices,						// of of slices of 32 outputs
		int filterslice_len,			// # of bytes in each slice (must be a multiple of 32)
		int zval,						// the filter's 'zero point'
		int32_t * scalefac_vecp,		// store scaleK as a fixed-point with 32 fractional bits
		int32_t *gemsumb				// store the sum of the scaled coeffs in each lane (if not NULL)

		)
{
	static const uint8_t divide_tables[2][128] __attribute__((aligned(128))) = {
		{  // 'A' case:   (128*256+127)/(i+128)   (but 0 where i <= 1)
			  0, 171,   0, 170,  253, 169, 251, 168,  249, 167, 247, 166,  245, 166, 243, 165,
			241, 164, 240, 163,  238, 162, 236, 162,  234, 161, 233, 160,  231, 159, 230, 158,
			228, 158, 226, 157,  225, 156, 223, 155,  222, 155, 220, 154,  219, 153, 217, 153,
			216, 152, 215, 151,  213, 150, 212, 150,  210, 149, 209, 148,  208, 148, 206, 147,
			205, 146, 204, 146,  203, 145, 201, 144,  200, 144, 199, 143,  198, 143, 196, 142,
			195, 141, 194, 141,  193, 140, 192, 139,  191, 139, 190, 138,  189, 138, 187, 137,
			186, 137, 185, 136,  184, 135, 183, 135,  182, 134, 181, 134,  180, 133, 179, 133,
			178, 132, 177, 132,  176, 131, 175, 131,  174, 130, 174, 130,  173, 129, 172, 129
		},
		{  // 'B' case:   (127*256+127)/(i+128)   (but 0 where i =0)
			  0, 169, 253, 169,  251, 168, 249, 167,  247, 166, 245, 165,  243, 164, 241, 164,
			239, 163, 238, 162,  236, 161, 234, 160,  233, 159, 231, 159,  229, 158, 228, 157,
			226, 156, 225, 156,  223, 155, 222, 154,  220, 153, 219, 153,  217, 152, 216, 151,
			214, 151, 213, 150,  211, 149, 210, 149,  209, 148, 207, 147,  206, 147, 205, 146,
			203, 145, 202, 145,  201, 144, 200, 143,  199, 143, 197, 142,  196, 141, 195, 141,
			194, 140, 193, 140,  191, 139, 190, 138,  189, 138, 188, 137,  187, 137, 186, 136,
			185, 135, 184, 135,  183, 134, 182, 134,  181, 133, 180, 133,  179, 132, 178, 132,
			177, 131, 176, 131,  175, 130, 174, 130,  173, 129, 172, 129,  171, 128, 170, 127
		}

	};

	int nvecper = filterslice_len/128u;	// filterslice_len must be a multiple of 128.

	// this is zval-128 as a signed byte
	// (we also use it as zval + 128 as an unsigned, for the 'B' path)
	HVX_Vector vdelt = Q6_Vb_vsplat_R( zval-128);			// value to offset, for non-scaled cases.

	int RND = 128;
	HVX_Vector k7fffffff = Q6_V_vsplat_R(0x7FFFFFFF);
	for(int islc = 0; islc < slices; islc++){
		HVX_Vector scaleK_vec = Q6_V_vzero();
		HVX_Vector offset_vec =  Q6_V_vzero();
		HVX_Vector scalefac_vec = k7fffffff;
		HVX_VectorPred noscale_lanes = Q6_Q_vcmp_eq_VwVw( Q6_V_vzero(), Q6_V_vzero());


		uint8_t *slice_weights = weights + islc * filterslice_len;

		int is_A = zval >= 130;			// use [A] formula based on min.
		if( is_A || zval <= 126 ) {		// use [A] or [B] ?
			HVX_Vector vlookup = *(HVX_Vector *) divide_tables[is_A?0:1];
			// first find min of all (if A) or max (if B)

			HVX_Vector maxvals = Q6_V_vzero();
			HVX_Vector xor_read = Q6_V_vsplat_R( is_A? -1:0);
			HVX_Vector const *vp = (HVX_Vector const *)slice_weights;
			for(int i = 0; i  < nvecper; i++ ){
				HVX_Vector x =  vp[i];
				maxvals = Q6_Vub_vmax_VubVub( maxvals, Q6_V_vxor_VV(x,xor_read));
			}
			// reduce across four bytes in each lane
			HVX_VectorPair shuf = Q6_Wb_vshuffoe_VbVb(maxvals, maxvals );
			maxvals = Q6_Vub_vmax_VubVub( Q6_V_lo_W(shuf), Q6_V_hi_W(shuf));
			shuf = Q6_Wh_vshuffoe_VhVh(maxvals, maxvals );
			maxvals = Q6_Vub_vmax_VubVub( Q6_V_lo_W(shuf), Q6_V_hi_W(shuf));

			HVX_Vector tbl_index;
			if( is_A){		// max( (zval-128)-minval, 0 )
				tbl_index = Q6_Vub_vsub_VubVub_sat( vdelt,  Q6_V_vnot_V(maxvals) );
			}else{			// max( maxval-(zval+128,0))
				tbl_index = Q6_Vub_vsub_VubVub_sat(  maxvals, vdelt);
			}
			// now we can map these in the table index.

			scaleK_vec = Q6_Vb_vlut32_VbVbI( tbl_index, vlookup, 0);
			scaleK_vec = Q6_Vb_vlut32or_VbVbVbI( scaleK_vec,tbl_index, vlookup, 1 );
			scaleK_vec = Q6_Vb_vlut32or_VbVbVbI( scaleK_vec,tbl_index, vlookup, 2 );
			scaleK_vec = Q6_Vb_vlut32or_VbVbVbI( scaleK_vec,tbl_index, vlookup, 3 );
			// find  offset = 128*256 +  RND - scaleK*zval;
			offset_vec = Q6_V_vsplat_R( -1 - (128*256 +  RND));	//1's complement of constant part
			offset_vec = Q6_Vuw_vrmpyacc_VuwVubRub( offset_vec, scaleK_vec, zval&0xFF );
			// dup to both lanes and 1's complement.
			offset_vec = Q6_V_vnot_V(Q6_Vh_vshuffe_VhVh(offset_vec, offset_vec));

			// make the 'scalefac_vec' value: K <<23 but in lanes where K =0, it's 0x7fffffff
			noscale_lanes = Q6_Q_vcmp_eq_VwVw( scaleK_vec, Q6_V_vzero());
			scalefac_vec = Q6_Vw_vasl_VwR( scaleK_vec, 23);			// bit 31 is garbage, though...
			scalefac_vec = Q6_V_vandor_VQR( scalefac_vec, noscale_lanes, -1);	// force to all 1's where K=0
			scalefac_vec = Q6_V_vand_VV(scalefac_vec, k7fffffff );	// fix bit 31.

			// note, the 'offset_vec' is a don't care in lanes where scaleK = 0.
		}
		*(HVX_Vector*)(scalefac_vecp+32*islc) = scalefac_vec;


		HVX_Vector   * vp = (HVX_Vector *) slice_weights;
		HVX_Vector k80 = Q6_Vb_vsplat_R(0x80);

		HVX_Vector sum = Q6_V_vsplat_R( -4*128 * nvecper ); // correction offset for gemsumb;
		for( int i = 0; i < nvecper; i++){
			HVX_Vector vx = vp[i];
			HVX_Vector val0 = Q6_Vub_vsub_VubVb_sat( vx, vdelt );      // adjust the value - non-scaled path
			// scaled path.
			HVX_VectorPair prod ;
			prod = Q6_Wuh_vmpy_VubVub(vx, scaleK_vec);    // k*x[i] + offs
			HVX_Vector sop0 = Q6_Vh_vadd_VhVh(offset_vec, Q6_V_lo_W(prod));
			HVX_Vector sop1 = Q6_Vh_vadd_VhVh(offset_vec, Q6_V_hi_W(prod));
			HVX_Vector val1 = Q6_Vb_vshuffo_VbVb( sop1, sop0);     // >> 8

			HVX_Vector val = Q6_V_vmux_QVV(noscale_lanes, val0, val1);
			sum = Q6_Vuw_vrmpyacc_VuwVubRub( sum,  val,  0x01010101);       // sum it
			vp[i] = Q6_V_vxor_VV( val, k80);                        // conv to signed and store
		}
		if( gemsumb != NULL){
			*(HVX_Vector *)(gemsumb + 32*islc) = sum;
		}
	} //for slices
}
#endif
static void repack_filter_for_d32_REF( struct nn_graph *nn, void *vrpfp );

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
//-> the vrpfp parameter is really a struct repack_filter_parms *.
//-> operation posts vrpfp->done_sem when done.

void
repack_filter_for_d32( struct nn_graph *nn, void *vrpfp )
{
	if(0){		// to use the reference instead
		repack_filter_for_d32_REF( nn, vrpfp );
		return;
	}

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

    //
    // WORKAROUND:
    // Currently the depth 'backfill' (done after the copy) is causing some issues (hangs) on v65
    // (possibly related to VM?)
    // So now, I'm prefilling the whole array with 'zero_offset' before doing the conversion,
    // and skipping the backfill. However, it's still weird: the prefill must be done
    //. even if filt_depth_pad4 == 0, and problems only occur in some cases.
    //
    if( 1 || filt_depth_pad4 != 0) {
    	vmemset_asm( outptr, rpfp->zero_offset, filt_hw *filt_depth_padded* filt_batches_padded );
    }
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
    // DISABLED due to workaround, see above
#if 0
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
#endif
    // remaining :
    //  - if applicable, convert to signed;
    //  - in the process (or, if not converting, as a separate process)
    //     find the gemsumb values.
    //
    // The 'signed' modes aren't compiled in for < v65
    // (since we don't need them, and also we don't have the Q6_Vub_vadd_VubVb_sat in v60)
    //
#if defined(V65) || defined(V66)
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

#if (defined(V65) || defined(V66)) && defined(ENABLE_FULL_V65_COEFF_SCALING)
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
                  logmsg(nn,2,"filt zero = %d",zval);
		int scaleK=0;
		//
		// pick the largest scale that won't overflow the calculation
		// (note that the overall scale calc is compensated for whatever scaleK
		// that we actually use here).
#if defined(V66) || defined(V65)
                int need_scale= 1;
#if 1
                if( zval < 127 ){       /// max = 255-zval = 
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
                        rpfp->coeffscale = (float)scaleK * (float)(1./256);
                }
#endif
#if 1	// fully vectorized per-lane-scaling
                do_perlane_weight_scaling( outptr, filt_batches_padded/32u, filt_hw*filt_depth_padded * 32,
                		zval, rpfp->scalefac_vec,  gemsumb );

#else
               // w groups of 4 scaleK
               uint32_t  * scaleK_vec = nn_memalign(128, 4*sizeof(uint8_t)  * filt_batches_padded);
               // w groups of 2 'offset'
               uint32_t * offset_vec = nn_memalign(128, 2*sizeof(uint16_t) * filt_batches_padded);
               int32_t offset, RND = 128 ;
               logmsg(nn,2,"in depth = %ld out depth = %ld zero = %ld",filt_depth, filt_batches, zval);

               union { HVX_Vector as_v ; uint8_t as_u8[128]; } minmax_union;
               minmax_union.as_u8[0] = 0; // (so it's used)

               for( int w= 0; w < filt_batches; w ++ ){
                   int32_t ms= -1, max = -2000;
#if 0 // vector min/max
                   int wlane = w&31;
                   if( wlane == 0 ) {
                	   // need to find min/max of batches w .. w+31
                	   // .. reading from the rearranged output data
                       int nslice = filt_hw*filt_depth_padded * 32;	// bytes per group of 32 batches
                       find_slice_minmax(  &minmax_union.as_v, outptr + w*filt_hw*filt_depth_padded, nslice);
                   }
                   int minwz = zval-minmax_union.as_u8[wlane*4];		// max( -(filt[i]-zval) )
                   int maxwz = minmax_union.as_u8[wlane*4+1]-zval;	// max( filt[i] - zval )
                   // at least one of minwz,maxwz is >= 0.
                   max = max_i32( minwz,maxwz);
                   ms = (maxwz>minwz)?1:-1;

#else
                   // note: if negative weight & positive weights reach the same peak, the vector code
                   // above will decide that ms=-1, whereas in the code below, it depends on whichever
                   // is found first. Otherwise they should be identical (it won't make any difference
                   // since this can only happen when max <128)
                   for( int h = 0; h < filt_hw*filt_depth; h ++ ){
                       int32_t tmp =  inptr[h*filt_batches+w] ;
                       if(tmp == 0) logmsg(nn,2," min found at %d", h);
                       if(tmp ==255) logmsg(nn,2," max found at %d", h);
                       tmp = tmp - zval;
                       int s = 1;
                       if(tmp <0) { tmp = -tmp; s = -1;}
                       if(tmp > max) { max = tmp; ms = s; }
                   }
#endif
                   if(ms == -1 && max > 129 ){     ///
                       scaleK = (128 * 256 + 127)/max;
                       offset = 128*256 +  RND - scaleK*zval;
                   }else if(ms == 1 && max > 128){
                       scaleK = (127 * 256 + 127)/max;
                       offset = 128*256 +  RND - scaleK*zval;
                   } else {
                       scaleK = 0;
                       offset = 0; //-256 *zval;
                   }
                   if(scaleK==0) rpfp->scalefac_vec[w] = 0x7fffffff ;   //1.0
                   else          rpfp->scalefac_vec[w] = (int)( (float)(2147483648.0/256.0) *(float)scaleK); //val 0.5 to 1
                
                   logmsg(nn,2,"sf[%d]     %f ",w, (float)scaleK *(1.0f/256.0f));
                   scaleK_vec[w] = Q6_R_vsplatb_R(scaleK);
                   offset_vec[w] = Q6_R_combine_RlRl(offset,offset);
               }
               for( int w= filt_batches; w < filt_batches_padded; w ++ ){
                   rpfp->scalefac_vec[w] = 0x7fffffff;
                   scaleK_vec[w] = 0;
                   offset_vec[w] = 0;
               }
                HVX_Vector k80 = Q6_Vb_vsplat_R( 0x80);
                HVX_Vector vdelt = Q6_Vb_vsub_VbVb(k80, v_fill); //v_fill is splat zoffset
                int nvecper = filt_hw * filt_depth_padded >> 2;
                int nb32 = filt_batches_padded >> 5;
                HVX_Vector   * vp = (HVX_Vector *) outptr;
                HVX_Vector suminit = Q6_V_vsplat_R( -4*128 * nvecper ); // correction offset

                HVX_Vector * K = (HVX_Vector *)scaleK_vec;
                HVX_Vector * L = (HVX_Vector *)offset_vec;

                if( nvecper > 0)        // avoid zero tests for inner loops
                 for( int ib32 = 0; ib32 < nb32; ib32++){
                        HVX_Vector sum = suminit;
                        HVX_Vector val;
                        HVX_Vector svec = K[ib32];
                        HVX_Vector voffs = L[ib32];
                        HVX_VectorPred  scale_sel = Q6_Q_vcmp_eq_VwVw(svec, Q6_V_vsplat_R(0));
                        for( int i = 0; i < nvecper; i++){
                                HVX_Vector vx = vp[i];
                                HVX_Vector hi, lo, val0, val1;
                                val0 = Q6_Vub_vadd_VubVb_sat( vx, vdelt );      // adjust the value

                                HVX_VectorPair prod ;
                                prod = Q6_Wuh_vmpy_VubVub(vx, svec);    // k*x[i] + offs
                                lo = Q6_Vh_vadd_VhVh(voffs, Q6_V_lo_W(prod));
                                hi = Q6_Vh_vadd_VhVh(voffs, Q6_V_hi_W(prod));
                                val1 = Q6_Vb_vshuffo_VbVb( hi, lo);     // >> 8

                                val = Q6_V_vmux_QVV(scale_sel, val0, val1);
                                sum = Q6_Vuw_vrmpyacc_VuwVubRub( sum,  val,  0x01010101);       // sum it
                                vp[i] = Q6_V_vxor_VV( val, k80);                        // conv to signed and store
                        }
                        vp += nvecper;
                        if( gemsumb != NULL){
                                *(HVX_Vector *)gemsumb = sum;
                                gemsumb += 32;
                        }
                }
                nn_free(scaleK_vec);
                nn_free(offset_vec);
#endif
               int32_t gm = 0x7fffffff;
               for(int i=0; i < filt_batches_padded; i++)
               {
                    logmsg(nn, 2, "%d) %08lx", i, rpfp->scalefac_vec[i]);
                    if(rpfp->scalefac_vec[i] < gm) gm = rpfp->scalefac_vec[i];
               }
               rpfp->coeffscale = (float)gm / (float)(0x7fffffff);
    }
#else   //V65 - effectively dead code with per channel model
		int scaling_offs=0;
		int need_scale= 1;
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
#endif
#endif	// (defined(V65) || defined(V66)) && defined(ENABLE_FULL_V65_COEFF_SCALING)
#if (defined(V65) || defined(V66)) && !defined(ENABLE_FULL_V65_COEFF_SCALING)

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

//
// reference (scalar version)
//

static void
repack_filter_for_d32_REF( struct nn_graph *nn, void *vrpfp )
{

	struct repack_filter_parms *rpfp = (struct repack_filter_parms *)vrpfp;

	struct tensor const * filt_tensor = rpfp->filt_tensor;

    int filt_height = filt_tensor->shape.filt_height;
    int filt_width = filt_tensor->shape.filt_width;
    int filt_depth = filt_tensor->shape.filt_depth;
    int filt_batches = filt_tensor->shape.filt_batches;
    const uint8_t* inptr = (const uint8_t*) filt_tensor->data;
    uint8_t* outptr = rpfp->out_data;

    // work out the 'padded' format
    int filt_depth_padded = (filt_depth+31)& ~31;
    int filt_batches_padded = (filt_batches+31)& ~31;
    int fdp32 = filt_depth_padded>>5;

    int fillval = rpfp->zero_offset;
    int32_t * sumb_ptr = rpfp->gemsumb;	// maybe NULL

    // copy, reorder, pad, sum
    for(int b = 0; b < filt_batches_padded; b++){
    	int bsum = 0;
    	for( int d = 0; d < filt_depth_padded; d++){
    		for( int h = 0; h < filt_height; h++ ){
    			for( int w = 0; w < filt_width; w++){
    				unsigned val = fillval;
    				if( b < filt_batches && d < filt_depth)
    					val = inptr[ b + filt_batches*( d + filt_depth * (w + filt_width *h))];
    				bsum += val;
    				unsigned dx = (d &3) | ((b&31)<<2) | ((d&0x1C)<<5);
    				outptr[ dx + 1024*(w + filt_width * ((d>>5) + fdp32*(h+filt_height*(b>>5)))) ]= val;
    			}
    		}
    	}
    	if(sumb_ptr!=0) sumb_ptr[b]= bsum;
    }

#if (defined(V65) || defined(V66)) && defined(ENABLE_FULL_V65_COEFF_SCALING)
    int filt_hw  = filt_height * filt_width;
    if( rpfp->signed_mode_sel) {
		int zval = rpfp->zero_offset;
		int scaleK=0;
		int scaling_offs=0;
		int need_scale= 1;
		//
		// pick the largest scale that won't overflow the calculation
		// (note that the overall scale calc is compensated for whatever scaleK
		// that we actually use here).
		if( zval < 127 ){	///
			// largest scaleK such that round( (255-zval)*scaleK/256 ) <= 127
			scaleK = (256*127+127)/(255-zval);
		}else if( zval > 129){
			// largest scaleK such that round( (0-zval)*scaleK/256 ) >= -128
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
		// Examples for need_scale = 1:
		//   zval  scaleK  scaling_off   0->  255->
		//   ====  ======   ==========   ==   ======
		//    0     127      32896        0   127
		//    1     128      32768        0   127    (0 and 1 both map to 0).
		//    20    138      30136      -11   127
		//    80    186      18016      -58   127
		//    120   241       3976     -113   127
		//    126   253       1018     -125   127
		//
		//    129   255          1     -128   126
		//    140   234        136     -128   105
		//    180   182        136     -128    53
		//    220   149        116     -128    20
		//    250   131        146     -128     3
		//    254   129        130     -128     1
		//    255   129          1     -128     0
		//
		//    The overall computation is equivalent to out[i] = round( (in[i]-zval)*scaleK/256 )
		//     i.e. out[i] =   ( in[i] * scaleK - (zval*scaleK) + 128 ) >> 8
		//                 =  (( in[i] * scaleK + 128*256 + 128 - (zval*scaleK)) >> 8  ) - 128
		//                 =  (( in[i] * scaleK + scaling_offs) >> 8  ) - 128
		//   (the last 2 cols of the table above are out[i] when in[i] = 0 and in[i] = 255)
		//
		// The scaling is done as follows:
		//    (1) find tmp = (in[i]*scaleK + scaling_off)>>8.
		//       By construction of scaleK and scaling_off, this is always in range 0..255, and
		//       furthermore, for in[i] = zval,  tmp= 128.
		//    (2) the signed result out[i] = tmp-128  (or tmp^0x80)
		//
		// The sum of the out[i] can be obtained by summing the 'tmp' and subtracting filt_depth_padded * filt_hw*128
		//
		int bscale = filt_depth_padded * filt_hw * 32;
		int8_t delt= 0x80-fillval;

		for( int b = 0; b < filt_batches_padded; b++){
			// go through each batch slice one at time so we can find the sums
			// separately ( lower 5 bits of 'b' index are inserted to form bits 6..2
			// of the address; the upper bits get multiplied by 'bscale' to form the upper part)
			//
			int bsum = -128*filt_depth_padded * filt_hw;
			if (!need_scale){
				for( int hwd = 0; hwd < filt_depth_padded * filt_hw; hwd++ ){
					unsigned pos = ((hwd& 3) + ((b&0x1f)<<2) + ( hwd>>2)*128) + bscale * (b>>5);
					int w = outptr[pos]-delt;
					if( w < 0) pos = 0;
					else if ( w > 255) w =255;
					bsum += w;
					outptr[pos] =  w^0x80;
				}
			}else{
				for( int hwd = 0; hwd < filt_depth_padded * filt_hw; hwd++ ){
					unsigned pos = ((hwd& 3) + ((b&0x1f)<<2) + ( hwd>>2)*128) + bscale * (b>>5);
					uint8_t w = (outptr[pos]*scaleK + scaling_offs) >> 8;
					bsum += w;
					outptr[pos] =  w^0x80;
				}
			}
	    	if(sumb_ptr!=0) sumb_ptr[b]= bsum;
		}
    }
#endif
#if (defined(V65) || defined(V66)) && !defined(ENABLE_FULL_V65_COEFF_SCALING)
#error
#endif
    nn_sem_post( &rpfp->done_sem);
}



