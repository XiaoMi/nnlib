/*
 * Copyright (c) 2016, 2017 The Linux Foundation. All rights reserved.
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

#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

#ifndef HVX_INTRINSIC_REFFUNC
#define HVX_INTRINSIC_REFFUNC(f) f
#endif
/*
 *  FUNCTIONS      : gvconv2dbbb_intrin
 *                                          
 *  DESCRIPTION                            
 *    Perform 2d convolution using elements of size in_depth < 32. Results are
 *    scaled and saturated to 8bits. Max and Min accumulations are kept.
 *                                       
 */


static inline
HVX_Vector q6op_Vw_vmpy_VwVw_s1_rnd_sat( HVX_Vector vu, HVX_Vector vv) {
	return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift( Q6_Vw_vmpye_VwVuh( vu, vv), vu, vv );
}
#define PTR_OFFSET(P,TYP,BYTES)  (TYP)( (char *)(P) + (BYTES))
// do two vrmpyaccs to a vector, using two input vecs vu1, vu0,
// and both halves of 'Rtt'.
static inline HVX_Vector
q6op_Vuw_vrmpyacc_VuwVubVubPub( HVX_Vector vx, HVX_Vector vu1, HVX_Vector vu0, int64_t Rtt ){
	HVX_Vector t = Q6_Vuw_vrmpyacc_VuwVubRub( vx,  vu0, (int32_t) Rtt );
	return Q6_Vuw_vrmpyacc_VuwVubRub( t, vu1,  (int32_t)(Rtt >> 32));
}
// input:  input rows,
//      row stride = in_width_pad * in_depth
//
//
// output: num_out_lines x out_width * 32 of u8 (vector aligned).
//
// minmax_buf[0..31]: store the max value (32 copies in a vector)
// minmax_buf[32..63]: store the min value (32 copies in a vector)
//
//** @@@ note: the reads of 'ptr_suma' are not compensated for vertical
// stride; so next_suma would need to be, by caller
//
void HVX_INTRINSIC_REFFUNC(inconv2dbbb_v60_asm)(
	const uint8_t * input,
	const uint8_t * weights,
	uint8_t * output,
	int in_width_pad,
	int next_out_width_row,
	int out_width,				// >=4, multiple of 4
	int in_depth,				// >=4, multiple of 4 (actually, must be 4...)
	int filt_width,
	int filt_height,			// >=1
	int num_out_lines,			// >=1
	int32_t * minmax_buf,
	int recip_level,
	const int32_t *biasbuf,
	const int32_t *ptr_suma,
	int next_suma,
	int stride_height_width)	// v stride in upper 16; h stride in lower (h stride must be even)
{
	/*printf("strides = %X; in_depth = %d; in_width= %d, out_width = %d; next_suma =%d\n",
			stride_height_width, in_depth, in_width_pad, out_width, next_suma);*/


	int in_width_depth = in_width_pad * in_depth;

	int stride_w = (uint16_t)stride_height_width;
	int in_width_stride_depth = in_width_depth * (stride_height_width>>16);
	int idepth = stride_w * in_depth >> 3;		// used to index a uint64 pointer
	int idepth3 = 3*idepth;

	int next_outputs = filt_height * in_width_depth - 16*stride_w;

	int filt_width_half = filt_width >> 1;
	int odd_wid = filt_width & 1;

	HVX_Vector recipvec = Q6_V_vsplat_R( recip_level );

	HVX_Vector min_val = Q6_V_vsplat_R( 0x7FFFFFFF);
	HVX_Vector max_val = Q6_Vw_vsub_VwVw(  Q6_V_vzero(), min_val);

	HVX_Vector wsum = *(HVX_Vector const *)biasbuf;

	HVX_Vector s0,s1,s2,s3;

	for(int irow = 0; irow < num_out_lines; irow++){
		HVX_Vector * ptr_z = (HVX_Vector *)(output + irow * next_out_width_row);
		int32_t const * active_sum = PTR_OFFSET( ptr_suma, int32_t const *, irow * next_suma);

		int64_t const * ptr_x0 = PTR_OFFSET(input, int64_t *,  irow * in_width_stride_depth);
		int64_t const * ptr_x1;
		int64_t x07x04_x03x00, x17x14_x13x10;
		int64_t x27x24_x23x20, x37x34_x33x30;


		for(int icol = 0; icol < out_width; icol+=4) {
			HVX_Vector const * ptr_w  = (HVX_Vector const *)weights;
			HVX_Vector w0,w1;
			int32_t sum0 = active_sum[0];	active_sum += stride_w;
			int32_t sum1 = active_sum[0];	active_sum += stride_w;
			int32_t sum2 = active_sum[0];	active_sum += stride_w;
			int32_t sum3 = active_sum[0];	active_sum += stride_w;

			s0 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum0));
			s1 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum1));
			s2 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum2));
			s3 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum3));

			// note, asm code does filt_height-1 iters; last is peeled off
			//
			for(int i = 0; i < filt_height; i++){
				x27x24_x23x20 = ptr_x0[idepth*2];
				x37x34_x33x30 = ptr_x0[idepth3*1];
				ptr_x1 = ptr_x0;
				ptr_x0 = PTR_OFFSET( ptr_x0, int64_t const *,  in_width_depth);

				for( int k = 0; k < filt_width_half; k++) {
						// inner loop
					w0 = *ptr_w ++;
					w1 = *ptr_w ++;
					x17x14_x13x10 = ptr_x1[idepth];
					x07x04_x03x00 = *ptr_x1 ++;

					s0 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s0, w1, w0, x07x04_x03x00);
					s1 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s1, w1, w0, x17x14_x13x10);
					s2 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s2, w1, w0, x27x24_x23x20);
					s3 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s3, w1, w0, x37x34_x33x30);

					/// This may over-read when filter wid is even! @@@
					x27x24_x23x20 = ptr_x1[idepth*2];
					x37x34_x33x30 = ptr_x1[idepth3*1];
				}
				if( odd_wid){
					w0 = *ptr_w ++;
					int32_t x13x10 = *(int32_t const*)&ptr_x1[idepth];
					int32_t x03x00 = *(int32_t const*)&ptr_x1[0];
					s0 = Q6_Vuw_vrmpyacc_VuwVubRub( s0, w0, x03x00);
					s1 = Q6_Vuw_vrmpyacc_VuwVubRub( s1, w0, x13x10);
					s2 = Q6_Vuw_vrmpyacc_VuwVubRub( s2, w0, (int32_t)x27x24_x23x20);
					s3 = Q6_Vuw_vrmpyacc_VuwVubRub( s3, w0, (int32_t)x37x34_x33x30);

				}
			} // endloop1

			ptr_x0 = PTR_OFFSET( ptr_x0, int64_t const *,  -next_outputs);

			min_val = Q6_Vw_vmin_VwVw( min_val,
					Q6_Vw_vmin_VwVw(Q6_Vw_vmin_VwVw(s0,s1),Q6_Vw_vmin_VwVw(s2,s3)));
			max_val = Q6_Vw_vmax_VwVw( max_val,
					Q6_Vw_vmax_VwVw(Q6_Vw_vmax_VwVw(s0,s1),Q6_Vw_vmax_VwVw(s2,s3)));

			HVX_Vector y0,y1,y2,y3;

			y0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s0, recipvec);
			y1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s1, recipvec);
			y2 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s2, recipvec);
			y3 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s3, recipvec);

			y3 = Q6_Vh_vpacke_VwVw( y3, y2);	// take upper 16 bits.
			y1 = Q6_Vh_vpacke_VwVw( y1, y0);
			*ptr_z ++ = Q6_Vub_vpack_VhVh_sat( y3, y1);	// sat to u8
		} // for icol
	} // for irow
	// h. reduce the min/max values
	//
	int r= 4;
	for( int i = 0; i < 5; i++){
		HVX_VectorPair shuf = Q6_W_vshuff_VVR( min_val, min_val, r);
		min_val = Q6_Vw_vmin_VwVw( Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));
		shuf =  Q6_W_vshuff_VVR( max_val, max_val, r);
		max_val = Q6_Vw_vmax_VwVw( Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));
		r <<= 1;
	}
	((HVX_Vector *)minmax_buf)[0] = max_val;
	((HVX_Vector *)minmax_buf)[1] = min_val;

}

