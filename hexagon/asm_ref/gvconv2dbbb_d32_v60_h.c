
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
 *  FUNCTIONS      : gvconv2dbbb_v60_asm
 *
 *  DESCRIPTION
 *    Perform 2d convolution using elements of size in_depth. Results are
 *    scaled and saturated to 8bits. Max and Min accumulations are kept.
 */
static inline
HVX_Vector q6op_Vw_vmpy_VwVw_s1_rnd_sat( HVX_Vector vu, HVX_Vector vv) {
	return Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift( Q6_Vw_vmpye_VwVuh( vu, vv), vu, vv );
}
// do two vrmpyaccs to a vector, using two input vecs vu1, vu0,
// and both halves of 'Rtt'.
static inline HVX_Vector
q6op_Vuw_vrmpyacc_VuwVubVubPub( HVX_Vector vx, HVX_Vector vu1, HVX_Vector vu0, int64_t Rtt ){
	HVX_Vector t = Q6_Vuw_vrmpyacc_VuwVubRub( vx,  vu0, (int32_t) Rtt );
	return Q6_Vuw_vrmpyacc_VuwVubRub( t, vu1,  (int32_t)(Rtt >> 32));
}

#define PTR_OFFSET(P,TYP,BYTES)  (TYP)( (char *)(P) + (BYTES))


void HVX_INTRINSIC_REFFUNC(gvconv2dbbb_v60_asm)(
		const uint8_t *input,
		const uint8_t *weights,
		const uint8_t *output,
		int32_t in_width,
		int32_t out_next_row,
		int32_t out_width,				// >= 1
		int32_t stride_height_width,
		int32_t in_depth,
		int32_t filt_width,
		int32_t filt_height,
		int32_t num_lines,
		const int32_t *biasbuf,
		const int32_t *suma,
		int32_t next_suma,
		int32_t *minmax_buf,
		int32_t const * recip_vals,		// points to 32 scales.
		int32_t zshift)

{
	int out_width_stride_depth = out_next_row;

	int filt_ht = filt_height * in_depth >> 5;			// must be >= 1
	int filt_wid = filt_width*4-1;				// inner loop count

	int in_width_4 = in_width *4;

	// unpack stride
	int stride_w = (uint16_t)stride_height_width;
	int stride_h = (uint32_t)stride_height_width >> 16;
	//
	// # amount (in int64's) to correct ptr_x0 after each output.
	//
	int next_outputs = filt_ht * in_width*4 - stride_w*16;

	int stride_w4 = stride_w * 4;	// unit for offsetting from ptr_x1

	int in_width_stride_depth = in_width * in_depth * stride_h;

	// get scales
	HVX_Vector recipvec = ((HVX_Vector const *)recip_vals)[0];
	// init min/max
	// (OK to init both to zero)
	HVX_Vector min_val = Q6_V_vsplat_R(0x7fffffff);
	HVX_Vector max_val = Q6_V_vsplat_R(-0x7fffffff);
	HVX_Vector wsum = *(HVX_Vector const *)biasbuf;

	HVX_Vector s0,s1,s2,s3;

	int64_t x07x04_x03x00, x17x14_x13x10, x27x24_x23x20, x37x34_x33x30;

	for( int irow = 0;  irow < num_lines ; irow++){
		HVX_Vector * ptr_z = (HVX_Vector *) ( output + irow * out_width_stride_depth );
		const uint64_t * ptr_x0 =  (const uint64_t *) ( input + irow * in_width_stride_depth );
		int32_t const * sumabuf = PTR_OFFSET( suma , int32_t const *  , irow * next_suma );

		for(int cols_remain = out_width; cols_remain > 0; cols_remain -=4 ){

			HVX_Vector const * ptr_w = (HVX_Vector const*)weights;
			HVX_Vector w0,w1;

			// Each loop generates 4 output pixels x 32 deep.
			//
			// Initialize sums, reading the  4 per-input values from 'sumabuf' according to stride_w;
			// Also each has an initialization component according to depth.
			// These compensate for weight offsets and input offset respectively.

			int32_t sum0 = sumabuf[0];	sumabuf += stride_w;
			int32_t sum1 = sumabuf[0];	sumabuf += stride_w;
			int32_t sum2 = sumabuf[0];	sumabuf += stride_w;
			int32_t sum3 = sumabuf[0];	sumabuf += stride_w;

			s0 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum0));
			s1 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum1));
			s2 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum2));
			s3 = Q6_Vw_vadd_VwVw( wsum , Q6_V_vsplat_R(sum3));

			// loop over filter height

			for(int i = 0; i < filt_ht-1; i++ ){
				const uint64_t * ptr_x1 = ptr_x0;
				ptr_x0 += in_width_4;		  // move down rows ...
				x27x24_x23x20 = ptr_x1[stride_w4*2];
				x37x34_x33x30 = ptr_x1[stride_w4*3];

				// loop over filter width (and input depth; every 4 iterations does a full
				// summation across the input depth.
				// Each iteration reads:
				//     2 weights of 32 x 4 x u8, which covers 4+4 depth slots at
				//     the current position (x 32 output depth);
				//    8 pixels from each of the four input positions corresponding to
				//     the four outputs we finding.
				//  .. and then group of 8 pixels is accumulated to the 4 sums, each
				//   using the same weights
				//
				for(int j = 0; j< filt_wid;j++){
					w0 = *ptr_w++;
					w1 = *ptr_w++;
					x17x14_x13x10 = ptr_x1[stride_w4];
					x07x04_x03x00 = ptr_x1[0];
					ptr_x1 ++;

					s0 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s0, w1, w0, x07x04_x03x00);
					s1 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s1, w1, w0, x17x14_x13x10);
					s2 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s2, w1, w0, x27x24_x23x20);
					s3 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s3, w1, w0, x37x34_x33x30);

					x27x24_x23x20 = ptr_x1[stride_w4*2];
					x37x34_x33x30 = ptr_x1[stride_w4*3];
				}
			}
			{
				// this below is a simple unpeel of last loop iteration; other than the preload
				// from ptr_x1 ( which is more intricate in the asm) it could be done
				// by 1 more iteration of the iloop
				const uint64_t * ptr_x1 = ptr_x0;
				ptr_x0 += in_width_4;
				for(int j = 0; j< filt_wid;j++){
					w0 = *ptr_w++;
					w1 = *ptr_w++;
					x17x14_x13x10 = ptr_x1[stride_w4];
					x07x04_x03x00 = ptr_x1[0];
					s0 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s0, w1, w0, x07x04_x03x00);
					s1 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s1, w1, w0, x17x14_x13x10);
					s2 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s2, w1, w0, x27x24_x23x20);
					s3 = q6op_Vuw_vrmpyacc_VuwVubVubPub( s3, w1, w0, x37x34_x33x30);
				}
			} //endloop1

			// correct for advance of ptr_x0 down rows; and move right by 4*stride_w*32 bytes
			ptr_x0 -= next_outputs;

			// find min/max (excluding right-hand columns in the padding margin)
			min_val = Q6_Vw_vmin_VwVw( min_val, s0);
			max_val = Q6_Vw_vmax_VwVw( max_val, s0);
			if(cols_remain >= 2 ){
				min_val = Q6_Vw_vmin_VwVw( min_val, s1);
				max_val = Q6_Vw_vmax_VwVw( max_val, s1);
				if(cols_remain >= 3 ){
					min_val = Q6_Vw_vmin_VwVw( min_val, s2);
					max_val = Q6_Vw_vmax_VwVw( max_val, s2);
					if(cols_remain >= 4 ){
						min_val = Q6_Vw_vmin_VwVw( min_val, s3);
						max_val = Q6_Vw_vmax_VwVw( max_val, s3);
					}
				}
			}

			// scale and reduce to 128 bytes
			HVX_Vector y0,y1,y2,y3, y0123;
			if(zshift>0){
				s0 = Q6_Vw_vasl_VwR( s0, zshift);
				s1 = Q6_Vw_vasl_VwR( s1, zshift);
				s2 = Q6_Vw_vasl_VwR( s2, zshift);
				s3 = Q6_Vw_vasl_VwR( s3, zshift);
			}


			y0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s0, recipvec);
			y1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s1, recipvec);
			y2 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s2, recipvec);
			y3 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( s3, recipvec);

			y3 = Q6_Vh_vpacke_VwVw( y3, y2);	// take upper 16 bits.
			y1 = Q6_Vh_vpacke_VwVw( y1, y0);
			y0123 = Q6_Vub_vpack_VhVh_sat( y3, y1);	// sat to u8

			/// store result
			*ptr_z ++  = y0123;

		} // for cols_remain


	}// for irow

	// scale the min/max according to scales
	min_val = Q6_Vw_vasl_VwR( min_val, zshift);
	max_val = Q6_Vw_vasl_VwR( max_val, zshift);
	min_val = q6op_Vw_vmpy_VwVw_s1_rnd_sat( min_val, recipvec);
	max_val = q6op_Vw_vmpy_VwVw_s1_rnd_sat( max_val, recipvec);
	// to accomodate -ve scales:
	HVX_Vector vmin = Q6_Vw_vmin_VwVw( min_val, max_val);
	HVX_Vector vmax = Q6_Vw_vmax_VwVw( min_val, max_val);
	// combine with previous min/max
	min_val = Q6_Vw_vmin_VwVw(vmin, ((HVX_Vector *)minmax_buf)[1]);
	max_val = Q6_Vw_vmax_VwVw(vmax, ((HVX_Vector *)minmax_buf)[0]);


	((HVX_Vector *)minmax_buf)[0] = max_val;
	((HVX_Vector *)minmax_buf)[1] = min_val;

}
