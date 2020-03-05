
/*
 * Copyright (c) 2016-2020, The Linux Foundation. All rights reserved.
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
#ifndef NN_ASM_OPS_H
#define NN_ASM_OPS_H 1
/*
 */

#include <stdint.h>
#if defined(__hexagon__)
#include <hexagon_protos.h>
#include <hexagon_types.h>
#endif
#include <math.h>
#include "hvx_hexagon_protos.h"

void avgpool_aligned_hvx(
	uint8_t *out,
	const uint8_t *in,
	int depth,
	int window_width,
	int window_height,
	int image_width,
	int scale);
void avgpool_nonaligned_hvx(
	uint8_t *out,
	const uint8_t *in,
	int depth,
	int window_width,
	int window_height,
	int image_width,
	int scale);

void maxpool_aligned_hvx(
	uint8_t *out,
	const uint8_t *in,
	int depth,
	int window_width,
	int window_height,
	int image_width);
void maxpool_nonaligned_hvx(
	uint8_t *out,
	const uint8_t *in,
	int depth,
	int window_width,
	int window_height,
	int image_width);

void memconvert_hvx(
	uint8_t *dest,
	const uint8_t *src,
	int len,
	short offset,
	short gain,
	int stride,
	int iters);

void quantize_asm(
	const int32_t *in,
	int offset_in,
	int gain_in,
	uint8_t* out,
	int n);

#if defined(__hexagon__)
void vmemcpy_asm(void *dst, const void *src, int len);
void vmemset_asm(void *dst, int val, int len);
void vmemset_nt_asm(void *dst, int val, int len);
void vmemset_short_asm(void *dst, int val, int len);
#endif

// 2d vector memcpy with src_pitch, dst_pitch being multiples of  vector
void vmemcpy_2d_asm(
  	 unsigned wid,			// bytes wide
      unsigned ht,			//rows
      void *dst,			// destination address, any allowed
      int dst_pitch,		// row pitch of dest; must be multiple of vector
      void const *src,		// source address, any allowed
      int src_pitch);		// row pitch of source; must be multiple of vector
// 2d vector memcpy with src_pitch, dst_pitch being anything
void vmemcpy_2d_general_asm(
  	 unsigned wid,			// bytes wide
      unsigned ht,			//rows
      void *dst,			// destination address, any allowed
      int dst_pitch,		// row pitch of dest; any allowed
      void const *src,		// source address, any allowed
      int src_pitch);		// row pitch of source; any allowed

// Like the above, but with built in prefetching

void vmemcpy_2d_general_with_prefetch( 
        int width, 
        int height,
        void * dst, 
        int d_stride,
        void const * src, 
        unsigned s_stride);


// 2d vector memset: stride must be a multiple of vector, but the
// start address and width can be arbitrary.
// There are three versions, for filling bytes, int16, or int32;
// note that for 16 and 32,
//    (1) Each fill row starts with the lsb of the value, even if the address is not aligned;
//    (2) The fill width in is *bytes*, not in elements, typically this would be a multiple of
//       the element size.
// Note: height <=0, and width <= 0 are tolerated cleanly.
//
void vmemset_32_2d_asm(
      void * dst,         // location
      int val,            // 32_bit value to fill
      int width,          // width of rectangle (bytes), >0
      int height,         // height of rectangle; rows > 0
      int stride );       // stride of buffer; must be multiple of vector
static inline void __attribute__((always_inline))
vmemset_16_2d_asm(
      void * dst,         // location
      int val,            // 16_bit value to fill
      int width,          // width of rectangle (bytes), >0
      int height,         // height of rectangle; rows > 0
      int stride )        // stride of buffer; must be multiple of vector
{
	vmemset_32_2d_asm(dst, Q6_R_combine_RlRl(val,val), width, height, stride);
}
static inline void __attribute__((always_inline))
vmemset_2d_asm(
      void * dst,         // location
      int val,            // 8 bit value to fill
      int width,          // width of rectangle (bytes), >0
      int height,         // height of rectangle; rows > 0
      int stride )        // stride of buffer; must be multiple of vector
{
	vmemset_32_2d_asm(dst, Q6_R_vsplatb_R(val), width, height, stride);
}
// the 'general' version of vmemset_2d is the same but does not require an aligned row stride.
// It will split the operation into as many sub-arrays as needed, and fill these using vmemset_32_2d_asm;
// for instance if the stride  is 5+1/4 vectors, and height is 15, it will split it into four parts, each with a stride
// of 21 vectors; the first part has rows 0,4,8,12 of the original; the other 3 have (1,5,9,13)
//  (2,6,10,14) and (3,7,11). So, the more aligned the stride is, the faster it will tend to go, but it
// will always work (even if it has to do each row separately).
void vmemset_32_2d_general_asm(
      void * dst,         // location
      int val,            // 32_bit value to fill
      int width,          // width of rectangle (bytes), >0
      int height,         // height of rectangle; rows > 0
      int stride );       // stride of buffer; any value
static inline void __attribute__((always_inline))
vmemset_16_2d_general_asm(
      void * dst,         // location
      int val,            // 16_bit value to fill
      int width,          // width of rectangle (bytes), >0
      int height,         // height of rectangle; rows > 0
      int stride )        // stride of buffer; any value
{
	vmemset_32_2d_general_asm(dst, Q6_R_combine_RlRl(val,val), width, height, stride);
}
static inline void __attribute__((always_inline))
vmemset_2d_general_asm(
      void * dst,         // location
      int val,            // 8 bit value to fill
      int width,          // width of rectangle (bytes), >0
      int height,         // height of rectangle; rows > 0
      int stride )        // stride of buffer; any value
{
	vmemset_32_2d_general_asm(dst, Q6_R_vsplatb_R(val), width, height, stride);
}


void gemacca_asm(const uint8_t * x, int N, int K, int *xsum, int y_offset);
void gemaccb_asm(const uint8_t * y, int * ysum, int K, int);
void gemmacbbw_asm(const uint8_t * x, const uint8_t * y, int * z_asm, int N, int M, int K);

void gemsuma_asm(const uint8_t * x, int N, int K, int *xsum, int y_offset, int z_offset);
void gemsumb_asm(const uint8_t * y, int * ysum, int K, int);
void gemmpybbw_asm(const uint8_t * x, const uint8_t * y, int * z_asm, int N, int M, int K);
void gemaddvvm_asm(const int * x0, const int * y0, int * z0, int N, int M, int * minmax, int reset);
void gemaddvvm_asm1(const int * x0, const int * y0, int * z0, int N, int M);

void gvconv2dbbw_asm(const uint8_t * x, const uint8_t * y, int * z, int in_width, int out_width, int zstride, int istride,
                     int filt_width, int filt_height, int out_height, int * ptr_xsum, int * ptr_ysum, int * ptr_max);
void gvconvsum2dbbw_asm(const uint8_t * x, const uint8_t * y, int * z, int in_width, int out_width, int zstride, int istride,
                     int filt_width, int filt_height, int out_height, int * ptr_xsum, int * ptr_ysum, int * ptr_max,
                     int in_offset, int zsum);

void gvconv2dbbb_asm(const uint8_t * x, const uint8_t * y, uint8_t * z, int in_width, int out_width, 
                     int zstride, int istride, int filt_width, int filt_height, int out_height, int * ptr_xsum,
                     int * ptr_ysum, int * ptr_max, int * biasbuf, int relu_scale);
void gvconvsum2dbbb_asm(const uint8_t * x, const uint8_t * y, uint8_t * z, int in_width, int out_width, int zstride, int istride,
                     int filt_width, int filt_height, int out_height, int * ptr_xsum, int * ptr_ysum, int * ptr_max,
                     int in_offset, int zsum, int * biasbuf, int relu_scale);
/*void gvconv2dbbb_v66_asm(const uint8_t * x, const uint8_t * y, uint8_t * z, int in_width, int out_width, 
                     int zstride, int istride, int filt_width, int filt_height, int out_height, int * ptr_xsum,
                     int * ptr_ysum, int * ptr_max, int * biasbuf, int relu_scale);*/


void gemvmpybbw_asm(
	const uint8_t * x,
	int x_offset,
	const uint8_t * y,
	int y_offset,
	int * z,
	int MSTEP,
	int K);

void im2col33322_hvx(
	const uint8_t * in,
	uint8_t * out,
	uint8_t in_offset,
	const uint8_t *vcontrols, 
	int start_149patch,
	int num_149patches);

void im2col7732_asm (
	const uint8_t * in,
	uint8_t * out,
	uint8_t in_offset,
	const uint8_t *vcontrols,
	int start_112patch,
	int num_112patches);

void biasadd_relu_requant_hvx(
	uint8_t *out,
	const int32_t *tmp_out,
	const int32_t *biasbuf,
	const uint32_t num_patches,
	const uint32_t depth,
	const uint32_t fixed_recip_level_size); 

void biasadd_relu_requant_nonaligned_hvx(
	uint8_t *out,
	const int32_t *tmp_out,
	const int32_t *biasbuf,
	const uint32_t num_patches,
	const uint32_t depth,
	const uint32_t fixed_recip_level_size); 

/* transpose the weights matrix and shufle blocks of 32 together */
void transpack_16(
	const uint16_t *in,
	int k,
	int m,
	uint16_t * out) ;

void pad2d_16(
	const uint16_t* input_data,
	int input_height,
	int input_width,
	uint16_t* output_data,
	int output_height,
	int output_width,
	int pad_value);
void transpack(
	const uint8_t *in,
	int k,
	int m,
	uint8_t * out) ;

void pad2d(
	const uint8_t* input_data,
	int input_height,
	int input_width,
	uint8_t* output_data,
	int output_height,
	int output_width,
	int pad_value);

void im2col_co(
	uint8_t* input_data, 
	int input_height,
	int input_width,
	int input_depth,
	int input_offset, 
	uint8_t* im2col_buffer,
	int filter_height,
	int filter_width,
	int stride,
	int output_height,
	int output_width,
	int filter_left_offset,
	int filter_top_offset);

void im2col_cn(
  uint8_t* input_data, int input_height, int input_width, int input_depth, int input_offset,
  uint8_t* im2col_buffer, int filter_height, int filter_width, int stride,
  int output_height, int output_width, int filter_left_offset, int filter_top_offset);

void unpad2d(const int* input_data, int input_height, int input_width,
            int* output_data, int output_height, int output_width);

void unpad2d_bytes(const uint8_t* input_data, int input_height, int input_width,
                       uint8_t * output_data, int output_height, int output_width);


void gemm_co(const uint8_t * a, int a_offset, const uint8_t * b, int b_offset, int * c,
             int N, int M, int K, int * suma, int * sumb) ;

void gemm_asm(const uint8_t * x, int x_offset,
              const uint8_t * yopt, int y_offset,
              int * z, int N, int M, int K,
              int NSTEP, int MSTEP, int KSTEP,
              int * xsum, int * ysum, int *minmax);

void im2col_slice_co(
  const uint8_t* input_data, int input_height, int input_width, int input_depth, int input_offset,
  uint8_t* im2col_buffer, int filter_height, int filter_width, int stride,
  int output_height, int output_width, int filter_left_offset, int filter_top_offset, int patch_start, int num_patches);

void fast_im2col_co(
  const uint8_t* in_data, int in_height, int in_width, int in_depth, int in_offset,
  uint8_t* im2col_buf, int filt_height, int filt_width, int stride,
  int start_line, int num_lines, int out_width, int pad_left, int pad_top, int skip_unpad_k);

void histogram_flat_asm(uint16_t * histo, uint8_t const * data, int depth, int nbatches, int batch_stride );
void histogram_d32_asm(uint16_t * histo, uint8_t const * data, int depth, int nbatches, int d32_stride );


static inline void __attribute__((always_inline)) l2pref(const void *p, uint32_t height, uint32_t width, uint32_t stride)
{
#if defined(__hexagon__)
	uint64_t control = Q6_P_combine_RR(stride,Q6_R_combine_RlRl(width,height));
	asm volatile (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
#endif
}

static inline void __attribute__((always_inline)) l2fetch(const void *p, uint32_t stride, uint32_t width, uint32_t height)
{
	return l2pref(p,height,width,stride);
}

static inline void __attribute__((always_inline)) l2fetch_v(const void *p, uint32_t stride, uint32_t width, uint32_t height)
{
#if defined(__hexagon__)
	uint64_t control = Q6_P_combine_RR(Q6_R_combine_RlRl(1,stride),Q6_R_combine_RlRl(width,height));
	asm volatile (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
#endif
}

static inline void __attribute__((always_inline)) wait_for_l2fetch()
{
#if defined(__hexagon__)
	int32_t usr;
	do {
		asm volatile ("nop ; %0 = usr ; nop" :"=r"(usr));
	} while(usr < 0);
#endif
}

static inline int32_t fast_roundf(float f)
{
#if defined(__hexagon__)
	return Q6_R_convert_sf2w_R(f);
#else
	return roundf(f);
#endif
}

static inline int64_t fast_i64_roundf(float f)
{
#if defined(__hexagon__)
       return Q6_P_convert_sf2d_R (f);
#else
       return roundf(f);
#endif
}


void gvmsumimw_asm(
	const uint8_t * x, 
	int *xsum, 
	int o_width, 
	int skip, 
	int stride, 
	int filt_width, 
	int o_height, 
	int y_offset, 
	int z_offset);
void gvmaccimw_asm(
	const uint8_t * x, 
	int *xsum, 
	int o_width, 
	int skip, 
	int stride, 
	int filt_width, 
	int o_height, 
	int y_offset);
void gvmmpybbw_asm(
	const uint8_t * x, 
	const uint8_t * y, 
	int * z_asm, 
	int, 
	int, 
	int, 
	int, 
	int, 
	int);
void gvmmacbbw_asm(
	const uint8_t * x,
	const uint8_t * y,
	int * z_asm,
	int, 
	int, 
	int, 
	int, 
	int, 
	int);
void gvmsumb_asm(
	const uint8_t * y,
	int * ysum,
	int K,
	int offset);
void gvmaccb_asm(
	const uint8_t * y,
	int * ysum,
	int K,
	int offset);
void gvmaddvvm_asm(
	int * x0,
	int * y0,
	int * z0,
	int N,
	int M,
	int * minmax,
	int reset);

int32_t qlrn_acc_asm(
		int32_t numelem,
		const int16_t *ptr_xvec);

void qmul_asm(
		uint8_t *a,
		uint8_t *b,
		int32_t *out,
		void *info,
		int32_t elem,
		int32_t aconst,
		int32_t bconst);

void qadd_asm(
		uint8_t *a,
		uint8_t *b,
		int32_t *out,
		void *info,
		int32_t elem,
		int32_t aconst,
		int32_t bconst);

void qsub_asm(
		uint8_t *a,
		uint8_t *b,
		int32_t *out,
		void *info,
		int32_t elem,
		int32_t aconst,
		int32_t bconst);

void qmaximum_asm(
		uint8_t *a,
		uint8_t *b,
		uint8_t *out,
		void *info,
		int32_t elem,
		int32_t aconst,
		int32_t bconst);

void qminimum_asm(
		uint8_t *a,
		uint8_t *b,
		uint8_t *out,
		void *info,
		int32_t elem,
		int32_t aconst,
		int32_t bconst);

void quant_add_spec_asm(
	uint8_t *aq,
	uint8_t *bq,
	int ialpha,
	int ikappa,
	int offset,
	int recip,
	uint8_t *ptr_c,
	int16_t *ptr_max,
	int length);
/*
 * Padzap W*32 bytes at some misaligned start
 * Crossing vectors not allowed.
 * W == 0 will give you the "rest of the vector"
 */

void padzap_part(
	uint8_t *start,
	uint8_t val,
	int d32_stride,
	int d32_iters,
	int row_stride,
	int row_iters,
	int w);

void padzap16_part(
	uint16_t *start,
	uint16_t val,
	int d32_stride,
	int d32_iters,
	int row_stride,
	int row_iters,
	int w);

void gvint_asm(
	const uint8_t *in_data_d32,
	int32_t *integral_out,
	int32_t next_d32,
	int32_t next_row,
	int32_t integral_width,
	int32_t in_depth,
	int32_t out_height,
	int32_t *scratch_128xW,
	int32_t filt_offset);

void gvsuma_asm(
	const int32_t *integral,
	int32_t *suma_out,
	int32_t in_width,
	int32_t next_output_width,
	int32_t stride_height,
	int32_t filt_width,
	int32_t filt_height,
	int32_t out_height,
	int32_t offset);

void gsum_asm(const uint8_t *xi,
              int32_t *zi,
              int in_width,
              int in_depth,
              int out_height,
              int stride_v,
              int filt_offset
              );

void gvconv2dbbb_v60_asm(
	const uint8_t *input,
	const uint8_t *weights,
	uint8_t *output,
	int32_t in_width,
	int32_t out_next_row,
	int32_t out_width,
	int32_t stride_height_width,
	int32_t in_depth,
	int32_t filt_width,
	int32_t filt_height,
	int32_t num_lines,
	const int32_t *biasbuf,
	const int32_t *suma,
	int32_t next_suma,
	int32_t *minmax_buf,
	uint32_t const *recip_vals,
	int32_t zshift );

void gvconv2dbbb_v66_asm(
	const uint8_t *input,
	const int8_t *weights,
	uint8_t *output,
	int32_t in_width,
	int32_t out_next_row,
	int32_t out_width,
	int32_t stride_height_width,
	int32_t in_depth,
	int32_t filt_width,
	int32_t filt_height,
	int32_t num_lines,
	const int32_t *biasbuf,
	int32_t *minmax_buf,
	const uint32_t * recip_val,
	int32_t out_align,
	int32_t skip_col,
	int32_t out_next_d32,
	int32_t nslice,
	int32_t recip_shamt); //const int32_t *equalize);

void gvconv2dbbbs1_d16_v66_asm( //special case for depths 48,80 etc.
	const uint8_t *input,
	const int8_t *weights,
	uint8_t *output,
	int32_t in_width,
	int32_t out_next_row,
	int32_t out_width,
	int32_t stride_height_width,
	int32_t in_depth,
	int32_t filt_width,
	int32_t filt_height,
	int32_t num_lines,
	const int32_t *biasbuf,
	int32_t *minmax_buf,
	const uint32_t * recip_val,
	int32_t out_align,
	int32_t skip_col,
	int32_t out_next_d32,
	int32_t nslice,
	int32_t recip_shamt); //const int32_t *equalize);

void gvconv2dbbbs1_v66_asm(
	const uint8_t *input,
	const int8_t *weights,
	uint8_t *output,
	int32_t in_width,
	int32_t out_next_row,
	int32_t out_width,
	int32_t stride_height_width,
	int32_t in_depth,
	int32_t filt_width,
	int32_t filt_height,
	int32_t num_lines,
	const int32_t *biasbuf,
	int32_t *minmax_buf,
	const uint32_t * recip_val,
	int32_t out_align,
	int32_t skip_col,
	int32_t out_next_d32,
	int32_t nslice,
	int32_t recip_shamt); //const int32_t *equalize);

void gvconv2dbbbs1x4_v66_asm(
	const uint8_t *input,
	const int8_t *weights,
	uint8_t *output,
	int32_t in_width,
	int32_t out_next_row,
	int32_t out_width,
	int32_t stride_height_width,
	int32_t in_depth,
	int32_t filt_width,
	int32_t filt_height,
	int32_t num_lines,
	const int32_t *biasbuf,
	int32_t *minmax_buf,
	const uint32_t * recip_val,
	int32_t nc1,
	int32_t nc2, 
	int32_t out_next_d32,
	int32_t nslice,
	int32_t recip_shamt); //const int32_t *equalize);

extern const unsigned char integral_control[];

extern const unsigned char integral_control[];

void dwconv2dbbb_v60_asm(
	const uint8_t *input, 
	const int8_t *weights,
	uint8_t *output,
	int32_t next_in_width_depth, 
	int32_t next_out_width_depth,
	int32_t next_in_width_32, 
	int32_t next_out_width_32,
	int32_t indepth, 
	int32_t out_width, 
	int32_t num_out_lines, 
	int32_t filt_height,
	int32_t *ptr_max,
	int32_t recip_level, 
	const int32_t *ptr_wsum,
	int32_t stride_height,
	int32_t zshift,
	const uint8_t *ctrl);

void dwconv2dbbb_unsigned_v60_asm(
	const uint8_t *input, 
	const uint8_t *weights,
	uint8_t *output,
	int32_t next_in_width_depth, 
	int32_t next_out_width_depth,
	int32_t next_in_width_32, 
	int32_t next_out_width_32,
	int32_t indepth, 
	int32_t out_width, 
	int32_t num_out_lines, 
	int32_t filt_height,
	int32_t *ptr_max,
	int32_t recip_level, 
	const int32_t *ptr_wsum,
	int32_t stride_height,
	int32_t zshift,
	const uint8_t *ctrl,
	int32_t filt_offset);

void dwconv3x3bbb_unsigned_v60_asm(
	const uint8_t *input, 
	const uint8_t *weights,
	const int32_t *ptr_wsum,
	uint8_t *output,
	int32_t next_in_width_depth, 
	int32_t next_in_width_32, 
	int32_t indepth, 
	int32_t out_width, 
	int32_t next_out_width_depth,
	int32_t num_out_lines, 
	int32_t recip_level, 
	int32_t zshift,
	int32_t *ptr_max,
	int32_t stride_height,
	int32_t filt_offset, 
	int32_t padding );

void dwconv2dbbb_s2_v60_asm(
	const uint8_t *input, 
	const int8_t *weights,
	uint8_t *output,
	int32_t next_in_width_depth, 
	int32_t next_out_width_depth,
	int32_t next_in_width_32, 
	int32_t next_out_width_32,
	int32_t indepth, 
	int32_t out_width, 
	int32_t num_out_lines, 
	int32_t filt_height,
	int32_t *ptr_max,
	int32_t recip_level, 
	const int32_t *ptr_wsum,
	int32_t stride_height,
	int32_t zshift,
	int32_t padding_shift);

void dwconv2dbbb_unsigned_s2_v60_asm(
	const uint8_t *input, 
	const uint8_t *weights,
	uint8_t *output,
	int32_t next_in_width_depth, 
	int32_t next_out_width_depth,
	int32_t next_in_width_32, 
	int32_t next_out_width_32,
	int32_t indepth, 
	int32_t out_width, 
	int32_t num_out_lines, 
	int32_t filt_height,
	int32_t *ptr_max,
	int32_t recip_level, 
	const int32_t *ptr_wsum,
	int32_t stride_height,
	int32_t zshift,
	int32_t padding_shift,
	int32_t filt_offset);

void dwconv3x3bbb_unsigned_s2_v60_asm(
	const uint8_t *input, 
	const uint8_t *weights,
	const int32_t *ptr_wsum,
	uint8_t *output,
	int32_t next_in_width_depth, 
	int32_t next_in_width_32, 
	int32_t indepth, 
	int32_t out_width, 
	int32_t next_out_width_depth,
	int32_t num_out_lines, 
	int32_t recip_level, 
	int32_t zshift,
	int32_t *ptr_max,
	int32_t stride_height,
	int32_t filt_offset,
	int32_t padding );

void dwconv2dhhh_MxN_asm(
        const uint16_t *in_buf,
        const int16_t  *filt,
        uint16_t  *out_buf,
        int next_in_width,
        int next_out_width,
        int next_in_width_32,
        int next_out_width_32,
        int depth,
        int out_width,
        int out_height,
        int filt_width,
        int filt_height,
        const int32_t *bias_sum,
        int32_t *max,
        int32_t recip_level,
        int recip_shift,
        int stride_v_h);
typedef void (*dwconv_t)(
	const uint8_t *input, 
	const uint8_t *weights,
	const int32_t *ptr_wsum,
	uint8_t *output,
	int32_t next_in_width_depth, 
	int32_t next_in_width_32, 
	int32_t indepth, 
	int32_t out_width, 
	int32_t next_out_width_depth,
	int32_t num_out_lines, 
	int32_t recip_level, 
	int32_t zshift,
	int32_t *ptr_max,
	int32_t stride_height,
	int32_t filt_offset, 
	int32_t padding );

typedef void (*dwconv2dbbb_t)(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s1_5xN_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s2_7xN_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s1_7xN_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s2_5xN_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s1_3xN_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s1_3x3_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s2_3x3_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);

void dwconv2dbbb_s2_3xN_asm(
   const uint8_t *in_buf,
   const uint8_t  *filt,
   uint8_t  *out_buf,
   int32_t next_in_width,
   int32_t next_out_width,
   int32_t next_in_width_32,
   int32_t next_out_width_32,
   int32_t depth,
   int32_t out_width,
   int32_t out_height,
   int32_t filt_height,
   int32_t filt_zero,
   const int32_t *bias_sum,
   int32_t *max,
   const uint32_t *recip_level,
   int32_t recip_shift,
   int32_t stride_height,
   HVX_Vector * scratch_buf,
   int32_t  left_skip);


void scalemem_d32_hvx(
	uint8_t * ptr_out,
	int32_t stride_out,
	uint8_t const * ptr_in,
	int32_t stride_in,
	int32_t height,
	int32_t width,
	int32_t scl_off);

typedef void (*inconv2d_t) (
	const uint8_t * input,
	const uint8_t * weights,
	uint8_t * output,
	int in_width_pad,
	int next_out_width_row,
	int out_width,
	int indepth,
	int filt_width,
	int filt_height,
	int num_out_lines,
	int32_t * minmax_buf,
	int recip_level,
	const int32_t *biasbuf,
	const int32_t *ptr_suma,
	int next_suma,
	int stride_height_width,
	int recip_shamt);

void inconv2dbbb_s1_v60_asm(
	const uint8_t * input,
	const uint8_t * weights,
	uint8_t * output,
	int in_width_pad,
	int next_out_width_row,
	int out_width,
	int indepth,
	int filt_width,
	int filt_height,
	int num_out_lines,
	int32_t * minmax_buf,
	int recip_level,
	const int32_t *biasbuf,
	const int32_t *ptr_suma,
	int next_suma,
	int stride_height_width,
	int recip_shamt);

void inconv2dbbb_v60_asm(
	const uint8_t * input,
	const uint8_t * weights,
	uint8_t * output,
	int in_width_pad,
	int next_out_width_row,
	int out_width,
	int indepth,
	int filt_width,
	int filt_height,
	int num_out_lines,
	int32_t * minmax_buf,
	int recip_level,
	const int32_t *biasbuf,
	const int32_t *ptr_suma,
	int next_suma,
	int stride_height_width,
	int recip_shamt);

void gvconv2dbbb_circ_d32_v65_asm(
        const uint8_t * input,
        const int8_t  * weights,
        uint8_t * output,
        int in_width_pad,
        int next_out_width_row,
        int out_width,
        int stride_w_h,
        int indepth,
        int filt_width,
        int filt_height,
        int num_out_lines,
        const int32_t * ptr_wsum,
        int32_t * ptr_max,
        const uint32_t * recip_level,
        int next_out_width,
        uint8_t * circ_buffer,
        int zshift, 
        int in_offset,
        const uint8_t * store_ctrl);

void gvconv2dbbb_circ_d64_v65_asm(
        const uint8_t * input,
        const int8_t  * weights,
        uint8_t * output,
        int in_width_pad,
        int next_out_width_row,
        int out_width,
        int stride_w_h,
        int indepth,
        int filt_width,
        int filt_height,
        int num_out_lines,
        const int32_t * ptr_wsum,
        int32_t * ptr_max,
        const uint32_t * recip_level,
        int next_out_width,
        uint8_t * circ_buffer,
        int zshift, 
        int in_offset,
        const uint8_t * store_ctrl);

void gvconv2dbbb_circ6_d32_v65_asm(
        const uint8_t * input,
        const int8_t  * weights,
        uint8_t * output,
        int in_width_pad,
        int next_out_width_row,
        int out_width,
        int stride_w_h,
        int indepth,
        int filt_width,
        int filt_height,
        int num_out_lines,
        const int32_t * ptr_wsum,
        int32_t * ptr_max,
        const uint32_t * recip_level,
        int next_out_width,
        uint8_t * circ_buffer,
        int zshift,
        int in_offset,
        const uint8_t * store_ctrl);

void gvconv2dbbb_circ6_d64_v65_asm(
        const uint8_t * input,
        const int8_t  * weights,
        uint8_t * output,
        int in_width_pad,
        int next_out_width_row,
        int out_width,
        int stride_w_h,
        int indepth,
        int filt_width,
        int filt_height,
        int num_out_lines,
        const int32_t * ptr_wsum,
        int32_t * ptr_max,
        const uint32_t * recip_level,
        int next_out_width,
        uint8_t * circ_buffer,
        int zshift,
        int in_offset,
        const uint8_t * store_ctrl);

typedef void (*conv2d_t)(
        const uint8_t *,
        const int8_t  *,
        uint8_t *,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        const int32_t *,
        int32_t *,
        const uint32_t *,
        int,
        uint8_t *,
        int,
        int,
        const uint8_t *);
#if 0
typedef void (*conv2d_t)(
        const uint8_t *,
        const int8_t  *,
        uint8_t *,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        const int32_t *,
        int32_t *,
        const uint32_t *,
        int,
        uint8_t *,
        int,
        int,
        const uint8_t *);
#endif

void repstream2_asm(
        const uint8_t * input,
        uint8_t * output,
        int     width,
        int     depth,
        int     fill_height,
        int     rpad_lpad,
        int     stride_w,
        const uint8_t * circ_base,
        int buf_height,
        int in_offset,
        int num_accs);  // not used

void repstreamN_asm(
        const uint8_t * input,
        uint8_t * output,
        int     width,
        int     depth,
        int     fill_height,
        int     rpad_lpad,
        int     stride_w,
        const uint8_t * circ_base,
        int buf_height,
        int in_offset,
        int num_accs);

typedef void (*repstream_t)(
        const uint8_t * input,
        uint8_t * output,
        int     width,
        int     depth,
        int     fill_height,
        int     rpad_lpad,
        int     stride_w,
        const uint8_t * circ_base,
        int buf_height,
        int in_offset,
        int num_accs);

void ivint_asm(
        const uint8_t * in_data,
        int32_t * integral,
        int32_t next_in_width,
        int32_t out_height,
        int32_t filt_offset,
        const unsigned char *);

void copyNto4_asm(const uint8_t * out4,
        const uint8_t * inN,
        int n,
        int in_offset,
        int in_depth,
        const unsigned char *);


// this refers to:
//   extern const int16_t lut_Log2_and_Pow2[6*64]
//   extern const uint8_t const_Count64[128];
//
void lrn_d32_hvx(uint8_t const * in_ptr, int in_depth, int in_offset, int radius,
		int32_t * tmp_buf, uint8_t * lrn_ptr, int kappa, int sigma, int beta,
		short recip, int out_offset, int next_d32_row,
		int next_logical_line, int width, int height, int depth_rng);

int maxpool_slice_hvx_3x3_stride1(
	uint8_t *out,
	const uint8_t *in, 
	int32_t in_next_row,
	int32_t out_next_row,
	int32_t out_vectors_wide,
	int32_t out_lines,
	int32_t out_lalign);

int maxpool_slice_hvx_3x3_stride2(
	uint8_t *out,
	const uint8_t *in, 
	int32_t in_next_row,
	int32_t out_next_row,
	int32_t out_vectors_wide,
	int32_t out_lines,
	int32_t out_lalign);

int maxpool_slice_hvx_2x2_stride2(
	uint8_t *out,
	const uint8_t *in, 
	int32_t in_next_row,
	int32_t out_next_row,
	int32_t out_vectors_wide,
	int32_t out_lines,
	int32_t out_lalign);

void to_d32_asm(
	const uint8_t *in_data, //any ptr 
	int32_t in_width,       //any width
	uint8_t * data_d32,     //ptr to d32 data aligned 128
	int32_t next_width_d32, //desired out width 32*(pad4+(in_width+3)&~3)
	int32_t in_height,      //any height
	int32_t in_depth);      //multiple of 32

void from_d32_asm(
	const uint8_t *data_d32,//ptr to d32 data aligned 128
	int32_t next_width_d32, //desired in width 32*(pad4+(in_width+3)&~3)
	uint8_t * out_data,     //any ptr 
	int32_t in_width,       //any width
	int32_t in_height,      //any height
	int32_t in_depth);      //multiple of 32

void vmemcpy_weights_asm(void *dst, const void *src, int length);
void vmemcpy_128(void *dst, const void *src, int length);

void gvrmaxmin(int32_t *pmaxmin);

void quantize_floats_to_8b_asm(
	const float *input,
	uint8_t *output,
	int32_t elements,
	uint32_t min_offset,
	uint32_t common_exp,
	uint32_t scaling);

void find_minmax_of_floats_asm(
	float const *ptr, 
	uint32_t elements,
	float *vminmax); 

void getstats_asm(HVX_Vector *in_data,
                  int32_t width32,
                  int32_t next_width,
                  int32_t height,
                  HVX_Vector * sum,
                  HVX_Vector * var,
                  HVX_Vector * max,
                  HVX_Vector * min);

void visqrt64_asm(HVX_Vector * am,
                  HVX_Vector * ae,
                  HVX_Vector * cm,
                  HVX_Vector * ce,
                  const short * lut);

void renorm_asm(HVX_Vector * in_vec,
                int width32,
                int next_row,
                int height,
                HVX_Vector * out_vec,
                HVX_Vector * qmean,
                HVX_Vector * qrsd);

void inconv2dbbb332_v60_asm(
        const uint8_t * input,
        const uint8_t * weights,
        uint8_t * output,
        int in_width_pad,
        int next_out_width_row,
        int out_width,
        int indepth,
        int filt_width,
        int filt_height,
        int num_out_lines,
        int32_t * minmax_buf,
        int recip_level,
        const int32_t *biasbuf,
        const int32_t *ptr_suma,
        int next_suma,
        int stride_height_width,
    	int recip_shamt);

void fcsuma_asm(const uint8_t * input,
                int width,
                HVX_Vector * suma);

void fullconnlayerbatch_asm(
        const uint8_t ** ptr_in_batches,
        const uint8_t *  filt_trans,
        uint8_t       ** ptr_out_batches,
        int              in_depth,
        int              batches,
        int32_t       *  max_asm,
        int32_t        fixed_recip_level_size,  //reciprocal of max for quatnization
        const int32_t *  biasadd,
        int32_t       *  batch_sum,
        int32_t          weight_offset,
        int32_t          recip_shamt
);
void fullconnlayerbatch1_asm(
        const uint8_t *  ptr_in_batches,
        const uint8_t *  filt_trans,
        uint8_t       * ptr_out_batches,
        int              in_depth,
        int              dummy,
        int32_t       *  max_asm,
        int32_t        fixed_recip_level_size,  //reciprocal of max for quatnization
        const int32_t *  biasadd,
        int32_t       *  batch_sum,
        int32_t          weight_offset,
        int32_t          recip_shamt
);


void conv3322bbb(
	const uint8_t *input, 
	const uint8_t *filt,
	const int32_t *bias,
	uint8_t *output,
	int32_t out_width, 
	int32_t out_height,
	int32_t recip_level, 
	int32_t zshift,
	int32_t filt_offset,
	int32_t in_next_row );

void load_indata_d2(
	const uint8_t *in, 
	int32_t in_width, 
	int32_t next_row,
	int32_t left_pad,
	int32_t right_pad,
	int32_t in_offset, 
	uint8_t *out,
	int32_t remains);

// this is in shape_utils.c, but convenient to declare
// in this header...
// These funcs have the same prototypes as memset.
// Note that dst must be aligned to the element size; 'n' is in elements,
// not bytes.
void *memset_32( void * dst, int val, size_t n);
void *memset_16( void * dst, int val, size_t n);

// more convenient wrappers
static inline void
memset_int32( int32_t * ptr, int32_t val, int n ){
	memset_32(ptr,val, n);
}
static inline void
memset_uint32( uint32_t * ptr, uint32_t val, int n)
{
	memset_32(ptr, (int)val, n );
}

static inline void
memset_float( float * ptr, float val, int n){
	union {
		float as_f;
		int32_t as_i32;
	} uu  = { val };
	memset_32(ptr, uu.as_i32, n );
}
static inline void
memset_int16( int16_t * ptr, int val, int n ){
	memset_16(ptr,val, n);
}
static inline void
memset_uint16( uint16_t * ptr, int val, int n)
{
	memset_16(ptr, val, n );
}

void d32_16_to_88_cn(
	uint8_t * ine8_ptr,
	uint8_t * ino8_ptr,
	const uint16_t * in16_ptr,
	int width,
	int height,
	int depth
);

#endif // NN_ASM_OPS_H

