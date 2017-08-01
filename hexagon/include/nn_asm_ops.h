
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
#ifndef NN_ASM_OPS_H
#define NN_ASM_OPS_H 1

/*
 */

#include <stdint.h>
#if defined(__hexagon__)
#include <hexagon_protos.h>
#endif

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
#endif

void gemacca_asm(const uint8_t * x, int N, int K, int *xsum, int y_offset);
void gemaccb_asm(const uint8_t * y, int * ysum, int K, int);
void gemmacbbw_asm(const uint8_t * x, const uint8_t * y, int * z_asm, int N, int M, int K);

void gemsuma_asm(const uint8_t * x, int N, int K, int *xsum, int y_offset, int z_offset);
void gemsumb_asm(const uint8_t * y, int * ysum, int K, int);
void gemmpybbw_asm(const uint8_t * x, const uint8_t * y, int * z_asm, int N, int M, int K);
void gemaddvvm_asm(const int * x0, const int * y0, int * z0, int N, int M, int * minmax, int reset);

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
                     int in_offset, int zsum, int * biasbuf, int relu_scale, int maxsum_shift);
void gvconv2dbbb_v66_asm(const uint8_t * x, const uint8_t * y, uint8_t * z, int in_width, int out_width, 
                     int zstride, int istride, int filt_width, int filt_height, int out_height, int * ptr_xsum,
                     int * ptr_ysum, int * ptr_max, int * biasbuf, int relu_scale);


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


static inline void l2pref(void *p, uint32_t height, uint32_t width, uint32_t stride)
{
	uint64_t control = Q6_P_combine_RR(stride,Q6_R_combine_RlRl(width,height));
#if defined(__hexagon__)
	asm volatile (" l2fetch(%0,%1) " : :"r"(p),"r"(control));
#endif
}


static inline void l2fetch(void *p, uint32_t stride, uint32_t width, uint32_t height)
{
	return l2pref(p,height,width,stride);
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

#endif
