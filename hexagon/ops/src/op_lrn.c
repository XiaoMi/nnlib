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
#include <nn_graph.h>
#include <string.h>
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include <op_lrn.h>
//#define DEBUG_PRINT_LRN_CYCLECOUNT
#define DEBUG_PRINT_LRN_PERFORMANCE
//#define DEBUG_USE_VEXTRACT
//#define DEBUG_USE_LRN_SCALAR_INNERLOOP

/*
 * LRN:
 * * Input tensor
 * * window_shape
 * * Bias
 * * Alpha
 * * Beta
 * out = in / (bias + ((alpha/window_size) * (sum(foreach (input element determined by window_shape)**2))))**beta
 * implementation equivalent: out = in*expf[logf{scaling*(bias/scaling+sum_squared_inputs_over_win)}*-beta]
*/

static inline int fourd_index(
	int32_t b,
	int32_t y,
	int32_t x,
	int32_t z,
	int32_t batches __attribute__((unused)),
	int32_t height,
	int32_t width,
	int32_t depth)
{
	return (b * height * width * depth)
		 + (y * width * depth)
		 + (x * depth)
		 + z;
}

static inline float compute_ref_lrn_at(
	int16_t *scratch,
	const uint8_t *in,
	const float min,
	const float max,
	const int32_t b,
	const int32_t y_start,
	const int32_t x_start,
	const int32_t z_start,
	const int32_t batches,
	const int32_t height,
	const int32_t width,
	const int32_t depth,
	const struct tensor *shape_tensor,
	const float in_step,
	const float bias,
	const float scaling,
	const float beta)
{
	int32_t x, y, z;
	int32_t window_y = shape_tensor->shape.height;
	int32_t window_x = shape_tensor->shape.width;
	int32_t window_z = shape_tensor->shape.depth;
	
	int32_t window_eachside_y = (window_y-1)/2;
	int32_t window_eachside_x = (window_x-1)/2;
	int32_t window_eachside_z = (window_z-1)/2;
	
	/* calc sum-squared-elemidx in the window */
	float dqelem = 0;
	float sum = 0;
	for (y = y_start - window_eachside_y; y < y_start + window_eachside_y + 1; y++) {
	  if (y < 0) continue;
	  if (y >= height) continue;
	  for (x = x_start - window_eachside_x; x < x_start + window_eachside_x + 1; x++) {
	    if (x < 0) continue;
	    if (x >= width) continue;
	    for (z = z_start - window_eachside_z; z < z_start + window_eachside_z + 1; z++) {
	      if (z < 0) continue;
	      if (z >= depth) continue;
	      /* save floats */
	      dqelem = min + in_step * in[fourd_index(b,y,x,z,batches,height,width,depth)];
	      sum += dqelem * dqelem;
	    }
	  }
	}
	
	/* Multiply by alpha-scaling, add bias */
	/* pow by -beta... that's the same as exp(ln(x)*-beta) */
	sum *= scaling;
	sum += bias;
	sum = expf(logf(sum) * -beta);
	return sum;
}

static int lrn_8_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *shape_tensor = self->inputs[3];
	const struct tensor *bias_tensor = self->inputs[4];
	const struct tensor *alpha_tensor = self->inputs[5];
	const struct tensor *beta_tensor = self->inputs[6];
	const float bias = tensor_get_float(bias_tensor,0);
	const float alpha = tensor_get_float(alpha_tensor,0);
	const float beta = tensor_get_float(beta_tensor,0);
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	const float in_step = (in_max-in_min)/255.0f;
	float in_data;
	uint8_t *in = (uint8_t *)in_tensor->data;
	
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float out_min = 0.0f;
	float out_max = 0.0f;
	float out_data;
	float *tmpdata = (float *)nn->scratch;
	uint8_t *out = (uint8_t *)out_tensor->data;
	
	const int32_t window_size = (int32_t) tensor_get_float(shape_tensor, 0);
	const float scaling = alpha / (float)window_size;
	float lrn_multiplier;
	
	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth = in_tensor->shape.depth;
	int32_t elemcount = batches*height*width*depth;
	
	int32_t b;
	int32_t x_start;
	int32_t y_start;
	int32_t z_start;
	int32_t i;
	
	/* check parameters and report errors to skip calc */
	if (in_tensor->data_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",
			out_tensor->max_size,
			in_tensor->data_size);
	}
	if (shape_tensor->shape.batches != 1) return errlog(nn,"LRN by batches?");
	if (nn->scratch_size < elemcount*sizeof(float)) return errlog(nn,"scratch too small");
	
	/* calc LRN at each idx */
	/* LRN math (normal formula): out = in * [{scaling *(bias/scaling + sum_squared_inputs_over_win)} ** -beta] */
	/* implementation equivalent: out = in*expf[logf{scaling*(bias/scaling+sum_squared_inputs_over_win)}*-beta] */
#ifdef DEBUG_PRINT_LRN_REF_PERFORMANCE
	int start_time =  nn_os_get_cycles(nn);
#endif
	
	/* set output shape/size */
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = in_tensor->data_size;
	
	/* Use elementwise calc */
	for (b = 0; b < batches; b++) {
	  tmpdata = (float *)nn->scratch;
	  for (y_start = 0; y_start < height; y_start++) {
	    for (x_start = 0; x_start < width; x_start++) {
	      for (z_start = 0; z_start < depth; z_start++) {
			/* REMOVED: reasonable optimization to skip 0 inputs since they will be 0 outputs. */
			/* read input value non-0 */
			/* get output sum-squared-elemidx */
			/* multiplied by alpha-scaling, add bias */
			/* then powd by -beta... that's the same as exp(ln(x)*-beta) */
			/* multiply by input value */
			/* then quantize and write output value */
			in_data = in_min + in_step * *in++;
			lrn_multiplier = compute_ref_lrn_at(
							(int16_t *)nn->scratch,
							(const uint8_t *)in_tensor->data,
							in_min,
							in_max,
							b,
							y_start,
							x_start,
							z_start,
							batches,
							height,
							width,
							depth,
							shape_tensor,
							in_step,
							bias,
							scaling,
							beta);
			out_data = lrn_multiplier * in_data;
			out_max = fmaxf(out_max,out_data);
			out_min = fminf(out_min,out_data);
			*tmpdata++ = out_data;
	      }
	    }
	  }
	  tmpdata = (float *)nn->scratch;
	  for (i = 0; i < height*width*depth; i++) {
	    *out++ = quantize_uint8(*tmpdata++,out_min,out_max);
	  }
	}
	
	/* report output min/max */
	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,out_min);
	out_min_tensor->data_size = sizeof(float);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_max_tensor,0,out_max);
	out_max_tensor->data_size = sizeof(float);
#ifdef DEBUG_PRINT_LRN_REF_PERFORMANCE
	int end_time =  nn_os_get_cycles(nn);
	int elem_size = elemcount;
	printf("qlrn ref cycles = %d (elements = %d)\n", (end_time-start_time), elem_size);
#endif
	
	return 0;
}

struct tdata {
	struct nn_node *self;
	int whoami;
	nn_sem_t donesem;
	void * iptr;
	void * iextraptr;
	void * optr;
	void * oextraptr;
	void * minptr;
	void * maxptr;
	void * padfillptr;
	void * padfillextraptr;
	int32_t elemfunctusecount;
	int32_t arg01;
	int32_t arg02;
	int32_t arg03;
	int32_t arg04;
	int32_t arg05;
	int32_t arg06;
	int32_t arg07;
};

static inline void copy_thread_args(struct tdata * othread, struct tdata ithread)
{
	othread->arg01 = ithread.arg01;
	othread->arg02 = ithread.arg02;
	othread->arg03 = ithread.arg03;
	othread->arg04 = ithread.arg04;
	othread->arg05 = ithread.arg05;
	othread->arg06 = ithread.arg06;
	othread->arg07 = ithread.arg07;
} 

static inline void update_q8_min_max_params(int32_t qinzero, float step, float *pmin, float *pmax)
{
	*pmin = -(step * (float)qinzero);
	*pmax = *pmin + (step * (float)UINT8_MAX);
} 

static inline void update_log2_out_step_factor_shift(float out_step, int32_t *pqlog2_out_step, int32_t *pfactor, int32_t *pshift)
{
	const int32_t BITS_SET_ONLY_LOWER_16_BITS_OF_32 = 0x0000FFFFL;
	float log2_of_recip_out_step = log2f(1.0f / out_step);
	if (log2_of_recip_out_step > 0.0f) {
		/* do compute log2(1/outstep) when log2(1/outstep)>=0 ie (1/outstep)>=1 ie outstep<=1 ie log2(outstep)<=0 */
		/* set leftshift and multfact to use as 1/outstep needed for compensation of exp2(-(x-log2(outstep))) */
		*pqlog2_out_step = -log2_of_recip_out_step * 65536.0f/*0x1.0p16f*/;
		*pshift = (int32_t)log2_of_recip_out_step + 1;
		*pfactor = exp2f(log2_of_recip_out_step - (float)(*pshift)) * 65536.0f/*0x1.0p16f*/;
	}
	else {
		/* skip compute log2(1/outstep) when log2(1/outstep)<0 ie 1/outstep<1 ie outstep>1 ie log2(outstep)>0 */
		/* set leftshift=0 and multfact=1 so 1/outstep=1 as unneeded for compensation of exp2(-(x-log2(outstep))) */
		*pqlog2_out_step = 0;
		*pshift = 0;
		*pfactor = BITS_SET_ONLY_LOWER_16_BITS_OF_32;
	}
}

static inline int32_t compute_expansionshift(int32_t inval, int32_t inshift, int32_t inshifadj) 
{
	int32_t outshift;
	{
		int32_t tempshift;
		int32_t unusedmsbmask = (1L << (inshift+inshifadj-1));
		int32_t unusedbitmask = (1L << (inshift+inshifadj)) - 1;
		int32_t usefulbitmask = 0xFFFFFFFFL ^ unusedbitmask;
		if ((inval & usefulbitmask) != 0) {
			/* skip compute expansion shift if useful bits occupy bitpositions where useful bits expected prior to downshift */
			outshift = 0;
		}
		else {
			/* do compute expansion shift if unused bits occupy bitpositions where bits expected prior to downshift */
			outshift = (inshifadj-1);
			inval &= unusedbitmask;
			for (tempshift = 0; tempshift < (inshift+inshifadj-1); tempshift++) {
				if ((inval & unusedmsbmask) != 0) {
					break;
				}
				outshift++;
				inval <<= 1;
			}
		}
	}
	return outshift;
}

#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
static inline int32_t get_and_display_step_time_cycles(struct nn_graph *nn, int32_t prev_step_time, const char * display_string)
{
	int32_t curr_step_time = nn_os_get_cycles(nn) ;
	printf("%s cycles = %8ld\n",display_string,(curr_step_time-prev_step_time));
	return curr_step_time;
}
#endif

static int32_t compute_qlrn_sum_sq_scalar_hvx_at(
	int16_t *scratch,
	const uint8_t *in,
	const float min,
	const float max,
	const int32_t b,
	const int32_t y_start,
	const int32_t x_start,
	const int32_t z_start,
	const int32_t batches,
	const int32_t height,
	const int32_t width,
	const int32_t depth,
	const int32_t window_eachside_b,
	const int32_t window_eachside_y,
	const int32_t window_eachside_x,
	const int32_t window_eachside_z,
	const int16_t qinzero)
{
	int32_t qwdsum;
	int32_t x, y, z;
	
	/* calc elemidx going from bytes to halfwords */
	int32_t numelem = 0;
	for (y = y_start - window_eachside_y; y < y_start + window_eachside_y + 1; y++) {
		if (y < 0) continue;
		if (y >= height) continue;
		for (x = x_start - window_eachside_x; x < x_start + window_eachside_x + 1; x++) {
			if (x < 0) continue;
			if (x >= width) continue;
			for (z = z_start - window_eachside_z; z < z_start + window_eachside_z + 1; z++) {
				if (z < 0) continue;
				if (z >= depth) continue;
				/* save halfwords at word-boundaries (so save every halfword at alternate halfword-boundaries) to ease vector processing commands */
				scratch[numelem*2] = ((int16_t)in[fourd_index(b,y,x,z,batches,height,width,depth)] & 0x00FF) - qinzero;
				numelem++;
			}
		}
	}
	
	/* calc sum-squared-elemidx going from halfwords to words (min=0 for square/sum-squares so no need to add min) */
	qwdsum = qlrn_acc_asm(numelem,scratch);
	return qwdsum;
}

/* Compute A = sum of input squares along window depth dimension */
static void compute_qlrn_sum_sq_vector_hvx(struct nn_graph *nn, void *vinfo)
{
	/* parameters */
	struct tdata *info = (struct tdata *)vinfo;
	uint8_t *in = (uint8_t *)(info->iptr);
	uint8_t *in_pad = (uint8_t *)(info->iextraptr);
	int32_t *out = (int32_t *)(info->optr);
	int32_t *out_pad = (int32_t *)(info->oextraptr);
	int32_t elemidx = (int32_t)(info->elemfunctusecount);
	int32_t depth = (int32_t)(info->arg01);
	int32_t window_depth = (int32_t)(info->arg02);
	int32_t qinzero = (int32_t)(info->arg03);
	
	/* depth + window on each side and then compute the pad */
	int32_t z_start;
	int32_t window_eachside_z = (window_depth - 1) / 2;
	int32_t depth_pad = (depth+2*window_eachside_z+ALIGN_SIZE-1)&~(ALIGN_SIZE-1);
	HVX_Vector vin_sq, vin_sq_next;
	HVX_Vector vsum;
	HVX_Vector vzero = Q6_V_vsplat_R(0x0);
	HVX_Vector va_val;
	HVX_VectorPair val1, outval1, outval2;
	HVX_VectorPair outval1_lin, outval2_lin;
	int32_t a_offset;
	a_offset = (qinzero<<8) | qinzero;
	a_offset = Q6_R_combine_RlRl(a_offset, a_offset);
	HVX_Vector va_offset = Q6_V_vsplat_R(a_offset);
	
	/* loop process across input & output */
	int32_t i;
	int32_t bhw;
	for (bhw = 0; bhw < elemidx; bhw++)
	{
		/* pad buffers */
		HVX_Vector * ptr_in = (HVX_Vector *)in_pad;
		HVX_Vector * ptr_out = (HVX_Vector *)out_pad;
		
		l2fetch((void *)(in+(4*ALIGN_SIZE)), L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		
		/* zero out in_pad buffer */
		vmemset_asm(in_pad,qinzero,depth_pad+window_eachside_z);
		
		/* Copy input to aligned buffer */
		vmemcpy_asm(in_pad+window_eachside_z,in,depth);
		
		/* Compute quantized input squares and save it in scratch */
		for (z_start = 0; z_start < depth_pad; z_start+=ALIGN_SIZE) {
			// val in...
			va_val = *ptr_in++;
			
			// val1e - 0, 2, 4..., val10 - 1, 3, 5, 7...
			val1 = Q6_Wh_vsub_VubVub(va_val, va_offset);
			
			// lo(outval2) = [0, 4,  8 ... 124] => Note: each element is 4 bytes and we are in 128 Byte mode
			// hi(outval2) = [2, 6, 10 ... 126]
			outval2 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(val1), Q6_V_lo_W(val1));
			
			// lo(outval1) = [1, 5,  9 ... 125]
			// hi(outval1) = [3, 7, 11 ... 127]
			outval1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(val1), Q6_V_hi_W(val1));
			
			/* Convert vectors that are out-of-order into linear order for storing in memory */
			// Input  => lo(outval2) = [0, 4,  8 ... 124] => Note: each element is 4 bytes and we are in 128 Byte mode
			// Input  => hi(outval2) = [2, 6, 10 ... 126]
			// Output => lo(outval2_lin) = [ 0,  2,  4, ...  62]
			// Output => hi(outval2_lin) = [64, 66, 68, ... 126]
			outval2_lin = Q6_W_vshuff_VVR(Q6_V_hi_W(outval2), Q6_V_lo_W(outval2), -4);
			
			// Input  => lo(outval1) = [1, 5,  9 ... 125]
			// Input  => hi(outval1) = [3, 7, 11 ... 127]
			// Output => lo(outval1_lin) = [ 1,  3,  5, ...  63]
			// Output => hi(outval1_lin) = [65, 67, 69, ... 127]
			outval1_lin = Q6_W_vshuff_VVR(Q6_V_hi_W(outval1), Q6_V_lo_W(outval1), -4);
			
			// Input  => lo(outval2_lin) = [ 0,  2,  4, ...  62]
			// Input  => lo(outval1_lin) = [ 1,  3,  5, ...  63]
			// Output => lo(outval1) = [ 0,   1,  2, ...  31]
			// Output => hi(outval1) = [32,  33, 34, ...  63]
			outval1 = Q6_W_vshuff_VVR(Q6_V_lo_W(outval1_lin), Q6_V_lo_W(outval2_lin), -4);
			
			// Input  => hi(outval2_lin) = [64, 66, 68, ... 126]
			// Input  => hi(outval1_lin) = [65, 67, 69, ... 127]
			// Output => lo(outval2) = [64, 65, 66, ...  95]
			// Output => hi(outval2) = [96, 97, 98, ... 127]
			outval2 = Q6_W_vshuff_VVR(Q6_V_hi_W(outval1_lin), Q6_V_hi_W(outval2_lin), -4);
			
			// val out...
			*ptr_out++ = Q6_V_lo_W(outval1);
			*ptr_out++ = Q6_V_hi_W(outval1);
			*ptr_out++ = Q6_V_lo_W(outval2);
			*ptr_out++ = Q6_V_hi_W(outval2);
		}
		
		/* Compute sum of input squares */
		ptr_out = (HVX_Vector *)out_pad;
		for (z_start = 0; z_start < depth_pad; z_start+=(ALIGN_SIZE/sizeof(int32_t))) {
			vsum = vzero;
			vin_sq = *ptr_out;
			vin_sq_next = *(ptr_out+1);
			for (i = 0; i < window_depth; i++) {
				va_val = Q6_V_valign_VVR(vin_sq_next,vin_sq,sizeof(int32_t)*i);
				vsum = Q6_Vw_vadd_VwVw_sat(vsum,va_val);
			}
			*ptr_out++ = vsum;
		}
		
		/* Copy relevant output in output buffer */
		vmemcpy_asm(out,out_pad,depth*sizeof(int32_t));
		
		/* Move pointers of input & output buffer */
		out += depth;
		in += depth;
	}
	nn_sem_post(&info->donesem);
}

/* Compute B = (qbias/qscaling+A) -> 16 bits integral + 16 bits fractional */
/* - B fractional part will be used as input of log2{} -> actual used range 1f to ~2f -> 0x4000 to 0x7FFF */
/* - B integer part will be used as bitshift to output of log2{} */
/* note B here does not yet include log2(scaling) + log2(sumsq_step) */
/* Compute log2{fractionalpart(B)} via nonlinear tool -> 0x4000/1f->0x7FFF/2f format in -> Q16 format out */
/* note alternate fract(B) values in 16 bits are values for log2 need - bits31-16 log2 unneeded set so log2out becomes 0 */
/* here log2 operates on 16bit-values provided with double the desired number of elements with alignment */
/* Compute C = log2{allparts(B)*qscaling}*qbeta (note qbeta is between 0q and 255q, float beta between 0f & 1f */
/* where log2{allparts(B)*qscaling} = shift_due_to_integ(B) + log2{fract(B)} + log2(sumsq_step) + log2(qscaling) */
/* also find max{integralpart(C)} & min{fractionalpart(C)} */
static void compute_qlrn_beta_mult_log_vector_hvx(struct nn_graph *nn, void *vinfo)
{
	/* parameters */
	struct tdata *info = (struct tdata *)vinfo;
	HVX_Vector * pvsumsq  = (HVX_Vector *)(info->iptr);
	HVX_Vector * pvbasefract = (HVX_Vector *)(info->optr);
	HVX_Vector * pvbaseinteg = (HVX_Vector *)(info->oextraptr);
	HVX_Vector * pvmin = (HVX_Vector *)(info->minptr);
	HVX_Vector * pvmax = (HVX_Vector *)(info->maxptr);
	int32_t * pfendpad = (int32_t *)(info->padfillptr);
	int32_t * piendpad = (int32_t *)(info->padfillextraptr);
	int32_t elemfunctusecount = info->elemfunctusecount;
	int32_t elempadcount = info->arg01;
	int32_t num_bits_used_by_qbeta = info->arg02;
	int32_t qbetaval = info->arg03;
	int32_t qbiasdivbyscalewrd = info->arg04;
	int32_t qlogscalewrd = info->arg05;
	int32_t qlogsumsqstep = info->arg06;
	int32_t scaledrecipstep_shift = info->arg07;
	
	const int32_t LOG2INPUT_BITS_DOWNSHIFT_16 = 16;
	const int32_t LOG2INPUT_FRACTBITS_DOWNSHIFT_2 = 2;
	
	int32_t regbetaval = ((qbetaval << 8) & 0x0000FF00L) | (qbetaval & 0x000000FFL);
	regbetaval = Q6_R_combine_RlRl(regbetaval, regbetaval);
	
	HVX_Vector vtemp_biasdivbyscalewrd = Q6_V_vsplat_R(qbiasdivbyscalewrd);
	HVX_Vector vtemp_offset_for_correcting_norm = Q6_V_vsplat_R(0x00000002L);
	HVX_Vector vtemp_and_mask_to_remove_upper16bits = Q6_V_vsplat_R(0x0000FFFFL);
	HVX_Vector vtemp_offset_for_adding_norm_lost_fracbit_at_bit14 = Q6_V_vsplat_R(0x00004000L);
	HVX_Vector vtemp_or_mask_to_set_upper16bits_equal_to_1f_equivalent = Q6_V_vsplat_R(0x40000000L);
	
	HVX_Vector vtemp_log2inputbits;
	HVX_Vector vtemp_log2inintegerbits;
	HVX_Vector vtemp_log2infractbits;
	
	const int32_t LOG2OUTPUT_INTEGERBITS_DOWNSHIFT_16 = 16;
	const int32_t EXP2INPUT_INTEGERBITS_DOWNSHIFT_16 = 16;
	
	HVX_Vector vtemp_logqscaleplusqsumsq15int16frac = Q6_V_vsplat_R(qlogscalewrd+qlogsumsqstep);
	HVX_Vector vtemp_normamtmax32minusshiftduetosumsqstep = Q6_V_vsplat_R(0x00000020L-scaledrecipstep_shift);
	HVX_Vector vtemp_negativeoneinsigned32bits = Q6_V_vsplat_R(0xFFFFFFFFL);
	HVX_Vector vtemp_zeroinsigned32bits = Q6_V_vsplat_R(0x00000000L);
	
	HVX_Vector vtemp_log2outbits;
	HVX_Vector vtemp_log2outintegerbits;
	HVX_Vector vtemp_log2outmultqbetabits;
	
	HVX_Vector vtemp_exp2inbits;
	HVX_Vector vtemp_exp2infractbits;
	
	HVX_VectorPred vpred_detect_negval_exp2in;
	
	HVX_Vector vtemp_rotatedintegermax;
	HVX_Vector vtemp_rotatedfractmin;
	
	HVX_Vector vtemp_exp2infractminval = Q6_V_vsplat_R(INT32_MAX); // intentional set min to highest value pre-search-min
	HVX_Vector vtemp_exp2inintegermaxval = Q6_V_vsplat_R(INT32_MIN); // intentional set max to lowest value pre-search-max
	
	/* HVX-intrinsic compute log2in integer bits & log2in fractional bits */
	#define COMPUTE_LOG2_IN(INT_PART, FRAC_PART) \
	/* get A=qsumsq<<shift */\
	vtemp_log2inputbits = Q6_Vw_vasl_VwR(*pvsumsq++, scaledrecipstep_shift);\
	\
	/* calc (qbias/qscaling+A) */\
	vtemp_log2inputbits = Q6_Vw_vadd_VwVw(vtemp_log2inputbits, vtemp_biasdivbyscalewrd);\
	\
	/* get separate norm value (leadingzeros/cl0bits) of (qbias/qscaling+A) */\
	INT_PART = Q6_Vw_vnormamt_Vw(vtemp_log2inputbits);\
	\
	/* add required correction offset to norm value */\
	INT_PART = Q6_Vw_vadd_VwVw(INT_PART, vtemp_offset_for_correcting_norm);\
	\
	/* left-normalize log2input B=(qbias/qscaling+A) based on norm value */\
	vtemp_log2inputbits = Q6_Vw_vasl_VwVw(vtemp_log2inputbits, INT_PART);\
	\
	/* down shift post-normalized value B */\
	vtemp_log2inputbits = Q6_Vw_vasr_VwR(vtemp_log2inputbits, LOG2INPUT_BITS_DOWNSHIFT_16);\
	\
	/* remove upper 16bits to get fract(B) in lower 16bits */\
	vtemp_log2infractbits = Q6_V_vand_VV(vtemp_log2inputbits, vtemp_and_mask_to_remove_upper16bits);\
	\
	/* ensure fract(B) 16bits have rng 0x4000->0x7FFF (equivalent to 1f->2f) as needed by nonlinear tool-based code */\
	vtemp_log2infractbits = Q6_Vw_vasr_VwR(vtemp_log2infractbits, LOG2INPUT_FRACTBITS_DOWNSHIFT_2);\
	\
	/* add back norm-lost most significant 1 bit back to fract(B) 16bits at bit corresponding to 0x4000 (equivalent 1f) */\
	vtemp_log2infractbits = Q6_Vw_vadd_VwVw(vtemp_log2infractbits,vtemp_offset_for_adding_norm_lost_fracbit_at_bit14);\
	\
	/* set actually unused 31-16bits of fract(B) at 0x4000(=>1f) so when sent to tool log2in their log2out becomes 0 */\
	vtemp_log2infractbits = Q6_V_vor_VV(vtemp_log2infractbits,vtemp_or_mask_to_set_upper16bits_equal_to_1f_equivalent);\
	FRAC_PART = vtemp_log2infractbits;
	
	/* HVX-intrinsic compute exp2in integer bits & exp2in fractional bits */
	#define COMPUTE_EXP2_IN(BASE_PART, INT_PART, FRAC_PART) \
	/* calc log2outintegerpart = ((32-log2inintegerbits-recipstepshift)<<16) */\
	vtemp_log2outintegerbits = Q6_Vw_vsub_VwVw(vtemp_normamtmax32minusshiftduetosumsqstep, BASE_PART);\
	vtemp_log2outintegerbits = Q6_Vw_vasl_VwR(vtemp_log2outintegerbits, LOG2OUTPUT_INTEGERBITS_DOWNSHIFT_16);\
	\
	/* calc log2out = log2outintegerpart + log2outfractpart + log2(qscale) + log2(qsumsqstep) */\
	INT_PART = Q6_Vw_vadd_VwVw(vtemp_log2outintegerbits, FRAC_PART);\
	INT_PART = Q6_Vw_vadd_VwVw(INT_PART, vtemp_logqscaleplusqsumsq15int16frac);\
	\
	/* calc C = log2out*beta product */\
	vtemp_log2outmultqbetabits = Q6_Vw_vmpyi_VwRb(INT_PART, regbetaval);\
	\
	/* downshift C log2out*beta product which is exp2in to get back in same format as log2out (ie 16init+16frac) */\
	vtemp_exp2inbits = Q6_Vw_vasr_VwR(vtemp_log2outmultqbetabits, num_bits_used_by_qbeta);\
	\
	/* shift C log2out*beta down by 16 to get exp2in integer part (which will become rshift at exp2output) */\
	INT_PART = Q6_Vw_vasr_VwR(vtemp_exp2inbits, EXP2INPUT_INTEGERBITS_DOWNSHIFT_16);\
	\
	/* detect and correct to 0 any C exp2in erroneously rounded-down to a negative val in range -1f to 0f */\
	vpred_detect_negval_exp2in = Q6_Q_vcmp_eq_VwVw(vtemp_negativeoneinsigned32bits, INT_PART);\
	Q6_Q_vcmp_gtand_QVwVw(vpred_detect_negval_exp2in, vtemp_zeroinsigned32bits, vtemp_exp2inbits);\
	vtemp_exp2infractbits = Q6_V_vmux_QVV(vpred_detect_negval_exp2in, vtemp_zeroinsigned32bits, vtemp_exp2inbits);\
	FRAC_PART = vtemp_exp2infractbits;
	
	// loop through 32bit/word-vectors
	for (int32_t idx = 0; idx < (elemfunctusecount - NUM_ELEM_WRD_VECTOR); idx += NUM_ELEM_WRD_VECTOR) {
		// prefetch qsumsq
		l2fetch(pvsumsq+NUM_ELEM_WRD_VECTOR, L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		l2fetch(pvbasefract+NUM_ELEM_WRD_VECTOR, L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		
		// compute log2in integer bits & log2in fractional bits
		COMPUTE_LOG2_IN(vtemp_log2inintegerbits, *pvbasefract)
		
		// nonlineartool-calc of log2out fractional bits (note use only of log2in fractional bits)
		non_lin_i_log2_16((signed short *)pvbasefract, (signed short *)pvbasefract, NUM_ELEM_HFW_VECTOR); 
		
		// compute exp2in integer bits & exp2in fractional bits (note use of log2in integer bits & log2out fractional bits)
		COMPUTE_EXP2_IN(vtemp_log2inintegerbits, vtemp_log2outbits, *pvbasefract)
		
		// search and find via vmin the vector containing exp2infractminval within its elements
		vtemp_exp2infractminval = Q6_Vw_vmin_VwVw(vtemp_exp2infractminval, *pvbasefract);
		pvbasefract++;
		
		// search and find via vmax the vector containing exp2inintegermaxval within its elements
		vtemp_exp2inintegermaxval = Q6_Vw_vmax_VwVw(vtemp_exp2inintegermaxval, vtemp_log2outbits);
		*pvbaseinteg++ = vtemp_log2outbits;
	}
	
	// Last vector which can be partial - do it outside the loop
	// compute log2in integer bits & log2in fractional bits
	COMPUTE_LOG2_IN(vtemp_log2inintegerbits, *pvbasefract)
	
	// nonlineartool-calc of log2out fractional bits (note use only of log2in fractional bits)
	non_lin_i_log2_16((signed short *)pvbasefract, (signed short *)pvbasefract, NUM_ELEM_HFW_VECTOR); 
	
	// compute exp2in integer bits & exp2in fractional bits (note use of log2in integer bits & log2out fractional bits)
	COMPUTE_EXP2_IN(vtemp_log2inintegerbits, *pvbaseinteg, *pvbasefract)
	
	// ensure min/max search doesn't pick ineligible values from junk in last vector pad beyond overall elemcount
	// (as pad region may have values outside range of valid vector elements and may be incorrectly picked as min/max)
	// - method used is to fill pad region with some value so even if picked min/max would be correct
	// loop through 32bit/word-scalars
	// note scalar loop should be ok to fill 32bit values in pad region
	// loop only runs for count of (elemusecountwithalignment-elemcount)
	// which would in worst case be count of vectorsize/sizeof(int32_t)
	if (info->whoami == 1) {
		for (int32_t count = 0; count < elempadcount; count++) {
			// fill exp2in fract part padded region with extreme scalar value (max-val-fill since min-search)
			*pfendpad++ = INT32_MAX;
			// fill exp2in integer part padded region with extreme scalar value (min-val-fill since max-search)
			*piendpad++ = INT32_MIN;
		}
	}
	
	// compute max and min on Last vector after overwriting pad region with INT32_MIN/MAX
	// search and find via vmin the vector containing exp2infractminval within its elements
	vtemp_exp2infractminval = Q6_Vw_vmin_VwVw(vtemp_exp2infractminval, *pvbasefract++);
	
	// search and find via vmax the vector containing exp2inintegermaxval within its elements
	vtemp_exp2inintegermaxval = Q6_Vw_vmax_VwVw(vtemp_exp2inintegermaxval, *pvbaseinteg++);
	
	// rotate found extremity-containing 32bit/word-vectors to get max/min values into elements at idx 0
	int32_t rotation_amount = (ALIGN_SIZE >> 1); // set rotate-amount initial for vector of words
	while (rotation_amount >= sizeof(int32_t)) {
		// rotate min-containing vector through possible rotate-amounts to get min into element at idx 0
		vtemp_rotatedfractmin = Q6_V_vror_VR(vtemp_exp2infractminval, rotation_amount);
		vtemp_exp2infractminval = Q6_Vw_vmin_VwVw(vtemp_exp2infractminval, vtemp_rotatedfractmin);
		
		// rotate max-containing vector through possible rotate-amounts to get max into element at idx 0
		vtemp_rotatedintegermax = Q6_V_vror_VR(vtemp_exp2inintegermaxval, rotation_amount); 
		vtemp_exp2inintegermaxval = Q6_Vw_vmax_VwVw(vtemp_exp2inintegermaxval, vtemp_rotatedintegermax);
		
		//set rotate-amount update for vector of words
		rotation_amount >>= 1; 
	}
	
	// extract scalar maxval at element at idx 0 of search-found max-vector
	// extract scalar minval at element at idx 0 of search-found min-vector
#ifdef DEBUG_USE_VEXTRACT
	*(int32_t *)(pvmin) = Q6_R_vextract_VR(vtemp_exp2infractminval, 0); // store scalar to memory
	*(int32_t *)(pvmax) = Q6_R_vextract_VR(vtemp_exp2inintegermaxval, 0); // store scalar to memory
#else
	*pvmin = vtemp_exp2infractminval; // store vector to memory
	*pvmax = vtemp_exp2inintegermaxval; // store vector to memory 
#endif
	nn_sem_post(&info->donesem);
}

/* Compute exp2[fractionalpart(-C)] after adjustments to C based on max{integralpart(C)} & min{fractionalpart(C)} */
/* Apply to C the out_step/min/max/qinzero adjustment based on max{integralpart(C)} & min{fractionalpart(C)} */
/* - given C as original output of log2 to be given as negated input of exp2 */
/* - apply outstep to get C (updated_log2_output) = C (original_log2_output) - log2(out_step) */
/* - -C fract part will be used as input of exp2{} -> actual used range of C 0f to 1f -> 0x0000 to 0x7FFF (-C 0 to -1f) */
/* - -C integer part will be used as shift to output of exp2{}  */
/* Compute exp2[fractionalpart(-C)] via nonlinear tool -> 0x0000/0f->0x7FFF/1f format in -> Q16 format out */
/* note alternate fract(-C) values in bits15-0 are the values for which exp2 needed - bits31-16 exp2 unneeded junk */
/* here exp2 operates on 16bit-values provided with double the desired number of elements with alignment */
static void compute_qlrn_fract_exp_hvx(struct nn_graph *nn, void *vinfo)
{
	/* parameters */
	struct tdata *info = (struct tdata *)vinfo;
	HVX_Vector * pvbasefract = (HVX_Vector *)(info->optr);
	HVX_Vector * pvbaseinteg = (HVX_Vector *)(info->oextraptr);
	int32_t elemfunctusecount = info->elemfunctusecount;
	int32_t qlog2_of_out_step = info->arg01;
	
	const int32_t EXP2INPUT_FRACTBITS_DOWNSHIFT_1 = 1;
	const int32_t EXP2INPUT_INTEGERBITS_DOWNSHIFT_16 = 16;
	
	HVX_Vector vtemp_qlog2ofoutstep = Q6_V_vsplat_R(qlog2_of_out_step);
	HVX_Vector vtemp_and_mask_to_remove_upper17bits = Q6_V_vsplat_R(0x00007FFFL);
	
	HVX_Vector vtemp_exp2inputbits;
	HVX_Vector vtemp_exp2infractbits;
	
	// HVX-intrinsic renormalize exp2in integer bits & exp2in fractional bits
	#define COMPUTE_RENORM_EXP2_IN(INT_PART, FRAC_PART) \
	/* subtract to get C (updated_log2_output) from C (original_log2_output) the log2(out_step) */\
	vtemp_exp2inputbits = Q6_Vw_vsub_VwVw(FRAC_PART, vtemp_qlog2ofoutstep);\
	\
	/* get integralpart(C) in Q16.0 from bits31-16 of allparts(C) by shifting into bits15-0 */\
	INT_PART = Q6_Vw_vasr_VwR(vtemp_exp2inputbits, EXP2INPUT_INTEGERBITS_DOWNSHIFT_16);\
	\
	/* get fractionalpart(C) in Q.15 from bits16-1 of allparts(C) by shifting into bits14-0 and masking Q15 */\
	vtemp_exp2infractbits = Q6_Vw_vasr_VwR(vtemp_exp2inputbits, EXP2INPUT_FRACTBITS_DOWNSHIFT_1);\
	FRAC_PART = Q6_V_vand_VV(vtemp_exp2infractbits, vtemp_and_mask_to_remove_upper17bits);
	
	// loop through 32bit/word-vectors
	for (int32_t idx = 0; idx < elemfunctusecount; idx += NUM_ELEM_WRD_VECTOR) {
		// prefetch exp2in
		l2fetch(pvbasefract+NUM_ELEM_WRD_VECTOR, L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		
		// renormalize exp2in integer bits & exp2in fractional bits
		COMPUTE_RENORM_EXP2_IN(*pvbaseinteg, *pvbasefract)
		
		// nonlineartool-calc of exp2out fractional bits (note use only of exp2in fractional bits)
		non_lin_i_exp2_16((signed short *)pvbasefract, (signed short *)pvbasefract, NUM_ELEM_HFW_VECTOR);
		
		// postcalc ptrupdate
		pvbaseinteg++;
		pvbasefract++;
	}
	nn_sem_post(&info->donesem);
}

/* Compute D = exp2[-C] = {<exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor}<<out_step_based_shift */
/* note after applying shifts and factors the D value is limited to 8bit-range 0x0000-0x00FF but is stored in 16bits */
/* also since exp2out quantized has its qinzero at 0 (minimum) there's need to include its qinzero in any computations */
/* Compute E = (qindata-qinzero) */
/* & Compute F = D*E = exp2[log2{qscaling*(qbias/qscaling+qsum_squared_inputs_over_win)}*-qbeta]*(qindata-qinzero) */
/* also find max{F} & min{F} */
/* note the F value is stored in 32bitvecpair */
static void compute_qlrn_exp_mult_data_hvx(struct nn_graph *nn, void *vinfo)
{
	/* parameters */
	struct tdata *info = (struct tdata *)vinfo;
	HVX_Vector * pvin = (HVX_Vector *)(info->iptr);
	HVX_Vector * pvbasefract = (HVX_Vector *)(info->optr);
	HVX_Vector * pvbaseinteg = (HVX_Vector *)(info->oextraptr);
	HVX_Vector * pvmax = (HVX_Vector *)(info->maxptr);
	int32_t elemfunctusecount = info->elemfunctusecount;
	int32_t elemcount = info->arg01;
	int32_t qinzero = info->arg02;
	int32_t exp2outmult16bitfactor = info->arg03;
	int32_t exp2outleftshift = info->arg04;
	
	const int32_t MULT_16BIT_FACTOR_BIT_MASK_REMOVE_UPPER_16BITS = 0x0000FFFFL;
	const int32_t MULT_16BIT_FACTOR_BIT_SHIFT_1 = 1;
	const int32_t BIT_SHIFT_1 = 1;
	const int32_t BIT_SHIFT_16 = 16;
	const int32_t BIT_SHIFT_7 = 7;
	const int32_t BIT_MASK_KEEP_LOWER_8_OF_32 = 0x000000FFL;
	
	exp2outmult16bitfactor &= MULT_16BIT_FACTOR_BIT_MASK_REMOVE_UPPER_16BITS;
	if (exp2outmult16bitfactor != MULT_16BIT_FACTOR_BIT_MASK_REMOVE_UPPER_16BITS) {
		exp2outmult16bitfactor += ((int32_t)1 << (MULT_16BIT_FACTOR_BIT_SHIFT_1 - 1));
	}
	exp2outmult16bitfactor >>= MULT_16BIT_FACTOR_BIT_SHIFT_1;
	int32_t regexp2outmult16bitfactor = Q6_R_combine_RlRl(exp2outmult16bitfactor, exp2outmult16bitfactor);
	
	int32_t regexp2outroundoffset = ((int32_t)1 << (BIT_SHIFT_7 - 1));
	regexp2outroundoffset = Q6_R_combine_RlRl(regexp2outroundoffset, regexp2outroundoffset);
	
	int32_t qlimitexp2out = ((int32_t)UINT8_MAX & BIT_MASK_KEEP_LOWER_8_OF_32);
	int32_t regqlimitexp2out = Q6_R_combine_RlRl(qlimitexp2out, qlimitexp2out);
	
	HVX_Vector vtemp_and_mask_to_remove_upper16bits = Q6_V_vsplat_R(MULT_16BIT_FACTOR_BIT_MASK_REMOVE_UPPER_16BITS);
	HVX_Vector vtemp_exp2outmult16bitfactor = Q6_V_vsplat_R(regexp2outmult16bitfactor);
	HVX_Vector vtemp_exp2outroundoffset = Q6_V_vsplat_R(regexp2outroundoffset);
	HVX_Vector vtemp_qlimitexp2out = Q6_V_vsplat_R(regqlimitexp2out);
	
	HVX_Vector vtemp_exp2inintegervectlow;
	HVX_Vector vtemp_exp2inintegervecthigh;
	
	HVX_Vector vtemp_exp2outvectlow;
	HVX_Vector vtemp_exp2outvecthigh;
	
	HVX_Vector vtemp_exp2outvectall_low, vtemp_exp2outvectall_high;
	
	HVX_VectorPair wtemp_product_exp2out_and_16bitfactor;
	
	HVX_VectorPred vpred_detect_exceed_qlimitexp2out;
	
	int32_t qinzeroadjust = ((int32_t)qinzero & BIT_MASK_KEEP_LOWER_8_OF_32);
	int32_t regqinzeroadjust = Q6_R_combine_RlRl(qinzeroadjust, qinzeroadjust);
	
	HVX_Vector vtemp_qinzeroadjust = Q6_V_vsplat_R(regqinzeroadjust);
	
	HVX_Vector vtemp_data16bitlow;
	HVX_Vector vtemp_data16bithigh;
	
	HVX_VectorPair wtemp_qdatatrue;
	
	HVX_VectorPair wtemp_qdataexpproduct32bitlow;
	HVX_VectorPair wtemp_qdataexpproduct32bithigh;
	
	HVX_Vector vtemp_rotatedproductmaxabs;
			
	HVX_Vector vtemp_exp2outqdataproductmaxabs = Q6_V_vsplat_R(INT32_MIN);// intentional set max to lowest value pre-search-max
			
	/* HVX-intrinsic compute exp2out fractional bits with adjustment */
	#define COMPUTE_EXP2OUT_FRAC_WITH_ADJUSTMENT(IN1, IN2, IN3, OUT)\
	/* get integerpart(-C) as pair of 32bitvectors */\
	vtemp_exp2inintegervectlow = IN1++;\
	vtemp_exp2inintegervecthigh = IN1++;\
	\
	/* get exp2[fractionalpart(-C)] as pair of 32bitvectors */\
	vtemp_exp2outvectlow = IN2;\
	vtemp_exp2outvecthigh = IN3;\
	\
	/* mask exp2[fractionalpart(-C)] to get from 32bits only lower 16 bits of fract parts of exp2out tool\
	   as alternate fract(-C) vals in bits15-0 exp2in are values for which exp2 needed - bits31-16 unneeded junk */\
	vtemp_exp2outvectlow = Q6_V_vand_VV(vtemp_exp2outvectlow, vtemp_and_mask_to_remove_upper16bits);\
	vtemp_exp2outvecthigh = Q6_V_vand_VV(vtemp_exp2outvecthigh, vtemp_and_mask_to_remove_upper16bits);\
	\
	/* calc exp2[fractionalpart(-C)] >> shift_due_to_integerpart(-C) as 32bitvector pair */\
	vtemp_exp2outvectlow = Q6_Vw_vasr_VwVw(vtemp_exp2outvectlow, vtemp_exp2inintegervectlow);\
	vtemp_exp2outvecthigh = Q6_Vw_vasr_VwVw(vtemp_exp2outvecthigh, vtemp_exp2inintegervecthigh);\
	\
	/* pack exp2[fractionalpart(-C)] >> shift_due_to_integerpart(-C) from 32bitvectorpair to 16bitvector\
	  (no actual bit loss since exp2out from nonlinear tool uses only bits15-0 - bits31-16 unneeded junk) */\
	OUT = Q6_Vuh_vpack_VwVw_sat(vtemp_exp2outvecthigh, vtemp_exp2outvectlow); \
	\
	/* calc 32bitvectorpair of 16bit <exp2[fract(-C)] >> shift_due_to_integ(-C)> * out_step_based_factor_16bit */\
	wtemp_product_exp2out_and_16bitfactor = Q6_Wuw_vmpy_VuhVuh(OUT,vtemp_exp2outmult16bitfactor);\
	\
	/* pack <exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor from 32bitvectorpair to 16bitvector */\
	vtemp_exp2outvectlow = Q6_Vw_vasr_VwR(Q6_V_lo_W(wtemp_product_exp2out_and_16bitfactor), BIT_SHIFT_1);\
	vtemp_exp2outvecthigh = Q6_Vw_vasr_VwR(Q6_V_hi_W(wtemp_product_exp2out_and_16bitfactor), BIT_SHIFT_1);\
	OUT =Q6_Vuh_vasr_VwVwR_sat(vtemp_exp2outvecthigh,vtemp_exp2outvectlow,(BIT_SHIFT_16-BIT_SHIFT_1));\
	\
	/* calc in 16bitvector D = <exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor<<out_step_based_shift\
	 then place in 8bits val D <exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor<<out_step_based_shift */\
	OUT = Q6_Vh_vasl_VhR(OUT, exp2outleftshift);\
	\
	/* limit to 8bits val of D = <exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor<<out_step_based_shift\
	   note the D value limited to 8bit-range-min-max 0x0000-0x00FF but is stored in 16bit values\
	   also since exp2out quantized has its qinzero at 0 (minimum) no need to include its qzero in any computations */\
	OUT = Q6_Vuh_vadd_VuhVuh_sat(OUT, vtemp_exp2outroundoffset);\
	OUT = Q6_Vuh_vlsr_VuhR(OUT, BIT_SHIFT_7);\
	vpred_detect_exceed_qlimitexp2out = Q6_Q_vcmp_gt_VhVh(OUT, vtemp_qlimitexp2out);\
	OUT = Q6_V_vmux_QVV(vpred_detect_exceed_qlimitexp2out, vtemp_qlimitexp2out, OUT);
	
	// loop through 8bit/byte-vectors
	for (int32_t idx = 0; idx < (elemfunctusecount - NUM_ELEM_BYT_VECTOR); idx += NUM_ELEM_BYT_VECTOR) {
		// prefetch exp2in, qindata
		l2fetch(pvbaseinteg+NUM_ELEM_HFW_VECTOR, L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		l2fetch(pvbasefract+NUM_ELEM_HFW_VECTOR, L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		l2fetch(pvin+NUM_ELEM_BYT_VECTOR, L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		
		// calc 8bit D = <exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor<<out_step_based_shift
		COMPUTE_EXP2OUT_FRAC_WITH_ADJUSTMENT(*pvbaseinteg, *pvbasefract, *(pvbasefract+1), vtemp_exp2outvectall_low)
		COMPUTE_EXP2OUT_FRAC_WITH_ADJUSTMENT(*pvbaseinteg, *(pvbasefract+2), *(pvbasefract+3), vtemp_exp2outvectall_high)
		
		// put qindata in 16bitvectorpair but still in 8bitformat
		wtemp_qdatatrue = Q6_Wuh_vunpack_Vub(*pvin++);
		
		// calc E = (qindata-qinzero) in 16bitvectorpair with 8bit qinzero upcasted to 16bitvectorsolo
		//   note E value is limited to range 16bit-range-min-max -255 to +255
		vtemp_data16bitlow = Q6_Vh_vsub_VhVh(Q6_V_lo_W(wtemp_qdatatrue), vtemp_qinzeroadjust);
		vtemp_data16bithigh = Q6_Vh_vsub_VhVh(Q6_V_hi_W(wtemp_qdatatrue), vtemp_qinzeroadjust);
		
		// calc F = product of 16bitvecs (qindata-qinzero)&exp2[log2{qscaling*(qbias/qscaling+qsumsq)}*-qbeta]
		//   store the calculated F = D*E in 32bitvecpair
		wtemp_qdataexpproduct32bitlow = Q6_Ww_vmpy_VhVuh(vtemp_data16bitlow, vtemp_exp2outvectall_low);
		wtemp_qdataexpproduct32bithigh = Q6_Ww_vmpy_VhVuh(vtemp_data16bithigh, vtemp_exp2outvectall_high);
		*pvbasefract = Q6_V_lo_W(wtemp_qdataexpproduct32bitlow);
		*(pvbasefract+1) = Q6_V_hi_W(wtemp_qdataexpproduct32bitlow);
		*(pvbasefract+2) = Q6_V_lo_W(wtemp_qdataexpproduct32bithigh);
		*(pvbasefract+3) = Q6_V_hi_W(wtemp_qdataexpproduct32bithigh);
		
		// Compute max on vector
		vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
		vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
		vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
		vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
	}
	
	// Last vector which can be partial - do it outside the loop
	// calc 8bit D = <exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor<<out_step_based_shift
	COMPUTE_EXP2OUT_FRAC_WITH_ADJUSTMENT(*pvbaseinteg, *pvbasefract, *(pvbasefract+1), vtemp_exp2outvectall_low)
	COMPUTE_EXP2OUT_FRAC_WITH_ADJUSTMENT(*pvbaseinteg, *(pvbasefract+2), *(pvbasefract+3), vtemp_exp2outvectall_high)
	
	// put qindata in 16bitvectorpair but still in 8bitformat
	wtemp_qdatatrue = Q6_Wuh_vunpack_Vub(*pvin++);
	
	// calc E = (qindata-qinzero) in 16bitvectorpair with 8bit qinzero upcasted to 16bitvectorsolo
	//   note E value is limited to range 16bit-range-min-max -255 to +255
	vtemp_data16bitlow = Q6_Vh_vsub_VhVh(Q6_V_lo_W(wtemp_qdatatrue), vtemp_qinzeroadjust);
	vtemp_data16bithigh = Q6_Vh_vsub_VhVh(Q6_V_hi_W(wtemp_qdatatrue), vtemp_qinzeroadjust);
	
	// calc F = product of 16bitvecs (qindata-qinzero)&exp2[log2{qscaling*(qbias/qscaling+qsumsq)}*-qbeta]
	//   store the calculated F = D*E in 32bitvecpair
	wtemp_qdataexpproduct32bitlow = Q6_Ww_vmpy_VhVuh(vtemp_data16bitlow, vtemp_exp2outvectall_low);
	wtemp_qdataexpproduct32bithigh = Q6_Ww_vmpy_VhVuh(vtemp_data16bithigh, vtemp_exp2outvectall_high);
	*pvbasefract = Q6_V_lo_W(wtemp_qdataexpproduct32bitlow);
	*(pvbasefract+1) = Q6_V_hi_W(wtemp_qdataexpproduct32bitlow);
	*(pvbasefract+2) = Q6_V_lo_W(wtemp_qdataexpproduct32bithigh);
	*(pvbasefract+3) = Q6_V_hi_W(wtemp_qdataexpproduct32bithigh);
	
	// ensure min/max search doesn't pick ineligible values from junk in last vector pad beyond overall elemcount
	// (as pad region may have values outside range of valid vector elements and may be incorrectly picked as min/max)
	// - method used is to fill pad region with a valid vector element so even if picked min/max would be correct
	// loop through 32bit/word-scalars
	// note scalar loop should be ok to fill 32bit values in pad region
	// loop only runs for count of (elemusecountwithalignment-elemcount)
	// which would in worst case be count of vectorsize/sizeof(int32_t)
	if (info->whoami == 1) {
		const int32_t BIT_SHIFT_FOR_NUM_ELEM_WRD_VECT_PAIR = (int32_t)log2f(2*NUM_ELEM_WRD_VECTOR);
		int32_t startelemindexvectpair = ((elemcount >> BIT_SHIFT_FOR_NUM_ELEM_WRD_VECT_PAIR) * NUM_ELEM_WRD_VECTOR * 2);
		int32_t endelemindexvectpair = (startelemindexvectpair + (NUM_ELEM_WRD_VECTOR * 2) - 1);
		int32_t oddcountofst = (elemcount & 0x1);
		int32_t firstjunkelemidxvectlow = startelemindexvectpair + ((elemcount + oddcountofst - startelemindexvectpair) >> 1);
		int32_t lastelemindexvectlow = endelemindexvectpair - NUM_ELEM_WRD_VECTOR;
		int32_t firstjunkelemidxvecthigh = firstjunkelemidxvectlow + (NUM_ELEM_WRD_VECTOR - oddcountofst);
		int32_t lastelemindexvecthigh = endelemindexvectpair;
		int32_t * tempwdptr = (int32_t *)(info->optr);
		for (int32_t idx = firstjunkelemidxvectlow; idx < (lastelemindexvectlow + 1); idx++) {
			// fill used vector pair low part padded/junk region with valid scalar value from a vector element
			tempwdptr[idx] = INT32_MIN;
		}
		for (int32_t idx = firstjunkelemidxvecthigh; idx < (lastelemindexvecthigh + 1); idx++) {
			// fill used vector pair high part padded/junk region with valid scalar value from a vector element
			tempwdptr[idx] = INT32_MIN;
		}
	}
	
	// Compute max on Last vector after overwriting pad region with INT32_MIN
	vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
	vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
	vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
	vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, Q6_Vh_vabs_Vh(*pvbasefract++));
	
	// rotate found extremity-containing 32bit/word-vectors to get max values into elements at idx 0
	int32_t rotation_amount = (ALIGN_SIZE >> 1); // set rotate-amount initial for vector of words
	while (rotation_amount >= sizeof(int32_t)) {
		// rotate max-containing vector through possible rotate-amounts to get max into element at idx 0
		vtemp_rotatedproductmaxabs = Q6_V_vror_VR(vtemp_exp2outqdataproductmaxabs, rotation_amount); 
		vtemp_exp2outqdataproductmaxabs = Q6_Vw_vmax_VwVw(vtemp_exp2outqdataproductmaxabs, vtemp_rotatedproductmaxabs);
		
		//set rotate-amount update for vector of words
		rotation_amount >>= 1; 
	}
	
	// extract scalar maxval at element at idx 0 of search-found max-vector
#ifdef DEBUG_USE_VEXTRACT
	*(int32_t *)(pvmax) = Q6_R_vextract_VR(vtemp_exp2outqdataproductmaxabs, 0); // store scalar to memory 
#else
	*pvmax = vtemp_exp2outqdataproductmaxabs; // store vector to memory 
#endif
	nn_sem_post(&info->donesem);
}

/* Compute G = qoutzero+F after applying to G and F the out_step adjustments based on max{F} & min{F} */
/* Compute 8bits output = G which only uses 8bits in 16bits */
/* note at this point output is already changed from 16bits to 8bits because of quantized zero addition */
/* so just need to pack the output to 8bit vectors from 16bit vectors (no bit loss) */
static void compute_qlrn_qzero_based_data_hvx(struct nn_graph *nn, void *vinfo)
{
	/* parameters */
	struct tdata *info = (struct tdata *)vinfo;
	HVX_Vector * pvout = (HVX_Vector *)(info->optr);
	HVX_Vector * pvbasefract = (HVX_Vector *)(info->oextraptr);
	int32_t elemfunctusecount = info->elemfunctusecount;
	int32_t out_qzeero = info->arg01;
	int32_t finalstagemultfactor = info->arg02;
	int32_t finalstagedwnshift = info->arg03;
	int32_t finalstageupshift = info->arg04;
	
	const int32_t BIT_MASK_KEEP_LOWER_8_OF_32 = 0x000000FFL;
	
	int32_t regfinalstagemultfactor = ((finalstagemultfactor << 8)&0x0000FF00L) | (finalstagemultfactor & 0x000000FFL);
	regfinalstagemultfactor = Q6_R_combine_RlRl(regfinalstagemultfactor, regfinalstagemultfactor);
	
	int32_t regfinalstageoffset = (((int32_t)out_qzeero & BIT_MASK_KEEP_LOWER_8_OF_32) << finalstagedwnshift);
	if (finalstagedwnshift > 0) {
		regfinalstageoffset += ((int32_t)1 << (finalstagedwnshift-1));
	}
	
	HVX_Vector vtemp_finalstageoffset = Q6_V_vsplat_R(regfinalstageoffset);
	
	HVX_Vector vtemp_finalstagevectlow;
	HVX_Vector vtemp_finalstagevecthigh;
	HVX_Vector vtemp_finalstgvectall;
	
	HVX_Vector vtempoutvectlow;
	HVX_Vector vtempoutvecthigh;
	
	// HVX-intrinsic compute qoutzero+((qindata-qinzero)*exp2(...))
	#define COMPUTE_SUM_QOUTZERO_AND_DATA_TIMES_EXP2_OP(OUT) \
	/* get 32bitvectorpair F = (qindata-qinzero)*exp2() and apply out_step factor adjustment based on max{F} & min{F} */\
	vtemp_finalstagevectlow = Q6_Vw_vmpyi_VwRb(*pvbasefract++, regfinalstagemultfactor);\
	vtemp_finalstagevecthigh = Q6_Vw_vmpyi_VwRb(*pvbasefract++, regfinalstagemultfactor);\
	\
	/* calc 32bitvecpair G = F+qoutzero+round (qoutzero&round preshifted to compensate for out_step shift adjust)*/ \
	vtemp_finalstagevectlow = Q6_Vw_vadd_VwVw(vtemp_finalstagevectlow, vtemp_finalstageoffset); \
	vtemp_finalstagevecthigh = Q6_Vw_vadd_VwVw(vtemp_finalstagevecthigh, vtemp_finalstageoffset); \
	\
	/* apply to G the out_step shift adjustment based on max{F} & min{F} \
	and store G converted from 32bitvectorpair to 16bitvector \
	note at this point output is already changed from 16bits to 8bits because quantized zero was added */\
	vtemp_finalstgvectall = Q6_Vh_vasr_VwVwR_sat(vtemp_finalstagevecthigh,vtemp_finalstagevectlow,finalstagedwnshift); \
	OUT = Q6_Vh_vasl_VhR(vtemp_finalstgvectall, finalstageupshift);
	
	// loop through 8bit/byte-vectors
	for (int32_t idx = 0; idx < elemfunctusecount; idx += NUM_ELEM_BYT_VECTOR) {
		// prefetch exp2out*(qindata-qinzero)
		l2fetch(pvbasefract+4*NUM_ELEM_HFW_VECTOR, L2_FETCH_STRIDE_LEN, L2_FETCH_BLOCK_SIZE, L2_FETCH_NUM_LINES);
		
		// get 16bitvectorpair G =((qindata-qinzero)*exp)+qoutzero where G actually only uses byte format
		COMPUTE_SUM_QOUTZERO_AND_DATA_TIMES_EXP2_OP(vtempoutvectlow)
		COMPUTE_SUM_QOUTZERO_AND_DATA_TIMES_EXP2_OP(vtempoutvecthigh)
		
		// store after packing in byte format output = G value ((qdata-qinzero)*exp)+qinzero)
		// note no bit loss as G actually only uses byte format because of quantized zero addition
		*pvout++ = Q6_Vub_vpack_VhVh_sat(vtempoutvecthigh, vtempoutvectlow); 
	}
	nn_sem_post(&info->donesem);
}
static int lrn_8_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *shape_tensor = self->inputs[3];
	const struct tensor *bias_tensor = self->inputs[4];
	const struct tensor *alpha_tensor = self->inputs[5];
	const struct tensor *beta_tensor = self->inputs[6];
	const float bias = tensor_get_float(bias_tensor,0);
	const float alpha = tensor_get_float(alpha_tensor,0);
	const float beta = tensor_get_float(beta_tensor,0);
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float in_step = (in_max - in_min) / (float)UINT8_MAX;
	
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float out_max;
	float out_min;
	float out_step;
	int32_t qlog2_of_out_step;
	uint8_t out_qzeero;
	
	const int32_t window_size = (int32_t) tensor_get_float(shape_tensor, 0);
	const float scaling = alpha / (float)window_size;
	
	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth = in_tensor->shape.depth;
	int32_t elemcount = batches*height*width*depth;
	
	struct tdata worker_info = {
			self,
			0,
	};
	struct tdata my_info = {
			self,
			1,
	};

	/* fallback to ref for certain parameters */
	if (beta >= 1.0f) {
		return lrn_8_execute_ref(self,nn);
	}
	
	/* check parameters and report errors to skip calc */
	if (in_tensor->data_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",
			out_tensor->max_size,
			in_tensor->data_size);
	}
	if (shape_tensor->shape.batches != 1) return errlog(nn,"LRN by batches?");
	if (nn->scratch_size < 5*elemcount*sizeof(float)) return errlog(nn,"scratch too small");
	
	/* calc LRN at all idx */
	/* LRN math (normal formula): out = in * [{scaling *(bias/scaling + sum_squared_inputs_over_win)} ** -beta] */
	/* impl: qoutdata = qoutzero+(qindata-qinzero)*exp2[log2{qscaling*(qbias/qscaling+qsum_squared_inputs_over_win)}*-qbeta] */
	/* => stepA: calc A = qsum_squared_inputs_over_win */
	/* => stepB: calc B = (qbiasbyscaling+A) */
	/* => stepC: calc C = log2{B*qscaling}*qbeta */
	/* => stepD: calc D = exp2[-C] */
	/* => stepE: calc E = (qindata-qinzero) */
	/* => stepF: calc F = D*E */
	/* => stepG: calc out/G = qoutzero+F */
#ifdef DEBUG_PRINT_LRN_PERFORMANCE
	int start_time = nn_os_get_cycles(nn);
#endif
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	int32_t step_time = nn_os_get_cycles(nn);
#endif
	
	/* set output shape/size */
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = in_tensor->data_size;
	
	/* set window shape/size */
	int32_t window_batches = shape_tensor->shape.batches;
	int32_t window_width = shape_tensor->shape.width;
	int32_t window_height = shape_tensor->shape.height;
	int32_t window_depth = shape_tensor->shape.depth;
	
	/* set quantized scaling value */
	uint8_t qinzero = quantize_uint8(0.0f,in_min,in_max);
	int32_t qlogscalewrd = log2f(scaling) * 65536.0f/*0x1.0p16f*/;
	
	/* set quantized bias-div-by-scaling values */
	const int32_t SCALED_RECIP_STEP_EXTRA_COMP_SHIFT = 4;
	float sumsq_step = in_step * in_step; // convert sum-squared-elem step from context of byte to hfwd (where bytes squared)
	int32_t scaledrecipstep_shift;
	if (sumsq_step <= 1.0f) {
		/* no need to compensate for bitloss in quantized(1/stepsize) with stepsize<=1 */
		scaledrecipstep_shift = 0;
	}
	else {
		/* do extra compensate for bitloss in quantized(1/stepsize) with stepsize>1 */
		scaledrecipstep_shift = (int32_t)(log2f(sumsq_step)) + (int32_t)SCALED_RECIP_STEP_EXTRA_COMP_SHIFT;
	}
	int16_t scaledrecipsumsq_step = (int16_t)((float)(0x1 << scaledrecipstep_shift)  / sumsq_step);
	int32_t qlogsumsqstep = log2f(sumsq_step) * 65536.0f/*0x1.0p16f*/;
	int32_t qbiasdivbyscalewrd = (int32_t)((bias * (float)scaledrecipsumsq_step) / scaling);
	
	/* set quantized beta value */
	const int32_t BIT_MASK_KEEP_LOWER_8_OF_32 = 0x000000FFL;
	const int32_t NUM_BITS_USED_BY_QBETA = 7; // Note Quantized beta cannot exceed 8bits as register use combines 4x to 32bits
	int32_t qbetaval = ((int32_t)(beta * (float)((int32_t)1 << NUM_BITS_USED_BY_QBETA) / 1.0f) & BIT_MASK_KEEP_LOWER_8_OF_32);
	
	/* set quantized-exp2-used values */
	int32_t exp2inintegermaxval = INT32_MIN;
	int32_t exp2infractminval = INT32_MAX;
	int32_t exp2outmult16bitfactor = 0xFFFFL;
	int32_t exp2outleftshift = 0;
	int32_t exp2outqdataproductmaxval = INT32_MIN;
	
	/* set final-stage-used values */
	const int32_t FINAL_STG_MULT_FACTOR_USE_NUM_BITS = 7;
	int32_t finalstagemultfactor = 0x7FL;
	int32_t finalstagedwnshift = 0;
	int32_t finalstageupshift = 0;
	
	/* allocate element-usage size for vector operations */
	int32_t elemusecountwithalignment = (elemcount + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
	int32_t elem32bitusesizewithpad = ((elemcount*sizeof(int32_t)) + MAXPAD - 1) & ~(MAXPAD - 1);
	
	/* allocate pointers for scratch as needed */
	/* note: nn->scratch initial pad reserved for sqlempad use at scratchpad space prior to other pads */
	int32_t * const integpartpadptr = (int32_t *)pad_and_align(nn->scratch, elem32bitusesizewithpad);
	int32_t * const sumsqpadptr = (int32_t *)pad_and_align(integpartpadptr, elem32bitusesizewithpad);
	int32_t * const fractpartpadptr = (int32_t *)pad_and_align(sumsqpadptr, elem32bitusesizewithpad);
	HVX_Vector * const vecpadptr = (HVX_Vector *)pad_and_align(fractpartpadptr,elem32bitusesizewithpad);
	HVX_Vector * const vecpadptr2 = vecpadptr + 1;
	HVX_Vector * const vecextrapadptr = vecpadptr2 + 1;
	HVX_Vector * const vecextrapadptr2 = vecextrapadptr + 1;
	int32_t use_multi_thread = ((elemusecountwithalignment <= ALIGN_SIZE) ? 0 : 1);
	int32_t idx;
	
	/* step1 -> calc A = qsum(squared_inputs_over_win) in 32-bit => sumsq_step: (in_step)^2 */
#ifdef DEBUG_USE_LRN_SCALAR_INNERLOOP
	int32_t force_use_lrn_scalar_innerloop = 1;
#else
	int32_t force_use_lrn_scalar_innerloop = 0;
#endif
	uint8_t * iptr = (uint8_t *)in_tensor->data;
	if ((window_batches != 1) || (window_width != 1) || (window_height != 1) || (force_use_lrn_scalar_innerloop == 1)) {
		/* Use non-optimal (scalar version) sum of in squares if window has more than depth dimension */
		/* No Intermediate buffers used */
		int32_t b;
		int32_t y_start;
		int32_t x_start;
		int32_t z_start;
		int32_t window_eachside_b = (window_batches - 1) / 2;
		int32_t window_eachside_y = (window_height - 1) / 2;
		int32_t window_eachside_x = (window_width - 1) / 2;
		int32_t window_eachside_z = (window_depth - 1) / 2;
		/* Compute sum of input squares along all dimensions */
		idx = 0;
		for (b = 0; b < batches; b++) {
			for (y_start = 0; y_start < height; y_start++) {
				for (x_start = 0; x_start < width; x_start++) {
					for (z_start = 0; z_start < depth; z_start++) {
						sumsqpadptr[idx++] = compute_qlrn_sum_sq_scalar_hvx_at(
									(int16_t *)nn->scratch,
									iptr,
									in_min,
									in_max,
									b,
									y_start,
									x_start,
									z_start,
									batches,
									height,
									width,
									depth,
									window_eachside_b,
									window_eachside_y,
									window_eachside_x,
									window_eachside_z,
									qinzero);
					}
				}
			}
		}
	}
	else {
		/* Use optimized sum of in squares only if window has depth dimension */
		/* Intermediate buffers - this scratch space is later reused */
		int32_t bhw = batches * height * width;
		int32_t window_eachside_z = (window_depth - 1) / 2;
		int32_t depth_pad = (depth + 2*window_eachside_z + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1);
		int32_t * const sumsqextrapadptr = (int32_t * const)pad_and_align(sumsqpadptr, elem32bitusesizewithpad);
		int32_t * const sumsqextrapadptr2 = (int32_t * const)pad_and_align(sumsqextrapadptr, depth_pad*sizeof(int32_t));
		uint8_t * const iextraptr = (uint8_t *)pad_and_align(sumsqextrapadptr2, depth_pad*sizeof(int32_t));
		uint8_t * const iextraptr2 = (uint8_t *)pad_and_align(iextraptr, (depth_pad+window_eachside_z)*sizeof(uint8_t));
		/* Compute sum of input squares along window depth dimension */
		if (use_multi_thread == 0) {
			my_info.iptr = iptr;
			my_info.iextraptr = iextraptr;
			my_info.optr = sumsqpadptr;
			my_info.oextraptr = sumsqextrapadptr;
			my_info.elemfunctusecount = bhw;
			my_info.arg01 = depth;
			my_info.arg02 = window_depth;
			my_info.arg03 = qinzero;
			
			nn_sem_init(&my_info.donesem,0);
			compute_qlrn_sum_sq_vector_hvx(nn, &my_info);
		}
		else {
			worker_info.iptr = iptr;
			worker_info.iextraptr = iextraptr;
			worker_info.optr = sumsqpadptr;
			worker_info.oextraptr = sumsqextrapadptr;
			worker_info.elemfunctusecount = (bhw / 2);
			worker_info.arg01 = depth;
			worker_info.arg02 = window_depth;
			worker_info.arg03 = qinzero;
			
			my_info.iptr = &iptr[worker_info.elemfunctusecount*worker_info.arg01];
			my_info.iextraptr = iextraptr2;
			my_info.optr = &sumsqpadptr[worker_info.elemfunctusecount*worker_info.arg01];
			my_info.oextraptr = sumsqextrapadptr2;
			my_info.elemfunctusecount = (bhw - worker_info.elemfunctusecount);
			copy_thread_args(&my_info, worker_info);
			
			nn_sem_init(&worker_info.donesem,0);
			nn_os_work_for_vector(nn, compute_qlrn_sum_sq_vector_hvx, &worker_info);
			nn_sem_init(&my_info.donesem,0);
			compute_qlrn_sum_sq_vector_hvx(nn, &my_info);
			nn_sem_wait(&worker_info.donesem);
		}
	}
	
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	step_time = get_and_display_step_time_cycles(nn, step_time, "qlrn step1  - calc A = sum_sq_inp_over_wind       :");
#endif
	
	/* step2 -> calc B = (qbias/qscaling+A) -> 16 bits integral + 16 bits fractional */
	/* - B fractional part will be used as input of log2{} -> actual used range 1f to ~2f -> 0x4000 to 0x7FFF */
	/* - B integer part will be used as bitshift to output of log2{} */
	/* note B here does not yet include log2(scaling) + log2(sumsq_step) */
	/* step3 -> calc log2{fractionalpart(B)} via nonlinear tool -> 0x4000/1f->0x7FFF/2f format in -> Q16 format out */
	/* note alternate fract(B) values in 16 bits are values for log2 need - bits31-16 log2 unneeded set so log2out becomes 0 */
	/* here log2 operates on 16bit-values provided with double the desired number of elements with alignment */
	/* step4 -> calc C = log2{allparts(B)*qscaling}*qbeta (note qbeta is between 0q and 255q, float beta between 0f & 1f */
    /* where log2{allparts(B)*qscaling} = shift_due_to_integ(B) + log2{fract(B)} + log2(sumsq_step) + log2(qscaling) */
	/* also find max{integralpart(C)} & min{fractionalpart(C)} */
	// Pseudocode:
	// int scaled_recip_in_stepshift;
	// if (in_sumsq_stepsize <= 1.0f) scaled_recip_in_stepshift = 0;
	// else scaled_recip_in_stepshift = (int)(log2f(in_sumsq_stepsize)) + (int)4;
	// int scaled_recip_in_stepsize = (int)((float)(0x1 << scaled_recip_in_stepshift)  / (float)in_sumsq_stepsize);
	// int isum = (int)((int)(sum) << scaled_recip_in_stepshift) + (int)((bias * (float)scaled_recip_in_stepsize) / scaling);
	// unsigned int leading_zeros = __builtin_clz(sum);
	// unsigned int shamt = leading_zeros + 1;
	// int normsum = ((isum << shamt) >> 16) & 0x0000FFFF;
	// int logfrac = frac_log_approx(normsum);
	// int ilogscal_q1516 = log2f(scaling)*0x1.0p16f;
	// int iloginstep_q1516 = log2f(in_sumsq_stepsize)*0x1.0p16f;
	// int ilogval_q1516 = logfrac + ((32-shamt) << 16) + ilogscal_q1516 + iloginstep_q1516;
	// float log_recip_out_stepsize = log2f(1.0f / out_stepsize);
	// int ilogoutstep_q1516 = -log_recip_out_stepsize * 0x1.0p16f;
	// int beta_frac = beta*0x1p7f;
	// long long int adjusted_ll = (long long int)(ilogval_q1516) * beta_frac;
	// if (adjusted_ll < 0) adjusted_ll = 0;
	// int adjusted = (adjusted_ll >> 7) - ilogoutstep_q1516;
	if (use_multi_thread == 0) {
		my_info.iptr = sumsqpadptr;
		my_info.optr = fractpartpadptr;
		my_info.oextraptr = integpartpadptr;
		my_info.minptr = vecpadptr;
		my_info.maxptr = vecextrapadptr;
		my_info.padfillptr = &fractpartpadptr[elemcount];
		my_info.padfillextraptr = &integpartpadptr[elemcount];
		my_info.elemfunctusecount = elemusecountwithalignment;
		my_info.arg01 = (elemusecountwithalignment - elemcount);
		my_info.arg02 = NUM_BITS_USED_BY_QBETA;
		my_info.arg03 = qbetaval;
		my_info.arg04 = qbiasdivbyscalewrd;
		my_info.arg05 = qlogscalewrd;
		my_info.arg06 = qlogsumsqstep;
		my_info.arg07 = scaledrecipstep_shift;
		
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_beta_mult_log_vector_hvx(nn, &my_info);
		
		exp2infractminval = *(int32_t *)(my_info.minptr);
		exp2inintegermaxval = *(int32_t *)(my_info.maxptr);
	}
	else {
		worker_info.iptr = sumsqpadptr;
		worker_info.optr = fractpartpadptr;
		worker_info.oextraptr = integpartpadptr;
		worker_info.minptr = vecpadptr;
		worker_info.maxptr = vecextrapadptr;
		worker_info.padfillptr = &fractpartpadptr[elemcount];
		worker_info.padfillextraptr = &integpartpadptr[elemcount];
		worker_info.elemfunctusecount = (((elemusecountwithalignment / 2) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));
		worker_info.arg01 = (elemusecountwithalignment - elemcount);
		worker_info.arg02 = NUM_BITS_USED_BY_QBETA;
		worker_info.arg03 = qbetaval;
		worker_info.arg04 = qbiasdivbyscalewrd;
		worker_info.arg05 = qlogscalewrd;
		worker_info.arg06 = qlogsumsqstep;
		worker_info.arg07 = scaledrecipstep_shift;
		
		my_info.iptr = &sumsqpadptr[worker_info.elemfunctusecount];
		my_info.optr = &fractpartpadptr[worker_info.elemfunctusecount];
		my_info.oextraptr = &integpartpadptr[worker_info.elemfunctusecount];
		my_info.minptr = vecpadptr2;
		my_info.maxptr = vecextrapadptr2;
		my_info.padfillptr = &fractpartpadptr[elemcount];
		my_info.padfillextraptr = &integpartpadptr[elemcount];
		my_info.elemfunctusecount = (elemusecountwithalignment - worker_info.elemfunctusecount);
		copy_thread_args(&my_info, worker_info);
		
		nn_sem_init(&worker_info.donesem,0);
		nn_os_work_for_vector(nn, compute_qlrn_beta_mult_log_vector_hvx, &worker_info);
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_beta_mult_log_vector_hvx(nn, &my_info);
		nn_sem_wait(&worker_info.donesem);
		
		int32_t worker_min = *(int32_t *)(worker_info.minptr);
		int32_t my_min = *(int32_t *)(my_info.minptr);
		int32_t worker_max = *(int32_t *)(worker_info.maxptr);
		int32_t my_max = *(int32_t *)(my_info.maxptr);
		exp2infractminval = ((worker_min <= my_min) ? worker_min: my_min);
		exp2inintegermaxval = ((worker_max >= my_max) ? worker_max: my_max);
	}
	
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	step_time = get_and_display_step_time_cycles(nn, step_time, "qlrn step 2+3+4 - calc C = qbeta*log2{allparts(B)}:");
#endif
		
	/* step5 -> compute out_step adjustment to C based on found max{integralpart(C)} & min{fractionalpart(C)} */
	/* compute & recompute out_step and equivalent out_shift/factors based on min{fract(C)} & max{integ(C)} */
	{
		const int32_t EXP2_OUT_SHIFT_LIMIT = 12; // shift-limit based on exp2 in/out bitpos used by Alexnet LRN
		int32_t calcshift;
		
		// compute out_step based on min{fract(C)}
		out_max = in_max * exp2f(-exp2infractminval / (float)UINT32_MAX);
		out_min = in_min * exp2f(-exp2infractminval / (float)UINT32_MAX);
		out_step = (out_max - out_min) / (float)UINT8_MAX;
		out_qzeero = quantize_uint8(0.0f,out_min,out_max);
		
		// compute out_step-equivalent out_shift/factors based on min{fract(C)}
		update_log2_out_step_factor_shift(out_step, &qlog2_of_out_step, &exp2outmult16bitfactor, &exp2outleftshift);
		calcshift = (exp2inintegermaxval + exp2outleftshift);
		
		// recompute out_step and equivalent out_shift/factors based on max{integ(C)}
		if ((calcshift > EXP2_OUT_SHIFT_LIMIT) && (exp2outleftshift > 0)) {
			// recompute out_step if calcshiftlimit exceeded by integ(C)-based_shift+out_step-based_shift
			while ((calcshift > EXP2_OUT_SHIFT_LIMIT) && (exp2outleftshift > 0)) {
				exp2outleftshift--; // update expout shift incrementally while shift limit exceeded
				out_step *= (float)((int32_t)1 << 1); // update out_step incrementally correspondingly 
				if (out_step > in_step) {
					out_step = in_step; // ensure out_step <= in_step to avoid loss of precision
					break;
				}
			}
			
			// recompute out_step when between half & one with expoutshift non-zero while in_step one or higher
			if ((out_step >= 0.5f) && (out_step < 1.0f) && (in_step >= 1.0f) && (exp2outleftshift >= 0)) {
				out_step = 1.0f; // recompute/set out_step to one when listed conditions are matched
			}
			
			// recompute out_step-equivalent out_shift/factors based on min{fract(C)}
			update_log2_out_step_factor_shift(out_step, &qlog2_of_out_step, &exp2outmult16bitfactor, &exp2outleftshift);
			update_q8_min_max_params(out_qzeero, out_step, &out_min, &out_max);
		}
	}
	
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	step_time = get_and_display_step_time_cycles(nn, step_time, "qlrn step5  - compute outstep-adjust for C        :");
#endif
	
	/* step6 -> apply to C the out_step/min/max/qinzero adjustment based on max{integralpart(C)} & min{fractionalpart(C)} */
	/* - given C as original output of log2 to be given as negated input of exp2 */
	/* - apply outstep to get C (updated_log2_output) = C (original_log2_output) - log2(out_step) */
	/* - -C fract part will be used as input of exp2{} -> actual used range of C 0f to 1f -> 0x0000 to 0x7FFF (-C 0 to -1f) */
	/* - -C integer part will be used as shift to output of exp2{}  */
	/* step7 -> calc exp2[fractionalpart(-C)] via nonlinear tool -> 0x0000/0f->0x7FFF/1f format in -> Q16 format out */
	/* note alternate fract(-C) values in bits15-0 are the values for which exp2 needed - bits31-16 exp2 unneeded junk */
	/* here exp2 operates on 16bit-values provided with double the desired number of elements with alignment */
	// Pseudocode:
	// int outfrac = exp_negx_q1516(adjusted);
	if (use_multi_thread == 0) {
		my_info.optr = fractpartpadptr;
		my_info.oextraptr = integpartpadptr;
		my_info.elemfunctusecount = elemusecountwithalignment;
		my_info.arg01 = qlog2_of_out_step;
		
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_fract_exp_hvx(nn, &my_info);
	}
	else {
		worker_info.optr = fractpartpadptr;
		worker_info.oextraptr = integpartpadptr;
		worker_info.elemfunctusecount = (((elemusecountwithalignment / 2) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));
		worker_info.arg01 = qlog2_of_out_step;
		
		my_info.optr = &fractpartpadptr[worker_info.elemfunctusecount];
		my_info.oextraptr = &integpartpadptr[worker_info.elemfunctusecount];
		my_info.elemfunctusecount = (elemusecountwithalignment - worker_info.elemfunctusecount);
		copy_thread_args(&my_info, worker_info);
		
		nn_sem_init(&worker_info.donesem,0);
		nn_os_work_for_vector(nn, compute_qlrn_fract_exp_hvx, &worker_info);
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_fract_exp_hvx(nn, &my_info);
		nn_sem_wait(&worker_info.donesem);
	}
	
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	step_time = get_and_display_step_time_cycles(nn, step_time, "qlrn step6+7  - calc exp2[fractionalpart(-C)]     :");
#endif
	
	/* step8 -> calc D = exp2[-C] = {<exp2[fract(-C)]>>shift_due_to_integ(-C)>*out_step_based_factor}<<out_step_based_shift */
	/* note after applying shifts and factors the D value is limited to 8bit-range 0x0000-0x00FF but is stored in 16bits */
	/* also since exp2out quantized has its qinzero at 0 (minimum) there's need to include its qinzero in any computations */
	/* step9 -> calc E = (qindata-qinzero) */
	/* step9 -> & calc F = D*E = exp2[log2{qscaling*(qbias/qscaling+qsum_squared_inputs_over_win)}*-qbeta]*(qindata-qinzero) */
	/* also find max{F} & min{F} */
	/* note the F value is stored in 32bitvecpair */
	iptr = (uint8_t *)in_tensor->data;
	if (use_multi_thread == 0) {
		my_info.iptr = iptr;
		my_info.optr = fractpartpadptr;
		my_info.oextraptr = integpartpadptr;
		my_info.maxptr = vecextrapadptr;
		my_info.elemfunctusecount = elemusecountwithalignment;
		my_info.arg01 = elemcount;
		my_info.arg02 = qinzero;
		my_info.arg03 = exp2outmult16bitfactor;
		my_info.arg04 = exp2outleftshift;
		
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_exp_mult_data_hvx(nn, &my_info);
		
		exp2outqdataproductmaxval = *(int32_t *)(my_info.maxptr);
	}
	else {
		worker_info.iptr = iptr;
		worker_info.optr = fractpartpadptr;
		worker_info.oextraptr = integpartpadptr;
		worker_info.maxptr = vecextrapadptr;
		worker_info.elemfunctusecount = (((elemusecountwithalignment / 2) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));
		worker_info.arg01 = elemcount;
		worker_info.arg02 = qinzero;
		worker_info.arg03 = exp2outmult16bitfactor;
		worker_info.arg04 = exp2outleftshift;
		
		my_info.iptr = &iptr[worker_info.elemfunctusecount];
		my_info.optr = &fractpartpadptr[worker_info.elemfunctusecount];
		my_info.oextraptr = &integpartpadptr[worker_info.elemfunctusecount];
		my_info.maxptr = vecextrapadptr2;
		my_info.elemfunctusecount = (elemusecountwithalignment - worker_info.elemfunctusecount);
		copy_thread_args(&my_info, worker_info);
		
		nn_sem_init(&worker_info.donesem,0);
		nn_os_work_for_vector(nn, compute_qlrn_exp_mult_data_hvx, &worker_info);
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_exp_mult_data_hvx(nn, &my_info);
		nn_sem_wait(&worker_info.donesem);
		
		int32_t worker_max = *(int32_t *)(worker_info.maxptr);
		int32_t my_max = *(int32_t *)(my_info.maxptr);
		exp2outqdataproductmaxval = ((worker_max >= my_max) ? worker_max: my_max);
	}
	
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	step_time = get_and_display_step_time_cycles(nn, step_time, "qlrn step8+9  - calc F = D*E = exp2[...]*(qin-qz) :");
#endif
		
	/* step10 -> recompute out_step adjustment to F based on found max{F} & min{F} */
	/* recompute out_step and equivalent out_shift/factors based on found max{F} & min{F} */
	{
		const int32_t PROD_WRD_TO_HFW_SHIFT_8 = 8; // shift amount based on final stage in/out bitpos used by all LRN
		const int32_t PROD_WRD_TO_HFW_SHIFADJ_4 = 4; // shift adjustment based on final stage in/out bitpos used by Alexnet LRN
		int32_t expansionshiftforprod;
		int32_t finalstagetotalshift;
		int32_t shiftforstepratio;
		float log2ofstepratio;
		
		// recompute out_step based on max{F} & min{F} dependent expansionshift at final stage
		expansionshiftforprod = compute_expansionshift(exp2outqdataproductmaxval,PROD_WRD_TO_HFW_SHIFT_8,PROD_WRD_TO_HFW_SHIFADJ_4);
		out_step /= exp2f(expansionshiftforprod);
		update_q8_min_max_params(out_qzeero, out_step, &out_min, &out_max);
		
		// recompute out_step-equivalent out_shift/factors based on max{F} & min{F} via dependent expansionshift at final stage
		log2ofstepratio = log2f(in_step / out_step);
		shiftforstepratio = (int32_t)(log2ofstepratio) + 1;
		finalstagetotalshift = (PROD_WRD_TO_HFW_SHIFT_8 + FINAL_STG_MULT_FACTOR_USE_NUM_BITS - shiftforstepratio);
		finalstagemultfactor = exp2f(log2ofstepratio-(float)(shiftforstepratio)) * ((1<<FINAL_STG_MULT_FACTOR_USE_NUM_BITS)-1);
		if (finalstagetotalshift < 0) {
			finalstagedwnshift = 0; // ensure non-negative rightshift
			finalstageupshift = -finalstagetotalshift; // ensure non-negative leftshift
		}
		else {
			finalstagedwnshift = finalstagetotalshift; // ensure non-negative rightshift
			finalstageupshift = 0; // ensure non-negative leftshift
		}
	}
	
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	step_time = get_and_display_step_time_cycles(nn, step_time, "qlrn step10 - compute outstep-adjust for F        :");
#endif
	
	/* step11 -> compute G = qoutzero+F after applying to G and F the out_step adjustments based on max{F} & min{F} */
	/* step12 -> compute 8bits output = G which only uses 8bits in 16bits */
	/* note at this point output is already changed from 16bits to 8bits because of quantized zero addition */
	/* so just need to pack the output to 8bit vectors from 16bit vectors (no bit loss) */
	uint8_t * optr = (uint8_t *)out_tensor->data;
	if (use_multi_thread == 0) {
		my_info.optr = optr;
		my_info.oextraptr = fractpartpadptr;
		my_info.elemfunctusecount = elemusecountwithalignment;
		my_info.arg01 = out_qzeero;
		my_info.arg02 = finalstagemultfactor;
		my_info.arg03 = finalstagedwnshift;
		my_info.arg04 = finalstageupshift;
		
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_qzero_based_data_hvx(nn, &my_info);
	}
	else {
		worker_info.optr = optr;
		worker_info.oextraptr = fractpartpadptr;
		worker_info.elemfunctusecount = (((elemusecountwithalignment / 2) + ALIGN_SIZE - 1) & ~(ALIGN_SIZE - 1));
		worker_info.arg01 = out_qzeero;
		worker_info.arg02 = finalstagemultfactor;
		worker_info.arg03 = finalstagedwnshift;
		worker_info.arg04 = finalstageupshift;
		
		my_info.optr = &optr[worker_info.elemfunctusecount];
		my_info.oextraptr = &fractpartpadptr[worker_info.elemfunctusecount];
		my_info.elemfunctusecount = (elemusecountwithalignment - worker_info.elemfunctusecount);
		copy_thread_args(&my_info, worker_info);
		
		nn_sem_init(&worker_info.donesem,0);
		nn_os_work_for_vector(nn, compute_qlrn_qzero_based_data_hvx, &worker_info);
		nn_sem_init(&my_info.donesem,0);
		compute_qlrn_qzero_based_data_hvx(nn, &my_info);
		nn_sem_wait(&worker_info.donesem);
	}
	
#ifdef DEBUG_PRINT_LRN_CYCLECOUNT
	step_time = get_and_display_step_time_cycles(nn, step_time, "qlrn step11+12 - calc G = qoutzero+F w/outstep-adj:");
#endif
	
	/* report output w/min/max */
	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,out_min);
	out_min_tensor->data_size = sizeof(float);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_max_tensor,0,out_max);
	out_max_tensor->data_size = sizeof(float);
#ifdef DEBUG_PRINT_LRN_PERFORMANCE
	int end_time =  nn_os_get_cycles(nn);
	int elem_size = elemcount;
	printf("qlrn hvx cycles = %d (elements = %d)\n", (end_time-start_time), elem_size);
#endif
	
	return 0;
}


static int lrn_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking q lrn node %p",self);
	if (self->n_inputs != 7) return errlog(nn,"LRN wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"LRN wrong # outs");
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	const int32_t window_size = (int32_t) tensor_get_float(self->inputs[3], 0);
	if (window_size < 1) {
		return errlog(nn, "LRN invalid window size (< 1)"); // int(window_size)>=1 check
	}
	const float bias = tensor_get_float(self->inputs[4], 0);
	if (bias < 1.0) {
		return errlog(nn, "LRN unsupported bias-value (< 1.0)"); // bias>=1 check
	}
	const float alpha = tensor_get_float(self->inputs[5], 0);
	if (alpha < 0.0) {
		return errlog(nn, "LRN unsupported alpha-value (< 0.0)"); // alpha>0 check
	}
	const float beta = tensor_get_float(self->inputs[6], 0);
	if (beta < 0.0) {
		return errlog(nn, "LRN unsupported beta-value (< 0.0)"); // beta>0 check
	}
	for (uint32_t i = 0; i < self->n_inputs; i++) {
		if (self->inputs[i] == NULL) {
			return errlog(nn,"input %d NULL",i);
		}
	}
	for (uint32_t i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) {
			return errlog(nn,"output %d NULL",i);
		}
	}
	logmsg(nn,2,"q lrn %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedLRN_8_ref = {
	SFINIT(.execute, lrn_8_execute_ref),
	SFINIT(  .check, lrn_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedLRN_8 = {
	SFINIT(.execute, lrn_8_execute_hvx),
	SFINIT(  .check, lrn_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};


