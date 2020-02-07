
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for convolution
 */
#define ENABLE_VECTOR_WEIGHT_ARRANGE
//#undef ENABLE_VECTOR_WEIGHT_ARRANGE
/*
 * FIXME: temporary minmax buf should be on stack
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <stdlib.h>
#include <stdio.h>
#include <supernode_utils.h>
#ifdef __hexagon__
#include "hexagon_types.h"
#else
#include <malloc.h>
#endif
#include "hvx_hexagon_protos.h"

#include "nn_bufferpool.h"

#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif

#ifdef ENABLE_VECTOR_WEIGHT_ARRANGE
#include "op_supernode_procweights.h"
#endif

#define ENABLE_FASTSUMA_1x1 0

/* 
 * Size of VTCM to reserve for circular buffer
 * Rest of VTCM will possibly be used for weights 
 * 
 * Note: at last calculation, max required circbuf size is 160K.
 * For Inception V3, max slice of weights is 128K.  
 * 
 * 0 will disable this
 */
#define VTCM_CIRCBUF_SIZE (0)

/* 8x8 convolution --> 32 bits, biasadd, relu, quantizedown to 8 bits  */

/*
 * Input and output have ordering BHWD
 * Filter has ordering HWDB (B is # of filters)
 */

#ifdef __hexagon__
#include <hexagon_protos.h>
#endif

static int __attribute__((unused)) supernode_execute_ref(struct nn_node *self, struct nn_graph *nn) 
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	const struct tensor *bias_tensor = self->inputs[7];
	const struct tensor *bias_min_tensor = self->inputs[8];
	const struct tensor *bias_max_tensor = self->inputs[9];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	int32_t out_batches = in_batches;
	int32_t out_width = nn_pad_compute_outsize(in_width,filt_width,stride_width,self->padding);
	int32_t out_height = nn_pad_compute_outsize(in_height,filt_height,stride_height,self->padding);
	int32_t out_depth = filt_batches;

	int32_t batch;
	int32_t filt_x;
	int32_t filt_y;
	int32_t filt_z;
	int32_t out_x;
	int32_t out_y;
	int32_t out_z;

	int32_t in_y_base;
	int32_t in_x_base;

	uint8_t *in = in_tensor->data;
	uint8_t *filt = filt_tensor->data;
	uint8_t *bias = bias_tensor->data;
	uint8_t *out = out_tensor->data;

	uint8_t *instripe;
	uint8_t *filtstripe;
	int32_t *outstripe;

	int32_t in_element;
	int32_t filt_element;
	int32_t sum;
	int32_t minsum = 0;
	int32_t maxsum = 0;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements;
	/* FIXME: if you pad depth you should adjust tmp_out_size here!!! */
	size_t biasbuf_size = out_depth*sizeof(int32_t);

	int32_t *biasbuf = nn->scratch;
	int32_t *tmp_out = biasbuf + biasbuf_size;

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	float bias_min_float = tensor_get_float(bias_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);

	int32_t adj_x = ((out_width-1) * stride_width + filt_width - in_width) / 2;
	int32_t adj_y = ((out_height-1) * stride_height + filt_height - in_height) / 2;

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */
	if( !flt_isfinite(in_max_float) || !flt_isfinite(in_min_float)){
		return errlog(nn,"input range to shortin supernode contains NaN or Inf");
	}
	float in_level_size = (in_max_float - in_min_float) / 255;
	float filt_level_size = (filt_max_float - filt_min_float) / 255;
	float bias_level_size = (bias_max_float - bias_min_float) / 255;
	float out_level_size = in_level_size * filt_level_size;

	float bias_mpy_amt = (bias_level_size / out_level_size);
	int bias_adder = (bias_max_float / out_level_size);

	//float conv_out_max_val = ((float)(INT32_MAX)) * out_level_size;
	//float conv_out_min_val = 0.0f;

	float final_out_max_val;
	float final_out_min_val;

	uint32_t fixed_recip_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t input_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	int32_t bias_offset = quantize_uint(0.0f,bias_min_float,bias_max_float);

	int i,j;
	int shamt;

	logmsg(nn,2,"supernode execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"supernode input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
	logmsg(nn,2,"supernode filt %dx%dx%dx%d",filt_batches,filt_height,filt_width,filt_depth);
	logmsg(nn,2,"supernode stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"supernode padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if (out_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",out_tensor->max_size,out_size);
	}
	if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");
	if (out_min->max_size < sizeof(float)) return errlog(nn,"min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn,"max too small");

	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_size;

	/* 
	 * This *could* be changed to fixed point and vectorized, but it shouldn't
	 * impact performance that much, just traversing depth once. 
	 */
	for (i = 0; i < out_depth; i++) {
		int32_t biasval = bias[i];
		biasbuf[i] = fast_roundf((biasval - bias_offset) * bias_mpy_amt);
	}

	/* BEGIN CONV2D. results in tmp_out buffer, also maxsum is updated */

	for (batch = 0; batch < out_batches; batch++) {
	  for (out_y = 0; out_y < out_height; out_y++) {
	    in_y_base = out_y * stride_height - adj_y;
	    for (out_x = 0; out_x < out_width; out_x++) {
	      in_x_base = out_x * stride_width - adj_x;
	      outstripe = tmp_out+(out_depth*(out_x+
	                       out_width*(out_y+
	                       out_height*(batch))));
	      for (out_z = 0; out_z < out_depth; out_z++) {
	        sum = 0;
	        for (filt_y = 0; filt_y < filt_height; filt_y++) {
	          if ((in_y_base + filt_y) >= in_height) continue;
	          if ((in_y_base + filt_y) < 0) continue;
	          for (filt_x = 0; filt_x < filt_width; filt_x++) {
	            if ((in_x_base + filt_x) >= in_width) continue;
	            if ((in_x_base + filt_x) < 0) continue;
		    instripe = in+(in_depth*(in_x_base+filt_x+
	                           in_width*(in_y_base+filt_y+
                                   in_height*(batch))));
	            filtstripe = filt+(out_z + 
					out_depth*filt_depth*(filt_x+
					filt_width*(filt_y)));
	            for (filt_z = 0; filt_z < filt_depth; filt_z++) {
	              in_element = instripe[filt_z];
	              filt_element = filtstripe[filt_z*out_depth];
	              in_element -= input_offset;
	              filt_element -= filt_offset;
	              sum += in_element*filt_element;
	//logmsg(nn,9,"[%d %d %d %d]: sum += %d*%d --> %d",
	//	batch,out_y,out_x,out_z,in_element,filt_element,sum);
	            }
	          }
	        }
	        if (sum < minsum) minsum = sum;
	        if (sum > maxsum) maxsum = sum;
	        outstripe[out_z] = sum;
	      }
	    }
	  }
	}

	/* Adjust the maximum by adding the maximum possible bias */
	/* Check maxsum vs. precalculated min/max values */
	maxsum += bias_adder;
	shamt = 16-__builtin_clz(maxsum);
	if (shamt < 0) shamt = 0;
	maxsum >>= shamt;
	fixed_recip_level_size = 0x00FF0000U/maxsum;	// chosen to align at bit 16

	/* Now go back through, add bias, clip to positive and requantize. */

	for (j = 0; j < out_batches*out_height*out_width; j++) {
	  for (i = 0; i < out_depth; i++) {
	    sum = biasbuf[i] + tmp_out[j*out_depth+i];
	    int32_t out_i = ((sum >> shamt) * fixed_recip_level_size + (1<<15));
	    out_i >>= 16;
	    if (out_i < 0) out_i = 0;
            if (out_i > 255) out_i = 255;
	    *out++ = out_i;
	  }
	}

	final_out_max_val = maxsum * out_level_size;
	final_out_min_val = 0.0f;

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,final_out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,final_out_max_val);
	out_max->data_size = sizeof(float);
	logmsg(nn,2,"supernode execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}


/*
 * Reference version of CONVSUM / CONV
 * 
 */

static int32_t filter_one(
	const uint8_t *in,
	const uint8_t *filt_u,
	int32_t in_next_row,
	int32_t in_next_d32,
	int32_t x,
	int32_t y,
	int32_t out_z,
	int32_t filt_height,
	int32_t filt_width,
	int32_t in_depth)
{
	int32_t sum = 0;
	const uint8_t *y_base;
	const uint8_t *x_base;
	const uint8_t *z_base;
	int32_t filt_x;
	int32_t filt_y;
	int32_t filt_z;
	int32_t filt_depth = in_depth;
	int32_t filt_sub_z;
	const int8_t *filt = (const int8_t *)filt_u;
	for (filt_y = 0; filt_y < filt_height; filt_y++) {
		/* from image base, in_y * in_next_row */
		y_base = in + (y+filt_y) * in_next_row;
		for (filt_x = 0; filt_x < filt_width; filt_x++) {
			/* From the row base, move over 32 * x offset */
			x_base = y_base + (x+filt_x)*32;
			for (filt_z = 0; filt_z < in_depth; filt_z += 32) {
				/* From the x base, we need to jump by in_next_d32 for every 32 z */
				z_base = x_base + (filt_z/32) * in_next_d32;
				for (filt_sub_z = 0; filt_sub_z < 32; filt_sub_z++) {
					/* For each 32 elements of input z, sequential. */
					int32_t z_byte_idx = (filt_sub_z) % 4;
					int32_t z_word_idx = (filt_sub_z) / 4;
					int32_t filt_idx = z_byte_idx
						+ 4*(out_z % 32)
						+ 128*z_word_idx
						+ 128*filt_depth*filt_x
						+ 128*filt_depth*filt_width*filt_y
						+ 128*filt_depth*filt_width*filt_height*(out_z/32);
					int32_t in_element = z_base[filt_sub_z];
					int32_t filt_element = filt[filt_idx];
					sum += in_element * filt_element;
				}
			}
		}
	}
	return sum;
}

static void __attribute__((unused)) convsum_ref(
	uint8_t *out,
	const uint8_t *in,
	const uint8_t *filt,
	const int32_t *bias,

	int32_t in_width,
	int32_t in_next_row,
	int32_t in_depth,
	int32_t in_next_d32,

	int32_t out_width,
	int32_t out_next_row,

	int32_t out_height,
	int32_t filt_width,
	int32_t filt_height,
	int32_t stride_width,
	int32_t stride_height,
	int32_t *minmax,
	int32_t recip_shamt,
	int32_t recip_val,
	int32_t *emsuma_buf,
	int32_t filt_offset)
{
	int out_y;
	int out_x;
	int out_z;
	int32_t minsum = 0;
	int32_t maxsum = 0;
	int32_t out_idx = 0;
	for (out_y = 0; out_y < out_height; out_y++) {
		for (out_x = 0; out_x < out_width; out_x++) {
			for (out_z = 0; out_z < 32; out_z++) {
				/* FIXME: might need to skip by 1 here depending on padding */
				int32_t in_x = out_x * stride_width;
				/* FIXME: might need to skip by 1 here depending on padding */
				int32_t in_y = out_y * stride_height;
				int32_t sum = filter_one(
					in,
					filt,
					in_next_row,
					in_next_d32,
					in_x,
					in_y,
					out_z,
					filt_height,
					filt_width,
					in_depth);
				sum += bias[out_z];
				if (sum < minsum) minsum = sum;
				if (sum > maxsum) maxsum = sum;
				sum >>= recip_shamt;
				sum *= recip_val;
				sum += 1<<15;
				sum >>= 16;
				if (sum < 0) sum = 0;
				if (sum > 255) sum = 255;
				out[out_idx++] = sum;
			}
		}
	}
}

static inline uint8_t *supernode_filtbuf_location(
	struct nn_graph *nn,
	struct supernode_info_new *info,
	int pingpong,
	const uint8_t *weights_in,
	const uint32_t inner_weight_size)
{
#if defined(V65) || defined(V66)
	if (info->use_v65 || info->use_v66) {
		if (/* nn->vtcm_ptr && */ (inner_weight_size <= (nn->vtcm_size-VTCM_CIRCBUF_SIZE))) {
			return nn->vtcm_ptr;
		} else {
			logmsg(nn,1,"weights %ld dont fit @ %p",inner_weight_size,nn->vtcm_ptr);
		} 
	} else {
		logmsg(nn,1,"V65/V66 compilation, but not using v65 or v66, do not copy to vtcm");
	}
#endif
	/* For now, just in-place */
	return (uint8_t *)weights_in;
}


#if 1
static inline void debug_value_range(
	struct nn_graph *nn,
	const uint8_t *out,
	int out_width,
	int n_lines,
	int out_next_row) {}
#else
static void __attribute__((unused)) debug_value_range(
	struct nn_graph *nn,
	const uint8_t *out,
	int out_width,
	int n_lines,
	int out_next_row)
{
	int i,j;
	int max,max_idx;
	for (i = 0; i < n_lines; i++) {
		max = 0;
		for (j = 0; j < out_width*32; j++) {
			if (out[j] > max) {
				max = out[j];
				max_idx = j;
			}
		}
		logmsg(nn,0,"line %d: max @ %d/%d: 0x%x",i,max_idx/32,max_idx%32,max);
		out += out_next_row;
	};
}
#endif

#if 0
void supernode_zapslice(struct nn_graph *nn, void * vinfo)
{
  struct workitem *work = vinfo;
  struct supernode_info_new *info = work->info;

  int rl_depths = work->zap_rl_depths;
  int pad_left  = work->zap_left_size;
  int pad_right = work->zap_right_size;
  int pad_top   = work->zap_top_size*rl_depths;
  int pad_bot   = work->zap_bot_size*rl_depths;

  int out_y, in_y, j ;
  uint8_t *optr ;
  int in_height = work->zap_height * rl_depths;
  int32_t delta  = (info->filt_height+2)/info->stride_height;

  //location in input
  in_y = work->start_line * info->stride_height * rl_depths - pad_top;

  if (work->need_initialize_suma)
  for (out_y = work->start_line; out_y < work->stop_line+delta; out_y++)
  {
      for(j=0; (j < info->stride_height*rl_depths) || (in_y >= in_height && in_y <= (in_height + pad_bot)); j++)
      {
          optr = work->zap_left + in_y * info->in_next_d32;
          if(in_y < 0 || in_y >= in_height) {
              vmemset_asm(optr, info->in_offset, info->in_next_d32);
          } else {
              if (pad_left  > 0) vmemset_asm(optr, work->zap_value, pad_left);
              optr = work->zap_right + in_y * info->in_next_d32;
              if (pad_right > 0) vmemset_asm(optr, work->zap_value, pad_right);
          }
          in_y++;
      }
  }//out_y
  return;
}
#endif

/*
  table used to control stores of 96 bytes aligned to 0,32,64,96 bytes
 */
const uint8_t store_cntrl[384] __attribute__ ((aligned(128))) = {
  0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,
  0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,
  0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,
  0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,
  0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,
  0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,0x7,
  0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,
  0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf,
  0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,
  0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,0x1,
  0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,
  0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,0x3,
  0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,
  0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,0x6,
  0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,
  0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,0xc,
  0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,
  0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,0x8,
  0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
  0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
  0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
  0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
  0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
  0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0
};


static void supernode_execute_hvx_conv_work(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	struct nn_node *self = work->self;
	struct supernode_info_new *info = self->opaque;

	int32_t start_line = work->start_line;
	int32_t stop_line = work->stop_line;
	int32_t skip_lines = work->skip_lines;
	int32_t in_next_row = info->in_next_row;
	int32_t in_depth = info->in_depth;
	int32_t in_next_d32 = info->in_next_d32;
	int32_t out_width = info->out_width_processed;
	int32_t out_next_row = info->out_next_row;
	int32_t filt_width = info->filt_width;
	int32_t filt_height = info->filt_height;
	int32_t stride_width = info->stride_width;
	int32_t stride_height = info->stride_height;
	int32_t next_suma_row = info->next_suma_off;
        
	int32_t recip_val = info->recip_val;
	//int32_t recip_shamt = info->recip_shamt;
	int32_t out_left_junk = info->out_left_junk;
	int32_t skip_col = info->skip_col;
	int32_t weight_chunks = work->weight_chunks;
	int32_t out_next_d32 = info->out_next_d32;
	int32_t weight_batch_size = info->weight_batch_size;

	const uint8_t *weights = work->weights;
	const int32_t *biasbuf = work->biases;
	const uint8_t *input = work->input + start_line*in_next_row*stride_height;
	uint8_t *output = work->output + start_line*out_next_row;

#ifdef V65
	int32_t use_v65 = info->use_v65;
#else
	int32_t use_v65 = 0;
#endif
	uint64_t start_cycles;
	uint64_t my_cycles;
	union {
		HVX_Vector vec[2];
		int32_t words[64];
	} minmax;

	int w, out_row; 

	//logmsg(nn,2,"min: %d max: %d",info->minval,info->maxval);

	start_cycles = nn_os_get_cycles(nn);
	//if (start_line >= stop_line) goto done;

#if 1
	logmsg(nn,2,"start/stop/skip_lines: %d/%d/%d output %p input %p filt %p bias %p in_depth %d in_next_row %d in_next_d32 %d out_width %d out_next_row %d out_next_d32 %d filt_width %d filt_height %d stride_width %d stride_height %d recip_shamt %d recip_val 0x%08x minmax_buf %p out_left_junk=%d in_left_skip=%d dummy_zeros=%p n_lines=%d suma_buf_start=%p suma_start=%d suma_buf=%p next_suma_row=%d skip_col=%d weight_chunks=%d weight_batch_size=%d cycles=%lld",
			start_line,
			stop_line,
			skip_lines,
			output,
			input,
			weights,
			biasbuf,
			in_depth,
			in_next_row,
			in_next_d32,
			out_width,
			out_next_row,
			out_next_d32,
			filt_width,
			filt_height,
			stride_width,
			stride_height,
			0,
			recip_val,
			minmax.words,
			out_left_junk,
			info->in_left_skip,
			NULL,
			(stop_line - start_line + (skip_lines-1)) / skip_lines,
			work->suma_buf,
			info->suma_start,
			work->suma_buf + info->suma_start,
			next_suma_row,
			skip_col,
			weight_chunks,
			weight_batch_size,
			start_cycles);
#endif

	nn_sem_t * done_sem = work->donesem;	// sem to post at the end

	minmax.vec[1] = Q6_V_vsplat_R(0x7FFFFFFF);
	minmax.vec[0] = Q6_V_vnot_V(minmax.vec[1]);
	/*-------------------------------------------------------------*/
	/*              V65 Implementations                            */
	/*-------------------------------------------------------------*/
#ifdef V65
	if (use_v65) {
        int cbuf_row;
		//int early1 = work->weight_chunks > 1 && work->weight_chunks & 1;
		//int early2 = work->weight_chunks > 2 && !(work->weight_chunks & 1) ;

		if((work->circ_buf = nn_os_bufstack_pop(&info->bufstack)) == NULL) {
			logmsg(nn,0,"bufstack returned a NULL pointer....");
			goto done;
		}

		l2fetch(input, in_next_row, in_next_row, filt_height); 

		int in_width_pad = ((in_next_d32/32) - info->in_left_skip + 3+info->in_right_padpad)&(-4);
		int buf_width  = in_width_pad*2*in_depth;
		int buf_height = (filt_height > stride_height) ? filt_height : stride_height;

		repstream_t repstream = (info->num_accs == 8 && stride_width < 3) ? &repstream2_asm : &repstreamN_asm;
		conv2d_t func64, func32;

		if(info->num_accs == 8) {
			func64 = &gvconv2dbbb_circ_d64_v65_asm;
			func32 = &gvconv2dbbb_circ_d32_v65_asm;
		} else if(info->num_accs == 6) {
			func64 = &gvconv2dbbb_circ6_d64_v65_asm;
			func32 = &gvconv2dbbb_circ6_d32_v65_asm;
		} else {
			errlog(nn,"%ld accumulators not supported",info->num_accs);
		}
		logmsg(nn,2,"repstream %p input %p to circ buf %p",repstream,input,work->circ_buf);
		if (work->circ_buf == 0) logmsg(nn,0,"oops: circ buff NULL");
		(*repstream)(
			input,
			work->circ_buf,
			in_next_d32>>5,
			in_depth, filt_height,
			Q6_R_combine_RlRl(info->in_right_padpad, info->in_left_skip),
			stride_width,
			work->circ_buf,
			buf_height,
			info->in_offset, info->num_accs);

		if (work->do_conv) nn_sem_wait(work->do_conv);

		int32_t n_in_rows = Q6_R_min_RR(filt_height, stride_height); 

		for(cbuf_row = 0, out_row = start_line; out_row < stop_line; out_row++) {

			if(out_row != (stop_line-1)) {
				l2fetch(input+buf_height*in_next_row,in_next_row, in_next_row, n_in_rows);
			}

			//if(early1 && cbuf_row == 0) nn_sem_wait(work->copy_done);
			w = 0; 
			//logmsg(nn,0,"v65 w=%d weight_chunks=%d conv input=%p",w,weight_chunks,work->circ_buf + cbuf_row*buf_width);
			if(weight_chunks & 1) {  //1st time round and weight chunks odd
				(*func32) ( //gvconv2dbbb_circ_d32_v65_asm
					work->circ_buf + cbuf_row*buf_width,
					(int8_t *)weights,
					output,
					in_width_pad,
					out_next_row,
					out_width,
					Q6_R_combine_RlRl(stride_height,stride_width),
					in_depth,
					filt_width,
					filt_height,
					1, 
					biasbuf,
					minmax.words,
					work->recip, //recip_val,
					out_next_d32>>5,
					work->circ_buf,
					info->recip_shamt,
					info->in_offset,
					store_cntrl);
				w+=1;
			}

			//if(early2 && cbuf_row == 0) nn_sem_wait(work->copy_done);
			if(w < weight_chunks) {                          //do 64 of out_depth at once
				(*func64) ( //gvconv2dbbb_circ_d64_v65_asm
					work->circ_buf + cbuf_row*buf_width,
					(int8_t *)weights + w*weight_batch_size,
					output + w*out_next_d32,
					in_width_pad,
					out_next_row,
					out_width,
					Q6_R_combine_RlRl(stride_height,stride_width),
					in_depth,
					filt_width,
					filt_height,
					(weight_chunks-w)/2,
					biasbuf+w*32,
					minmax.words,
					work->recip + w*32, //recip_val,
					out_next_d32,
					work->circ_buf,
					info->recip_shamt,
					info->in_offset,
					store_cntrl);
			}//end w

			if(out_row < stop_line-1) { //dont do another repstream at end
				(*repstream)(
					input + buf_height*in_next_row,
					work->circ_buf + cbuf_row*buf_width,
					in_next_d32>>5,
					in_depth, n_in_rows,
					Q6_R_combine_RlRl(info->in_right_padpad, info->in_left_skip),
					stride_width,
					work->circ_buf,
					buf_height,
					info->in_offset, info->num_accs);
			}

			input  += in_next_row*stride_height;
			output += out_next_row;

			cbuf_row += stride_height;
			if (cbuf_row >= buf_height) cbuf_row -= buf_height;
		}//end acts
		//nn_checkpoint_arrival(nn,self,&info->conv_slices[work->start_chunk].checkpoint);
		if(!work->last_chunk){
			if (work->conv_done) nn_sem_post(work->conv_done);	// post this
		}
	} else 
#endif //#ifdef V65
    { 
    /*-------------------------------------------------------------*/
    /*              V60 Implementations                            */
    /*-------------------------------------------------------------*/
		//  The 'suma_scratch buffer' is used for the intermediate integral buffer;
		// only grab it if/when we need it.
		int suma_scratch_index;
		int32_t * suma_scratch = NULL;
		// actual suma goes here. Current vertical extent of current batch.
		int32_t *suma = work->suma_buf + start_line*next_suma_row/sizeof(int32_t);


		int32_t  proc_rows = work->num_lines;
		int32_t  pf_offset = Q6_R_max_RR(filt_height-stride_height, 0);

		int32_t n_lines = Q6_R_min_RR(stop_line-start_line, proc_rows);
		int32_t n_in_rows = (n_lines-1)*stride_height + filt_height; 

		// prefetch initial activations
		l2fetch(input, in_next_row, in_next_row, n_in_rows);
		for(out_row = start_line; out_row < stop_line; out_row+=proc_rows) {

			int32_t next_n_lines = Q6_R_min_RR(stop_line-out_row-proc_rows, proc_rows);
			int32_t next_n_in_rows = (next_n_lines-1)*stride_height + filt_height; 

			if( work->need_initialize_suma ){
				/* OK do sumabuf.
				 */
				if(filt_width == 1 && filt_height == 1 && ENABLE_FASTSUMA_1x1) {
					gsum_asm(
						input,
						suma,
						in_next_d32>>5,
						in_depth,
						n_lines,
						stride_height,
						info->filt_offset);
				} else {
					if( suma_scratch == NULL){
						suma_scratch =  bufpool_take( &info->bufpool_suma_scratch, &suma_scratch_index);
						if( suma_scratch == NULL){
							errlog(nn,"failed to get suma scratch");
							goto done;
						}
					}
					// four lines of "scratch128xW; the rest is for integral
					int32_t *scratch_128xW = suma_scratch;
					int32_t *integral_tmp = scratch_128xW + info->suma_width*4;
					// zero the first row of integral_tmp
					vmemset_asm(integral_tmp, 0, info->suma_width*sizeof(int32_t));

					gvint_asm(
						input,
						integral_tmp + info->suma_width, 	// second row of integral_tmp
						in_next_d32,
						in_next_row,
						info->suma_width,
						in_depth,
						n_in_rows,
						scratch_128xW,             // extra scratch buffer (4 rows)
						info->filt_offset);

					gvsuma_asm(
						integral_tmp,
						suma,
						info->suma_width,
						next_suma_row,
						stride_height,
						filt_width,
						filt_height,
						n_lines,
						filt_height*filt_width*in_depth*info->filt_offset*info->in_offset);
				}
			} else {
				//logmsg(nn,2,"waiting for progress...");
				//nn_progress_waitfor(suma_progress,suma_progress_need);
				// prefetch, if it was already done.
				// likely just been found.
				l2fetch(suma+info->suma_start, next_suma_row, info->in_width*sizeof(int32_t), n_lines);
				//logmsg(nn,2,"progress complete...");
			}
			// convolution
			for (w = 0; w < weight_chunks; w++) {

				// prefetch next batch of weights
				if (w < (weight_chunks-1)) {
					if (out_row==0) 
						l2fetch(weights + (w+1)*weight_batch_size, weight_batch_size/32, weight_batch_size/32,32);
				} else {
					if (work->pf_inp && out_row==(stop_line-n_lines)) 
						l2fetch(weights + weight_chunks*weight_batch_size, weight_batch_size/32, weight_batch_size/32,32);

						// prefetch activations
						if (next_n_lines > 0)
							l2fetch(input+(proc_rows*stride_height+pf_offset)*in_next_row, in_next_row, in_next_row, next_n_in_rows-pf_offset);
				}
				gvconv2dbbb_v60_asm(
					input,
					weights + w*weight_batch_size,
					output + w*out_next_d32,
					in_next_d32>>5,
					out_next_row,
					out_width,
					Q6_R_combine_RlRl(stride_height,stride_width),
					in_depth,
					filt_width,
					filt_height,
					n_lines,
					biasbuf+w*32,
					suma + info->suma_start,
					next_suma_row,
					minmax.words,
					work->recip + w*32,
					info->recip_shamt);
			}
			input += proc_rows*in_next_row*stride_height;
			output+= proc_rows*out_next_row;
			suma  += proc_rows*next_suma_row/sizeof(int32_t);
			n_lines   = next_n_lines;
			n_in_rows = next_n_in_rows;
		}
		// release suma_scratch, if needed.
		if( suma_scratch != NULL)
			bufpool_release( &info->bufpool_suma_scratch, suma_scratch_index);
	}
	gvrmaxmin(minmax.words);
	nn_atomic_min(&info->minval,minmax.words[32]);
	nn_atomic_max(&info->maxval,minmax.words[ 0]);

	my_cycles = nn_os_get_cycles(nn) - start_cycles;
	nn_atomic_add64(&info->cycles,my_cycles);
	//asm volatile ("":::"memory");
	logmsg(nn,2,"min=%d(%d) max=%d(%d) cycles=%lld",minmax.words[32],info->minval,minmax.words[0],info->maxval,my_cycles);
	debug_value_range(nn,work->output+start_line*out_next_row,out_width,stop_line-start_line,out_next_row*skip_lines);
	//logmsg(nn,0,"posting to %p",work->donesem);
done:
	if (use_v65) {
		if (work->circ_buf) nn_os_bufstack_push(&info->bufstack,work->circ_buf);
	}
	nn_checkpoint_arrival(nn,self,&info->conv_slices[work->start_chunk].checkpoint);
	if (done_sem != NULL) nn_sem_post(done_sem);
}


/*
    The v66 - Hana - convolution code
 */


static void supernode_execute_hvx_conv_work_v66(struct nn_graph *nn, void * vinfo)
{
#ifdef V66
	struct workitem *work = vinfo;
	struct nn_node *self = work->self;
	struct supernode_info_new *info = self->opaque;
	//int whoami = info->whoami;
	int32_t start_line = work->start_line;
	int32_t stop_line = work->stop_line;
	int32_t skip_lines = work->skip_lines;
	int32_t in_next_row = info->in_next_row;
	int32_t in_depth = info->in_depth;
	int32_t in_next_d32 = info->in_next_d32;
	//int32_t in_left_skip = info->in_left_skip;
	int32_t out_width = info->out_width;
	int32_t out_next_row = info->out_next_row;
	//int32_t out_depth = info->out_depth;
	//int32_t out_next_d32 = info->out_next_d32;
	//int32_t out_height = info->out_height;
	int32_t filt_width = info->filt_width;
	int32_t filt_height = info->filt_height;
	int32_t stride_width = info->stride_width;
	int32_t stride_height = info->stride_height;

	const uint8_t *input = work->input + start_line*in_next_row*stride_height;
	const uint8_t *weights = work->weights;
	const int32_t *biasbuf = work->biases;
	uint8_t *output = work->output + start_line * out_next_row;

	int32_t recip_val = info->recip_val;
	//int32_t recip_shamt = info->recip_shamt;
	int32_t out_left_junk = info->out_left_junk;
	int32_t skip_col = info->skip_col;
	int32_t weight_chunks = work->weight_chunks;
				
	int out_row, w;
	int32_t out_next_d32 = info->out_next_d32;
//	int32_t weight_batch_size = info->weight_batch_size;
	//uint64_t start_cycles;
	//uint64_t my_cycles;
	union {
		HVX_Vector vec[2];
		int32_t words[64];
	} minmax;

	if (unlikely(info->use_v66 != 1)) {
		errlog(nn,"trying to use v66 convolution but not in v66 mode!!");
		return;
	}
	/* EJP: I think this works, total lines - start line / stride, but round up */
	/* We may need to factor in stride in addition to this */
	//logmsg(nn,2,"min: %d max: %d",info->minval,info->maxval);
	//start_cycles = nn_os_get_cycles(nn);

	minmax.vec[1] = Q6_V_vsplat_R(0x7FFFFFFF);
	minmax.vec[0] = Q6_V_vnot_V(minmax.vec[1]);



	if (start_line >= stop_line) return; 
	/* Prefetch activation chunk */
	if (0) logmsg(nn,0,"l2fetch %p and %d %d",
		input+start_line*in_next_row*stride_height,
		in_next_row,
		(stop_line - start_line + (skip_lines-1)) / skip_lines);
#if 1
	logmsg(nn,1,"start/stop/skip_lines: %d/%d/%d output %p input %p filt %p bias %p in_depth %d in_next_row %d in_next_d32 %d out_width %d out_next_row %d out_next_d32 %d filt_width %d filt_height %d stride_width %d stride_height %d recip_shamt %d recip_val 0x%08x minmax_buf %p out_left_junk=%d in_left_skip=%d dummy_zeros=%p n_lines=%d skip_col=%d weight_chunks=%d",
			start_line,
			stop_line,
			skip_lines,
			output,
			input,
			weights,
			biasbuf,
			in_depth,
			in_next_row,
			in_next_d32,
			out_width,
			out_next_row,
			out_next_d32,
			filt_width,
			filt_height,
			stride_width,
			stride_height,
			0,
			recip_val,
			minmax.words,
			out_left_junk,
			info->in_left_skip,
			NULL,
			(stop_line - start_line + (skip_lines-1)) / skip_lines,
                        skip_col,
			weight_chunks);
#endif
    /*-------------------------------------------------------------*/
    /*              V66 Implementation                             */
    /*-------------------------------------------------------------*/
	int proc_rows = 2;
	int next_proc_rows = work->num_lines;
	int pf_offset = Q6_R_max_RR(filt_height-stride_height, 0);
	int n_lines = Q6_R_min_RR(stop_line-start_line, proc_rows); 
	int n_in_rows = (n_lines-1)*stride_height + filt_height;

	//logmsg(nn, 2, "convolving chunk # ptr %08x", (int) (weights+0*weight_batch_size));
	l2fetch(input, in_next_row, in_next_row, n_in_rows);

	if(stride_width == 2) {
		for(out_row = start_line; out_row < stop_line; ) {
			int next_n_lines = Q6_R_min_RR(stop_line-out_row-proc_rows, next_proc_rows);
			//wait_for_l2fetch();
			if(next_n_lines > 0) {
				int next_n_in_rows = (next_n_lines-1)*stride_height + filt_height;
				l2fetch(input+(proc_rows*stride_height+pf_offset)*in_next_row, in_next_row, in_next_row, next_n_in_rows-pf_offset);
			}
			gvconv2dbbb_v66_asm(
				input,
				(int8_t *)weights,
				output,
				in_next_d32>>5,
				out_next_row,
				out_width - 0*skip_col,
				Q6_R_combine_RlRl(stride_height*skip_lines,stride_width),
				in_depth,
				filt_width,
				filt_height,
				n_lines,
				biasbuf,
				minmax.words,
				work->recip, //recip_val,
				32*((4-out_left_junk)&3),
				Q6_R_combine_RlRl(skip_col, 32*info->in_left_skip+(stride_width == 2)),
				out_next_d32,weight_chunks,
				info->recip_shamt); //(int32_t const*)work->recip);

			input  += proc_rows*in_next_row*stride_height;
			output += proc_rows*out_next_row;
			n_lines = next_n_lines;
			out_row += proc_rows;
			proc_rows = next_proc_rows;
		}//end activation rows

	} else if(work->fix_overcompute==0) {

		for(out_row = start_line; out_row < stop_line; ) {
			int next_n_lines = Q6_R_min_RR(stop_line-out_row-proc_rows, next_proc_rows);
			//wait_for_l2fetch();
			if(next_n_lines > 0) {
				int next_n_in_rows = (next_n_lines-1)*stride_height + filt_height;
				l2fetch(input+(proc_rows*stride_height+pf_offset)*in_next_row, in_next_row, in_next_row, next_n_in_rows-pf_offset);
			}

			gvconv2dbbbs1_v66_asm(
				input,
				(int8_t *)weights,
				output,
				in_next_d32>>5,
				out_next_row,
				out_width - (skip_col & 4),
				Q6_R_combine_RlRl(stride_height*skip_lines,stride_width),
				in_depth,
				filt_width,
				filt_height,
				n_lines,
				biasbuf,
				minmax.words,
				work->recip, //recip_val,
				32*((4-out_left_junk)&3),
				skip_col,
				out_next_d32, weight_chunks,
				info->recip_shamt); //(int32_t const*)work->recip);
                
			input  += proc_rows*in_next_row*stride_height;
			output += proc_rows*out_next_row;
			n_lines = next_n_lines;
			out_row += proc_rows;
			proc_rows = next_proc_rows;
		}//end activation rows

	} else if(work->fix_overcompute==1) {

		for(out_row = start_line; out_row < stop_line; ) {
			int next_n_lines = Q6_R_min_RR(stop_line-out_row-proc_rows, next_proc_rows);
			//wait_for_l2fetch();
			if(next_n_lines > 0) {
				int next_n_in_rows = (next_n_lines-1)*stride_height + filt_height;
				l2fetch(input+(proc_rows*stride_height+pf_offset)*in_next_row, in_next_row, in_next_row, next_n_in_rows-pf_offset);
			}

			//int windex = (work->start_chunk + w) % work->vtcm_chunks;
			gvconv2dbbbs1x4_v66_asm(
				input,
				(int8_t *)weights,
				output,
				in_next_d32>>5,
				out_next_row,
				out_width,
				Q6_R_combine_RlRl(stride_height*skip_lines,stride_width),
				in_depth,
				filt_width,
				filt_height,
				n_lines,
				biasbuf,
				minmax.words,
				work->recip, //recip_val,
				0,
				0,
				out_next_d32,weight_chunks,
				info->recip_shamt); //(int32_t const*)work->recip);

			input  += proc_rows*in_next_row*stride_height;
			output += proc_rows*out_next_row;
			n_lines = next_n_lines;
			out_row += proc_rows;
			proc_rows = next_proc_rows;
		}//end activation rows

	} else { //handle cases of in_depth = 48 80 etc.

		for(out_row = start_line; out_row < stop_line; ) {
			int next_n_lines = Q6_R_min_RR(stop_line-out_row-proc_rows, next_proc_rows);
			//wait_for_l2fetch();
			if(next_n_lines > 0) {
				int next_n_in_rows = (next_n_lines-1)*stride_height + filt_height;
				l2fetch(input+(proc_rows*stride_height+pf_offset)*in_next_row, in_next_row, in_next_row, next_n_in_rows-pf_offset);
			}

			//int windex = (work->start_chunk + w) % work->vtcm_chunks;
			gvconv2dbbbs1_d16_v66_asm(
				input,
				(int8_t *)weights,
				output,
				in_next_d32>>5,
				out_next_row,
				out_width - (skip_col & 4),
				Q6_R_combine_RlRl(stride_height*skip_lines,stride_width),
				in_depth,
				filt_width,
				filt_height,
				n_lines,
				biasbuf,
				minmax.words,
				work->recip, //recip_val,
				32*((4-out_left_junk)&3),
				skip_col,
				out_next_d32,weight_chunks,
				info->recip_shamt); //(int32_t const*)work->recip);

			input  += proc_rows*in_next_row*stride_height;
			output += proc_rows*out_next_row;
			n_lines = next_n_lines;
			out_row += proc_rows;
			proc_rows = next_proc_rows;
		}//end activation rows
	}
	gvrmaxmin(minmax.words);
	nn_atomic_min(&info->minval,minmax.words[32]); 
	nn_atomic_max(&info->maxval,minmax.words[ 0]); 

	//logmsg(nn,2," posting back conv done sem %d sem %08x", 0, (int)(work->conv_done + 0));
	for(w=0; w < weight_chunks; w++) {
		//int windex = (work->start_chunk + w) % work->vtcm_chunks;
		nn_checkpoint_arrival(nn,self,&info->conv_slices[work->start_chunk + w].checkpoint);
		//nn_sem_post(work->conv_done + windex);
	}
	//my_cycles = nn_os_get_cycles(nn) - start_cycles;
	//nn_atomic_add64(&info->cycles,my_cycles);
	//logmsg(nn,2,"min=%d(%d) max=%d(%d) cycles=%lld",minmax.words[32],info->minval,minmax.words[0],info->maxval,my_cycles);
	//debug_value_range(nn,output+start_line*out_next_row,out_width,stop_line-start_line,out_next_row*skip_lines);
#endif
	return;
}

#if 0
static void supernode_execute_hvx_suma_work(struct nn_graph *nn, void *vinfo)
{
    struct workitem *work = vinfo;
    const struct supernode_info_new *info = work->info;
    int32_t in_depth = info->in_depth;
    int32_t in_next_row = info->in_next_row;
    int32_t in_next_d32 = info->in_next_d32;
    int32_t stride_height = info->stride_height;
    int32_t filt_width  = info->filt_width;
    int32_t filt_height = info->filt_height;

    int32_t start_line = work->start_line;
    int32_t stop_line  = work->stop_line;
    int32_t in_start_row   = start_line * stride_height * in_next_row;
    int32_t suma_start_row = start_line * info->next_suma_off/sizeof(int32_t);
    int32_t n_lines = stop_line - start_line;
    int32_t n_in_rows = (n_lines-1) * stride_height + filt_height;

    const uint8_t *suma_in = work->input    + in_start_row;
    int32_t *suma_out      = work->suma_buf + suma_start_row; 

    l2fetch(suma_in, in_next_row, in_next_row, n_in_rows);

    if(filt_width == 1 && filt_height == 1 && ENABLE_FASTSUMA_1x1) {
        gsum_asm(
            suma_in,
            suma_out,
            in_next_d32>>5, 
            in_depth, 
            n_lines, 
            stride_height, 
            info->filt_offset);
    } else {

        int32_t *scratch_128xW = work->suma_scratch;
        int32_t *integral_tmp = scratch_128xW + info->suma_width*4;

        vmemset_asm(integral_tmp, 0, info->suma_width*sizeof(int32_t));

        gvint_asm(
            suma_in,
            integral_tmp + info->suma_width, 
            in_next_d32,
            in_next_row,
            info->suma_width,          // width to compute // FIXME: make sure to round up from padded width
            in_depth,
            n_in_rows,
            scratch_128xW,             // extra scratch buffer
            info->filt_offset);

        gvsuma_asm(
            integral_tmp, 
            suma_out,
            info->suma_width,           // in width
            info->next_suma_off,        // next output width
            stride_height,
            filt_width,
            filt_height,
            n_lines,
            filt_height*filt_width*in_depth*info->filt_offset*info->in_offset);
    }
}
#endif

#if 0
static void supernode_execute_hvx_work(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	struct supernode_info_new *info = work->info;

        if(info->use_v66)
	    supernode_execute_hvx_conv_work_v66(nn, work);
        else
	    supernode_execute_hvx_conv_work(nn, work);

        return;
}
#endif
#if 0
static void __attribute__((unused)) supernode_execute_hvx_work_v66s1(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	struct nn_node *self = work->self;
	struct supernode_info_new *info = self->opaque;
	//int whoami = info->whoami;
	const uint8_t *input = work->input;
	const uint8_t *weights = work->weights;
	const int32_t *biasbuf = work->biases;
	uint8_t *output = work->output;
	int32_t start_line = work->start_line;
	int32_t stop_line = work->stop_line;
	int32_t skip_lines = work->skip_lines;
	int32_t *minmax_buf = work->minmax_buf;
	int32_t *suma_buf = work->suma_buf + info->suma_start;
	int32_t next_suma_row = info->next_suma_off;
	//int32_t in_width = info->in_width;
	int32_t in_next_row = info->in_next_row;
	int32_t in_depth = info->in_depth;
	int32_t in_next_d32 = info->in_next_d32;
	//int32_t in_left_skip = info->in_left_skip;
	int32_t out_width = info->out_width;
	int32_t out_next_row = info->out_next_row;
	//int32_t out_depth = info->out_depth;
	//int32_t out_next_d32 = info->out_next_d32;
	//int32_t out_height = info->out_height;
	int32_t filt_width = info->filt_width;
	int32_t filt_height = info->filt_height;
	int32_t stride_width = info->stride_width;
	int32_t stride_height = info->stride_height;
	int32_t recip_val = info->recip_val;
	//int32_t recip_shamt = info->recip_shamt;
	int32_t out_left_junk = info->out_left_junk;
	// EJP: FIXME: pass n_lines so it's precomputed and we don't get the divide.
	int32_t n_lines = (stop_line - start_line + (skip_lines-1)) / skip_lines;
	uint64_t start_cycles;
	uint64_t my_cycles;
	//uint32_t line;
	/* EJP: I think this works, total lines - start line / stride, but round up */
	/* We may need to factor in stride in addition to this */
	//memset(minmax_buf,0,128*2);
	//logmsg(nn,2,"min: %d max: %d",info->minval,info->maxval);
#if 0
	logmsg(nn,2,"start/stop/skip_lines: %d/%d/%d output %p input %p filt %p bias %p in_depth %d in_next_row %d in_next_d32 %d out_width %d out_next_row %d out_next_d32 %d filt_width %d filt_height %d stride_width %d stride_height %d recip_shamt %d recip_val 0x%08x minmax_buf %p out_left_junk=%d in_left_skip=%d dummy_zeros=%p n_lines=%d",
			start_line,
			stop_line,
			skip_lines,
			output,
			input,
			weights,
			biasbuf,
			in_depth,
			in_next_row,
			in_next_d32,
			out_width,
			out_next_row,
			0xcafebabe,
			filt_width,
			filt_height,
			stride_width,
			stride_height,
			0,
			recip_val,
			minmax_buf,
			out_left_junk,
			info->in_left_skip,
			dummy_zeros,
			n_lines);
#endif
	start_cycles = nn_os_get_cycles(nn);
	gvconv2dbbbs1_v66_asm(
		input+start_line*in_next_row*stride_height,
		weights,
		output + start_line*out_next_row,
		in_next_d32>>5,
		out_next_row*skip_lines,
		out_width,
		Q6_R_combine_RlRl(stride_height*skip_lines,stride_width),
		in_depth,
		filt_width,
		filt_height,
		n_lines,
		biasbuf,
		suma_buf,
		next_suma_row,
		minmax_buf,
		recip_val,
		32*((4-out_left_junk)&3),
		0);
	my_cycles = nn_os_get_cycles(nn) - start_cycles;
	nn_atomic_min(&info->minval,minmax_buf[32]);
	nn_atomic_max(&info->maxval,minmax_buf[0]);
	nn_atomic_add64(&info->cycles,my_cycles);
	//logmsg(nn,0,"min=%d(%d) max=%d(%d) cycles=%lld",minmax_buf[32],info->minval,minmax_buf[0],info->maxval,my_cycles);
	//debug_value_range(nn,output+start_line*out_next_row,out_width,n_lines,out_next_row*skip_lines);
	//logmsg(nn,0,"posting to %p",work->donesem);
	nn_sem_post(work->donesem);
}
#endif

#if 0
static inline void supernode_do_memcpy(struct nn_graph *nn, uint8_t *dst, const uint8_t *src, uint32_t size)
{
#ifdef V66
	logmsg(nn,2,"copy %d bytes %p --> %p",size,src,dst);
        asm volatile (
                "M0 = %2; memcpy(%0,%1,M0)"
        ::"r"(dst),"r"(src),"r"(size-1)
        :"m0");
	logmsg(nn,2,"memcpy complete",size,src,dst);
#elif defined(V65)
	logmsg(nn,2,"copy %d bytes %p --> %p",size,src,dst);
	nn_graph_hvx_blockcopy(dst,src,size);
	logmsg(nn,2,"memcpy complete",size,src,dst);
#endif
}
#endif

#if 0
//do the memcpy from ddr to vtcm
int supernode_memcpy(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
#if defined(V66) || defined(V65)
        if (work->copy_out == NULL) return 0;
        if(work->wait_before) {
          nn_sem_wait_n_times(work->conv_done, work->threads);
        }
        logmsg(nn,2,"memcpy'ing %p -> %p %ld bytes. waited on: %p",work->copy_in,work->copy_out,work->copy_size,work->conv_done);
	nn_graph_memcpy(nn,work->copy_out,work->copy_in,work->copy_size);
#endif
        return 0;
}

//wait for the final convoltions to complete
int supernode_check_sems(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
#ifdef V66
        int j;
        for(j=0; j < work->weight_chunks; j++) {
           logmsg(nn,2,"pending %08x ",(int)(work->conv_done+j));
           nn_sem_wait_n_times(work->conv_done+j, work->threads);
        }
#endif
        return 0;
}
#endif

// This workitem does the following:
//     (1) wait for 'conv_done', 'threads' times  (but only if conv_done is not NULL)
//     (2) Start a vector thread which:
//          (a) runs the copy, then
//          (b) does post(do_conv) 'work.threads' times
//
//   (step (2) is skipped if work->copy_out is null)
//
// This is used between groups of executions where weights need switching. The subsequent
// convolution workitems all take( do_conv) before reading the weights.
// The 'wait for conv_done' is used when this is not the first copy, in order to enure
//  the previous convolutions are finished.

#if 0
// EJP: FIXME: wait for all convs to complete.  Should be able to be triggered by checkpoint?
static inline void supernode_convs_done(
        struct nn_node *self,
        struct nn_graph *nn,
        struct supernode_info_new *info,
        uint32_t num_weight_chunks,
        uint32_t threads)
{
#ifdef V66
          struct workitem work;
          work.info = info;
          work.threads = threads;
          work.weight_chunks = num_weight_chunks;
          work.execute = supernode_check_sems;
          work.conv_done = &info->semaphores[0];
          logmsg(nn,2," check sem for conv %d", work.weight_chunks);
          supernode_add_work_item(self,nn,info,work);
#endif
          return;
}
#endif

// V66 version of adding weights to be copied.  
#if 0
static inline int supernode_weight_chunk(
        struct nn_node *self,
        struct nn_graph *nn,
        struct supernode_info_new *info,
        int wait_before,	// on v65/66, 1 => need wait for prev convs to finish; 0=> no wait
        const uint8_t *weights,
        uint32_t weight_chunk_size,
        uint32_t index,
	uint32_t threads)
{
#ifdef V66
        logmsg(nn,2,"chunk size = %ld, chunk num = %ld", weight_chunk_size, index);

        if (weight_chunk_size <= nn->vtcm_size) {

          int vtcm_num;
          struct workitem work = {0};
          work.info = info;

          vtcm_num = index % info->n_weight_batches; //num_weight_chunks;
          work.threads = threads;
          work.copy_in = weights + index*weight_chunk_size,  //set to zero to all get same weights
          work.copy_out = (uint8_t *)nn->vtcm_ptr + vtcm_num*weight_chunk_size;
          work.copy_size = weight_chunk_size;
          work.conv_done = &info->semaphores[vtcm_num];
          work.wait_before = wait_before;
          work.execute = supernode_memcpy;
          logmsg(nn,0," queue weights job %d src: %p dst: %p  continue %d sem %08x", index, work.copy_in,work.copy_out,vtcm_num, (int)work.conv_done);
          return supernode_add_work_item(self,nn,info,work);
        }
#endif
        return supernode_add_l2fetch(self,nn,info,weights+index*weight_chunk_size,weight_chunk_size/32,32);
}
#endif

#if 0
// Spawn off a convolution work job
//int supernode_execute_workitem_vector_dispatch(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
void supernode_execute_workitem_vector_dispatch(struct nn_graph *nn, void *vwork) // struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	//struct workitem *work = vwork;
	//logmsg(nn,0,"hvx launch work semaphore %p",work->donesem);
	nn_os_work_for_vector(nn,supernode_execute_hvx_work,vwork);
}
#endif

// Wait for threads to finish
int supernode_execute_workitem_vector_join(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	nn_sem_wait_n_times( work->donesem, NUM_THREADS);
	return 0;
}

int supernode_execute_workitem_join_some(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	nn_sem_wait_n_times( work->donesem, work->join_iters);
	return 0;
}

/*
 * We are computing (a-a_offset)*(b-b_offset)
 * If a_offset != 0, we need to compute a_offset * sum(b-b_offset)
 * We can sum the elements of the filter (offsetted) and multiply by a_offset
 * This we can do ahead of time.
 */
#ifndef ENABLE_VECTOR_WEIGHT_ARRANGE
#if 1
/*
   float model
 */
float supernode_rearrange_vector_d32(
  uint8_t *out_data,
  const uint8_t* in_data,
  int filt_height,
  int filt_width,
  int filt_depth,
  int filt_depth_total,
  int filt_batches,
  int filt_batches_total,
  int filt_offset,
  int32_t *weight_scale,
  int32_t *sumb
) {
  int x,y,z,d,s,v,i;
  int inval;
  int32_t max ;
  int32_t sign, max_sign ;
  int32_t sum, count=0;
  int32_t tmp, found0, found255;
  int32_t h, w, b, dhw = filt_depth*filt_height*filt_width;
  float   ftmp,global_scale,total_mean =0.f;
  float scaling; //, delta ;
  float RND;
         printf(" wzero = %d depth = %d\n",filt_offset, filt_depth);
        if( filt_offset < 127 ){     
             global_scale = (127.f)/(float)(255-filt_offset);
        }else if( filt_offset > 129){
             global_scale = (128.f)/(float)filt_offset;
        }else{
             global_scale =1.f;
        }
        found0 = found255 = 0;
        for (b = 0; b < filt_batches; b++) {
                max = 0;
                sum = 0;
                for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < filt_width; w++) {
                                for (d = 0; d < filt_depth_total; d++) {
                                        if (d >= filt_depth) tmp = filt_offset;
                                        else tmp = in_data[
                                                  h*filt_width*filt_depth*filt_batches
                                                + w*filt_depth*filt_batches
                                                + d*filt_batches
                                                + b];
                                        if(tmp == 0) found0 = 1;
                                        if(tmp == 255) found255 = 1;
                                        tmp -= filt_offset;
                                        if(tmp < 0) { sign = -1; tmp = -tmp; } else sign = 1;
                                        if(tmp > max) { max_sign = sign; max = tmp; }
                                        sum += tmp ;
                                }
                        }
                }
                if(max_sign == -1 && max > 129) {
                   scaling = (128.f)/(float)max;
                } else if(max_sign == 1 && max > 128) {
                   scaling = (127.f)/(float)max;
                } else {
                   scaling = 1.f;
                }
                if(scaling < 1.f) {
                     float mean = (float) sum / (float) dhw;
                     mean = (float) max / mean; 
                     total_mean += mean;
                     count += 1;
                }
                weight_scale[b] = (int32_t) ( 2147483648.f * scaling); 
        }
        //attempt to categorize the channels
        if(scaling < 1.f) { 
          //printf("total mean = %f\n", total_mean / (float) count);
          RND = (1.f/64.f)*total_mean / (float) count;
          if(RND > 0.5f) RND = 0.5f;
          //printf("rnd val = %f\n", RND);
        }
        if(!found0 && !found255) printf("max and min not found\n");
        for (b = filt_batches; b < filt_batches_total; b++) {
                weight_scale[b] = 0x7fffffff; //1.0
        }
        for (b = 0; b < filt_batches; b++) {
                sum = 0;
                scaling = (float)weight_scale[b] /2147483648.f ;  
                for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < filt_width; w++) {
                                for (d = 0; d < filt_depth_total; d++) {
                                        if (d >= filt_depth) inval = filt_offset;
                                        else inval = in_data[
                                                  h*filt_width*filt_depth*filt_batches
                                                + w*filt_depth*filt_batches
                                                + d*filt_batches
                                                + b];
                                        tmp = inval - filt_offset;
                                       if(scaling < 1.f) {
                                        ftmp = (float)tmp * scaling;
                                        if(ftmp < 0.f) { sign = 1; ftmp = -ftmp; } else sign = 0;
                                        tmp = ftmp + RND;
                                        if(sign) tmp = -tmp;
                                       }
                                        if(tmp > 127) tmp = 127;
                                        if(tmp <-128) tmp =-128;
                                        sum += tmp; //inval - filt_offset; //tmp;
                                }
                        }
                }
                sumb[b] = sum;
        }
        for (b = filt_batches; b < filt_batches_total; b++) {
                sumb[b] = 0;
        }
        for (x = 0; x < filt_batches_total; x+=32)
        {
          for (y = 0; y < filt_height; y++)
          {
            for (d = 0; d < filt_depth_total; d+=32)
            {
              for (z = 0; z < filt_width; z++)
              {
                  for (v = 0; v < 32; v+=4)
                  for (s = 0; s < 32; s+=1)
                  for (i = 0; i < 4; i+=1)
                  {
                    int in_d = d+v+i;
                    int in_b = x+s;

                    scaling = (float)weight_scale[in_b] /2147483648.f ;  
                    int in_idx = y*filt_width*filt_depth*filt_batches
                                + z*filt_depth*filt_batches
                                + in_d*filt_batches
                                + in_b;
                    int out_idx = x*filt_height*filt_width*filt_depth_total
                                + y*filt_width*filt_depth_total*32
                                + z*32*32
                                + d*filt_width*32
                                + v*32
                                + 4*s
                                + i;
                    if ((in_d >= filt_depth) || (in_b >= filt_batches)) {
                       inval = filt_offset;
                    } else { 
                       inval = in_data[in_idx];
                    }

                    tmp  = (inval - filt_offset) ;
                    if(scaling < 1.f) {
                      ftmp = (float)tmp * scaling ;
                      if(ftmp < 0.f) { sign = 1; ftmp = -ftmp; } else sign = 0;
                      tmp = ftmp + RND;
                      if(sign) tmp = -tmp;
                    }
                    if(tmp > 127) tmp = 127;
                    if(tmp <-128) tmp =-128;
                    out_data[out_idx] = (int8_t)tmp;
                  }
              }//filt_width
            }//in_depth
          }//filt_height
        }//out_depth
  return(global_scale);
}
#else

/* 
   fixed point model to match supernode_procweights 
 */
float supernode_rearrange_vector_d32(
  uint8_t *out_data,
  const uint8_t* in_data,
  int filt_height,
  int filt_width,
  int filt_depth,
  int filt_depth_total,
  int filt_batches,
  int filt_batches_total,
  int filt_offset,
  int32_t *weight_scale,
  int32_t *sumb
) {
  int x,y,z,d,s,v,i;
  int inval;
  int32_t max = 0;
  int32_t sign, max_sign = 0;
  int32_t sum;
  int32_t tmp, RND;
  int32_t h, w, b;
  float   global_scale ;
  int32_t scaleK, offsetK;
  float * scaling = nn_memalign(128,  4*filt_batches_total);

        if( filt_offset < 128 ){     
             scaleK = (256*127+127)/(255-filt_offset);
        }else if( filt_offset > 128){
             scaleK = (256*128+127)/filt_offset;
        }else{
             scaleK = 0;
        }
        if(scaleK == 0)
          global_scale = 1.f;
        else
          global_scale = (float)scaleK*(float)(1./256.f);

        for (b = 0; b < filt_batches; b++) {
                max = 0;
                for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < filt_width; w++) {
                                for (d = 0; d < filt_depth_total; d++) {
                                        if (d >= filt_depth) tmp = filt_offset;
                                        else tmp = in_data[
                                                  h*filt_width*filt_depth*filt_batches
                                                + w*filt_depth*filt_batches
                                                + d*filt_batches
                                                + b];
                                        tmp -= filt_offset;

                                        if(tmp < 0) { sign = -1; tmp = -tmp; } else sign = 1;
                                        if(tmp > max) { max_sign = sign; max = tmp; }
                                }
                        }
                }
                if(max_sign == -1 && max > 128) {
                   scaleK = (128*256+127)/max;
                } else if(max_sign == 1 && max > 127) {
                   scaleK = (127*256+127)/max;
                } else {
                   scaleK = 0;
                }
                weight_scale[b] = scaleK; 
        }
        for (b = filt_batches; b < filt_batches_total; b++) {
                weight_scale[b] = 0;
        }
        for (b = 0; b < filt_batches; b++) {
                scaleK = weight_scale[b];
                if(scaleK == 0) sum = 0;
                else sum = -128*filt_height*filt_width*filt_depth_total;
                offsetK = 128*256 + 128 - scaleK*filt_offset;
                for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < filt_width; w++) {
                                for (d = 0; d < filt_depth_total; d++) {
                                        if (d >= filt_depth) tmp = filt_offset;
                                        else tmp = in_data[
                                                  h*filt_width*filt_depth*filt_batches
                                                + w*filt_depth*filt_batches
                                                + d*filt_batches
                                                + b];
                                        if(scaleK == 0) {
                                          tmp = tmp - filt_offset;
                                          if(tmp > 127) tmp = 127;
                                          if(tmp <-128) tmp =-128;
                                          sum += tmp;
                                        } else {
                                          uint8_t w = (tmp * scaleK + offsetK) >> 8;
                                          sum += w;
                                        }
                                }
                        }
                }
                sumb[b] = sum; 
        }
        for (b = filt_batches; b < filt_batches_total; b++) {
                sumb[b] = 0;
        }
        for (x = 0; x < filt_batches_total; x+=32)
        {
          for (y = 0; y < filt_height; y++)
          {
            for (d = 0; d < filt_depth_total; d+=32)
            {
              for (z = 0; z < filt_width; z++)
              {
                  for (v = 0; v < 32; v+=4)
                  for (s = 0; s < 32; s+=1)
                  for (i = 0; i < 4; i+=1)
                  {
                    int in_d = d+v+i;
                    int in_b = x+s;

                    scaleK = weight_scale[in_b];  
                    offsetK = 128*256 + RND - scaleK*filt_offset;
                    int in_idx = y*filt_width*filt_depth*filt_batches
                                + z*filt_depth*filt_batches
                                + in_d*filt_batches
                                + in_b;
                    int out_idx = x*filt_height*filt_width*filt_depth_total
                                + y*filt_width*filt_depth_total*32
                                + z*32*32
                                + d*filt_width*32
                                + v*32
                                + 4*s
                                + i;
                    if ((in_d >= filt_depth) || (in_b >= filt_batches)) {
                       inval = filt_offset;
                    } else { 
                       inval = in_data[in_idx];
                    }

                    if(scaleK==0) {
                      tmp  = inval - filt_offset;
                      if(tmp > 127) tmp = 127;
                      if(tmp <-128) tmp =-128;
                      out_data[out_idx] = tmp; //^0x80;
                    } else {
                      uint8_t w = (inval * scaleK + offsetK) >> 8;
                      out_data[out_idx] = w ^ 0x80;
                    }
                  }
              }//filt_width
            }//in_depth
          }//filt_height
        }//out_depth
        for (b = 0; b < filt_batches_total; b++) {
                int32_t p;
                scaleK = weight_scale[b];
                if(scaleK == 0) {
                  p = 0x7fffffff;
                } else {
                  p = (int32_t) (2147483648.f * (float)scaleK/256.f);
                }
                weight_scale[b] = p;
        }
        nn_free(scaling);
  return(global_scale);
}
#endif
/* current v65 model */
#if 1
float supernode_rearrange_scaler_d32(
  uint8_t *out_data,
  const uint8_t* in_data,
  int filt_height,
  int filt_width,
  int filt_depth,
  int filt_depth_total,
  int filt_batches,
  int filt_batches_total,
  int filt_offset,
  int32_t *sumb
) {
  int x,y,z,d,s,v,i;
  int inval;
  int32_t sum;
  int32_t tmp, RND = 128;
  int32_t h, w, b;
  float   global_scale ;
  int32_t scaleK, offsetK;

        if( filt_offset < 128 ){     
             scaleK = (256*127+127)/(255-filt_offset);
             offsetK = 128*256 + RND - scaleK*filt_offset;
        }else if( filt_offset > 128){
             scaleK = (256*128+127)/filt_offset;
             offsetK = 128*256 + RND - scaleK*filt_offset;
        }else{
             scaleK = 0;
             offsetK = 0;
        }
        if(scaleK == 0)
          global_scale = 1.f;
        else
          global_scale = (float)scaleK*(float)(1./256.f);

        for (b = 0; b < filt_batches; b++) {
                if(scaleK == 0) sum = 0;
                else sum = -128*filt_height*filt_width*filt_depth_total;
                offsetK = 128*256 + 128 - scaleK*filt_offset;
                for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < filt_width; w++) {
                                for (d = 0; d < filt_depth_total; d++) {
                                        if (d >= filt_depth) tmp = filt_offset;
                                        else tmp = in_data[
                                                  h*filt_width*filt_depth*filt_batches
                                                + w*filt_depth*filt_batches
                                                + d*filt_batches
                                                + b];
                                        if(scaleK == 0) {
                                          tmp = tmp - filt_offset;
                                          if(tmp > 127) tmp = 127;
                                          if(tmp <-128) tmp =-128;
                                          sum += tmp;
                                        } else {
                                          uint8_t w = (tmp * scaleK + offsetK) >> 8;
                                          sum += w;
                                        }
                                }
                        }
                }
                sumb[b] = sum; 
        }
        for (b = filt_batches; b < filt_batches_total; b++) {
                sumb[b] = 0;
        }
        for (x = 0; x < filt_batches_total; x+=32)
        {
          for (y = 0; y < filt_height; y++)
          {
            for (d = 0; d < filt_depth_total; d+=32)
            {
              for (z = 0; z < filt_width; z++)
              {
                  for (v = 0; v < 32; v+=4)
                  for (s = 0; s < 32; s+=1)
                  for (i = 0; i < 4; i+=1)
                  {
                    int in_d = d+v+i;
                    int in_b = x+s;

                    int in_idx = y*filt_width*filt_depth*filt_batches
                                + z*filt_depth*filt_batches
                                + in_d*filt_batches
                                + in_b;
                    int out_idx = x*filt_height*filt_width*filt_depth_total
                                + y*filt_width*filt_depth_total*32
                                + z*32*32
                                + d*filt_width*32
                                + v*32
                                + 4*s
                                + i;
                    if ((in_d >= filt_depth) || (in_b >= filt_batches)) {
                       inval = filt_offset;
                    } else { 
                       inval = in_data[in_idx];
                    }

                    if(scaleK==0) {
                      tmp  = inval - filt_offset;
                      if(tmp > 127) tmp = 127;
                      if(tmp <-128) tmp =-128;
                      out_data[out_idx] = tmp; //^0x80;
                    } else {
                      uint8_t w = (inval * scaleK + offsetK) >> 8;
                      out_data[out_idx] = w ^ 0x80;
                    }
                  }
              }//filt_width
            }//in_depth
          }//filt_height
        }//out_depth
  return(global_scale);
}
#else
float supernode_rearrange_scaler_d32(
  uint8_t *out_data,
  const uint8_t* in_data,
  int filt_height,
  int filt_width,
  int filt_depth,
  int filt_depth_total,
  int filt_batches,
  int filt_batches_total,
  int filt_offset,
  int32_t *sumb
) {
  int x,y,z,d,s,v,i;
  int inval;
  int32_t sum;
  int32_t tmp;
  int32_t h, w, b;
  float global_scale ;
        if( filt_offset < 127 ){       /// max = 255-zval = 
             global_scale = 127.0/(float)(255-filt_offset);
        }else if( filt_offset > 129){
             global_scale = 128.0/(float)(filt_offset);
        }else{
             global_scale = 1.0;
        }
        for (b = 0; b < filt_batches; b++) {
                sum = 0;
                for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < filt_width; w++) {
                                for (d = 0; d < filt_depth_total; d++) {
                                        if (d >= filt_depth) tmp = filt_offset;
                                        else tmp = in_data[
                                                  h*filt_width*filt_depth*filt_batches
                                                + w*filt_depth*filt_batches
                                                + d*filt_batches
                                                + b];
                                        tmp -= filt_offset;
                                        tmp = (float) tmp * global_scale;
                                        if(tmp > 127) tmp = 127;
                                        if(tmp <-128) tmp =-128;
                                        sum += tmp;
                                }
                        }
                }
                sumb[b] = sum;
        }
        for (b = filt_batches; b < filt_batches_total; b++) {
                sumb[b] = 0;
        }
        for (x = 0; x < filt_batches_total; x+=32)
        {
          for (y = 0; y < filt_height; y++)
          {
            for (d = 0; d < filt_depth_total; d+=32)
            {
              for (z = 0; z < filt_width; z++)
              {
                  for (v = 0; v < 32; v+=4)
                  for (s = 0; s < 32; s+=1)
                  for (i = 0; i < 4; i+=1)
                  {
                    int in_d = d+v+i;
                    int in_b = x+s;

                    int in_idx = y*filt_width*filt_depth*filt_batches
                                + z*filt_depth*filt_batches
                                + in_d*filt_batches
                                + in_b;
                    int out_idx = x*filt_height*filt_width*filt_depth_total
                                + y*filt_width*filt_depth_total*32
                                + z*32*32
                                + d*filt_width*32
                                + v*32
                                + 4*s
                                + i;
                    if ((in_d >= filt_depth) || (in_b >= filt_batches)) inval = filt_offset;
                    else inval = in_data[in_idx];

                    inval  = (inval - filt_offset) * global_scale;
                    if(inval > 127) inval = 127;
                    if(inval <-128) inval =-128;
                    out_data[out_idx] = (int8_t)inval;
                  }
              }//filt_width
            }//in_depth
          }//filt_height
        }//out_depth
  return(global_scale);
}
#endif

/*
    v60 case where weights are unsigned
 */
void supernode_rearrange_d32(
  uint8_t *out_data,
  const uint8_t* in_data,
  int filt_height,
  int filt_width,
  int filt_depth,
  int filt_depth_total,
  int filt_batches,
  int filt_batches_total,
  int filt_offset,
  int32_t *sumb
) {
  int x,y,z,d,s,v,i;
  int inval;
        int32_t sum;
        int32_t tmp;
        int32_t h, w, b;

        for (b = 0; b < filt_batches; b++) {
                sum = 0;
                for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < filt_width; w++) {
                                for (d = 0; d < filt_depth_total; d++) {
                                        if (d >= filt_depth) tmp = filt_offset;
                                        else tmp = in_data[
                                                  h*filt_width*filt_depth*filt_batches
                                                + w*filt_depth*filt_batches
                                                + d*filt_batches
                                                + b];
                                        sum += tmp;
                                }
                        }
                }
                sumb[b] = sum ;
        }
        for (b = filt_batches; b < filt_batches_total; b++) {
		sumb[b] = filt_height*filt_width*filt_depth_total*filt_offset;
        }
        for (x = 0; x < filt_batches_total; x+=32)
        {
          for (y = 0; y < filt_height; y++)
          {
            for (d = 0; d < filt_depth_total; d+=32)
            {
              for (z = 0; z < filt_width; z++)
              {
                  for (v = 0; v < 32; v+=4)
                  for (s = 0; s < 32; s+=1)
                  for (i = 0; i < 4; i+=1)
                  {
                    int in_d = d+v+i;
                    int in_b = x+s;

                    int in_idx = y*filt_width*filt_depth*filt_batches
                                + z*filt_depth*filt_batches
                                + in_d*filt_batches
                                + in_b;
                    int out_idx = x*filt_height*filt_width*filt_depth_total
                                + y*filt_width*filt_depth_total*32
                                + z*32*32
                                + d*filt_width*32
                                + v*32
                                + 4*s
                                + i;
                    if ((in_d >= filt_depth) || (in_b >= filt_batches)) inval = filt_offset;
                    else inval = in_data[in_idx];

                    out_data[out_idx] = inval;
                  }
              }//filt_width
            }//in_depth
          }//filt_height
        }//out_depth
  return;
}

#endif //!ENABLE_VECTOR_WEIGHT_ARRANGE



#if 0

static inline int supernode_filt_d32_in_idx(
	int h,
	int w,
	int id,
	int od,
	int height,
	int width,
	int in_depth,
	int out_depth)
{
	return od 
		+ id * out_depth
		+ w * out_depth * in_depth
		+ h * width * out_depth * in_depth;
}
/*
 * Most major: 4-wide along input depth dimension
 * Next most major: group of 32 output depth
 * Next most major: back along input depth dimension, but just 8 of them to get to a group of ID=32
 * Next most major: width
 * Next most major: the rest of the input depth
 * Next most major: height
 * Least major: the rest of the output depth
 */

static inline int supernode_filt_d32_out_idx(
	int h,
	int w,
	int id,
	int od,
	int height,
	int width,
	int in_depth,
	int out_depth)
{
	int id_chunk = id / 32;
	int od_chunk = od / 32;
	int id_bigoff = (id % 32) / 4;
	int id_smoff = id % 4;
	int od_off = od % 32;
	return id_smoff
		+ od_off * 4
		+ id_bigoff * 4 * 32
		+ w * 32 * 32
		+ id_chunk * 32 * width * 32 
		+ h * width * in_depth * 32
		+ od_chunk * 32 * height * width * in_depth;
}



static inline void supernode_rearrange_for_d32(
	uint8_t *dst,
	const uint8_t *src,
	int filt_height,
	int filt_width,
	int in_depth,
	int out_depth)
{
	int h,w,od,id;
	int in_idx;
	int out_idx;
	for (od = 0; od < out_depth; od++) {
		for (h = 0; h < filt_height; h++) {
			for (w = 0; w < filt_width; w++) {
				for (id = 0; id < in_depth; id ++) {
					in_idx = supernode_filt_d32_in_idx(h,w,od,id,
						filt_height,filt_width,in_depth,out_depth);
					out_idx = supernode_filt_d32_out_idx(h,w,od,id,
						filt_height,filt_width,in_depth,out_depth);
					dst[out_idx] = src[in_idx];
				}
			}
		}
	}
}
#else
void fast_rearrange_32x32(
  uint8_t *out_data,
  const uint8_t *in_data,
  int filt_batches)
{
  /* Input is 32 batches consecutive at any alignment, and we have 32 depth at filt_batches stride */
  /* Output should be 4D * 32B * 8D */

  /* Read 4 unaligned vectors @ in+0, in+filt_batches,in+filt_batches*2,in+filt_batches*3 */
  /* Shuffle data together */
  /* Store out aligned vector */
  /* Increment in by filt_batches*4 */
  /* Increment out by vector size */

  /* Repeat 8x */
  /* Reference: */
  int v,s,i;
  for (v = 0; v < 32; v+=4)
  for (s = 0; s < 32; s+=1)
  for (i = 0; i < 4; i+=1) {
    out_data[v*32+s*4+i] = in_data[(v+i)*filt_batches+s];
  }
}
static void __attribute__((unused))
supernode_rearrange_for_d32(
  uint8_t *out_data,
  const uint8_t* in_data,
  int filt_height,
  int filt_width,
  int filt_depth,
  int filt_depth_total,
  int filt_batches,
  int filt_batches_total,
  int filt_offset) {
  int x,y,z,d,s,v,i;

  //out_width = 32*filt_width*filt_height*32;
  //out_height = out_depth/32 * in_depth/32;
  uint8_t inval;
  for (x = 0; x < filt_batches_total; x+=32)
  {
    for (y = 0; y < filt_height; y++)
    {
      for (d = 0; d < filt_depth_total; d+=32)
      {
        for (z = 0; z < filt_width; z++)
        {
	  if (((d+32) < filt_depth) && (x+32) < filt_batches) {
            int in_idx = y*filt_width*filt_depth*filt_batches
                       + z*filt_depth*filt_batches
                       + d*filt_batches
                       + x;
            int out_idx = x*filt_height*filt_width*filt_depth_total
                        + y*filt_width*filt_depth_total*32
                        + z*32*32
                        + d*filt_width*32;
            fast_rearrange_32x32(out_data+out_idx,in_data+in_idx,filt_batches);
          } else {
            for (v = 0; v < 32; v+=4)
            for (s = 0; s < 32; s+=1)
            for (i = 0; i < 4; i+=1)
            {
              int in_d = d+v+i;
              int in_b = x+s;
              int in_idx = y*filt_width*filt_depth*filt_batches
                     + z*filt_depth*filt_batches
                     + in_d*filt_batches
                     + in_b;
              int out_idx = x*filt_height*filt_width*filt_depth_total
                          + y*filt_width*filt_depth_total*32
                          + z*32*32
                          + d*filt_width*32
                          + v*32
                          + 4*s
                          + i;
              if ((in_d >= filt_depth) || (in_b >= filt_batches)) inval = filt_offset;
              else inval = in_data[in_idx];
              out_data[out_idx] = inval;
            }
          }//filt_width
        }//segment
      }//in_depth
    }//filt_height
  }//out_depth
  return;
}

#endif

#if 0
static void __attribute__((unused)) supernode_execute_suma(struct nn_graph *nn, void *vinfo)
{
	struct workitem *work = vinfo;
	//struct nn_node *self = work->self;
	const struct supernode_info_new *node = work->info;
	const uint8_t *input = work->suma_in;
	int32_t *integral_tmp = work->suma_int_tmp;
	int32_t *sumabuf = work->suma_output;
	int32_t *scratch_128xW = work->suma_scratch;
	int32_t vertical_pad = 1;
	int32_t num_lines = work->suma_num_lines;
	//int32_t in_width_total = node->in_width;
	int32_t in_depth = node->in_depth;
	int32_t in_next_row = node->in_next_row;
	int32_t stride_height = node->stride_height;
	int32_t integral_width = node->suma_width;
	//int32_t integral_offset = node->integral_off;
	int32_t next_integral_offset = node->next_suma_off;
	int32_t filt_height = node->filt_height;
	int32_t filt_width = node->filt_width;
	int32_t in_offset = node->in_offset;
	int32_t filt_offset = node->filt_offset;
	int32_t integral_out_height = (num_lines-1) * stride_height + filt_height + vertical_pad;
	int32_t in_next_d32 = node->in_next_d32;
	int32_t in_width = node->in_width;

#if 1
	logmsg(nn,3,"in_offset=%d filt_offset=%d num_lines=%d", in_offset,filt_offset,num_lines);
	logmsg(nn,3,"in_next_row=%d,next_integral_offset=%d,integral_width=%d",in_next_row,next_integral_offset,integral_width);
	logmsg(nn,3,"int_out_h=%d,intctl=%p,sumabuf=%p",integral_out_height,integral_control,sumabuf);
	
	//vmemset_asm(nn->scratch,0xAB,nn->scratch_size);
	//vmemset_asm(nn->scratch,0x00,next_integral_offset*integral_out_height*4);
	logmsg(nn,3,"gvint_asm(%p,%p,0x%x,0x%x,0x%x,0x%x,0x%x,%p,0x%x,%p,%d,%d)",
		input,
		integral_tmp,
		in_next_d32,
		in_next_row,
		//next_integral_offset,		// Bytes to next integral
		integral_width,			// width to compute
		in_depth,
		integral_out_height,
		scratch_128xW, 			// extra scratch buffer
		filt_offset,
		integral_control,
		in_offset,
		in_width);
#endif

        //point wise convolution run special fast path
        if(filt_width == 1 && filt_height == 1 && ENABLE_FASTSUMA_1x1) {
            gsum_asm(input,  sumabuf, in_next_d32>>5, in_depth, num_lines, stride_height, filt_offset);
        } else {
        //int i;
        //for(i=0; i < 128; i++) printf("%d,",input[i]);
	gvint_asm(
		input,
		integral_tmp,
		in_next_d32,
		in_next_row,
		//next_integral_offset,		// Bytes to next integral
		integral_width,			// width to compute // FIXME: make sure to round up from padded width
		in_depth,
		integral_out_height,
		scratch_128xW, 			// extra scratch buffer
		filt_offset,
		integral_control,
		in_offset);
		//in_width);
#if 0
	do {
		int *p = (int *)integral_tmp;
		int i,j;
		logmsg(nn,0,"integral_tmp=%p",p);
		for (i = 0; i < work->suma_num_lines; i++) {
			if (i > 7) break;
			for (j = 0; j < integral_width; j+=8) {
				logmsg(nn,0,"(%d,%d-%d) 0x%08x %08x %08x %08x %08x %08x %08x %08x",
					i,j,j+7,
					p[j+0],
					p[j+1],
					p[j+2],
					p[j+3],
					p[j+4],
					p[j+5],
					p[j+6],
					p[j+7]);
			}
			p += integral_width;
		}
	} while (0);
#endif
	logmsg(nn,3,"gvsuma_asm(%p,%p,0x%x,0x%x,0x%x,0x%x,0x%x,0x%x,0x%x)",
		integral_tmp,
		sumabuf,
		integral_width,			// in width
		next_integral_offset,		// next output width
		stride_height,
		filt_width,
		filt_height,
		num_lines+vertical_pad,
		filt_height*filt_width*in_depth*filt_offset*in_offset);
		


         //gvsuma does all the fiddly bit now conv2d only needs horizontal offset
	gvsuma_asm(
		integral_tmp,
		sumabuf,
		integral_width,			// in width
		next_integral_offset,		// next output width
		stride_height,
		filt_width,
		filt_height,
		num_lines,
		filt_height*filt_width*in_depth*filt_offset*in_offset);
        }//end  1x1
	//do { int i;
	//for (i = 0; i < num_lines+1; i++) sumabuf[i*integral_width-3] = sumabuf[i*integral_width-4];
	//for (i = 0; i < num_lines+1; i++) sumabuf[i*integral_width-2] = sumabuf[i*integral_width-3];
	//for (i = 0; i < num_lines+1; i++) sumabuf[i*integral_width-1] = sumabuf[i*integral_width-2]; } while (0);
	//for (i = 0; i < num_lines+1; i++) sumabuf[i*integral_width-1] = sumabuf[i*integral_width]; } while (0);
#if 0
	do {
		int *p = (int *)sumabuf;
		int i,j, K = in_depth*filt_offset*in_offset;
		logmsg(nn,0,"sumabuf=%p",p);
		for (i = 0; i < work->suma_num_lines; i++) {
			//if (i > 7) break;
			for (j = 0; j < in_width; j+=8) {
				logmsg(nn,0,"(%d,%d-%d) 0x%08x %08x %08x %08x %08x %08x %08x %08x",
					i,j,j+7,
					K+p[j+0],
					K+p[j+1],
					K+p[j+2],
					K+p[j+3],
					K+p[j+4],
					K+p[j+5],
					K+p[j+6],
					K+p[j+7]);
			}
			p += in_width;
		}
	} while (0);
#endif
	nn_sem_post(work->donesem);
}

static int __attribute__((unused)) supernode_spawn_suma(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	uint64_t start;
	uint64_t stop;
	nn_sem_t donesem;
	nn_sem_init(&donesem,0);
	work->donesem = &donesem;
	start = nn_os_get_cycles(nn);
	nn_os_work_for_vector(nn,supernode_execute_suma,work);
	nn_sem_wait(&donesem);
	stop = nn_os_get_cycles(nn);
	record_usertime(nn,node,NN_GRAPH_PERFEVENT_USER1,stop-start);
	if (0) logmsg(nn,0,"node id=%x suma_time=%lld",node->node_id,stop-start);
	return 0;
}

/*
 * Add any work items needed for SUMA, and return pointer to SUMA buffer 
 */
static inline __attribute__((unused)) void *supernode_add_suma(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info,
	const uint8_t *input)
{
#if defined(V65) || defined(V66)
	logmsg(nn,0,"I thought this was if zero'd out");
	return NULL;
#endif
	if (info->use_signed_weights) return NULL;
	struct workitem suma_work;
	//int32_t in_width_total = info->in_width;
	//int32_t in_height = info->in_height;
        //int32_t suma_buf_size = 2*((((in_width_total+16+16+31)&(~31))+8)*in_height); //math error nonalgnd ptr
	//int32_t suma_buf_size = 2*((((in_width_total+16+16+31)&(~31)))*(8+in_height));
	//int32_t integral_tmp_size = suma_buf_size;
	//int32_t scratch_size = in_width_total * 128;
	//int32_t *sumabuf = nn_scratch_alloc(nn, suma_buf_size + integral_tmp_size + scratch_size);
	int32_t suma_buf_size = (info->next_suma_off*info->out_height + 256)/sizeof(int32_t); 
	int32_t integral_tmp_size = info->suma_width*(info->in_height+1); 
	int32_t scratch_size = 4*info->suma_width; 
	int32_t *sumabuf = nn_scratch_alloc(nn, (suma_buf_size + integral_tmp_size + scratch_size)*sizeof(int32_t));
	int32_t *integral_tmp = sumabuf + suma_buf_size;
	int32_t *suma_scratch = integral_tmp + integral_tmp_size;
	suma_work.self = self;
	suma_work.info = info;
	suma_work.suma_in = input;
	suma_work.suma_output = sumabuf;
	suma_work.suma_int_tmp = integral_tmp;
	suma_work.suma_scratch = suma_scratch;
	suma_work.suma_num_lines = info->out_height;
	suma_work.execute = supernode_spawn_suma;
	supernode_add_work_item(self,nn,info,suma_work);
	return sumabuf;
}
#endif

/* 
 * EJP: FIXME:
 * When we have large activations and small weights, it's better to chop up the activation
 * into pieces and bring through the entire set of weights rather than run through the 
 * activation multiple times with a fixed set of weights.
 * 
 * Find minimum reasonable weights L2$ footprint.  2x weight slices for cache-only system
 * 0 for memcpy into vtcm
 * 1x weight slices for vmemcpy to vtcm?
 * 
 * Find activation total size
 * If activation size + weights L2$ footprint < L2$ size, we're good
 * Find total number+size of weight slices
 * While doesn't fit (based on L2$ footprint of weights & activation slice size):
 *   Increase slicing factor that minimizes traffic
 *   If we try and slice up the activation to fit better, we bring in 2x the weights
 *   If we try and slice up the weights to fit better, we bring in 2x the activation
 *   We have to bring in the whole activation + whole weights once
 *   Activation might be hot in the cache, weights probably not
 *   Early in the graph, activations are big and weights are comparatively small
 *   Later in ght graph, activations are small and weights are comparatively big
 * 
 *   Since we have to bring in all the weights at least once, if the whole activation
 *   fits + 2 slices of weights that's probably the best option.  If not, we want to
 *   find some combination of slices of weights and slices of activations that fits nicely.
 */

static inline int32_t cache_weight_footprint(int weight_slice_size, int weight_slices)
{
#ifdef V66
	return 0;
#elif V65
	return weight_slice_size;
#else
	return weight_slice_size * ((weight_slices >= 2) ? 2 : 1);
#endif
}

static inline int32_t friendly_cache_footprint() {
#ifdef V66
	return 900*1024;
#else
	return 400*1024;
#endif
}

static inline int32_t cache_footprint(
	int activation_slice_size,
	int activation_slices,
	int weight_slice_size,
	int weight_slices,
	int out_slice_size)
{
	int32_t cache_footprint = cache_weight_footprint(weight_slice_size,weight_slices) 
		+ activation_slice_size * ((activation_slices >= 2) ? 2 : 1)
		+ 0*out_slice_size;
	return cache_footprint;
}

static inline __attribute__((unused)) int32_t cache_friendly(
	int activation_slice_size,
	int activation_slices,
	int weight_slice_size,
	int weight_slices,
	int out_slice_size)
{
	int footprint = cache_footprint(activation_slice_size,activation_slices,
		weight_slice_size,weight_slices,out_slice_size);
	if (footprint < friendly_cache_footprint()) return 1;
	return 0;
}

static inline int32_t slice_for_cache_v66(
        int32_t node_id,
        /* General stuff */
        struct nn_graph *nn,
        /* activation information */
        int32_t batches,
        int32_t d32_slices,
        int32_t width_total,
        int32_t out_width,
        int32_t out_height,
        int32_t stride_height,
        int32_t filt_width,
        int32_t filt_height,
        /* Weights information */
        int32_t weight_d32_size,
        int32_t weight_d32_slices,
        /* Results! */
        int32_t *inner_act_batches,
        int32_t *inner_act_rows,
        int32_t *inner_weight_chunks,
        int32_t *burst_chunks)
{
        int32_t input_height = out_height*stride_height+filt_height-stride_height;
        int32_t size_per_line = 32*d32_slices*width_total;
        int32_t act_total_size = size_per_line*out_height*stride_height;
        int32_t weight_total_size = weight_d32_size * weight_d32_slices;
        logmsg(nn,2,"id=%d in_h=%d out_h=%d sz_per_line=%d w_d32_sz=%d w_d32_slices=%d act_tot=%d w_tot_sz=%d fw=%d fh=%d",
                   node_id,
                input_height, out_height, size_per_line,
                weight_d32_size, weight_d32_slices,
                act_total_size, weight_total_size,filt_width,filt_height);

        if (weight_d32_size > nn->vtcm_size) {
                logmsg(nn,1,"weight slice too big %d", weight_d32_size);
                *inner_act_batches = 1;
                *inner_weight_chunks = 1;
                *burst_chunks = 1;
        } else if (weight_total_size <= nn->vtcm_size) {
                logmsg(nn,1,"weights fit");
                *inner_act_batches = batches;
                *inner_weight_chunks = weight_d32_slices;
                if(weight_d32_size <= (nn->vtcm_size/2)) {
                  *burst_chunks = Q6_R_min_RR(2, weight_d32_slices);
                } else {
                  *burst_chunks = 1;
                }
        } else {
                logmsg(nn,1,"weights dont fit");
                *inner_act_batches = 1; //batches;
                int32_t nchunks  = (nn->vtcm_size)/weight_d32_size;
                *inner_weight_chunks = nchunks;
                if(weight_d32_size <= (nn->vtcm_size/2)) {
                  *burst_chunks = Q6_R_min_RR(2, weight_d32_slices);
                } else {
                  *burst_chunks = 1;
                }
        }
	//*inner_weight_chunks &= (-(*burst_chunks));
        int32_t overhead = Q6_R_max_RR(filt_height,stride_height)*size_per_line;
        int32_t nchunks = *inner_weight_chunks;

        //(overhead+2*nlines1*size_per_line)*NUM_THREADS + weight_d32_size*nchunks = 384*1024
        int32_t nlines0 = (28*1024 + size_per_line*stride_height - 1)/(size_per_line*stride_height);
        int32_t nlines1 = (768*1024-NUM_THREADS*32*out_width*nchunks-NUM_THREADS*overhead +
                          2*NUM_THREADS*size_per_line*stride_height -1)/(2*NUM_THREADS*size_per_line*stride_height);
        int32_t nlines2 = ((stride_height*out_height+1)/4 - filt_height + stride_height + 1)/(4*stride_height);
        nlines1 = Q6_R_min_RR(nlines0,nlines1);
        *inner_act_rows = Q6_R_max_RR(Q6_R_min_RR(nlines2,nlines1), 1);
	return 0;
}
#if 1
static inline int32_t slice_for_cache(
        int32_t node_id,
        /* General stuff */
        struct nn_graph *nn,
        /* activation information */
        int32_t batches,
        int32_t d32_slices,
        int32_t width_total,
        int32_t out_width,
        int32_t out_height,
        int32_t stride_height,
        int32_t filt_width,
        int32_t filt_height,
        /* Weights information */
        int32_t weight_d32_size,
        int32_t weight_d32_slices,
        /* Results! */
        int32_t *inner_act_batches,
        int32_t *inner_act_rows,
        int32_t *inner_weight_chunks)
{
#ifdef V65
        int32_t input_height = out_height*stride_height+filt_height-stride_height;
        int32_t size_per_line = 32*d32_slices*width_total;
        int32_t buf_height = (filt_height > stride_height) ? filt_height+1 : stride_height+1;
        int32_t act_total_size = 2*NUM_THREADS*size_per_line*buf_height;
        int32_t weight_total_size = weight_d32_size * weight_d32_slices;
        logmsg(nn,1,"id=%d in_h=%d out_h=%d sz_per_line=%d w_d32_sz=%d w_d32_slices=%d act_tot=%d w_tot_sz=%d",
                   node_id,
                input_height,
                out_height,
                size_per_line,
                weight_d32_size,
                weight_d32_slices,
                act_total_size,
                weight_total_size);

        *inner_act_rows = 1;
        if (weight_d32_size > nn->vtcm_size) {
                logmsg(nn,1,"weight slice too big %d", weight_d32_size);
                *inner_act_batches = 1;
                *inner_weight_chunks = 1;
        } else if (weight_total_size <= nn->vtcm_size) {
                logmsg(nn,1,"weights fit");
                *inner_act_batches = 1;
                *inner_weight_chunks = weight_d32_slices;
        } else {
                logmsg(nn,1,"weights dont fit");
                *inner_act_batches = batches;
#if 0
                /* See how many weight slices fit in cache, 1 current and 1 on the way  try to do 2 else do 1 at a time */
                *inner_weight_chunks = (nn->vtcm_size)/(weight_d32_size);
                if (*inner_weight_chunks < 1) *inner_weight_chunks = 1;
                *inner_weight_chunks = (*inner_weight_chunks + 1)&(~1); //round up to 2

                if(*inner_weight_chunks * weight_d32_size > nn->vtcm_size) *inner_weight_chunks -= 1;

                int num_blocks = weight_d32_slices / *inner_weight_chunks;
                num_blocks = (weight_d32_slices+num_blocks-1) / num_blocks;

                /*if chunks still odd make even if it balances load more equally use more od64 */
                if(*inner_weight_chunks & 1 && num_blocks < *inner_weight_chunks) *inner_weight_chunks -= 1;

                if(*inner_weight_chunks < 1) *inner_weight_chunks = 1;
#else
                int32_t nchunks  = (nn->vtcm_size)/weight_d32_size;
                int32_t nchunks2 = Q6_R_max_RR(nchunks&~1, 1);     //round to 2
                int32_t niter  = (weight_d32_slices + nchunks -1)/nchunks;
                int32_t niter2 = (weight_d32_slices + nchunks2-1)/nchunks2;

                *inner_weight_chunks = (niter2 == niter) ? nchunks2 : nchunks;
#endif
        }
#else
        int32_t size_per_line = 32*d32_slices*width_total;
        //luc
        logmsg( nn,1,"id=%d in_h=%d out_h=%d sz_per_line=%d w_d32_sz=%d w_d32_slices=%d act_tot=%d w_tot_sz=%d",
                node_id,
                out_height*stride_height+filt_height-stride_height,
                out_height,
                size_per_line,
                weight_d32_size,
                weight_d32_slices,
                size_per_line*batches*(out_height*stride_height+filt_height-stride_height),
                weight_d32_size * weight_d32_slices);

        int32_t nchunks = Q6_R_min_RR(Q6_R_max_RR(128*1024/weight_d32_size, 1), weight_d32_slices);
        int32_t overhead = (filt_height-stride_height)*size_per_line; 

        if(filt_width!=1 || filt_height!=1) {
            overhead += (filt_height+2)*width_total*sizeof(int32_t);
        }

        int32_t datasize_for_outrow = stride_height*size_per_line + width_total*sizeof(int32_t) + 32*out_width*nchunks;
        int32_t nlines0 = 32*1024/(size_per_line*stride_height);
        int32_t nlines1 = (nn->vtcm_size-(nchunks+1)*weight_d32_size-NUM_THREADS*overhead)/(NUM_THREADS*datasize_for_outrow);

        *inner_act_batches = batches;
        *inner_act_rows = Q6_R_max_RR(Q6_R_min_RR(nlines0,nlines1), 1); 
        *inner_weight_chunks = nchunks;
#endif
        return 0;
}
#endif

void note_chunk_checkpoint_arrival(struct nn_graph *nn, struct nn_node *self, void *opaque)
{
	struct supernode_info_new *info = self->opaque;
	struct weight_slice_info *myslice = opaque;
	//long idx = myslice - info->conv_slices;
#if 1
	logmsg(nn,2,"chunk checkpoint did %d... copy %d bytes %p --> %p, wakeup %d @ offset %d alldone=%d/%d",
		myslice->checkpoint.required,
		myslice->copy_size,myslice->copy_in,myslice->copy_out,
		myslice->wakeup_items,myslice->batch_start_offset,
		info->alldone_checkpoint.count,info->alldone_checkpoint.required);
#endif
	if (myslice->copy_size > 0) supernode_do_memcpy(nn,myslice->copy_out,myslice->copy_in,myslice->copy_size);
	supernode_execute_some_strategy(self,nn,info->batch_start_idx + myslice->batch_start_offset, myslice->wakeup_items);
	nn_checkpoint_arrival(nn,self,&info->alldone_checkpoint);
}

//luc
//extern const char *info_id2name(unsigned int id);
//extern const char *info_id2opname(unsigned int id);

/*
 * TODO: have multiple strategies, and go through them to find one that matches.
 * Common code will be refactored into helper functions
 * Scenarios:
 * * Tiny enough to just do reference / scalar version
 * * Input in D32, large output depth (currently implemented)
 * * Input in normal format, want D32, large output depth
 * * Input in normal format, want D32, short output depth
 * * Input in D32, short output depth
 * * Short input depth, large output depth
 */

/*
 * What do we have to do before calling the work function?
 *
 * Some things are cacheable:
 * * Biasbuf number range conversion
 * * Adding min_out value (converted) into biasbuf
 * * Adding gemsumb values (if needed) into biasbuf
 * * Adding in_offset * weight_offset * N into biasbuf (if needed)
 * * Partitioning scheme
 * * Reciprocal for range conversion
 * * Min and Max output information
 * * Strategy for partitioning
 * * Work function
 * * output shape/padding
 *
 * * Ensure that cached value is still accurate given these inputs (if not, recalc)
 * * Any preproc (im2col) or padding adjustments
 * * Add work_for_vector items
 * * Wait for workers to complete
 * * Any fixup / postproc 
 * 
 * * The work function is passed to work_for_vector.  It can be different for different architectures.
 * * The strategy for partitioning could be a set of functions.
 */
int supernode_recalculate_strategy(struct nn_node *self, struct nn_graph *nn)
{
	/* Pull out the inputs we need */
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	//const struct tensor *min_in_tensor = self->inputs[2];
	//const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	//const struct tensor *bias_tensor = self->inputs[7];
	//const struct tensor *bias_min_tensor = self->inputs[8];
	//const struct tensor *bias_max_tensor = self->inputs[9];
	/* Find the output tensors */
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	/* Structures with auxillary information */
	struct supernode_info_new *info = self->opaque;
	/* 
	 * Find the dimensions of the input data, 
	 * both dimensions of data as well as padding
	 */
	info->in_shape = in_tensor->shape;

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_left_pad = in_tensor->format.width_pad[0];
	int32_t in_right_pad = in_tensor->format.width_pad[1];
	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];
	int32_t in_top_pad = in_tensor->format.height_pad[0];
	int32_t in_bottom_pad = in_tensor->format.height_pad[1];
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;

	/* Find the dimensions of the filter weights.  filt_batches == out_depth */
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	//int32_t filt_depth = filt_tensor->shape.filt_depth;

	/* Find the stride values */
	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	/* Calculate output dimensions */
	int32_t out_batches = in_batches;
	int32_t out_depth = filt_batches;

	int32_t required_w_before, required_w_after;
	int32_t required_h_before, required_h_after;

	/* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */

	int32_t out_width = nn_pad_compute_outsize_and_pad(in_width,filt_width,stride_width,self->padding,
			& required_w_before, & required_w_after);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height,filt_height,stride_height,self->padding,
		& required_h_before, & required_h_after);

	// total number of inputs on each row actually needed (including the required_w before and after).
	// this is normally the same as required_w_before + in_width + required_w_after
	//  but it may be a bit less, when required_w_after = 0 and stride_width >1
	//
	int32_t required_w_total = filt_width + (out_width-1)*stride_width;

	//logmsg(nn,1, "h: %ld(%ld)%ld  w %ld(%ld)%ld\n", required_h_before, out_height, required_h_after,
	//		required_w_before, out_width, required_w_after);
	/* Set up pointers to bulk data */
	uint8_t *in = in_tensor->data;
	//uint8_t *filt = filt_tensor->data;
	//uint8_t *bias = bias_tensor->data;
	uint8_t *out = out_tensor->data;
	//int32_t *bias32_ptr = bias_tensor->data;

	/* Calculate conversion ratio from bias to product space */
	//float bias_to_prod_ratio = (bias_level_size / prod_level_size);
	/* What is the value of the output minimum in the product space? */
	/* We need to add it to the products to move the smallest valid value to zero */
	//float min_out_prod_offset = -info->out_minval / prod_level_size;

	/* 
	 * Set output padding values to sensible defaults.
	 * FUTURE WORK: find optimized padding values instead of default ones
	 */
	int32_t out_right_pad = ((-out_width) & 3);
	int32_t out_left_pad = 4;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
//	int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;

	//int32_t filt_depth_total = in_depth_total;
	//int32_t filt_batches_total = out_depth_total;

	/* How much storage for each frame in the batch? */
	int32_t input_batch_size = in_height_total * in_width_total * in_depth_total;
	//int32_t output_batch_size = out_height_total * out_height_total * out_depth_total;

	/* 
	 * If we are striding, we need to ensure that the total left padding is compatible with 
	 * the filter width / stride combination, otherwise we won't be hitting the right pixels exactly.
	 */

	/* Where does our output data start? */
	uint8_t *out_data_start = out + out_top_pad * out_width_total * out_depth_total;

	int num_out_slices = (out_depth+31)/32;

	/* Grab some precomputed weight size information */
	//int n_weight_batches = info->n_weight_batches;
	int weight_batch_size = info->weight_batch_size;

	//int i;
	//int d,d2,b,t;

	//uint32_t maxval_leading_zeros;

	//int32_t input_skip_lines;

	struct workitem work = {0}; // klocwork
	//struct workitem waitwork = work;
	//struct workitem copywork = work;
	struct workitem startwork = work;
	//int workidx = 0;
	//int32_t tmpval32;


	logmsg(nn,1,"Supernode %x: Recalculating Strategy...",self->node_id);
	//logmsg(nn,0,"Weight batch size: %d. Per %d: %d",nn->vtcm_size,weight_batch_size,nn->vtcm_size/weight_batch_size);

	/* Some sanity checks... */
	if (num_out_slices <= 0) return errlog(nn,"no out depth to iterate?");
	if (((in_width_total) % 4) != 0) return errlog(nn,"width fail");
	if ((in_depth_total % 32) != 0) return errlog(nn,"depth fail");
	if (((out_width_total) % 4) != 0) return errlog(nn,"width math fail");
	if ((out_depth_total % 32) != 0) return errlog(nn,"depth math fail");
	if (in_depth_before_pad != 0) return errlog(nn,"depth before pad not supported");


	// check all the padding requirements:
	//   (1) required_w_before  <= in_left_pad
	//   (2) required_w_after  <= in_right_pad+in_left_pad
	//     (because the  right side can use the left-padding of the following row).
	//      Condition (2) can fail when (1) passes, if you have an even-width filter and the right-side
	//      padding is 0.
	//  NOTE: if required_w_after > in_right_pad, it means that right-side
	//  output pixels are using left-padding on the next line, and the bottom right corner
	//  will need one extra row of bottom padding to support it.
	//
	// and
	//  (3) required_h_before <= in_top_pad
	//  (4) required_h_after + extra_bottom_row <= in_bottom_pad
	//   (where 'extra_bottom_row' is 1 if the right edge overlap mentioned above is needed.


	/* EJP: hopefully handle arbitrary in left pad... */
	if (in_left_pad != 4) logmsg(nn,2,"Caution: left pad == %d",in_left_pad);

	// 1 if an extra bottom row is needed to supply RHS padding in the last input row.
	int32_t extra_bottom_row = 0;
	if(required_w_after > in_right_pad){
		extra_bottom_row = 1;
		// However when the vertical stride > 1, and the required_h_after = 0, it's possible
		// that we don't really need an extra bottom row, since in some cases the
		// last input row is itself not used.
		// Check for that, to avoid unnecessary requirement for extra padding.
		//
		if( stride_height > 1 && required_h_after == 0){
			// number of input rows needed to generate output (including padding)
			int h_needed = filt_height + (out_height-1)*stride_height;
			if( h_needed < required_h_before + in_height ){
				// we have at least one superfluous bottom row, so don't need
				// to artificially add padding.
				extra_bottom_row = 0;
			}
		}
	}

	int32_t required_h_after_augmented = required_h_after + extra_bottom_row;

	if( required_w_before > in_left_pad || required_w_after > (in_right_pad + in_left_pad)){
		return errlog(nn, "insufficient W padding: %d(%d)%d, filter needs %d()%d\n",
				(int)in_left_pad, (int)in_width, (int)in_right_pad, (int)required_w_before, (int)required_w_after);
	}
	if(  required_h_before > in_top_pad || required_h_after_augmented > in_bottom_pad ){
		return errlog(nn, "insufficient H padding: %d(%d)%d, filter needs %d()%d+%d\n",
				(int)in_top_pad, (int)in_height, (int)in_bottom_pad,
				(int)required_h_before, (int)required_h_after,(int)extra_bottom_row);
	}


	logmsg(nn,2,"in: %p h/w/d: %d/%d/%d total h/w/d: %d/%d/%d first valid row: %p first valid data: %p",
		in,
		in_height,in_width,in_depth,
		in_height_total,in_width_total,in_depth_total,
		in+(in_depth_total*in_width_total*in_top_pad),
		in+(in_depth_total*in_width_total*in_top_pad)+(in_depth_total*in_left_pad));

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	/*
	 * Update info / supernode info values
	 * These values are stored in structures for use during normal execution
	 */
	{
		struct tensor_addressing tin = tensor_addressing_d32( in_tensor);
		info->in_next_d32 = tin.d32_stride;
		info->in_next_row = tin.height_stride;
		info->in_next_batch = tin.batch_stride;
		info->in_width = tin.d32_stride / 32u;
		info->in_depth = 32*tin.nd32_total;
	}

	info->out_width_processed = out_width;

	{
		struct tensor_addressing tout = tensor_addressing_d32	( out_tensor);
		info->out_next_d32 = tout.d32_stride;
		info->out_next_row = tout.height_stride;
		info->out_width = tout.d32_stride /32u;				// includes width padding
		info->out_depth_total = 32*tout.nd32_total;			// includes all padding
	}
	info->out_depth_valid = out_depth + out_depth_after_pad;	// know how much we'll compute

	info->stride_height = stride_height;
	info->stride_width = stride_width;


	// input_base0, in_height:  this omits any top/bottom padding rows that we don't need

	info->input_base0 = in + (in_top_pad - required_h_before)*info->in_next_row;
	info->in_height = in_height + required_h_before + required_h_after;
	info->weights_base = info->weights;

	/* 
	 * If we are striding, we need to ensure that the total left padding is compatible with 
	 * the filter width / stride combination, otherwise we won't be hitting the right pixels exactly.
	 */
	/* If we did normal convolution, how many junk values would we have on the left? */
	info->out_left_junk = (in_left_pad - required_w_before) / info->stride_width;
	/* Do we need to skip an element on the input so that we stride to the right starting point? (covers even/odd leftover from out_left_junk) */
	info->in_left_skip = in_left_pad - (info->out_left_junk * info->stride_width + required_w_before);
	if (info->in_left_skip < 0
			|| (info->stride_width==2 && info->in_left_skip > 1) ){
		return errlog(nn,"wrong in left skip");
	}

	/* Is skip_col the same as in_left_skip? */
        //info->skip_col = ((out_width & 3) <= ((4-out_left_junk)&3) && (out_width & 3)!=0);
        info->skip_col = info->in_left_skip;

	info->out_height = out_tensor->shape.height;
	info->filt_width = filt_width;
	info->filt_height = filt_height;

	// Note: this may expand the output range
	if(  fill_info_minmax_basics(nn,self,info) != 0 )
		return -1;
	//fill_info_dim_basics(nn,self,info);
	logmsg(nn,2,"out_maxval=%f in_level_size=%f filt_level_size=%f prod_level_size=%f maxvalid ~= %d",
		info->out_maxval,
		info->prod_level_size/info->weights_level_size,
		info->weights_level_size,
		info->prod_level_size,
		info->max_valid_val);
	/*
	 * Recompute bias values
	 * We need to incorporate the input bias values, min offset, and gemsumb values into the bias buffer
	 * The bias buffer is added in the "product" (in_stepsize * filt_stepsize) number system
	 */
	int filt_offset = info->weights_offset;
	int input_offset = info->in_offset;

	int bias32 = (self->node_type == OP_Supernode_8x8p32to8_d32);
	int32_t bias_extra = 0;
        if (!info->use_v65 && filt_height==1 && filt_width==1 && ENABLE_FASTSUMA_1x1) bias_extra = info->in_depth*input_offset*filt_offset;
	logmsg(nn,2,"in_depth_total=%d input_offset=%d filt_offset=%d bias_extra=%d",in_depth_total,input_offset,filt_offset,bias_extra);
	if (fill_bias_buf(nn,self,info,bias32,bias_extra) != 0)
		return -1;

	/*
	 * Prepare output tensors
	 */
	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail max_size=%d @ %p",out_max->max_size,&out_max->max_size);
	}
	tensor_set_float(out_min,0,info->out_minval);
	tensor_set_float(out_max,0,info->out_maxval);

	/*
	 * Overcompute Reduction
	 * We may have too much input padding.  If we can avoid computing it, that's great!
	 * We should make this very robust so we can have truly arbitrary padding
	 */

//#ifndef HEXAGON_V66
#if 1
	/*
	 * In V60-V65, we should just be able to move the pointer to 
	 *   (b,-required_h_before,required_w_before,0)
	 * since there is no strict vector alignment requirement for the activation (for this node...)
	 */
	// maybe info->in_width = in_width + required_w_before + required_w_after;
	info->out_width = out_tensor->shape.width + out_tensor->format.width_pad[1];
	info->out_left_junk = 0;
	if (info->use_v65) {
		int naccs = 8;		// default, multiples of 8
		//if(((out_width+7)%8) < (2+((out_width+5)%6))) { //too general for now
#if 1
		if( info->out_width >= 11 && info->out_width <= 36){
			// use multiples of 6?
			//big gains for iv3
			// use 6 for 11,12,17,18,35,36
			// 9,10,33,34 would be good too but currently asm needs last group to be 5 or 6
			int div6 = (info->out_width+5)/6u;	// # of groups of 6
			int pad6 = div6*6 - info->out_width;	// amount to pad to reach mul of 6
			if(pad6 <= 1 && (info->out_width <=18 || info->out_width >= 35)){
				naccs = 6;
				info->out_width = div6*6;
                                logmsg(nn,3," choosing 6 accs");
			}
		}
#endif
		info->num_accs = naccs;

		//v65 chops off what it doesn't want and puts it in a buffer
		info->in_left_skip = in_left_pad - required_w_before ; //1,2,3 or 4
		int edge = (in_left_pad - required_w_before) & ~3;
		info->input_base = tensor_location_bhw_d32(in_tensor,0,-required_h_before,edge-in_left_pad);
	} else {
		// this may not be vector aligned, but it is 32 aligned: it points to the first h-padding unit we need
		info->input_base = tensor_location_bhw_d32(in_tensor,0,-required_h_before,-required_w_before);
		info->in_left_skip = 0;
	}
	info->skip_col = info->in_left_skip;
	out_data_start = tensor_location_bhw_d32(out_tensor,0,0,0);
	logmsg(nn,2,"now out_width=%d out_left_junk=%d out_data_start=%p",info->out_width,info->out_left_junk,out_data_start);
//	info->suma_start = 3 + in_left_pad - required_w_before;
#else

	info->suma_start = (required_w_before & (stride_width >> 1)) + 3;

	/* If we don't require any left padding, we can skip it */
	//return errlog(nn,"FIXME: left padding");
	/* 
	 * EJP: move this (maybe) and refactor
	 * 
	 * Try and skip over input left padding if possible
	 *  - Adjust input pointer right by a vector
	 *  - Decrease in width
	 *  - Now we really need to recompute out_width / out_left_junk / in_left_skip / pointers
	 * So this maybe should be moved up.
	 * Perhaps a better strategy would be to reconsider as:
	 *  - For this architecture, how do I adjust the input pointers/values?
	 *  - Based on those values, how do I set output pointers/values?
	 */
	if ((required_w_before == 0) && (in_left_pad == 4)) {
		info->input_base += 128;
		info->in_width -= 4;
		/* FIXME: this is wrong if we have 1x1 filters with 2x stride or something maybe? */
		info->out_width -= 4;
		info->out_left_junk = 0;
		info->in_left_skip = 0;
		info->skip_col = info->in_left_skip;
		out_data_start += 128;
		info->suma_start += 4;
	}
#endif

	/* Skip initial padding lines */
	//input_skip_lines = in_top_pad-required_h_before;
	//info->input_base += info->in_next_row*input_skip_lines;


	/*
	 * Preparing the work list
	 * We create a work list, which is just a list of items for the main thread to do
	 * We need to put in all the things we need to do to execute the node:
	 * * Padding zapping
	 * * GEMSUMA computation
	 * * Actual convolution
	 * * l2fetch / memcpy of values
	 * * Waiting on semaphores for things to finish
	 * The work list is currently executed by the main thread
	 * This means that vector work needs to be passed to nn_os_work_for_vector
	 */

	supernode_softreset_work_items(self,nn,info);

	/*
	 * Set up work items
	 */

#if 0
	/* FIXME for v66: copy work has to actually work. */
	copywork.execute = supernode_execute_workitem_copy;
	copywork.info = info;
	copywork.self = self;
#endif

	//waitwork.execute = supernode_execute_workitem_join_some;

	work.info = info;
	work.self = self;
	//work.input = info->input_base;
	//work.output  = out;
	//work.weights = filtdst;
	work.stop_line = info->out_height;
	work.skip_lines = 1;
	//work.new_execute = supernode_execute_workitem_vector_dispatch;	// FIXME, pick the right function
	work.new_execute = supernode_execute_hvx_conv_work;	// FIXME, pick the right function

#if 0
        if(required_h_before < in_top_pad) work.zap_top_size = required_h_before + 1; //allow for integration
        else work.zap_top_size = required_h_before;

        work.zap_bot_size = required_h_after_augmented;
        work.zap_right = tensor_location_d32(in_tensor,0,0,in_width,0);
        work.zap_left = tensor_location_d32(in_tensor,0,0,-in_left_pad,0);
        work.zap_left_size = in_left_pad*32;
        work.zap_right_size = in_right_pad*32;  // DJH just keep it clean
        work.zap_height = in_height;

        /*
         * If we have padding zap work to do, add it to the list
         * Some extra zapping happens due to getting GEMSUMA to work, perhaps it could be optimized
         * FIXME: maybe just set zap_right and zap_right_size and let it be on
         * the next row if enough left pad exists.
         */
        if ((out_right_pad > 0) && (work.zap_left_size == 0)) {
                if (out_right_pad > in_left_pad) return errlog(nn,"oops, not enough left pad");
                /* EJP: FIXME: this probably doesn't work if zap_right_size goes byond the vector size */
        }
#else
	startwork.self = self;
	startwork.info = info;
	// top/bottom zapping based on entire rows (all padding) so they use input_base0.
	startwork.zap_top = (uint8_t *)info->input_base0;
	startwork.zap_bot = (uint8_t *)info->input_base0
		+ (required_h_before + in_height)*info->in_next_row;

	startwork.zap_top_size = info->in_next_row * required_h_before;
	startwork.zap_bot_size = info->in_next_row * required_h_after_augmented;

	uint8_t  * z0 = tensor_location_bhw_d32(in_tensor,0,0,0);	// first real pixel on first row
	startwork.zap_left = z0 - 32*in_left_pad;					// start of left padding on first row
	startwork.zap_right = z0 + 32*in_width;					// start of right padding on first row
	// do not zap beyond the right padding
	// EJP: are there other cases where we need to zap right pad?
	startwork.zap_right_size = Q6_R_min_RR(in_right_pad,required_w_after);
	startwork.zap_rl_depths = in_depth_total / 32;

	// zap all of the left padding if
	// (a) any left padding is needed, or
	// (b) the req. for right padding exceeds the actual right padding (so left is needed for that)
	if (required_w_before != 0 || required_w_after > in_right_pad) {
		startwork.zap_left_size = in_left_pad;
	} else {
		startwork.zap_left_size = 0;
	}
	startwork.zap_height = in_height;
	startwork.zap_value = input_offset;
	logmsg(nn,2,"padding=%d required_w_before: %d required_w_after: %d required_h_before: %d required_h_after: %d",self->padding,required_w_before,required_w_after,required_h_before,required_h_after);

	/*
	 * If we have padding zap work to do, add it to the list
	 * Some extra zapping happens due to getting GEMSUMA to work, perhaps it could be optimized
	 * FIXME: maybe just set zap_right and zap_right_size and let it be on
	 * the next row if enough left pad exists.
	 */
	/* GLS: leaving this stuff in; it may be needed to avoid contaminating min/max in output width padding */
	if ((out_right_pad > 0) && (startwork.zap_left_size == 0)) {
		//if (out_right_pad > in_left_pad) return errlog(nn,"oops, not enough left pad");
		/* EJP: FIXME: this probably doesn't work if zap_right_size goes byond the vector size */
		startwork.zap_left_size = in_left_pad;
		startwork.zap_right_size = in_right_pad;
	}
//#ifndef V66
#if 1
	/* EJP: this is extra padding for integral/suma, but maybe it's no longer needed? */
	if (startwork.zap_left_size == 0) {
		startwork.zap_left_size = in_left_pad;
	}
	if(0)if (startwork.zap_top_size == 0 && in_top_pad > 0) {
		startwork.zap_top_size = info->in_next_row;
		startwork.zap_top = (uint8_t *)info->input_base0 - info->in_next_row;
	}
#endif
#endif
	work.zap_rl_depths = in_depth_total/32;
	work.zap_value = input_offset;

	/* 
	 * Slicing was a good first attempt, but needs refactoring.
	 * If all the weights fit in cache easily, load them all and do all the depths for a chunk of activation.
	 * Else, if the activation fits in cache easily, do a d32 output at a time
	 * Else, find some number of weights and some amount of activation that both fit
	 *   Traffic ~= activation_chunk_size * weight_slices + weight_chunk_size * activation_slices
	 *   ... but maybe not quite so simple.  Want to maximize use of locality
	 *   Maybe need outer_activation, outer_weights, inner_activation, inner_weights nested loops
	 * What's the right optimization technique?
	 *  Weights have to be at least 32*h*w*in_depth
	 *  Probably need at least a few rows of activation
	 *  Weight chunks outer, activation inner --> more activation BW
	 *  Activation chunks outer, weight chunks inner --> more weight BW
	 *  Looking at the literature, it looks like it's a non-trivial problem?
	 * If we have a huge activation, we want to favor activation locality.
	 * If we have huge weights, we want to favor weight locality.
	 * So... our constraints are:
	 *  * Minimum effective weights size (d32 chunk size), granularity
	 *  * Minimum effective activation size (currently some rows, but we could subdivide further...), granularity
	 *  * Weights + Activation + prefetching room < cache_size
	 *  * Either Weights or Activations needs to have multiple chunks for multithreading!
	 * Given those things, try and find something that matches the activation/weight size ratio.
	 *   Start with minimum amounts of activation/weights
	 *   Add to activation/weight size to try and get closer to the ratio, see if it's still cache-friendly
	 *   If we can't add the thing that makes the ratio better, try and add a little more of the other one
	 * Once we have activation/weight chunk sizes, how should we proceed to the next chunk of work?
	 *   Probably the one that needs less bandwidth immediately.
	 *   We will need to iterate over all the weight and activation chunks, but the instantaneous BW requirements are lower if going to the next small thing.
	 * Need to be aware of threading...
	 *   Let's start out with always threading by dividing the activation, easier to handle suma/integral/zapping
	 * So.......
	 * for (outer_weights) {
	 *   for (outer_activations) {
	 *     for (inner_weights) {
	 *       for (inner_activations) {
	 *       }
	 *     }
	 *   }
	 * }
	 * If weights_fit: outer_weights = 1, outer_activations = n_act_chunks, inner_weights = all, inner_activations=chunksize/threads
	 * else if activations_fit: outer_weights = 1, outer_activations = 1, inner_weights = all, inner_activations=all/threads
	 * else outer_weights = n_weight_chunks, outer_activations = n_act_chunks, inner_weights = weight_chunksize, inner_activations=chunksize/threads
	 * 
	 * TBD: should we consider batches just part of the activation size?
	 *   * Probably, but it's hard to index and iterate over. 
	 *   * Maybe split above loop activations into outer_batches, outer_rows
	 */

	/* Determine how we want to slice the work */
        //int32_t weight_batch_size0 = (info->out_depth_slice == 64 && info->out_depth % 64) ? weight_batch_size : weight_batch_size*2;
	//int32_t batch_slice_factor;
	//int32_t height_slice_factor;
	//int32_t weight_slice_factor;
	int32_t inner_act_batches;
	int32_t inner_act_rows;
	int32_t inner_weight_chunks;
	//luc
//	logmsg(nn,0,"opname=%s name=%s ",info_id2opname(self->node_id),info_id2name(self->node_id));

	slice_for_cache(
		self->node_id,
		nn,
		out_batches,
		in_depth_total/32,
		info->in_width,
		//info->in_height,
                info->out_width,
		info->out_height,
		info->stride_height,
		info->filt_width,
		info->filt_height,
		weight_batch_size,
		num_out_slices,
		&inner_act_batches,
                &inner_act_rows,
		&inner_weight_chunks);

	work.num_lines = inner_act_rows;
	inner_act_rows = (out_height+NUM_THREADS-1)/(NUM_THREADS);

	/* Ignore batch / weight slice factor for now ... */
	int32_t outer_act_batches = (out_batches + inner_act_batches - 1)/inner_act_batches;
	int32_t outer_act_iters = (out_height + inner_act_rows - 1)/inner_act_rows;
	int32_t outer_weight_chunks = (num_out_slices + inner_weight_chunks - 1) / inner_weight_chunks;

	// NOTE: outer_act_iters <= NUM_THREADS.

	logmsg(nn,1,"batch/row/weight chks: inner=%d,%d,%d outer=%d,%d,%d out_height=%d out_depth_chunks=%d",
		inner_act_batches,work.num_lines,inner_weight_chunks,
		outer_act_batches,outer_act_iters,outer_weight_chunks,
		out_height,
		num_out_slices);

    /*-------------------------------------------------------------*/
    /*  Setup parameters and allocate scratch for SUMA computation */
    /*-------------------------------------------------------------*/
    int32_t scratch_size;		// in int32 units
    int32_t suma_buf_rowstride;	// in int32 units

    // sumabuf is allocated in scratch, for all batches
    int32_t *sumabuf_batch0 = NULL;
    int sumabuf_batch_stride= 0;	// in int32's

    if (!info->use_v65) {
        if(info->filt_width ==1 && info->filt_height == 1 && ENABLE_FASTSUMA_1x1) {
        	// each row of the suma has
        	//   (a) one slot for each 'unused' left-padding element (if any); 0..3
        	//   (b) one slot for each input element which is used by the convolution
        	//       including req. left & right (i.e. required_w_total )
        	//   (c) padding to a multiple of 32.
        	//  suma_start points to section (b).
        	//
            info->suma_start = (in_left_pad-required_w_before)&3; 
            suma_buf_rowstride = roundup(info->suma_start+required_w_total,32);
            scratch_size = 32;	 // not actually used; minimal size
        } else {
        	// each row of the integral buffer will have (in i32 slots):
        	//   (a) 8 zeros
        	//   (b) one slot for each 'unused' left-padding element (if any); 0..3
        	//   (c) one slot for each input element which is used by the convolution
        	//       including req. left & right (i.e. required_w_total )
        	//   (d) padding up to a multiple of 32.
        	// The 'suma_start' points to the first element before section (c)
        	// 'suma buf' has the same width, and the the first 'proper' output
        	// will be at position 'suma_start'.
        	//
            info->suma_start = 7 + ((in_left_pad-required_w_before)&3);
            suma_buf_rowstride = roundup(info->suma_start + 1 + required_w_total,32);
            info->suma_width = suma_buf_rowstride;
            // this is 4 rows for 'scratch_128xW' plus enough rows for the integral buffer
            int scratch_rows = 4 + (work.num_lines-1)*info->stride_height+info->filt_height+1;
            scratch_size = suma_buf_rowstride*scratch_rows + 32;
        }

        // allocate suma_scratch buffers,x NUM_THREADS
        uint32_t sumatmp_size =  scratch_size * sizeof(int32_t);
        void * tbufp = nn_scratch_alloc( nn, sumatmp_size*NUM_THREADS);
        bufpool_init( &info->bufpool_suma_scratch, NUM_THREADS, tbufp, sumatmp_size );

        info->next_suma_off = suma_buf_rowstride * sizeof(int32_t);
        sumabuf_batch_stride = suma_buf_rowstride*info->out_height;	// size of 1 plane in int32's
        int32_t suma_buf_size = sumabuf_batch_stride * in_batches;	// size of whole buffer in int32's

        sumabuf_batch0 = nn_scratch_alloc(nn,  suma_buf_size *sizeof(int32_t));
        if( sumabuf_batch0 == NULL){
        	return errlog( nn, "failed to get %d bytes for sumabuf", (int)(suma_buf_size *sizeof(int32_t)));
        }
    }

	int32_t semaphore_count = 0;
	int ow,ob,or,ib;


	int curr_suma_progress = 0;
	int batchstart_idx = 0;
	//if (hlines_per_slice <= NUM_THREADS) return errlog(nn,"OOPS: chopped too fine");
	for (ob = 0; ob < outer_act_batches; ob++) {
	for (ib = 0; ib < inner_act_batches; ib++) {
		int b = ob * inner_act_batches + ib;
		if (b >= in_batches)
			continue;
		work.suma_buf = sumabuf_batch0 + b*sumabuf_batch_stride;

#if 0
		/* l2fetch first weight chunk */
		logmsg(nn,1,"adding l2fetch: %p %d %d",info->weights,weight_batch_size,inner_weight_chunks);
// Add workitem: weights
// EJP: merge into startwork
		supernode_weights(self,nn,info,b!=0, /*don't wait for previous in batch 0*/
				info->weights,weight_batch_size,info->use_v65 ? inner_weight_chunks : 1, outer_act_iters);
#endif
		/* Zap padding is back */
		//supernode_add_padding_zap(self,nn,info,zapwork,b*input_batch_size,required_h_before,required_w_before);
		curr_suma_progress ++;	// advance per batch
// Add workitem: padzap
// EJP: make startwork
		info->startup_info.batch_start_offset = 0;
		if (info->use_v65) {
			uint32_t weights_total = inner_weight_chunks * weight_batch_size;
			if (unlikely(weights_total > (nn->vtcm_size - VTCM_CIRCBUF_SIZE))) return errlog(nn,"oops: v65 selected but weights too big");
			startwork.copy_in = info->weights;
			startwork.copy_out = nn->vtcm_ptr;
			startwork.copy_size = inner_weight_chunks * weight_batch_size;
			info->startup_info.wakeup_items = outer_act_iters;
			//if (b == 0) supernode_note_earlywork(nn,self,info,startwork.copy_out,startwork.copy_in,startwork.copy_size);
		} else {
			/* For V60, we need to finish the gemsuma before the rest can continue */
			startwork.pf_inp = info->weights;
			startwork.pf_width = startwork.pf_stride = weight_batch_size/32;
			/* EJP: V60 code prefetches next chunk of weights automatically, just PF first chunk */
			//startwork.pf_height = 1 * 32;
			startwork.pf_height = inner_weight_chunks * 32;
			//info->startup_info.wakeup_items = outer_act_iters * outer_weight_chunks;
			info->startup_info.wakeup_items = outer_act_iters;
		}
		startwork.join_iters = outer_weight_chunks;
	        if (b > 0) info->work_items[batchstart_idx].next_startup_offset = info->n_work_items - batchstart_idx - 1;
		int NOW_BATCHES=1;
		batchstart_idx = supernode_add_batch_startup(self,nn,info,startwork,b*input_batch_size,required_h_before,required_w_before,NOW_BATCHES);

	for (ow = 0; ow < outer_weight_chunks; ow++) {

            int last_chunk_in_batch = (ow == (outer_weight_chunks-1));
	    int start_weights = ow * inner_weight_chunks;
	    int now_chunks = Q6_R_min_RR(num_out_slices - start_weights,inner_weight_chunks);
	    int32_t next_weight_chunks = inner_weight_chunks;
	    int32_t max_next_weight_chunks = num_out_slices-start_weights-now_chunks;
	    next_weight_chunks = Q6_R_min_RR(max_next_weight_chunks,next_weight_chunks);
	    for (or = 0; or < outer_act_iters; or++) {
               // work.suma_scratch = scratch[n_scratch++%2];

		int pf_outer_act = (or == (outer_act_iters-1));
		//int pf_outer_act = (or == 0);
		int needs_next_outer_weights = pf_outer_act && (ow != (outer_weight_chunks-1));

		int start_row = or * inner_act_rows;
		int n_rows = Q6_R_min_RR(out_height-start_row,inner_act_rows);


		/* FILL OUT NORMAL WORK INFORMATION */
		work.need_initialize_suma = (ow ==0);
		//work.suma_progress_need = curr_suma_progress;
		//work.suma_progress_index = or;
		const uint8_t *filtsrc = info->weights_base + start_weights*weight_batch_size;
		const uint8_t *filtdst = supernode_filtbuf_location(nn,info,ow,filtsrc,now_chunks*weight_batch_size);
		work.weights = filtdst;
		work.weight_chunks = now_chunks;
		work.input = info->input_base+b*input_batch_size;
		work.output = tensor_location_d32(out_tensor,b,0,0,start_weights*32);
		//work.output = out_data_start+b*output_batch_size+start_weights*info->out_next_d32;
		work.biases = info->biasbuf + start_weights*32;
		work.recip = info->recip + start_weights*32;
		work.start_line = start_row;
		work.stop_line = work.start_line + n_rows;
		//work.do_conv = &info->semaphores[1];			// sem to wait for, before starting conv
		work.do_conv = NULL;
		work.conv_done = &info->semaphores[2];			// sem to post, after conv if not last_chunk (v65 only)
//		work.copy_done = &info->semaphores[3];
		work.start_chunk = start_weights;
		work.last_chunk = last_chunk_in_batch && ( b == in_batches-1);

		if (needs_next_outer_weights) {
			work.pf_inp = filtsrc + now_chunks*weight_batch_size;
			work.pf_width = weight_batch_size;
			work.pf_stride = work.pf_width;
			work.pf_height = next_weight_chunks;

			logmsg(nn,1,"or=%d ow=%d/%d set up weight pf ptr=%p width=%d height=%d",
					or,ow,outer_weight_chunks,work.pf_inp,work.pf_width,work.pf_height);
		} else {
			work.pf_inp = NULL;
			work.pf_width = 0;
			work.pf_stride = 0;
			work.pf_height = 0;
			logmsg(nn,1,"or=%d ow=%d/%d no pf",or,ow,outer_weight_chunks);
		}
		work.donesem = &info->semaphores[0];
		semaphore_count++;
// Add workitem: work
		supernode_add_work_item(self,nn,info,work);
	    }//or
	    memset(&info->conv_slices[start_weights],0,sizeof(info->conv_slices[ow]));
            if ((ow == 0) && (!info->use_v65)) {
                /* For V60 code, wake up rest of items once first batch is done */
                /* This ensures ordering of gemsuma with other work items */
                info->conv_slices[ow].wakeup_items = (outer_weight_chunks-1)*outer_act_iters;
                info->conv_slices[ow].batch_start_offset = (ow+1)*outer_act_iters;
            }
	    if (!last_chunk_in_batch && info->use_v65) {
		info->conv_slices[start_weights].copy_in = work.pf_inp;
		info->conv_slices[start_weights].copy_out = nn->vtcm_ptr;
		info->conv_slices[start_weights].copy_size = next_weight_chunks * weight_batch_size;
		info->conv_slices[start_weights].batch_start_offset = (ow+1)*outer_act_iters;
		info->conv_slices[start_weights].wakeup_items = outer_act_iters;
		logmsg(nn,1,"v65: Setting up copy %p-->%p %d bytes, %d items @ %d",
			info->conv_slices[start_weights].copy_in,
			info->conv_slices[start_weights].copy_out,
			info->conv_slices[start_weights].copy_size,
			info->conv_slices[start_weights].wakeup_items,
			info->conv_slices[start_weights].batch_start_offset);
	    }
            nn_checkpoint_init(&info->conv_slices[start_weights].checkpoint,outer_act_iters,note_chunk_checkpoint_arrival,(void *)&info->conv_slices[start_weights]);
#if 0
	if(!last_chunk_in_batch && info->use_v65)
// Add workitem: weights
// Needs to be merged into work checkpoint work
		    supernode_weights(self,nn,info,1, /*wait for previous*/
				work.pf_inp, work.pf_width,work.pf_height, outer_act_iters);
#endif
        }//ow
        }//ib
        }//ob
	//logmsg(nn,0,"semaphore_count / join_iters=%d",semaphore_count);

	// 'semaphore_count is the # of convolves in the whole operation; number
	// of donesem posts; this is used in the final join.
	// However, for v65, the work thread only posts done_sem in units where last_chunk is set.
	// (earlier ones are joined by the weight-copy ops).

	//if( info->use_v65) semaphore_count = outer_act_iters;
	//waitwork.join_iters = semaphore_count;
	//waitwork.donesem = &info->semaphores[0];
// Add workitem: wait for shutdown
	//supernode_add_work_item(self,nn,info,waitwork);
#if 0
	/*
	 * If we have padding zap work to do, add it to the list
	 * Some extra zapping happens due to getting GEMSUMA to work, perhaps it could be optimized
	 * FIXME: maybe just set zap_right and zap_right_size and let it be on
	 * the next row if enough left pad exists.
	 */
		if ((out_right_pad > 0) && (work.zap_left_size == 0)) {
			if (out_right_pad > in_left_pad) return errlog(nn,"oops, not enogh left pad");
			work.zap_left = tensor_location_d32(in_tensor,b,start_row-required_h_before,-in_left_pad,0);
			work.zap_left_size = in_left_pad;
		}


/* EJP FIXME NOW: Set up work item for inner rows + inner batches */
		
		//l2fetch_inner_activation;
		if (ow == 0) {
			const uint8_t *filtsrc = info->weights_base + ow*weight_batch_size;
			filtdst = supernode_filtbuf_location(info,ow,filtsrc);
			logmsg(nn,0,"copy first inner weight chunk... ow=%d ob=%d or=%d fsrc=%p fdst=%p",
				ow,ob,or,filtsrc,filtdst);
			// copy_first_inner_weight_chunk;
		}
		for (iw = 0; iw < inner_weight_chunks; iw++) {
			if ((iw != (inner_weight_chunks-1)) || needs_next_outer_weights) {
				//logmsg(nn,0,"copy next inner weight chunk... ow=%d ob=%d or=%d iw=%d",ow,ob,or,iw);
				//copy_next_weight_chunk
			}
		for (ib = 0; ib < inner_act_batches; ib++) {
		for (ir = 0; ir < inner_act_rows; ir++) {
			//logmsg(nn,0,"maybe zap: ow=%d ob=%d or=%d iw=%d ib=%d ir=%d",ow,ob,or,iw,ib,ir);
			//logmsg(nn,0,"work item: ow=%d ob=%d or=%d iw=%d ib=%d ir=%d",ow,ob,or,iw,ib,ir);
			
			//if (weight_slice == 0) zap;
			//else disable_zap;
			//setup rest of work
			//supernode_add_work_item(self,nn,info,work);
		}
		}
		}
	}
	}
	}
#endif
#if 0
	for (hslice = 0; hslice < height_slice_factor; hslice++) {
		const uint8_t *batch_input = info->input_base + b*input_batch_size;
		/*
		 * For the first slice, we want to zap the padding and l2fetch the initial data
		 */
		if (hslice == 0) supernode_add_l2fetch(
				self,
				nn,
				info,
				batch_input,
				in_width_total*in_depth_total,
				in_hlines_per_slice+required_h_before+required_h_after_augmented);
		if (hslice == 0) supernode_add_padding_zap(
				self,
				nn,
				info,
				zapwork,
				b*input_batch_size,
				required_h_before,
				required_w_before);
		if (hslice == 0) work.suma_buf = supernode_add_suma(self,nn,info,batch_input);
		for (d = 0; d < num_out_slices; d++) {
			//const uint8_t *filtsrc = info->weights_base + workidx*weight_batch_size;
			const uint8_t *filtsrc = info->weights_base + d*weight_batch_size;
			int workidx_mod_batches = workidx % n_weight_batches;
			filtdst = supernode_filtbuf_location(info,workidx_mod_batches, filtsrc);
			copywork.copy_in = filtsrc;
			copywork.copy_out = filtdst;
			copywork.copy_size = weight_batch_size;
			supernode_add_work_item(self,nn,info,copywork);
			/* FIXME: padding zap: 
				Always zap right pad
				If required_w_before, zap left pad (4 w's)
				If required_h_before, zap top pad
				If required_h_after_augmented, zap bottom pad
			*/
			/* as we near the end of this activation slice, get the next one */
			if ((d == num_out_slices-1) && ((hslice+1) < height_slice_factor)) supernode_add_l2fetch(
				self,
				nn,
				info,
				batch_input + in_width_total*in_depth_total*in_hlines_per_slice*(hslice+1),
				in_width_total*in_depth_total,
				in_hlines_per_slice+required_h_before+required_h_after_augmented);
			work.weights = filtdst;
			work.donesem = &info->semaphores[workidx_mod_batches];

			work.input = batch_input;
			work.output = out_data_start + b*output_batch_size + d*info->out_next_d32;
			work.biases = info->biasbuf + d*32;
			/* EJP: FIXME: broken for batches, really */
			for (t = 0; t < NUM_THREADS; t++) {
				work.minmax_buf = info->minmax_buf+(NUM_THREADS*workidx_mod_batches+t)*64;
				work.start_line = hslice*out_hlines_per_slice+t;
				work.stop_line = work.start_line + out_hlines_per_slice-t;
				if (work.stop_line > info->out_height) {
					work.stop_line = info->out_height;
				}
				supernode_add_work_item(self,nn,info,work);
			}
			waitwork.donesem = &info->semaphores[(workidx+1) % n_weight_batches];
			if (workidx >= (n_weight_batches-1)) supernode_add_work_item(self,nn,info,waitwork);
			if (workidx >= (n_weight_batches-1)) logmsg(nn,2,"add wait %d for %d",(workidx+1)%n_weight_batches,workidx);
			workidx++;
		}
	}
	}
	for (i = workidx - (n_weight_batches-1); i < workidx; i++) {
		if (i < 0) continue;
		logmsg(nn,2,"end: add wait %d",i%n_weight_batches);
		waitwork.donesem = &info->semaphores[i%n_weight_batches];
		supernode_add_work_item(self,nn,info,waitwork);
	}
#endif
	/* Add work to check the output min/max and see if we need to adjust and try again */
	//work.execute = supernode_execute_workitem_check_for_retry;
	//supernode_add_work_item(self,nn,info,work);

	/*
	 * Sometimes we want to collect some statistics...
	 */
	if (0) supernode_statistics(nn,info,self);

	/*
	 * We've calculated the strategy, mark that the work is done. Hopefully it sticks!
	 */
	supernode_compile_worklist(nn,info,self);
	info->prepared_vtcm_addr = nn->vtcm_ptr;
	info->needs_retry = 0;
	info->strategy_valid = 1;
	return 0;
}


/*
 * What do we have to do before calling the work function?
 *
 * Some things are cacheable:
 * * Biasbuf number range conversion
 * * Adding min_out value (converted) into biasbuf
 * * Adding gemsumb values (if needed) into biasbuf
 * * Adding in_offset * weight_offset * N into biasbuf (if needed)
 * * Partitioning scheme
 * * Reciprocal for range conversion
 * * Min and Max output information
 * * Strategy for partitioning
 * * Work function
 * * output shape/padding
 *
 * * Ensure that cached value is still accurate given these inputs (if not, recalc)
 * * Add work_for_vector items
 * * Wait for workers to complete
 * * Any fixup / postproc 
 * 
 * * The work function is passed to work_for_vector.  It can be different for different architectures.
 * * The strategy for partitioning could be a set of functions.
 */
int supernode_recalculate_strategy_v66(struct nn_node *self, struct nn_graph *nn)
{
	/* Pull out the inputs we need */
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	//const struct tensor *min_in_tensor = self->inputs[2];
	//const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	//const struct tensor *bias_tensor = self->inputs[7];
	//const struct tensor *bias_min_tensor = self->inputs[8];
	//const struct tensor *bias_max_tensor = self->inputs[9];
	/* Find the output tensors */
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	/* Structures with auxillary information */
	struct supernode_info_new *info = self->opaque;

	info->in_shape = in_tensor->shape;
	/* 
	 * Find the dimensions of the input data, 
	 * both dimensions of data as well as padding
	 */
	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_left_pad = in_tensor->format.width_pad[0];
	int32_t in_right_pad = in_tensor->format.width_pad[1];
	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];
	int32_t in_top_pad = in_tensor->format.height_pad[0];
	int32_t in_bottom_pad = in_tensor->format.height_pad[1];
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;

        info->in_depth_after = in_depth_after_pad;

	/* Find the dimensions of the filter weights.  filt_batches == out_depth */
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	//int32_t filt_depth = filt_tensor->shape.filt_depth;

	/* Find the stride values */
	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	/* Calculate output dimensions */
	int32_t out_batches = in_batches;
	int32_t out_depth = filt_batches;

        int32_t required_w_before, required_w_after;
        int32_t required_h_before, required_h_after;

        /* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */

        int32_t out_width = nn_pad_compute_outsize_and_pad(in_width,filt_width,stride_width,self->padding,
                        & required_w_before, & required_w_after);
        int32_t out_height = nn_pad_compute_outsize_and_pad(in_height,filt_height,stride_height,self->padding,
                & required_h_before, & required_h_after);

	/* Set up pointers to bulk data */
	uint8_t *in = in_tensor->data;

	/* 
	 * Set output padding values to sensible defaults.
	 * FUTURE WORK: find optimized padding values instead of default ones
	 */
	int32_t out_right_pad = ((-out_width) & 3);
	int32_t out_left_pad = 4;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
//	int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;

	//int32_t filt_batches_total = out_depth_total;
	//int32_t filt_depth_total = in_depth_total;

	/* How much storage for each frame in the batch? */
	int32_t input_batch_size = in_height_total * in_width_total * in_depth_total;
	//int32_t output_batch_size = out_height_total * out_height_total * out_depth_total;

	int num_out_slices = (out_depth+31)/32;
	int num_in_slices = (in_depth+31)/32;

	/* Grab some precomputed weight size information */
	//int n_weight_batches = info->n_weight_batches;
	int weight_batch_size = info->weight_batch_size;

	struct workitem work = {0};
	//struct workitem waitwork = work;
	struct workitem startwork = work;
	//int workidx = 0;

	logmsg(nn,1,"Supernode %x: Recalculating 66 Strategy...",self->node_id);
	//logmsg(nn,0,"Weight batch size: %d. Per %d: %d",nn->vtcm_size,weight_batch_size,nn->vtcm_size/weight_batch_size);

	/* Some sanity checks... */
	if (num_out_slices <= 0) return errlog(nn,"no out depth to iterate?");
	if (((in_width_total) % 4) != 0) return errlog(nn,"width fail");
	if ((in_depth_total % 32) != 0) return errlog(nn,"depth fail");
	if (((out_width_total) % 4) != 0) return errlog(nn,"width math fail");
	if ((out_depth_total % 32) != 0) return errlog(nn,"depth math fail");
	if (in_depth_before_pad != 0) return errlog(nn,"depth before pad not supported");

	/* EJP: hopefully handle arbitrary in left pad... */
	if (in_left_pad != 4) logmsg(nn,2,"Caution: left pad == %d",in_left_pad);
	if (required_w_before > in_left_pad) {
		//logmsg(nn,0,"in_left_pad: %d layout: %d",in_left_pad,in_tensor->format.layout);
		return errlog(nn,"FIXME: insufficient left pad");
	}
	if (required_w_after > (in_right_pad + in_left_pad)) {
		return errlog(nn,"FIXME: insufficient right pad? Strange...");
	}
	if (required_h_before > in_top_pad) return errlog(nn,"top pad");
	if (required_h_after > in_bottom_pad) return errlog(nn,"bot pad");


	logmsg(nn,2,"in: %p h/w/d: %d/%d/%d total h/w/d: %d/%d/%d first valid row: %p first valid data: %p = %p",
		in,
		in_height,in_width,in_depth,
		in_height_total,in_width_total,in_depth_total,
		in+(in_depth_total*in_width_total*in_top_pad),
	       in+(in_depth_total*in_width_total*in_top_pad)+(32*in_left_pad),
	       tensor_location_d32(in_tensor, 0,0,0,0));


	/* Where does our output data start? */
	int fail;
	if ((fail = tensor_out_prepare_padded_d32(out_tensor,
						  out_batches,
						  out_height,out_top_pad,out_bottom_pad,
						  out_width,out_left_pad,out_right_pad,
						  out_depth,out_depth_before_pad,out_depth_after_pad,
						  NN_TYPE_QUINT8)) != 0) {
		errlog(nn,"output tensor prep fail %d", fail);
		return fail;
	}

	/*
	 * Update info / supernode info values
	 * These values are stored in structures for use during normal execution
	 */
	{
		struct tensor_addressing tin = tensor_addressing_d32( in_tensor);
		info->in_next_d32 = tin.d32_stride;
		info->in_next_row = tin.height_stride;
		info->in_next_batch = tin.batch_stride;
		info->in_width = tin.d32_stride/32u;
		info->in_depth = 32 * tin.nd32;
	}
	{
		struct tensor_addressing tout = tensor_addressing_d32( out_tensor);
		info->out_next_d32 = tout.d32_stride;
		info->out_next_row = tout.height_stride;
		info->out_width = tout.d32_stride/32u;
		info->out_depth_valid = 32 * tout.nd32;
		info->out_depth_total = 32 * tout.nd32_total;
	}

	info->stride_height = stride_height;
	info->stride_width = stride_width;


	info->input_base0 = in + (in_top_pad - required_h_before)*info->in_next_row;
	info->in_height = in_height + required_h_before + required_h_after;
	info->weights_base = info->weights;

	/* 
	 * If we are striding, we need to ensure that the total left padding is compatible with 
	 * the filter width / stride combination, otherwise we won't be hitting the right pixels exactly.
	 */
	/* If we did normal convolution, how many junk values would we have on the left? */
	info->out_left_junk = (in_left_pad - required_w_before) / info->stride_width;
	/* Do we need to skip an element on the input so that we stride to the right starting point? (covers even/odd leftover from out_left_junk) */
	info->in_left_skip = in_left_pad - (info->out_left_junk * info->stride_width + required_w_before);
	if (info->in_left_skip < 0
			|| (info->stride_width==2 && info->in_left_skip > 1) ){
		return errlog(nn,"wrong in left skip");
	}

	/* Is skip_col the same as in_left_skip? */
        info->skip_col = info->in_left_skip;

	info->out_height = out_tensor->shape.height;
	info->filt_width = filt_width;
	info->filt_height = filt_height;

	// Note: this may expand the output range
	if (fill_info_minmax_basics(nn,self,info) != 0) return -1;
	//fill_info_dim_basics(nn,self,info);
	logmsg(nn,2,"out_maxval=%f in_level_size=%f filt_level_size=%f prod_level_size=%f maxvalid ~= %d",
		info->out_maxval,
		info->prod_level_size/info->weights_level_size,
		info->weights_level_size,
		info->prod_level_size,
		info->max_valid_val);
	/*
	 * Recompute bias values
	 * We need to incorporate the input bias values, min offset, and gemsumb values into the bias buffer
	 * The bias buffer is added in the "product" (in_stepsize * filt_stepsize) number system
	 */

	int bias32 = (self->node_type == OP_Supernode_8x8p32to8_d32);
	int32_t bias_extra = 0;
	int input_offset = info->in_offset;
	int filt_offset = info->weights_offset;

	logmsg(nn,2,"in_depth_total=%d input_offset=%d filt_offset=%d bias_extra=%d",in_depth_total,input_offset,filt_offset,bias_extra);
	if (fill_bias_buf(nn,self,info,bias32,bias_extra) != 0)
		return -1;

	/* 
	 * Recompute weights
	 * The weights need to be arranged so that they are in a depth32 compatible format.
	 * We keep note that we've rearranged the weights so we only do this once.
	 * For architectures that support signed weights, we convert weights to signed
	 * FIXME: maybe we should move this to check time instead of recalculation time
	 */
	//FIXME: RECALC / REARRANGE WEIGHTS

	/*
	 * Prepare output tensors
	 */
	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail");
	}
	tensor_set_float(out_min,0,info->out_minval);
	tensor_set_float(out_max,0,info->out_maxval);

	/*
	 * Overcompute Reduction
	 * We may have too much input padding.  If we can avoid computing it, that's great!
	 * We should make this very robust so we can have truly arbitrary padding
	 */

	/* If we don't require any left padding, we can skip it */
	//return errlog(nn,"FIXME: left padding");
	/* 
	 * EJP: move this (maybe) and refactor
	 * 
	 * Try and skip over input left padding if possible
	 *  - Adjust input pointer right by a vector
	 *  - Decrease in width
	 *  - Now we really need to recompute out_width / out_left_junk / in_left_skip / pointers
	 * So this maybe should be moved up.
	 * Perhaps a better strategy would be to reconsider as:
	 *  - For this architecture, how do I adjust the input pointers/values?
	 *  - Based on those values, how do I set output pointers/values?
	 */
	//HANA values
	logmsg(nn,2,"padding=%d required_w_before: %d required_w_after: %d required_h_before: %d required_h_after: %d",self->padding,required_w_before,required_w_after,required_h_before,required_h_after);
	logmsg(nn,2," v66 chosen");
	work.fix_overcompute = 0;
	if (required_w_before > 0 && stride_width == 1 && filt_width > 1 && (out_width & 3) == 0) {
		info->input_base = tensor_location_d32(in_tensor,0,-required_h_before,-required_w_before,0);
		info->skip_col = 0;
		info->out_width = out_width;
		work.fix_overcompute = 1;
		logmsg(nn,1," choose width overcompute ");
	} else if (required_w_before == 0 && in_left_pad == 4) {
		info->input_base = tensor_location_d32(in_tensor,0,-required_h_before,0,0);
		//info->input_base += 32*in_left_pad;
		info->in_width -= in_left_pad;
		/* FIXME: this is wrong if we have 1x1 filters with 2x stride or something maybe? */
		info->out_left_junk = 0;
		info->in_left_skip = 0;
		info->out_width -= out_left_pad;
		info->skip_col = (4-(out_width & 3))%4;
		logmsg(nn,1," valid padding only");
	} else {
		//only skip col if oput wdth is over 3 otherwise pipe is not flushed correctly
		info->skip_col = (((out_width & 3) <= ((4-info->out_left_junk)&3) && (out_width & 3)!=0 && out_width > 4)) ? 4:0;
                info->skip_col |= ((4-((out_width + info->out_left_junk) & 3)) % 4);
		info->input_base = tensor_location_d32(in_tensor,0,-required_h_before,-in_left_pad,0);
		logmsg(nn,1," same or valid (with non mod4 in_left_pad) padding ");
	}

	//handle cases of in_depth = 48 80 etc.
	if(work.fix_overcompute == 0 && stride_width==1 && info->in_depth_after == 16 && info->in_depth > 32 && filt_width > 1 && filt_height > 1) {
		work.fix_overcompute = 2;
	}

	/* Skip initial padding lines */
	//input_skip_lines = in_top_pad-required_h_before;
	//info->input_base += info->in_next_row*input_skip_lines;


	/*
	 * Preparing the work list
	 * We create a work list, which is just a list of items for the main thread to do
	 * We need to put in all the things we need to do to execute the node:
	 * * Padding zapping
	 * * Actual convolution
	 * * l2fetch / memcpy of values
	 * * Waiting on semaphores for things to finish
	 * The work list is currently executed by the main thread
	 * This means that vector work needs to be passed to nn_os_work_for_vector
	 */

	logmsg(nn,1,"softreset work items ...");
	supernode_softreset_work_items(self,nn,info);
	logmsg(nn,1,"softreset work items done");
	/*
	 * Set up work items
	 */

	/* FIXME for v66: copy work has to actually work. */
	//copywork.execute = supernode_execute_workitem_copy;
	//copywork.info = info;
	//copywork.self = self;

	//waitwork.execute = supernode_execute_workitem_join_some;

	work.info = info;
	work.self = self;
	//work.input = info->input_base;
	//work.output  = out;
	//work.weights = filtdst;
	work.stop_line = info->out_height;
	work.skip_lines = 1;
	//work.new_execute = supernode_execute_workitem_vector_dispatch;	// FIXME, pick the right function
	work.new_execute = supernode_execute_hvx_conv_work_v66;			// FIXME, pick the right function

	startwork.self = self;
	startwork.info = info;
	startwork.zap_top = (uint8_t *)info->input_base0;
	startwork.zap_bot = (uint8_t *)info->input_base0
		+ (required_h_before + in_height)*info->in_next_row;
	startwork.zap_top_size = info->in_next_row * required_h_before;
	startwork.zap_bot_size = info->in_next_row * required_h_after;
	/* If we need extra right padding that we don't have, add another row to the bottom zapping to handle overflow */
	if ((required_w_after > in_right_pad) && (required_h_after < in_bottom_pad)) {
		startwork.zap_bot_size += info->in_next_row;
	}
	startwork.zap_left = tensor_location_d32(in_tensor,0,0,-in_left_pad,0);
	startwork.zap_right = tensor_location_d32(in_tensor,0,0,in_width,0);
	if (required_w_after) startwork.zap_right_size = Q6_R_max_RR(in_right_pad,required_w_after);
	else startwork.zap_right_size = 0;	// EJP: are there other cases where we need to zap right pad?
	startwork.zap_rl_depths = in_depth_total / 32;
	//startwork.zap_left_size = in_left_pad; //always zap
	if (required_w_before || required_w_after > in_right_pad) startwork.zap_left_size = in_left_pad;
	else startwork.zap_left_size = 0;  //DJH always pad

	startwork.zap_height = in_height;
	startwork.zap_value = input_offset;

	/*
	 * If we have padding zap work to do, add it to the list
	 * Some extra zapping happens due to getting GEMSUMA to work, perhaps it could be optimized
	 * FIXME: maybe just set zap_right and zap_right_size and let it be on
	 * the next row if enough left pad exists.
	 */
	if ((out_right_pad > 0) && (startwork.zap_left_size == 0)) {
		//if (out_right_pad > in_left_pad) return errlog(nn,"oops, not enough left pad");
		/* EJP: FIXME: this probably doesn't work if zap_right_size goes byond the vector size */
		startwork.zap_left_size = in_left_pad;
		startwork.zap_right_size = in_right_pad;
	}
	//work.zap_rl_depths = in_depth_total/32;
	//work.zap_value = input_offset;

	/* 
	 * Slicing was a good first attempt, but needs refactoring.
	 *   Traffic ~= activation_chunk_size * weight_slices + weight_chunk_size * activation_slices
	 *   ... but maybe not quite so simple.  Want to maximize use of locality
	 *   Maybe need outer_activation, outer_weights, inner_activation, inner_weights nested loops
	 * What's the right optimization technique?
	 *  Weights have to be at least 32*h*w*in_depth
	 *  Probably need at least a few rows of activation
	 *  Weight chunks outer, activation inner --> more activation BW
	 *  Activation chunks outer, weight chunks inner --> more weight BW
	 *  Looking at the literature, it looks like it's a non-trivial problem?
	 * If we have a huge activation, we want to favor activation locality.
	 * If we have huge weights, we want to favor weight locality.
	 * So... our constraints are:
	 *  * Minimum effective weights size (d32 chunk size), granularity
	 *  * Minimum effective activation size (currently some rows, but we could subdivide further...), granularity
	 *  * Weights + Activation + prefetching room < cache_size
	 *  * Either Weights or Activations needs to have multiple chunks for multithreading!
	 * Given those things, try and find something that matches the activation/weight size ratio.
	 *   Start with minimum amounts of activation/weights
	 *   Add to activation/weight size to try and get closer to the ratio, see if it's still cache-friendly
	 *   If we can't add the thing that makes the ratio better, try and add a little more of the other one
	 * Once we have activation/weight chunk sizes, how should we proceed to the next chunk of work?
	 *   Probably the one that needs less bandwidth immediately.
	 *   We will need to iterate over all the weight and activation chunks, but the instantaneous BW requirements are lower if going to the next small thing.
	 * Need to be aware of threading...
	 *   Let's start out with always threading by dividing the activation, easier to handle suma/integral/zapping
	 * So.......
	 * for (outer_weights) {
	 *   for (outer_activations) {
	 *     for (inner_weights) {
	 *       for (inner_activations) {
	 *       }
	 *     }
	 *   }
	 * }
	 * If weights_fit: outer_weights = 1, outer_activations = n_act_chunks, inner_weights = all, inner_activations=chunksize/threads
	 * else if activations_fit: outer_weights = 1, outer_activations = 1, inner_weights = all, inner_activations=all/threads
	 * else outer_weights = n_weight_chunks, outer_activations = n_act_chunks, inner_weights = weight_chunksize, inner_activations=chunksize/threads
	 * 
	 * TBD: should we consider batches just part of the activation size?
	 *   * Probably, but it's hard to index and iterate over. 
	 *   * Maybe split above loop activations into outer_batches, outer_rows
	 */

	/* Determine how we want to slice the work */
	int32_t inner_act_batches;
	int32_t inner_act_rows;
	int32_t inner_weight_chunks;
        int32_t burst_chunks; //how many weight chuks to process per activation each time

	slice_for_cache_v66(
		self->node_id,
		nn,
		out_batches,
		num_in_slices,
		info->in_width,
                info->out_width,
		info->out_height,
		info->stride_height,
		info->filt_width,
		info->filt_height,
		weight_batch_size,
		num_out_slices,
		&inner_act_batches,
                &inner_act_rows,
		&inner_weight_chunks,
                &burst_chunks);
	work.num_lines = inner_act_rows;
        work.vtcm_chunks = inner_weight_chunks;
	inner_act_rows = (out_height+NUM_THREADS-1)/(NUM_THREADS);

	/* Ignore batch / weight slice factor for now ... */
	int32_t outer_act_batches = (out_batches + inner_act_batches - 1)/inner_act_batches;
	int32_t outer_act_iters = (out_height + inner_act_rows - 1)/inner_act_rows;
	int32_t outer_weight_chunks = (num_out_slices + inner_weight_chunks - 1) / inner_weight_chunks;

	// NOTE: outer_act_iters <= NUM_THREADS.

	logmsg(nn,1,"batch/row/weight chks: inner=%d,%d,%d outer=%d,%d,%d out_height=%d out_depth_chunks=%d",
		inner_act_batches,work.num_lines,inner_weight_chunks,
		outer_act_batches,outer_act_iters,outer_weight_chunks,
		out_height,
		num_out_slices);

    /*-------------------------------------------------------------*/
    /*  Setup parameters  */
    /*-------------------------------------------------------------*/
	int i,iw,ow,or,ib,ob,b;
        int total_chunks = num_out_slices;

	float underload_ratio = 1.0f;
	for (or = 0; or < outer_act_iters; or++) {
		int start_row = or * inner_act_rows;
		int n_rows = Q6_R_min_RR(out_height-start_row,inner_act_rows);
		if (n_rows < inner_act_rows) underload_ratio = (float)n_rows/inner_act_rows;
	};


	info->startup_info.batch_start_offset = 0;
	//info->startup_info.wakeup_items = outer_act_iters * inner_weight_chunks;

	//if (hlines_per_slice <= NUM_THREADS) return errlog(nn,"OOPS: chopped too fine");
	int batchstart_idx = 0;
	int weights_fit = (total_chunks == inner_weight_chunks);

	for (ob = 0; ob < outer_act_batches; ob++) {
	inner_act_batches = Q6_R_min_RR(inner_act_batches,(in_batches-ob*inner_act_batches));
	int NOW_BATCHES=inner_act_batches;



	info->startup_info.wakeup_items = 0; // figure out below, refigure each batch
	//if (b >= in_batches) continue;
	//if (b > 0) return errlog(nn,"Fixme: batches");

#if 0
        for(iw = 0; iw < inner_weight_chunks; iw++)
        {
	   supernode_weight_chunk(self,nn,info, 0, info->weights, weight_batch_size, iw, outer_act_iters);
        }
#else
	//supernode_weight_chunk(self,nn,info,0,info->weights,weight_batch_size*inner_weight_chunks,0,outer_act_iters);
#endif
	if ((ob == 0) || (!weights_fit)) {
		logmsg(nn,2,"adding memcpy: %p %d*%d",info->weights, weight_batch_size, inner_weight_chunks);
		startwork.copy_in = info->weights;
		startwork.copy_out = nn->vtcm_ptr;
		startwork.copy_size = weight_batch_size*inner_weight_chunks;
	} else {
		logmsg(nn,2,"Memcpy should be OK already:");
		startwork.copy_in = NULL;
		startwork.copy_out = NULL;
		startwork.copy_size = 0;
	}
	if (ob == 0) supernode_note_earlywork(nn,self,info,startwork.copy_out,startwork.copy_in,startwork.copy_size);

	/* Zap padding is back */
	if (ob > 0) info->work_items[batchstart_idx].next_startup_offset = info->n_work_items - batchstart_idx - 1;
	startwork.join_iters = total_chunks;
	batchstart_idx = supernode_add_batch_startup(self,nn,info,startwork,ob*input_batch_size,required_h_before,required_w_before,NOW_BATCHES);
	logmsg(nn,2,"batchstart idx: %d",batchstart_idx);
	/* If we reuse VTCM, set burst_chunks to 1, because it's hard to figure out wakeup and wrapround below */
	if (total_chunks == inner_weight_chunks) burst_chunks = inner_weight_chunks;
	else burst_chunks = 1;
	if ((in_batches == 1) && (underload_ratio < 0.67) && (burst_chunks > 2)) {
		logmsg(nn,2,"Underload ratio=%f, changing burst chunks: %d-->%d",underload_ratio,burst_chunks,1);
		burst_chunks = 1;
	}


	for (ow = 0; ow < total_chunks; ) {

          int32_t now_chunks = Q6_R_min_RR(total_chunks-ow, burst_chunks);
	  //int32_t next_chunks = Q6_R_min_RR(total_chunks-(ow+now_chunks),burst_chunks);
	  int32_t copy_chunks = Q6_R_max_RR(0,Q6_R_min_RR(burst_chunks,total_chunks-(inner_weight_chunks + ow)));
          if (ow < inner_weight_chunks) info->startup_info.wakeup_items += outer_act_iters * inner_act_batches;
	for (ib = 0; ib < inner_act_batches; ib++) {
	b = ob * inner_act_batches + ib;
	  for (or = 0; or < outer_act_iters; or++) {

		int start_row = or * inner_act_rows;
		int n_rows = Q6_R_min_RR(out_height-start_row,inner_act_rows);
                logmsg(nn,2,"%d) %d line", or,n_rows);

		/* FILL OUT NORMAL WORK INFORMATION */
		work.weights = (uint8_t *) nn->vtcm_ptr + (ow % info->n_weight_batches) * weight_batch_size;
		work.start_chunk = ow;
		work.weight_chunks = now_chunks;
		work.input = info->input_base+b*input_batch_size;
		work.output = tensor_location_d32(out_tensor,b,0,0,32*ow);
		work.biases = info->biasbuf + 32 * ow;
		work.recip = info->recip + 32 * ow;
		work.start_line = start_row;
		work.stop_line = work.start_line + n_rows;
                work.conv_done = info->semaphores;
                work.threads = outer_act_iters;
                work.last_chunk = ((ow + now_chunks) == total_chunks );
		supernode_add_work_item(self,nn,info,work);
	  }//or
        }//ib
	  for (i = 0; i < now_chunks; i++) {
              //int refill_chunk = inner_weight_chunks + ow + i;
              //int weight_chunk_id = -1;
              //int n_weights = 0;
              //weight_chunk_id = supernode_weight_chunk(self,nn,info,1,info->weights,weight_batch_size,iw,outer_act_iters);
              if (i < copy_chunks) {
                iw  = inner_weight_chunks + ow + i ;
	        //weight_chunk_id = supernode_weight_chunk(self,nn,info, 1, info->weights,weight_batch_size,iw,outer_act_iters);
		info->conv_slices[ow+i].copy_in = info->weights + iw * weight_batch_size;
		info->conv_slices[ow+i].copy_out = ((uint8_t *)nn->vtcm_ptr) + (iw % info->n_weight_batches) * weight_batch_size;
		info->conv_slices[ow+i].copy_size = weight_batch_size;
		//info->conv_slices[ow+i].batch_start_offset = batchstart_idx + iw * outer_act_iters;
		info->conv_slices[ow+i].batch_start_offset = iw * outer_act_iters * inner_act_batches;
		info->conv_slices[ow+i].wakeup_items = outer_act_iters*inner_act_batches;
              } else {
		iw = -1;
		memset(&info->conv_slices[ow+i],0,sizeof(info->conv_slices[ow+i]));
              }
              nn_checkpoint_init(&info->conv_slices[ow+i].checkpoint,outer_act_iters*inner_act_batches,note_chunk_checkpoint_arrival,(void *)&info->conv_slices[ow+i]);
	  }
#if 0
          for (i = 0; i < next_chunks; i++)
          {
             //iw  = inner_weight_chunks + ow + i ;
             if(iw < total_chunks) {
	       //supernode_weight_chunk(self,nn,info, 1, info->weights,weight_batch_size,iw,outer_act_iters);
             }
          }
#endif
          ow += now_chunks;
        }//ow
	//nn_checkpoint_init(&info->alldone_checkpoint,total_chunks,note_alldone_checkpoint_arrival,(void *)(-1));
	//supernode_convs_done(self,nn,info,inner_weight_chunks,outer_act_iters); //check off all outstadning jobs

        }//ob
	/* Add work to check the output min/max and see if we need to adjust and try again */
	//work.execute = supernode_execute_workitem_check_for_retry;
	//supernode_add_work_item(self,nn,info,work);

	/*
	 * Sometimes we want to collect some statistics...
	 */
	if (0) supernode_statistics(nn,info,self);

	/*
	 * We've calculated the strategy, mark that the work is done. Hopefully it sticks!
	 */
	supernode_compile_worklist(nn,info,self);
	info->prepared_vtcm_addr = nn->vtcm_ptr;
	info->needs_retry = 0;
	info->strategy_valid = 1;
	return 0;
}

#if defined(V65) || defined(SCRATCH)
int get_circ_buf_size(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *stride_tensor = self->inputs[6];
	struct supernode_info_new *info = self->opaque;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_left_pad = in_tensor->format.width_pad[0];
	int32_t in_right_pad = in_tensor->format.width_pad[1];
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t stride_height = stride_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
        int32_t buf_size = 2*((filt_height > stride_height) ? filt_height : stride_height);
	logmsg(nn,2,"filt_height=%d stride_height=%d initial buf_size=%d in_right_padpad=%d in_width_total=%d in_depth_total=%d result=%d",
		filt_height,stride_height,buf_size,info->in_right_padpad,in_width_total,in_depth_total,
        	buf_size*(info->in_right_padpad + in_width_total)*in_depth_total);

        buf_size = buf_size*(info->in_right_padpad + in_width_total)*in_depth_total;
        return(buf_size);
}
#else
int get_circ_buf_size(struct nn_node *self, struct nn_graph *nn)
{
	return 0;
}
#endif


static int supernode_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info_new *nodeinfo = self->opaque;
	struct tensor *out = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	unsigned long long int total_time;
	//logmsg(nn,0,"NEW supernode id=%d",self->node_id);
	//logmsg(nn,0,"FIXME: memset out for debug... remove");
	//memset(out->data,0xAB,out->max_size);
	if (out->format.layout == NN_LAYOUT_D32_LOVINGLY_PREPARED) {
		logmsg(nn,2,"Prepared output.");
		logmsg(nn,2,"out tensor info: bhwd=%d,%d,%d,%d paddings=(%d,%d)x(%d,%d)x(%d,%d)",
			out->shape.batches,out->shape.height,out->shape.width,out->shape.depth,
			out->format.height_pad[0],out->format.height_pad[1],
			out->format.width_pad[0],out->format.width_pad[1],
			out->format.depth_pad[0],out->format.depth_pad[1]);
	}
	nn_scratch_reset(nn);			// if we recurse, reset scratch ptr
	l2fetch(nodeinfo,sizeof(*nodeinfo),sizeof(*nodeinfo),1);
	if ((nodeinfo->work_items != NULL) && (nodeinfo->n_work_items > 0)) {
		l2fetch(nodeinfo->work_items,sizeof(struct workitem),sizeof(struct workitem),nodeinfo->n_work_items);
	}

        nodeinfo->circ_buf_size = (get_circ_buf_size(self, nn) + 127) & (~127); //used in v65 
#if defined(V65) && !defined(V66)
        int i;
	int total_size = nodeinfo->circ_buf_size * NUM_THREADS;
	char *buf;
	if ((total_size <= VTCM_CIRCBUF_SIZE) && (total_size <= nn->vtcm_size)) {
		buf = nn->vtcm_ptr;
		buf += (nn->vtcm_size-VTCM_CIRCBUF_SIZE);
	} else {
		buf = nn_scratch_alloc(nn,total_size);
	}
	if (buf == NULL) return errlog(nn,"scratch failed to alloc %d bytes for bufstack",(int)total_size);
	nn_os_bufstack_init(&nodeinfo->bufstack);
	for (i = 0; i < NUM_THREADS; i++) {
		nn_os_bufstack_push(&nodeinfo->bufstack,buf+i*nodeinfo->circ_buf_size);
	}
#endif
	//nn_scratch_grow(nn,nodeinfo->circ_buf_size * NUM_THREADS);

	if (likely(supernode_strategy_valid(self,nn,nodeinfo))) {
		if (supernode_execute_strategy(self,nn) != 0) {
			return errlog(nn,"execute strategy failed");
		}
	} else {
                if(nodeinfo->use_v66 == 1) {
			if (supernode_recalculate_strategy_v66(self,nn) != 0) {
				return errlog(nn,"recalc strategy for v66 failed");
			}
                } else {
			if (supernode_recalculate_strategy(self,nn) != 0) {
				return errlog(nn,"recalc strategy failed");
			}
                }
		if (supernode_execute_strategy(self,nn) != 0) {
			return errlog(nn,"execute strategy fail after recalc");
		}
	}
	/* Replay if self-calculated min/max are insufficient */
	if (nodeinfo->needs_retry) {
		nodeinfo->recursion_depth++;
		if (nodeinfo->recursion_depth < 3) {
			return supernode_execute_hvx(self,nn);
		} else {
			logmsg(nn,0,"Extreme recursion detected, problem finding min/max?");
		}
	}
	nodeinfo->recursion_depth = 0;
	tensor_set_float(out_min,0,nodeinfo->out_minval);
	tensor_set_float(out_max,0,nodeinfo->out_maxval);
	/* Record cycles (divide by # of vector worker threads somehow?) */
	total_time = nodeinfo->cycles;
	record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER0,total_time);
	logmsg(nn,2,"out tensor info: bhwd=%d,%d,%d,%d paddings=(%d,%d)x(%d,%d)x(%d,%d)",
		out->shape.batches,out->shape.height,out->shape.width,out->shape.depth,
		out->format.height_pad[0],out->format.height_pad[1],
		out->format.width_pad[0],out->format.width_pad[1],
		out->format.depth_pad[0],out->format.depth_pad[1]);
	logmsg(nn,2,"Supernode execute done!");
	return 0;
}

int supernode_check(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info_new *info = self->opaque;

	// ctor checks that n_inputs = 12 or 13 (13th is ChannelScale
	// and n_outputs = 3

	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *filt_min_tensor = self->inputs[4];
	const struct tensor *filt_max_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	float filt_max_float = tensor_get_float(filt_max_tensor,0);
	float filt_min_float = tensor_get_float(filt_min_tensor,0);
	float filt_level_size;
	int32_t filt_offset = get_qu8_level_size_zero(filt_min_float,filt_max_float,&filt_level_size);
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_batches_roundup = (filt_batches + 31) & ~31;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	int32_t filt_depth_roundup = (filt_depth + 31) & ~31;
	uint32_t filt_elements = filt_width * filt_height * filt_depth_roundup;
	uint32_t weights_size = filt_elements * filt_batches_roundup;
	int32_t stride_width = stride_tensor->shape.width;
	uint32_t out_depth = filt_batches_roundup;
	int32_t weight_batch_size = filt_width * filt_height * filt_depth_roundup * 32;
	int32_t n_weight_batches = supernode_n_weight_batches(weight_batch_size, nn->vtcm_size);
	int32_t total_weight_batches = filt_batches_roundup/32;
	float specified_minval = tensor_get_float(self->inputs[10],0);
	float specified_maxval = tensor_get_float(self->inputs[11],0);
	int i;
	int use_v66 = 0;

	int circ_buf_est = 2*(filt_height+stride_tensor->shape.height)*filt_depth*(self->output_defs[0].max_sizes[1]*stride_width+8);
	nn_scratch_grow(nn,circ_buf_est*NUM_THREADS);

#ifdef HEXAGON_V66

#if 0
        nn_os_vtcm_acquire(nn);
        uint8_t * ts_buf = nn_memalign(128,512);
        uint8_t * td_buf = (uint8_t *) nn->vtcm_ptr;

        for(i=0; i < 256; i++) ts_buf[i] = i;

        asm volatile (
                "M0 = %2; memcpy(%0,%1,M0)"
        ::"r"(td_buf),"r"(ts_buf),"r"(256-1)
        :"m0");

        for(i=0; i < 256; i++) {
             logmsg(nn, 0, " %02x -> buf[%d] -> %02x" , ts_buf[i], i, td_buf[i]);
             //if(ts_buf[i] != td_buf[i]) return errlog(nn,"FIXME: memcpy doesnt work properly");
        }

        nn_free(ts_buf);
        nn_os_vtcm_release(nn);
#endif
        
	/*
	 * In V66, we need chunks of weights to not cross a page boundary,
	 * but if weights_size is too big then memalign can fail
	 */
	weights_size = 1U << (32-__builtin_clz(weights_size-1));
	int weights_align = weights_size;
        if(weight_batch_size <= nn->vtcm_size)
        {
	  use_v66 = 1;
        }
	if (stride_width > 2) use_v66 = 0;
	logmsg(nn,2,"stride_width = %d use_v66=%d",stride_width,use_v66);
#else
	int weights_align = 128;
#endif
	int use_v65 = 0;
#ifdef V65
        /*
         * bail out to use v60 imlpementation if weights chunks dont fit into VTCM
         */
        if(weight_batch_size <= nn->vtcm_size)
        {
	  use_v65 = 1;
        }
#endif
#if 0
	if (stride_width > 2) {
		logmsg(nn,1,"Trying to disable V65 code, stride width > 2");
		use_v65 = 0;
	}
#endif
	if ((filt_elements % 32) != 0) return errlog(nn,"FIXME: < 32 depth");
	if ((filt_batches_roundup % 32) != 0) return errlog(nn,"FIXME: < 32 filts");
	if (info != NULL) {
		/* Already set up, invalidate strategy and return */
		info->strategy_valid = 0;
		logmsg(nn,0,"info was already set up?");
		return 0;
	}

	// check if we have a channel-scaling input (#12)
	float const *channel_scale_flts = NULL;
	if( check_channelscale_present(nn,self, filt_batches, &channel_scale_flts)!=0)
		return -1;

	if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
		return errlog(nn,"couldn't allocate info");
	}
	if ((info->minmax_buf = nn_memalign(128,NUM_THREADS*n_weight_batches*64*sizeof(int))) == NULL) {
		return supernode_check_error_return(nn, info,"minmax");
	}
	if ((info->weights = nn_memalign(weights_align,weights_size)) == NULL) {
		return supernode_check_error_return(nn, info,"weights");
	}
	if ((info->biasbuf = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"biasbuf");
	}
	if ((info->semaphores = nn_calloc(3+n_weight_batches,sizeof(nn_sem_t))) == NULL) {
		return supernode_check_error_return(nn, info,"semaphores");
	}
	if ((info->gemsumb = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"gemsumb");
	}
	if ((info->conv_slices = nn_calloc(total_weight_batches,sizeof(*info->conv_slices))) == NULL) {
		return supernode_check_error_return(nn, info,"conv_slices");
	}
	if ((info->recip = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"recip");
	}
	if ((info->k_factor = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"k_factor");
	}
	if ((info->k_factor_recip = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"k_factor_recip");
	}
	// load channel-scale (if any)
	if( load_channel_scales(nn,info,channel_scale_flts,filt_batches) != 0 ){
		return supernode_check_error_return(nn, info,NULL);
	}
#if 0
	for (i = 0; i < total_weight_batches; i++) {
		nn_checkpoint_init(&info->conv_checkpoints[i],1,note_conv_checkpoint_arrival,(void *)(i));
	}
#endif
	int use_signed_weights = use_v65 | use_v66;
	info->use_v65 = use_v65;
	info->use_v66 = use_v66;
	info->use_signed_weights = use_signed_weights;
        info->num_accs = 8;
	for (i = 0; i < n_weight_batches+3; i++) {
		nn_sem_init(&info->semaphores[i],0);
	}
	float weight_scalefac = 1.0;	// how much weights were scaled (if at all);  0.5 .. 1
	// (weight_scalefac is not used, since we look at all the weights & channel scales
	// and find the largest net scaling factor, max_k_factor)
	//
	// weight_scale[d] (as a fixed-point, with 31 fractional bits)
	// is laid on top of k_factor_recip buffer, until k_factor and k_factor_recip are found
	//
	int32_t * weight_scale = (int32_t*) info->k_factor_recip;


#ifdef ENABLE_VECTOR_WEIGHT_ARRANGE
	// @@@ NOTE @@@
	// (1) the below is acquiring and releasing a vector context, since we don't currently
	// have one any other way in the check function
	// (2) 'repack_filter_for_d32' returns a result in rpfparms.coeffscale, which is either 1.0 or 0.5,
	// used to compensate info->weights_level_size
	//
	{
		info->has_weight_scale = use_signed_weights;
		struct repack_filter_parms rpfparms;
		rpfparms.out_data = info->weights;
		rpfparms.filt_tensor = filt_tensor;
		rpfparms.zero_offset = filt_offset;
		rpfparms.signed_mode_sel = use_signed_weights;
		rpfparms.gemsumb = info->gemsumb;
		rpfparms.scalefac_vec = weight_scale;

		nn_sem_init( & rpfparms.done_sem, 0);
		// call it directly.
		repack_filter_for_d32( nn, &rpfparms);
		weight_scalefac = rpfparms.coeffscale;
		if( weight_scalefac == 1.0f){
			info->has_weight_scale = 0;	// signed, but no scaling needed
		}
		logmsg(nn,2,"weights global scale factor= %f with %d zero",weight_scalefac,filt_offset);
	}


#else // ! ENABLE_VECTOR_WEIGHT_ARRANGE
        if(info->use_signed_weights) {
           weight_scalefac = supernode_rearrange_vector_d32(
                info->weights,
                filt_tensor->data,
                filt_height,
                filt_width,
                filt_depth,
                filt_depth_roundup,
                filt_batches,
                filt_batches_roundup,
                filt_offset,
                weight_scale,
                info->gemsumb);
#if 0
        } else if(info->use_v65) {
           weight_scalefac = supernode_rearrange_scaler_d32(
                info->weights,
                filt_tensor->data,
                filt_height,
                filt_width,
                filt_depth,
                filt_depth_roundup,
                filt_batches,
                filt_batches_roundup,
                filt_offset,
                info->gemsumb);
#endif
        } else {
            supernode_rearrange_d32(
                info->weights,
                filt_tensor->data,
                filt_height,
                filt_width,
                filt_depth,
                filt_depth_roundup,
                filt_batches,
                filt_batches_roundup,
                filt_offset,
                info->gemsumb);
        }
#endif// ! ENABLE_VECTOR_WEIGHT_ARRANGE

	//
	// set up the k_factor and k_factor_recip
	// In general, we are setting k_factor = k_factor/weight_scale
	//                            k_factor_inv = weight_scale/k_factor
	find_k_kinv( nn, info, filt_batches_roundup);

	logmsg(nn,3,"has_channel_scale= %d, has_weight_scale= %d, max_k = %f",
			info->has_channel_scale, info->has_weight_scale, info->max_k_factor);

	supernode_cleaninv_weights(
		info->weights,
		filt_height*filt_width*filt_depth_roundup*filt_batches_roundup);
	//logmsg(nn,2,"weights cksum: %08x",data_cksum(info->weights,filt_height*filt_width*filt_depth_roundup*filt_batches_roundup));
	info->strategy_valid = 0;	/* Redundant w/ calloc */
	self->opaque = info;
	info->weight_batch_size = weight_batch_size;
	info->n_weight_batches = n_weight_batches;
        info->in_right_padpad = 8*stride_width; //tack on this to circular buffer to avoid bad max's

        info->weights_offset = use_signed_weights ? 0: filt_offset;
        info->filt_offset = filt_offset;
        info->weights_level_size = filt_level_size;
        logmsg(nn,2,"filt_offset  = %d", filt_offset);

	logmsg(nn,2,"stride_width=%d in_right_padpad=%d",stride_width,info->in_right_padpad);

	nn_sem_init(&info->alldone_sem,0);
	if(setup_initial_output_range( nn, info, specified_minval, specified_maxval, 0.0f, 0.5f)) return -1;

	return 0;

}

static int supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info_new *info = self->opaque;
	if (info != NULL) {
		supernode_reset_work_items(self,nn,info);
		nn_free(info->k_factor_recip);
		nn_free(info->k_factor);
		nn_free(info->recip);
		nn_free(info->conv_slices);
		nn_free(info->gemsumb);
		nn_free(info->semaphores);
		nn_free(info->biasbuf);
		nn_free(info->weights);
		nn_free(info->minmax_buf);
		nn_free(info);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}


static int supernode_earlywork_note_pred(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor)
{
	struct supernode_info_new *info = self->opaque;
	if (info == NULL) return errlog(nn,"Oops: no supernode info available");
	if (predecessor == NULL) return errlog(nn,"Oops: NULL predecessor");
	if (predecessor->ops->earlywork_register != NULL) {
		if (predecessor->ops->earlywork_register(predecessor,nn,&info->my_earlywork) != 0) {
			return errlog(nn,"registering early work with %p failed?",predecessor);
		}
		info->earlywork_pred = predecessor;
	} else {
		logmsg(nn,2,"node %p does not take early work.",predecessor);
	}
	return 0;
}

static int supernode_earlywork_register(struct nn_node *self, struct nn_graph *nn, struct nn_early_work *work)
{
	struct supernode_info_new *info = self->opaque;
	if (info == NULL) return errlog(nn,"Oops: no supernode info available");
	info->next_earlywork = work;
	return 0;
}

struct nn_node_ops nn_ops_for_Supernode_8x8p8to8_d32 = {
	.execute = supernode_execute_hvx,
	.check = supernode_check,
	.ctor = node_alloc_common,
	.dtor = supernode_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),

	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT | NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION,
	.earlywork_note_pred = supernode_earlywork_note_pred,
	.earlywork_register = supernode_earlywork_register,
};

struct nn_node_ops nn_ops_for_Supernode_8x8p32to8_d32 = {
	.execute = supernode_execute_hvx,
	.check = supernode_check,
	.ctor = node_alloc_common,
	.dtor = supernode_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),

	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT | NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION,
	.earlywork_note_pred = supernode_earlywork_note_pred,
	.earlywork_register = supernode_earlywork_register,
};

#if 1
/*
 * On Vector Thread, take a slice of input and do the work for it.
 * 
 */

/*
   Table to control vdelta and conditional add operations for 3to4 expand code.
 */
const unsigned char copy3to4_cntrl[128] __attribute__ ((aligned (128))) = {
//expand 3 to 4 bytes for input layer
0x00,0x00,0x00,0x00,0x01,0x03,0x01,0x06,0x02,0x02,0x06,0x04,0x03,0x05,0x0D,0x0C,
0x04,0x04,0x04,0x00,0x0D,0x0F,0x09,0x0A,0x06,0x02,0x0A,0x08,0x1B,0x19,0x19,0x18,
0x08,0x08,0x08,0x08,0x09,0x0B,0x01,0x06,0x1A,0x1A,0x1E,0x1C,0x13,0x15,0x15,0x14,
0x0C,0x0C,0x04,0x00,0x15,0x17,0x11,0x12,0x36,0x32,0x32,0x30,0x33,0x31,0x31,0x30,
0x10,0x10,0x10,0x10,0x11,0x13,0x11,0x16,0x12,0x12,0x16,0x14,0x03,0x05,0x0D,0x0C,
0x34,0x34,0x34,0x30,0x3D,0x3F,0x39,0x3A,0x26,0x22,0x2A,0x28,0x2B,0x29,0x29,0x28,
0x18,0x18,0x18,0x18,0x09,0x0B,0x01,0x06,0x2A,0x2A,0x2E,0x2C,0x23,0x25,0x25,0x24,
0x6C,0x6C,0x64,0x60,0x65,0x67,0x61,0x62,0x66,0x62,0x62,0x60,0x63,0x61,0x61,0x60,
};

#define LPAD 2   //this is the extra left pad needed to make sure integration gives regular results

#if 0
#if 1
void shortin_zapslice_depth4(struct nn_graph *nn, void * vinfo)
{
  struct workitem *zapwork = vinfo;
  struct supernode_info_new *info = zapwork->info;
  struct nn_node *self = zapwork->self;
  const struct tensor *in_tensor = self->inputs[0];
  int32_t in_image_depth =  in_tensor->shape.depth;  //raw input tensor
  int32_t in_width = in_tensor->shape.width;  
  int32_t in_height = in_tensor->shape.height;

  int pad_left = zapwork->zap_left_size;
  int pad_right = zapwork->zap_right_size;
  int pad_top= zapwork->zap_top_size;  //overloading this as its just the height not the area
  int pad_bot= zapwork->zap_bot_size;  //same

  int out_y, in_y, j ;
  uint8_t *in_data = (uint8_t *)info->raw_input;
  uint8_t *in_data_d4_pad = zapwork->zap_left;
  uint8_t *iptr ;
  uint8_t *optr ;
  int32_t delta  = (info->filt_height+1)/info->stride_height;
  in_y = (zapwork->start_line * info->stride_height) - pad_top;

  l2fetch(in_data,in_width*in_image_depth,in_width*in_image_depth,in_height);

  for (out_y = zapwork->start_line; out_y < zapwork->stop_line+delta; out_y++)
  {
      for(j=0; (j < info->stride_height) || (in_y >= in_height && in_y <= (in_height + pad_bot)); j++)
      {
          optr = in_data_d4_pad + in_y * info->in_next_row; //in_depth*in_width_pad;
          if(in_y < 0 || in_y >= in_height) {
              logmsg(nn,2,"zapping row @ y=%d optr=%p in_next_row=%d",in_y,optr,info->in_next_row);
              vmemset_asm(optr, info->in_offset, info->in_next_row);
          } else {
              iptr = in_data + in_y * in_image_depth * in_width - (in_image_depth * pad_left) / 4;
//logmsg(nn,2,"in_data=%p in_y=%d in_image_depth=%d in_width=%d pad_left=%d iptr=%p", in_data,in_y,in_image_depth,in_width,pad_left,iptr);
              copyNto4_asm(optr, iptr, info->in_next_row/128, info->in_offset, in_image_depth, copy3to4_cntrl); //
//logmsg(nn,2,"in_data_d4_pad=%p optr=%p in_next_row=%d",in_data_d4_pad,optr,info->in_next_row); vmemset_asm(optr, info->in_offset, pad_left);
              if (pad_left > 0) vmemset_asm(optr,info->in_offset,pad_left);
              if(pad_right > 0) vmemset_asm(optr+pad_left+4*in_width, info->in_offset, pad_right);
//const uint32_t *ptmp = (const uint32_t *)optr;
//logmsg(nn,2,"optr %p [0..7]: %x %x %x %x %x %x %x %x", ptmp,ptmp[0],ptmp[1],ptmp[2],ptmp[3],ptmp[4],ptmp[5],ptmp[6],ptmp[7]);
          }
          in_y++;
      }
  }//out_y
  return;
}
#else
static void shortin_convert_image_Nto4(void * vinfo)
{
    struct workitem *zapwork = vinfo;
    struct supernode_info_new *info = zapwork->info;
    struct nn_node *self = zapwork->self;
    const struct tensor *in_tensor = self->inputs[0];
    int32_t in_image_depth =  in_tensor->shape.depth;  //raw input tensor

    const uint8_t * optr = zapwork->zap_left; //info->input_base;
    const uint8_t * iptr = info->raw_input - (zapwork->zap_left_size*in_image_depth)/4;
    int i;
    for(i=0; i < zapwork->zap_height; i++)
    {
       copyNto4_asm(optr, iptr, info->in_next_row/128, info->in_offset, in_image_depth, copy3to4_cntrl);
       optr += info->in_next_row;
       iptr += in_image_depth*zapwork->nonzap_width;
    }
    return;
}
static void shortin_supernode_execute_zap_right(void * vinfo)
{
    struct workitem *zapwork = vinfo;
    struct supernode_info_new *info = zapwork->info;
    int i;
    uint8_t * ptr = zapwork->zap_right;
     
    for(i=0; i < zapwork->zap_height; i++)
    {
       vmemset_asm(ptr, zapwork->zap_value, zapwork->zap_right_size);
       ptr += info->in_next_row;
    }
    return;
}
static void shortin_supernode_execute_zap_left(void * vinfo)
{
    struct workitem *zapwork = vinfo;
    struct supernode_info_new *info = zapwork->info;
    int i;
    uint8_t * ptr = zapwork->zap_left;
    for(i=0; i < zapwork->zap_height; i++)
    {
       memset(ptr, zapwork->zap_value, zapwork->zap_left_size);
       ptr += info->in_next_row;
    }
    return;
}
static void shortin_supernode_execute_zap_top(void * vinfo)
{
    struct workitem *zapwork = vinfo;
    //struct supernode_info_new *info = zapwork->info;
    uint8_t * ptr = zapwork->zap_top;
    vmemset_asm(ptr, zapwork->zap_value, zapwork->zap_top_size);
    return;
}
static void shortin_supernode_execute_zap_bot(void * vinfo)
{
    struct workitem *zapwork = vinfo;
    //struct supernode_info_new *info = zapwork->info;
    uint8_t * ptr = zapwork->zap_bot;
    vmemset_asm(ptr, zapwork->zap_value, zapwork->zap_bot_size);
    return;
}
#endif

static int shortin_supernode_prepare_input(struct nn_graph *nn, void *vwork)
{
	struct workitem *work = vwork;
	//struct nn_node *self = work->self;
	//struct supernode_info_new *info = work->info;

        //int32_t * integral_buf = info->suma_int_tmp;
        //int32_t * suma_buf = info->suma_buf;
        //int32_t filter_count = info->in_depth*info->filt_height*info->filt_width;
        //int32_t offset = info->filt_offset*info->in_offset*filter_count;
	//vmemset_asm(nn->scratch,0,4*1024*1024);

        //printf(" filt_offset = %d\n", info->filt_offset);
        /* fill in my temp tensor */
	//logmsg(nn,2,"scratch=%p--%p integral_buf=%p suma_buf=%p",nn->scratch,nn->scratch+nn->scratch_size,integral_buf,suma_buf);
        shortin_zapslice_depth4(nn,work);

#if 0
	int i=0,j=0;
        for(i=0; i < 24; i++) {
           printf("%d	:",i);
           //for(j=0; j < info->suma_width; j++) printf("%ld,",integral_buf[info->suma_width*(i+0)+j]);
           //for(j=0; j < info->suma_width; j++) printf("%ld,",suma_buf[info->suma_width*(i+0)+j]);
           //for(j=0; j < info->in_next_row; j++) printf("%d,",work->input[info->in_next_row*(i+0)+j+2*4]);
           printf("\n");
        }
        //   for(j=0; j < info->out_depth*info->in_depth*info->filt_height*info->filt_width; j++) printf("%d,\n",work->weights[j]);
        printf(" in hvx_slice %ld %ld\n", info->out_depth, info->out_next_d32);
#endif
	return 0;
}
#else

static void shortin_prepare_indata(struct workitem *zapwork, int32_t ypos, int32_t nlines, int ibatch)
{
    struct supernode_info_new *info = zapwork->info;
    struct nn_node *self = zapwork->self;
    const struct tensor *in_tensor = self->inputs[0];

    int32_t in_image_depth =  in_tensor->shape.depth;   //raw input tensor
    int32_t in_width  = in_tensor->shape.width;  
    int32_t in_height = in_tensor->shape.height;

    int32_t in_width_depth = in_width * in_image_depth;

    int pad_left = zapwork->zap_left_size;
    int pad_right= zapwork->zap_right_size;
    int pad_top  = zapwork->zap_top_size;  //overloading this as its just the height not the area

    uint8_t *in_data = (uint8_t *)info->raw_input + ibatch * in_height*in_width_depth;

    int32_t y, in_y;

    // select an 'expand-to-4' function based on in_image_depth
    typedef void (*expfunc_fp)(uint8_t *, uint8_t const * , int, int, int, int);
    expfunc_fp exfuncp;

    static const expfunc_fp funcs[4] = {
    		incopy_expand_1to4,incopy_expand_2to4, incopy_expand_3to4, incopy_expand_4to4
    };
    exfuncp = (in_image_depth >=1 && in_image_depth <= 4)?  funcs[in_image_depth-1]: NULL;

    for (y = ypos; y < (ypos + nlines); y++) {

        uint8_t *optr = (uint8_t *)zapwork->input + y*info->in_next_row;

        in_y = y - (pad_top -1);

        if (in_y < 0 || in_y >= in_height) {
            vmemset_asm(optr, info->in_offset, info->in_next_row);
        } else {
            uint8_t *iptr = in_data + in_y * in_width_depth;
            if (exfuncp) {
                (*exfuncp)( optr, iptr, in_width, info->in_offset, pad_left>>2, pad_right>>2);
            }
        }
    }
}

static void shortin_prefetch_input(struct workitem *zapwork, int32_t ypos, int32_t nlines)
{
    struct supernode_info_new *info = zapwork->info;
    struct nn_node *self = zapwork->self;
    const struct tensor *in_tensor = self->inputs[0];

    int32_t in_image_depth =  in_tensor->shape.depth;   //raw input tensor
    int32_t in_width  = in_tensor->shape.width;  
    int32_t in_height = in_tensor->shape.height;

    int32_t in_width_depth = in_width * in_image_depth;
    int pad_top  = zapwork->zap_top_size;  //overloading this as its just the height not the area

    uint8_t *in_data = (uint8_t *)info->raw_input;

    int32_t start = ypos - (pad_top-1);
    int32_t stop  = start + nlines;

    if (start < 0) {
        nlines += start;
        start = 0;
    }

    if (stop >= in_height) {
       nlines -= (stop - in_height);
    }

    if (nlines > 0) {
        l2fetch(in_data + start*in_width_depth, in_width_depth, in_width_depth, nlines); 
    }
}
#endif

static void shortin_supernode_execute_hvx_slice(struct nn_graph *nn, void *vwork)
{
    struct workitem *work = vwork;
    //struct nn_node *self = work->self;
    struct supernode_info_new *info = work->info;
    
    union {
        HVX_Vector vec[4];
        int32_t words[256];
    } minmax;

    int32_t * integral_buf = nn_os_bufstack_pop(&info->bufstack);
    int32_t * suma_buf = work->suma_buf;
    int32_t filter_count = info->in_depth*info->filt_height*info->filt_width;
    int32_t offset = info->filt_offset*info->in_offset*filter_count;
    int32_t in_next_row = info->in_next_row;
    int32_t filt_height = info->filt_height;
    int32_t stride_height = info->stride_height;
    int32_t start_line = work->start_line;
    int32_t stop_line  = work->stop_line;

    int32_t ibatch = work->batch_index;

    const uint8_t *input = work->input + in_next_row*stride_height*start_line;
    // offset output area for current batch
    uint8_t *output_batch =  work->output + info->out_next_batch * ibatch;
    //uint8_t *output = output_batch + info->out_next_row*start_line;

    int32_t  proc_rows = work->num_lines;
    int32_t  pf_offset = Q6_R_max_RR(filt_height-stride_height, 0);

    int32_t n_lines = Q6_R_min_RR(stop_line-start_line, proc_rows);
    int32_t n_in_rows = (n_lines-1)*stride_height + filt_height; 
#if 1
    int32_t path223 = (info->stride_width == 2 && stride_height == 2 && info->filt_width == 3);
    inconv2d_t conv = (info->stride_width == 1) ? &inconv2dbbb_s1_v60_asm :
                                                  (path223) ? &inconv2dbbb332_v60_asm : &inconv2dbbb_v60_asm;
#else
    inconv2d_t conv = (info->stride_width == 1) ? &inconv2dbbb_s1_v60_asm : &inconv2dbbb_v60_asm;
#endif
    int32_t out_row, d;
    minmax.vec[1] = Q6_V_vsplat_R(0x7FFFFFFF);
    minmax.vec[0] = Q6_V_vnot_V(minmax.vec[1]);

    // prefetch initial activations
    shortin_prefetch_input(work, start_line*stride_height, n_in_rows);

    for (out_row = start_line; out_row < stop_line; out_row += proc_rows) {

        vmemset_asm(integral_buf, 0, info->suma_width*4);

        wait_for_l2fetch();
        {
        	int32_t load_from = out_row*stride_height;
        	int32_t load_rows = n_in_rows;
        	if( out_row > start_line){	// skip the rows we already have from before
        		load_from += pf_offset;
        		load_rows -= pf_offset;
        	}
        	 shortin_prepare_indata(work, load_from , load_rows, ibatch);
        }

        ivint_asm( input,
                   integral_buf + info->suma_width,
                   info->in_width,
                   n_in_rows,
                   info->filt_offset,
                   integral_control);
#if 0
        logmsg(nn,2,"start=%d input_start=%p integral_buf=%p filt_offset=%d in_width=%d stride_height=%d num_lines=%d filt_height=%d values=%08x %08x %08x %08x,%08x %08x %08x %08x",
            out_row,input,integral_buf,info->filt_offset,
            info->in_width,
            info->stride_height,work->num_lines,info->filt_height,
            integral_buf[0],
            integral_buf[1],
            integral_buf[2],
            integral_buf[3],
            integral_buf[info->in_width+0],
            integral_buf[info->in_width+1],
            integral_buf[info->in_width+2],
            integral_buf[info->in_width+3]);
#endif
        gvsuma_asm(integral_buf, suma_buf,
                   info->suma_width, info->next_suma_off,  info->stride_height, 
                   info->filt_width, info->filt_height, n_lines, offset);
#if 0
        logmsg(nn,2,"start=%d suma_width=%d next_suma_off=%d suma_buf=%p-->%p integral_buf=%p num_lines=%d values=%08x %08x %08x %08x",
            out_row,
            info->suma_width,
            info->next_suma_off,
            suma_buf,
            suma_buf,
            integral_buf,
            n_lines, 
            suma_buf[0],
            suma_buf[1],
            suma_buf[2],
            suma_buf[3]);
#endif
        int32_t next_n_lines = Q6_R_min_RR(stop_line-out_row-proc_rows, proc_rows);
        int32_t next_n_in_rows = (next_n_lines-1)*stride_height + filt_height; 

        /* do conv2d */
        for (d = 0; d < info->out_depth_valid; d+=info->weight_batch_size) {

            if (next_n_lines > 0 && (d+info->weight_batch_size) >= info->out_depth_valid) {
                shortin_prefetch_input(work, (out_row+proc_rows)*stride_height + pf_offset, next_n_in_rows-pf_offset);
            }
#if 0
            logmsg(nn,2,"d=%d input=%p-->%p weights=%p-->%p filter_count=%d output=%p//%p out_next_row=%d biasbuf=%p recip_val=%x num_lines=%d filt_width=%d filt_height=%d in_depth=%d stride=%08x shamt=%d",
                d,
                work->input,
                input + info->in_depth*LPAD,
                work->weights,
                work->weights + d*filter_count,
                filter_count,
                output_batch,
                (char*)0,//output+d*info->out_next_d32/32,
                info->out_next_row,
                info->biasbuf + d,
                info->recip_val,
                n_lines,
                info->filt_width,
                info->filt_height,
                info->in_depth,
                Q6_R_combine_RlRl(info->stride_height,info->stride_width),
                info->recip_shamt);

            logmsg(nn,2,"start=%d, suma_buf=%p my_suma_ptr=%p values=%08x %08x %08x %08x",
                out_row, suma_buf,
                suma_buf+info->suma_start,
                suma_buf[info->suma_start+0],
                suma_buf[info->suma_start+1],
                suma_buf[info->suma_start+2],
                suma_buf[info->suma_start+3]);
#endif
            (*conv)(input + info->in_depth*LPAD, //adjust for left integral pad
                    work->weights + d*filter_count,
                    output_batch+d*info->out_next_d32/32u + info->out_next_row * out_row,
                    info->in_width,
                    info->out_next_row,
                    info->out_width,
                    info->in_depth,
                    info->filt_width,
                    info->filt_height,
                    n_lines,
                    minmax.words,
                    info->recip_val,
                    info->biasbuf + d,
                    suma_buf + info->suma_start,
                    info->next_suma_off,
                    Q6_R_combine_RlRl(info->stride_height,info->stride_width),
                    info->recip_shamt );

#if 0
            logmsg(nn,2,"oldmin=%d oldmax=%d found_min=%d found_max=%d",t_min,t_max,minmax.words[32],minmax.words[0]);

            logmsg(nn,2,"start=%d maxe: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
                out_row,
                minmax.words[64+0+0],minmax.words[64+0+1],minmax.words[64+0+2],minmax.words[64+0+3],
                minmax.words[64+4+0],minmax.words[64+4+1],minmax.words[64+4+2],minmax.words[64+4+3],
                minmax.words[64+8+0],minmax.words[64+8+1],minmax.words[64+8+2],minmax.words[64+8+3],
                minmax.words[64+12+0],minmax.words[64+12+1],minmax.words[64+12+2],minmax.words[64+12+3],
                minmax.words[64+16+0],minmax.words[64+16+1],minmax.words[64+16+2],minmax.words[64+16+3],
                minmax.words[64+20+0],minmax.words[64+20+1],minmax.words[64+20+2],minmax.words[64+20+3],
                minmax.words[64+24+0],minmax.words[64+24+1],minmax.words[64+24+2],minmax.words[64+24+3],
                minmax.words[64+28+0],minmax.words[64+28+1],minmax.words[64+28+2],minmax.words[64+28+3]);

            logmsg(nn,2,"start=%d mine: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
                out_row, 
                minmax.words[96+0+0],minmax.words[96+0+1],minmax.words[96+0+2],minmax.words[96+0+3],
                minmax.words[96+4+0],minmax.words[96+4+1],minmax.words[96+4+2],minmax.words[96+4+3],
                minmax.words[96+8+0],minmax.words[96+8+1],minmax.words[96+8+2],minmax.words[96+8+3],
                minmax.words[96+12+0],minmax.words[96+12+1],minmax.words[96+12+2],minmax.words[96+12+3],
                minmax.words[96+16+0],minmax.words[96+16+1],minmax.words[96+16+2],minmax.words[96+16+3],
                minmax.words[96+20+0],minmax.words[96+20+1],minmax.words[96+20+2],minmax.words[96+20+3],
                minmax.words[96+24+0],minmax.words[96+24+1],minmax.words[96+24+2],minmax.words[96+24+3],
                minmax.words[96+28+0],minmax.words[96+28+1],minmax.words[96+28+2],minmax.words[96+28+3]);
#endif
            //t_min = Q6_R_min_RR(t_min,minmax.words[32]);
            //t_max = Q6_R_max_RR(t_max,minmax.words[0]);
        }

        input  += proc_rows*stride_height*in_next_row;
        //output += proc_rows*info->out_next_row;
        n_lines   = next_n_lines;
        n_in_rows = next_n_in_rows;
    }
    gvrmaxmin(minmax.words);
    nn_atomic_min(&info->minval,minmax.words[32]);
    nn_atomic_max(&info->maxval,minmax.words[ 0]);
    
    nn_os_bufstack_push(&info->bufstack,integral_buf);
    nn_sem_post(work->donesem);
}

#define SHORTIN_WORKITEMS 8
/* NOTE: this returns:
 *    0   : OK
 *    1   : need to call again (retry)
 *    < 0 : error
 */

static int shortin_supernode_execute_everything(struct nn_graph *nn, void *vself)
{
	/* Since this can recurse, rewind our scratch allocator */
	nn_scratch_reset(nn);
	struct nn_node *self = vself;
	struct supernode_info_new *info = self->opaque;
	struct workitem input_work = {0};	// klockwork
	struct workitem filtwork[NUM_THREADS];
	//struct workitem zapwork;
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	//const struct tensor *min_in_tensor = self->inputs[2];
	//const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	//const struct tensor *bias_tensor = self->inputs[7];
	//const struct tensor *bias_min_tensor = self->inputs[8];
	//const struct tensor *bias_max_tensor = self->inputs[9];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	info->in_shape = in_tensor->shape;
	/*
	 * Since the input is not in D32 format, it doesn't have padding and stuff
	 */
	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;       //could be 1,2,3 or 4

        //printf(" wxh = %ld %ld %ld\n", in_width, in_height, (int32_t)self->padding);
	/*
	 * Filt information
	 */
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	//int32_t filt_depth = filt_tensor->shape.filt_depth;

	/*
	 * stride information
	 */
	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	/*
	 * Compute output dims 
	 */
	int32_t out_batches = in_batches;
	int32_t out_depth = filt_batches;

	/* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */
	int32_t required_w_before, required_h_before, required_h_after;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,filt_width,stride_width,self->padding, &required_w_before);
	int32_t out_width_pad = roundup(out_width, 4);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height,filt_height,stride_height,self->padding, &required_h_before, &required_h_after);
	required_w_before += LPAD;

	int32_t out_left_pad = 4;
	int32_t out_right_pad = out_width_pad-out_width;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31; // padding amount in case out depth != 32

	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;
	info->out_width = out_width_total - out_left_pad;
	info->out_depth_valid = out_depth_total;
	info->out_depth_total = out_depth_total;
	info->out_height = out_height;
	info->out_next_d32 = out_width_total * 32;
	info->out_next_row = out_width_total * out_depth_total;

	info->stride_width = stride_width;
	info->stride_height = stride_height;
	info->filt_width = filt_width;
	info->filt_height = filt_height;

	info->out_next_batch = info->out_next_row * out_height_total;



	//printf(" %ld, %ld\n", required_h_before, required_w_before);

	int32_t in_width_pad = required_w_before + in_width;
	int32_t next_in_width = roundup(in_width_pad+required_w_before, 32);
	if((stride_width*out_width_pad+filt_width-1+LPAD) >= next_in_width) next_in_width += 32; //prevent spurious max's
	int32_t required_w_after = next_in_width - in_width_pad;
	//printf(" lpad = %ld rpad = %ld\n", required_w_before, required_w_after);
	//printf(" %ld -> %ld %ld\n", in_width_pad, next_in_width, in_width_pad);

	//int32_t in_depth_before_pad = 0;
	//int32_t in_depth_after_pad = 0;
	int32_t in_depth_total = 4; //in_depth + in_depth_before_pad + in_depth_after_pad;

	info->weight_batch_size = 32;

	info->in_width = next_in_width;
	info->in_depth = in_depth_total; 
	info->in_next_row = next_in_width*in_depth_total;
	info->in_next_d32 = 0; //depth 4
	info->in_height = in_height + required_h_before + required_h_after;
	uint8_t *in = nn_scratch_alloc(nn,(2+required_h_before+info->in_height)*info->in_next_row);
	//info->input_base = in + (4-required_h_before)*info->in_next_row;
	info->input_base = in + (2+required_h_before)*info->in_next_row; // + (4-required_h_before)*info->in_next_row;
         
	info->integral_off = next_in_width;
	info->suma_width= info->in_width;
	info->next_suma_off= info->in_next_row;     //integral is same size as image
	info->suma_start= 1;
	logmsg(nn,2,"scratch=%p input_base=%p next_in_width=%d in_height=%d required_w_before=%d",in,info->input_base,next_in_width,info->in_height,required_w_before);

	// disallow gain > 1.0 (not any more, shortinv supports it now)
	//	info->recip_shamt_must_be_zero =1;

	// find input range, output scaling & range
	// Note: this may expand the output range
	if( fill_info_minmax_basics(nn,self,info) != 0) return -1;

	//float bias_to_prod_ratio = (bias_level_size / prod_level_size);
	float min_out_prod_offset;
	min_out_prod_offset = -info->out_minval / info->prod_level_size;
	//float prod_to_out_ratio = prod_level_size / out_level_size;

	info->weights_base = info->weights;
	info->raw_input = in_tensor->data;

	info->minval = 0;
	info->maxval = 0;
	/* Put more assumption checks here */
	if (in_depth > 4) return errlog(nn,"Make this code work for input depth > 4"); 
	if ((stride_width > 4)) return errlog(nn,"Add support for stride > 4");

	int fail = 0;
	if ((fail = tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8)) != 0) {
		return errlog(nn,"output tensor prep fail: %d", fail);
	}
	if (tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"minout prep fail");
	}
	if (tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"maxout prep fail");
	}

	tensor_set_float(out_min_tensor,0,info->out_minval);
	tensor_set_float(out_max_tensor,0,info->out_maxval);

	// prefetch weights
	l2fetch(info->weights, 128, 128, filt_width*filt_height*out_depth_total/32);

	/* Compute bias buffer */
	/* 
	 * We will need to recompute the bias buffer any time the input offset changes
	 */
	int bias32 = (self->node_type == OP_InputSupernode_8x8p32to8_outd32);
	if (fill_bias_buf(nn,self,info,bias32,0) != 0)
		return -1;

	nn_sem_t donesem;

	nn_sem_init(&donesem,0);

	//work load set up 
//	input_work.zap_top = (uint8_t *)info->input_base - (1)*info->in_next_row;;
//	input_work.zap_bot = (uint8_t *)info->input_base + (required_h_before + in_height)*info->in_next_row;
#if 0
	work.zap_top_size = info->in_next_row * (required_h_before + 1);
	work.zap_bot_size = work.zap_top_size ; 
#else
	input_work.zap_top_size = (required_h_before + 1);
	input_work.zap_bot_size = required_h_after ; 
#endif
	//logmsg(nn,2,"zap top size=%d zap bot size=%d",work.zap_top_size,work.zap_bot_size);
	input_work.zap_left = (uint8_t *)info->input_base + info->in_next_row * required_h_before;
//	input_work.zap_right = input_work.zap_left + (info->in_width - required_w_after) * info->in_depth;
//	input_work.zap_rl_depths = 0; //depth 4 so n.a.
	input_work.zap_left_size = required_w_before*info->in_depth;
	input_work.zap_right_size = required_w_after*info->in_depth;
//	input_work.zap_height = in_height;
//	input_work.nonzap_width = in_width;
//	input_work.zap_value = input_offset;
	/* Fill out work information */
	input_work.info = info;
	input_work.self = self;
	input_work.input = info->input_base;
	input_work.output = tensor_location_bhw_d32(out_tensor,0,0,0); //point to line 0, depth 0, elem 0
	input_work.weights = info->weights;
	input_work.biases = info->biasbuf;
//	input_work.start_line = 0;
//	input_work.stop_line = out_height;
//	input_work.num_lines = out_height;
//	input_work.skip_lines = 1;
	input_work.donesem = &donesem;

//	nn_os_vector_call(nn,shortin_supernode_prepare_input,&input_work);

	int32_t inner_act_rows  = (out_height+NUM_THREADS-1)/(NUM_THREADS);
	int32_t outer_act_iters = (out_height + inner_act_rows - 1)/inner_act_rows;
	int32_t slice = Q6_R_max_RR(out_height/(2*SHORTIN_WORKITEMS), 1);
	int i;

	nn_os_bufstack_init(&info->bufstack);
	for (i = 0; i < NUM_THREADS; i++) {
		void *tmp;
		int int_buf_rows = slice*info->stride_height+info->filt_height+1;
		int int_buf_size = (4*int_buf_rows*info->suma_width + 127)&(~127);
		if ((tmp = nn_scratch_alloc(nn,int_buf_size)) == NULL) {
			return errlog(nn,"scratch fail");
		}
		nn_os_bufstack_push(&info->bufstack,tmp);
	}

	int32_t buf_size =  roundup(info->suma_width*(slice+1)*4,128);
	int32_t *suma_buf = nn_scratch_alloc(nn,outer_act_iters*buf_size);	// FIXME: scratch alloc

	if (suma_buf == NULL) {
		return errlog(nn,"scratch fail");
	}


	for( int ibatch = 0; ibatch < in_batches; ibatch++){
		input_work.batch_index = ibatch;

		for (i = 0; i < outer_act_iters; i++) {
			filtwork[i] = input_work;
			filtwork[i].start_line = i*inner_act_rows;
			filtwork[i].stop_line  = Q6_R_min_RR((i+1)*inner_act_rows, out_height);
			filtwork[i].suma_buf = suma_buf + i*buf_size/sizeof(int32_t);
			filtwork[i].num_lines = slice;
			nn_os_work_for_vector(nn,shortin_supernode_execute_hvx_slice,&filtwork[i]);
		}
		nn_sem_wait_n_times(&donesem, outer_act_iters);
	}
	// check input/output ranges; if needed, adjust range and return 1. If OK, return 0.
	// shortin conv now does min/max in output units, not sum-of-product units.

	int need_redo = 0;
	int fixed_flags = info->minmax_precalc_flags;				// parm for adjust_minmax_for_zero_with_constraints
	float ominval = info->out_minval;	// for transforming minval,maxval to app range
	float output_level_size = info->output_level_size;
	
	if( (fixed_flags&2)==0){	// max not fixed
		if( info->maxval > 255){
			float app_maxval = output_level_size*(info->maxval + 0.5f)+ ominval;
			info->out_maxval = round_up_quarter_octave( fmaxf( app_maxval, 0x1.0p-4f));

			logmsg(nn,1,"Maxval too small (%x > 255), retrying... output_level_size=%f new outmax = %f (>= %f)",
					info->maxval,output_level_size,info->out_maxval, app_maxval);

			need_redo = 1;
		}
	}
	if((fixed_flags&1)==0){		// min not fixed
		if( info->minval < 0){		// i.e. if < 0 ..
			float app_minval = output_level_size *(info->minval-0.5f) + ominval;
			info->out_minval = round_up_quarter_octave( fminf( app_minval, -0x1.0p-8f));

			logmsg(nn,1,"Minval too large (%x < 0), retrying... output_level_size=%f new outmin = %f (<= %f)",
					info->minval,output_level_size,info->out_minval, app_minval);

			need_redo = 1;

		}
	}
	if( need_redo ){
		// restore 'fixed' endpoints to original spec (in case they were tweaked by previous endpoint corrs)
		if( fixed_flags & 1) info->out_minval = info->out_minval_spec;
		if( fixed_flags & 2) info->out_maxval = info->out_maxval_spec;
		adjust_minmax_for_zero_with_constraints( & info->out_minval, &info->out_maxval, fixed_flags);
		logmsg(nn,1, "adjusted output range: %f ... %f", info->out_minval, info->out_maxval );
	}
	return need_redo;
}

/*
 * This executes on a scalar thread
 * 
 * Make sure everything is OK, then spawn off work onto the vector threads.
 */
static int shortin_supernode_execute_call(struct nn_node *self, struct nn_graph *nn)
{
	//return nn_os_vector_call(nn,shortin_supernode_execute_everything,self);
	int count = 0;
	while(count < 3){
		int res = shortin_supernode_execute_everything(nn,self);
		if( res <= 0) return res;
		++ count;
	}
	logmsg(nn,0,"Extreme recursion detected in shortin_supernode, problem finding min/max?");
	return -1;
}

//special rearrangement of weights insert filt offset in gaps
static void shortin_rearrange_weights_Ndto4(
  uint8_t* in_filt, int filter_count, int in_depth, int filt_batches, int out_depth, uint8_t* out_filt, int filt_offset)
{
	int v,w,x,y,z;
	int inval;

	for (x = 0; x < out_depth; x+=32) {
		for (w=0,y = 0; y < filter_count; y+=4,w+=in_depth) {
			for (z = 0; z < 32; z+=1) {
				//in data is depth 1-4 so address neededs converting
				for(v=0; v < 4; v++) {
					if ((v < in_depth) && ((x+z) < filt_batches)) {
						inval = in_filt[filt_batches*(w+v)+x+z];
					} else {
						inval = filt_offset;
					}
					out_filt[filter_count*x + 32*y + 4*z+v] = inval;
				}
			}
		}
	}
	return;
}

//create base vector for computing new bias with inoffset
static void shortin_filt_sumb(uint8_t * filt_trans, int out_depth, int filter_count, int32_t * filt_sum) {
    int i, j, k;
    int32_t sumw;

    for (i=0; i < out_depth; i+=32) {
      for (j=0; j < 32; j++) {
         sumw = 0;
         for (k=0; k < filter_count/4; k++) {
           sumw += filt_trans[i*filter_count+128*k+4*j+0];
           sumw += filt_trans[i*filter_count+128*k+4*j+1];
           sumw += filt_trans[i*filter_count+128*k+4*j+2];
           sumw += filt_trans[i*filter_count+128*k+4*j+3];
         }
         filt_sum[i+j] = sumw;
      }
    }
    return;
}

static int shortin_supernode_check(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info_new *info = self->opaque;
	//
	// shortin supernode may wind up with 13 inputs due to ChannelScale - but by
	// the time we get to here, it should be dealt with and will be just a scalar [1.0].
	// so we can ignore it.
	// inputs = 12 or 13; outputs = 3; checked in ctor

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_batches_roundup = (filt_batches + 31) & ~31;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	int32_t filt_depth_roundup = (filt_depth + 3) & ~3;
	uint32_t filt_elements = filt_width * filt_height * filt_depth_roundup;
	uint32_t weight_batch_size = filt_elements * 32;
	uint32_t weights_size = filt_elements * filt_batches_roundup;
	uint32_t out_depth = filt_batches_roundup;
	int32_t in_depth = in_tensor->shape.depth;       //could be 1,2,3 or 4
	float specified_minval = tensor_get_float(self->inputs[10],0);
	float specified_maxval = tensor_get_float(self->inputs[11],0);
        const struct tensor *min_filt_tensor = self->inputs[4];
        const struct tensor *max_filt_tensor = self->inputs[5];
        uint8_t *filt = filt_tensor->data;
        float filt_max_float = tensor_get_float(max_filt_tensor,0);
        float filt_min_float = tensor_get_float(min_filt_tensor,0);
        float filt_level_size;
        int32_t filt_offset = get_qu8_level_size_zero(filt_min_float,filt_max_float, &filt_level_size);
	int32_t out_height_max = self->output_defs[0].max_sizes[1];
	int32_t out_width_max = self->output_defs[0].max_sizes[2];
	const struct tensor *stride_tensor = self->inputs[6];
	int32_t stride_height = stride_tensor->shape.height;
	int32_t stride_width = stride_tensor->shape.height;
	int32_t input_height = (out_height_max+1) * stride_height + filt_height;
	int32_t input_width = (out_width_max+1) * stride_width + filt_width;
	int32_t input_depth = filt_depth_roundup;
	int32_t input_size = input_height*input_width*input_depth + 256;
	logmsg(nn,2,"estimating input %dx%dx%d, growing scratch to %d",input_height,input_width,input_depth,input_size*6);
	//nn_scratch_grow(nn,input_size*6+input_height*input_width*4*SHORTIN_WORKITEMS);
	nn_scratch_grow(nn,input_size*6+input_height*input_width*4*NUM_THREADS);


	/* At check time or ctor time, we need to allocate all the intermediate storage we will need */

	weights_size = 1U << (32-__builtin_clz(weights_size-1));
	if (info != NULL) {
		/* Already set up, invalidate strategy and return */
		info->strategy_valid = 0;
		return 0;
	}
	if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
		return errlog(nn,"couldn't allocate info");
	}
	if ((info->minmax_buf = nn_memalign(128,64*sizeof(int))) == NULL) {
		nn_free(info);
		return errlog(nn,"malloc/memalign");
	}
	if ((info->weights = nn_memalign(128,weights_size)) == NULL) {
		nn_free(info->minmax_buf);
		nn_free(info);
		return errlog(nn,"alloc weights");
	}
	if ((info->biasbuf = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info);
		return errlog(nn,"alloc biasbuf");
	}
	if ((info->gemsumb = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		nn_free(info->biasbuf);
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info);
		return errlog(nn,"alloc biasbuf");
	}
        //printf("max buf = %p\n", info->minmax_buf);
	logmsg(nn,2,"filt_elements=%d in_depth=%d out_depth=%d filt_batches=%d",
		filt_elements,in_depth,out_depth,filt_batches);
        shortin_rearrange_weights_Ndto4(filt, filt_elements, in_depth, filt_batches, out_depth, info->weights, filt_offset);
	/* Precalculate gemsumb */
        shortin_filt_sumb(info->weights, out_depth, filt_elements, info->gemsumb);
        info->weights_level_size = filt_level_size;
        info->weights_offset = 0;
	info->filt_offset = filt_offset;

	info->max_k_factor = 1.0f;

#if 0
	if ((info->semaphores = nn_calloc(n_weight_batches,sizeof(nn_sem_t))) == NULL) {
		nn_free(info->biasbuf);
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info);
		return errlog(nn,"alloc semaphores");
	}
	for (i = 0; i < n_weight_batches; i++) {
		nn_sem_init(&info->semaphores[i],0);
	}
#endif
	info->strategy_valid = 0;	/* Redundant w/ calloc */
	info->weight_batch_size = weight_batch_size;
	self->opaque = info;

	if(setup_initial_output_range( nn, info, specified_minval, specified_maxval, -1.0f/128, 1.0f/128)) return -1;

	return 0;
}


static int shortin_supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info_new *info = self->opaque;
	if (info != NULL) {
		supernode_reset_work_items(self,nn,info);
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info->biasbuf);
		nn_free(info->gemsumb);
		nn_free(info);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}

//
// shortin supernode may wind up with 13 inputs due to ChannelScale - but by
// the time we get to here, it should be dealt with and will be just a scalar [1.0].
// so we can ignore it.

struct nn_node_ops nn_ops_for_InputSupernode_8x8p8to8_outd32 = {
	.execute = shortin_supernode_execute_call,
	.check = shortin_supernode_check,
	.ctor = node_alloc_common,
	.dtor = shortin_supernode_dtor,
	.flags = NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_InputSupernode_8x8p32to8_outd32 = {
	.execute = shortin_supernode_execute_call,
	.check = shortin_supernode_check,
	.ctor = node_alloc_common,
	.dtor = shortin_supernode_dtor,
	.flags = NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),
};

#endif
