
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for convolution
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef __hexagon__
#include <malloc.h>
#else
#endif

#if defined(__hexagon__)
static int32_t max(int a, int32_t b) { return((a>b) ? a : b); }
#endif

/* 8x8 convolution --> 32 bits */


#define ALIGN_SIZE 128
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((ALIGN_SIZE)-1)))
static inline void *pad_and_align(void *ptr, unsigned long minsize)
{
	uintptr_t ptrval = (uintptr_t)(ptr);
	ptrval += minsize + (ALIGN_SIZE-1);
	ptrval &= ~(ALIGN_SIZE-1);
	return (void *)ptrval;
}


static int __attribute__((unused)) conv2d_execute_ref_im2col(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
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
	int32_t adj_x, adj_y;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,filt_width,stride_width,self->padding, & adj_x);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,filt_height,stride_height,self->padding, & adj_y);
	int32_t out_depth = filt_batches;

	int32_t batch;
	int32_t out_x;
	int32_t out_y;
	int32_t out_z;

	int32_t in_y_base;
	int32_t in_x_base;

	uint8_t *in = in_tensor->data;
	uint8_t *filt = filt_tensor->data;
	int32_t *out = out_tensor->data;

	uint8_t *filtstripe;
	int32_t *outstripe;

	int32_t in_element;
	int32_t filt_element;
	int32_t sum;

	int32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements*sizeof(int32_t);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float in_level_size = (in_max_float - in_min_float) / 255;
	float filt_level_size = (filt_max_float - filt_min_float) / 255;
	float out_level_size = in_level_size * filt_level_size;

	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t input_offset = quantize_int(0.0f,in_min_float,in_max_float);
	int32_t filt_offset = quantize_int(0.0f,filt_min_float,filt_max_float);

	uint8_t *im2col_row;
	int32_t i;
	int32_t filt_total_length = filt_depth*filt_width*filt_height;

	logmsg(nn,2,"conv2d execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"conv2d input min/max=%f/%f",self,in_min_float,in_max_float);
	logmsg(nn,2,"conv2d input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
	logmsg(nn,2,"conv2d filt %dx%dx%dx%d",filt_batches,filt_height,filt_width,filt_depth);
	logmsg(nn,2,"conv2d stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"conv2d padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	logmsg(nn,2,"expect %lld MACs",(long long int)out_batches*out_height*out_width*out_depth*in_depth);
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if (out_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",out_tensor->max_size,out_size);
	}
	if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");
	if (out_min->max_size < sizeof(float)) return errlog(nn,"min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn,"max too small");
	if (self->padding == NN_PAD_NA) return errlog(nn,"This op might pad");

	if ((im2col_row = nn_malloc(filt_total_length)) == NULL) return errlog(nn,"tmp data storage fail");
	logmsg(nn,3,"malloced im2col row %p",im2col_row);

	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_size;

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,out_max_val);
	out_max->data_size = sizeof(float);

	/* We'll do one im2col followed by one multiply with filter */
	/* For optimization, you may wish to consider more im2col at a time. */

	for (batch = 0; batch < out_batches; batch++) {
	  for (out_y = 0; out_y < out_height; out_y++) {
	    in_y_base = out_y * stride_height - adj_y;
	    for (out_x = 0; out_x < out_width; out_x++) {
	      in_x_base = out_x * stride_width - adj_x;
	      im2col_stripe(im2col_row,
			in+(in_height*in_width*in_depth*batch),
			in_x_base,
			in_width,
			filt_width,
			in_y_base,
			in_height,
			filt_height,
			filt_depth,
			input_offset);
	      outstripe = out+(out_depth*(out_x+
	                       out_width*(out_y+
	                       out_height*(batch))));
	      // For each im2col-generated row of input data, 
	      // dot product by each filter
	      for (out_z = 0; out_z < out_depth; out_z++) {
	        // For each filter, start a new sum and get a base ptr
	        sum = 0;
	        filtstripe = filt+out_z;
	        for (i = 0; i < filt_total_length; i++) {
	          // Dot product inner loop
	          in_element = im2col_row[i];
		  // Note: coefficients for different filters are contiguous
	          filt_element = filtstripe[i*out_depth];
	          in_element -= input_offset;
	          filt_element -= filt_offset;
	          sum += in_element*filt_element;
	        }
	        outstripe[out_z] = sum;
	      }
	    }
	  }
	}
	logmsg(nn,3,"freeing im2col row %p",im2col_row);
	nn_free(im2col_row);
	logmsg(nn,2,"conv2d execute (im2col ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}
/*
 * Input and output have ordering BHWD
 * Filter has ordering HWDB (B is # of filters)
 */

static int conv2d_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
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
	int32_t adj_x, adj_y;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,filt_width,stride_width,self->padding, &adj_x);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,filt_height,stride_height,self->padding, & adj_y);
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
	int32_t *out = out_tensor->data;

	uint8_t *instripe;
	uint8_t *filtstripe;
	int32_t *outstripe;

	int32_t in_element;
	int32_t filt_element;
	int32_t sum;

	int32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements*sizeof(int32_t);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);


	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float in_level_size = (in_max_float - in_min_float) / 255;
	float filt_level_size = (filt_max_float - filt_min_float) / 255;
	float out_level_size = in_level_size * filt_level_size;

	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t input_offset = quantize_int(0.0f,in_min_float,in_max_float);
	int32_t filt_offset = quantize_int(0.0f,filt_min_float,filt_max_float);

	logmsg(nn,2,"conv2d execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"conv2d input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
	logmsg(nn,2,"conv2d filt %dx%dx%dx%d",filt_batches,filt_height,filt_width,filt_depth);
	logmsg(nn,2,"conv2d stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"conv2d padding %d",self->padding);
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

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,out_max_val);
	out_max->data_size = sizeof(float);

	for (batch = 0; batch < out_batches; batch++) {
	  for (out_y = 0; out_y < out_height; out_y++) {
	    in_y_base = out_y * stride_height - adj_y;
	    for (out_x = 0; out_x < out_width; out_x++) {
	      in_x_base = out_x * stride_width - adj_x;
	      outstripe = out+(out_depth*(out_x+
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
	if (0 && out_z == 10) logmsg(nn,9,"[%d %d %d %d]: sum += %d*%d --> %d [%d %d %d] [ioff: -%d foff: -%d]",
		batch,out_y,out_x,out_z,in_element,filt_element,sum,
		filt_y,filt_x,filt_z,input_offset,filt_offset);
	            }
	          }
	        }
	        outstripe[out_z] = sum;
	      }
	    }
	  }
	}
	logmsg(nn,2,"conv2d execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}
#ifdef __hexagon__

#define VPAD 8
#define HPAD 16
#define DPAD 32



//
// NOTE:
//  -this function is unused
//  -it calls nn_scratch_grow in an execute method

static int __attribute__((unused)) conv2d_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
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
	int32_t adj_x, adj_y;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,filt_width,stride_width,self->padding, &adj_x);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,filt_height,stride_height,self->padding,&adj_y);
	int32_t out_depth = filt_batches;

	int32_t batch;

//	int32_t in_y_base;
//	int32_t in_x_base;

	uint8_t *in = in_tensor->data;
	uint8_t *filt = filt_tensor->data;
	int32_t *out = out_tensor->data;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements*sizeof(int32_t);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float in_level_size = (in_max_float - in_min_float) / 255;
	float filt_level_size = (filt_max_float - filt_min_float) / 255;
	float out_level_size = in_level_size * filt_level_size;

	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t input_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);

	logmsg(nn,2,"conv2d execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"conv2d input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
	logmsg(nn,2,"conv2d filt %dx%dx%dx%d",filt_batches,filt_height,filt_width,filt_depth);
	logmsg(nn,2,"conv2d stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"conv2d padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if (out_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d @ %x",out_tensor->max_size,out_size,self->node_id);
	}
	if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");
	if (out_min->max_size < sizeof(float)) return errlog(nn,"min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn,"max too small");

	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_size;

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,out_max_val);
	out_max->data_size = sizeof(float);

        /* intermediate buffer generation */
        int patches = out_height*out_width;
        int patches_pad = (patches+VPAD-1)&~(VPAD-1);
        int out_depth_pad = (out_depth + DPAD - 1) & ~(DPAD-1);
        int32_t filter_value_count = filt_width*filt_height*filt_depth; //aka K 
        int32_t filter_value_count_pad = (filter_value_count+(HPAD-1))&~(HPAD-1); //K rounding
        // filter count has to be multiple of 16 and should have minimum value of 32 for gemmpybbw_asm to work properly
        filter_value_count_pad = max(32,filter_value_count_pad);
	int32_t im2col_buf_size = (patches_pad*filter_value_count_pad);
	int32_t minmax_size = sizeof(int)*64;
	int32_t suma_size = patches_pad*sizeof(int);
	int32_t sumb_size = out_depth_pad*sizeof(int);
	int32_t filt_pad_size = filter_value_count_pad * out_depth_pad;
	int32_t filt_pad_trans_size = filt_pad_size;
	int32_t out_pad_size = out_depth_pad * patches_pad * sizeof(int) + 128;

	if(nn_scratch_grow(nn,ROUNDUP(im2col_buf_size) + ROUNDUP(minmax_size) + ROUNDUP(suma_size)
		+ ROUNDUP(sumb_size) + ROUNDUP(filt_pad_trans_size) + ROUNDUP(filt_pad_size) + ROUNDUP(out_pad_size))){
		return errlog(nn,"failed to get scratch");
	}

        uint8_t* im2col_buf = nn->scratch;
        int *minmax = (int *) pad_and_align(im2col_buf, im2col_buf_size);
        int * suma = (int *) pad_and_align(minmax, minmax_size);
        int * sumb = (int *) pad_and_align(suma, suma_size);
        uint8_t* filt_pad = (uint8_t*)pad_and_align(sumb, sumb_size);
        uint8_t* filt_pad_trans = (uint8_t*)pad_and_align(filt_pad, filt_pad_size);
        int* out_pad = (int*)pad_and_align(filt_pad_trans, filt_pad_trans_size);
		//printf("CCCCC alloc size scratch addr %p, : scratch_size: %d",nn->scratch, nn->scratch_size);
        /* pad out the filter weights matrix to M x K */
	/* Zero out output since we accumulate with it */
	memset(out_pad,0,out_pad_size);
	logmsg(nn,2,"im2col_buf_size = %d @ %p\n",patches_pad * filter_value_count_pad,im2col_buf);
	logmsg(nn,2,"filt_pad = %d @ %p\n",out_depth_pad * filter_value_count_pad,filt_pad);
	logmsg(nn,2,"out_pad = %d @ %p\n",out_depth_pad * patches_pad * 4,out_pad);
        pad2d(filt, filter_value_count, out_depth,
              filt_pad, filter_value_count_pad, out_depth_pad, filt_offset);
        transpack(filt_pad, filter_value_count_pad, out_depth_pad, filt_pad_trans) ;

	for (batch = 0; batch < out_batches; batch++) {

          /*pad data matrix horizontally to tuples of HPAD */
          im2col_cn(&in[batch*in_height*in_width*in_depth], in_height,in_width,in_depth, input_offset,
                    im2col_buf, filt_height, filt_width, stride_width,
                    out_height, out_width, adj_x, adj_y);

          gemm_asm (im2col_buf,     -input_offset,
                    filt_pad_trans, -filt_offset,
                    out_pad,
                    patches_pad, out_depth_pad, filter_value_count_pad, //N M K
                    patches_pad, 32           , filter_value_count_pad, suma, sumb, minmax); 

          //int gmax = minmax[0];
          //int gmin = minmax[32];
          //printf(" gemm max min %d %d\n", gmax, gmin);

          /* strip out the padding from the output */
          unpad2d(out_pad, patches_pad, out_depth_pad,
                  (void *)(&out[batch*patches*out_depth]), patches, out_depth);
	}//end batch
		//free(temp32);
	logmsg(nn,2,"conv2d execute (hvx) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}
#endif

#if 0
static inline void logmsg_input(
	struct nn_graph *nn,
	int logval,
	int index,
	const struct tensor *tens)
{
	logmsg(nn,logval,"input %d: BHWD=%d,%d,%d,%d data %d bytes @ %p",
		index,
		tens->shape.batches,
		tens->shape.height,
		tens->shape.width,
		tens->shape.depth,
		tens->data_size,
		tens->data);
}

static int conv2d_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking conv2d node %p",self);
	if (self->n_inputs != 7) return errlog(nn,"conv2d id %x wrong # inputs",self->node_id);
	if (self->n_outputs != 3) return errlog(nn,"conv2d wrong # outputs");
	if (self->inputs == NULL) return errlog(nn,"NULL inputs");
	if (self->outputs == NULL) return errlog(nn,"NULL outputs");
	for (i = 0; i < self->n_inputs; i++) {
		if (self->inputs[i] == NULL) {
			return errlog(nn,"input %d NULL",i);
		}
	}
	for (i = 0; i < self->n_outputs; i++) {
		if (self->outputs[i] == NULL) {
			return errlog(nn,"output %d NULL",i);
		}
	}
	logmsg(nn,3,"conv2d node %p inputs: "
		"[in, filt, min_in, max_in, min_filt, max_filt, stride]:",
		self);
	for (i = 0; i < self->n_inputs; i++) {
		logmsg_input(nn,3,i,self->inputs[i]);
	}
	logmsg(nn,2,"conv2d node %p check OK",self);
	return 0;
}
#endif

struct nn_node_ops nn_ops_for_QuantizedConv2d_8x8to32 = {
#if 1
	.execute = conv2d_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
#else
	.execute = conv2d_execute_ref_im2col, // <-- not working yet
	//.execute = conv2d_execute_ref,
	.check = conv2d_check_ref,
	.ctor = conv2d_ctor,
	.dtor = node_free_common,
#endif
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedConv2d_8x8to32_ref = {
	.execute = conv2d_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

/*
 * Input and output have ordering BHWD
 * Filter has ordering HWDB (B is # of filters)
 */

static int conv2d_u16_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
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
	int32_t adj_x, adj_y;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width, filt_width, stride_width, self->padding, &adj_x);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height, filt_height, stride_height, self->padding, &adj_y);
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

	uint16_t *in = in_tensor->data;
	uint16_t *filt = filt_tensor->data;
	int32_t *out = out_tensor->data;

	uint16_t *instripe;
	uint16_t *filtstripe;
	int32_t *outstripe;

	int32_t in_element;
	int32_t filt_element;
	int64_t sum;

	int32_t out_elements = out_batches * out_height*out_width*out_depth;
	size_t out_size = out_elements * sizeof(int32_t);

	float in_max_float = tensor_get_float(max_in_tensor, 0);
	float in_min_float = tensor_get_float(min_in_tensor, 0);
	float filt_max_float = tensor_get_float(max_filt_tensor, 0);
	float filt_min_float = tensor_get_float(min_filt_tensor, 0);


	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float in_level_size = (in_max_float - in_min_float) / 65536;
	float filt_level_size = (filt_max_float - filt_min_float) / 65536;
	uint32_t mul_per_out = filt_depth * filt_width * filt_height;
	uint32_t scale = 31 - __builtin_clz(mul_per_out);
	if (mul_per_out & ((1 << scale) - 1)) scale++; // take the ceiling
	float out_level_size = in_level_size * filt_level_size * (1<<scale);

	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t input_offset = quantize_uint16(0.0f, in_min_float, in_max_float);
	int32_t filt_offset = quantize_uint16(0.0f, filt_min_float, filt_max_float);

	logmsg(nn, 2, "conv2d execute. node=%p id=%x", self, self->node_id);
	logmsg(nn, 2, "conv2d input %dx%dx%dx%d", in_batches, in_height, in_width, in_depth);
	logmsg(nn, 2, "conv2d filt %dx%dx%dx%d", filt_batches, filt_height, filt_width, filt_depth);
	logmsg(nn, 2, "conv2d stride %dx%d", stride_height, stride_width);
	logmsg(nn, 2, "conv2d padding %d", self->padding);
	logmsg(nn, 2, "expected out shape %dx%dx%dx%d", out_batches, out_height, out_width, out_depth);
	if (in_depth != filt_depth) return errlog(nn, "oops, depth != depth");
	if (out_size > (out_tensor->max_size)) {
		return errlog(nn, "output too small, %d < %d", out_tensor->max_size, out_size);
	}
	if (stride_tensor->shape.batches != 1) return errlog(nn, "bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn, "bad stride depth");
	if (out_min->max_size < sizeof(float)) return errlog(nn, "min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn, "max too small");

	tensor_set_shape(out_tensor, out_batches, out_height, out_width, out_depth);
	out_tensor->data_size = out_size;

	tensor_set_shape(out_min, 1, 1, 1, 1);
	tensor_set_float(out_min, 0, out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max, 1, 1, 1, 1);
	tensor_set_float(out_max, 0, out_max_val);
	out_max->data_size = sizeof(float);

	for (batch = 0; batch < out_batches; batch++) {
		for (out_y = 0; out_y < out_height; out_y++) {
			in_y_base = out_y * stride_height - adj_y;
			for (out_x = 0; out_x < out_width; out_x++) {
				in_x_base = out_x * stride_width - adj_x;
				outstripe = out + (out_depth*(out_x +
					out_width * (out_y +
						out_height * (batch))));
				for (out_z = 0; out_z < out_depth; out_z++) {
					sum = 0;
					for (filt_y = 0; filt_y < filt_height; filt_y++) {
						if ((in_y_base + filt_y) >= in_height) continue;
						if ((in_y_base + filt_y) < 0) continue;
						for (filt_x = 0; filt_x < filt_width; filt_x++) {
							if ((in_x_base + filt_x) >= in_width) continue;
							if ((in_x_base + filt_x) < 0) continue;
							instripe = in + (in_depth*(in_x_base + filt_x +
								in_width * (in_y_base + filt_y +
									in_height * (batch))));
							filtstripe = filt + (out_z +
								out_depth * filt_depth*(filt_x +
									filt_width * (filt_y)));
							for (filt_z = 0; filt_z < filt_depth; filt_z++) {
								in_element = instripe[filt_z];
								filt_element = filtstripe[filt_z*out_depth];
								in_element -= input_offset;
								filt_element -= filt_offset;
								sum += (int64_t)in_element * filt_element;
							}
						}
					}
					outstripe[out_z] = min_u32(0xffffffff, max_u32(0, (sum + (scale? (1<<(scale-1)):0)) >> scale));
				}
			}
		}
	}
	logmsg(nn, 2, "conv2d execute (ref) done! %dx%dx%dx%d",
		out_batches, out_height, out_width, out_depth);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedConv2d_16x16to32 = {
	.execute = conv2d_u16_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};