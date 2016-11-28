
/*
 * Copyright (c) 2016, The Linux Foundation. All rights reserved.
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

#include <vpred3332.h>
#include <vpred7732.h>
#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef __hexagon__
#include <malloc.h>
#endif

/* 8x8 convolution --> 32 bits, biasadd, relu, quantizedown to 8 bits  */

/*
 * Input and output have ordering BHWD
 * Filter has ordering HWDB (B is # of filters)
 */

#ifdef __hexagon__
#include <hexagon_standalone.h>
#include <hexagon_protos.h>

struct tdata {
        struct nn_node *self;
        int whoami;
        int mat_max;
        void * iptr;
        void * optr;
        int    arg0;
        int    arg1;
        int    arg2;
        int    arg3;
        int    arg4;
        nn_sem_t donesem;
};

#define RESET_PMU() __asm__ __volatile__ (" r0 = #0x48 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DUMP_PMU() __asm__ __volatile__ (" r0 = #0x4a ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")

#define DISABLE_PMU() __asm__ __volatile__ (" r0 = #0x42 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define ENABLE_PMU() __asm__ __volatile__ (" r0 = #0x41 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")


#define VPAD 8
#define HPAD 16
#define DPAD 32

#define K_CHUNKSIZE (768)
#define DATA_PREFETCH_BLOCK (72*1024)


static inline int rounddn(int x, int pad) { x = x  & ~(pad-1); return(x); }

//void l2pref(void *, int height, int width, int stride);


#endif

#define ALIGN_SIZE 128
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((X)-1)))
#define MAXPAD (ALIGN_SIZE)
static inline void *pad_and_align(void *ptr, unsigned long minsize)
{
	uintptr_t ptrval = (uintptr_t)(ptr);
	ptrval += minsize + (MAXPAD-1);
	ptrval &= ~(ALIGN_SIZE-1);
	return (void *)ptrval;
}

static inline void __attribute__((unused)) l2prefetch(void *ptr, int height, int width, int stride)
{
        if (width != stride) {
                //printf(" doing 2D prefetch of H=%d W=%d Stride=%d\n",height,width,stride);
                l2pref(ptr,height,width,stride);
                return;
        }

//        printf("called l2prefetch with h=%d w=%d stride=%d\n",height,width,stride);

        height = ((height*width + 63)/64);
        l2pref(ptr,height,64,64);
//printf("   single l2fetch is getting %d 64B lines @ %x\n",height,ptr);
}

/* hack for sim fastforward */
int supernode_count=0;

static int supernode_execute_ref(struct nn_node *self, struct nn_graph *nn)
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

	uint32_t in_batches = in_tensor->shape.batches;
	uint32_t in_width = in_tensor->shape.width;
	uint32_t in_height = in_tensor->shape.height;
	uint32_t in_depth = in_tensor->shape.depth;

	uint32_t filt_batches = filt_tensor->shape.byidx[0];
	uint32_t filt_height = filt_tensor->shape.byidx[3];
	uint32_t filt_width = filt_tensor->shape.byidx[2];
	uint32_t filt_depth = filt_tensor->shape.byidx[1];

	uint32_t stride_width = stride_tensor->shape.width;
	uint32_t stride_height = stride_tensor->shape.height;

	uint32_t out_batches = in_batches;
	uint32_t out_width = nn_pad_compute_outsize(in_width,filt_width,stride_width,self->padding);
	uint32_t out_height = nn_pad_compute_outsize(in_height,filt_height,stride_height,self->padding);
	uint32_t out_depth = filt_batches;

	uint32_t batch;
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
	int32_t maxsum = 0;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements;
	/* FIXME: if you pad depth you should adjust tmp_out_size here!!! */
	size_t tmp_out_size = out_elements*sizeof(int32_t);
	size_t biasbuf_size = out_depth*sizeof(int32_t);

	int32_t *biasbuf = nn->scratch;
	int32_t *tmp_out = pad_and_align(biasbuf,biasbuf_size);

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

	logmsg(nn,2,"supernode execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"supernode input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
	logmsg(nn,2,"supernode filt %dx%dx%dx%d",filt_batches,filt_height,filt_width,filt_depth);
	logmsg(nn,2,"supernode stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"supernode padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if ((tmp_out_size + biasbuf_size + MAXPAD) > nn->scratch_size) {
		return errlog(nn,"scratch too small (%d>%d)",tmp_out_size,nn->scratch_size);
	}
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
		biasbuf[i] = roundf((biasval - bias_offset) * bias_mpy_amt);
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
	        if (sum > maxsum) maxsum = sum;
	        outstripe[out_z] = sum;
	      }
	    }
	  }
	}

	/* Adjust the maximum by adding the maximum possible bias */
	maxsum += bias_adder;
	fixed_recip_level_size = 0x00FF0000U/maxsum;	// chosen to align at bit 16

	/* Now go back through, add bias, clip to positive and requantize. */

	for (j = 0; j < out_batches*out_height*out_width; j++) {
	  for (i = 0; i < out_depth; i++) {
	    sum = biasbuf[i] + tmp_out[j*out_depth+i];
	    int32_t out_i = (sum * fixed_recip_level_size + (1<<15));
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

static inline void __attribute__((unused)) biasadd_relu_requant_ref(
	uint8_t *out,
	const int32_t *tmp_out,
	const int32_t *biasbuf,
	const uint32_t num_patches,
	const uint32_t depth,
	const uint32_t fixed_recip_level_size)
{
	int32_t sum;
	int32_t outval;
	int32_t i,j;
	for (j = 0; j < num_patches; j++) {
		for (i = 0; i < depth; i++) {
			sum = biasbuf[i] + tmp_out[j*depth+i];
			outval = sum * fixed_recip_level_size + (1<<15);
			outval >>= 16;
			if (outval < 0) outval = 0;
			if (outval > 255) outval = 255;
			*out++ = outval;
		}
	}
}

static void biasadd_relu_requant_execute_hvx_slice(struct nn_graph *nn, void * vinfo) 
{
        struct tdata *info = vinfo;
        struct nn_node *self = info->self;
        int whoami = info->whoami;
        int32_t *biasbuf = nn->scratch;
	struct tensor *out_tensor = self->outputs[0];
        uint8_t *out = out_tensor->data;

        int length, offset, k;
        int * tmp_out = info->iptr;
        int patches = info->arg0;
        int depth = info->arg1;
        uint32_t fixed_recip_level_size = info->arg2;
        k = patches/2;
        k = (k+4-1)&~(4-1);   //multiple of 4 patches makes 4xdepth always 128byte 'tiple
           
        if(whoami ==0) {
           offset = 0;
           length = k;
        } else {
           offset = k*depth;
           length = patches - k;
        }
        if((depth % 32)==0) {
           biasadd_relu_requant_hvx(
               out + offset,
               (int32_t *)tmp_out + offset,
               biasbuf,
               length,
               depth,
               fixed_recip_level_size);
             
        } else {
           biasadd_relu_requant_nonaligned_hvx(
               out + offset,
               (int32_t *)tmp_out + offset,
               biasbuf,
               length,
               depth,
               fixed_recip_level_size);
        }
        nn_sem_post(&info->donesem);
}

static void __attribute__((unused)) gemsuma_execute_hvx_slice(struct nn_graph *nn, void * vinfo) 
{
        struct tdata *info = vinfo;
        int chunk = info->arg0;
        int num_patches = info->arg1;
        
                if(chunk==0) {
                    gemsuma_asm(
                                (uint8_t *) info->iptr,
                                num_patches,
                                info->arg2,
                                (int *) info->optr,
                                info->arg3,  //xoffset
                                info->arg4); //zoffset
                } else {
                    gemacca_asm(
                                (uint8_t *) info->iptr,
                                num_patches,
                                info->arg2,
                                (int *) info->optr,
                                info->arg3);  //xoffset
                }

        nn_sem_post(&info->donesem);
}

static void supernode_execute_hvx_slice(struct nn_graph *nn, void * vinfo) 
{
        struct tdata *info = vinfo;
        struct nn_node *self = info->self;
        int whoami = info->whoami;

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	//const struct tensor *bias_tensor = self->inputs[7];
	//const struct tensor *bias_min_tensor = self->inputs[8];
	//const struct tensor *bias_max_tensor = self->inputs[9];
	//struct tensor *out_tensor = self->outputs[0];
	//struct tensor *out_min = self->outputs[1];
	//struct tensor *out_max = self->outputs[2];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t filt_batches = filt_tensor->shape.byidx[0];
	int32_t filt_height = filt_tensor->shape.byidx[3];
	int32_t filt_width = filt_tensor->shape.byidx[2];
	int32_t filt_depth = filt_tensor->shape.byidx[1];

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	int32_t out_batches = in_batches;
	int32_t out_width = nn_pad_compute_outsize(in_width,filt_width,stride_width,self->padding);
	int32_t out_height = nn_pad_compute_outsize(in_height,filt_height,stride_height,self->padding);
	int32_t out_depth = filt_batches;

	int32_t batch;

	//int32_t in_y_base;
	//int32_t in_x_base;

	uint8_t *in = in_tensor->data;
	//uint8_t *filt = filt_tensor->data;
	//uint8_t *bias = bias_tensor->data;

	//uint8_t *instripe;
	//uint8_t *filtstripe;
	//int32_t *outstripe;

	//int32_t in_element;
	//int32_t filt_element;
	//int32_t sum;
	int32_t maxsum = 0;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	//size_t out_size = out_elements;
	/* FIXME: if you pad depth you should adjust tmp_out_size here!!! */
	size_t tmp_out_size = out_elements*sizeof(int32_t);
	size_t biasbuf_size = out_depth*sizeof(int32_t);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	//float bias_min_float = tensor_get_float(bias_min_tensor,0);
	//float bias_max_float = tensor_get_float(bias_max_tensor,0);

	int32_t adj_x = ((out_width-1) * stride_width + filt_width - in_width) / 2;
	int32_t adj_y = ((out_height-1) * stride_height + filt_height - in_height) / 2;

        //printf("PADDING %d %d %d\n", self->padding, adj_x, adj_y);

        struct tdata sub_info = {
                .self = self,
                .whoami = whoami,
        };
        nn_sem_init(&sub_info.donesem,0);

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	//float in_level_size = (in_max_float - in_min_float) / 255;
	//float filt_level_size = (filt_max_float - filt_min_float) / 255;
	//float bias_level_size = (bias_max_float - bias_min_float) / 255;
	//float out_level_size = in_level_size * filt_level_size;

	//float bias_mpy_amt = (bias_level_size / out_level_size);
	//int bias_adder = (bias_max_float / out_level_size);

	//float conv_out_max_val = ((float)(INT32_MAX)) * out_level_size;
	//float conv_out_min_val = 0.0f;

	//float final_out_max_val;
	//float final_out_min_val;

	//uint32_t fixed_recip_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t input_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	//int32_t bias_offset = quantize_uint(0.0f,bias_min_float,bias_max_float);

	//int i;

	/* intermediate buffer generation */
	int patches = out_height*out_width;
	int patches_pad = (patches+VPAD-1)&~(VPAD-1);
	int out_depth_pad = (out_depth + DPAD - 1) & ~(DPAD-1);
	int32_t filter_value_count = filt_width*filt_height*filt_depth; //aka K 
	int32_t filter_value_count_pad = (filter_value_count+(HPAD-1))&~(HPAD-1); //K rounding
	uint32_t im2col_bufsize = sizeof(uint8_t) * patches_pad * filter_value_count_pad;
	//uint32_t out_padsize = sizeof(int32_t) * patches_pad * out_depth_pad;
	uint32_t minmax_size = 2*sizeof(int)*64;
	uint32_t suma_size = patches_pad*sizeof(int);
	//uint32_t sumb_size = out_depth_pad*sizeof(int);
	uint32_t sumb_size = 2*32*sizeof(int); //one for each thread

	uint8_t* filt_pad_trans = self->opaque;
	int32_t *biasbuf = nn->scratch;
	uint8_t *im2col_buf = pad_and_align(biasbuf,biasbuf_size);
	int *minmax_buf = pad_and_align(im2col_buf,im2col_bufsize);
	int *suma = pad_and_align(minmax_buf,minmax_size);
	int *sumb_buf = pad_and_align(suma,suma_size);
	int *tmp_out = pad_and_align(sumb_buf,sumb_size);
	int *out_pad = pad_and_align(tmp_out,tmp_out_size);
        int *sumb, *minmax;
#if 0
	uint32_t totalsize = biasbuf_size
		+ im2col_bufsize
		+ minmax_size
		+ suma_size
		+ sumb_size
		+ out_padsize
		+ tmp_out_size
		+ ALIGN_SIZE*7;
#endif
	uint32_t skip_unpad = (out_depth == out_depth_pad);
	uint32_t skip_im2col = ((skip_unpad)
			&& (filter_value_count == filter_value_count_pad)
			//&& (patches == patches_pad)
			&& (filt_height == 1)
			&& ((filt_width == 1) || (adj_x == 0))
			);
	uint32_t stage_one = 
                  ((filter_value_count_pad == 32)
		&& (stride_width == 2)
		&& (filt_depth == 3)
		&& (filt_height == 3)
		&& (filt_width == 3));
	uint32_t stage_one_v1 = ((filter_value_count_pad ==160)
		&& (stride_width == 2)
		&& (filt_depth == 3)
		&& (filt_height == 7)
		&& (filt_width == 7));

	//uint8_t* filt_pad = (uint8_t*)memalign(128, filter_value_count_pad*out_depth_pad);
	//uint8_t* filt_pad_trans = (uint8_t*)memalign(128, filter_value_count_pad*out_depth_pad);
	//int has_precalc_max = (self->n_inputs == 11);
	//float precalc_max = has_precalc_max ? tensor_get_float(self->inputs[10],0) : 0.0;
	if (skip_unpad) out_pad = tmp_out;

        int wpf, spf; //slice prefetch and weights prefetch
        int N, K = filter_value_count_pad;
        int chunk, num_chunks, k;
        int total_slices = (patches_pad*K+DATA_PREFETCH_BLOCK-1)/DATA_PREFETCH_BLOCK; //divide into 64K
        total_slices = (total_slices + 1) & ~1; //try to make even for symmetric threads
        int weights, slice, fetch_size;
        int num_patches, patch_start, patch_base;
        int NSTEP = rounddn(patches_pad/total_slices, VPAD);
        int M = out_depth_pad;
        uint8_t * im2col_patch_buf;
        int KSTEP, total_chunks ; //how much K is split

        total_slices = total_slices/2; //equal split on each thread

        if(stage_one) {
           if(whoami == 0) {
            total_slices = 9; 
            NSTEP = 8*149; 
           } else {
            total_slices = 9; //total slices 149*(8*9*2+5)
            NSTEP = 8*149; 
           }
        }
        if(stage_one_v1) {
           NSTEP = 4*112; 
           total_slices =14;  //total slices 112/2
        }
        if(whoami == 0) {
             patch_base = 0;
             N = total_slices*NSTEP;
             sumb = sumb_buf;
             minmax = minmax_buf;
        } else {
             patch_base = total_slices*NSTEP;
             N = patches_pad;
             sumb = sumb_buf+32;
             minmax = minmax_buf+64;
        }
        //printf("/////////////////// total slices = %d NSTEP = %d / %d\n",total_slices, NSTEP, N);

        if(K > K_CHUNKSIZE) total_chunks = (K+K_CHUNKSIZE-1)/K_CHUNKSIZE; else total_chunks = 1;
        KSTEP = rounddn(K / total_chunks, HPAD);
        //if(NSTEP == 96) K0 = 768; else K0 =4800/(NSTEP-96); if(K0 < 0) K0 = 32;
        //printf(" KSTEP = %d K = %d NSTEP = %d N = %d\n", KSTEP,K,NSTEP,N);

        //prefetch first filter weights and data slice
        //data is fetches as total of NSTEP*s2 * in_depth 
        fetch_size = ((NSTEP*stride_width*stride_height)+(adj_y+filt_height)*in_width)*in_depth;
	for (batch=0; batch < out_batches; batch++) {
             uint8_t * in_batch = &in[batch*in_height*in_width*in_depth];
             l2prefetch(in_batch+K*patch_base, 8, fetch_size/8, fetch_size/8);
             l2prefetch(filt_pad_trans,  8, 4*KSTEP, 4*KSTEP);
        }
        //printf(" %08x %d\n", (int) in, fetch_size);
        //printf(" NSTEP = %d N = %d KSTEP = %d K = %d\n", NSTEP, N, KSTEP, K);
        //printf(" raw %08X to %08X	%d\n", (int) in, (int) &in[fetch_size], fetch_size);
        //printf(" im2col %08X\n", (int) im2col_buf);
        //printf(" filtr %08X\n", (int) filt_pad_trans);

#if 0
	if (++supernode_count==2) {
		RESET_PMU();
		ENABLE_PMU();
	}
#endif

	/* BEGIN CONV2D. results in tmp_out buffer, also maxsum is updated */
	/* pad out the filter weights matrix to M x K and transpose done in check ref*/
	/* pad data matrix horizontally to tuples of HPAD */
	/* Do convolutions */
	for (batch = 0; batch < out_batches; batch++)
	{
          uint8_t * in_batch = &in[batch*in_height*in_width*in_depth];
          for(weights = 0; weights < M; weights += 32)
          {
            for(slice=0, patch_start=patch_base; slice < total_slices; slice++, patch_start += NSTEP)
            {
              int last_slice = (slice == total_slices - 1);
              if(last_slice) num_patches = N - patch_start;
	      else num_patches = NSTEP;

              //prefetch im2col data after creation in first column of weights
              if(skip_im2col) 
	        im2col_patch_buf = in_batch + patch_start*K;
              else
	        im2col_patch_buf = im2col_buf + patch_start*K;
              //prefetch data for im2col in L2
              if(!skip_im2col && weights ==0) {
                /* first time need to prefetch raw input data scaled by stride^2 approx*/
                spf = (patch_start+NSTEP)/out_width;
                spf = spf*stride_height*in_width ;

                fetch_size = (NSTEP*stride_width*stride_height+filt_height*in_width)*in_depth;
                l2prefetch(&in_batch[spf*in_depth], 8, fetch_size/8, fetch_size/8);
                //printf(" %08x - %08x %d\n", (int) &in_batch[spf*in_depth], (int) &in_batch[spf*in_depth+fetch_size], fetch_size);

                /* first time no need to prefetch im2col computed live */
                if(stage_one)  {
                  im2col33322_hvx(in_batch,
                                 im2col_patch_buf,
                                 input_offset,
                                 preds3332,
                                 patch_start/149, num_patches/149
                  );
                } else if(stage_one_v1)  {
                  im2col7732_asm (in_batch,
                                im2col_patch_buf,
                                input_offset,
                                vpred7732,
                                patch_start/112,  num_patches/112
                  );
                } else  {
                  im2col_slice_co(in_batch, 
			        in_height,
	         		in_width,
			        in_depth,
			        input_offset,
			        im2col_patch_buf,
	         		filt_height,
			        filt_width,
	         		stride_width,
			        out_height,
		         	out_width,
	         		adj_x,
			        adj_y,
                                patch_start, num_patches
                  );
                }
              } //im2col
              for(chunk=0, k=0 ; chunk < total_chunks; chunk++, k+= KSTEP)
              {
                int last_chunk = (chunk == total_chunks - 1);
                if(last_chunk) num_chunks = K - chunk*KSTEP; else num_chunks = KSTEP;

                if(last_chunk)
                  spf = num_patches*K;
                else
                  spf = k + num_chunks;
                l2prefetch(&im2col_patch_buf[spf], NSTEP, num_chunks, K);

                //prefetch weights in chunks
                //when end of slices and chunks next weights
                if(last_chunk && last_slice)
                  wpf = (weights+32)*K ;
                else if(last_chunk && !last_slice) //stay in this  K block
                  wpf = weights*K ;
                else
                  wpf = (weights*K) + (k+num_chunks)*32;
                if (whoami == 1) l2prefetch(&filt_pad_trans[wpf], 8, 4*num_chunks, 4*num_chunks);
                //printf("filter fetching %d from %08X\n", 32*KSTEP, (int)&filt_pad_trans[wpf]);
                sub_info.iptr = im2col_patch_buf+k;
                sub_info.optr = suma + patch_start;
                sub_info.arg0 = chunk;
                sub_info.arg1 = num_patches;
                sub_info.arg2 = Q6_R_combine_RlRl(num_chunks,K);
                sub_info.arg3 = -filt_offset;
                sub_info.arg4 = K*input_offset*filt_offset;
                //only dcfetch chunks
                if(weights==0)nn_os_work_for_scalar(nn,gemsuma_execute_hvx_slice,&sub_info);
                //gemsuma_execute_hvx_slice(nn,&sub_info);

                if(chunk==0) {
                    gemmpybbw_asm(im2col_patch_buf,  
                                &filt_pad_trans[weights*K],  
                                &out_pad[M*patch_start+weights], 
                                num_patches, 
                                M, 
                                Q6_R_combine_RlRl(num_chunks,K));
                /*
                    if(weights==0) gemsuma_asm(
                                im2col_patch_buf,
                                num_patches,
                                Q6_R_combine_RlRl(num_chunks,K),
                                &suma[patch_start],
                                -filt_offset,
                                K*input_offset*filt_offset);
                 */
                    if(slice == 0) gemsumb_asm(
                                &filt_pad_trans[weights*K],  
                                sumb,        //reused for each slice 
                                num_chunks,
                                -input_offset);
                } else {
                    gemmacbbw_asm(im2col_patch_buf+k,  
                                &filt_pad_trans[weights*K+32*k],  
                                &out_pad[M*patch_start+weights], 
                                num_patches, 
                                M, 
                                Q6_R_combine_RlRl(num_chunks,K));
                 /*
                    if(weights==0) gemacca_asm(
                                im2col_patch_buf+k,
                                num_patches,
                                Q6_R_combine_RlRl(num_chunks,K),
                                &suma[patch_start],
                                -filt_offset);
                 */
                    if(slice == 0) gemaccb_asm(
                                &filt_pad_trans[weights*K+32*k],  
                                sumb,        //reused for each slice 
                                num_chunks,
                                -input_offset);
                }//endif
                if(weights==0) nn_sem_wait(&sub_info.donesem);
              }//end chunks
              gemaddvvm_asm(&suma[patch_start], 
                            sumb, 
                            &out_pad[M*patch_start+weights],
                            num_patches, 
                            M, 
                            minmax, 
                            (weights==0 && slice == 0));
            }//end slice
          }//end weights
	  int gmax = minmax[32];
	  if (maxsum < gmax) maxsum = gmax;
	  /* strip out the padding from the output */
	  if (!skip_unpad) unpad2d(out_pad, patches_pad, out_depth_pad,
		&tmp_out[batch*patches*out_depth], patches, out_depth);
	}//end batch
	//DISABLE_PMU();
	/* Adjust the maximum by adding the maximum possible bias */
        info->mat_max = maxsum;
#if 0
	DISABLE_PMU();
#endif
        nn_sem_post(&info->donesem);
}

static int supernode_execute_hvx(struct nn_node *self, struct nn_graph *nn)
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

	int32_t filt_batches = filt_tensor->shape.byidx[0];
	int32_t filt_height = filt_tensor->shape.byidx[3];
	int32_t filt_width = filt_tensor->shape.byidx[2];
	int32_t filt_depth = filt_tensor->shape.byidx[1];

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	int32_t out_batches = in_batches;
	int32_t out_width = nn_pad_compute_outsize(in_width,filt_width,stride_width,self->padding);
	int32_t out_height = nn_pad_compute_outsize(in_height,filt_height,stride_height,self->padding);
	int32_t out_depth = filt_batches;

	//int32_t batch;
	//int32_t filt_x;
	//int32_t filt_y;
	//int32_t filt_z;
	//int32_t out_x;
	//int32_t out_y;
	//int32_t out_z;

	//int32_t in_y_base;
	//int32_t in_x_base;

	//uint8_t *in = in_tensor->data;
	//uint8_t *filt = filt_tensor->data;
	uint8_t *bias = bias_tensor->data;
	//uint8_t *out = out_tensor->data;

	//uint8_t *instripe;
	//uint8_t *filtstripe;
	//int32_t *outstripe;

	//int32_t in_element;
	//int32_t filt_element;
	//int32_t sum;
	int32_t maxsum = 0;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements;
	/* FIXME: if you pad depth you should adjust tmp_out_size here!!! */
	size_t tmp_out_size = out_elements*sizeof(int32_t);
	size_t biasbuf_size = out_depth*sizeof(int32_t);

        struct tdata worker_info = {
                .self = self,
                .whoami = 0,
        };
        nn_sem_init(&worker_info.donesem,0);
        struct tdata my_info = {
                .self = self,
                .whoami = 1,
        };

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	float bias_min_float = tensor_get_float(bias_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);

	//int32_t adj_x = ((out_width-1) * stride_width + filt_width - in_width) / 2;
	//int32_t adj_y = ((out_height-1) * stride_height + filt_height - in_height) / 2;

        //printf("PADDING %d %d %d\n", self->padding, adj_x, adj_y);

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float in_level_size = (in_max_float - in_min_float) / 255;
	float filt_level_size = (filt_max_float - filt_min_float) / 255;
	float bias_level_size = (bias_max_float - bias_min_float) / 255;
	float out_level_size = in_level_size * filt_level_size;

	float bias_mpy_amt = (bias_level_size / out_level_size);
	int bias_adder = ceil(bias_max_float / out_level_size);

	//float conv_out_max_val = ((float)(INT32_MAX)) * out_level_size;
	//float conv_out_min_val = 0.0f;

	float final_out_max_val;
	float final_out_min_val;

	//float final_recip_level_size;
	uint32_t fixed_recip_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	//int32_t input_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
	//int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	int32_t bias_offset = quantize_uint(0.0f,bias_min_float,bias_max_float);

	//int i,j;
	int i;

	/* intermediate buffer generation */
	int patches = out_height*out_width;
	int patches_pad = (patches+VPAD-1)&~(VPAD-1);
	int out_depth_pad = (out_depth + DPAD - 1) & ~(DPAD-1);
	int32_t filter_value_count = filt_width*filt_height*filt_depth; //aka K 
	int32_t filter_value_count_pad = (filter_value_count+(HPAD-1))&~(HPAD-1); //K rounding
	uint32_t im2col_bufsize = sizeof(uint8_t) * patches_pad * filter_value_count_pad;
	uint32_t out_padsize = sizeof(int32_t) * patches_pad * out_depth_pad;
	uint32_t minmax_size = 2*sizeof(int)*64; //two threads need access
	uint32_t suma_size = patches_pad*sizeof(int);
	//uint32_t sumb_size = out_depth_pad*sizeof(int);
	uint32_t sumb_size = 2*32*sizeof(int);

#if 0
	uint8_t* filt_pad_trans = self->opaque;
#endif
	int32_t *biasbuf = nn->scratch;
	uint8_t *im2col_buf = pad_and_align(biasbuf,biasbuf_size);
	int *minmax_buf = pad_and_align(im2col_buf,im2col_bufsize);
	int *suma = pad_and_align(minmax_buf,minmax_size);
	int *sumb = pad_and_align(suma,suma_size);
	int *tmp_out = pad_and_align(sumb,sumb_size);
	//int *out_pad = pad_and_align(tmp_out,tmp_out_size);
	uint32_t totalsize = biasbuf_size
		+ im2col_bufsize
		+ minmax_size
		+ suma_size
		+ sumb_size
		+ out_padsize
		+ tmp_out_size
		+ ALIGN_SIZE*7;
	uint32_t skip_unpad = (out_depth == out_depth_pad);
	uint32_t skip_im2col = ((skip_unpad)
		&& (filter_value_count == filter_value_count_pad)
		//&& (patches == patches_pad)
		&& (filt_height == 1)
		&& (filt_width == 1));

	uint64_t conv_start;
	uint64_t nonconv_start;
	uint64_t nonconv_end;
#if 0
	uint32_t stage_one = 
                 ((filter_value_count_pad == 32)
		&& (stride_width == 2)
		&& (filt_depth == 3)
		&& (filt_height == 3)
		&& (filt_width == 3));
#endif
        
	//uint8_t* filt_pad = (uint8_t*)memalign(128, filter_value_count_pad*out_depth_pad);
	//uint8_t* filt_pad_trans = (uint8_t*)memalign(128, filter_value_count_pad*out_depth_pad);
	int has_precalc_max = (self->n_inputs == 11);
	float precalc_max = has_precalc_max ? tensor_get_float(self->inputs[10],0) : 0.0;
	//if (skip_unpad) out_pad = tmp_out;

	logmsg(nn,2,"supernode execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"supernode input %dx%dx%dx%d [%f,%f]",in_batches,in_height,in_width,in_depth,in_min_float,in_max_float);
	logmsg(nn,2,"supernode filt %dx%dx%dx%d [%f,%f]",filt_batches,filt_height,filt_width,filt_depth,filt_min_float,filt_max_float);
	logmsg(nn,2,"supernode stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"supernode padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	if (has_precalc_max) logmsg(nn,2,"Precalculated maximum value: %f",precalc_max);
	else logmsg(nn,2,"No precalculated maximum value");
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if (totalsize > nn->scratch_size) {
		return errlog(nn,"scratch too small (%d>%d)",totalsize,nn->scratch_size);
	}
	if (out_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",out_tensor->max_size,out_size);
	}
	if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");
	if (out_min->max_size < sizeof(float)) return errlog(nn,"min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn,"max too small");
	if (skip_im2col) logmsg(nn,2,"Woo, able to skip im2col");
	if (skip_unpad) logmsg(nn,2,"Woo, able to skip unpad");
	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_size;

#if 0
	if (++supernode_count==2) {
		RESET_PMU();
		ENABLE_PMU();
	}
#endif
	conv_start = nn_os_get_cycles();
        nn_os_work_for_vector(nn,supernode_execute_hvx_slice,&worker_info);
        //supernode_execute_hvx_slice(nn,&worker_info);
        supernode_execute_hvx_slice(nn,&my_info);

        nn_sem_wait(&worker_info.donesem);
	nonconv_start = nn_os_get_cycles();

	/* 
	 * This *could* be changed to fixed point and vectorized, but it shouldn't
	 * impact performance that much, just traversing depth once. 
	 */
	for (i = 0; i < out_depth; i++) {
		int32_t biasval = bias[i];
		biasbuf[i] = ((biasval - bias_offset) * bias_mpy_amt)+0.5f;
	}

	//return 0;
        maxsum = my_info.mat_max;
        if(worker_info.mat_max > maxsum) maxsum = worker_info.mat_max;

	if (has_precalc_max) {
		maxsum = round(precalc_max / out_level_size);
	} else {
		/* Adjust the maximum by adding the maximum possible bias */
		maxsum += bias_adder;
	}
	fixed_recip_level_size = 0x00FF0000U/maxsum;	// chosen to align at bit 16

	/* Now go back through, add bias, clip to positive and requantize. */

	worker_info.iptr = tmp_out;
	worker_info.arg0 = out_batches * patches;
	worker_info.arg1 = out_depth;
	worker_info.arg2 = fixed_recip_level_size;
	my_info.iptr = tmp_out;
	my_info.arg0 = out_batches * patches;
	my_info.arg1 = out_depth;
	my_info.arg2 = fixed_recip_level_size;
	nn_os_work_for_vector(nn,biasadd_relu_requant_execute_hvx_slice,&worker_info);
	biasadd_relu_requant_execute_hvx_slice(nn,&my_info);
	nn_sem_wait(&worker_info.donesem);

	final_out_max_val = maxsum * out_level_size;
	final_out_min_val = 0.0f;

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,final_out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,final_out_max_val);
	out_max->data_size = sizeof(float);
#if 0
	DISABLE_PMU();
#endif
	nonconv_end = nn_os_get_cycles();
	record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER0,nonconv_start-conv_start);
	record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER1,nonconv_end-nonconv_start);

	logmsg(nn,2,"supernode execute (hvx) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}

static int supernode_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking supernode node %p",self);
	if (self->n_inputs < 10) return errlog(nn,"supernode wrong # inputs");
	if (self->n_inputs > 11) return errlog(nn,"supernode wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"supernode wrong # outputs");
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
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	uint32_t filt_batches = filt_tensor->shape.byidx[0];
	uint32_t filt_height = filt_tensor->shape.byidx[3];
	uint32_t filt_width = filt_tensor->shape.byidx[2];
	uint32_t filt_depth = filt_tensor->shape.byidx[1];
	uint32_t out_depth = filt_batches;
	uint8_t *filt = filt_tensor->data;
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	uint32_t filt_elements = filt_width*filt_height*filt_depth;
	uint32_t filt_elements_pad = (filt_elements + HPAD - 1) & (~(HPAD - 1));
	int out_depth_pad = (out_depth + DPAD - 1) & ~(DPAD-1);
	uint32_t consts_size = filt_elements_pad * out_depth_pad;
	int vec_id;

	if ((self->opaque = memalign(ALIGN_SIZE,consts_size)) == NULL) {
		return errlog(nn,"couldn't allocate buffer for const rearrangement");
	}
	vec_id = nn_os_vector_acquire();
	pad2d(filt,filt_elements,out_depth,nn->scratch,filt_elements_pad,out_depth_pad,filt_offset);
	transpack(nn->scratch,filt_elements_pad,out_depth_pad,self->opaque);
	nn_os_vector_release(vec_id);
	logmsg(nn,2,"supernode node %p check OK",self);
	return 0;
}


static int supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
	if (self->opaque) free(self->opaque);
	return node_free_common(self,nn);
}

#if 0
struct nn_node_ops nn_ops_for_QuantizedConv2d_8x8to32 = {
	.execute = conv2d_execute_hvx,
	.check = conv2d_check_ref,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
#endif

struct nn_node_ops nn_ops_for_Supernode_8x8p8to8 = {
	.execute = supernode_execute_hvx,
	.check = supernode_check_ref,
	.ctor = node_alloc_common,
	.dtor = supernode_dtor,
};

struct nn_node_ops nn_ops_for_Supernode_8x8p8to8_ref = {
	.execute = supernode_execute_ref,
	.check = supernode_check_ref,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

