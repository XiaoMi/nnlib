
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
#include <hexagon_protos.h>

#ifdef HEXAGON_V66
#define GVCONV_ASM gvconv2dbbb_v66_asm
#define GVCONVSUM_ASM(BUF,FILT,OUT,W,OW,OD,WD,FW,FH,NL,IMSUM,FSUM,MAXBUF,U0,U1,BIASBUF,RLEV) \
	GVCONV_ASM(BUF,FILT,OUT,W,OW,OD,WD,FW,FH,NL,IMSUM,FSUM,MAXBUF,BIASBUF,RLEV)
#else
#define GVCONV_ASM gvconv2dbbb_asm
#define GVCONVSUM_ASM gvconvsum2dbbb_asm
#endif

#endif //__hexagon__

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
	int    arg5;
	int    arg6;
	int    arg7;
	nn_sem_t donesem;
	uint64_t cycles;
};

struct supernode_info {
	void *filt_pad;
	float maxval;
	float minval;
	int maxval_precalculated;
	int minval_precalculated;
};

#define VPAD 4   //gvm loops now only unrolled by 4
#define HPAD 16
#define DPAD 32

#define MAX_BUF (384*1024)

//static inline int rounddn(int x, int pad) { x = x  & ~(pad-1); return(x); }

//void l2pref(void *, int height, int width, int stride);


//#define SUPERNODE_DEBUG

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

// Calculate right shift value for maxsum > 2^24. Maxsum > 2^16 are also adjusted correctly. This is to ensure maxsum*reciprocal gives 0x00ff0000 most of the time.
static int __attribute__((unused)) get_shr_val(int maxsum)
{
	int shr=0;

	if (Q6_R_cl0_R(maxsum) < 8)
        shr =  8 - Q6_R_cl0_R(maxsum);

	return shr;
}

static int __attribute__((unused)) get_shr_val_maxsum16(int maxsum)
{
	int shr=0;

	if (Q6_R_cl0_R(maxsum) < 16)
		shr =  16 - Q6_R_cl0_R(maxsum);
	return shr;
}

static inline void __attribute__((unused)) l2prefetch(void *ptr, int height, int width, int stride)
{
        if (width != stride) {
                l2pref(ptr,height,width,stride);
        } else {
                height = ((height*width + 63)/64);
                l2pref(ptr,height,64,64);
        }
        return;
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
	struct supernode_info *nodeinfo = (struct supernode_info *)self->opaque;
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

	uint8_t *in = (uint8_t *)in_tensor->data;
	uint8_t *filt = (uint8_t *)filt_tensor->data;
	uint8_t *bias = (uint8_t *)bias_tensor->data;
	uint8_t *out = (uint8_t *)out_tensor->data;

	uint8_t *instripe;
	uint8_t *filtstripe;
	int32_t *outstripe;

	int32_t in_element;
	int32_t filt_element;
	int32_t sum;
	int32_t maxsum = 0;
	int32_t minsum = 0;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements;
	/* FIXME: if you pad depth you should adjust tmp_out_size here!!! */
	size_t tmp_out_size = out_elements*sizeof(int32_t);
	size_t biasbuf_size = out_depth*sizeof(int32_t);

	int32_t *biasbuf = (int32_t *)nn->scratch;
	int32_t *tmp_out = (int32_t *)pad_and_align(biasbuf,biasbuf_size);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	float bias_min_float = tensor_get_float(bias_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);

	//int32_t adj_x = ((out_width-1) * stride_width + filt_width - in_width) / 2;
	//int32_t adj_y = ((out_height-1) * stride_height + filt_height - in_height) / 2;
	int32_t adj_x = nn_pad_compute_before(out_width,filt_width,stride_width,self->padding);
	int32_t adj_y = nn_pad_compute_before(out_height,filt_height,stride_height,self->padding);

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

	float final_out_max_val = nodeinfo->maxval;
	float final_out_min_val = nodeinfo->minval;
	float minval_offset = -final_out_min_val / out_level_size;
	//int shr_val;

	int32_t maxrange = (final_out_max_val - final_out_min_val) / out_level_size + 0.5f;
	uint32_t fixed_recip_level_size = 0x00ff0000 / maxrange;

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
//logmsg(nn,9,"[%d %d %d %d]: sum += %d*%d --> %d", batch,out_y,out_x,out_z,in_element,filt_element,sum);
	            }
	          }
	        }
	        if (sum > maxsum) maxsum = sum;
	        if (sum < minsum) minsum = sum;
	        outstripe[out_z] = sum;
	      }
	    }
	  }
	}

	/* Adjust the maximum by adding the maximum possible bias */
	maxsum += minval_offset;
	minsum += minval_offset;

	// To handle max-sum > 2^24, we find a right shift value. All the  conv value are right shifted before
    //  multiplying with reciprocal.
    // Before calculating the reciprocal, calculate a shift amount: 0, or 16-Q6_R_cl0_R(val))
    //  Shift the max value right by shift amount before finding the reciprocal
    // Shift all the sums in core asm loop, right by shift amount before multiplying by the reciprocal.
	//shr_val = get_shr_val_maxsum16(maxsum);
	//fixed_recip_level_size = 0x00FF0000U/(maxsum >> shr_val);	// chosen to align at bit 16

	/* Now go back through, add bias, clip to positive and requantize. */

	for (j = 0; j < out_batches*out_height*out_width; j++) {
	  for (i = 0; i < out_depth; i++) {
	    sum = biasbuf[i] + tmp_out[j*out_depth+i] + minval_offset;
	    int32_t out_i = ( ((sum) * fixed_recip_level_size) + (1<<15));
	    out_i >>= 16;
	    if (out_i < 0) out_i = 0;
            if (out_i > 255) out_i = 255;
	    *out++ = out_i;
            //printf("conv_out=%ld, biasbuf=%ld, out=%ld\n", tmp_out[j*out_depth+i], biasbuf[i],out_i);
	  }
	}

	if ((!nodeinfo->maxval_precalculated) && 
		((maxsum * out_level_size) > (final_out_max_val-final_out_min_val))) {
		while ((nodeinfo->maxval-nodeinfo->minval) < (maxsum * out_level_size)) {
			nodeinfo->maxval *= 2;
		}
		logmsg(nn,1,"Supernode %lx maxvalue too small, retrying w/ %f... maxsum=%ld, bias_adder=%d, out_level=%f final_out_max_val=%f [new minval=%f]\n",
			self->node_id,nodeinfo->maxval, maxsum, bias_adder, out_level_size, final_out_max_val, nodeinfo->minval);
		return supernode_execute_ref(self,nn);
	}

	if ((!nodeinfo->minval_precalculated) && 
		(minsum * out_level_size) < 0) {
		while ((minsum * out_level_size) < 0) {
			nodeinfo->minval *= 2;
			minsum -= minval_offset;
			minval_offset = -nodeinfo->minval / out_level_size;
			minsum += minval_offset;
		}
		logmsg(nn,1,"Supernode %lx minvalue too big, retrying w/ %f... minsum=%ld, bias_adder=%d, out_level=%f final_out_max_val=%f [new minval=%f]\n",
			self->node_id,nodeinfo->minval, minsum, bias_adder, out_level_size, final_out_max_val, nodeinfo->minval);
		return supernode_execute_ref(self,nn);
	}

#ifdef SUPERNODE_DEBUG
	printf("REF: fixed_recip_level_size = %lu, shr = %d outmin/max = [%f %f], maxsum = %ld, out_level=%f, bias_adder=%d\n", fixed_recip_level_size, shr_val, final_out_min_val, final_out_max_val, maxsum, out_level_size, bias_adder);
#endif
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

static void __attribute__((unused)) biasadd_relu_requant_execute_hvx_slice(struct nn_graph *nn, void * vinfo) 
{
        struct tdata *info = (struct tdata *)vinfo;
        struct nn_node *self = info->self;
        int whoami = info->whoami;
        int32_t *biasbuf = (int32_t *)nn->scratch;
	struct tensor *out_tensor = self->outputs[0];
        uint8_t *out = (uint8_t *)out_tensor->data;

        int length, offset, k;
        int * tmp_out = (int *)info->iptr;
        int patches = info->arg0;
        int depth = info->arg1;
        if(whoami == 0) l2prefetch(biasbuf, 1, depth, depth);
        uint32_t fixed_recip_level_size = info->arg2;
	uint64_t start_cycles = nn_os_get_cycles(nn);
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
	info->cycles = nn_os_get_cycles(nn)-start_cycles;
        nn_sem_post(&info->donesem);
}

static void __attribute__((unused)) gemsuma_execute_hvx_slice(struct nn_graph *nn, void * vinfo) 
{
        struct tdata *info = (struct tdata *)vinfo;
        //struct nn_node *self = info->self;
        //int whoami = info->whoami;
        int filt_y = info->arg0;
        
                if(filt_y==0) {
                  gvmsumimw_asm( (uint8_t *) info->iptr,
                                 (int *) info->optr,
                                 info->arg1, //out_width;
                                 info->arg2, //skip*in_depth_pad;
                                 info->arg3, //stride_width*in_depth_pad;
                                 info->arg4, //filt_width*in_depth_pad;
                                 info->arg5, //num_lines;
                                 info->arg6, //-filt_offset;
                                 info->arg7);//K*input_offset*filt_offset;
                } else {
                  gvmaccimw_asm( (uint8_t *) info->iptr,
                                 (int *) info->optr,
                                 info->arg1, //out_width;
                                 info->arg2, // skip*in_depth_pad;
                                 info->arg3, // stride_width*in_depth_pad;
                                 info->arg4, //filt_width*in_depth_pad;
                                 info->arg5, //num_lines;
                                 info->arg6);//-filt_offset;
                }

        nn_sem_post(&info->donesem);
}

static void supernode_execute_hvx_slice(struct nn_graph *nn, void * vinfo) 
{
	struct tdata *info = (struct tdata *)vinfo;
	struct nn_node *self = info->self;
	int whoami = info->whoami; //do i use the high half or the low half.

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

	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;
        int     fetch_stride = stride_width;

	int32_t out_batches = in_batches;
	int32_t out_width = nn_pad_compute_outsize(in_width,filt_width,stride_width,self->padding);
	int32_t out_height = nn_pad_compute_outsize(in_height,filt_height,stride_height,self->padding);
	int32_t out_depth = filt_batches;
	int out_depth_pad = (out_depth + DPAD - 1) & ~(DPAD-1); //number of filters

	int32_t batch;
	//int32_t filt_x;
	//int32_t filt_y;
	//int32_t filt_z;
	//int32_t out_x;
	//int32_t out_y;
	//int32_t out_z;
	//int32_t skip, skip4;
	int32_t weights;
	int32_t num_lines;

	//int32_t in_y_base;
	//int32_t in_x_base;

	uint8_t *in = (uint8_t *)in_tensor->data;
	//uint8_t *filt = filt_tensor->data;
	//uint8_t *bias = bias_tensor->data;

	//int32_t in_element;
	//int32_t filt_element;
	int32_t maxsum = 0;

	struct tensor *out_tensor = self->outputs[0];
	uint8_t *out = (uint8_t *)out_tensor->data;
	uint32_t fixed_recip_level_size = info->arg0;

	uint32_t out_elements = out_batches*out_height*out_width*out_depth;
	//size_t out_size = out_elements;
	size_t tmp_out_size = out_elements*sizeof(int8_t);
	size_t biasbuf_size = out_depth*sizeof(int32_t);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	//float bias_min_float = tensor_get_float(bias_min_tensor,0);
	//float bias_max_float = tensor_get_float(bias_max_tensor,0);

	//int32_t pad_x = ((out_width-1) * stride_width + filt_width - in_width) / 2;
	//int32_t pad_y = ((out_height-1) * stride_height + filt_height - in_height) / 2;
	int32_t pad_x = nn_pad_compute_before(out_width,filt_width,stride_width,self->padding);
	int32_t pad_y = nn_pad_compute_before(out_height,filt_height,stride_height,self->padding);


	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t input_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
	int32_t filt_offset __attribute__((unused)) = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	//int32_t bias_offset = quantize_uint(0.0f,bias_min_float,bias_max_float);

	//printf(" in_offset = %d filt_offset = %d bias_offset = %d\n", input_offset, filt_offset, bias_offset);
	//int i,j;

	/* intermediate buffer generation */
	int patches = out_width * out_height;
	int patches_pad = (patches+VPAD-1)&~(VPAD-1);
	//int N = patches;
	int in_depth_pad = (in_depth + HPAD - 1) & ~(HPAD-1);
	int     im2col_bufsize;
	int32_t filter_value_count = filt_width*filt_height*filt_depth; //aka K
	int32_t filter_value_count_pad = (filter_value_count+(HPAD-1))&~(HPAD-1); //K rounding
	//int     out_width4 = (out_width + 3) & ~3;
	int     K = filter_value_count_pad; //filt_width*filt_height*depth;
    int shr_val;

	//uint32_t out_padsize = sizeof(int32_t) * patches_pad * out_depth_pad;
	uint32_t max_size = 2*sizeof(int)*32; //one for each thread
	uint32_t suma_size = patches_pad*sizeof(int);
	uint32_t sumb_size = 2*32*sizeof(int); //one for each thread
	uint32_t bad_depth = (filter_value_count_pad != filter_value_count) || (self->padding == NN_PAD_SAME_CAFFE);

	uint32_t stage_one_v3 = ((filter_value_count_pad == 32)
					&& (stride_width == 2)
					&& (in_depth == 3)
					&& (self->padding == NN_PAD_VALID)
					&& (in_height == 299)
					&& (in_width == 299)
					&& (filt_height == 3)
					&& (filt_width == 3));
	// TODO HEXNN-53: causing Caffe GoogleNet (Inception v1) to crash
	uint32_t stage_one_v1 = 0;
//  uint32_t stage_one_v1 = ((filter_value_count_pad ==160)
//                        && (stride_width == 2)
//                        && (in_depth == 3)
//                        && (filt_height == 7)
//                        && (filt_width == 7));
    uint32_t stage_one = (stage_one_v1 || stage_one_v3);
    uint32_t skip_unpad_m = (out_depth == out_depth_pad);
    uint32_t skip_unpad_k = (filter_value_count == filter_value_count_pad);

	int skip_im2col = (self->padding == NN_PAD_VALID || (filt_width == 1 && filt_height ==1))
				   && (!stage_one) && (skip_unpad_k);

	if(stage_one || bad_depth) {
	   im2col_bufsize = out_width * (out_height+ 2*filt_height + 2) *filter_value_count_pad;
	} else {
	   im2col_bufsize = (in_width+pad_x)*(2*pad_y+in_height+2*filt_height)*in_depth_pad;
	}
	struct supernode_info *nodeinfo = (struct supernode_info *)self->opaque;
	uint8_t * filt_pad_trans = (uint8_t *)nodeinfo->filt_pad;
	int32_t * biasbuf = (int32_t *)nn->scratch;
	uint8_t * im2col_patch_buf = (uint8_t *)pad_and_align(biasbuf,biasbuf_size);
	int     * max_buf = (int *)pad_and_align(im2col_patch_buf,im2col_bufsize);
	int     * im2col_sum = (int *)pad_and_align(max_buf,max_size);
	int     * filt_sum = (int *)pad_and_align(im2col_sum,suma_size);
	uint8_t  * tmp_out = (uint8_t *)pad_and_align(filt_sum,sumb_size);
	uint8_t   * out_pad = (uint8_t *)pad_and_align(tmp_out,tmp_out_size);
        //int     * sumb, * max;
	if (skip_unpad_m) out_pad = out;
	uint64_t start_cycles = nn_os_get_cycles(nn);

	int base_line = 0;
	int my_height = out_height/2;
	if(whoami == 1) {
		 base_line = out_height/2;
		 my_height = out_height - out_height/2;
	}

	// printf("in %d %d out %d %d %d %d K = %d\n", in_width, in_height,out_width, out_height, pad_x, pad_y, K);

	//uint8_t* filt_pad = (uint8_t*)memalign(128, filter_value_count_pad*out_depth_pad);
	//uint8_t* filt_pad_trans = (uint8_t*)memalign(128,filter_value_count_pad*out_depth_pad);
	//int has_precalc_max = (self->n_inputs == 11);
	//float precalc_max = has_precalc_max ? tensor_get_float(self->inputs[10],0) : 0.0;

	//for stage 1 after im2col treated as a non padded 1x1 filter
    int batch_size = in_height*in_width*in_depth;
	if (bad_depth && !stage_one) {
	    /* EJP: FIXME: batches are now broken */
	    logmsg(nn,2,"im2col o=%p in=%p hwd=%d,%d,%d filt=%d,%d stride=%d,%d out=%d-%d,%d,%d zero=%d",
		im2col_patch_buf,
		in,
		in_height,
		in_width,
		in_depth,
		filt_height,
		filt_width,
		stride_width,
		stride_height,
		base_line,
		base_line+my_height,
		out_width,
		filter_value_count_pad,
		input_offset);
	    im2col_full(
		im2col_patch_buf,
		in,
		in_height,
		in_width,
		in_depth,
		filt_height,
		filt_width,
		stride_width,
		stride_height,
		base_line,
		base_line+my_height,
		out_height,
		out_width,
		filter_value_count_pad,
		input_offset,
		self->padding);

	}
	if(stage_one || !skip_unpad_k || bad_depth) {
		//printf(" stage one or unpad needed\n");
		in_width = out_width;
		in_height = out_height;
		in_depth_pad = filter_value_count_pad;
		filt_height = 1;
		filt_width = 1;
		stride_width = 1;
		pad_x = 0;
		pad_y = 0;
	}

	//compute amount of work per slice based on buffer sizes solution for lines per slice
	int W = (in_width + pad_x)*in_depth_pad;
	int lines_per_slice = (MAX_BUF - in_depth_pad*filt_height*filt_width * 64)/(4*stride_width*W);
	if (lines_per_slice <= 1) lines_per_slice = 2;
	 //printf(" lines per slice %d  out height %d\n",lines_per_slice, out_height);
	int num_slices = (out_height + lines_per_slice-1)/lines_per_slice;
	num_slices = (num_slices + 1)/2; // & ~1; //make even for split between threads
	if (num_slices == 1) num_slices = 2;
	lines_per_slice = (my_height) / num_slices;
	//logmsg(nn,1,"whoami? %d. base_line=%d my_height=%d num_slices=%d lines/slice=%d",whoami,base_line,my_height,num_slices,lines_per_slice);
	// printf("num slices %d lines per slice %d\n", num_slices, lines_per_slice);

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
	l2prefetch(filt_pad_trans, 1,  filt_height*filt_width*in_depth_pad*32,
								   filt_height*filt_width*in_depth_pad*32);
	/* BEGIN CONV2D. results in tmp_out buffer, also maxsum is updated pad */
	/* out the filter weights matrix to M x K and transpose done in check  */
        /* ref pad data matrix horizontally to tuples of HPAD, Do convolutions */

	//reset max_buf
	memset(max_buf+32*whoami,0,32*sizeof(max_buf[0]));
    for (batch = 0; batch < out_batches; batch++) {
          uint8_t * in_batch = &in[batch*batch_size];
          uint8_t * out_batch = &out_pad[batch*out_depth_pad*patches];
          //printf("%d, init fetch ptr = %08X\n",whoami,in_batch + base_line*fetch_stride*in_width*in_depth);
          l2prefetch(in_batch + base_line*fetch_stride*in_width*in_depth,
                              lines_per_slice + filt_height + 1, 
                              fetch_stride*in_width*in_depth,
                              fetch_stride*in_width*in_depth);
          for(weights = 0; weights < out_depth_pad; weights += 32) {
             //printf("%d weights = %d\n",whoami,weights);
             uint8_t * im2col_buf;
             int lines, start_line, slice;
             int total_height = my_height;
             int total_slices = num_slices;

             for(slice=0, start_line=base_line; slice < num_slices; slice++, start_line += num_lines) {
               int delta = (slice ==0) ? 0 : filt_height-1; //prevent overlap of im2col gen,
               int last_slice = (slice == num_slices - 1);
               //printf(" %d last slice %d ? \n",slice,last_slice);

               //rate matching to spread workload out evenly
               if(total_height * num_slices > my_height * total_slices)
                    lines = lines_per_slice + 1;
               else
                    lines = lines_per_slice ;
               if(last_slice) num_lines = base_line+my_height - start_line; else num_lines = lines;
               total_height -= num_lines;
               total_slices -= 1;

               //printf("%d num lines %d, start line %d\n",whoami, num_lines, start_line);

               //first time around weights or if im2col is skipped prefetch raw data
               int data_index = start_line + 1*num_lines;

               if(last_slice) data_index = base_line;

               if (bad_depth) {
            	   im2col_buf = im2col_patch_buf;// + whoami*base_line*W;
               } else {
            	   im2col_buf = im2col_patch_buf + whoami*(filt_height+pad_y)*W;
               }

               if(skip_im2col || weights == 0) {
                   //printf("%d, fetch ptr = %08X\n",whoami,in_batch + data_index*fetch_stride*in_width*in_depth);
                   l2prefetch(in_batch + data_index*fetch_stride*in_width*in_depth,
                              num_lines + filt_height + 1, 
                              fetch_stride*in_width*in_depth,
                              fetch_stride*in_width*in_depth);
               } else {
            	   //logmsg(nn,1,"whoami=%d im2col_patch_buf=%p im2col_buf=%p",whoami,im2col_patch_buf,im2col_buf);
                   //printf("%d fetch ptr = %08X\n",whoami,im2col_buf + data_index*fetch_stride*W);
                   l2prefetch(im2col_buf + data_index*fetch_stride*W,
                              num_lines + filt_height, fetch_stride*W, fetch_stride*W);
               }
               if(skip_im2col) {
                 im2col_buf = in_batch;
               } else {
            	   //allow filt_height extra for each thrread
            	   //im2col_buf = im2col_patch_buf + whoami*(filt_height+pad_y)*W;
            	   //logmsg(nn,1,"whoami=%d im2col_patch_buf=%p im2col_buf=%p",whoami,im2col_patch_buf,im2col_buf);
            	   	   if(weights == 0) {
            	   		   if(stage_one_v1) {
            	   			   im2col7732_asm( in_batch,
                                   im2col_buf + start_line * W,
                                   input_offset,
                                   vpred7732,
                                   start_line, num_lines);
            	   		   } else if(stage_one_v3) {
            	   			   im2col33322_hvx(in_batch,
                                   im2col_buf + start_line * W,
                                   input_offset,
                                   preds3332,
                                   start_line, num_lines);
            	   		   } else if (bad_depth) {
            	   			   /* already im2col? */
            	   			   //logmsg(nn,1,"whoami=%d in empty bad depth block");
            	   		   } else {
            	   			   fast_im2col_co(in_batch,  //offset computed internally
                                  in_height,   in_width,   in_depth, input_offset,
                                  im2col_buf + (start_line+delta) * stride_width * W,
                                  filt_height, filt_width, stride_width,
                                  start_line+delta, //
                                  num_lines-delta, //max of out_height,
                                  out_width, pad_x, pad_y, skip_unpad_k);
            	   		   } // im2col
            	   	   } // weights == 0
               } // else skip_im2col
               int weight_index ;
               if(last_slice) {
                    weight_index = (weights+32);
                    if(weight_index >= out_depth_pad) weight_index = 0;
                    weight_index *= K;
               } else {
                    weight_index = weights*K ;
               }
               if(whoami == 0)
                  l2prefetch(filt_pad_trans + weight_index, 1, 
                             filt_height*filt_width*in_depth_pad*32, 
                             filt_height*filt_width*in_depth_pad*32);

               //to be moved to set up section
               gvmsumb_asm(  filt_pad_trans + weights*K, filt_sum+32*whoami, K, -input_offset);


               shr_val = info->arg1;

               //new mega loop
               //printf("%d use ptr = %08X\n",whoami,im2col_buf + start_line*stride_width*W);
               if(num_lines == 0) {
            	   logmsg(nn,2," WARNING!!! %d) Corner case: num_lines=0 lines_per_slice (%d) = my_height (%d) / num_slices (%d)\n", whoami,
            			   lines_per_slice,
            			   my_height,
            			   num_slices);
               }
               if(weights == 0)
                   GVCONVSUM_ASM(im2col_buf + start_line * stride_width * W,
                                  filt_pad_trans,
                                  out_batch + start_line * out_width* out_depth_pad,
                                  W, //padded in width
                                  out_width,
                                  out_depth_pad,
                                  Q6_R_combine_RlRl(stride_width,in_depth_pad),
                                  filt_width, //in_depth_pad
                                  filt_height,
                                  num_lines,
                                  im2col_sum + start_line * out_width,
                                  filt_sum+32*whoami,
                                  max_buf + 32*whoami, -filt_offset, input_offset*filt_offset*K,
                                  (int *)biasbuf,
                                  fixed_recip_level_size,shr_val);
               else
                   GVCONV_ASM (  im2col_buf + start_line * stride_width * W,
                                  filt_pad_trans + weights*K,
                                  out_batch + weights + start_line * out_width* out_depth_pad,
                                  W, //padded in width
                                  out_width,
                                  out_depth_pad,
                                  Q6_R_combine_RlRl(stride_width,in_depth_pad),
                                  filt_width, //in_depth_pad
                                  filt_height,
                                  num_lines,
                                  im2col_sum + start_line * out_width,
                                  filt_sum+32*whoami,
                                  max_buf + 32*whoami,
                                  (int *)(biasbuf + weights),
                                  fixed_recip_level_size); 
             }//slices
          }//end weights
	  int gmax = max_buf[32*whoami];
	  if (maxsum < gmax) maxsum = gmax;
	  logmsg(nn,2,"whoami=%d maxsum=%d",whoami,maxsum);
	  /* strip out the padding from the output */
	  if (!skip_unpad_m) {
		  int pad_h = pad_y*out_width*out_depth_pad; //compensate im2col pad_top part
		  unpad2d_bytes(out_batch+ pad_h + base_line * out_width * out_depth_pad,
                                     my_height*out_width, out_depth_pad,
	                             &out[batch*patches*out_depth + base_line * out_width * out_depth],
                                     my_height*out_width, out_depth);
	   }
	}//end batch
	//DISABLE_PMU();
	/* Adjust the maximum by adding the maximum possible bias */
        info->mat_max = maxsum;
	info->cycles = nn_os_get_cycles(nn) - start_cycles;
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
	struct supernode_info *nodeinfo = (struct supernode_info *)self->opaque;

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
	int out_depth_pad = (out_depth + DPAD - 1) & ~(DPAD-1);

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
	uint8_t *bias = (uint8_t *)bias_tensor->data;
	//uint8_t *out = out_tensor->data;

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
			self,
			0,
	};
	nn_sem_init(&worker_info.donesem,0);
	struct tdata my_info = {
			self,
			1,
	};
	nn_sem_init(&my_info.donesem,0);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	float bias_min_float = tensor_get_float(bias_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);

	//int32_t pad_x = ((out_width-1) * stride_width + filt_width - in_width) / 2;
	//int32_t pad_y = ((out_height-1) * stride_height + filt_height - in_height) / 2;
	int32_t pad_x = nn_pad_compute_before(out_width,filt_width,stride_width,self->padding);
	int32_t pad_y = nn_pad_compute_before(out_height,filt_height,stride_height,self->padding);

        //printf("PARAM filt w,h %x %d stride = %d in_depth = %d out_depth %d\n",filt_width, filt_height, stride_width, in_depth, out_depth);

        //printf("PADDING %d %d %d\n", self->padding, pad_x, pad_y);

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
	float minval_offset = -nodeinfo->minval/out_level_size;
	int bias_adder = ceil((bias_max_float-nodeinfo->minval) / out_level_size);

	//float conv_out_max_val = ((float)(INT32_MAX)) * out_level_size;
	//float conv_out_min_val = 0.0f;

	float final_out_max_val = nodeinfo->maxval;
	float final_out_min_val = nodeinfo->minval;

	//float final_recip_level_size;
	uint32_t fixed_recip_level_size;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	//int32_t input_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
	//int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	int32_t bias_offset = quantize_uint(0.0f,bias_min_float,bias_max_float);

	//int i,j;
	int i;
	int shr=0;

	/* intermediate buffer generation */
	int patches = out_height*out_width;
	int patches_pad = (patches + VPAD-1)& ~(VPAD-1);
	int in_depth_pad = (in_depth + HPAD - 1) & ~(HPAD-1);
    int im2col_bufsize;
	int32_t filter_value_count = filt_width*filt_height*filt_depth; 
	int32_t filter_value_count_pad = (filter_value_count+(HPAD-1))&~(HPAD-1); //K 
	uint32_t stage_one_v3 = ((filter_value_count_pad == 32)
			&& (stride_width == 2)
			&& (in_depth == 3)
			&& (filt_height == 3)
			&& (filt_width == 3));
	// TODO HEXNN-53: causing Caffe GoogleNet (Inception v1) to crash
	uint32_t stage_one_v1 = 0;
//  uint32_t stage_one_v1 = ((filter_value_count_pad ==160)
//                        && (stride_width == 2)
//                        && (in_depth == 3)
//                        && (filt_height == 7)
//                        && (filt_width == 7));
    uint32_t stage_one = (stage_one_v1 || stage_one_v3);
    uint32_t bad_depth = (filter_value_count_pad != filter_value_count) || (self->padding == NN_PAD_SAME_CAFFE);
    if(stage_one || bad_depth) {
	    //if (bad_depth) logmsg(nn,1,"unfortunate input depth, should do im2col");
            im2col_bufsize = out_width * (out_height+2*filt_height+2) *filter_value_count_pad;
	} else {
            im2col_bufsize = (in_width+pad_x)*(2*pad_y+in_height+2*filt_height)*in_depth_pad;
    }
	uint32_t out_padsize = sizeof(int32_t) * patches_pad * out_depth_pad;
	uint32_t max_size = 2*sizeof(int)*32; //two threads need access
	uint32_t suma_size = patches_pad*sizeof(int);
	uint32_t sumb_size = 2*32*sizeof(int);
#if 0
	uint8_t* filt_pad_trans = self->opaque->filt_pad;
#endif
	int32_t *biasbuf = (int32_t *)nn->scratch;
	//uint8_t *im2col_buf = pad_and_align(biasbuf,biasbuf_size);
	//int *max_buf = pad_and_align(im2col_buf,im2col_bufsize);
	//int *suma = pad_and_align(max_buf,max_size);
	//int *sumb = pad_and_align(suma,suma_size);
	//int *tmp_out = pad_and_align(sumb,sumb_size);
	//int *out_pad = pad_and_align(tmp_out,tmp_out_size);
	uint32_t totalsize = biasbuf_size
		+ im2col_bufsize
		+ max_size
		+ suma_size
		+ sumb_size
		+ out_padsize
		+ tmp_out_size
		+ ALIGN_SIZE*7;
#if 0
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
#else
	uint32_t skip_unpad = (out_depth == out_depth_pad); // && filter_value_count == filter_value_count_pad);
    uint32_t skip_im2col = (self->padding == 2 || (filt_width == 1 && filt_height ==1)) && (!stage_one) && (skip_unpad) && (!bad_depth);
#endif
        
	//uint8_t* filt_pad = (uint8_t*)memalign(128, filter_value_count_pad*out_depth_pad);
	//uint8_t* filt_pad_trans = (uint8_t*)memalign(128, filter_value_count_pad*out_depth_pad);
	//if (skip_unpad) out_pad = tmp_out;

	logmsg(nn,2,"supernode execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"supernode input %dx%dx%dx%d [%f,%f]",in_batches,in_height,in_width,in_depth,in_min_float,in_max_float);
	logmsg(nn,2,"supernode filt %dx%dx%dx%d [%f,%f]",filt_batches,filt_height,filt_width,filt_depth,filt_min_float,filt_max_float);
	logmsg(nn,2,"supernode stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"supernode padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	logmsg(nn,2,"Maximum value: %f%s",final_out_max_val,
		nodeinfo->maxval_precalculated ? " (precalculated)" : "");
	logmsg(nn,2,"Minimum value: %f%s",final_out_min_val,
		nodeinfo->minval_precalculated ? " (precalculated)" : "");
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if (bad_depth && (in_batches > 1)) {
		return errlog(nn,"Batch size > 1 not supported with odd in depth for now");
	}
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
	if (out_height <= 3) {
		logmsg(nn,1,"Out very short, ditching to ref version");
		return supernode_execute_ref(self,nn);
	}

#if 0
	if (++supernode_count==5) {
		RESET_PMU();
		ENABLE_PMU();
	}
#endif
	/* 
	 * This *could* be changed to fixed point and vectorized, but it shouldn't
	 * impact performance that much, just traversing depth once. 
	 */
	if (bias_max_float > 1073741824.0f /*0x1.0p30*/ * out_level_size) return errlog(nn,"bias magnitude too large");
	if (bias_min_float < -1073741824.0f /*0x1.0p30*/ * out_level_size) return errlog(nn,"bias magnitude too large");
	for (i = 0; i < out_depth; i++) {
		int32_t biasval = bias[i];
		biasbuf[i] = (biasval - bias_offset) * bias_mpy_amt+minval_offset+0.5f;
	}
	maxsum = ((final_out_max_val-final_out_min_val) / out_level_size) + 0.5f;
    // To handle max-sum > 2^24, we find a right shift value. All the  conv value are right shifted before 
    //  multiplying with reciprocal.
    // Before calculating the reciprocal, calculate a shift amount: 0, or 16-Q6_R_cl0_R(val))
    //  Shift the max value right by shift amount before finding the reciprocal
    // Shift all the sums in core asm loop, right by shift amount before multiplying by the reciprocal.
	shr = get_shr_val(maxsum);

	fixed_recip_level_size = 0x00ff0000/(maxsum >> shr);

#ifdef SUPERNODE_DEBUG
	printf("Initial values - HVX: fixed_recip_level_size = %lu, shr = %d outmin/max = [%f %f], maxsum = %ld, out_level=%f, bias_adder=%d\n", fixed_recip_level_size, shr, final_out_min_val, final_out_max_val, maxsum, out_level_size, bias_adder);
#endif
	worker_info.arg0 = fixed_recip_level_size;
	my_info.arg0 = fixed_recip_level_size;
	worker_info.arg1 = shr;
	my_info.arg1 = shr;

	//conv_start = HAP_perf_get_pcycles();
	nn_os_work_for_vector(nn,supernode_execute_hvx_slice,&worker_info);
	//nn_os_work_for_vector(nn,supernode_execute_hvx_slice,&my_info);
	//supernode_execute_hvx_slice(nn,&worker_info);
	supernode_execute_hvx_slice(nn,&my_info);

	nn_sem_wait(&worker_info.donesem);
	//nn_sem_wait(&my_info.donesem);
	record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER0,(my_info.cycles+worker_info.cycles)/2);
	//nonconv_start = HAP_perf_get_pcycles();


	//return 0;
	maxsum = my_info.mat_max;
	if(worker_info.mat_max > maxsum) maxsum = worker_info.mat_max;

	if ((!nodeinfo->maxval_precalculated) && 
		(((maxsum+bias_adder) * out_level_size) > (final_out_max_val-final_out_min_val))) {
		while ((nodeinfo->maxval-nodeinfo->minval) < ((maxsum+bias_adder) * out_level_size)) {
			nodeinfo->maxval *= 2;
			if (!nodeinfo->minval_precalculated) nodeinfo->minval = -nodeinfo->maxval;
		}
		logmsg(nn,1,"Supernode %lx maxvalue too small, retrying w/ %f... maxsum=%ld, bias_adder=%d, out_level=%f final_out_max_val=%f [new minval=%f]\n",
			self->node_id,nodeinfo->maxval, maxsum, bias_adder, out_level_size, final_out_max_val, nodeinfo->minval);
		return supernode_execute_hvx(self,nn);
	}
#ifdef SUPERNODE_DEBUG
	printf("HVX: fixed_recip_level_size = %lu, shr = %d outmin/max = [%f %f], maxsum = %ld, out_level=%f, bias_adder=%d\n", fixed_recip_level_size, shr, final_out_min_val, final_out_max_val, maxsum, out_level_size, bias_adder);
#endif
	/* Now go back through, add bias, clip to positive and requantize. */

	//nn_os_work_for_vector(nn,biasadd_relu_requant_execute_hvx_slice,&worker_info);
	//biasadd_relu_requant_execute_hvx_slice(nn,&my_info);
	//nn_sem_wait(&worker_info.donesem);

	//record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER1,(my_info.cycles+worker_info.cycles)/2);

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,final_out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,final_out_max_val);
	out_max->data_size = sizeof(float);
#if 0
	DISABLE_PMU();
#endif
	//nonconv_end = HAP_perf_get_pcycles();
	//record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER0,nonconv_start-conv_start);
	//record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER1,nonconv_end-nonconv_start);

	logmsg(nn,2,"supernode execute (hvx) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);

	return 0;
}

static int supernode_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking supernode node %p",self);
	if (self->n_inputs != 12) return errlog(nn,"supernode wrong # inputs (%d)",self->n_inputs);
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
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	uint32_t out_depth = filt_batches;
	uint8_t *filt = (uint8_t *)filt_tensor->data;
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
	uint32_t filt_elements = filt_width*filt_height*filt_depth;
	uint32_t filt_elements_pad = (filt_elements + HPAD - 1) & (~(HPAD - 1));
	int out_depth_pad = (out_depth + DPAD - 1) & ~(DPAD-1);
	uint32_t consts_size = filt_elements_pad * out_depth_pad;
	int vec_id;
	struct supernode_info *info;
	if ((info = (struct supernode_info *)malloc(sizeof(*info))) == NULL) {
		return errlog(nn,"couldn't allocate info");
	}
	info->maxval_precalculated = 1;
	info->maxval = tensor_get_float(self->inputs[11],0);
	info->minval_precalculated = 1;
	info->minval = tensor_get_float(self->inputs[10],0);
	if (info->maxval == INFINITY) {
		info->maxval_precalculated = 0;
		info->maxval = 0.5;
	}
	if (info->minval == -INFINITY) {
		info->minval_precalculated = 0;
		info->minval = -info->maxval;
	}
#if 0
	if (self->n_inputs == 11) {
		info->maxval_precalculated = 1;
		info->maxval = tensor_get_float(self->inputs[10],0);
	} else {
		info->maxval_precalculated = 0;
		info->maxval = 0.5;
	};
#endif
	self->opaque = info;
	if ((info->filt_pad = memalign(ALIGN_SIZE,consts_size)) == NULL) {
		return errlog(nn,"couldn't allocate buffer for const rearrangement");
	}
	vec_id = nn_os_vector_acquire();
	pad2d(filt,filt_elements,out_depth,(uint8_t*)nn->scratch,filt_elements_pad,out_depth_pad,filt_offset);
	transpack((const uint8_t*)nn->scratch,filt_elements_pad,out_depth_pad,(uint8_t*)info->filt_pad);
	nn_os_vector_release(vec_id);
	logmsg(nn,2,"supernode node %p check OK",self);
	return 0;
}


static int supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info *info = (struct supernode_info *)self->opaque;
	if (info != NULL) {
		free(info->filt_pad);
		free(info);
	}
	return node_free_common(self,nn);
}

#if 0
struct nn_node_ops nn_ops_for_QuantizedConv2d_8x8to32 = {
	SFINIT(.execute, conv2d_execute_hvx),
	SFINIT(  .check, conv2d_check_ref),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};
#endif

struct nn_node_ops nn_ops_for_Supernode_8x8p8to8 = {
	SFINIT(.execute, supernode_execute_hvx),
	//supernode_execute_ref,
	SFINIT(  .check, supernode_check_ref),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, supernode_dtor),
};

struct nn_node_ops nn_ops_for_Supernode_8x8p8to8_ref = {
	SFINIT(.execute, supernode_execute_ref),
	SFINIT(  .check, supernode_check_ref),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

