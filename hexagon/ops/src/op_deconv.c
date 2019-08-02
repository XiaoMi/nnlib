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


#include <nn_graph.h>
#include <stdlib.h>
#include <stdio.h>
#include <quantize.h>
#ifndef __hexagon__
#include <malloc.h>
#else
#endif

static int deconv_execute_ref(struct nn_node *self, struct nn_graph *nn)
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
	int32_t out_width = nn_pad_compute_outsize_inverse(in_width,filt_width,stride_width,self->padding);
	int32_t out_height = nn_pad_compute_outsize_inverse(in_height,filt_height,stride_height,self->padding);
	int32_t out_depth = filt_batches;

	int32_t adj_x;
	int32_t adj_y;

	int32_t batch;
	int32_t filt_x;
	int32_t filt_y;
	int32_t filt_z;
	int32_t out_x;
	int32_t out_y;
	int32_t out_z;

	int32_t in_y_base;
	int32_t in_x_base;
	int32_t in_y;
	int32_t in_x;

	const uint8_t *in = in_tensor->data;
	const uint8_t *filt = filt_tensor->data;
	int32_t *out = out_tensor->data;

	const uint8_t *instripe;
	const uint8_t *filtstripe;
	int32_t *outstripe;

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);

	int32_t in_element;
	int32_t filt_element;
	int32_t sum;

	float in_level_size = (in_max_float - in_min_float) / 255;
	float filt_level_size = (filt_max_float - filt_min_float) / 255;
	float out_level_size = in_level_size * filt_level_size;
	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	int32_t input_offset = quantize_int(0.0f,in_min_float,in_max_float);
	int32_t filt_offset = quantize_int(0.0f,filt_min_float,filt_max_float);

#if 0
	printf("in: %dx%dx%d filt=%dx%d stride=%dx%d,padding=%d\n",
		(int)in_height,(int)in_width,(int)in_depth,
		(int)filt_height,(int)filt_width,
		(int)stride_height,(int)stride_width,
		(int)self->padding);
#endif
	// note, this is based on *output* size
	nn_pad_compute_outsize_and_padbefore( out_width, filt_width, stride_width, self->padding , & adj_x);
	nn_pad_compute_outsize_and_padbefore( out_height, filt_height, stride_height, self->padding , & adj_y);

	//printf("adj_x = %d adj_y = %d\n",(int)adj_x,(int)adj_y);

	int32_t out_size = out_batches * out_width * out_height * out_depth * sizeof(int32_t);
	
	logmsg(nn,2,"deconv execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"deconv input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
	logmsg(nn,2,"deconv filt %dx%dx%dx%d",filt_batches,filt_height,filt_width,filt_depth);
	logmsg(nn,2,"deconv stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"deconv padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);

	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if (out_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",out_tensor->max_size,out_size);
	}
	if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");

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
	    in_y_base = out_y + adj_y;
	    for (out_x = 0; out_x < out_width; out_x++) {
	      in_x_base = out_x + adj_x;
	      outstripe = out+out_depth*(out_x+out_width*(out_y+(out_height*batch)));
	      for (out_z = 0; out_z < out_depth; out_z++) {
	        sum = 0.0f;
	        for (filt_y = 0; filt_y < filt_height; filt_y++) {
	          if ((out_y - filt_y) % stride_height) continue;
	          in_y = (in_y_base - filt_y) / stride_height;
	          if (in_y >= in_height) continue;
	          if (in_y < 0) continue;
	          for (filt_x = 0; filt_x < filt_width; filt_x++) {
	            if ((in_x_base - filt_x) % stride_width) continue;
	            in_x = (in_x_base - filt_x) / stride_width;
	            if (in_x >= in_width) continue;
	            if (in_x < 0) continue;
	            instripe = in+in_depth*(in_x
                      +in_width*(in_y+in_height*batch));
	            filtstripe = filt+(out_z + out_depth*filt_depth*(filt_x
                      +filt_width*filt_y));
	            for (filt_z = 0; filt_z < filt_depth; filt_z++) {
	              in_element = instripe[filt_z];
	              in_element -= input_offset;
	              filt_element = filtstripe[filt_z*out_depth];
	              filt_element -= filt_offset;
	              sum += in_element * filt_element;
#if 0
	              printf("@oy=%d ox=%d od=%d iy=%d ix=%d id=%d instripe off=%d in_depth=%d in_width=%d: sum += %f*%f --> %f total\n",
	                (int)out_y,(int)out_x,(int)out_z,
	                (int)in_y,(int)in_x,(int)filt_z,
			(int)(instripe-in),(int)in_depth,(int)in_width,
	                in_element,filt_element,sum);
#endif
	              //printf("[%d %d]  ", instripe -in, filtstripe - filt);

	            }
	          }
	        }
	        //printf("@ %d %d %d: out=%f\n",(int)out_y,(int)out_x,(int)out_z,sum);
	        outstripe[out_z] = sum;
	        //printf("deconv-ref [%ld][%ld][%ld]  [%ld] = %ld \n",batch,out_y, out_x, out_z, outstripe[out_z]);

	      }
	    }
	  }
	}

	logmsg(nn,2,"deconv execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}

//Temp changes
#ifdef __hexagon__


#define VPAD 8
#define HPAD 16
#define DPAD 32
static int32_t max(int a, int32_t b) { return((a>b) ? a : b); }



#define ALIGN_SIZE 128
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((ALIGN_SIZE)-1)))
static inline void *pad_and_align(void *ptr, unsigned long minsize)
{
	uintptr_t ptrval = (uintptr_t)(ptr);
	ptrval += minsize + (ALIGN_SIZE-1);
	ptrval &= ~(ALIGN_SIZE-1);
	return (void *)ptrval;
}




static int conv2d_execute_hvx_mod(struct nn_node *self, struct nn_graph *nn,  uint8_t *filt)
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
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,filt_height,stride_height,self->padding, &adj_y);
	int32_t out_depth = filt_batches;

	int32_t batch;

//	int32_t in_y_base;
//	int32_t in_x_base;

	uint8_t *in = in_tensor->data;
	//uint8_t *filt = filt_tensor->data;
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
          im2col_co(&in[batch*in_height*in_width*in_depth], in_height,in_width,in_depth, input_offset,
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

/*
 To compute deconv- HVX -> We can re-use conv2d - HVX, with following procedure.

  1) Pad the inputs around height x weight  (IE Call Quantized pad2d_frame OP, before calling this OP)
  2) Reverse the filter - a) If filter->batches ==1,  reverse the filter across height and weight.
 										{ [a,b],      { [d,c],
									      [c,d]} --->   [b,a]  }

					      b) If filter->batches > 1, reverse the filter's' , only. The internal height x weight remains unchanged.
								Filter batches=4 - 4x2x2x1   { [a], [b],[c],[d]}, each a,b,c,d is 2x2.
										           ------->  { [d], [c],[b],[a]}


  3) Now call the conv2d-HVX as before. Note: filter are const.- for modify function with filter being passed additionally.
  	  	  If Step (2) is done initially - while - creating the filters, we could avoid 1) mallocs for the filters.2) conv2d_execute_hvx_mod
*/

static __attribute__((unused)) int deconv_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{

	const struct tensor *filt_tensor = self->inputs[1];

	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	const uint8_t *filt = filt_tensor->data;
	int i,j,k,l, index_new, index;

	uint8_t    *filt_new;
	uint32_t   filt_size = filt_batches*filt_height*filt_width*filt_depth*sizeof(int8_t);

	if ((filt_new = nn_malloc(filt_size)) == NULL) return errlog(nn,"tmp data storage fail");
	logmsg(nn,3,"malloced -temp data-  %p, size %ld",filt_new, nn->scratch_size);



	// Reverse the filter......
	for(i=0;i< filt_batches;i++)
	for(j=0;j< filt_height;j++)
	for(k=0;k< filt_width;k++)
	for(l=0;l< filt_depth ;l++)
	{
		//Flip height & width
		index     = (                  (i *filt_height  +                  j  )*filt_width +                 k )*filt_depth + l;

		if(filt_batches > 1)
			index_new = (((filt_batches -1 -i)*filt_height  +                  j  )*filt_width +                 k )*filt_depth + l;
		else
			index_new = ((                  i *filt_height  + (filt_height-1 - j) )*filt_width + (filt_width-1  -k))*filt_depth + l;

		filt_new[index_new] = filt[index];
		filt_new[index]     = filt[index_new];

	}

	conv2d_execute_hvx_mod(self, nn,  filt_new);


	nn_free(filt_new);
	return 0;
}
#endif


struct nn_node_ops nn_ops_for_QuantizedDeconv_8x8to32 = {
//	.execute = deconv_execute_hvx,
	.execute = deconv_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedDeconv_8x8to32_ref = {
	.execute = deconv_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};


