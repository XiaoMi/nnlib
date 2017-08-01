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
	int32_t out_width = nn_pad_compute_outsize(in_width,filt_width,stride_width,self->padding);
	int32_t out_height = nn_pad_compute_outsize(in_height,filt_height,stride_height,self->padding);
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

	const uint8_t *in = (uint8_t *)in_tensor->data;
	const uint8_t *filt = (uint8_t *)filt_tensor->data;
	int32_t *out = (int32_t *)out_tensor->data;

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
	while (in_width != nn_pad_compute_outsize(
		out_width,
		filt_width,
		stride_width,
		self->padding)) out_width++;
	while (in_height != nn_pad_compute_outsize(
		out_height,
		filt_height,
		stride_height,
		self->padding)) out_height++;

	adj_x = ((in_width-1) * stride_width + filt_width - out_width) / 2;
	adj_y = ((in_height-1) * stride_height + filt_height - out_height) / 2;
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





static int deconv_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking deconv node %p",self);
	if (self->n_inputs != 7) return errlog(nn,"deconv id %x wrong # inputs",self->node_id);
	if (self->n_outputs != 3) return errlog(nn,"deconv wrong # outputs");
	logmsg(nn,2,"deconv node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedDeconv_8x8to32 = {
	.execute = deconv_execute_ref,
	.check = deconv_check_ref,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedDeconv_8x8to32_ref = {
	.execute = deconv_execute_ref,
	.check = deconv_check_ref,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};


