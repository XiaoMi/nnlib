
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
 * This contains the code for "depthwise convolution"
 */

#include <nn_graph.h>
#include <quantize.h>

static int depthwiseconv2d_execute_ref(struct nn_node *self, struct nn_graph *nn)
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


	const uint8_t *in = (uint8_t *)in_tensor->data;
	const uint8_t *filt = (uint8_t *)filt_tensor->data;
	int32_t *out = (int32_t *)out_tensor->data;
	int32_t *outstripe;

	int32_t out_batches = in_batches;
	int32_t adj_x, adj_y;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,filt_width,stride_width,self->padding, & adj_x);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,filt_height,stride_height,self->padding, & adj_y);
	int32_t out_depth = in_depth * filt_batches;

	int32_t out_elements = out_batches*out_height*out_width*out_depth;
	size_t out_size = out_elements*sizeof(int32_t);

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);


	float in_level_size = (in_max_float - in_min_float) / 255;
	float filt_level_size = (filt_max_float - filt_min_float) / 255;
	float out_level_size = in_level_size * filt_level_size;

	float out_max_val = ((float)(INT32_MAX)) * out_level_size;
	float out_min_val = ((float)(INT32_MIN)) * out_level_size;

	int32_t input_offset = quantize_int(0.0f,in_min_float,in_max_float);
	int32_t filt_offset = quantize_int(0.0f,filt_min_float,filt_max_float);

	int32_t batch;
	int32_t out_y;
	int32_t out_x;
	int32_t z;
	int32_t mult;
	int32_t filt_y;
	int32_t filt_x;
	int32_t in_element;
	int32_t filt_element;
	int32_t in_x_base;
	int32_t in_y_base;
	int32_t sum;

	logmsg(nn,2,"depthwiseconv2d execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"depthwiseconv2d input %dx%dx%dx%d [%f..%f]",in_batches,in_height,in_width,in_depth,in_min_float,in_max_float);
	logmsg(nn,2,"depthwiseconv2d filt %dx%dx%dx%d [%f..%f]",filt_batches,filt_height,filt_width,filt_depth,filt_min_float,filt_max_float);
	logmsg(nn,2,"depthwiseconv2d stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"depthwiseconv2d padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	logmsg(nn,2,"out_level_size: %f out_min=%f out_max=%f",out_level_size,out_max_val,out_min_val);
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");
	if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");

	if( tensor_out_prepare_normal(out_tensor,out_batches,out_height,out_width,out_depth, NN_TYPE_INT32)!=0 ){
		return errlog(nn,"output too small, %d < %d",out_tensor->max_size,out_size);
	}
	if( tensor_set_single_float( out_min, out_min_val)!= 0
		|| tensor_set_single_float( out_max, out_max_val) ){
		return errlog(nn,"min or max too small");
	}

	for (batch = 0; batch < out_batches; batch++) {
	  for (out_y = 0; out_y < out_height; out_y++) {
	    in_y_base = out_y * stride_height - adj_y;
	    for (out_x = 0; out_x < out_width; out_x++) {
	      in_x_base = out_x * stride_width - adj_x;
	      outstripe = out+(out_depth*(out_x+
	                       out_width*(out_y+
	                       out_height*(batch))));
	      for (z = 0; z < in_depth; z++) {
	        for (mult = 0; mult < filt_batches; mult++) {
	          sum = 0;
	          for (filt_y = 0; filt_y < filt_height; filt_y++) {
	            if ((in_y_base + filt_y) >= in_height) continue;
	            if ((in_y_base + filt_y) < 0) continue;
	            for (filt_x = 0; filt_x < filt_width; filt_x++) {
	              if ((in_x_base + filt_x) >= in_width) continue;
	              if ((in_x_base + filt_x) < 0) continue;
		      in_element = in[z+in_depth*(in_x_base+filt_x+
	                             in_width*(in_y_base+filt_y+
                                     in_height*(batch)))];
	              filt_element = filt[(z*filt_batches+mult)+ 
					  filt_batches*filt_depth*(filt_x+
					  filt_width*(filt_y))];
	              in_element -= input_offset;
	              filt_element -= filt_offset;
	              sum += in_element*filt_element;
	            }
	          }
	          outstripe[z*filt_batches+mult] = sum;
	        }
	      }
	    }
	  }
	}
	logmsg(nn,2,"conv2d execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedDepthwiseConv2d_8x8to32 = {
	.execute = depthwiseconv2d_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedDepthwiseConv2d_8x8to32_ref = {
	.execute = depthwiseconv2d_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

