
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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

static int depthwiseconv2d_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *stride_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];

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


	const float *in = in_tensor->data;
	const float *filt = filt_tensor->data;
	float *out = out_tensor->data;
	float *outstripe;

	int32_t out_batches = in_batches;
	int32_t adj_x, adj_y;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,filt_width,stride_width,self->padding, & adj_x);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,filt_height,stride_height,self->padding, & adj_y);
	int32_t out_depth = in_depth * filt_batches;

	int32_t batch;
	int32_t out_y;
	int32_t out_x;
	int32_t z;
	int32_t mult;
	int32_t filt_y;
	int32_t filt_x;
	float in_element;
	float filt_element;
	int32_t in_x_base;
	int32_t in_y_base;
	float sum;

	logmsg(nn,2,"depthwiseconv2d f execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"depthwiseconv2d f input %dx%dx%dx%d",in_batches,in_height,in_width,in_depth);
	logmsg(nn,2,"depthwiseconv2d f filt %dx%dx%dx%d",filt_batches,filt_height,filt_width,filt_depth);
	logmsg(nn,2,"depthwiseconv2d f stride %dx%d",stride_height,stride_width);
	logmsg(nn,2,"depthwiseconv2d f padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%dx%dx%d",out_batches,out_height,out_width,out_depth);
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");

	if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");

	if ( tensor_out_prepare_normal( out_tensor, out_batches,out_height,out_width,out_depth, NN_TYPE_FLOAT)!=0 ){
		return errlog(nn,"output too small");
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
	              sum += in_element*filt_element;
	            }
	          }
	          outstripe[z*filt_batches+mult] = sum;
	        }
	      }
	    }
	  }
	}
	logmsg(nn,2,"depthwiseconv2d f execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}

static int depthwiseconv2d_check_f(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 3) return errlog(nn,"depthwiseconv2d f id %x wrong # inputs",self->node_id);
	if (self->n_outputs != 1) return errlog(nn,"depthwiseconv2d f wrong # outputs");
	return 0;
}

struct nn_node_ops nn_ops_for_DepthwiseConv2d_f = {
	.execute = depthwiseconv2d_execute_f,
	.check = depthwiseconv2d_check_f,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_CLS_DWCONVF,
};
// 'reference' (same thing, but immune to being transformed by prepare.c)
struct nn_node_ops nn_ops_for_DepthwiseConv2d_f_ref = {
	.execute = depthwiseconv2d_execute_f,
	.check = depthwiseconv2d_check_f,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
