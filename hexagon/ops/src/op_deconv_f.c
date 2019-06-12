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


static inline int32_t Minimum( int32_t a, int32_t b) { return a < b? a: b;}

static int deconv_f_execute_ref(struct nn_node *self, struct nn_graph *nn)
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

	const float *in = in_tensor->data;
	const float *filt = filt_tensor->data;
	float *out = out_tensor->data;

	const float *instripe;
	const float *filtstripe;
	float *outstripe;

	float in_element;
	float filt_element;
	float sum;

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

	int32_t out_size = out_batches * out_width * out_height * out_depth * sizeof(float);
	
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

	for (batch = 0; batch < out_batches; batch++) {
	  for (out_y = 0; out_y < out_height; out_y++) {
	    in_y_base = out_y + adj_y;
	    // range of filter to use on this row: all the 0 <= fy < filt_height but
	    //   (1) (out_y-fy)% stride_height must be 0
	    //   (2) in_y_base - fy   must be >= 0  			   		 => fy < in_y_base+1
	    //   (3) (in_y_base-fy)/stride_height must be < in_height      => fy > in_y_base - in_height*stride_height
	    int32_t fy0 = out_y % stride_height;										// first fy to meet (1)
	    int32_t fylim = Minimum(in_y_base+1,filt_height);							// fy < fylim for (2)
	    int32_t dely = (in_y_base - in_height*stride_height) - fy0;					// should be < 0 for (3)
	    if( dely >= 0) fy0 += stride_height*(dely/stride_height+1 );				// increase fy0 to meet (3)

	    for (out_x = 0; out_x < out_width; out_x++) {
	      in_x_base = out_x + adj_x;
	      int32_t fx0 = out_x % stride_width;
	      int32_t fxlim = Minimum(in_x_base+1,filt_width);
	      int32_t delx = (in_x_base - in_width*stride_width) - fx0;
	      if( delx >= 0) fx0 += stride_width*(delx/stride_width+1 );
	      outstripe = out+out_depth*(out_x+out_width*(out_y+(out_height*batch)));
	      for (out_z = 0; out_z < out_depth; out_z++) {
	        sum = 0.0f;
	        // only the filt_y where (filt_y-out_y) is a multiple of stride_height and  0 <= in_y < in_height
	        for (filt_y = fy0; filt_y < fylim; filt_y += stride_height) {
	          in_y = (in_y_base - filt_y) / stride_height;
		      // only the filt_x where (filt_x-out_x) is a multiple of stride_width and 0 <= in_x < in_width
	          for (filt_x = fx0; filt_x < fxlim; filt_x += stride_width) {
	            in_x = (in_x_base - filt_x) / stride_width;
	            instripe = in+in_depth*(in_x
                      +in_width*(in_y+in_height*batch));
	            filtstripe = filt+(out_z + out_depth*filt_depth*(filt_x
                      +filt_width*filt_y));
	            for (filt_z = 0; filt_z < filt_depth; filt_z++) {
	              in_element = instripe[filt_z];
	              filt_element = filtstripe[filt_z*out_depth];
	              sum += in_element * filt_element;
#if 0
	              printf("@oy=%d ox=%d od=%d iy=%d ix=%d id=%d instripe off=%d in_depth=%d in_width=%d: sum += %f*%f --> %f total\n",
	                (int)out_y,(int)out_x,(int)out_z,
	                (int)in_y,(int)in_x,(int)filt_z,
			(int)(instripe-in),(int)in_depth,(int)in_width,
	                in_element,filt_element,sum);
#endif
	            }
	          }
	        }
	        //printf("@ %d %d %d: out=%f\n",(int)out_y,(int)out_x,(int)out_z,sum);
	        outstripe[out_z] = sum;
	      }
	    }
	  }
	}

	logmsg(nn,2,"deconv_f execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}


struct nn_node_ops nn_ops_for_Deconv_f = {
	.execute = deconv_f_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};


