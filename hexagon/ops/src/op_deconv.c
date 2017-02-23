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
	            }
	          }
	        }
	        //printf("@ %d %d %d: out=%f\n",(int)out_y,(int)out_x,(int)out_z,sum);
	        outstripe[out_z] = sum;
	        //printf("deconv-ref [%ld][%ld]  [%ld] = %ld \n",out_y, out_x, out_z, outstripe[out_z]);

	      }
	    }
	  }
	}

	logmsg(nn,2,"deconv execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}

#if 0
int conv2d_execute_ref_mod(struct nn_node *self, struct nn_graph *nn, uint8_t *in , int32_t in_batches, int32_t in_width , int32_t in_height, int32_t in_depth, uint8_t *filt )
{
	//const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	//int32_t in_batches = in_tensor->shape.batches;
	//int32_t in_width = in_tensor->shape.width;
	//int32_t in_height = in_tensor->shape.height;
	//int32_t in_depth = in_tensor->shape.depth;

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

	//uint8_t *in = in_tensor->data;

	//uint8_t *filt = filt_tensor->data;
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

	printf("%s Output Dimenstion %ld x %ld x %ld x %ld", __FUNCTION__, out_batches,out_height,out_width,out_depth);
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
	//logmsg(nn,9,"[%d %d %d %d]: sum += %d*%d --> %d",
	//	batch,out_y,out_x,out_z,in_element,filt_element,sum);
	            }
	          }
	        }
	        outstripe[out_z] = sum;
	        printf("conv-mod [%ld][%ld]  [%ld] = %ld \n",out_y, out_x, out_z, outstripe[out_z]);
	      }
	    }
	  }
	}
	logmsg(nn,2,"conv2d execute (ref) done! %dx%dx%dx%d",
		out_batches,out_height,out_width,out_depth);
	return 0;
}


static int deconv_execute_conv_ref(struct nn_node *self, struct nn_graph *nn)
{

	const struct tensor *in_tensor = self->inputs[0];

	const struct tensor *filt_tensor = self->inputs[1];
	/*
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	*/
	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;
	const uint8_t *indata = in_tensor->data;

	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	const uint8_t *filt = filt_tensor->data;
	uint8_t    filt_new[100*100];
	{
		int i,j,k,l, index_new, index;

		int h_adj2 =  filt_height/2+1;
		int w_adj2 =  filt_width/2 +1;

		int in_height_new = in_height+h_adj2*2;
		int in_width_new = in_width+w_adj2*2;
		uint8_t *temp_data = nn->scratch;

		printf("<--------------Input data %ld x %ld x %ld x %ld\n", in_batches,in_height, in_width,in_depth);
		for(i=0;i< in_batches;i++)
		for(j=0;j< in_height;j++)
		{
		for(k=0;k< in_width;k++)
		for(l=0;l< in_depth ;l++)
		{
			index     = ((i*in_height     + j)*in_width     + k)*in_depth + l;
			printf(" %d, ", indata[index]);
		}
		printf("\n");
		}

		printf("<---------------Modified Data----%ld x %d x %d x %ld\n", in_batches,in_height_new, in_width_new,in_depth);
		for(i=0;i< in_batches;i++)
		for(j=0;j< in_height_new;j++)
		{
		for(k=0;k< in_width_new;k++)
		for(l=0;l< in_depth ;l++)
		{
			index_new = ((i*in_height_new + j          )*in_width_new + k          )*in_depth + l;
			index     = ((i*in_height     + (j-h_adj2))*in_width     + (k -w_adj2))*in_depth + l;

			if((j < h_adj2) || (j >= (in_height+h_adj2)) ||
			   (k < w_adj2) || (k >= (in_width +w_adj2))  )
				temp_data[index_new] =0;
			else
				temp_data[index_new] = indata[index];

			printf(" %d, ", temp_data[index_new]);
		}
		printf("\n");
		}

		in_width  = in_width_new;
		in_height = in_height_new;

		printf("------------- Original Filter-----------%ld x %ld x %ld x %ld----------->\n",filt_batches,filt_height,filt_width ,  filt_depth);
		for(i=0;i< filt_batches;i++)
		for(j=0;j< filt_height;j++)
		{
		for(k=0;k< filt_width;k++)
		{
		for(l=0;l< filt_depth ;l++)
		{
			index     = ((i*filt_height +               j )*filt_width + k)*filt_depth + l;
			printf(" %d ",  filt[index]);
		}
		printf(" ], [ ");
		}
		printf("\n");
		}



		// Reverse the filter......
		printf("\n");
		for(i=0;i< filt_batches;i++)
		for(j=0;j< filt_height;j++)
		{
		for(k=0;k< filt_width/2+1;k++)
		for(l=0;l< filt_depth ;l++)
		{
			index     = ((i*filt_height + j )*filt_width +                 k )*filt_depth + l;
			index_new = ((i*filt_height + j )*filt_width + (filt_width-1  -k))*filt_depth + l;

			filt_new[index_new] = filt[index];
			filt_new[index]     = filt[index_new];

			//printf("[%d] %d [%d] = %d, \n",index, filt_new[index],index_new, filt_new[index_new]);
		}
		printf("\n");
		}


		printf("------------- Modified Filter---------------------->\n");
		for(i=0;i< filt_batches;i++)
		for(j=0;j< filt_height;j++)
		{
		for(k=0;k< filt_width;k++)
		{
		for(l=0;l< filt_depth ;l++)
		{
			index     = ((i*filt_height +               j )*filt_width + k)*filt_depth + l;
			printf(" %d ",  filt_new[index]);
		}
		printf(" ], [ ");
		}
		printf("\n");
		}

	}

	conv2d_execute_ref_mod(self, nn, nn->scratch, in_batches, in_width , in_height, in_depth, filt_new);


	return 0;
}
#endif

static int deconv_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking deconv node %p",self);
	if (self->n_inputs != 7) return errlog(nn,"deconv id %x wrong # inputs",self->node_id);
	if (self->n_outputs != 3) return errlog(nn,"deconv wrong # outputs");
	logmsg(nn,2,"deconv node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedDeconv_8x8to32 = {
//	.execute = deconv_execute_conv_ref,
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


