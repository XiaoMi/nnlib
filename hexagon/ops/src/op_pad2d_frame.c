
/* FIXME: should not be used, remove me */

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
#include <string.h>
#include <quantize.h>
#include <math.h>
#include <nn_broadcast.h>
#include <stdio.h>



static int pad2d_frame_q8_execute_ref(struct nn_node *self, struct nn_graph *nn)
{


	const struct tensor *in_tensor = self->inputs[0];

	const struct tensor *filt_tensor = self->inputs[1];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;


	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];


	const uint8_t *indata = in_tensor->data;
	uint8_t    *out_data = out_tensor->data;

	//int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	//int32_t filt_depth = filt_tensor->shape.filt_depth;

	int32_t out_batches = in_batches;
	int32_t out_width = in_width + (filt_width-1)*2;

	int32_t out_height = in_height + (filt_height-1)*2;
	int32_t out_depth  = in_depth;

	int32_t out_size   = out_batches * out_width * out_height * out_depth * sizeof(int8_t);

	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);

	if (out_size > (out_tensor->max_size)) {
		return errlog(nn,"output too small, %d < %d",out_tensor->max_size,out_size);
	}
	//if (stride_tensor->shape.batches != 1) return errlog(nn,"bad stride batch");
	//if (stride_tensor->shape.depth != 1) return errlog(nn,"bad stride depth");
	if (out_min->max_size < sizeof(float)) return errlog(nn,"min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn,"max too small");

	tensor_set_shape(out_tensor,out_batches,out_height,out_width,out_depth);

//	printf("%s Output Dimenstion %ld x %ld x %ld x %ld", __FUNCTION__, out_batches,out_height,out_width,out_depth);
	out_tensor->data_size = out_size;

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,in_min_float); //no change in min/max values
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,in_max_float);

	out_max->data_size = sizeof(float);


	{
		int i,j,k,l, index_new, index;

		int h_adj2 =  filt_height - 1;
		int w_adj2 =  filt_width  - 1;

		int in_height_new = in_height+h_adj2*2;
		int in_width_new = in_width+w_adj2*2;
		//uint8_t *out_data = nn->scratch;


#if 0 //Reference
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
				out_data[index_new] =0;
			else
				out_data[index_new] = indata[index];

			printf(" %d, ", out_data[index_new]);
		}
		printf("\n");
		}
#else
		int size;
		for(i=0;i< in_batches;i++)
		for(j=0;j< in_height_new;j++)
		{
			//for(k=0;k< in_width_new;k++)
			//for(l=0;l< in_depth ;l++)

			if((j < h_adj2) || (j >= in_height_new - h_adj2))
			{
				//Top fill
				l =0; k=0;
				index_new = ((i*in_height_new + j          )*in_width_new + k          )*in_depth + l;
				size      = in_width_new*in_depth;
				//memset(&out_data[index_new], 0, size);
				vmemset_asm(&out_data[index_new], 0, size);
			}
			else
			{
				l =0; k=0;
				index_new = ((i*in_height_new + j          )*in_width_new   )*in_depth;
				index     = ((i*in_height     + (j-h_adj2))*in_width        )*in_depth;

				size = w_adj2*in_depth;
				//Insert at the beginning of the row
				memset(&out_data[index_new], 0, size);

				//printf("new %d, Old %d, size %d\n", index_new, index, size);
				index_new +=size;
				//memcpy(&out_data[index_new+size], &indata[index], in_width*in_depth);
				vmemcpy_asm(&out_data[index_new], &indata[index], in_width*in_depth);
				index_new +=in_width*in_depth;

				memset(&out_data[index_new], 0, size*in_depth);

			}

		}

#endif

	}
	return 0 ;
}


/*
struct nn_node_ops nn_ops_for_pad2d_frame_int32 = {
	.execute = pad2d_frame_int32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};
*/

struct nn_node_ops nn_ops_for_Quantizedpad2d_frame_8p = {
	.execute = pad2d_frame_q8_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Quantizedpad2d_frame_8p_ref = {
	.execute = pad2d_frame_q8_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

