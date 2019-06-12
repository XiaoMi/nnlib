
/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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

/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains a tile node
 */

static int tile_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"tile execute. self=%p ",self);

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *multiples_tensor = self->inputs[1];

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	uint8_t *in_data = in_tensor->data;
	uint32_t *multiples = multiples_tensor->data;
	float in_min = tensor_get_float(self->inputs[2], 0);
	float in_max = tensor_get_float(self->inputs[3], 0);

	uint8_t *out_data = out_tensor->data;
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;

	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;

	int dwh = depth*width*height;
	int dw = depth*width;

    int multiple_size = multiples_tensor->shape.batches * multiples_tensor->shape.height * multiples_tensor->shape.width * multiples_tensor->shape.depth;

	int b_multiple = multiple_size > 3 ? multiples[0] : 1;
	int h_multiple = multiple_size > 2 ? multiples[multiple_size-3] : 1;
	int w_multiple = multiple_size > 1 ? multiples[multiple_size-2] : 1;
	int d_multiple = multiples[multiple_size-1];

	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_tensor, batches*b_multiple, height*h_multiple, width*w_multiple, depth*d_multiple, NN_TYPE_QUINT8);

	*out_min = in_min;
	*out_max = in_max;

	int chunk_size_d = depth;
	int chunk_size_w = chunk_size_d * d_multiple*width;
	int chunk_size_h = chunk_size_w * w_multiple*height;
	int chunk_size_b = chunk_size_h * h_multiple*batches;

	int counter = 0;

	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );

	for(int b = 0; b < batches; b++){
		int c1 = b*dwh;
		uint8_t *copy_start_w = out_data+counter;
		for(int h = 0; h < height; h++){
			int c2 = h*dw + c1;
			uint8_t *copy_start_d = out_data+counter;
			for(int w = 0; w < width; w++){
				int c3 = w*depth + c2;
				for(int d_mul = 0; d_mul < d_multiple; d_mul++){
					nn_mcmanager_vmemcpy( nn, &mcman, out_data+counter, in_data+c3, chunk_size_d );
					counter += chunk_size_d;
				}
			}
			nn_mcmanager_wait( nn, &mcman );
			for(int w_mul = 1; w_mul < w_multiple; w_mul++){
				nn_mcmanager_vmemcpy( nn, &mcman, out_data+counter, copy_start_d, chunk_size_w );
				counter += chunk_size_w;
			}
		}
		nn_mcmanager_wait( nn, &mcman );
		for(int h_mul = 1; h_mul < h_multiple; h_mul++){
			nn_mcmanager_vmemcpy( nn, &mcman, out_data+counter, copy_start_w, chunk_size_h );
			counter += chunk_size_h;
		}
	}
	nn_mcmanager_wait( nn, &mcman );
	for(int b_mul = 1; b_mul < b_multiple; b_mul++){
		nn_mcmanager_vmemcpy( nn, &mcman, out_data+counter, out_data, chunk_size_b );
		counter += chunk_size_b;
	}

	nn_mcmanager_wait( nn, &mcman );

	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedTile_8 = {
	.execute = tile_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

