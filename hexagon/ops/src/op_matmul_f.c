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
#include <stdlib.h>

//
// matmul a * b
//      A shape = [b,h,w,din ]
//      B shape = [1,1,din, dout]
// result shape = [b,h,w,dout]
//
// or in general (to be more robust to use cases):
//
// If b.wid divides a.depth:
//
//      A shape = [b,h,w,k*din ]
//      B shape = [1,1,din, dout]
// result shape = [b,h,w*k, dout ]
//
// or if b.wid doesn't divide a.dep, but it divides a.ht * a.wid * a.dep,
// flatten the whole batch:
//      A shape = [ a.bat, a.ht, a.wid. a.dep ]
//      B shape = [ 1,1, b.wid, b.dep ]
//   result = [ a.bat, 1,  (a.ht *  a.wid * a.dep)/b.wid,  b.dep ]
//


static int matmul_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	uint32_t a_batches = a_tensor->shape.batches;
	uint32_t a_width = a_tensor->shape.width;
	uint32_t a_height = a_tensor->shape.height;
	uint32_t a_depth = a_tensor->shape.depth;

	uint32_t b_batches = b_tensor->shape.batches;
	uint32_t b_width = b_tensor->shape.width;
	uint32_t b_height = b_tensor->shape.height;
	uint32_t b_depth = b_tensor->shape.depth;

	uint32_t out_batches = a_batches;
	uint32_t out_height = a_height;			// may change
	uint32_t out_width = a_width;			// may change
	uint32_t out_depth = b_depth;

	int32_t x;
	int32_t y;
	int32_t i;

	const float *a = a_tensor->data;
	const float *b = b_tensor->data;
	float *out = out_tensor->data;

	float adata;
	float bdata;
	float sum;

	logmsg(nn,2,"matmul execute. self=%p",self);
	logmsg(nn,2,"matmul in dims: %dx%dx%dx%d * %dx%dx%dx%d",
		a_batches,a_height,a_width,a_depth,
		b_batches,b_height,b_width,b_depth);
	if (b_batches != 1 || b_height != 1) return errlog(nn,"b-side must be 2-d");

	uint32_t prod_ahw = a_height * a_width;

	if( a_depth != b_width){		// special cases
		unsigned k;
		if( a_depth >= 2*b_width && ( k = a_depth/b_width,   b_width*k == a_depth)){
			// a_depth = k * b_width
			out_width = a_width * k;
			prod_ahw *= k;		// pretend a is [ a_batches, a_height, a_width*k, b_width]
		}else{
			uint32_t prod_ahwd =  prod_ahw * a_depth;
			k = prod_ahwd / b_width;
			if( b_width * k == prod_ahwd ){		// pretend a is [a_batches, 1, prod_ahwd/b_width, b_width ]
				out_height = 1;
				out_width = k;
				prod_ahw = k;
			}else{
				return errlog(nn, "shape mismatch");
			}
		}
	}

	logmsg(nn,2,"matmul out dims: %dx%dx%dx%d", (unsigned) out_batches, (unsigned) out_height, (unsigned)out_width, (unsigned)out_depth);


	if( tensor_out_prepare_normal( out_tensor, out_batches,out_height,out_width,out_depth, NN_TYPE_FLOAT)!= 0){
		return errlog(nn,"output too small");
	}
	uint32_t a_outerdims = a_batches * prod_ahw;
	//
	// the product is equivalent to [ a_outerdims, b_width] * [ b_width, b_depth ] -> [a_outerdims, b_depth]
	// and then reshape the result to [a_batches, out_height, out_width, b_depth]
	//

	for (y = 0; y < a_outerdims; y++) {
		for (x = 0; x < b_depth; x++) {
			sum = 0.0f;
			for (i = 0; i < b_width; i++) {
				adata = a[i+y*b_width];
				bdata = b[x+i*b_depth];
				sum += adata * bdata;
				//printf("y=%ld, x=%ld, i=%ld, adata=%f, bdata=%f, sum=%f\n",y,x,i,adata,bdata,sum);
			}
			out[x+y*out_depth] = sum;
		}
	}
	logmsg(nn,2,"matmul execute (ref) done!");
	return 0;
}



struct nn_node_ops nn_ops_for_MatMul_f = {
	.execute = matmul_execute_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

