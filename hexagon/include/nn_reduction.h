
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
#ifndef NN_REDUCTION_H
#define NN_REDUCTION_H 1

#include <nn_graph.h>

/*
 * We have a 4D shape.
 * 
 * The reductions can reudce along any dimension 
 * 
 * For example, we could reduce all the dimensions and get a scalar output.
 * 
 * We could reduce the depth and still have height and width
 * 
 * We could reduce the height and width and still have depth
 * 
 * We will use a 1-pass algorithm for the output.
 * 
 * for (ob = 0; ob < out_batches; ob++) {
 *  for (oh = 0; oh < out_height; oh++) {
 *    for (ow = 0; ow < out_width; ow++) {
 *      for (od = 0; od < out_depth; od++) {
 *        // calculate initial input
 *        tmp = initalizer;
 *        for (rb = 0; rb < batches_to_reduce; rb++) {
 *          for (rh = 0; rh < height_to_reduce; rh++) {
 *            for (rw = 0; rw < width_to_reduce; rw++) {
 *              for (rd = 0; rd < depth_to_reduce; rd++) {
 *                // calculate this input position
 *                tmp = f(tmp,*pos);
 * }}}}}}}}
 * 
 * alternatively, copy to scratch, and then
 * if (reduce_batches) reduce_inplace(scratch,batches,height*width*depth)
 * for (i = 0; i < ob; i++) {
 *   if (reduce_height) reduce_inplace(scratch+ob*h*w*d,width*depth);
 * }
 * for (i = 0; i < oh; i++) {
 *   if (reduce_width) reduce_inplace(scratch+oh*w*d,depth);
 * }
 * for (i = 0; i < ow; i++) {
 *   if (reduce_depth) reduce_inplace(scratch+ow*d,depth);
 * }
 * But I think this leaves it strided somehow in the case that we reduce depth but not batches...
 * Going the other way should work... reduce depth in place, then width then height then batches
 * But I Think the big loops above are the best bet.
 */

#define CREATE_REDUCTION_INLINE(TYPENAME,TYPE) \
static inline TYPE nn_do_reductions_##TYPENAME( \
	const TYPE *in, \
	int b_in, \
	int h_in, \
	int w_in, \
	int d_in, \
	int rb, \
	int rh, \
	int rw, \
	int rd, \
	TYPE (*f)(TYPE,TYPE), \
	TYPE initializer) \
{ \
	TYPE tmp = initializer; \
	const TYPE *p; \
	int b,h,w,d; \
	for (b = 0; b < rb; b++) \
	  for (h = 0; h < rh; h++) \
	    for (w = 0; w < rw; w++) \
	      for (d = 0; d < rd; d++) { \
	        p = in + b * h_in * w_in * d_in + h * w_in * d_in + w * d_in + d; \
		tmp = f(tmp,*p); \
	} \
	/* finalizer? */ \
	return tmp; \
} \
static inline int nn_reduction_##TYPENAME( \
	struct nn_node *self, \
	struct nn_graph *nn, \
	TYPE (*f)(TYPE, TYPE), \
	TYPE initializer) \
{ \
	const struct tensor *input_tensor = (const struct tensor *)self->inputs[0]; \
	struct tensor *out_tensor = (struct tensor *)self->outputs[0]; \
	struct shape out_shape = input_tensor->shape; \
	int b_in = input_tensor->shape.batches; \
	int h_in = input_tensor->shape.height; \
	int w_in = input_tensor->shape.width; \
	int d_in = input_tensor->shape.depth; \
	int b_out = b_in; \
	int h_out = h_in; \
	int w_out = w_in; \
	int d_out = d_in; \
	int rb; \
	int rh; \
	int rw; \
	int rd; \
	int b,h,w,d; \
	int out_elements; \
	int out_bytes; \
	const TYPE *in; \
	const TYPE *base = (const TYPE *)input_tensor->data; \
	TYPE *out = (TYPE *)out_tensor->data; \
	if (self->n_inputs == 3) { \
		const struct tensor *reduction_dims_tensor = (const struct tensor *)self->inputs[1]; \
		const int32_t true_rank = tensor_get_int32(self->inputs[2],0); \
		const int32_t *dims = (const int32_t *)reduction_dims_tensor->data; \
		int32_t dim; \
		int i; \
		for (i = 0; i < reduction_dims_tensor->shape.depth; i++) { \
			dim = true_rank - dims[i] - 1; \
			if (dim > 3) d_out = w_out = h_out = b_out = 1; \
			if (dim == 0) d_out = 1; \
			if (dim == 1) w_out = 1; \
			if (dim == 2) h_out = 1; \
			if (dim == 3) b_out = 1; \
		} \
		out_shape.batches = b_out; \
		out_shape.height = h_out; \
		out_shape.width = w_out; \
		out_shape.depth = d_out; \
		if (self->padding == NN_PAD_VALID) { \
			/* Eliminate the dimensions that are reduced */ \
			for (i = 0; i < reduction_dims_tensor->shape.depth; i++) { \
				dim = true_rank - dims[i] - 1; \
				if (dim > 3) out_shape.depth =  \
					out_shape.width =  \
					out_shape.height =  \
					out_shape.batches = 0; \
				if (dim == 0) out_shape.depth = 0; \
				if (dim == 1) out_shape.width = 0; \
				if (dim == 2) out_shape.height = 0; \
				if (dim == 3) out_shape.batches = 0; \
			} \
			if (out_shape.batches == 0) { \
				out_shape.batches = 1; \
			} \
			if (out_shape.height == 0) { \
				out_shape.height = out_shape.batches; \
				out_shape.batches = 1; \
			} \
			if (out_shape.width == 0) { \
				out_shape.width = out_shape.height; \
				out_shape.height = out_shape.batches; \
				out_shape.batches = 1; \
			} \
			if (out_shape.depth == 0) { \
				out_shape.depth = out_shape.width; \
				out_shape.width = out_shape.height; \
				out_shape.height = out_shape.batches; \
				out_shape.batches = 1; \
			} \
		} \
	} else { \
		/* assume flatten */ \
		b_out = h_out = w_out = d_out = 1; \
		out_shape.batches = 1; \
		out_shape.height = 1; \
		out_shape.width = 1; \
		out_shape.depth = 1; \
	} \
	out_elements = b_out * h_out * w_out * d_out; \
	out_bytes = out_elements * sizeof(TYPE); \
	if (out_bytes > out_tensor->max_size) return errlog(nn,"out too small"); \
	out_tensor->shape = out_shape; \
	out_tensor->data_size = out_bytes; \
	/* calculate number of elements to reduce in each dimension */ \
	rb = (b_out == 1) ? b_in : 1; \
	rh = (h_out == 1) ? h_in : 1; \
	rw = (w_out == 1) ? w_in : 1; \
	rd = (d_out == 1) ? d_in : 1; \
	for (b = 0; b < b_out; b++) \
	  for (h = 0; h < h_out; h++) \
	    for (w = 0; w < w_out; w++) \
	      for (d = 0; d < d_out; d++) { \
	        in = base + b * h_in * w_in * d_in + h * w_in * d_in + w * d_in + d; \
	        *out++ = nn_do_reductions_##TYPENAME(in,b_in,h_in,w_in,d_in,rb,rh,rw,rd,f,initializer); \
	} \
	return 0; \
}


CREATE_REDUCTION_INLINE(float,float)
CREATE_REDUCTION_INLINE(int32,int32_t)

#undef CREATE_REDUCTION_INLINE



#endif

