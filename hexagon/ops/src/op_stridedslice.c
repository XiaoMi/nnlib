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
 * Given a start offset and width and a stride for each dimention in the input tensor,
 * create a new output tensor with just the slice specified.
 * 
 * There were a lot of fancy features in the docs for another environment,
 * so we'll do a subset.
 */

#include <nn_graph.h>
#include <string.h>
#include "nn_gentranspose.h"
#include "quantize.h"

#define MASK_UPDATE_RANGE(BIT, DIM) \
{ \
	if (BIT & shrink_mask) { \
		DIM##_stop = DIM##_start + 1; \
		DIM##_step = 1; \
	} else { \
		if (BIT & begin_mask) DIM##_start = (DIM##_step < 0) ? DIM##_in-1 : 0; \
		if (BIT & end_mask) DIM##_stop = (DIM##_step < 0) ? -1 : DIM##_in; \
	}\
}

#define HANDLE_NEGATIVE_INDEX(DIM) \
{ \
    if(DIM##_start < 0) DIM##_start = max_i32(0, DIM##_start + DIM##_in); \
    if(DIM##_stop < 0) DIM##_stop = max_i32(-1, DIM##_stop + DIM##_in); \
}

// find the output size. It will be 0 (invalid) if stop-start is 0 or
// has the opposite sign as 'step'.
//
static inline unsigned get_out_size(int start, int stop, int step)
{
	int range = stop-start;
	if( step < 0){
		range = -range;
		step = -step;
	}
	if( range <= 0) return 0;
	return ((unsigned)range + (step-1))/(unsigned)step;
}

static inline void update_out(int shrink_mask, struct shape *shp) {
	struct shape scratch_shape = *shp;
	struct shape out_shape = { { { 1, 1, 1, 1 } } };
	int read_idx, write_idx;
	write_idx = 3;
	// collect dimensions from right
	for (read_idx = 3; read_idx >= 0; read_idx--) {
		if (0 == ((1 << read_idx) & shrink_mask)) {
			out_shape.dimension[write_idx--] = scratch_shape.dimension[read_idx];
		}
	}
	*shp = out_shape;
}
static inline int check_index( int start, int step, int outsize, int insize)
{
	int idx0 =start;					// first sample
	int idx1 = start + step*(outsize-1); // last
	if( step < 0){
		int t = idx1; idx1 = idx0; idx0 = t;		// put in order
	}
	return (0 <= idx0) && (idx1 < insize);
}


static int strided_slice_impl(
	struct nn_node *self,
	struct nn_graph *nn,
	int element_size,
	int typecode)
{
	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *start_tensor = self->inputs[1];
	const struct tensor *stop_tensor = self->inputs[2];
	const struct tensor *step_tensor = self->inputs[3];
	// optional parameters - cannot be optional for quantized op as input min/max come after
	//const struct tensor *begin_mask_tensor = self->inputs[4];
	//const struct tensor *end_mask_tensor = self->inputs[5];
	//const struct tensor *shrink_mask_tensor = self->inputs[6];
	struct tensor *out_tensor = self->outputs[0];
	int b_in = input_tensor->shape.batches;
	int h_in = input_tensor->shape.height;
	int w_in = input_tensor->shape.width;
	int d_in = input_tensor->shape.depth;
	const char *in = input_tensor->data;
	char *out = out_tensor->data;
	int32_t order = start_tensor->shape.depth;
	int b_start = (order < 4) ? 0 : tensor_get_int32(start_tensor,order-4);
	int h_start = (order < 3) ? 0 : tensor_get_int32(start_tensor,order-3);
	int w_start = (order < 2) ? 0 : tensor_get_int32(start_tensor,order-2);
	int d_start = (order < 1) ? 0 : tensor_get_int32(start_tensor,order-1);
	int b_stop = (order < 4) ? b_in : tensor_get_int32(stop_tensor,order-4);
	int h_stop = (order < 3) ? h_in : tensor_get_int32(stop_tensor,order-3);
	int w_stop = (order < 2) ? w_in : tensor_get_int32(stop_tensor,order-2);
	int d_stop = (order < 1) ? d_in : tensor_get_int32(stop_tensor,order-1);
	int b_step = (order < 4) ? 1 : tensor_get_int32(step_tensor,order-4);
	int h_step = (order < 3) ? 1 : tensor_get_int32(step_tensor,order-3);
	int w_step = (order < 2) ? 1 : tensor_get_int32(step_tensor,order-2);
	int d_step = (order < 1) ? 1 : tensor_get_int32(step_tensor,order-1);
	int begin_mask = 0;
	int end_mask = 0;
	int shrink_mask = 0;

    //convert negative slice indices
    HANDLE_NEGATIVE_INDEX(b);
    HANDLE_NEGATIVE_INDEX(h);
    HANDLE_NEGATIVE_INDEX(w);
    HANDLE_NEGATIVE_INDEX(d);

	if (self->n_inputs > 6) {
		begin_mask = tensor_get_int32(self->inputs[4], 0);
		end_mask = tensor_get_int32(self->inputs[5], 0);
		shrink_mask = tensor_get_int32(self->inputs[6], 0);

		MASK_UPDATE_RANGE(0x1, b);
		MASK_UPDATE_RANGE(0x2, h);
		MASK_UPDATE_RANGE(0x4, w);
		MASK_UPDATE_RANGE(0x8, d);
	}

	// check stride before dividing with it
	if (0 == b_step) return errlog(nn,"invalid b_step");
	if (0 == h_step) return errlog(nn,"invalid h_step");
	if (0 == w_step) return errlog(nn,"invalid w_step");
	if (0 == d_step) return errlog(nn,"invalid d_step");

	// for setting output shape only
	struct shape out_shape;

	out_shape.batches = get_out_size(b_start, b_stop, b_step);
	out_shape.height = get_out_size(h_start, h_stop, h_step);
	out_shape.width = get_out_size(w_start, w_stop, w_step);
	out_shape.depth = get_out_size(d_start, d_stop, d_step);

	int out_elements = shape_element_count( & out_shape);
	if (0 == out_elements) return errlog(nn,"no output");

	// check that all generated indices are in range (before we 'shrink' output shape)
	if(    !check_index(b_start,  b_step, out_shape.batches, b_in)
		|| !check_index(h_start,  h_step, out_shape.height, h_in)
		|| !check_index(w_start,  w_step, out_shape.width, w_in)
		|| !check_index(d_start,  d_step, out_shape.depth, d_in)){
		return errlog(nn,"indices beyond input range");
	}
	// output dims which are marked by 'shrink_mask' are removed from the output shape
	// keep the unmodified shape for the loop
	struct shape out_shape_actual = out_shape;
	if (shrink_mask != 0)
		update_out(shrink_mask, &out_shape_actual);

	uint32_t total_bytes = out_elements * element_size;
	int b,h,w,d;
	int offset;

	logmsg(nn,2,"begin_mask: %x end_mask: %x shrink_mask: %x", begin_mask, end_mask, shrink_mask);
	logmsg(nn,2,"slice node %p execute order=%d in=%dx%dx%dx%d start=%dx%dx%dx%d stop=%dx%dx%dx%d step=%dx%dx%dx%d out=%dx%dx%dx%d",
		self,order,
		b_in,h_in,w_in,d_in,
		b_start,h_start,w_start,d_start,
		b_stop,h_stop,w_stop,d_stop,
		b_step,h_step,w_step,d_step,
		out_shape_actual.batches,out_shape_actual.height,out_shape_actual.width,out_shape_actual.depth);

	if( tensor_out_prepare_normal_fromshape( out_tensor, &out_shape_actual, typecode)!=0){
		return errlog(nn,"out too small, %d > %d",total_bytes,out_tensor->max_size);
	}
	int copy_size = element_size;		// inner loop copy size
	if( d_step == 1){	// copy depth as a unit when possible
		copy_size *= out_shape.depth;
		out_shape.depth= 1;
	}

	// this is just so that the compiler doesn't need to make "0-loop-count" tests on all the inner loops.
	if( (int)out_shape.height < 1 || (int)out_shape.width< 1
			|| (int)out_shape.depth < 1) return 0;

	// if the inner copy size is the same as element size, maybe we can use a strided copy instead
	// of memcpy
	void (*strided_copy_2d_fp)( uint8_t * outp, uint8_t const *inp, int h, int w, int hsi, int wsi, int hso, int wso) = NULL;
	if( copy_size == element_size){	 // we didn't chunk depth at all...
		if( out_shape.width >=2 &&  out_shape.depth >= 2 )
			strided_copy_2d_fp = (element_size==1)?strided_copy_2d_1b:(element_size==2)? strided_copy_2d_2b
					:(element_size==4)? strided_copy_2d_4b: NULL;
	}

	if( strided_copy_2d_fp!=NULL){
		// let the strided copy do the innermost 2 loop levels.
		//printf("%d x %d x %d\n",  (int)out_shape.width,(int)out_shape.depth, element_size );
		for (int ib = 0; ib < (int)out_shape.batches; ib ++){		// output index
			b = b_start + ib*b_step;								// input index
			for (int ih = 0; ih < (int)out_shape.height; ih ++){
				h = h_start + ih*h_step;
				offset = element_size*(b*h_in*w_in*d_in + h*w_in*d_in + w_start *d_in + d_start);

				(*strided_copy_2d_fp)( (uint8_t*)out, (uint8_t const*)in+offset, out_shape.width, out_shape.depth,
						w_step * element_size*d_in,				// input stride on w dim
						d_step * element_size,						// input stride on d dim,
						element_size * out_shape.depth,				// output stride on w dim
						element_size );								// output stride on d dim
				out += element_size * out_shape.depth *out_shape.width;
			}
		}
		return 0;
	}

	// general case

	for (int ib = 0; ib < (int)out_shape.batches; ib ++){		// output index
		b = b_start + ib*b_step;								// input index
		for (int ih = 0; ih < (int)out_shape.height; ih ++){
			h = h_start + ih*h_step;
			for (int iw = 0; iw < (int)out_shape.width; iw ++){
				w = w_start + iw*w_step;
				for (int id = 0; id < (int)out_shape.depth; id ++){
					d = d_start + id*d_step;
					offset = element_size*(b*h_in*w_in*d_in
						+ h*w_in*d_in
						+ w*d_in
						+ d);
					memcpy(out,in+offset,copy_size);
					out += copy_size;
				}
			}
		}
	}

	return 0;
}

static int sslice_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return strided_slice_impl(self,nn,sizeof(float), NN_TYPE_FLOAT);
}
static int sslice_execute_int32(struct nn_node *self, struct nn_graph *nn)
{
	return strided_slice_impl(self,nn,sizeof(int32_t), NN_TYPE_INT32);
}

static int sslice_execute_1b(struct nn_node *self, struct nn_graph *nn)
{
	return strided_slice_impl(self,nn,1,NN_TYPE_UINT8);
}

static int sslice_execute_q8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[7]);
	tensor_copy(self->outputs[2],self->inputs[8]);
	return strided_slice_impl(self,nn,1, NN_TYPE_QUINT8);
}

// # inputs must be  4 or 7;
// IOCOUNT allows 4...7, this function excludes 5 and 6
static int sslice_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 4 && self->n_inputs != 7) return errlog(nn,"num inputs");
	return 0;
}


struct nn_node_ops nn_ops_for_StridedSlice_f = {
	.execute = sslice_execute_f,
	.check = sslice_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(4,7),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_StridedSlice_int32 = {
	.execute = sslice_execute_int32,
	.check = sslice_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(4,7),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_StridedSlice_uint8 = {
	.execute = sslice_execute_1b,
	.check = sslice_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(4,7),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_QuantizedStridedSlice_8 = {
	.execute = sslice_execute_q8,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(9),
	.n_outputs = NN_IOCOUNT(3),
};
