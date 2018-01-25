
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
 * Given a start offset and width for each dimention in the input tensor,
 * create a new output tensor with just the slice specified.
 * 
 */
/*
 *  inputs:
 *     0 - data tensor
 *     1 - start spec (see below)
 *     2 - size spec (see below)
 *     3,4 (if quantized): input min,max
 *
 *     The 'start spec' and 'size spec' are both tensors of shape (1,1,1,k)
 *     where k = 1..4.
 *     e.g. you can have
 *         start_spec [1,1,1,4] = { b0, h0, w0, d0 }
 *         size_spec [1,1,1,4] = { b_size, h_size, w_size, d_size }
 *
 *    And then the slice will be  [b0 ... b0+b_size-1] on batches, etc.
 *
 *    - For k <= 4, both 'start spec' and 'size spec' must still have the same shape,
 *     and the missing dimensions are dropped on the left; e.g.
 *         start_spec [1,1,1,2] = { w0, d0 }
 *         size_spec [1,1,1,2] = { w_size, d_size }
 *       .. and the 'batch' and 'height' dimensions are not sliced.
 *     Note:
 *         a 'size' of -1 means all available  (so start = 0, size=-1 means retain all)

 */
#include <nn_graph.h>
#include <string.h>

static int slice_impl(
	struct nn_node *self,
	struct nn_graph *nn,
	int element_size,
	int typeid )
{
	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *start_tensor = self->inputs[1];
	const struct tensor *size_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	int b,b_in,b_start,b_size;
	int h,h_in,h_start,h_size;
	int w,w_in,w_start,w_size;
	int d_in,d_start,d_size;

	logmsg(nn,2,"slice node %p execute",self);

	int nspec = start_tensor->shape.depth;
	if( start_tensor->shape.batches != 1 || start_tensor->shape.height != 1
		|| start_tensor->shape.width != 1 || nspec < 1 || nspec > 4
		|| !shape_matches( &start_tensor->shape, &size_tensor->shape) ){
		return errlog(nn,"bad size/shape spec for 'slice'");
	}

	b_in = input_tensor->shape.batches;
	h_in = input_tensor->shape.height;
	w_in = input_tensor->shape.width;
	d_in = input_tensor->shape.depth;

	b_start = 0;	b_size = -1;
	h_start = 0; 	h_size = -1;
	w_start = 0;	w_size = -1;
	d_start = tensor_get_int32( start_tensor, nspec-1);
	d_size = tensor_get_int32( size_tensor, nspec-1);

	if( nspec >= 2 ){
		w_start = tensor_get_int32(start_tensor,nspec-2);
		w_size = tensor_get_int32(size_tensor,nspec-2);
		if( nspec >= 3 ){
			h_start = tensor_get_int32(start_tensor,nspec-3);
			h_size = tensor_get_int32(size_tensor,nspec-3);
			if( nspec >= 4){
				b_start = tensor_get_int32(start_tensor,0);
				b_size = tensor_get_int32(size_tensor,0);
			}
		}
	}

	if (b_size == -1) b_size = b_in - b_start;
	if (h_size == -1) h_size = h_in - h_start;
	if (w_size == -1) w_size = w_in - w_start;
	if (d_size == -1) d_size = d_in - d_start;


	logmsg(nn,2,"in/start/size: b: %d/%d/%d h: %d/%d/%d w: %d/%d/%d d: %d/%d/%d order_skip=%d",
		b_in,b_start,b_size,
		h_in,h_start,h_size,
		w_in,w_start,w_size,
		d_in,d_start,d_size,
		4-nspec);

	if (b_size <= 0) return errlog(nn,"bad b_size");
	if (h_size <= 0) return errlog(nn,"bad h_size");
	if (w_size <= 0) return errlog(nn,"bad w_size");
	if (d_size <= 0) return errlog(nn,"bad d_size");
	if ((b_start+b_size) > b_in) return errlog(nn,"in b too small");
	if ((h_start+h_size) > h_in) return errlog(nn,"in h too small");
	if ((w_start+w_size) > w_in) return errlog(nn,"in w too small");
	if ((d_start+d_size) > d_in) return errlog(nn,"in d too small");

	if( tensor_out_prepare_normal( out_tensor, b_size,h_size,w_size,d_size, typeid ) !=  0) {
		return errlog(nn,"out too small");
	}

	const char *indata = input_tensor->data;
	// skip the 'offset'
	indata += element_size*( d_start + d_in * (w_start + w_in * (h_start + h_in*b_start)));

	const char *in;
	char *out = out_tensor->data;

	// try to simplify the layout, move things into inner loops.
	// (for now, all strides are in elements, not bytes).
	//
	struct ddesc { int32_t siz, in_stride; }
	layout[4] = {
			// dimens.   in_stride
			{   d_size,   1, },
			{   w_size,   d_in },
			{   h_size,   w_in*d_in},
			{   b_size,   h_in*w_in*d_in },
	};
	// if a dimension's input stride is the product of the previous dim's size and in stride,
	// then they can be merged into one with the lower stride, and the product of sizes.
	// The other dims are moved down and a size=1 dim added at the outside.
	// layout[0].in_stride will remain = 1.
	//
	int ndims = 4;
	for( int i = 0; i < ndims-1; ){
		if( layout[i+1].in_stride == layout[i].siz * layout[i].in_stride ){
			layout[i].siz *= layout[i+1].siz;
			for( int j = i+1; j < ndims-1; j++){
				layout[j] = layout[j+1];		// squeeze others down.
			}
			ndims --;
			layout[ndims].siz = 1;		// and stride doesn't matter
		}else{
			i++;
		}
	}
	int b_in_stride, h_in_stride, w_in_stride;

	d_size = layout[0].siz;
	w_size = layout[1].siz;
	w_in_stride = layout[1].in_stride * element_size;
	h_size = layout[2].siz;
	h_in_stride = layout[2].in_stride * element_size;
	b_size = layout[3].siz;
	b_in_stride = layout[3].in_stride * element_size;

	for (b = 0; b < b_size; b++) {
		for (h = 0; h < h_size; h++) {
			for (w = 0; w < w_size; w++) {
				in = indata + b_in_stride * b + h_in_stride * h + w_in_stride * w;
				memcpy(out,in,d_size*element_size);
				out += d_size * element_size;
			}
		}
	}
	return 0;
}

static int slice_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	return slice_impl(self,nn,sizeof(float), NN_TYPE_FLOAT);
}

static int slice_execute_8(struct nn_node *self, struct nn_graph *nn)
{
	return slice_impl(self,nn,sizeof(uint8_t), NN_TYPE_UINT8);
}

static int slice_execute_int32(struct nn_node *self, struct nn_graph *nn)
{
	return slice_impl(self,nn,sizeof(int32_t), NN_TYPE_INT32);
}

static int slice_execute_q8(struct nn_node *self, struct nn_graph *nn)
{
	tensor_copy(self->outputs[1],self->inputs[3]);
	tensor_copy(self->outputs[2],self->inputs[4]);
	return slice_impl(self,nn,sizeof(uint8_t), NN_TYPE_QUINT8);
}

static int slice_check_f(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

static int slice_check_8(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

static int slice_check_int32(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 3) return errlog(nn,"num inputs");
	if (self->n_outputs != 1) return errlog(nn,"num outputs");
	return 0;
}

static int slice_check_q8(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"checking slice node %p",self);
	if (self->n_inputs != 5) return errlog(nn,"num inputs");
	if (self->n_outputs != 3) return errlog(nn,"num outputs");
	return 0;
}


struct nn_node_ops nn_ops_for_Slice_f = {
	.execute = slice_execute_f,
	.check = slice_check_f,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Slice_8 = {
	.execute = slice_execute_8,
	.check = slice_check_8,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_Slice_int32 = {
	.execute = slice_execute_int32,
	.check = slice_check_int32,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedSlice_8 = {
	.execute = slice_execute_q8,
	.check = slice_check_q8,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

