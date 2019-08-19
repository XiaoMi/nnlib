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
#include <stdio.h>

//
// QuantizedPad_8_d32 node
// Currently this is constrained to support only B,H,W padding (not d).
//
//
//
//   4 inputs:
//        0 : input tensor, qu8
//        1 : scalar float (input min)
//        2 : scalar float (input max)
//        3 : 'padding' tensor, int32 [1,1,n,2] where n = 1..4  (see input #1 of Pad_f)
//       4 : (optional) scalar float, value to pad (default 0.0).
//    3 output:
//        0: output tensor, qu8
//        1 : scalar float (output min)
//        2 : scalar float (output max)

struct pad_d32_runstate {
	struct shape inshape;
	struct shape outshape;
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	int pad_byte;
	int b_add_before;			// batches to add before
	unsigned obatch_core_len; 	// amount to fill each output batch which is padding.

	// H,W padding and the core copy are described 'per batch';

	int h_add_before_offset;	// offset to skip top padding in output
	int h_add_before_fill_len;	// 0, or amount to fill in top padding
	int h_add_after_offset;		// additional offset to reach bottom padding
	int h_add_after_fill_len;	// 0, or amount to fill in bottom padding.

	int core_copy_width;		// width of the 'core copy' in bytes
	int core_copy_height;	// height in rows (really, total # nd32 rows)
	int core_copy_dstoff;	// offset in bytes to leave room for padding on left

	// w_before uses core_copy_height, and is done to the output start address
	int w_add_before_width;	// 32 * amount of left-padding added
	// w_after uses core_copy_height, and is done to the output start address + core_copy_dstoff + core_copy_width
	int w_add_after_width;	// 32 * amount of right-padding added

	nn_sem_t done_sem;
};
static void hvx_pad_d32_workfunc( struct nn_graph * nn, void * rstpv);

static int pad_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const  *in_tensor = self->inputs[0];
	struct tensor const *in_min_tensor = self->inputs[1];
	struct tensor const *in_max_tensor = self->inputs[2];
	struct tensor const * pads_tensor = self->inputs[3];
	if( pads_tensor->shape.depth != 2) return errlog(nn,"bad pad tensor");


	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	
	float pad_val_f = 0.0f;
	if (self->n_inputs > 4) {
		pad_val_f = tensor_get_float(self->inputs[4],0);
		if (self->n_inputs > 5 && tensor_get_int32(self->inputs[5],0)) {
			if (pad_val_f < in_min || pad_val_f > in_max) {
				return errlog(nn, "Pad value (%f) is outside the range of inputs (%f,%f)", pad_val_f, in_min, in_max);
			}
		}
	}
	int pad_byte;		// byte for padding
	if( fabsf(pad_val_f) != INFINITY ){
		int padv = roundf_i32(255.0f * (pad_val_f - in_min) /(in_max-in_min));
		pad_byte = saturate_u8( padv );
	}else{
		// treat -/+ inf as 0 or 255
		pad_byte = (pad_val_f < 0)? 0 : 255;
	}


	const int32_t *pads = pads_tensor->data;

	// extract the pads, based on w dimension; ensure all are >=0
	//
	unsigned padt_len = pads_tensor->shape.width;
	if( padt_len> 4) padt_len = 4;		// ignore > 4
	unsigned padby[4*2] = {0,0, 0,0, 0,0, 0,0};
	for( int i = 0; i < (int)padt_len*2; i++ ){
		int p = pads[i];
		if( p < 0) return errlog(nn,"pad bad tensor");
		padby[i] = p;
	}
	// find the new shape; validate sanity
	struct shape out_shape;
	uint32_t new_shape_count = 1;
	for( int i =0; i < 4; i++){
		unsigned p_before = padby[2*i];
		unsigned p_after = padby[2*i+1];
		unsigned old_dim = in_tensor->shape.dimension[i];
		uint64_t all_dim = (uint64_t)old_dim + (uint64_t)p_before + (uint64_t)p_after;
		if( all_dim > (uint64_t)0x7FFFFFFF) return errlog(nn,"padded size overflow");
		uint32_t new_dim = (uint32_t)all_dim;
		out_shape.dimension[i] = new_dim;
		new_shape_count = mulu32_sat( new_shape_count, new_dim);
	}
	if (new_shape_count ==0 || new_shape_count == (uint32_t)-1)
		return errlog(nn,"padded size overflow");
	if( padby[6]>0 || padby[7]> 0 )
		return errlog( nn,"does not currently support depth padding");


	struct pad_d32_runstate runstate;
	runstate.inshape = in_tensor->shape;
	runstate.outshape = out_shape;

	int w_in_before_pad = in_tensor->format.width_pad[0];
	int w_out_before_pad = w_in_before_pad;
	int w_out_after_pad = (-(out_shape.width + w_out_before_pad))&3;
	int d_out_after_pad = (-out_shape.depth)& 31;

	int res = tensor_out_prepare_padded_d32( out_tensor,
			out_shape.batches,
			out_shape.height, in_tensor->format.height_pad[0], in_tensor->format.height_pad[1],
			out_shape.width, w_out_before_pad, w_out_after_pad,
			out_shape.depth, 0, d_out_after_pad, NN_TYPE_QUINT8 );
	if( res != 0){
		return errlog(nn,"output too small");
	}
	runstate.tin = tensor_addressing_d32( in_tensor );
	runstate.tout = tensor_addressing_d32( out_tensor);
	runstate.b_add_before = padby[0];
	runstate.pad_byte = pad_byte;
	// figure out how many bytes we need to zero in a batch, starting from the first 'real' byte
	// (thus skipping all the top padding, and the left padding on first d32 slice, and right on last d32 slice).
	{
		int out_nd32 = runstate.tout.nd32;
		int out_d32_stride = runstate.tout.d32_stride;
		int od32_rows = runstate.outshape.height * out_nd32;
		int core_width = runstate.outshape.width * 32;
		runstate.obatch_core_len =  core_width + (od32_rows-1)*out_d32_stride;

		int h_add_before = padby[2];
		int od32_hbefore = out_nd32 * h_add_before;
		runstate.h_add_before_offset = out_d32_stride * od32_hbefore;
		runstate.h_add_before_fill_len = 0;
		if( h_add_before > 0){
			runstate.h_add_before_fill_len = core_width + (od32_hbefore -1)*out_d32_stride;
		}
		int h_add_after = padby[3];
		int od32_hafter = out_nd32 * h_add_after;
		// h_add_after_offset is where the bottom padding rows are, relative to adding h_add_before_offset
		runstate.h_add_after_offset = out_d32_stride * out_nd32 * runstate.inshape.height;
		runstate.h_add_after_fill_len = 0;
		if( h_add_after > 0){
			runstate.h_add_after_fill_len = core_width + (od32_hafter -1)*out_d32_stride;
		}

		runstate.core_copy_width = runstate.inshape.width*32;
		runstate.core_copy_height = out_nd32 * runstate.inshape.height;
		runstate.core_copy_dstoff = padby[4]*32;		// offset into dest

		runstate.w_add_before_width = padby[4]*32;
		runstate.w_add_after_width = padby[5]*32;
	}

	nn_sem_init(&runstate.done_sem, 0);

	nn_os_work_for_vector(nn, hvx_pad_d32_workfunc, &runstate);

	tensor_set_single_float(out_min_tensor, in_min);
	tensor_set_single_float(out_max_tensor, in_max);
	nn_sem_wait( &runstate.done_sem);

	return 0;

}
//
// Work function
// note that we don't fill the 'padding' zones in the output, only the padding.
// Exception is that the left & right
static void
hvx_pad_d32_workfunc( struct nn_graph * nn, void * rstpv)
{
	struct pad_d32_runstate * rstp = (struct pad_d32_runstate *)rstpv;
	int pad_byte = rstp->pad_byte;

	int b_add_before = rstp->b_add_before;
	int obatches = rstp->outshape.batches;
	int ibatches = rstp->inshape.batches;

	uint8_t *obat_ptr0 = rstp->tout.data;
	uint8_t const* ibat_ptr0 = rstp->tin.data ;

	int core_copy_width = rstp->core_copy_width;
	int core_copy_height = rstp->core_copy_height;
	int out_d32_stride = rstp->tout.d32_stride;


	for( int obat = 0; obat < obatches; obat++ ){
		uint8_t *obat_ptr = obat_ptr0 + rstp->tout.batch_stride * obat;
		int ibat = obat-b_add_before;
		if( (unsigned)ibat >= (unsigned)ibatches){	// is a padding batch
			vmemset_asm( obat_ptr,pad_byte, rstp->obatch_core_len);
		}else{
			int h_before_len = rstp->h_add_before_fill_len;
			if( h_before_len > 0){
				vmemset_asm( obat_ptr, pad_byte, h_before_len );
				obat_ptr += rstp->h_add_before_offset;
			}
			int h_after_len = rstp->h_add_after_fill_len;
			if( h_after_len > 0 ){
				vmemset_asm( obat_ptr + rstp->h_add_after_offset, pad_byte, h_after_len );
			}

			// core 2d copy
			uint8_t const * ibat_ptr = ibat_ptr0 + rstp->tin.batch_stride * ibat;
			uint8_t *obat_copy_at = obat_ptr + rstp->core_copy_dstoff;	// dest rectangle

			//printf("core copy: [%d x %d] @ %p stride = %d\n", core_copy_height, core_copy_width, obat_copy_at, out_d32_stride);
			vmemcpy_2d_asm(
					core_copy_width,	// width of copy
					core_copy_height,	// height
					obat_copy_at,			// dest rectangle
					out_d32_stride,			// stride
					ibat_ptr,				// source rectangle
					rstp->tin.d32_stride);	// stride

			int w_add_bef_wid = rstp->w_add_before_width;
			if( w_add_bef_wid > 0){
				vmemset_2d_asm( obat_ptr, pad_byte, w_add_bef_wid, core_copy_height, out_d32_stride);
			}
			int w_add_aft_wid = rstp->w_add_after_width;
			if( w_add_aft_wid > 0){
				//printf("wafter copy: [%d x %d] @ %p stride = %d\n", core_copy_height,w_add_aft_wid, obat_copy_at + core_copy_width, out_d32_stride);
				vmemset_2d_asm( obat_copy_at + core_copy_width, pad_byte, w_add_aft_wid, core_copy_height,out_d32_stride );
			}
		}
	}
	nn_sem_post ( &rstp->done_sem);
};



struct nn_node_ops nn_ops_for_QuantizedPad_8_d32 = {
	.execute = pad_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(4,6),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

