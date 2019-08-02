/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
#include <math.h>
#include "quantize.h"
/*
 * This operator takes a d32 input and copies it to the output,
 * with exactly the same padding; however, you can specify a value to put in the 'padding' areas.
 * This is intended to assist in unit tests, and is not optimized for speed.
 *
 *
 * Input 0:   input d32 tensor
 * Input 1:   (optional) fill value for top/bottom/left/right pad. default is 0xFF
 * Input 2    (optional) fill value for depth padding.  default is same as spatial padding.
 *
 * Notes:
 *  (1) if either of the values is -1, a pseudo-random fill is used .
 *  (2) The spatial fill policy is continued beyond the end of the allocated output area, to at most 4 rows; according
 *      to the space actually available.
 */

typedef void (*general_fill_fp)(  uint8_t *out, int len, int fillval,  uint64_t *state);
static void general_fill_op_fixed( uint8_t *out, int len, int fillval,  uint64_t *state);
static void general_fill_op_random( uint8_t *out, int len, int fillval,  uint64_t *state);


static int padfill_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *spatfill_tensor = self->inputs[1];
	const struct tensor *depthfill_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];

	int spatial_fill_val = 255;
	int depth_fill_val = 255;


	logmsg(nn,2,"padfill_d32 execute. self=%p ",self);

	if( self->n_inputs >= 2){
		spatial_fill_val = depth_fill_val = tensor_get_int32( spatfill_tensor ,0);
		if( self->n_inputs >= 3){
			depth_fill_val = tensor_get_int32( depthfill_tensor ,0);
		}
	}
	uint64_t random_state = 0x827Ea91D91924124ull;
	// allocate the output
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;

	int pad_top = in_tensor->format.height_pad[0];
	int pad_bottom = in_tensor->format.height_pad[1];
	int pad_left = in_tensor->format.width_pad[0];
	int pad_right = in_tensor->format.width_pad[1];
	int pad_d0 = in_tensor->format.depth_pad[0];
	int pad_d1 = in_tensor->format.depth_pad[1];

	int res = tensor_out_prepare_padded_d32( out_tensor,
			batches,
			height, pad_top, pad_bottom,
			width, pad_left, pad_right,
			depth, pad_d0, pad_d1, NN_TYPE_QUINT8);
	if( res != 0){
		return errlog(nn,"output too small");
	}

	memcpy( out_tensor->data, in_tensor->data, out_tensor->data_size);

	general_fill_fp spatial_fill = (spatial_fill_val == -1)? general_fill_op_random: general_fill_op_fixed;
	general_fill_fp depth_fill = (depth_fill_val == -1)? general_fill_op_random: general_fill_op_fixed;

	struct tensor_addressing tout = tensor_addressing_d32( out_tensor );


	// pad all the rows above and below.
	int rowbytes = tout.height_stride;		// size of a full row
	if( pad_top > 0 || pad_bottom > 0 ){
		int top_pad_bytes = rowbytes * pad_top;
		int bot_pad_bytes = rowbytes * pad_bottom;
		int bot_offset = rowbytes * (pad_top + height);

		for( int b = 0; b < batches; b++){
			uint8_t * bbase = (uint8_t *)out_tensor->data + b * tout.batch_stride;
			if( top_pad_bytes > 0)
				(*spatial_fill)( bbase, top_pad_bytes, spatial_fill_val, & random_state);
			if( bot_pad_bytes > 0)
				(*spatial_fill)( bbase+bot_offset, bot_pad_bytes, spatial_fill_val, & random_state);
		}
	}
	// pad any past the last?
	int extra_bytes = out_tensor->max_size - out_tensor->data_size;
	if( extra_bytes > 4*rowbytes) extra_bytes = 4*rowbytes;
	if( extra_bytes > 0 ){
		uint8_t *p = (uint8_t *)out_tensor->data + out_tensor->data_size;
		(*spatial_fill)( p, extra_bytes, spatial_fill_val, & random_state);
	}

	// left and right width padding.
	// this only needs to be done in the 'height*nd32' rows
	// (using d32 stride)
	if( pad_left > 0 || pad_right > 0){
		int extht = height * tout.nd32;	// # of places to do it...
		int pad_left_bytes = 32 * pad_left;
		int pad_right_bytes = 32* pad_right;
		int d32_str = tout.d32_stride;

		for( int b = 0; b < batches; b++){
			uint8_t * bbase = (uint8_t *)out_tensor->data + b * tout.batch_stride + pad_top * rowbytes;
			if( pad_left_bytes > 0){
				for( int ih=0; ih < extht; ih++){
					(*spatial_fill)(bbase + ih*d32_str, pad_left_bytes , spatial_fill_val, & random_state);
				}
			}
			if( pad_right_bytes > 0){
				bbase += pad_left_bytes + 32*width;
				for( int ih=0; ih < extht; ih++){
					(*spatial_fill)(bbase + ih*d32_str, pad_right_bytes , spatial_fill_val, &random_state);
				}
			}
		}
	}
	// depth fill. first left side, if any
	if( pad_d0 > 0){
		for( int b = 0; b < batches; b++){
			uint8_t * bbase = (uint8_t *)out_tensor->data + b * tout.batch_stride + pad_top * rowbytes + 32*pad_left;
			for(int ih =0; ih < height; ih++){
				for( int iw = 0; iw < width ; iw++){
					(*depth_fill)( bbase + ih*rowbytes + iw*32, pad_d0, depth_fill_val, &random_state);
				}
			}
		}
	}
	// right side, if any.
	if( pad_d1 > 0){
		// offset to depth-end-padding:
		int doffset = (tout.nd32-1)*tout.d32_stride + (32-pad_d1);
		for( int b = 0; b < batches; b++){
			uint8_t * bbase = (uint8_t *)out_tensor->data + b * tout.batch_stride + pad_top * rowbytes + 32*pad_left + doffset;
			for(int ih =0; ih < height; ih++){
				for( int iw = 0; iw < width ; iw++){
					(*depth_fill)( bbase + ih*rowbytes + iw*32, pad_d1, depth_fill_val, &random_state);
				}
			}
		}
	}
	logmsg(nn,2,"padfill_d32 %p done",self);
	return 0;
}

static void
general_fill_op_fixed( uint8_t *out, int len, int fillval,  uint64_t *state)
{
	memset( out, fillval, len);
}

static void
general_fill_op_random( uint8_t *out, int len, int fillval,  uint64_t *state)
{
	uint64_t rstt = *state;

	if( ((unsigned)(size_t)out & 3) == 0 && len >= 4 ){
		int n32 = len >> 2;
		uint32_t *o32 = (uint32_t*) out;
		for( int i = 0; i < n32; i++){
			 o32[i] = rstt >>32;
			 rstt = rstt* 6364136223846793005ull + 1442695040888963407ull;
		}
		out += n32*4;
		len &= 3;
	}
	for( int i  =0; i < len; i++){
		 out[i] = (uint8_t)( rstt >> 32);
		 rstt = rstt* 6364136223846793005ull + 1442695040888963407ull;
	}
	*state = rstt;
}



struct nn_node_ops nn_ops_for_FillPadding_8_d32 = {
	.execute = padfill_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,3),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};

