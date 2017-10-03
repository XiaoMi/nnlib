
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
#include <string.h>

struct tdata {
	struct nn_node *self;
	int retval;
	nn_sem_t donesem;
};

static int convert_from_d32_execute(struct nn_graph *nn, void *vself)
{
	const struct nn_node *self = vself;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	const uint8_t *in;
	uint8_t *out = out_tensor->data;
	int b_in = in_tensor->shape.batches;
	int h_in = in_tensor->shape.height;
	int w_in = in_tensor->shape.width;
	int d_in = in_tensor->shape.depth;
	int in_d32_stride = tensor_d32_stride_d32( in_tensor);
	int b,h,w,d;
	if (tensor_out_prepare_normal(out_tensor,b_in,h_in,w_in,d_in,NN_TYPE_QUINT8)) {
		return errlog(nn,"can't prepare output");
	}

	int d_left = in_tensor->format.depth_pad[0];
	int d_before = in_tensor->format.depth_pad[0];
	int d_after = in_tensor->format.depth_pad[1];
	int left_pad = in_tensor->format.width_pad[0];
	int right_pad = in_tensor->format.width_pad[1];
	int width_total = left_pad + right_pad + w_in;

	if ( 	((d_in % 32) == 0)
		&& (d_before == 0)
		&& (d_after == 0)
		&& (left_pad == 4)
		&& (((w_in + right_pad) % 4) == 0)) {
		/* Fast Path */
		for (b = 0; b < b_in; b++) {
			in = tensor_location_d32(in_tensor,b,0,0,0);
			out = out_tensor->data;
			out += b*h_in*w_in*d_in;
			logmsg(nn,2,"in=%p wtot=%d out=%p w_in=%d h_in=%d d_in=%d",
				in,width_total*32,out,w_in,h_in,d_in);
			from_d32_asm(in,width_total*32,out,w_in,h_in,d_in);
		}
		return 0;
	}

	if( d_left == 0 && d_in == 32){	// full house, depthwise
		int rowlen = w_in * d_in;
		for (b = 0; b < b_in; b++) {
			for (h = 0; h < h_in; h++) {
				// point to first in row
				in = tensor_location_bhw_d32(in_tensor,b,h,0);
				vmemcpy_asm(out,in,rowlen);
				out += rowlen;
			}
		}
	}else{

		for (b = 0; b < b_in; b++) {
			for (h = 0; h < h_in; h++) {
				for (w = 0; w < w_in; w++) {
					int d_remain = d_in;
					d = d_left;
					// point to start of first depth slice
					in = tensor_location_bhw_d32(in_tensor,b,h,w);
					while(1){
						int dnow = 32-d;
						dnow = (dnow < d_remain)? dnow: d_remain;	// copy this many
						memcpy(out,in+d,dnow);
						out += dnow;
						d_remain -= dnow;
						if( d_remain <= 0)
							break;
						d = 0;
						in += in_d32_stride;
					}
				} //w
			} //h
		} //b
	}
	return 0;
}

static int get_option( struct nn_graph * nn, const struct tensor * tens_in, int default_val, char const *option_desc, int maxval)
{
	if( tens_in == NULL) return default_val;
	int newval = tensor_get_int32( tens_in,  0 );
	if( newval < 0 || newval > maxval){
		logmsg(nn,2,"convert_d32: value %d out of range (0..%d) for %s; using default of %d",
				newval, maxval, option_desc, default_val );
		return default_val;
	}else{
		return newval;
	}
}

static int convert_to_d32_execute(struct nn_graph *nn, void *vself)
{
	const struct nn_node *self = vself;
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	const uint8_t *in = in_tensor->data;
	uint8_t *out;
	int b_in = in_tensor->shape.batches;
	int h_in = in_tensor->shape.height;
	int w_in = in_tensor->shape.width;
	int d_in = in_tensor->shape.depth;


	// process optional padding
	int d_pad_before = 0;		// defaults
	int w_pad_left = 4;
	int w_pad_right_min = 0;
	int h_pad_top = 4;

	if( self->n_inputs >=2){
		d_pad_before = get_option( nn, self->inputs[1], d_pad_before, "depth padding", MAX_PADDING_DEPTH );
		if( self->n_inputs >=3 )
			w_pad_left = get_option( nn, self->inputs[2], w_pad_left, "width padding(left)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=4 )
			w_pad_right_min = get_option( nn, self->inputs[3], w_pad_right_min, "width padding (min right)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=5 )
			h_pad_top = get_option( nn, self->inputs[4], h_pad_top, "height padding", MAX_PADDING_HEIGHT );
	}
	int wtotal = (w_pad_left + w_in + w_pad_right_min + 3)&~3;
	int w_pad_right = wtotal - (w_pad_left + w_in);
	if( w_pad_right > MAX_PADDING_WIDTH) w_pad_right -= 4;

	int d_pad_after = (-(d_in+d_pad_before))&31;
	int h_pad_bottom = h_pad_top;

	int b,h,w,d;
	logmsg(nn,2,"h: %d|%d|%d w: %d|%d|%d d: %d|%d|%d avail=%d",
			h_pad_top,h_in,h_pad_bottom,
			w_pad_left,w_in,w_pad_right,
			d_pad_before,d_in,d_pad_after,
			out_tensor->max_size);
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		b_in,
		h_in,h_pad_top,h_pad_bottom,
		w_in,w_pad_left,w_pad_right,
		d_in,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8) != 0) {
		logmsg(nn,2,"h: %d|%d|%d w: %d|%d|%d d: %d|%d|%d avail=%d",
			h_pad_top,h_in,h_pad_bottom,
			w_pad_left,w_in,w_pad_right,
			d_pad_before,d_in,d_pad_after,
			out_tensor->max_size);
		return errlog(nn,"out prepare fail (tensor %p)", out_tensor);
	}
	if ((w_pad_left == 4) && ((d_in % 32) == 0) && (((w_in+w_pad_right)%32)==0)) {
		for (b = 0; b < b_in; b++) {
			in = in_tensor->data;
			in += b*h_in*w_in*d_in;
			out = tensor_location_d32(out_tensor,b,0,0,0);
			logmsg(nn,2,"in=%p w_in=%d out=%p w_total=%d h_in=%d d_in=%d",
				in,w_in,out,wtotal*32,h_in,d_in);
			to_d32_asm(in,w_in,out,32*wtotal,h_in,d_in);
		}
		return 0;
	} else for (b = 0; b < b_in; b++) {
		for (h = 0; h < h_in; h++) {
			for (w = 0; w < w_in; w++) {
				for (d = 0; d < d_in; /* see below */) {
					int d_left = d_in-d;
					int d_valid = (d_left < 32) ? d_left : 32;
					out = tensor_location_d32(out_tensor,b,h,w,d);
					memcpy(out,in,d_valid);
					in += d_valid;
					d += d_valid;
				}
			}
		}
	}
	return 0;
}

static int convert_from_d32_spawn(struct nn_node *self, struct nn_graph *nn)
{
	return nn_os_vector_call(nn,convert_from_d32_execute,self);
}

static int convert_to_d32_spawn(struct nn_node *self, struct nn_graph *nn)
{
	return nn_os_vector_call(nn,convert_to_d32_execute,self);
}
// input 0: 'flat 'u8' tensor
//  input 1:  (optional) scalar int: depth padding start - default 0  (0..31)
//  input 2:  (optional) scalar int: width padding start - default 4  (0..MAX_PADDING_WIDTH)
//  input 3:  (optional) scalar int: width padding end (min) - default 0  (0..MAX_PADDING_WIDTH)
//    The 'end' padding will be adjusted up so that the width total is a multiple of 4. If it exceeds
//    MAX_PADDING_WIDTH as a result, it will then be adjusted down by 4.
//  input 4:  (optional): scalar int: top/bottom padding for height  default 4 (0..MAX_PADDING_HEIGHT)
//
// output : d32 u8 tensor

static int convert_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	int k = node_check_inputs_range( self, nn, "convert_d32", 1, -5 );
	if(k==0) k = node_check_outputs_n( self, nn, "convert_d32", 1);
	return k;
}




struct nn_node_ops nn_ops_for_Convert_from_d32 = {
	.execute = convert_from_d32_spawn,
	.check = convert_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT,
};

struct nn_node_ops nn_ops_for_Convert_to_d32 = {
	.execute = convert_to_d32_spawn,
	.check = convert_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_OUTPUT,
};

