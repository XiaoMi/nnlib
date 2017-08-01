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
#include <quantize.h>
#include <math.h>

static inline void do_mirrorpad(
	void *outpv,
	const void *inpv,
	const int32_t w_in,
	const int32_t h_in,
	const int32_t pre_w,
	const int32_t post_w,
	const int32_t pre_h,
	const int32_t post_h,
	const int32_t depth_byte_size,
	const int32_t reflect)
{
	const char *inp = (const char *)inpv;
	char *outp = (char *)outpv;
	const int32_t in_w_size = depth_byte_size * w_in;
	const int32_t out_w_size = depth_byte_size * (w_in + pre_w + post_w);
	//const int32_t pre_w_skip = pre_w * depth_byte_size;
	//const int32_t post_w_skip = post_w * depth_byte_size;
	//const int32_t pre_h_skip = pre_h * out_w_size;
	//const int32_t post_h_skip = post_h * out_w_size;
	int w;
	int h;
	const char *in;
	char *out;
	const char *pad_in;
	const int w_mirror_start_offset = reflect * depth_byte_size;
	const int h_mirror_start_offset = reflect * out_w_size;
	in = inp;
	out = outp + pre_h*out_w_size;
	for (h = 0; h < h_in; h++) {
		for (w = pre_w-1; w >= 0; w--) {
			pad_in = in + w_mirror_start_offset + w*depth_byte_size;
			memcpy(out,pad_in,depth_byte_size);
			out += depth_byte_size;
		}
		memcpy(out,in,in_w_size);
		in += in_w_size;
		out += in_w_size;
		for (w = 0; w < post_w; w++) {
			pad_in = in - (1+w)*depth_byte_size - w_mirror_start_offset;
			memcpy(out,pad_in,depth_byte_size);
			out += depth_byte_size;
		}
	}
	out = outp;
	in = outp + pre_h * out_w_size;
	for (h = pre_h-1; h >= 0; h--) {
		pad_in = in + h_mirror_start_offset + h*out_w_size;
		memcpy(out,pad_in,out_w_size);
		out += out_w_size;
	}
	out = outp + out_w_size * (pre_h + h_in);
	in = out; 
	for (h = 0; h < post_h; h++) {
		pad_in = in - (1+h)*out_w_size - h_mirror_start_offset;
		memcpy(out,pad_in,out_w_size);
		out += out_w_size;
	}
}


static int mirrorpad_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *pads_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	const int32_t *pads = (const int32_t *)pads_tensor->data;
	const int32_t pad_d_before = pads[0+0];
	const int32_t pad_d_after = pads[0+1];
	const int32_t pad_w_before = pads[2+0];
	const int32_t pad_w_after = pads[2+1];
	const int32_t pad_h_before = pads[4+0];
	const int32_t pad_h_after = pads[4+1];
	const int32_t pad_b_before = pads[6+0];
	const int32_t pad_b_after = pads[6+1];
	const int32_t d_in = in_tensor->shape.depth;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t d_out = d_in + pad_d_before + pad_d_after;
	const int32_t w_out = w_in + pad_w_before + pad_w_after;
	const int32_t h_out = h_in + pad_h_before + pad_h_after;
	const int32_t b_out = b_in + pad_b_before + pad_b_after;
	const uint32_t elements_out = d_out * w_out * h_out * b_out;
	const uint32_t element_size = sizeof(float);
	const uint32_t bytes_out = elements_out * element_size;
	const float *in_base = (const float *)in_tensor->data;
	float *out_base = (float *)out_tensor->data;
	const float *inp;
	float *outp;
	int b;
	logmsg(nn,2,"in tensor: %dx%dx%dx%d",b_in,h_in,w_in,d_in);
	if (pads_tensor->shape.depth != 2) return errlog(nn,"bad pad shape");
	if (pads_tensor->shape.width != 4) return errlog(nn,"bad pad shape");
	if (pad_b_before || pad_b_after) return errlog(nn,"can't pad batches");
	if (pad_d_before || pad_d_after) return errlog(nn,"can't pad depth");
	if (pad_w_before >= w_in) return errlog(nn,"width too small (%d>=%d)",pad_w_before,w_in);
	if (pad_w_after >= w_in) return errlog(nn,"width too small (%d>=%d)",pad_w_after,w_in);
	if (pad_h_before >= h_in) return errlog(nn,"height too small");
	if (pad_h_after >= h_in) return errlog(nn,"height too small");
	if (bytes_out > out_tensor->max_size) return errlog(nn,"out too small");
	tensor_set_shape(out_tensor,b_out,h_out,w_out,d_out);
	out_tensor->data_size = bytes_out;

	for (b = 0; b < b_in; b++) {
		inp = in_base + b*h_in*w_in*d_in;
		outp = out_base + b*h_out*w_out*d_out;
		do_mirrorpad(
			outp,
			inp,
			w_in,
			h_in,
			pad_w_before,
			pad_w_after,
			pad_h_before,
			pad_h_after,
			d_in*element_size,
			(self->padding == NN_PAD_MIRROR_REFLECT));

	}
	return 0;
}

static int mirrorpad_f_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"mirrorpad node %p",self);
	if ((self->padding != NN_PAD_MIRROR_REFLECT) 
		&& (self->padding != NN_PAD_MIRROR_SYMMETRIC)) {
			return errlog(nn,"bad mirror pad type");
	}
	if (self->n_inputs != 2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"mirrorpad %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_MirrorPad_f = {
	SFINIT(.execute, mirrorpad_f_execute),
	SFINIT(  .check, mirrorpad_f_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

