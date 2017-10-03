
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains implementations for quantized max pooling node
 */


#include <nn_graph.h>
#include <string.h>

/*
 * A note about strategy
 * In Depth32, we should have some sufficient padding around H and W dimensions
 * 
 * So we should be able to zap borders and compute pretty liberally 
 * In particular, if we don't care about output values, we don't even care about X and Y position 
 * relative to the border, since we will just compute padding in the same format.
 */

struct tdata {
	struct nn_node *self;
	const uint8_t *indata;
	uint8_t *outdata;
	int32_t in_next_row;
	int32_t in_next_d32;
	int32_t in_height;
	union {
		struct {
			int32_t out_next_row;
			int32_t out_vectors_wide;
			int32_t out_lines;
			int32_t out_lalign;
			int32_t which_slice;
		};
		struct {
			int32_t zap_left_w;
			int32_t zap_w_skip;
			int32_t zap_top;
			int32_t zap_bot;
			int32_t zap_d32_iters;
		};
	};
	nn_sem_t donesem;
	int (*f)(uint8_t *, const uint8_t *, int32_t, int32_t, int32_t, int32_t, int32_t);
};

static void maxpool_execute_slice(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	uint8_t *outdata = td->outdata;
	const uint8_t *indata = td->indata;
	int in_next_row = td->in_next_row;
	int out_next_row = td->out_next_row;
	int out_vectors_wide = td->out_vectors_wide;
	int out_lines = td->out_lines;
	int out_lalign = td->out_lalign;
	l2fetch(indata,in_next_row,td->in_next_d32,td->in_height);
	td->f(outdata,indata,in_next_row,out_next_row,out_vectors_wide,out_lines,out_lalign);
	nn_sem_post(&td->donesem);
}

static void maxpool_zap(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	uint8_t *indata = (uint8_t *)td->indata;
	int32_t in_next_row = td->in_next_row;
	int32_t in_next_d32 = td->in_next_d32;
	int32_t in_height = td->in_height;
	int32_t zap_top = td->zap_top;
	int32_t zap_bot = td->zap_bot;
	int32_t zap_left_w = td->zap_left_w;
	int32_t zap_w_skip = td->zap_w_skip;
	int32_t zap_r_woff = zap_w_skip+zap_left_w;
	int32_t zap_r_amt = (-zap_r_woff) & 3;
	int32_t zap_d32_iters = td->zap_d32_iters;
	vmemset_asm(indata,0,in_next_row*zap_top);
	vmemset_asm(indata+in_next_row*(in_height+zap_top),0,in_next_row*zap_bot);
	padzap_part(indata+in_next_row*zap_top,0,in_next_d32,zap_d32_iters,in_next_row,in_height+2,zap_left_w);
	if (zap_r_amt) {
		padzap_part(indata+in_next_row*zap_top+32*(zap_w_skip+zap_left_w),0,in_next_d32,zap_d32_iters,in_next_row,in_height+1,zap_r_amt);
	}
	nn_sem_post(&td->donesem);
}

static int maxpool_d32_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *window_tensor = self->inputs[3];
	const struct tensor *stride_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t window_width = window_tensor->shape.width;
	int32_t window_height = window_tensor->shape.height;

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	int32_t out_batches = in_batches;
	int32_t required_w_before, required_h_before;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,window_width,stride_width,self->padding, &required_w_before);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,window_height,stride_height,self->padding, &required_h_before);
	int32_t out_depth = in_depth;

	int32_t out_left_pad = 4;
	int32_t out_right_pad = (-out_width) & 3;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = 4;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;

	const uint8_t *inptr;
	uint8_t *outptr;

	int32_t b,h,w,d,out_h,out_w,win_h,win_w;

	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail");
	}
	tensor_set_float(out_min,0,tensor_get_float(in_min_tensor,0));
	tensor_set_float(out_max,0,tensor_get_float(in_max_tensor,0));

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	for (b = 0; b < out_batches; b++) {
	for (out_h = 0; out_h < out_height; out_h++) {
	for (out_w = 0; out_w < out_width; out_w++) {
	for (d = 0; d < out_depth; d++) {
		int32_t max = 0;
		outptr = tensor_location_d32(out_tensor,b,out_h,out_w,d);
		for (win_h = 0; win_h < window_height; win_h++) {
		for (win_w = 0; win_w < window_width; win_w++) {
			h = out_h * stride_height - required_h_before + win_h;
			w = out_w * stride_width - required_w_before + win_w;
			if (h < 0) continue;
			if (w < 0) continue;
			if (h >= in_height) continue;
			if (w >= in_width) continue;
			inptr = tensor_location_d32(in_tensor,b,h,w,d);
			if (max < *inptr) max = *inptr;
		}}
		*outptr = max;
	}}}}
	return 0;
}

static int maxpool_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *window_tensor = self->inputs[3];
	const struct tensor *stride_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t in_left_pad = in_tensor->format.width_pad[0];
	int32_t in_right_pad = in_tensor->format.width_pad[1];
	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];
	int32_t in_top_pad = in_tensor->format.height_pad[0];
	int32_t in_bottom_pad = in_tensor->format.height_pad[1];

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	int32_t window_width = window_tensor->shape.width;
	int32_t window_height = window_tensor->shape.height;

	int32_t out_batches = in_batches;

	int32_t required_w_before;
	int32_t required_h_before, required_h_after;

	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,window_width,stride_width,self->padding, & required_w_before);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height,window_height,stride_height,self->padding,
			&required_h_before, &required_h_after);
	int32_t out_depth = in_depth;

	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	//int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;

	int32_t out_left_pad = 4;
	int32_t out_right_pad = (-out_width) & 3;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = 4;
	int32_t out_depth_before_pad = in_depth_before_pad;
	int32_t out_depth_after_pad = in_depth_after_pad;

	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	//int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;

	int32_t in_next_row = in_width_total*in_depth_total;
	int32_t in_next_d32 = in_width_total*32;

	int32_t out_next_row = out_width_total*out_depth_total;
	//int32_t out_next_d32 = out_width_total*32;

	int32_t out_vectors_wide = out_width_total*32/128;

	//uint8_t *in_data = in_tensor->data;
	//const uint8_t *in_data_start = in_data + (in_next_row*(in_top_pad-required_h_before));
	//uint8_t *out_data = out_tensor->data;
	//uint8_t *out_data_start = out_data + out_next_row*out_top_pad;
	int in_d32_slices_per_batch = in_depth_total/32;
	int in_d32_slices_total = in_batches*in_d32_slices_per_batch;
	int (*f)(uint8_t *, const uint8_t *, int32_t, int32_t, int32_t, int32_t, int32_t);
	struct tdata tds[in_d32_slices_total];
	int i;
	int d;
	int b;
	int in_w_offset = -in_left_pad;
	int out_w_offset = -out_left_pad;


	if( required_h_before > in_top_pad || required_h_after > in_bottom_pad){
		return errlog(nn,"can't zap top/bottom padding");
	}
	/*
	 * For VALID padding we want to start after the IN_LEFT_OFFSET
	 * For stride == 1, out_lalign == 32 for SAME.
	 * For stride == 2, out_lalign == 64 for SAME.
	 */

	//memset(out_data,0,out_width_total*out_height_total*out_depth_total);

	struct tdata zap_td = {
		.self = self,
		.in_next_row = in_next_row,
		.in_next_d32 = in_next_d32,
		.in_height = in_height,
		.zap_top = required_h_before,
		.zap_bot = required_h_after,
		.zap_left_w = in_left_pad,
		.zap_w_skip = in_width,
		.zap_d32_iters = in_depth_total/32,
	};
	nn_sem_init(&zap_td.donesem,0);
	for (b = 0; b < in_batches; b++) {
		zap_td.indata = tensor_location_d32(in_tensor,b,-required_h_before,-in_left_pad,0);
		nn_os_work_for_vector(nn,maxpool_zap,&zap_td);
		nn_sem_wait(&zap_td.donesem);
	}

#if 0
	int h,d;
	/* ZAP INPUT BORDERS */
	memset(in_data+in_next_row*(in_top_pad-required_h_before),0,in_next_row*required_h_before);
	for (h = 0; h < in_height; h++) {
		uint8_t *rowstart = in_data + in_next_row*(h+in_top_pad);
		for (d = 0; d < in_depth_total/32; d++) {
			uint8_t *d32start = rowstart + in_next_d32 * d;
			memset(d32start,0,in_left_pad*32);
			if (in_right_pad) memset(d32start+(in_left_pad+in_width)*32,0,in_right_pad*32);
		}
	}
	memset(in_data+in_next_row*(in_top_pad+in_height),0,in_next_row*required_h_after+128);
#endif

	if ((window_tensor->shape.batches != 1)
		|| (window_tensor->shape.depth != 1)
		|| (window_height != window_width)
		|| (stride_height != stride_width)
		|| (stride_tensor->shape.depth != 1)
		|| (stride_tensor->shape.batches != 1)) {
		return maxpool_d32_ref(self,nn);
	}
	if ((window_tensor->shape.width == 3) 
		&& (stride_width == 1)) f = maxpool_slice_hvx_3x3_stride1;
	else if ((window_tensor->shape.width == 3) 
		&& (stride_width == 2)) f = maxpool_slice_hvx_3x3_stride2;
	else if ((window_tensor->shape.width == 2)
		&& (stride_width == 2)) f = maxpool_slice_hvx_2x2_stride2;
	else return maxpool_d32_ref(self,nn);

	//if (self->padding == NN_PAD_VALID) 
	if (required_w_before == 0) {
		out_vectors_wide -= out_left_pad/4;
		in_w_offset = 0;
		out_w_offset = 0;
	} else if (stride_width == 2) {
		/* EJP: this is a bit clunky. Works for 3x3 but kind of in a funny way. */
		required_w_before += 2;
	}

	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail");
	}
	tensor_set_float(out_min,0,tensor_get_float(in_min_tensor,0));
	tensor_set_float(out_max,0,tensor_get_float(in_max_tensor,0));

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	for (b = 0; b < in_batches; b++) {
		for (i = 0; i < in_d32_slices_per_batch; i++) {
			int idx = b*in_d32_slices_per_batch+i;
			struct tdata this_td = {
				.self = self,
				.indata = tensor_location_d32(in_tensor,b,-required_h_before,in_w_offset,i*32),
				.outdata = tensor_location_d32(out_tensor,b,0,out_w_offset,i*32),
				.in_next_row = in_next_row,
				.in_next_d32 = in_next_d32,
				.in_height = in_height+required_h_before+required_h_after,
				.out_vectors_wide = out_vectors_wide,
				.out_next_row = out_next_row,
				.out_lines = out_height,
				.out_lalign = required_w_before*32,
				.which_slice = idx,
				.f = f,
			};
			tds[idx] = this_td;
			nn_sem_init(&tds[idx].donesem,0);
		}
	}


#if 0
	logmsg(nn,0,"in_data=%p in_data_start=%p in_next_row=0x%x out_vectors_wide=0x%x out_next_row=0x%x out_lines=0x%x in_height=0x%x out_lalign=%d f=%p out=%p out_data_start=%p",
		in_data,in_data_start,in_next_row,td.out_vectors_wide,out_next_row,out_height,in_height,td.out_lalign,td.f,out_data,out_data_start);
#endif
	for (d = 0; d < in_d32_slices_total; d++) {
		nn_os_work_for_vector(nn,maxpool_execute_slice,&tds[d]);
	}
	for (d = 0; d < in_d32_slices_total; d++) {
		nn_sem_wait(&tds[d].donesem);
	}
	return 0;
}

static int maxpool_check(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 5) return errlog(nn,"maxpool wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"maxpool wrong # outs");
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedMaxPool_8_d32 = {
	.execute = maxpool_execute,
	.check = maxpool_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedMaxPool_8_d32_ref = {
	.execute = maxpool_execute,
	.check = maxpool_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

