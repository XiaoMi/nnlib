
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
 * This contains implementations for quantized avg pooling node
 */


#include <nn_graph.h>
#include <string.h>

/*
 * A note about strategy
 * In Depth32, we should have some sufficient padding around H and W dimensions
 * 
 * We don't want to have to go around calculating different reciprocals.  If we assume common 3x3 filter,
 * we always want to divide by 9.  
 * 
 * Around the borders, this is a problem.  We can work around it by padding with the average of the 
 * rest of the values around the border.  
 * So for left padding we zap with the average of the leftmost two valid columns.
 * So for right padding we zap with the average of the rightmost two valid columns.
 * So for top padding we zap with the average of the topmost two valid rows
 * So for bottom padding we zap with the average of the bottommost two valid rows
 * 
 * After zapping around the edges, we should be good to rip through the data.
 * 
 */

struct tdata {
	struct nn_node *self;
	const uint8_t *indata;
	uint8_t *outdata;
	int32_t in_next_row;
	int32_t out_next_row;
	int32_t out_vectors_wide;
	int32_t out_lines;
	int32_t out_lalign;
	int32_t out_width;
	int32_t in_next_d32;
	int32_t out_next_d32;
	int32_t depth;
	int32_t width;
	int32_t left_pad;
	int32_t window_h;
	int32_t window_w;
	int32_t which_depth;
	nn_sem_t donesem;
};

int avgpool_slice_hvx_3x3_stride1(
	uint8_t *out,
	const uint8_t *in, 
	int32_t in_next_row,
	int32_t out_next_row,
	int32_t out_vectors_wide,
	int32_t out_lines,
	int32_t out_lalign);

int avgpool_hvx_d32(
	uint8_t *out,
	const uint8_t *in, 
	int32_t in_next_row,
	int32_t out_next_row,
	int32_t in_stride,
	int32_t iters,
	int32_t window_h,
	int32_t window_w,
	int32_t recip,
	int32_t lalign);

int avgpool_zap_lr(
	uint8_t *ptr,
	int32_t height,
	int32_t width,
	int32_t left_pad,
	int32_t next_row);

int avgpool_zap_row(
	uint8_t *outptr,
	uint8_t *inptr,
	int32_t next_row);

static void avgpool_execute_3x3_zap(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	int in_next_row = td->in_next_row;
	int out_lines = td->out_lines;
	uint8_t *zapin = (uint8_t *)td->indata;
	int d;

	int nd32 = (td->depth+31) >>5;
	if( out_lines > 1 ){
		avgpool_zap_row(zapin,zapin+in_next_row,in_next_row);
		avgpool_zap_row(zapin+in_next_row*(out_lines+1),zapin+in_next_row*(out_lines-1),in_next_row);
	}else{
		// just one input row; copy up & down
		vmemcpy_asm( zapin, zapin+in_next_row,  in_next_row);
		vmemcpy_asm( zapin+2*in_next_row, zapin+in_next_row,  in_next_row);
	}
	// FIXME: won't work when input width = 1

	for (d = 0; d < nd32; d++) {
		avgpool_zap_lr(
			zapin+d*td->in_next_d32,
			td->out_lines+2,
			td->width,
			td->left_pad,
			in_next_row);
	}

	nn_sem_post(&td->donesem);
}

static void avgpool_execute_3x3(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	uint8_t *outdata = td->outdata + td->out_next_d32 * td->which_depth;
	const uint8_t *indata = td->indata + td->in_next_d32 * td->which_depth;
	int in_next_row = td->in_next_row;
	int out_next_row = td->out_next_row;
	int out_vectors_wide = td->out_vectors_wide;
	int out_lines = td->out_lines;

#if 0
	logmsg(nn,0,"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x "
		"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		indata[0], indata[1], indata[2], indata[3],
		indata[4], indata[5], indata[6], indata[7],
		indata[8], indata[9], indata[10], indata[11],
		indata[12], indata[13], indata[14], indata[15],
		indata[16], indata[17], indata[18], indata[19],
		indata[20], indata[21], indata[22], indata[23],
		indata[24], indata[25], indata[26], indata[27],
		indata[28], indata[29], indata[30], indata[31]);

	logmsg(nn,0,"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x "
		"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		indata[32+0], indata[32+1], indata[32+2], indata[32+3],
		indata[32+4], indata[32+5], indata[32+6], indata[32+7],
		indata[32+8], indata[32+9], indata[32+10], indata[32+11],
		indata[32+12], indata[32+13], indata[32+14], indata[32+15],
		indata[32+16], indata[32+17], indata[32+18], indata[32+19],
		indata[32+20], indata[32+21], indata[32+22], indata[32+23],
		indata[32+24], indata[32+25], indata[32+26], indata[32+27],
		indata[32+28], indata[32+29], indata[32+30], indata[32+31]);

	logmsg(nn,0,"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x "
		"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		indata[64+0], indata[64+1], indata[64+2], indata[64+3],
		indata[64+4], indata[64+5], indata[64+6], indata[64+7],
		indata[64+8], indata[64+9], indata[64+10], indata[64+11],
		indata[64+12], indata[64+13], indata[64+14], indata[64+15],
		indata[64+16], indata[64+17], indata[64+18], indata[64+19],
		indata[64+20], indata[64+21], indata[64+22], indata[64+23],
		indata[64+24], indata[64+25], indata[64+26], indata[64+27],
		indata[64+28], indata[64+29], indata[64+30], indata[64+31]);
#elif 0
	logmsg(nn,0,"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		indata[0+0],indata[0+32],indata[0+64],indata[0+96],
		indata[128+0],indata[128+32],indata[128+64],indata[128+96],
		indata[256+0],indata[256+32],indata[256+64],indata[256+96]);
	logmsg(nn,0,"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		indata[384+0],indata[384+32],indata[384+64],indata[384+96],
		indata[512+0],indata[512+32],indata[512+64],indata[512+96],
		indata[640+0],indata[640+32],indata[640+64],indata[640+96]);
	logmsg(nn,0,"%02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		indata[768+0],indata[768+32],indata[768+64],indata[768+96],
		indata[896+0],indata[896+32],indata[896+64],indata[896+96],
		indata[1024+0],indata[1024+32],indata[1024+64],indata[1024+96]);
#endif

	avgpool_slice_hvx_3x3_stride1(
		outdata,
		indata,
		in_next_row,
		out_next_row,
		out_vectors_wide,
		out_lines,
		td->out_lalign);
	nn_sem_post(&td->donesem);
}

static void avgpool_execute_non3x3(struct nn_graph *nn, void *vtdata)
{
	struct tdata *td = vtdata;
	uint8_t *outdata = td->outdata;
	const uint8_t *indata = td->indata;
	int in_next_row = td->in_next_row;
	int out_next_row = td->out_next_row;
	int out_lines = td->out_lines;
	int32_t window_h = td->window_h;
	int32_t window_w = td->window_w;
	int32_t in_next_d32 = td->in_next_d32;
	int32_t out_next_d32 = td->out_next_d32;
	int d,h;
	uint32_t recip = (1<<17)/(unsigned)(window_h*window_w);
	recip =  (recip+1)>>1;

	int nd32 = (td->depth+31)>>5;
	int wout_vecs = (td->out_width + 3)>>2;

	for (h = 0; h < out_lines; h++) {
		for (d = 0; d <  nd32; d++) {
			avgpool_hvx_d32(
				outdata+d*out_next_d32,
				indata+d*in_next_d32,
				in_next_row,
				out_next_row,
				128,
				wout_vecs,
				window_h,
				window_w,
				recip,
				td->out_lalign);
		}
		indata += in_next_row;
		outdata += out_next_row;
	}
	nn_sem_post(&td->donesem);
}



static int avgpool_execute(struct nn_node *self, struct nn_graph *nn)
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
	int32_t required_w_before, required_h_before;

	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,window_width,stride_width,self->padding, & required_w_before);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,window_height,stride_height,self->padding, & required_h_before);
	int32_t out_depth = in_depth;

	//int32_t required_w_after = nn_pad_compute_after(in_width,window_width,stride_width,self->padding);
	//int32_t required_h_after = nn_pad_compute_after(in_height,window_height,stride_height,self->padding);

	int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_batch_size = in_height_total*in_width_total*in_depth_total;

	int32_t out_left_pad = 4;
	int32_t out_right_pad = (-out_width) & 3;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = 4;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-in_depth) & 31;

	int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_batch_size = out_height_total*out_width_total*out_depth_total;

	int32_t in_next_row = in_width_total*in_depth_total;
	int32_t in_next_d32 = in_width_total*32;

	int32_t out_next_row = out_width_total*out_depth_total;
	int32_t out_next_d32 = out_width_total*32;

	const uint8_t *in_data = (uint8_t const *)in_tensor->data;
	const uint8_t *in_data_start = in_data + (in_next_row*(in_top_pad-required_h_before));
	uint8_t *out_data = (uint8_t *)out_tensor->data;
	uint8_t *out_data_start = out_data + out_next_row*out_top_pad;
	int32_t n_out_d32s = out_depth_total/32;
	int i;
	int b;

	logmsg(nn,2,"avgpool %dx%d %dx%dx%dx%d %dx%dx%dx%d n_d32=%d",window_height,window_width,
		in_batches,in_width,in_height,in_depth,
		out_batches,out_width,out_height,out_depth,n_out_d32s);

	struct tdata td = {
		.self = self,
		.indata = in_data_start,
		.outdata = out_data_start,
		.in_next_row = in_next_row,
		.out_vectors_wide = out_width_total*32/128,
		.out_next_row = out_next_row,
		.out_lines = out_height,
		.depth = in_depth,
		.width = in_width,
		.out_width = out_width,
		.window_h = window_height,
		.window_w = window_width,
		.left_pad = in_left_pad,
		.in_next_d32 = in_next_d32,
		.out_next_d32 = out_next_d32,
		.out_lalign = required_w_before*32,
	};

	struct tdata tds[n_out_d32s];

	nn_sem_init(&td.donesem,0);
	for (i = 0; i < n_out_d32s; i++) {
		tds[i] = td;
	}

	//if (self->padding == NN_PAD_VALID) 
	if (required_w_before == 0) {
		td.out_vectors_wide -= out_left_pad/4;
		td.outdata += out_left_pad * 32;
		td.indata += in_left_pad * 32;
		td.out_lalign = 0;
	}

	//memset(out_data,0,out_width_total*out_height_total*out_depth_total);

	if ((window_tensor->shape.batches != 1)
		|| (window_tensor->shape.depth != 1)
		|| (window_height != window_width)
		|| (stride_height != stride_width)
		|| (stride_tensor->shape.depth != 1)
		|| (stride_tensor->shape.batches != 1)) {
		return errlog(nn,"bad window/stride shape");
	}


	if (tensor_set_single_float(out_min,tensor_get_float(in_min_tensor,0)) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_set_single_float(out_max,tensor_get_float(in_max_tensor,0)) != 0) {
		return errlog(nn,"max out prep fail");
	}
	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail (%d+%d+%d)x(%d+%d+%d)x(%d+%d+%d) %d > %d?",
			out_height,out_top_pad,out_bottom_pad,
			out_width,out_left_pad,out_right_pad,
			out_depth,out_depth_before_pad,out_depth_after_pad,
			out_width_total*out_height_total*out_depth_total,
			out_tensor->max_size);
	}

	for (b = 0; b < out_batches; b++) {
		if ((window_tensor->shape.width == 3)
			&& (stride_tensor->shape.width == 1)) {
			if (required_w_before > 0) {
				nn_os_work_for_vector(nn,avgpool_execute_3x3_zap,&td);
				nn_sem_wait(&td.donesem);
			}
			for (i = 0; i < n_out_d32s; i++) {
				tds[i].which_depth = i;
				nn_os_work_for_vector(nn,avgpool_execute_3x3,&tds[i]);
			}
			for (i = 0; i < n_out_d32s; i++) {
				td.out_vectors_wide = out_width_total*out_depth_total/128;
				nn_sem_wait(&tds[i].donesem);
			}
		} else {
			nn_os_work_for_vector(nn,avgpool_execute_non3x3,&td);
			nn_sem_wait(&td.donesem);
		}
		td.indata += in_batch_size;
		td.outdata += out_batch_size;
		for (i = 0; i < n_out_d32s; i++) {
			tds[i].indata += in_batch_size;
			tds[i].outdata += out_batch_size;
		}
	}
	return 0;
}

static int avgpool_check(struct nn_node *self, struct nn_graph *nn)
{
	if (self->n_inputs != 5) return errlog(nn,"avgpool wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"avgpool wrong # outs");
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedAvgPool_8_d32 = {
	.execute = avgpool_execute,
	.check = avgpool_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedAvgPool_8_d32_ref = {
	.execute = avgpool_execute,
	.check = avgpool_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

