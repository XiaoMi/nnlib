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
#include <stdlib.h>
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#include <nn_broadcast.h>
#include <op_add_sub.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif


CREATE_OP_ADD_SUB(add, Add, +)

struct qadd_888_info {
	float out_min;
	float out_max;
	int min_precalculated;
	int max_precalculated;
};

static inline void do_quantized_add_888(
	struct nn_graph *nn,
	uint8_t *aq,
	float amax,
	float amin,
	uint8_t *bq,
	float bmax,
	float bmin,
	float gmax,
	float gmin,   //guess
	uint8_t * cq,
	float *cmax,
	float *cmin,
	int length)
{
	float stepa = amax-amin;
	float stepb = bmax-bmin;
	float step, lmin, lmax;
	float alpha  = stepa/stepb;
	if(alpha >= 256.0f) {
		vmemcpy_asm(cq, aq, length);
		*cmax = amax;
		*cmin = amin;
		return;
	}
	int16_t *ptr_max = nn_scratch_alloc(nn,256);
	short ialpha = 128.0f*alpha;
	float kappa  = 128.0f*alpha +(255.0f*amin + 255.0f*bmin)/stepb ;
	short ikappa = (int) (kappa+.0f); //+ialpha is because input is 08 ^
	//compute local max,min by updating local
	lmin = (gmin * 255.0f)/stepb;
	lmax = (gmax * 255.0f)/stepb;
	step = lmax - lmin;
	float frecip = (255.0f * 32768.0f) / step;
	float foffset = (255.0f * lmin) / step;
	if (frecip >= 32767.0f) frecip = 32767.0f;
	short recip = (int) (frecip +0.0f);
	short offset = (int) (foffset-0.5f);
	//printf("frecip=%f foffset=%f recip=%x offset=%x step=%f stepa=%f stepb=%f gmax=%f gmin=%f alpha=%f kappa=%f\n",frecip,foffset,recip,offset,step,stepa,stepb,gmax,gmin,alpha,kappa);
	quant_add_spec_asm(aq, bq, ialpha, ikappa, offset, recip, cq, ptr_max, length);
	lmax = (float)ptr_max[0];
	lmin = (float)ptr_max[64];
	//turn back to global max
	*cmin = (lmin*stepb)/255.0f;
	*cmax = (lmax*stepb)/255.0f;
}

static inline uint8_t *expand(
	struct nn_graph *nn,
	uint8_t *data,
	const struct shape srcshape,
	const struct shape dstshape)
{
	uint8_t *ret;
	uint32_t bytes = dstshape.batches * dstshape.height * dstshape.width * dstshape.depth;
	if (likely((srcshape.batches == dstshape.batches)
		&& (srcshape.height == dstshape.height)
		&& (srcshape.width == dstshape.width)
		&& (srcshape.depth == dstshape.depth))) {
		return data;
	}
	if ((ret = nn_scratch_alloc(nn,bytes)) == NULL) {
		logmsg(nn,0,"scratch fail");
		return NULL;
	}
	int32_t b,h,w,d;
	uint8_t *dst;
	uint8_t *src;
	for (b = 0; b < dstshape.batches; b++) {
		for (h = 0; h < dstshape.height; h++) {
			for (w = 0; w < dstshape.width; w++) {
				src = data
					+ b*srcshape.height*srcshape.width*srcshape.depth
					+ h*srcshape.width*srcshape.depth
					+ w*srcshape.depth;
				dst = ret
					+ b*dstshape.height*dstshape.width*dstshape.depth
					+ h*dstshape.width*dstshape.depth
					+ w*dstshape.depth;
				if (dstshape.depth == srcshape.depth) {
					vmemcpy_asm(dst,src,dstshape.depth);
				} else for (d = 0; d < dstshape.depth; d++) {
					dst[d] = src[0];
				}
			}
		}
	}
	return ret;
}


static int qadd_888_hvx(struct nn_graph *nn, void *vself)
{
	struct nn_node *self = vself;
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	const struct tensor *a_min_tensor = self->inputs[2];
	const struct tensor *a_max_tensor = self->inputs[3];
	const struct tensor *b_min_tensor = self->inputs[4];
	const struct tensor *b_max_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	struct qadd_888_info *info = self->opaque;
	float a_min_float = tensor_get_float(a_min_tensor,0);
	float a_max_float = tensor_get_float(a_max_tensor,0);
	float b_min_float = tensor_get_float(b_min_tensor,0);
	float b_max_float = tensor_get_float(b_max_tensor,0);
	float a_level_size = (a_max_float - a_min_float)/255;
	float b_level_size = (b_max_float - b_min_float)/255;
	int a_is_big = a_level_size >= b_level_size;
	const struct tensor *big_tensor = a_is_big ? a_tensor : b_tensor;
	const struct tensor *small_tensor = a_is_big ? b_tensor : a_tensor;
	float big_min_float = a_is_big ? a_min_float : b_min_float;
	float big_max_float = a_is_big ? a_max_float : b_max_float;
	float small_min_float = a_is_big ? b_min_float : a_min_float;
	float small_max_float = a_is_big ? b_max_float : a_max_float;
	float out_min_float = info->out_min;
	float out_max_float = info->out_max;
	int min_precalculated = info->min_precalculated;
	int max_precalculated = info->max_precalculated;
	float discovered_min_out;
	float discovered_max_out;
	tensor_set_single_float(out_min_tensor,out_min_float);
	tensor_set_single_float(out_max_tensor,out_max_float);
	struct shape total_shape;
	uint8_t *big_data;
	uint8_t *small_data;
	/* Handle broadcasting */
	if (!are_dims_compatible(a_tensor->shape,b_tensor->shape)) return errlog(nn,"incompatible shapes");
	total_shape.batches = output_dim(a_tensor->shape.batches,b_tensor->shape.batches);
	total_shape.height = output_dim(a_tensor->shape.height,b_tensor->shape.height);
	total_shape.width = output_dim(a_tensor->shape.width,b_tensor->shape.width);
	total_shape.depth = output_dim(a_tensor->shape.depth,b_tensor->shape.depth);
	if(tensor_out_prepare_normal_fromshape( out_tensor, &total_shape, NN_TYPE_QINT8)!=0){
		return errlog(nn,"output too small");
	}
	big_data = expand(nn,big_tensor->data,big_tensor->shape,total_shape);
	small_data = expand(nn,small_tensor->data,small_tensor->shape,total_shape);
	//logmsg(nn,0,"big range = %f small = %f",big_max_float-big_min_float,small_max_float-small_min_float);
	//logmsg(nn,0,"out ptr=%p",out_tensor->data);
	//logmsg(nn,0,"out max=%f min=%f",out_max_float,out_min_float);
	do_quantized_add_888(nn,
		big_data,
		big_max_float,
		big_min_float,
		small_data,
		small_max_float,
		small_min_float,
		out_max_float,
		out_min_float,
		out_tensor->data,
		&discovered_max_out,
		&discovered_min_out,
		total_shape.batches*total_shape.height*total_shape.width*total_shape.depth);
	//logmsg(nn,0,"discovered max = %f min = %f",discovered_max_out,discovered_min_out);
	if (0) logmsg(nn,0,"out data: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		((uint8_t *)out_tensor->data)[0],
		((uint8_t *)out_tensor->data)[1],
		((uint8_t *)out_tensor->data)[2],
		((uint8_t *)out_tensor->data)[3],
		((uint8_t *)out_tensor->data)[4],
		((uint8_t *)out_tensor->data)[5],
		((uint8_t *)out_tensor->data)[6],
		((uint8_t *)out_tensor->data)[7],
		((uint8_t *)out_tensor->data)[0+8],
		((uint8_t *)out_tensor->data)[1+8],
		((uint8_t *)out_tensor->data)[2+8],
		((uint8_t *)out_tensor->data)[3+8],
		((uint8_t *)out_tensor->data)[4+8],
		((uint8_t *)out_tensor->data)[5+8],
		((uint8_t *)out_tensor->data)[6+8],
		((uint8_t *)out_tensor->data)[7+8]);
	if (!max_precalculated && (discovered_max_out > out_max_float)) {
		logmsg(nn,0,"Precalculated max: %f > %f, retrying...",discovered_max_out,out_max_float);
		info->out_max = discovered_max_out * 1.2f;
		return qadd_888_hvx(nn,self);
	}
	if (!min_precalculated && (discovered_min_out < out_min_float)) {
		logmsg(nn,0,"Precalculated min: %f < %f, retrying...",discovered_min_out,out_min_float);
		info->out_min = discovered_min_out * 1.2f;
		return qadd_888_hvx(nn,self);
	}
	if (!max_precalculated && ((info->out_max - info->out_min) < (big_max_float-big_min_float))) {
		logmsg(nn,0,"making out range at least as large as in range");
		info->out_max = big_max_float-big_min_float + info->out_min;
		return qadd_888_hvx(nn,self);
	}
	return 0;
}

static int qadd_888_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
#if 0
	if ( (a_tensor->shape.batches != b_tensor->shape.batches)
	   ||(a_tensor->shape.height != b_tensor->shape.height)
	   ||(a_tensor->shape.width != b_tensor->shape.width)
	   ||(a_tensor->shape.depth != b_tensor->shape.depth)) {
		return errlog(nn,"incompatible shapes, must be same");
	}
	if (a_tensor->data_size > out_tensor->max_size) return errlog(nn,"out too small");
#endif
	if (sizeof(float) > out_min_tensor->max_size) return errlog(nn,"min too small");
	if (sizeof(float) > out_max_tensor->max_size) return errlog(nn,"max too small");
	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);
	logmsg(nn,2,"qadd %dx%dx%dx%d %dx%dx%dx%d",
		a_tensor->shape.batches,
		a_tensor->shape.height,
		a_tensor->shape.width,
		a_tensor->shape.depth,
		b_tensor->shape.batches,
		b_tensor->shape.height,
		b_tensor->shape.width,
		b_tensor->shape.depth);
	return nn_os_vector_call(nn,qadd_888_hvx,self);
}

static int qadd_888_check(struct nn_node *self, struct nn_graph *nn)
{
	struct qadd_888_info *info;
	if (self->n_inputs != 8) return errlog(nn,"Wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"Wrong # outputs");
	if ((info = nn_calloc(1,sizeof(struct qadd_888_info))) == NULL) {
		return errlog(nn,"calloc");
	}
	self->opaque = info;
	const struct tensor *out_min_tensor = self->inputs[6];
	const struct tensor *out_max_tensor = self->inputs[7];
	float out_max_float = tensor_get_float(out_max_tensor,0);
	float out_min_float = tensor_get_float(out_min_tensor,0);
	uint32_t out_size = self->output_defs[0].elementsize;

	out_size *= self->output_defs[0].max_sizes[0];
	out_size *= self->output_defs[0].max_sizes[1];
	out_size *= self->output_defs[0].max_sizes[2];
	out_size *= self->output_defs[0].max_sizes[3];

	// round up
	out_size = (out_size+127) & (~127);

	// ensure enough scratch
	nn_scratch_grow(nn,out_size*2+256);

	if (out_min_float == -INFINITY) {
		info->min_precalculated = 0;
		info->out_min = 0.0f;
	} else {
		info->min_precalculated = 1;
		info->out_min = out_min_float;
	}
	if (out_max_float == INFINITY) {
		info->max_precalculated = 0;
		info->out_max = 0.5f;
	} else {
		info->max_precalculated = 1;
		info->out_max = out_max_float;
	}
	return 0;
}

static int qadd_888_dtor(struct nn_node *self, struct nn_graph *nn)
{
	if (self->opaque) nn_free(self->opaque);
	self->opaque = NULL;
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8 = {
	.execute = qadd_888_execute,
	.check = qadd_888_check,
	.ctor = node_alloc_common,
	.dtor = qadd_888_dtor,
};

#if 0

static int qadd_888_hvx_d32(struct nn_graph *nn, void *vself)
{
	struct nn_node *self = vself;
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	const struct tensor *a_min_tensor = self->inputs[2];
	const struct tensor *a_max_tensor = self->inputs[3];
	const struct tensor *b_min_tensor = self->inputs[4];
	const struct tensor *b_max_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	struct qadd_888_info *info = self->opaque;
	float a_min_float = tensor_get_float(a_min_tensor,0);
	float a_max_float = tensor_get_float(a_max_tensor,0);
	float b_min_float = tensor_get_float(b_min_tensor,0);
	float b_max_float = tensor_get_float(b_max_tensor,0);
	float a_level_size = (a_max_float - a_min_float)/255;
	float b_level_size = (b_max_float - b_min_float)/255;
	int a_is_big = a_level_size >= b_level_size;
	const struct tensor *big_tensor = a_is_big ? a_tensor : b_tensor;
	const struct tensor *small_tensor = a_is_big ? b_tensor : a_tensor;
	float big_min_float = a_is_big ? a_min_float : b_min_float;
	float big_max_float = a_is_big ? a_max_float : b_max_float;
	float small_min_float = a_is_big ? b_min_float : a_min_float;
	float small_max_float = a_is_big ? b_max_float : a_max_float;
	float out_min_float = info->out_min;
	float out_max_float = info->out_max;
	int min_precalculated = info->min_precalculated;
	int max_precalculated = info->max_precalculated;
	float discovered_min_out;
	float discovered_max_out;
	tensor_set_single_float(out_min_tensor,out_min_float);
	tensor_set_single_float(out_max_tensor,out_max_float);
	uint8_t *big_data;
	uint8_t *small_data;
	/* FIXME: Handle broadcasting */
	if ((a_tensor->shape.batches != b_tensor->shape.batches)
		|| (a_tensor->shape.height != b_tensor->shape.height)
		|| (a_tensor->shape.width != b_tensor->shape.width)
		|| (a_tensor->shape.depth != b_tensor->shape.depth)) return errlog(nn,"d32 add same shape");
	if (a_tensor->shape.batches != 1) return errlog(nn,"fixme: handle batches");
	if (a_tensor->format.raw != b_tensor->format.raw) {
		return errlog(nn,"d32 same params also");
	}
	int32_t rows = a_tensor->shape.height;
	int32_t row_stride = tensor_row_stride_d32(a_tensor);
	int32_t left_pad = a_tensor->format.width_pad[0];
	tensor_out_prepare_d32_sameas(out_tensor,a_tensor);
	big_data = tensor_location_d32(big_tensor,0,0,-left_pad,0);
	small_data = tensor_location_d32(small_tensor,0,0,-left_pad,0);

#if 0
	int32_t right_pad = a_tensor->format.width_pad[1];
	int32_t big_zero = quantize_int(0.0f,big_min_float,big_max_float);
	int32_t small_zero = quantize_int(0.0f,small_min_float,small_max_float);
	int32_t d32_stride = tensor_d32_stride_d32(a_tensor);
	int32_t d32_iters = (a_tensor->shape.depth+31)/32;
	int32_t width = a_tensor->shape.width;
	padzap_part(
		tensor_location_d32(big_tensor,0,0,-left_pad,0),
		big_zero,
		d32_stride,
		d32_iters,
		row_stride,
		rows,
		left_pad);
	padzap_part(
		tensor_location_d32(small_tensor,0,0,-left_pad,0),
		small_zero,
		d32_stride,
		d32_iters,
		row_stride,
		rows,
		left_pad);
	if (right_pad > 0) {
		padzap_part(
			tensor_location_d32(big_tensor,0,0,width,0),
			big_zero,
			d32_stride,
			d32_iters,
			row_stride,
			rows,
			right_pad);
		padzap_part(
			tensor_location_d32(small_tensor,0,0,width,0),
			small_zero,
			d32_stride,
			d32_iters,
			row_stride,
			rows,
			right_pad);
	}
#endif

	//logmsg(nn,0,"big range = %f small = %f",big_max_float-big_min_float,small_max_float-small_min_float);
	//logmsg(nn,0,"out ptr=%p",out_tensor->data);
	//logmsg(nn,0,"out max=%f min=%f",out_max_float,out_min_float);
	do_quantized_add_888(nn,
		big_data,
		big_max_float,
		big_min_float,
		small_data,
		small_max_float,
		small_min_float,
		out_max_float,
		out_min_float,
		tensor_location_d32(out_tensor,0,0,-left_pad,0),
		&discovered_max_out,
		&discovered_min_out,
		rows*row_stride);
	//logmsg(nn,0,"discovered max = %f min = %f",discovered_max_out,discovered_min_out);
	if (0) logmsg(nn,0,"out data: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		((uint8_t *)out_tensor->data)[0],
		((uint8_t *)out_tensor->data)[1],
		((uint8_t *)out_tensor->data)[2],
		((uint8_t *)out_tensor->data)[3],
		((uint8_t *)out_tensor->data)[4],
		((uint8_t *)out_tensor->data)[5],
		((uint8_t *)out_tensor->data)[6],
		((uint8_t *)out_tensor->data)[7],
		((uint8_t *)out_tensor->data)[0+8],
		((uint8_t *)out_tensor->data)[1+8],
		((uint8_t *)out_tensor->data)[2+8],
		((uint8_t *)out_tensor->data)[3+8],
		((uint8_t *)out_tensor->data)[4+8],
		((uint8_t *)out_tensor->data)[5+8],
		((uint8_t *)out_tensor->data)[6+8],
		((uint8_t *)out_tensor->data)[7+8]);
	if (!max_precalculated && (discovered_max_out > out_max_float)) {
		logmsg(nn,0,"Precalculated max: %f > %f, retrying...",discovered_max_out,out_max_float);
		info->out_max = discovered_max_out * 1.2f;
		return qadd_888_hvx_d32(nn,self);
	}
	if (!min_precalculated && (discovered_min_out < out_min_float)) {
		logmsg(nn,0,"Precalculated min: %f < %f, retrying...",discovered_min_out,out_min_float);
		info->out_min = discovered_min_out * 1.2f;
		return qadd_888_hvx_d32(nn,self);
	}
	if (!max_precalculated && ((info->out_max - info->out_min) < (big_max_float-big_min_float))) {
		logmsg(nn,0,"making out range at least as large as in range");
		info->out_max = big_max_float-big_min_float + info->out_min;
		return qadd_888_hvx_d32(nn,self);
	}
	return 0;
}

static int qadd_888_execute_d32(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
#if 0
	if ( (a_tensor->shape.batches != b_tensor->shape.batches)
	   ||(a_tensor->shape.height != b_tensor->shape.height)
	   ||(a_tensor->shape.width != b_tensor->shape.width)
	   ||(a_tensor->shape.depth != b_tensor->shape.depth)) {
		return errlog(nn,"incompatible shapes, must be same");
	}
	if (a_tensor->data_size > out_tensor->max_size) return errlog(nn,"out too small");
#endif
	if (sizeof(float) > out_min_tensor->max_size) return errlog(nn,"min too small");
	if (sizeof(float) > out_max_tensor->max_size) return errlog(nn,"max too small");
	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);
	logmsg(nn,2,"qadd %dx%dx%dx%d %dx%dx%dx%d",
		a_tensor->shape.batches,
		a_tensor->shape.height,
		a_tensor->shape.width,
		a_tensor->shape.depth,
		b_tensor->shape.batches,
		b_tensor->shape.height,
		b_tensor->shape.width,
		b_tensor->shape.depth);
	return nn_os_vector_call(nn,qadd_888_hvx_d32,self);
}

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8_d32 = {
	.execute = qadd_888_execute_d32,
	.check = qadd_888_check,
	.ctor = node_alloc_common,
	.dtor = qadd_888_dtor,
};

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8_d32_ref = {
	.execute = qadd_888_execute_d32,
	.check = qadd_888_check,
	.ctor = node_alloc_common,
	.dtor = qadd_888_dtor,
};

struct nn_node_ops nn_ops_for_QuantizedSub_8p8to8_d32 = {
	.execute = qadd_888_execute_d32,
	.check = qadd_888_check,
	.ctor = node_alloc_common,
	.dtor = qadd_888_dtor,
};


struct nn_node_ops nn_ops_for_QuantizedSub_8p8to8_d32_ref = {
	.execute = qadd_888_execute_d32,
	.check = qadd_888_check,
	.ctor = node_alloc_common,
	.dtor = qadd_888_dtor,
};
#endif

