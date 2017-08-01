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

static inline void do_quantized_add_888(uint8_t *aq, float amax, float amin,
                         uint8_t *bq, float bmax, float bmin,
                         float gmax, float gmin,   //guess
                         uint8_t * cq, float *cmax, float *cmin, int length, int16_t *scratch)
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
	int16_t *ptr_max = scratch;
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


static int qadd_888_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	const struct tensor *a_min_tensor = self->inputs[2];
	const struct tensor *a_max_tensor = self->inputs[3];
	const struct tensor *b_min_tensor = self->inputs[4];
	const struct tensor *b_max_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	struct qadd_888_info *info = (struct qadd_888_info *)self->opaque;
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
	tensor_set_shape(out_min_tensor,1,1,1,1);
	tensor_set_shape(out_max_tensor,1,1,1,1);
	tensor_set_float(out_min_tensor,0,out_min_float);
	tensor_set_float(out_max_tensor,0,out_max_float);

	if (sizeof(float) > out_min_tensor->max_size) return errlog(nn,"min too small");
	if (sizeof(float) > out_max_tensor->max_size) return errlog(nn,"max too small");

	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);

	if ( (a_tensor->shape.batches != b_tensor->shape.batches)
	   ||(a_tensor->shape.height != b_tensor->shape.height)
	   ||(a_tensor->shape.width != b_tensor->shape.width)
	   ||(a_tensor->shape.depth != b_tensor->shape.depth)) {
		return errlog(nn,"incompatible shapes, must be same");
	}

	if (a_tensor->data_size > out_tensor->max_size) return errlog(nn,"out too small");
	out_tensor->shape = a_tensor->shape;
	out_tensor->data_size = a_tensor->data_size;
	//logmsg(nn,0,"big range = %f small = %f",big_max_float-big_min_float,small_max_float-small_min_float);
	//logmsg(nn,0,"out ptr=%p",out_tensor->data);
	//logmsg(nn,0,"out max=%f min=%f",out_max_float,out_min_float);
	do_quantized_add_888((uint8_t *)big_tensor->data,
		big_max_float,
		big_min_float,
		(uint8_t *)small_tensor->data,
		small_max_float,
		small_min_float,
		out_max_float,
		out_min_float,
		(uint8_t *)out_tensor->data,
		&discovered_max_out,
		&discovered_min_out,
		a_tensor->data_size,
		(int16_t *)nn->scratch);
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
		info->out_max = discovered_max_out * 1.2;
		return qadd_888_execute(self,nn);
	}
	if (!min_precalculated && (discovered_min_out < out_min_float)) {
		logmsg(nn,0,"Precalculated min: %f < %f, retrying...",discovered_min_out,out_min_float);
		info->out_min = discovered_min_out * 1.2;
		return qadd_888_execute(self,nn);
	}
	if (!max_precalculated && ((info->out_max - info->out_min) < (big_max_float-big_min_float))) {
		logmsg(nn,0,"making out range at least as large as in range");
		info->out_max = big_max_float-big_min_float + info->out_min;
		return qadd_888_execute(self,nn);
	}
	return 0;
}

static int qadd_888_check(struct nn_node *self, struct nn_graph *nn)
{
	struct qadd_888_info *info;
	if (self->n_inputs != 8) return errlog(nn,"Wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"Wrong # outputs");
	if ((info = (struct qadd_888_info *)calloc(1,sizeof(struct qadd_888_info))) == NULL) {
		return errlog(nn,"calloc");
	}
	self->opaque = info;
	const struct tensor *out_min_tensor = self->inputs[6];
	const struct tensor *out_max_tensor = self->inputs[7];
	float out_max_float = tensor_get_float(out_max_tensor,0);
	float out_min_float = tensor_get_float(out_min_tensor,0);
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

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8 = {
	SFINIT(.execute, qadd_888_execute),
	SFINIT(  .check, qadd_888_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

