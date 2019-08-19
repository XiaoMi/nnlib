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
#include <stdlib.h>
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#include <nn_broadcast.h>
//#include <op_add_sub.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_hexagon_protos.h"
#include "hvx_inlines.h"

// This creates:
//  nn_ops_for_QuantizedAdd_8p8to32
//  nn_ops_for_QuantizedAdd_8p8to32_ref

//CREATE_OP_ADD_SUB(add, Add, +)

struct qadd_888_info
{
	float out_min;
	float out_max;
	int8_t min_precalculated;
	int8_t max_precalculated;
	int8_t has_run_before;
};
#ifdef HEXAGON_V66
#define MAX_THREADS 4
#else
#define MAX_THREADS 2
#endif
struct addsub_flat_direct_runstate;
typedef void (*core_oper_fp)(struct addsub_flat_direct_runstate *rstp, HVX_Vector const *ptrA, HVX_Vector const *ptrB, HVX_Vector *ptrout, int n_elements, int nvecs, int16_t *minmax_buf);

static inline void do_quantized_add_888(
	struct nn_graph *nn,
	uint8_t *aq,
	float amax,
	float amin,
	uint8_t *bq,
	float bmax,
	float bmin,
	float gmax,
	float gmin, //guess
	uint8_t *cq,
	float *cmax,
	float *cmin,
	int length)
{
	float stepa = amax - amin;
	float stepb = bmax - bmin;
	float step, lmin, lmax;
	float alpha = stepa / stepb;
	if (alpha >= 256.0f)
	{
		vmemcpy_asm(cq, aq, length);
		*cmax = amax;
		*cmin = amin;
		return;
	}
	int16_t *ptr_max;
	if ((ptr_max = nn_scratch_alloc(nn, 256)) == NULL)
	{
		errlog(nn, "scratch alloc fail");
		return;
	}
	short ialpha = 128.0f * alpha;
	float kappa = 128.0f * alpha + (255.0f * amin + 255.0f * bmin) / stepb;
	short ikappa = (int)(kappa + .0f); //+ialpha is because input is 08 ^
	//compute local max,min by updating local
	lmin = (gmin * 255.0f) / stepb;
	lmax = (gmax * 255.0f) / stepb;
	step = lmax - lmin;
	float frecip = (255.0f * 32768.0f) / step;
	float foffset = (255.0f * lmin) / step;
	if (frecip >= 32767.0f)
		frecip = 32767.0f;
	short recip = (int)(frecip + 0.0f);
	short offset = (int)(foffset - 0.5f);
	//printf("frecip=%f foffset=%f recip=%x offset=%x step=%f stepa=%f stepb=%f gmax=%f gmin=%f alpha=%f kappa=%f\n",frecip,foffset,recip,offset,step,stepa,stepb,gmax,gmin,alpha,kappa);
	quant_add_spec_asm(aq, bq, ialpha, ikappa, offset, recip, cq, ptr_max, length);
	lmax = (float)ptr_max[0];
	lmin = (float)ptr_max[64]; //if you're changing this update the scrach size check in qadd_888_execute
	//turn back to global max
	*cmin = (lmin * stepb) / 255.0f;
	*cmax = (lmax * stepb) / 255.0f;
}

static inline uint8_t *expand(
	struct nn_graph *nn,
	uint8_t *data,
	const struct shape srcshape,
	const struct shape dstshape)
{
	uint8_t *ret;
	if (likely(shape_matches(&srcshape, &dstshape)))
	{
		return data;
	}
	uint32_t bytes = dstshape.batches * dstshape.height * dstshape.width * dstshape.depth;
	if ((ret = nn_scratch_alloc(nn, bytes)) == NULL)
	{
		logmsg(nn, 0, "scratch fail");
		return NULL;
	}
	int32_t b, h, w, d;
	uint8_t *dst;
	uint8_t *src;
	for (b = 0; b < dstshape.batches; b++)
	{
		for (h = 0; h < dstshape.height; h++)
		{
			for (w = 0; w < dstshape.width; w++)
			{
				src = data + b * srcshape.height * srcshape.width * srcshape.depth 
					   + h * srcshape.width * srcshape.depth 
					   + w * srcshape.depth;
				dst = ret + b * dstshape.height * dstshape.width * dstshape.depth 
					  + h * dstshape.width * dstshape.depth 
					  + w * dstshape.depth;
				if (dstshape.depth == srcshape.depth)
				{
					vmemcpy_asm(dst, src, dstshape.depth);
				}
				else
					for (d = 0; d < dstshape.depth; d++)
					{
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
	float a_min_float = tensor_get_float(a_min_tensor, 0);
	float a_max_float = tensor_get_float(a_max_tensor, 0);
	float b_min_float = tensor_get_float(b_min_tensor, 0);
	float b_max_float = tensor_get_float(b_max_tensor, 0);
	float a_level_size = (a_max_float - a_min_float) / 255;
	float b_level_size = (b_max_float - b_min_float) / 255;
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
	tensor_set_single_float(out_min_tensor, out_min_float);
	tensor_set_single_float(out_max_tensor, out_max_float);
	struct shape total_shape;
	uint8_t *big_data;
	uint8_t *small_data;

	nn_scratch_reset(nn);

	/* Handle broadcasting */
	if (!are_dims_compatible(a_tensor->shape, b_tensor->shape))
		return errlog(nn, "incompatible shapes");
	total_shape.batches = output_dim(a_tensor->shape.batches, b_tensor->shape.batches);
	total_shape.height = output_dim(a_tensor->shape.height, b_tensor->shape.height);
	total_shape.width = output_dim(a_tensor->shape.width, b_tensor->shape.width);
	total_shape.depth = output_dim(a_tensor->shape.depth, b_tensor->shape.depth);
	if (tensor_out_prepare_normal_fromshape(out_tensor, &total_shape, NN_TYPE_QINT8) != 0)
	{
		return errlog(nn, "output too small");
	}
	big_data = expand(nn, big_tensor->data, big_tensor->shape, total_shape);
	small_data = expand(nn, small_tensor->data, small_tensor->shape, total_shape);
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
			     total_shape.batches * total_shape.height * total_shape.width * total_shape.depth);
	//logmsg(nn,0,"discovered max = %f min = %f",discovered_max_out,discovered_min_out);
	if (0)
		logmsg(nn, 0, "out data: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
		       ((uint8_t *)out_tensor->data)[0],
		       ((uint8_t *)out_tensor->data)[1],
		       ((uint8_t *)out_tensor->data)[2],
		       ((uint8_t *)out_tensor->data)[3],
		       ((uint8_t *)out_tensor->data)[4],
		       ((uint8_t *)out_tensor->data)[5],
		       ((uint8_t *)out_tensor->data)[6],
		       ((uint8_t *)out_tensor->data)[7],
		       ((uint8_t *)out_tensor->data)[0 + 8],
		       ((uint8_t *)out_tensor->data)[1 + 8],
		       ((uint8_t *)out_tensor->data)[2 + 8],
		       ((uint8_t *)out_tensor->data)[3 + 8],
		       ((uint8_t *)out_tensor->data)[4 + 8],
		       ((uint8_t *)out_tensor->data)[5 + 8],
		       ((uint8_t *)out_tensor->data)[6 + 8],
		       ((uint8_t *)out_tensor->data)[7 + 8]);
	if (!max_precalculated && (discovered_max_out > out_max_float))
	{
		logmsg(nn, 0, "Precalculated max: %f > %f, retrying...", discovered_max_out, out_max_float);
		info->out_max = discovered_max_out * 1.2f;
		return qadd_888_hvx(nn, self);
	}
	if (!min_precalculated && (discovered_min_out < out_min_float))
	{
		logmsg(nn, 0, "Precalculated min: %f < %f, retrying...", discovered_min_out, out_min_float);
		info->out_min = discovered_min_out * 1.2f;
		return qadd_888_hvx(nn, self);
	}
	if (!max_precalculated && ((info->out_max - info->out_min) < (big_max_float - big_min_float)))
	{
		logmsg(nn, 0, "making out range at least as large as in range");
		info->out_max = big_max_float - big_min_float + info->out_min;
		return qadd_888_hvx(nn, self);
	}
	return 0;
}

static int addsub_flat_direct_execute(struct nn_node *self, struct nn_graph *nn);

static int qadd_888_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	if (shape_matches(&a_tensor->shape, &b_tensor->shape))
	{
		return addsub_flat_direct_execute(self, nn);
	}
#if 0
	if ( (a_tensor->shape.batches != b_tensor->shape.batches)
	   ||(a_tensor->shape.height != b_tensor->shape.height)
	   ||(a_tensor->shape.width != b_tensor->shape.width)
	   ||(a_tensor->shape.depth != b_tensor->shape.depth)) {
		return errlog(nn,"incompatible shapes, must be same");
	}
	if (a_tensor->data_size > out_tensor->max_size) return errlog(nn,"out too small");
#endif
	if (sizeof(float) > out_min_tensor->max_size)
		return errlog(nn, "min too small");
	if (sizeof(float) > out_max_tensor->max_size)
		return errlog(nn, "max too small");
	out_min_tensor->data_size = sizeof(float);
	out_max_tensor->data_size = sizeof(float);

	logmsg(nn, 2, "qadd %dx%dx%dx%d %dx%dx%dx%d",
		   a_tensor->shape.batches,
		   a_tensor->shape.height,
		   a_tensor->shape.width,
		   a_tensor->shape.depth,
		   b_tensor->shape.batches,
		   b_tensor->shape.height,
		   b_tensor->shape.width,
		   b_tensor->shape.depth);
	return nn_os_vector_call(nn, qadd_888_hvx, self);
}
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

enum
{
	op_add,
	op_subtract
}
operator;
// scaling parms for add/sub
//
struct addsub_scaling_parms
{
	int16_t a_scale;
	int16_t b_scale;
	uint16_t scales_hi; // repacked for the hvx code.
	uint16_t scales_lo;
	int16_t operator; // op_add, op_sub
	int32_t offset;
	int16_t final_rsh;
	int16_t intermed_zero;				// the step (2) value corresponding to a 'zero'.
	int32_t intermed_min, intermed_max; // intermed values in this range won't clip at the output.
	float netscale;						// used to transform 'intermediate' result to application result

	int16_t ab_zero[2];
	float ab_stepsize[2];
	int underrange;
};

struct addsub_flat_direct_runstate
{
	struct addsub_scaling_parms osp;

	uint8_t const *inA;
	uint8_t const *inB;
	uint8_t *outp;
	float out_min_specified, out_max_specified;
	int8_t need_minmax;
	unsigned elements_total; // # of elements to do
	int elements_chunk;		 // chunk size (# per work unit; multiple of 128)
	volatile int next_pos;   // current position
	volatile int thrindx;	// used to assign thread indices
	int16_t *minmax_buf;
	nn_sem_t done_sem;
};
static inline int set_addsub_scaling_for_output_range(struct qadd_888_info *info, struct addsub_scaling_parms *osp, int fixed_range);
static void addsub_flat_work(struct nn_graph *nn, void *rstpv);
static int check_addsub_need_rerun(struct nn_graph *nn, struct addsub_flat_direct_runstate *rstp, struct qadd_888_info *info, int nthreads);
//
// This is an add/subract for quantized add, which only supports flat, no broadcasting case,
// using the arithmetic algo transplanted from add_d32.
//
static int
addsub_flat_direct_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct qadd_888_info *info = (struct qadd_888_info *)self->opaque;
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	int operator= op_add;

	if (tensor_out_prepare_normal_fromshape(out_tensor, &a_tensor->shape, NN_TYPE_QUINT8) != 0)
	{
		return errlog(nn, "output too small");
	}
	///////////////////////////////////////////////////////
	// set up ranges.
	///////////////////////////////////////////////////////
	float a_in_min = tensor_get_float(self->inputs[2], 0);
	float a_in_max = tensor_get_float(self->inputs[3], 0);
	float b_in_min = tensor_get_float(self->inputs[4], 0);
	float b_in_max = tensor_get_float(self->inputs[5], 0);

	int min_precalc = info->min_precalculated;
	int max_precalc = info->max_precalculated;

	struct addsub_flat_direct_runstate runstate;
	runstate.osp.operator= operator;
	runstate.osp.ab_zero[0] = get_qu8_level_size_zero(a_in_min, a_in_max, &runstate.osp.ab_stepsize[0]);
	runstate.osp.ab_zero[1] = get_qu8_level_size_zero(b_in_min, b_in_max, &runstate.osp.ab_stepsize[1]);

	// obtain the pre-calc values where applicable...
	if (min_precalc)
	{
		runstate.out_min_specified = tensor_get_float(self->inputs[6], 0);
	}
	if (max_precalc)
	{
		runstate.out_max_specified = tensor_get_float(self->inputs[7], 0);
	}

	if (!min_precalc || !max_precalc)
	{
		// given {a,b}_in_{min,max}, and assuming
		//   in_min <= 0 ,  in_max >=0 , in_max > in_min:
		// - largest and smallest output values are given by the below
		float out_max_all; // should be >= 0
		float out_min_all; // should be <=0
		if (operator== op_add)
		{
			out_max_all = a_in_max + b_in_max;
			out_min_all = a_in_min + b_in_min;
		}
		else
		{
			out_max_all = a_in_max - b_in_min;
			out_min_all = a_in_min - b_in_max;
		}
		out_max_all = fmaxf(out_max_all, out_min_all + 0.001f);

		if (!info->has_run_before)
		{
			// make up our own endpoints; make the range 1/8 of
			// the theoretical range
			float ored_min = 0.125f * out_min_all;
			float ored_max = 0.125f * out_max_all;

			if (min_precalc)
			{ // move max only
				info->out_min = runstate.out_min_specified;
				info->out_max = fmax(0.0f, info->out_min + (ored_max - ored_min));
			}
			else if (info->max_precalculated)
			{ // == 2: move min only
				info->out_max = runstate.out_max_specified;
				info->out_min = fminf(0.0f, info->out_max - (ored_max - ored_min));
			}
			else
			{
				info->out_min = ored_min;
				info->out_max = ored_max;
			}
			adjust_minmax_for_zero_with_constraints(
				&info->out_min, &info->out_max, min_precalc + 2 * max_precalc);
			info->has_run_before = 1;
		}
		else
		{
			// make sure that the output range is at least 1/32 of the
			// total input range.
			float out_range_all = out_max_all - out_min_all;
			float current_range = info->out_max - info->out_min;
			float d = out_range_all - 32.0f * current_range;
			if (d > 0)
			{ // expand endpoints
				float r = d / (32.0f * (out_range_all - current_range));
				if (!min_precalc)
				{ // can move min?
					float adj_min = (out_min_all - info->out_min) * r;
					if (adj_min < 0.0f)
					{
						info->out_min += adj_min;
					}
				}
				else
				{ // reset to keeo from drifting
					info->out_min = runstate.out_min_specified;
				}
				if (!max_precalc)
				{ // can move max ?
					float adj_max = (out_max_all - info->out_max) * r;
					if (adj_max > 0.0f)
					{
						info->out_max += adj_max;
					}
				}
				else
				{
					info->out_max = runstate.out_max_specified;
				}
				adjust_minmax_for_zero_with_constraints(
					&info->out_min, &info->out_max, min_precalc + 2 * max_precalc);
			}
		}
	}

	int res = set_addsub_scaling_for_output_range(info, &runstate.osp, min_precalc && info->max_precalculated);
	if (res != 0)
		return errlog(nn, "scaling failed");

	runstate.inA = a_tensor->data;
	runstate.inB = b_tensor->data;
	runstate.outp = out_tensor->data;
	runstate.need_minmax = !max_precalc || !max_precalc;
	runstate.minmax_buf = (int16_t *)nn->scratch;
	//
	// how many elements do we need to add ?
	//
	int elements_total = tensor_element_count(a_tensor);

#if (MAX_THREADS != 2) && (MAX_THREADS != 4)
#error "assuming MAX_THREADS==2 or 4 here"
#endif
	// try to work in chunks of 128 vectors, though we reduce it
	// if the size of the work is small. The overhead
	// per chunk is not high, and the prefetch
	// considerations mean we don't want chunks too large.
	//
	int elements_chunk = 128 * 128;
	// elements_chunk must be a multiple of 128; it's ok
	// if it's > elements_total

	int nthreads = MAX_THREADS;
	if (elements_total < 2 * MAX_THREADS * elements_chunk)
	{
		if (elements_total < 8 * 128)
		{ // leave that as is
			nthreads = 1;
		}
		if (MAX_THREADS == 2 || elements_total < 64 * 128)
		{
			// just split in 2 or 4 roughly equal (for NTHREADS=4, only 2)
			int et = elements_total;
			if (et > 256 * 128)
				et = (et + 1) >> 1;
			elements_chunk = ((et + 255) >> 8) * 128;
			nthreads = 2;
		}
		else
		{ // split in 4 or 8 roughly equal
			int et = elements_total;
			if (et > 512 * 128)
				et = (et + 1) >> 1;
			elements_chunk = ((et + 511) >> 9) * 128;
			nthreads = 4;
		}
	}
	runstate.elements_total = elements_total;
	runstate.elements_chunk = elements_chunk;

	//printf("len = %d; %d threads doing chunks of %d\n", elements_total, nthreads, elements_chunk);

	while (1)
	{ // once or twice
		runstate.next_pos = 0;
		runstate.thrindx = 0;
		nn_sem_init(&runstate.done_sem, 0);

		for (int i = 0; i < nthreads; i++)
		{
			nn_os_work_for_vector(nn, addsub_flat_work, &runstate);
		}
		nn_sem_wait_n_times(&runstate.done_sem, nthreads);

		if (runstate.need_minmax == 0)
			break;
		int res = check_addsub_need_rerun(nn, &runstate, info, nthreads);
		if (res <= 0)
		{
			if (res < 0)
				return -1;
			break;
		}
		runstate.need_minmax = 0; // don't need to range check second run.
	}
	// set the outputs
	tensor_set_single_float(out_min_tensor, info->out_min);
	tensor_set_single_float(out_max_tensor, info->out_max);
	return 0;
}
// this converts a value at step (2) of the process to a float output value.
//
static inline float __attribute__((unused))
convert_intermed_to_outval(struct addsub_scaling_parms const *osp, int val)
{
	//printf("convert %d:  - %d * %.5f --> %.5f\n", val, osp->intermed_zero,
	//osp->netscale, (val - osp->intermed_zero)*osp->netscale );

	return (val - osp->intermed_zero) * osp->netscale;
}
// check if we need to run again:
//  0 : no, 1: yes, -1 :error
//
static int
check_addsub_need_rerun(struct nn_graph *nn, struct addsub_flat_direct_runstate *rstp, struct qadd_888_info *info, int nthreads)
{
	// collect the min and max ...
	int16_t const *mmp = rstp->minmax_buf;
	int minvali = mmp[0];
	int maxvali = mmp[1];
	for (int i = 1; i < nthreads; i++)
	{
		minvali = max_i32(minvali, mmp[64 * i]); // max, since min is encoded as ~min
		maxvali = max_i32(maxvali, mmp[64 * i + 1]);
	}
	minvali = -1 - minvali; // now is proper min

	float actual_min = convert_intermed_to_outval(&rstp->osp, minvali);
	float actual_max = convert_intermed_to_outval(&rstp->osp, maxvali);
	float out_range = actual_max - actual_min;

	//printf("range found is: %d.. %d  [%f ... %f]\n", minvali, maxvali, actual_min, actual_max + out_range*0);

	// pad the range out a bit
	out_range = out_range + fmaxf(0.15f * out_range, 1e-4);
	float padded_min = fmin(0.0f, actual_max - out_range);
	float padded_max = fmax(0.0f, actual_min + out_range);
	int any_adj = 0;
	float out_min = info->out_min;
	float out_max = info->out_max;

	int min_max_precalc = ((info->min_precalculated) ? 1 : 0) + ((info->max_precalculated) ? 2 : 0);

	if ((min_max_precalc & 1) == 0 && actual_min < out_min)
	{
		logmsg(nn, 2, "adjusting out_min from %f to %f (actual min is %f)", out_min, padded_min, actual_min);
		out_min = padded_min;
		any_adj = 1;
	}
	if ((min_max_precalc & 2) == 0 && actual_max > out_max)
	{
		logmsg(nn, 2, "adjusting out_max from %f to %f (actual max is %f)", out_max, padded_max, actual_max);
		out_max = padded_max;
		any_adj = 1;
	}
	if (!any_adj)
		return 0; // all done, if no adjustment needed

	// reset original endpoints, where applic.
	// (so that small tweaks for zero correction don't accumulate)
	//
	if ((min_max_precalc & 1))
		out_min = rstp->out_min_specified;
	if ((min_max_precalc & 2))
		out_max = rstp->out_max_specified;
	info->out_min = out_min;
	info->out_max = out_max;

	if (out_min < 0.0f)
	{
		// correct range and reload
		adjust_minmax_for_zero_with_constraints(&info->out_min, &info->out_max, min_max_precalc);
	}
	//printf("rerange to %f  ... %f\n", info->out_min, info->out_max);
	int res = set_addsub_scaling_for_output_range(info, &rstp->osp, 0);
	if (res < 0)
		return -1;
	return 1;
}

////////
///// HVX code for add
////////
static inline HVX_VectorPair first_stage(HVX_Vector vinA, HVX_Vector vinB, int32_t sclab_hi, int32_t sclab_lo)
{
	HVX_VectorPair vinAB = Q6_W_vcombine_VV(vinB, vinA);

	HVX_VectorPair hisop = Q6_Wh_vmpa_WubRb(vinAB, sclab_hi);
	HVX_VectorPair losop = Q6_Wh_vmpa_WubRb(vinAB, sclab_lo);
	HVX_Vector hisop_0 = Q6_V_lo_W(hisop);
	HVX_Vector hisop_1 = Q6_V_hi_W(hisop);
	HVX_Vector losop_0 = Q6_V_lo_W(losop);
	HVX_Vector losop_1 = Q6_V_hi_W(losop);
	// >> these by 7
	losop_0 = Q6_Vuh_vlsr_VuhR(losop_0, 7);
	losop_1 = Q6_Vuh_vlsr_VuhR(losop_1, 7);
	// average lo & hi to get result.
	return Q6_W_vcombine_VV(
		Q6_Vh_vavg_VhVh(hisop_1, losop_1),
		Q6_Vh_vavg_VhVh(hisop_0, losop_0));
}
// the rest of the op is: add offset (with sat) and then >> rsh
static inline HVX_Vector second_stage(HVX_VectorPair vin, int32_t offset, int rsh)
{
	HVX_Vector voffs = Q6_V_vsplat_R(offset);

	HVX_Vector p0 = Q6_Vh_vadd_VhVh_sat(Q6_V_lo_W(vin), voffs);
	HVX_Vector p1 = Q6_Vh_vadd_VhVh_sat(Q6_V_hi_W(vin), voffs);
	return Q6_Vub_vasr_VhVhR_rnd_sat(p1, p0, rsh);
}

static inline HVX_Vector_x4
first_stage_underrange(HVX_Vector va, HVX_Vector vb, int scalea, int scaleb, int initial_rsh)
{
	HVX_VectorPair xta = Q6_Wb_vshuffoe_VbVb(Q6_V_vzero(), va);
	HVX_VectorPair xtb = Q6_Wb_vshuffoe_VbVb(Q6_V_vzero(), vb);
	HVX_VectorPair p02 = Q6_Ww_vmpy_VhRh(Q6_V_lo_W(xta), scalea); // even 32-bit prods
	HVX_VectorPair p13 = Q6_Ww_vmpy_VhRh(Q6_V_hi_W(xta), scalea); // odd 32-bit prods
	p02 = Q6_Ww_vmpyacc_WwVhRh_sat(p02, Q6_V_lo_W(xtb), scaleb);
	p13 = Q6_Ww_vmpyacc_WwVhRh_sat(p13, Q6_V_hi_W(xtb), scaleb);

	HVX_Vector_x4 result;
	result.val[0] = Q6_Vw_vasr_VwR(Q6_V_lo_W(p02), initial_rsh);
	result.val[1] = Q6_Vw_vasr_VwR(Q6_V_lo_W(p13), initial_rsh);
	result.val[2] = Q6_Vw_vasr_VwR(Q6_V_hi_W(p02), initial_rsh);
	result.val[3] = Q6_Vw_vasr_VwR(Q6_V_hi_W(p13), initial_rsh);
	return result;
}
// the rest of the op is: add offset (with sat) and then >> rsh
static inline HVX_Vector
second_stage_underrange(HVX_Vector_x4 first, int offset, int final_rsh)
{
	HVX_Vector voffs = Q6_V_vsplat_R(offset);
	HVX_Vector p0 = Q6_Vw_vadd_VwVw_sat(first.val[0], voffs);
	HVX_Vector p1 = Q6_Vw_vadd_VwVw_sat(first.val[1], voffs);
	HVX_Vector p2 = Q6_Vw_vadd_VwVw_sat(first.val[2], voffs);
	HVX_Vector p3 = Q6_Vw_vadd_VwVw_sat(first.val[3], voffs);
	HVX_Vector s02 = Q6_Vh_vsat_VwVw(p2, p0);
	HVX_Vector s13 = Q6_Vh_vsat_VwVw(p3, p1);
	return Q6_Vub_vasr_VhVhR_rnd_sat(s13, s02, final_rsh);
}

static inline void __attribute__((always_inline))
add_core_hvx(struct addsub_flat_direct_runstate *rstp, HVX_Vector const *ptrA, HVX_Vector const *ptrB, HVX_Vector *ptrout, int n_elements, int nvecs, int16_t *minmax_buf)
{
	int scales_hi = rstp->osp.scales_hi;
	int scales_lo = rstp->osp.scales_lo;
	int offset = rstp->osp.offset;
	int final_rsh = rstp->osp.final_rsh;
	scales_hi = Q6_R_combine_RlRl(scales_hi, scales_hi);
	scales_lo = Q6_R_combine_RlRl(scales_lo, scales_lo);
	offset = Q6_R_combine_RlRl(offset, offset);
	HVX_Vector vCenter = q6op_Vh_vsplat_R(rstp->osp.intermed_zero);
	HVX_Vector vminmax_all = *(HVX_Vector *)minmax_buf;

	int need_minmax = rstp->need_minmax;
	if (0 && !need_minmax)
	{ // fast loop  (NOTE: it doesn't really turn out to be faster)
		HVX_VectorPair p1 = first_stage(ptrA[0], ptrB[0], scales_hi, scales_lo);
		for (int i = 0; i < nvecs - 1; i++)
		{
			*ptrout++ = second_stage(p1, offset, final_rsh);
			p1 = first_stage(ptrA[i + 1], ptrB[i + 1], scales_hi, scales_lo);
		}
		*ptrout = second_stage(p1, offset, final_rsh);
	}
	else
	{
		int lastn = n_elements - (nvecs - 1) * 128; // 1..128: # in use on last iteration.
		HVX_Vector vmin = vCenter;
		HVX_Vector vmax = vCenter;
		HVX_VectorPair p1 = first_stage(ptrA[0], ptrB[0], scales_hi, scales_lo);
		for (int i = 0; i < nvecs - 1; i++)
		{
			vmin = Q6_Vh_vmin_VhVh(vmin, Q6_Vh_vmin_VhVh(Q6_V_lo_W(p1), Q6_V_hi_W(p1)));
			vmax = Q6_Vh_vmax_VhVh(vmax, Q6_Vh_vmax_VhVh(Q6_V_lo_W(p1), Q6_V_hi_W(p1)));
			*ptrout++ = second_stage(p1, offset, final_rsh);
			p1 = first_stage(ptrA[i + 1], ptrB[i + 1], scales_hi, scales_lo);
		}
		*ptrout = second_stage(p1, offset, final_rsh);
		if (need_minmax)
		{ // last needs special treatment in minmax
			HVX_Vector vt0 = Q6_V_lo_W(p1);
			HVX_Vector vt1 = Q6_V_hi_W(p1);
			// mask according to lastn
			if (lastn < 128)
			{
				vt1 = Q6_V_vmux_QVV(Q6_Q_vsetq_R(lastn & ~1), vt1, vCenter);
				if (lastn < 127)
				{
					vt0 = Q6_V_vmux_QVV(Q6_Q_vsetq_R((lastn + 1) & ~1), vt0, vCenter);
				}
			}
			vmin = Q6_Vh_vmin_VhVh(vmin, Q6_Vh_vmin_VhVh(vt0, vt1));
			vmax = Q6_Vh_vmax_VhVh(vmax, Q6_Vh_vmax_VhVh(vt0, vt1));
			// interleave ~min amd max; reduce to one vector and apply to global.
			HVX_VectorPair vmmshuf = Q6_Wh_vshuffoe_VhVh(vmax, Q6_V_vnot_V(vmin));
			vminmax_all = Q6_Vh_vmax_VhVh(vminmax_all, Q6_Vh_vmax_VhVh(Q6_V_lo_W(vmmshuf), Q6_V_hi_W(vmmshuf)));
		}
	}
	if (need_minmax)
	{
		// horizontal reduce and store.
		// first two 16-bit outputs will be: ~min, and max
		for (int i = 0; i < 5; i++)
		{
			HVX_VectorPair vmmshuf = Q6_W_vshuff_VVR(vminmax_all, vminmax_all, -4);
			vminmax_all = Q6_Vh_vmax_VhVh(Q6_V_lo_W(vmmshuf), Q6_V_hi_W(vmmshuf));
		}
		*(HVX_Vector *)minmax_buf = vminmax_all;
	}
}

static inline void __attribute__((always_inline))
add_core_hvx_underrange(struct addsub_flat_direct_runstate *rstp, HVX_Vector const *ptrA, HVX_Vector const *ptrB, HVX_Vector *ptrout, int n_elements, int nvecs, int16_t *minmax_buf)
{
	int scale_a = rstp->osp.a_scale;
	int scale_b = rstp->osp.b_scale;
	int offset = rstp->osp.offset;
	int initial_rsh = 8;
	int final_rsh = rstp->osp.final_rsh;
	if (rstp->osp.underrange)
	{
		initial_rsh += final_rsh - 1;
		final_rsh = 1;
	}
	scale_a = Q6_R_combine_RlRl(scale_a, scale_a);
	scale_b = Q6_R_combine_RlRl(scale_b, scale_b);
	HVX_Vector_x4 p1 = first_stage_underrange(ptrA[0], ptrB[0], scale_a, scale_b, initial_rsh);
	for (int i = 0; i < nvecs - 1; i++)
	{
		*ptrout++ = second_stage_underrange(p1, offset, final_rsh);
		p1 = first_stage_underrange(ptrA[i + 1], ptrB[i + 1], scale_a, scale_b, initial_rsh);
	}
	*ptrout = second_stage_underrange(p1, offset, final_rsh);
}

static void
addsub_flat_work(struct nn_graph *nn, void *rstpv)
{
	struct addsub_flat_direct_runstate *rstp = (struct addsub_flat_direct_runstate *)rstpv;

	int thrindex = __sync_fetch_and_add(&rstp->thrindx, 1); // obtain a unique thread index

	int elements_total = rstp->elements_total;
	int elements_chunk = rstp->elements_chunk;

	int curposn;
	uint8_t const *ptrA0 = rstp->inA;
	uint8_t const *ptrB0 = rstp->inB;
	uint8_t *ptrout0 = rstp->outp;
	int16_t *minmax_buf = rstp->minmax_buf + 64 * thrindex;
	HVX_Vector * min_max_vec = (HVX_Vector *)minmax_buf;
	*min_max_vec = Q6_V_vsplat_R(0x80008000);
	core_oper_fp funcp = add_core_hvx;
	if (rstp->osp.underrange)
		funcp = add_core_hvx_underrange;

	while (curposn = __sync_fetch_and_add(&rstp->next_pos, elements_chunk), curposn < elements_total)
	{
		int n_elements = min_i32(elements_chunk, elements_total - curposn);
		int nvecs = (n_elements + 127) / 128u;
		HVX_Vector const *ptrA = (HVX_Vector const *)(ptrA0 + curposn);
		HVX_Vector const *ptrB = (HVX_Vector const *)(ptrB0 + curposn);

		int delt = 0;
		if (nvecs > 16)
		{ // fetch first 16 vecs from both A & B, then do the rest.
			l2fetch(ptrA, 128, 128, 16);
			l2fetch(ptrB, 128, 128, 16);
			delt = 16;
			wait_for_l2fetch();
		}
		l2fetch(ptrA + delt, 128, 128, nvecs - delt);
		l2fetch(ptrB + delt, 128, 128, nvecs - delt);

		HVX_Vector *ptrout = (HVX_Vector *)(ptrout0 + curposn);
		(*funcp)(rstp, ptrA, ptrB, ptrout, n_elements, nvecs, minmax_buf);
		//Figure out if we need underrange here, call one function or the other
	}
	nn_sem_post(&rstp->done_sem);
}

// The add/sub operation is done as:
//   (1) multiply a[i]*256 by ascl, and b[i]*256 by bscl, add products together in 32 bits; result is  >= 0, <2^31
//   (2) >> 16 and treat as  i16  ( >= 0, <= 2^15)
//   (3) add a 16-bit offset using saturated add
//   (4) >>final_rsh and saturate to u8.
// ascl and bscl are the two output scales, relative to a common exponent (defined by final_rsh).
// We have a constraint that each of ascl,bscl must be <= 16383, which means the sum in (1) is < 2^31).
// Also, for 'add', we need their sum to be <= 16383, due to how the hvx calculation works.
// 'offset' is chosen so that 0+0 = 0
//  I.e. after working out the scaling, 'offset' is the amount which gives for 'zero' inputs, a value at (3) of
//    output_zero << final_rsh.
// If the calculated offset falls outside the i16 range, it can be saturated; this situation means that all
// possible outputs shoulde be 0 or 0xff, based on the ranges.
//
// for adaptive ranging, we keep track (at step (2) of the min and max values, which can be translated to application-level
// floats.
//
// To do subtraction with the same datapath:
//   - ascl can range over  0..32767
//   - bscl is a negative number  -32767..0
//   (this will only work if the 'b' mul can be s16 x u16)
//
//
// There is a limitation:
//    a_scale_f and b_scale_f must both <= 63.984, otherwise the method is infeasible
//  (the 'rsh' would need to be < 0).
// In terms of ranges, this means that the output range must be at least 1/63.98. of the largest
// of the input ranges; if we make it at least 1/63.98 of the *sum* of the input ranges (which
// is the actual worst-case output range) then that's sufficient. In order to make it less likely
// to need re-running, we use 1/8 of the hypothetical input range when there's no other information.
//
//

//Underrange core

//This is the normal core

static inline int
set_addsub_scaling_for_output_range(struct qadd_888_info *info, struct addsub_scaling_parms *osp, int fixed_range)
{
	float out_min = info->out_min;
	float out_max = info->out_max;

	float out_scl = 255.0f / (out_max - out_min); // out scale
	float out_z = -out_min * out_scl;			  // the 'zero point'

	//printf("output range = %.6f..%.6f; z = %.6f\n", out_min, out_max, out_z);

	float a_scale_f = out_scl * osp->ab_stepsize[0];
	float b_scale_f = out_scl * osp->ab_stepsize[1];
	float scaleval = a_scale_f + b_scale_f;

	if (osp->operator!= op_add)
	{
		scaleval = fmaxf(a_scale_f, b_scale_f);
		b_scale_f = -b_scale_f;
	}
	//printf("float scales: %.7f, %.7f\n", a_scale_f, b_scale_f);

	// find exponent. The scale factor here is so that neither of
	// a_scale, b_scale will exceed 16384. (the value is 16384./16380.)
	// scexp should be >= -1 generally; if there is range expansion, it could
	// be less, and we limit rsh to <= 7 which will force smaller a_scale,b_scale.
	//
	// For subtraction, a_scale and b_scale have opposite signs; and both
	// must have abs value < 16384.
	// A value of scexp > 6 is infeasible; this would require rsh < 0
	//
	int scexp = flt_getexp(scaleval * (float)(16384. / 16380.));
	if (scexp > 6 && (!fixed_range || scexp > 12)) // req. scale is too large
		return -1;
	int rsh = min_i32(6 - scexp, 7); // should be 0..6 (maybe 7 sometimes)

	// important: rsh=0 is 'underrange' when fixed_range, and not otherwise (this lets us get
	// a rounding up in, using an actual final_rsh of 1, when fixed_range)
	int preshift = 0;
	if (fixed_range && rsh <= 0)
	{
		preshift = 1 - rsh;
	}
	int is_underrange = preshift != 0;
	osp->underrange = is_underrange;

	// determine the quantized scale factors
	float rsh_scl = flt_ldexp(1.0f,rsh);
	int a_scale = roundf_i32(a_scale_f * (rsh_scl * 256.0f));
	int b_scale = roundf_i32(b_scale_f * (rsh_scl * 256.0f));
	// work out the zero point
	//
	float rsh_scl2 = is_underrange ? 2.0f : rsh_scl; // rsh_scl *2^preshift

	int intzero = (osp->ab_zero[0] * a_scale + osp->ab_zero[1] * b_scale) >> (8 - preshift); // step 2 result for 'zero' input
	int offset = roundf_i32(out_z * rsh_scl2) - intzero;

	/*printf("scale = %d, %d; offset = %d; final_rsh = %d; int_zero = %d\n",
	a_scale, b_scale, offset, rsh, intzero );*/

	osp->final_rsh = rsh;
	osp->a_scale = a_scale;
	osp->b_scale = b_scale;
	osp->intermed_zero = intzero;
	osp->offset = is_underrange? offset: saturate_i16(offset);
	osp->netscale = 1.0f / (out_scl * rsh_scl);

	// special packing for hvx code
	// scales_lo has the 7 lsbs of each scale (zero extended)
	// scales_hi has bits [14:7] of each scale
	osp->scales_lo = (a_scale & 0x7f) | ((b_scale & 0x7f) << 8);
	osp->scales_hi = ((a_scale >> 7) & 0xFF) | ((b_scale << 1) & 0xFF00);
	// --> underrange is only supported when output range is fixed; so the below params
	// --> are not meaningful in underrange mode.
	//
	// find intermed_min, intermed_max ; range of values at (2) which won't
	// clip  at the output.
	// note that these may fall outside the range (0..32767) of (2) results,
	// which just means that clipping can't occur at that endpoint in the particular setup.
	// So it's important to store these as i32, they may not fit in i16.
	//
	int rbias = is_underrange?0: ((1<<rsh)>>1);
	// result from (3) must be at least -rbias to not clip as < 0
	osp->intermed_min = -(offset + rbias);
	// result from (3) must be at most (256<<rsh)-(rbias+1) to not clip as > 255
	osp->intermed_max = osp->intermed_min + (1 << (rsh + 8)) - 1;
	return 0;
}

static int qadd_888_check(struct nn_node *self, struct nn_graph *nn)
{
	struct qadd_888_info *info;
	int n_in = self->n_inputs;

	if ((info = nn_calloc(1, sizeof(struct qadd_888_info))) == NULL)
	{
		return errlog(nn, "calloc");
	}
	self->opaque = info;
	float out_min_float = -INFINITY;
	float out_max_float = INFINITY;
	if (n_in >= 7)
	{
		const struct tensor *out_min_tensor = self->inputs[6];
		out_min_float = tensor_get_float(out_min_tensor, 0);
		if (n_in >= 8)
		{
			const struct tensor *out_max_tensor = self->inputs[7];
			out_max_float = tensor_get_float(out_max_tensor, 0);
		}
	}
	uint32_t out_size = self->output_defs[0].elementsize;

	out_size *= self->output_defs[0].max_sizes[0];
	out_size *= self->output_defs[0].max_sizes[1];
	out_size *= self->output_defs[0].max_sizes[2];
	out_size *= self->output_defs[0].max_sizes[3];

	// round up
	out_size = (out_size + 127) & (~127);

	// ensure enough scratch
	// (the worst case is 256 for min/max, plus room for broadcast-expanded
	// 'A' and broadcast-expanded 'B').
	nn_scratch_grow(nn, out_size * 2 + 256);

	if (out_min_float == -INFINITY)
	{
		info->min_precalculated = 0;
		info->out_min = 0.0f;
	}
	else
	{
		info->min_precalculated = 1;
		info->out_min = out_min_float;
	}
	if (out_max_float == INFINITY)
	{
		info->max_precalculated = 0;
		info->out_max = 0.5f;
	}
	else
	{
		info->max_precalculated = 1;
		info->out_max = out_max_float;
		if (info->min_precalculated)
		{
			adjust_minmax_for_zero_with_constraints(&info->out_min, &info->out_max, 1 | 2);
			info->has_run_before = 1; // don't need any range calc.
		}
	}
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to8 = {
	.execute = qadd_888_execute,
	.check = qadd_888_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(6, 8),
	.n_outputs = NN_IOCOUNT(3),
};

static int qsub_reject(struct nn_node *self, struct nn_graph *nn)
{
	return errlog(nn, "non d32 sub not supported");
}
struct nn_node_ops nn_ops_for_QuantizedSub_8p8to8 = {
	.execute = qadd_888_execute,
	.check = qsub_reject,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(6, 8),
	.n_outputs = NN_IOCOUNT(3),
};
