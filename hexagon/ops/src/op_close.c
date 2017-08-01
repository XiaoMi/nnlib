
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
#include <math.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for a node that can check that 
 * input 0 is close to input 1
 */

#define CHECK(THING) \
	if (dut->THING != ref->THING) return errlog(nn,"check fail " #THING " (%d vs %d)",dut->THING,ref->THING);


#define FUDGE_FACTOR 0.05

static inline int close_execute(struct nn_node *self, struct nn_graph *nn, 
	int (*cmp)(struct nn_graph *nn, void *a, void *b, uint32_t size))
{
	const struct tensor *dut = self->inputs[0];
	const struct tensor *ref = self->inputs[1];
	logmsg(nn,2,"close execute. self=%p ",self);
	/* Copy input tensor to output */
	CHECK(shape.batches);
	CHECK(shape.height);
	CHECK(shape.width);
	CHECK(shape.depth);
	CHECK(data_size);
	if (cmp(nn,dut->data,ref->data,ref->data_size) != 0) {
		return errlog(nn,"data mismatch");
	}
	logmsg(nn,2,"close node %p OK",self);
	return 0;
}

static int close_execute_f(struct nn_node *self, struct nn_graph *nn)
{

	const struct tensor *dut = self->inputs[0];
	const struct tensor *ref = self->inputs[1];
	const struct tensor *eps;
	float *a = (float *)dut->data;
	float *b = (float *)ref->data;
	float fudge;
	int count = ref->data_size/sizeof(float);
	int i;
	float ref_max = 0.0f;
	float ref_min = 0.0f;
	float range;
	float epsilon;
	float cur_diff;
	float max_diff = 0.0f;
	int max_diff_idx = 0;
	if (self->n_inputs > 2) {
		eps = self->inputs[2];
		fudge = tensor_get_float(eps,0);
	} else {
		fudge = FUDGE_FACTOR;
	}
	logmsg(nn,2,"close check_fvals execute. self=%p ",self);

	/* Copy input tensor to output */
	CHECK(shape.batches);
	CHECK(shape.height);
	CHECK(shape.width);
	CHECK(shape.depth);
	CHECK(data_size);

	for (i = 0; i < count; i++) {
		ref_max = fmaxf(b[i],ref_max);
		ref_min = fminf(b[i],ref_min);
		cur_diff = fabsf(a[i] - b[i]);
		if (cur_diff > max_diff) {
			max_diff = cur_diff;
			max_diff_idx = i;
		}

	}
	range = ref_max - ref_min;
	epsilon = range*fudge;

	if (max_diff > epsilon) {
		i = max_diff_idx;
		errlog(nn,"data not close. Worst offender: i: %d/%d a[i]: %a %f b[i]: %a %f max_diff=%f range=%f epsilon=%f",
			max_diff_idx,count,a[i],a[i],b[i],b[i],
			max_diff,range,epsilon);

		errlog(nn,"\t\tActual\t\tExpected\tDiff");
		for (i = 0; i < count; i++) {
			if (i == max_diff_idx)
				errlog(nn,"%d)\t%f\t%f\t%f <====",i,a[i],b[i],a[i]-b[i]);
			else
				errlog(nn,"%d)\t%f\t%f\t%f",i,a[i],b[i],a[i]-b[i]);
		}
		return 1;
	}
	logmsg(nn,2,"close node %p OK",self);
	return 0;
}

static inline int check_i32vals(struct nn_graph *nn, void *av, void *bv, uint32_t size)
{
	int32_t *a = (int32_t *)av;
	int32_t *b = (int32_t *)bv;
	int count = size/sizeof(int32_t);
	int i;
	uint32_t max = 0;
	int32_t tmp;
	for (i = 0; i < count; i++) {
		tmp = b[i];
		if (tmp < 0) tmp = -tmp;
		if (max < tmp) max = tmp;
	}
	for (i = 0; i < count; i++) {
		if (fabsf(((float)(a[i]) - (float)(b[i])) / (float)(max)) > FUDGE_FACTOR) {
			logmsg(nn,0,"i: %d/%d a[i]: %08x b[i]: %08x",i,count,a[i],b[i]);
			return 1;
		}
	}
	return 0;
}

static inline int check_u8vals(struct nn_graph *nn, void *av, void *bv, uint32_t size)
{
	uint8_t *a = (uint8_t *)av;
	uint8_t *b = (uint8_t *)bv;
	uint32_t max = 0;
	int count = size/sizeof(uint8_t);
	int i;
	for (i = 0; i < count; i++) {
		if (max < b[i]) max = b[i];
	}
	for (i = 0; i < count; i++) {
		if (fabsf(((float)(a[i]) - (float)(b[i])) / (float)(max)) > FUDGE_FACTOR) {
			logmsg(nn,0,"i: %d/%d a[i]: %08x b[i]: %08x",i,count,a[i],b[i]);
			return 1;
		}
	}
	return 0;
}

static inline int __attribute__((unused)) check_novals(struct nn_graph *nn, void *av, void *bv, uint32_t size) 
{
	return 0;
}

static int close_execute_i32(struct nn_node *self, struct nn_graph *nn)
{
	return close_execute(self,nn,check_i32vals);
	//return close_execute(self,nn,check_novals);
}

static int close_execute_u8(struct nn_node *self, struct nn_graph *nn)
{
	return close_execute(self,nn,check_u8vals);
	//return close_execute(self,nn,check_novals);
}

static int close_execute_q_u8(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *dut = self->inputs[0];
	const struct tensor *dut_min = self->inputs[1];
	const struct tensor *dut_max = self->inputs[2];
	const struct tensor *ref = self->inputs[3];
	const struct tensor *ref_min = self->inputs[4];
	const struct tensor *ref_max = self->inputs[5];
	float dut_min_float = tensor_get_float(dut_min,0);
	float dut_max_float = tensor_get_float(dut_max,0);
	float ref_min_float = tensor_get_float(ref_min,0);
	float ref_max_float = tensor_get_float(ref_max,0);
	float dut_range = (dut_max_float - dut_min_float);
	float ref_range = (ref_max_float - ref_min_float);
	float dut_stepsize = dut_range / 255.0f;
	float ref_stepsize = ref_range / 255.0f;
	const uint8_t *dutdata = (const uint8_t *)dut->data;
	const uint8_t *refdata = (const uint8_t *)ref->data;
	float dutval,refval;
	int count = ref->data_size / sizeof(uint8_t);
	int i;
	int err = 0;

	logmsg(nn,2,"close q execute. self=%p ",self);
	CHECK(shape.batches);
	CHECK(shape.height);
	CHECK(shape.width);
	CHECK(shape.depth);
	CHECK(data_size);

	for (i = 0; i < count; i++) {
		dutval = dutdata[i] * dut_stepsize + dut_min_float;
		refval = refdata[i] * ref_stepsize + ref_min_float;
		if (fabsf((dutval-refval)/ref_range) > FUDGE_FACTOR) {
			logmsg(nn,1,"dut min/max = %f/%f ref min/max = %f/%f dut/ref q=%02x/%08x",
				dut_min_float,
				dut_max_float,
				ref_min_float,
				ref_max_float,
				dutdata[i],
				refdata[i]);
			logmsg(nn,1,"i: %d/%d dut: %f ref %f",i,count,dutval,refval);
			err++;
		}
	}
	if (err) return errlog(nn,"data mismatch");
	logmsg(nn,2,"close q node %p OK",self);
	return 0;
}

static int close_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking close node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"check: wrong # inputs");
	if (self->n_outputs != 0) return errlog(nn,"check: wrong # outputs");
	logmsg(nn,2,"close node %p check OK",self);
	return 0;
}

static int close_check_f(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking close node %p",self);
	if (self->n_inputs < 2) return errlog(nn,"check: wrong # inputs");
	if (self->n_inputs > 3) return errlog(nn,"check: wrong # inputs");
	if (self->n_outputs != 0) return errlog(nn,"check: wrong # outputs");
	logmsg(nn,2,"close node %p check OK",self);
	return 0;
}

static int close_check_q(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking close q node %p",self);
	if (self->n_inputs != 6) return errlog(nn,"check: wrong # inputs");
	if (self->n_outputs != 0) return errlog(nn,"check: wrong # outputs");
	logmsg(nn,2,"close q node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Close_f = {
	SFINIT(.execute, close_execute_f),
	SFINIT(  .check, close_check_f),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Close_int32 = {
	SFINIT(.execute, close_execute_i32),
	SFINIT(  .check, close_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Close_qint32 = {
	SFINIT(.execute, close_execute_i32),
	SFINIT(  .check, close_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Close_quint8 = {
	SFINIT(.execute, close_execute_u8),
	SFINIT(  .check, close_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_Close_q_quint8 = {
	SFINIT(.execute, close_execute_q_u8),
	SFINIT(  .check, close_check_q),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

