
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
#include <math.h>
#include "quantize.h"
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for a node that can check that 
 * input 0 is close to input 1
 */

#define CHECK(THING) \
	if (dut->THING != ref->THING) return errlog(nn,"check fail " #THING " (%d vs %d)",dut->THING,ref->THING);

#define CHECK_LE(THING) \
	if (dut->THING > ref->THING) return errlog(nn,"check fail " #THING " (%d > %d)",dut->THING,ref->THING);

#define CHECK_LE_WITH_OFFSET(THING) \
	if (offset+dut->THING > ref->THING) return errlog(nn,"check fail " #THING " (%d > %d)",offset+dut->THING,ref->THING);

#define FUDGE_FACTOR 0.07
#define MAX_TO_SHOW 2000

static inline int close_execute(struct nn_node *self, struct nn_graph *nn,
	int (*cmp)(struct nn_graph *nn, void *a, void *b, uint32_t size))
{
#ifdef TIMING_MODE
	return 0;
#endif
	if( nn_option_get(nn,debug_skip_check))  return 0;

	const struct tensor *dut = self->inputs[0];
	const struct tensor *ref = self->inputs[1];
	uint32_t offset = 0;
	int32_t offset_index;
	logmsg(nn,2,"close execute. self=%p ",self);
	/* Copy input tensor to output */
	if(nn->loopstack.n == 0){
		CHECK(shape.batches);
		CHECK(shape.height);
		CHECK(shape.width);
		CHECK(shape.depth);
		CHECK(data_size);
	}
	else{
		offset_index = tensor_get_int32(self->inputs[2],0);
		offset = nn_loopstack_get_offset(nn, offset_index);
		CHECK_LE(shape.batches);
		CHECK_LE(shape.height);
		CHECK_LE(shape.width);
		CHECK_LE(shape.depth);
		CHECK_LE_WITH_OFFSET(data_size);
	}

	uint8_t* ref_data = ref->data;

	if (cmp(nn,dut->data,ref_data+offset,dut->data_size) != 0) {
		return errlog(nn,"data mismatch");
	}
	if(nn->loopstack.n > 0)
	{
		nn_loopstack_increment_offset(nn, offset_index, dut->data_size);
	}
	logmsg(nn,2,"close node %p OK",self);
	return 0;
}

static int close_execute_f(struct nn_node *self, struct nn_graph *nn)
{
	if( nn_option_get(nn,debug_skip_check))  return 0;

	const struct tensor *dut = self->inputs[0];
	const struct tensor *ref = self->inputs[1];
	const struct tensor *eps;
	float *a = dut->data;
	float *b = ref->data;
	float fudge;
	int count = ref->data_size/sizeof(float);
	int i;
	float ref_max = 0.0f;
	float ref_min = 0.0f;
	float mean_err = 0.0f;
	float prev_mean_err = 0.0f;
	float S_err = 0.0f;
	float err = 0.0f;
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

	// find min/max deviant elements,
	// and compute mean and standard-deviation of error
	int any_NAN = 0;
	prev_mean_err = a[0]-b[0];
	for (i = 0; i < count; i++) {
		ref_max = fmaxf(b[i],ref_max);
		ref_min = fminf(b[i],ref_min);
		cur_diff = fabsf(a[i] - b[i]);
		if (!(cur_diff <= max_diff)) {
			if( flt_isnan(cur_diff)){
				if( any_NAN < 20)
					logmsg(nn,0,"NaN found at location %d", i);
				any_NAN++;
			}else{
				max_diff = cur_diff;
				max_diff_idx = i;
			}
		}
		err = a[i]-b[i];
		prev_mean_err = mean_err;
		mean_err = mean_err + (err - mean_err) / (i+1);
		S_err = S_err + (err - mean_err) * (err - prev_mean_err);
	}
	range = ref_max - ref_min;
	epsilon = range*fudge;
	if (epsilon < FUDGE_FACTOR) epsilon = FUDGE_FACTOR;
	if( any_NAN){
		errlog(nn,"%d NAN values found!!", any_NAN);
	}
	if (max_diff > epsilon) {
		i = max_diff_idx;
		errlog(nn,"data stats: mean_error=%f sd_error=%f", mean_err, sqrtf(S_err/count));
		errlog(nn,"data not close. Worst offender: i: %d/%d a[i]: %a %f b[i]: %a %f max_diff=%f range=%f epsilon=%f shape=%d,%d,%d,%d",
			max_diff_idx,count,a[i],a[i],b[i],b[i],
			max_diff,range,epsilon,
			ref->shape.batches,ref->shape.height,ref->shape.width,ref->shape.depth);

		errlog(nn,"\t\tActual\t\tExpected\tDiff");
		for (i = 0; i < count; i++) {
			int h,w,d;
			d = i % ref->shape.depth;
			w = (i / (ref->shape.depth)) % ref->shape.width;
			h = (i / (ref->shape.depth*ref->shape.width)) % ref->shape.height;
			if (i == max_diff_idx) {
				logmsg(nn,0,"%d[%d,%d,%d])\t%f\t%f\t%f <====",i,h,w,d,a[i],b[i],a[i]-b[i]);
			} else {
				logmsg(nn,0,"%d[%d,%d,%d])\t%f\t%f\t%f",i,h,w,d,a[i],b[i],a[i]-b[i]);
			}
			if (i >= MAX_TO_SHOW) {
				logmsg(nn,0,"Stopping at %d elements");
				return 1;
			}
		}
		return 1;
	}
	logmsg(nn,2,"close node %p OK",self);
	return any_NAN!=0;
}

static inline int check_i32vals(struct nn_graph *nn, void *av, void *bv, uint32_t size)
{
	int32_t *a = av;
	int32_t *b = bv;
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
	uint8_t *a = av;
	uint8_t *b = bv;
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

static inline int check_mse_u8(struct nn_graph *nn, void *av, void *bv, uint32_t size)
{
    uint8_t *a = av;
    uint8_t *b = bv;
    int count = size/sizeof(uint8_t);
    int i;
    uint64_t se = 0;
    for (i = 0; i < count; i++) {
        int d = a[i]-b[i];
        se += d*d;
    }
    float mse = (float)se/(float)count;
    errlog(nn, "mse %f", mse);
    if (mse > 0.5) {
        logmsg(nn,0,"mse %f, expected %f", mse, 0.5);
        return 1;
    }
    return 0;
}

static inline int check_u16vals(struct nn_graph *nn, void *av, void *bv, uint32_t size)
{
	uint16_t *a = av;
	uint16_t *b = bv;
	uint32_t max = 0;
	int count = size/sizeof(uint16_t);
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

static inline int is_same_box(uint16_t *a, uint16_t *b){
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

static inline int check_u16boxes(struct nn_graph *nn, void *av, void *bv, uint32_t size)
{
	uint16_t *a = av;
	uint16_t *b = bv;
	int count = size/sizeof(uint16_t);
	int total = count / 4;
	int matches = 0;
	int i, j;
	for (i = 0; i < total; i++) {
		for(j = 0; j < total; j++){
			if(is_same_box(a+i*4, b+j*4)){
				matches++;
				break;
			}
		}
	}

	int different = (total - matches) / 2;

	if((float)different / (float)total > FUDGE_FACTOR){
		
		logmsg(nn,0,"%d boxes out of %d are different", different, total);
		return 1;
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

static int close_execute_u16(struct nn_node *self, struct nn_graph *nn)
{
	return close_execute(self,nn,check_u16vals);
}

static int close_execute_boxes_u16(struct nn_node *self, struct nn_graph *nn)
{
	return close_execute(self,nn,check_u16boxes);
}

static int close_execute_q_u8(struct nn_node *self, struct nn_graph *nn)
{
#ifdef TIMING_MODE
	return 0;
#endif
	if( nn_option_get(nn,debug_skip_check))  return 0;

	const struct tensor *dut = self->inputs[0];
	const struct tensor *dut_min = self->inputs[1];
	const struct tensor *dut_max = self->inputs[2];
	const struct tensor *ref = self->inputs[3];
	const struct tensor *ref_min = self->inputs[4];
	const struct tensor *ref_max = self->inputs[5];
	const struct tensor *error_ratio = (self->n_inputs == 7 && nn->loopstack.n == 0) || (self->n_inputs == 8 && nn->loopstack.n > 0)? self->inputs[self->n_inputs-1] : NULL;
	float dut_min_float = tensor_get_float(dut_min,0);
	float dut_max_float = tensor_get_float(dut_max,0);
	float ref_min_float = tensor_get_float(ref_min,0);
	float ref_max_float = tensor_get_float(ref_max,0);
	float error_ratio_float = (NULL != error_ratio) ? tensor_get_float(error_ratio,0) : FUDGE_FACTOR;
	float dut_range = (dut_max_float - dut_min_float);
	float ref_range = (ref_max_float - ref_min_float);
	float dut_stepsize = dut_range / 255.0f;
	float ref_stepsize = ref_range / 255.0f;
	float max_error_ratio = 0.0;
	float curr_error_ratio = 0.0;
	const uint8_t *dutdata = dut->data;
	const uint8_t *refdata = ref->data;
	float dutval,refval;
	int count = dut->data_size / sizeof(uint8_t);
	int i;
	int max_error_idx = 0;
	uint32_t offset = 0;
	int32_t offset_index;
	logmsg(nn,2,"close q execute. self=%p ",self);
	if(nn->loopstack.n == 0){
		CHECK(shape.batches);
		CHECK(shape.height);
		CHECK(shape.width);
		CHECK(shape.depth);
		CHECK(data_size);
	}
	else{
		offset_index = tensor_get_int32(self->inputs[self->n_inputs-2],0);
		offset = nn_loopstack_get_offset(nn, offset_index);
		CHECK_LE(shape.batches);
		CHECK_LE(shape.height);
		CHECK_LE(shape.width);
		CHECK_LE(shape.depth);
		CHECK_LE_WITH_OFFSET(data_size);
	}

	logmsg(nn,2,"Closeness checking... dut min/max: %f/%f ref min/max: %f/%f",
		dut_min_float,dut_max_float,ref_min_float,ref_max_float);

	for (i = 0; i < count; i++) {
		dutval = dutdata[i] * dut_stepsize + dut_min_float;
		refval = refdata[i+offset] * ref_stepsize + ref_min_float;
		curr_error_ratio = fabsf((dutval-refval)/ref_range);
		if (curr_error_ratio > error_ratio_float) {
			logmsg(nn,2,"%d, h/w/d=%d/%d/%d dut=%f ref=%f",
				i,
				i/(dut->shape.depth*dut->shape.width),
				(i/dut->shape.depth)%dut->shape.width,
				i%(dut->shape.depth),
				dutval,
				refval);
		} else if (max_error_ratio > error_ratio_float) {
			logmsg(nn,9,"%d, h/w/d=%d/%d/%d dut=%f ref=%f",
				i,
				i/(dut->shape.depth*dut->shape.width),
				(i/dut->shape.depth)%dut->shape.width,
				i%(dut->shape.depth),
				dutval,
				refval);
		}
		if (curr_error_ratio > max_error_ratio) {
			max_error_ratio = curr_error_ratio;
			max_error_idx = i;
		}

	}
	if (max_error_ratio > error_ratio_float) {
		errlog(nn, "max error ratio / test error ratio = %f/%f, max error index %d count %d",
			max_error_ratio,
			error_ratio_float,
			max_error_idx,
			count);
		errlog(nn, "dut min/max = %f/%f ref min/max = %f/%f",
			dut_min_float,
			dut_max_float,
			ref_min_float,
			ref_max_float);
		errlog(nn, "dut q/f = %d/%f ref q/f = %d/%f",
			dutdata[max_error_idx],
			dutdata[max_error_idx] * dut_stepsize + dut_min_float,
			refdata[max_error_idx],
			refdata[max_error_idx] * ref_stepsize + ref_min_float);
		return errlog(nn,"data mismatch");
	}
	
	if(nn->loopstack.n > 0)
	{
		nn_loopstack_increment_offset(nn, offset_index, count);
	}
	
	logmsg(nn,2,"close qu8 node %p OK",self);
	return 0;
}

static int close_execute_q_i16(struct nn_node *self, struct nn_graph *nn)
{
#ifdef TIMING_MODE
	return 0;
#endif
	const struct tensor *dut = self->inputs[0];
	const struct tensor *dut_min = self->inputs[1];
	const struct tensor *dut_max = self->inputs[2];
	const struct tensor *ref = self->inputs[3];
	const struct tensor *ref_min = self->inputs[4];
	const struct tensor *ref_max = self->inputs[5];
	const struct tensor *error_ratio = (self->n_inputs == 7 && nn->loopstack.n == 0) || (self->n_inputs == 8 && nn->loopstack.n > 0)? self->inputs[self->n_inputs-1] : NULL;
	float dut_min_float = tensor_get_float(dut_min,0);
	float dut_max_float = tensor_get_float(dut_max,0);
	float ref_min_float = tensor_get_float(ref_min,0);
	float ref_max_float = tensor_get_float(ref_max,0);
	float error_ratio_float = (NULL != error_ratio) ? tensor_get_float(error_ratio,0) : FUDGE_FACTOR;
	float dut_range = (dut_max_float - dut_min_float);
	float ref_range = (ref_max_float - ref_min_float);
	float dut_stepsize = dut_range /  65535.0f;
	float ref_stepsize = ref_range /  65535.0f;
	float max_error_ratio = 0.0;
	float curr_error_ratio = 0.0;
	const int16_t *dutdata = dut->data;
	const int16_t *refdata = ref->data;
	float dutval,refval;
	int count = dut->data_size / sizeof(int16_t);
	int i;
	int max_error_idx = 0;
	uint32_t offset = 0;
	int32_t offset_index;

	logmsg(nn,2,"close q execute. self=%p ",self);
	if(nn->loopstack.n == 0){
		CHECK(shape.batches);
		CHECK(shape.height);
		CHECK(shape.width);
		CHECK(shape.depth);
		CHECK(data_size);
	}
	else{
		offset_index = tensor_get_int32(self->inputs[self->n_inputs-2],0);
		offset = nn_loopstack_get_offset(nn, offset_index);
		CHECK_LE(shape.batches);
		CHECK_LE(shape.height);
		CHECK_LE(shape.width);
		CHECK_LE(shape.depth);
		CHECK_LE_WITH_OFFSET(data_size);
	}

	logmsg(nn,2,"Closeness checking... dut min/max: %f/%f ref min/max: %f/%f",
		dut_min_float,dut_max_float,ref_min_float,ref_max_float);
	for (i = 0; i < count; i++) {
		dutval = dutdata[i] * dut_stepsize ;
		refval = refdata[i+offset] * ref_stepsize;
		curr_error_ratio = fabsf((dutval-refval)/ref_range);

		if (curr_error_ratio > error_ratio_float) {
			logmsg(nn,2,"%d, h/w/d=%d/%d/%d dut=%f ref=%f",
				i,
				i/(dut->shape.depth*dut->shape.width),
				(i/dut->shape.depth)%dut->shape.width,
				i%(dut->shape.depth),
				dutval,
				refval);
		} else if (max_error_ratio > error_ratio_float) {
			logmsg(nn,9,"%d, h/w/d=%d/%d/%d dut=%f ref=%f",
				i,
				i/(dut->shape.depth*dut->shape.width),
				(i/dut->shape.depth)%dut->shape.width,
				i%(dut->shape.depth),
				dutval,
				refval);
		}
		if (curr_error_ratio > max_error_ratio) {
			max_error_ratio = curr_error_ratio;
			max_error_idx = i;
		}

	}
	if (max_error_ratio > error_ratio_float) {
		errlog(nn, "max error ratio / test error ratio = %f/%f, max error index %d count %d",
			max_error_ratio,
			error_ratio_float,
			max_error_idx,
			count);
		errlog(nn, "dut min/max = %f/%f ref min/max = %f/%f",
			dut_min_float,
			dut_max_float,
			ref_min_float,
			ref_max_float);
		errlog(nn, "dut q/f = %d/%f ref q/f = %d/%f",
			dutdata[max_error_idx],
			dutdata[max_error_idx] * dut_stepsize + dut_min_float,
			refdata[max_error_idx],
			refdata[max_error_idx] * ref_stepsize + ref_min_float);
		return errlog(nn,"data mismatch");
	}
	
	if(nn->loopstack.n > 0)
	{
		nn_loopstack_increment_offset(nn, offset_index, count);
	}
	
	logmsg(nn,2,"close qi16 node %p OK",self);
	return 0;
}

static int close_execute_q_u16(struct nn_node *self, struct nn_graph *nn)
{
#ifdef TIMING_MODE
	return 0;
#endif
	const struct tensor *dut = self->inputs[0];
	const struct tensor *dut_min = self->inputs[1];
	const struct tensor *dut_max = self->inputs[2];
	const struct tensor *ref = self->inputs[3];
	const struct tensor *ref_min = self->inputs[4];
	const struct tensor *ref_max = self->inputs[5];
	const struct tensor *error_ratio = (self->n_inputs == 7 && nn->loopstack.n == 0) || (self->n_inputs == 8 && nn->loopstack.n > 0)? self->inputs[self->n_inputs-1] : NULL;
	float dut_min_float = tensor_get_float(dut_min,0);
	float dut_max_float = tensor_get_float(dut_max,0);
	float ref_min_float = tensor_get_float(ref_min,0);
	float ref_max_float = tensor_get_float(ref_max,0);
	float error_ratio_float = (NULL != error_ratio) ? tensor_get_float(error_ratio,0) : FUDGE_FACTOR;
	float dut_range = (dut_max_float - dut_min_float);
	float ref_range = (ref_max_float - ref_min_float);
	float dut_stepsize = dut_range /  65535.0f;
	float ref_stepsize = ref_range /  65535.0f;
	float max_error_ratio = 0.0;
	float curr_error_ratio = 0.0;
	const uint16_t *dutdata = dut->data;
	const uint16_t *refdata = ref->data;
	float dutval,refval;
	int count = dut->data_size / sizeof(uint16_t);
	int i;
	int max_error_idx = 0;
	uint32_t offset = 0;
	int32_t offset_index;

	logmsg(nn,2,"close q execute. self=%p ",self);
	if(nn->loopstack.n == 0){
		CHECK(shape.batches);
		CHECK(shape.height);
		CHECK(shape.width);
		CHECK(shape.depth);
		CHECK(data_size);
	}
	else{
		offset_index = tensor_get_int32(self->inputs[self->n_inputs-2],0);
		offset = nn_loopstack_get_offset(nn, offset_index);
		CHECK_LE(shape.batches);
		CHECK_LE(shape.height);
		CHECK_LE(shape.width);
		CHECK_LE(shape.depth);
		CHECK_LE_WITH_OFFSET(data_size);
	}

	logmsg(nn,2,"Closeness checking... dut min/max: %f/%f ref min/max: %f/%f",
		   dut_min_float,dut_max_float,ref_min_float,ref_max_float);
	for (i = 0; i < count; i++) {
		dutval = dutdata[i] * dut_stepsize ;
		refval = refdata[i+offset] * ref_stepsize;
		curr_error_ratio = fabsf((dutval-refval)/ref_range);

		if (curr_error_ratio > error_ratio_float) {
			logmsg(nn,2,"%d, h/w/d=%d/%d/%d dut=%f ref=%f",
				   i,
				   i/(dut->shape.depth*dut->shape.width),
				   (i/dut->shape.depth)%dut->shape.width,
				   i%(dut->shape.depth),
				   dutval,
				   refval);
		} else if (max_error_ratio > error_ratio_float) {
			logmsg(nn,9,"%d, h/w/d=%d/%d/%d dut=%f ref=%f",
				   i,
				   i/(dut->shape.depth*dut->shape.width),
				   (i/dut->shape.depth)%dut->shape.width,
				   i%(dut->shape.depth),
				   dutval,
				   refval);
		}
		if (curr_error_ratio > max_error_ratio) {
			max_error_ratio = curr_error_ratio;
			max_error_idx = i;
		}

	}
	if (max_error_ratio > error_ratio_float) {
		errlog(nn, "max error ratio / test error ratio = %f/%f, max error index %d count %d",
			   max_error_ratio,
			   error_ratio_float,
			   max_error_idx,
			   count);
		errlog(nn, "dut min/max = %f/%f ref min/max = %f/%f",
			   dut_min_float,
			   dut_max_float,
			   ref_min_float,
			   ref_max_float);
		errlog(nn, "dut q/f = %d/%f ref q/f = %d/%f",
			   dutdata[max_error_idx],
			   dutdata[max_error_idx] * dut_stepsize + dut_min_float,
			   refdata[max_error_idx],
			   refdata[max_error_idx] * ref_stepsize + ref_min_float);
		return errlog(nn,"data mismatch");
	}

	if(nn->loopstack.n > 0)
	{
		nn_loopstack_increment_offset(nn, offset_index, count);
	}

	logmsg(nn,2,"close qi16 node %p OK",self);
	return 0;
}

static int close_execute_mse_u8(struct nn_node *self, struct nn_graph *nn)
{
    return close_execute(self,nn,check_mse_u8);
}


struct nn_node_ops nn_ops_for_Close_f = {
	.execute = close_execute_f,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(2,3),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_int32 = {
	.execute = close_execute_i32,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(2,3),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_qint32 = {
	.execute = close_execute_i32,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_quint8 = {
	.execute = close_execute_u8,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_quint16 = {
	.execute = close_execute_u16,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_q_quint8 = {
	.execute = close_execute_q_u8,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_q_qint16 = {
	.execute = close_execute_q_i16,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_q_quint16 = {
		.execute = close_execute_q_u16,
		.check = NULL,
		.ctor = node_alloc_common,
		.dtor = node_free_common,
		.n_inputs = NN_IOCOUNT_RANGE(6,8),
		.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_boxes_quint16 = {
	.execute = close_execute_boxes_u16,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_Close_mse_u8 = {
        .execute = close_execute_mse_u8,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(2),
        .n_outputs = NN_IOCOUNT(0),
};
