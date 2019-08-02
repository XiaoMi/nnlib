
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a reciprocal (ie. 1/x) node
 */

#include <nn_graph.h>
#include <string.h>
#include "hvx_inlines.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#include "hvx_inlines.h"
typedef long HVX_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN*)(A))
#endif
#define MIN_Q_VAL 0
#define MAX_Q_VAL 255


struct nearzero_tdata {
	struct nn_node *self;
	int whoami;
	nn_sem_t donesem;
	int start_batch;
	int end_batch;
	uint8_t * data;
	uint8_t quantized_zero;
	uint8_t * result;
};

struct minmax_tdata {
    uint8_t *in_data;
    uint8_t *out_data;
    int32_t reduction_batches;
    int32_t num_blobs;
    int32_t blob_size;
    nn_sem_t donesem;
};

struct mapdata {
    const uint8_t *in_data;
    uint8_t *map_data;
    uint8_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};


static void min_positive_execute_slice(struct nn_graph *nn, void *vinfo)
{
	struct nearzero_tdata *info = vinfo;
	uint8_t real_min = 255;
	
	for(int i = info->start_batch; i < info->end_batch; i++){
		if(info->data[i] > info->quantized_zero && info->data[i] < real_min) real_min = info->data[i];
	}
	
	*(info->result) = real_min;

	nn_sem_post(&info->donesem);
}

static void max_negative_execute_slice(struct nn_graph *nn, void *vinfo)
{
	struct nearzero_tdata *info = vinfo;
	uint8_t real_max = 0;
	
	for(int i = info->start_batch; i < info->end_batch; i++){
		if(info->data[i] < info->quantized_zero && info->data[i] > real_max) real_max = info->data[i];
	}

	*(info->result) = real_max;

	nn_sem_post(&info->donesem);
}

static HVX_Vector find_min_vec (HVX_Vector in) {
    HVX_Vector in_rot = Q6_V_vror_VR(in, 64);
    HVX_Vector cur_min = Q6_Vub_vmin_VubVub(in, in_rot);
    in_rot = Q6_V_vror_VR(cur_min, 32);
    cur_min = Q6_Vub_vmin_VubVub(cur_min, in_rot);
    in_rot = Q6_V_vror_VR(cur_min, 16);
    cur_min = Q6_Vub_vmin_VubVub(cur_min, in_rot);
    in_rot = Q6_V_vror_VR(cur_min, 8);
    cur_min = Q6_Vub_vmin_VubVub(cur_min, in_rot);
    in_rot = Q6_V_vror_VR(cur_min, 4);
    cur_min = Q6_Vub_vmin_VubVub(cur_min, in_rot);
    in_rot = Q6_V_vror_VR(cur_min, 2);
    cur_min = Q6_Vub_vmin_VubVub(cur_min, in_rot);
    in_rot = Q6_V_vror_VR(cur_min, 1);
    cur_min = Q6_Vub_vmin_VubVub(cur_min, in_rot);
    return cur_min;
}

static HVX_Vector find_max_vec (HVX_Vector in) {
    HVX_Vector in_rot = Q6_V_vror_VR(in, 64);
    HVX_Vector cur_max = Q6_Vub_vmax_VubVub(in, in_rot);
    in_rot = Q6_V_vror_VR(cur_max, 32);
    cur_max = Q6_Vub_vmax_VubVub(cur_max, in_rot);
    in_rot = Q6_V_vror_VR(cur_max, 16);
    cur_max = Q6_Vub_vmax_VubVub(cur_max, in_rot);
    in_rot = Q6_V_vror_VR(cur_max, 8);
    cur_max = Q6_Vub_vmax_VubVub(cur_max, in_rot);
    in_rot = Q6_V_vror_VR(cur_max, 4);
    cur_max = Q6_Vub_vmax_VubVub(cur_max, in_rot);
    in_rot = Q6_V_vror_VR(cur_max, 2);
    cur_max = Q6_Vub_vmax_VubVub(cur_max, in_rot);
    in_rot = Q6_V_vror_VR(cur_max, 1);
    cur_max = Q6_Vub_vmax_VubVub(cur_max, in_rot);
    return cur_max;
}

static uint8_t __attribute__((noinline,unused))
vec_extract( HVX_Vector v)
{
    union {
        HVX_Vector v;
        uint8_t as_u8[128];
    } uu = { v };
    return uu.as_u8[0];
}

static void reduce_min_all_axes_hvx(struct nn_graph *nn, void *vtd) {

    struct minmax_tdata *td = vtd;
    uint8_t *in_data = td->in_data;
    uint8_t *out_data = td->out_data;
    const int blob_size = td->blob_size;
    const int leftovers = blob_size % sizeof(HVX_Vector);
    int xd;
    HVX_Vector cur_min = q6op_Vb_vsplat_R(MAX_Q_VAL);
    for (xd = 0; xd + 128 <=blob_size; xd +=128) {
        cur_min = Q6_Vub_vmin_VubVub(*(HVX_Vector * ) & in_data[xd], cur_min);
    }
    if (leftovers) {
        HVX_VectorPred mux_pred = Q6_Q_vsetq_R(leftovers);
        HVX_Vector in = vmemu(&in_data[xd]);
        HVX_Vector last_in = Q6_V_vmux_QVV(mux_pred, in, q6op_Vb_vsplat_R(MAX_Q_VAL));
        cur_min = Q6_Vub_vmin_VubVub(last_in, cur_min);
    }
    cur_min = find_min_vec(cur_min);
    out_data[0] = vec_extract(cur_min);
    nn_sem_post(&td->donesem);
}

static void reduce_max_all_axes_hvx(struct nn_graph *nn, void *vtd) {

    struct minmax_tdata *td = vtd;
    uint8_t *in_data = td->in_data;
    uint8_t *out_data = td->out_data;
    const int blob_size = td->blob_size;
    const int leftovers = blob_size % sizeof(HVX_Vector);
    int xd;
    HVX_Vector cur_max = q6op_Vb_vsplat_R(MIN_Q_VAL);
    for (xd = 0; xd + 128 <=blob_size; xd +=128) {
        cur_max = Q6_Vub_vmax_VubVub(*(HVX_Vector * ) & in_data[xd], cur_max);
    }
    if (leftovers) {
        HVX_VectorPred mux_pred = Q6_Q_vsetq_R(leftovers);
        HVX_Vector in = vmemu(&in_data[xd]);
        HVX_Vector last_in = Q6_V_vmux_QVV(mux_pred, in, q6op_Vb_vsplat_R(MIN_Q_VAL));
        cur_max = Q6_Vub_vmax_VubVub(last_in, cur_max);
    }
    cur_max = find_max_vec(cur_max);
    out_data[0] = vec_extract(cur_max);
    nn_sem_post(&td->donesem);
}

static void get_reciprocals_continuous_range(float input_min, float input_max, uint8_t real_min, uint8_t real_max, float* output_min, float* output_max, unsigned char* output_data){
	float input_step = (input_max - input_min) / 255.f;
	
	*output_min = 2147483648.0f;
	*output_max = -2147483648.0f;
	
	float inverses[256];

	for (int i = real_min; i < real_max+1; i++) {
		float float_val = input_min + ((float)i) * input_step;
		if(float_val != 0.f){
			inverses[i] = 1/float_val;
			if(inverses[i] > *output_max) *output_max = inverses[i];
			if(inverses[i] < *output_min) *output_min = inverses[i];
		}
	}
	
	if(*output_min < 0 && *output_max < 0) *output_max = 0.f;
	if(*output_min > 0 && *output_max > 0) *output_min = 0.f;
	
	float out_step = (*output_max - *output_min) / 255.f;
	
	for (int i = real_min; i < real_max+1; i++){
		output_data[i] = round((inverses[i] - *output_min) / out_step);
	}
}

static void get_reciprocals_split_range(float input_min, float input_max, uint8_t real_min, uint8_t real_max, uint8_t skip_start, uint8_t skip_end, float* output_min, float* output_max, unsigned char* output_data){
	float input_step = (input_max - input_min) / 255.f;
	
	*output_min = 2147483648.0f;
	*output_max = -2147483648.0f;
	
	float inverses[256];

	for (int i = real_min; i < skip_start+1; i++) {
		float float_val = input_min + ((float)i) * input_step;
		if(float_val != 0.f){
			inverses[i] = 1/float_val;
			if(inverses[i] > *output_max) *output_max = inverses[i];
			if(inverses[i] < *output_min) *output_min = inverses[i];
		}
	}
	for (int i = skip_end; i < real_max+1; i++) {
		float float_val = input_min + ((float)i) * input_step;
		if(float_val != 0.f){
			inverses[i] = 1/float_val;
			if(inverses[i] > *output_max) *output_max = inverses[i];
			if(inverses[i] < *output_min) *output_min = inverses[i];
		}
	}

	if(*output_min < 0 && *output_max < 0) *output_max = 0.f;
	if(*output_min > 0 && *output_max > 0) *output_min = 0.f;
	
	float out_step = (*output_max - *output_min) / 255.f;
	
	for (int i = real_min; i < real_max+1; i++){
		output_data[i] = round((inverses[i] - *output_min) / out_step);
	}
}


static void map_values(struct nn_graph *nn, void *vtd)
{
    struct mapdata *td = vtd;
    uint8_t* in_data = (uint8_t*) td->in_data;
    uint8_t* out_data = td->out_data;
    unsigned char* map_data = td->map_data;

    const int num_loops = 1 + ((td->num_elements - 1) / 128); //ceiling

    for (int i=0; i<num_loops; i++) {
        HVX_Vector vin = *(HVX_Vector *) in_data;
        HVX_Vector *vout = (HVX_Vector *) out_data;
        // byte shuffle table
        HVX_Vector luta = *(HVX_Vector *) map_data;
        HVX_Vector lutb = *(HVX_Vector *) & map_data[128];
        HVX_Vector lut0 = Q6_Vb_vshuff_Vb(luta);
        HVX_Vector lut1 = Q6_Vb_vshuff_Vb(lutb);

        // look up value in table
        // only 32 bytes can be done at a time, so we need to do 8 lookups and OR the results

        *vout = q6op_Vb_vlut32_VbVbI(vin, lut0, 0);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut0, 1);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut0, 2);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut0, 3);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 4);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 5);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 6);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 7);
        // move pointers to process next 128 bytes
        in_data += 128;
        out_data += 128;
    }

    nn_sem_post(&td->donesem);
}


static int recip_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"recip execute. self=%p ",self);
	
	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *input_min_tensor = self->inputs[1];
	const struct tensor *input_max_tensor = self->inputs[2];
	
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	
	uint8_t *input_data = input_tensor->data;
	float input_min_float = tensor_get_float(input_min_tensor,0);
	float input_max_float = tensor_get_float(input_max_tensor,0);
	
	uint8_t *out_data = out_tensor->data;
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;
	
	float in_step = (input_max_float - input_min_float) / 255.f;
	
	int elements = input_tensor->shape.batches * input_tensor->shape.height * input_tensor->shape.width * input_tensor->shape.depth;
	
	tensor_out_prepare_normal(out_tensor, input_tensor->shape.batches, input_tensor->shape.height, input_tensor->shape.width, input_tensor->shape.depth, NN_TYPE_QUINT8);
	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	
	if(elements < 256){
		float out_max_float = -2147483648.0f;
		float out_min_float = 2147483648.0f;

		float* inverses = nn_malloc(elements * sizeof(float));

		for (int i = 0; i < elements; i++) {
			float float_val = input_min_float + input_data[i] * in_step;
			inverses[i] = 1/float_val;
			if(inverses[i] > out_max_float) out_max_float = inverses[i];
			if(inverses[i] < out_min_float) out_min_float = inverses[i];
		}
		
		if(out_min_float < 0 && out_max_float < 0) out_max_float = 0.f;
		if(out_min_float > 0 && out_max_float > 0) out_min_float = 0.f;
		
		float out_step = (out_max_float - out_min_float) / 255.f;
		
		for (int i = 0; i < elements; i++){
			out_data[i] = round((inverses[i] - out_min_float) / out_step);
		}
		
		out_max[0] = out_max_float;
		out_min[0] = out_min_float;

		nn_free(inverses);
	} else {
		unsigned char inverse_lookup[256] __attribute__ ((aligned(128)));
		uint8_t real_min = 0;
		uint8_t real_max = 255;
		
		if(input_min_float < 0 && input_max_float > 0){
			uint8_t quantized_zero = round((-input_min_float) / in_step);
			uint8_t skip_start;
			uint8_t skip_end;
		
			struct nearzero_tdata worker0_info = {
					.self = self,
					.whoami = 0,
					.start_batch = 0,
					.end_batch = elements,
					.data = input_data,
					.quantized_zero = quantized_zero,
					.result = &skip_start,
			};
			struct nearzero_tdata worker1_info = {
					.self = self,
					.whoami = 0,
					.start_batch = 0,
					.end_batch = elements,
					.data = input_data,
					.quantized_zero = quantized_zero,
					.result = &skip_end,
			};

			nn_sem_init(&worker0_info.donesem,0);
			nn_sem_init(&worker1_info.donesem,0);
			nn_os_work_for_vector(nn,max_negative_execute_slice,&worker0_info);
			nn_os_work_for_vector(nn,min_positive_execute_slice,&worker1_info);
			nn_sem_wait(&worker0_info.donesem);
			nn_sem_wait(&worker1_info.donesem);

			get_reciprocals_split_range(input_min_float, input_max_float, real_min, real_max, skip_start, skip_end, out_min, out_max, inverse_lookup);
		}
		else if(input_min_float == 0 && input_max_float > 0){
			struct minmax_tdata td = {
					.in_data = input_data,
					.out_data = &real_min,
					.reduction_batches = 1,
					.num_blobs = 1,
					.blob_size = elements
			};
			nn_sem_init(&td.donesem, 0);
			nn_os_work_for_vector(nn, reduce_min_all_axes_hvx, &td);
			nn_sem_wait(&td.donesem);

			get_reciprocals_continuous_range(input_min_float, input_max_float, real_min, real_max, out_min, out_max, inverse_lookup);
		}
		else if(input_min_float < 0 && input_max_float == 0){
			struct minmax_tdata td = {
					.in_data = input_data,
					.out_data = &real_max,
					.reduction_batches = 1,
					.num_blobs = 1,
					.blob_size = elements
			};
			nn_sem_init(&td.donesem, 0);
			nn_os_work_for_vector(nn, reduce_max_all_axes_hvx, &td);
			nn_sem_wait(&td.donesem);

			get_reciprocals_continuous_range(input_min_float, input_max_float, real_min, real_max, out_min, out_max, inverse_lookup);
		}

		int elements_128aligned = elements / 128;
		elements_128aligned *= 128;
		
		struct mapdata td = {
				.in_data = input_data,
				.map_data = inverse_lookup,
				.out_data = out_data,
				.num_elements = elements_128aligned
		};
		nn_sem_init(&td.donesem,0);
		nn_os_work_for_vector(nn,map_values,&td);
		nn_sem_wait(&td.donesem);

		for(int i = elements_128aligned; i < elements; i++) out_data[i] = inverse_lookup[input_data[i]];
	}

	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedRecip_8 = {
	.execute = recip_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

