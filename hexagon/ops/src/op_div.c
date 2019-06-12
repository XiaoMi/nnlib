
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
#include "hvx_inlines.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains an elementwise div op for 1-D denominators
 */

struct mapdata {
    const uint8_t *in_data;
    uint8_t *map_data;
    uint8_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};

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

static int div_depthwise_execute(struct nn_node *self, struct nn_graph *nn)
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
	uint8_t *a_data = a_tensor->data;
	uint8_t *b_data = b_tensor->data;
	float a_min_float = tensor_get_float(a_min_tensor,0);
	float a_max_float = tensor_get_float(a_max_tensor,0);
	float a_step = (a_max_float - a_min_float) / 255.f;
	float b_min_float = tensor_get_float(b_min_tensor,0);
	float b_max_float = tensor_get_float(b_max_tensor,0);
	float b_step = (b_max_float - b_min_float) / 255.f;
	uint8_t *out_data = out_tensor->data;
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;

	int elements = a_tensor->shape.batches * a_tensor->shape.height * a_tensor->shape.width * a_tensor->shape.depth;
	int depth = b_tensor->shape.depth;

	if(self->n_inputs > 6){
		*out_min = tensor_get_float(self->inputs[6],0);
		*out_max = tensor_get_float(self->inputs[7],0);
	}
	else{
		*out_min = 2147483648.0f;
		*out_max = -2147483648.0f;

		for(int d = 0; d < depth; d++){
			float denominator = b_min_float + ((float)b_data[d]) * b_step;

			if(a_max_float/denominator > *out_max) *out_max = a_max_float/denominator;
			if(a_min_float/denominator < *out_min) *out_min = a_min_float/denominator;
		}
		
		if(*out_min < 0 && *out_max < 0) *out_max = 0;
		if(*out_min > 0 && *out_max > 0) *out_min = 0;
	}
	

	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_tensor, a_tensor->shape.batches, a_tensor->shape.height, a_tensor->shape.width, a_tensor->shape.depth, NN_TYPE_QUINT8);

	float out_step = (*out_max - *out_min) / 255.f;
	
	// For each depth:
	//   1. Calculate a map from the depths' true range to quantized output
	//   2. Go through the values at that depth and map them to the output
	uint8_t map[256];
	for(int d = 0; d < depth; d++){
		float denominator = b_min_float + ((float)b_data[d]) * b_step;

		if(denominator == 0.f) return errlog(nn,"division by zero");

		// Calculate map
		for(int i = 0; i <= 255 ; i++){
			float numerator = a_min_float + ((float)i) * a_step;
			float quotient = numerator / denominator;
			map[i] = round((quotient - *out_min) / out_step);
		}

		// Map the input to the output
		for(int i = d; i < elements; i+=depth){
			out_data[i] = map[a_data[i]];
		}
	}

	return 0;
}

static int div_scalar_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"div execute. self=%p ",self);

	float in_min = tensor_get_float(self->inputs[2],0);
	float in_max = tensor_get_float(self->inputs[3],0);
	
	const struct tensor *b_tensor = self->inputs[1];
	
	uint8_t *b_data = b_tensor->data;
	float b_min = tensor_get_float(self->inputs[4],0);
	float b_max = tensor_get_float(self->inputs[5],0);
	float b_step = (b_max - b_min) / 255.f;

	float scalar = b_min + ((float)b_data[0]) * b_step;
	
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;

	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);

	out_min[0] = in_min / scalar;
	out_max[0] = in_max / scalar;

	tensor_copy(self->outputs[0],self->inputs[0]);

	return 0;
}

static int div_scalar_static_minmax_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"div execute. self=%p ",self);

	const struct tensor *input_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	uint8_t *input_data = input_tensor->data;
	float in_min = tensor_get_float(self->inputs[2],0);
	float in_max = tensor_get_float(self->inputs[3],0);
	float in_step = (in_max - in_min) / 255.f;
	
	uint8_t *b_data = b_tensor->data;
	float b_min = tensor_get_float(self->inputs[4],0);
	float b_max = tensor_get_float(self->inputs[5],0);
	float b_step = (b_max - b_min) / 255.f;
	
	float static_min = tensor_get_float(self->inputs[6],0);
	float static_max = tensor_get_float(self->inputs[7],0);
	float static_step = (static_max - static_min) / 255.f;

	uint8_t *out_data = out_tensor->data;
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;

	float scalar = b_min + ((float)b_data[0]) * b_step;

	tensor_out_prepare_normal(out_tensor, input_tensor->shape.batches, input_tensor->shape.height, input_tensor->shape.width, input_tensor->shape.depth, NN_TYPE_QUINT8);
	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);

	out_min[0] = static_min;
	out_max[0] = static_max;
	
	unsigned char div_lookup[256] __attribute__ ((aligned(128)));
	
	for(int i = 0; i < 256; i++){
		float float_value = (in_min + ((float)i) * in_step) / scalar;
		div_lookup[i] = (float_value - static_min)/static_step;
	}
	
	int elements = input_tensor->shape.batches * input_tensor->shape.height * input_tensor->shape.width * input_tensor->shape.depth;
	int elements_128aligned = elements / 128;
	elements_128aligned *= 128;
	
	struct mapdata td = {
			.in_data = input_data,
			.map_data = div_lookup,
			.out_data = out_data,
			.num_elements = elements_128aligned
	};
	nn_sem_init(&td.donesem,0);
	nn_os_work_for_vector(nn,map_values,&td);
	nn_sem_wait(&td.donesem);

	for(int i = elements_128aligned; i < elements; i++) out_data[i] = div_lookup[input_data[i]];

	return 0;
}

static int div_execute(struct nn_node *self, struct nn_graph *nn){

	if(self->inputs[1]->shape.depth == 1){
		if(self->n_inputs == 6){
			return div_scalar_execute(self, nn);
		} else {
			return div_scalar_static_minmax_execute(self, nn);
		}
	} else {
		return div_depthwise_execute(self, nn);
	}
}

static int div_check(struct nn_node *self, struct nn_graph *nn){
	if(self->n_inputs != 6 && self->n_inputs != 8)
		return errlog(nn,"must have 6 or 8 inputs");
	if(self->inputs[1]->shape.batches != 1 
		|| self->inputs[1]->shape.height != 1 
		|| self->inputs[1]->shape.width != 1){
		return errlog(nn,"op only supported for scalar and 1d tensors");
	}
	logmsg(nn,2,"div node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedDiv_8 = {
	.execute = div_execute,
	.check = div_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(6,8),
	.n_outputs = NN_IOCOUNT(3),
};

