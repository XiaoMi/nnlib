/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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

//  Group and transpose along a given axis
//
//  The operation is equivalent to doing the following (e.g. axis is w)
//  [b,h,wa*wb,d] -> [b,h,wa,wb,d]  (reshape)
//  [b,h,wb,wa,d]                   (transpose)
//  [b,h,wb*wa,d]                   (reshape)
//
//  5 inputs:
//      0: input data       (uint8_t) Need to set up the input shape
//      1: axis             (int32_t)
//      2: number of groups (int32_t)
//      3: input min val    (scalar float)
//      4: input max val    (scalar float)
//  3 output:
//      0: output data      (uint8_t)
//      1: output min val   (scalar float - same as input min)
//      2: output max val   (scalar float - same as input max)
//
// There is also AxisShuffle_16, which is the same, but handles 16 bit data;
// and AxisShuffle_f, AxisShuffle_int32 which are the same but have
// no min/max inputs or outputs (in 3, out 1).
//

#include <stdlib.h>
#include <nn_graph.h>
#include "nn_gentranspose.h"
#include "nn_axis.h"

#define OP_AXISSHUFFLE_INPUT_NUM 3
#define OP_AXISSHUFFLE_OUTPUT_NUM 1

#define OP_AXISSHUFFLE_Q_INPUT_NUM 5
#define OP_AXISSHUFFLE_Q_OUTPUT_NUM 3
#define DIM_NUM 4                   //number of dimensions

#define INPUT_DATA_IDX 0
#define INPUT_AXIS_IDX 1
#define INPUT_NUMGROUP_IDX 2
#define INPUT_DATA_MIN 3
#define INPUT_DATA_MAX 4

#define OUTPUT_DATA_IDX 0
#define OUTPUT_DATA_MIN 1
#define OUTPUT_DATA_MAX 2



struct axisshuffle_info {
	int16_t elementsize;				// size of element in bytes (set in 'check')
	int16_t is_quant;					// has min/max (set in 'check')
	int16_t eltype;						// NN_TYPE_XX (set in 'check')
	int16_t strategy_valid;
	struct shape opshape;				// the current operand shape
	int axisno;							// current dimension
	int numGroup;						// current shuffle block
    struct nn_transpose_desc txdesc;	// the 'transpose plan'
};

static int
axishuffle_check_strategy(struct nn_node *self, struct nn_graph *nn )
{
    const struct tensor *in_data_tensor = self->inputs[INPUT_DATA_IDX];
    const struct tensor *in_axis_tensor = self->inputs[INPUT_AXIS_IDX];
    const struct tensor *in_numGroup_tensor = self->inputs[INPUT_NUMGROUP_IDX];
    struct tensor *out_tensor = self->outputs[OUTPUT_DATA_IDX];

    struct axisshuffle_info  *info = (struct axisshuffle_info *)self->opaque;
    int32_t in_axis = tensor_get_int32( in_axis_tensor, 0);
    int numGroup = tensor_get_int32( in_numGroup_tensor, 0);

	if( info->strategy_valid
		&& shape_matches( &info->opshape, &in_data_tensor->shape )
		&& info->axisno == in_axis
		&& info->numGroup == numGroup ){
		return 0;			// we are good to go.
	}
	// fill in...
	info->axisno = in_axis;
	info->numGroup = numGroup;
	info->opshape = in_data_tensor->shape;
	info->strategy_valid = 0;

    int res = handle_negative_axes(nn, &in_axis, 1);
    if (res)
        return errlog(nn, "AxisShuffle: cannot group elements along the axis\n");

    uint32_t dimsize_to_shuffle = info->opshape.dimension[in_axis];
    uint32_t num_elems_per_group;
    if( numGroup <= 0 || dimsize_to_shuffle == 0 ||
    		( num_elems_per_group = dimsize_to_shuffle/numGroup,
    				num_elems_per_group * numGroup != dimsize_to_shuffle) ){
    	return errlog(nn, "AxisShuffle: cannot group elements along the axis \n");
    }
    if (tensor_out_prepare_normal_fromshape(out_tensor, &info->opshape, info->eltype) !=0) return errlog(nn,"out too small");

    if( info->is_quant){	// prepare min/max outputs
    	struct tensor *out_min_tensor = self->outputs[OUTPUT_DATA_MIN];
    	struct tensor *out_max_tensor = self->outputs[OUTPUT_DATA_MAX];
    	struct shape shp_1111 = { .batches = 1, .height = 1, .width = 1, .depth = 1 };
        if( tensor_out_prepare_normal_fromshape(out_min_tensor, &shp_1111, NN_TYPE_FLOAT)!=0
        		|| tensor_out_prepare_normal_fromshape(out_max_tensor, &shp_1111, NN_TYPE_FLOAT)!=0 ){
    		return errlog(nn,"min or max out too small");
    	}
    	// the _16 version does QINT16 as well; take care to propagate that from the input.
    	if( info->eltype == NN_TYPE_QUINT16 && in_data_tensor->format.type == NN_TYPE_QINT16 ){
    		out_tensor->format.type = NN_TYPE_QINT16;
    	}
    }
    // work out the transpose
    uint32_t temp_in_dim[DIM_NUM+1];
    int32_t perm_arr[DIM_NUM+1];
    uint32_t temp_dim_idx = 0;
    for (uint32_t i = 0; i < DIM_NUM; ++i, ++temp_dim_idx) {
        if(i == in_axis) {
            temp_in_dim[temp_dim_idx] = numGroup;
            temp_in_dim[temp_dim_idx+1] = num_elems_per_group;

            perm_arr[temp_dim_idx] = temp_dim_idx+1;
            perm_arr[temp_dim_idx+1] = temp_dim_idx;
            ++temp_dim_idx;
            continue;
        }
        temp_in_dim[temp_dim_idx] = info->opshape.dimension[i];
        perm_arr[temp_dim_idx] = temp_dim_idx;
    }
    res = nn_transpose_analyze_direct( &info->txdesc, info->elementsize, perm_arr, DIM_NUM+1, temp_in_dim, DIM_NUM+1 );
    if(res) return errlog( nn,"AxisShuffle: transpose analyze error %d", res);

    if(info->txdesc.buffer_needed > nn->scratch_size){
        res = nn_scratch_grow(nn, info->txdesc.buffer_needed);
        if( res != 0)return errlog(nn, "failed to grow scratch memory");
    }

    info->strategy_valid = 1;
    return 0;
}



static int axisshuffle_execute(struct nn_node *self, struct nn_graph *nn)
{

	int res = axishuffle_check_strategy( self, nn);
	if (res !=0) return res;

    const struct tensor *in_data_tensor = self->inputs[INPUT_DATA_IDX];

    struct tensor *out_tensor = self->outputs[OUTPUT_DATA_IDX];
    struct axisshuffle_info  *info = (struct axisshuffle_info *)self->opaque;

    res = nn_transpose_execute( nn, &info->txdesc, nn->scratch, (uint8_t*)(out_tensor->data), (uint8_t const*)(in_data_tensor->data));
    if(res) return errlog( nn, "AxisShuffle: transpose exec error %d", res);
    if( info->is_quant ){
        const struct tensor *in_data_min_tensor = self->inputs[INPUT_DATA_MIN];
        const struct tensor *in_data_max_tensor = self->inputs[INPUT_DATA_MAX];
        struct tensor *out_min_tensor = self->outputs[OUTPUT_DATA_MIN];
        struct tensor *out_max_tensor = self->outputs[OUTPUT_DATA_MAX];
    	*(float*)(out_min_tensor->data) = *(float*)(in_data_min_tensor->data);
    	*(float*)(out_max_tensor->data) = *(float*)(in_data_max_tensor->data);
    }

    return 0;
}
//
// 'check' allocates the 'info' object, setting it all to 0,
// except for 3 fields which define the particular op variant

static int axisshuffle_check(struct nn_node *self, struct nn_graph *nn)
{
	struct axisshuffle_info  *info = (struct axisshuffle_info  *) nn_calloc(1, sizeof(struct axisshuffle_info));
	if( info == NULL) return errlog(nn,"calloc failed");
	self->opaque = info;
	int elsize, eltype, isq;
	switch( self->node_type ){
	 case OP_AxisShuffle_int32:
		 elsize = sizeof(int32_t);
		 eltype = NN_TYPE_INT32;
		 isq = 0;
		 break;
	 case OP_AxisShuffle_f:
		 elsize = sizeof(float);
		 eltype = NN_TYPE_FLOAT;
		 isq = 0;
		 break;
	 case OP_AxisShuffle_8:
		 elsize = sizeof(uint8_t);
		 eltype = NN_TYPE_QUINT8;
		 isq = 1;
		 break;
	 case OP_AxisShuffle_16:
		 elsize = sizeof(uint16_t);
		 eltype = NN_TYPE_QUINT16;
		 isq = 1;
		 break;

	 default:
		 return errlog(nn,"unexpected node_type %d", (int)self->node_type);
	}
	info->elementsize = elsize;
	info->eltype = eltype;
	info->is_quant = isq;

	nn_scratch_grow(nn,64*1024);	/// to cover all current transpose cases
	return 0;
}
struct nn_node_ops nn_ops_for_AxisShuffle_int32 = {
    .execute = axisshuffle_execute,
    .check = axisshuffle_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
    .n_inputs = NN_IOCOUNT(OP_AXISSHUFFLE_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_AXISSHUFFLE_OUTPUT_NUM),
};

struct nn_node_ops nn_ops_for_AxisShuffle_f = {
    .execute = axisshuffle_execute,
    .check = axisshuffle_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
    .n_inputs = NN_IOCOUNT(OP_AXISSHUFFLE_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_AXISSHUFFLE_OUTPUT_NUM),
};


struct nn_node_ops nn_ops_for_AxisShuffle_8 = {
    .execute = axisshuffle_execute,
    .check = axisshuffle_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
    .n_inputs = NN_IOCOUNT(OP_AXISSHUFFLE_Q_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_AXISSHUFFLE_Q_OUTPUT_NUM),
};
// this does 16 and u16; the data type will be copied from the input.
struct nn_node_ops nn_ops_for_AxisShuffle_16 = {
    .execute = axisshuffle_execute,
    .check = axisshuffle_check,
    .ctor = node_alloc_common,
    .dtor = node_free_common_release_opaque,
    .n_inputs = NN_IOCOUNT(OP_AXISSHUFFLE_Q_INPUT_NUM),
    .n_outputs = NN_IOCOUNT(OP_AXISSHUFFLE_Q_OUTPUT_NUM),
};
