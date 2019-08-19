
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
#ifndef NN_GRAPH_IF_H
#define NN_GRAPH_IF_H 1
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains things applicable to the interface outside the DSP.
 */

#include <nn_graph_types.h>
#include <hexagon_nn.h>
#include <stdint.h>
#include <stdio.h>

struct input {
	uint32_t src_id;
	uint32_t output_idx;
};

#define NODE_ID_RESERVED_CONSTANT 0


#define MAX_DIMENSIONS 8
struct output {
	uint32_t rank; // dimensions in the tensor
	uint32_t max_sizes[MAX_DIMENSIONS]; // max num elements in each dimension
	uint32_t elementsize; // size of each element
	int32_t zero_offset; // 0 for float / integer values
	float stepsize; // 0 for float/integer values
};

struct perfinfo {
	uint32_t node_id;
	uint32_t executions;
	union {
		uint64_t counter;
		struct {
			uint32_t counter_lo;
			uint32_t counter_hi;
		};
	};
};

struct initinfo {
	int32_t priority;
};

int hexagon_nn_get_dsp_offset(uint32_t *libhexagon_addr, uint32_t *fastrpc_shell_addr);
int hexagon_nn_version(int *ver);
int hexagon_nn_last_execution_cycles(nn_id_t id, unsigned int *cycles_lo, unsigned int *cycles_hi);
int hexagon_nn_multi_execution_cycles(nn_id_t id, unsigned int *cycles_lo, unsigned int *cycles_hi);
int hexagon_nn_init(hexagon_nn_nn_id *g);
int hexagon_nn_config();
int hexagon_nn_get_power(int type);
int hexagon_nn_snpprint(nn_id_t id, unsigned char *buf, uint32_t length);
int hexagon_nn_getlog(nn_id_t id, unsigned char *buf, uint32_t length);
int hexagon_nn_set_debug_level(nn_id_t id, int level);
int print_node_perf(nn_id_t id);
int hexagon_nn_get_power(int type);

struct almost_a_tensor ;

int hexagon_nn_config_with_options(
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
	);
int hexagon_nn_graph_config(
	nn_id_t id,
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
	);


/* MUST MATCH IDL !! */
typedef struct {
        hexagon_nn_execute_result result;
        unsigned char* extraInfo;
        int extraInfoLen;
        int extraInfoValidLen; //like data_valid_len in tensordef
} hexagon_nn_execute_info;

/* 
 * Definition / I/O for a Tensor
 * Must match IDL
 * FIXME: could remove unused
 */
typedef struct {
	unsigned int batches;
	unsigned int height;
	unsigned int width;
	unsigned int depth;
	unsigned char *data;
	int dataLen;		/* For input and output */
	unsigned int data_valid_len; /* for output only */
	unsigned int unused;
} hexagon_nn_tensordef;

int hexagon_nn_append_node(
	nn_id_t id,
	uint32_t node_id, 
	op_type operation, 
	padding_type padding, 
	const struct input *inputs, 
	uint32_t num_inputs, 
	const struct output *outputs,
	uint32_t num_outputs);

int hexagon_nn_append_const_node(
	nn_id_t id,
	uint32_t node_id, 
	uint32_t batches, 
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len);

int hexagon_nn_append_empty_const_node(
	nn_id_t id,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	uint32_t data_len);

int hexagon_nn_populate_const_node(
	nn_id_t id,
	uint32_t node_id,
	const uint8_t *data,
	uint32_t data_len,
	uint32_t target_offset);

int hexagon_nn_prepare(nn_id_t id);
int hexagon_nn_execute(nn_id_t id, 
	uint32_t batches_in,
	uint32_t height_in,
	uint32_t width_in,
	uint32_t depth_in,
	const uint8_t *data_in,
	uint32_t data_len_in,
	uint32_t *batches_out,
	uint32_t *height_out,
	uint32_t *width_out,
	uint32_t *depth_out,
	uint8_t *data_out,
	uint32_t data_out_max,
	uint32_t *data_out_size);
int hexagon_nn_execute_new(nn_id_t id,
	const hexagon_nn_tensordef *tensors_in,
	uint32_t n_tensors_in,
	hexagon_nn_tensordef *tensors_out,
	uint32_t n_tensors_out);
int hexagon_nn_execute_with_info(nn_id_t id,
	const hexagon_nn_tensordef *tensors_in,
	uint32_t n_tensors_in,
	hexagon_nn_tensordef *tensors_out,
	uint32_t n_tensors_out, 
	hexagon_nn_execute_info *execute_info);
int hexagon_nn_teardown(nn_id_t id);
int hexagon_nn_reset_perfinfo(nn_id_t id, uint32_t event);
int hexagon_nn_get_perfinfo(nn_id_t id, 
	struct perfinfo *info_out, 
	unsigned int info_out_len,
	unsigned int *n_items_out);

int hexagon_nn_get_nodetype(nn_id_t graph_id,
			    nn_id_t node_id, 
			    uint32_t *node_type);

int hexagon_nn_variable_read( nn_id_t id, uint32_t node_id, int32_t output_index,
	uint32_t *b_out, uint32_t *h_out, uint32_t *w_out, uint32_t *d_out,
	uint8_t *data_out,	uint32_t data_out_max, 	uint32_t *data_out_len);
int hexagon_nn_variable_write (	nn_id_t id, uint32_t node_id, int32_t output_index,
	uint32_t batches_in, uint32_t height_in, uint32_t width_in, uint32_t depth_in,
	const uint8_t *data_in, uint32_t data_len_in);
int hexagon_nn_variable_write_flat( nn_id_t id, 	uint32_t node_id, int32_t output_index,
	const uint8_t *data_in, uint32_t data_len_in);

int hexagon_nn_op_name_to_id(const char *name, unsigned int *id);
int hexagon_nn_op_id_to_name(const unsigned int id, char *name, int name_len);
int hexagon_nn_set_powersave_level(unsigned int level);
int hexagon_nn_set_powersave_details(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency);

int hexagon_nn_set_graph_option( nn_id_t id, char const *opname, int value);

#endif
