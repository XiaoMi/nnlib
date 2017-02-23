
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
#ifndef NN_GRAPH_IF_H
#define NN_GRAPH_IF_H 1
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains things applicable to the interface outside the DSP.
 */

#include <nn_graph_types.h>
#include <stdint.h>

struct input {
	uint32_t src_id;
	uint32_t output_idx;
};

#define NODE_ID_RESERVED_CONSTANT 0

struct output {
	uint32_t max_size;
	uint32_t unused;
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

int hexagon_nn_version(int *ver);
int hexagon_nn_last_execution_cycles(nn_id_t id, unsigned int *cycles_lo, unsigned int *cycles_hi);
nn_id_t hexagon_nn_init();
int hexagon_nn_snpprint(nn_id_t id, unsigned char *buf, uint32_t length);
int hexagon_nn_getlog(nn_id_t id, unsigned char *buf, uint32_t length);
int hexagon_nn_set_debug_level(nn_id_t id, int level);

typedef struct almost_a_tensor hexagon_nn_tensordef;
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
int hexagon_nn_teardown(nn_id_t id);
int hexagon_nn_reset_perfinfo(nn_id_t id, uint32_t event);
int hexagon_nn_get_perfinfo(nn_id_t id, 
	struct perfinfo *info_out, 
	unsigned int info_out_len,
	unsigned int *n_items_out);

int hexagon_nn_op_name_to_id(const char *name, unsigned int *id);
int hexagon_nn_op_id_to_name(const unsigned int id, char *name, int name_len);
int hexagon_nn_set_powersave_level(unsigned int level);



#endif
