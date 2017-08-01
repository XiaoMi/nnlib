
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the interface code.
 */

#include <nn_graph.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef __hexagon__
#include <malloc.h>
#endif

#if defined(USE_OS_QURT) // should be only for QURT
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include <qurt.h>
#endif

static inline void fast_strncpy(char *dst, const char *src, int len)
{
	//int real_len = strnlen(src,len);
	int real_len = strlen(src)+1;
	if (real_len > len) real_len = len;
	memcpy(dst,src,real_len);
}

static inline struct nn_graph *nn_id_to_graph(nn_id_t id) {
	return (struct nn_graph *)(id);
}

static inline nn_id_t nn_graph_to_id(struct nn_graph *graph) {
	return (nn_id_t)(graph);
}

nn_id_t hexagon_nn_init()
{
	/* allocate new ID */
	struct nn_graph *graph;
	int ret;
	if ((graph = (struct nn_graph *)calloc(1,sizeof(*graph))) == NULL) {
		return 0;
	}
	graph->state = NN_GRAPH_CONSTRUCTION;
	if ((graph->scratch = memalign(128,SCRATCH_SIZE)) == NULL) {
		free(graph);
		return 0;
	}
	graph->scratch_size = SCRATCH_SIZE;
	if ((graph->logbuf = (char *)calloc(1,LOGBUF_SIZE)) == NULL) {
		free(graph->scratch);
		free(graph);
		return 0;
	}
	graph->logbuf_size = LOGBUF_SIZE-1;
	graph->logbuf_pos = 0;
	if ((ret=nn_os_workers_spawn(graph)) != 0) return ret;
	return nn_graph_to_id(graph);
}

int hexagon_nn_getlog(nn_id_t id, unsigned char *buf, uint32_t length)
{
	struct nn_graph *graph;
	fast_strncpy((char *)buf,"id not found\n",length);
	if ((graph = nn_id_to_graph(id)) == NULL) return -1;
	buf[length-1] = '\0';
	fast_strncpy((char *)buf,graph->logbuf,length-1);
	graph->logbuf_pos = 0;
	graph->logbuf[graph->logbuf_pos] = '\0';
	return 0;
}

int hexagon_nn_snpprint(nn_id_t id, unsigned char *buf, uint32_t length)
{
	struct nn_graph *graph;
	strncat((char *)buf,"id not found\n",length);
	if ((graph = nn_id_to_graph(id)) == NULL) return -1;
	do_snpprint(graph,(char *)buf,length);
	return 0;
}

int hexagon_nn_set_debug_level(nn_id_t id, int level)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) return -1;
	if (level < 0) level = 0;
	graph->debug_level = level;
	return 0;
}

int hexagon_nn_prepare(nn_id_t id)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_prepare(graph);
}

int hexagon_nn_append_node(
	nn_id_t id,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	const struct input *inputs,
	uint32_t num_inputs,
	const struct output *outputs,
	uint32_t num_outputs)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if (graph->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(graph,"append: graph not under construction");
	}
	return do_append_node(
		graph,
		node_id,
		operation,
		padding,
		num_inputs,
		num_outputs,
		inputs,
		outputs);
}

int hexagon_nn_append_const_node(
	nn_id_t id,
	uint32_t node_id,
	uint32_t batches, 
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if (graph->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(graph,"append: graph not under construction");
	}
	return do_append_const_node(
		graph,
		node_id,
		batches,
		height,
		width,
		depth,
		data,
		data_len);
}

int hexagon_nn_execute_new(
	nn_id_t id,
	const hexagon_nn_tensordef *inputs,
	uint32_t n_inputs,
	hexagon_nn_tensordef *outputs,
	uint32_t n_outputs)
{
	struct nn_graph *graph;
	const struct tensor *ins;
	struct tensor *outs;
	/* 
	 * hexagon_nn_tensordef should be compatible with struct tensor for
	 * input and output nodes
	 */
	ins = (const struct tensor *)inputs;
	outs = (struct tensor *)outputs;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if (graph->state != NN_GRAPH_PREPARED) {
		return errlog(graph,"graph not prepared");
	}
	logmsg(graph,2,"in hexagon_nn_execute_new, %d in %d out",n_inputs,n_outputs);
	return do_execute(graph,ins,n_inputs,outs,n_outputs);
}

int hexagon_nn_execute(
	nn_id_t id,
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
	uint32_t *data_out_size)
{
	struct nn_graph *graph;
	struct tensor in;
	struct tensor out;
	int ret;
	tensor_set_shape(&in,batches_in,height_in,width_in,depth_in);
	in.data_size = in.max_size = data_len_in;
	in.data = (uint8_t *)data_in;
	out.data = data_out;
	/* Put max size in data_size for output tensor to match RPC layout */
	out.data_size = out.max_size = data_out_max;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if (graph->state != NN_GRAPH_PREPARED) {
		return errlog(graph,"graph not prepared");
	}
	ret = do_execute(graph,&in,1,&out,1);
	*batches_out = out.shape.batches;
	*height_out = out.shape.height;
	*width_out = out.shape.width;
	*depth_out = out.shape.depth;
	*data_out_size = out.data_size;
	return ret;
}

int hexagon_nn_teardown(nn_id_t id)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_teardown(graph);
}

int hexagon_nn_get_perfinfo(nn_id_t id, 
	struct perfinfo *info_out, 
	unsigned int info_out_len,
	unsigned int *n_items_out)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if ((*n_items_out=do_perfinfo_get(graph,info_out,info_out_len))==~0U) {
		return -1;
	} else {
		return 0;
	}
}

int hexagon_nn_reset_perfinfo(nn_id_t id, uint32_t event)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_perfinfo_reset(graph,event);
}

int hexagon_nn_version(int *ver)
{
	*ver = 92;
	return 0;
}

int hexagon_nn_last_execution_cycles(nn_id_t id, unsigned int *cycles_lo, unsigned int *cycles_hi)
{
	struct nn_graph *graph;
	uint64_t total;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	total = graph->execution_total_cycles;
	*cycles_hi = total >> 32;
	*cycles_lo = total;
	return 0;
}

int hexagon_nn_GetHexagonBinaryVersion(int *ver)
{
	return hexagon_nn_version(ver);
}

int hexagon_nn_PrintLog(const uint8_t data_in, unsigned int data_in_len)
{
	return 0;
}

int hexagon_nn_op_name_to_id(const char *name, unsigned int *id)
{
	int i;
	for (i = 0; i < NN_OPS_MAX; i++) {
		if (0==strcmp(name,hexagon_nn_op_names[i])) {
			*id = i;
			return 0;
		}
	}
	return -1;
}

int hexagon_nn_op_id_to_name(const unsigned int id, char *name, int name_len)
{
	if (id >= NN_OPS_MAX) return -1;
	const char *opname = hexagon_nn_op_names[id];
	if ((strlen(opname)+1) > name_len) return -1;
	strcpy(name,opname);
	return 0;
}

#if defined(USE_OS_QURT) // should be only for QURT
int hexagon_nn_disable_dcvs()
{
    HAP_power_request_t request;
    request.type = HAP_power_set_DCVS;
    request.dcvs.dcvs_enable = FALSE;
    request.dcvs.dcvs_option = HAP_DCVS_ADJUST_UP_DOWN;
    return HAP_power_set(NULL, &request);
}

int hexagon_nn_set_powersave_level(unsigned int level)
{
	if (level == 0) return hexagon_nn_disable_dcvs();
	return 0;
}

int hexagon_nn_config()
{
	HAP_mem_set_grow_size(0x1000000, MAX_UINT64);
	return 0;
}
#else
int hexagon_nn_set_powersave_level(unsigned int level) { return 0; }
int hexagon_nn_disable_dcvs() { return 0; }
int hexagon_nn_config() { return 0; }

#endif
