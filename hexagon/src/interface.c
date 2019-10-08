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
 * This contains the interface code.
 */

#include <hexagon_nn.h>
#include <hexnn_graph_wrapper_interface.h>
#include <nn_graph.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "nn_string_map.h"
#ifndef __hexagon__
#include <malloc.h>
#endif


#if defined(USE_OS_QURT) // should be only for QURT
#include "remote.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "AEEStdErr.h"
#include <qurt.h>
#else // defined(USE_OS_QURT)
#define AEE_EBADCLASS 10
typedef void* remote_handle64;
#endif

#define UNUSED_PARAM(x) (void)(x)
#define NN_VERSION 0x00020F00

#define ROUNDUP_8BYTES(X)   ((X+7)&(~7))

// TODO - Move these from globals to a graph property
extern int Num_Vector_Threads;
extern int Total_Threads;
extern int Stack_Size;
extern int VTCM_User_Req;

// Lookup table from qurt's qurt_sysenv_arch_version&0xffff onto known thread-counts
struct thread_count_t {
	int arch;
	int threads;
	int hvx_threads;
	int vtcm_size;
};
struct thread_count_t Arch_Thread_Counts[] = {
	{ .arch=0x8466, .threads=4, .hvx_threads=4, .vtcm_size=262144 }, //8150
	{ .arch=0x4066, .threads=4, .hvx_threads=2, .vtcm_size=262144 }, //Talos
	{ .arch=0,      .threads=0, .hvx_threads=0, .vtcm_size=0 },
};


//////////// HASH TABLE for mapping graph id -> graph pointer.
// The table is just NN_GRAPH_HASHN linked lists; a graph is stored
// on table indicated by find_hash(graph_id).
// We move a graph to the front of the list each time it's accessed.
///////////////////////////
// this is the size of the hash table; should be a power of 2
#define NN_GRAPH_HASHN 32

// each entry in the table is a linked-list
struct graph_hashtable_entry {
	struct nn_graph *graph_list;		// pointer to the first graph, or NULL
};
static struct graph_hashtable_entry graph_table[NN_GRAPH_HASHN];
// single mutex for the whole table - will allow us to 'visit all graphs'
// if we want to.
// It also protects graph_table_count and graph_id_seqno.
static nn_mutex_t graph_table_mutex = NN_MUTEX_INIT;
static uint32_t graph_table_count;
static uint32_t graph_id_seqno;

static inline int find_hash( nn_id_t grid){
	return grid % (unsigned)NN_GRAPH_HASHN;
}
//
// 'inner' lookup function:
//   - lock the mutex
//   - find the graph with the given id;
//   - move it to top of list, if it is not already;
//   - provide (indirectly) pointer to the record (if entryp is not NULL)
//   - does not unlock mutex if entryp is not NULL.
//  - if not found, it always unlocks the mutex and returns NULL.
//
static
struct nn_graph * find_graph_inner(nn_id_t grid, struct graph_hashtable_entry **entryp )
{
	if( grid == 0) return NULL;
	struct graph_hashtable_entry * tabp = &graph_table[ find_hash(grid)];

	nn_mutex_lock( &graph_table_mutex);

	if( entryp != NULL)
		*entryp = tabp;
	struct nn_graph *grp = tabp->graph_list;
	// fast path
	if( grp == NULL || grp->id == grid){
		if( grp == NULL || entryp ==NULL ){
			nn_mutex_unlock(&graph_table_mutex);
		}
		return grp;
	}
	// look down the list. There will be at least one
	// ahead of the one we find.
	struct nn_graph **pos = &grp->next_graph;
	while( grp = *pos, grp != NULL){
		if( grp->id == grid){		// found it
			// move to front
			*pos = grp->next_graph;
			grp->next_graph = tabp->graph_list;
			tabp->graph_list = grp;
			if( entryp != NULL) return grp;
			break;	// unlock and return;
		}
		pos = & grp->next_graph;
	}
	nn_mutex_unlock(& graph_table_mutex);
	return grp;
}

// given a new graph, assign an id to it, and add it to the hash table.
// returns the new id.
//
static nn_id_t
add_graph_to_hash( struct nn_graph *grp)
{
	uint32_t grid;
	nn_mutex_lock( &graph_table_mutex);
	do{
		grid = ++graph_id_seqno & 0x7FFFFFFF;
	}while( grid==0);

	struct graph_hashtable_entry * tabp = &graph_table[ find_hash(grid)];

	grp->next_graph = tabp->graph_list;
	tabp->graph_list = grp;
	grp->id = grid;
	graph_table_count ++;
	nn_mutex_unlock(&graph_table_mutex);
	return grid;
}
// map an id to a graph *; returns null if invalid.
// The graph will be moved to the front of its list, if not already
// there.
struct nn_graph *
nn_id_to_graph(nn_id_t id) {
	return find_graph_inner( id, NULL);
}
// map an id to a graph *, and remove from table; returns null if invalid.
static struct nn_graph *
nn_id_to_graph_and_remove(nn_id_t id) {
	struct graph_hashtable_entry * entryp;
	struct nn_graph * res = find_graph_inner( id, &entryp);
	if( res != NULL ){
		entryp->graph_list = res->next_graph;
		res->next_graph = NULL;
		graph_table_count--;
		nn_mutex_unlock(&graph_table_mutex);
	}
	return res;
}
/////////////// END OF GRAPH HASH TABLE CODE /////////////////////////


static inline void fast_strncpy(char *dst, const char *src, int len)
{
	//int real_len = strnlen(src,len);
	int real_len = strlen(src)+1;
	if (real_len > len) real_len = len;
	memcpy(dst,src,real_len);
}



// Each symbol on the right is from the library on the left of =.
// Send back the current address for subraction from objdump symbol map.

int hexagon_nn_get_dsp_offset(uint32_t *libhexagon_addr, uint32_t *fastrpc_shell_addr)
{

        int retVal = 0;
#if defined(USE_OS_QURT) // should be only for QURT
	*fastrpc_shell_addr = (uint32_t)&qurt_sem_add;
	*libhexagon_addr    = (uint32_t)&hexagon_nn_get_dsp_offset;
#else // defined(USE_OS_QURT)
	*fastrpc_shell_addr = 0;
	*libhexagon_addr    = 0;
#endif
	return retVal;
}

int hexagon_nn_domains_get_dsp_offset(
	remote_handle64 h,
	uint32_t *libhexagon_addr,
	uint32_t *fastrpc_shell_addr
	)
{
	UNUSED_PARAM(h);
	return hexagon_nn_get_dsp_offset(libhexagon_addr, fastrpc_shell_addr);
}

int hexagon_nn_init_with_info(hexagon_nn_nn_id* g, const struct initinfo* info) {
	if (!g) return AEE_EBADCLASS;

	*g = 0;
	struct nn_graph *graph;
	int ret;
	if ((graph = nn_calloc(1,sizeof(*graph))) == NULL) {
		return -1;
	}
	graph->graph_options = Default_graphoptions;

	nn_os_vtcm_choose_size(graph);
	graph->priority = info->priority;

	graph->state = NN_GRAPH_CONSTRUCTION;
	nn_mutex_init(&graph->log_mutex);
	if ((graph->scratch = nn_memalign(128,SCRATCH_SIZE)) == NULL) {
		nn_free(graph);
		return -1;
	}
	graph->scratch_size = SCRATCH_SIZE;
	if ((graph->logbuf = nn_calloc(1,LOGBUF_SIZE)) == NULL) {
		nn_free(graph->scratch);
		nn_free(graph);
		return -1;
	}
	graph->logbuf_size = LOGBUF_SIZE-1;
	graph->logbuf_pos = 0;
	if ((ret=nn_os_workers_spawn(graph)) != 0) {
		nn_free(graph->logbuf);
		nn_free(graph->scratch);
		nn_free(graph);
		return -1;
	}
	/* allocate new ID */
	*g = add_graph_to_hash(graph);
	return 0;
}

int hexagon_nn_domains_init_with_info(remote_handle64 h, hexagon_nn_nn_id* g, const struct initinfo* info)
{
	UNUSED_PARAM(h);
	return hexagon_nn_init_with_info(g,info);
}


int hexagon_nn_init(hexagon_nn_nn_id *g)
{
	struct initinfo info;
	info.priority = 0; // 0 is default
	return hexagon_nn_init_with_info(g, &info);
}

int hexagon_nn_domains_init(remote_handle64 h, hexagon_nn_nn_id* g)
{
	UNUSED_PARAM(h);
	return hexagon_nn_init(g);
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

int hexagon_nn_domains_getlog(remote_handle64 h, nn_id_t id, unsigned char *buf, uint32_t length)
{
	UNUSED_PARAM(h);
	return hexagon_nn_getlog(id, buf, length);
}

int hexagon_nn_snpprint(nn_id_t id, unsigned char *buf, uint32_t length)
{
	struct nn_graph *graph;
	strncat((char *)buf,"id not found\n",length);
	if ((graph = nn_id_to_graph(id)) == NULL) return -1;
	do_snpprint(graph,(char *)buf,length);
	return 0;
}

int hexagon_nn_domains_snpprint(remote_handle64 h, nn_id_t id, unsigned char *buf, uint32_t length)
{
	UNUSED_PARAM(h);
	return hexagon_nn_snpprint(id, buf, length);
}

int hexagon_nn_set_debug_level(nn_id_t id, int level)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) return -1;
	if (level < 0) level = 0;
	graph->debug_level = level;
	return 0;
}

int hexagon_nn_domains_set_debug_level(remote_handle64 h, nn_id_t id, int level) {
	UNUSED_PARAM(h);
	return hexagon_nn_set_debug_level(id, level);
}

int hexagon_nn_set_graph_option(nn_id_t id, char const * optname, int value)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) return -1;
	return nn_option_set_int( graph, optname, value);
}

int hexagon_nn_domains_set_graph_option(remote_handle64 h, nn_id_t id, char const * optname, int value) {
	UNUSED_PARAM(h);
	return hexagon_nn_set_graph_option(id, optname, value);
}

int hexagon_nn_prepare(nn_id_t id)
{
	struct nn_graph *graph;

	uint64_t pcycle_start;
	uint64_t pcycle_stop;

	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	pcycle_start = nn_os_get_cycles(graph);

	int retval = do_prepare(graph);

	pcycle_stop = nn_os_get_cycles(graph);
	graph->execution_total_cycles = pcycle_stop - pcycle_start;
	graph->multi_execution_total_cycles += graph->execution_total_cycles;

	return retval;
}

int hexagon_nn_domains_prepare(remote_handle64 h, nn_id_t id) {
	UNUSED_PARAM(h);
	return hexagon_nn_prepare(id);
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

int hexagon_nn_append_node_list(
	nn_id_t id,
	const struct hexagon_nn_op_node *ops,
	int len) {
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL, "nn id %x not found", id);
	}
	if (graph->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(graph, "append: graph not under construction");
	}
	for (int i = 0; i < len; ++i) {
		if (0 != do_append_node(graph,
								ops[i].node_id,
								ops[i].operation,
								ops[i].padding,
								ops[i].inputsLen,
								ops[i].outputsLen,
								ops[i].inputs,
								ops[i].outputs))
			return errlog(NULL, "append node error %x", id);
	}
	return 0;
}

int hexagon_nn_domains_append_node(
	remote_handle64 h,
	nn_id_t id,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	const struct input *inputs,
	uint32_t num_inputs,
	const struct output *outputs,
	uint32_t num_outputs)
{
	UNUSED_PARAM(h);
	return hexagon_nn_append_node(
		id, node_id, operation, padding, inputs, num_inputs, outputs, num_outputs);
}

int hexagon_nn_domains_append_node_list(
    nn_id_t id,
    const struct hexagon_nn_op_node *ops,
    int len) {
  return hexagon_nn_append_node_list(id, ops, len);
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

int hexagon_nn_append_const_node_list(
	nn_id_t id,
	struct hexagon_nn_const_node *consts,
	int len) {
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL, "nn id %x not found", id);
	}
	if (graph->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(graph, "append: graph not under construction");
	}
	for (int i = 0; i < len; ++i) {
		if (0 != do_append_const_node(graph,
									  consts[i].node_id,
									  consts[i].tensor.batches,
									  consts[i].tensor.height,
									  consts[i].tensor.width,
									  consts[i].tensor.depth,
									  consts[i].tensor.data,
									  consts[i].tensor.dataLen))
			return errlog(NULL, "append const node error %x", id);
	}
	return 0;
}

int hexagon_nn_domains_append_const_node(
	remote_handle64 h,
	nn_id_t id,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len)
{
	UNUSED_PARAM(h);
	return hexagon_nn_append_const_node(
		id, node_id, batches, height, width, depth, data, data_len);
}

int hexagon_nn_domains_append_const_node_list(
    nn_id_t id,
    struct hexagon_nn_const_node *consts,
    int len) {
  return hexagon_nn_append_const_node_list(id, consts, len);
}

int hexagon_nn_append_empty_const_node(
	nn_id_t id,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	uint32_t data_len)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if (graph->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(graph,"append: graph not under construction");
	}
	return do_append_empty_const_node(
		graph,
		node_id,
		batches,
		height,
		width,
		depth,
		data_len);
}

int hexagon_nn_domains_append_empty_const_node(
	remote_handle64 h,
	nn_id_t id,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	uint32_t data_len)
{
	UNUSED_PARAM(h);
	return hexagon_nn_append_empty_const_node(
		id, node_id, batches, height, width, depth, data_len);
}


int hexagon_nn_populate_const_node(
	nn_id_t id,
	uint32_t node_id,
	const uint8_t *data,
	uint32_t data_len,
	uint32_t target_offset)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if (graph->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(graph,"append: graph not under construction");
	}
	return do_populate_const_node(
		graph,
		node_id,
		data,
		data_len,
		target_offset);
}

int hexagon_nn_domains_populate_const_node(
	remote_handle64 h,
	nn_id_t id,
	uint32_t node_id,
	const uint8_t *data,
	uint32_t data_len,
	uint32_t target_offset)
{
	UNUSED_PARAM(h);
	return hexagon_nn_populate_const_node(id, node_id, data, data_len, target_offset);
}



/*
 * FIXME: hexagon_nn_tensordef will no longer be compatible with struct tensor
 * as we make it more complex.
 * Instead, create struct tensors here and copy from hexagon_nn_tensordef values.
 * You should be able to avoid copying the bulk data though!
 *
 * Note that in C99 you can create an array on the stack from a function argument.
 */

int hexagon_nn_execute_new(
	nn_id_t id,
	const hexagon_nn_tensordef *inputs,
	uint32_t n_inputs,
	hexagon_nn_tensordef *outputs,
	uint32_t n_outputs)
{
	struct nn_graph *graph;
	uint64_t pcycle_start;
	uint64_t pcycle_stop;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	pcycle_start = nn_os_get_cycles(graph);
	if (graph->n_inputs != n_inputs) {
		struct tensor *inputs_tmp;
		if ((inputs_tmp = nn_realloc((void *)graph->inputs,sizeof(*inputs_tmp)*n_inputs)) == NULL) {
			return errlog(graph,"can't allocate for %d inputs",n_inputs);
		} else {
			graph->inputs = inputs_tmp;
			graph->n_inputs = n_inputs;
		}
	}
	if (graph->n_outputs != n_outputs) {
		struct tensor *outputs_tmp;
		if ((outputs_tmp = nn_realloc(graph->outputs,sizeof(*outputs_tmp)*n_outputs)) == NULL) {
			return errlog(graph,"can't allocate for %d outputs",n_outputs);
		} else {
			graph->outputs = outputs_tmp;
			graph->n_outputs = n_outputs;
		}
	}
	int i;
	int ret;
	struct tensor *input_tensors = (struct tensor *)graph->inputs;
	struct tensor *output_tensors = (struct tensor *)graph->outputs;
	for (i = 0; i < n_inputs; i++) {
		const hexagon_nn_tensordef *in = inputs+i;
		struct tensor *t = input_tensors+i;
		t->shape.batches = in->batches;
		t->shape.height = in->height;
		t->shape.width = in->width;
		t->shape.depth = in->depth;
		t->data = in->data;
		t->max_size = in->dataLen;
		t->data_size = in->data_valid_len;
		t->format.raw0 = t->format.raw1 = 0;
		int elementsize = in->data_valid_len / (in->batches * in->height * in->width * in->depth);
		if (elementsize == 4) {
			t->format.type = NN_TYPE_FLOAT; // Just a best guess
		} else if (elementsize == 2) {
			t->format.type = NN_TYPE_QINT16; // Just a best guess
		} else {
			t->format.type = NN_TYPE_QUINT8; // Just a best guess
		}
	}
	for (i = 0; i < n_outputs; i++) {
		hexagon_nn_tensordef *out = outputs+i;
		struct tensor *t = output_tensors+i;
		t->data = out->data;
		t->max_size = out->dataLen;
		t->data_size = 0;
	}
	if (graph->state != NN_GRAPH_PREPARED) {
		return errlog(graph,"graph not prepared");
	}
	logmsg(graph,2,"in hexagon_nn_execute_new, %d in %d out",n_inputs,n_outputs);
	ret = do_execute(graph);
	for (i = 0; i < n_outputs; i++) {
		hexagon_nn_tensordef *out = outputs+i;
		struct tensor *t = output_tensors+i;
		out->batches = t->shape.batches;
		out->height = t->shape.height;
		out->width = t->shape.width;
		out->depth = t->shape.depth;
		out->data_valid_len = t->data_size;
	}
	pcycle_stop = nn_os_get_cycles(graph);
	graph->execution_total_cycles = pcycle_stop - pcycle_start;
	graph->multi_execution_total_cycles += graph->execution_total_cycles;
	if (ret) errlog(graph,"fail in execute_new()");
	return ret;
}

int hexagon_nn_execute_with_info(
	nn_id_t id,
	const hexagon_nn_tensordef *inputs,
	uint32_t n_inputs,
	hexagon_nn_tensordef *outputs,
	uint32_t n_outputs,
	hexagon_nn_execute_info *execute_info) {

    /* Prototype implementation just wraps hexagon_execute_new.
       Eventually it should be the other way around */
	int result = hexagon_nn_execute_new(id, inputs, n_inputs, outputs, n_outputs);

    /* initialize extra info to 0 */
    if(execute_info->extraInfo != NULL|| execute_info->extraInfoLen != 0){
       memset(execute_info->extraInfo,0, execute_info->extraInfoLen);
	}
    /* Set the returned extraInfoValidLen to 0. */
    execute_info->extraInfoValidLen = 0;
    /* Just handle basic errors right now */
    if (result == 0) {
        execute_info->result = NN_EXECUTE_SUCCESS;
    } else if(result == -1) {
        execute_info->result = NN_EXECUTE_ERROR;
    } else{
    	execute_info->result = result;
    }
	return 0;
}

int hexagon_nn_domains_execute_new(
	remote_handle64 h,
	nn_id_t id,
	const hexagon_nn_tensordef *inputs,
	uint32_t n_inputs,
	hexagon_nn_tensordef *outputs,
	uint32_t n_outputs)
{
	UNUSED_PARAM(h);
	return hexagon_nn_execute_new(id, inputs, n_inputs, outputs, n_outputs);
}

int hexagon_nn_domains_execute_with_info(
        remote_handle64 h,
	nn_id_t id,
	const hexagon_nn_tensordef *inputs,
	uint32_t n_inputs,
	hexagon_nn_tensordef *outputs,
        uint32_t n_outputs,
        hexagon_nn_execute_info *execute_info) {
        UNUSED_PARAM(h);
        return hexagon_nn_execute_with_info(id, inputs, n_inputs, outputs, n_outputs, execute_info);
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

	hexagon_nn_tensordef in;
	hexagon_nn_tensordef out = {0}; // klocwork
	int ret;

	in.batches = batches_in;
	in.height = height_in;
	in.width = width_in;
	in.depth = depth_in;
	in.data_valid_len = in.dataLen = data_len_in;
	in.data = (uint8_t *)data_in;
	out.data = data_out;
	out.dataLen = data_out_max;
	out.data_valid_len = 0;  // Handle no-output case
	ret = hexagon_nn_execute_new(id,&in,1,&out,1);
	*batches_out = out.batches;
	*height_out = out.height;
	*width_out = out.width;
	*depth_out = out.depth;
	*data_out_size = out.data_valid_len;
	return ret;
}

int hexagon_nn_domains_execute(
	remote_handle64 h,
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
	UNUSED_PARAM(h);
	return hexagon_nn_execute(
		id,
		batches_in, height_in, width_in, depth_in, data_in, data_len_in,
		batches_out, height_out, width_out, depth_out, data_out, data_out_max, data_out_size);
}

int hexagon_nn_teardown(nn_id_t id)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph_and_remove(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_teardown(graph);
}

int hexagon_nn_domains_teardown(remote_handle64 h, nn_id_t id) {
	UNUSED_PARAM(h);
	return hexagon_nn_teardown(id);
}

int hexagon_nn_get_nodetype(nn_id_t graph_id,
			    nn_id_t node_id,
			    uint32_t *node_type)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(graph_id)) == NULL) {
		return errlog(NULL,"nn graph-id %x not found",graph_id);
	}
	const struct nn_node *node;
	if ((node = get_node(graph,node_id)) == NULL) {
		return errlog(NULL,"node-id %x not found",node_id);
	}

	*node_type = node->node_type;
	return 0;
}

int hexagon_nn_domains_get_nodetype(
	remote_handle64 h,
	nn_id_t graph_id,
	nn_id_t node_id,
	uint32_t *nodetype)
{
	UNUSED_PARAM(h);
	return hexagon_nn_get_nodetype(graph_id, node_id, nodetype);
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

int hexagon_nn_domains_get_perfinfo(
	remote_handle64 h,
	nn_id_t id,
	struct perfinfo *info_out,
	unsigned int info_out_len,
	unsigned int *n_items_out)
{
	UNUSED_PARAM(h);
	return hexagon_nn_get_perfinfo(id, info_out, info_out_len, n_items_out);
}

int hexagon_nn_reset_perfinfo(nn_id_t id, uint32_t event)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_perfinfo_reset(graph,event);
}


int hexagon_nn_domains_reset_perfinfo(
	remote_handle64 h,
	nn_id_t id,
	int32_t event)
{
	UNUSED_PARAM(h);
	return hexagon_nn_reset_perfinfo(id, event);
}

int hexagon_nn_version(int *ver)
{
	*ver = NN_VERSION;
	return 0;
}

int hexagon_nn_domains_version(remote_handle64 h, int* ver)
{
	UNUSED_PARAM(h);
	return hexagon_nn_version(ver);
}

int hexagon_nn_multi_execution_cycles(nn_id_t id, unsigned int *cycles_lo, unsigned int *cycles_hi)
{
	struct nn_graph *graph;
	uint64_t total;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	total = graph->multi_execution_total_cycles;
	*cycles_hi = total >> 32;
	*cycles_lo = total;
	return 0;
}

int hexagon_nn_domains_multi_execution_cycles(
	remote_handle64 h,
	nn_id_t id,
	unsigned int *cycles_lo,
	unsigned int *cycles_hi)
{
	UNUSED_PARAM(h);
	return hexagon_nn_multi_execution_cycles(id, cycles_lo, cycles_hi);
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

int hexagon_nn_domains_last_execution_cycles(
	remote_handle64 h,
	nn_id_t id,
	unsigned int *cycles_lo,
	unsigned int *cycles_hi)
{
	UNUSED_PARAM(h);
	return hexagon_nn_last_execution_cycles(id, cycles_lo, cycles_hi);
}

int print_node_perf(nn_id_t id)
{
	struct nn_graph *graph;
	struct nn_node *node;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	for (node = graph->head; node != NULL; node = node->next) {
		if (node->node_type != OP_Const)
			logmsg(graph,0,"Node performance: %9llu - %s",node->iter_cycles,hexagon_nn_op_names[node->node_type]);
	}
	return 0;
}
int hexagon_nn_variable_read(
	nn_id_t id,
	uint32_t node_id,
	int32_t output_index,
	uint32_t *b_out,
	uint32_t *h_out,
	uint32_t *w_out,
	uint32_t *d_out,
	uint8_t *data_out,
	uint32_t data_out_max,
	uint32_t *data_out_len)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_variable_read( graph, node_id, output_index,
		b_out, h_out, w_out, d_out, data_out, data_out_max, data_out_len );
}

int hexagon_nn_domains_variable_read(
	remote_handle64 h,
	nn_id_t id,
	uint32_t node_id,
	int32_t output_index,
	uint32_t *b_out,
	uint32_t *h_out,
	uint32_t *w_out,
	uint32_t *d_out,
	uint8_t *data_out,
	uint32_t data_out_max,
	uint32_t *data_out_len)
{
	UNUSED_PARAM(h);
	return hexagon_nn_variable_read(
		id, node_id, output_index, b_out, h_out, w_out, d_out,
		data_out, data_out_max, data_out_len);
}


int hexagon_nn_variable_write (
	nn_id_t id,
	uint32_t node_id,
	int32_t output_index,
	uint32_t batches_in,
	uint32_t height_in,
	uint32_t width_in,
	uint32_t depth_in,
	const uint8_t *data_in,
	uint32_t data_len_in)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_variable_write( graph, node_id, output_index,
		batches_in, height_in, width_in, depth_in, data_in, data_len_in );
}

int hexagon_nn_domains_variable_write (
	remote_handle64 h,
	nn_id_t id,
	uint32_t node_id,
	int32_t output_index,
	uint32_t batches_in,
	uint32_t height_in,
	uint32_t width_in,
	uint32_t depth_in,
	const uint8_t *data_in,
	uint32_t data_len_in)
{
	UNUSED_PARAM(h);
	return hexagon_nn_variable_write(
		id, node_id, output_index, batches_in, height_in, width_in, depth_in,
		data_in, data_len_in);
}

int hexagon_nn_variable_write_flat(
	nn_id_t id,
	uint32_t node_id,
	int32_t output_index,
	const uint8_t *data_in,
	uint32_t data_len_in)
{
	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	return do_variable_write_flat( graph, node_id, output_index, data_in, data_len_in );
}

int hexagon_nn_domains_variable_write_flat(
	remote_handle64 h,
	nn_id_t id,
	uint32_t node_id,
	int32_t output_index,
	const uint8_t *data_in,
	uint32_t data_len_in)
{
	UNUSED_PARAM(h);
	return hexagon_nn_variable_write_flat(id, node_id, output_index, data_in, data_len_in);
}


int hexagon_nn_GetHexagonBinaryVersion(int *ver)
{
	return hexagon_nn_version(ver);
}

int hexagon_nn_domains_GetHexagonBinaryVersion(remote_handle64 h, int *ver)
{
	UNUSED_PARAM(h);
	return hexagon_nn_GetHexagonBinaryVersion(ver);
}

int hexagon_nn_PrintLog(const uint8_t data_in, unsigned int data_in_len)
{
	return 0;
}

int hexagon_nn_domains_PrintLog(remote_handle64 h,
	const uint8_t data_in, unsigned int data_in_len)
{
	UNUSED_PARAM(h);
	return hexagon_nn_PrintLog(data_in, data_in_len);
}

int hexagon_nn_op_name_to_id(const char *name, unsigned int *id)
{
	int res = op_type_from_string( name );
	*id = res;
	return (res <0)? -1:0;
}

int hexagon_nn_domains_op_name_to_id(remote_handle64 h,
	const char *name, unsigned int *id)
{
	UNUSED_PARAM(h);
	return hexagon_nn_op_name_to_id(name, id);
}

int hexagon_nn_op_id_to_name(const unsigned int id, char *name, int name_len)
{
	const char *opname = op_type_to_string(id);
	if ( opname == NULL ||  (strlen(opname)+1) > name_len) return -1;
	strcpy(name,opname);
	return 0;
}

int hexagon_nn_domains_op_id_to_name(remote_handle64 h,
	const unsigned int id, char *name, int name_len)
{
	UNUSED_PARAM(h);
	return hexagon_nn_op_id_to_name(id, name, name_len);
}


int hexagon_nn_get_num_nodes_in_graph(nn_id_t id, unsigned int *num_nodes){
	struct nn_graph *graph;
	struct nn_node *node;
	*num_nodes = 0;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}
	if (graph->state != NN_GRAPH_PREPARED) {
		return errlog(graph,"graph not prepared");
	}
	uint32_t i = 0;
	for (node = graph->head; node != NULL; node = node->next) {
		i++;
	}
	if (i==0){
		return errlog(graph,"0 nodes in present graph");
	}
	*num_nodes = i;
	return 0;
}

int hexagon_nn_domains_get_num_nodes_in_graph(remote_handle64 h,
	nn_id_t id, unsigned int *num_nodes)
{
	UNUSED_PARAM(h);
	return hexagon_nn_get_num_nodes_in_graph(id, num_nodes);
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

int hexagon_nn_get_power(int type)
{
	(void) type;
	HAP_power_response_t response;
	response.type = HAP_power_get_clk_Freq;
	(void) HAP_power_get(NULL, &response);
	return response.clkFreqHz;
}

#if (__HEXAGON_ARCH__ >= 62)
static int hexagon_nn_vote(unsigned int level)
{
    HAP_power_request_t request;
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = FALSE;

    // Bounds created by an expected level from 0-255 split into 5 powersave modes
    const unsigned int RELEASE_VOTE = 0x7FFFFFFF;
    const unsigned int TURBO_UPPER_BOUNDS = 51;
    const unsigned int NOMINAL_PLUS_UPPER_BOUNDS = 102;
    const unsigned int NOMINAL_UPPER_BOUNDS = 153;
    const unsigned int SVS_PLUS_UPPER_BOUNDS = 204;
    const unsigned int SVS_UPPER_BOUNDS = 255;

    // Suggested latency in microseconds
    const int LOW_LATENCY = 100;
    const int MEDIUM_LATENCY = 500;
    const int HIGH_LATENCY = 1000;

    if (level == RELEASE_VOTE){ // Release any vote
        request.dcvs_v2.dcvs_option = HAP_DCVS_V2_POWER_SAVER_MODE;
    }
    else {
        request.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
        request.dcvs_v2.set_latency = TRUE;
        request.dcvs_v2.set_dcvs_params = TRUE;
        if (level < TURBO_UPPER_BOUNDS){
            request.dcvs_v2.latency = LOW_LATENCY;
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
            request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_TURBO;
            request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
        } else if (level < NOMINAL_PLUS_UPPER_BOUNDS){
            request.dcvs_v2.latency = LOW_LATENCY;
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_NOMPLUS;
            request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_NOMPLUS;
            request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_NOMPLUS;
        } else if (level < NOMINAL_UPPER_BOUNDS){
            request.dcvs_v2.latency = LOW_LATENCY;
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_NOM;
            request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_NOM;
            request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_NOM;
        } else if (level < SVS_PLUS_UPPER_BOUNDS){
            request.dcvs_v2.latency = MEDIUM_LATENCY;
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_SVSPLUS;
            request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_SVSPLUS;
            request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_SVSPLUS;
        } else if (level < SVS_UPPER_BOUNDS){
            request.dcvs_v2.latency = HIGH_LATENCY;
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_SVS;
            request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_SVS;
            request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_SVS;
        } else {
            request.dcvs_v2.set_latency = FALSE;
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_SVS2;
            request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_SVS2;
            request.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_SVS2;
        }
    }
    return HAP_power_set(NULL, &request);
}

int hexagon_nn_set_powersave_details(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
    HAP_power_request_t request;
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = FALSE;

    // Suggested latency in microseconds
    const int LOW_LATENCY = 100;
    const int MEDIUM_LATENCY = 500;
    const int HIGH_LATENCY = 1000;

    if (corner == NN_CORNER_RELEASE){ // Release any vote
        request.dcvs_v2.dcvs_option = HAP_DCVS_V2_POWER_SAVER_MODE;
    }
    else {
        request.dcvs_v2.dcvs_enable = (dcvs == NN_DCVS_DISABLE ? FALSE : TRUE);
        request.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
        request.dcvs_v2.set_latency = TRUE;
        request.dcvs_v2.set_dcvs_params = TRUE;
#if (__HEXAGON_ARCH__ >= 66)
        if (corner == NN_CORNER_TURBOPLUS){
            request.dcvs_v2.dcvs_params.max_corner = 10;
            request.dcvs_v2.latency = (latency == 0 ? LOW_LATENCY : latency);
        } else
#endif // (__HEXAGON_ARCH__ >= 66)
	if (corner == NN_CORNER_TURBO){
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
            request.dcvs_v2.latency = (latency == 0 ? LOW_LATENCY : latency);
        } else if (corner == NN_CORNER_NOMPLUS){
            request.dcvs_v2.latency = (latency == 0 ? LOW_LATENCY : latency);
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_NOMPLUS;
        } else if (corner == NN_CORNER_NOMINAL){
            request.dcvs_v2.latency = (latency == 0 ? LOW_LATENCY : latency);
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_NOM;
        } else if (corner == NN_CORNER_SVSPLUS){
            request.dcvs_v2.latency = (latency == 0 ? MEDIUM_LATENCY : latency);
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_SVSPLUS;
        } else if (corner == NN_CORNER_SVS){
            request.dcvs_v2.latency = (latency == 0 ? HIGH_LATENCY : latency);
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_SVS;
        } else {
            request.dcvs_v2.set_latency = FALSE;
            request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_SVS2;
        }
        request.dcvs_v2.dcvs_params.min_corner = request.dcvs_v2.dcvs_params.max_corner;
        request.dcvs_v2.dcvs_params.target_corner = request.dcvs_v2.dcvs_params.max_corner;
    }
    return HAP_power_set(NULL, &request);
}

#else
static int hexagon_nn_vote(unsigned int level)
{
    HAP_power_request_t request;
    request.type = HAP_power_set_DCVS;
    request.dcvs.dcvs_enable = FALSE;
    request.dcvs.dcvs_option = HAP_DCVS_ADJUST_UP_DOWN;
    int ret;
    if ((ret = HAP_power_set(NULL, &request)) != 0)
        return errlog(NULL,"unable to vote DCVS off ret=%d",ret);

    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_mips_bw;
    request.mips_bw.set_mips = request.mips_bw.set_bus_bw = request.mips_bw.set_latency = TRUE;


    // Bounds created by an expected level from 0-255 split into 5 powersave modes
    const unsigned int RELEASE_VOTE = 0x7FFFFFFF;
    const unsigned int TURBO_UPPER_BOUNDS = 51;
    const unsigned int NOMINAL_UPPER_BOUNDS = 153;
    const unsigned int SVS_UPPER_BOUNDS = 255;

    // Suggested latency in microseconds
    const int LOW_LATENCY = 100;
    const int MEDIUM_LATENCY = 500;
    const int HIGH_LATENCY = 1000;

    if (level == RELEASE_VOTE){ // Release any vote
        request.mips_bw.latency = -1;
    }
    else {
        if (level < TURBO_UPPER_BOUNDS){
            request.mips_bw.mipsPerThread = 500;
            request.mips_bw.mipsTotal = 1000;
            request.mips_bw.bwBytePerSec = (uint64)(12000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = LOW_LATENCY;
        } else if (level < NOMINAL_UPPER_BOUNDS){
            request.mips_bw.mipsPerThread = 100;
            request.mips_bw.mipsTotal = 600;
            request.mips_bw.bwBytePerSec = (uint64)(6000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = LOW_LATENCY;
        } else if (level < SVS_UPPER_BOUNDS){
            request.mips_bw.mipsPerThread = 50;
            request.mips_bw.mipsTotal = 400;
            request.mips_bw.bwBytePerSec = (uint64)(3000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = MEDIUM_LATENCY;
        } else {
            request.mips_bw.mipsPerThread = 50;
            request.mips_bw.mipsTotal = 200;
            request.mips_bw.bwBytePerSec = (uint64)(1000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = HIGH_LATENCY;
        }
    }
    return HAP_power_set(NULL, &request);
}

int hexagon_nn_set_powersave_details(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
    HAP_power_request_t request;
    request.type = HAP_power_set_DCVS;
    request.dcvs.dcvs_enable = (dcvs == NN_DCVS_ENABLE ? TRUE : FALSE);
    request.dcvs.dcvs_option = HAP_DCVS_ADJUST_UP_DOWN;
    int ret;
    if ((ret = HAP_power_set(NULL, &request)) != 0)
        return errlog(NULL,"unable to vote DCVS ret=%d",ret);

    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_mips_bw;
    request.mips_bw.set_mips = request.mips_bw.set_bus_bw = request.mips_bw.set_latency = TRUE;


    // Suggested latency in microseconds
    const int LOW_LATENCY = 100;
    const int MEDIUM_LATENCY = 500;
    const int HIGH_LATENCY = 1000;

    if (corner == NN_CORNER_RELEASE){ // Release any vote
        request.mips_bw.latency = -1;
    }
    else {
        if (corner == NN_CORNER_TURBO || corner == NN_CORNER_NOMPLUS){
            request.mips_bw.mipsPerThread = 500;
            request.mips_bw.mipsTotal = 1000;
            request.mips_bw.bwBytePerSec = (uint64)(12000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = (latency == 0 ? LOW_LATENCY : latency);
        } else if (corner == NN_CORNER_NOMINAL || corner == NN_CORNER_SVSPLUS){
            request.mips_bw.mipsPerThread = 100;
            request.mips_bw.mipsTotal = 600;
            request.mips_bw.bwBytePerSec = (uint64)(6000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = (latency == 0 ? LOW_LATENCY : latency);
        } else if (corner == NN_CORNER_SVS){
            request.mips_bw.mipsPerThread = 50;
            request.mips_bw.mipsTotal = 400;
            request.mips_bw.bwBytePerSec = (uint64)(3000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = (latency == 0 ? MEDIUM_LATENCY : latency);
        } else {
            request.mips_bw.mipsPerThread = 50;
            request.mips_bw.mipsTotal = 200;
            request.mips_bw.bwBytePerSec = (uint64)(1000) * 1000000;
            request.mips_bw.busbwUsagePercentage = (unsigned short) 50;
            request.mips_bw.latency = (latency == 0 ? HIGH_LATENCY : latency);
        }
    }
    return HAP_power_set(NULL, &request);
}

#endif


int hexagon_nn_set_powersave_level(unsigned int level)
{
    return hexagon_nn_vote(level);
}


int hexagon_nn_graph_config(
	nn_id_t id,
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
	)
{

	struct nn_graph *graph;
	if ((graph = nn_id_to_graph(id)) == NULL) {
		return errlog(NULL,"nn id %x not found",id);
	}

//TODO - move hexagon_nn_config_with_options code into here
//   Don't forget to ensure that everything still works for people who
//   do standard config()..prepare() without calling graph_config() explicitly!
//   (Probably call graph_config during prepare() if it hasn't already been done.)

	for (int i=0; i<num_string_options; i++) {
		int string_length;
		switch(string_options[i].option_id) {
		case NN_OPTION_ENABLE_GRAPH_PRINT:
			graph->enable_graph_print = 1;
			string_length = strlen(string_options[i].string_data);
			if ((graph->enable_graph_print_prefix = nn_calloc(1,string_length+1)) == NULL) {
				return -1;
			}
			strcpy(graph->enable_graph_print_prefix, string_options[i].string_data);
			break;
		case NN_OPTION_ENABLE_CONST_PRINT:
			graph->enable_const_print = 1;
			string_length = strlen(string_options[i].string_data);
			if ((graph->enable_const_print_prefix = nn_calloc(1,string_length+1)) == NULL) {
				return -1;
			}
			strcpy(graph->enable_const_print_prefix, string_options[i].string_data);
			break;
		case NN_OPTION_ENABLE_TENSOR_PRINT:
			if (graph->debug_level==0) {
				graph->debug_level = 1; // Ensure debug level is high enough to enable tensor printing
			}
			graph->enable_tensor_print = 1;
			string_length = strlen(string_options[i].string_data);
			if ((graph->enable_tensor_print_prefix = nn_calloc(1,string_length+1)) == NULL) {
				return -1;
			}
			strcpy(graph->enable_tensor_print_prefix, string_options[i].string_data);
			break;
		case NN_OPTION_TENSOR_PRINT_FILTER:
			string_length = strlen(string_options[i].string_data);
			if ((graph->tensor_print_filter = nn_calloc(1,string_length+1)) == NULL) {
				return -1;
			}
			strcpy(graph->tensor_print_filter, string_options[i].string_data);
			break;
		default:
			// UNSUPPORTED option.  Too bad we can't logmsg (TODO, FIXME)
			errlog(graph,"DEBUG: Unknown option %d",string_options[i].option_id);
			return 1;
			break;
		}
	}

	return 0;
}

int hexagon_nn_config_with_options(
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
	)
{
	// Always set this to some default, might get overriden by one
	// of the options.
#ifndef NN_GRAPH_ON_SIMULATOR // API not available on simulator
	HAP_mem_set_grow_size(0x1000000, MAX_UINT64);
#endif // NN_GRAPH_ON_SIMULATOR

	// Gather DSP info from Qurt
	// vector_mode_mask 00000200=2xHVX, 00000400=4xHVX
	int vector_mode_mask = qurt_hvx_get_units();
	qurt_sysenv_max_hthreads_t num_threads;
	qurt_sysenv_get_max_hw_threads(&num_threads);

	Total_Threads = num_threads.max_hthreads;
	Num_Vector_Threads = (vector_mode_mask & 0xF00) >> 8;
	Stack_Size = 16384;


	// Don't trust Qurt for thread-counts if we don't have to
	qurt_arch_version_t av;
	qurt_sysenv_get_arch_version(&av);
	int arch = av.arch_version & 0xffff;
	for (int i=0; Arch_Thread_Counts[i].arch; i++) {
		if (arch == Arch_Thread_Counts[i].arch) {
			Total_Threads = Arch_Thread_Counts[i].threads;
			Num_Vector_Threads = Arch_Thread_Counts[i].hvx_threads;
			VTCM_User_Req = Arch_Thread_Counts[i].vtcm_size;
			break;
		}
	}


	// Explicitly-provided numbers override any auto-set values
	int i;
	for (i=0; i<num_uint_options; i++) {
		if (uint_options[i].uint_value != (uint32_t) -1) {  // "-1" is uint equivalent of "use defaults"
			switch(uint_options[i].option_id) {
			case NN_OPTION_NOSUCHOPTION:
				break;
			case NN_OPTION_SCALAR_THREADS:
				Total_Threads = uint_options[i].uint_value;
				break;
			case NN_OPTION_HVX_THREADS:
				Num_Vector_Threads = uint_options[i].uint_value;
				break;
			case NN_OPTION_VTCM_REQ:
				VTCM_User_Req = uint_options[i].uint_value;
				break;
			case NN_OPTION_HAP_MEM_GROW_SIZE:
#ifndef NN_GRAPH_ON_SIMULATOR // API not available on simulator
			    HAP_mem_set_grow_size(uint_options[i].uint_value, MAX_UINT64);
#endif // NN_GRAPH_ON_SIMULATOR
			    break;
			default:
				// UNSUPPORTED option.  Too bad we can't logmsg (TODO, FIXME)
				return 1;
				break;
			}
		}
	}

	return 0;
}



int hexagon_nn_config()
{
	return hexagon_nn_config_with_options(NULL,0,NULL,0);
}


#else
int hexagon_nn_set_powersave_level(unsigned int level) { return 0; }
int hexagon_nn_set_powersave_details(hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency) { return 0; }
int hexagon_nn_disable_dcvs() { return 0; }
int hexagon_nn_get_power(int type) { return 0; }

int hexagon_nn_graph_config(
	nn_id_t id,
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
	)
{
	// TODO - All targets probably need graph_config, as the options it controls grows.
	//        Older options like num-threads have tight interactions with the OS-specific code,
	//        so require careful treatment
	//        Newer options like enable_tensor_print should be enabled for all targets.

	return 0;
}
int hexagon_nn_config_with_options(
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
)
{
	return 0;
}

#if !defined(USE_OS_H2)
int hexagon_nn_config() {

	pmu_init();
	return 0;
}
#else
int hexagon_nn_config() {

	return 0;
}
#endif

#endif

int hexagon_nn_domains_config(remote_handle64 h)
{
	UNUSED_PARAM(h);
	return hexagon_nn_config();
}

int hexagon_nn_domains_config_with_options(
	remote_handle64 h,
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
	)
{
	UNUSED_PARAM(h);
	return hexagon_nn_config_with_options(
		uint_options, num_uint_options, string_options, num_string_options);
}

int hexagon_nn_domains_graph_config(
	nn_id_t id,
	const struct uint_option_t *uint_options,
	uint32_t num_uint_options,
	const struct string_option_t *string_options,
	uint32_t num_string_options
	)
{
	return hexagon_nn_graph_config(
		id, uint_options, num_uint_options, string_options, num_string_options);
}

int hexagon_nn_domains_set_powersave_level(remote_handle64 h, unsigned int level)
{
	UNUSED_PARAM(h);
	return hexagon_nn_set_powersave_level(level);
}


int hexagon_nn_domains_disable_dcvs(remote_handle64 h)
{
	UNUSED_PARAM(h);
	return hexagon_nn_disable_dcvs();
}


int hexagon_nn_domains_get_power(remote_handle64 h, int a)
{
	return hexagon_nn_get_power(a);
}

int hexagon_nn_domains_set_powersave_details(remote_handle64 h,
	hexagon_nn_corner_type corner, hexagon_nn_dcvs_type dcvs, unsigned int latency)
{
	UNUSED_PARAM(h);
	return hexagon_nn_set_powersave_details(corner, dcvs, latency);
}


int hexagon_nn_domains_open (const char* uri, remote_handle64* handle)
{
	if(handle == NULL) {
        errlog(NULL,"DEBUG:Null domain handle");
        return 0;
    }
    /* can be any value or ignored, rpc layer doesn't care
     * also ok
     * *handle = 0;
     * *handle = 0xdeadc0de;
     */
    *handle = (remote_handle64) 0xdeadc0de;
    return 0;
}

int hexagon_nn_domains_close(remote_handle64 h)
{
	return 0;
}

int hexagon_nn_populate_graph(nn_id_t id,
        const unsigned char* graph_data, int graph_dataLen)
{
    int sts = 0;

    flat_batch_ops_params *fbo_poi;

    fbo_poi = (flat_batch_ops_params *)graph_data;

	unsigned long size = 0;

	while (size < graph_dataLen && sts == 0)
    {
		hexagon_nn_input *inputs;
		unsigned long inputsLen;

		hexagon_nn_output *outputs;
		unsigned long outputsLen;

		unsigned char *data;
		unsigned long dataLen;

		size += sizeof(flat_batch_ops_params)-sizeof(unsigned char);

        switch (fbo_poi->op) {
            case HEXNN_BATCH_OP_APPEND_NODE:
				inputs = (hexagon_nn_input *)fbo_poi->c;
				inputsLen = fbo_poi->U.an_params.inputsLen/sizeof(hexagon_nn_input);

				outputs = (hexagon_nn_output *)(fbo_poi->c + ROUNDUP_8BYTES(fbo_poi->U.an_params.inputsLen));
				outputsLen = fbo_poi->U.an_params.outputsLen/sizeof(hexagon_nn_output);

                sts = hexagon_nn_append_node(id, fbo_poi->U.an_params.node_id,
                        fbo_poi->U.an_params.operation, fbo_poi->U.an_params.padding,
                        inputs, inputsLen,
                        outputs, outputsLen);
                if(sts !=0 ) {
                    errlog(NULL,"DEBUG:batched op error at HEXNN_BATCH_OP_APPEND_NODE, node_id = %d, sts = %d", fbo_poi->U.an_params.node_id, sts);
                }

                size += (ROUNDUP_8BYTES(fbo_poi->U.an_params.inputsLen) + ROUNDUP_8BYTES(fbo_poi->U.an_params.outputsLen));

                break;

            case HEXNN_BATCH_OP_APPEND_CONST_NODE:
				data = fbo_poi->c;
				dataLen = fbo_poi->U.acn_params.dataLen/sizeof(unsigned char);

                sts = hexagon_nn_append_const_node(id, fbo_poi->U.acn_params.node_id,
                        fbo_poi->U.acn_params.batches, fbo_poi->U.acn_params.height,
                        fbo_poi->U.acn_params.width, fbo_poi->U.acn_params.depth, 
						data, dataLen);
                if(sts !=0 ) {
                    errlog(NULL,"DEBUG:batched op error at HEXNN_BATCH_OP_APPEND_CONST_NODE, node_id = %d, sts = %d", fbo_poi->U.acn_params.node_id, sts);

                }

				size += ROUNDUP_8BYTES(fbo_poi->U.acn_params.dataLen);

                break;

            case HEXNN_BATCH_OP_APPEND_EMPTY_CONST_NODE:
                sts = hexagon_nn_append_empty_const_node(id, fbo_poi->U.aecn_params.node_id,
                        fbo_poi->U.aecn_params.batches, fbo_poi->U.aecn_params.height,
                        fbo_poi->U.aecn_params.width, fbo_poi->U.aecn_params.depth,
                        fbo_poi->U.aecn_params.size);
                if(sts !=0 ) {
                    errlog(NULL,"DEBUG:batched op error at HEXNN_BATCH_OP_APPEND_EMPTY_CONST_NODE, node_id = %d, sts = %d", fbo_poi->U.aecn_params.node_id, sts);
                }

				size += ROUNDUP_8BYTES(8);

                break;

            case HEXNN_BATCH_OP_POPULATE_CONST_NODE:
				data = fbo_poi->c;
				dataLen = fbo_poi->U.pcn_params.dataLen/sizeof(unsigned char);

                sts = hexagon_nn_populate_const_node(id, fbo_poi->U.pcn_params.node_id,
                        data, dataLen,
                        fbo_poi->U.pcn_params.target_offset);
                if(sts !=0 ) {
                    errlog(NULL,"DEBUG:batched op error at HEXNN_BATCH_OP_POPULATE_CONST_NODE, node_id = %d, sts = %d", fbo_poi->U.pcn_params.node_id, sts);
                }

				size += ROUNDUP_8BYTES(fbo_poi->U.pcn_params.dataLen);

                break;

            default:
                // error, should not reach here
                sts = -1;
                break;
        }

        if(sts != 0){
            break;
        }
        else
		{
			fbo_poi = (flat_batch_ops_params *)(graph_data+size);
		}
    }

    if (sts == 0)
    {
        sts = hexagon_nn_prepare(id);
        if(sts !=0 ) {
            errlog(NULL,"DEBUG:batched op error at  hexagon_nn_prepare(), sts = %d", sts);
        }
    }

    return sts;
}

int hexagon_nn_domains_populate_graph(remote_handle64 h, nn_id_t id,
        const unsigned char* data, int dataLen)
{
    return hexagon_nn_populate_graph(id, data, dataLen);
}
