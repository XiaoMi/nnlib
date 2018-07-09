
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
#ifndef NN_GRAPH_H
#define NN_GRAPH_H 1
/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains definitions for things used internally.
 */

#include <nn_graph_ops.h>
#include <nn_graph_types.h>
#include <nn_graph_im2col.h>
#include <nn_graph_if.h>
#include <nn_asm_ops.h>
#include <nn_graph_os.h>
#include <nn_atomic.h>
#include <platform.h>
#include <stdio.h>

#define likely(cond)	(__builtin_expect(!!(cond), 1))
#define unlikely(cond)	(__builtin_expect(!!(cond), 0))

struct nn_node_ops;
struct freelist_node;

struct nn_node {
	struct nn_node_ops *ops;	// Operations for this NN
	const struct tensor **inputs;	// Inputs
	struct tensor **outputs;	// Outputs
	struct input *input_refs;	// References to node outputs
	struct output *output_defs;	// Output definitions
	uint32_t n_inputs;		// Number of inputs
	uint32_t n_outputs;		// Number of outputs
	uint32_t node_type;		// node type
	uint32_t node_id;		// node ID
	padding_type padding;		// kind of padding
	struct nn_node *next;		// ptr to next node
	void *opaque;			// whatever the node wants to put here
	uint32_t executions;		// how many times the node executes
	uint32_t refs;			// time op was referenced by any output
	uint64_t perfcounter;		// performance counter
	uint64_t iter_cycles;		// cycles consumed in last execution
};

enum nn_graph_state {
	NN_GRAPH_INVALID,
	NN_GRAPH_CONSTRUCTION,
	NN_GRAPH_PREPARED,
};

enum {
	NN_GRAPH_PERFEVENT_CYCLES = 0,
	NN_GRAPH_PERFEVENT_USER0 = 1,
	NN_GRAPH_PERFEVENT_USER1 = 2,
	NN_GRAPH_PERFEVENT_HWPMU = 3,
	NN_GRAPH_PERFEVENT_UTIME = 5,
};

struct nn_graph {
	struct nn_node *head;		// First node in graph list
	struct nn_node *tail;
	struct nn_node *nonconst_head;
	struct nn_node *nonconst_tail;
	void *scratch;			// temporary storage
	size_t scratch_size;		// size of scratch
	int32_t scratch_nextalloc;	// next allocation offset
	nn_id_t id;			// ID of this nn
	unsigned int debug_level;	// Debug level of this NN
	const struct tensor *inputs;
	struct tensor *outputs;
	uint32_t n_inputs;
	uint32_t n_outputs;
	enum nn_graph_state state;
	struct freelist_node *root;	// root of internal free list
	void *bulk;			// bulk memory pointer
	unsigned long watermark_offset;	// most memory allocated
	unsigned int perf_event;
	char *logbuf;
	nn_mutex_t log_mutex;
	uint32_t logbuf_size;
	uint32_t logbuf_pos;
	nn_pipe_t *vec_work;
	nn_pipe_t *nonvec_work;
	uint64_t execution_total_cycles;
	void *os_opaque;		// data for the OS layer
	uint32_t internal_node_id;	// Internal Node ID
	uint32_t const_inf_id;		// ID for infinity
	uint32_t const_ninf_id;		// ID for -infinity
	uint32_t const_zero_id;		// ID for -infinity
	void *vtcm_ptr;			// ptr to VTCM
	uint32_t vtcm_size;		// size of VTCM
	struct nn_node **nonconst_head_ptr;	// ptr to head of non-const nodes
	void *find_node_opaque;
};

// Within an execution function,
// you can call nn_scratch_reset, and then
// use nn_scratch_alloc to divvy up the scratch into
// parts.


static inline void nn_scratch_reset(struct nn_graph *nn) { nn->scratch_nextalloc = 0; }

// nn_scratch_alloc is safe to call in multiple threads (but not thread_safe
// with respect to any of the other nn_scratch functions).
//

static inline void *nn_scratch_alloc(struct nn_graph *nn, size_t bytes)
{
	char *scratch_base = (char*) nn->scratch;
	size_t oldoff,oldoff0;
	size_t newoff;
	size_t total = nn->scratch_size;
	// Round up to multiple of 128 to keep things vector aligned.
	// Even for smaller requests, it may help to keep each one cache aligned.
	bytes = (bytes+127)&~(size_t)127;

	volatile int32_t * nextalloc_p = & nn->scratch_nextalloc;
	oldoff = *nextalloc_p;
	do{
		newoff = oldoff + bytes;
		if (newoff > total) return NULL;
		oldoff0 = oldoff;
		oldoff = nn_atomic_cas32(nextalloc_p,oldoff0,newoff);
	}while( unlikely(oldoff0 != oldoff) );

	return scratch_base + oldoff;
}

//
// nn_scratch_remain returns the remaining # of bytes in scratch not yet allocated;
// nn_scratch_nextaddr returns the address of that remaining mem
//
static inline int32_t nn_scratch_remain( struct nn_graph const *nn){
	return (int32_t)nn->scratch_size - nn->scratch_nextalloc;
}
static inline void * nn_scratch_nextaddr( struct nn_graph const *nn){
	return (char*)nn->scratch + nn->scratch_nextalloc;
}

int nn_scratch_grow(struct nn_graph *nn, size_t bytes);


static inline uint32_t nn_graph_new_internal_node_id(struct nn_graph *nn)
{
	return --(nn->internal_node_id);
}

static inline void record_usertime(struct nn_graph *nn, struct nn_node *op, uint32_t type, uint64_t time)
{
	if (nn->perf_event == type) op->perfcounter = time;
}

const struct nn_node *get_node(struct nn_graph *nn, uint32_t id);
void debug_print_node(struct nn_graph *nn, struct nn_node *node);
void debug_print_graph(struct nn_graph *nn);
void graphviz_print_node(struct nn_graph *nn, struct nn_node *node, FILE *dotfile);
void graphviz_print_graph(struct nn_graph *nn);
void print_node_checksum(struct nn_graph *nn, struct nn_node *node);
void print_graph_checksum(struct nn_graph *nn);

#define SCRATCH_SIZE (1024*1024*8)
#define LOGBUF_SIZE (1024*512)

#define NN_NODE_FLAG_D32_INPUT (1<<0)
#define NN_NODE_FLAG_D32_OUTPUT (1<<16)

struct nn_node_ops {
	int (*execute)(struct nn_node *self, struct nn_graph *nn);
	int (*check)(struct nn_node *self, struct nn_graph *nn);
	struct nn_node *(*ctor)(
		struct nn_graph *nn,
		uint32_t node_id,
		op_type operation,
		padding_type padding,
		uint32_t num_inputs,
		uint32_t num_outputs,
		const struct input *inputs,
		const struct output *outputs);
	int (*dtor)(struct nn_node *self, struct nn_graph *nn);
	int (*padding_hint)(struct nn_node *self, struct nn_graph *nn);
	unsigned int flags;
};
extern struct nn_node_ops *optab[];

int do_execute(struct nn_graph *nn);
int do_append_node(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs);
int do_append_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len);
int do_append_empty_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	uint32_t data_len);
int do_populate_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	const uint8_t *data,
	uint32_t data_len,
	uint32_t target_offset);

int do_teardown(struct nn_graph *nn);
void do_snpprint(struct nn_graph *nn, char *buf, uint32_t length);
int do_prepare(struct nn_graph *nn);

int do_perfinfo_reset(struct nn_graph *nn, uint32_t event);
int do_perfinfo_get(struct nn_graph *nn, struct perfinfo *info, uint32_t info_len);


int node_free_common(struct nn_node *node, struct nn_graph *nn);
struct nn_node *node_alloc_common(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs);

struct nn_node *alloc_node(
	uint32_t node_id,
	op_type operation,
	padding_type padding);
struct nn_node* find_last_consumer(
	struct nn_graph *nn,
	struct nn_node *producer,
	int out_idx);
struct nn_node* find_first_consumer(
	struct nn_graph *nn,
	struct nn_node *producer,
	int out_idx);
struct nn_node* find_unique_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int out_idx);
int check_single_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int out_idx,
	struct nn_node *consumer);

//
// this is given a list of one or more node items
//    rmnodes[0..n_remove-1]
//  which must exist in the list in the order given, possibly
//  with intervening items. 'anchor' is an upstream anchor for these
//   (i.e. *anchor = rmnodes[0]; or (*anchor)->next = rmnodes[0], etc)
//
// All of those are removed from the list, and the first one rmnodes[0] is replaced by new_node.
// it will return -1 if it can't find the removed nodes; this should not
// occur if the assumptions are met. If a non-zero return occurs, it may
// be that some of the items were not deleted. If the first item could not be found,
// the new item is not inserted.
//
// if anchor is NULL, &nn->head is used.
// if new_node is NULL, only the removal of nodes in rmnodes is done.
// if any of the rmnodes[1..n_remove-1] is NULL, it is taken as a list
//  terminator (but there must be at least one item).
//

int replace_node_sequence (
		struct nn_graph *nn,
		struct nn_node ** anchor,	// list anchor, at items[0] or upstream
		struct nn_node * new_node,	// node to replace (may be null)
    	struct nn_node * const * rmnodes,
        int n_remove);
// 'varargs' version of replace_node_sequence: the '...' parms become the rmnodes array.
// Evalutes to return value of replace_node_sequence.
//
#define replace_nodes(NN,ANCHOR,NEWNODE,...)\
 ({ struct nn_node *replace_node_list[] = { __VA_ARGS__}; \
   replace_node_sequence(NN,ANCHOR,NEWNODE, replace_node_list, sizeof(replace_node_list)/sizeof(replace_node_list[0]));})

static inline uint32_t nn_align_up(uint32_t align_amt, uint32_t val)
{
	uint32_t minusone = align_amt - 1;
	return ((val + minusone) & (~minusone));
}


//
// utilites for checking nodes
//  (can be called from 'check' functions)
//

// check if #inputs in range min_no .. max_no; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.
// max_no < 0 can be used to indicate that extra inputs may be NULL;
// e.g. min_no =2, max_no = -5 means inputs must be in range 2..5, and inputs 0,1 may not be
// null, but inputs 2,3,4 may be NULL; caller will need to check.
//
int node_check_inputs_range( struct nn_node *self, struct nn_graph *nn, char const *name, int32_t min_no, int32_t max_no);
// check if #inputs =n; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.
int node_check_inputs_n( struct nn_node *self, struct nn_graph *nn, char const *name, int32_t n);
// check if #outputs in range min_no .. max_no; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.
// max_no < 0 can be used to indicate that extra outputs may be NULL;
// e.g. min_no =2, max_no = -5 means outputs must be in range 2..5, and outputs 0,1 may not be
// null, but inputs 2,3,4 may be NULL; caller will need to check.
int node_check_outputs_range( struct nn_node *self, struct nn_graph *nn, char const *name, int32_t min_no, int32_t max_no);
// check if #outputs =n; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.
int node_check_outputs_n(  struct nn_node *self, struct nn_graph *nn, char const *name, int32_t n);
// check if #inputs = n_in, and outputs = n_out; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.
int node_check_inputs_outputs_n(  struct nn_node *self, struct nn_graph *nn, char const *name, int32_t n_in, int32_t n_out);


#ifdef H2_H

#define RESET_PMU() __asm__ __volatile__ (" r0 = #0x48 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DUMP_PMU() __asm__ __volatile__ (" r0 = #0x4a ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DISABLE_PMU() __asm__ __volatile__ (" r0 = #0x42 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define ENABLE_PMU() __asm__ __volatile__ (" r0 = #0x41 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")

#else

#ifndef RESET_PMU
#define RESET_PMU() /* NOTHING */
#endif
#ifndef DUMP_PMU
#define DUMP_PMU() /* NOTHING */
#endif
#ifndef DISABLE_PMU
#define DISABLE_PMU() /* NOTHING */
#endif
#ifndef ENABLE_PMU
#define ENABLE_PMU() /* NOTHING */
#endif

#endif

#include <nn_graph_log.h>
#include <nn_graph_padding.h>
#include <nn_graph_allocator.h>
#include <nn_graph_find_node.h>

#endif
