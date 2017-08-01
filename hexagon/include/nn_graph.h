
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
#include <platform.h>

#define likely(cond)	(__builtin_expect(!!(cond), 1))
#define unlikely(cond)	(__builtin_expect(!!(cond), 0))

struct nn_node_ops;
struct freelist_node;

struct nn_node {
	struct nn_node_ops *ops;	// Operations for this NN
	const struct tensor **inputs;	// Inputs
	struct tensor **outputs;	// Outputs
	struct input *input_refs;	// References to node outputs
	uint32_t n_inputs;		// Number of inputs
	uint32_t n_outputs;		// Number of outputs
	uint32_t node_type;		// node type
	uint32_t node_id;		// node ID
	padding_type padding;		// kind of padding
	struct nn_node *next;		// ptr to next node
	void *opaque;			// whatever the node wants to put here
	uint32_t executions;
	uint64_t perfcounter;
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
	void *scratch;			// temporary storage
	size_t scratch_size;		// size of scratch
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
};

static inline uint32_t nn_graph_new_internal_node_id(struct nn_graph *nn)
{
	return --(nn->internal_node_id);
}

static inline void record_usertime(struct nn_graph *nn, struct nn_node *op, uint32_t type, uint64_t time)
{
	if (nn->perf_event == type) op->perfcounter = time;
}

#define SCRATCH_SIZE (1024*1024*192)
#define LOGBUF_SIZE (1024*1024*2)

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
};
extern struct nn_node_ops *optab[];

int do_execute(
	struct nn_graph *nn,
	const struct tensor *inputs,
	uint32_t n_inputs,
	struct tensor *outputs,
	uint32_t n_outputs);
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



static inline uint32_t nn_align_up(uint32_t align_amt, uint32_t val)
{
	uint32_t minusone = align_amt - 1;
	return ((val + minusone) & (~minusone));
}

#ifdef H2_H

#define RESET_PMU() __asm__ __volatile__ (" r0 = #0x48 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DUMP_PMU() __asm__ __volatile__ (" r0 = #0x4a ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define DISABLE_PMU() __asm__ __volatile__ (" r0 = #0x42 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")
#define ENABLE_PMU() __asm__ __volatile__ (" r0 = #0x41 ; trap0(#0); \n" : : : "r0","r1","r2","r3","r4","r5","r6","r7","memory")

#else

#define RESET_PMU() /* NOTHING */
#define DUMP_PMU() /* NOTHING */
#define DISABLE_PMU() /* NOTHING */
#define ENABLE_PMU() /* NOTHING */

#endif

#include <nn_graph_log.h>
#include <nn_graph_padding.h>
#include <nn_graph_allocator.h>

#endif
