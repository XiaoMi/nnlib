
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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

#include <nn_graph_builtin.h>
#include <nn_graph_ops.h>
#include <nn_graph_types.h>
#include <nn_graph_im2col.h>
#include <nn_graph_if.h>
#include <nn_asm_ops.h>
#include <nn_graph_os.h>
#include <nn_atomic.h>
#include <platform.h>
#include <stdio.h>
#include <nn_graph_earlywork.h>

#define likely(cond)	(__builtin_expect(!!(cond), 1))
#define unlikely(cond)	(__builtin_expect(!!(cond), 0))

struct nn_node_ops;
struct freelist_node;

typedef uint32_t noderefhash_set_t;

struct nn_node {
	struct nn_node_ops *ops;	// Operations for this NN
	const struct tensor **inputs;	// Inputs
	struct tensor **outputs;	// Outputs
	struct input *input_refs;	// References to node outputs
	struct output *output_defs;	// Output definitions
	uint32_t n_inputs;		// Number of inputs
	uint32_t n_outputs;		// Number of outputs
	noderefhash_set_t noderefhash;     // 'or' of noderefhash_mask( input_refs[i].src_id ) for all inputs
	uint32_t node_type;		// node type
	uint32_t node_id;		// node ID
	padding_type padding;		// kind of padding
	struct nn_node *next;		// ptr to next node
	void *opaque;			// whatever the node wants to put here
	uint32_t flags;
	uint32_t executions;		// how many times the node executes
	uint32_t refs;			// time op was referenced by any output
	uint64_t perfcounter;		// performance counter
	uint64_t iter_cycles;		// cycles consumed in last execution
};

enum nn_node_flags {			// 'flags' field
	NN_NODE_FLAG_RETAIN = (1<<0),		// don't remove this node in prepare, even if it has no consumers (set in ctor)
										// RETAIN is set if n_outputs==0, also for things like Variable and Assign.
	NN_NODE_FLAG_NO_CONVERT_D32 = (1<<1), // don't convert to d32. Used for nodes generated e.g. by metanodes.
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
////////////////////////// Batch Sequence data structures ////////////////

struct nn_batchseq_portdesc
{
	uint32_t batchsize;			// the batch size on this port
};

//
// nn_graph_batchseqstate (member 'batchseq' of struct nn_graph)
// When not iterating, this will be entirely 0
//
struct nn_graph_batchseqstate {
	uint8_t have_batchseqconf_yet;			// set when BatchSeqConfig is encountered (check for 2x)
	uint32_t  graph_batches;			// # of batches graph is built for.
	uint32_t  batch_quant;				// preferred multiple of batches ( or 1 if none; factor of graph_batches);
	uint32_t  options;
	
	// variables set up when we start a run (in INPUT node).
	uint32_t    iterno;					// 0 ... n_iters-1; the iteration #
	uint32_t	n_iters;				// # of iterations.
	uint32_t	total_batches;			// total # batches we are doing
	uint32_t	batchn;					// # of batches this run (set in INPUT node)
	uint32_t	batchoffs;				// # of batches prior to this run (updated in exec loop)
	// when n_iters_1 < n_iters, the first n_iters1 runs are of size batch_n1, and the rest are of size batch_n2.
	// (otherwise all are batch_n1)
	uint32_t	n_iters_1;
	uint32_t	batch_n1, batch_n2;		// two batch sizes that may be used in the run

	// some arrays
	//
	int32_t n_dimsel_in;			// # of elements in dimsel_in
	int32_t * dimsel_in;			// array of dimsel
	int32_t n_dimsel_out;			// # of elements in dimsel_out
	int32_t * dimsel_out;			// array of dimsel
	
	// the inseq_desc and outseq_desc arrays are allocated the first time INPUT and OUTPUT run,
	// if bath sequencing is in effect.
	// they are equal in size to the # of outputs of INPUT, and # of inputs of OUTPUT.
	struct nn_batchseq_portdesc *inseq_desc;
	struct nn_batchseq_portdesc *outseq_desc;
};
// some methods of the nn_graph_iterstate object:
static inline void nn_batchseqstate_init( struct nn_graph_batchseqstate *p) { memset(p, 0, sizeof(struct nn_graph_batchseqstate));}
static inline void nn_batchseqstate_free( struct nn_graph_batchseqstate *p) {
	if( p->outseq_desc != NULL ) nn_free( p->outseq_desc );
	if( p->inseq_desc != NULL ) nn_free( p->inseq_desc );
	if( p->dimsel_out != NULL ) nn_free(p->dimsel_out);
	if( p->dimsel_in != NULL ) nn_free(p->dimsel_in);
}	

// this is always done before each 'outer' execute operation, to make sure it starts properly.
static inline void nn_batchseqstate_before_outer_exec( struct nn_graph_batchseqstate *p)
{
	p->n_iters = 0;		// forget that we are iterating
	p->iterno = 0;		// reset itercount
	p->batchoffs = 0;
}

// this is used at the end of do..while in execute operation
//
static inline int nn_batchseqstate_loop_update(struct nn_graph_batchseqstate *p)
{
	int iterno = p->iterno+1;
	if( iterno >= p->n_iters ) return 0;		// all done
	p->iterno = iterno;					// update iter count
	p->batchoffs  += p->batchn;			// update batch offset
	if( iterno == p->n_iters_1 ){		// change batches/loop?
		p->batchn = p->batch_n2;
	}
	return 1;
}

////////////////////////////////////////

// graph->pstate is NULL, except during
// prepare where it points to an 'nn_prepare_state',
// this is used to e.g. cache constants
//
struct nn_prepare_state;

struct nn_graph {
	struct nn_node *head;		// First node in graph list
	struct nn_node *tail;		// 'weak' tail pointer
								// (may be null or not last; usually last though).
	uint32_t node_count;		// may be inaccurate (use to e.g. estimate hash size)

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
	struct nn_prepare_state *pstate;
	void *vtcm_ptr;			// ptr to VTCM
	uint32_t vtcm_size;		// size of VTCM
	struct nn_node **nonconst_head_ptr;	// ptr to head of non-const nodes
	void *find_node_opaque;
	void *fake_vtcm_ptr;
	// this is the 'or' of all the node->ops->flags values (anded with NN_NODE_FLAGS_SET)
	// which can be used to skip optimization passes (if no nodes exist of a given class,
	// the flag for the class will be zero). Only valid during prepare phase.
	uint32_t op_class_set;
	int32_t priority;
	struct nn_graph_batchseqstate batchseq;
	
	// Commandline options
	int enable_graph_print;
	int enable_tensor_print;
	char *enable_graph_print_prefix;
	char *enable_tensor_print_prefix;
	char *tensor_print_filter;
};

// this sets the noderefhash field on a node. Call after changing src_id
void node_rehash_inputrefs( struct nn_node *);

// convert a node_id to a 'noderefhash_set_t' value with exactly 1 bit set
static inline noderefhash_set_t
noderefhash_mask( uint32_t node_id ){
    uint32_t prod = node_id * 2654435761U;
    return 1u << ( prod >> (32-5));
}

//////////////////////////////////////////////
// mechanism for issuing a series of memcpys; this can be done from a non-vector
// thread, and all of the memcpy (at least the big ones) are done in vector
// threads; the mechanism does not need to be used in a vector thread.
//
//  struct nn_memcpy_manager  mcman;
//  nn_mcmanager_init(nn, &mcman );
//    ... as many as you like of ..
//        nn_mcmanager_vmemcpy( nn, &mcman, dst, src, len );
//        nn_mcmanager_vmemset32( nn, &mcman, dst, val32, len );
//        nn_mcmanager_tensor_copy( nn, &mcman, dst_tensor, src_tensor );
//        nn_mcmanager_vmemcpy_2d( nn, &mcman, width, height, dst, dst_stride, src_stride );
//        nn_mcmanager_vmemset32_2d( nn, &mcman, dst, val , width, height, dst_stride );
//  .. and then finally 
//  nn_mcmanager_wait( nn, &mcman );
//  ... which waits until all are done. This *must* done before mcman goes
//  out of scope.
//  You can now let mcman go out of scope, or start with more copies.
//
//  The various memcpy/memset operations done before 'wait' may be done in parallel
//  threads and may be deeply reordered. No more than NN_MEMCPY_MANAGER_MAX_THREADS
//  will be outstanding at any time.
// Note: the API is not itself thread-safe; the init, copy requests, and wait must
// all be done in the same thread.
//
// regarding nn_mcmanager_vmemset32: the 'value' is 32 bits, but the 'len' is in
// bytes; and the len can be any number, start pointer any alignment.
//
#define NN_MEMCPY_MANAGER_MAX_THREADS 2		// max # of pending ops.
#define NN_MEMCPY_MANAGER_SLOTS 4			// of slots in struct (>= MAX_THREADS+2)

// this encodes one of 3 operations:
//  if src != NULL, rows = 0:
//          vmemcpy( dst,src, len )
//  if src != NULL, rows >=2:
//      2-d memcpy rows x len; src_stride = val; dst_stride = dst_stride
//  if src = NULL:
//      2-d memset32 @ dst with dst_stride, max(1,rows) x len.
struct nn_memcpy_manager_op {
	void *dst;
	void const *src;
	unsigned len;
	unsigned val;
	unsigned rows;
	unsigned dst_stride;
};
struct nn_memcpy_manager
{
	nn_sem_t done_sem;			// issued after each completion
	int pending;				// # of wait(done_sem) pending
	unsigned avglen;			// rough 'running average' of operation lengths.
	volatile unsigned avail_set;			// bit set of available operation slots
	volatile unsigned ready_set;			// slots ready but not picked up by run thread
	struct nn_memcpy_manager_op ops[NN_MEMCPY_MANAGER_SLOTS];
};
void nn_mcmanager_init(struct nn_graph *nn, struct nn_memcpy_manager * );
void nn_mcmanager_vmemcpy_or_set(struct nn_graph *nn, struct nn_memcpy_manager *,
	void *dst, void const * src, unsigned len, unsigned fillval );
void nn_mcmanager_wait(struct nn_graph *nn, struct nn_memcpy_manager *);

static inline void nn_mcmanager_vmemcpy(struct nn_graph *nn, struct nn_memcpy_manager *mcm,
	void *dst, void const * src, unsigned len )
{
	nn_mcmanager_vmemcpy_or_set( nn, mcm, dst, src, len, 0);
}
static inline void nn_mcmanager_vmemset32(struct nn_graph *nn, struct nn_memcpy_manager *mcm,
	void *dst, unsigned fillval, unsigned len )
{
	nn_mcmanager_vmemcpy_or_set( nn, mcm, dst, NULL, len, fillval);
}
//
// this is wrapper for nn_mcmanager_vmemcpy, to handle tensors
// Returns -1 if the output is too small.
//
int nn_mcmanager_tensor_copy(struct nn_graph *nn, struct nn_memcpy_manager *mcm,
		struct tensor *dst, const struct tensor *src);


// 2D operations via memset
//
// ** this is an internal function, to implement the 2d memcpy and fill.
void
nn_manager_vmemcpy_or_set_2d( int width, int height, void* dst, void const * src,
	unsigned dst_stride, unsigned src_stride_or_fillval, struct nn_graph *nn, struct nn_memcpy_manager *mcm);

static inline void nn_mcmanager_vmemcpy_2d(struct nn_graph *nn, struct nn_memcpy_manager *mcm,
		int width, int height, void * dst, unsigned dst_stride, void const * src, unsigned src_stride)
{
	if( src != NULL) // don't allow it to look like a 2d fill
		nn_manager_vmemcpy_or_set_2d( width, height, dst, src, dst_stride, src_stride, nn, mcm );
}

static inline void nn_mcmanager_vmemset32_2d(struct nn_graph *nn, struct nn_memcpy_manager *mcm,
		 void * dst, unsigned fillval, int width, int height,unsigned dst_stride)
{
	if( height >0)
		nn_manager_vmemcpy_or_set_2d( width, height, dst, NULL, dst_stride, fillval, nn, mcm );
}

//////////////////////////////////////////////

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
void print_graph_to_file(struct nn_graph *nn);
void print_node_checksum(struct nn_graph *nn, struct nn_node *node);
void print_graph_checksum(struct nn_graph *nn);

#define SCRATCH_SIZE (1024*1024*8)
#define LOGBUF_SIZE (1024*512)

#define NN_NODE_FLAG_D32_INPUT (1<<0)
#define NN_NODE_FLAG_D32_OUTPUT (1<<16)
#define NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION (1<<17)
//
// This flag means that the output range will be the same as the input,
// *and* that you can move a range-limiting op such as Relu from after the op to
// before, without changing the result (so it e.g. applies to maxpool but not avgpool)
#define NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE (1<<18)

// some flags on certain ops to put them in 'classes'; if there is no
// op in a given class in the graph we can skip entire passes in prepare.
//
#define NN_NODE_FLAG_CLS_QUANTIZE 		(1u<<31)		// Quantize
#define NN_NODE_FLAG_CLS_REQUANTRANGE 	(1<<30)	// RequantizationRange_32
#define NN_NODE_FLAG_CLS_QUANTMUL8TO32  (1<<29)  // QuantizeMul_8x8to32
#define NN_NODE_FLAG_CLS_DWCONVF        (1<<28)  // DepthwiseConv2d_f
#define NN_NODE_FLAG_CLS_CHANSHUFFLE  	(1<<27)	// QuantizedChannelShuffle_8
#define NN_NODE_FLAG_CLS_OEMNODE        (1<<26)  // OemNode
#define NN_NODE_FLAG_CLS_SUPPORTS_ALIAS (1<<25)		// Supports output 0 stored at same address as input 0 (e.g. reshape)

// set of all 'classes' flags
#define NN_NODE_FLAGS_SET\
	(NN_NODE_FLAG_CLS_QUANTIZE|NN_NODE_FLAG_CLS_REQUANTRANGE|NN_NODE_FLAG_CLS_QUANTMUL8TO32\
	|NN_NODE_FLAG_CLS_DWCONVF|NN_NODE_FLAG_CLS_CHANSHUFFLE|NN_NODE_FLAG_CLS_OEMNODE|NN_NODE_FLAG_CLS_SUPPORTS_ALIAS)

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
	int (*earlywork_note_pred)(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor);
	int (*earlywork_register)(struct nn_node *self, struct nn_graph *nn, struct nn_early_work *work);
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

int do_variable_read(
	struct nn_graph * nn,
	uint32_t node_id,
	int output_index,
	uint32_t *b_out,
	uint32_t *h_out,
	uint32_t *w_out,
	uint32_t *d_out,
	uint8_t *data_out,
	uint32_t data_out_max,
	uint32_t *data_out_len);

int do_variable_write(
	struct nn_graph *nn,
	uint32_t node_id,
	int output_index,
	uint32_t b,
	uint32_t h,
	uint32_t w,
	uint32_t d,
	const uint8_t *data_in,
	uint32_t data_in_size);

int do_variable_write_flat(
	struct nn_graph *nn,
	uint32_t node_id,
	int output_index,
	const uint8_t *data_in,
	uint32_t data_in_size);




int do_perfinfo_reset(struct nn_graph *nn, uint32_t event);
int do_perfinfo_get(struct nn_graph *nn, struct perfinfo *info, uint32_t info_len);


// Convert an nn_id_t to a pointer to struct nn_graph
// (this subverts the API, but some test frameworks which are linked to the
// library need to do this)
// a NULL return means the graph id is not valid.
struct nn_graph * nn_id_to_graph( nn_id_t id );


int node_free_common(struct nn_node *node, struct nn_graph *nn);
// if you have a (possibly null) opaque pointer which just needs to be free'd, this
// can be the node dtor.
int node_free_common_release_opaque(struct nn_node *node, struct nn_graph *nn );

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
void set_last_consumers(struct nn_graph *nn);
struct nn_node* find_first_producer(
	struct nn_graph *nn,
	struct nn_node *producer);
struct nn_node* find_first_consumer(
	struct nn_graph *nn,
	struct nn_node *producer,
	int out_idx);
enum { CONSUMER_NOINCHECK=1 };
struct nn_node* find_unique_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
    int req_node_type,
    const int *node_types,
	int options);

static inline
struct nn_node* find_unique_consumer_mustbe(
	struct nn_graph *nn, 
	struct nn_node *producer, 
    int req_node_type,      // must be this type
	int options)
{ return find_unique_consumer( nn, producer, req_node_type, NULL, options );
}
static inline
struct nn_node* find_unique_consumer_anytype(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int options)
{ return find_unique_consumer( nn, producer, -1, NULL, options );
}

//
// These functions create a new 'const' node with the given int or float value.
// They are intended to be used during the 'prepare' phase; they use a caching
// mechanism to avoid repeated nodes of the same value.
//
uint32_t create_const_int32_op(struct nn_graph *nn, int32_t const_float);
uint32_t create_const_float_op(struct nn_graph *nn, float const_float);

// This is used in 'prepare' phase; make a new const node and place it
// at the *top* of the list.
//
int do_prepend_const_node(struct nn_graph *nn,
		uint32_t node_id,
		uint32_t batches, uint32_t height, uint32_t width, 	uint32_t depth,
		const uint8_t *data, uint32_t data_len);

//
// returns 0 iff
//  - 'consumer' has at least one input connected to an output of producer
//  - no other node has inputs connected to connected to producer
// returns -1 otherwise
//
int check_single_consumer_all(
	struct nn_graph *nn, 
	struct nn_node *producer, 
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
// 'dtor' is called on all removed nodes.
//
// if anchor is NULL, &nn->head is used.
// if new_node is NULL, only the removal of nodes in rmnodes is done.
// if any of the rmnodes[1..n_remove-1] is NULL, it is ignored
//  terminator (but rnodes[0] must not be NULL)
//

int replace_node_sequence (
		struct nn_graph *nn,
		struct nn_node ** anchor,	// list anchor, at items[0] or upstream
		struct nn_node * new_node,	// node to replace (may be null)
    	struct nn_node * const * rmnodes,
        int n_remove);
// 'varargs' version of replace_node_sequence: the '...' parms become the rmnodes array.
// Evaluates to return value of replace_node_sequence.
//
#define replace_nodes(NN,ANCHOR,NEWNODE,...)\
 ({ struct nn_node *replace_node_list[] = { __VA_ARGS__}; \
   replace_node_sequence(NN,ANCHOR,NEWNODE, replace_node_list, sizeof(replace_node_list)/sizeof(replace_node_list[0]));})


//
// replace a single node 'delnode' with a sequence of 0 or more nodes
//    addnodes[0..n_add-1].
// -  'inspos' is where we start looking for delnode (default &nn->head, if NULL)
// -  if delnode is NULL, we don't remove anything and we just insert all the nodes
//     after *inspos.
// - It is ok for some (or all) of the addnodes to be NULL, these are just ignored.
//   The remainder are inserted in sequence.
//
// returns -1 if delnode !=NULL and was not found; otherwise returns # inserted nodes.
// 'dtor' is called on delnode, if not NULL.
//
int
replace_node_with_sequence(
		struct nn_graph *nn,
		struct nn_node ** inspos,
		struct nn_node * delnode,
    	struct nn_node * const * addnodes,
        int n_add);
// 'varargs' version of replace_node_with_sequence: the '...' parms become the addnodes array.
// Evaluates to return value of replace_node_with_sequence.
//

#define replace_node_with(NN,ANCHOR,DELNODE,...)\
 ({ struct nn_node *add_node_list[] = { __VA_ARGS__}; \
   replace_node_with_sequence(NN,ANCHOR, DELNODE, add_node_list, sizeof(add_node_list)/sizeof(add_node_list[0]));})

// insert_nodes(nn, posn, node1, node2...) just inserts them all at 'posn'.

#define insert_nodes(NN,ANCHOR,...)\
 ({ struct nn_node *add_node_list[] = { __VA_ARGS__}; \
   replace_node_with_sequence(NN,ANCHOR, NULL, add_node_list, sizeof(add_node_list)/sizeof(add_node_list[0]));})

//
// change refs to outputs of nodeid 'old_nodeid' to refs to 'new_nodeid'.
// 'pattern' is a mapping table encoded in nybbles, when a reference to 'old_node_id' output 'k'
// is found, we look at nybble k (indexed starting from lsb), and
//   nybble = 0:			=> error, there should be no reference to this output of old_node_id
//   nybble = 0xF:			=> leave this one as is
//   nybble = n = 1..0xE:	=> change the reference to be to 'new_nodeid' output n-1
// an output index >= 16 is considered an error (we don't have that many nybbles in pattern)
//
// returns:
//  -1 if error (all 'ok' replacements are done anyway)
//  otherwise, number of replacements done, >= 0.
//
// Example:
//   pattern = 0x321  -> remap refs to outputs (0,1,2), to new_nodeid outputs (0,1,2); inputs >=3 will be error
//   pattern = 0xFFF3 -> remap refs to output 0 to output 2 of new_nodeid; leave refs to 1,2,3 unchanged; >= 4 will be error.
//
int
change_output_refs( struct nn_graph * nn,
		struct nn_node * begin,			// can specify the 'old_nodeid' node to speed things up; or NULL.
		uint32_t old_nodeid,
		uint32_t new_nodeid,
		uint64_t pattern );

//
// this is for rewiring the consumers of a 'Split' or 'ChannelShuffle' to point at the
// Convert_from_d32 nodes which have been attached to its outputs.
// 'newnodes' is an array of pointers to those conversion nodes.
//
// Operation is:
//
//   -- look for all node input refs of the form { old_nodeid, idx }
//          where idx is in range 0 .. n_outputs-1
//   -- change these to { newnodes[idx]->node_id, 0 }
// in cases where newnodes[idx] is NULL, the references are not changed.
//
int
change_multi_output_refs( struct nn_graph * nn,
		struct nn_node * begin,			// can specify the 'old_nodeid' node to speed things up; or NULL.
		uint32_t old_nodeid,
		int n_outputs,					// number of outputs to rewire
		struct nn_node const * newnodes[] );  // array of new nodes to point to [0..n_outputs-1]

//
// this is for rewiring when a single node has been arbitrarily replaced
// by several nodes which supply the signals previously generated by 'old_nodeid'.
//
// Operation is:
//
//   -- look for all node input refs of the form { old_nodeid, idx }
//          where idx is in range 0 .. n_outputs-1
//   -- change these to new_inpref[idx],
// in cases where newnodes[idx].src_id is 0, the references are not changed.

int
change_multi_output_refs_table( struct nn_graph * nn,
		struct nn_node * begin,			// can specify the 'old_nodeid' node to speed things up; or NULL.
		uint32_t old_nodeid,
		int n_outputs,					// number of outputs to rewire
		struct input const new_inpref[] );  // array of new

static inline uint32_t nn_align_up(uint32_t align_amt, uint32_t val)
{
	uint32_t minusone = align_amt - 1;
	return ((val + minusone) & (~minusone));
}

//
// special ctors in op_const.c
extern struct nn_node *hexagon_nn_const_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len);

extern struct nn_node *hexagon_nn_empty_const_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	uint32_t data_len);

extern int hexagon_nn_populate_const(
	struct nn_graph *nn,
	uint32_t node_id,
	const uint8_t *data,
	uint32_t data_len,
	uint32_t target_offset);

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

enum {
	GRAPHCHECK_DEADNODES =1,	// report nodes which have outputs but no reference
	GRAPHCHECK_HASH = 2,		// check hash table
    GRAPHCHECK_NONCONST= 4,     // check NONCONST partitioning
};

#if !defined(NN_LOG_MAXLEV) || NN_LOG_MAXLEV >= 1
int check_graph( struct nn_graph *nn, int options);
#else
static inline int check_graph( struct nn_graph *nn, int options){ return 0; }
#endif

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
#include <nn_graph_memcpy.h>

#endif
