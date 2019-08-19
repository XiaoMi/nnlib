
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
#ifndef NN_GRAPH_LOOPING_H
#define NN_GRAPH_LOOPING_H 1

// #include <nn_graph.h>  // included in nn_graph.h!



#define NN_MAX_LOOPSTACK 4		// max # of loop control nodes in graph
#define NN_MAX_OUTPUTS 10		// max # of outputs supported by looping

struct nn_graph;
struct nn_node;


struct nn_loopcounts {
	uint32_t itercount;		    // no. of iterations, not including current
	uint32_t prev_batches;	    // no. of batches previously generated in loop
	uint32_t current_batches;	// no. of batches in current iterations
	uint32_t total_batches;		// if prev + current >= total, this is last run.
	uint32_t prev_outer_batches;	// previous batches in outer loops
	uint32_t next_batches;
	uint32_t offsets[NN_MAX_OUTPUTS];
};
typedef int (*nn_loopend_fp)( struct nn_graph *nn, struct nn_node *node,
			struct nn_loopcounts *counts, void *opaque);


struct nn_loopstack_entry {
	nn_loopend_fp loopend_function;
	struct nn_node * nodep;
	void *opaque;
	struct nn_loopcounts counts;
};

struct nn_loopstack {
	int n;
	struct nn_loopstack_entry entries[NN_MAX_LOOPSTACK+1];
};

////////////////////////// API for 'Loop Control Node' to use /////////////////////
// API to read the current loop state
struct nn_loopcounts const * nn_loopstack_get_counts(struct nn_graph const *nn);
uint32_t nn_loopstack_get_itercount(struct nn_graph const *nn);
uint32_t nn_loopstack_get_prev_batches(struct nn_graph const *nn);
uint32_t nn_loopstack_get_current_batches(struct nn_graph const *nn);
uint32_t nn_loopstack_get_offset(struct nn_graph const *nn, uint32_t index);
uint32_t nn_graph_output_expanded(struct nn_graph const *nn, uint32_t index);
void nn_loopstack_set_next_batches(struct nn_graph *nn, uint32_t size);
void nn_loopstack_increment_offset(struct nn_graph *nn, uint32_t index, uint32_t size);
void nn_graph_set_output_expanded(struct nn_graph *nn, uint32_t index);

// API to push a loop using default loopend_function
// loop-control node .execute() must always call one of these two functions.
// The total_batches can be an estimate;
//   if prev_batches + current_batches < total_batches, the LCN
//   will be executed again, with prev_batches updated; if not,
//   it will not be executed again.

// push using standard loop-end function

int nn_loopstack_push( struct nn_graph *nn, struct nn_node * self,
	unsigned current_batches,
	unsigned total_batches);

// API to push a loop using specified loopend_function
int nn_loopstack_push_withfunc( struct nn_graph *nn,  struct nn_node * self,
	unsigned current_batches,
	unsigned total_batches,
	nn_loopend_fp loopend_function,
	void *opaque);

// default loop-end function
//
int nn_loopend_default(struct nn_graph *nn, struct nn_node *node, struct nn_loopcounts *counts, void *opaque);

////////////////////////////////////////////////////////////////////////////////
// The pass in 'prepare' which checks the graph and makes any adjustments needed
//
int nn_graphloop_prepare_graph( struct nn_graph * nn);
int nn_dynamictensor_prepare_graph( struct nn_graph * nn);

////////////////////////////////////////////////////////////////////////////////
//
//  Internal API for execute function to use
//
static inline void nn_loopstack_pre_execute(struct nn_graph *nn, struct nn_loopstack *nnloops )
{
	memset( nnloops, 0, sizeof(struct nn_loopstack));
}

// call this at end of graph.
//   if errcode != 0: stop with error
//   else if rerun_node is NULL:
//      graph is done
//   else
//      go back up to rerun_node.
// It is safe to call this at the end of any graph ( returns {0,NULL} when loopstack is empty)
//
struct nn_loop_end_action {
	int errcode;
	struct nn_node * rerun_node;
};
struct nn_loop_end_action nn_loopstack_post_execute_slowpath( struct nn_graph *nn);

static inline struct nn_loop_end_action
nn_loopstack_post_execute( struct nn_graph *nn, struct nn_loopstack *nnloops )
{
	if( nnloops->n <= 0){
		struct nn_loop_end_action result={ 0, NULL};
		return result;
	}else{
		return nn_loopstack_post_execute_slowpath(nn);
	}

}



#endif // NN_GRAPH_LOOPING_H

