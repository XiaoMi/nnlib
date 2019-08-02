
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

#include "nn_graph.h"
#include "nn_prepare.h"



struct nn_loop_end_action
nn_loopstack_post_execute_slowpath( struct nn_graph *nn)
{
	int n = nn->loopstack.n;	// function is only called when this is >0.
	struct nn_loopstack_entry *entp = &nn->loopstack.entries[n-1];

	struct nn_loop_end_action result = {0, NULL};
	// call the loop-end function
	struct nn_node * nodep = entp->nodep;
	int res = (*entp->loopend_function)( nn, nodep, &entp->counts, entp->opaque);
	if( res < 0 ) {
		result.errcode = res;
		return result;
	}
	if( res > 0) {
		result.rerun_node = nodep;
	}
	else{
		nn->loopstack.n--;
	}
	return result;

}

/////////////////////////////////////////////////////////////////////////////////////
// functions to read loop stack
//
struct nn_loopcounts const * nn_loopstack_get_counts(struct nn_graph const *nn){
	return & nn->loopstack.entries[nn->loopstack.n-1].counts;
}
uint32_t nn_loopstack_get_itercount(struct nn_graph const *nn)
{
	return nn->loopstack.entries[nn->loopstack.n-1].counts.itercount;
}
uint32_t nn_loopstack_get_prev_batches(struct nn_graph const *nn)
{
	return nn->loopstack.entries[nn->loopstack.n-1].counts.prev_batches;
}
uint32_t nn_loopstack_get_current_batches(struct nn_graph const *nn)
{
	return nn->loopstack.entries[nn->loopstack.n-1].counts.current_batches;
}
uint32_t nn_loopstack_get_offset(struct nn_graph const *nn, uint32_t index){
	return nn->loopstack.entries[nn->loopstack.n-1].counts.offsets[index];
}
uint32_t nn_graph_output_expanded(struct nn_graph const *nn, uint32_t index){
	return nn->expanded_outputs[index];
}
void nn_loopstack_set_next_batches(struct nn_graph *nn, uint32_t size){
	nn->loopstack.entries[nn->loopstack.n-1].counts.next_batches = size;
}
void nn_loopstack_increment_offset(struct nn_graph *nn, uint32_t index, uint32_t size){
	nn->loopstack.entries[nn->loopstack.n-1].counts.offsets[index] = size;
}
void nn_graph_set_output_expanded(struct nn_graph *nn, uint32_t index){
	nn->expanded_outputs[index] = 1;
}

// standard loop-end function
//
int
nn_loopend_default(struct nn_graph *nn, struct nn_node *node,
			struct nn_loopcounts *counts, void *opaque)
{
	uint32_t new_batches = counts->prev_batches + counts->current_batches;
	if ( new_batches < counts->total_batches ){
		counts->itercount ++;
		counts->prev_batches = new_batches;
		counts->current_batches = counts->next_batches;
		return 1;			// keep looping
	}else{
		counts->prev_outer_batches += new_batches;
		counts->itercount = 0;
		counts->prev_batches = 0;
		counts->next_batches = 0;
		return 0;				// done looping.
	}
}


// push using standard loop-end function

int nn_loopstack_push( struct nn_graph *nn, struct nn_node * self,
	unsigned current_batches,
	unsigned total_batches)
{
	return nn_loopstack_push_withfunc( nn,self, current_batches, total_batches, nn_loopend_default, NULL);
}

// API to push a loop using specified loopend_function
int nn_loopstack_push_withfunc( struct nn_graph *nn,  struct nn_node * self,
	unsigned current_batches,
	unsigned total_batches,
	nn_loopend_fp loopend_function,
	void *opaque)
{
	int n = nn->loopstack.n;

	if( n < 0 || n >= NN_MAX_LOOPSTACK) return errlog(nn,"loop stack error");	// stack invalid, or full
	struct nn_loopstack_entry *entp = &nn->loopstack.entries[n];

	logmsg(nn,2,"entering loop at lev=%d under control of node 0x%X, prev=%d, cur=%d, total=%d",
			n, (unsigned)self->node_id, (int)entp->counts.prev_batches, (int)current_batches, (int)total_batches);

	entp->loopend_function = loopend_function;
	entp->nodep = self;
	entp->opaque = opaque;
	entp->counts.current_batches = current_batches;
	entp->counts.total_batches = total_batches;
	nn->loopstack.n++;

	return 0;
}

/// 'inref_set' used in nn_graphloop_prepare_graph

// this lazily represents a set of 'struct input'.
// (lazy, because duplicates may not be dropped at insert time, but they
// will be dropped when you do a 'sort'.
//
// - other then 'init' and 'free',
//    the only operations are 'add' and 'sort'. 'sort' removes duplicates.
// - when 'add' is done, the new is discarded if it matches the previous last
//   entry. Otherwise it is appended, if there is room. So no two adjacent elements are ever
//   the same. If it doesn't fit, we try sorting, and if that doesn't make enough space, grow
//   the allocation.
// - the 'allocated' size (alloc) is never small, and only grows.
//

struct inref_set {
	struct input *tbl;
	int size;
	int alloc;
};
static  int inref_set_insert_slowpath(struct inref_set *tp, struct input const * newin );
static  void inref_set_sort(struct inref_set *tp );
static int nn_graphloop_sort_partitions( struct nn_graph * nn);

static inline int inref_set_init(struct inref_set *tp, int n ){
	tp->tbl = nn_malloc( n*sizeof(struct input));
	if( tp->tbl == NULL) return -1;
	tp->size = 0;
	tp->alloc = n;
	return 0;
}
static void inref_set_free( struct inref_set *tp)
{
	nn_free(tp->tbl);
}
static inline int inref_matches( struct input const * a, struct input const * b){
	return (a->src_id == b->src_id)
		&& (a->output_idx == b->output_idx);
}
static inline int inref_cmp( struct input const * a, struct input const * b){
	if (a->src_id == b->src_id){
		return (a->output_idx < b->output_idx)? -1: (a->output_idx > b->output_idx);
	}
	return (a->src_id < b->src_id)? -1: 1;
}

static inline int inref_set_insert(struct inref_set *tp, struct input const * newin ){
	if( tp->size > 0 &&  inref_matches( &tp->tbl[tp->size-1], newin ) ){
		return 0;	// discard
	}
	if( tp->size < tp->alloc){
		tp->tbl[tp->size++]= *newin;
		return 0;
	}
	return inref_set_insert_slowpath( tp, newin);
}

// inserting in a 'full list'.
static  int __attribute__((noinline))
inref_set_insert_slowpath(struct inref_set *tp, struct input const * newin )
{
	// first try 'sorting' the list
	inref_set_sort( tp);
	if( tp-> size >= tp->alloc){	// it didn't free up any room
		int add_to = (tp->alloc < 512)? tp->alloc: 512;
		int newalloc = tp->alloc + add_to;
		void * newmem = nn_realloc( tp->tbl, sizeof(struct input)*newalloc );
		if( newmem == NULL) return -1;
		tp->alloc = newalloc;
		tp->tbl = newmem;
	}
	if( !inref_matches( &tp->tbl[tp->size-1], newin) ){
		tp->tbl[tp->size++]= *newin;
	}
	return 0;
}
// sort and eliminate dups.

static  void __attribute__((noinline))
inref_set_sort(struct inref_set *tp )
{
	int n = tp->size;
	if( n < 2) return;
	struct input *tbl = tp->tbl;

	// put the first two in order
	if( inref_cmp( &tbl[0], &tbl[1])>0){
		struct input tmp = tbl[0]; tbl[0] = tbl[1], tbl[1] = tmp;
	}
	int nsorted = 2;
	for( int ns = 2; ns < n; ns++){
		// binary search to figure out where element [ns] goes in the first
		// nsorted elements. If it is not already there, insert it in the
		// proper position.
		struct input newval = tbl[ns];
		int ilo = 0;
		int ihi = nsorted;
		while( ilo < ihi ){
			int imid = (ilo+ihi)>>1;
			int cmp = inref_cmp( &newval, &tbl[imid] );
			if( cmp < 0){
				ihi = imid;
			}else if( cmp != 0){
				ilo = imid+1;
			}else{
				break;				// found a duplicate
			}
		}
		if( ilo == ihi){			// 0..nsorted: insert before this point.
			if( ilo < nsorted){
				memmove( &tbl[ilo+1], &tbl[ilo], (nsorted-ilo)*sizeof(struct input));
			}
			tbl[ilo] = newval;
			nsorted++;
		}
	}
	tp->size = nsorted;
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
// Special graph-prepare pass when 'loop control' nodes are present
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

static inline uint32_t compute_max_size_with_pad(
		struct nn_graph *nn,
		struct nn_node *newnode,
		const struct shape *shape,
		uint32_t elementsize,
		uint32_t output_index)
{
	uint32_t b = shape->batches;
	uint32_t h = shape->height;
	uint32_t w = shape->width;
	uint32_t d = shape->depth;
	uint32_t probably_scalar_float = ((b == 1) && (h == 1) && (w == 1) && (d == 1)
									  && (elementsize == 4) && (output_index > 0));
	if (!probably_scalar_float && (newnode->ops->flags & NN_NODE_FLAG_D32_OUTPUT)) {
		h = h + 8;
		w = (w + 7) & (~3);
		d = (d + 31) & (~31);
	}
	return (b*h*w*d*elementsize+127) & ~127;
}

int ensure_output_batch_size(struct nn_graph * nn, struct nn_node *nodep, int output_idx, int max_batch_size){
	int resized = 0;
	
    if(nodep->outputs[output_idx]->shape.batches == 0
	   || nodep->output_defs[output_idx].max_sizes[0] == 0){

        nodep->outputs[output_idx]->shape.batches = max_batch_size;
        nodep->output_defs[output_idx].max_sizes[0] = max_batch_size;
		
		resized++;
    }
	
    if(nodep->outputs[output_idx]->shape.height == 0
	   || nodep->output_defs[output_idx].max_sizes[1] == 0){

        nodep->outputs[output_idx]->shape.height = max_batch_size;
        nodep->output_defs[output_idx].max_sizes[1] = max_batch_size;
		
		resized++;
    }

    if(nodep->outputs[output_idx]->shape.width == 0
	   || nodep->output_defs[output_idx].max_sizes[2] == 0){

        nodep->outputs[output_idx]->shape.width = max_batch_size;
        nodep->output_defs[output_idx].max_sizes[2] = max_batch_size;
		
		resized++;
    }

    if(nodep->outputs[output_idx]->shape.depth == 0
	   || nodep->output_defs[output_idx].max_sizes[3] == 0){

        nodep->outputs[output_idx]->shape.depth = max_batch_size;
        nodep->output_defs[output_idx].max_sizes[3] = max_batch_size;
		
		resized++;
    }

	if(resized > 1){
		return errlog(nn,"more than one dynamic dimension");
	}

	if(resized){
		uint32_t new_size = compute_max_size_with_pad(nn,nodep,&nodep->outputs[output_idx]->shape,
													  nodep->output_defs[output_idx].elementsize,output_idx);

        if(nodep->outputs[output_idx]->max_size < new_size) {
            nodep->outputs[output_idx]->max_size = new_size;
            nodep->outputs[output_idx]->data_size = new_size;
        }
	}
	
	return 0;
}

// look for nodes that have dynamic output size and widen the 0-sized dimensions of all nodes after them
int
nn_dynamictensor_prepare_graph( struct nn_graph * nn)
{
    struct nn_node **nodepp = nn->nonconst_head_ptr;		// ptr to pointer to first
    struct nn_node * nodep;
    struct nn_node * output_node = NULL;

    memset( nn->expanded_outputs, 0, NN_MAX_OUTPUTS*sizeof(uint32_t));

    for( ; (nodep = *nodepp) != NULL; nodepp = &nodep->next ){
        if( nodep->node_type == OP_OUTPUT){
            output_node = nodep;
            break;
        }
    }

    if(output_node != NULL && output_node->node_type == OP_OUTPUT){
        for(int i = 0; i < output_node->n_inputs; i++){
            struct nn_node * input_node = find_node( nn, output_node->input_refs[i].src_id);
            uint32_t idx = output_node->input_refs[i].output_idx;
            if(input_node->outputs[idx]->shape.batches == 0
               || input_node->outputs[idx]->shape.height == 0
               || input_node->outputs[idx]->shape.width == 0
               || input_node->outputs[idx]->shape.depth == 0
               || input_node->output_defs[idx].max_sizes[0] == 0
               || input_node->output_defs[idx].max_sizes[1] == 0
               || input_node->output_defs[idx].max_sizes[2] == 0
               || input_node->output_defs[idx].max_sizes[3] == 0)
            {
                nn_graph_set_output_expanded(nn, i);
            }
        }
    }

    nodepp = nn->nonconst_head_ptr;		// ptr to pointer to first
	int max_batch_size = -1;
	for( ; (nodep = *nodepp) != NULL; nodepp = &nodep->next ){
		if( nodep->node_type == OP_BoxWithNmsLimit_q8q16){
			struct nn_node * limit_node = find_node( nn, nodep->input_refs[4].src_id);
			struct tensor *limit_tensor = limit_node->outputs[0];
			int* limit_data = limit_tensor->data;

			max_batch_size = limit_data[0];
		}
		else if(nodep->node_type == OP_Proposal_q8q16){
			struct nn_node * limit_node = find_node( nn, nodep->input_refs[15].src_id);
			struct tensor *limit_tensor = limit_node->outputs[0];
			int* limit_data = limit_tensor->data;

			max_batch_size = limit_data[0];
		}

		if(nodep->node_type != OP_Const && max_batch_size > 0){
			for(int i = 0; i < nodep->n_outputs; i++){
				int result = ensure_output_batch_size(nn, nodep, i, max_batch_size);
				if(result){
					return result;
				}
			}
		}
	}
	
	return 0;
}

//
// This must be called immediately after the optimation passes; after the
// 'const' nodes have been sorted to the front, and before the 'input' arrays
// are set up.
// Only needs calling when there is a node with flag NN_NODE_FLAG_CLS_LOOP_CONTROL_NODE
// somewhere in the graph.
//
int
nn_graphloop_prepare_graph( struct nn_graph * nn)
{
	struct inref_set needsink;		// table of inputs that need 'sink'.
	if( inref_set_init(&needsink,128)){
		return errlog(nn,"alloc failed");
	}

	int need_sort = 0;
	int cur_graph_partition = 0;
	struct nn_node * output_node_ptr = NULL;

	struct nn_node **nodepp = nn->nonconst_head_ptr;		// ptr to pointer to first
	struct nn_node * nodep;
	for( ; (nodep = *nodepp) != NULL; nodepp = &nodep->next ){
		if( nodep->node_type == OP_Const) continue;		// should not be
		int is_lcn = (nodep->ops->flags & NN_NODE_FLAG_CLS_LOOP_CONTROL_NODE) != 0;

		if( cur_graph_partition == 0 && !is_lcn){		// trivial fast case
			nodep->refs =0;
			continue;
		}

		//
		// scan all the non-const inputs to the node.
		// if we see the same nid twice in a row, we can skip the second one;
		// we are just looking for the range of the 'refs' value in the non-const nodes.
		//
		int max_partn = 0;
		int min_partn = 999;
		int n_in = nodep->n_inputs;
		unsigned prev_nid = 0;
		for( int i =0; i < n_in; i++ ){
			unsigned nid = nodep->input_refs[i].src_id;
			if(nid != prev_nid ){
				struct nn_node * src = find_node( nn, nid);
				if( src == NULL){
					errlog(nn,"failed to find node");
					goto fail;
				}
				if( src->node_type != OP_Const){
					int other_partn = src->refs;
					max_partn  = max_i32( max_partn, other_partn);
					min_partn  = min_i32( min_partn, other_partn);
				}
				prev_nid = nid;
			}
		}
		if( min_partn> max_partn) min_partn = 0;		// node has no non-const inputs.

		int cur_node_partn = max_partn;
		if( is_lcn ){
			// this introduces a new partition.
			if (cur_graph_partition != max_partn ){
				errlog(nn,"Bad LCN topology at node 0x%p", (unsigned)nodep->node_id);
				goto fail;
			}
			if( cur_graph_partition >= NN_MAX_LOOPSTACK){
				errlog(nn,"Too many LCN at node 0x%p", (unsigned)nodep->node_id);
				goto fail;
			}
			cur_node_partn = ++cur_graph_partition;
			logmsg(nn,3,"LCN node 0x%p is first in partition 0x%X",
					(unsigned)nodep->node_id, cur_graph_partition);
		}else{
			if( nodep->node_type == OP_OUTPUT){
				if( output_node_ptr != NULL){
					errlog(nn,"second OUTPUT node found");
					goto fail;
				}
				output_node_ptr = nodep;
			}
		}
		nodep->refs = cur_node_partn;

		if( cur_node_partn < cur_graph_partition){
			need_sort = 1;
		}
		// is the cur_node_partn > min_partn? this will always be true for an LCN.
		// Find all the non-const inputs which span from a lower-partition
		//
		if( cur_node_partn > min_partn){
			unsigned prev_nid = 0;
			struct nn_node *srcnode = NULL;
			for( int i =0; i < n_in; i++){
				unsigned nid = nodep->input_refs[i].src_id;
				if(nid != prev_nid ){
					srcnode = find_node( nn, nid);
					prev_nid = nid;
				}
				if( srcnode == NULL){
					errlog(nn,"failed to find node");
					goto fail;
				}
				if( srcnode->node_type != OP_Const && srcnode->refs < cur_node_partn){
					// we need this input to be connected to the sink.
					int e = inref_set_insert( &needsink, &nodep->input_refs[i]);
					if( unlikely(e)){
						errlog(nn,"alloc fail in insert");
						goto fail;
					}
				}
			}
		}
	}
	// remove any duplicates in 'needsink'
	// (and put them in order as a side-effect)
	inref_set_sort( &needsink);

	logmsg(nn,2,"%d partitions found; need_sort = %d; %d tensors need sink",cur_graph_partition, need_sort, needsink.size  );


	if( needsink.size > 0){
		// make the 'Sink' node.
		struct nn_node * sink_node = create_node( nn,0, OP_Sink, NN_PAD_NA,
					needsink.size, 0,		// inputs and output count
					needsink.tbl,			// inputs
					 NULL );				// outputs
		if( sink_node == NULL ){
			errlog(nn,"failed to make Sink with %d inputs", (int)needsink.size);
			goto fail;
		}
		sink_node->refs = cur_graph_partition;	// in case we need to sort
		insert_nodes( nn, nodepp,sink_node);
	}
	inref_set_free(&needsink);

	if( output_node_ptr != NULL
			&& output_node_ptr->refs != cur_graph_partition ){
		return errlog(nn,"OUTPUT node is not downstream from the last loop-control node");
	}

	int result = 0;
	if( need_sort){
		logmsg(nn,3,"need partition sort...");
		result = nn_graphloop_sort_partitions(nn);
	}

	return result;
 fail:
	inref_set_free(&needsink);
	return -1;
}

// This is called if nn_graphloop_prepare_graph finds that there are some nodes which are
// 'out of order' in the graph; when called, all the non-const nodes beyond
// nonconst_head_ptr will have 'refs' fields in the range 0..NN_MAX_LOOPSTACK
// and all we have to do is sort them in order of 'refs', without changing the order
// of the subsets which all have the same 'refs'.
// This is done by stripping the ones with refs > 0 out into separate lists, according
// to the number, and then just stringing the lists back together again.
//
static int
nn_graphloop_sort_partitions( struct nn_graph * nn)
{
	struct sublist{
		struct nn_node * headp;
		struct nn_node * tailp;
	};
	struct sublist sublists[NN_MAX_LOOPSTACK] = { {NULL,NULL}, };

	// if we see a 'refs' which is not in range 0..NN_MAX_LOOPSTACK, we treat
	// it as if it was NN_MAX_LOOPSTACK, keep going, and record the error for return.
	// that way the graph at least gets put back together.

	// make a pass through the list, remove all the nodes which have refs >1,
	// place them on the sublists in the same order.
	int errs = 0;

	struct nn_node **nodepp = nn->nonconst_head_ptr;		// ptr to pointer to first
	struct nn_node * nodep;
	for( ; (nodep = *nodepp) != NULL; nodepp = &nodep->next ){
		if( nodep->node_type != OP_Const){
			unsigned partn = nodep->refs;
			if( partn > 0){
				if( partn > NN_MAX_LOOPSTACK){
					errs = 1;
					partn = NN_MAX_LOOPSTACK;
				}
				// select the sublist we need...
				struct sublist * lsp = &sublists[partn-1];
				*nodepp = nodep->next;	// remove from existing list.
				// append to sublist
				nodep->next = NULL;
				if( lsp->headp==NULL){
					lsp->headp = nodep;
				}else{		// not the first...
					lsp->tailp->next = nodep;
				}
				lsp->tailp = nodep;
			}
		}
	}
	// Now.. string them back together. Note that nodepp points to the 'next' field
	// of the last 'partition 0' node, and (*nodepp) is NULL.
	for(int i = 0; i < NN_MAX_LOOPSTACK; i++){
		struct sublist * lsp = &sublists[i];
		if(lsp->headp!=NULL){		// something there...
			*nodepp = lsp->headp;	// string it up
			nodepp = &lsp->tailp->next;
		}
	}

	if( errs){
		return errlog(nn,"found invalid partition tags when sorting partitions");
	}
	return 0;
}
