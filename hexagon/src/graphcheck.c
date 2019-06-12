
/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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
#include <nn_graph.h>
#include "nn_string_map.h"

#if !defined(NN_LOG_MAXLEV) || NN_LOG_MAXLEV >= 1
//
// This contains code which does 'sanity checks' on a graph,
// looking for things like dangling references, operations out of order in the graph,
// etc.
// It makes no changes to the graph, and constructs its own index, so that
// the hash table can also be checked.
//
//

// The index is a table of these structs, sorted by node_id.
struct gcheck_entry {
	uint32_t node_id;			// id of the node
	struct nn_node const * nodep;	// pointer to it
	int32_t position;			// index of position in linked_list.
	int32_t refcount;				// used to count refs to each node.
};


struct gchk_context {
	struct gcheck_entry *index;		// pointer to allocated array of
	int index_size;					// # of values
	int tail_posn;					// index of what 'tail' is pointing at (-1 if not found)
	int nonconst_head_posn;			// index of what 'nonconst_head_ptr' is pointing at (-1 of not found)
	// for hash checking
	int hash_count;			// # of nodes in hash table
	int hash_errs;			// # of improper entries in hash table
};

static int build_index( struct nn_graph *nn, struct gchk_context* ctxp );
static void release_index( struct nn_graph *nn, struct gchk_context* ctxp );
static struct gcheck_entry * lookup_node( struct gchk_context* ctxp, uint32_t node_id );
extern void traverse_hash_for_check(struct nn_graph *nn,
		void (*callback)(struct nn_graph *, uint32_t nid, struct nn_node *, void *),
		void *info );
static void hash_check_callback( struct nn_graph *nn,uint32_t nid, struct nn_node *nodep, void * vctxp );


int
check_graph( struct nn_graph *nn, int options)
{
	struct gchk_context  ctx;
	if(nn->debug_level == 0 && !nn_option_get(nn,test_force_graph_check)) return 0;

	int c = build_index( nn, &ctx);
	if( c != 0) return c;
	int numerr = 0;
	{
		//  check for duplicate node ids
		struct gcheck_entry const *gip = ctx.index;

		for( int i = 0; i < ctx.index_size-1; i++ ){
			if( gip[i].node_id == gip[i+1].node_id ){
				logmsg(nn,0,"duplicate node %X: %s at pos=%d and %s at pos=%d",
						(unsigned)gip[i].node_id, op_type_to_string(gip[i].nodep->node_type),(int) gip[i].position,
						op_type_to_string(gip[i+1].nodep->node_type), (int)gip[i+1].position);
				numerr++;
			}
		}
		if( numerr){
			release_index(nn,&ctx);
			return errlog(nn, "duplicate node ids found; giving up");
		}
	}
	if( ctx.index_size != nn->node_count){
		logmsg(nn,0,"there are %d nodes, not %d as recorded", ctx.index_size, (int)nn->node_count);
	}
	if( nn->tail != NULL && ctx.tail_posn < 0 ){
		logmsg(nn,0,"Tail pointer %p is not on the list!", nn->tail);
		numerr++;
	}
	//
	// check references.
	// for each input ref:
	//   - referenced node (producer) must exist;
	//   - producer must have sufficient outputs;
	//   - producer must be earlier in the sequence than consumer.
	//
	{
		struct nn_node const * cons;
		struct gcheck_entry  * cidx;
		int hasherrs = 0;

		for( int icons = 0; icons < ctx.index_size; icons++ ){
			cidx = &ctx.index[icons];
			cons = cidx->nodep;
			int n_in = cons->n_inputs;
			for( int i = 0; i < n_in; i++){
				struct input const * in = &cons->input_refs[i];
				uint32_t prod_id = in->src_id;
				struct gcheck_entry  * pidx = lookup_node( &ctx, prod_id);
				if( pidx == NULL){
					logmsg(nn,0,"node %X input %d refs. non-existent node %X",
							(unsigned)cons->node_id, i, (unsigned)prod_id );
					numerr++;
					continue;
				}
				struct nn_node const * prod = pidx->nodep;
				if( in->output_idx >= prod->n_outputs ){
					logmsg(nn,0,"node %X input %d refs. %X output %d, but only %d outputs",
							(unsigned)cons->node_id, i, (unsigned)prod_id, (int)in->output_idx, (int)prod->n_outputs);
					numerr++;
					continue;
				}
				pidx->refcount ++;
				if( pidx->position >= cidx->position){
					logmsg(nn,0,"node %X (%s) input %d (pos %d) refs later node  %X (%s) output %d (pos %d)",
							(unsigned)cons->node_id,op_type_to_string(cons->node_type), i,(int)cidx->position,
							(unsigned)prod_id, op_type_to_string(prod->node_type), (int)in->output_idx,(int)pidx->position);
					numerr++;

				}
			}
			// check the noderefhash is OK (by forcing recalc and comparing to previous)
			//
			noderefhash_set_t old_hash = cons->noderefhash;
			//logmsg(nn,2,"noded_id %08X inputs = %2d noderefhash = %08X", (unsigned)cons->node_id, (int)cons->n_inputs, (unsigned)cons->noderefhash);
			node_rehash_inputrefs((struct nn_node*)cons);
			if( (cons->noderefhash & ~old_hash) !=0 ){
				logmsg(nn,0, "Node %X has bad noderefhash -- was 0x%08X, should be 0x%08X", (unsigned)cons->node_id,
						(unsigned) old_hash, (unsigned) cons->noderefhash);
				hasherrs++;
			}
		}
		if( numerr){
			release_index(nn,&ctx);
			return errlog(nn,"cross-indexing errors found");
		}
		numerr = hasherrs;
	}
	if( options & GRAPHCHECK_NONCONST){
		int ncpos = ctx.nonconst_head_posn;
		if( ncpos < 0){
			logmsg(nn,0,"nonconst_head_ptr=%p is not in list!", nn->nonconst_head_ptr);
			numerr++;
		}else{
			for( int i = 0; i < ctx.index_size; i++ ){
				struct gcheck_entry  const * idxp = &ctx.index[i];
				struct nn_node const * np = idxp->nodep;
				int partition_is = idxp->position >= ncpos;
				int partition_should = np->node_type != OP_Const;
				if( partition_is !=partition_should ){
					logmsg(nn,0, "node %X (%s) found at pos %d, %s nonconst_head_ptr",
						(unsigned)np->node_id, op_type_to_string(np->node_type), (int)idxp->position,
						partition_is ? "after":"before");
					numerr++;
				}
			}
		}
	}
	// look for dead nodes ( no NN_NODE_FLAG_RETAIN flag, and 0 references). Not considered an error.
	// The NN_NODE_FLAG_RETAIN is set for nodes which have 0 outputs (Sink, Check, OUTPUT) and for a few
	// special cases like Assign and Variable which should be kept even if their consumers are removed.
	//
	if( options & GRAPHCHECK_DEADNODES){
		for( int i = 0; i < ctx.index_size; i++ ){
			struct gcheck_entry  const * idxp = &ctx.index[i];
			struct nn_node const * np = idxp->nodep;
			if( idxp->refcount == 0 && (np->flags&NN_NODE_FLAG_RETAIN)==0){
				logmsg(nn,0,"node %X (%s) with %d outputs has no references",
					(unsigned)np->node_id, op_type_to_string(np->node_type), (int)np->n_outputs );
				numerr++;
			}
		}
	}
	if( options & GRAPHCHECK_HASH){
		ctx.hash_count = 0;
		ctx.hash_errs = 0;
		traverse_hash_for_check(nn, hash_check_callback, (void*)&ctx);
		if(ctx.hash_errs ){
			numerr += ctx.hash_errs;
			logmsg(nn,0,"%d bad entries in hash; %d ok out of %d", ctx.hash_errs, ctx.hash_count, ctx.index_size );
		}else{
			logmsg(nn,2,"hash ok; %d of %d nodes are in hash", ctx.hash_count, ctx.index_size );
		}
	}
	if( numerr ==0 ) logmsg(nn,2,"graph check ok, options %X", options);
	release_index(nn,&ctx);
	return (numerr ==0)? 0:-1;
}

// the call to traverse_hash_for_check results in this being called for each entry (nid, nodep) in the hash.
//

static void
hash_check_callback( struct nn_graph *nn,uint32_t nid, struct nn_node *nodep, void * vctxp )
{
	struct gchk_context * ctxp = (struct gchk_context *)vctxp;
	struct gcheck_entry * idxp = lookup_node(ctxp, nid);
	if( idxp == NULL){
		logmsg(nn,0,"hash has node %X(%s) @%p which is not on linked-list!", (unsigned)nid, op_type_to_string(nodep->node_type), nodep );
		ctxp->hash_errs++;
	}else if( idxp->nodep != nodep){
		logmsg(nn,0,"hash has node %X(%s) @%p, linked list has it (%s) @%p !",
				(unsigned)nid,op_type_to_string(nodep->node_type),  nodep,
				op_type_to_string(idxp->nodep->node_type), idxp->nodep );
		ctxp->hash_errs++;
	}else{
		ctxp->hash_count++;
	}

}

static int
gindex_compare_func( void const *pva, void const *pvb)
{
	struct gcheck_entry const * pa = (struct gcheck_entry const*)pva;
	struct gcheck_entry const * pb = (struct gcheck_entry const*)pvb;
	if( pa->node_id < pb->node_id) return -1;
	return pa->node_id > pb->node_id;
}

// allocate and build index.
// Also, sets tail_posn and nonconst_head_posn
static
int build_index( struct nn_graph *nn, struct gchk_context* ctxp )
{

	// first count the nodes in the list
	struct nn_node const * tmp  = nn->head;
	int node_count = 0;
	for( ; tmp!= NULL; tmp = tmp->next){
		if( ++node_count == 100000){
			return errlog(nn,"appears to be loop in list!");
		}
	}
	// construct the index;
	ctxp->index = NULL;
	ctxp->index_size = 0;
	if( node_count ==0){
		return 0;
	}
	struct gcheck_entry *gindex = nn_calloc( node_count, sizeof(struct gcheck_entry));
	if( gindex == NULL) return errlog(nn,"alloc failed");

	struct nn_node const *tail = nn->tail;
	struct nn_node  **nonconst_head = nn->nonconst_head_ptr;
	int tail_posn = -1;
	int nonconst_posn = (nonconst_head == &nn->head)?0:-1;

	tmp = nn->head;
	int cnt = 0;
	for(; tmp!= NULL; tmp = tmp->next){
		if( cnt >= node_count){ cnt++; break;}
		if( tmp == tail) tail_posn = cnt;
		if( &tmp->next == nonconst_head) nonconst_posn = cnt+1;
		struct gcheck_entry * gep = &gindex[cnt];
		gep->node_id = tmp->node_id;
		gep->nodep = tmp;
		gep->position = cnt++;
	}
	ctxp->tail_posn = tail_posn;
	ctxp->nonconst_head_posn = nonconst_posn;

	if( cnt != node_count){
		nn_free(gindex);
		return errlog( nn, "list size different on second pass??");
	}
	if( node_count > 1){
		qsort(gindex, node_count, sizeof(struct gcheck_entry), gindex_compare_func);
	}
	ctxp->index = gindex;
	ctxp->index_size = node_count;

	return 0;

}
static void release_index( struct nn_graph *nn, struct gchk_context* ctxp )
{
	if(ctxp->index != NULL){
		nn_free(ctxp->index);
		ctxp->index = NULL;
	}
}
// find a node in the index, based on its node id.
// returns null when not found.
//
static struct gcheck_entry *
lookup_node( struct gchk_context* ctxp, uint32_t node_id )
{
	struct gcheck_entry *gindex = ctxp->index;
	int lo = 0;
	int hi = ctxp->index_size;
	while( lo <hi){
		int mid = (lo+hi)>>1;
		struct gcheck_entry *pmid = &gindex[mid];
		uint32_t idmid  =pmid->node_id;
		if( idmid <= node_id){
			if( idmid == node_id) return pmid;
			lo = mid+1;
		}else{
			hi = mid;
		}
	}
	return NULL;
}

#endif
