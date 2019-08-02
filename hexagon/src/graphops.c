
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
 * This contains operations on the graph
 */

#include <nn_graph.h>
#include "nn_prepare.h"

static inline void log_causality(
	struct nn_graph *nn, 
	struct nn_node *tmp, 
	struct nn_node *producer)
{
	logmsg(nn,0,
		"CAUSALITY VIOLATION: "
		"node %p (id=0x%x) referenced output of node %p (id=0x%x) "
		"before instantiated in the graph",
		tmp,
		tmp->node_id,
		producer,
		producer->node_id);
}

/* Returns the first node in the graph that produces an output */
/* Excludes Const nodes */

struct nn_node* find_first_producer(
	struct nn_graph *nn, 
	struct nn_node *consumer)
{
	struct nn_node *tmp;
	struct input *in;
	int i;
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		if (tmp == consumer) return NULL;
		if (tmp->node_type == OP_Const) continue;
		for (i = 0; i < consumer->n_inputs; i++) {
			in = &consumer->input_refs[i];
			if (in->src_id == tmp->node_id) return tmp;
		}
	}
	return NULL;
}

void set_last_consumers(struct nn_graph *nn)
{
	struct nn_node *tmp;
	int i;
	struct tensor *t;
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		for (i = 0; i < tmp->n_inputs; i++) {
			t = (struct tensor *)tmp->inputs[i];
			t->last_consumer = tmp;
		}
	}
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		for (i = 0; i < tmp->n_outputs; i++) {
			t = tmp->outputs[i];
			if (t->last_consumer == NULL) {
				// If we haven't set the last consumer, it is a dead output
				// That's OK, we can put it back in the free list immediately.
				t->last_consumer = tmp;
			}
		}
	}
}

/* Returns the last node in the graph to reference the input. */
/* If no node references the input, producer is returned.  */

struct nn_node* find_last_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int out_idx)
{
	struct nn_node *tmp;
	struct nn_node *last_node = producer;
	struct input *in;
	int i;
	int seen_producer = 0;
	uint32_t prod_id = producer->node_id;
	noderefhash_set_t prod_hashmask = noderefhash_mask(prod_id);
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		if( (tmp->noderefhash & prod_hashmask)!=0){
			for (i = 0; i < tmp->n_inputs; i++) {
				in = &tmp->input_refs[i];
				if (in->src_id != prod_id) continue;
				if (in->output_idx != out_idx) continue;
				if (!seen_producer) {
					log_causality(nn,tmp,producer);
				} else {
					last_node = tmp;
				}
			}
		}
		if (tmp == producer) seen_producer = 1;
	}
	return last_node;
}

/* Returns the last node in the graph to reference the input. */
/* If no node references the input, producer is returned.  */

struct nn_node* find_first_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int out_idx)
{
	struct nn_node *tmp;
	struct input *in;
	int i;
	int seen_producer = 0;
	uint32_t prod_id = producer->node_id;
	noderefhash_set_t prod_hashmask = noderefhash_mask(prod_id);

	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		if( (tmp->noderefhash & prod_hashmask)!=0){
			for (i = 0; i < tmp->n_inputs; i++) {
				in = &tmp->input_refs[i];
				if (in->src_id != prod_id) continue;
				if (in->output_idx != out_idx) continue;
				if (!seen_producer) {
					log_causality(nn,tmp,producer);
				} else {
					return tmp;
				}
			}
		}
		if (tmp == producer) seen_producer = 1;
	}
	return producer;
}
/*
 *  Find the 'unique consumer' for a given 'producer' node
 *  If a 'consumer' node exists which meets all of the criteria below, a pointer is returned to it; otherwise
 *  NULL is returned. If producer == NULL, NULL is returned.
 *
 * The criteria are (in order of checking, more or less):
 *
 *  (1) consumer must have an input connected to producer.
 *  (2) consumer must be of specified node type(s), see below
 *  (3) consumer must be the *only* node with an input connected to producer
 *  (4) All of consumer's inputs must be connected to producer, except for inputs which are connected to const.
 *
 *  if CONSUMER_NOINCHECK is in options, (4) is skipped.
 *
 *  The 'node_type' is specified via req_node_type and node_types:
 *      req_node_type == -1, node_types = NULL:   any type is OK
 *      req_node_type >= 0, node_types == NULL:	  must match req_node_type
 *      req_node_type >= 1, node_types != NULL:	   must match one of node_types[0] .. node_types[req_node_type-1]
 */

struct nn_node* find_unique_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int req_node_type,			// -1, or desired node_type, or length of table (if node_types != NULL)
	const int *node_types,		// table of node-types, if not NULL.
	int options)
{
	struct nn_node *tmp;
	struct nn_node *consumer;
	struct input const *in;

	if( producer == NULL)
		return NULL;
	uint32_t prod_id = producer->node_id;
	noderefhash_set_t prod_hashmask = noderefhash_mask(prod_id);

	// (1) find a candidate
	consumer = NULL;
	for( tmp = producer->next; tmp != NULL; tmp = tmp->next) {
		if( (tmp->noderefhash & prod_hashmask)!=0){
			for (int i = 0; i < tmp->n_inputs; i++) {
				in = &tmp->input_refs[i];
				if (in->src_id == prod_id && in->output_idx == 0 ){
					consumer = tmp;
					goto found1;
				}
			}
		}
	}
	return NULL;
 found1:
 	// (2) found a consumer. make sure it's of specified type
 	 if( req_node_type >= 0){
 		 int nt = consumer->node_type;

 		 if( node_types == NULL){
 			 if( nt !=req_node_type) return NULL;	// doesn't match specified type
 		 }else{
 			 for( int i =0; i <req_node_type; i++ ){
 				 if( nt == node_types[i]) goto nt_ok;
 			 }
 			 return NULL;
 		 }
 	   nt_ok:;
 	 }
 	 // (3) make sure that no later nodes are also reading 'producer'
 	 //
 	for( tmp = consumer->next; tmp != NULL; tmp = tmp->next) {
		if( (tmp->noderefhash & prod_hashmask)!=0){
			for (int i = 0; i < tmp->n_inputs; i++) {
				in = &tmp->input_refs[i];
				if (in->src_id == prod_id ) return NULL;
			}
		}
 	}
 	//
 	// (4) check that all inputs to consumer are from producer, or from 'const' nodes.
 	//
 	if( (options & CONSUMER_NOINCHECK )== 0)
 	{
 		for (int i = 0; i < consumer->n_inputs; i++) {
 			in = &consumer->input_refs[i];
 			if( in->src_id != prod_id){
 				tmp = find_node(nn, in->src_id);
 				if( tmp == NULL || tmp->node_type != OP_Const){
 					return NULL;
 				}
 			}
 		}
 	}
 	return consumer;
}


//
// returns 0 iff
//  - 'consumer' has at least one input connected an output of producer
//  - no other node has inputs connected to connected to producer
// returns -1 otherwise
//
int check_single_consumer_all(
	struct nn_graph *nn, 
	struct nn_node const *producer,
	struct nn_node const *consumer)
{
	struct nn_node *tmp;
	uint32_t prod_id = producer->node_id;
    // check at least one connection
    int n = consumer->n_inputs;
    for( int i = 0; i < n; i++ ){
        if( consumer->input_refs[i].src_id == prod_id ) goto found1;
    }
    return -1;
  found1:;

	noderefhash_set_t prod_hashmask = noderefhash_mask(prod_id);
    // only look downstream from producer
	for (tmp = producer->next; tmp != NULL; tmp = tmp->next) {
        if (tmp == consumer) continue;      // already seen it
        if( (tmp->noderefhash & prod_hashmask) ==0) continue;	// none here
		for (int i = 0; i < tmp->n_inputs; i++) {
			if( tmp->input_refs[i].src_id == prod_id) {
				logmsg(nn,2,"also found consumer %x",tmp->node_id);
				return -1;
			}
		}
	}
	return 0;
}
#if 0
//
// returns true iff the output
//  (producer, out_idx) 
//  goes only to a single input, and that
//  input is on 'consumer'
//
int check_single_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int out_idx,
	struct nn_node *consumer)
{
	struct nn_node *tmp;
	struct input *in;
	int i;
	int seen_producer = 0;
    int count = 0;
	uint32_t prod_id = producer->node_id;

	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		for (i = 0; i < tmp->n_inputs; i++) {
			in = &tmp->input_refs[i];
			if (in->src_id != prod_id) continue;
			if (in->output_idx != out_idx) continue;
			if (!seen_producer) {
				log_causality(nn,tmp,producer);
			} else {
                count ++;
                if( tmp != consumer || count > 1) return 0;
			}
		}
		if (tmp == producer) seen_producer = 1;
	}
	return count == 1;
}
#endif
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
        int n_remove)
{
	if( anchor == NULL) anchor = &nn->head;
	// find the first one
    struct nn_node * searchfor;
    if( n_remove < 1 || ( searchfor = rmnodes[0]) == NULL )
    	return -1;
    struct nn_node * current = *anchor;
    while( current != searchfor){
    	if( current == NULL)
    		return -1;		// couldn't find the first one...
    	anchor = &current->next;
    	current = *anchor;
    }
    int node_count = nn->node_count;
    // OK, the anchor points to the first node in rmnodes[].
    if( new_node != NULL){
    	// insert node before the one we're about to delete
    	*anchor = new_node;
    	anchor = &new_node->next;
    	*anchor = current;
    	node_count++;
    }
    nn->tail = new_node;	// ensure this is valid
    // remove the first item
    current = current->next;
    *anchor = current;
    searchfor->ops->dtor(searchfor,nn);
    node_count--;

    // remove the rest
    for( int i = 1; i < n_remove; i++){
    	searchfor = rmnodes[i];
    	if (searchfor != NULL) {
            while( current != searchfor){
            	if( current == NULL){
            		nn->node_count = (node_count <0)?0:node_count;
            		return -1;		// couldn't find
            	}
            	anchor = &current->next;
            	current = *anchor;
            }
            // remove it
            current = current->next;
            *anchor = current;
            searchfor->ops->dtor(searchfor,nn);
            node_count--;
        }
    }
    // add the new node to the hash (must do this *after* the deletes,
    // often one of the deleted nodes will have the same node id)
    if( new_node != NULL)
    	insert_node_to_hash(nn,new_node);
	nn->node_count = (node_count <0)?0:node_count;
    return 0;
}

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
        int n_add)
{
	if (inspos == NULL) inspos = &nn->head;
    int node_count = nn->node_count;
	if( delnode != NULL){
		// find, delete, position
		struct nn_node *searchnode = *inspos;
		while(searchnode!= delnode){
			if( searchnode == NULL) return -1;	// failed to find delnode
			inspos = &searchnode->next;
			searchnode = *inspos;
		}
		// *inspos == delnode
		*inspos = delnode->next;
		if( nn->tail == delnode) nn->tail = nn->head;
		delnode->ops->dtor(delnode,nn);
		node_count--;
	}
	// now the inserts.
	struct nn_node * lastp = NULL;
	struct nn_node * firstp = NULL;
	int nins = 0;
	// first, string them together
	for( int i =n_add-1; i >=0; i--){
		struct nn_node *np = addnodes[i];
		if( np != NULL){
			if(nins == 0){
				lastp = np;
			}else{
				np->next = firstp;
			}
			firstp = np;		// will be first non-null
			++nins;
			insert_node_to_hash(nn,np);
		}
	}
	if( nins > 0){
		lastp->next = *inspos;
		*inspos = firstp;
		node_count += nins;
	}
	nn->node_count = (node_count <0)?0:node_count;
	return nins;
}



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
		uint64_t pattern )
{
	noderefhash_set_t hashmask = noderefhash_mask(old_nodeid);
	int replace_count = 0;
	int errs = 0;
	struct nn_node *np = (begin == NULL)? nn->head: begin;
	for(; np != NULL ; np = np->next){
		if( (np->noderefhash & hashmask) != 0){		// may have matching refs in it
			int n_in = np->n_inputs;
			int any = 0;
			for( int  i =0; i < n_in; i++){
				if( np->input_refs[i].src_id == old_nodeid){
					unsigned idx = np->input_refs[i].output_idx;
					if( idx >= 16) {			// error, must be <= 15
						errs = 1;
					}else{
						int map = (pattern >>(4*idx)) &15;
						if( map == 0 ){
							errs = 1;			// error, output should not exist
						}else if( map < 15){					// ok do this one
							np->input_refs[i].src_id = new_nodeid;
							np->input_refs[i].output_idx = map-1;
							any = 1;
							replace_count++;
						}
					}
				}
			}
			if( any ) node_rehash_inputrefs( np);
		}
	}
	return errs?-1: replace_count;
}

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
		struct nn_node const * newnodes[] )  // array of new nodes to point to [0..n_outputs-1]
{
	noderefhash_set_t hashmask = noderefhash_mask(old_nodeid);
	int replace_count = 0;
	int errs = 0;
	struct nn_node *np = (begin == NULL)? nn->head: begin;
	for(; np != NULL ; np = np->next){
		if( (np->noderefhash & hashmask) != 0){		// may have matching refs in it
			int n_in = np->n_inputs;
			int any = 0;
			for( int  i =0; i < n_in; i++){
				if( np->input_refs[i].src_id == old_nodeid){
					unsigned idx = np->input_refs[i].output_idx;
					if( idx < (unsigned)n_outputs){		// in range ...
						struct nn_node const* newnode = newnodes[idx];
						if( newnode != NULL){		// ok, do this one.
							np->input_refs[i].src_id = newnode->node_id;
							np->input_refs[i].output_idx = 0;
							any = 1;
							replace_count++;
						}
					}
				}
			}
			if( any ) node_rehash_inputrefs( np);
		}
	}
	return errs?-1: replace_count;
}

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
		struct input const new_inpref[] )  // array of new
{
	noderefhash_set_t hashmask = noderefhash_mask(old_nodeid);
	int replace_count = 0;
	int errs = 0;
	struct nn_node *np = (begin == NULL)? nn->head: begin;
	for(; np != NULL ; np = np->next){
		if( (np->noderefhash & hashmask) != 0){		// may have matching refs in it
			int n_in = np->n_inputs;
			int any = 0;
			for( int  i =0; i < n_in; i++){
				if( np->input_refs[i].src_id == old_nodeid){
					unsigned idx = np->input_refs[i].output_idx;
					if( idx < (unsigned)n_outputs){		// in range ...
						struct input const* newinref = &new_inpref[idx];
						if( newinref->src_id != 0){		// ok, do this one.
							np->input_refs[i] = *newinref;
							any = 1;
							replace_count++;
						}
					}
				}
			}
			if( any ) node_rehash_inputrefs( np);
		}
	}
	return errs?-1: replace_count;
}

