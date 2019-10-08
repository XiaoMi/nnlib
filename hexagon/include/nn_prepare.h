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

#ifndef NN_PREPARE_H
#define NN_PREPARE_H 1

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains common routines used in the graph prepare step 
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <quantize.h>
#include <math.h>
#include <stdio.h>
#include "nn_oemnode.h"
#include "transpose_conv_procweights.h"
#include "nn_axis.h"

// output desc for scalar floats.
extern const struct output Output_ScalarFloat;
/*
	.rank = 4,
	.max_sizes = {1, 1, 1, 1},
	.elementsize = 4};
*/

void make_outputdesc_from_shape(struct output *outp, struct shape const *shp, int elsize, int add_d32_padding_unused);

// extract shape from output desc
void shape_from_outputdesc(struct shape *shp, struct output const *outp);

//legacy
static inline
void shape_from_outdesc(struct shape *shp, struct output const *outp, int add_d32_padding){
	shape_from_outputdesc(shp,outp);
}

struct nn_node *create_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t operation,
	padding_type padding,
	int num_inputs,
	int num_outputs,
	struct input *input_refs,
	struct output *output_defs);

//
// currently all the uses of this are for Convert to/from d32;
// so elsize =1.
//
struct nn_node *create_convert(
	struct nn_graph *nn,
	int src_id,
	int output_idx,
	struct shape const *outsize,
	uint32_t operation);

// find a node, but only if node_type == ntype
inline struct nn_node *
find_node_must_be(struct nn_graph *nn, uint32_t node_id, op_type ntype)
{
	struct nn_node *n = find_node(nn, node_id);
	if (n != NULL && n->node_type == ntype)
		return n;
	return NULL;
}

// find a node, but only if node_type == OP_Const
struct nn_node *find_node_must_be_Const(struct nn_graph *nn, uint32_t node_id);
// find a const node from a struct input. returrns
// NULL if the src_id != 0.
//
struct nn_node *find_node_must_be_Const_from_ref(struct nn_graph *nn, struct input const *iref);

void free_node_array(struct nn_node **node_list, uint32_t num_nodes);


struct nn_node* find_last_consumer(
	struct nn_graph *nn,
	struct nn_node *producer,
	int out_idx);
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
// these return a node_id, or 0 if an error occurred.
//
uint32_t create_const_int32_op(struct nn_graph *nn, int32_t const_float);
uint32_t create_const_float_op(struct nn_graph *nn, float const_float);

//
// const nodes made in 'prepare' are added at the *front* of the list,
// so that they can be referenced by any other node.
// You can call this with data =NULL, data_len >0, and it will create
// a const with 'garbage' value that you can then fill in.
//
// This is just like do_prepend_const_node
// but it returns a pointer to the new node's tensor; NULL on error
// Useful when you are passing NULL as 'data' and want to fill
// in the data yourself.
//
struct tensor const *
do_prepend_const_node_ptr(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches, uint32_t height, uint32_t width, 	uint32_t depth,
	const uint8_t *data, uint32_t data_len);

// const nodes made in 'prepare' are added at the *front* of the list,
// so that they can be referenced by any other node.
// You can call this with data =NULL, data_len >0, and it will create
// a const with 'garbage' value that you can then fill in.
//
static inline int
do_prepend_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches, uint32_t height, uint32_t width, 	uint32_t depth,
	const uint8_t *data, uint32_t data_len)
{
	struct tensor const *tens;
	tens = do_prepend_const_node_ptr( nn, node_id, batches, height, width, depth,
			data, data_len);
	return (tens==NULL)?-1:0;
}
//
// two functions to handle oversize d32 supernodes:
// if a node meets the criteria, we prepend a QuantizedPadForConv_8_d32
// (or QuantizedPadForConv_u16_d32) in front of it, and make it a VALID operation.
// The PadForConv node sees the same window and stride tensors as are used in the conv,
// so it can work out what padding is needed, and apply it.
// There are two functions here:
//   - oversize_d32_supernode_quick_check( node const * ) returns 1 if the node
//        '*may*' be a candidate. This is a simple inline function, for use in graph
//         traversal loop.
//   - handle_oversize_d32_supernode( nn_graph *, node ** nodep )
//        will do the proper checking, and change if needed.
//      It returns <0 if there is a fatal error.
//
static inline int
oversize_d32_supernode_quick_check( struct nn_graph *nn, struct nn_node const * node)
{
	// this is just a check to see if (a) is one of the supernode types
	//  (b) is not 'VALID' padding.
	int ty = node->node_type;
	if( ! (
		   ty == OP_Supernode_8x8p8to8_d32
		|| ty == OP_DepthwiseSupernode_8x8p8to8_d32
		|| ty == OP_Supernode_8x8p32to8_d32
		|| ty == OP_DepthwiseSupernode_8x8p32to8_d32
		|| ty == OP_Supernode_u16x16p16to16_d32
		|| ty == OP_Supernode_u16x16p32to16_d32 )){
		return 0;
	}
	if( node->padding == NN_PAD_VALID){
		return 0;
	}
	return 1;
}
// in prepare_utils.c
int handle_oversize_d32_supernode( struct nn_graph *nn,  struct nn_node ** nodep);



//
// returns 0 iff
//  - 'consumer' has at least one input connected to an output of producer
//  - no other node has inputs connected to connected to producer
// returns -1 otherwise
//
int check_single_consumer_all(
	struct nn_graph *nn,
	struct nn_node const *producer,
	struct nn_node const *consumer);

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


// used to convert a 13-input supernode (with channelscale) to 12
int handle_channelscaled_supernode( struct nn_graph *nn, struct nn_node *nodep);

#endif //NN_PREPARE_H
