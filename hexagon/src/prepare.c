
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

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains things applicable to the interface outside the DSP.
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <quantize.h>
#include <math.h>
#include <stdio.h>
#include "nn_oemnode.h"
#include "nn_axis.h"
// int hexagon_nn_prepare(nn_id id);


struct prep_const_cache_entry
{
	uint32_t value;
	uint32_t node_id;
};
// This is pointed to by nn->pstate, but only during
// prepare (it is allocated on local stack frame of prepare function)
//
#define SCALAR_CACHE_SIZE 32

struct nn_prepare_state {
	// cache of recently built float or int consts.
	// the first 3 slots are dedicated to 0, -inf, inf.
	// The rest are managed on a 'least recently used is evicted' basis
	// (last one is most recently used).
	// Note: any of the node_id in here may be zero, due to calls to 'purge_const_cache',
	// these need to be treated as empty slots).
	// entries in the range [3+scalar_cache_n ... 3+SCALAR_CACHE_SIZE-1] are considered garbage.
	int scalar_cache_n;			// # of valid (after first 3)
	struct prep_const_cache_entry scalar_cache[3+SCALAR_CACHE_SIZE];


	// TODO: info to avoid creating multiple Convert_to_d32 ops.
};

// const nodes made in 'prepare' are added at the *front* of the list,
// so that they can be referenced by any other node.
// You call this with data =NULL, data_len >0, and it will create
// a const with 'garbage' value that you can then fill in.
//
int
do_prepend_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches, uint32_t height, uint32_t width, 	uint32_t depth,
	const uint8_t *data, uint32_t data_len)
{
	struct nn_node *node;
	if ((node = hexagon_nn_empty_const_ctor(
		  nn,	node_id,
		  batches, height,width,depth,
		  data_len)) == NULL) {
		return errlog(nn,"node id=0x%x ctor fail",node_id);
	}
	if( data != NULL)
		memcpy(node->outputs[0]->data,data,data_len);
	insert_nodes(nn,NULL,node);	// insert at head
	return 0;
}


// make new const scalar nodes with a given float or int32 value.
// if in 'prepare' phase, these are cached.
//
uint32_t create_const_float_op(struct nn_graph *nn, float floatval)
{
	union {
		float fval;
		int32_t ival;
	} u = { floatval };
	return create_const_int32_op(nn,u.ival);
}

uint32_t create_const_int32_op(struct nn_graph *nn, int32_t value)
{
	struct nn_prepare_state *pcp = nn->pstate;
	struct prep_const_cache_entry * ep = NULL;
	if( pcp != NULL){	//
		if( (value & 0x7FFFF) == 0 ){		// could be 0,-inf, inf
			if( value == 0){
				ep = &pcp->scalar_cache[0];		// zero goes here
			}else if( value == -0x00800000){	// -inf
				ep = &pcp->scalar_cache[1];		// goes here...
			}else if (value == 0x7F800000){		// +inf
				ep = &pcp->scalar_cache[2];		// goes here...
			}
		}
		if( ep == NULL){		// wasn't one of the above
			int ncache = pcp->scalar_cache_n;
			struct prep_const_cache_entry * cache = &pcp->scalar_cache[3];
			int ipos;
			for(  ipos =ncache-1; ipos >= 0; ipos--){
				if( cache[ipos].value == value )break;
			}
			if( ipos < 0 ){			// did not find it.
				if( ncache < SCALAR_CACHE_SIZE){	// cache not full
					ncache++;
					pcp->scalar_cache_n = ncache;	 // add it at the end
				}else{
					// if full, throw out the oldest 4.
					#if SCALAR_CACHE_SIZE <12
					#error "need to look at this..."
					#endif
					int ndiscard = 4;
					memmove( &cache[0], &cache[ndiscard], sizeof(cache[0])*(SCALAR_CACHE_SIZE-ndiscard));
					ncache = SCALAR_CACHE_SIZE+1-ndiscard;
				}
				ep = & cache[ncache-1];		// this is where we are at.
				ep->value = value;
				ep->node_id = 0;			// make sure it's not valid
			}else{	// found it...
				ep = & cache[ncache-1];
				if( ipos < ncache-1){		// move it to the end
					uint32_t keep = cache[ipos].node_id;
					memmove( &cache[ipos], &cache[ipos+1], sizeof(cache[0])*(ncache-1-ipos));
					ep->value = value;
					ep->node_id = keep;
				}
			}
		}
	}
	// was it a hit?
	if( ep != NULL && ep->node_id != 0) {
		return ep->node_id;
	}

	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	int32_t val = value;
	if ((do_prepend_const_node(nn,new_node_id,1,1,1,1,(const uint8_t *)&val,sizeof(int32_t))) != 0) {
		errlog(nn,"Can't make a const scalar node");
		return 0;
	}
	if( ep!= 0){
		ep->value = value;
		ep->node_id = new_node_id;
	}
	return new_node_id;
}


// remove a nid from the const cache,  e.g. if its value was changed.
//
static void
purge_const_cache( struct nn_graph *nn, uint32_t nid)
{
	struct nn_prepare_state *pcp = nn->pstate;
	if( pcp != NULL){
		int ncache = pcp->scalar_cache_n;
		struct prep_const_cache_entry * cache = &pcp->scalar_cache[0];
		for( int i= 0; i < ncache+3 ; i++){
			if( cache[i].node_id == nid){
				cache[i].node_id = 0;
				return;
			}
		}
	}
}
static inline
uint32_t get_inf_node(struct nn_graph *nn)
{
	return create_const_float_op(nn, INFINITY);
}
static inline
uint32_t get_ninf_node(struct nn_graph *nn)
{
	return create_const_float_op(nn, -INFINITY);
}
static inline
uint32_t get_zero_node(struct nn_graph *nn)
{
	return create_const_float_op(nn, 0.0f);
}

static float
read_float_from_qu8( struct tensor const * qu8_tensor, struct tensor const * min_tensor, struct tensor const *max_tensor)
{
	float maxval = tensor_get_float(max_tensor,0);
	int quantized_val = *(uint8_t const*)qu8_tensor->data;
	if( quantized_val == 255) return maxval;	// common case
	float minval = tensor_get_float(min_tensor,0);
	if( quantized_val == 0) return minval;
	return minval + ((maxval-minval) * quantized_val / 255.0f);
}
// get a zero qu8 node of shape (1,1,1,depth)
// The result is returned via an array of 3 'struct input': data, min, max.
// returns 0 if ok.
static int
get_zero_flat_qu8_const( struct nn_graph * nn, int depth , struct input inrefs[3] )
{
	// make an array...
	if( depth < 1 || depth > (1<<19))
		return -1;
	uint8_t * tptr;
	int ok_for_var_array = (depth <= 4096);
	uint8_t tmp_array[ ok_for_var_array ?  depth : 4];
	if( ok_for_var_array){
		tptr = tmp_array;
		memset( tmp_array, 0, depth);
	}else{
		tptr = (uint8_t*)nn_calloc( 1, depth);
		if( tptr== NULL)
			return -1;
	}
	uint32_t new_const_node_id = nn_graph_new_internal_node_id(nn);
	int res = do_prepend_const_node( nn, new_const_node_id, 1,1,1, depth, tptr, depth );
	if(!ok_for_var_array){
		nn_free(tptr);
	}
	if( res != 0) return -1;

	uint32_t min_node_id = get_zero_node(nn);
	uint32_t max_node_id = create_const_float_op( nn, 1.0f);
	if( min_node_id == 0 || max_node_id == 0 ) return -1;
	inrefs[0].src_id = new_const_node_id;
	inrefs[0].output_idx = 0;
	inrefs[1].src_id = min_node_id;
	inrefs[1].output_idx = 0;
	inrefs[2].src_id = max_node_id;
	inrefs[2].output_idx = 0;
	return 0;
}

////////////////////////////////////////// Shapes and padding ////////////////

// output desc for scalar floats.
static const struct output Output_ScalarFloat = {
	.rank = 4,
	.max_sizes = {1,1,1,1},
	.elementsize = 4
};
#if 0
// "official" policy on d32 padding
static void
add_d32_padding_u32( uint32_t *bhwd)
{
	bhwd[0] += MAX_PADDING_BATCHES;
	bhwd[1] += MAX_PADDING_HEIGHT;
#if 1 // original policy
	bhwd[2] += MAX_PADDING_WIDTH;
	bhwd[3] += MAX_PADDING_DEPTH;
#else // more frugal
	bhwd[2] = ( bhwd[2] +  7) & ~3u;	// add 4, round up to mult. of 4
	bhwd[3] = ( bhwd[3] + 31) & ~31u;	// round up to mult of 32
#endif
}
// add padding to a struct output
inline static void
output_add_d32_padding(struct output *outp )
{
	add_d32_padding_u32( &outp->max_sizes[0]);
}
// add padding to a struct shape
inline static void
shape_add_d32_padding(struct shape *shp )
{
	add_d32_padding_u32( &shp->batches);
}
// make output desc from shape; optionally add d32 padding
#endif

static void __attribute__((noinline))
make_outputdesc_from_shape( struct output *outp, struct shape const *shp, int elsize, int add_d32_padding_unused)
{
	outp->rank = 4;
	outp->max_sizes[0] = shp->batches;
	outp->max_sizes[1] = shp->height;
	outp->max_sizes[2] = shp->width;
	outp->max_sizes[3] = shp->depth;
	for(int i = 4; i < (int)(sizeof(outp->max_sizes)/sizeof(outp->max_sizes[0])); i++ ){
		outp->max_sizes[i] = 0;
	}
	outp->elementsize = elsize;
	outp->zero_offset =  0;
	outp->stepsize  = 0.0f;
	//if( add_d32_padding ) output_add_d32_padding(outp);
}
// extract shape from output desc; optionally add d32 padding

static void __attribute__((noinline))
shape_from_outdesc( struct shape *shp, struct output const *outp, int add_d32_padding)
{
	shp->batches = outp->max_sizes[0];
	shp->height = outp->max_sizes[1];
	shp->width = outp->max_sizes[2];
	shp->depth = outp->max_sizes[3];
	//if( add_d32_padding) shape_add_d32_padding(shp);
}


int const_depth_extend_8(struct nn_node *node, int amt, int val)
{
	struct tensor *t = node->outputs[0];
	int b = t->shape.batches;
	int w = t->shape.width;
	int h = t->shape.height;
	int d = t->shape.depth;
	uint8_t *srcdata = t->data;
	int i;
	int new_size = b*w*h*(d+amt);
	uint8_t *new_data;
	uint8_t *dstdata;
	if ((new_data = nn_malloc(new_size + 128)) == NULL) return -1;
	for (i = 0, dstdata=new_data; i < b*w*h; i++) {
		memcpy(dstdata,srcdata,d);
		dstdata += d;
		srcdata += d;
		memset(dstdata,val,amt);
		dstdata += amt;
	}
	nn_free(t->data);
	t->data = new_data;
	t->shape.depth = d+amt;
	node->outputs[0]->max_size = new_size;
	return 0;
}


int const_width_extend_8(struct nn_node *node, int amt, int val)
{
	struct tensor *t = node->outputs[0];
	int b = t->shape.batches;
	int w = t->shape.width;
	int h = t->shape.height;
	int d = t->shape.depth;
	uint8_t *srcdata = t->data;
	int i;
	int new_size = b*(w+amt)*h*d;
	uint8_t *new_data;
	uint8_t *dstdata;
	if ((new_data = nn_malloc(new_size + 128)) == NULL) return -1;
	for (i = 0, dstdata=new_data; i < b*h; i++) {
		memcpy(dstdata,srcdata,w*d);
		dstdata += w*d;
		srcdata += w*d;
		memset(dstdata,val,amt*d);
		dstdata += amt*d;
	}
	nn_free(t->data);
	t->data = new_data;
	t->shape.width = w+amt;
	node->outputs[0]->max_size = new_size;
	return 0;
}
static inline int shape_1111( struct shape const * shp){
	return shp->batches == 1 && shp->height ==1 && shp->width ==1 && shp->depth == 1;
}
static inline int shape_111X( struct shape const * shp){
	return shp->batches == 1 && shp->height ==1 && shp->width ==1;
}
static inline int shape_X11X( struct shape const * shp){
	return shp->height ==1 && shp->width ==1;
}
//////////////////////////////////////////////////////
static struct nn_node * __attribute__((unused))create_convert_to_d32(struct nn_graph *nn, int src_id, int output_idx);
static struct nn_node *create_convert_from_d32(struct nn_graph *nn, int src_id, int output_idx, struct shape outsize);
static int change_refs(struct nn_graph *nn, int old_id, int old_out_idx, int new_id, int new_out_idx);

// find a node, but only if node_type == ntype
static inline struct nn_node *
find_node_must_be( struct nn_graph *nn, uint32_t node_id, op_type ntype){
	struct nn_node * n = find_node(nn,node_id);
	if( n != NULL && n->node_type == ntype)
		return n;
	return NULL;
}
// find a node, but only if node_type == OP_Const
static struct nn_node * __attribute__((noinline))
find_node_must_be_Const( struct nn_graph *nn, uint32_t node_id){
	return find_node_must_be( nn, node_id, OP_Const);
}
// find a const node from a struct input. returrns
// NULL if the src_id != 0.
//
static struct nn_node * __attribute__((noinline))
find_node_must_be_Const_from_ref( struct nn_graph *nn, struct input const *iref){
	if( iref->output_idx != 0) return NULL;
	return find_node_must_be( nn, iref->src_id, OP_Const);
}


//
// This is given a pointer to an array of 3 src refs presumed to
// reference three consts, defining u8 data:  data_arr, min, max.
// It will convert these to int32 symmetric, and update the refs in-place.
// returns -1 if there is a problem.
//  You can supply min_size, max_size parameters which indicate the allowed range;
//  values of <= 0 for these ae ignored.
static int convert_qu8_const_to_qint32( struct nn_graph *nn, struct input refs[3], int min_size, int max_size )
{
	// collect the consts.
	struct nn_node const * node_data = find_node_must_be_Const_from_ref( nn, &refs[0]);
	struct nn_node const * node_min = find_node_must_be_Const_from_ref( nn, &refs[1]);
	struct nn_node const * node_max = find_node_must_be_Const_from_ref( nn, &refs[2]);
	if( node_data== NULL || node_min == NULL || node_max == NULL) return errlog(nn,"can't get qu8 consts");

	//
	// get source data, check size
	//
	struct tensor const * data_tensor = node_data->outputs[0];

	uint8_t const * srcp = (uint8_t const*)data_tensor->data;
	int32_t size = tensor_element_count(data_tensor);
	if( size < min_size || (max_size>0 && size >max_size) || size > data_tensor->data_size ){
		return errlog(nn,"improper size %d in qu8_const_to_qint32", (int)size);
	}
	// get the floats
	float minval = tensor_get_float( node_min->outputs[0],0);
	float maxval = tensor_get_float( node_max->outputs[0],0);

	// here's what we do: shift the data left by 16 and subtract delt.
	float in_step = flt_div_255(maxval-minval);		// input step.
	// output max is 2^31 times the output step, which is 2^-16 times the input step, so..
	float output_max = in_step * (float)(1<<15);

	// find the offset.
	int delt = roundf_i32( minval * -(float)(1<<16) / in_step);

	// make the new const
	uint32_t new_const_nid = nn_graph_new_internal_node_id( nn);
	int res = do_prepend_const_node(nn, new_const_nid,
			data_tensor->shape.batches, data_tensor->shape.height, data_tensor->shape.width, data_tensor->shape.depth,
			NULL, size * sizeof(int32_t));
	if(res != 0){
		return errlog(nn,"could not alloc node for %d int32's", size);
	}
	struct nn_node * newconst = nn->head;		// should be here
	if( newconst == NULL || newconst->node_id != new_const_nid){
		newconst = find_node_must_be_Const(nn,new_const_nid);
	}
	if( newconst == NULL) return errlog(nn,"lost node!");

	// OK now we can fill the thing in...
	int32_t *outp = (int32_t *)newconst->outputs[0]->data;
	for( int i =0; i < size; i++){
		outp[i] = srcp[i]*(1<<16) - delt;
	}
	uint32_t out_min_nid = create_const_float_op( nn, -output_max);
	uint32_t out_max_nid = 0;
	if( out_min_nid != 0) out_max_nid = create_const_float_op(nn, output_max );

	if( out_max_nid ==0) return errlog(nn,"alloc failed");

	logmsg(nn,5,"converted %d of %f ... %f qu8 to  %f .. %f qint32, using (x<<16)-%d",
			size, minval, maxval, -output_max, output_max, delt );
	refs[0].src_id = new_const_nid;
	refs[1].src_id = out_min_nid;
	refs[2].src_id = out_max_nid;
	return 0;
}



static inline int prepare_input(struct nn_graph *nn, struct nn_node *node, int i)
{
	struct input inp;
	struct nn_node *srcnode;
	inp = node->input_refs[i];
	if (inp.src_id != NODE_ID_RESERVED_CONSTANT) {
		if ((srcnode = find_node(nn,inp.src_id)) == NULL) {
			return errlog(nn,"FATAL: source node ID 0x%x from node %x not found",inp.src_id,node->node_id);
		}
		if (srcnode->n_outputs <= inp.output_idx) {
			return errlog(nn,"FATAL: Bad output %d on node ID 0x%x\n",inp.output_idx,inp.src_id);
		}
		node->inputs[i] = srcnode->outputs[inp.output_idx];
		return 0;
	} else {
		logmsg(nn,1,"nn %p node %p idx %d: not handling constant input");
		return 0;
	}
}
//
// for each node in the graph set up the node->inputs[..] as pointers to the proper tensors
// (copied from other node's ->output[..], according node->input_refs[..])
//
static int prepare_inputs(struct nn_graph *nn)
{
	struct nn_node *node;
	int i;
	int err;
	for (node = nn->head; node != NULL; node = node->next) {
		for (i = 0; i < node->n_inputs; i++) {
			if ((err = prepare_input(nn,node,i)) != 0) return err;
		}
	}
	return 0;
}
#if 0
static int run_op_setup(struct nn_graph *nn)
{
	struct nn_node *node;
	int err;
	for (node = nn->head; node != NULL; node = node->next) {
		if ((err = node->ops->setup(node)) != 0) return err;
	}
}
#endif

// 'check' all of the ops in the graph.
static int run_op_check(struct nn_graph *nn)
{
	struct nn_node *node;
	int err;
	for (node = nn->head; node != NULL; node = node->next) {
		if ((err = node->ops->check(node,nn)) != 0) return err;
	}
	return 0;
}

// return 0 if all outputs of 'producer' go only to 'consumer'; otherwise -1

static inline int check_all_outputs(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	struct nn_node *consumer)
{
	return check_single_consumer_all(nn,producer, consumer);
/*
	int i;
	for (i = 0; i < producer->n_outputs; i++) {
		if (find_first_consumer(nn,producer,i) != consumer) return -1;
		if (find_last_consumer(nn,producer,i) != consumer) return -1;
	}
	return 0;
*/
}
// return 0 if the first 'n' inputs of nodes 'a' and 'b' are identically connected
// otherwise -1.
//
int check_same_inputs(struct nn_graph *nn, struct nn_node *a, struct nn_node *b, int n_inputs)
{
	int i;
	if (a->n_inputs < n_inputs) return -1;
	if (b->n_inputs < n_inputs) return -1;
	for (i = 0; i < n_inputs; i++) {
		if (a->input_refs[i].src_id != b->input_refs[i].src_id) return -1;
		if (a->input_refs[i].output_idx != b->input_refs[i].output_idx) return -1;
	}
	return 0;
}

/*
 * EJP: we really need a new syntax here for pattern matching graph parts 
 */
#define DO_DTOR(NODE,NN) ((NODE)->ops->dtor(NODE,NN))

static int try_fuse_requantization_range(struct nn_graph *nn, struct nn_node **requant_range_node_p)
{
	struct nn_node *requant_range_node = *requant_range_node_p;
	struct nn_node *requantize_node;
	struct nn_node *new_node;
	struct output new_outputs[3];
	int i;
	op_type operation = OP_QuantizeDownAndShrinkRange_32to8;
	/* Make sure the start node is the right kind */
	if (requant_range_node->node_type != OP_RequantizationRange_32) return 0;
	logmsg(nn,9,"found requantizationrange");
	/* Make sure the consumer is a requantization node and there's only one of them */
	requantize_node = find_unique_consumer_mustbe( nn, requant_range_node,OP_Requantize_32to8,CONSUMER_NOINCHECK);
	if( requantize_node == NULL) return 0;

	/* Make sure the inputs are pointing to the right place */
	if (check_same_inputs(nn,requant_range_node,requantize_node,3) != 0) return 0;
	// TODO: check the next 2 inputs of requantize_node are from requant_range_node
	//
	logmsg(nn,9,"Found matching requantize");
	/* Just to make sure we're not cheating, we will create a new op and dtor the old ones */
	for (i = 0; i < 3; i++) {
		new_outputs[i] = requantize_node->output_defs[i];
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		requantize_node->node_id,
		operation,
		requantize_node->padding,
		requant_range_node->n_inputs,
		3,
		requant_range_node->input_refs,
		new_outputs)) == NULL) return errlog(nn,"new node ctor");

	if( replace_nodes(nn, requant_range_node_p, new_node,
			requant_range_node, requantize_node )!= 0){
		return errlog(nn,"replace_node failed in fuse_requant_range");
	}
	logmsg(nn,3,"Changed to autorequantize node id=%x (was requantize ID)",new_node->node_id);
	return 0;
}

/* Turn Requantize with const max value followed by Relu into the same Requantize followed by ReluX */
/* EJP: FIXME: could support Requantize->ReluX where we take the min of the two max values */
static int try_make_reluX(struct nn_graph *nn, struct nn_node **requant_node_p)
{
	struct nn_node *requantize_node = *requant_node_p;
	struct nn_node *relu_node;
	struct nn_node *max_node;
	struct nn_node *new_node;
	struct input new_inputs[4];
	struct output new_outputs[3];
	uint32_t max_node_id;
	int i;
	op_type operation = OP_QuantizedReluX_8;
	if (requantize_node->node_type != OP_Requantize_32to8) return 0;
	/* Find range max node */
	if (requantize_node->input_refs[4].output_idx != 0) return 0;
	max_node_id = requantize_node->input_refs[4].src_id;
	if ((max_node = find_node_must_be_Const(nn,max_node_id)) == NULL) return 0;
	/* Make sure consumer is relu and there's only one of them */
	relu_node = find_unique_consumer_mustbe( nn, requantize_node, OP_QuantizedRelu_8,0);
	if( relu_node == NULL) return 0;
	logmsg(nn,9,"found matching relu");
	/* Create inputs and outputs */
	for (i = 0; i < 3; i++) {
		new_outputs[i] = relu_node->output_defs[i];
		new_inputs[i] = relu_node->input_refs[i];
	}
	new_inputs[3] = requantize_node->input_refs[4];
	if ((new_node = optab[operation]->ctor(
		nn,
		relu_node->node_id,
		operation,
		requantize_node->padding,
		4,
		3,
		new_inputs,
		new_outputs)) == NULL) return errlog(nn,"ctor fail");

	if( replace_nodes( nn, &requantize_node->next, new_node, relu_node )!=0){
		return errlog(nn,"replace relu failed");
	}

	logmsg(nn,3,"Changed requantize w/ const max --> Relu to --> ReluX");
	return 0;
}

//Replace axisshuffle with chanshuffle if the axis is depth
static int try_optimize_axisshuffle(struct nn_graph *nn, struct nn_node **in_node_p)
{
#define AXISSHUFFLE_INPUT_DATA_IDX 0
#define AXISSHUFFLE_INPUT_AXIS_IDX 1
#define AXISSHUFFLE_INPUT_NUMGROUP_IDX 2
#define AXISSHUFFLE_INPUT_DATA_MIN 3
#define AXISSHUFFLE_INPUT_DATA_MAX 4
#define AXISSHUFFLE_CHANNEL_IDX   3

#define CHANSHUFFLE_NUM_INPUT 4
#define CHANSHUFFLE_NUM_OUPUT 3
#define CHANSHUFFLE_INPUT_NUMGROUP_IDX 0
#define CHANSHUFFLE_INPUT_DATA_IDX 1
#define CHANSHUFFLE_INPUT_DATA_MIN_IDX 2
#define CHANSHUFFLE_INPUT_DATA_MAX_IDX 3

#define SIZE_THRESHOLD 512

    struct nn_node *in_node = *in_node_p;
    int ntyp = in_node->node_type;
	if(OP_AxisShuffle_8 == ntyp) {

        struct nn_node * axis_node = find_node_must_be_Const_from_ref(nn, &in_node->input_refs[AXISSHUFFLE_INPUT_AXIS_IDX]);
        if(axis_node != NULL) {

            int32_t in_axis = tensor_get_int32(in_node->inputs[AXISSHUFFLE_INPUT_AXIS_IDX],0);
            in_axis = handle_negative_axis(in_axis);
            if (-1 == in_axis) return errlog(nn, "AxisShuffle: axis is out of range \n");

            struct nn_node * group_node = find_node_must_be_Const_from_ref(nn, &in_node->input_refs[AXISSHUFFLE_INPUT_NUMGROUP_IDX]);
            if (AXISSHUFFLE_CHANNEL_IDX == in_axis && group_node != NULL) {  //in_axis is depth

                const struct tensor *in_numGroup_tensor = in_node->inputs[AXISSHUFFLE_INPUT_NUMGROUP_IDX];
                const int32_t in_numGroup = tensor_get_int32(in_numGroup_tensor,0);

                if(in_numGroup != 2 && in_numGroup != 4)  return 0;   //only replace op when using OP_QuantizedChannelShuffle_8_d32

                const struct tensor *in_data_tensor = in_node->inputs[0];
                uint32_t size = in_data_tensor->data_size;

                if(size < SIZE_THRESHOLD)  return 0;   //only use OP_QuantizedChannelShuffle_8_d32 when the size is bigger than 512

                int32_t n_new_inputs = CHANSHUFFLE_NUM_INPUT;   //number of inputs of chanshuffle
                int32_t n_new_outputs = CHANSHUFFLE_NUM_OUPUT;

                struct input new_input_refs[n_new_inputs];  //new inputs
                new_input_refs[CHANSHUFFLE_INPUT_NUMGROUP_IDX] = in_node->input_refs[AXISSHUFFLE_INPUT_NUMGROUP_IDX];
                new_input_refs[CHANSHUFFLE_INPUT_DATA_IDX]     = in_node->input_refs[AXISSHUFFLE_INPUT_DATA_IDX];
                new_input_refs[CHANSHUFFLE_INPUT_DATA_MIN_IDX] = in_node->input_refs[AXISSHUFFLE_INPUT_DATA_MIN];
                new_input_refs[CHANSHUFFLE_INPUT_DATA_MAX_IDX] = in_node->input_refs[AXISSHUFFLE_INPUT_DATA_MAX];

                struct output new_output_defs[n_new_outputs];   //new outputs
                memcpy(new_output_defs, in_node->output_defs, n_new_outputs*sizeof(struct output));

                struct nn_node *chan_shuffle_node;  //new node
                if((chan_shuffle_node = optab[OP_QuantizedChannelShuffle_8]->ctor(
                    nn,
                    in_node->node_id,
                    OP_QuantizedChannelShuffle_8,
                    in_node->padding,
                    n_new_inputs,
                    n_new_outputs,
                    new_input_refs,
                    new_output_defs)) == NULL) {
                        return errlog(nn,"can't make new chanshuffle node");
                }

                if( replace_node_with( nn, in_node_p,in_node, chan_shuffle_node) < 0){
                    return errlog(nn,"failed to replace_node for AxisShuffle");
                }
            }
        }
    }
    return 0;
}

static void
requantize_op_minmax_inputs( struct nn_graph *nn, struct nn_node *qdown1_node,
		struct input minmax_in[2])	// set these
{
	if (qdown1_node->node_type == OP_Requantize_32to8) {		// copy from  qdown1_node inputs
		minmax_in[0] = qdown1_node->input_refs[3];
		minmax_in[1] = qdown1_node->input_refs[4];
	}else{
		if (qdown1_node->node_type != OP_QuantizeDownAndShrinkRange_32to8)
			logmsg(nn,0,"bad call to requantize_op_minmax_inputs");
		minmax_in[0].src_id = get_ninf_node(nn);		// set to -/+ infinity
		minmax_in[0].output_idx = 0;
		minmax_in[1].src_id = get_inf_node(nn);
		minmax_in[1].output_idx = 0;
	}
}

static inline struct input gen_zero_input(struct nn_graph *nn)
{
	struct input zero_input = {
		.src_id = get_zero_node(nn),
		.output_idx = 0,
	};
	return zero_input;
};

// if node is one of:
//  OP_QuantizedRelu_8, OP_QuantizedReluX_8, OP_QuantizedClamp_8
// then return 0, and also modify minmax_in[0] and minmax_in[1] to appropriate min and max refs.
//  If it is NULL, or not one of these types, return -1, and don't change anything.
// Also, in case of OP_QuantizedRelu_8, the 'max_in_p' is not changed (presumed to be +inf already).
//
static int __attribute__((unused))
extract_min_max_from_relu(struct nn_graph *nn, struct nn_node const *node, struct input minmax_in[2] )
{
	if( node != NULL){
		if (node->node_type == OP_QuantizedRelu_8) {
			minmax_in[0] = gen_zero_input(nn);
			return 0;
		}else if(node->node_type == OP_QuantizedReluX_8) {
			minmax_in[0] = gen_zero_input(nn);
			minmax_in[1] = node->input_refs[3];
			return 0;
		}else if(node->node_type == OP_QuantizedClamp_8) {
			minmax_in[0] = node->input_refs[3];
			minmax_in[1] = node->input_refs[4];
			return 0;
		}
	}
	return -1;
}

static int maybe_morph_supernode_to_superfc( struct nn_graph *nn, uint32_t node_id,
		int supernode_op, int padding,int num_in, struct input * new_inputs,
		struct output *new_outputs, struct nn_node **resultp);

struct supernode_replacement_desc {
	op_type old_node_type;
	int new_node_type;		// -1 if no 'bias8' support
	int new_node_type_bias32;	// -1 if no 'bias32' support
	int (*extra_checks)(struct nn_graph *, struct nn_node *);	// func to call for extra checks
	int old_n_inputs:8;			// # of inputs on the 'old' operation (usually 7; 6 if no 'stride')
	int skip_if_no_bias:1;		// skip this conversion if the add-bias is missing.
};

//
// before:
//  conv_node:       [ descp->old_node_type ]
//  qdown0_node:     [ QuantizeDownAndShrinkRange_32to8 | OP_Requantize_32to8 ]
//  biasadd_node:    [ OP_QuantizedBiasAdd_8p8to32 ]
//  qdown1_node:     [ QuantizeDownAndShrinkRange_32to8 | OP_Requantize_32to8 ]
//  relu_node (optonal):  [ OP_QuantizedRelu_8 | OP_QuantizedReluX_8 | OP_QuantizedClamp_8 ]
//
// ... replace with supernode based on descp->new_node_type
//
// ** OR ** if descp->new_node_type_bias32 is >=0, we can also accept
//
//  conv_node:       [ descp->old_node_type ]
//  biasadd_node:    [ OP_QuantizedBiasAdd_32p32to32 ]
//  qdown1_node:     (same as above)
//  relu_node (optonal, same as above)
//
// ... and replace with supernode based on descp->new_node_type_bias32
//
// (new: it's allowed for descp->new_node_type to be -1, in which case only the bias32 form is matched.


static int try_make_supernode_flavored(struct nn_graph *nn, struct nn_node **conv_node_p,
		struct supernode_replacement_desc const * descp )
{
	struct nn_node *conv_node = *conv_node_p;
	struct nn_node *qdown0_node;
	struct nn_node *biasadd_node;
	struct nn_node *qdown1_node = NULL;
	struct nn_node *relu_node = NULL;
	struct nn_node *supernode;
	struct nn_node *lastop;
	const int max_num_inputs = 12;
	struct input new_inputs[max_num_inputs];
	struct input minmax_inputs[2];
	struct output new_outputs[3];
	int i;
	op_type old_op_flavor = descp->old_node_type;
	int operation = descp->new_node_type;
	/* Make sure start node is the right kind... */
	if (conv_node->node_type != old_op_flavor) return 0;
	logmsg(nn,9,"found conv flavor %d (node id=%d)",old_op_flavor,conv_node->node_id);
	if ((descp->extra_checks != NULL) && (descp->extra_checks(nn,conv_node) != 0)) return 0;
	// FIXME if (!is_QuantizedConv_with_const_filter(conv_node)) return;
	/* Find the consumer node */
	// note that requant_ops[2] is OP_QuantizedBiasAdd_32p32to32 and this is just for detecting
	// the bias32  case
	int newop_bias32 = descp->new_node_type_bias32;

	static const int requant_ops[3] = { OP_QuantizeDownAndShrinkRange_32to8, OP_Requantize_32to8 , OP_QuantizedBiasAdd_32p32to32};
	static const int relu_ops[3] = { OP_QuantizedRelu_8, OP_QuantizedReluX_8, OP_QuantizedClamp_8 };

	qdown0_node = find_unique_consumer(nn, conv_node, (newop_bias32>=0)?3: 2, requant_ops, 0);
	if( qdown0_node == NULL) return 0;

	// is it a bias32 case?
	if( qdown0_node->node_type == OP_QuantizedBiasAdd_32p32to32 ){	// looks like it is...
		if( newop_bias32 < 0) return 0;			// op doesn't support that
		biasadd_node = qdown0_node;		// this is actually the bias node
		qdown0_node = NULL;				// this is unused
		operation = newop_bias32;
	}else{
		// must be an 8-bit bias situation. Some cases don't support bias8; but we look at that
		// later, and try to transform the bias input to 32 in these cases.
		logmsg(nn,9,"found qdown0");
		biasadd_node =  find_unique_consumer_mustbe( nn, qdown0_node,OP_QuantizedBiasAdd_8p8to32,0 );
		if( biasadd_node == NULL){
			// we can stop matching here, if the qdown0 node was OP_QuantizeDownAndShrinkRange_32to8.
			if( qdown0_node->node_type != OP_QuantizeDownAndShrinkRange_32to8){
				return 0;
			}
		}
	}
	if( biasadd_node != NULL){
		qdown1_node = find_unique_consumer(nn, biasadd_node, 2, requant_ops, 0);
		if( qdown1_node == NULL){
			biasadd_node = NULL;	// pattern incomplete for bias add.
			if( operation == newop_bias32)	// bias32 incomplete, stop .
				return 0;
		}
	}
	if( biasadd_node != NULL){
		logmsg(nn,9,"found biasadd");
		// look for qdown1 and optional relu

		/* And repeat for QuantizeDown #1 */
		requantize_op_minmax_inputs(nn,qdown1_node, minmax_inputs);
		logmsg(nn,9,"found qdown1");

		/* EJP: FIXME: optimize RELU to Requantize_32to8 with const 0 / INF */
		/* EJP: FIXME: optimize RELUX to Requantize_32to8 with const 0 / MAX */
		/* EJP: FIXME: allow plain autorequantize with -INF / INF */
		/* Now repeat for Relu */
		relu_node = find_unique_consumer( nn, qdown1_node, 3,relu_ops, 0);
		if( extract_min_max_from_relu( nn, relu_node,  minmax_inputs ) == 0 ){
			logmsg(nn,9,"found relu/clamp");
			lastop = relu_node;
		}else{
			relu_node = NULL;
			lastop = qdown1_node;
		}
	}else{
		return 0;				// @@@@ Temporary: don't convert if no bias (conversion causes inaccuracy on V65 alexnet)
		if( descp->skip_if_no_bias ) 		// don't do this conversion if there's no bias.
			return 0;
		// don't have an addbias or relu.
		// Fake it with a zero bias node.
		// Also, set the minmax to -inf +inf.
		minmax_inputs[0].src_id = get_ninf_node(nn);
		minmax_inputs[0].output_idx = 0;
		minmax_inputs[1].src_id = get_inf_node(nn);
		minmax_inputs[1].output_idx = 0;
		lastop = qdown0_node;
	}
	//
	// if operation < 0, it means we have an 8-bit bias situation
	// but the pattern doesn't support that. We can still try to
	// convert the bias const to 32 and proceed from there... make sure there's one
	// in the pattern, and set the flag.
	int need_bias32_convert = 0;
	if( operation < 0){
		if( biasadd_node ==NULL || biasadd_node->node_type != OP_QuantizedBiasAdd_8p8to32 || newop_bias32 < 0 ){
			return 0;
		}
		need_bias32_convert = 1;
		operation = newop_bias32;
	}
	/*** WOO we are a good candidate to make a supernode */
	/* 
	 * Embiggen the inputs
	 * Copy the inputs from the nodes:
	 * * all the input args from conv
	 * * Followed by values/min/max for biasadd
	 */
	int n_in = descp->old_n_inputs;

	for (i = 0; i < n_in; i++) {
		new_inputs[i] = conv_node->input_refs[i];
	}
	// we add 5 more inputs: 3 for bias, 2 for  output min/max
	if( biasadd_node != NULL){
		new_inputs[n_in] = biasadd_node->input_refs[1];
		new_inputs[n_in+1] = biasadd_node->input_refs[4];
		new_inputs[n_in+2] = biasadd_node->input_refs[5];
	}else{
		// Need a zero bias input for supernode.
		// depth of supernode output...
		int depth  = conv_node->output_defs[0].max_sizes[3];
		int k = get_zero_flat_qu8_const(nn, depth, &new_inputs[n_in]);	 // fills in 3 slots.
		if( k!= 0){
			errlog(nn, "Failed to make 0-bias size %d\n", depth);
			return -1;
		}
	}

	// try to apply 8->32 bit bias conversion, if applicable.
	// if successful, it creates new bias and range nodes, and updates the three
	// input refs in-place.
	if( need_bias32_convert){
		int res = convert_qu8_const_to_qint32( nn, & new_inputs[n_in],0, 16384 );
		if(res !=0) return 0;		// didn't work...
	}


	new_inputs[n_in+3] = minmax_inputs[0];
	new_inputs[n_in+4] = minmax_inputs[1];
	n_in += 5;

	/* FIXME: struct vals / ordering. */
	/* FIXME: Time to merge fastrpc branch back into master */
	for (i = 0; i < 3; i++) {
		new_outputs[i] = lastop->output_defs[i]; //TODO - When requantize_node->outputs is array of tensors with rank, copy rank and all sizes from it.
	}

	supernode = NULL;
	// see if we can morph this to a SuperFC.
	// the function will return :
	//   -1 if error
	//    0, and supernode = NULL, if it can't;
	//    0  and supernode != NULL, if it did.
	//  Note that the function is allowed to scribble on new_inputs and new_outputs.
	//.
	if( operation == OP_Supernode_8x8p8to8 || operation == OP_Supernode_8x8p32to8){
		int res = maybe_morph_supernode_to_superfc( nn,
				lastop->node_id , operation, conv_node->padding, n_in, new_inputs , new_outputs , &supernode);
		if( res != 0) return -1;
		// supernode != NULL means it did the morphing.
	}

	/* Reuse the outputs & ID from the relu node */
	/* Allocate new node */
	if( supernode == NULL){

		if ((supernode = optab[operation]->ctor(
			nn,
			lastop->node_id,
			operation,
			conv_node->padding,
			n_in,
			3,
			new_inputs,
			new_outputs)) == NULL) return errlog(nn,"ctor fail");
	}
	/* Clean up the old not needed nodes */

	/* note:qdown0_node, relu_node may be null */


	if( replace_nodes(nn,conv_node_p,supernode,
			conv_node, qdown0_node, biasadd_node, qdown1_node, relu_node) !=0 ){
		return errlog(nn, "replace_nodes failed in make_supernode");
	}

	logmsg(nn,3,"Created supernode id=%x (was relu ID)",supernode->node_id);
	return 0;
}

//
// This is called when a OP_Supernode_8x8p8to8 or OP_Supernode_8x8p32to8 is about to
// be created; it checks to see if it can be done as a SuperFC_8x8p32to8.
// If not, return 0 and *resultp = NULL;
// if so, make the new node, return 0 with *resultp set to it.
// 'new_input' and new_output are the arrays which were to be used to make the supernode;
//  we can scribble on these (e.g. we need to delete input #6 (stride) and also
//  we may need to convert bias to 32 bits.
//
// Conditions are:
//   (1) output shape must be [*,1,1,*]  ( for now - batches must be 1).
//   (2) filter shape must be (h,w,*,*) where h,w are the h,w,shape of the input.
//   (3) if h>1 or w>1, the padding must be 'valid'.
//   (4) input depth & output depth of the proposed superfc must be ok.
//
//
int
maybe_morph_supernode_to_superfc( struct nn_graph *nn,
		uint32_t node_id,
		int supernode_op,			// one of OP_Supernode_8x8p8to8, OP_Supernode_8x8p32to8
		int padding,
		int num_in,
		struct input * new_inputs,	// 'new input' array
		struct output *new_outputs,
		struct nn_node **resultp)
{
	*resultp = NULL;
	if( num_in != 12) return errlog(nn,"wrong # inputs");

	if( new_outputs[0].max_sizes[0] != 1 )		// until I figure out how to handle that.
		return 0;

	if( new_outputs[0].max_sizes[1] != 1 || new_outputs[1].max_sizes[2] != 1)
		return 0;
	// find the filter weights
	struct nn_node const *wts = find_node_must_be_Const_from_ref( nn, &new_inputs[1]);
	if( wts == NULL) return errlog(nn,"weights input must be const");
	struct tensor const *wts_tensor = wts->outputs[0];
	int filt_h = wts_tensor->shape.filt_height;
	int filt_w = wts_tensor->shape.filt_width;
	int out_depth = wts_tensor->shape.filt_batches;
	int in_depth = wts_tensor->shape.filt_depth;

	if( (filt_h >1 ||  filt_w > 1) && padding != NN_PAD_VALID ) return 0;
	if( (out_depth&15)!=0 || out_depth < 32) return 0;	// superfc can't do that output depth.

	if( ((filt_h *filt_w * in_depth)&15)!=0 || (filt_h *filt_w * in_depth) < 32) return 0;	// superfc can't do that input depth

	struct nn_node * source_node = find_node( nn, new_inputs[0].src_id);
	struct output const *source_odef = &source_node->output_defs[new_inputs[0].output_idx];
	if( source_odef->max_sizes[1] != filt_h  || source_odef->max_sizes[2] != filt_w ){
		return 0;			// input shape doesn't match filter.
	}
	logmsg(nn,4,"About to morph node 0x%x to SuperFC [%d x %d]", (unsigned)node_id, filt_h, filt_w );

	// that is a good candidate! after here we do it or fail.

	// upgrade u8 bias to int32 if needed.
	if( supernode_op == OP_Supernode_8x8p8to8){
		int res = convert_qu8_const_to_qint32( nn, &new_inputs[7], out_depth, out_depth);
		if( res != 0) return -1;
	}else if( supernode_op != OP_Supernode_8x8p32to8 ) {
		logmsg(nn,0, "bad type!");
		return -1;
	}

	// we delete the stride input 6 now. Note that with the conditions already established,
	// we don't care what the stride is.
	//
	for( int i = 6; i < 11; i++) new_inputs[i] = new_inputs[i+1];
	int new_op = OP_SuperFC_8x8p32to8;

	struct nn_node * newnode = optab[new_op]->ctor( nn,
		node_id, new_op, padding,
		11,3, new_inputs, new_outputs);

	if( newnode == NULL) return errlog(nn,"constructor failed!");

	*resultp = newnode;
	return 0;

}

static inline int warnlog(struct nn_graph *nn, const char *msg)
{
	logmsg(nn,2,msg);
	return -1;
}

static int dwconv_extrachecks(struct nn_graph *nn, struct nn_node *conv_node)
{
	struct nn_node *weights_node;
	struct nn_node *stride_node;
	int weights_id = conv_node->input_refs[1].src_id;
	int weights_idx = conv_node->input_refs[1].output_idx;
	int stride_id = conv_node->input_refs[6].src_id;
	int stride_idx = conv_node->input_refs[6].output_idx;
	if (weights_idx != 0 || (weights_node = find_node_must_be_Const(nn,weights_id)) == NULL)
		return warnlog(nn,"weights node not found, or not const");
	if (stride_idx!= 0 || (stride_node = find_node_must_be_Const(nn,stride_id)) == NULL)
		return warnlog(nn,"stride node not found, or not const");
	if (stride_node->outputs[stride_idx]->shape.width > 2) return warnlog(nn,"horiz stride");
	if (weights_node->output_defs[weights_idx].max_sizes[3] != 1) return warnlog(nn,"depth mult");
	if (weights_node->output_defs[weights_idx].max_sizes[1] != 3) return warnlog(nn,"filt width");
	//if ((weights_node->output_defs[weights_idx].max_sizes[2] % 32) != 0) return errlog(nn,"FIXME: for supernode, depth must be mult of 32 for now.");
	return 0;
}
// batchnorm can only convert multiply when the 'B' size of mult is (1,1,1,d)
static int batchnorm_extrachecks(struct nn_graph *nn, struct nn_node *conv_node)
{
	struct nn_node *weights_node;
	int weights_id = conv_node->input_refs[1].src_id;
	int weights_idx = conv_node->input_refs[1].output_idx;
	if (weights_idx != 0 || (weights_node = find_node_must_be_Const(nn,weights_id)) == NULL)
		return -1;
	for(int i =0; i < 3; i++){
		if (weights_node->output_defs[0].max_sizes[i] != 1) return -1;
	}
	return 0;
}

// matmult can only convert to supernode when the input & output depth
// are supported multiples.
//
static int matmult_extrachecks(struct nn_graph *nn, struct nn_node *conv_node)
{
	struct nn_node *weights_node;
	int weights_id = conv_node->input_refs[1].src_id;
	int weights_idx = conv_node->input_refs[1].output_idx;
	if (weights_idx != 0 || (weights_node = find_node_must_be_Const(nn,weights_id)) == NULL)
		return -1;

	int in_depth = weights_node->outputs[0]->shape.filt_depth;
	int out_depth = weights_node->outputs[0]->shape.filt_batches;
	if( (in_depth&15) != 0		 	// input depth must be multiple of 16
		|| (out_depth &15)!=0		// output depth  must be multiple of 16
		|| out_depth < 32 )			// .. and output depth must be >= 32
		return -1;
	return 0;
}

// don't change a conv2d directly to d32 since there are cases
//  where it will become an input_conv instead.
static const
struct supernode_replacement_desc ReplaceConv_SuperNode = {
	.old_node_type = OP_QuantizedConv2d_8x8to32,
	.new_node_type = OP_Supernode_8x8p8to8,
	.new_node_type_bias32 = OP_Supernode_8x8p32to8,
	.extra_checks = NULL,
	.old_n_inputs = 7,
};
// we could replace depthwise directly to _d32
// but that would defeat merging a 'mul-by-scalar' which follows it.
static const
struct supernode_replacement_desc ReplaceConv_DwiseSuperNode = {
	.old_node_type = OP_QuantizedDepthwiseConv2d_8x8to32,
	.new_node_type = OP_DepthwiseSupernode_8x8p8to8,
	.new_node_type_bias32 = OP_DepthwiseSupernode_8x8p32to8,
	.extra_checks = dwconv_extrachecks,
	.old_n_inputs = 7,
};

static const
struct supernode_replacement_desc ReplaceMult_BatchNorm = {
	.old_node_type = OP_QuantizedMul_8x8to32,
	.new_node_type = OP_QuantizedBatchNorm_8x8p8to8,
	.new_node_type_bias32 = OP_QuantizedBatchNorm_8x8p32to8,
	.extra_checks = batchnorm_extrachecks,
	.old_n_inputs = 6,		// no 'stride' so only 6
	.skip_if_no_bias = 1	// don't convert this if no 'biasadd
};

// replace OP_QuantizedMatMul_8x8to32 and following biasadd, quantize; with OP_SuperFC_8x8p32to8
static const
struct supernode_replacement_desc ReplaceMatMult_Supernode = {
	.old_node_type = OP_QuantizedMatMul_8x8to32,
	.new_node_type = -1,			// no support for 8-bit bias
	.new_node_type_bias32 = OP_SuperFC_8x8p32to8,
	.extra_checks = matmult_extrachecks,
	.old_n_inputs = 6,		// no 'stride' so only 6
	.skip_if_no_bias = 1	// don't convert this if no 'biasadd'.
};

static int try_make_qadd_supernode(struct nn_graph *nn, struct nn_node **qadd_node_p);

// given a 'Shape_int32', check to see if the upstream node is a Dequantize
// and if so, reconnect the shape to the input of that node.
// TODO: many other ops besides Quantize don't change the shape.
// This combo is in inception_resnet_v2 near the end.
static int
try_bypass_for_shape( struct nn_graph *nn, struct nn_node **shape_node_p)
{
	struct nn_node * shape_node = *shape_node_p;
	if( shape_node->n_inputs <1 || shape_node->input_refs[0].output_idx != 0) return 0;
	uint32_t prod_node_id = shape_node->input_refs[0].src_id;
	struct nn_node * prod_node = find_node_must_be( nn, prod_node_id, OP_Dequantize);
	if( prod_node == NULL || prod_node->n_inputs < 1 ) return 0;
	// rewire shape node to quant node source
	shape_node->input_refs[0] = prod_node->input_refs[0];
	node_rehash_inputrefs( shape_node );
	logmsg(nn,4,"moved input of Shape(%x) to input of previous node", (unsigned)shape_node->node_id);
	return 0;
}
//
// make supernode
// Also look for Shape_int32 which are connected output of Quantize
//
static int try_make_supernode(struct nn_graph *nn, struct nn_node **conv_node_p)
{
	struct supernode_replacement_desc const * sdescp = NULL;

	int ntyp = (*conv_node_p)->node_type;

	switch( ntyp){
	 case OP_Shape_int32:
		// not a supernode, but can be done in this pass..
		return try_bypass_for_shape( nn, conv_node_p);

	 case OP_QuantizedAdd_8p8to32:
		return try_make_qadd_supernode( nn, conv_node_p);

	 case OP_QuantizedConv2d_8x8to32:
		sdescp = &ReplaceConv_SuperNode;
		break;

	 case OP_QuantizedDepthwiseConv2d_8x8to32:
		sdescp = &ReplaceConv_DwiseSuperNode;
		break;

	 case OP_QuantizedMul_8x8to32:
		sdescp = &ReplaceMult_BatchNorm;
		break;

	 case OP_QuantizedMatMul_8x8to32:
		 sdescp = &ReplaceMatMult_Supernode;
		 break;

	 default:
		break;
	}

	if( sdescp != NULL)
		return try_make_supernode_flavored(nn,conv_node_p, sdescp);
	return 0;
}


static inline int graph_iterator(struct nn_graph *nn, int (*f)(struct nn_graph *nn, struct nn_node **trynode))
{
	struct nn_node **root;
	for (root = &nn->head;  *root != NULL; root = (*root == NULL) ? root : &((*root)->next)) {
		if (f(nn,root) != 0) return errlog(nn,"function returned error");
	}
	return 0;
}

static int make_autorequantize(struct nn_graph *nn)
{
	return graph_iterator(nn,try_fuse_requantization_range);
}

static int make_supernodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_supernode);
}

static int make_reluX_nodes(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_reluX);
}

static int make_optimize_axisshuffle (struct nn_graph *nn)
{
	return graph_iterator(nn,try_optimize_axisshuffle);
}

static int try_pad_bad_supernodes(struct nn_graph *nn, struct nn_node **src_node_p)
{
	struct nn_node *src_node = *src_node_p;
	struct nn_node *dst_node;
	int depth;
	int depth_pad;
	int pad_amt;
	struct nn_node *src_filts;
	struct nn_node *src_bias;
	struct nn_node *dst_filts;
	struct nn_node *dst_filts_min;
	struct nn_node *dst_filts_max;
	float filt_min_val;
	float filt_max_val;
	int dst_filt_offset;
	/* Make sure start node is the right kind... */
	if (src_node->node_type != OP_Supernode_8x8p8to8) return 0;
	/* Find the consumer node */
	dst_node = find_unique_consumer_mustbe( nn,src_node, OP_Supernode_8x8p8to8,0);
	if( dst_node == NULL) return 0;

	/* Don't pad if sufficiently aligned */
	if ((src_bias = find_node_must_be_Const(nn,src_node->input_refs[7].src_id)) == NULL) return 0;
	if (((depth = src_bias->outputs[0]->shape.depth) % 32) == 0) return 0;
	/* Don't pad too much, it increases later node complexity a lot */
	/* FIXME: needs better support for quantizing back down in op_supernode. */
	//if (depth <= 48) return 0;
	//if (depth == 80) return 0;
	/* FIXME: make sure const nodes are not referenced more than once, or dup them? */
	depth_pad = (depth + 31) & (~31);
	pad_amt = depth_pad - depth;
	/* Find dst_filt_max / dst_filt_min */
	if ((src_filts = find_node_must_be_Const(nn,src_node->input_refs[1].src_id)) == NULL) return 0;
	if ((dst_filts = find_node_must_be_Const(nn,dst_node->input_refs[1].src_id)) == NULL) return 0;
	if ((dst_filts_min = find_node_must_be_Const(nn,dst_node->input_refs[4].src_id)) == NULL) return 0;
	if ((dst_filts_max = find_node_must_be_Const(nn,dst_node->input_refs[5].src_id)) == NULL) return 0;
	filt_min_val = tensor_get_float(dst_filts_min->outputs[0],0);
	filt_max_val = tensor_get_float(dst_filts_max->outputs[0],0);
	dst_filt_offset = quantize_uint8(0.0f,filt_min_val,filt_max_val);
	logmsg(nn,2,"found bad depth %d @ nodeid %x padding by %d to %d offset %d",
		depth,
		src_node->node_id,
		pad_amt,
		depth_pad,
		dst_filt_offset);
	if (const_depth_extend_8(src_bias,pad_amt,0) != 0) return -1;
	if (const_depth_extend_8(src_filts,pad_amt,0) != 0) return -1;
	if (const_width_extend_8(dst_filts,pad_amt,dst_filt_offset) != 0) return -1;
	src_node->outputs[0]->max_size = ((uint64_t)src_node->outputs[0]->max_size * depth_pad) / depth;
	logmsg(nn,2,"Successfully prepadded supernodes %x and %x",src_node->node_id,dst_node->node_id);
	return 0;
}

static int pad_bad_supernodes(struct nn_graph *nn)
{
	struct nn_node **root;
	for (root = &nn->head; *root != NULL; root = &((*root)->next)) {
		if (try_pad_bad_supernodes(nn,root) != 0) return errlog(nn,"bad failure");
	}
	return 0;
}
//
// compare the shapes of two output descriptors
//  - return -1 if they are different and incompatible
//  - return an 8-bit code if they are different, but are compatible by
//    broadcasting B to match A. I.e in all non-matching dims, B size = 1.
//
// The code has each of bit 0..3 set if broadcast is needed on dims 0..3 (B..D).
// bits 4..7 is the same, except that when both dims are 1, these bits are 1 (whereas the lower bit is 0)
//
// (Usually, if you need a dim to *not* have broadcast, then you will check for a 0 in the lower
// 4 bits; if you need it to *have* broadcast, you will check for a 1 in the upper 4 bits). A 1 vs 1
// situation will satisfy both of these).
//
// In the case where shapes are identical, the result will be xxxx0000 (and the bits 7..4 indicate
//  which dims are 1). See first 3 examples
//
//   Examples
//      (2,4,5,32)   (2,4,5,32)    result =  0x00  0000 0000
//      (2,4,1,32)   (2,4,1,32)    result =  0x40  0100 0000
//      (1,1,5,32)   (1,1,5,32)    result =  0x30  0011 0000
//      (2,4,2,32)   (1,4,1,32)    result =  0x55  0101 0101
//      (2,4,5,32)   (2,4,5,1)     result =  0x88  1000 1000
//      (2,4,2,32)   (1,1,1,32)	   result =  0x77  0111 0111
//      (2,1,1,32)   (1,1,1,32)    result =  0x71  0111 0001
//      (2,1,1,32)   (2,1,5,32)    result = -1 (only B broadcasts)
//      (2,4,2,32)   (3,4,2,32)    result = -1
//
static int check_broadcast_dims( struct output const * outA, struct output const * outB )
{
	int res = 0;
	for( int i = 0; i < 4; i++ ){
		uint32_t szA = outA->max_sizes[i];
		uint32_t szB = outB->max_sizes[i];
		if( szA != szB ){
			if( szB != 1) return -1;
			res |= (0x11<<i);
		}else if( szA == 1){
			// This is the case where they are equal, but both == 1...
			res |= (0x10<<i);
		}
	}
	return res;
}
// this is like check_broadcast_dims except it supports reverse broadcasting, A onto B.
//
static int check_broadcast_dims_either_way( struct output const * outA, struct output const * outB )
{
	int result = check_broadcast_dims( outA, outB);
	if( result < 0 ) result = check_broadcast_dims(outB,outA);
	return result;
}

static int check_dims_equal(struct output const *outA, struct output const *outB)
{
	for( int i = 0; i < 4; i++ ) {
		uint32_t szA = outA->max_sizes[i];
		uint32_t szB = outB->max_sizes[i];
		if (szA != szB) return -1;
	}
	return 0;
}

static int check_spaceop_ok_for_d32(struct nn_graph * nn, struct nn_node * srcnode );
static int check_batchop_ok_for_d32(struct nn_graph * nn, struct nn_node * srcnode );
static int get_blocksize_values( struct nn_graph *nn, struct input inref,  int * blocksize_h, int * blocksize_w);

static int depth32_replacement_op(struct nn_graph * nn, struct nn_node * srcnode )
{
	int try_op = srcnode->node_type;
	if( (srcnode->flags &NN_NODE_FLAG_NO_CONVERT_D32) !=0)
		return -1;
	int new_op = -1;
	switch (try_op){
	 case OP_Supernode_8x8p8to8:
	 case OP_Supernode_8x8p32to8:
	 {
			struct nn_node *weights;
			if ((weights = find_node_must_be_Const(nn,srcnode->input_refs[1].src_id)) == NULL) {
				return errlog(nn,"supernode weights not found, or not const");
			}
			struct shape const * wshape = & weights->outputs[0]->shape;
			/* Don't use D32 supernode if filter is large and we have SAME padding */
			if( wshape->filt_height > 9 || wshape->filt_width > 9 ){
				if( (srcnode->padding == NN_PAD_SAME)
						|| (srcnode->padding == NN_PAD_SAME_CAFFE))  return -1;
			}
			/* If the input depth is small, we want a different op */
			/* return OP_InputSupernode_8x8p8to8_outd32 to flag this */
			if (wshape->filt_depth <= 4)
				return OP_InputSupernode_8x8p8to8_outd32;
			return (try_op==OP_Supernode_8x8p8to8)? OP_Supernode_8x8p8to8_d32: OP_Supernode_8x8p32to8_d32;
	 }

	 case OP_DepthwiseSupernode_8x8p8to8:   return OP_DepthwiseSupernode_8x8p8to8_d32;
	 case OP_DepthwiseSupernode_8x8p32to8:  return OP_DepthwiseSupernode_8x8p32to8_d32;
	 case OP_QuantizedBatchNorm_8x8p8to8:	return OP_QuantizedBatchNorm_8x8p8to8_d32;
	 case OP_QuantizedBatchNorm_8x8p32to8:	return OP_QuantizedBatchNorm_8x8p32to8_d32;
	 case OP_QuantizedMaxPool_8:            return OP_QuantizedMaxPool_8_d32;
	 case OP_QuantizedAvgPool_8:            return OP_QuantizedAvgPool_8_d32;
	 case OP_QuantizedPRelu_8:              return OP_QuantizedPRelu_8_d32;
	 case OP_QuantizedConcat_8:             return OP_QuantizedConcat_8_d32;
	 case OP_Convert_from_aix:              return OP_Convert_from_aix_d32;
	 case OP_QuantizedSoftmax_8:
	   {  // convert only if
		  //    (1) input is a Convert_from_d32 (i.e. input already in d32)
		  // or (2) height >=4 and (width >= 8 or width == 4) and depth != 2
		  //
		  // Also, for now we *don't* convert if the input
		  // is from a SuperFC_8x8p32to8 (due to shape disagreement issues)
		  //
		  op_type converted_op = OP_QuantizedSoftmax_8_d32;
		  struct nn_node * src = find_node(nn,srcnode->input_refs[0].src_id);
		  if( src != NULL ){
			  if(src->node_type == OP_Convert_from_d32 )
				  return converted_op;
			  if(src->node_type == OP_SuperFC_8x8p32to8)
				  return -1;
		  }
		  int op_ht = srcnode->output_defs[0].max_sizes[1];
		  int op_width = srcnode->output_defs[0].max_sizes[2];
		  int op_depth = srcnode->output_defs[0].max_sizes[3];

		  if( op_ht >= 4 && (op_width >= 8 ||  op_width == 4 ) && op_depth >2)
			  return converted_op;
		  return -1;
	   }
	 case OP_QuantizedLRN_8:
	  {
		struct nn_node *window;
		if ((window = find_node_must_be_Const(nn,srcnode->input_refs[3].src_id)) == NULL) {
			return errlog(nn,"LRN window not found, or not const");
		}
		if( ! shape_X11X(&window->outputs[0]->shape )) return -1;
		return OP_QuantizedLRN_8_d32;
	  }
	 case OP_QuantizedSub_8p8to8:  // this op is currently not supported in non-d32
		 return OP_QuantizedSub_8p8to8_d32;

	 case OP_QuantizedAdd_8p8to8:
	  {
		struct nn_node *src0;
		struct nn_node *src1;
		if ((src0 = find_node(nn,srcnode->input_refs[0].src_id)) == NULL) return errlog(nn,"src0 not found");
		if ((src1 = find_node(nn,srcnode->input_refs[1].src_id)) == NULL) return errlog(nn,"src1 not found");
		int src0_idx = srcnode->input_refs[0].output_idx;
		int src1_idx = srcnode->input_refs[1].output_idx;
		// add_d32/sub_32: supports:
		// - any case broadcasting from B and/or H only (detected as 00xx in lower 4 bits of bcode)
		// - any case where B_shape = (1,1,*,*)  (broadcast on B and H, and optionally on W and/or D)
		//		(detected as xx11 in upper 4 bits). For this case 'A'  can be size 1 on B and/or D.
		// Also supports broadcasting from A onto B with the same restrictions.
		//
		// Convert only if input 0 or 1 is convert_from_d32 (i.e. input already in d32) and no broadcast
		if (    (!((src0->node_type == OP_Convert_from_d32) || (src1->node_type == OP_Convert_from_d32))) 
			&& (check_dims_equal(&src0->output_defs[src0_idx],&src1->output_defs[src1_idx]) == 0)) {
#if 0
			logmsg(nn,2,"src0.od[0]: %d,%d,%d,%d src1.od[0]: %d,%d,%d,%d",
				src0->output_defs[src0_idx].max_sizes[0],
				src0->output_defs[src0_idx].max_sizes[1],
				src0->output_defs[src0_idx].max_sizes[2],
				src0->output_defs[src0_idx].max_sizes[3],
				src1->output_defs[src1_idx].max_sizes[0],
				src1->output_defs[src1_idx].max_sizes[1],
				src1->output_defs[src1_idx].max_sizes[2],
				src1->output_defs[src1_idx].max_sizes[3]);
			logmsg(nn,2,"add or sub: dims equal and non-d32");
#endif
			return -1;
		}
		int bcode = check_broadcast_dims_either_way( &src0->output_defs[0], &src1->output_defs[0]);
		if( bcode < 0		// # <0 means incompatible even with broadcasting
			|| !( (bcode& 0x0C) == 0 || ( bcode & 0x30) == 0x30 ))
			  return -1;
		 return (try_op==OP_QuantizedAdd_8p8to8)? OP_QuantizedAdd_8p8to8_d32: OP_QuantizedSub_8p8to8_d32;
	  }
	 case OP_QuantizedPad_8:		// this can be converted if it does not do depth padding.
	 {
		if( srcnode->n_inputs < 4) return -1;
		struct nn_node * src = find_node(nn,srcnode->input_refs[0].src_id);
		if (src == NULL) return errlog(nn,"Oops, can't find src of pad op");
		if (src->node_type != OP_Convert_from_d32) return -1;		// Only use pad d32 if input is d32 format
		struct nn_node * dims_node = find_node_must_be_Const(nn,srcnode->input_refs[3].src_id);
		if( dims_node == NULL || srcnode->input_refs[3].output_idx != 0 ){
			 return -1;
		}
		// the dims tensor is of shape [n,2] where n can be 1,2,3,4; each row is pad_before, pad_after.
		// if n <= 3, there is no depth padding.
		struct tensor const * dims = dims_node->outputs[0];
		int nvals = dims->data_size / (2*sizeof(int32_t));		// this is 'n'
		if( nvals > 3 ){
			// may include depth padding. Can't convert if so.
			int32_t const *ptr = (int32_t const*)dims->data;
			if( ptr[3*2]!= 0 || ptr[3*2+1]!= 0) return -1;
		}
		return OP_QuantizedPad_8_d32;
	 }
      //only for testing instance norm hvx for now
	 case OP_QuantizedInstanceNorm_8:       return OP_QuantizedInstanceNorm_8_d32;
	 case OP_QuantizedInstanceNormBG_8:     return OP_QuantizedInstanceNormBG_8_d32;
	 case OP_QuantizedInstanceNormBG_8_ref: return OP_QuantizedInstanceNormBG_8_d32_ref;

	 //case OP_QuantizedRelu_8:
	 //	 new_op = OP_QuantizedRelu_8_d32;	// convert only if input 0 is already convert-from-d32.
	//	 break;
	 case OP_QuantizedTanh_8:
		 new_op = OP_QuantizedTanh_8_d32;	// convert only if input 0 is already convert-from-d32.
		 break;
	 case OP_QuantizedSigmoid_8:
		 new_op = OP_QuantizedSigmoid_8_d32;	// convert only if input 0 is already convert-from-d32.
		 break;

	 case OP_QuantizedNeg_8:
		 new_op = OP_QuantizedNeg_8_d32;	// convert only if input 0 is already convert-from-d32.
		 break;

	 // QuantizedChannelShuffle can only be converted if it has an interleave factor supported by the d32 node
	 case OP_QuantizedChannelShuffle_8:
	  {
		int n_out = srcnode->n_outputs-2;	// actual # of outputs
		struct nn_node *const_node = find_node_must_be_Const( nn, srcnode->input_refs[0].src_id);
		if( n_out < 1 || const_node==NULL) return -1;
		int k_val=  tensor_get_int32( const_node->outputs[0], 0);

		if( k_val != 4 && k_val != 2) return -1;	 // << add supported k values here.
		return OP_QuantizedChannelShuffle_8_d32;
	  }
	  // currently: QuantizedSplit will only convert if the split is on dim #3.
	  // There is a convention that if inputs #0 and inputs #1 are wired to
	  // the same place, that means split on 3; otherwise #0 is connected to a constant which must be 3.
	  //
	 case OP_QuantizedSplit_8:
	  {
		if( srcnode->n_inputs != 4) return -1;
		struct input const * inps = srcnode->input_refs;
		if( inps[0].src_id == inps[1].src_id && inps[0].output_idx == inps[1].output_idx ){
			return OP_QuantizedSplit_8_d32;			// this is ok to convert;
		}
		struct nn_node *const_node = find_node_must_be_Const( nn, inps[0].src_id);
		if( srcnode->n_outputs < 4 || const_node==NULL) {
			return -1;
		}
		int dimno =  tensor_get_int32( const_node->outputs[0], 0);
		// when QuantizedSplit_8_d32 can handle splits on other dims, add that here.
		if( dimno != 3) return -1;
		return OP_QuantizedSplit_8_d32;
	  }
	 case OP_DepthToSpace_8:
	  {
		if( check_spaceop_ok_for_d32(nn,srcnode) == 0 )
			return OP_DepthToSpace_8_d32;
		return -1;
	  }
	  case OP_QuantizedResizeBilinear_8: 
	  {
		if (srcnode->output_defs[0].max_sizes[3] > 2) return OP_QuantizedResizeBilinear_8_d32;
		return -1;
	  }
	  case OP_BatchToSpaceND_8:
		if( check_batchop_ok_for_d32(nn,srcnode) != 0 ) return -1;
		new_op = OP_BatchToSpaceND_8_d32;
		break;
	  case OP_SpaceToBatchND_8:
		if( check_batchop_ok_for_d32(nn,srcnode) != 0 ) return -1;
		new_op = OP_SpaceToBatchND_8_d32;
		break;
	 default:
			return -1;
	}
	// if it falls out of the switch, new_op is the new operation,
	// but only if input0 currently comes from a Convert_from_d32 node.
	// otherwise, leave it as is. This is for ops which are not really
	// any more efficient in d32 mode.
	if( new_op > 0){
		struct nn_node * inp0_node = NULL;
		if( srcnode->n_inputs >=1 ){
			inp0_node = find_node_must_be( nn, srcnode->input_refs[0].src_id, OP_Convert_from_d32);
			if( inp0_node == NULL )
				return -1;
		}
	}
	return new_op;
}

// return  0  if an OP_BatchToSpaceND_8 or OP_SpaceToBatchND_8 is OK for conversion
// to d32.
//   - inputs #1 and #2 must be const
//   - if input #1 has depth = 2, the second element (blocksize_w) must be 1..4
//

static int
check_batchop_ok_for_d32(struct nn_graph * nn, struct nn_node * srcnode )
{
	if( srcnode->n_inputs < 5) return -1;		// not right
	struct nn_node *bsize_node  = find_node_must_be_Const_from_ref(nn, &srcnode->input_refs[1]);
	struct nn_node *padding_node  = find_node_must_be_Const_from_ref(nn, &srcnode->input_refs[2]);

	if( bsize_node == NULL || padding_node == NULL)return -1;

	struct tensor const * bsize_tensor = bsize_node->outputs[0];
	if( bsize_tensor->shape.depth < 2) return 0;		// it's ok, bsize_w is implied = 1
	int bsize_w = tensor_get_int32( bsize_tensor, 1);
	if( bsize_w >=1 && bsize_w <= 4) return 0;
	return 1;
}

//
// return 0 if the supplied OP_DepthToSpace_8 is suitable to be transformed
// to OP_DepthToSpace_8_d32 (expected to also handle SpaceToDepth, BatchToSpace etc
// as these are implemented)
//
//  constraints for OP_DepthToSpace_8_d32:
//     - blocksize_w must be 1,2,3 or 4
//     - output depth must be a multiple of 32.
//  OR
//     - blocksize_w must be 2
//     - output depth must be 16.
//
static int
check_spaceop_ok_for_d32(struct nn_graph * nn, struct nn_node * srcnode )
{
	int blocksize_h, blocksize_w;
	if( srcnode->n_inputs < 4) return -1;		// not right
	// get input depth
	struct input datain_ref = srcnode->input_refs[0];
	struct nn_node * datain_node = find_node( nn, datain_ref.src_id );
	if(datain_node == NULL) return -1;
	int input_depth = datain_node->output_defs[ datain_ref.output_idx].max_sizes[3];
	if( input_depth%32u!= 0)
		return -1;

	if( get_blocksize_values( nn, srcnode->input_refs[1], &blocksize_h, &blocksize_w)!= 0){
		return -1;
	}
	if( blocksize_w > 4) return -1;			// not eligible for d32
	// now find output depth. Must
	unsigned bsprod = mulu32_sat(blocksize_h, blocksize_w );
	int out_depth = (unsigned)input_depth/bsprod;
	if( out_depth == 0 || out_depth * bsprod != input_depth  ){
		return -1;		// can't convert, not sane
	}
	if(  !(out_depth==16 && blocksize_w==2) && (out_depth&31) != 0 ){
		return -1;      // can't convert, not supported
	}
	return 0;
}


// get the blocksize values, given a 'struct input' containing the reference to the const.
//  - reference must be to a const
//  - shape must be [1,1,1,1] or [1,1,1,2] of 4-byte elements
//  - elements in array are [blocksize] or [blocksize_h, blocksize_w]
//  - elements must be >=1
//
static int
get_blocksize_values( struct nn_graph *nn, struct input inref,  int * blocksize_h, int * blocksize_w)
{
	struct nn_node * const_node = find_node_must_be_Const_from_ref( nn, &inref);
	if( const_node == NULL ) return -1;

	struct tensor const * ctens = const_node->outputs[0];
	int nel = ctens->shape.depth;
	if( nel < 1 ||  nel > 2 || ctens->data_size != nel*sizeof(int32_t)) return -1;
	int32_t const * bsp = (int32_t const*) ctens->data;
	int bsh = bsp[0];
	int bsw = bsp[nel-1];
	*blocksize_h = bsh;
	*blocksize_w = bsw;
	return min_i32(bsh,bsw)>=1 ? 0: -1;
}


//
// currently all the uses of this are for Convert to/from d32;
// so elsize =1.
//
static struct nn_node *create_convert(
	struct nn_graph *nn,
	int src_id,
	int output_idx,
	struct shape outsize,
	uint32_t operation)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp = { .src_id = src_id, .output_idx = output_idx, };
	struct output outp;
	make_outputdesc_from_shape( &outp, &outsize, /*elsize=*/1, /*pad_d32=*/ 0);
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,1,1,&inp,&outp);
	return new_node;
}

static struct nn_node *create_convert_to_d32(struct nn_graph *nn, int src_id, int output_idx)
{
	struct shape outsize;
	struct nn_node *producer;
	// Validate that data producer feeding this node has been initialized
	if ((producer = find_node(nn,src_id)) == NULL) {
		logmsg(nn,0,"producer not found");
		return NULL;
	}
	shape_from_outdesc( &outsize, &producer->output_defs[output_idx], /*add_d32_pad=*/ 0);
	return create_convert(nn,src_id,output_idx,outsize,OP_Convert_to_d32);
}

//
// given a source ref 'input_ref', and a node 'must_precede' in the linklist,
// look for a Convert_to_d32 which:
//    (1) is reading from the source ref (and therefore after it)
//    (2) has exactly one input (no 'special padding' options)
//    (3) is before 'must_precede'
// .. return a pointer to that node, or NULL if there is none.
//
static struct nn_node *
find_existing_tod32( struct nn_graph *nn,
		struct input input_ref,					// look for a conversion from this.
		struct nn_node const *src_node,			// the node corresponding to input_ref.src_id (or NULL)
		struct nn_node const * must_precede )
{
	if( src_node == NULL){
		src_node = find_node(nn,input_ref.src_id);
		if( src_node == NULL) return NULL;
	}
	struct nn_node * tmp = src_node->next;
	// look downstream for a conversion before must_precede
	//
	while( tmp != NULL){
		if( tmp == must_precede) return NULL;
		if( tmp->node_type == OP_Convert_to_d32
			&& tmp->n_inputs == 1
			&& tmp->input_refs[0].src_id == input_ref.src_id
			&& tmp->input_refs[0].output_idx == input_ref.output_idx ){
			return tmp;
		}
		tmp = tmp->next;
	}
	return NULL;
}


// This is like create_convert_to_d32, but it will check to see if the source is a OP_Convert_from_d32 and
// if it is, it will bypass that.
//
// Otherwise - it will look for an existing Convert_to_d32 of the same source; the existing one is only used
// if it appears *before* target_node in the list, and does not have any extra ('config') inputs. If that
// is found, its output will be used.
// If both of those fail, it will make a Convert_to_d32
//
// Normally returns 0:
//   If it made a new node, then *result_nodep points to it, and *result_inpref is set up to point to that node;
//   If it didn't need to make a new node, *result_nodep is set to NULL, and *result_inpref is set to the source
//     of a d32 tensor (the input of from_d32, or output of exisiting to_d32).
// Returns -1 on error.

static int
need_convert_to_d32(struct nn_graph *nn,
		struct nn_node * target_node,					// point to node which needs the convert as its input
		struct input current_inpref,					// current input ref
		struct input * result_inpref,						// the id:out_index is returned here (new input ref)
		struct nn_node ** result_nodep )					// new convert node, or NULL if we didn't need one.
{
	*result_nodep = NULL;
	struct nn_node *producer =  find_node(nn,current_inpref.src_id);
	if (producer == NULL) {
		logmsg(nn,0,"producer not found");
		return -1;
	}
	if( current_inpref.output_idx == 0 && producer->node_type == OP_Convert_from_d32){
		// we can use the input of that node
		*result_inpref = producer->input_refs[0];
		//printf("bypassed from 0x%X:%d\n", (int)result_inpref->src_id, (int)result_inpref->output_idx );
		return 0;
	}
	//
	// is there an existing tod32 with the same input that we can use?
	//
	{
		struct nn_node * existing_tod32 = find_existing_tod32( nn, current_inpref, producer, target_node);
		if( existing_tod32 != NULL){
			logmsg(nn,3,"reuse node %X for convert_to_d32({%X,%d})", existing_tod32->node_id,
					current_inpref.src_id, current_inpref.output_idx );
			result_inpref->src_id = existing_tod32->node_id;
			result_inpref->output_idx = 0;
			return 0;
		}
	}

	// otherwise make a new one
	struct shape outsize;
	shape_from_outdesc( &outsize, &producer->output_defs[current_inpref.output_idx], /*add_d32_pad=*/ 0 );
	struct nn_node * new_node =  create_convert(nn,current_inpref.src_id,current_inpref.output_idx,outsize,OP_Convert_to_d32);

	if( new_node == NULL) return -1;
	result_inpref->src_id = new_node->node_id;
	result_inpref->output_idx = 0;
	*result_nodep = new_node;
	return 0;

}


static struct nn_node *create_convert_from_d32(struct nn_graph *nn, int src_id, int output_idx, struct shape outsize)
{
	return create_convert(nn,src_id,output_idx,outsize,OP_Convert_from_d32);
}
static struct nn_node *create_autoquantize(struct nn_graph *nn, struct nn_node *srcnode, int src_idx)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp = { .src_id = srcnode->node_id, .output_idx = src_idx, };
	struct output outp[3];
	uint32_t operation = OP_AutoQuantize;

	/* output 0 is the same size, but elementsize 1 */
	outp[0] = srcnode->output_defs[0];
	outp[0].elementsize = 1;
	/* Ouptut 1 and 2 are float max and min */
	outp[1] = Output_ScalarFloat;
	outp[2] = Output_ScalarFloat;
	logmsg(nn,4,"creating autoquant op %x",new_node_id);
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,1,3,&inp,&outp[0]);
	return new_node;
}
//
// make a 'requantize' mode which reads from the given node.
//
static struct nn_node *create_requantize(struct nn_graph *nn, struct nn_node *srcnode)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp[3] = {
		{ .src_id = srcnode->node_id, .output_idx = 0, },
		{ .src_id = srcnode->node_id, .output_idx = 1, },
		{ .src_id = srcnode->node_id, .output_idx = 2, },};
	struct output outp[3];
	uint32_t operation = OP_QuantizeDownAndShrinkRange_32to8;
	/* output 0 is the same size, but elementsize 1 */
	outp[0] = srcnode->output_defs[0];
	outp[0].elementsize = 1;
	/* Ouptut 1 and 2 are float max and min */
	outp[1] = Output_ScalarFloat;
	outp[2] = Output_ScalarFloat;
	logmsg(nn,4,"creating requant op %x",new_node_id);
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,3,3,&inp[0],&outp[0]);
	return new_node;
}
//
// make a 'dequantize' mode which reads from the given node.
//
static struct nn_node *create_dequantize(struct nn_graph *nn, struct nn_node *srcnode)
{
	struct nn_node *new_node;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input inp[3] = {
		{ .src_id = srcnode->node_id, .output_idx = 0, },
		{ .src_id = srcnode->node_id, .output_idx = 1, },
		{ .src_id = srcnode->node_id, .output_idx = 2, },};
	struct output outp = srcnode->output_defs[0];
	uint32_t operation = OP_Dequantize;
	/* output 0 is the same size, but elementsize 4 */
	outp.elementsize = 4;
	logmsg(nn,4,"creating dequant op %x",new_node_id);
	new_node = optab[operation]->ctor(nn,new_node_id,operation,NN_PAD_NA,3,1,&inp[0],&outp);
	return new_node;
}

static int change_refs(struct nn_graph *nn, int old_id, int old_out_idx, int new_id, int new_out_idx)
{
	struct nn_node *node;
	int i;
	noderefhash_set_t hashmask = noderefhash_mask(old_id);

	for (node = nn->head; node != NULL; node = node->next) {
		int changed = 0;
		if( (node->noderefhash & hashmask)!=0){
			for (i = 0; i < node->n_inputs; i++) {
				if ((node->input_refs[i].src_id == old_id)
					&& (node->input_refs[i].output_idx == old_out_idx)) {
					//logmsg(nn,0,"changing node %x to %x:%d",node->node_id,new_id,new_out_idx);
					node->input_refs[i].src_id = new_id;
					node->input_refs[i].output_idx = new_out_idx;
					changed = 1;
				}
			}
			if( changed) node_rehash_inputrefs( node);
		}
	}
	return 0;
}


/*
 *  This is used to convert the following ops to d32:
 *    QuantizedConcat_8
 *    QuantizedChannelShuffle_8
 *    QuantizedSplit_8
 *  The 'new' optype is supplied as new_operation, and is one of the above with _d32 added
 *
 *   All of these node types have the same port pattern:
 *
 *   Inputs: 3*n_in+1
 *        0                 -> a scalar int paramater
 *        1 .. n_in         -> tensor inputs
 *        n_in+1 .. 2*n_in  -> the minima
 *        2*n_in+1 .. 3*n_in  -> the maxima
 *
 *   Outputs: n_out+2
 *        0 .. n_out-1     -> tensor outputs
 *        n_out            -> output minimum
 *        n_out+1          -> output maximum
 *
 *  for   QuantizedConcat_8, n_out is always = 1
 *  for  QuantizedSplit_8, n_in is always = 1.
 *
 *
 * * Are input dims OK?
 * Create convert_to_d32 before
 * Create convert_from_d32 after
 * Change source to convert_to_d32 output
 * Point all consumers to new node 
 * Replace the one node with all the new ones.
 *
 * A special case: for OP_QuantizedSplit_8 , we need to check if inputs #0,#1 are wired to the same source;
 * if so, the #0 needs to be reconnected in the same way as #1.
 */
static int do_convert_concat_depth32(struct nn_graph *nn, struct nn_node **nodeptr, int new_operation)
{
	//struct nn_node *convert_to_headnode = NULL;
	struct nn_node *srcnode = *nodeptr;
	struct nn_node *convert_from_node;
	struct nn_node *new_d32node;
	//struct nn_node *tmp;

	int node_ins = srcnode->n_inputs;
	int node_outs = srcnode->n_outputs;
	int i;
	int src_id;
	int src_oidx;

	// sanity test; protect from breaking the stack
	if( node_ins < 4 || node_ins >(1+3*1024) || node_outs < 3 || node_outs > 1024+2){
		return 0;
	}
	// actual # of tensors in and out (the number of conversions needed)

	int n_inputs = (node_ins-1)/3;			// 1..1024
	int n_outputs = node_outs -2;			// 1..1024

	// some variable sized arrays ...
	struct output new_outputs[node_outs];
	struct input new_inputs[node_ins];
	// pointers to all the new nodes, in order of execution:
	// [0.. n_inputs-1]:  conv_to_d32 nodes
	// [n_inputs]      :   new concat node
	// [n_inputs+1 .. n_inputs+n_outputs]   : conv_from_d32 node
	struct nn_node *new_nodes[n_inputs+1+n_outputs];
	const struct nn_node ** new_from_d32_nodes = (const struct nn_node **)&new_nodes[n_inputs+1];

	if( node_ins != 3*n_inputs+1){
		logmsg(nn,0,"%s node has %d inputs", hexagon_nn_op_names[srcnode->node_type], node_ins );
		return 0;	// that's not right
	}

	logmsg(nn, 2, "Converting %s %p to %s", hexagon_nn_op_names[srcnode->node_type], srcnode, hexagon_nn_op_names[new_operation]);
	if ((srcnode->output_defs[0].max_sizes[1] == 1) && (srcnode->output_defs[0].max_sizes[2] == 1)) {
		logmsg(nn,2,"Don't try and convert 1x1xD tensors to D32 format");
		return 0;
	}
	memcpy(new_inputs,srcnode->input_refs, node_ins*sizeof(new_inputs[0]));
	memcpy(new_outputs, srcnode->output_defs, node_outs*sizeof(new_outputs[0]));

	// d32 padding on output tensors
	//for( int i = 0; i < n_outputs; i++)
	//	output_add_d32_padding( &new_outputs[i]);

	// create cvt->32 nodes for all of the inputs; modify the 'new_inputs' array
	// to reflect those sources. Entries in new_nodes[] will be set to point to the
	// new nodes, and to NULL when the nodes were not actually needed (i.e. we were wired
	// to the input of an existing Convert_from_d32)
	for (i = 0; i < n_inputs; i++) {
		src_id = srcnode->input_refs[1+i].src_id;
		src_oidx = srcnode->input_refs[1+i].output_idx;

		int res = need_convert_to_d32( nn , srcnode,
				new_inputs[1+i],	// old input ref
				&new_inputs[1+i],		// input ref will be placed here
				& new_nodes[i] );		// new node ptr placed here (or NULL if existing node used)
		if( res != 0){
			return errlog(nn,"Can't make new d32 node");
		}
	}
	// if a split node input #0 is wired to same place as #1, make sure we rewire the new one in the same way.
	if( new_operation == OP_QuantizedSplit_8_d32 ){
		struct input const *old_inputs = srcnode->input_refs;
		if( old_inputs[0].src_id == old_inputs[1].src_id
			&& old_inputs[0].output_idx == old_inputs[1].output_idx ){
			new_inputs[0] = new_inputs[1];
		}
	}

	// Create the new operation that operates in d32-format
	if ((new_d32node = optab[new_operation]->ctor(
		nn,
		srcnode->node_id,
		new_operation,
		srcnode->padding,
		node_ins,
		node_outs,
		new_inputs,
		new_outputs)) == NULL) {
		return errlog(nn,"Can't make new d32 node");
	}
	// stash that in the list of new nodes
	new_nodes[n_inputs] = new_d32node;

	// Create the conversion(s) back from d32 format
	for(i = 0; i < n_outputs; i++ ){
		if ((convert_from_node = create_convert_from_d32( nn,
			new_d32node->node_id,	// each connected to new node
			i,		// each connected to a separate output
			srcnode->outputs[i]->shape)) == NULL) {
			return errlog(nn,"Can't make convert to d32");
		}
		new_from_d32_nodes[i] = convert_from_node;
	}

	// move consumers of srcnode to read the convert_from_node
	change_multi_output_refs( nn, srcnode,
			srcnode->node_id,		// replace all refs to this node
			n_outputs,				// .. with output index  in range 0..n_outputs-1
			new_from_d32_nodes);	// to these nodes, output index 0.

	//
	// replace one node with the whole list
	//
	if( replace_node_with_sequence( nn, nodeptr,srcnode, new_nodes, n_inputs+1+n_outputs ) < 0){
		return errlog(nn,"failed to replace nodes for %s",hexagon_nn_op_names[new_operation]);
	}
	return 0;
}

static int do_convert_to_short_conv(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *srcnode = *nodeptr;
	int32_t new_operation = OP_InputSupernode_8x8p8to8_outd32;
	struct nn_node *convert_from_node;
	struct nn_node *new_d32node;
	struct nn_node *stride;
	int i;
	struct output new_outputs[srcnode->n_outputs];
	struct nn_node *weights;
	int stride_width;

	// DISABLE FOR NOW
	if (0) return 0;
	if (srcnode->node_type == OP_Supernode_8x8p32to8) {
		new_operation = OP_InputSupernode_8x8p32to8_outd32;
	}

	logmsg(nn,2,"checking for %p to short conv conversion",srcnode);
	if ((srcnode->node_type != OP_Supernode_8x8p8to8) && (srcnode->node_type != OP_Supernode_8x8p32to8)) {
		return 0;
	}
	/* Ensure output depth multiple of 32 for now... */
	if ((weights = find_node_must_be_Const(nn,srcnode->input_refs[1].src_id)) == NULL) {
		return errlog(nn,"weights not found, or not const");
	}
	//if ((weights->outputs[0]->shape.filt_batches % 32) != 0) return 0;
	//if (weights->outputs[0]->shape.filt_batches < 32) return 0;
	/* Ensure stride even / > 1 */
	if ((stride = find_node_must_be_Const(nn,srcnode->input_refs[6].src_id)) == NULL) {
		return errlog(nn,"stride not found, or not const");
	}
	stride_width = stride->outputs[0]->shape.width;
	/* Odd strides greater than 1 are unsupported for now */
	if ((stride_width > 1) && ((stride_width & 1) != 0)) return 0;
	//int stride_height;
	//stride_height = stride->outputs[0]->shape.height;
	//if ((stride->outputs[0]->shape.height > 1) return 0;

	make_outputdesc_from_shape( &new_outputs[0], &srcnode->outputs[0]->shape, /*elsize=*/sizeof(char), /*pad_d32=*/ 0);

	logmsg(nn, 10, "Will create d32_out padded from %p to %dx%dx%dx%d",
	       srcnode->outputs[0],
	       new_outputs[0].max_sizes[0],new_outputs[0].max_sizes[1],
	       new_outputs[0].max_sizes[2],new_outputs[0].max_sizes[3]);


	for (i = 1; i < srcnode->n_outputs; i++) {
		new_outputs[i] = srcnode->output_defs[i];
	}

	// Hack to ensure 16KB minimum allocation for this output
	// TODO - Is it a bug that we resize new_outputs[0] here, but not d32_shape, which is
	//    later used for create_convert_to_d32?
	uint32_t size_per_depth =
		new_outputs[0].elementsize *
		new_outputs[0].max_sizes[0] *
		new_outputs[0].max_sizes[1] *
		new_outputs[0].max_sizes[2];
	if ( size_per_depth * new_outputs[0].max_sizes[3] < 16 * 1024 ) {
		// round up to nearest depth for >=16KB
		new_outputs[0].max_sizes[3] = ( ((16 * 1024) + (size_per_depth - 1)) / size_per_depth);
	}

	// Create the new operation that operates in d32-format
	if ((new_d32node = optab[new_operation]->ctor(
		nn,
		srcnode->node_id,
		new_operation,
		srcnode->padding,
		srcnode->n_inputs,
		srcnode->n_outputs,
		srcnode->input_refs,
		new_outputs)) == NULL) {
		return errlog(nn,"can't make new d32node");
	}
	// Create the conversion back from d32 format
	//logmsg(nn,0,"srcnode: %d max_size: %d convert_to max_size: %d",srcnode->node_id,srcnode->outputs[0]->max_size,convert_to_node->outputs[0]->max_size);
	if ((convert_from_node = create_convert_from_d32(
		nn,
		new_d32node->node_id,
		0,
		srcnode->outputs[0]->shape)) == NULL) {
		return errlog(nn,"can't make convert from d32");
	}
	change_output_refs( nn, srcnode, new_d32node->node_id, convert_from_node->node_id, 0xFF1);

	// replace {srcnode} with { new_d32node, convert_from_node }

	if( replace_node_with(nn, nodeptr, srcnode, new_d32node, convert_from_node)<0){
		return errlog(nn,"replace failed in convert_to_short_conv");
	}

	return 0;
}

/*
  Take an input node (nodeptr) and transform it into a new subgraph
  which performs a similar operation but using d32-format, with
  conversion operators sandwiching the main operation.
  e.g. "Add" becomes "into_d32 -> d32_Add -> from_d32"
*/
static int do_convert_to_depth32(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *srcnode = *nodeptr;
	int32_t new_operation;
	struct nn_node *convert_to_nodes[2] = {NULL,NULL};		// conversion for inputs 0, 1
	struct nn_node *convert_from_node;
	struct nn_node *new_d32node;

	int n_inputs = srcnode->n_inputs;
	int n_outputs = srcnode->n_outputs;


	unsigned convinput_set = 1;		// bit 0 = convert input 0; bit 1 = convert input 1.

	// Does this operation have a d32 version?
	if ((new_operation = depth32_replacement_op(nn, srcnode)) < 0) return 0;

	// some special cases...
    	if ( new_operation == OP_Convert_from_aix_d32)
    		convinput_set = 0;
	else if( new_operation == OP_InputSupernode_8x8p8to8_outd32)
		return do_convert_to_short_conv(nn,nodeptr);		// (also includes bias32 variant)
	else if( new_operation == OP_QuantizedConcat_8_d32  || new_operation == OP_QuantizedSplit_8_d32)
		return do_convert_concat_depth32(nn,nodeptr, new_operation);

	// ChannelShuffle:
	// if it has 1 input, 1 output, then the code below will handle it.
	// Otherwise use do_convert_concat_depth32
	//
	if( new_operation == OP_QuantizedChannelShuffle_8_d32 ){
		if(srcnode->n_inputs > 4 || srcnode->n_outputs > 3 ){
			return do_convert_concat_depth32(nn,nodeptr, new_operation);
		}
		convinput_set = 2;		// convert input #1, not #0
	}
	if(new_operation == OP_QuantizedAdd_8p8to8_d32 || new_operation ==  OP_QuantizedSub_8p8to8_d32){
		convinput_set = 1 + 2;		// convert both inputs.
	}

	//////////////////////////// general boring normal convert-to-d32 /////////////////////////

	// (protect the stack)
	if( n_inputs < 1 || n_inputs > 64 || n_outputs < 1 || n_outputs > 64) return 0;

	struct input new_inputs[n_inputs];
	struct output new_outputs[n_outputs];


	logmsg(nn, 2, "Converting %s %p to %s", hexagon_nn_op_names[srcnode->node_type], srcnode, hexagon_nn_op_names[new_operation]);

	make_outputdesc_from_shape( &new_outputs[0], &srcnode->outputs[0]->shape, /*elsize=*/sizeof(char), /*pad_d32=*/ 0);

	logmsg(nn, 10, "Will create d32_out padded from %p to %dx%dx%dx%d",
	       srcnode->outputs[0],
	       new_outputs[0].max_sizes[0],new_outputs[0].max_sizes[1],
	       new_outputs[0].max_sizes[2],new_outputs[0].max_sizes[3]);

	// copy the other output descs, and inputs
	if( n_outputs > 1)
		memcpy( &new_outputs[1], &srcnode->output_defs[1], sizeof( struct output)* (n_outputs-1));
	memcpy( new_inputs, srcnode->input_refs, sizeof(struct input)* n_inputs);


	// create the convert-to-d32 as needed
	for( int i = 0; i < 2; i++){
		if( convinput_set & (1<<i) ){
			int k = need_convert_to_d32( nn,srcnode,
					new_inputs[i],			// old input ref
					&new_inputs[i],			// input refs go here
					& convert_to_nodes[i] );   // new node goes here (or null)
			if( k != 0) return errlog(nn,"can't make conv to d32");
		}
	}

	// Create the new operation that operates in d32-format
	if ((new_d32node = optab[new_operation]->ctor(
		nn,
		srcnode->node_id,
		new_operation,
		srcnode->padding,
		n_inputs,
		n_outputs,
		new_inputs,
		new_outputs)) == NULL) {
		return errlog(nn,"can't make new d32node");
	}

	// Create the conversion back from d32 format
	//logmsg(nn,0,"srcnode: %d max_size: %d convert_to max_size: %d",srcnode->node_id,srcnode->outputs[0]->max_size,convert_to_node->outputs[0]->max_size);
	if ((convert_from_node = create_convert_from_d32(
		nn,
		new_d32node->node_id,
		0,
		srcnode->outputs[0]->shape)) == NULL) {
		return errlog(nn,"can't make convert from d32");
	}

	// Stitch in the conversion-to-d32, d32-operation, conversion-from-d32
	/* Before we stitch in the new elements, rewrite the graph to point consumers the right way */
	/* We do this before we stitch in so that we don't rewrite the convert_from to point to itself */
	change_output_refs( nn, srcnode, new_d32node->node_id, convert_from_node->node_id, 0xFFFF1);

	// replace 'srcnode' with { convert_to_node, [convert_to_node2,] new_d32node, convert_from_node }

	if( replace_node_with( nn, nodeptr,srcnode,
			convert_to_nodes[0], convert_to_nodes[1], new_d32node, convert_from_node) < 0){
		return errlog(nn,"failed to replace_node for conv_depth_32");
	}
	return 0;
}

static int convert_to_depth32(struct nn_graph *nn)
{
	return graph_iterator(nn,do_convert_to_depth32);
}

//
// look for:
//   add_node:  [ OP_QuantizedAdd_8p8to32]		.. with constant 'b' input shaped as [1,1,1,d]
//  .. replace with
//        [ OP_QuantizedBiasAdd_8p8to32 ]

static int do_mark_biasadd_node(struct nn_graph *nn, struct nn_node **add_node_p)
{
	struct nn_node *add_node = *add_node_p;
	struct nn_node *bias_val_node;
	struct nn_node *new_node;
	struct tensor *bias_val;
	struct output new_outputs[3];
	int i;
	int operation = OP_QuantizedBiasAdd_8p8to32;
	if (add_node->node_type != OP_QuantizedAdd_8p8to32) return 0;
	if ((bias_val_node = find_node_must_be_Const(nn,add_node->input_refs[1].src_id)) == NULL) return 0;
	bias_val = bias_val_node->outputs[0];
	if( !shape_111X(&bias_val->shape)) return 0;
	/* We could probably handle add with scalar, but for now just ditch it */
	if (bias_val->shape.depth <= 1) return 0;
	for (i = 0; i < 3; i++) {
		new_outputs[i] = add_node->output_defs[i];
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		add_node->node_id,
		operation,
		add_node->padding,
		add_node->n_inputs,
		add_node->n_outputs,
		add_node->input_refs,
		new_outputs)) == NULL) return errlog(nn,"ctor fail");

	if( replace_nodes( nn,add_node_p, new_node, add_node )!=0){
		return errlog(nn,"failed to replace add\n");
	}
	logmsg(nn,2,"Converted QAdd to QBiasAdd id=%x",new_node->node_id);
	return 0;
}

static int mark_biasadd_nodes(struct nn_graph *nn)
{
	return graph_iterator(nn,do_mark_biasadd_node);
}

static int try_make_qadd_supernode(struct nn_graph *nn, struct nn_node **qadd_node_p)
{
	struct nn_node *qadd_node = *qadd_node_p;
	struct nn_node *qdown_node;
	struct nn_node *relu_node;
	struct nn_node *supernode;
	struct nn_node *lastop;
	const int num_inputs = 8;
	struct input new_inputs[num_inputs];
	struct input minmax_inputs[2];
	struct output new_outputs[3];
	int i;

	static const int requant_ops[2] = { OP_QuantizeDownAndShrinkRange_32to8, OP_Requantize_32to8 };
	static const int relu_ops[3] = { OP_QuantizedRelu_8, OP_QuantizedReluX_8, OP_QuantizedClamp_8 };


	op_type operation = OP_QuantizedAdd_8p8to8;
	/* Make sure start node is the right kind... */
	if (qadd_node->node_type != OP_QuantizedAdd_8p8to32) return 0;
	logmsg(nn,4,"found add id=%x",qadd_node->node_id);
	/* Find the consumer node */
	qdown_node = find_unique_consumer(nn,qadd_node,2,requant_ops,0);
	if( qdown_node == NULL) return 0;

	logmsg(nn,4,"found qdown");
	/* Now repeat for Relu */
	requantize_op_minmax_inputs( nn, qdown_node, minmax_inputs);

	relu_node = find_unique_consumer( nn,qdown_node, 3, relu_ops, 0 );
	if( extract_min_max_from_relu( nn, relu_node,  minmax_inputs ) == 0 ){
		logmsg(nn,4,"found relu/clamp\n");
		lastop = relu_node;
	}else{
		logmsg(nn,4,"RELU/Clamp missing");
		relu_node = NULL;
		lastop = qdown_node;
	}

	/*** WOO we are a good candidate to make a supernode */
	/* 
	 * Embiggen the inputs
	 * Copy the inputs from the nodes:
	 * * all the input args from conv
	 * * Followed by values/min/max for biasadd
	 */
	for (i = 0; i < 6; i++) {
		new_inputs[i] = qadd_node->input_refs[i];
	}
	new_inputs[6] = minmax_inputs[0];
	new_inputs[7] = minmax_inputs[1];
	/* FIXME: struct vals / ordering. */
	/* FIXME: Time to merge fastrpc branch back into master */
	for (i = 0; i < 3; i++) {
		new_outputs[i] = lastop->output_defs[i];
	}
	/* Reuse the outputs & ID from the relu node */
	/* Allocate new node */
	if ((supernode = optab[operation]->ctor(
		nn,
		lastop->node_id,
		operation,
		qadd_node->padding,
		num_inputs,
		3,
		new_inputs,
		new_outputs)) == NULL) return 0;
	/* Clean up the old not needed nodes */
	/* note: relu_node may be NULL */

	if( replace_nodes(nn,qadd_node_p, supernode,
			qadd_node, qdown_node, relu_node)!= 0){
		return errlog(nn,"error replacing nodes for qadd_supernode");
	}

	logmsg(nn,3,"Created qadd supernode id=%x (was relu ID)",supernode->node_id);
	return 0;
}
//
// 'nodep' is an AutoQuantize node.
// if the input is a const, and is not too large, make min/max/u8 consts and replace
// all the refs to them.
// This function makes no edits to the list at the targets location (it only adds const
// nodes at the front of list, and edits source refs).
//
static int try_subst_autoquantize_const( struct nn_graph *nn, struct nn_node * nodep)
{

	if( nodep->node_type != OP_AutoQuantize || nodep->n_inputs != 1 || nodep->input_refs[0].output_idx!= 0)
		return 0;
	struct nn_node *const_node = find_node_must_be_Const( nn, nodep->input_refs[0].src_id);

	const int max_els = 8192;	// if bigger than this, we won't do it.

	if( const_node == NULL || const_node->outputs[0]->data_size > sizeof(float)*max_els ){
		return 0;	// not a const, or too large.
	}
	struct tensor const * const_tensor = const_node->outputs[0];
	int n_els = tensor_element_count(const_tensor);
	if( n_els > max_els) return 0;	// ?? shape didn't match len ??

	const float *fdata = (const float *)const_tensor->data;

	float minval,maxval;

	// find range, checking for inf, nan
	int res = find_range_of_floats( fdata, n_els, &minval, &maxval);
	if( res != 0 )
		return errlog(nn,"Autoquantize const inputs contain nan or inf");
	if(minval == maxval){
		maxval = 0.125f;	// all 0.0 -> make it [0,0.125] range
	}else if(minval != 0.0f){
		// make sure we have a 'clean' zero
		adjust_minmax_for_zero( &minval, &maxval);
	}
	// make a buffer on the stack, or alloc
	void *tbuf = NULL;
	int toobig = n_els >= 2048;			// this threshold doesn't need to match max_els
	uint8_t xbuf[ toobig? 1:  (n_els+1)];
	uint8_t * buf_u8 = xbuf;
	if( toobig){
		tbuf = nn_malloc( n_els * sizeof(uint8_t));
		if( tbuf == NULL) return errlog(nn,"alloc failed");
		buf_u8 = (uint8_t*)tbuf;
	}
	logmsg(nn,9,"bypassing AutoQuantize node 0x%X with new const: %d vals over range %f ... %f",
			(unsigned) nodep->node_id, n_els, minval, maxval);

	// now convert all to tbuf.
	float scale = 255.0f/(maxval-minval);
	for( int i =0; i < n_els; i++ ){
		buf_u8[i] = saturate_u8( roundf_i32( (fdata[i]-minval)*scale));
	}
	// make the nodes...
	unsigned new_const_nid = nn_graph_new_internal_node_id(nn);
	res = do_prepend_const_node(nn, new_const_nid,
			const_tensor->shape.batches, const_tensor->shape.height, const_tensor->shape.width, const_tensor->shape.depth,
			buf_u8, n_els);
	if( tbuf != NULL) nn_free(tbuf);

	unsigned min_nid, max_nid;
	if( res!=0
		|| ( min_nid = create_const_float_op(nn, minval)) == 0
		|| ( max_nid = create_const_float_op(nn, maxval)) == 0){
		return errlog(nn,"failed to make consts for AutoQuantize(const)");
	}
	// OK, the nodes are in the graph - we just need to replace refs to AutoQuant outputs (0,1,2) to the new nodes
	//
	unsigned autoquant_nid = nodep->node_id;
	struct input new_inprefs[3] = {		// replacement references
			{ new_const_nid, 0 },
			{ min_nid, 0 },
			{ max_nid, 0 }
	};
	res = change_multi_output_refs_table(nn, nodep,
			autoquant_nid,					// old node id
			3,					// number of outputs to rewire
			new_inprefs);
	return res <0? res: 0;
}

static int try_make_autoquantize(struct nn_graph *nn, struct nn_node **quantize_node_p)
{
	struct nn_node *quantize_node = *quantize_node_p;
	struct nn_node *min_node;
	struct nn_node *max_node;
	struct nn_node *new_node;
	op_type operation = OP_AutoQuantize;
	struct output new_outputs[3];
	int i;
	if (quantize_node->node_type != OP_Quantize){
		if( quantize_node->node_type == OP_AutoQuantize ){
			return try_subst_autoquantize_const( nn, quantize_node);
		}
		return 0;
	}
	logmsg(nn,4,"Found quantize node id=%x",quantize_node->node_id);
	/* Try to find min and max nodes, otherwise abort */
	/* Check that Min and Max nodes are Min and Max ops */
	if ((min_node = find_node_must_be(nn,quantize_node->input_refs[1].src_id, OP_Min_f)) == NULL) return 0;
	if ((max_node = find_node_must_be(nn,quantize_node->input_refs[2].src_id, OP_Max_f)) == NULL) return 0;

	for (i = 0; i < 3; i++) {
		new_outputs[i] = quantize_node->output_defs[i];
	}
	if ((new_node = optab[operation]->ctor(
		nn,
		quantize_node->node_id,
		operation,
		quantize_node->padding,
		1,	// inputs
		3,	// outputs
		&quantize_node->input_refs[0],
		new_outputs)) == NULL) return errlog(nn,"ctor fail");

	if( replace_nodes(nn,quantize_node_p,  new_node,  quantize_node )!=0)
		return errlog(nn, "error replacing nodes for autoquantize");

	logmsg(nn,2,"Formed AutoQUantize id=%x",new_node->node_id);
	return try_subst_autoquantize_const( nn, new_node);
}


static int make_autoquantize(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_autoquantize);
}

static int try_make_quantized_dwise(struct nn_graph *nn, struct nn_node **dwise_node_p)
{
	struct nn_node *dwise_node = *dwise_node_p;
	struct nn_node *input_node;
	struct nn_node *filter_node;
	struct nn_node *new_quant_in_node;
	struct nn_node *new_quant_filter_node;
	struct nn_node *new_dwise_node;
	struct nn_node *new_requant_node;
	struct nn_node *new_dequant_node;
	op_type operation = OP_QuantizedDepthwiseConv2d_8x8to32;
	struct output new_outputs[3];
	struct input new_inputs[7];
	if (dwise_node->node_type != OP_DepthwiseConv2d_f) return 0;
	if (dwise_node->n_inputs < 3) return 0;
	uint32_t input_id = dwise_node->input_refs[0].src_id;
	uint32_t input_idx = dwise_node->input_refs[0].output_idx;
	uint32_t filter_id = dwise_node->input_refs[1].src_id;
	uint32_t filter_idx = dwise_node->input_refs[1].output_idx;
	uint32_t stride_id = dwise_node->input_refs[2].src_id;
	uint32_t stride_idx = dwise_node->input_refs[2].output_idx;
	logmsg(nn,4,"Found dconv2d_f node id=%x",dwise_node->node_id);

	/* Try to find min and max nodes, otherwise abort */
	if ((input_node = find_node(nn,input_id)) == NULL) return 0;
	if ((filter_node = find_node(nn,filter_id)) == NULL) return 0;

	logmsg(nn,4,"Found input node %x filter node %x and stride node %x",
		input_node->node_id,
		filter_node->node_id,
		stride_id);

	/* Quantized ops: output 0 is same shape but element size 4 */
	new_outputs[0] = dwise_node->output_defs[0];
	new_outputs[0].elementsize = 4;

	/* Quantized ops: output 1 and 2 are min/max float vals */
	new_outputs[1] = Output_ScalarFloat;
	new_outputs[2] = Output_ScalarFloat;

	if ((new_quant_in_node = create_autoquantize(nn,input_node,input_idx)) == NULL) {
		return errlog(nn,"can't create autoquant");
	}
	if ((new_quant_filter_node = create_autoquantize(nn,filter_node,filter_idx)) == NULL) {
		return errlog(nn,"can't create autoquant");
	}
	logmsg(nn,4,"made quant in %x and quant filter %x",
		new_quant_in_node->node_id,
		new_quant_filter_node->node_id); 

	new_inputs[0].src_id = new_quant_in_node->node_id;
	new_inputs[1].src_id = new_quant_filter_node->node_id;
	new_inputs[2].src_id = new_quant_in_node->node_id;
	new_inputs[3].src_id = new_quant_in_node->node_id;
	new_inputs[4].src_id = new_quant_filter_node->node_id;
	new_inputs[5].src_id = new_quant_filter_node->node_id;
	new_inputs[6].src_id = stride_id;

	new_inputs[0].output_idx = 0;
	new_inputs[1].output_idx = 0;
	new_inputs[2].output_idx = 1;
	new_inputs[3].output_idx = 2;
	new_inputs[4].output_idx = 1;
	new_inputs[5].output_idx = 2;
	new_inputs[6].output_idx = stride_idx;
	if ((new_dwise_node = optab[operation]->ctor(
		nn,
		dwise_node->node_id,
		operation,
		dwise_node->padding,
		7,	// inputs
		3,	// outputs
		new_inputs,
		new_outputs)) == NULL) return errlog(nn,"ctor fail");

	if ((new_requant_node = create_requantize(nn,new_dwise_node)) == NULL) {
		return errlog(nn,"can't create requant");
	}
	if ((new_dequant_node = create_dequantize(nn,new_requant_node)) == NULL) {
		return errlog(nn,"can't create dequant");
	}
	change_output_refs( nn, dwise_node, dwise_node->node_id, new_dequant_node->node_id, 0xFFFF1);

	// replace dwise_node with
	//  { new_quant_filter_node, new_quant_in_node, new_dwise_node, new_requant_node,
	//     new_dequant_node}
	//
	if( replace_node_with( nn, dwise_node_p, dwise_node,
			new_quant_filter_node, new_quant_in_node, new_dwise_node, new_requant_node, new_dequant_node) <0 ){
		return errlog(nn,"failed to replace in make_quantized_dwise");
	}
	logmsg(nn,2,"Formed QuantizedDepthwiseConv2d id=%x",new_dwise_node->node_id);
	return 0;
}


static int make_quantized_dwise(struct nn_graph *nn)
{
	return graph_iterator(nn,try_make_quantized_dwise);
}


#if 0
/*
 * If a const could change (nonzero and finite) and has more than one consumer, it's bad to change it.
 */
static inline int is_bad_const_node_for_fold(struct nn_graph *nn, struct nn_node *src, struct nn_node *dest)
{
	float oldval = tensor_get_float(src->outputs[0],0);
	if (oldval == 0.0f) return 0;
	if (oldval == INFINITY) return 0;
	if (oldval == -INFINITY) return 0;
	if (check_all_outputs(nn,src,dest) == 0) return 0;
	return 1;
}
#endif

/*
 * Try to take a multiply by a scalar value and fold it back into the previous supernode
 */
// matches the end of a chain like this:
//       supernode: [ <one of several types> ]
//	     mul_node:   [ OP_QuantizedMul_8x8to32 ]			- with scalar constant 'B' input
//  --> requant_node: [ OP_QuantizeDownAndShrinkRange_32to8 ]
//
// if it matches, the weight, bias, output range of the supernode are scaled, and the
// references  to requant node are pointed directly to its output, thus leaving the mul & requant unused.
// There are cases where the scale factor is 1.0, so we don't have to scale the supernode in those
// cases, just remove the mul.
//
//
static int try_fold_scalar_mpy(struct nn_graph *nn, struct nn_node **requant_node_p)
{
	struct nn_node *requant_node = *requant_node_p;
	struct nn_node *mul_node;
	struct nn_node *supernode;
	struct nn_node *mul_const_val;
	struct nn_node *mul_const_min;
	struct nn_node *mul_const_max;
	float scaleval;

	if (requant_node->node_type != OP_QuantizeDownAndShrinkRange_32to8) return 0;
	if ((mul_node = find_node(nn,requant_node->input_refs[0].src_id)) == NULL) return 0;
	if (mul_node->node_type != OP_QuantizedMul_8x8to32) return 0;
	if (((mul_const_val = find_node_must_be_Const(nn,mul_node->input_refs[1].src_id)) == NULL)
	    || ((mul_const_min = find_node_must_be_Const(nn,mul_node->input_refs[4].src_id)) == NULL)
	    || ((mul_const_max = find_node_must_be_Const(nn,mul_node->input_refs[5].src_id)) == NULL)) {
		logmsg(nn,4,"mpy B val not found / not const");
		return 0;
	}
	if( ! shape_1111(&mul_const_val->outputs[0]->shape) ){
		logmsg(nn,4,"not mpy by scalar");
		return 0;
	}

	scaleval = read_float_from_qu8(mul_const_val->outputs[0], mul_const_min->outputs[0], mul_const_max->outputs[0]);

	logmsg(nn,2,"candidate mpy %x by scalar %f",mul_node->node_id,scaleval);
	if ((supernode = find_node(nn,mul_node->input_refs[0].src_id)) == NULL) return 0;
	//
	// is it one of the supernodes we can scale (including batchnorm)?
	int supernode_nt = supernode->node_type;
	int is_batchnorm = ( supernode_nt == OP_QuantizedBatchNorm_8x8p8to8 )
					|| ( supernode_nt == OP_QuantizedBatchNorm_8x8p32to8 );
	if (is_batchnorm
		|| (supernode_nt == OP_Supernode_8x8p8to8)
		|| (supernode_nt == OP_Supernode_8x8p32to8)
		|| (supernode_nt == OP_DepthwiseSupernode_8x8p8to8)
		|| (supernode_nt == OP_DepthwiseSupernode_8x8p32to8)) {
		logmsg(nn,2,"Found supernode-like op id=%x",supernode->node_id);
	} else {
		logmsg(nn,4,"Op %x was not a supernode",supernode->node_id);
		return 0;
	}
	// If the scale is 1.0, we don;t need to change the supernode, just reconnect consumers
	// of the 'mul' to the supernode (and this is ok even if supernode has other consumers)
	if( scaleval != 1.0f){
		if (check_all_outputs(nn,supernode,mul_node) != 0) {
			logmsg(nn,4,"Supernode %x has consumer other than mul node %x",supernode->node_id,mul_node->node_id);
			return 0;
		}
		// process the six min/max in this array:
		struct nn_node *minmax_nodes[6];
		// the set of inputs is different for batchnorm, which has no 'stride' input
		static const uint8_t range_inputs_convolution[6] = { 4,5,8,9,10,11};
		static const uint8_t range_inputs_batchnorm[6]   = { 4,5,7,8, 9,10};
		uint8_t const * range_inputs = is_batchnorm ? range_inputs_batchnorm : range_inputs_convolution;
		// [0]  weights_min_node      arg 4
		// [1]  weights_max_node      arg 5
		// [2]  bias_min_node         arg 8  (or 7)
		// [3]  bias_max_node         arg 9  (or 8)
		// [4]  out_min_node          arg 10 (or 9)
		// [5]  out_max_node          arg 11; (or 10)

		for( int i=0; i<6; i++){
			int inp_no = range_inputs[i];
			struct nn_node * minmax_node = find_node_must_be_Const(nn,supernode->input_refs[inp_no].src_id);
			if(minmax_node == NULL){
				logmsg(nn,4,"supernode op %x min/max not found or not const",supernode->node_id);
				return 0;
			}
			minmax_nodes[i] = minmax_node;
		}
		// scale them. When necessary, make new const nodes.
		int any_mods = 0;
		for(int i = 0; i < 6;i++){
			struct nn_node * minmax_node = minmax_nodes[i];
			float oldval = tensor_get_float(minmax_node->outputs[0],0);
			float newval = scaleval*oldval;
			// value doesn't change if it's 0.0, -inf, or +inf; don't need to touch those.
			if( newval != oldval){
				if( check_all_outputs(nn,minmax_node,supernode)!=0){	// has other refs - make a new one.
					uint32_t newnode = create_const_float_op(nn, newval );
					if( newnode == 0) return -1;	// create failed (with errlog)
					supernode->input_refs[range_inputs[i]].src_id = newnode;
					any_mods = 1;
				}else{	 // just modify the existing const
					tensor_set_float(minmax_node->outputs[0],0,newval);
					// remove it from cache if it's there.
					purge_const_cache(nn,minmax_node->node_id);
				}
			}
		}
		if( any_mods )node_rehash_inputrefs(supernode);
	}
	// connect all the outputs of QuantizedMul_8x8to32>QuantizeDownAndShrinkRange_32to8
	// to the outputs of the supernode.
	change_output_refs( nn, supernode, requant_node->node_id, supernode->node_id, 0x321);
	return 0;
}

static int fold_scalar_mpys(struct nn_graph *nn)
{
	return graph_iterator(nn,try_fold_scalar_mpy);
}

static struct nn_node * find_concat_for_chanshuf( struct nn_graph *nn, struct nn_node *dshuf_node, int dq_parm );
static struct nn_node * find_split_for_chanshuf( struct nn_graph *nn, struct nn_node *dshuf_node, int dq_parm );
/*
 *  Try to combine a QuantizedChannelShuffle_8 with either or both of
 *   - upstream QuantizedConcat_8
 *   - downstream QuantizedSplit_8
 */
static int try_combine_chanshuffle(struct nn_graph *nn, struct nn_node **chanshuf_node_p)
{
	struct nn_node *dshuf_node = *chanshuf_node_p;

	// must be single input, single output
	if( dshuf_node->node_type != OP_QuantizedChannelShuffle_8
		|| dshuf_node->n_inputs != 4 || dshuf_node->input_refs[0].output_idx != 0
		|| dshuf_node->n_outputs != 3 ){
		return 0;
	}
	// get the 'dq' parameter; output_depth/k
	struct nn_node * k_node = find_node_must_be_Const( nn,dshuf_node->input_refs[0].src_id );
	if( k_node == NULL) return 0;
	int k_val = tensor_get_int32( k_node->outputs[0], 0);
	if( k_val <= 1 ) return 0;
	// dq*k = out_depth
	unsigned all_out_depth = dshuf_node->output_defs[0].max_sizes[3];
	int dq_val = all_out_depth/k_val;
	if( dq_val * k_val != all_out_depth){
		warnlog(nn,"chanshuffle - k does not divide depth");
		return 0;
	}

	// check for an upstream node
	struct nn_node * concat_node = find_concat_for_chanshuf(nn, dshuf_node, dq_val);

	// Now see if there is downstream 'split' node we could use

	struct nn_node *split_node = find_split_for_chanshuf( nn, dshuf_node, dq_val);

	// anything to do here?
	if( concat_node == NULL && split_node == NULL) return 0;

	// OK. we will combine.
	// We will remove the old node, and the 'split' node, but not the 'concat' (it will become
	// dead if there are no other consumers).
	//
	//  - make a new node with the same node_id as the split (if one exists) or the chanshuf (if not).
	//    We then don't need to relabel any downstream consumers.
	//  If there is no concat, the new node has the same inputs as the old; if not, it has the same
	//   inputs as the concat, except for #0, which is retained from the channelshuf.
	//
	struct nn_node * last_node = (split_node != NULL)? split_node : dshuf_node;
	struct nn_node * new_node;
	op_type new_node_optype = OP_QuantizedChannelShuffle_8;
	int num_in = (concat_node == NULL)? 4: concat_node->n_inputs;
	if ( num_in < 4 || num_in > 2048*3) return 0;	// yikes

	struct input new_input_tab[num_in];
	struct input const * new_inputs = dshuf_node->input_refs;
	if( concat_node != NULL){
		new_input_tab[0] = new_inputs[0];	// the 'dq' input stays there...
		memcpy( &new_input_tab[1], &concat_node->input_refs[1], (num_in-1)*sizeof(struct input));
		new_inputs = new_input_tab;
	}
	logmsg(nn,4,"consolidating ChannelShuffle: %d inputs, %d outputs, dq=%d\n",
			num_in/3u, last_node->n_outputs-2, dq_val);
	if ((new_node = optab[new_node_optype]->ctor(
		nn,
		last_node->node_id,
		new_node_optype,
		dshuf_node->padding,
		num_in,	// inputs
		last_node->n_outputs,		// outputs
		new_inputs,
		last_node->output_defs)) == NULL) return errlog(nn,"ctor fail");

	// ok now replace 1 or 2 nodes with 'newnode'

	if( replace_nodes(nn, chanshuf_node_p, new_node,
			dshuf_node, split_node )!= 0){
		return errlog(nn,"replace_node failed in try_combine_chanshuffle");
	}
	return 0;
}

static int combine_chanshuffle(struct nn_graph *nn)
{
	return graph_iterator(nn,try_combine_chanshuffle);
}

// helper for try_combine_chanshuffle
// Finds an eligible upstream QuantizedConcat_8, or returns NULL if there isn't one.
// The concat must be
//    - on dim 3
//    - all of the same size
//    - depth of all inputs divisible by dq.
//  Once we identify the concat with N inputs, we check that its output depth is divisible
//  by N*dq, and then we can figure the size all its inputs should be.
//
static struct nn_node *
find_concat_for_chanshuf( struct nn_graph *nn, struct nn_node *dshuf_node, int dq_parm )
{
	// find producer id;
	unsigned nid = dshuf_node->input_refs[1].src_id;
	struct nn_node * concat_node = find_node_must_be( nn, nid , OP_QuantizedConcat_8);
	if( concat_node == NULL) return NULL;
	//
	// make sure that the ChannelShuffle inputs are wired to it as expected
	//
	for( int i  =0; i < 3; i++ ){
		if( dshuf_node->input_refs[i+1].src_id != nid || dshuf_node->input_refs[i+1].output_idx != i)
			return NULL;
	}
	// make sure that the Concat is on #3 and it has at least 7 inputs (3n+1, n >=2 )
	int conc_num_in = concat_node->n_inputs;
	if( conc_num_in < 7 || concat_node->n_outputs < 3) return NULL;
	struct nn_node * dim_node = find_node_must_be_Const( nn,concat_node->input_refs[0].src_id );

	if( dim_node == NULL  || tensor_get_int32( dim_node->outputs[0], 0) !=3 )
		return NULL;
	int concat_n = (conc_num_in-1)/3u;
	if( conc_num_in != 3*concat_n+1) return NULL;

	struct output const *concat_out = &concat_node->output_defs[0];
	if( concat_out->rank != 4) return NULL;
	unsigned all_depth = concat_out->max_sizes[3];
	unsigned tmp = all_depth/(dq_parm*concat_n);
	unsigned in_depth = tmp* dq_parm;		// inputs must all be this depth
	if ( in_depth * concat_n != all_depth) return NULL;

	struct output const * outdesc;
	// survey all the inputs, check if the same size
	for( int i = 0; i < concat_n; i++){
		unsigned srcnid = concat_node->input_refs[i+1].src_id;
		int out_idx = concat_node->input_refs[i+1].output_idx;
		struct nn_node * src_node = find_node(nn, srcnid);
		if( src_node == NULL || src_node->n_outputs <= out_idx ) return NULL;
		outdesc = &src_node->output_defs[out_idx];
		if(  outdesc->max_sizes[0] != concat_out->max_sizes[0]
			  ||  outdesc->max_sizes[1] != concat_out->max_sizes[1]
			  ||  outdesc->max_sizes[2] != concat_out->max_sizes[2]
			  ||  outdesc->max_sizes[3] != in_depth ){
				return NULL;
		}
	}
	return concat_node;
}
//
// another helper for try_combine_chanshuffle
// find an eligible downstream 'split', or return NULL.
// The split must be on depth, and the number of outputs must divide dq_parm.
//
static struct nn_node *
find_split_for_chanshuf( struct nn_graph *nn, struct nn_node *dshuf_node, int dq_parm )
{
	struct nn_node * split_node = find_unique_consumer_mustbe( nn, dshuf_node, OP_QuantizedSplit_8, 0);
	if( split_node == NULL) return NULL;
	// make sure that the Concat is on #3 and it has at least 2 outputs
	int split_num = split_node->n_outputs-2;
	if( split_num < 2 || split_node->n_inputs < 4 || (unsigned)dq_parm % split_num != 0) return NULL;
	struct nn_node * dim_node = find_node_must_be_Const( nn,split_node->input_refs[0].src_id );
	if( dim_node == NULL  || tensor_get_int32( dim_node->outputs[0], 0) !=3 )
		return NULL;
	return split_node;
}



/*
 * Find Requantize_32to8 with Const min/max --> Dequantize --> Quantize with Const Min/Max that are the same values
 */

static int try_get_const_float_val(struct nn_graph *nn, uint32_t src_id, float *val_out)
{
	struct nn_node *node;
	static const struct shape one_shape  = { .batches=1, .height=1, .width=1, .depth = 1};

	if ((node = find_node_must_be_Const(nn,src_id)) == NULL) return -1;
	struct tensor const * tens = node->outputs[0];
	if( ! shape_matches( &tens->shape, & one_shape )) return -1;
	if (tens->max_size != 4) return -1;
	memcpy(val_out,tens->data,4);
	return 0;
}

static int do_remove_unnecessary_dequant_quants(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *quantize_node = *nodeptr;
	struct nn_node *dequantize_node;
	struct nn_node *requantize_node;
	float requantize_min_val;
	float requantize_max_val;
	float quantize_min_val;
	float quantize_max_val;
	uint32_t src_id;
	if (quantize_node->node_type != OP_Quantize) return 0;
	logmsg(nn,4,"Found quantize ID %x",quantize_node->node_id);
	src_id = quantize_node->input_refs[0].src_id;
	if ((dequantize_node=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (dequantize_node->node_type != OP_Dequantize) return 0;
	logmsg(nn,4,"Found dequantize ID %x",dequantize_node->node_id);
	src_id = dequantize_node->input_refs[0].src_id;
	if ((requantize_node=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (requantize_node->node_type != OP_Requantize_32to8) return 0;
	if (try_get_const_float_val(nn,requantize_node->input_refs[3].src_id,&requantize_min_val) != 0) return 0;
	if (try_get_const_float_val(nn,requantize_node->input_refs[4].src_id,&requantize_max_val) != 0) return 0;
	if (try_get_const_float_val(nn,quantize_node->input_refs[1].src_id,&quantize_min_val) != 0) return 0;
	if (try_get_const_float_val(nn,quantize_node->input_refs[2].src_id,&quantize_max_val) != 0) return 0;
	logmsg(nn,4,"requantize_min=%f requantize_max=%f quantize_min=%f quantize_max=%f",
		requantize_min_val,requantize_max_val,quantize_min_val,quantize_max_val);
	if (requantize_min_val != quantize_min_val) return 0;
	if (requantize_max_val != quantize_max_val) return 0;
	logmsg(nn,4,"changing %d refs to %d...",quantize_node->node_id,requantize_node->node_id);

	change_output_refs( nn, quantize_node, quantize_node->node_id, requantize_node->node_id, 0x321);
	return 0;
}

static int remove_unnecessary_dequant_quants(struct nn_graph *nn)
{
	return graph_iterator(nn,do_remove_unnecessary_dequant_quants);
}


/*
 * Find Dequantize followed by AutoQuantize
 */

static int do_remove_unnecessary_quants(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	struct nn_node *srcnode;
	uint32_t src_id;
	uint32_t srcsrc_id;
	uint32_t srcsrc_idx;
	int i;
	if (node->node_type != OP_AutoQuantize) return 0;
	src_id = node->input_refs[0].src_id;
	if ((srcnode=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (srcnode->node_type != OP_Dequantize) {
		//logmsg(nn,0,"src node %x not from_32 for to_d32 node %x",srcnode->node_id,node->node_id);
		return 0;
	}
	for (i = 0; i < 3; i++) {
		srcsrc_id = srcnode->input_refs[i].src_id;
		srcsrc_idx = srcnode->input_refs[i].output_idx;
		logmsg(nn,4,"quants: trying to convert %x:%d to %x:%d everywhere...",
			node->node_id,i,srcsrc_id,srcsrc_idx);
		if (change_refs(nn,node->node_id,i,srcsrc_id,srcsrc_idx) != 0) {
			return errlog(nn,"change refs failed");
		}
	}
	return 0;
}

static int remove_unnecessary_quants(struct nn_graph *nn)
{
	return graph_iterator(nn,do_remove_unnecessary_quants);
}

/*
 * Find convert_to_d32 nodes
 * Find producer
 * If producer is not convert from D32, continue
 * Point all consumers to producer's input
 * The op should become dead
 * Producer might also become dead
 */

static int do_remove_unnecessary_d32_converts(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	struct nn_node *srcnode;
	uint32_t src_id;
	uint32_t srcsrc_id;
	uint32_t srcsrc_idx;
	if (node->node_type != OP_Convert_to_d32) return 0;
	src_id = node->input_refs[0].src_id;
	if ((srcnode=find_node(nn,src_id)) == NULL) return errlog(nn,"src %d not found",src_id);
	if (srcnode->node_type != OP_Convert_from_d32 && srcnode->node_type != OP_Convert_from_aix_d32) {
		//logmsg(nn,0,"src node %x not from_32 for to_d32 node %x",srcnode->node_id,node->node_id);
		return 0;
	}
    if (srcnode->node_type == OP_Convert_from_d32) {
        srcsrc_id = srcnode->input_refs[0].src_id;
        srcsrc_idx = srcnode->input_refs[0].output_idx;
    }
    else if (srcnode->node_type == OP_Convert_from_aix_d32)
    {
        srcsrc_id = srcnode->node_id;
        srcsrc_idx = node->input_refs[0].output_idx;
    }
    else {
        return errlog(nn, "Shouldn't hit this code branch when trying to remove uneeded d32 conversion nodes");
    }
	//logmsg(nn,0,"trying to convert %x:0 to %x:%d everywhere...",node->node_id,srcsrc_id,srcsrc_idx);
	return change_refs(nn,node->node_id,0,srcsrc_id,srcsrc_idx);
}

static int remove_unnecessary_d32_converts(struct nn_graph *nn)
{
	return graph_iterator(nn,do_remove_unnecessary_d32_converts);
}


/*
 * Find Supernodes
 * That have filter shape 3x3x2x2
 * Convert them to the special case
 */

static int do_make_supernode_3322(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	struct nn_node *weights;
	struct nn_node *newnode;
	struct tensor const *weights_tensor;
	int operation;
	if (node->node_type == OP_Supernode_8x8p8to8) {
		operation = OP_Supernode3322_8x8p8to8;
	} else if (node->node_type == OP_Supernode_8x8p32to8) {
		operation = OP_Supernode3322_8x8p32to8;
	} else {
		return 0;
	}
	if ((weights = find_node_must_be_Const(nn,node->input_refs[1].src_id)) == NULL) {
		logmsg(nn,2,"Hmmm... weights not const or not found");
		return 0;
	}
	weights_tensor = weights->outputs[0];
	if (       (weights_tensor->shape.filt_height != 3)
		|| (weights_tensor->shape.filt_width != 3)
		|| (weights_tensor->shape.filt_depth != 2)
		|| (weights_tensor->shape.filt_batches != 2)) return 0;
	if ((newnode = optab[operation]->ctor(
		nn,
		node->node_id,
		operation,
		node->padding,
		node->n_inputs,
		node->n_outputs,
		node->input_refs,
		node->output_defs)) == NULL) return errlog(nn,"ctor fail");
	if (replace_nodes(nn,nodeptr,newnode,node) != 0) {
		return errlog(nn,"replace_nodes in supernode_3322");
	}
	return 0;
}

static int make_supernode_3322(struct nn_graph *nn)
{
	return graph_iterator(nn,do_make_supernode_3322);
}



// for each OemNode:
//  - get the first input, an integer which defines what it is;
//  - dispatch to a function which does the graph replacement.

static int do_expand_oem_nodes(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	if( node != NULL && node->node_type == OP_OemNode ){
		int nt = -1;
		if( node->n_inputs >=1 ){
			struct nn_node * cnode = find_node_must_be_Const( nn, node->input_refs[0].src_id);
			if( cnode != NULL && node->input_refs[0].output_idx ==0 ){
				struct tensor const * t = cnode->outputs[0];
				if( t->data_size >= 4){
					nt = tensor_get_int32(t,0);
				}
			}
		}
		int result;
		switch(nt){
#if NN_GRAPH_WITH_LENS_LSTM
		 case NN_OEMNODE_METANODEID_LENS_LSTM:
			 result = transform_metanode_GL_Lstm( nn, nodeptr);
			  break;
#endif
		 default:
			result = errlog(nn,"unrecognized OemNode 0x%x\n", (unsigned)node->node_id);
			break;
		}
		if( result !=0) return result;
	}
	return 0;
}

static int expand_oem_nodes(struct nn_graph *nn)
{
	return graph_iterator(nn,do_expand_oem_nodes);
}

static inline int node_has_flags(struct nn_graph *nn, int src_id, uint32_t flagmask)
{
	struct nn_node *target = find_node(nn,src_id);
	if (target == NULL) return 0;
	return ((target->ops->flags & flagmask) != 0);
}

/*
 * find_range_node
 * Look backward in graph to find the op producing the range.
 * Maybe beef this up a bit to support both FakeConcat and move_relus
 * Probably feed a function pointer that we can have as a visitor for each min or max producing node.
 */

/*
 * Instead of real input, this uses the min value, since sometimes we bypass that around data rearrangement ops 
 */
static inline int find_range_node_visitor(struct nn_graph *nn, int src_id, int (*f)(struct nn_graph *, struct nn_node *, void *), void *opaque)
{
	struct nn_node *target;
	if ((target = find_node(nn,src_id)) == NULL) {
		return errlog(nn,"Can't find node id %d",src_id);
	}
	// the src_id can't have more than one consumer
	if(  find_unique_consumer_anytype(nn, target, 0) == NULL){
		logmsg(nn,3,"can't move range back to node %X - has multiple outputs", (unsigned) target->node_id );
		return 1;
	}

	if (target->ops->flags & NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE) {
		int min_idx = target->n_inputs - 2;
		logmsg(nn,2,"Recursing past op id=%x (%s) n_inputs=%d picked %d as min",
			target->node_id,
			hexagon_nn_op_names[target->node_type],
			target->n_inputs,
			min_idx);
		return find_range_node_visitor(nn,target->input_refs[min_idx].src_id,f,opaque);
	}
	if (target->node_type == OP_QuantizedConcat_8) {
		int n_inputs = (target->n_inputs - 1) / 3;
		int i;
		int pred_id;
		int err = 0;
		for (i = 0; i < n_inputs; i++) {
			pred_id = target->input_refs[1+n_inputs+i].src_id;
			logmsg(nn,2,"Recursing past concat id=%x n_inputs=%d picking %d as min (targ id %x)",
				target->node_id,target->n_inputs,1+n_inputs+i,pred_id);
			if ((err=find_range_node_visitor(nn,pred_id,f,opaque)) != 0) return err;
		}
		/* Done recursing through all concat sources */
		return err;
	}
	return f(nn,target,opaque);
}


static inline struct nn_node *find_range_node(struct nn_graph *nn, int src_id)
{
	struct nn_node *target;
	while(1){
		target = find_node(nn,src_id);
		if (target == NULL) return NULL;
		if ( !(target->ops->flags & NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE))
			break;
		int idx = 0;
		// only used for concat placement - since we're fixing ranges, don't fix ones past fake concat
		if (target->node_type == OP_QuantizedFakeConcat_8_d32) break;
		logmsg(nn,3,"node %x uses input range, recursing (input %d)",src_id,idx);
		src_id = target->input_refs[idx].src_id;
	}
	if (!((target->node_type == OP_Supernode_8x8p8to8_d32)
	||    (target->node_type == OP_Supernode_8x8p32to8_d32))) {
		logmsg(nn,3,"unknown op type on node %x",src_id);
		return NULL;
	}
	return target;
}
// if 'src_id' is a supernode, return its output range values
// if it's a node that just copies its range from the input, then see if the one before is
//  a supernode, and so on. If it fails, the generated range is -INF, +INF
//
static inline int update_range_minmax(struct nn_graph *nn, int src_id, float minmax[2], int src_ids[2])
{
	struct nn_node *target = find_range_node(nn,src_id);
	if (target == NULL) return warnlog(nn,"Couldn't find range node");;
	/* FIXME: maybe use last two args instead of hard coding 11/12? */
	if ((target->node_type != OP_Supernode_8x8p8to8_d32) 
		&& (target->node_type != OP_Supernode_8x8p32to8_d32)) {
		return warnlog(nn,"bad op");
	}
	struct nn_node *minnode = find_node_must_be_Const(nn,target->input_refs[10].src_id);
	struct nn_node *maxnode = find_node_must_be_Const(nn,target->input_refs[11].src_id);
	if (minnode == NULL) {
		logmsg(nn,3,"Can not find min node, or not const");
		minmax[0] = -INFINITY;
	} else {
		float minval = tensor_get_float(minnode->outputs[0],0);
		if (minval < minmax[0]) {
			logmsg(nn,2,"Changing src %x minval %f to %f @ op %x",src_id,minmax[0],minval,minnode->node_id);
			minmax[0] = minval;
			src_ids[0] = minnode->node_id;
		}
	}
	if (maxnode == NULL) {
		logmsg(nn,3,"Can not find max node, or not const");
		minmax[1] = INFINITY;
	} else {
		float maxval = tensor_get_float(maxnode->outputs[0],0);
		if (maxval > minmax[1]) {
			logmsg(nn,2,"Changing src %x maxval %f to %f @ op %x",src_id,minmax[1],maxval,maxnode->node_id);
			minmax[1] = maxval;
			src_ids[1] = maxnode->node_id;
		}
	}
	return 0;
}

#if 0
static inline float find_range_max(struct nn_graph *nn, int src_id)
{
	struct nn_node *target = find_range_node(nn,src_id);
	if (target == NULL) return INFINITY;
	/* FIXME: maybe use last two args instead of hard coding 11/12? */
	struct nn_node *maxnode = find_node(nn,target->input_refs[11].src_id);
	if (maxnode == NULL) {
		logmsg(nn,3,"Can not find max node");
		return INFINITY;
	}
	if (maxnode->node_type != OP_Const) {
		logmsg(nn,3,"max not const");
		return INFINITY;
	}
	return tensor_get_float(maxnode->outputs[0],0);
}

static inline float find_range_min(struct nn_graph *nn, int src_id)
{
	struct nn_node *target = find_range_node(nn,src_id);
	if (target == NULL) return -INFINITY;
	/* FIXME: maybe use last two args instead of hard coding 11/12? */
	struct nn_node *minnode = find_node(nn,target->input_refs[10].src_id);
	if (minnode == NULL) {
		logmsg(nn,3,"Can not find min node");
		return -INFINITY;
	}
	if (minnode->node_type != OP_Const) {
		logmsg(nn,3,"min not const");
		return -INFINITY;
	}
	return tensor_get_float(minnode->outputs[0],0);
}
#endif

static inline int try_fix_minmax(struct nn_graph *nn, int src_id, int node_ids[2])
{
	struct nn_node *target = find_range_node(nn,src_id);
	if (target == NULL) return warnlog(nn,"Couldn't find op");
	if ((target->node_type != OP_Supernode_8x8p8to8_d32) 
		&& (target->node_type != OP_Supernode_8x8p32to8_d32)) return errlog(nn,"bad op");
	target->input_refs[10].src_id = node_ids[0];
	target->input_refs[11].src_id = node_ids[1];
	node_rehash_inputrefs(target);
	return 0;
}

static int do_remove_concats_with_placement(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	if (node->node_type != OP_QuantizedConcat_8_d32) return 0;
	int n_inputs = (node->n_inputs - 1) / 3;
	int i;
	uint32_t depth_so_far;
	float maxval = 0.0f;
	float minval = 0.0f;
	int node_ids[2] = { get_zero_node(nn), get_zero_node(nn) };
	float tmpvals[2] = { 0.0f, 0.0f, };
	for (i = 0; i < n_inputs; i++) {
		if (update_range_minmax(nn,node->input_refs[1+i].src_id,tmpvals,node_ids)) return 0; // failed to find range nodes
		minval = fminf(minval,tmpvals[0]);
		maxval = fmaxf(maxval,tmpvals[1]);
	}
	/* We can only do this optimization if the inputs have fixed, uniform ranges */
	if (nn_isinf(maxval)) {
		logmsg(nn,2,"Self-calculating maxima");
		return 0;
	}
	if (nn_isinf(minval)) {
		logmsg(nn,2,"Self-calculating minima");
		return 0;
	}
	for (i = 0; i < n_inputs; i++) {
		// logmsg(nn,2,"Checking for multi-dest on a concat input...");
		struct nn_node *src_node = find_node(nn,node->input_refs[1+i].src_id);
		if (src_node == NULL) return errlog(nn,"Oops: node disappeared");
		if (check_single_consumer_all(nn,src_node,node) != 0) return 0;
	}
	/* Try to fix all the minima and maxima.  This will probably help accuracy even if we abort later */
	for (i = 0; i < n_inputs; i++) {
		logmsg(nn,2,"Changing %x to min=%x/%f max=%x/%f...",
			node->input_refs[1+i].src_id,
			node_ids[0], tmpvals[0],
			node_ids[1], tmpvals[1]);
		if (try_fix_minmax(nn,node->input_refs[1+i].src_id,node_ids) != 0) {
			logmsg(nn,2,"Can't fix max val");
			return 0;
		}
	}
	/* We can also only do this optimization if the inputs all accept careful placement */
	for (i = 0; i < n_inputs; i++) {
		if (!node_has_flags(nn,node->input_refs[1+i].src_id,NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION)) {
			logmsg(nn,2,"node %x does not accept preparation",node->input_refs[1+i].src_id);
			return 0;
		}
	}
	/* OK, all inputs have the same const maxima and minima, and they all accept careful placement */
	/* Ensure we have appropriate sized depth for each node... */
	depth_so_far = 0;
	for (i = 0; i < n_inputs; i++) {
		struct nn_node *src = find_node(nn,node->input_refs[1+i].src_id);
		if (src == NULL) return errlog(nn,"late inability to find source node");
		if (src->output_defs[0].max_sizes[3] % 32) {
			logmsg(nn,2,"Aborting, depth != 0 mod 32");
			return 0;
		}
		depth_so_far += src->output_defs[0].max_sizes[3];
		logmsg(nn,3,"depth from op id %x adding %d, total %d",src->node_id,src->output_defs[0].max_sizes[3],depth_so_far);
	}
	if (depth_so_far != node->output_defs[0].max_sizes[3]) {
		logmsg(nn,2,"concat out def doesn't match sum of inputs: %d != %d",depth_so_far,node->output_defs[0].max_sizes[3]);
		return 0;
	}
	/* now, create the new op and replace old op with it */
	op_type operation = OP_QuantizedFakeConcat_8_d32;
	struct nn_node *new_node;
	if ((new_node = optab[operation]->ctor(
		nn,
		node->node_id,
		operation,
		node->padding,
		node->n_inputs,
		node->n_outputs,
		node->input_refs,
		node->output_defs)) == NULL) {
		return errlog(nn,"ctor fail");
	}
	logmsg(nn,2,"created node %p id=%x",new_node,new_node->node_id);
	if (replace_nodes(nn,nodeptr,new_node,node) != 0) {
		return errlog(nn,"replace concat fail");
	}
	/* We don't have the old node any more... */
	node = new_node;
	logmsg(nn,2,"*nodeptr=%p newnode->next=%p",*nodeptr,new_node->next);
	logmsg(nn,3,"Fake Concat Success?");
	
	/* OK, now what did we get ourselves into? 
	 * We need to mark all the sources' output tensors, but that can wait for prepare time.
	 * I'm more concerned about memory allocation in the next pass, we need to ensure that the full buffer 
	 * is allocated for the first node producing partial output.  Hopefully this also helps the allocator
	 * not allocate the small outputs of the contributing ops.
	 */
	/* I think here is a winning strategy:
	 * 0) Find the first producer in the graph 
	 * 1) Change first producer's max sizes to this node's
	 * 2) Find last consumer of concat node
	 * 3) Add a sink to first producer's output after last concat node, to prevent it from being freed
	 * 4) Avoid allocation for all other producers and this node by marking data pointer as non-NULL (maybe point to first producer output)
	 * 
	 * Allocator should see correct size for first producer and should keep it alive all the way through the Sink.  
	 * This is the correct lifetime.
	 * Sink just returns, so executing it is inexpensive to have in the graph.
	 *
	 * After allocation, we need to fix up all the pointers: the concat pointer should point to the start of the data,
	 * and the other nodes need to be adjusted to the right d32 offset and have their shapes set just right.
	 * I think this can be done at prepare time
	 */
	/* Find first producer */
	struct nn_node *first = find_first_producer(nn,node);
	if (first == NULL) return errlog(nn,"Oops, couldn't find that producer?");
	/* Change first producer's max size to this node's */
	first->outputs[0]->max_size = node->outputs[0]->max_size;
	/* Find last consumer */
	struct nn_node *last = find_last_consumer(nn,node,0);
	if (last == NULL) return errlog(nn,"Oops, couldn't find last consumer?");
	/* Create Sink node */
	operation = OP_Sink;
	uint32_t new_node_id = nn_graph_new_internal_node_id(nn);
	struct input newnode_input = { .src_id = first->node_id, .output_idx = 0, };
	if ((new_node = optab[operation]->ctor(
		nn,
		new_node_id,
		operation,
		node->padding,
		1,	// inputs
		0,	// outputs
		&newnode_input,
		NULL)) == NULL) {
		return errlog(nn,"ctor fail");
	}
	/* Add sink after last consumer of concat */
	if (insert_nodes(nn, &last->next, new_node) != 1) {
		return errlog(nn,"replace node fail");
	}
	/* Mark this and other contributing nodes' data pointers */
	node->outputs[0]->data = first;
	for (i = 0; i < n_inputs; i++) {
		struct nn_node *tmp = find_node(nn,node->input_refs[1+i].src_id);
		if (tmp == NULL) return errlog(nn,"OOPS: can't find node now");
		if (tmp == first) continue;
		tmp->outputs[0]->data = first;
	}
	/* We can clean up those data pointers, pad values, etc at check() time. */
	logmsg(nn,2,"got all the way here");
	return 0;
}

static int remove_concats_with_placement(struct nn_graph *nn)
{
	return graph_iterator(nn,do_remove_concats_with_placement);
}



/*
 * Convert crazy depthwise convolutions
 * 
 * There are a few cases where "depthwise convolution" is really another op:
 * * 1x1 depthwise convolution with no depth multiplier is elementwise broadcasting multiply.
 * * Depthwise convolution with input channels == 1 is really normal convolution.
 *
 * Convert these into the better representation.
 */

static int do_convert_insane_dwise_to_Mul(struct nn_graph *nn, struct nn_node **nodeptr, struct nn_node *node, struct nn_node *filt)
{
	int new_operation = OP_QuantizedMul_8x8to32;
	struct nn_node *newnode;
	if ((newnode = optab[new_operation]->ctor(
		nn,
		node->node_id,
		new_operation,
		NN_PAD_NA,
		6,3,
		node->input_refs,
		node->output_defs)) == NULL) return errlog(nn,"ctor fail");
	if (replace_nodes(nn,nodeptr,newnode,node) != 0) return errlog(nn,"replace_nodes");
	logmsg(nn,2,"1x1 Depthwise Conv converted to Mul");
	return 0;
}

static int do_convert_insane_dwise_to_Conv2d(struct nn_graph *nn, struct nn_node **nodeptr, struct nn_node *node, struct nn_node *filt, struct nn_node *stride)
{
	int new_operation = OP_QuantizedConv2d_8x8to32;
	struct nn_node *newnode;
	if ((newnode = optab[new_operation]->ctor(
		nn,
		node->node_id,
		new_operation,
		NN_PAD_NA,
		7,3,
		node->input_refs,
		node->output_defs)) == NULL) return errlog(nn,"ctor fail");
	if (replace_nodes(nn,nodeptr,newnode,node) != 0) return errlog(nn,"replace_nodes");
	logmsg(nn,2,"Short Input Depthwise Conv converted to regular Conv2d");
	return 0;
}


static int do_convert_insane_dwise(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *conv = *nodeptr;
	struct nn_node *filt_node;
	struct nn_node *stride_node;
	if (conv->node_type != OP_QuantizedDepthwiseConv2d_8x8to32) return 0;
	if ((filt_node = find_node_must_be_Const(nn,conv->input_refs[1].src_id)) == NULL) return 0;
	if ((stride_node = find_node_must_be_Const(nn,conv->input_refs[6].src_id)) == NULL) return 0;
	/* Check for Elementwise Multiply Case */
	if ((filt_node->outputs[0]->shape.filt_height == 1) 
		&& (filt_node->outputs[0]->shape.filt_width == 1)
		&& (filt_node->outputs[0]->shape.filt_batches == 1)
		&& (stride_node->outputs[0]->shape.height == 1)
		&& (stride_node->outputs[0]->shape.width == 1)) {
		return do_convert_insane_dwise_to_Mul(nn,nodeptr,conv,filt_node);
	}
	/* 
	 * EJP: FIXME: we could expand depth_multiplier depthwise convs of input sizes <= 4 to normal convs 
	 * and it would probably be a win, should be converted to inconv.
	 */
	if (filt_node->outputs[0]->shape.filt_depth == 1) {
		/* Input channels == 1, use normal conv */
		return do_convert_insane_dwise_to_Conv2d(nn,nodeptr,conv,filt_node,stride_node);
	}
	return 0;
}


static int convert_insane_dwise(struct nn_graph *nn)
{
	return graph_iterator(nn,do_convert_insane_dwise);
}


/*
 * Move relus
 * If we see a bare relu (FIXME: TBD: /reluX/Clamp?), it hasn't been fused with supernode.  So try harder.
 * 
 * Look at min input source.
 * * Does it have NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE? Recurse.
 * * Is it a Concat? Recurse for each input.
 * * Is it a data-rearranging op? Maybe we can skip it if we're using min/max instead of input 0, but if not, recurse (or add OUTPUT_USES_INPUT_RANGE to op...).
 * Hopefully we can reuse some find_range_node code.
 * Once we find the min-producing node, is it Supernode? Replace min.
 */

static int move_relu_visitor(struct nn_graph *nn, struct nn_node *node, void *opaque)
{
	if (!((node->node_type == OP_Supernode_8x8p8to8)
	   || (node->node_type == OP_Supernode_8x8p32to8))) {
		logmsg(nn,2,"node %p id=%x type=%s: unknown type",
			node,node->node_id,hexagon_nn_op_names[node->node_type]);
		return -1;
	}
	node->input_refs[10].src_id = get_zero_node(nn);
	node->input_refs[10].output_idx = 0;
	return 0;
}

static int do_move_relus(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	if (node->node_type != OP_QuantizedRelu_8) return 0;
	int src_id = (node->input_refs[1].src_id);
	/* FIXME: look for Relu/ReluX/Clamp */
	if (find_range_node_visitor(nn,src_id,move_relu_visitor,NULL) != 0) {
		logmsg(nn,2,"Couldn't hide relu in all inputs, not removing relu");
		return 0;
	}
	/* Changed all the inputs... now we need to change references to the predecessor.  Dead code removal should remove it. */
	change_refs(nn,node->node_id,0,node->input_refs[0].src_id,node->input_refs[0].output_idx);
	change_refs(nn,node->node_id,1,node->input_refs[1].src_id,node->input_refs[1].output_idx);
	change_refs(nn,node->node_id,2,node->input_refs[2].src_id,node->input_refs[2].output_idx);
	logmsg(nn,2,"Moved RELU to supernodes, dead op removal should get %x",node->node_id);
	return 0;
}

static int move_relus(struct nn_graph *nn)
{
	return graph_iterator(nn,do_move_relus);
}


static int bump_refs(struct nn_graph *nn, struct nn_node **nodeptr)
{
	int i;
	struct nn_node *consumer = *nodeptr;
	struct nn_node *producer = NULL;
	int prod_id = 0;			// often are dups, can skip 'find_node'
	int node_id;
	for (i = 0; i < consumer->n_inputs; i++) {
		node_id = consumer->input_refs[i].src_id;
		if( producer == NULL || (prod_id != node_id)){
			if ((producer = find_node(nn,node_id)) == NULL) {
				return errlog(nn,"can't find id 0x%x",node_id);
			}
			prod_id = node_id;
		}
		producer->refs++;
	}
	return 0;
}


/*
 * To update reference counts:
 * * Clear all reference counts
 * * Go through the graph.  For each input, increment reference count 
 * (at the same time, we update the node count and tail)
 */
static void update_refs(struct nn_graph *nn)
{
	struct nn_node * p = nn->head;
	struct nn_node * tail = NULL;
	int count = 0;
	while( p != 0){
		p->refs = 0;
		tail = p;
		p = p->next;
		++count;
	}
	if( (int)nn->node_count != count ){
		logmsg(nn,0,"===== node count: recorded %d, actual %d\n", (int)nn->node_count, count);
		nn->node_count = count;
	}
	nn->tail = tail;
	graph_iterator(nn,bump_refs);
}

// call after removing dead nodes:
// check to see if the cached nodes are still present and delete from cache if not.
// Also, squeeze out any empty slots which may be thus formed (or which may be already
// in the cache)
//
static void
do_const_cache_cleanup( struct nn_graph *nn)
{
	struct nn_prepare_state *psp = nn->pstate;
	int ncache = psp->scalar_cache_n;
	int last_void = 0;
	for( int i = 0; i < ncache+3; i++){
		uint32_t * nidp = &psp->scalar_cache[i].node_id;
		if( *nidp != 0){
			struct nn_node * np = find_node(nn, *nidp);
			if( np == NULL){	// whoops, it's gone
				*nidp = 0;
				last_void = i;
			}
		}else{
			last_void = i;
		}
	}
	// if last_void >= 3, empty slots were found or created in the 'variable' part.
	if( last_void >= 3){
		struct prep_const_cache_entry * wp = &psp->scalar_cache[3];
		struct prep_const_cache_entry const *rp = wp;
		int new_count = 0;
		for( int i = 0; i < ncache; i++ ){
			if( rp[i].node_id != 0 ){
				*wp++ = rp[i];		// keep it...
				new_count++;
			}
		}
		psp->scalar_cache_n = new_count;
	}
}

// policy for deciding if a node is 'dead'...
// (1) must have zero refs
// (2) must not have NN_NODE_FLAG_RETAIN.
//   NN_NODE_FLAG_RETAIN is set in the ctor, for all nodes with 0 outputs, and
//   also for certain ops (e.g. Variable) which we don't want to scavenge.
//
static inline int
__attribute__((always_inline))
is_node_dead( struct nn_node const *node )
{
	return unlikely(node->refs == 0) &&  (node->flags & NN_NODE_FLAG_RETAIN)==0 ;
}

#if 1
/*
 * Calculate reference counts
 * Find nodes that are dead:
 * * Node must have more than zero outputs (OUTPUT, PPRINT, CHECK nodes, etc)
 * * Every output has zero references
 * Remove Nodes
 * If we removed more than zero nodes, repeat... requires plumbing, just do it 8 times
 */
#if 0
static int do_remove_dead_node(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	//logmsg(nn,0,"nodeptr=%p node=%p",nodeptr,node);
	if (node == NULL) return 0;
	if( !is_node_dead(node)) return 0;
	logmsg(nn,8,"freeing %p node->next=%p",node,node->next);
	*nodeptr = node->next;
	if( nn->tail == node) nn->tail = node->next;	// make sure tail is valid
	DO_DTOR(node,nn);
	if( nn->node_count>0) --nn->node_count;
	return do_remove_dead_node(nn,nodeptr);
}
#endif


static int remove_dead_nodes(struct nn_graph *nn)
{
	unsigned nc0 = nn->node_count;
	int niter = 0;
	while(1){
		update_refs(nn);
		// make a pass through...
		unsigned n_del = 0;				// count of deleted nodes
		unsigned new_node_count = 0;	// count of surviving nodes
		struct nn_node **nodeptr = &nn->head;
		struct nn_node *node;
		while( node = *nodeptr, node!= NULL){
			if (is_node_dead(node)){
				// delete this one
				logmsg(nn,8,"freeing %p node->next=%p",node,node->next);
				*nodeptr = node->next;					// take it out of the list
				DO_DTOR(node,nn);
				n_del ++;
				// don't change nodeptr.
			}else{
				nodeptr = &node->next;
				new_node_count ++;
			}
		}
		++niter;
		nn->node_count = new_node_count;
		if( n_del == 0) break;			// no deletions, stop..

		// if any deletions, ensure tail still is valid.
		nn->tail = nn->head;
	}
	logmsg(nn,2,"deleted %u dead nodes out of %u in %d iterations",
			nc0-nn->node_count, nc0, niter);
	do_const_cache_cleanup(nn);
	return 0;
}
#else
//#define DEADNODES_WITH_QSORT
// compare for qsort; the items being sorted are pointers.
#ifdef DEADNODES_WITH_QSORT
static int deadlist_comp( void const * a, void const *b)
{
	char *ptra = *(char**)a;
	char *ptrb = *(char**)b;
	return ptra-ptrb;
}
#endif
//
// More efficient remove_dead_nodes
//  (1) update ref counts
//  (2) find all nodes which are dead, put pointers in array deadlist (it may fill up)
//  (3) For each of these, update the ref count; if any refs go to 0 and are officially 'dead',
//      add those to the list too. This is done without changing the linked
//      list. Done when we've processed them all (all nodes in deadlist must
//      have their references-to updated, here, even if we've run out of space to
//      remember nodes which become dead as a result).
//  (4) delete the nodes: sort the pointers in deadlist, and then
//      make a pass through the linked-list, removing all dead nodes which appear
//      in deadlist, or which have 0 inputs (dead nodes which don't appear in deadlist
//      can't be deleted, since they have not been accounted for yet).
//  (5) If we ran out of space in (2) or (3) to store dead nodes, go
//      back to (2).
// in step (4), if the list did *not* fill up in (2) and (3), we can safely delete all
//  of the dead nodes we find without sorting and checking the list, since all of them
// must be in the list.
//
// SIMPLIFICATION (if no DEADNODES_WITH_QSORT):
//    - table is not sorted; we always remove all nodes whose reference counts go to
//      zero even if the table was too small to account for them all.
//    - if the table size was too small, we repeat and rebuild the ref counts from scratch
//      on the next attempt.
//
static int
remove_dead_nodes(struct nn_graph *nn)
{
	struct nn_node *deadlist[512];
	int deadlist_max = sizeof(deadlist)/sizeof(deadlist[0]);

	int deadlist_incomplete = 0;

	// (1) update ref counts
#ifdef DEADNODES_WITH_QSORT
	update_refs(nn);
	int allcount = nn->node_count;
	do{
#else
	do{
		update_refs(nn);
		int allcount = nn->node_count;
#endif
		struct nn_node * node;
		int dead_count = 0;
		deadlist_incomplete = 0;
		// (2) find all the dead nodes
		//     or as many as we have room to store pointers to
		for( node = nn->head; node != NULL; node = node->next ){
			if( is_node_dead(node)){	// dead
				if( dead_count >= deadlist_max){
					deadlist_incomplete = 1;
					break;
				}
				deadlist[dead_count++] = node;
			}
		}
		if( dead_count == 0)		// all done
			break;
		// (3) account the references of the ones in the list; keep adding more if there's room
		for( int k = 0; k < dead_count; k++){	// <<< deadcount may ++ in loop
			struct nn_node * doomed_node = deadlist[k];
			struct nn_node * producer = NULL;
			int prod_node_id=0;		// retain previous lookup

			for (int i = 0; i < doomed_node->n_inputs; i++) {
				int node_id = doomed_node->input_refs[i].src_id;
				if( producer == NULL || (node_id != prod_node_id)){
					if ((producer = find_node(nn,node_id)) == NULL) {
						return errlog(nn,"can't find id 0x%x",node_id);
					}
					prod_node_id = node_id;
				}
				if( producer->refs <=1 ){		// it will be 0 now
					if( producer->refs == 0) return errlog(nn, "ref accounting error");
					producer->refs = 0;
					if( is_dead(producer)){
						if( dead_count < deadlist_max){
							deadlist[dead_count++] = producer;
						}else{
							deadlist_incomplete = 1; // will have to go back for that one
						}
					}
				}else{
					producer->refs--;
				}
			}
		}
#ifdef DEADNODES_WITH_QSORT
		// (4) sort the pointers in increasing order.
		// (only needed if the deadlist is incomplete)
		if( deadlist_incomplete ){
			qsort( deadlist, dead_count, sizeof( deadlist[0]), deadlist_comp);
		}
#endif
		// now, go back through the list of nodes. Any dead node we find in there will be removed if:
		//   - deadlist did not fill up; or
		//   - the node has no inputs; or
		//   - the node pointer appears in the (now sorted) deadlist.
		//   Any other 'dead' node is not removed yet, since its references-to have not been
		//   accounted for, and we will process it on the next pass.
		//
		struct nn_node ** nodeptr = &nn->head;
		int rmcount = 0;
		while(  (node = * nodeptr)!= NULL){
			struct nn_node ** next_nodeptr = &node->next;
			if( is_dead(node) ){	// candidate for removal
				struct nn_node * nfound = node;
				// make sure it's in the deadlist, if that's incomplete (and if the node has inputs)
#ifdef DEADNODES_WITH_QSORT
				if( deadlist_incomplete && node->n_inputs > 0){
					int ilo = 0;
					int ihi =  dead_count;
					while( ilo < ihi){
						int imid = (ilo+ihi)>>1;
						nfound= deadlist[imid];
						if( nfound == node) break;		// found it
						if( nfound < node){
							ilo = imid+1;
						}else{
							ihi = imid;
						}
					}
				}
#endif
				if( nfound == node){		// OK to delete
					rmcount++;
					logmsg(nn,8,"freeing %p node",node);
					*nodeptr = node->next;
					next_nodeptr = nodeptr;	// visit this one again
					DO_DTOR(node,nn);
				}
			}
			nodeptr = next_nodeptr;
		}
		if(rmcount)nn->tail = nn->head;
		allcount -= rmcount;
		nn->node_count = (allcount <0)? 0: allcount;
		// not all of the identified nodes were encountered in the deletion pass?
		// possible cause: A node which is (incorrectly) in the hash but not in the linked list,
		// could wind up in the  deadlist, and will not be deleted.
#ifdef DEADNODES_WITH_QSORT
		if( rmcount < dead_count){
			logmsg(nn,0, "unused node removal mismatch: identified %d, removed only %d", dead_count, rmcount);
		}else{
			logmsg(nn,4, "removed %d dead nodes",(int)rmcount);
		}
#else
		logmsg(nn,4, "removed %d dead nodes",(int)rmcount);
#endif
	} while(deadlist_incomplete);

	do_const_cache_cleanup(nn);
	return 0;
}
#endif

#if 0
static void move_nonconst_head_ptr(struct nn_graph *nn)
{
	struct nn_node *tmp;
	do {
		tmp = *nn->nonconst_head_ptr;
		if (tmp == NULL) break;
		if (tmp->node_type != OP_Const) break;
		tmp = tmp->next;
	} while (1);
}
#endif

static int do_gather_const_node(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *tmp = *nodeptr;
	//logmsg(nn,4,"nn=%p nodeptr=%p *nodeptr=%p",nn,nodeptr,*nodeptr);
	if (tmp == NULL) return 0;
	if (tmp->node_type != OP_Const) return 0;
	//logmsg(nn,4,"Const node %x @ %p type %d",tmp->node_id,tmp,tmp->node_type);

	// if it's at nonconst_head_ptr, just move that to after current node.
	if (tmp == *nn->nonconst_head_ptr) {
		nn->nonconst_head_ptr = &tmp->next;
		return 0;
	}
	// remove from list and insert after existing consts (before nonconst_head_ptr)
	*nodeptr = tmp->next;
	tmp->next = *nn->nonconst_head_ptr;
	*nn->nonconst_head_ptr = tmp;
	nn->nonconst_head_ptr = &tmp->next;
	return do_gather_const_node(nn,nodeptr);
}

static int gather_const_nodes(struct nn_graph *nn)
{
	nn->nonconst_head_ptr = &nn->head;
	graph_iterator(nn,do_gather_const_node);
	if (nn->head == NULL) return errlog(nn,"oops: head pointing to NULL");
	return 0;
}

#if 0
// This is a thing we need until we clean up all the
// duplicate const nodes in unit_test
static int
is_duplicate_const_node( struct nn_node * a, struct nn_node * b )
{
	if( a->node_type != OP_Const || b->node_type != OP_Const) return 0;
	int elsize = a->output_defs[0].elementsize;
	if( elsize != b->output_defs[0].elementsize) return 0;
	struct tensor const *ta = a->outputs[0];
	struct tensor const *tb = b->outputs[0];
	if( !shape_matches( &ta->shape, & tb->shape) ) return 0;
	if( elsize ==0) return 1;
	return ( memcmp( ta->data, tb->data, elsize * shape_element_count(&ta->shape)) == 0 );

}
#endif

//
// init hash tables.
// While adding nodes to hash, discover duplicate nodes.
// if any dups are found:
//   - fatal error if either is not Const
//   - if both are const, there is a warning and most recently added is discarded.
// (compatible with previous behaviour, in which the node added later was invisible).
//
//
// This also finds the 'or' of all the flags fields of the ops (done so we can find
// if any nodes in certain classes exist), stores it at nn->op_class_set.

static int init_hashtable( struct nn_graph *nn)
{
	int err = initialize_hash(nn);
	if( err !=0 ) return err;

	struct nn_node * p;
	struct nn_node ** pp;
	int errs = 0;

	int const_node_dups = 0;
	unsigned flags_or = 0;

	for( pp = &nn->head; (p = *pp, p != NULL); ){
		flags_or |= p->ops->flags;
		struct nn_node * px = insert_node_to_hash(nn, p);
		if( px != p){
			if( px == NULL) return -1;
			if( px->node_type == OP_Const && p->node_type == OP_Const){
				logmsg(nn,1,"Const node using id %X ignored; was also used for previous Const", (unsigned)p->node_id);
				const_node_dups++;
				*pp = p->next;
				DO_DTOR(p,nn);
				continue;		// stay on pp
			}else{
				errlog(nn,"re-use of node id 0x%X", (unsigned)p->node_id);
				errs++;
			}
		}
		pp = &p->next;
	}
	if( const_node_dups){
		nn->node_count = (nn->node_count >= const_node_dups)? (nn->node_count - const_node_dups): 0;
		nn->tail = nn->head;
	}
	// this should already be accurate, based on the ctors; but
	// doesn't hurt to start clean
	nn->op_class_set = flags_or & NN_NODE_FLAGS_SET;
	return errs? -1:0;
}


#define CHECK(OPTS) 	if ((err= check_graph(nn,(OPTS)))!=0) return err;

//#define CHECK_PERFORMANCE_PREPARE 1

static int optimize(struct nn_graph *nn)
{
	int err;

#ifdef CHECK_PERFORMANCE_PREPARE
	uint32_t cyc0 = nn_os_get_cycles(nn);
	unsigned runtimes[18];
	int nodecount0 = nn->node_count;
#define OPTIMIZE_PERF(n) { uint32_t cyc1 = nn_os_get_cycles(nn); if(n>=0) runtimes[n] = (cyc1-cyc0)>>10; cyc0 = cyc1;}
#else
#define OPTIMIZE_PERF(n)
#endif

	init_hashtable(nn);
	OPTIMIZE_PERF(0)
	CHECK(GRAPHCHECK_HASH)  OPTIMIZE_PERF(-1)

	if( (nn->op_class_set & NN_NODE_FLAG_CLS_REQUANTRANGE)!=0){		// any RequantizationRange_32?
		if ((err = make_autorequantize(nn)) != 0) return err;
	}
	OPTIMIZE_PERF(1)
	if( (nn->op_class_set & NN_NODE_FLAG_CLS_QUANTIZE)!=0){			// any Quantize?
		if ((err = make_autoquantize(nn)) != 0) return err;
	}
	OPTIMIZE_PERF(2)
	if( (nn->op_class_set & NN_NODE_FLAG_CLS_DWCONVF)!= 0){			// any DepthwiseConv2d_f ?
		if ((err = make_quantized_dwise(nn)) != 0) return err;
	}
	if (1) if ((err = convert_insane_dwise(nn)) != 0) return err;			// Convert QuantizedDepthwiseConv to regular Conv for some cases
	OPTIMIZE_PERF(3)
	if ((err = remove_unnecessary_quants(nn)) != 0) return err;
	OPTIMIZE_PERF(4)
	if ((err = remove_unnecessary_dequant_quants(nn)) != 0) return err;
	OPTIMIZE_PERF(5)
	if ((err = make_optimize_axisshuffle(nn)) != 0) return err;

	if( (nn->op_class_set & NN_NODE_FLAG_CLS_CHANSHUFFLE)!= 0){			// any QuantizedChannelShuffle_8 ?
		if ((err = combine_chanshuffle(nn)) != 0) return err;
	}
	OPTIMIZE_PERF(6)
	CHECK(GRAPHCHECK_HASH)  OPTIMIZE_PERF(-1)
	if ((err = remove_dead_nodes(nn)) != 0) return err;
	OPTIMIZE_PERF(7);
	CHECK(GRAPHCHECK_DEADNODES|GRAPHCHECK_HASH)  OPTIMIZE_PERF(-1)
	if ((err = make_reluX_nodes(nn)) != 0) return err;
	OPTIMIZE_PERF(8);
	if ((err = mark_biasadd_nodes(nn)) != 0) return err;
	OPTIMIZE_PERF(9);
	if ((err = gather_const_nodes(nn)) != 0) return err;
	OPTIMIZE_PERF(10);
	if ((err = make_supernodes(nn)) != 0) return err;
	OPTIMIZE_PERF(11);
	if ((err = make_supernode_3322(nn)) != 0) return err;
	if((nn->op_class_set & NN_NODE_FLAG_CLS_OEMNODE)!=0){
		if ((err = expand_oem_nodes(nn)) != 0) return err;
	}
	OPTIMIZE_PERF(12);
	CHECK(GRAPHCHECK_HASH)  OPTIMIZE_PERF(-1)
	if( (nn->op_class_set & NN_NODE_FLAG_CLS_QUANTMUL8TO32)!=0){			// any QuantizeMul_8x8to32?
		if ((err = fold_scalar_mpys(nn)) != 0) return err;
	}
	OPTIMIZE_PERF(13);
	if(0) if ((err = pad_bad_supernodes(nn)) != 0) return err;
	if ((err = move_relus(nn)) != 0) return err;
	if ((err = convert_to_depth32(nn)) != 0) return err;
	OPTIMIZE_PERF(14);
	if ((err = remove_unnecessary_d32_converts(nn)) != 0) return err;
	OPTIMIZE_PERF(15);
	CHECK(GRAPHCHECK_HASH)  OPTIMIZE_PERF(-1)
	// We have to remove dead nodes before we remove concats with placement,
	// or we will end up with stale D32 converts also showing as consumers.
	if ((err = remove_dead_nodes(nn)) != 0) return err;
	if (1) if ((err = remove_concats_with_placement(nn)) != 0) return err;
	if ((err = remove_dead_nodes(nn)) != 0) return err;
	OPTIMIZE_PERF(16);
	if ((err = gather_const_nodes(nn)) != 0) return err;
	OPTIMIZE_PERF(17);
	CHECK(GRAPHCHECK_DEADNODES|GRAPHCHECK_HASH|GRAPHCHECK_NONCONST)

#ifdef CHECK_PERFORMANCE_PREPARE
	logmsg(nn,0,"optimize %d->%d nodes (x1k cyc): %d %d %d %d; %d %d %d %d; %d %d %d %d; %d %d %d %d; %d %d",
		nodecount0, (int)nn->node_count,
		runtimes[0],runtimes[1], runtimes[2], runtimes[3],
		runtimes[4],runtimes[5], runtimes[6], runtimes[7],
		runtimes[8],runtimes[9], runtimes[10], runtimes[11],
		runtimes[12],runtimes[13], runtimes[14], runtimes[15],
		runtimes[16],runtimes[17]);
#endif
	return 0;
}

static int note_a_predecessor(struct nn_graph *nn, struct nn_node **nodeptr)
{
	struct nn_node *node = *nodeptr;
	if (node == NULL) return 0;
	if (node->next == NULL) return 0;
	if (node->next->ops->earlywork_note_pred == NULL) return 0;
	return node->next->ops->earlywork_note_pred(node->next,nn,node);
}

static int note_predecessors(struct nn_graph *nn)
{
	return graph_iterator(nn,note_a_predecessor);
}

static int do_prepare_inner(struct nn_graph *nn)
{
	int err;
	nn_os_hvx_power_on(nn);
	if (nn->state != NN_GRAPH_CONSTRUCTION) {
		return errlog(nn,"prepare: Graph not under construction");
	}
	//if ((err = run_op_setup(nn)) != 0) return err; /* FIXME: needed? Or just call ctor? */
	if ((err = optimize(nn)) != 0) return err;
	if ((err = prepare_inputs(nn)) != 0) return err;
	if ((err = allocate_graph_storage(nn)) != 0) return err;
	if ((err = run_op_check(nn)) != 0) return err;
	if (0) if ((err = note_predecessors(nn)) != 0) return err;
	nn_os_hvx_power_off(nn);
	nn->state = NN_GRAPH_PREPARED;
#ifdef SHOWY_DEBUG
	graphviz_print_graph(nn);
#endif

	// Print the final graph structure in YAML format, for debug
	if (nn->enable_graph_print) {
		print_graph_to_file(nn);
	}

	return 0;
}

void check_processor_version(struct nn_graph *nn) {
	// Check the DSP processor version against what we compiled for.
	// See also:   (/prj/dsp/qdsp6/arch/hexagon_sdk_34/Hexagon_SDK/3.4.0/libs/common/qurt/computev65/include/qurt/qurt_event.h )
	// qurt_sysenv_procname.asid
	// qurt_sysenv_procname.name
	// qurt_sysenv_get_hw_timer()
	// qurt_sysenv_get_process_name()
	//
#ifdef USE_OS_QURT
	qurt_arch_version_t av;
	qurt_sysenv_get_arch_version(&av);
	//   v66 (8150) e.g. 0x00018466
	//      (talos) e.g. 0x00004066
	logmsg(nn,1,"Arch_Version: %08x", av.arch_version);
	int shortver = av.arch_version & 0xff;
#ifdef HEXAGON_V68
	int expver = 0x68;
#else
#ifdef HEXAGON_V66
	int expver = 0x66;
#else
#ifdef HEXAGON_V65
	int expver = 0x65;
#else
	int expver = 0x60;
#endif // HEXAGON_V65
#endif // HEXAGON_V66
#endif // HEXAGON_V68
	if (shortver != expver) {
		errlog(nn,"WARN: ARCH-MISMATCH (compiled for v%x != actual v%x)", expver, shortver);
	}
#endif // USE_QURT_OS
}

int do_prepare(struct nn_graph *nn)
{
	check_processor_version(nn);

	struct nn_prepare_state prepstate;
	memset( &prepstate, 0, sizeof(prepstate));
	nn->pstate = &prepstate;
	int res = do_prepare_inner(nn);
	nn->pstate = NULL;
	return res;
}


