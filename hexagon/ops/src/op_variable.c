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
 * This contains the code for variable nodes.
 */

#include <nn_graph.h>
#include <stdlib.h>

//
// Variable:
// This has 'n' inputs (presumed to be 'const' nodes) and 'n' outputs.
// During 'check': the outputs are initialized by copying them from the inputs.
// During execute: nothing happens.
//  Variables can be changed in two ways:
//    - a variable referenced as an even-numbered input of an 'Assign' node
//      will be updated (from  the odd input) when the Assign node is executed.
//    - there are apis variable_write( graph, node, out_index, ... )
//                 and variable_write_flat( graph, node, out_index ... )
//      which may be used to load variables. Also 'variable_read' to read. See
//      functions below for details.
//
// Enhancement: an 'empty' const node (with a non-empty shape, but with elementsize == 0)
//  may be used as the 'init' input to a Variable with a non-zero elementsize on the
//  corresponding output. In this case, during 'check', the output tensor is initialized
//  with the shape of the const, and based on the elementsize, with all-"zero" values. When
//  the elementsize is 1, the output_defs[i].zero_offset is used as the zero value; for other
// element sizes, actual zero  is used.

static int variable_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"variable execute. self=%p ",self);
	/* Nothing to do! */
	return 0;
}

static int variable_check(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	logmsg(nn,2,"Checking variable node %p",self);
	if (self->n_inputs > self->n_outputs) return errlog(nn,"too many inputs");
	for (i = 0; (i < self->n_inputs) && (i < self->n_outputs); i++) {
		struct tensor const *src_tensor = self->inputs[i];
		struct tensor *dst_tensor = self->outputs[i];
		uint32_t elsize;
		if( src_tensor->data_size == 0			// is an empty tensor
			&& ( elsize = self->output_defs[i].elementsize, elsize > 0 )){
			// fill with zeros
			// don't trust that an 'empty' shape is sane
			uint32_t elements = mulu32_x4_sat( src_tensor->shape.batches,
					src_tensor->shape.height, src_tensor->shape.width, src_tensor->shape.depth);
			uint32_t totalsize = mulu32_sat( elements, elsize );
			if( (int32_t)totalsize <= 0){
				return errlog(nn,"bad shape/elsize for zero-init");
			}else if( totalsize > dst_tensor->max_size){
				return errlog(nn,"variable too small for zero-init");
			}else{
				int fillval = 0;
				if (elsize == 1) fillval = self->output_defs[i].zero_offset;
				memset( dst_tensor->data, fillval, totalsize);
				dst_tensor->shape = src_tensor->shape;
				dst_tensor->data_size = totalsize;
				dst_tensor->format.raw0 = dst_tensor->format.raw1 = 0;
			}
		}else{
			if (tensor_copy(dst_tensor, src_tensor) != 0) {
				return errlog(nn,"out too small");
			}
		}
	}
	return 0;
}

// this ctor is used for Variable and Assign
// (for Assign, it only sets the NN_NODE_FLAG_RETAIN flag).
//

static struct nn_node *variable_assign_ctor(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	int i;
	uint32_t data_size = 0;
	uint8_t *p;
	struct nn_node *self;
	if ((self = node_alloc_common(
		nn,
		node_id,
		operation,
		padding,
		num_inputs,
		num_outputs,
		inputs,
		outputs)) == NULL) {
		errlog(nn,"alloc node");
		return NULL;
	}
	if( operation == OP_Variable){
		// Compute the total memory we'll need for all outputs
		for (i = 0; i < num_outputs; i++) {
			uint32_t size_of_output = outputs[i].elementsize;
			for (int j=0; j<outputs[i].rank; j++) {
				size_of_output *= outputs[i].max_sizes[j];
			}
			data_size += nn_align_up(128, size_of_output);
		}
		// Request the memory we'll need to store outputs consecutively
		if ((self->opaque = nn_memalign(128,data_size)) == NULL) {
				errlog(nn,"tensor storage");
		}
		// Divide and assign the memory to the various outputs
		p = self->opaque;
		for (i = 0; i < num_outputs; i++) {
			self->outputs[i]->data = p;
			uint32_t size_of_output = outputs[i].elementsize;
			for (int j=0; j<outputs[i].rank; j++) {
				size_of_output *= outputs[i].max_sizes[j];
			}
			p += nn_align_up(128, size_of_output);
		}
	}
	// prevent removal of the node, even if outputs are not used
	self->flags |= NN_NODE_FLAG_RETAIN;

	return self;
}

static int variable_dtor(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"variable node %p dtor",self);
	if (self->opaque) nn_free(self->opaque);
	self->opaque = NULL;
	return node_free_common(self,nn);
}


//
// gatekeeper for variable access from API:
// maps node_id to struct node *, 
// if all these check out:
//   (1) graph has been prepared
//   (2) node exists and is a variable
// 	 (3) specified output # is in range
//
static struct nn_node *
locate_variable_node( struct nn_graph * nn, uint32_t node_id, int out_idx, char const *op )
{
	if( nn->state != NN_GRAPH_PREPARED){
		errlog( nn, "variable_%s: must prepare graph first",op);
		return NULL;
	}
	struct nn_node * node = find_node( nn, node_id );
	if( node == NULL || node->node_type != OP_Variable){
		errlog( nn, "variable_%s: node 0x%X not found or not Variable", op,(unsigned)node_id);
		return NULL;
	}
	if( out_idx < 0 || out_idx >= node->n_outputs ){
		errlog( nn, "variable_%s: node 0x%X does not have output %d", op, 
		(unsigned)node_id, out_idx);
		return NULL;
	}
	return node;
}

//
//
// variable read API
// check:
//   (1) graph has been prepared
//   (2) node exists and is a variable
// 	 (3) specified output # is in range
//	 (4) buffer must be large enough for variable's data
//
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
	uint32_t *data_out_len)
{
	struct nn_node *self = locate_variable_node( nn, node_id, output_index, "read" );
	if( self == NULL ) return -1;
	struct tensor *data = self->outputs[output_index];
	if (data_out_max < data->data_size) return errlog(nn,"too small");
	*b_out = data->shape.batches;
	*h_out = data->shape.height;
	*w_out = data->shape.width;
	*d_out = data->shape.depth;
	*data_out_len = data->data_size;
	if( data -> data_size > 0 )
		memcpy(data_out,data->data,data->data_size);
	return 0;
}
//
// variable write API
// check:
//   (1) graph has been prepared
//   (2) node exists and is a variable
// 	 (3) specified output # is in range
//   (4) dimensions must be sane
//   (5) prod(dimensions)*elsize must == data_in_size
//	 (6) data_in_size <= max_size
//
int do_variable_write(
	struct nn_graph *nn,
	uint32_t node_id,
	int output_index,
	uint32_t b,
	uint32_t h,
	uint32_t w,
	uint32_t d,
	const uint8_t *data_in,
	uint32_t data_in_size)
{
	struct nn_node *self = locate_variable_node( nn, node_id, output_index, "write" );
	if( self == NULL ) return -1;
	struct tensor *data = self->outputs[output_index];
	
	unsigned elbytes = self->output_defs[output_index].elementsize;
	uint32_t elements = mulu32_x4_sat( b,h,w,d );
	uint32_t total_bytes = mulu32_sat( elements,elbytes );
	
	if( elements == 0 || total_bytes == (uint32_t)-1 )
		return errlog(nn,"variable_write: invalid shape");
	if( total_bytes != data_in_size )
		 return errlog(nn,"variable_write: size/shape mismatch for elsize=%u", elbytes);
	if( total_bytes > data->max_size )
		 return errlog(nn,"variable_write: not enough space");

	data->shape.batches = b;
	data->shape.height = h;
	data->shape.width = w;
	data->shape.depth = d;
	data->data_size = data_in_size;
	if( data_in_size > 0 )
		memcpy(data->data,data_in,data_in_size);
	return 0;
}

// variable 'flat' write API
// Like variable_write, except:
//   - does not change shape of tensor; supplied data
//    must exactly match existing length
//   - you can specify '-1' as the output_idx, and
//    load all outputs (with concatenated data) 
// check:
//   (1) graph has been prepared
//   (2) node exists and is a variable
// 	 (3) specified output # is in range
//   (4) data_in_size matches data_size (or sum of)
//
int do_variable_write_flat(
	struct nn_graph *nn,
	uint32_t node_id,
	int output_index,
	const uint8_t *data_in,
	uint32_t data_in_size)
{
	struct nn_node *self = locate_variable_node( nn, node_id, 
		(output_index==-1)?0:output_index, "write_flat");
	if( self == NULL ) return -1;
	// select range: a single output, or all of them
	int first_output = output_index;
	int end_output  = output_index+1;
	if( output_index == -1 ){
		first_output = 0;
		end_output = self->n_outputs;
	}
	// sum the sizes
	uint32_t allsize = 0;
	for( int i = first_output; i < end_output; i++ ){
		allsize += self->outputs[i]->data_size;
	}
	if( data_in_size != allsize )
		return errlog(nn,"variable_write_flat: expected %u bytes, got %u",
			(unsigned)allsize, (unsigned)data_in_size);
	// do the copies
	for( int i = first_output; i < end_output; i++ ){
		unsigned tsize = self->outputs[i]->data_size;
		if( tsize > 0 ){
			memcpy( self->outputs[i]->data, data_in, tsize );
			data_in += tsize;
		}
	}
	return 0;
}

// assign has 2*n inputs, and at most n outputs.
// Action is:
//  (1) copy data from odd inputs to even inputs (the even 'inputs' are expected
//   to be connected to upstream Variable outputs).
//
//  (2) for each output (and there may be none) also copy the data from corresponding
//    odd input. If an output's elementsize is zero, that copy will be skipped.
//


static int assign_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* Copy odd inputs to even inputs */
	int i;
	struct nn_memcpy_manager mcman;
	nn_mcmanager_init(nn, &mcman );
	int res = 0;

	logmsg(nn,2,"assign execute. self=%p inputs=%d",self,self->n_inputs);

	for (i = 0; i < self->n_inputs; i += 2) {
		res = nn_mcmanager_tensor_copy( nn,&mcman, (struct tensor *)self->inputs[i],self->inputs[i+1] );
		if( res != 0){
			errlog(nn,"can't copy to input %d",i);
			goto finish;
		}
	}

	/* Copy input to output... or fudge output ptr in setup? */

	for (i = 0; i < self->n_outputs; i++) {
		if( self->output_defs[i].elementsize > 0){
			res = nn_mcmanager_tensor_copy(nn, &mcman, self->outputs[i],self->inputs[2*i+1] );
			if( res != 0){
				return errlog(nn,"can't copy to output %d",i);
				goto finish;
			}
		}
	}
 finish:
	nn_mcmanager_wait( nn, &mcman);
	return res;
}

// 'assign' must have
//  (1) even # of inputs and at least 2
//  (2) # of outputs must be at most inputs/2.
//

static int assign_check(struct nn_node *self, struct nn_graph *nn)
{
	/* Check 2N inputs,  <=1N output */
	int n_inpair = self->n_inputs >>1;

	if ( n_inpair < 1 || (self->n_inputs & 1)!= 0)  return errlog(nn,"bad # inputs (must be even, >=2)");
	if (self->n_outputs > n_inpair ) return errlog(nn,"too many outs");

	// check that all the even inputs are connected to 'Variable' nodes
	for( int i= 0; i < n_inpair; i++ ){
		struct input const * inref = &self->input_refs[2*i];
		struct nn_node const * varnode = find_node(nn, inref->src_id);
		if( varnode == NULL || varnode->node_type != OP_Variable  || inref->output_idx >= varnode->n_outputs ){
			return errlog(nn,"assign node id=0x%X: input %d not connected to valid Variable output",
					(unsigned)self->node_id, 2*i);
		}
	}
	return 0;
}


struct nn_node_ops nn_ops_for_Assign = {
	.execute = assign_execute,
	.check = assign_check,
	.ctor = variable_assign_ctor,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_GE(2),
	.n_outputs = NN_IOCOUNT_GE(0),
};

struct nn_node_ops nn_ops_for_Variable = {
	.execute = variable_execute,
	.check = variable_check,
	.ctor = variable_assign_ctor,
	.dtor = variable_dtor,
	.n_inputs = NN_IOCOUNT_GE(1),
	.n_outputs = NN_IOCOUNT_GE(1),
};


