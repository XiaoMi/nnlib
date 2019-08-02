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
#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>

#include "nn_prepare.h"
#include "nn_oemnode.h"
#if NN_GRAPH_WITH_LENS_LSTM
//
// 'Meta Node' for Lens Lstm
// This node contains no real 'check' or 'execute' - it is intended to be transformed
// into a group of other nodes during graph prepare.
//
//  14 inputs:
//     0  :  scalar int 0, reserved to mark variations
//     1  :  input array qu8 [b,h,w,d_newin]		# 'new input'
//     2  :  input array qu8 [b,h,w,d_oldin]		# previous data input
//     3  :  scalar float, min for  1,2				# expected: -1.0
//     4  :  scalar float, max for  1,2				# expected: 0.99219 = 127/128
//     5  :  weights for fully connected: qu8, [1,1,d_newin+d_newout,d_out*4]
//     6  :  scalar float, min for weights
//     7  :  scalar float, max for weights
//     8  :  bias tensor, qint32,  [1,1,1,d_out*4]
//	   9  :  scalar float, min for bias 
//	   10 :  scalar float, max for bias
//     11 :  previous internal state [b,w,d, d_out]
//     12 :  scalar float, min for internal state		#expected: -16.0
//     13 :  scalar float, max for internal state		#expected: 16.0
//
//  6,9 or 12 outputs:
//     0  :  output array qu8 [b,h,w,d_out]
//     1  :  scalar float, min for output 0 (fixed at -1.0)
//     2  :  scalar float, max for output 0 (fixed at 0.99219 = 127/128)
//     3  :  output internal state array int16 [b,h,w,d_out]
//     4  :  scalar float, min for output 3 (same as inp # 12)
//     5  :  scalar float, max for output 3 (same as inp # 13)
//     --
//     6  :  output internal node qu8 [b,h,w,d_newin+d_newout] (concat of inputs 1,2)
//     7  :  scalar float, min for output 6 (same as inp # 3)
//     8  :  scalar float, max for output 6 (same as inp # 4)
//     --
//     9  :  output internal node int16 [b,h,w,4*d_out] (output of matmul)
//     10 :  scalar float, min for output 9 (=-8.0)
//     11 :  scalar float, max for output 9 (=+8.0)
//
//
//  NOTE:
//    - ranges for bias and for int32 are symmetric, so 'max' on these are redundant.
//    - all inputs must be const nodes except for #1,#2,#11.
//
//
// This is called when the node at *nodeloc
// is a LSTM node; it will replace it with the required subgraph
// return 0 if successfully transformed, or negative if an error raised.
//
//
int __attribute__((cold))
transform_metanode_GL_Lstm( struct nn_graph * nn, struct nn_node ** nodeloc )
{
	struct nn_node *meta_node = *nodeloc;
	if( meta_node == NULL || meta_node->n_inputs != 14 ){
		return errlog(nn,"metanode does not have 14 inputs");
	}
	int n_out = meta_node->n_outputs;
	if( n_out != 6 &&  n_out != 9 && n_out != 12 ){
		return errlog(nn,"metanode has improper # of outputs");
	}

	// collect all the source nodes (not #0)
	struct nn_node * source_nodes[14];
	struct output const * source_odescs[3];		// for non-const inputs
	source_nodes[0] = NULL;
	for(int i= 1; i < 14; i++){
		struct input iref = meta_node->input_refs[i];
		struct nn_node *srcnode = find_node(nn,iref.src_id);
		if( srcnode == NULL ) return errlog(nn,"node %X not found", iref.src_id );
		// must be const except 1,2,11
		if( i >= 3 && i != 11 ){
			if( iref.output_idx!=0 || srcnode->node_type != OP_Const || srcnode->outputs[0]==NULL
					|| srcnode->outputs[0]->data == NULL )
				return errlog(nn,"input %d is not const (or is empty)", i );
		}else{
			if( iref.output_idx >= srcnode->n_outputs )
				return errlog(nn,"bad source ref %X.%d", iref.src_id, iref.output_idx);
			source_odescs[(i>=3)?2:(i-1)] = & srcnode->output_defs[iref.output_idx];

		}
		source_nodes[i] = srcnode;
	}

	// ranges of input and weights can determine scaling
	//
	float in_min = tensor_get_float( source_nodes[3]->outputs[0],0);
	float in_max = tensor_get_float( source_nodes[4]->outputs[0],0);
	float wt_min = tensor_get_float( source_nodes[6]->outputs[0],0);
	float wt_max = tensor_get_float( source_nodes[7]->outputs[0],0);
	float SOP_step = (in_max-in_min)*(wt_max-wt_min)/(255.0f*255.0f);
	// That is the SOP step; it must be <= output step.
	float min_range = SOP_step*32768.0f;

	float matmul_output_range = 8.0f;
	if( matmul_output_range < min_range){
		// need to use a larger range.
		// round up to a multiple of 2/3.
		// (and if it's close to exact, round it up one more).
		int k = (int)(min_range*1.5f + 1.001f);
		matmul_output_range = (float)k/1.5f;
		logmsg(nn,2,"bumping lstm matmul range to %f", matmul_output_range);
	}

	// collect the dimensions of the overall operation
	//
	//  source_odescs[0]:   input #1
	//  source_odescs[1]:	input #2
	//  source_odescs[2]:   input #11
	//
	int d_in_new = source_odescs[0]->max_sizes[3];
	int d_in_old = source_odescs[1]->max_sizes[3];
	int d_concat = d_in_new + d_in_old;
	int d_out = meta_node->output_defs[0].max_sizes[3];

	struct shape const *weights_shp = &source_nodes[5]->outputs[0]->shape;
	if( weights_shp->batches != 1 || weights_shp->height != 1
		 || weights_shp->width != d_concat || weights_shp->depth != 4*d_out ){
		return errlog(nn, "lstm weights shape not correct");
	}
	struct shape const *bias_shp = &source_nodes[8]->outputs[0]->shape;
	if( bias_shp->batches != 1 || bias_shp->height != 1
		 || bias_shp->width != 1 || bias_shp->depth != 4*d_out ){
		return errlog(nn, "lstm bias shape not correct");
	}
	// check all the upper dims are consistent
	for( int i =0; i < 4; i++){
		int d =  source_odescs[0]->max_sizes[i];
		if( i < 3){
			if( source_odescs[1]->max_sizes[i] != d || source_odescs[2]->max_sizes[i] != d ){
				return errlog(nn,"input shapes not consistent");
			}
		}else{
			d = d_out;
		}
		// check outputs shapes too (don't bother with optional)
		if( meta_node->output_defs[0].max_sizes[i] != d || meta_node->output_defs[3].max_sizes[i] != d ){
			return errlog(nn,"output shape not matching input");
		}
	}
	// check to see of the configuration is supported by the nodes.
	//
	if( (d_out&15) !=0 )return errlog(nn,"Lstm nodes output must be a multipe of 16");



	////// OK start building it //////////////////////
	// Nodes wil be added in this order. The lstmout node inherits the node id.
	//
	struct nn_node * concat_node = NULL;
	struct nn_node * fcon_node = NULL;
	struct nn_node * lstmin_node = NULL;
	struct nn_node * lstmout_node = NULL;

	//
	// make the consts we need.
	//
	uint32_t three_nid = create_const_int32_op(nn,3);
	uint32_t plus8_nid = create_const_float_op(nn,matmul_output_range);
	uint32_t minus8_nid = create_const_float_op(nn,-matmul_output_range);
	if(three_nid == 0 || plus8_nid == 0 || minus8_nid == 0) return -1;

	uint32_t concat_nid = 0, fcon_nid =0, lstm_in_nid = 0, lstm_out_nid = 0;
	//
	// we need two 'config' arrays
	//
	uint32_t lstm_in_config_nid =  nn_graph_new_internal_node_id(nn);
	uint32_t lstm_out_config_nid =  nn_graph_new_internal_node_id(nn);
	{
		int32_t lstm_input_conf[8]= { d_out, 0,
						0, 1*d_out,  0, 0*d_out, 0, 2*d_out };
		int res1 = do_prepend_const_node( nn, lstm_in_config_nid, 1,1,1,8, (uint8_t const*)lstm_input_conf, 8*sizeof(int32_t));
		int32_t lstm_output_conf[3]= { d_out, 0, 3*d_out };
		int res2 = do_prepend_const_node( nn, lstm_out_config_nid, 1,1,1,3, (uint8_t const*)lstm_output_conf, 3*sizeof(int32_t));
		if( res1 != 0 || res2 != 0)
			return errlog(nn,"failed to make config const for LSTM");
		// TODO: put something in the prepare_state to cache these.
	}


	// Make OP_QuantizedConcat_8
	{
		concat_nid = nn_graph_new_internal_node_id(nn);
		struct input node_ins[7] = {
				{ three_nid, 0 },
				meta_node->input_refs[1], meta_node->input_refs[2],
				meta_node->input_refs[3], meta_node->input_refs[3],
				meta_node->input_refs[4], meta_node->input_refs[4]
		};
		struct output node_outs[3] = {
				*source_odescs[0],		// need to modify depth later
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 },
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 }
		};
		node_outs[0].max_sizes[3] = d_concat;
		int optype = OP_QuantizedConcat_8;

		concat_node = optab[optype]->ctor(
			nn,
			concat_nid,
			optype,
			NN_PAD_NA,
			7, 3,
			node_ins, node_outs);
		// prevent it from converting to d32.
		if( concat_node != NULL) concat_node->flags |= NN_NODE_FLAG_NO_CONVERT_D32;
	}
	// Make OP_QuantizedMatMulDims_8x8p32to16 (this is a variant which
	// maintains the upper shape dimensions as they are)

	if( concat_node != NULL){
		fcon_nid = nn_graph_new_internal_node_id(nn);
		struct input node_ins[11] = {
				{ concat_nid, 0 },			// input to matmul (from concat)
				meta_node->input_refs[5],		// weights array
				{ concat_nid, 1}, {concat_nid, 2},	// input range
				meta_node->input_refs[6], meta_node->input_refs[7],	// weights range
				meta_node->input_refs[8],					// bias
				meta_node->input_refs[9], meta_node->input_refs[10],	// bias range
				{  minus8_nid, 0 }, {plus8_nid, 0}			// output range
		};
		struct output node_outs[3] = {
				*source_odescs[0],		// need to modify depth, elementsize later
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 },
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 }
		};
		node_outs[0].max_sizes[3] = d_out*4;
		node_outs[0].elementsize = 2;
		int optype = OP_QuantizedMatMulDims_8x8p32to16;
		fcon_node = optab[optype]->ctor(
			nn,
			fcon_nid,
			optype,
			NN_PAD_NA,
			11, 3,
			node_ins, node_outs);
	}

	// Make OP_QuantizedLstmInput_16x16to16

	if( fcon_node!= NULL){
		lstm_in_nid = nn_graph_new_internal_node_id(nn);
		struct input node_ins[7] = {
				{ lstm_in_config_nid, 0 },
				{ fcon_nid, 0 },				// output from fcon
				meta_node->input_refs[11],		// old internal state
				{ fcon_nid, 1 },  meta_node->input_refs[12],	// mins
				{ fcon_nid, 2 },  meta_node->input_refs[13],	// maxes
		};
		struct output node_outs[3] = {
				meta_node->output_defs[3],
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 },
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 }
		};
		int optype = OP_QuantizedLstmInput_16x16to16;
		lstmin_node = optab[optype]->ctor(
			nn,
			lstm_in_nid,
			optype,
			NN_PAD_NA,
			7, 3,
			node_ins, node_outs);
	}

	// Make OP_QuantizedLstmOutput_16x16to8
	// (keep original node_id)
	//
	if( lstmin_node != NULL)
	{
		lstm_out_nid = meta_node->node_id;
		struct input node_ins[7] = {
				{ lstm_out_config_nid, 0 },
				{ fcon_nid, 0 },				// output from fcon
				{ lstm_in_nid, 0 },				// output from lstm_in
				{ fcon_nid, 1},{ fcon_nid, 2},	// range from fcon
				{ lstm_in_nid, 1}, {lstm_in_nid,2}	// range from lstm_in
		};
		struct output node_outs[3] = {
				meta_node->output_defs[0],
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 },
				{.rank=4, .max_sizes={1,1,1,1}, .elementsize = 4 }
		};
		int optype = OP_QuantizedLstmOutput_16x16to8;
		lstmout_node = optab[optype]->ctor(
			nn,
			lstm_out_nid,
			optype,
			NN_PAD_NA,
			7, 3,
			node_ins, node_outs);
	}
	if( lstmout_node == NULL){
		if( lstmin_node != NULL){
			lstmin_node->ops->dtor(lstmin_node,nn);
		}
		if( fcon_node != NULL){
			fcon_node->ops->dtor(fcon_node,nn);
		}
		if( concat_node != NULL){
			concat_node->ops->dtor(concat_node,nn);
		}
		return errlog(nn,"error creating nodes");
	}
	int res = replace_node_with(nn, nodeloc, meta_node, concat_node, fcon_node, lstmin_node, lstmout_node);
	if( res < 0) return errlog(nn,"replace failed");

	//
	// now we have to rewire all the downstream...
	//
	{
		// make a table of the new mapping, one for each output of the
		// meta node
		struct input new_inrefs[12] = {
				{0,0}, {0,0},{0,0},		// these can stay as-is (lstmout gets old node_id)
				{lstm_in_nid,0}, {lstm_in_nid,1}, {lstm_in_nid,2},	// remap these to lstmin
				{concat_nid,0}, {concat_nid,1}, {concat_nid,2},		// concat result (if applicable)
				{fcon_nid,0}, {fcon_nid,1}, {fcon_nid,2}			// fcon result (if applicable)
		};
		res = change_multi_output_refs_table( nn, lstmout_node, lstm_out_nid, n_out,	new_inrefs );
		if( res < 0)return errlog(nn,"failed to remap refs");
	}

	return 0;
}

#endif //NN_GRAPH_WITH_LENS_LSTM
