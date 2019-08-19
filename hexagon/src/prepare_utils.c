
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
/*
 */
#include <nn_prepare.h>

const struct output Output_ScalarFloat = {
	.rank = 4,
	.max_sizes = {1, 1, 1, 1},
	.elementsize = 4
};

void __attribute__((noinline))
make_outputdesc_from_shape(struct output *outp, struct shape const *shp, int elsize, int add_d32_padding_unused)
{
	outp->rank = 4;
	outp->max_sizes[0] = shp->batches;
	outp->max_sizes[1] = shp->height;
	outp->max_sizes[2] = shp->width;
	outp->max_sizes[3] = shp->depth;
	for (int i = 4; i < (int)(sizeof(outp->max_sizes) / sizeof(outp->max_sizes[0])); i++)
	{
		outp->max_sizes[i] = 0;
	}
	outp->elementsize = elsize;
	outp->zero_offset = 0;
	outp->stepsize = 0.0f;
	//if( add_d32_padding ) output_add_d32_padding(outp);
}

void __attribute__((noinline))
shape_from_outputdesc(struct shape *shp, struct output const *outp)
{
	shp->batches = outp->max_sizes[0];
	shp->height = outp->max_sizes[1];
	shp->width = outp->max_sizes[2];
	shp->depth = outp->max_sizes[3];
	//if( add_d32_padding) shape_add_d32_padding(shp);
}

struct nn_node *create_node(
	struct nn_graph *nn,
	uint32_t node_id,			// uses this node_id; or assigns one, if this is 0.
	uint32_t operation,
	padding_type padding,
	int num_inputs,
	int num_outputs,
	struct input *input_refs,
	struct output *output_defs)
{
	if (operation >= NN_OPS_MAX)
		return NULL;
	if( node_id == 0)
		node_id = nn_graph_new_internal_node_id(nn);
	return optab[operation]->ctor(nn, node_id, operation, padding, num_inputs, num_outputs, input_refs, output_defs);
}
//
// create a conversion.
// 'operation' must be one of the to/from _d32 nodes.
//
struct nn_node *create_convert(
	struct nn_graph *nn,
	int src_id,
	int output_idx,
	struct shape const * outsize,
	uint32_t operation)
{
	struct input inp = {
		.src_id = src_id,
		.output_idx = output_idx,
	};
	int elbytes = 1;
	switch( operation){
		case OP_Convert_from_d32:
		case OP_Convert_to_d32:
			break;
		case OP_Convert_from_d32_16b:
		case OP_Convert_to_d32_16b:
			elbytes = 2;
			break;
		default:
			errlog(nn,"Bad op for convert");
			return  NULL;

	}

	struct output outp;
	make_outputdesc_from_shape(&outp, outsize, /*elsize=*/elbytes, /*pad_d32=*/0);
	struct nn_node *new_node = create_node(nn, 0, operation, NN_PAD_NA, 1, 1, &inp, &outp);
	return new_node;
}

struct nn_node *__attribute__((noinline))
find_node_must_be_Const(struct nn_graph *nn, uint32_t node_id)
{
	return find_node_must_be(nn, node_id, OP_Const);
}

extern struct nn_node *find_node_must_be(struct nn_graph *nn, uint32_t node_id, op_type ntype);

struct nn_node *__attribute__((noinline))
find_node_must_be_Const_from_ref(struct nn_graph *nn, struct input const *iref)
{
	if (iref->output_idx != 0)
		return NULL;
	return find_node_must_be(nn, iref->src_id, OP_Const);
}

void __attribute__((cold)) free_node_array(struct nn_node **node_list, uint32_t num_nodes)
{
	for (uint32_t i = 0; i < num_nodes; i++)
	{
		if (node_list[i] != NULL)
			nn_free(node_list[i]);
	}
}



//////////////////////////////////////////////////////////////////////////
// This is used to 'clean' up supernodes which have had input #12 'channelscale'
// added to them, but did not convert to d32 for some reason.
// Generally it will rescale the input weights to emulate the channel scaing, and
// will replace input #12 with a scalar '1.0'.
// Caller must establish that the nodep refers to a supernode type that can't support
// channel scale (but has an optional dummy slot for it, since the d32 version can).
//
// if the node only has 12 inputs, nothing happens. Otherwise we expect:
//
//  input 1 -> const weight tensor
//  input 4,5-> consts, weight tensor min/max
//  input 12 -> const, scale array, floats, [1,1,1,d].
//
//  The weight tensor must be shape [.., d] or [ ..., d,1]
// (if d=1, this is a special case; any weight tensor is allowed, and
// we rescale the weights via their min/max, or do nothing if the scale is 1.0. Thus
// the operation is idempotent.
//
//  The weights are scaled in-place if the weight const has no other use, otherwise
// a new one is made.
//
int
handle_channelscaled_supernode( struct nn_graph *nn, struct nn_node *nodep)
{
	if( nodep->n_inputs < 13) return 0;			// nothing to do

	struct nn_node const * wts_node = find_node_must_be_Const_from_ref( nn, &nodep->input_refs[1]);
	struct nn_node const * min_node = find_node_must_be_Const_from_ref( nn, &nodep->input_refs[4]);
	struct nn_node const * max_node = find_node_must_be_Const_from_ref( nn, &nodep->input_refs[5]);
	struct nn_node const * scale_node = find_node_must_be_Const_from_ref( nn, &nodep->input_refs[12]);

	// all must be const.
	if( wts_node == NULL || min_node == NULL || max_node == NULL || scale_node == NULL)
	{
		return errlog(nn,"node 0x%X has ChannelScale but non-const inputs", (unsigned)nodep->node_id);
	}
	float wmin =  tensor_get_float( min_node->outputs[0], 0);
	float wmax =  tensor_get_float( max_node->outputs[0], 0);

	if( !(wmin <= 0.0f &&  wmin < wmax && wmax >= 0.0f)){
		return errlog(nn,"node has bad weight range %.6g .. %.6g", wmin, wmax);
	}
	struct tensor const * scale_tensor = scale_node->outputs[0];
	int depth = scale_tensor->data_size/sizeof(float);
	if( scale_tensor->shape.depth != depth) return errlog(nn,"bad ChannelScale shape");

	float const * scalep = (float const*)scale_tensor->data;
	if (depth == 1){
		// looks like one float... easier to handle
		float scaleval = scalep[0];
		if( scaleval == 1.0f) return 0;					// no change needed
		if( !(scaleval > 1e-6f && scaleval < 1e6f)){		// sanity check (and Nan)
			return errlog(nn,"?? won't scale node weights by %.8g", scaleval);
		}
		unsigned new_min_nodeid = create_const_float_op( nn, scaleval * wmin);
		unsigned new_max_nodeid = create_const_float_op( nn, scaleval * wmax);
		unsigned new_one_nodeid = create_const_float_op( nn, 1.0f);
		if( new_min_nodeid ==0 || new_max_nodeid == 0 || new_one_nodeid == 0) return -1;
		nodep->input_refs[4].src_id = new_min_nodeid;
		nodep->input_refs[5].src_id = new_max_nodeid;
		nodep->input_refs[12].src_id = new_one_nodeid;
		node_rehash_inputrefs(nodep);
		return 0;
	}
	struct tensor const * wts_tensor = wts_node->outputs[0];
	if( wts_tensor->shape.depth != depth && !(wts_tensor->shape.depth == 1 && wts_tensor->shape.width == depth )){
		// not [...,d] and not [ ...,d,1]
		return errlog(nn,"node 0x%X: weight shape does not match channelscale depth=%d", (unsigned)nodep->node_id, depth );
	}
	unsigned wts_elements = tensor_element_count( wts_tensor);
	unsigned new_wts_nid = 0;
	uint8_t * wrtp = (uint8_t*)wts_tensor->data;
	uint8_t const * rdp = wrtp;

	if( check_single_consumer_all(nn,wts_node, nodep)!=0){	// we are not the only consumer
		new_wts_nid = nn_graph_new_internal_node_id(nn);
		struct nn_node * new_nodep = NULL;
		int k = -1;
		if(new_wts_nid != 0)
		  k = do_prepend_const_node( nn, new_wts_nid,
				wts_tensor->shape.batches, wts_tensor->shape.height, wts_tensor->shape.width, wts_tensor->shape.depth,
				NULL, wts_elements * sizeof(uint8_t));	// create uninitialized
		if( k==0)
			new_nodep = find_node_must_be_Const( nn, new_wts_nid);

		if( new_nodep==NULL) return errlog(nn,"can't allocate new node");
		// will store output to new node
		wrtp = (uint8_t*)new_nodep->outputs[0]->data;
	}
	// now, rescale,reading from *rdp,write to *wrtp.
	int rescale_rows = wts_elements/depth;			// # of 'rows' to rescale
	// we need to find the 'zero' of the weight values
	int wt_zero = saturate_u8( roundf_i32( -wmin*255.0f/(wmax-wmin)));

	// make an array [depth] of uint16 to hold scale factors.
	//
	int need_alloc = depth> 512;
	uint64_t tmparr_arr[need_alloc? 1: (depth+3)/4];
	uint64_t *tmparr = &tmparr_arr[0];
	if( need_alloc) {
		tmparr = nn_malloc( depth * sizeof(uint16_t));
		if(tmparr == NULL) return errlog(nn,"alloc failed");
	}
	uint16_t * sclarr = (uint16_t *)tmparr;
	// the values in the array (are 1.0-scale)* 32768 rounded, and must be in range 0..32767
	for( int i = 0; i < depth;i++){
		float scl = scalep[i];
		int qval = 32768 - roundf_i32(scl*32768.0f);
		if( qval < 0 || qval > 32767){
			logmsg(nn,0,"bad value for channelscale[%d]= %f",i, scl);
			qval = (qval < 0)? 0: 32767;
		}
		sclarr[i] = qval;
	}

	if(depth%4u != 0){
		for(int i = 0; i < rescale_rows; i++){
			for( int j = 0; j < depth; j++){
				int inval = *rdp++;
				int delt =  (inval-wt_zero);		// range -255 .. 255
				delt = (delt * sclarr[j] + 16384)>>15;	// adjustment to make
				*wrtp++ = saturate_u8(  inval - delt );
			}
		}
	}else{
		// do 4 at a time
		uint32_t const * rp32 = (uint32_t const *)rdp;
		uint32_t *  wp32 = (uint32_t  *)wrtp;
		uint64_t const *  sp64 = (uint64_t const *)sclarr;
		uint64_t wz_4h = Q6_P_vzxtbh_R( Q6_R_vsplatb_R(wt_zero));

		for( int  i=0; i <rescale_rows; i++){
			for(int j =0; j < depth/4u; j++){
				uint64_t invals_h = Q6_P_vzxtbh_R(*rp32++);	//zero extend
				uint64_t scales_h = sp64[j];
				uint64_t deltas_h = Q6_P_vsubh_PP(invals_h,wz_4h);		/// in[i]-wt_zero;
				uint64_t prod01 = Q6_P_vmpyh_RR_sat( (uint32_t)deltas_h, (uint32_t)scales_h );
				uint64_t prod23 = Q6_P_vmpyh_RR_sat( (uint32_t)(deltas_h>>32), (uint32_t)(scales_h>>32) );
				prod01 = Q6_P_vasrw_PI( prod01,14);
				prod23 = Q6_P_vasrw_PI( prod23,14);
				// need >>1 more with rounding but can pack to h now
				uint64_t prod_h = Q6_P_combine_RR( Q6_R_combine_RlRl( prod23>>32, prod23),
						Q6_R_combine_RlRl( prod01>>32, prod01));
				prod_h = Q6_P_vasrh_PI_rnd( prod_h, 1);		// that is delta to subtract from original
				uint32_t result_b = Q6_R_vsathub_P( Q6_P_vsubh_PP(invals_h, prod_h));
				*wp32 ++ = result_b;
			}
		}
	}

	if(need_alloc){
		nn_free(tmparr);
	}
	// just about done.
	// install new weights - if applicable - and new input 12
	unsigned new_one_nodeid = create_const_float_op( nn, 1.0f);
	if( new_one_nodeid == 0) return -1;

	if( new_wts_nid != 0 ){
		nodep->input_refs[1].src_id = new_wts_nid;
	}
	nodep->input_refs[12].src_id = new_one_nodeid;
	node_rehash_inputrefs(nodep);
	return 0;
}

//////////////////////////////////////////////////////////////////////////

// handle supernodes which have filters so large that d32 may not provide enough padding
// space for them.
// it is assumed that oversize_d32_supernode_quick_check(nn, *nodep ) has already returned 1.
//
int handle_oversize_d32_supernode( struct nn_graph *nn,  struct nn_node ** nodep)
{
	struct nn_node * conv_node = *nodep;
	// get the 'weights' tensor.
	struct nn_node const * wts_node = find_node_must_be_Const_from_ref( nn, &conv_node->input_refs[1]);
	if( wts_node == NULL) return 0;
	struct shape const *wts_shape = &wts_node->outputs[0]->shape;
	int fht = wts_shape->filt_height;
	int fwid = wts_shape->filt_width;
	if( fht < 1 || fwid < 1 || (fht==1 && fwid ==1)) return 0;
	// below are the only supported padding types; do nothing if the
	// padding type is not supported.
	// (VALID is also supported, but we do nothing for that)
	// we want to find the worst-case padding needed for this filter size, for any input size.
	//
	int pad = conv_node->padding;
	int pad_adj=1;
	switch( pad){
	 case NN_PAD_NA:
		 break;
	 case NN_PAD_SAME:
		 break;
	 case NN_PAD_SAME_CAFFE:
		 pad_adj = 0;
		 break;
	 default:
		 return 0;
	}
	// for NN_PAD_NA, we need to look at the stride, and effectively reduce the filter size by stride-1.
	if( pad == NN_PAD_NA){
		struct nn_node const * stride_node = find_node_must_be_Const_from_ref( nn, &conv_node->input_refs[6]);
		if(stride_node == NULL) return 0;
		struct shape const *stride_shape = &stride_node->outputs[0]->shape;
		int str_h = stride_shape->height;
		int str_w = stride_shape->width;
		fht = (fht > str_h)? (fht+1-str_h) : 1;
		fwid = (fwid > str_w)? (fwid+1-str_w) : 1;
	}
	// all of these should be >=0, since the fht and fwid are >=1.
	int need_top = (fht-pad_adj)>>1;
	int need_bottom = fht-1-need_top;
	int need_left = (fwid-pad_adj)>>1;
	int need_right = fwid-1-need_left;

	// conditions for 'at-risk' supernode:
	//      need_top or need_bottom : either > 4
	//      need_left or need_right : either > 3
	//      need_right > 0 and need_bottom >3
	//
	if( ! (  max_i32(need_top, need_bottom)> 4
		  || max_i32(need_left, need_right)> 3
		  || (need_right >0 && need_bottom > 3)))
		return 0;

	int tmp = conv_node->input_refs->output_idx;
	struct nn_node *upstream = find_node( nn, conv_node->input_refs[0].src_id);
	if( upstream ==NULL || upstream->n_outputs <= tmp) return 0;

	logmsg(nn,3,"Replacing supernode 0x%X with PadForConv + Supernode: padding = %d %d %d %d\n",
			(unsigned)conv_node->node_id, need_top, need_bottom, need_left, need_right );
	struct nn_node * qfc_node = NULL;
	struct nn_node *newconv_node = NULL;
	// OK it needs conversion.
	// (1) make QuantizedPadForConv
	// inputs are :  data, min,max; filter, stride
	// it has the same 'padding' as the conv we are replacing.
	{
		struct input const * convin = conv_node->input_refs;
		struct input qfc_in[5] = {
				convin[0], convin[2], convin[3],	// data, min, max
				convin[1], convin[6]				// weights, stride
		};
		struct output qfc_out[3] = {
				upstream->output_defs[tmp],
				Output_ScalarFloat,
				Output_ScalarFloat
		};
		// pad the sizes with worst-case padding
		qfc_out[0].max_sizes[1] += need_top + need_bottom;
		qfc_out[0].max_sizes[2] += need_left + need_right;

		int pad_for_conv_op = (qfc_out[0].elementsize ==2)? OP_QuantizedPadForConv_u16_d32 : OP_QuantizedPadForConv_8_d32;

		qfc_node = create_node( nn, 0, pad_for_conv_op, pad, 5, 3, qfc_in, qfc_out);
		if( qfc_node == NULL) return errlog(nn,"can't make QuantizedPadForConv node!");
	}
	// (2) make a new conv: same as original - same node id - but with VALID padding and with
	//   input from QuantizedPadForConv node.
	//
	{
		int n_in = conv_node->n_inputs;
		struct input newconvin[n_in];
		memcpy( newconvin, conv_node->input_refs, n_in*sizeof(struct input));
		// replace inputs 0,2,3:
		unsigned qfc_nid = qfc_node->node_id;
		newconvin[0].src_id = qfc_nid;
		newconvin[0].output_idx = 0;
		newconvin[2].src_id = qfc_nid;
		newconvin[2].output_idx = 1;
		newconvin[3].src_id = qfc_nid;
		newconvin[3].output_idx = 2;
		newconv_node = create_node( nn, conv_node->node_id,conv_node->node_type, NN_PAD_VALID, n_in, 3, newconvin, conv_node->output_defs);
		if( newconv_node == NULL) {
			qfc_node->ops->dtor( qfc_node, nn);
			return errlog(nn,"can't make new supernode!");
		}
	}

	// replace one node with two, and we are done.
	if( replace_node_with(nn,nodep, conv_node, qfc_node, newconv_node) < 0)
		return errlog(nn,"replace with QuantizedPadForConv failed");

	return 0;
}

