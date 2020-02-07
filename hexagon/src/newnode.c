
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
 * This contains the code to append a node.
 */

#include <nn_graph.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn_string_map.h"
#include "udo_impl_dsp_hexnn_internal_v2.h"
#include "SnpeUdo/UdoFlatten.h"

const char *TypeStrings[] = {
        "void",
        "qint8",
        "quint8",
        "int8",
        "uint8",
        "qint16",
        "quint16",
        "int32",
        "float32"
};


struct nn_node *alloc_node(uint32_t node_id, 
	op_type operation, padding_type padding)
{
	struct nn_node *newnode;
	if ((newnode = nn_malloc(sizeof(*newnode))) == NULL) {
		return newnode;
	}
	newnode->node_id = node_id;
	newnode->ops = optab[operation];
	newnode->node_type = operation;
	newnode->padding = padding;
	newnode->perfcounter = 0;
	newnode->executions = 0;
	newnode->opaque = NULL;
	newnode->flags = 0;
        newnode->refs = 0;
        newnode->iter_cycles = 0;
        newnode->executions = 0;

	return newnode;
}


int udo_common_execute (struct nn_node *node, struct nn_graph *nn) {
	SnpeUdo_ExternalNotify_t temp = NULL;
        int n_hexnn_tenosrs_per_udo_tensor_tf = 3;
        int n_hexnn_tenosr_per_udo_tensor_nonquant = 1;
        SnpeUdo_QuantizationType_t* q_types = (SnpeUdo_QuantizationType_t*)(node->udo_info).udo_input_q_types;
        int hexnn_in_tensor_ind = 0;
        if (check_udo_library_existence((node->udo_info).udo_lib_id) != 0) {
                return UDO_EXE_LIB_NOT_REGISTERED_INTERNAL;
        }
        for (int i=0; i<(node->udo_info).udo_num_input_tensors; i++) {
                if (q_types[i] == SNPE_UDO_QUANTIZATION_TF) {
                        (((((SnpeUdo_TensorParam_t*)((node->udo_info).udo_input_tensors))[i]).quantizeParams).TFParams).minValue = tensor_get_float((node->inputs)[hexnn_in_tensor_ind+1], 0);
                        (((((SnpeUdo_TensorParam_t*)((node->udo_info).udo_input_tensors))[i]).quantizeParams).TFParams).maxValue = tensor_get_float((node->inputs)[hexnn_in_tensor_ind+2], 0);
                        hexnn_in_tensor_ind += n_hexnn_tenosrs_per_udo_tensor_tf;
                } else if (q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        hexnn_in_tensor_ind += n_hexnn_tenosr_per_udo_tensor_nonquant;
                } else {
                        return UDO_EXE_INVALID_INPUTS_OUTPUTS_QUANTIZATION_TYPE_INTERNAL;
                }
        }
        q_types = (SnpeUdo_QuantizationType_t*)(node->udo_info).udo_output_q_types;
        int hexnn_out_tensor_ind = 0;
        for (int i=0; i<(node->udo_info).udo_num_output_tensors; i++) {
                if (q_types[i] == SNPE_UDO_QUANTIZATION_TF || q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        uint32_t ele_size = ((node->output_defs)[hexnn_out_tensor_ind]).elementsize;
                        SnpeUdo_DataType_t out_data_type = SNPE_UDO_DATATYPE_FIXED_8;
                        switch(ele_size) {
                                case 1: out_data_type = SNPE_UDO_DATATYPE_FIXED_8; break;
                                case 2: out_data_type = SNPE_UDO_DATATYPE_FIXED_16; break;
                                case 4: out_data_type = SNPE_UDO_DATATYPE_FIXED_32; break;
                        }
                        (((SnpeUdo_TensorParam_t*)((node->udo_info).udo_output_tensors))[i]).dataType = out_data_type;
                }
                if (q_types[i] == SNPE_UDO_QUANTIZATION_TF) {
                        (((((SnpeUdo_TensorParam_t*)((node->udo_info).udo_output_tensors))[i]).quantizeParams).TFParams).minValue = tensor_get_float((node->inputs)[hexnn_in_tensor_ind], 0);
                        (((((SnpeUdo_TensorParam_t*)((node->udo_info).udo_output_tensors))[i]).quantizeParams).TFParams).maxValue = tensor_get_float((node->inputs)[hexnn_in_tensor_ind+1], 0);

                        tensor_copy((node->outputs)[hexnn_out_tensor_ind+1], (node->inputs)[hexnn_in_tensor_ind]); 
                        tensor_copy((node->outputs)[hexnn_out_tensor_ind+2], (node->inputs)[hexnn_in_tensor_ind+1]);

                        hexnn_in_tensor_ind += 2;
                        hexnn_out_tensor_ind += n_hexnn_tenosrs_per_udo_tensor_tf;
                } else if (q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        hexnn_out_tensor_ind += n_hexnn_tenosr_per_udo_tensor_nonquant;
                } else {
                        return UDO_EXE_INVALID_INPUTS_OUTPUTS_QUANTIZATION_TYPE_INTERNAL;
                }
        }
        if (check_udo_library_existence((node->udo_info).udo_lib_id) != 0) {
                return UDO_EXE_LIB_NOT_REGISTERED_INTERNAL;
        }
	SnpeUdo_ErrorType_t e = (*((SnpeUdo_ExecuteOpFunction_t)((node->udo_info).udo_exe))) ((node->udo_info).udo_operation, true, 0, temp);

	return (int)e;
}


struct nn_node* alloc_udo_node (struct nn_graph *nn, uint32_t node_id, uint32_t num_inputs, uint32_t num_outputs, uint32_t ops_flag) {
        struct nn_node* newnode = nn_malloc(sizeof(struct nn_node));
        if (newnode == NULL) {
                return NULL;
        }
        struct nn_node_ops* udo_ops = nn_malloc(sizeof(struct nn_node_ops));
        if (udo_ops == NULL) {
                nn_free(newnode);
                return NULL;
        }
        graph_node* udo_op_inf = nn_malloc(sizeof(graph_node));
        if (udo_op_inf == NULL) {
                nn_free(udo_ops);
                nn_free(newnode);
                return NULL;
        }

        newnode->node_id = node_id;
        newnode->node_type = NN_OPS_MAX;
        newnode->ops = udo_ops;
        newnode->padding = NN_PAD_NA;
        newnode->refs = 0;
        newnode->iter_cycles = 0;
        newnode->executions = 0;
        udo_ops->execute = udo_common_execute;
        udo_ops->check = NULL;
        udo_ops->ctor = NULL;
        udo_ops->dtor = node_free_udo_common_release;
        udo_ops->padding_hint = NULL;
        udo_ops->flags = ops_flag;
        udo_ops->earlywork_note_pred = NULL;
        udo_ops->earlywork_register = NULL;
        (newnode->udo_info).udo_op_factory = NULL;
        (newnode->udo_info).udo_operation = NULL;
        (newnode->udo_info).udo_release_op = NULL;
        (newnode->udo_info).udo_release_op_factory = NULL;
        // assumes append_udo_node n inputs/outputs match op implementation and n inputs/outputs cannot vary
        struct nn_node_io_range in_r = {(uint8_t)num_inputs,(uint8_t)(num_inputs+1)};
        struct nn_node_io_range out_r = {(uint8_t)num_outputs,(uint8_t)(num_outputs+1)};
        udo_ops->n_inputs = in_r;
        udo_ops->n_outputs = out_r;
        newnode->perfcounter = 0;
        newnode->executions = 0;
        newnode->opaque = NULL;
        newnode->flags = 0;
        udo_op_inf->node = newnode;
        udo_op_inf->graph = nn;
        (newnode->udo_info).udo_op_infra = udo_op_inf;
        (newnode->udo_info).udo_input_tensors = NULL;
        (newnode->udo_info).udo_num_input_tensors = 0;
        (newnode->udo_info).udo_output_tensors = NULL;
        (newnode->udo_info).udo_num_output_tensors = 0;
        (newnode->udo_info).udo_added_d32_converts = 0;

        return newnode;
}


static inline void free_inputs(struct nn_node *node)
{
	if (node->inputs) nn_free(node->inputs);
	if (node->input_refs) nn_free(node->input_refs);
}

const struct nn_node *get_node(struct nn_graph *nn, uint32_t id) {
	const struct nn_node *node = nn->head;
	while (node) {
		if (node->node_id == id) {
			return node;
		}
		node = node->next;
	}
	return NULL;
}
static inline int alloc_inputs(
	struct nn_graph *nn,
	struct nn_node *newnode, 
	uint32_t n, 
	const struct input *inputs)
{
	unsigned int tmpsize;
	int i;
	newnode->n_inputs = n;
	newnode->inputs = NULL;
	newnode->input_refs = NULL;
	if (n == 0) {
		return 0;
	}
	tmpsize = n*sizeof(newnode->input_refs[0]);
	/* allocate inputs */
	if ((newnode->input_refs = nn_calloc(1,tmpsize)) == NULL) {
		return errlog(nn,"input refs alloc failed");
	}
	if ((newnode->inputs = nn_calloc(n,sizeof(void *))) == NULL) {
		nn_free(newnode->input_refs);
		return errlog(nn,"input ptr storage alloc failed");
	}

	/* Copy input refs */
	for (i = 0; i < n; i++) {
		if (inputs[i].src_id == 0) {
			/* Or we could handle and dup tensor here */
			nn_free(newnode->input_refs);
			nn_free(newnode->inputs);
			return errlog(nn,"fatal: const tensor in generic input");
		}
		newnode->input_refs[i] = inputs[i];
		// Copy the shape from source to this input
		const struct nn_node *source_node = get_node(nn, inputs[i].src_id);
		if (source_node) {
			const struct tensor *source_output = source_node->outputs[inputs[i].output_idx];
			newnode->inputs[i] = source_output;
		}
	}
	node_rehash_inputrefs(newnode);
	return 0;
}

static inline void free_outputs(struct nn_node *node)
{
	int i;
	for (i = 0; i < node->n_outputs; i++) {
		node->outputs[i]->data = NULL;
		tensor_free(node->outputs[i]);
	}
	if (node->outputs) nn_free(node->outputs);
	if (node->output_defs) nn_free(node->output_defs);
}


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

static inline int alloc_outputs(
	struct nn_graph *nn,
	struct nn_node *newnode, 
	uint32_t n, 
	const struct output *outputs)
{
	int i;
	struct shape tshape;
	tshape.depth = tshape.width = tshape.height = tshape.batches = 0;
	newnode->n_outputs = n;
	if (n == 0) {
		newnode->outputs = NULL;
		newnode->output_defs = NULL;
		return 0;
	}
	/* Allocate outputs */
	if ((newnode->outputs = nn_calloc(n,sizeof(void *))) == NULL) {
		return errlog(nn,"output ptr storage alloc failed");
	}
	if ((newnode->output_defs = nn_calloc(n,sizeof(struct output))) == NULL) {
		nn_free(newnode->outputs);
		return errlog(nn,"output def storage alloc failed");
	}
	memcpy(newnode->output_defs,outputs,n*sizeof(struct output));
	/* Allocate outputs */
	/*
	 * Allocate base tensor struct but don't allocate storage until later.
	 * We could postpone longer, but this works pretty well.
	 */
	for (i = 0; i < n; i++) {
		struct tensor * out_tensor = tensor_alloc(&tshape,0);
		newnode->outputs[i] = out_tensor;
		if ( out_tensor == NULL) {
			goto err_free_allocated_outputs;
		}
		/* EJP: now unnecessary, we keep output shape definitions separate */
		uint32_t rank = outputs[i].rank;
		uint32_t elsize = outputs[i].elementsize;
		uint32_t max_size = elsize;
		if( rank > MAX_DIMENSIONS){
			logmsg(nn,0,"rank %u tensor?", (unsigned)rank );
			rank = MAX_DIMENSIONS;
		}else{
			// fill 'garbage' entries in max_sizes with 1
			struct output *new_odef = &newnode->output_defs[i];
			for( int j = rank; j < MAX_DIMENSIONS; j++){
				new_odef->max_sizes[j] = 1;
			}
		}

		out_tensor->shape.batches = (rank < 4) ? 1 : outputs[i].max_sizes[rank-4];
		out_tensor->shape.height =  (rank < 3) ? 1 : outputs[i].max_sizes[rank-3];
		out_tensor->shape.width =   (rank < 2) ? 1 : outputs[i].max_sizes[rank-2];
		out_tensor->shape.depth =   (rank < 1) ? 1 : outputs[i].max_sizes[rank-1];
		max_size = compute_max_size_with_pad(nn,newnode,&out_tensor->shape,elsize,i);
#if 0
		if( elsize > 0){
			for (uint32_t j=0; j< rank; j++) {
				max_size= mulu32_sat( max_size, outputs[i].max_sizes[j]);
			}
			if( (int32_t)max_size <= 0) {	// 0, -ve or insane dims
				errlog(nn,"output calls for %u bytes, of elsize = %u", (unsigned) max_size, (unsigned)elsize);
				goto err_free_allocated_outputs;
			}
			if (newnode->ops->flags & NN_NODE_FLAG_D32_OUTPUT) max_size *= 2;
		}
#endif
		out_tensor->max_size = max_size;
		if (max_size > 100*1024*1024) {
			logmsg(nn,3,"Node %p id %x has large output #%d of %d bytes: max sizes [rank %d] %d,%d,%d,%d shape %d,%d,%d,%d elsize %d",
				newnode,
				newnode->node_id,
				i,
				max_size,
				rank,
				outputs[i].max_sizes[0],
				outputs[i].max_sizes[1],
				outputs[i].max_sizes[2],
				outputs[i].max_sizes[3],
				out_tensor->shape.batches,
				out_tensor->shape.height,
				out_tensor->shape.width,
				out_tensor->shape.depth,
				elsize);
		}
	}
	return 0;
err_free_allocated_outputs:
	for (i = 0; i < n; i++) {
		if (newnode->outputs[i]) tensor_free(newnode->outputs[i]);
	}
	nn_free(newnode->outputs);
	nn_free(newnode->output_defs);
	return errlog(nn,"output tensor malloc failed");
}



int node_alloc_common_helper(
	struct nn_graph *nn,
	uint32_t node_id,
        struct nn_node* newnode,
	op_type operation,
        const char* op_type_print_alt,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{

	// check input & output counts.
	//  must be >= min_ports
	//  must be < max1_ports (or max1_ports =0)
	{
		struct nn_node_ops const * ops = newnode->ops;
		if( num_inputs < ops->n_inputs.min_ports ||
				(ops->n_inputs.max1_ports!=0 && num_inputs >= ops->n_inputs.max1_ports)){
			errlog(nn,"node %X (%s): bad input count %d",
					(unsigned)node_id, op_type_to_string_alt(operation,op_type_print_alt), (int)num_inputs);
                        return -1;
		}
		if( num_outputs < ops->n_outputs.min_ports ||
				(ops->n_outputs.max1_ports!=0 && num_outputs >= ops->n_outputs.max1_ports)){
			errlog(nn,"node %X (%s): bad output count %d",
					(unsigned)node_id, op_type_to_string_alt(operation,op_type_print_alt), (int)num_outputs);
                        return -1;
		}
	}
	if (alloc_inputs(nn, newnode, num_inputs, inputs) != 0) {
		errlog(nn,"input alloc failed");
                return -1;
	}
	if (alloc_outputs(nn, newnode, num_outputs, outputs) != 0) {
		errlog(nn,"output alloc failed");
	        free_inputs(newnode);
                return -1;
	}
	if( num_outputs == 0 ){
		newnode->flags |= NN_NODE_FLAG_RETAIN;
	}
	return 0;
}


struct nn_node *node_alloc_common(
        struct nn_graph *nn,
        uint32_t node_id,
        op_type operation,
        padding_type padding,
        uint32_t num_inputs,
        uint32_t num_outputs,
        const struct input *inputs,
        const struct output *outputs)
{
        struct nn_node *newnode;
        if ((newnode = alloc_node(node_id,operation,padding)) == NULL) {
                errlog(nn,"common alloc id %x malloc fail",node_id);
                return NULL;
        }

        if (node_alloc_common_helper(nn, node_id, newnode, operation, "??", num_inputs, num_outputs, inputs, outputs) == -1) {
                nn_free(newnode);
                return NULL;
        }
        return newnode;

}

struct nn_node *node_alloc_udo_common(
        struct nn_graph *nn,
        uint32_t node_id,
        uint32_t num_inputs,
        uint32_t num_outputs,
        const struct input *inputs,
        const struct output *outputs,
        uint32_t ops_flag)
{
        struct nn_node *newnode;
        if ((newnode = alloc_udo_node(nn, node_id,num_inputs,num_outputs, ops_flag)) == NULL) {
                errlog(nn,"udo common alloc id %x malloc fail",node_id);
                return NULL;
        }

        if (node_alloc_common_helper(nn, node_id, newnode, NN_OPS_MAX, "udo", num_inputs, num_outputs, inputs, outputs) == -1) {
                nn_free((newnode->udo_info).udo_op_infra);
                nn_free(newnode->ops);
                nn_free(newnode);
                return NULL;
        }
        return newnode;

}


// this sets the noderefhash field on a node.
// Call after changing src_id
void node_rehash_inputrefs( struct nn_node * node)
{
	noderefhash_set_t hashall = 0;
	uint32_t prev_nid = 0;
	for( int i = 0;  i < node->n_inputs; i++ ){
		uint32_t nid = node->input_refs[i].src_id;
		if( nid != prev_nid){
			hashall |= noderefhash_mask(nid);
			prev_nid = nid;
		}
	}
	node->noderefhash = hashall;
}
int node_free_common(struct nn_node *node, struct nn_graph *nn)
{
	logmsg(nn,3,"freeing node %p id=%x",node,node->node_id);
	free_inputs(node);
	free_outputs(node);
	del_node_from_hash(nn,node->node_id,node);
	nn_free(node);
	return 0;
}
//
// if you have a (possibly null) opaque pointer which just needs to be free'd, this
// can be the node dtor.
int node_free_common_release_opaque(struct nn_node *node, struct nn_graph *nn )
{
	if( node->opaque != NULL ){
		nn_free(node->opaque);
		node->opaque = NULL;
	}
	return node_free_common( node, nn);
}


int node_free_udo_common_release(struct nn_node *node, struct nn_graph *nn )
{
        int release_op_err = 0;
        if (check_udo_library_existence((node->udo_info).udo_lib_id) == 0) {
                if((node->udo_info).udo_release_op && (node->udo_info).udo_operation) {
	                if(((*((SnpeUdo_ReleaseOpFunction_t)((node->udo_info).udo_release_op))) ((node->udo_info).udo_operation))!=SNPE_UDO_NO_ERROR) {
                                errlog(nn, "udo node id=0x%x release operation function failed", node->node_id);
                                release_op_err = 1;
                        }
                        (node->udo_info).udo_operation = NULL;
                }
        }
        nn_free((node->udo_info).udo_op_infra);
        (node->udo_info).udo_op_infra = NULL;
        if ((node->udo_info).udo_input_tensors)  nn_free((node->udo_info).udo_input_tensors);
        (node->udo_info).udo_input_tensors = NULL;
        if ((node->udo_info).udo_output_tensors)  nn_free((node->udo_info).udo_output_tensors);
        (node->udo_info).udo_output_tensors = NULL;
        nn_free(node->ops);
        node->ops = NULL;
        int e = node_free_common(node, nn);
        if (release_op_err) {
                return -1;
        } else {
                return e;
        }
}

int free_unflattened_udo_parameters (int num_params, SnpeUdo_Param_t* unflattened_params) 
{
        if(unflattened_params) {
                for(int i=0; i<num_params; i++) {
                        if(unflattened_params[i].paramName) {
                                nn_free(unflattened_params[i].paramName);
                                unflattened_params[i].paramName = NULL;
                        }
                        if((unflattened_params[i].paramType) == SNPE_UDO_PARAMTYPE_TENSOR) {
                                if((unflattened_params[i].tensorParam).maxDimensions) {
                                        nn_free((unflattened_params[i].tensorParam).maxDimensions);
                                        (unflattened_params[i].tensorParam).maxDimensions = NULL;
                                }
                                if((unflattened_params[i].tensorParam).currDimensions) {
                                        nn_free((unflattened_params[i].tensorParam).currDimensions);
                                        (unflattened_params[i].tensorParam).currDimensions = NULL;
                                }
                                if((unflattened_params[i].tensorParam).tensorData) {
                                        nn_free((unflattened_params[i].tensorParam).tensorData);
                                        (unflattened_params[i].tensorParam).tensorData = NULL;
                                }
                        } else if((unflattened_params[i].paramType) == SNPE_UDO_PARAMTYPE_STRING) {
                                if(unflattened_params[i].stringParam) {
                                        nn_free(unflattened_params[i].stringParam);
                                        unflattened_params[i].stringParam = NULL;
                                }
                        }
                }
        }
        return 0;
}

int unflatten_udo_parameters(void* flattened, uint32_t size_flattened, uint32_t* num_params, SnpeUdo_Param_t** unflattened_params)
{
        dspStaticParams_t* flattened_params = flattened;
        *num_params = (flattened_params->meta).numParams;
        if((*num_params)==0 || size_flattened<=sizeof(dspStaticParamsMeta_t)) {
                *num_params = 0;
                *unflattened_params = NULL;
                return 0;
        }
        *unflattened_params = nn_malloc((*num_params)*sizeof(SnpeUdo_Param_t));
        if (*unflattened_params == NULL)  return -1;
        dspStaticParamDescriptor_t* cur_desc;
        SnpeUdo_Param_t* cur_param;
        for(int i=0; i<(*num_params); i++) {
                if(i==0){
                        cur_desc = &(flattened_params->paramDesc);
                } else {
                        cur_desc = (dspStaticParamDescriptor_t*) (((uint8_t*)cur_desc)+cur_desc->size);
                }
                if(((uint8_t*)cur_desc) >= ((uint8_t*)flattened)+size_flattened) {  // sanity check
                        free_unflattened_udo_parameters(i, *unflattened_params);
                        nn_free(*unflattened_params);
                        *unflattened_params = NULL;
                        return -1; // errlog
                }
                cur_param = (*unflattened_params)+i;
                cur_param->paramType = cur_desc->paramType;
                if(cur_desc->paramType == SNPE_UDO_PARAMTYPE_SCALAR) {
                        (cur_param->scalarParam).dataType = (cur_desc->scalarInfo).dataType;
                        (cur_param->scalarParam).dataValue.floatValue = (cur_desc->scalarInfo).dataValue.floatValue;
                } else if(cur_desc->paramType == SNPE_UDO_PARAMTYPE_TENSOR) {
                        (cur_param->tensorParam).layout = (cur_desc->tensorInfo).layout;
                        (cur_param->tensorParam).quantizeParams = (cur_desc->tensorInfo).quantizeInfo;
                        (cur_param->tensorParam).dataType = (cur_desc->tensorInfo).dataType;
                }
                char* name_start = ((char*)cur_desc) + sizeof(dspStaticParamDescriptor_t);
                if((cur_desc->name).lengthString == 0 && (cur_desc->name).sizeStruct == sizeof(udoString_t)) {
                        cur_param->paramName = NULL;
                } else {
                        cur_param->paramName = nn_malloc((cur_desc->name).lengthString+1);
                        if (cur_param->paramName == NULL) {
                                free_unflattened_udo_parameters(i, *unflattened_params);
                                nn_free(*unflattened_params);
                                *unflattened_params = NULL;
                                return -1; // errlog
                        }
                        strncpy(cur_param->paramName, name_start, (cur_desc->name).lengthString+1);
                }
                if(cur_desc->paramType == SNPE_UDO_PARAMTYPE_TENSOR) {
                        if ((cur_param->tensorParam).layout == SNPE_UDO_LAYOUT_NULL) {
                                (cur_param->tensorParam).maxDimensions = NULL;
                                (cur_param->tensorParam).currDimensions = NULL;
                                (cur_param->tensorParam).tensorData = NULL;
                                (cur_param->tensorParam).tensorRank = 0;
                        } else {
                                dims_t* dims_start = (dims_t*)(name_start + (cur_desc->name).sizeStruct - sizeof(udoString_t));
                                uint32_t rank = dims_start->rank;
                                (cur_param->tensorParam).tensorRank = rank;
                                (cur_param->tensorParam).maxDimensions = nn_malloc(rank*sizeof(uint32_t));
                                if ((cur_param->tensorParam).maxDimensions == NULL) {
                                        if(cur_param->paramName)  nn_free(cur_param->paramName);
                                        free_unflattened_udo_parameters(i, *unflattened_params);
                                        nn_free(*unflattened_params);
                                        *unflattened_params = NULL;
                                        return -1; // errlog
                                }
                                (cur_param->tensorParam).currDimensions = nn_malloc(rank*sizeof(uint32_t));
                                if ((cur_param->tensorParam).currDimensions == NULL) {
                                        if(cur_param->paramName)  nn_free(cur_param->paramName);
                                        if((cur_param->tensorParam).maxDimensions)  nn_free((cur_param->tensorParam).maxDimensions);
                                        free_unflattened_udo_parameters(i, *unflattened_params);
                                        nn_free(*unflattened_params);
                                        *unflattened_params = NULL;
                                        return -1; // errlog
                                }
                                uint32_t* ds = &(dims_start->ds);
                                for(int j=0; j<rank; j++) {
                                        (cur_param->tensorParam).maxDimensions[j] = ds[j];
                                        (cur_param->tensorParam).currDimensions[j] = ds[j+rank];
                                }
                                tensorData_t* data_start = (tensorData_t*)(((uint8_t*)dims_start)+dims_start->size);
                                (cur_param->tensorParam).tensorData = nn_malloc(data_start->dataSize);
                                if ((cur_param->tensorParam).tensorData == NULL) {
                                        if(cur_param->paramName)  nn_free(cur_param->paramName);
                                        if((cur_param->tensorParam).maxDimensions)  nn_free((cur_param->tensorParam).maxDimensions);
                                        if((cur_param->tensorParam).currDimensions)  nn_free((cur_param->tensorParam).currDimensions);
                                        free_unflattened_udo_parameters(i, *unflattened_params);
                                        nn_free(*unflattened_params);
                                        *unflattened_params = NULL;
                                        return -1; // errlog
                                }
                                memcpy((cur_param->tensorParam).tensorData, ((uint8_t*)data_start)+sizeof(tensorData_t), data_start->dataSize);
                        }
                } else if(cur_desc->paramType == SNPE_UDO_PARAMTYPE_STRING) {
                        udoString_t* string_data_start = (udoString_t*)(name_start + (cur_desc->name).sizeStruct - sizeof(udoString_t));
                        char* string_data_char_start = ((char*)string_data_start)+sizeof(udoString_t);
                        
                        cur_param->stringParam = nn_malloc(string_data_start->lengthString+1);
                        if (cur_param->stringParam == NULL) {
                                free_unflattened_udo_parameters(i, *unflattened_params);
                                nn_free(*unflattened_params);
                                *unflattened_params = NULL;
                                return -1; // errlog
                        }
                        strncpy(cur_param->stringParam, string_data_char_start, ((string_data_start->lengthString)+1));
                }
        }
        if(((uint8_t*)cur_desc)+cur_desc->size != ((uint8_t*)flattened)+size_flattened) {   // sanity check
                free_unflattened_udo_parameters(*num_params, *unflattened_params);
                nn_free(*unflattened_params);
                *unflattened_params = NULL;
                return -1;   // errlog
        }
        return 0;
}

int udo_append_to_list(struct nn_graph *nn, struct nn_node *node, struct udo_node* udo_n) {
        if(udo_n == NULL)  return -1;
        udo_n->node = node;
        udo_n->next = NULL;
        if(nn->num_udos == 0) {
                nn->udo_list_start = udo_n;
                nn->udo_list_end = nn->udo_list_start;
                nn->num_udos = 1;
        } else {
                (nn->udo_list_end)->next = udo_n;
                nn->udo_list_end = udo_n;
                nn->num_udos += 1;
        }
        return 0;
}
//
// append newnode to end of linked-list
// 'tail' pointer may be NULL, or may point to anything in the list.
// So, we need to find the end, in general.
//
static inline void node_append(struct nn_graph *nn, struct nn_node *newnode)
{

	struct nn_node *tmp = nn->tail;
	struct nn_node **ptr = (tmp==NULL)? &nn->head : &tmp->next;

    struct nn_node *p = *ptr;
    while( p != NULL ){ // look for last
        ptr = &p->next;
        p = *ptr;
    }
	newnode->next = NULL;
    *ptr = newnode;
    nn->tail = newnode;
    nn->node_count ++;
}

int do_append_node(
	struct nn_graph *nn,
	uint32_t node_id,
	op_type operation,
	padding_type padding,
	uint32_t num_inputs,
	uint32_t num_outputs,
	const struct input *inputs,
	const struct output *outputs)
{
	/* Allocate new node */
	/* Set default parameters and ops */
	/* Call node->ctor(node) */
	if( node_id==0) return errlog(nn,"node id cannot be 0");
	struct nn_node *node;
	if ((node = optab[operation]->ctor(
		     nn,
		     node_id,
		     operation,
		     padding,
		     num_inputs,
		     num_outputs,
		     inputs,
		     outputs)) == NULL) {
		return errlog(nn,"node id=0x%x ctor fail",node_id);
	}
	// add the new node's class flags
	// to the class set.
	uint32_t class_flags = node->ops->flags & NN_NODE_FLAGS_SET;
	nn->op_class_set |= class_flags;
	node_append(nn,node);
	return 0;
}


int do_append_udo_node(
        nn_id_t id,
        struct nn_graph *nn,
        uint32_t node_id,
	const char* package_name,
	char* op_type,
        void* flattened_static_params,
        uint32_t flattened_static_params_size,
        uint32_t num_inputs,
        uint32_t num_outputs,
        const struct input *inputs,
        const struct output *outputs,
        hexagon_nn_udo_err* err)
{
        if(node_id==0) {
                errlog(nn,"node id cannot be 0");
                *err = UDO_INVALID_NODE_ID;
                return 0;
        }

        for (int i=0; i<num_outputs; i++) {
                uint32_t ele_s = outputs[i].elementsize;
                if (ele_s != 1 && ele_s != 2 && ele_s != 4) {
                        errlog(nn,"output element size should be 1, 2 or 4 bytes");
                        *err = UDO_INVALID_INPUTS_OUTPUTS_ELEMENT_SIZE;
                        return 0;
                }
        }

        for (int i=0; i<num_inputs; i++) {
                struct nn_node* src_node = find_node(nn, inputs[i].src_id);
                uint32_t ele_s = (src_node->output_defs[inputs[i].output_idx]).elementsize;
                if (ele_s != 1 && ele_s != 2 && ele_s != 4) {
                        errlog(nn,"input element size should be 1, 2 or 4 bytes");
                        *err = UDO_INVALID_INPUTS_OUTPUTS_ELEMENT_SIZE;
                        return 0;
                }
        }

        impl_library* udo_lib = find_udo_library(package_name, op_type);
        if (udo_lib == NULL) {
                errlog(nn, "node id=0x%x append fail, udo library %s has not been registered or operation %s is not found in library %s",node_id,package_name,op_type,package_name);
                *err = UDO_LIB_NOT_REGISTERED_WITH_THIS_OP;
                return 0;
        }
        uint32_t udo_lib_id = udo_lib->index;

        struct udo_node* new_udo = nn_malloc(sizeof(struct udo_node));
        if (new_udo == NULL) {
                errlog(nn, "node id=0x%x append fail, memory allocation for keeping track of udo nodes in graph failed",node_id);
                *err = UDO_MEMORY_ALLOCATION_FAILURE;
                return 0;
        }


        SnpeUdo_Param_t* unflattened_static_params;
        uint32_t num_static_params;
        if (flattened_static_params==NULL || flattened_static_params_size==0) {   // 0 static parameter
                unflattened_static_params = NULL;
                num_static_params = 0;
        } else {
                // unflatten to SnpeUdo_Param_t
                if(unflatten_udo_parameters(flattened_static_params, flattened_static_params_size, &num_static_params, &unflattened_static_params) != 0) {
                        nn_free(new_udo);
                        errlog(nn, "node id=0x%x append fail, static parameters of operation %s in udo library %s cannot be saved",node_id,op_type,package_name);
                        *err = UDO_FAILED_TO_CREATE_OP_FACTORY;
                        return 0;
                }
        }

        if (check_udo_library_existence(udo_lib_id) != 0)  {
                nn_free(new_udo);
                errlog(nn, "node id=0x%x append fail, udo library %s has not been registered",node_id,package_name);
                *err = UDO_LIB_NOT_REGISTERED_WITH_THIS_OP;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                return 0;
        }

        SnpeUdo_String_t op_t = op_type;
        uint32_t udo_n_in = 0;
        uint32_t udo_n_out = 0;
        SnpeUdo_QuantizationType_t* udo_in_q_types = NULL;
        SnpeUdo_QuantizationType_t* udo_out_q_types = NULL;
        SnpeUdo_HexNNTensorLayout_t* udo_in_layouts = NULL;
        SnpeUdo_HexNNTensorLayout_t* udo_out_layouts = NULL;

        if ((udo_lib->validate_op) && ((udo_lib->validate_op)(op_t, num_static_params, unflattened_static_params)!=SNPE_UDO_NO_ERROR)) {
                nn_free(new_udo);
                errlog(nn, "node id=0x%x append fail, udo library %s cannot validate op %s with provided configuration",node_id,package_name,op_type);
                *err = UDO_FAILED_TO_CREATE_OP_FACTORY;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                return 0;
        }

        if (check_udo_library_existence(udo_lib_id) != 0)  {
                nn_free(new_udo);
                errlog(nn, "node id=0x%x append fail, udo library %s has not been registered",node_id,package_name);
                *err = UDO_LIB_NOT_REGISTERED_WITH_THIS_OP;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                return 0;
        }
        if (((udo_lib->query_op)(op_t, num_static_params, unflattened_static_params, &udo_n_in, &udo_in_q_types, &udo_in_layouts, &udo_n_out, &udo_out_q_types, &udo_out_layouts)!=SNPE_UDO_NO_ERROR) || (udo_n_in==0 && udo_n_out==0) || (udo_in_q_types==NULL && udo_n_in!=0) || (udo_out_q_types==NULL && udo_n_out!=0)) {
                nn_free(new_udo);
                errlog(nn, "node id=0x%x append fail, udo library %s cannot query op %s",node_id,package_name,op_type);
                *err = UDO_LIB_FAILED_TO_QUERY_OP;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                return 0;
        }
 
        uint32_t exp_hexnn_n_in = 0;
        uint32_t n_hexnn_tenosrs_per_udo_tensor_tf = 3;
        uint32_t n_hexnn_tenosr_per_udo_tensor_nonquant = 1;
        for (int i=0;i<udo_n_in;i++){
                if (udo_in_q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        exp_hexnn_n_in += n_hexnn_tenosr_per_udo_tensor_nonquant;
                } else if (udo_in_q_types[i] == SNPE_UDO_QUANTIZATION_TF) {
                        exp_hexnn_n_in += n_hexnn_tenosrs_per_udo_tensor_tf;
                } else {  // not supported
                        nn_free(new_udo);
                        errlog(nn, "node id=0x%x append fail, udo library %s op %s has supported quantization type for inputs",node_id,package_name,op_type);
                        *err = UDO_LIB_UNSUPPORTED_QUANTIZATION_TYPE;
                        free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                        return 0;
                }
        }

        uint32_t exp_hexnn_n_out = 0;
        uint32_t udo_n_quant_out = 0;
        for (int i=0;i<udo_n_out;i++){ 
                if (udo_out_q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        exp_hexnn_n_out += n_hexnn_tenosr_per_udo_tensor_nonquant; 
                } else if (udo_out_q_types[i] == SNPE_UDO_QUANTIZATION_TF) {
                        exp_hexnn_n_out += n_hexnn_tenosrs_per_udo_tensor_tf;
                        udo_n_quant_out += 1;
                } else {  // not supported
                        nn_free(new_udo);
                        errlog(nn, "node id=0x%x append fail, udo library %s op %s has supported quantization type for outputs",node_id,package_name,op_type);
                        *err = UDO_LIB_UNSUPPORTED_QUANTIZATION_TYPE;
                        free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                        return 0;
                }
        }

        uint32_t exp_hexnn_n_in_w_out_ranges = exp_hexnn_n_in + udo_n_quant_out*2;

        if ((num_inputs!=exp_hexnn_n_in_w_out_ranges) || (num_outputs!=exp_hexnn_n_out)) {
                nn_free(new_udo);
                errlog(nn, "node id=0x%x append fail, udo library %s  op %s cannot be appended due to unexpected number of inputs or outputs",node_id,package_name,op_type);
                *err = UDO_INVALID_INPUTS_OUTPUTS_NUMBER;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                return 0;
        }


        uint32_t node_ops_flag = 0;
        if (udo_in_layouts) {
                for (int i=0; i<udo_n_in; i++) {
                        if (udo_in_layouts[i] == SNPE_UDO_DSP_TENSOR_LAYOUT_D32) {
                               node_ops_flag |= NN_NODE_FLAG_D32_INPUT;
                               break;
                        }
                }
        }
        if (udo_out_layouts) {
                for (int i=0; i<udo_n_out; i++) {
                        if (udo_out_layouts[i] == SNPE_UDO_DSP_TENSOR_LAYOUT_D32) {
                               node_ops_flag |= NN_NODE_FLAG_D32_OUTPUT;
                               break;
                        }
                }
        }


        /* Allocate new node */
        /* Set default parameters and ops */
        /* Call node->ctor(node) */
        struct nn_node *node;
        if ((node = node_alloc_udo_common(
                     nn,
                     node_id,
                     num_inputs,
                     num_outputs,
                     inputs,
                     outputs,
                     node_ops_flag)) == NULL) {
                nn_free(new_udo);
                errlog(nn,"udo node id=0x%x node allocation failed",node_id);
                *err = UDO_NODE_ALLOCATION_FAILURE;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                return 0;
        }


        if (check_udo_library_existence(udo_lib_id) != 0)  {
                nn_free(new_udo);
                errlog(nn, "node id=0x%x append fail, udo library %s has not been registered",node_id,package_name);
                *err = UDO_LIB_NOT_REGISTERED_WITH_THIS_OP;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                node_free_udo_common_release(node, nn);
                return 0;
        }

        SnpeUdo_CreateOpFactoryFunction_t create_op_factory = udo_lib->create_op_factory;
        SnpeUdo_OpFactory_t this_factory = NULL;
        SnpeUdo_HexNNv2OpFactoryInfra_t op_fac_infra = {id};
        if (create_op_factory(SNPE_UDO_CORETYPE_DSP, &op_fac_infra, op_t, num_static_params, unflattened_static_params, &this_factory)!=SNPE_UDO_NO_ERROR || this_factory==NULL) {
                nn_free(new_udo);
                errlog(nn, "node id=0x%x append fail, udo op factory for operation %s cannot be created",node_id,op_type);
                *err = UDO_FAILED_TO_CREATE_OP_FACTORY;
                free_unflattened_udo_parameters(num_static_params, unflattened_static_params);
                node_free_udo_common_release(node, nn);
                return 0;
        }

        free_unflattened_udo_parameters(num_static_params, unflattened_static_params);

        (node->udo_info).udo_lib_id = udo_lib_id;
        (node->udo_info).udo_op_factory = this_factory;
        (node->udo_info).udo_operation = NULL;
        (node->udo_info).udo_create_operation = udo_lib->create_operation;
        (node->udo_info).udo_exe = udo_lib->execute_op;
        (node->udo_info).udo_release_op = udo_lib->release_op;
        (node->udo_info).udo_release_op_factory = udo_lib->release_op_factory;
        (node->udo_info).udo_input_q_types = udo_in_q_types;
        (node->udo_info).udo_input_layouts = udo_in_layouts;
        (node->udo_info).udo_num_input_tensors = udo_n_in;
        (node->udo_info).udo_output_q_types = udo_out_q_types;
        (node->udo_info).udo_output_layouts = udo_out_layouts;
        (node->udo_info).udo_num_output_tensors = udo_n_out;

        // add the new node's class flags
        // to the class set.
        uint32_t class_flags = node->ops->flags & NN_NODE_FLAGS_SET;
        nn->op_class_set |= class_flags;

        node_append(nn,node);
        udo_append_to_list(nn, node, new_udo);
        *err = UDO_SUCCESS;
        return 0;
}


int do_append_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len)
{
	/* Allocate new node */
	/* Set default parameters and ops */
	/* Call node->ctor(node) */
	if( node_id==0) return errlog(nn,"node id cannot be 0");
	struct nn_node *node;
	if ((node = hexagon_nn_const_ctor(
		nn,
		node_id,
		batches,
		height,
		width,
		depth,
		data,
		data_len)) == NULL) {
		return errlog(nn,"node id=0x%x ctor fail",node_id);
	}
	node_append(nn,node);
	return 0;
}

int do_append_empty_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	uint32_t data_len) {
	struct nn_node *node;
	if( node_id ==0) return errlog(nn,"node id cannot be 0");

	if ((node = hexagon_nn_empty_const_ctor(
		     nn,
		     node_id,
		     batches,
		     height,
		     width,
		     depth,
		     data_len)) == NULL) {
		return errlog(nn,"node id=0x%x ctor fail",node_id);
	}
	node_append(nn,node);
	return 0;
}
int do_populate_const_node(
	struct nn_graph *nn,
	uint32_t node_id,
	const uint8_t *data,
	uint32_t data_len,
	uint32_t target_offset) {
	return hexagon_nn_populate_const(nn, node_id, data, data_len, target_offset);
}

int do_teardown(struct nn_graph *nn)
{
	struct nn_node *node;
	int err;
        int udo_fail = 0;
        int dtor_fail = 0;
	nn_os_workers_kill(nn);
	nn->state = NN_GRAPH_INVALID;

        if (nn->num_udos>0) {
                struct udo_node* cur_udo_node = nn->udo_list_start;
                struct udo_node* cur_factory_check_udo_node = nn->udo_list_start;
                void* cur_factory = NULL;
                int factory_freed = 0;
                while(cur_udo_node != NULL) {
                        cur_factory = ((cur_udo_node->node)->udo_info).udo_op_factory;
                        cur_factory_check_udo_node = nn->udo_list_start;
                        factory_freed = 0;
                        while (cur_factory_check_udo_node!=cur_udo_node) {
                                if (((cur_factory_check_udo_node->node)->udo_info).udo_op_factory == cur_factory) { 
                                        factory_freed = 1;
                                        break;
                                }
                                cur_factory_check_udo_node = cur_factory_check_udo_node->next;
                        }
                        if (factory_freed == 0) {
                                if (check_udo_library_existence(((cur_udo_node->node)->udo_info).udo_lib_id) == 0) {
                                        if (((SnpeUdo_ReleaseOpFactoryFunction_t)(((cur_udo_node->node)->udo_info).udo_release_op_factory))(cur_factory)!=SNPE_UDO_NO_ERROR) {
			                        errlog(nn,"udo node id=0x%x release op factory failed in teardown", (cur_udo_node->node)->node_id);
                                                udo_fail = 1;
                                        }
                                }
                        }
                        cur_udo_node = cur_udo_node->next;
                }
        }

        if (nn->num_udos>0) {
                struct udo_node* cur_udo_node = nn->udo_list_start;
                struct udo_node* next_udo_node = NULL;
                while(cur_udo_node != NULL) {
                        next_udo_node = cur_udo_node->next;
                        nn_free(cur_udo_node);
                        cur_udo_node = next_udo_node;
                }
        }

        while (nn->head != NULL) {
                node = nn->head;
                nn->head = nn->head->next;
                if ((err = node->ops->dtor(node,nn)) != 0) {
                        errlog(nn,"node id=0x%x dtor failed in teardown", node->node_id);
                        dtor_fail = 1;
                }
        }

	allocator_teardown(nn);
	find_node_teardown(nn);
	if (nn->fake_vtcm_ptr) nn_free(nn->fake_vtcm_ptr);
	if (nn->inputs) nn_free((void *)nn->inputs);
	if (nn->outputs) nn_free(nn->outputs);
	nn_batchseqstate_free( & nn->batchseq );
	nn_free(nn->scratch);
	nn_free(nn->logbuf);
	nn_free(nn);
        if (udo_fail || dtor_fail) {
                return -1;
        } else {
	        return 0;
        }
}

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

int node_check_inputs_range( struct nn_node *self, struct nn_graph *nn, char const *name, int32_t min_no, int32_t max_no)
{
	uint32_t n = self->n_inputs;
	uint32_t i;
	int maxabs = (max_no < 0)? -max_no: max_no;
	int nullcheck = (max_no < 0)? min_no: maxabs;
	if( nullcheck > n) nullcheck = n;

	if( n < min_no || n > maxabs )
		return errlog(nn, "%s: wrong # inputs %d (range: %d...%d)", name, n, min_no, max_no);
	if( nullcheck > 0){
		struct tensor const **inputs = self->inputs;
		if( inputs == NULL) return errlog(nn,"%s: input pointer is null", name);
		for( i = 0; i < nullcheck; i++){
			if ( inputs[i] == NULL)
				return errlog(nn,"%s: NULL input %d", name, i);
		}
	}
	return 0;
}
// check if #inputs =n; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.

int node_check_inputs_n( struct nn_node *self, struct nn_graph *nn, char const *name, int32_t n)
{
	return node_check_inputs_range(self,nn, name, n, n);	// should compile to move;jump
}
// check if #outputs in range min_no .. max_no; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.
// max_no < 0 can be used to indicate that extra outputs may be NULL;
// e.g. min_no =2, max_no = -5 means outputs must be in range 2..5, and outputs 0,1 may not be
// null, but inputs 2,3,4 may be NULL; caller will need to check.
//

int node_check_outputs_range( struct nn_node *self, struct nn_graph *nn, char const *name, int32_t min_no, int32_t max_no)
{
	uint32_t n = self->n_outputs;
	uint32_t i;
	int maxabs = (max_no < 0)? -max_no: max_no;
	int nullcheck = (max_no < 0)? min_no: maxabs;
	if( nullcheck > n) nullcheck = n;

	if( n < min_no || n > maxabs )
		return errlog(nn, "%s: wrong # outputs %d (range: %d...%d)", name, n, min_no, max_no);
	if( nullcheck > 0){
		struct tensor **outputs = self->outputs;
		if( outputs == NULL) return errlog(nn,"%s: output pointer is null", name);
		for( i = 0; i < nullcheck; i++){
			if (outputs[i] == NULL)
				return errlog(nn,"%s: NULL output %d", name, i);
		}
	}
	return 0;
}
// check if #outputs =n; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.

int node_check_outputs_n(  struct nn_node *self, struct nn_graph *nn, char const *name, int32_t n)
{
	return node_check_outputs_range(self,nn, name, n,n);
}

// check if #inputs = n_in, and outputs = n_out; and check non-null.
// if not, log error and return non-zero. "name" is the node name for error messages.
int node_check_inputs_outputs_n(  struct nn_node *self, struct nn_graph *nn, char const *name, int32_t n_in, int32_t n_out)
{
	int k = node_check_inputs_range( self,nn, name, n_in,n_in);
	if( k == 0 ) k = node_check_outputs_range( self,nn, name, n_out,n_out);
	return k;
}

void graphviz_print_node(struct nn_graph *nn, struct nn_node *node, FILE *dotfile)
{
#ifdef SHOWY_DEBUG
#ifdef LINUX_DEBUG
#define SAY_GRAPH(...) printf(__VA_ARGS__);
#else
#define SAY_GRAPH(...) fprintf(dotfile, __VA_ARGS__);
#endif
	if (nn == NULL || node == NULL) {
		return;
	}
	SAY_GRAPH("    MEM%p [label=\"%s id=%lu\"];\n",
            node, hexagon_nn_op_names[node->node_type], node->node_id);
	const struct tensor **t1 = node->inputs;
	if (t1 == NULL) {
	} else {
		for (int i=0; i<node->n_inputs; i++) {
			SAY_GRAPH("    MEM%p -> MEM%p [label=\"%d\"];\n", t1[i], node, i);
		}
	}
	struct tensor **t2 = node->outputs;
	if (t2 == NULL) {
	} else {
		for (int i=0; i<node->n_outputs; i++) {
			char *red = "red";
			char *blue = "blue";
			char *black = "black";
			char *color = black;
			int num_elements = t2[i]->shape.batches * t2[i]->shape.height *
				t2[i]->shape.width * t2[i]->shape.depth;
			if (t2[i]->max_size < num_elements) {
				color = red;
			} else if (t2[i]->max_size < (4 * num_elements)) {
				color = blue;
			}
			SAY_GRAPH("    MEM%p [shape=box,label=\"%lu*%lu*%lu*%lu %lu\",color=\"%s\"];\n",
				  t2[i], t2[i]->shape.batches, t2[i]->shape.height,
				  t2[i]->shape.width, t2[i]->shape.depth, t2[i]->max_size, color);
			SAY_GRAPH("    MEM%p -> MEM%p;\n", node, t2[i]);
		}
	}
#endif
}

void graphviz_print_graph(struct nn_graph *nn)
{
#ifdef SHOWY_DEBUG
        FILE *dotfile;
        char filename[255];

        if (nn == NULL) {
                return;
        }

        uint64_t pcycle = nn_os_get_cycles(NULL);

#ifndef LINUX_DEBUG
        snprintf(filename, 255, "debug/%llu_%p.dot", pcycle, nn);
        if ((dotfile = fopen(filename, "w")) == NULL) {
                printf("Ooops... Couldn't open file %s\n", filename);
                return;
        }
#endif

        struct nn_node *node = nn->head;

        SAY_GRAPH("digraph {\n");
        while (node) {
                graphviz_print_node(nn, node, dotfile);
                node = node->next;
        }
        SAY_GRAPH("}\n");
#ifndef LINUX_DEBUG
        fclose(dotfile);
#endif
#endif
}

void print_graph_to_file(struct nn_graph *nn)
{
	/* YAML node example:
	   nodes:
	     10fa0:
	       optype: OP_INPUT
	       outputs:
	         - size: [1,1,1,1000]
		   elsize: 4
		   file: foo_104a0_0.dat
	     104a1:
	       optype: OP_Add_f
	       inputs:
	         - 104a0_0
		 - 10010_0
	       outputs:
	         - size: [1,1,1,1000]
		   elsize: 4
		   file: foo_104a1_0.dat
	*/

#if defined(V66)
	FILE *outfile;
	char charbuf[512];
	char minibuf[256];
	int i,j,k;

        if (nn == NULL) {
                return;
        }

	snprintf(charbuf,512, "%s_graph.yaml", nn->enable_graph_print_prefix);
	if ((outfile = fopen(charbuf, "w")) == NULL) {
		errlog(nn,"Ooops... Couldn't open file '%s'", charbuf);
		return;
	} else {
		logmsg(nn,1,"INFO: Writing '%s'", charbuf);
	}

	fputs("---\nnodes:\n", outfile);

	struct nn_node *node = nn->head;
	while (node) {
		uint32_t id = node->node_id;
		const char *node_type_name = hexagon_nn_op_names[node->node_type];

		snprintf(charbuf,512, "  0x%x:\n    optype: %s\n", (unsigned) id, node_type_name);
		fputs(charbuf, outfile);

		if (node->n_inputs>0) {
			fputs("    inputs:\n", outfile);
			for (i=0; i<node->n_inputs; i++) {
				snprintf(charbuf,512, "      - [0x%x,%u]\n", (unsigned) node->input_refs[i].src_id, (unsigned) node->input_refs[i].output_idx);
				fputs(charbuf, outfile);
			}
		}

		if (node->n_outputs>0) {
                        // Hack to heuristically convert all "8,1111x32,1111x32" triples into quantized triples
                        int use_hack_for_quants = 0;
                        char *hack_for_quant_strings[] = { "quint8", "float32", "float32" };
                        if ( (node->n_outputs == 3)
                             && (node->output_defs[0].elementsize == 1)
                             && (node->output_defs[1].elementsize == 4)
                             && (node->output_defs[2].elementsize == 4)
                             && (node->outputs[1]->shape.dimension[0] == 1)
                             && (node->outputs[1]->shape.dimension[1] == 1)
                             && (node->outputs[1]->shape.dimension[2] == 1)
                             && (node->outputs[1]->shape.dimension[3] == 1)
                             && (node->outputs[2]->shape.dimension[0] == 1)
                             && (node->outputs[2]->shape.dimension[1] == 1)
                             && (node->outputs[2]->shape.dimension[2] == 1)
                             && (node->outputs[2]->shape.dimension[3] == 1) ) {
                                use_hack_for_quants = 1;
                        }

			fputs("    outputs:\n", outfile);
			for (i=0; i<node->n_outputs; i++) {
				// Calculate the rank-string, e.g.  "1,1,1,1000"
				j=0;
				struct output out_def = node->output_defs[i];
				struct tensor *out = node->outputs[i];
				for (k=0; k<out_def.rank; k++) {
					j += snprintf(minibuf+j, 256-j, "%u, ", (unsigned) out->shape.dimension[k]);
				}
				if (j) {
					minibuf[j-2] = 0;  // Remove trailing comma, terminate string
				} else {
					minibuf[j] = 0;
				}

                                int eltype = out->format.type;
                                int elsize = out_def.elementsize;
                                const char *elstring = "unk";
                                if (eltype == 0) {
                                        if (elsize == 1) {
                                                elstring = "unk8";
                                        } else if (elsize == 2) {
                                                elstring = "unk16";
                                        } else if (elsize == 4) {
                                                elstring = "unk32";
                                        }
                                } else {
                                        elstring = TypeStrings[eltype];
                                }

                                if (use_hack_for_quants) {
                                        elstring = hack_for_quant_strings[i];
                                }

				snprintf(charbuf,512, "      - size: [%s]\n        eltype: %s\n        file: %s_%x_%u.dat\n",
					 minibuf,
                                         elstring,
					 nn->enable_tensor_print_prefix,
					 (unsigned) id,
					 i
					);
				fputs(charbuf, outfile);
			}
                        if (use_hack_for_quants) {
                                fputs("    quantized_outputs: [0,1,2]\n", outfile);
                        }
		}

                node = node->next;
	}

	fclose(outfile);
#endif
}

void debug_print_node(struct nn_graph *nn, struct nn_node *node)
{
	if (nn == NULL || node == NULL || nn->debug_level < 2) {
		return;
	}
	logmsg(nn, 2, "Node %p type=%s id=%lu  %lu-in %lu-out",
	       node, hexagon_nn_op_names[node->node_type], node->node_id,
	       node->n_inputs, node->n_outputs);
	const struct tensor **t1 = node->inputs;
	if (t1 == NULL) {
		logmsg(nn, 2, "Node %p expected %d inputs, got nullptr",
		       node, node->n_inputs);
	} else {
		for (int i=0; i<node->n_inputs; i++) {
			logmsg(nn, 2, "Node INPUT %d@%p %lu*%lu*%lu*%lu %lu", i, t1[i],
			       t1[i]->shape.batches, t1[i]->shape.height,
			       t1[i]->shape.width, t1[i]->shape.depth, t1[i]->max_size);
		}
	}
	struct tensor **t2 = node->outputs;
	if (t2 == NULL) {
		logmsg(nn, 2, "Node %p expected %d outputs, got nullptr",
		       node, node->n_outputs);
	} else {
		for (int i=0; i<node->n_outputs; i++) {
			logmsg(nn, 2, "Node OUTPUT %d@%p %lu*%lu*%lu*%lu %lu", i, t2[i],
			       t2[i]->shape.batches, t2[i]->shape.height,
			       t2[i]->shape.width, t2[i]->shape.depth, t2[i]->max_size);
		}
	}
}

void debug_print_graph(struct nn_graph *nn)
{
	if (nn == NULL || nn->debug_level < 2) {
		return;
	}
	struct nn_node *node = nn->head;
	while (node) {
		debug_print_node(nn, node);
		node = node->next;
	}
}

uint32_t fletcher32_tensor(struct nn_graph *nn,   struct tensor *t )
{
	// TODO - JONWOLFE - Fix implicit assumption of elementsize=1
	size_t words = t->shape.batches * t->shape.height *
		t->shape.width * ((t->shape.depth + 1) / 2);

        uint32_t sum1 = 0xffff, sum2 = 0xffff;
        size_t tlen;

	uint32_t b=0;
	uint32_t h=0;
	uint32_t w=0;
	uint32_t d=0;
	uint8_t *data;
	uint16_t val;
        while (words) {
                tlen = ((words >= 359) ? 359 : words);
                words -= tlen;
                do {
			// Get the next data word from tensor,
			//   and deal with odd-length depths
			data = tensor_location_d32(t, b, h, w, d);
			if (d == t->shape.depth - 1) {
				// zero-pad the last word if depth is odd.
				val = *data << 8;
				d=0;
				w++;
			} else if (d == t->shape.depth - 2) {
				val = (*data << 8) + *(data + 1);
				d=0;
				w++;
			} else {
				val = (*data << 8) + *(data + 1);
				d+=2;
			}
			if (w == t->shape.width) {
				w=0;
				h++;
			}
			if (h == t->shape.height) {
				h=0;
				b++;
			}

			// Fletcher-32 maths
                        sum2 += sum1 += val;
                        tlen--;
                } while (tlen);
                sum1 = (sum1 & 0xffff) + (sum1 >> 16);
                sum2 = (sum2 & 0xffff) + (sum2 >> 16);
        }
        /* Second reduction step to reduce sums to 16 bits */
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);

	if ( b != t->shape.batches || h || w || d ) {
		errlog(nn, "ERROR while computing checksum... Got to %d,%d,%d,%d on a %d,%d,%d,%d tensor",
		       b, h, w, d, t->shape.batches, t->shape.height,
		       t->shape.width, t->shape.depth);
		return 0; // Not a legal checksum... Some error.
	}
        return (sum2 << 16) | sum1;
}

void print_node_checksum(struct nn_graph *nn,  struct nn_node *node )
{
	if (nn == NULL ||node == NULL ||  nn->debug_level < 2) {
		return;
	}
	struct tensor **out = node->outputs;
	if (out == NULL) {
	} else {
		for (int i=0; i<node->n_outputs; i++) {
			uint32_t checksum = fletcher32_tensor(nn, out[i]);
			logmsg(nn, 2, "CHECKSUM: %p %08x", out[i], checksum);
		}
	}
	if (node->next) {
		debug_print_node(nn, node->next);
	}
}

void print_graph_checksum(struct nn_graph *nn)
{
	if (nn == NULL || nn->debug_level < 2) {
		return;
	}
	struct nn_node *node = nn->head;
	while (node) {
		print_node_checksum(nn, node);
		node = node->next;
	}
}
