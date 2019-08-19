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

#include "nn_const_prep_share.h"

static nn_mutex_t nn_const_share_mutex = NN_MUTEX_INIT;

struct nn_node* nn_cpshare_get_const_node( struct nn_graph *nn, struct nn_node* self, int input_no )
{
	struct nn_node * res = NULL;
	if( self->n_inputs > input_no ){
		uint32_t nid = self->input_refs[input_no].src_id;
		res = find_node(nn, nid );
		if( res->node_type == OP_Const) return res;
	}
	return NULL;
}

struct nn_cpshare_base *nn_cpshare_get_existing( struct nn_graph * nn,
	struct nn_cpshare_typedesc const*td, struct nn_node const * cnode )
{
	struct nn_cpshare_base *result = NULL;
	if( cnode != NULL && cnode->node_type == OP_Const ){ 
		nn_mutex_lock(&nn_const_share_mutex);
		struct nn_cpshare_base *p = (struct nn_cpshare_base *) cnode->opaque;
		if( p != NULL && p->typedesc == td ){
			p->ref_count++;
			result = p;
		}
		nn_mutex_unlock(&nn_const_share_mutex);
	}
	return result;
}


static void
nn_cpshare_call_dtor( struct nn_graph *nn, struct nn_cpshare_base * cpshare )
{
	nn_cpshare_dtor_fp dtor = cpshare->typedesc->dtor;
	if( dtor != NULL){
		(*dtor)(nn,cpshare);
	}else{
		if( cpshare->ptr_x != NULL ) nn_free( cpshare->ptr_x);
		if( cpshare->ptr_sumb != NULL ) nn_free( cpshare->ptr_sumb);
		if( cpshare->ptr_w != NULL ) nn_free( cpshare->ptr_w);
		nn_free( cpshare );
	}
}

void
nn_cpshare_attach( struct nn_graph *nn, struct nn_node* const_node, void * cpsharev )
{
	struct nn_cpshare_base * cpshare = (struct nn_cpshare_base *)cpsharev;
	nn_mutex_lock( &nn_const_share_mutex);
	if( const_node->opaque == NULL ){
		const_node->opaque = cpshare;
		cpshare->ref_count++;
	}
	nn_mutex_unlock( &nn_const_share_mutex);
}
void
nn_cpshare_decref( struct nn_graph *nn, void * cpsharev )
{
	struct nn_cpshare_base * cpshare = (struct nn_cpshare_base *)cpsharev;
	nn_mutex_lock( &nn_const_share_mutex);
	cpshare->ref_count--;
	if( cpshare->ref_count <= 0){
		nn_cpshare_call_dtor( nn, cpshare);
	}
	nn_mutex_unlock( &nn_const_share_mutex);
}
