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
#include <quantize.h>

#include "hvx_inlines.h"

// contains implementations for:
//   Neg_f, Abs_f
//   Neg_i32, Abs_i32
//   QuantizedNeg_8
//



struct unary_vec_op_runstate {
	uint8_t const * inptr;
	uint8_t *outptr;
	volatile int vnext;
	int vtotal;
	int vchunk;
	nn_sem_t done_sem;
};

#define MAX_THREAD_COUNT 2


//
// template for making a vector function
//
#define RUNFUNCTION_DEF(FUNCNAME,OPERATOR,OPINIT)\
static void FUNCNAME( struct nn_graph * nn, void *rstpv){								\
	struct unary_vec_op_runstate *rstp= (struct unary_vec_op_runstate*)rstpv;	\
	int vtotal = rstp->vtotal; /* total vecs */					\
	int vchunk = rstp->vchunk; /* number vecs per unit */		\
	int ivec;													\
	while( ivec = __sync_fetch_and_add(&rstp->vnext, vchunk),  ivec < vtotal ){	\
		HVX_Vector const *vsrc = (HVX_Vector const*)rstp->inptr + ivec;	\
		HVX_Vector *vdst = (HVX_Vector*)rstp->outptr + ivec;	\
		int nvec = min_i32( vchunk, vtotal-ivec);		/*# of vecs to do */ \
		if( nvec & 1){							\
			vdst[0] = OPERATOR( vsrc[0]);		\
			vdst++;								\
			vsrc++;								\
		}										\
		for( int i =0; i < nvec>>1; i++){		\
			HVX_Vector v0 = OPERATOR(vsrc[0]);	\
			HVX_Vector v1 = OPERATOR(vsrc[1]);	\
			vdst[0] = v0;						\
			vdst[1] = v1;						\
			vsrc+= 2;							\
			vdst+=2;							\
		}				\
	}					\
	nn_sem_post(&rstp->done_sem);\
}
#define OPER_NEGF(V) Q6_V_vxor_VV(V,Q6_V_vsplat_R(0x80000000))
#define OPER_NEGI32(V) Q6_Vw_vsub_VwVw_sat(Q6_V_vzero(),V)
#define OPER_1SCOMP(V) Q6_V_vnot_V(V)
#define OPER_ABSF(V) Q6_V_vand_VV(V,Q6_V_vsplat_R(0x7FFFFFFF))
#define OPER_ABSI32(V) Q6_Vw_vabs_Vw_sat(V)

RUNFUNCTION_DEF(hvx_run_neg_f, OPER_NEGF,)
RUNFUNCTION_DEF(hvx_run_abs_f, OPER_ABSF,)
RUNFUNCTION_DEF(hvx_run_neg_i32, OPER_NEGI32,)
RUNFUNCTION_DEF(hvx_run_abs_i32, OPER_ABSI32,)
RUNFUNCTION_DEF(hvx_run_1scomp, OPER_1SCOMP,)

typedef void (*run_fp)(struct nn_graph *nn,void*);


static int unary_op_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor * out_tensor = self->outputs[0];

	unsigned dtype = NN_TYPE_INT32;
	run_fp run_functionp;

	switch(self->node_type){
		case OP_Neg_f:
			run_functionp = hvx_run_neg_f;
			dtype = NN_TYPE_FLOAT;
			break;
		case OP_Abs_f:
			run_functionp = hvx_run_abs_f;
			dtype = NN_TYPE_FLOAT;
			break;
		case OP_Neg_int32:
			run_functionp = hvx_run_neg_i32;
			break;
		case OP_Abs_int32:
			run_functionp = hvx_run_abs_i32;
			break;
		case OP_QuantizedNeg_8:
			run_functionp = hvx_run_1scomp;
			dtype = NN_TYPE_QUINT8;
			break;
		default:
			return errlog(nn,"bad node type");
	}

	if(tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, dtype)!= 0 ){
		return errlog(nn, "output too small");
	}
	unsigned total_v = (out_tensor->data_size+127)/128u;		// # of vectors

	unsigned vchunk = 512;			// process 64K at once
	if( total_v < vchunk*3){
		if( total_v > 256){
			vchunk = 2*((total_v+3)>>2);	// half; rounded up to even
		}else{
			vchunk = total_v;
		}
	}
	int nthreads = MAX_THREAD_COUNT;
	if( total_v <= (MAX_THREAD_COUNT-1)*vchunk){
		nthreads = (total_v + (vchunk-1))/vchunk;
	}

	struct unary_vec_op_runstate rst;
	rst.inptr  = in_tensor->data;
	rst.outptr = out_tensor->data;
	rst.vnext = 0;
	rst.vtotal = total_v;
	rst.vchunk = vchunk;
	nn_sem_init( &rst.done_sem,0);

	for( int i = 0; i < nthreads; i++)
		nn_os_work_for_vector( nn, run_functionp, &rst);

	int result = 0;
	if( self->node_type == OP_QuantizedNeg_8){
		// propogate (min,max) to (-max,-min)
		struct tensor const * min_tensor = self->inputs[1];
		struct tensor const * max_tensor = self->inputs[2];
		float in_min = tensor_get_float(min_tensor,0);
		float in_max = tensor_get_float(max_tensor,0);

		if( tensor_set_single_float(self->outputs[1],-in_max)!=0
		  || tensor_set_single_float(self->outputs[2],-in_min)!=0 ){
			errlog(nn,"output too small for min or max");
			result = -1;
		}

	}

	nn_sem_wait_n_times( & rst.done_sem, nthreads);
	return result;
}


// these operators all support aliasing because they can work in-place
// (writing the output back to the same memory).

struct nn_node_ops nn_ops_for_Neg_f = {
	.execute = unary_op_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS
};
struct nn_node_ops nn_ops_for_Neg_int32 = {
	.execute = unary_op_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS
};
struct nn_node_ops nn_ops_for_Abs_f = {
	.execute = unary_op_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS
};
struct nn_node_ops nn_ops_for_Abs_int32 = {
	.execute = unary_op_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS
};

struct nn_node_ops nn_ops_for_QuantizedNeg_8 = {
	.execute = unary_op_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_CLS_SUPPORTS_ALIAS
};
