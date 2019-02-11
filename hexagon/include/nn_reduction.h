
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
#ifndef NN_REDUCTION_H
#define NN_REDUCTION_H 1

#include <nn_graph.h>


//
// When reducing selected dimensions on a linearly-addressed tensor,
// it is possible to 'munge' adjacent dims together if they are both
// reduced, or if both are not reduced. (where the input dim is 1,
// the dimension may be combined with either).
// for instance [2,3,5,6] -> [2,3,1,1] can be done as 6 1-d reductions [2*3,5*6]->[2*3,1]

// Thus with 4 dimensions, the worst case is to have 2 loops
// of reduction  (e.g [2,3,4,12] -> [2,1,4,1] we need reductions on 3->1, 12->1
// but  [2,3,1,12] -> [2,1,1,1] can be done as [2,3*1*12]->[2,1]
// We also have at most 2 dims of iteration.
//
// General case can be done by mapping to a 5-dimensional case where
// the reduction is always done on the 2nd and 4th dims:
//    [p,q,r,s,t]->[p,1,r,1,t]
// (where at least one of p,t is 1; and s>1 (unless there is no reduction
// at all).
//
// Examples:
//  [2,3,5,6]->[1,1,1,1]     [1,1,1,2*3*5*2,1] -> [1,1,1,1,1]
//  [2,3,5,6]->[2,3,1,1]	 [1,1,2*3,5*6,1]	->[1,1,2*3,1,1]
//  [2,3,5,6]->[2,1,5,6]     [1,1,2,3,5*6] -> [1,1,2,1,5*6]
//
// This function finds the p,q,r,s,t parms to map a given
// reduction problem to [p,q,r,s,t]-> [p,1,r,1,t]
// It promises that:
//     -all are >= 1
//     -at least one of p,r,t is 1
//     -if s=1, then p=q=r=1 as well (this is a 'no reduction' case)
//     -if q=1, then p = 1  (1d reduction needs only r,s,t)
//     -if q >1, then r != 1  (2d reduction is only spec when needed)
//
// So there are four possible outcomes:
//     [1,1,1,1,t] :         no reduction, copy 't'
//     [1,1,r,s,t], s>1:     'r' of 1d-reduction(s) of t-vector
//                            r=t=1 is a full reduction.
//     [p,q,r,s,1], q,r,s>1  2-d reduction.
//     [1,q,r,s,t], q,r,s,t>1 :  2-d reduction
//
// IMPORTANT:
//  caller needs to ensure all dims >=1, and all dims
// match across in/out except where shape_out dim is 1.
//
void	/// this is in shape_util.c
nn_find_generic_reduction_dims(
		struct shape const *shape_in,
		struct shape const *shape_out,
		int generic_reduction_dims[5]);


/*
 * Generic Tensor reduction:
 *  If there is only one input, it is fully reduced to [1,1,1,1].
 *  If there are three, the third input is a single integer defining 'true rank'
 *  And the second is [1,1,1,d] of integers defining dimensions to reduce along.
 *   'true rank' indicates the numbering system for dimensions.
 *    true_rank = 4:
 *          0,1,2,3  -> B,H,W,D
 *    true_rank = 3:
 *          0,1,2  -> H,W,D
 *    true_rank = 2:
 *          0,1  -> W,D
 *    true_rank = 1:
 *          0  -> D
 *  If any dimension index is negative, all dimensions will be reduced.
 *
 *  In addition, if padding == NN_PAD_VALID, then the reduced dimensions are 'squeezed'
 *  to the end, e.g. a [2,5,7,32] tensor reduced on H & W becomes [1,1,2,32] rather than [2,1,1,32].
 *
 *  The third input can be omitted and defaults to 4.
 */

//
// This function finds the output shape.
// The 'generic_reduction_dims' is obtained by by passing
// the input and output shape to find_generic_reduction_dims,
// but in this case the 'output shape' is the shape *before* squeezing
// reduced dims (which can be different from the output shape
// when padding == NN_PAD_VALID).
//
void
nn_find_reduction_shape(	struct nn_node *self,struct nn_graph *nn,
		struct shape *out_shape_p, int generic_reduction_dims[5]);


#define CREATE_REDUCTION_INLINE(TYPENAME,TYPE,OUTTYPECODE) \
static inline TYPE __attribute__((always_inline)) nn_do_1d_reduction_##TYPENAME( \
	const TYPE *in, \
	int n,\
	int stride,\
	TYPE (*f)(TYPE,TYPE), \
	TYPE initializer) \
{\
	TYPE tmp = initializer; \
	int i;\
	for(i=0;i<n;i++) tmp = (*f)(tmp,in[i*stride]);\
	return tmp;\
}\
static inline int __attribute__((always_inline)) nn_reduction_##TYPENAME( \
	struct nn_node *self, \
	struct nn_graph *nn, \
	TYPE (*f)(TYPE, TYPE), \
	TYPE initializer) \
{ \
	const struct tensor *input_tensor = self->inputs[0]; \
	struct tensor *out_tensor = self->outputs[0]; \
	struct shape out_shape;\
	const TYPE *in = (const TYPE *)input_tensor->data; \
	int rdims[5];\
	TYPE *out;\
	nn_find_reduction_shape(self,nn,&out_shape,rdims);\
	if( tensor_out_prepare_normal_fromshape(out_tensor,&out_shape,OUTTYPECODE)!=0)\
		return errlog(nn,"out too small"); \
	out = (TYPE *)out_tensor->data; \
	if( rdims[3]==1){ /* no reduction, copy */	\
		memcpy(out, in, rdims[4]*sizeof(TYPE));	\
	}else{ /* general case */	\
		int n_out= rdims[0];	\
		int r_out = rdims[1];	\
		int n_in = rdims[2];	\
		int r_in = rdims[3];	\
		int n_vec =rdims[4];	/* often 1*/	\
		int i_out, i_in, ir_out, ivec;	\
		if( n_vec == 1){ \
			/* 2d reduction { n_out, [r_out], n_in, [r_in] } -> { n_out, n_in } */\
			/* this case includes 'full-reduce' with n_out=r_out=n_in= 1*/\
			for(i_out = 0; i_out < n_out; i_out++){\
				for( i_in = 0; i_in < n_in; i_in++){\
					TYPE result = initializer;\
					for( ir_out = 0; ir_out < r_out; ir_out++ ){\
						TYPE const *inp = in + i_in*r_in + ir_out*(r_in*n_in) + i_out*(r_in*n_in*r_out);\
						TYPE inres = nn_do_1d_reduction_##TYPENAME(inp, r_in, 1, f, initializer);\
						result = (*f)(result,inres);\
					}\
					*out++ = result;\
				}\
			}\
		}else{\
			/* =>n_out=1, so 2d reduction {[r_out], n_in, [r_in], n_vec } -> {n_in, n_vec } */\
			for(i_in = 0; i_in < n_in; i_in++){\
				for( ivec = 0; ivec < n_vec; ivec++){\
					TYPE result = initializer;\
					for( ir_out = 0; ir_out < r_out; ir_out++ ){\
						TYPE const *inp = in + ivec + i_in*(r_in*n_vec) + ir_out*(n_in*r_in*n_vec);\
						TYPE inres = nn_do_1d_reduction_##TYPENAME(inp, r_in, n_vec, f, initializer);\
						result = (*f)(result,inres);\
					}\
					*out++ = result;\
				}\
			}\
		}\
	}\
	return 0; \
}


CREATE_REDUCTION_INLINE(float,float,NN_TYPE_FLOAT)
CREATE_REDUCTION_INLINE(int32,int32_t,NN_TYPE_INT32)

#undef CREATE_REDUCTION_INLINE



#endif

