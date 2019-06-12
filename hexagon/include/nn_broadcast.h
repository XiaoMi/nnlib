
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
#ifndef NN_BROADCAST_H
#define NN_BROADCAST_H 1

#include <nn_graph.h>

/* Is this dimension compatible (1 or same?) */
static inline int is_dim_compatible(int a, int b)
{
	return ((a == b) || (a == 1) || (b == 1));
}
/* Are all the dimensions compatible? */
static inline int are_dims_compatible(const struct shape a, const struct shape b)
{
	return is_dim_compatible(a.batches,b.batches)
		&& is_dim_compatible(a.height,b.height)
		&& is_dim_compatible(a.width,b.width)
		&& is_dim_compatible(a.depth,b.depth);
}
/* If the dimensions are compatible, what's the output dimension? */
static inline int output_dim(int a, int b)
{
	return (a == 1) ? b : a;
}

/*
 * In order to handle multiple types, we need to make a macro.
 * If we could use C++ I guess we would have type-templated function here
 */

/*
 * Elementwise Broadcasting Operators
 * 
 * For either input, if the dimension of one input is 1 and the dimension of 
 * the other input is greater than one, we automatically broadcast the values
 * N times to match the other dimension.
 * 
 * An example examples: 1x2x1x1 + 1x1x1x2 would output 1x2x1x2
 * 1x2x3x4 + 1x1x1x1 would output 1x2x3x4
 * But 1x2x3x4 + 1x1x1x2 are not compatible shapes.
 * 
 * First we calculate the output shape.
 *
 * Iterating over the output shape, each level we compute how 
 * we should stride over the dimension.  This is either 0 
 * (to repeat) or the number of elements in the lower dimensions.
 *
 * As we go through all the input dimensions, for the unity dimensions
 * that have stride zero, we will repeat the values as we increment over
 * the number of outputs at that dimension.
 *
 * There is probably a more elegant / vectorizable method here, but this
 * is the simplest I could think of.
 */

#define CREATE_ELEMENTWISE_INLINE(NAME,INTYPE,OUTTYPE,OUTTYPECODE) \
static inline int __attribute__((always_inline)) NAME( \
	struct nn_node *self, \
	struct nn_graph *nn, \
	OUTTYPE (*f)(INTYPE, INTYPE, void *), \
	void *opaque) \
{ \
	const struct tensor *a_tensor = self->inputs[0]; \
	const struct tensor *b_tensor = self->inputs[1]; \
	struct tensor *out_tensor = self->outputs[0]; \
	int32_t ab = a_tensor->shape.batches; \
	int32_t ah = a_tensor->shape.height; \
	int32_t aw = a_tensor->shape.width; \
	int32_t ad = a_tensor->shape.depth; \
	int32_t bb = b_tensor->shape.batches; \
	int32_t bh = b_tensor->shape.height; \
	int32_t bw = b_tensor->shape.width; \
	int32_t bd = b_tensor->shape.depth; \
	int32_t ob = output_dim(ab,bb); \
	int32_t oh = output_dim(ah,bh); \
	int32_t ow = output_dim(aw,bw); \
	int32_t od = output_dim(ad,bd); \
	size_t elements = ob*ow*oh*od; \
	size_t bytes = elements * sizeof(OUTTYPE); \
	/* \
	 * Need to precompute strides for BHWD \
	 * If broadcasting, stride == 0 \
	 * Else, product of lower dimensions \
	 */ \
	int32_t adstride = (ad == 1) ? 0 : 1; \
	int32_t bdstride = (bd == 1) ? 0 : 1; \
	int32_t awstride = (aw == 1) ? 0 : (ad); \
	int32_t bwstride = (bw == 1) ? 0 : (bd); \
	int32_t ahstride = (ah == 1) ? 0 : (ad*aw); \
	int32_t bhstride = (bh == 1) ? 0 : (bd*bw); \
	int32_t abstride = (ab == 1) ? 0 : (ad*aw*ah); \
	int32_t bbstride = (bb == 1) ? 0 : (bd*bw*bh); \
	const INTYPE *a_data = a_tensor->data; \
	const INTYPE *b_data = b_tensor->data; \
	OUTTYPE *out_data = out_tensor->data; \
	int32_t b,w,h,d; \
	const INTYPE *abstart; \
	const INTYPE *bbstart; \
	const INTYPE *ahstart; \
	const INTYPE *bhstart; \
	const INTYPE *aptr; \
	const INTYPE *bptr; \
	INTYPE aval; \
	INTYPE bval; \
	logmsg(nn,2,"elementwise execute. self=%p ",self); \
	if (!are_dims_compatible(a_tensor->shape,b_tensor->shape)) { \
		return errlog(nn,"incompatible shapes (%dx%dx%dx%d) (%dx%dx%dx%d)", \
			a_tensor->shape.batches, \
			a_tensor->shape.height, \
			a_tensor->shape.width, \
			a_tensor->shape.depth, \
			b_tensor->shape.batches, \
			b_tensor->shape.height, \
			b_tensor->shape.width, \
			b_tensor->shape.depth); \
	} \
	logmsg(nn,2,"shapes: %dx%dx%dx%d %dx%dx%dx%d --> %dx%dx%dx%d", \
			a_tensor->shape.batches, \
			a_tensor->shape.height, \
			a_tensor->shape.width, \
			a_tensor->shape.depth, \
			b_tensor->shape.batches, \
			b_tensor->shape.height, \
			b_tensor->shape.width, \
			b_tensor->shape.depth, \
			ob,oh,ow,od); \
	if( tensor_out_prepare_normal( out_tensor, ob,oh,ow,od, OUTTYPECODE)!= 0)\
		return errlog(nn,"out too small (id=%x): %d > %d",self->node_id,bytes,out_tensor->max_size); \
	for (b = 0; b < ob; b++) { \
		abstart = a_data + b*abstride; \
		bbstart = b_data + b*bbstride; \
		for (h = 0; h < oh; h++) { \
			ahstart = abstart + h*ahstride; \
			bhstart = bbstart + h*bhstride; \
			for (w = 0; w < ow; w++) { \
				aptr = ahstart + w*awstride; \
				bptr = bhstart + w*bwstride; \
				for (d = 0; d < od; d++) { \
					aval = *aptr; \
					bval = *bptr; \
					/* FUNCTION */ \
					*out_data++ = f(aval,bval,opaque); \
					/* END FUNCTION */ \
					aptr += adstride; \
					bptr += bdstride; \
				} \
			} \
		} \
	} \
	logmsg(nn,2,"elementwise done. self=%p ",self); \
	return 0; \
}

#if 0	// no longer in use
CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_f,float,float,NN_TYPE_QUINT8)
CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_int32,int32_t,int32_t,NN_TYPE_INT32)
#endif

CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_qint32_quint8,uint8_t,int32_t,NN_TYPE_INT32)
CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_quint8,uint8_t,uint8_t,NN_TYPE_QUINT8)

#undef CREATE_ELEMENTWISE_INLINE

#define ABPAD 128
#define ALIGN_SIZE 128
static inline void *pad_and_align(void *ptr, unsigned long minsize)
{
	uintptr_t ptrval = (uintptr_t)(ptr);
	ptrval += minsize + (ALIGN_SIZE-1);
	ptrval &= ~(ALIGN_SIZE-1);
	return (void *)ptrval;
}

struct hvx_info {
	uint8_t *a_data_pad;
	uint8_t *b_data_pad;
	int a_const_value;
	int b_const_value;
	int elements;
};

/* Look for patterns to use HVX intrinsics version of the code and broadcast/prepare the data */
int nn_check_prepare_hvx_opt(
		struct nn_graph *nn,
		const struct tensor *a_tensor,
		const struct tensor *b_tensor,
		struct tensor *out_tensor,
		const uint8_t *a_data,
		const uint8_t *b_data,
		struct hvx_info *opt_info);

static inline int __attribute__((always_inline))
check_prepare_hvx_opt(
		struct nn_graph *nn,
		const struct tensor *a_tensor,
		const struct tensor *b_tensor,
		struct tensor *out_tensor,
		const uint8_t *a_data,
		const uint8_t *b_data,
		struct hvx_info *opt_info)
{
	return nn_check_prepare_hvx_opt( nn, a_tensor, b_tensor, out_tensor, a_data, b_data, opt_info);
}

#define NEW_BROADCAST

#ifdef NEW_BROADCAST

struct elementwise_funcs {
	// pointers to 'work' functions for scalar mode
	void (* op_stride_11)( void *out, void const *in1, void const *in2, int n, void *opaque);
	void (* op_stride_10)( void *out, void const *in1, void const *in2, int n, void *opaque);
	void (* op_rev_stride_01)( void *out, void const *in1, void const *in2, int n, void *opaque);
	// pointers for HVX mode (any may be null)
	void (* op_hvx_stride_11)( void *out, void const *in1, void const *in2, int n, void *opaque);
	void (* op_hvx_stride_10)( void *out, void const *in1, void const *in2, int n, void *opaque);
	void (* op_hvx_rev_stride_01)( void *out, void const *in1, void const *in2, int n, void *opaque);
	uint8_t in_elbytes;		// elementsize (1 or 4)
	uint8_t out_elbytes;	// elementsize (1 or 4)
	uint16_t out_typecode;	// type for the output tensor
	uint16_t minlen_hvx;	// don't use hvx if n < minlen_hvx
	uint8_t hvx_need_align;	// if set, don't use hvx unless all pointers (except in2, for _10 and 01 cases) are aligned.
	uint8_t scratch_elbytes;	// normally zero, see comments above nn_elementwise_with_broadcast for usage.
};
//
// strategy:
//   - if shapes are both (b,h,w,d),   make one call to op_stride_11;
//   - if we have  (b,h,w,d) and (1,1,1,1)
//                      make one call to op_stride_01
//   - if we have  (b,w,h,d) and (1,1,1,d)
//         make (b*h*w) calls to op_stride_01
//
//
int nn_elementwise_with_broadcast(
	struct nn_node *self,
	struct nn_graph *nn,
	struct elementwise_funcs const * functabp,
	void * intermed_a,
	void * intermed_b,
	void *opaque);

// use to make most "op_stride_11" functions
// OPER is a function accepting two vars; can be a #define
#define BROADCAST_STRIDE_11_FUNC( FNAME,DTYPE_IN, DTYPE_OUT, OPER)\
static void FNAME( void *out, void const *in1, void const *in2, int n, void *opaque)\
{\
	DTYPE_OUT * op = (DTYPE_OUT*)out;\
	DTYPE_IN const * inp1 = (DTYPE_IN const *)in1;\
	DTYPE_IN const * inp2 = (DTYPE_IN const *)in2;\
	for( int i =0; i < n; i++) op[i] = OPER(inp1[i],inp2[i]);\
}
// use to make most op_stride_10 funcs
#define BROADCAST_STRIDE_10_FUNC( FNAME, DTYPE_IN, DTYPE_OUT, OPER)\
static void  FNAME( void *out, void const *in1, void const *in2, int n, void *opaque)\
{\
	DTYPE_OUT * op = (DTYPE_OUT*)out;\
	DTYPE_IN const * inp1 = (DTYPE_IN const *)in1;\
	DTYPE_IN xin2 = *(DTYPE_IN const *)in2;\
	for( int i =0; i < n; i++) op[i] = OPER(inp1[i],xin2);\
}

// use to make most op_rev_stride_10 funcs, where needed
#define BROADCAST_REV_STRIDE_01_FUNC( FNAME, DTYPE_IN, DTYPE_OUT, OPER)\
static void  FNAME( void *out, void const *in1, void const *in2, int n, void *opaque)\
{\
	DTYPE_OUT * op = (DTYPE_OUT*)out;\
	DTYPE_IN const * inp1 = (DTYPE_IN const *)in1;\
	DTYPE_IN xin2 = *(DTYPE_IN const *)in2;\
	printf("HHHH\n");\
	for( int i =0; i < n; i++) op[i] = OPER(xin2,inp1[i]) ;\
}

#endif

#endif

