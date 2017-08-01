
/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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
static inline int are_dims_compatible(struct shape a, struct shape b)
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

#define CREATE_ELEMENTWISE_INLINE(NAME,INTYPE,OUTTYPE) \
static inline int NAME( \
	struct nn_node *self, \
	struct nn_graph *nn, \
	OUTTYPE (*f)(INTYPE, INTYPE, void *), \
	void *opaque) \
{ \
	const struct tensor *a_tensor = (const struct tensor *)self->inputs[0]; \
	const struct tensor *b_tensor = (const struct tensor *)self->inputs[1]; \
	struct tensor *out_tensor = (struct tensor *)self->outputs[0]; \
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
	const INTYPE *a_data = (const INTYPE *)a_tensor->data; \
	const INTYPE *b_data = (const INTYPE *)b_tensor->data; \
	OUTTYPE *out_data = (OUTTYPE *)out_tensor->data; \
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
	if (bytes > out_tensor->max_size) return errlog(nn,"out too small (id=%x): %d > %d",self->node_id,bytes,out_tensor->max_size); \
	tensor_set_shape(out_tensor,ob,oh,ow,od); \
	out_tensor->data_size = bytes; \
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

CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_f,float,float)
CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_int32,int32_t,int32_t)
CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_qint32_quint8,uint8_t,int32_t)
CREATE_ELEMENTWISE_INLINE(broadcast_elementwise_execute_quint8,uint8_t,uint8_t)

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
static inline int check_prepare_hvx_opt(
		struct nn_graph *nn,
		const struct tensor *a_tensor,
		const struct tensor *b_tensor,
		struct tensor *out_tensor,
		const uint8_t *a_data,
		const uint8_t *b_data,
		struct hvx_info *opt_info) {

	uint32_t ab, ah, aw, ad;
	uint32_t bb, bh, bw, bd;
	int opt_flag=0;
	uint8_t *a_data_pad = NULL;
	uint8_t *b_data_pad = NULL;
	int elements, bhw, i;
	int a_const_value = 0;
	int b_const_value = 0;

	tensor_get_shape(a_tensor,&ab,&ah,&aw,&ad);
	tensor_get_shape(b_tensor,&bb,&bh,&bw,&bd);
	//t2 =  nn_os_get_cycles(nn);

	/* One of the operands is scalar */
	if((ab==1)&&(ah==1)&&(aw==1)&&(ad==1))	{
		elements = bb*bh*bw*bd;

		b_data_pad = (uint8_t *)b_data;// since b_data is already aligned to 128, no need to create padded buffer
		a_data_pad = (uint8_t *)nn->scratch;

		tensor_set_shape(out_tensor,bb,bh,bw,bd);

		opt_flag = 1;
		a_const_value = a_data[0];
	}
	else if((bb==1)&&(bh==1)&&(bw==1)&&(bd==1))	{
		elements = ab*ah*aw*ad;

		a_data_pad = (uint8_t *)a_data;// since a_data is already aligned to 128, no need to create padded buffer
		b_data_pad = (uint8_t *)nn->scratch;

		tensor_set_shape(out_tensor,ab,ah,aw,ad);

		opt_flag = 1;
		b_const_value = b_data[0];
	}

	/* Both operands are of same dimensions */
	else if ((ab==bb)&&(ah==bh)&&(aw==bw)&&(ad==bd))	{
		//logmsg(nn,0,"Entering qmul asm - a=%p, b=%p, elem=%d, elem_pad=%d", a_data_pad, b_data_pad, elements, elements_pad);
		//t3 =  nn_os_get_cycles(nn);
		elements = ab*ah*aw*ad;

		a_data_pad = (uint8_t *)a_data;// since a_data is already aligned to 128, no need to create padded buffer
		b_data_pad = (uint8_t *)b_data;// since b_data is already aligned to 128, no need to create padded buffer

		tensor_set_shape(out_tensor,ab,ah,aw,ad);

		opt_flag = 1;
	}

	/* Depth matches on both operands - Broadcast elements in one operand to match dimensions of other operand */
	else if ((bb==1)&&(bh==1)&&(bw==1)&&(bd==ad))	{

		elements = ab*ah*aw*ad;
		bhw = ab*ah*aw;

		a_data_pad = (uint8_t *)a_data;// since a_data is already aligned to 128, no need to create padded buffer
		b_data_pad = (uint8_t *)nn->scratch;

		// Broadcast elements in b to match a dimensions
		for(i=0;i<bhw;i++) {
			vmemcpy_asm(b_data_pad,b_data,bd);
			b_data_pad += bd;
		}
		b_data_pad = (uint8_t *)nn->scratch;//pad_and_align(a_data_pad, elements_pad);

		tensor_set_shape(out_tensor,ab,ah,aw,ad);

		opt_flag = 1;
		//printf("Depth optimization: a: %lux%lux%lux%lu b: %lux%lux%lux%lu\n", ab,ah,aw,ad, bb,bh,bw,bd);

	}
	else if ((ab==1)&&(ah==1)&&(aw==1)&&(bd==ad))	{

		elements = bb*bh*bw*bd;
		bhw = bb*bh*bw;

		b_data_pad = (uint8_t *)b_data ;// since b_data is already aligned to 128, no need to create padded buffer
		a_data_pad = (uint8_t *)nn->scratch;

		// Broadcast elements in b to match a dimensions
		for(i=0;i<bhw;i++) {
			vmemcpy_asm(a_data_pad,a_data,ad);
			a_data_pad += ad;
		}
		a_data_pad = (uint8_t *)nn->scratch;

		tensor_set_shape(out_tensor,bb,bh,bw,bd);

		opt_flag = 1;
		//printf("Depth optimization: a: %lux%lux%lux%lu b: %lux%lux%lux%lu\n", ab,ah,aw,ad, bb,bh,bw,bd);

	}

	if(opt_flag) {
		opt_info->a_data_pad = a_data_pad;
		opt_info->b_data_pad = b_data_pad;
		opt_info->elements = elements;
		opt_info->a_const_value = a_const_value;
		opt_info->b_const_value = b_const_value;
	}

	return opt_flag;
}



#endif

