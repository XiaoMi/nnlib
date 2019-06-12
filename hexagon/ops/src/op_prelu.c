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
 * This contains implementations for quantized Prelu node
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <stdio.h>

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

#define MAX_THREAD (2)   //Works only for thread 2 and less
#define ALIGN_SIZE 128
#define NUM_BYT_PERVECTOR (ALIGN_SIZE)
#define ROUNDUP(X) (((X) + ALIGN_SIZE - 1) & (~((ALIGN_SIZE)-1)))
#define NUM_BYT_PERVECTOR_MASK (NUM_BYT_PERVECTOR - 1)
#define MAXPAD (ALIGN_SIZE)
//#define  DEBUG_PRINT_PRELU_REF_PERFORMANCE
//#define  DEBUG_PRINT_PRELU_PERFORMANCE
#define MIN_ELEM_MULTI_THRED      (200)    // For float alpha multi threading
#define MIN_INP_ELEL_MULTITHREAD  (6000)   // For input data multi threading



struct tdata_prelu {
	struct nn_node *self;
	void * iptr;
	void * optr;
	void * ptr1;
	void * ptr2;
	void * ptr3;
	void * ptr4;
	int    arg0;
	int    arg1;
	int    arg2;
	int    arg3;
	int    arg4;
	float  arg_f;
	nn_sem_t donesem;
	uint64_t cycles;
};

static inline void *pad_and_align(void *ptr, unsigned long minsize)
{
	uintptr_t ptrval = (uintptr_t)(ptr);
	ptrval += minsize + (MAXPAD-1);
	ptrval &= ~(ALIGN_SIZE-1);
	return (void *)ptrval;
}

//	Reference fixed point Prelu implementation
static int prelu_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *in_alpha_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t alpha_depth = in_alpha_tensor->shape.depth;
	size_t bytes = in_tensor->shape.batches 
		* in_tensor->shape.height
		* in_tensor->shape.width
		* in_tensor->shape.depth;
	uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	uint32_t i,j,idx;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	if (nn_scratch_grow(nn,sizeof(float)*alpha_depth)){
		return errlog(nn,"failed to get scratch");
	}
	float *alpha = nn->scratch;
	uint8_t quantized_zero = quantize_uint8(0.0f,in_min,in_max);
	//uint32_t alpha_frac = (1<<16) * alpha;
	//uint32_t alpha_offset = quantized_zero - ((quantized_zero * alpha_frac + 0x08000) >> 16);
	uint32_t alpha_frac;
	uint32_t alpha_offset;

	/* Assert min and max are size 1,1,1,1 ? */

	if ((alpha_depth > 1) && (alpha_depth != in_tensor->shape.depth)) return errlog(nn,"Input depth mismatch ");

	for(j =0; j <  alpha_depth; j++) {
		alpha[j] = tensor_get_float(in_alpha_tensor,j);
		if (alpha[j] < 0.0f) return errlog(nn,"negative alpha %f, %d", alpha[j],j);
		if (alpha[j] > 1.0f) return errlog(nn,"alpha must be <= 1.0f");
	}

	logmsg(nn,2,"Prelu execute. self=%p ",self);
	logmsg(nn,2,"Prelu in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));
	//logmsg(nn,2,"alpha=%f alpha_frac=%x/%x,alpha_offset=%x\n",
	//	alpha,alpha_frac,1<<16,alpha_offset);

	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = bytes;

#ifdef DEBUG_PRINT_PRELU_REF_PERFORMANCE
		int start_time =  nn_os_get_cycles(nn);
#endif

	for (i = 0; i < bytes/alpha_depth; i++) {
		for (j = 0; j < alpha_depth; j++) {
			idx = i*alpha_depth +j;
			out_data[idx] = in_data[idx];
			if (in_data[idx] < quantized_zero) {
				alpha_frac = (1<<16) * alpha[j];
				alpha_offset = quantized_zero - ((quantized_zero * alpha_frac + 0x08000) >> 16);
				out_data[idx] = ((in_data[idx] * alpha_frac+0x08000) >> 16) + alpha_offset;
				logmsg(nn,2,"in_data[%d] = %d, out_data = %d\n",i,in_data[i],out_data[i]);
			}
			//printf(" in_data[%lu] %x, alpha_frac %lx , alpha_offset %lx, out_data[] = %x\n", idx, in_data[idx], alpha_frac, alpha_offset, out_data[idx]);
		}
	}

	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

#ifdef DEBUG_PRINT_PRELU_REF_PERFORMANCE
	int end_time =  nn_os_get_cycles(nn);
	printf("Prelu-ref cycles = %d (elements = %d)\n", (end_time-start_time), bytes);
#endif
	logmsg(nn,2,"Prelu out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"Prelu %p done",self);
	return 0;
}


// Concat two vectors , along with rotation. Used in circular buffer.
#define concat_vec( fr_ping,  fr_pong,  circ_res,  rot_pong, result)\
{\
	fr_pong = Q6_V_vror_VR(fr_pong, rot_pong);\
	result = Q6_V_valign_VVR(fr_ping, fr_pong, circ_res);\
}

// Generic module to handle different scenarios to concat two arrays of size 128 bytes.
//  Concat the arrays at any given byte location.  

#define concat_arry( al_fr, ping_idx, pong_idx, circ_res, result) {\
	HVX_Vector fr_ping, fr_pong ;\
\
	fr_ping   = *(al_fr+ping_idx);\
	fr_pong   = *(al_fr+pong_idx);\
	result =  Q6_V_valign_VVR(fr_ping, fr_pong, circ_res);\
}



static inline HVX_Vector prelu_1d_intrinsic(HVX_Vector V_in_val, HVX_Vector V_frac, HVX_Vector V_offset, HVX_Vector V_round, HVX_Vector V_zero, int32_t Shift_R)
{
	
	HVX_Vector V_temp8, V_temp16, V_temp16_2;
	HVX_VectorPair V_pair;
	HVX_VectorPred   inp_neg_pred;

	inp_neg_pred = Q6_Q_vcmp_gt_VubVub(V_zero, V_in_val);
		
			//(in_data[i] * alpha_frac)
	V_pair   = Q6_Wuh_vmpy_VubVub(V_in_val, V_frac);

	//(in_data[i] * alpha_frac+0x080)
	V_temp16   = Q6_Vuh_vadd_VuhVuh_sat(Q6_V_lo_W(V_pair), V_round);
	V_temp16_2 = Q6_Vuh_vadd_VuhVuh_sat(Q6_V_hi_W(V_pair), V_round);

	//out_data[i] = ((in_data[i] * alpha_frac+0x080) >> 8) + alpha_offset;
	V_temp16   = Q6_Vuh_vlsr_VuhR(V_temp16  , Shift_R);
	V_temp16_2 = Q6_Vuh_vlsr_VuhR(V_temp16_2, Shift_R);
	V_temp8    =  Q6_Vb_vshuffe_VbVb(V_temp16_2, V_temp16);

	//out_data[i] = ((in_data[i] * alpha_frac+0x080) >> 8) + alpha_offset;
	V_temp8 =  Q6_Vb_vadd_VbVb(V_temp8, V_offset);
 	// if( in_data < quantized_zero) out_data = alpha*in_data else out_data = in_data
	return  Q6_V_vmux_QVV(inp_neg_pred, V_temp8, V_in_val);

}

// Common function which can be used in other OP.
//		
//uint32_t alpha_frac = (1<<16) * alpha;   << This part still not vectorized -TBD.  

struct tdata_float2u16 {
	struct nn_node *self;
	void * iptr;
	void * optr;
	int    arg0;
	nn_sem_t donesem;
	uint64_t cycles;
};

void float_to_u16(struct nn_graph *nn, void *vinfo)
{
	struct tdata_float2u16 *info = vinfo;
	int j;

	float    *alpha     = (float *)info->iptr;
	uint16_t *out_u16   = (uint16_t *)info->optr;
	uint32_t elem       = (uint32_t) info->arg0;


	for (j = 0; j < elem; j++) {
		out_u16[j] = alpha[j] * (1 << 16);
	}

	nn_sem_post(&info->donesem);
}

// Generic function : Converts input  float  , to fixed point u16
//	out_u16[j] = inp_f[j] * (1 << 16);
// 	This function can be used plugged in other modules.
static inline void hvx_float2_u16( float *inp_f, uint16_t *out_u16, uint32_t elem, struct nn_node *self, struct nn_graph *nn)
{
	int      idx, num_thread,thr;
	struct   tdata_float2u16  td[MAX_THREAD];

	if(elem < MIN_ELEM_MULTI_THRED)
		num_thread = 1;
	else 
		num_thread = MAX_THREAD; 

	idx =0;
	for(thr=0;thr < num_thread; thr++){
		td[thr].self = self;
		td[thr].iptr = &inp_f[idx];
		td[thr].optr = &out_u16[idx];
		if(thr == (num_thread -1) )
			idx = elem - idx;
		else
			idx += elem/num_thread; 
		td[thr].arg0 = idx;
		nn_sem_init(&td[thr].donesem,0);
		nn_os_work_for_vector(nn,float_to_u16,  &td[thr]);
	}
	for(thr=0;thr < num_thread; thr++){
		nn_sem_wait(&td[thr].donesem);
	}
}

/*
	for(j=0;j < alpha_depth;j++){
		alpha_offset[j] = quantized_zero - ((quantized_zero * temp16_al[j] + 0x08000) >> 16);
		alpha_frac[j] = temp16_al[j] >> 8;
	}
*/
void do_hvx_quant_frac_offst(struct nn_graph *nn, void *vinfo)
{
		struct tdata_prelu *info = vinfo;

		uint8_t  *alpha_frac    = (uint8_t *)info->ptr1;
		uint8_t  *alpha_offset  = (uint8_t *)info->ptr2;
		uint8_t  *temp16_al     = (uint8_t *)info->ptr3;
		//uint32_t bytes          = (uint32_t) info->arg0;
		uint32_t alpha_depth    = (uint32_t) info->arg1;
		uint32_t quantized_zero = (uint32_t) info->arg2;


		uint32_t quantized_zero_16 = quantized_zero << 16 | quantized_zero, quantized_zero32, j;
		HVX_Vector  V_round;
		V_round        = Q6_V_vsplat_R(0x00008000);
		
		quantized_zero32 = quantized_zero << 8    | quantized_zero;
		quantized_zero32 = quantized_zero32 << 16 | quantized_zero32;

		HVX_Vector *frac = (HVX_Vector *)alpha_frac;
		HVX_Vector *offt = (HVX_Vector *)alpha_offset;
		HVX_Vector quant_zero = Q6_V_vsplat_R(quantized_zero32);
		HVX_Vector *in_alpha = (HVX_Vector *)temp16_al;
		HVX_Vector temp16,temp16_1, temp32_1, temp32_2, frac_16_1, frac_16_2, temp8;
		HVX_VectorPair temp_pair;

		
		for(j=0;j < alpha_depth/NUM_BYT_PERVECTOR+1;j++){

			frac_16_1 = *in_alpha++; 
			frac_16_2 = *in_alpha++;
			
			
			temp_pair = Q6_Wuw_vmpy_VuhRuh(frac_16_1, quantized_zero_16); //(quantized_zero * temp16_al[j]
			
			temp32_1  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(temp_pair), V_round);   //_zero * temp16_al[j] + 0x08000)
			temp32_2  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(temp_pair), V_round);   //_zero * temp16_al[j] + 0x08000)
			temp16    = Q6_Vh_vshuffo_VhVh(temp32_2,temp32_1);            //al[j] + 0x08000) >> 16)
			
			temp_pair = Q6_Wuw_vmpy_VuhRuh(frac_16_2, quantized_zero_16);
			
			temp32_1  = Q6_Vw_vadd_VwVw(Q6_V_lo_W(temp_pair), V_round);
			temp32_2  = Q6_Vw_vadd_VwVw(Q6_V_hi_W(temp_pair), V_round);
			temp16_1  = Q6_Vh_vshuffo_VhVh(temp32_2,temp32_1);            //0x08000) >> 16
						
			temp8     = Q6_Vb_vpacke_VhVh(temp16_1, temp16);  			  //
			*offt++   = Q6_Vb_vsub_VbVb(quant_zero , temp8); 			  //quantized_zero - ((quantized_zero * temp16_al[j] + 0x08000) >> 16)
			*frac++   = Q6_Vb_vpacko_VhVh(frac_16_2, frac_16_1); 		  // alpha_frac[j] = temp16_al[j] >> 8;
		}
	nn_sem_post(&info->donesem);
}

// Main function which works on quantized - alpha of depth > 1.
//  Multi threading capable.
//  Circular buffer for- high speed - odd depth size. 
//  If depth is < 128, Circular buffer duplicated to keep it > 128 .
void hvx_intr_prelu_circ(struct nn_graph *nn, void *vinfo)
{

	struct tdata_prelu *info = vinfo;

	uint8_t  *in_data       = (uint8_t *)info->iptr;
	uint8_t  *out_data      = (uint8_t *)info->optr;
	uint8_t  *alpha_frac    = (uint8_t *)info->ptr1;
	uint8_t  *alpha_offset  = (uint8_t *)info->ptr2;
	uint8_t  *buf_pad       = (uint8_t *)info->ptr3;
	uint32_t bytes          = (uint32_t) info->arg0;
	uint32_t alpha_depth    = (uint32_t) info->arg1;
	uint32_t quantized_zero = (uint32_t) info->arg2;



	uint32_t circ_res=0, circ_idx=0, circlen=0 , max_circlen_slot;
	int      i, Shift_R =8;
	int32_t  ping_idx=0,pong_idx;
	uint32_t rot_pong=0;
	uint32_t quantized_zero32= quantized_zero;
	
	HVX_Vector V_alpha_frac , V_alpha_offset, V_round, V_zero;
	HVX_Vector *in_ptr, *out_ptr;

	//Prepare circular buffer of size > 128. 
	if(alpha_depth <NUM_BYT_PERVECTOR){
		uint8_t *i1_ptr=alpha_frac, *i2_ptr=alpha_offset;
		uint32_t vec_depth=alpha_depth;
		i1_ptr += alpha_depth;
		i2_ptr += alpha_depth;
		while(vec_depth < NUM_BYT_PERVECTOR){
			vmemcpy_asm(i1_ptr, alpha_frac, alpha_depth);
			vmemcpy_asm(i2_ptr, alpha_offset, alpha_depth);
			i1_ptr    += alpha_depth;
			i2_ptr    += alpha_depth;
			vec_depth += alpha_depth;
		}
		circlen +=vec_depth;

	} else {
		circlen = alpha_depth;
	}


	in_ptr  = (HVX_Vector *) in_data;
	out_ptr = (HVX_Vector *) out_data;


	max_circlen_slot   = ((circlen+NUM_BYT_PERVECTOR)/NUM_BYT_PERVECTOR);

	quantized_zero32   = quantized_zero32 << 8 | quantized_zero32;
	quantized_zero32   = quantized_zero32 << 16 | quantized_zero32;

	V_round            = Q6_V_vsplat_R(0x00800080);
	V_zero             = Q6_V_vsplat_R(quantized_zero32);

	HVX_Vector *al_fr  = (HVX_Vector *) &alpha_frac[0];
	HVX_Vector *al_of  = (HVX_Vector *) &alpha_offset[0];

	V_alpha_frac       = *al_fr;
	V_alpha_offset     = *al_of;
	
	HVX_Vector fr_ping0   = V_alpha_frac;  //First 128 vector
	HVX_Vector of_ping0   = V_alpha_offset;

	
	for (i = 0; i < bytes/NUM_BYT_PERVECTOR; i++)
	{
		*out_ptr++ = prelu_1d_intrinsic(*in_ptr++, V_alpha_frac, V_alpha_offset, V_round, V_zero, Shift_R);
		
		//------------------------------------------------------------------
		// --------------------------Handle circular buffer - for efficient alpha 1 D vec access.

		//circ_idx = (circ_idx+128) % circlen;
		circ_idx +=NUM_BYT_PERVECTOR; 
		if(circ_idx >= circlen)	circ_idx -= circlen;

		pong_idx = circ_idx /NUM_BYT_PERVECTOR;
		ping_idx = pong_idx+1;
		circ_res = circ_idx & NUM_BYT_PERVECTOR_MASK; // circ_idx  - (circ_idx/NUM_BYT_PERVECTOR)*NUM_BYT_PERVECTOR;
		
		if(circ_idx + NUM_BYT_PERVECTOR > circlen){
			//first stage concat
			
			if(ping_idx >= max_circlen_slot){ //128*((circlen+128)/128))
				//corner case when we still have 2 buffers span , this can happen when circ_idx is multiple of 128
				rot_pong = circ_res;//circ_idx & NUM_BYT_PERVECTOR_MASK; 

				V_alpha_frac   = *(al_fr+pong_idx);
				V_alpha_offset = *(al_of+pong_idx);
				V_alpha_frac   = Q6_V_vror_VR(V_alpha_frac, rot_pong);
				V_alpha_offset = Q6_V_vror_VR(V_alpha_offset, rot_pong);
				
			} else
			{
				concat_arry(al_fr, ping_idx, pong_idx,circ_res, V_alpha_frac); 
				concat_arry(al_of, ping_idx, pong_idx,circ_res, V_alpha_offset); 
			}

			//second stage concat - when out buffer spans across 3 different chunk location. 
			// Eg,  1) last part of buf1, 2) first part of buf2 3)  again first part of buf1
			// Use result from previous output.
			rot_pong =  (circlen - circ_idx);
			circ_res = NUM_BYT_PERVECTOR - rot_pong; 
			
			concat_vec(fr_ping0 , V_alpha_frac  ,circ_res,rot_pong, V_alpha_frac);
			concat_vec(of_ping0 , V_alpha_offset,circ_res,rot_pong, V_alpha_offset);
			


		} else {

			concat_arry(al_fr, ping_idx, pong_idx,circ_res, V_alpha_frac); 
			concat_arry(al_of, ping_idx, pong_idx,circ_res, V_alpha_offset); 

			//printf("HVX2.0 circlen %lu circ_idx %3lu: circ_res %3lu , ping_idx %3ld, pong_idx %3ld rot ping  %3lu, %3lu \n",circlen, circ_idx, circ_res, ping_idx, pong_idx, rot_ping, rot_pong);
		}


	}//for(


	// Handle last part of unpadded inputs
	if( bytes & NUM_BYT_PERVECTOR_MASK)
	{
		out_ptr = (HVX_Vector *)&buf_pad[0];
		*out_ptr++ = prelu_1d_intrinsic(*in_ptr++, V_alpha_frac, V_alpha_offset, V_round, V_zero, Shift_R);
		vmemcpy_asm(&out_data[bytes & (~NUM_BYT_PERVECTOR_MASK)], buf_pad, bytes &  NUM_BYT_PERVECTOR_MASK);
	}


	nn_sem_post(&info->donesem);
}

//   Prelu function - to handle Alpha > 1 depth.
//  Convert float alpha to quantized (8 bit) , alpha_frac & alpha_offset.
//  	   Float to fixed 16 bit is - multi threaded inside hvx_compute_frac_offset.
//     Added duplicate alpha_frac & offset buffers, for multi threading the main prelu hvx_intr_prelu_circ function.
//
void prelu_1D_alpha(uint8_t *in_data, uint8_t *out_data, float *alpha, uint32_t quantized_zero,size_t bytes , size_t alpha_depth ,struct nn_node *self, struct nn_graph *nn)
{
	//If you're changing any of the sizes here, please update them in prelu_execute_hvx
	uint8_t  *alpha_frac1   = pad_and_align(alpha, sizeof(float)*alpha_depth+2*NUM_BYT_PERVECTOR);
	uint8_t  *alpha_frac2   = pad_and_align(alpha_frac1,   alpha_depth+2*NUM_BYT_PERVECTOR);
	uint8_t  *alpha_offset1 = pad_and_align(alpha_frac2,   alpha_depth+2*NUM_BYT_PERVECTOR);
	uint8_t  *alpha_offset2 = pad_and_align(alpha_offset1, alpha_depth+2*NUM_BYT_PERVECTOR);
	uint8_t  *buf_pad1      = pad_and_align(alpha_offset2,  alpha_depth+2*NUM_BYT_PERVECTOR);
	uint8_t  *buf_pad2      = pad_and_align(buf_pad1     ,  2*NUM_BYT_PERVECTOR);
	uint16_t *temp16_al     = pad_and_align(buf_pad2, 2*NUM_BYT_PERVECTOR);
	uint32_t idx;

	struct tdata_prelu  td[MAX_THREAD];
	td[0].self = self;
	td[1].self = self;




	hvx_float2_u16(alpha, temp16_al, alpha_depth, self, nn);

	td[0].iptr = in_data;
	td[0].optr = out_data;
	td[0].ptr1 = alpha_frac1;
	td[0].ptr2 = alpha_offset1;
	td[0].ptr3 = temp16_al;
	//td[0].ptr4 = alpha_offset2;
	td[0].arg1 = alpha_depth;
	td[0].arg2 = quantized_zero;



	nn_sem_init(&td[0].donesem,0);
	nn_os_work_for_vector(nn,do_hvx_quant_frac_offst,  &td[0]);
	nn_sem_wait(&td[0].donesem);

	if(bytes < 32*1024)
		l2fetch(in_data, 1 , NUM_BYT_PERVECTOR , bytes/NUM_BYT_PERVECTOR);
	else
		l2fetch(in_data, 1 , NUM_BYT_PERVECTOR , (32*1024)/NUM_BYT_PERVECTOR);

	// If the input data is split across the thread, the alpha vector needs to start at correct offset for second thread
	if(bytes > MIN_INP_ELEL_MULTITHREAD){
		uint32_t res;
		idx = (4+bytes/256)*128;
		res = idx % alpha_depth;

		memcpy(alpha_frac2,         &alpha_frac1[res], alpha_depth-res);
		memcpy(&alpha_frac2[alpha_depth-res],   &alpha_frac1[0], res);
		
		memcpy(alpha_offset2,       &alpha_offset1[res], alpha_depth-res);
		memcpy(&alpha_offset2[alpha_depth-res], &alpha_offset1[0]  , res);	

		
	}	
	else {
		idx = bytes;   //Disable multi threading for small input_data sizes.
	}
	
	//all other arg remain same
	td[0].iptr = in_data;
	td[0].optr = out_data;
	td[0].ptr1 = alpha_frac1;
	td[0].ptr2 = alpha_offset1;
	td[0].ptr3 = buf_pad1;
	td[0].arg0 = idx;
	td[0].arg1 = alpha_depth;
	td[0].arg2 = quantized_zero;



	td[1].iptr = &in_data[idx];
	td[1].optr = &out_data[idx];
	td[1].ptr1 = alpha_frac2;
	td[1].ptr2 = alpha_offset2;
	td[1].ptr3 = buf_pad2;
	td[1].arg0 = bytes - idx;
	td[1].arg1 = alpha_depth;
	td[1].arg2 = quantized_zero;


	nn_sem_init(&td[0].donesem,0);
	nn_os_work_for_vector(nn,hvx_intr_prelu_circ,  &td[0]);
	if(td[1].arg0) {
		nn_sem_init(&td[1].donesem,0);
		nn_os_work_for_vector(nn,hvx_intr_prelu_circ,  &td[1]);
		nn_sem_wait(&td[1].donesem);
	} 
	nn_sem_wait(&td[0].donesem);
	


}


//   Prelu function - to handle Alpha == 1 depth.
static void hvx_prelu_scalar_alpha(struct nn_graph *nn, void *vinfo)
{
	struct tdata_prelu *info = vinfo;

	float    alpha     = (float )info->arg_f;
	uint32_t bytes       = (uint32_t) info->arg1;
	uint8_t *in_data    = (uint8_t *) info->iptr;
	uint8_t *out_data   = (uint8_t *) info->optr;
	uint32_t quantized_zero = (uint32_t) info->arg0;


		int bhw;
		int32_t Shift_R =8;
		uint32_t alpha_offset_v;
		uint32_t alpha_frac = (1<<16) * alpha, alpha_frac_v;
		uint32_t alpha_offset = quantized_zero - ((quantized_zero * alpha_frac + 0x08000) >> 16);



		uint32_t quantized_zero32= quantized_zero;
		//logmsg(nn,2,"alpha=%f alpha_frac=%x/%x,alpha_offset=%x\n",
		//	alpha,alpha_frac,1<<16,alpha_offset);

		alpha_frac_v = alpha_frac >>8;
		alpha_frac_v = alpha_frac_v << 8  | alpha_frac_v;
		alpha_frac_v = alpha_frac_v << 16 | alpha_frac_v;

		alpha_offset_v = alpha_offset   << 8  | alpha_offset;
		alpha_offset_v = alpha_offset_v << 16 | alpha_offset_v;

		quantized_zero32 = quantized_zero32 << 8 | quantized_zero32;
		quantized_zero32 = quantized_zero32 << 16 | quantized_zero32;

		//printf("alpha_frac %lx, alpha_offset %lx, quantized_zero %lx\n ", alpha_frac, alpha_offset, quantized_zero32);

		HVX_Vector V_out_val;
		HVX_Vector V_in_val;
		HVX_Vector V_alpha_frac   = Q6_V_vsplat_R(alpha_frac_v);
		HVX_Vector V_alpha_offset = Q6_V_vsplat_R(alpha_offset_v);
		HVX_Vector V_round        = Q6_V_vsplat_R(0x00800080);
		HVX_Vector V_zero         = Q6_V_vsplat_R(quantized_zero32);

		HVX_Vector V_temp8, V_temp16, V_temp16_2;
		HVX_VectorPair V_pair;
		//HVX_VectorPair V_pair_2;

		HVX_VectorPred   inp_neg_pred;

		HVX_Vector *in_ptr = (HVX_Vector *) in_data;
		HVX_Vector *out_ptr = (HVX_Vector *) out_data;

		for (bhw = 0; bhw < bytes/NUM_BYT_PERVECTOR; bhw ++)
		{

			V_in_val = *in_ptr++;
			V_out_val = V_in_val;

			inp_neg_pred = Q6_Q_vcmp_gt_VubVub(V_zero, V_in_val);
			
			// alpha*in_data
		
			//(in_data[i] * alpha_frac)
			V_pair   = Q6_Wuh_vmpy_VubVub(V_in_val, V_alpha_frac);

			//(in_data[i] * alpha_frac+0x080)
			V_temp16   = Q6_Vuh_vadd_VuhVuh_sat(Q6_V_lo_W(V_pair), V_round);
			V_temp16_2 = Q6_Vuh_vadd_VuhVuh_sat(Q6_V_hi_W(V_pair), V_round);
			//V_pair_2 = Q6_Wuh_vadd_WuhWuh_sat(V_pair, V_pair_round);

			//out_data[i] = ((in_data[i] * alpha_frac+0x080) >> 8) + alpha_offset;
			V_temp16   = Q6_Vuh_vlsr_VuhR(V_temp16  , Shift_R);
			V_temp16_2 = Q6_Vuh_vlsr_VuhR(V_temp16_2, Shift_R);

			V_temp8    =  Q6_Vb_vshuffe_VbVb(V_temp16_2, V_temp16);

			//out_data[i] = ((in_data[i] * alpha_frac+0x080) >> 8) + alpha_offset;
			V_temp8 =  Q6_Vb_vadd_VbVb(V_temp8, V_alpha_offset);
 			// if( in_data < quantized_zero) out_data = alpha*in_data else out_data = in_data
			V_out_val = Q6_V_vmux_QVV(inp_neg_pred, V_temp8, V_in_val);

			*out_ptr++ = V_out_val;
		}

		if(bytes % NUM_BYT_PERVECTOR){
			int i, offset= (bytes/NUM_BYT_PERVECTOR)* NUM_BYT_PERVECTOR;
			for(i = offset; i < bytes;i++){
				out_data[i] = in_data[i];
				if (in_data[i] < quantized_zero) {
					out_data[i] = ((in_data[i] * alpha_frac+0x08000) >> 16) + alpha_offset;
					//logmsg(nn,2,"in_data[%d] = %d, out_data = %d\n",i,in_data[i],out_data[i]);
				}
			}
		}

	nn_sem_post(&info->donesem);

}


static inline void prelu_scalar_alpha(uint8_t *in_data, uint8_t *out_data, float alpha, uint32_t quantized_zero,size_t bytes ,struct nn_node *self, struct nn_graph *nn )
{

	struct tdata_prelu  td[2];

	td[0].self = self;

	td[0].iptr = in_data;
	td[0].optr = out_data;
	td[0].arg0 = quantized_zero;
	td[0].arg1 = bytes;
	td[0].arg_f = alpha;

	nn_sem_init(&td[0].donesem,0);
	nn_os_work_for_vector(nn, hvx_prelu_scalar_alpha,  &td[0]);
	nn_sem_wait(&td[0].donesem);


}



//	Prelu fixed-pt- HVX  module.
//	 Note: Multi threading is enabled for this op.
static int prelu_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *in_alpha_tensor = self->inputs[3];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	size_t alpha_depth = in_alpha_tensor->shape.depth;

	size_t bytes = in_tensor->shape.batches 
		* in_tensor->shape.height
		* in_tensor->shape.width
		* in_tensor->shape.depth;
	uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);

	//values obtained from prelu_1d_alpha
	size_t alpha_size = (sizeof(float)*alpha_depth+2*NUM_BYT_PERVECTOR);
	size_t temp16_al_size = (sizeof(uint16_t)*alpha_depth);
	size_t alpha_frac_size = (alpha_depth+2*NUM_BYT_PERVECTOR);
	size_t alpha_offset_size = (alpha_depth+2*NUM_BYT_PERVECTOR);
	size_t buf_pad_size = (2*NUM_BYT_PERVECTOR);
	size_t total_size = ROUNDUP(alpha_size) + ROUNDUP(alpha_frac_size)*2 + 
		ROUNDUP(alpha_offset_size)*2 + ROUNDUP(buf_pad_size)*2 + ROUNDUP(temp16_al_size);

	if (nn_scratch_grow(nn,total_size)) {
		return errlog(nn,"failed to get scratch");
	}

	float *alpha = nn->scratch;
	uint8_t quantized_zero = quantize_uint8(0.0f,in_min,in_max);
	uint32_t j;
	/* Assert min and max are size 1,1,1,1 ? */

	logmsg(nn,2,"Prelu execute. self=%p ",self);
	logmsg(nn,2,"Prelu in min/max=%f/%f ",
		tensor_get_float(in_min_tensor,0),
		tensor_get_float(in_max_tensor,0));
	/* Assert min and max are size 1,1,1,1 ? */

	if ((alpha_depth > 1) && (alpha_depth != in_tensor->shape.depth)) return errlog(nn,"Input depth mismatch ");

	for(j =0; j <  alpha_depth; j++) {
		alpha[j] = tensor_get_float(in_alpha_tensor,j);
		//printf("Check Alpha \n");
		if (alpha[j] < 0.0f) return errlog(nn,"negative alpha %f, %d", alpha[j],j);
		if (alpha[j] > 1.0f) return errlog(nn,"alpha must be <= 1.0f");
	}



	if (bytes > out_tensor->max_size) return errlog(nn,"out too small");
	out_tensor->shape = in_tensor->shape;
	out_tensor->data_size = bytes;


#ifdef DEBUG_PRINT_PRELU_PERFORMANCE
	int start_time =  nn_os_get_cycles(nn);
#endif
	if(alpha_depth == 1 )
		prelu_scalar_alpha(in_data, out_data, alpha[0], quantized_zero, bytes, self, nn);
	else
		prelu_1D_alpha(in_data, out_data, alpha, quantized_zero, bytes, alpha_depth, self,  nn);

#ifdef DEBUG_PRINT_PRELU_PERFORMANCE
	int end_time =  nn_os_get_cycles(nn);
	printf("Prelu cycles = %d (elements = %d)\n", (end_time-start_time), bytes);
#endif
	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

	logmsg(nn,2,"Prelu out min/max=%f/%f ",
		tensor_get_float(out_min_tensor,0),
		tensor_get_float(out_max_tensor,0));
	logmsg(nn,2,"Prelu %p done",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedPRelu_8_ref = {
	.execute = prelu_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedPRelu_8 = {
	.execute = prelu_execute_hvx,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

