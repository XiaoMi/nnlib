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
#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#include <stdio.h>
//#define  DEBUG_PRINT_GENERIC_PAD_REF_PERFORMANCE
//#define  DEBUG_PRINT_EDGE_PAD_HVX_PERFORMANCE
#define ALIGN_SIZE 128
#define NUM_BYT_PERVECTOR (ALIGN_SIZE)
#define NUM_BYT_PERVECTOR_MASK (NUM_BYT_PERVECTOR - 1)
#define MAXPAD (ALIGN_SIZE)
#define MAX_THREAD (2)  // Thread <= 2 is supported
#define MIN_COPY_SIZE (32)
#define USE_HVX_FLAG  (1)

#if 0
static inline void fast_memcpy(void *out_ptr, void *in_ptr, uint32_t len)
{
	if(len < MIN_COPY_SIZE)
		memcpy(out_ptr, in_ptr, len);
	else
		vmemcpy_asm(out_ptr,in_ptr, len);

}
#endif

struct tdata_pad {
	struct nn_node *self;
	void * iptr;
	void * optr;
	int    h_in;
	int    w_in;
	int    d_in;
	int    pad_h_before;
	int    pad_h_after;
	int    pad_w_before;
	int    pad_w_after;
	int    pad_d_before;
	int    pad_d_after;
	int    element_size;
	int    padval;
	nn_sem_t donesem;
};



static inline void do_pad(
	void *outpv,
	const void *inpv,
	const int32_t h_in,
	const int32_t w_in,
	const int32_t d_in,
	const int32_t pre_h,
	const int32_t post_h,
	const int32_t pre_w,
	const int32_t post_w,
	const int32_t pre_d,
	const int32_t post_d,
	const int32_t element_size,
	const int32_t padval)
{
	const char *in = inpv;
	char *out = outpv;
	//int h_out = h_in + pre_h + post_h;
	int w_out = w_in + pre_w + post_w;
	int d_out = d_in + pre_d + post_d;
	int out_depth_size = d_out * element_size;
	int out_width_size = w_out * out_depth_size;
	int pre_h_size = out_width_size * pre_h;
	int post_h_size = out_width_size * post_h;
	int pre_w_size = out_depth_size * pre_w;
	int post_w_size = out_depth_size * post_w;
	int pre_d_size = element_size * pre_d;
	int post_d_size = element_size * post_d;
	int in_d_size = d_in * element_size;
	int h,w;

	memset(out,padval,pre_h_size); out += pre_h_size;
	for (h = 0; h < h_in; h++) {
		memset(out,padval,pre_w_size); out += pre_w_size;
		for (w = 0; w < w_in; w++) {
			memset(out,padval,pre_d_size); out += pre_d_size;
			memcpy(out,in,in_d_size); in += in_d_size; out += in_d_size;
			memset(out,padval,post_d_size); out += post_d_size;
		}
		memset(out,padval,post_w_size); out += post_w_size;
	}
	memset(out,padval,post_h_size); out += post_h_size;
}



//  Hvx implementation . Intrinsic/ASM 
static void do_pad_edge_hvx(struct nn_graph *nn, void *vinfo)

/*
static inline void do_pad_edge_hvx(
	void *outpv,
	const void *inpv,
	const int32_t h_in,
	const int32_t w_in,
	const int32_t d_in,
	const int32_t pre_h,
	const int32_t post_h,
	const int32_t pre_w,
	const int32_t post_w,
	const int32_t pre_d,
	const int32_t post_d,
	const int32_t element_size,
	const int32_t padval)
*/
{
	struct tdata_pad *td = vinfo;
	void *inpv  =  td->iptr;
	void *outpv =  td->optr;
	int32_t h_in = td->h_in;
	int32_t w_in = td->w_in;
	int32_t d_in = td->d_in;
	int32_t pre_h  = td->pad_h_before;
	int32_t post_h = td->pad_h_after;
	int32_t pre_w  = td->pad_w_before;
	int32_t post_w = td->pad_w_after;
	int32_t pre_d  = td->pad_d_before;
	int32_t post_d = td->pad_d_after;
	int32_t element_size = td->element_size;
	int32_t padval       = td->padval;

	const char *in = inpv;
	char *out = outpv;
	//int h_out = h_in + pre_h + post_h;
	int w_out = w_in + pre_w + post_w;
	int d_out = d_in + pre_d + post_d;
	int out_depth_size = d_out * element_size;
	int out_width_size = w_out * out_depth_size;
	int pre_h_size = out_width_size * pre_h;
	int post_h_size = out_width_size * post_h;
	int pre_w_size = out_depth_size * pre_w;
	int post_w_size = out_depth_size * post_w;
	int pre_d_size = element_size * pre_d;
	int post_d_size = element_size * post_d;
	int in_d_size = d_in * element_size;

	if( element_size==1) padval = Q6_R_vsplatb_R(padval);
	else if ( element_size == 2) padval = Q6_R_combine_RlRl(padval,padval);


	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );
	// !! Do not return from this function without doing nn_mcmanager_wait( nn, &mcman ); !!

	// top
	if( pre_h_size > 0){
		nn_mcmanager_vmemset32_2d(nn, &mcman,  out, padval, pre_h_size, 1, 0);	// 'single row' fill
		out += pre_h_size;
	}

	int any_d = pre_d_size | post_d_size;
	int any_w = pre_w_size | post_w_size;
	
	if(any_d == 0 || any_w == 0)
	{
		char * out1 = out;

		// paramaters of the copy
		int rows, copywid,pre_wid, post_wid, instride,outstride;
		if( any_d == 0){
			// no padding in d; maybe in w
			rows = h_in;
			instride = in_d_size * w_in;
			outstride = out_width_size;
			pre_wid = pre_w_size;
			copywid =  in_d_size * w_in;
			post_wid = post_w_size;
			if( any_w == 0){
				// neither! make it a '1-row copy'
				copywid *= rows;
				rows  = 1;
				// the pre_wid, post_wid are 0 and the strides don't matter now
			}
		}else{		// d padding, but no w: combine h*w dims
			rows = h_in * w_in;
			instride = in_d_size;
			outstride = out_depth_size;
			pre_wid = pre_d_size;
			copywid = in_d_size;
			post_wid = post_d_size;
		}
		// 2d memset the left side
		if( pre_wid> 0 ){
			nn_mcmanager_vmemset32_2d(nn, &mcman,  out1, padval, pre_wid, rows, outstride );
			out1 += pre_wid;
		}
		// 2d copy the middle
		nn_mcmanager_vmemcpy_2d( nn, &mcman,
				copywid, rows,		// width, height
				out1,   outstride,				// outp, out_stride
				in, instride );						// inp, in_stride
		out1 += copywid;

		if( post_wid > 0){
			nn_mcmanager_vmemset32_2d(nn, &mcman,  out1, padval, post_wid, rows, outstride );
		}

	} else {
		// W *and * D padding....
		int in_w_size =  in_d_size * w_in;
		char * out1 = out;
		if(pre_w_size){
			nn_mcmanager_vmemset32_2d(nn, &mcman, out1, padval,pre_w_size, h_in, out_width_size );
			out1 += pre_w_size;
		}
		// loop over h, doing memsets for the depth padding
		for( int h = 0; h < h_in; h++){
			char * out2 = out1 + out_width_size * h;		// posn in row, after width padding
			char const * in2 = in + in_w_size*h;
			if( pre_d_size > 0){
				nn_mcmanager_vmemset32_2d(nn, &mcman,  out2, padval, pre_d_size, w_in, out_depth_size);
				out2 += pre_d_size;
			}
			// 2d copy the middle
			nn_mcmanager_vmemcpy_2d( nn, &mcman,
					in_d_size, w_in,		// width, height
					out2,  out_depth_size,				// outp, out_stride
					in2, in_d_size );						// inp, in_stride
			out2 += in_d_size;
			if( post_d_size >  0)
				nn_mcmanager_vmemset32_2d(nn, &mcman,  out2, padval, post_d_size, w_in, out_depth_size);
		}
		if(post_w_size){
			out1 += out_depth_size*w_in;
			nn_mcmanager_vmemset32_2d(nn, &mcman, out1, padval,post_w_size, h_in, out_width_size );
		}
	}
	out += h_in * out_width_size;		// skip all the core rows

	// bottom
	if( post_h_size > 0){
		nn_mcmanager_vmemset32_2d(nn, &mcman,  out, padval, post_h_size, 1, 0);	// 'single row' fill
	}
	nn_mcmanager_wait( nn, &mcman );
	//nn_sem_post(&td->donesem);

}

// 'pads_tensor is an array [1,1,n,2]:
//     [  pad_b_before, pad_b_after],
//     [  pad_h_before, pad_h_after],
//     [  pad_w_before, pad_w_after],
//     [  pad_d_before, pad_d_after]]
// (if w < 4, elements on the bottom are assumed to be 0)



static int pad_generic_execute(struct nn_node *self, 
	struct nn_graph *nn, 
	const struct tensor *in_tensor, 
	const struct tensor *pads_tensor, 
	uint32_t element_type,
	int padval, int hvx_flag)
{
	//const struct tensor *in_tensor = self->inputs[0];
	//const struct tensor *pads_tensor = self->inputs[1];
	if( pads_tensor->shape.depth != 2) return errlog(nn,"bad pad tensor");
	unsigned padt_len = pads_tensor->shape.width;
	if( padt_len> 4) padt_len = 4;		// ignore > 4

	struct tensor *out_tensor = self->outputs[0];
	const int32_t *pads = pads_tensor->data;
	const uint32_t element_size = tensor_type_size( element_type);

	// extract the pads, based on w dimension; ensure all are >=0 and
	//
	unsigned padby[4*2] = {0,0, 0,0, 0,0, 0,0};
	for( int i = 0; i < (int)padt_len*2; i++ ){
		int p = pads[i];
		if( p < 0) return errlog(nn,"pad bad tensor");
		padby[i] = p;
	}
	// find the new shape; validate sanity
	struct shape out_shape;
	uint32_t new_shape_count = 1;
	for( int i =0; i < 4; i++){
		unsigned p_before = padby[2*i];
		unsigned p_after = padby[2*i+1];
		unsigned old_dim = in_tensor->shape.dimension[i];
		uint64_t all_dim = (uint64_t)old_dim + (uint64_t)p_before + (uint64_t)p_after;
		if( all_dim > (uint64_t)0x7FFFFFFF) return errlog(nn,"padded size overflow");
		uint32_t new_dim = (uint32_t)all_dim;
		out_shape.dimension[i] = new_dim;
		new_shape_count = mulu32_sat( new_shape_count, new_dim);
	}
	if (new_shape_count ==0 || new_shape_count == (uint32_t)-1
			|| mulu32_sat( new_shape_count, element_size) == (uint32_t)-1 )
		return errlog(nn,"padded size overflow");

	const int32_t pad_b_before = padby[0];
	const int32_t pad_b_after  = padby[1];
	const int32_t pad_h_before = padby[2];
	const int32_t pad_h_after  = padby[3];
	const int32_t pad_w_before = padby[4];
	const int32_t pad_w_after  = padby[5];
	const int32_t pad_d_before = padby[6];
	const int32_t pad_d_after  = padby[7];

	const int32_t d_in = in_tensor->shape.depth;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t d_out = out_shape.depth;
	const int32_t w_out = out_shape.width;
	const int32_t h_out = out_shape.height;
	const int32_t b_out = out_shape.batches;
	uint8_t *in_base = in_tensor->data;
	uint8_t *inp;
	uint8_t *out_base = out_tensor->data;
	uint8_t *outp;
	int b;
	struct tdata_pad td;

	logmsg(nn,2,"in tensor: %dx%dx%dx%d",b_in,h_in,w_in,d_in);
	logmsg(nn,2,"pads: %d,%dx%d,%dx%d,%dx%d,%d",
		pad_b_before,pad_b_after,
		pad_h_before,pad_h_after,
		pad_w_before,pad_w_after,
		pad_d_before,pad_d_after);
	if (pad_b_before || pad_b_after) return errlog(nn,"can't pad batches");

	if( tensor_out_prepare_normal_fromshape( out_tensor, &out_shape, element_type)!=0)
		 return errlog(nn,"out too small");

	for (b = 0; b < b_out; b++) {
		inp = in_base + b*h_in*w_in*d_in*element_size;
		outp = out_base + b*h_out*w_out*d_out*element_size;

	
		if(hvx_flag)
		{
		    int bytes = h_in*w_in*d_in*element_size;

		    td.self = self;
		    td.iptr = inp;
		    td.optr = outp;
		    td.h_in = h_in;
		    td.w_in = w_in;
		    td.d_in = d_in;
		    td.pad_h_before = pad_h_before;
		    td.pad_h_after  = pad_h_after;
		    td.pad_w_before = pad_w_before;
		    td.pad_w_after  = pad_w_after;
		    td.pad_d_before = pad_d_before;
		    td.pad_d_after  = pad_d_after;
		    td.padval       = padval;
		    td.element_size = element_size;


		    if(bytes < 32*1024)
			l2fetch(inp, 1 , NUM_BYT_PERVECTOR , bytes/NUM_BYT_PERVECTOR);
		    else
			l2fetch(inp, 1 , NUM_BYT_PERVECTOR , (32*1024)/NUM_BYT_PERVECTOR);

		    // do_pad_edge doesn't need vector thread any more
		    do_pad_edge_hvx(nn, &td);

		    //nn_sem_init(&td.donesem,0);
		    //nn_os_work_for_vector(nn,do_pad_edge_hvx, &td);
		    //nn_sem_wait(&td.donesem);

		} else {
		    do_pad( outp,
			inp,
			h_in,
			w_in,
			d_in,
			pad_h_before,
			pad_h_after,
			pad_w_before,
			pad_w_after,
			pad_d_before,
			pad_d_after,
			element_size,
			padval);
		}
	}
	return 0;
}

static int pad_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	int ret;
#ifdef DEBUG_PRINT_GENERIC_PAD_REF_PERFORMANCE
	uint32_t start_time =  nn_os_get_cycles(nn);
#endif
	
	ret = pad_generic_execute(self,nn,self->inputs[0],self->inputs[1],NN_TYPE_FLOAT,0, 0);
#ifdef DEBUG_PRINT_GENERIC_PAD_REF_PERFORMANCE
	uint32_t end_time =  nn_os_get_cycles(nn);
	printf("Pad -f cycles = %ld (elements = %d)\n", (end_time-start_time), self->outputs[0].data_size);
#endif
	return ret;
}

static int pad_q_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *in_pads_tensor = self->inputs[3];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float pad_val_f = 0.0f;
	int ret;

	if (self->n_inputs > 4) {
		pad_val_f = tensor_get_float(self->inputs[4],0);
	}

	if (self->n_inputs > 5 && tensor_get_int32(self->inputs[5],0)) {
		if (pad_val_f < in_min || pad_val_f > in_max) {
			return errlog(nn, "Pad value (%f) is outside the range of inputs (%f,%f)", pad_val_f, in_min, in_max);
		}
	}

	int padval;
	if( fabsf(pad_val_f) != INFINITY ){
		padval = roundf_i32(255.0f * (pad_val_f - in_min) /(in_max-in_min));
		// quantization creates inaccuracies if the pad val is outside the range of the inputs
		// allow some leeway for rounding errors
		if (padval  < -2 || padval > 257 ) {
			return errlog(nn,"Pad value outside the range of inputs");
		}
		padval = saturate_u8( padval );
	}else{
		// treat -/+ inf as 0 or 255
		padval = (pad_val_f < 0)? 0 : 255;
	}
	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

#ifdef	DEBUG_PRINT_EDGE_PAD_HVX_PERFORMANCE
	uint32_t start_time =  nn_os_get_cycles(nn);
#endif
	ret = pad_generic_execute(self,nn,in_tensor,in_pads_tensor, NN_TYPE_QUINT8,padval, USE_HVX_FLAG);
#ifdef	DEBUG_PRINT_EDGE_PAD_HVX_PERFORMANCE
	uint32_t end_time=  nn_os_get_cycles(nn);
	printf("Pad  HVX cycles = %lu (elements = %d)\n",  (end_time-start_time), self->outputs[0].data_size);
#endif

	return ret;
}

static int pad_f_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"pad node %p",self);
	int k = node_check_inputs_outputs_n( self,nn, "pad", 2,1);
	if (k!=0)
		return k;
	logmsg(nn,2,"pad %p check OK",self);
	return 0;
}

static int pad_q_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"qpad node %p",self);
	int k = node_check_inputs_range( self, nn, "pad", 4, 6 );
	if(k==0) k = node_check_outputs_n( self, nn, "pad", 3 );
	if (k!=0)
		return k;
	logmsg(nn,2,"pad %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Pad_f = {
	.execute = pad_f_execute,
	.check = pad_f_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedPad_8 = {
	.execute = pad_q_execute,
	.check = pad_q_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

