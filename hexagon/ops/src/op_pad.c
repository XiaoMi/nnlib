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
#define BUF_PAD_SIZE (ALIGN_SIZE*4)

static inline void *pad_and_align(void *ptr, unsigned long minsize)
{
	uintptr_t ptrval = (uintptr_t)(ptr);
	ptrval += minsize + (MAXPAD-1);
	ptrval &= ~(ALIGN_SIZE-1);
	return (void *)ptrval;
}


static inline void fast_memcpy(void *out_ptr, void *in_ptr, uint32_t len)
{
	if(len < MIN_COPY_SIZE)
		memcpy(out_ptr, in_ptr, len);
	else
		vmemcpy_asm(out_ptr,in_ptr, len);

}

static inline void fill_pad(char *out_ptr, uint32_t out_len, char *pad_ptr, uint32_t pad_len) {
    while (out_len > 0) {
        if (out_len >= pad_len) {
            fast_memcpy(out_ptr, pad_ptr, pad_len); out_ptr += pad_len;
            out_len -= pad_len;
        } else {
            // done
            fast_memcpy(out_ptr, pad_ptr, out_len);
            out_len = 0;
        }
    }
}

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
	const char *in = (const char *)inpv;
	char *out = (char *)outpv;
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
	struct tdata_pad *td = (struct tdata_pad *)vinfo;
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

	const char *in = (const char *)inpv;
	char *out = (char *)outpv;
	char buf_pad [ALIGN_SIZE*30];
	char *buf_pad_ptr = (char *)pad_and_align(buf_pad, ALIGN_SIZE);
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

	memset(buf_pad_ptr,padval,BUF_PAD_SIZE);
	
	fill_pad(out, pre_h_size, buf_pad_ptr, BUF_PAD_SIZE); out += pre_h_size;

	if((pre_d_size == 0) && (post_d_size == 0))
	{
	    for (h = 0; h < h_in; h++) {
		fill_pad(out, pre_w_size, buf_pad_ptr, BUF_PAD_SIZE); out += pre_w_size;
		fast_memcpy(out,(void *)in,in_d_size*w_in); in += in_d_size*w_in; out += in_d_size*w_in;
		fill_pad(out, post_w_size, buf_pad_ptr, BUF_PAD_SIZE); out += post_w_size;
	    }

	} else {
	    for (h = 0; h < h_in; h++) {
		fill_pad(out, pre_w_size, buf_pad_ptr, BUF_PAD_SIZE); out += pre_w_size;
		for (w = 0; w < w_in; w++)
		{
			fill_pad(out, pre_d_size, buf_pad_ptr, BUF_PAD_SIZE); out += pre_d_size;
			fast_memcpy(out,(void *)in,in_d_size*w_in); in += in_d_size*w_in; out += in_d_size*w_in;
			fill_pad(out, post_d_size, buf_pad_ptr, BUF_PAD_SIZE); out += post_d_size;
		}
		fill_pad(out, post_w_size, buf_pad_ptr, BUF_PAD_SIZE); out += post_w_size;
	    }
	}

	fill_pad(out, post_h_size, buf_pad_ptr, BUF_PAD_SIZE); out += post_h_size;

	nn_sem_post(&td->donesem);

}



static int pad_generic_execute(struct nn_node *self, 
	struct nn_graph *nn, 
	const struct tensor *in_tensor, 
	const struct tensor *pads_tensor, 
	uint32_t element_size, 
	int padval, int hvx_flag)
{
	//const struct tensor *in_tensor = self->inputs[0];
	//const struct tensor *pads_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	const int32_t *pads = (const int32_t *)pads_tensor->data;
	const int32_t pad_b_before = (pads_tensor->shape.width >= 1) ? pads[0+0] : 0;
	const int32_t pad_b_after = (pads_tensor->shape.width >= 1) ? pads[0+1] : 0;
	const int32_t pad_h_before = (pads_tensor->shape.width >= 2) ? pads[2+0] : 0;
	const int32_t pad_h_after = (pads_tensor->shape.width >= 2) ? pads[2+1] : 0;
	const int32_t pad_w_before = (pads_tensor->shape.width >= 3) ? pads[4+0] : 0;
	const int32_t pad_w_after = (pads_tensor->shape.width >= 3) ? pads[4+1] : 0;
	const int32_t pad_d_before =(pads_tensor->shape.width >= 4) ?  pads[6+0] : 0;
	const int32_t pad_d_after = (pads_tensor->shape.width >= 4) ? pads[6+1] : 0;
	const int32_t d_in = in_tensor->shape.depth;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t d_out = d_in + pad_d_before + pad_d_after;
	const int32_t w_out = w_in + pad_w_before + pad_w_after;
	const int32_t h_out = h_in + pad_h_before + pad_h_after;
	const int32_t b_out = b_in + pad_b_before + pad_b_after;
	const uint32_t elements_out = d_out * w_out * h_out * b_out;
	const uint32_t bytes_out = elements_out * element_size;
	uint8_t *in_base = (uint8_t *)in_tensor->data;
	uint8_t *inp;
	uint8_t *out_base = (uint8_t *)out_tensor->data;
	uint8_t *outp;
	int b;
	struct tdata_pad td;

	logmsg(nn,2,"in tensor: %dx%dx%dx%d",b_in,h_in,w_in,d_in);
	logmsg(nn,2,"pads: %d,%dx%d,%dx%d,%dx%d,%d",
		pad_b_before,pad_b_after,
		pad_h_before,pad_h_after,
		pad_w_before,pad_w_after,
		pad_d_before,pad_d_after);
	//if (pads_tensor->shape.depth != 2) return errlog(nn,"bad pad shape");
	//if (pads_tensor->shape.width > 4) return errlog(nn,"bad pad shape");
	//if (pads_tensor->shape.width < 4) return errlog(nn,"bad pad shape");
	//if (pads_tensor->shape.height != 1) return errlog(nn,"bad pad shape");
	//if (pads_tensor->shape.batches != 1) return errlog(nn,"bad pad shape");
	if (pad_b_before || pad_b_after) return errlog(nn,"can't pad batches");
	if (bytes_out > out_tensor->max_size) return errlog(nn,"out too small");
	tensor_set_shape(out_tensor,b_out,h_out,w_out,d_out);
	out_tensor->data_size = bytes_out;

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

		    do_pad_edge_hvx(nn,  &td);

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
	return bytes_out;
}

static int pad_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	int ret;
#ifdef DEBUG_PRINT_GENERIC_PAD_REF_PERFORMANCE
	uint32_t start_time =  nn_os_get_cycles(nn);
#endif
	
	ret = pad_generic_execute(self,nn,self->inputs[0],self->inputs[1],sizeof(float),0, 0);
#ifdef DEBUG_PRINT_GENERIC_PAD_REF_PERFORMANCE
	uint32_t end_time =  nn_os_get_cycles(nn);
	printf("Pad -f cycles = %ld (elements = %d)\n", (end_time-start_time), ret);
#endif
	return 0;
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

	if (self->n_inputs == 5) {
		pad_val_f = tensor_get_float(self->inputs[4],0);
	}
	int padval = quantize_uint8(pad_val_f,in_min,in_max);
	tensor_copy(out_min_tensor,in_min_tensor);
	tensor_copy(out_max_tensor,in_max_tensor);

#ifdef	DEBUG_PRINT_EDGE_PAD_HVX_PERFORMANCE
	uint32_t start_time =  nn_os_get_cycles(nn);
#endif
	ret = pad_generic_execute(self,nn,in_tensor,in_pads_tensor,sizeof(uint8_t),padval, USE_HVX_FLAG);
#ifdef	DEBUG_PRINT_EDGE_PAD_HVX_PERFORMANCE
	uint32_t end_time=  nn_os_get_cycles(nn);
	printf("Pad  HVX cycles = %lu (elements = %d)\n",  (end_time-start_time), ret);
#endif

	return 0;
}

static int pad_f_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"pad node %p",self);
	if (self->n_inputs != 2) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 1) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"pad %p check OK",self);
	return 0;
}

static int pad_q_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"qpad node %p",self);
	if (self->n_inputs < 4) return errlog(nn,"wrong # inputs");
	if (self->n_inputs > 5) return errlog(nn,"wrong # inputs");
	if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");
	logmsg(nn,2,"pad %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Pad_f = {
	SFINIT(.execute, pad_f_execute),
	SFINIT(  .check, pad_f_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

struct nn_node_ops nn_ops_for_QuantizedPad_8 = {
	SFINIT(.execute, pad_q_execute),
	SFINIT(  .check, pad_q_check),
	SFINIT(   .ctor, node_alloc_common),
	SFINIT(   .dtor, node_free_common),
};

