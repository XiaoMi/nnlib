/*
 * Copyright (c) 2017, The Linux Foundation. All rights reserved.
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
#include <math.h>
#include "hvx_inlines.h"
#include "quantize.h"
#include "nn_asm_ops.h"
#define LRN_MAXTHREADS 2


//#define TEST_PERFORMANCE
/*
 * resize_bilinear
 * -  Input tensor (with min and max)
 * - new_dims tensor (with height and width )
 */
struct bilin_runstate;
typedef void (*run_slice_fp)( struct bilin_runstate * rstp, uint8_t * out_ptr, uint8_t const * in_ptr);

struct bilin_runstate {
	struct tensor_addressing tin;
	struct tensor_addressing tout;

	int depth;
	int batches;

	int in_ht;		// input height, width
	int in_width;
	int out_ht;		// output size
	int out_width;

	run_slice_fp run_slice_func;
	int with_hvx;

	nn_sem_t done_sem;
	int jobs;		// total # of 'subsections' to run
	volatile int next_job;	 // index of next subsection
};

static void do_bilin_interp_x2_single_slice(struct bilin_runstate * rstp, uint8_t * out_ptr, uint8_t const * in_ptr );
static void bilin_interp_work( struct nn_graph *nn, void *rstpv );
static void do_bilin_interp_x2_single_slice_HVX(struct bilin_runstate * rstp, uint8_t * out_ptr, uint8_t const * in_ptr );


static int resize_bilinear_d32_execute(struct nn_node *self, struct nn_graph *nn)
{

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *newdims_tensor = self->inputs[3];

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	logmsg(nn,2,"lrn_d32 execute. node=%p id=%x",self,self->node_id);

	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;

	// get new dims
	int new_height = tensor_get_int32( newdims_tensor, 0);
	int new_width = tensor_get_int32( newdims_tensor, 1);

	if( new_height != height*2
		||  new_width != width*2 ){
		return errlog( nn, "resizebilinear_d32: currently only x2");
	}
	int width_pad_before = in_tensor->format.width_pad[0];
	int out_width_pad_before = 4;
	int out_width_pad_after = (-(out_width_pad_before + new_width))&3;

	// construct output tensor, just like input
	//
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		batches,
		new_height, in_tensor->format.height_pad[0],in_tensor->format.height_pad[1],
		new_width,  out_width_pad_before,out_width_pad_after,
		depth, in_tensor->format.depth_pad[0],in_tensor->format.depth_pad[1],
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			out_tensor->max_size,batches,height,width,depth);
	}

	struct bilin_runstate runstate;
	runstate.tin = tensor_addressing_d32(in_tensor );
	runstate.tout = tensor_addressing_d32(out_tensor );
	runstate.in_ht = height;
	runstate.in_width = width;
	runstate.out_ht = new_height;
	runstate.out_width = new_width;
	runstate.batches = batches;
	runstate.depth = depth;
	runstate.jobs = batches * runstate.tin.nd32;
	runstate.next_job = 0;
	runstate.run_slice_func = do_bilin_interp_x2_single_slice_HVX;
	runstate.with_hvx = 1;

	// we require aligned vectors for hvx. the output is always aligned, with pad_before = 4;
	// If the input padding is multiple of 4; we're ok; if it's an odd multiple of 2,
	// we can fix by expanding the input on the left by 2, and the output by 4.
	//

	if( (width_pad_before&1) != 0 ){
		runstate.run_slice_func = do_bilin_interp_x2_single_slice;	// use scalar code
		runstate.with_hvx = 0;
	}else if((width_pad_before&2) != 0 ){
		runstate.in_width += 2;
		runstate.tin.data -= 64;	// include 2 pixels in 'gutter'
		runstate.out_width += 4;
		runstate.tout.data -= 128;	// put 4 pixels in 'gutter'.
	}

	int n_threads = min_i32(2, runstate.jobs);
	nn_sem_init( &runstate.done_sem,0);
	for(int i =0; i < n_threads; i++)
		nn_os_work_for_vector( nn, bilin_interp_work, &runstate );

	// copy the min and max through
	tensor_copy( out_min_tensor, in_min_tensor );
	tensor_copy( out_max_tensor, in_max_tensor );
	for(int i =0; i < n_threads; i++)
		nn_sem_wait( &runstate.done_sem);

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("resizebilinear_d32  cycles = %d (elements = %d)\n",
		(end_time-start_time),	(int)tensor_element_count(out_tensor));
#endif

	logmsg(nn,2,"lrn_d32 %p done",self);
	return 0;
}

static void
bilin_interp_work( struct nn_graph *nn, void *rstpv )
{
	struct bilin_runstate *rstp = (struct bilin_runstate *)rstpv;

	int nd32 = rstp->tin.nd32;				// # of depth slices to do
	///int depth = rstp->depth + rstp->tin.d0;		// include any pre-padding;

	run_slice_fp run_slice_func = rstp->run_slice_func;

	int job_idx;

	uint32_t pf_stride = rstp->tin.height_stride;
	uint32_t pf_width = rstp->in_width * 32;
	uint32_t pf_height = rstp->in_ht;

	while( job_idx = __sync_fetch_and_add(&rstp->next_job, 1), job_idx < rstp->jobs) {
		int batch_idx = 0;
		int d32_idx = job_idx;
		if( d32_idx >= nd32){
			batch_idx = (unsigned)job_idx/nd32;
			d32_idx -= batch_idx* nd32;
		}
		// ok, get pointers;
		uint8_t const *in_ptr = rstp->tin.data  + batch_idx * rstp->tin.batch_stride  + d32_idx * rstp->tin.d32_stride;
		l2fetch( in_ptr, pf_stride, pf_width, pf_height);

		uint8_t * out_ptr =     rstp->tout.data + batch_idx * rstp->tout.batch_stride + d32_idx * rstp->tout.d32_stride;

		(*run_slice_func)( rstp, out_ptr, in_ptr);

	}
	nn_sem_post( &rstp->done_sem);
}


// 'reference' single slice.
// Note, pointers here are x32 aligned but not generally 128 aligned
//
static void
do_bilin_interp_x2_single_slice(struct bilin_runstate * rstp, uint8_t * out_ptr, uint8_t const * in_ptr )
{
	int in_row_stride = rstp->tin.height_stride;
	int out_row_stride = rstp->tout.height_stride;

	int in_height = rstp->in_ht;
	int in_width = rstp->in_width;

	int64_t * prev_outp = NULL;

	for( int  irow = 0; irow < in_height; irow++ ){
		int64_t *outp = (int64_t *)(out_ptr + out_row_stride * irow*2);
		// interpolate horizontally
		// only process 2 x uint64 (16 pixels) per pass.
		//
		int64_t const *inp = (int64_t const *)(in_ptr + in_row_stride * irow);
		for( int k = 0; k < 2; k ++){
			int64_t a0 = inp[2*k];
			int64_t a1 = inp[2*k+1];
			for(int j = 0; j < in_width-1; j++){
				outp[8*j + 2*k]= a0;
				outp[8*j + 2*k + 1]= a1;
				int64_t b0 = inp[4*j + 2*k + 4];		// get next input
				int64_t b1 = inp[4*j + 2*k + 5];
				outp[8*j + 2*k + 4]= Q6_P_vavgub_PP( a0, b0);
				outp[8*j + 2*k + 5]= Q6_P_vavgub_PP( a1, b1);
				a0 = b0;
				a1 = b1;
			}
			outp[in_width*8 -8 +  2*k]= a0;			// last input to 2nd last output
			outp[in_width*8 -8 + 2*k + 1]= a1;
			outp[in_width*8 -4 +  2*k]= a0;			// andt to last as well.
			outp[in_width*8 -4 + 2*k + 1]= a1;
		}
		if( irow >= 1){
			int64_t  *op1 = (int64_t *)(out_ptr + out_row_stride *(irow*2-1));
			// irow = 1 .. in_ht-1
			// prev_outp = output irow*2-2
			// outp = output irow*2
			// op1 = output irow*2-1
			// interpolate vertically between surrounding two rows.
			// (rounding down in the h, up in the v, to balance the bias)

			for(int j = 0; j < in_width*4; j++){
				int64_t x0 = Q6_P_vavgub_PP_rnd( prev_outp[2*j], outp[2*j]);
				int64_t x1 = Q6_P_vavgub_PP_rnd( prev_outp[2*j+1], outp[2*j+1]);
				op1[2*j] = x0;
				op1[2*j+1] = x1;
			}
		}
		prev_outp= outp;
	}
	// prev_outp points to the last (even) output row.
	int64_t *outp_last = (int64_t *)(out_ptr + out_row_stride * (in_height*2-1));
	memcpy( outp_last, prev_outp, in_width*2 * 32 );	// last row same as previous)
}

//
// hvx version
//
// H-scaling
// input (AAAA =32 bytes)
//
// AAAA BBBB CCCC DDDD    EEEE FFFF GGGG HHHH
//
// output:
// AAAA A::B BBBB B::C    CCCC C::D DDDD D::E ...
//
// this inline takes the ABCD and EFGH vectors, and makes two outputs
//
static inline HVX_VectorPair
horiz_resize_x2( HVX_Vector vABCD, HVX_Vector vEFGH)
{
	HVX_Vector vBCDE = Q6_V_valign_VVR( vEFGH, vABCD, 32 );
	HVX_Vector vMid = Q6_Vub_vavg_VubVub( vABCD, vBCDE);	// A:B B:C C:D D:E
	return Q6_W_vshuff_VVR( vMid, vABCD, -32 );
}
//
// Note: this requires both input and output to be vector aligned
//
static void
do_bilin_interp_x2_single_slice_HVX(struct bilin_runstate * rstp, uint8_t * out_ptr, uint8_t const * in_ptr )
{
	int in_row_stride = rstp->tin.height_stride;
	int out_row_stride = rstp->tout.height_stride;

	int in_height = rstp->in_ht;
	int in_width = rstp->in_width;
	// if inwidth >=5 we do 1 loop
	// if  >= 9 we do do 2, etc
	int wloops = (in_width-1)>>2;		// could be 0...
	int wspare = in_width & 3;
	int store_extra = (in_width-1) & 2;
	HVX_VectorPred qright = q6op_Q_vsetq2_R( 32* wspare);

	//
	// Do first row as a single pass.
	//
	{
		HVX_Vector const *vpin = (HVX_Vector const *) in_ptr;
		HVX_Vector *vpout = (HVX_Vector *) out_ptr;
		HVX_Vector vL = *vpin++;
		for( int i =0; i < wloops; i++ ){
			HVX_Vector vR = *vpin++;
			HVX_VectorPair vout = horiz_resize_x2( vL, vR);
			vpout[0] = Q6_V_lo_W(vout);
			vpout[1] = Q6_V_hi_W(vout);
			vpout += 2;
			vL = vR;
		}
		// End-of-row business:
		// when wloops = 0
		//            vL =   A B C D
		//            vrol = D A B C
		//  we want to scale:
		//    inwid = 1      A A x x  x x x x  qright = 0..31
		//    inwid = 2      A B B x  x x x x  qright = 0..63
		//    inwid = 3      A B C C  x x x x  qright = 0..95
		//    inwid = 4      A B C D  D x x x  qright = (all)
		//
		HVX_Vector vrol = Q6_V_vror_VR( vL,-32); // DABC
		HVX_VectorPair vout = horiz_resize_x2(  Q6_V_vmux_QVV(qright, vL,vrol), vrol);
		vpout[0] = Q6_V_lo_W(vout);
		if( store_extra )	// width = 3,4,  7,8,  11,12, ...
			vpout[1] = Q6_V_hi_W(vout);
	}
	// now do two input rows, four output rows at a time
	int nrowpair = (in_height-1)>>1;
	for(int irowp = 0; irowp < nrowpair; irowp++ ){
		HVX_Vector const *vpin0 = (HVX_Vector const *) (in_ptr + (2*irowp)*in_row_stride);
		HVX_Vector const *vpin1 = (HVX_Vector const *) ( (char const*) vpin0 + in_row_stride);
		HVX_Vector const *vpin2 = (HVX_Vector const *) ( (char const*) vpin0 + 2*in_row_stride);
		HVX_Vector * vpout = (HVX_Vector *)( out_ptr + (4*irowp+1)*out_row_stride);	// first of four
		HVX_Vector vL0 = *vpin0++;
		HVX_Vector vL1 = *vpin1++;
		HVX_Vector vL2 = *vpin2++;

		for( int i =0; i < wloops; i++ ){
			HVX_Vector vR0 = *vpin0++;
			HVX_Vector vR1 = *vpin1++;
			HVX_Vector vR2 = *vpin2++;
			HVX_VectorPair vout0 = horiz_resize_x2( vL0, vR0);
			HVX_VectorPair vout1 = horiz_resize_x2( vL1, vR1);
			HVX_VectorPair vout2 = horiz_resize_x2( vL2, vR2);
			// v interp between  0 and 1 for 1st row
			vpout[0] = Q6_Vub_vavg_VubVub_rnd( Q6_V_lo_W(vout0), Q6_V_lo_W(vout1));
			vpout[1] = Q6_Vub_vavg_VubVub_rnd( Q6_V_hi_W(vout0), Q6_V_hi_W(vout1));
			// 2nd row
			HVX_Vector *vpo = (HVX_Vector  *) ( (char*) vpout + out_row_stride);
			vpo[0] = Q6_V_lo_W(vout1);
			vpo[1] = Q6_V_hi_W(vout1);
			// v interp between  1 and 2 for 3rd row
			vpo = (HVX_Vector  *) ( (char*) vpout + 2*out_row_stride);
			vpo[0] = Q6_Vub_vavg_VubVub_rnd( Q6_V_lo_W(vout1), Q6_V_lo_W(vout2));
			vpo[1] = Q6_Vub_vavg_VubVub_rnd( Q6_V_hi_W(vout1), Q6_V_hi_W(vout2));
			// 4th row
			vpo = (HVX_Vector  *) ( (char*) vpout + 3*out_row_stride);
			vpo[0] = Q6_V_lo_W(vout2);
			vpo[1] = Q6_V_hi_W(vout2);
			vpout += 2;
			vL0 = vR0;  vL1 = vR1; vL2 = vR2;
		}
		{
			// and the stuff at the end
			HVX_Vector vrol0 = Q6_V_vror_VR( vL0,-32);
			HVX_Vector vrol1 = Q6_V_vror_VR( vL1,-32);
			HVX_Vector vrol2 = Q6_V_vror_VR( vL2,-32);
			HVX_VectorPair vout0 = horiz_resize_x2(  Q6_V_vmux_QVV(qright, vL0,vrol0), vrol0);
			HVX_VectorPair vout1 = horiz_resize_x2(  Q6_V_vmux_QVV(qright, vL1,vrol1), vrol1);
			HVX_VectorPair vout2 = horiz_resize_x2(  Q6_V_vmux_QVV(qright, vL2,vrol2), vrol2);
			vpout[0] = Q6_Vub_vavg_VubVub_rnd( Q6_V_lo_W(vout0), Q6_V_lo_W(vout1));
			*( HVX_Vector *)( (char *)vpout + out_row_stride) = Q6_V_lo_W(vout1);
			*( HVX_Vector *)( (char *)vpout + 2*out_row_stride) = Q6_Vub_vavg_VubVub_rnd( Q6_V_lo_W(vout1), Q6_V_lo_W(vout2));
			*( HVX_Vector *)( (char *)vpout + 3*out_row_stride) = Q6_V_lo_W(vout2);
			if( store_extra ){	// width = 3,4,  7,8,  11,12, ...
				vpout++;
				vpout[0] = Q6_Vub_vavg_VubVub_rnd( Q6_V_hi_W(vout0), Q6_V_hi_W(vout1));
				*( HVX_Vector *)( (char *)vpout + out_row_stride) = Q6_V_hi_W(vout1);
				*( HVX_Vector *)( (char *)vpout + 2*out_row_stride) = Q6_Vub_vavg_VubVub_rnd( Q6_V_hi_W(vout1), Q6_V_hi_W(vout2));
				*( HVX_Vector *)( (char *)vpout + 3*out_row_stride) = Q6_V_hi_W(vout2);

			}
		}
	}
	// we've either
	//    - done the whole thing except the last row (which is copy of of the previous);
	//           (this is the case when the input height is odd)
	//    - or the whole thing except the last 3
	//           (this is the case when the input height is even)
	//
	HVX_Vector *vpo_a = (HVX_Vector *)( out_ptr + (in_height*2-2)*out_row_stride);	// 2nd last output
	HVX_Vector *vpo_b = (HVX_Vector *)( (char*)vpo_a  + out_row_stride);	// last output

	if( (in_height &1)!= 0 ){	// copy a to b
		int nvec = (in_width-1)>>1;
		HVX_Vector v = *vpo_a ++;
		for( int i = 0; i < nvec; i++){
			*vpo_b++ = v;
			v = *vpo_a++;
		}
		*vpo_b = v;
	}else{
		// process 2 input rows, make one v-interpolated output row and two h-interpolated
		HVX_Vector const *vpin0 = (HVX_Vector const *) (in_ptr + (in_height-2)*in_row_stride);	// 2nd last input
		HVX_Vector const *vpin1 = (HVX_Vector const *) ( (char const*) vpin0 + in_row_stride);	// last
		HVX_Vector vL0 = *vpin0++;
		HVX_Vector vL1 = *vpin1++;
		for( int i =0; i < wloops; i++ ){
			HVX_Vector vR0 = *vpin0++;
			HVX_Vector vR1 = *vpin1++;
			HVX_VectorPair vout0 = horiz_resize_x2( vL0, vR0);
			HVX_VectorPair vout1 = horiz_resize_x2( vL1, vR1);
			// v interp between  0 and 1 for 1st row of the 3
			HVX_Vector *vpo_x = (HVX_Vector  *) ( (char*) vpo_a - out_row_stride);
			vpo_x[0] = Q6_Vub_vavg_VubVub_rnd( Q6_V_lo_W(vout0), Q6_V_lo_W(vout1));
			vpo_x[1] = Q6_Vub_vavg_VubVub_rnd( Q6_V_hi_W(vout0), Q6_V_hi_W(vout1));
			// 2nd and 3rd row
			vpo_a[0] = Q6_V_lo_W(vout1);
			vpo_a[1] = Q6_V_hi_W(vout1);
			vpo_b[0] = Q6_V_lo_W(vout1);
			vpo_b[1] = Q6_V_hi_W(vout1);
			vpo_a += 2;
			vpo_b += 2;
			vL0 = vR0;  vL1 = vR1;
		}
		{
			// and the stuff at the end
			HVX_Vector vrol0 = Q6_V_vror_VR( vL0,-32);
			HVX_Vector vrol1 = Q6_V_vror_VR( vL1,-32);
			HVX_VectorPair vout0 = horiz_resize_x2(  Q6_V_vmux_QVV(qright, vL0,vrol0), vrol0);
			HVX_VectorPair vout1 = horiz_resize_x2(  Q6_V_vmux_QVV(qright, vL1,vrol1), vrol1);
			HVX_Vector *vpo_x = (HVX_Vector  *) ( (char*) vpo_a - out_row_stride);
			vpo_x[0] = Q6_Vub_vavg_VubVub_rnd( Q6_V_lo_W(vout0), Q6_V_lo_W(vout1));
			vpo_a[0] = Q6_V_lo_W(vout1);
			vpo_b[0] = Q6_V_lo_W(vout1);
			if( store_extra ){	// width = 3,4,  7,8,  11,12, ...
				vpo_x[1] = Q6_Vub_vavg_VubVub_rnd( Q6_V_hi_W(vout0), Q6_V_hi_W(vout1));
				vpo_a[1] = Q6_V_hi_W(vout1);
				vpo_b[1] = Q6_V_hi_W(vout1);
			}
		}
	}
}



static int resize_bilinear_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking resize_bilinear_d32 node %p",self);

	int k = node_check_inputs_outputs_n( self, nn, "lrn_d32", 4, 3 );
	if( k != 0 ) return k;
	logmsg(nn,2,"resize_bilinear_d32 %p check OK",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedResizeBilinear_8_d32 = {
	.execute = resize_bilinear_d32_execute,
	.check = resize_bilinear_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
