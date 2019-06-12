
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
 * This contains implementations for quantized l2 pooling node
 */
//#define TEST_PERFORMANCE

// Parameters are:
// - input tensor
// -  window shape [1,WinH,WinW,1]
// -  stride shape [1,StrH,StrW,1]
//
// roughly:
//   - at each point [h,w]:
//       (1) find sum-of-squares of input, over the window [WinH,WinW] centered
//         at [h,w]
//       (2) find the mean of all of these, over the window size (not counting values
//           'outside' the input, if any)
//       (3) the result is square root of the mean.
//   - if the 'stride' dimensions are >1 the output is decimated by the given
//     amounts
//
//  Same operation is iterated across B and D dimensions. Output size
// is roughly [b_in,  h_in/StrH, w_in/StrW, d_in]
// .. where the actual h_out and w_out depend on the window size and padding modes, as well
// as the input size; and window size when PAD_VALID.


//
// For the case where the integral buffer is used
//   - the integral buffer finds 32-bit accumulations of pixels; each 'pixel' is actually
//     (pixval-inzero)**2
//
//   - If any output windows are reduced by edge clipping, then padding rows/cols
//    will be added to integral buffer so that the correct results are produced by including
//     these padded rows/cols in the constant window size.
//

// Scaling:
//    The input has in_range = in_max - in_min		=> in_step = in_range/255
//    The output has _out_range = out_max           => out_step = out_max/255
//
// Scaling is done in two steps:
//   (1) the 'square' of each input will be in range 0..65025 worst case, so if the windowsize
//      is w, the 32-bit sums will be < w *64k
//      We can record the range of this, to determine the output range needed.
//   (2) these sums are >> initial_shift, which brings all to < 64k.
//      An upper limit on 'initial_shift' can be calculated from the window size and 'zero point' if
//      the data.
//   (3) For the output, we take the square root of that result, with an additional scale factor, and
//     that becomes the output.
//
// For instance, if the input zero point is 45, the max per-pixel value is (255-45)^2 = 44100;
// if the window is 5x5, the max sum is 25*44100 = 1102500; so initial_shift of 5 is guaranteed
//   to give results <= 34453.
//
// If we know (or suspect) that the actual range of data allows a smaller right shift, it can be made smaller
// to preserve precision.
//  Once the value is < 64K, we find its square root (with 4 fractional bits); the upper limit of that
// will be 2900..4096 , depending on how the largest value from step (2) falls in range 32K ..64K
//  The result is then multiplied by a scale factor (15 fractional bits) which maps the upper limit
//  to 255 (this will be in range 2000 .2900 roughly).
//
//
#include <nn_graph.h>
#include <string.h>

#include "nn_atomic.h"
#include "quantize.h"
#include "hvx_mathops.h"


#define NN_INTEGRAL_BUF_FOR_L2POOL
#include "nn_integral_buffer.h" // much of the code for integral buffer management

#define L2POOL_MAX_THREADS 2



struct l2pool_runstate;
typedef void( *l2pool_process_slice_fp)( struct l2pool_runstate const *ibp,  void * integ_buf, int batch_idx, int d32_idx  );

struct l2pool_thrinfo {
	struct l2pool_runstate * rstp;		// pointer to container;
	void  * integ_buf;					// integral buf allocated to thread
	int job0, job_end;			// thread does jobs with  job0 <= i < job_end
	int16_t thr_index;			// 0,1 ..
};


struct l2pool_runstate
{
	struct integral_buffer_plan const *ibp;
	int window_size;			// wid * ht
	float input_range_min, input_range_max;
	float input_step;			// each input quant = this value
	float output_range_max;		// output_range_min is always 0.

	int n_threads;		// # of threads running.
	int jobs;
	nn_sem_t done_sem;
	l2pool_process_slice_fp process_slice_func;

	struct l2pool_thrinfo thrinfo[L2POOL_MAX_THREADS];
};

////////////////////////////////////////
// subproblem for ranging:
// given a 'maxsum' which is the largest integer sum-of-squares,
//  find:
//     -  output_max value
//     - the initial_shift (before square root) and
//     - the scale_factor applied after.
//
static inline void
do_scaling_based_on_max_winsum( struct l2pool_runstate *rstp, struct integral_buffer_plan *ibp, int max_winsum )
{
	max_winsum = max_i32( max_winsum, 256);	// don't consider values < this

	// determine the max output value in application units.
	float out_max = sqrtf( max_winsum/(float)rstp->window_size) * rstp->input_step;
	// how much do I need to >> max_winsum to make it fit in u16;
	int initial_shift = (max_winsum <= 0xFFFF)? 0 : (floor_log2( max_winsum)-15);

	// the following calc. will be done on each input sum to make an output code:
	//    (a)   t1 = input_sum * 2^-initial_shift
	//    (b)   t2 =  sqrt(t1)*16
	//    (c)   out_code = scale_fac * t2 * 2^-15
	//
	// So out_code = sqrt( input_sum ) * scale_fac * 2^-(initial_shift/2+11)
	//  The scaling we want is:
	//      out_code = (1/out_step) * sqrt(  input_sum * in_step^2 / winsize )
	//  => out_code =  sqrt(input_sum) *   in_step/out_step * sqrt(1/winsize);
	//
	// so we need to resolve
	//       scale_fac * 2^-(initial_shift/2+11)  = in_step/out_step * sqrt(1/winsize);
	// =>
	//      scale_fac = in_step/ ( out_step * sqrt( winsize * 2^-(initial_shift+22)) )
	//            =     in_step*255/( out_max * sqrt( winsize * 2^-(initial_shift+22)) )

	float sqrtw = sqrtf( flt_ldexp( (float)rstp->window_size, -(initial_shift+22)) );
	float scale_fac = rstp->input_step *255.0f/ ( out_max * sqrtw);

	// in extreme cases - small window and/or most things close to zero -
	// the scale factor will be

	if( scale_fac > 32767.0f){
		// should only happen with small window, or small values, initial_shift pegged at 0.
		// reduce scale_fac by expanding output range.
		scale_fac = 32767.0f;
		out_max = 255.0f*rstp->input_step/(scale_fac * sqrtw);
	}
	// @@ remove these from runstate if not used
	ibp->recip_shift = initial_shift;
	ibp->recip_mant =  roundf_i32( scale_fac);
	rstp->output_range_max = out_max;
}

//
// do the initial scaling calc
//
static void do_l2pool_scaling( struct l2pool_runstate *rstp, struct integral_buffer_plan *ibp, float in_min, float in_max )
{
	float in_range = in_max - in_min;
	float inscl = 255.0f/in_range;
	rstp->input_range_min = in_min;
	rstp->input_range_max = in_max;
	rstp->input_step = flt_div_255(in_range);

	// find zero code for input; this is needed for the 'producer' operation
	int inzero = saturate_u8( roundf_i32( -in_min*inscl));
	ibp->zero_code = inzero;

	// find the worst-case sum
	int winsize = ibp->window_ht * ibp->window_wid;
	rstp->window_size = winsize;
	int max_sum = max_i32( inzero, 255-inzero);	// largest input code (abs value)
	max_sum = max_sum *max_sum * winsize;
	do_scaling_based_on_max_winsum( rstp, ibp, max_sum);
}


static void l2pool_worker_thread( struct nn_graph *nn, void *thrpv );

static void l2pool_process_slice( struct l2pool_runstate const *ibp,  void * integ_buf, int batch_idx, int d32_idx  );
static void l2pool_process_slice_hvx( struct l2pool_runstate const *ibp,  void * integ_buf, int batch_idx, int d32_idx  );


static int l2pool_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *window_tensor = self->inputs[3];
	const struct tensor *stride_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];


#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif


	int use_hvx = self->node_type == OP_QuantizedL2Pool_8_d32;
	// integral_buffer_plan is attached as persistent memory, (initially cleared to 0)
	// so that it doesn't need to be recalculated every time.
	struct integral_buffer_plan * ibp = (struct integral_buffer_plan *) self->opaque;
	if( ibp == NULL)
		return errlog(nn,"no IPB!!");
	
	// set up the plan. Note, this doesn't check batches = depth = 1 on stride and window.
	// 'check' does that.

	int k = setup_integral_buffer_plan( self, nn, ibp, in_tensor, window_tensor, stride_tensor);
	if( k != 0){			// new plan was made
		if( k <0)   // there was a problem.
			return k;
	}

	// if ok, then ibp->outshape is the output shape
	int top_bottom_pad = (ibp->outshape.height == 1) ? 1 : 4;
	int out_wpad_0 =  4;   /*always 4*/ //(ibp->outshape.width == 1)? 0:4;
	int out_wpad_1 = (-(out_wpad_0 + ibp->outshape.width)) & 3;
	int out_dpad_0 = 0;
	int out_dpad_1 = (-ibp->outshape.depth)&31;


	if (tensor_out_prepare_padded_d32(out_tensor,
		ibp->outshape.batches,
		ibp->outshape.height, top_bottom_pad,top_bottom_pad,
		ibp->outshape.width,	out_wpad_0, out_wpad_1,
		ibp->outshape.depth, out_dpad_0, out_dpad_1,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail %dx(%d+%d+%d)x(%d+%d+%d)x(%d+%d+%d) %d > %d?",
				ibp->outshape.batches,
				ibp->outshape.height, top_bottom_pad,top_bottom_pad,
				ibp->outshape.width,	out_wpad_0, out_wpad_1,
				ibp->outshape.depth, out_dpad_0, out_dpad_1,
				shape_element_count( &ibp->outshape),
			out_tensor->max_size);
	}
	// find all the strides etc
	ibp->tout = tensor_addressing_d32( out_tensor );

	logmsg(nn,2,"l2pool %dx%d s=%dx%d (pad %d) %dx%dx%dx%d %dx%dx%dx%d n_d32=%d",
			ibp->window_ht, ibp->window_wid,
			ibp->stride_ht, ibp->stride_wid, self->padding,
			ibp->inshape.batches, ibp->inshape.height, ibp->inshape.width, ibp->inshape.depth,
			ibp->outshape.batches, ibp->outshape.height, ibp->outshape.width, ibp->outshape.depth,
			ibp->tout.nd32);


	struct l2pool_runstate runstate;
	runstate.ibp = ibp;

	do_l2pool_scaling( &runstate, ibp, tensor_get_float(in_min_tensor,0), tensor_get_float(in_max_tensor,0) );

	logmsg(nn,2,"l2pool: winsize=%d,  in_range = %f..%f in_zero = %d; shift=%d, fact=%d, out_range= %f",
			runstate.window_size,
			runstate.input_range_min, runstate.input_range_max,
			ibp->zero_code, ibp->recip_shift, ibp->recip_mant ,
			runstate.output_range_max );

	runstate.process_slice_func = (!use_hvx) ? l2pool_process_slice: l2pool_process_slice_hvx;

	// figure out how many 'jobs' we need, how many threads to launch
	int njobs = ibp->inshape.batches * ibp->tin.nd32;
	runstate.jobs = njobs;
	int n_threads = min_i32(L2POOL_MAX_THREADS, njobs );
	nn_scratch_reset( nn );		// reset 'scratch' pointer



#if 0
printf("%d threads for %d jobs; each with %d bytes scratch\n", n_threads, njobs, integ_buf_bytes );
printf("buffer rows: %d row0 =%d, initial load = %d\n", ibp->ibuf_rows, ibp->ibuf_row0, ibp->ibuf_initial_load);
printf("top_padding = %d; bottom = %d; infeas_h = %d\n",ibp->wpad_top, ibp->wpad_bottom, ibp->infeas_pad_ht );
printf("left_padding = %d; right = %d; infeas_w = %d\n",ibp->wpad_left, ibp->wpad_right, ibp->infeas_pad_wid );
#endif


	void * integ_buf = NULL;
	int integ_buf_bytes =0;

	void (*l2pool_worker_thread_func)( struct nn_graph *, void *);

	l2pool_worker_thread_func = l2pool_worker_thread;

	{
		// allocate scratch for the integral buffers
		integ_buf_bytes = ibp->ibuf_total_bytes;
		integ_buf = nn_scratch_alloc(nn, integ_buf_bytes*n_threads);
		if( integ_buf == NULL){
			return errlog(nn, "could not alloc %d bytes of scratch",integ_buf_bytes*n_threads );
		}
	}
	// fill in the 'thrinfo' and launch threads
	runstate.n_threads = n_threads;
	nn_sem_init(&runstate.done_sem,0);
	runstate.thrinfo[0].job0 = 0;
	runstate.thrinfo[0].job_end = njobs;
	runstate.thrinfo[0].thr_index = 0;
#if L2POOL_MAX_THREADS > 1
#if L2POOL_MAX_THREADS > 2
#error "assuming <=2 threads"
#endif
	if( n_threads > 1){
		int split = (njobs+1)>>1;
		runstate.thrinfo[0].job_end = split;
		runstate.thrinfo[1].job0 = split;
		runstate.thrinfo[1].job_end = njobs;
		runstate.thrinfo[1].thr_index = 1;
	}
#endif

	for(int i=0; i < n_threads; i++){
		runstate.thrinfo[i].rstp = &runstate;
		runstate.thrinfo[i].integ_buf = integ_buf;
		integ_buf = (void*)( (char*)integ_buf + integ_buf_bytes );
		nn_os_work_for_vector(nn, l2pool_worker_thread_func , &runstate.thrinfo[i]);
	}

	for(int i=0; i < n_threads; i++){
		nn_sem_wait( &runstate.done_sem);
	}

	if (tensor_set_single_float(out_min_tensor,0.0f) != 0
		|| tensor_set_single_float(out_max_tensor,runstate.output_range_max) != 0) {
		return errlog(nn,"min or max out prep fail");
	}


#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("l2pool_d32 %s cycles = %d (elements = %d->%d) thr=%d\n",
			use_hvx?"hvx": "ref",(end_time-start_time),
					(int)tensor_element_count(in_tensor),
			(int)tensor_element_count(out_tensor), n_threads);
#endif
	logmsg(nn,2,"l2pool_d32 %p done", self);
	return 0;
}

// worker thread - calls l2pool_process_slice on the next slice until they are all done
static void
l2pool_worker_thread( struct nn_graph *nn, void *thrpv )
{
	struct l2pool_thrinfo *thrp = (struct l2pool_thrinfo *)thrpv;
	struct l2pool_runstate * rstp = thrp->rstp;
	void *integ_buf = thrp->integ_buf;
	struct integral_buffer_plan const *ibp = rstp->ibp;
	l2pool_process_slice_fp slice_func = rstp->process_slice_func;

	int job_idx = thrp->job0;				// get range of jobs to do
	int job_end = thrp->job_end;
	int batch_idx=0;
	int d32_idx= 0;

	int nd32 = ibp->tin.nd32;		// # of depth slices
	if( job_idx> 0){				// starting in the middle
		batch_idx =  job_idx/(unsigned)nd32;
		d32_idx = job_idx - batch_idx * nd32;
	}

	for( ; job_idx  < job_end; job_idx++){
		// process the current batch
		(*slice_func)( rstp, integ_buf, batch_idx, d32_idx);
		if( ++d32_idx >= nd32){
			d32_idx = 0;
			batch_idx++;
		}
	}

	nn_sem_post( &rstp->done_sem);
}


////////////////////////////////////////////////////////
//
//'reference' function to process an output row
//
static void
consumer_run_row_ref( struct integ_buff_vscan const *vscan,  struct integral_buffer_plan const *ibp, int out_wid )
{
	uint8_t * optr = vscan->consumer_out_row;
	int32_t const *pup  = vscan->consumer_top;
	int32_t const *pdn = vscan->consumer_bot;
	int win_w = ibp->window_wid;
	int stride_w = ibp->stride_wid;

	int recip_rsh = ibp->recip_shift;
	int recip_mant = ibp->recip_mant;

	int depth = 32;

	int wbump = stride_w*32- depth;	// amount to bump pointers at end of w loop

	for( int j = 0; j < out_wid; j++ ){
		for( int id = 0; id < depth; id++){
			int32_t areasum =  (pdn[win_w*32]-pup[win_w*32])-(pdn[0]-pup[0]);
			// >> recip_rsh, result should fit in u16
			int areasum1 = saturate_u16( areasum >> recip_rsh);
			// find square root * 16
			int sqrtval = ref_ruh_sqrt_RuhI ( areasum1, 4);
			//int sqrtval = roundf_i32( sqrtf((float)areasum1*256.0f));
			// final scaling
			int32_t scaled_sum = (sqrtval * recip_mant + 0x4000) >> 15;
//if( j==4 &&  id == 20)printf(" areasum=%d; >>= %d; sqrt = %d ; scaled = %d\n", (int)areasum, areasum1, sqrtval, (int)scaled_sum);
			optr[j*32+id] = saturate_u8(  scaled_sum);
			pup++;		// to next i32 unit
			pdn++;
		}
		pup += wbump;
		pdn += wbump;
	}
}
static int __attribute__((noinline,unused))
vec_extract( HVX_Vector val, int index)
{
	union { HVX_Vector v; int as_int[32];} uu = { val };
	return uu.as_int[index&31];
}
static inline
HVX_Vector
win_sum_4( HVX_Vector vUL,HVX_Vector vUR, HVX_Vector vDL, HVX_Vector vDR)
{
	return Q6_Vw_vsub_VwVw(
		 Q6_Vw_vsub_VwVw( vDR,vUR),
	     Q6_Vw_vsub_VwVw( vDL,vUL));
}
//
//'hvx' function to process an output row
//
static void
consumer_run_row_hvx( struct integ_buff_vscan const *vscan,  struct integral_buffer_plan const *ibp, int out_wid )
{
	HVX_Vector * optr = (HVX_Vector*)vscan->consumer_out_row;
	HVX_Vector const *pup  = (HVX_Vector const *)vscan->consumer_top;
	HVX_Vector const *pdn = (HVX_Vector const *)vscan->consumer_bot;
	int win_w = ibp->window_wid;
	int stride_w = ibp->stride_wid;

	// do loop_count*4 outputs, followed by 1..4 at the end.
	//	int loop_count = (out_wid-1)>>2;

	// ** due to the size of the loop, this rounds the width
	// up to a multiple of 4. The extra stores go into width-after padding.
	// There may be up to 4 extra reads of the input (4 vectors)
	// but since it's in scratch area that should not be a problem.
	// If this is changed to keep track of the range of values, it will need
	// to be modified to avoid including the extraneous values in the min/max.
	//
	int loop_count = (out_wid+3)>>2;	// do all and up to 3 extra...

	int recip_rsh = ibp->recip_shift;
	int recip_mant = ibp->recip_mant;
	recip_mant = Q6_R_combine_RlRl( recip_mant,recip_mant);

	HVX_Vector vULa, vDLa, vURa;	// something to 'read ahead'
	vULa = pup[0];
	vDLa = pdn[0];
	vURa = pup[win_w];

	for(int j = 0; j < loop_count; j++){
		//  four deltas
		HVX_Vector sumA = win_sum_4(vULa, vURa,vDLa, pdn[win_w]);
		pup += stride_w;
		pdn += stride_w;
		HVX_Vector sumB = win_sum_4( pup[0], pup[win_w],pdn[0], pdn[win_w]);
		pup += stride_w;
		pdn += stride_w;
		HVX_Vector sumC = win_sum_4( pup[0], pup[win_w],pdn[0], pdn[win_w]);
		pup += stride_w;
		pdn += stride_w;
		HVX_Vector sumD = win_sum_4( pup[0], pup[win_w],pdn[0], pdn[win_w]);
		pup += stride_w;
		pdn += stride_w;
		// >> by recip_rsh, and trunc/round to 16 bits
		HVX_Vector tmpAC = Q6_Vuh_vasr_VwVwR_sat( sumC, sumA, recip_rsh);
		HVX_Vector tmpBD = Q6_Vuh_vasr_VwVwR_sat( sumD, sumB, recip_rsh);
		// deal these out so we have
		//      A0 A1 ... A31 B0.. B31
		// and  C0 C1 ... C31 D0.. D31
		HVX_VectorPair dealt= Q6_W_vdeal_VVR( tmpBD, tmpAC, -2);
		// square roots (with 4 fraction bits; range will be about 0.. 4096)
		HVX_Vector sqrtAB = hvx_Vuh_sqrt_VuhI( Q6_V_lo_W(dealt), 4 );
		HVX_Vector sqrtCD = hvx_Vuh_sqrt_VuhI( Q6_V_hi_W(dealt), 4 );

		// now scale by recip_mant
		HVX_Vector scAB = Q6_Vh_vmpy_VhRh_s1_rnd_sat(sqrtAB, recip_mant);
		HVX_Vector scCD = Q6_Vh_vmpy_VhRh_s1_rnd_sat(sqrtCD, recip_mant);

		// pack/sat
		HVX_Vector result = Q6_Vub_vpack_VhVh_sat( scCD,scAB);
		vULa = pup[0];
		vDLa = pdn[0];
		vURa = pup[win_w];
		*optr++ = result;
	}

	// final 1..4
	// ** removed
	if(0){
		HVX_Vector sumA=Q6_V_vzero();
		HVX_Vector sumB=Q6_V_vzero();
		HVX_Vector sumC=Q6_V_vzero();
		HVX_Vector sumD=Q6_V_vzero();

		// load sumA, and sumB, sumC, sumD  according to what's actually used
		int k= (out_wid&3);
		sumA = win_sum_4(vULa, vURa,vDLa, pdn[win_w]);
		pup += stride_w;
		pdn += stride_w;
		if( k != 1){
			sumB = win_sum_4( pup[0], pup[win_w],pdn[0], pdn[win_w]);
			pup += stride_w;
			pdn += stride_w;
			if( k != 2){
				sumC = win_sum_4( pup[0], pup[win_w],pdn[0], pdn[win_w]);
				pup += stride_w;
				pdn += stride_w;
				if( k== 0)
					sumD = win_sum_4( pup[0], pup[win_w],pdn[0], pdn[win_w]);
			}
		}
		HVX_Vector tmpAC = Q6_Vuh_vasr_VwVwR_sat( sumC, sumA, recip_rsh);
		HVX_Vector tmpBD = Q6_Vuh_vasr_VwVwR_sat( sumD, sumB, recip_rsh);
		HVX_VectorPair dealt= Q6_W_vdeal_VVR( tmpBD, tmpAC, -2);
		HVX_Vector sqrtAB = hvx_Vuh_sqrt_VuhI( Q6_V_lo_W(dealt), 4);
		HVX_Vector sqrtCD = hvx_Vuh_sqrt_VuhI( Q6_V_hi_W(dealt), 4);
		HVX_Vector scAB = Q6_Vh_vmpy_VhRh_s1_rnd_sat( sqrtAB, recip_mant);
		HVX_Vector scCD = Q6_Vh_vmpy_VhRh_s1_rnd_sat( sqrtCD, recip_mant);

		HVX_Vector result = Q6_Vub_vpack_VhVh_sat( scCD,scAB);
		*optr = result;
	}
}
// sumA:      A0   A1 ....  A31           (w)
// tmpAC:     A0 C0  A1  C1   ... A31 C31 (h)
// tmpBD:     B0 D0  B1  D1  .... B31 D31 (h)
// dealt.lo:  A0 A1 .. A31 B0 ... B31 (h)
// dealt.hi:  C0 C1 .. C31 D0 ..  D31 (h)
// result0:   A0 C0 A1 C1 ... A31 C31 B0 D0 .. .B31 D31
// result:    A0 A1 ..A31 B0 B1.. B31 C0 C1 ..   D31


// process a slice
//
static void l2pool_process_slice( struct l2pool_runstate const *rstp,  void * integ_buf, int batch_idx, int d32_idx  )
{
	struct integral_buffer_plan const *ibp= rstp->ibp;
	// NOTE: if vsc is declared here and only
	// passed to functions that are expanded inline,
	// the  compiler should treat it as a series of local
	// variables (and be able to see all of the places where
	// they are modified).
	//
	struct integ_buff_vscan vsc;		// 'vertical scan' state
	vsc.buffer_base = integ_buf;

	int stride_h = ibp->stride_ht;
	int out_ht = ibp->outshape.height;
	int out_wid = ibp->outshape.width;
	int inrows_remain = ibp->inshape.height;

	// first set up the scan (this includes finding the read
	// and write pointers for the current job)
	//
	setup_integ_buffer_vscan( &vsc, ibp, batch_idx, d32_idx);

	vscan_clear_initial( &vsc, ibp);

	//
	// The first time we must load this may rows,
	// so that top padding can be constructed.
	// 'rows_loaded_per' is the number of rows we load
	// in subsequent loops.
	//
	int rows_to_load = ibp->ibuf_initial_load;
	const int rows_loaded_per = 4;

	// run until we have generated the required # of output rows.
	while(1){
		if( inrows_remain > 0){
			// This loads rows, and generates left/right/top padding as needed.
			// It will set up for bottom padding after loading the last row.
			// note that the last load will generally be < rows_loaded_per, but that
			// won't break anything; it may cause the consumer to miss a step.
			l2pool_producer_load_rows( &vsc, ibp, rows_to_load);
			inrows_remain -= rows_to_load;
			rows_to_load = min_i32(rows_loaded_per,inrows_remain);
		}else{
			// add 4 more padding rows, or as many as possible. Also advances 'consumer_rows_generated'
			producer_add_bottom_padding( &vsc,ibp, rows_loaded_per);
		}
		// generate output rows
		// depending on the stride_h, we may generate up to 4 output rows after each 'load',
		// or it may be that we need a few loads between each time it's generated (if stride_h > 4).
		// TODO: probably should ensure that we can make at least one per loop. This requires
		// expanding the buffer height to be >= win_h+stride_h, and then loading max(4,stride)
		// per loop.
		while( vsc.consumer_rows_generated * stride_h < vsc.producer_nrows-vsc.row_offset ){
			// we can generate the next output row...
			consumer_run_row_ref( &vsc, ibp, out_wid);
			// bump down...
			// advance the window pointers and the output pointer.
			if(++ vsc.consumer_rows_generated >= out_ht ) goto done;
			vscan_bump_consumer_pointers(&vsc, ibp);
		}
	}
 done:
 	 return;
}
static void
l2pool_process_slice_hvx(struct l2pool_runstate const *rstp,  void * integ_buf, int batch_idx, int d32_idx  )
{
	struct integral_buffer_plan const *ibp= rstp->ibp;
	// NOTE: if vsc is declared here and only
	// passed to functions that are expanded inline,
	// the  compiler should treat it as a series of local
	// variables (and be able to see all of the places where
	// they are modified).
	//
	struct integ_buff_vscan vsc;		// 'vertical scan' state
	vsc.buffer_base = integ_buf;

	int stride_h = ibp->stride_ht;
	int out_ht = ibp->outshape.height;
	int out_wid = ibp->outshape.width;
	int inrows_remain = ibp->inshape.height;

	// first set up the scan (this includes finding the read
	// and write pointers for the current job)
	//
	setup_integ_buffer_vscan( &vsc, ibp, batch_idx, d32_idx);
	vscan_clear_initial_hvx( &vsc, ibp);

	//
	// The first time we must load this may rows,
	// so that top padding can be constructed.
	// 'rows_loaded_per' is the number of rows we load
	// in subsequent loops.
	//
	int rows_to_load = ibp->ibuf_initial_load;
	const int rows_loaded_per = 4;

	// run until we have generated the required # of output rows.
	while(1){
		if( inrows_remain > 0){
			// This loads rows, and generates left/right/top padding as needed.
			// It will set up for bottom padding after loading the last row.
			// note that the last load will generally be < rows_loaded_per, but that
			// won't break anything; it may cause the consumer to miss a step.
			l2pool_producer_load_rows_hvx( &vsc, ibp, rows_to_load);

			inrows_remain -= rows_to_load;
			rows_to_load = min_i32(rows_loaded_per,inrows_remain);
		}else{
			// add 4 more padding rows, or as many as possible. Also advances 'consumer_rows_generated'
			producer_add_bottom_padding_hvx( &vsc,ibp, rows_loaded_per);
		}
		// generate output rows
		// depending on the stride_h, we may generate up to 4 output rows after each 'load',
		// or it may be that we need a few loads between each time it's generated (if stride_h > 4).
		// TODO: probably should ensure that we can make at least one per loop. This requires
		// expanding the buffer height to be >= win_h+stride_h, and then loading max(4,stride)
		// per loop.
		while( vsc.consumer_rows_generated * stride_h < vsc.producer_nrows-vsc.row_offset ){
			// we can generate the next output row...
			consumer_run_row_hvx( &vsc, ibp, out_wid);
			// bump down...
			// advance the window pointers and the output pointer.
			if(++ vsc.consumer_rows_generated >= out_ht ) goto done;
			vscan_bump_consumer_pointers(&vsc, ibp);
		}
	}
 done:
 	 return;
}



static int l2pool_check(struct nn_node *self, struct nn_graph *nn)
{
	//if (self->n_inputs != 5) return errlog(nn,"l2pool wrong # inputs");
	//if (self->n_outputs != 3) return errlog(nn,"l2pool wrong # outs");

	const struct tensor *window_tensor = self->inputs[3];
	const struct tensor *stride_tensor = self->inputs[4];

	if( window_tensor->shape.batches != 1 || window_tensor->shape.depth != 1 ){
		return errlog(nn, "bad window shape");
	}
	if( stride_tensor->shape.batches != 1 || stride_tensor->shape.depth != 1 ){
		return errlog(nn, "bad stride shape");
	}
	// attach the 'integral buffer plan'
	// initialized to all 0

	void *ibp = nn_calloc( sizeof(struct integral_buffer_plan),1);
	self->opaque = ibp;
	if( ibp == NULL)
		return errlog(nn, "can't allocate %d bytes for buffer plan", (int)sizeof(struct integral_buffer_plan));
	return 0;
}

static int l2pool_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct integral_buffer_plan * ibp = (struct integral_buffer_plan *) self->opaque;
	if( ibp!= NULL){
		nn_free( ibp->edgepad_scale_alloc);
		nn_free( ibp);
		self->opaque= NULL;
	}
	return node_free_common(self,nn);
}


struct nn_node_ops nn_ops_for_QuantizedL2Pool_8_d32 = {
	.execute = l2pool_execute,
	.check = l2pool_check,
	.ctor = node_alloc_common,
	.dtor = l2pool_dtor,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};

struct nn_node_ops nn_ops_for_QuantizedL2Pool_8_d32_ref = {
	.execute = l2pool_execute,
	.check = l2pool_check,
	.ctor = node_alloc_common,
	.dtor = l2pool_dtor,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};

