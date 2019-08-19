
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
 * This contains implementations for quantized avg pooling node
 */

//#define TEST_PERFORMANCE

//
// Avgpool_d32 uses one of three strategies:
//   - if the situation is such that the output tensor is (B,1,1,D), a separate
//     method is used which just sums the appropriate window of the input (usually all of it,
//     but may not be when there's a stride) and generates the output from that
//   - there is a hand-coded assembler routine for win = 3x3 and stride = 1x1; this requires
//     'zapping' of the input edges to implement padding, and can't always be used when the
//     input padding is not sufficient. It also requires in_ht, in_wid >=2
//   - in all other cases, an integral buffer is used to find the window sums.
//
//
// For the case where the integral buffer is used
//   - the integral buffer finds 32-bit accumulations of pixels, multiplied by 256
//       (the x256 is to provide extra fractional bits, to bury rounding errors
//       when calculating edge padding)
//   - If any output windows are reduced by edge clipping, then padding rows/cols
//    will be added to integral buffer so that the correct results are produced by including
//     these padded rows/cols in the constant window size.
//   - each window's sum needs to be divided by (win_h * winw * 256), which is expressed
//     as a 16-bit 'mantissa' (in range 16k..32k) and a shift.
//   - the 32-bit sums from the accumulation are:
//         (1) right-shifted by a predetermined amount; result rounded to 16 bits
//         (2) scaled by a 16 bit fraction (which we  keep in range (0.5..0.999)
//         (3) 8 bit result is then located in bits 13..6 and can be extracted using >>6
//
//  Example
//   win size        (1)>> by      (2)  mul by mant, >>15
//    3x3              5               29127
//    5x5              6               20927
//   19x19            10               23237
//   63x63            13               16908
//   1x3               3               21845
//    1x2              2               16384
//    1x1              1               16384
// The right shift is ceiling(log2(wh*ww))+1
// The scale factor is (2^(13+rsh))/(wh*ww)
//
// The above is correct for the case when the output min/max are the same is the input min/max. In some
// cases, specifically with large windows, we may want to reduce the output range dynamically for
// better resolution; this would be as above, except
//     (2) add an offset (saturated to i16) and then mul by a 16-bit fraction
//     (3)  obtain u8 result using >> final_rsh (may be < 6).
//
//=============================
//  All-average-to-one case:
//=============================
//  This is a special case, where the output h x w is  1x1. No integral image is used,
//    but we store an intermediate 32-bit result:
//    - work is split up into batches/depth-slices as usual
//    - for each work unit, result is the 32-bit sums of all 32 of the depth indices (one vector)
//    - these are offset so that '0' means application-level 0; and we find the min and max
//      of all of them.
//    - the range is then computed, and the values are all quantized to 8 bits.
//
//
#include <nn_graph.h>
#include <string.h>

#include "nn_integral_buffer.h" // much of the code for integral buffer management
#include "nn_atomic.h"
#include "hvx_mathops.h"

#ifdef HEXAGON_V66
#define AVGPOOL_MAX_THREADS 4
#else
#define AVGPOOL_MAX_THREADS 2
#endif

/////////////////////////////////////////////////
// prototypes of asm routines for particular cases
//
//
// This does one 'depth slice' for win = 3x3 stride = 1x1
// Note:
//    - all pointers are vector aligned
//    - The first output is based on the upper 3x3 of the indicated input area.
//    - The 'out_vectors_wide' is the number of output vectors.
//    - if out_lalign is 32,64, or 96, then that many bytes of 'padding' are inserted
//      at the start of each output row, and the rest are shifted along (the 'out_vectors wide'
//      will need to account for this, to ensure you get the full output row you want).
//    - the number of output rows is rounded up to even; if you want an odd #, you'll need to have bottom
//     padding in the output to support an extra row.

//  If edge padding is needed, this has to be done beforehand (see avgpool_zap_row and avgpool_zap_lr).
//
// For instance, suppose you have in_wid = out_wid = 11, input_wid_pad_left = output_wid_pad_left = 4:
// (with NN_PAD_SAME):
//
//   * will need to zap input l&r first; thus padding is needed on both sides
//   * set input pointer to the start of the input padding, so that the extra left is included
//   * first 3 outputs will be garbage, so set out_lalign = 32; thus you have a full vector of
//     garbage at the start of each row, and you should set outp to the first output padding row;
//   * that makes a total of 1+3+11 = 15 valid outputs, so set out_vectors_wide to 4.
//
// Likewise if you have 7 rows in, 7 rows out, with NN_PAD_SAME:
//   * will need at least one padding row at top and bottom of input; these will be zapped
//   * you will also need >1 padding row at the bottom of the output, for overshoot.
//   * set the 'input' pointer to the first padding row.
//
//

int avgpool_slice_hvx_3x3_stride1(
	uint8_t *out,			// pointer to output
	const uint8_t *in, 		// pointer to input
	int32_t in_next_row,	// input row pitch
	int32_t out_next_row,	// output row pitch
	int32_t out_vectors_wide,	// output vectors
	int32_t out_lines,		// output height (= input height)
	int32_t out_lalign);



 // This does left/right zapping on each row.
 //  ptr-> points to 1st vector on 1st row
 // This should done before top/bot zapping, with the range included
 // to cover the extra top/bot rows if needed.
 //
  int avgpool_zap_lr(
  	uint8_t *ptr,		// pointer to start of row, including width-before padding
  	int32_t height,		// rows to process
  	int32_t width,		// actual width
  	int32_t left_pad,  // width_before padding
  	int32_t next_row);	// row stride
///////////////////////////////////////////////////////////////////////

// is there enough padding to zap and use the assembler version?
// it's ok to have insufficient right padding, provided (1) left+right padding
//  is <= total left+right needed; and (2) extra row at the bottom.
// also, the zap routine does L & R together, so if either padding is
// needed, both are needed.
//
static inline int
padding_is_enough_to_zap( struct integral_buffer_plan const *ibp, struct tensor const * in_tensor )
{
	if( ibp->wpad_flags ==0) return 1;
	int bot_needed = ibp->wpad_bottom;
	int rt_have = in_tensor->format.width_pad[1];

	int lr_pad_needed= max_i32(ibp->wpad_left, ibp->wpad_right);
	if( lr_pad_needed > rt_have) bot_needed ++;

	if( ibp->wpad_top > in_tensor->format.height_pad[0]
	    || bot_needed > in_tensor->format.height_pad[1]
	    || lr_pad_needed > in_tensor->format.width_pad[0]
	    || lr_pad_needed*2 > in_tensor->format.width_pad[0] + rt_have){
		return 0;
	}
	return 1;
}


struct avgpool_runstate;
typedef void( *avgpool_process_slice_fp)( struct avgpool_runstate const *ibp,  void * integ_buf, int batch_idx, int d32_idx  );

struct avgpool_thrinfo {
	struct avgpool_runstate * rstp;		// pointer to container;
	void  * integ_buf;					// integral buf allocated to thread
	int job0, job_end;			// thread does jobs with  job0 <= i < job_end
	int16_t thr_index;			// 0,1 ..
};


////////////////////////////////////////
// parameters for using avgpool_slice_hvx_3x3_stride1
// The pointers are correct only for batch 0, d32 index = 0;
// all need to be offset for other cases.
struct avgpool_asm_parms {
	uint8_t const * in_ptr_base;	// 'input' pointer for the hvx routine
	uint8_t * out_ptr_base;			// output pointer
	int in_next_row;					// row stride bytes
	int out_next_row;				//
	int vecs_wide;				// full # of output vecs
	int out_shift;				// # of bytes to pad into 1st vec to align
	uint8_t * top_zap_loc;		// location for top zap (input row 0); null if not needed
	uint8_t * bot_zap_loc;		// location for bottom zap (input row ht-1); null if not needed.
	uint8_t * lr_zap_loc;		// location for left/right zap; null if not needed. Done after top/bot
	int row_zap_skip;			// = 0,32,64, or 96; # of bytes to skip in first vector of top/bottom zap
	int row_zap_nvecs;			// width of top/bot zap ( in vectors)
	int lr_zap_height;			// rows to zap LR
	int in_wpad_left;			// used for LR zap
	int in_width;				// used for LR zap
};

static void avgpool_earlywork_v(struct nn_graph *nn, void *vinfo)
{
	struct integral_buffer_plan *plan = vinfo;
	struct nn_early_work *work = plan->misc;
	if (work == NULL) return;
	if (work->vtcm_addr != nn->vtcm_ptr) return;
	if (work->src_addr == NULL) return;
	if (work->dst_addr == NULL) return;
	if (work->bytes == 0) return;
	nn_graph_memcpy(nn,work->dst_addr,work->src_addr,work->bytes);
	work->valid = 1;
}
static int avgpool_earlywork(struct nn_graph *nn, void *vinfo)
{
	avgpool_earlywork_v(nn,vinfo);
	return 0;
}

// set up the 'avgpool_asm_parms'
//
static inline void
setup_avgpool_asm_parms(struct avgpool_asm_parms *asp, struct integral_buffer_plan const *ibp)
{
	int in_next_row = ibp->tin.height_stride;

	uint8_t const * in_ptr = ibp->tin.data;
	uint8_t * out_ptr = ibp->tout.data;
	// if the input needs padding on left/top, treat that as moving the pointer
	in_ptr -= 32*ibp->wpad_left + in_next_row * ibp->wpad_top;
	// last byte we need to generate.
	uint8_t *out_last = out_ptr + 32*ibp->outshape.width-1;

	int in_w_lpad = (((int) (size_t)in_ptr) >> 5)&3;
	int out_w_lpad = (((int) (size_t)out_ptr) >> 5)&3;

	in_ptr -= 32*in_w_lpad;
	out_ptr -= 32*out_w_lpad;

	int lshift = out_w_lpad - in_w_lpad;		// -3 ..3
	if( lshift < 0) out_ptr -= 128;			// outpad < inpad; must make room

	int input_wpad_left = ibp->input_wpad_left;

	asp->in_ptr_base = in_ptr;
	asp->out_ptr_base = out_ptr;
	asp->in_next_row = in_next_row;
	asp->out_next_row = ibp->tout.height_stride;
	asp->vecs_wide= (unsigned)(out_last-out_ptr)/128u + 1;
	asp->out_shift = (lshift&3)* 32;
	asp->in_wpad_left = input_wpad_left;
	asp->in_width = ibp->inshape.width;


	asp->top_zap_loc = NULL;
	asp->bot_zap_loc = NULL;
	asp->lr_zap_loc = NULL;

	int wpad_flags = ibp->wpad_flags;
	if( wpad_flags != 0 ){
		uint8_t * row0ptr= ibp->tin.data - 32 *input_wpad_left;
		int in_height = ibp->inshape.height;
		int lrzap_rows = in_height;

		if( (wpad_flags & (intbuf_PAD_T|intbuf_PAD_B))!= 0 ){
			// skip lanes in first vector (avoid overlap with right-zap in previous d32 slice)
			asp->row_zap_skip = (input_wpad_left&3)*32;		// 0,32,64, or 96
			// # of vectors to zap including any partial left or right
			asp->row_zap_nvecs = ( (input_wpad_left&3) +ibp->inshape.width + 3 )>>2;
			// offset from start of left padding to first zapped vector (0 or 128).
			int left_offs = (input_wpad_left*32)&~127;
			// bottom zap? extend LR zap range if so
			if( (wpad_flags & intbuf_PAD_B) !=0){
				asp->bot_zap_loc = row0ptr + left_offs+ (in_height-1)*in_next_row;
				lrzap_rows ++;
			}
			// top zap? extend LR zap range if so
			if( (wpad_flags & intbuf_PAD_T) !=0){
				asp->top_zap_loc = row0ptr + left_offs;
				lrzap_rows ++;
				row0ptr -= in_next_row;
			}
		}

		// do LR zap ?
		if( (wpad_flags & (intbuf_PAD_L|intbuf_PAD_R)) !=0 ){
			asp->lr_zap_loc = row0ptr;
			asp->lr_zap_height = lrzap_rows;
		}
	}
}

struct avgpool_runstate
{
	struct integral_buffer_plan const *ibp;
	struct avgpool_asm_parms asmparms;		// where needed
	float range_min, range_max;

	int n_threads;		// # of threads running.
	int jobs;
	nn_sem_t done_sem;
	avgpool_process_slice_fp process_slice_func;

	// These are only applicable when doing 1x1 reduce;
	// 'reduce1x1_result' is an array of [32xi32], one for each batch, each d32 slice,
	// and are the total sums. It is allocated in scratch area. The min & max are the min/max sums for the
	// The first thread waits for others and then finishes the computation.
	int32_t * reduce1x1_result;
	volatile int32_t reduce1x1_min_not, reduce1x1_max;


	// 2x2 are sliced differently: by batches, and then vertically into 1,2 or 4 parts.
	// so jobs = batches * (1<<slice_shift)
	volatile int next_job;			// used to select next job
	int hslice_shift;				// 0,1 or 2 according to how h is sliced
	int hbreaks[4+1];				// height breaks. hbreaks[0] = 0, hbreaks[1<<hslice_shift]==out.height


	struct avgpool_thrinfo thrinfo[AVGPOOL_MAX_THREADS];
};

////////////////////////////////////////






static void avgpool_worker_thread( struct nn_graph *nn, void *thrpv );
static void avgpool_reduce_to_1x1_worker_thread( struct nn_graph *nn, void *thrpv );
static void avgpool_reduce_to_1x1_post_worker_thread( struct nn_graph *nn, void *rstpv );
static void avgpool_2x2_worker_thread_func( struct nn_graph * nn, void *rstpv);

static void avgpool_process_slice( struct avgpool_runstate const *ibp,  void * integ_buf, int batch_idx, int d32_idx  );
static void avgpool_process_slice_hvx( struct avgpool_runstate const *ibp,  void * integ_buf, int batch_idx, int d32_idx  );

static void avgpool_process_slice_special( struct avgpool_runstate const *ibp,  void * integ_buf, int batch_idx, int d32_idx  );

enum avgpool_special_case {
	avgpool_special_NONE = 0,
	avgpool_special_3x3_hvx,
	avgpool_special_2x2_hvx,
	avgpool_special_reduce_to_1x1
};

static int avgpool_execute(struct nn_node *self, struct nn_graph *nn)
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


	int use_hvx = self->node_type == OP_QuantizedAvgPool_8_d32;
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
		int special_handler = avgpool_special_NONE;		//zero
		// check to see if we can use special-case code
		if( use_hvx ){
			if( ibp->outshape.height == 1 && ibp->outshape.width == 1){
				special_handler = avgpool_special_reduce_to_1x1;
			}else if(ibp->window_ht ==3 && ibp->window_wid == 3 && ibp->stride_ht == 1 && ibp->stride_wid == 1
				&& ibp->inshape.width >1 && ibp->inshape.height > 1 ){
				ibp->hvx_specialized_handler = avgpool_slice_hvx_3x3_stride1;
				special_handler = avgpool_special_3x3_hvx;
			}else if( ibp->window_ht == 2 && ibp->window_wid == 2 && ibp->stride_wid == 2){
				// can use specialized 2x2 handler (any stride_h is supported).
				special_handler = avgpool_special_2x2_hvx;
			}
		}
		ibp->hvx_specialized_handler_code = special_handler;
	}
	// for output wid = 1, the left-padding will be 4, unless we can use the 'reduce_to_1x1' special case
	// handler in which case it will be 2.
	int width_before_pad = 4;
	if(  ibp->hvx_specialized_handler_code == avgpool_special_reduce_to_1x1){
		width_before_pad = 2;
	}

	// if ok, then ibp->outshape is the output shape
	int top_bottom_pad = (ibp->outshape.height == 1) ? 1 : 4;
	int out_wpad_0 = width_before_pad;
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
	//
	// can we use the special-case handler?
	//
	int special_handler = ibp->hvx_specialized_handler_code;
	if( special_handler  == avgpool_special_3x3_hvx ){
		// if not enough padding to zap, can't use it
		// if the output height is odd and we don't have a row of padding to contain it, can't use it.
		// because of the way left-padding is handled,
		// the minimum output width-padding is (in_wid_pad - left_pad)%4
		if( ! padding_is_enough_to_zap( ibp, in_tensor)
		 || ((out_tensor->shape.height&1)!=0  && out_tensor->format.height_pad[1] == 0 )
		 || out_tensor->format.width_pad[0] <  ((in_tensor->format.width_pad[0]-ibp->wpad_left) &3)){
			special_handler = avgpool_special_NONE;
		}
	}

	logmsg(nn,2,"avgpool %dx%d s=%dx%d (pad %d) %dx%dx%dx%d %dx%dx%dx%d n_d32=%d special=%d pad %d:%d:%d:%d",
			ibp->window_ht, ibp->window_wid,
			ibp->stride_ht, ibp->stride_wid, self->padding,
			ibp->inshape.batches, ibp->inshape.height, ibp->inshape.width, ibp->inshape.depth,
			ibp->outshape.batches, ibp->outshape.height, ibp->outshape.width, ibp->outshape.depth,
			ibp->tout.nd32 , special_handler,
			ibp->wpad_top, ibp->wpad_bottom, ibp->wpad_left, ibp->wpad_right );


	struct avgpool_runstate runstate;
	runstate.ibp = ibp;
	runstate.next_job = 0;

	runstate.range_min = tensor_get_float(in_min_tensor,0);
	runstate.range_max = tensor_get_float(in_max_tensor,0);

	runstate.process_slice_func = (!use_hvx) ? avgpool_process_slice:
				(special_handler == avgpool_special_3x3_hvx)  ? avgpool_process_slice_special : avgpool_process_slice_hvx;

	// figure out how many 'jobs' we need, how many threads to launch
	int njobs = ibp->inshape.batches * ibp->tin.nd32;
	runstate.jobs = njobs;
	int n_threads = min_i32(AVGPOOL_MAX_THREADS, njobs);
	nn_scratch_reset( nn );		// reset 'scratch' pointer



#if 0
printf("%d threads for %d jobs; each with %d bytes scratch\n", n_threads, njobs, integ_buf_bytes );
printf("buffer rows: %d row0 =%d, initial load = %d\n", ibp->ibuf_rows, ibp->ibuf_row0, ibp->ibuf_initial_load);
printf("top_padding = %d; bottom = %d; infeas_h = %d\n",ibp->wpad_top, ibp->wpad_bottom, ibp->infeas_pad_ht );
printf("left_padding = %d; right = %d; infeas_w = %d\n",ibp->wpad_left, ibp->wpad_right, ibp->infeas_pad_wid );
#endif


	void * integ_buf = NULL;
	int integ_buf_bytes =0;
	int need_post_work = 0;

	void (*avgpool_worker_thread_func)( struct nn_graph *, void *);

	avgpool_worker_thread_func = avgpool_worker_thread;
	int need_thrinfo = 1;

	if( special_handler == avgpool_special_3x3_hvx){
		// if using the 3x3 special handler, set up its parms
		setup_avgpool_asm_parms( &runstate.asmparms, ibp);
	}else if( special_handler == avgpool_special_2x2_hvx){
		// need alternate slicing setup; by batches and then height by 1,2 or 4.
		// no d32 slicing.
		int hslice_shift = 1;	// assume 2 for now...
		int oht = ibp->outshape.height;
		int batches = ibp->outshape.batches;
		if( oht <2 || batches >= 4*AVGPOOL_MAX_THREADS){	// don't slice at all
			hslice_shift = 0;
		}else if( AVGPOOL_MAX_THREADS >=4 && oht >=4 && batches<=3){	// slice into 4
			hslice_shift = 2;
		}
		runstate.jobs = njobs = batches<< hslice_shift;
		n_threads = min_i32(AVGPOOL_MAX_THREADS, njobs);
		runstate.hslice_shift = hslice_shift;
		// set up the 'hbreaks' array
		runstate.hbreaks[0] = 0;
		int nh = 1<<hslice_shift;	// 1,2, or 4
		runstate.hbreaks[nh] = oht;
		if( hslice_shift > 0){
			int hmid = (oht+1)>>1;
			if( hslice_shift < 2){
				runstate.hbreaks[1] = hmid;
			}else{
				runstate.hbreaks[1] = (hmid+1)>>1;
				runstate.hbreaks[2] = hmid;	// midpoint
				runstate.hbreaks[3] = (hmid+oht+1)>>1;
			}
		}
		need_thrinfo = 0;
		avgpool_worker_thread_func = avgpool_2x2_worker_thread_func;

	}else if( special_handler == avgpool_special_reduce_to_1x1){
		runstate.reduce1x1_min_not = ~0x7fffffff;
		runstate.reduce1x1_max = ~0x7fffffff;
		// allocate buffer for results (in 32-bit form)
		int nvec = runstate.jobs;		// total # of units (batch *d32)
		// round up to even so we can do two at once
		nvec = (nvec+1)& ~1;
		void *mbuf = nn_scratch_alloc(nn, nvec * 128);
		if( mbuf == NULL)
			return errlog(nn, "could not alloc %d bytes of scratch",nvec*128 );
		runstate.reduce1x1_result = (int32_t*)mbuf;
		avgpool_worker_thread_func = avgpool_reduce_to_1x1_worker_thread;
		need_post_work = 1;
	}else{
		// else
		// allocate scratch for the integral buffers
		
		integ_buf_bytes = ibp->ibuf_total_bytes;
		if (nn_scratch_grow(nn, integ_buf_bytes*n_threads)){
			return errlog(nn, "scratch too small");
		}
		nn_scratch_reset(nn);
		integ_buf = nn_scratch_alloc(nn, integ_buf_bytes*n_threads);
		if( integ_buf == NULL){
			return errlog(nn, "could not alloc %d bytes of scratch %d",integ_buf_bytes*n_threads,nn->scratch_size);
		}
	}
	// fill in the 'thrinfo' and launch threads
	runstate.n_threads = n_threads;
	nn_sem_init(&runstate.done_sem,0);

	if(need_thrinfo){
		for(int i=0; i < n_threads; i++){
			runstate.thrinfo[i].job0 = (i*njobs) / n_threads;
			runstate.thrinfo[i].job_end = ((i+1)*njobs) / n_threads;
			runstate.thrinfo[i].thr_index = i;
			runstate.thrinfo[i].rstp = &runstate;
			runstate.thrinfo[i].integ_buf = integ_buf;
			integ_buf = (void*)( (char*)integ_buf + integ_buf_bytes );
			nn_os_work_for_vector(nn, avgpool_worker_thread_func , &runstate.thrinfo[i]);
		}
	}else{
		for(int i=0; i < n_threads; i++){
			nn_os_work_for_vector(nn, avgpool_worker_thread_func , &runstate);
		}
	}
	if(!need_post_work) nn_os_vector_call(nn,avgpool_earlywork,ibp);

	// wait for those to be done
	nn_sem_wait_n_times( &runstate.done_sem, n_threads);

	// do post-work for 1x1 case
	if( need_post_work ){
		nn_os_work_for_vector(nn, avgpool_reduce_to_1x1_post_worker_thread , &runstate);
		nn_os_vector_call(nn,avgpool_earlywork,ibp);
		nn_sem_wait( &runstate.done_sem);
	}

	if (tensor_set_single_float(out_min_tensor,runstate.range_min) != 0
		|| tensor_set_single_float(out_max_tensor,runstate.range_max) != 0) {
		return errlog(nn,"min or max out prep fail");
	}

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("avgpool_d32 %s cycles = %d (elements = %d->%d) thr=%d\n",
			use_hvx?"hvx": "ref",(end_time-start_time),
					(int)tensor_element_count(in_tensor),
			(int)tensor_element_count(out_tensor), n_threads);
#endif
	logmsg(nn,2,"avgpool_d32 %p done", self);
	return 0;
}

// worker thread - calls avgpool_process_slice on the next slice until they are all done
static void
avgpool_worker_thread( struct nn_graph *nn, void *thrpv )
{
	struct avgpool_thrinfo *thrp = (struct avgpool_thrinfo *)thrpv;
	struct avgpool_runstate * rstp = thrp->rstp;
	void *integ_buf = thrp->integ_buf;
	struct integral_buffer_plan const *ibp = rstp->ibp;
	avgpool_process_slice_fp slice_func = rstp->process_slice_func;

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

//
// special case where we reduce all to 1x1.
//
// We find the int32 sum in each depth slice (32 for each); find the min
// and max of these, and then calculate ranging.
// In order to avoid the range being artificially expanded by values in the
// padding, depth slice results in the first and last slice are 'trimmed' by
// forcing the 'depth padding' results  to  -in_min* winsize * 255/(in_max-in_max), which is
// the sum you get for all 'zero' input.
// This is run in multiple threads to do the work, with a 32-bit result at rstp->reduce1x1_result
// and then avgpool_reduce_to_1x1_post_worker_thread is called (once the range is known)
// to convert that to u8.
//

static void
avgpool_reduce_to_1x1_worker_thread( struct nn_graph *nn, void *thrpv )
{
	struct avgpool_thrinfo *thrp = (struct avgpool_thrinfo *)thrpv;
	struct avgpool_runstate * rstp = thrp->rstp;
	struct integral_buffer_plan const *ibp = rstp->ibp;
	int nd32 = ibp->tin.nd32;			// # of depth slices

	int job_idx = thrp->job0;				// get range of jobs to do
	int job_end = thrp->job_end;
	int batch_idx=0;
	int d32_idx= 0;
	int in_d32_stride = ibp->tin.d32_stride;
	int in_height_stride = ibp->tin.height_stride;
	float winsize = ibp->window_ht * ibp->window_wid;
	float range_min = rstp->range_min;
	float range_max = rstp->range_max;
	// figure out the current range.
	// By considering the 'input' (sums) to be scaled over
	// 0.. 255*winsize, instead of 0..255, we get the /winsize in here too.
	float rdiff= (range_max-range_min)/(float)(255 * winsize);
	// this is the 'zero' in the acc'd input values.
	float in_zero = -range_min/rdiff;
	HVX_Vector v_acczero = Q6_V_vsplat_R( roundf_i32( in_zero));

	// make a mask vector:
	//  bit 0 = 1 in slots which are depth-padding in the first depth slice
	//  bit 1 = 1 in slots which are depth-padding in the last depth slice (when nd32 >= 2)
	// all other = 0
	//
	int d0 = ibp->tin.d0;
	int depth_all = d0 + ibp->inshape.depth;
	HVX_Vector depthmask = Q6_V_vand_QR( Q6_Q_vsetq_R(4*d0),1);	// correct result for 'left' depth padding
	if( (depth_all&31) != 0){	// last one has depth padding on right
		HVX_VectorPred qright = Q6_Q_not_Q( Q6_Q_vsetq_R(4*depth_all));
		depthmask  = Q6_V_vandor_VQR( depthmask, qright , (nd32==1)?1:2);
	}

	union { HVX_Vector as_v;
		int32_t as_i32[32];
	}uu;
	HVX_Vector all_max = Q6_V_vsplat_R(0x80000000);
	HVX_Vector all_min_not = all_max;

	// The address of the slice data
	uint8_t const * src_data0 = ibp->tin.data;
	uint8_t const * src_data = src_data0;

	if( job_idx> 0){				// starting in the middle
		batch_idx =  job_idx/(unsigned)nd32;
		d32_idx = job_idx - batch_idx * nd32;
		src_data = src_data0 + batch_idx * ibp->tin.batch_stride + d32_idx * in_d32_stride;
	}
	// it's important to use the height & wid from the plan, which may be trimmed from the actual input tensor.
	int in_height = ibp->inshape.height;
	int in_width = ibp->inshape.width;

	for( ;  job_idx < job_end; job_idx ++) {
		// find the sum of the whole slice
		HVX_Vector slc_sum = hvx_sum_hxw_d32_slice( src_data, in_height, in_width, in_height_stride);
		((HVX_Vector*)rstp->reduce1x1_result)[ job_idx] = slc_sum;	// save for later

		// Change 'padding' lanes to v_acczero, for range calculation.
		// Extract the applicable mask from 'depthmask' (for general case, mask is zero).
		HVX_VectorPred qpadding = Q6_Q_vand_VR(depthmask, (d32_idx ==0)?1 : (d32_idx==nd32-1)?2:0 );
		slc_sum = Q6_V_vmux_QVV( qpadding, v_acczero, slc_sum);

		// reduce it with previous slices
		all_max = Q6_Vw_vmax_VwVw( all_max, slc_sum);
		all_min_not = Q6_Vw_vmax_VwVw( all_min_not, Q6_V_vnot_V(slc_sum));

		// bump indices & pointer
		if( ++d32_idx >= nd32){
			d32_idx = 0;
			++batch_idx;
			src_data = src_data0 + batch_idx * ibp->tin.batch_stride ;
		}else{
			src_data += in_d32_stride;
		}
	}
	// reduce min/max range laterally
	int k = 4;
	HVX_VectorPair minmax2 = Q6_W_vshuff_VVR( all_max, all_min_not,k );
	HVX_Vector  minmax = Q6_Vw_vmax_VwVw( Q6_V_hi_W(minmax2), Q6_V_lo_W(minmax2));
	// even word lanes have ~xmin; odd have xmax. keep going...
	for( int i  =0; i < 4; i++ ){
		k <<= 1;
		minmax2 = Q6_W_vshuff_VVR( minmax, minmax, k);
		minmax = Q6_Vw_vmax_VwVw( Q6_V_hi_W(minmax2), Q6_V_lo_W(minmax2));
	}
	uu.as_v = minmax;
	Q6_dcfetch_A(&uu);
	// -------

	int32_t minval_not = uu.as_i32[0];
	int32_t maxval = uu.as_i32[1];

	// accumulate min & max to these variables
	nn_atomic_max( &rstp->reduce1x1_min_not, minval_not);
	nn_atomic_max( &rstp->reduce1x1_max, maxval);
	// and we are done, except for the final post

	nn_sem_post( &rstp->done_sem);
}
// this runs once after all the work is done on avgpool_reduce_to_1x1_worker_thread

static void
avgpool_reduce_to_1x1_post_worker_thread( struct nn_graph *nn, void *rstpv )
{
	struct avgpool_runstate * rstp = (struct avgpool_runstate *)rstpv;
	struct integral_buffer_plan const *ibp = rstp->ibp;

	int nd32 = ibp->tin.nd32;			// # of depth slices
	int batches = ibp->inshape.batches;	// # of batches

	float winsize = ibp->window_ht * ibp->window_wid;
	float range_min = rstp->range_min;
	float range_max = rstp->range_max;
	// figure out the current range.
	// By considering the 'input' (sums) to be scaled over
	// 0.. 255*winsize, instead of 0..255, we get the /winsize in here too.
	float rdiff= (range_max-range_min)/(float)(255 * winsize);
	// this is the 'zero' in the acc'd input values.
	float in_zero = -range_min/rdiff;

	// get the range found in the first phase
	int global_min = ~rstp->reduce1x1_min_not;
	int global_max = rstp->reduce1x1_max;


	// We now have all the results, in vectors of int32; one per 'job',
	// and we have the min and max of the whole thing. Determine how
	// to scale it for output.
	// first, find the output range in application units; respect
	//   min <=0, max >= 0
	//
	float fglobal_min = fminf(0.0f,global_min - in_zero) * rdiff;
	float fglobal_max = fmaxf( fmaxf(0.0f,global_max - in_zero) * rdiff, fglobal_min + 0.001f);
	//
	// now we need to work out how to scale the values to get them to that range.
	//
	// The data path is:
	//       (1) double, and add offset;
	//                (the result will now be 0 if the input was at the min end of the range)
	//       (2) scale by a 16-bit 'scale_frac' (>>15) and then >> scale_rsh
	//       (3) divide by 1 with rounding and then limit to 0..FF.
	//
	// The overall scaling is scale_frac * 2^-(15+scale_rsh)
	// since scale_rsh >=0 and we need scale_frac <=32K, that means
	/// the overall scale <= 1. In cases where it wants to be >1 we need to artificially
	// expand the output range.
	//
	float out_range = fglobal_max - fglobal_min; // range of output (prior to adjusting)
	// the corresponding range of codes in the i32 buffer is (out_range/rdiff)
	// So the gain we want to have is 255/(out_range/rdiff), which needs to be < 1.
	// Note that we are working with the non-corrected output range; but correcting
	// it can only make it larger, and the gain smaller.
	//
	if( 255.1f * rdiff >=  out_range){	// the '.1' is a fudge to get some margin
		float expfac = (255.1f*rdiff)/out_range;
		fglobal_min *= expfac;
		fglobal_max *= expfac;
	}
	// now fix up the range
	float out_min, out_max;
	float step;				// (max-min)/255
	float recip_step;		// 255/(max-min)

	quantize_adjust_range( & out_min, &out_max, &step, & recip_step, fglobal_min, fglobal_max );
	rstp->range_min = out_min;
	rstp->range_max = out_max;

	// now, the scale we need is rdiff * recip_step
	// this is from the int32's to the u8's.
	float scale_target = rdiff * recip_step;

	// find the scale_rsh
	int scale_rsh = min_i32( 31, max_i32(0,-flt_getexp(scale_target)));
	float scale_norm = flt_ldexp(scale_target, scale_rsh+15);

	int scale_fac= saturate_i16( roundf_i32(scale_norm ));

	// now - figure out the offset. it is the solution to this:
	//    (2*in_zero + offset)/2 * scale_fac = out_zero
	//
	//  so offset = 2*(out_zero/scale_fac - in_zero)
	// but out_zero = -out_min*255/(out_max-out_min) =  -out_min * recip_step
	//
	float outz_over_scale = 0.0f;
	if( out_min < 0 ){
		// use the actual (quantized) scale for higher precision
		float actual_scale = flt_ldexp( (float)scale_fac, -(scale_rsh+15));
		outz_over_scale = -out_min * recip_step/actual_scale;
	}
	int inp_offset = roundf_i32( 2.0f*( outz_over_scale - in_zero));

	logmsg(nn,2,"reduce to 1x1: offs %d, scale %d, >> %d, out range is %f.. %f",
		inp_offset, scale_fac, scale_rsh, out_min, out_max );

	// now scale all the values to u8. We do 2 at once, in-place in the buffer,
	// since it's more efficient; then copy them out.

	HVX_Vector  * sumptr = (HVX_Vector *)rstp->reduce1x1_result;
	HVX_Vector voffs = Q6_V_vsplat_R(inp_offset);
	HVX_Vector vscl = q6op_Vh_vsplat_R(scale_fac);	// need it in odd words
	int npairs = (rstp->jobs+1)>>1;		// number of pairs

	for( int i =0; i < npairs; i++){
		HVX_Vector sums0 = sumptr[2*i];
		HVX_Vector sums1 = sumptr[2*i+1];

		sums0 = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(sums0,sums0),voffs);	//*2 * offs
		HVX_Vector scaled0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( sums0, vscl);	// * scale
		scaled0= Q6_Vw_vasr_VwR( scaled0, scale_rsh);
		sums1 = Q6_Vw_vadd_VwVw(Q6_Vw_vadd_VwVw(sums1,sums1),voffs);	//*2 * offs
		HVX_Vector scaled1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( sums1, vscl);	// * scale
		scaled1= Q6_Vw_vasr_VwR( scaled1, scale_rsh);

		HVX_Vector val_h = Q6_Vh_vasr_VwVwR_rnd_sat( scaled1,scaled0,1);	// >> 1 and round
		val_h = Q6_Vh_vdeal_Vh( val_h);	// 0..31 [0] followed by 0..31 [1]
		HVX_Vector val_ub = Q6_Vub_vpack_VhVh_sat( val_h, val_h);  		// pack/sat to u8
		// now we have 32 of [0], 32 of[1], 32 of [0], 32 of [1]
		HVX_VectorPair xpos = Q6_W_vdeal_VVR( val_ub, val_ub, 32);	// transpose them out
		// have 4 of each so any alignment will work.
		sumptr[2*i] = Q6_V_lo_W(xpos);
		sumptr[2*i+1] = Q6_V_hi_W(xpos);
	}
	//
	// now copy them all out
	//
	uint8_t * outp0 = ibp->tout.data;
	// outp0 may be misaligned due to left-padding. This is ok, since
	// we are only writing w=1, and all 4 width units are the same in the source data.
	// align the address to make checking tools happy.
	outp0 = (uint8_t *)((size_t)outp0 & ~(size_t)127);
	int out_batch_stride = ibp->tout.batch_stride;
	int out_d32_stride = ibp->tout.d32_stride;
	for(int ib = 0; ib < batches; ib ++){
		for( int id32 = 0; id32 < nd32; id32++){
			uint8_t *outp = outp0 + ib * out_batch_stride + id32 * out_d32_stride;
			*(HVX_Vector *)outp = *sumptr++;
		}
	}

	/* *** scalar version of the above
	int32_t const * ptr = rstp->reduce1x1_result;
	for( int ib = 0; ib < batches; ib++ ){
		for(int id32 = 0; id32 < nd32; id32++){
			uint8_t * outp = ibp->tout.data + ib * ibp->tout.batch_stride + id32* ibp->tout.d32_stride;
			for(int i= 0; i < 32; i++ ){
				int delt = 2*ptr[i] + inp_offset;
				delt = ((int64_t)delt*scale_fac + 0x4000) >> 15;
				delt >>= scale_rsh;
				outp[i] = saturate_u8( (delt+1)>>1 );
			}
			ptr += 32;
		}
	}
	 */

	nn_sem_post( &rstp->done_sem);
}

////////////////////////////////////////////////////////
// Special case for 2x2 with stride_h = 2
////////////////////////////////////////////////////////

static void avgpool_2x2_inner_func(	struct integral_buffer_plan const *ibp,
	int h0, int outht,  int batch );

static void avgpool_2x2_worker_thread_func( struct nn_graph * nn, void *rstpv)
{
	struct avgpool_runstate * rstp = (struct avgpool_runstate *)rstpv;
	struct integral_buffer_plan const *ibp = rstp->ibp;

	int ijob;
	int njobs = rstp->jobs;
	int hslice_shift= rstp->hslice_shift;	// 0,1 or 2
	int hslice_mask = (1<<hslice_shift)-1;	// 0,1 or 3

	while( ijob = __sync_fetch_and_add(&rstp->next_job, 1), ijob < njobs ){
		int ibatch = ijob >> hslice_shift;		// this is the job #
		int const * hbrks = &rstp->hbreaks[ ijob & hslice_mask];	// point to 'breaks' table
		int h0 = hbrks[0];		// first output row we do
		int oht = hbrks[1] - h0;		// number of rows we do
		avgpool_2x2_inner_func( ibp, h0, oht, ibatch);
	}
	nn_sem_post( &rstp->done_sem);
}


//////////////////////////////////////////////////////////////////////////////
// code for doing avgpool with 2x2 window, h_stride = 2.
// The operation does top/bottom/left padding, as needed, on-the-fly;
// if right-padding is needed, it will zap it into the input buffer
// prior to starting work (the padding is done by replicating the last col,
// so the average works out properly). Most uses of this operation
// don't need padding.
//
// If left padding is needed, we operate as if the padding had been done,
// and then actually do it using a vdelta op to copy bytes within
// the first vector.
// So, the (padded) input width is always even, and the output width is
// always half as much.
//
// Procedure is:
//  - read 2 input rows, via a 'vlalign' operation, so that we start by
//    reading the first 4 pixels in each vector (include 'padding' if applicable).
//  - widen-add the vectors from the two rows; result is two vecs of 16 bit
//  - shuffle these, and add together to add the even and odd w slices
//  Result is a vector with 64 sums, 32 for even 32 for odd output w.
//  given 2 of these, we reduce-asr to 8 bit, and then 'deal' the result
//  to get one output vector.
//
//   h  | w1 w0 d4 d3 d2 d1 d0  // inputs
//   d0 | w1 w0 d4 d3 d2 d1 --  // widening add
//   w0 | w1 d4 d3 s2 d1 d0 --  // shuffle n = 62
//   -  | w1 d4 d3 s2 d1 d0 --  // add together
//
// Combining two results:
//   -  | w1 d4 d3 s2 d1 d0 w2  // reducing >>2 with rnd/sat to u8
//      | w2 w1 d4 d3 d2 d1 d0  // byte deal
//
static inline
HVX_Vector avgpool_2x2s2_stage1( HVX_Vector vin0, HVX_Vector vin1 )
{
	HVX_VectorPair vsum = Q6_Wh_vadd_VubVub( vin1, vin0 );	// vertical add
	HVX_VectorPair vshuf = Q6_W_vshuff_VVR( Q6_V_hi_W(vsum),Q6_V_lo_W(vsum),64-2);
	return Q6_Vh_vadd_VhVh( Q6_V_hi_W(vshuf),Q6_V_lo_W(vshuf));
}
// take two results from stage1; round, pack, reorder for the output.
// Each input vector contains the results for two output pixels.
static inline
HVX_Vector avgpool_2x2s2_stage2( HVX_Vector vin23, HVX_Vector vin01 )
{
	HVX_Vector rounded = Q6_Vub_vasr_VhVhR_rnd_sat( vin23, vin01, 2);	// >>2 and pack
	// but pixels 0,1 are in even bytes, and 2,3 in odd, so...
	return Q6_Vb_vdeal_Vb( rounded);
}

// process one batch of the work, starting at output row h0, for 'outht'
// output rows.

static void avgpool_2x2_inner_func(	struct integral_buffer_plan const *ibp,
	int h0, int outht,  int batch )
{
	int margin_left = ibp->wpad_left;	// 0 or 1 only!
	int margin_top = ibp->wpad_top;		// 0 or 1 only!
	int stride_h = ibp->stride_ht;		// >= 1
	int nd32 = ibp->tin.nd32;

	int in_height_stride = ibp->tin.height_stride;
	int in_d32_stride = ibp->tin.d32_stride;

	uint8_t const * in_batch = ibp->tin.data + batch * ibp->tin.batch_stride;

	// work out the actual range of input rows need to cover this output span
	// (used for l2 prefetch).
	int hbase = h0*stride_h-margin_top;  // start row (may be -1, if margin_top = 1)
	int h_begin = max_i32(hbase,0);	// first actual input row we need
	int h_end = min_i32(hbase+stride_h*(outht-1)+2, ibp->inshape.height); //last+1
	int h_inrows = h_end - h_begin;	// # of input rows we need, starting at hbegin

	// work out the end-of-row, for top row, for l2 prefetch calc.
	// (doesn't include right padding .. yet)
	uint8_t const *in_batch_endrow = in_batch + ibp->inshape.width*32;

	if( ibp->wpad_right>0){	// ok we need to do right padding....
		// do all the input rows needed for this output span. when stride =1
		// there will be overlap with other extents, but that should be ok.
		uint8_t * posn = // dest position
				(uint8_t*)in_batch_endrow
				                   + h_begin * in_height_stride;
		vmemcpy_2d_asm( 32, nd32*h_inrows,	// wid, height
				posn, in_d32_stride,			// dest, stride
				posn-32, in_d32_stride );		// ht, stride;
		in_batch_endrow += 32;	// since we'll need to read that
	}
	int wout = ibp->outshape.width;  //output wid
	int out_height_stride = ibp->tout.height_stride;
	int out_d32_stride = ibp->tout.d32_stride;

	uint8_t * out_batch = ibp->tout.data + batch * ibp->tout.batch_stride;

	//
	// figure out how to align the data.
	// using a vlalign will bring the first input to the start of vector.
	// if a 'preload' is needed on each row, the pointer will be bumped
	// so the preload is the previous vector.
	in_batch -= margin_left*32;	// adjust as if we need to read margin
	unsigned lalign = (size_t) in_batch & 0x60;	// intra-vector offset.
	int valamt = 0;
	int need_preload = 0;	// do we need to preload 'previous'?
	if( lalign != 0){ // preload is needed, unless valamt=0x20 and margin_left=1
		valamt = 128-lalign;		// 0x20,0x40,0x60
		if( valamt > 32*margin_left){	// we need preload
			need_preload = 1;
		}
		in_batch += valamt;		// first 'non-preload' read.
	}

	uint8_t* out_pos0 = out_batch + h0 * out_height_stride;		// output starts here.

	// determine the start for first
	// pass across: stride_h*h0-margin_top rows down. When h0=0 and margin_top =1,
	// this will be one row above the input; we make an adjustment later.
	uint8_t const * in_pos0 = in_batch + in_height_stride*hbase;
	// the start of the last row, for bottom margin
	uint8_t const* in_pos_last = in_batch + in_height_stride*(ibp->inshape.height-1);


	// prefetch: point to the first thing we need.
	uint8_t const *pf_ptr = in_batch - 128*need_preload;	// first vec we need
	unsigned pf_width = min_u32( in_d32_stride, in_batch_endrow - pf_ptr);			// bytes we need to read
	pf_ptr += h_begin*in_height_stride;						// adjust for start position.

	// we initially issue a certain # of prefetch rows; then, before starting each row,
	// issue enough to cover stride_h more rows.
	int pf_rows_issued = (in_height_stride >=16384 )? 2: (1 << (Q6_R_cl0_R(in_height_stride)-16));
	pf_rows_issued = min_i32( pf_rows_issued, h_inrows);		// max is the whole thing

	l2fetch( pf_ptr, in_d32_stride, pf_width, nd32*pf_rows_issued);
	pf_ptr += pf_rows_issued*in_height_stride;

	// lpad_ctl is all zero, unless we need 'margin_left' in which case it is 32 in the
	// first 32 bytes.
	HVX_Vector lpad_ctl = Q6_V_vand_QR( Q6_Q_vsetq_R( margin_left?32:0), 0x20202020);
	// # if input vectors to process (after alignment,counting margins)
	// int winvecs = (wout+1)>>1;   // >=1
	// int in_vecs_even = (winvecs&1)==0;	// used in loop
	int in_vecs_even = ((wout-1)&2)!=0;
	// # of times to run the w vector loop (one vector out per loop)
	int wloopcnt = (wout-1)>>2;		// >= 0

	//wout  winvecs in_vecs_even wloopcnt
	//    1     1         0         0
	//    2     1         0         0
	//    3     2         1         0
	//    4     2         1         0
	//    5     3         0         1
	//    6     3         0         1
	//    7     4         1         1
	//    8     4         1         1
	//    9     5         0         2

	HVX_Vector vin0_prev = Q6_V_vzero();
	HVX_Vector vin1_prev = Q6_V_vzero();

	for(int ih =0; ih < outht; ih++){
		// prefetch?
		if( pf_rows_issued < h_inrows){
			int pf_next = min_i32( pf_rows_issued+stride_h, h_inrows);
			int pf_now = pf_next-pf_rows_issued;
			l2fetch( pf_ptr, in_d32_stride, pf_width, nd32*pf_now);
			pf_ptr += pf_now*in_height_stride;
			pf_rows_issued = pf_next;
		}

		uint8_t * outp = out_pos0 + ih * out_height_stride;
		// set up two input pointers, to adjacent rows (usually)
		uint8_t const * inp0 = in_pos0+ ih*stride_h*in_height_stride;
		uint8_t const * inp1 = inp0 + in_height_stride;
		// don't fall off the top
		if(inp0 < in_batch)inp0 = inp1;
		// don't fall off the bottom.
		if(inp1 > in_pos_last) inp1 = inp0;
				// row_delta is either in_height_stride, or 0; it is used so
		// we don't need to keep updating inp1
		uint32_t row_delta = inp1-inp0;

		// d32 loop.
		for(int id32 = 0; id32 < nd32; id32++){
			HVX_Vector const * vinp0 = (HVX_Vector const * )inp0;
			HVX_Vector const * vinp1 = (HVX_Vector const * )(inp0 + row_delta);
			HVX_Vector * voutp = (HVX_Vector *)outp;
			inp0 += in_d32_stride;
			outp += out_d32_stride;

			// prepare for w loop
			if( need_preload){
				vin0_prev = vinp0[-1];
				vin1_prev = vinp1[-1];
			}
			// the inner loop reads 2 vectors from 2 rows and writes one.
			// But it is unpeeled, so the 'prolog' reads one vector from each row,
			// and passes a half-result (2 output pixels) to the loop;
			// In some cases the loop has 0 iterations.
			HVX_Vector vin0 = *vinp0++;
			HVX_Vector vin1 = *vinp1++;
			HVX_Vector va0 = Q6_V_vlalign_VVR( vin0,vin0_prev,valamt);
			HVX_Vector va1 = Q6_V_vlalign_VVR( vin1,vin1_prev,valamt);
			// if 'left padding' is needed, this replaces the first 32
			// with a copy of the second 32. (if not, lpad_ctl=0).
			va0 = Q6_V_vdelta_VV( va0, lpad_ctl);
			va1 = Q6_V_vdelta_VV( va1, lpad_ctl);
			HVX_Vector vsum01 = avgpool_2x2s2_stage1( va0,va1);
			vin0_prev = vin0;
			vin1_prev = vin1;

			for( int i = 0; i < wloopcnt; i++){  // generate one out vec per loop.
				HVX_Vector vin0r = *vinp0++;
				HVX_Vector vin1r = *vinp1++;
				va0 = Q6_V_vlalign_VVR( vin0r,vin0_prev,valamt);
				va1 = Q6_V_vlalign_VVR( vin1r,vin1_prev,valamt);
				HVX_Vector vsum23 = avgpool_2x2s2_stage1( va0,va1);
				vin0 = *vinp0++;
				vin1 = *vinp1++;
				// OK we have vsum01 (from prev iter) and vsum23, so
				// make an output...
				*voutp++ = avgpool_2x2s2_stage2( vsum23, vsum01);

				va0 = Q6_V_vlalign_VVR( vin0,vin0r,valamt);
				va1 = Q6_V_vlalign_VVR( vin1,vin1r,valamt);
				vin0_prev = vin0;
				vin1_prev = vin1;
				vsum01 = avgpool_2x2s2_stage1( va0,va1);
			}
			// vsum01 will always be a valid partial result here; if the
			// # input vectors is even, we need to complete it
			// with one more vector read
			HVX_Vector vsumlast=vsum01;
			if(in_vecs_even){
				va0 = Q6_V_vlalign_VVR( *vinp0,vin0_prev,valamt);
				va1 = Q6_V_vlalign_VVR( *vinp1,vin1_prev,valamt);
				vsumlast = avgpool_2x2s2_stage1( va0,va1);
			}
			*voutp++ = avgpool_2x2s2_stage2( vsumlast, vsum01);
		} // d32 loop;
	}// h loop
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
	int rsh_rnd = (1<<recip_rsh)>>1;

	int depth = 32;

	int wbump = stride_w*32- depth;	// amount to bump pointers at end of w loop

	for( int j = 0; j < out_wid; j++ ){
		for( int id = 0; id < depth; id++){
			int32_t areasum =  (pdn[win_w*32]-pup[win_w*32])-(pdn[0]-pup[0]);
			areasum = saturate_i16( (areasum + rsh_rnd)>> recip_rsh);
			int32_t scaled_sum = (areasum * recip_mant + 0x8000) >> 15;
			optr[j*32+id] = saturate_u8(  (scaled_sum+32)>>6);
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
	int loop_count = (out_wid-1)>>2;

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
		HVX_Vector tmpAC = Q6_Vh_vasr_VwVwR_rnd_sat( sumC, sumA, recip_rsh);
		HVX_Vector tmpBD = Q6_Vh_vasr_VwVwR_rnd_sat( sumD, sumB, recip_rsh);
		// deal these out so we have
		//      A0 A1 ... A31 B0.. B31
		// and  C0 C1 ... C31 D0.. D31
		HVX_VectorPair dealt= Q6_W_vdeal_VVR( tmpBD, tmpAC, -2);
		// now scale by recip_mant
		HVX_Vector scAB = Q6_Vh_vmpy_VhRh_s1_rnd_sat( Q6_V_lo_W(dealt), recip_mant);
		HVX_Vector scCD = Q6_Vh_vmpy_VhRh_s1_rnd_sat( Q6_V_hi_W(dealt), recip_mant);
		// final >>6 and round
		HVX_Vector result = Q6_Vub_vasr_VhVhR_rnd_sat( scCD,scAB, 6);
		result = Q6_Vb_vdeal_Vb(result);
		vULa = pup[0];
		vDLa = pdn[0];
		vURa = pup[win_w];
		*optr++ = result;
	}
	// final 1..4
	{
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
		HVX_Vector tmpAC = Q6_Vh_vasr_VwVwR_rnd_sat( sumC, sumA, recip_rsh);
		HVX_Vector tmpBD = Q6_Vh_vasr_VwVwR_rnd_sat( sumD, sumB, recip_rsh);
		HVX_VectorPair dealt= Q6_W_vdeal_VVR( tmpBD, tmpAC, -2);
		HVX_Vector scAB = Q6_Vh_vmpy_VhRh_s1_rnd_sat( Q6_V_lo_W(dealt), recip_mant);
		HVX_Vector scCD = Q6_Vh_vmpy_VhRh_s1_rnd_sat( Q6_V_hi_W(dealt), recip_mant);
		HVX_Vector result = Q6_Vub_vasr_VhVhR_rnd_sat( scCD,scAB, 6);
		result = Q6_Vb_vdeal_Vb(result);
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
static void avgpool_process_slice( struct avgpool_runstate const *rstp,  void * integ_buf, int batch_idx, int d32_idx  )
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
			avgpool_producer_load_rows( &vsc, ibp, rows_to_load);
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
avgpool_process_slice_hvx(struct avgpool_runstate const *rstp,  void * integ_buf, int batch_idx, int d32_idx  )
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
			avgpool_producer_load_rows_hvx( &vsc, ibp, rows_to_load);

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

//// avgpool_process_slice_special //////
// calls HVX routine for 3x3 case
//
// This is for zapping top & bottom;
// row '-1' will be set to the average of rows 0 and 1;
// row 'ht' will be set to the average of rows h-2 and h-1.
//
// The first store must be masked exactly to the proper start position of the data,
// otherwise we risk corrupting right-zap done in the adjacent depth slice by another
// thread.
//  nvec = # of vectors (>=1)
//   skip_bytes = # of bytes to skip at start of first store (0,32,64 or 96)
//
static inline void
row_zap_2rows( uint8_t * out, uint8_t const *in0, uint8_t const * in1 , int nvec, int skip_bytes )
{
	HVX_Vector const *vinp0 = (HVX_Vector const *)in0;
	HVX_Vector const *vinp1 = (HVX_Vector const *)in1;
	HVX_Vector *voutp = (HVX_Vector *)out;
	HVX_Vector av0 = Q6_Vub_vavg_VubVub_rnd( vinp0[0], vinp1[0]);
	q6op_vstcc_QnAV( Q6_Q_vsetq_R(skip_bytes), &voutp[0], av0);
	for( int i = 1; i < nvec;i++){
		voutp[i]= Q6_Vub_vavg_VubVub_rnd( vinp0[i], vinp1[i]);
	}
}

static void avgpool_process_slice_special( struct avgpool_runstate const *rstp,  void * integ_buf, int batch_idx, int d32_idx  )
{
	struct integral_buffer_plan const *ibp= rstp->ibp;
	struct avgpool_asm_parms const *asp = &rstp->asmparms;

	int input_offset = batch_idx * ibp->tin.batch_stride  + d32_idx * ibp->tin.d32_stride;
	int in_row_stride = asp->in_next_row;

	uint8_t const * in_ptr = asp->in_ptr_base + input_offset;
	l2fetch(in_ptr, in_row_stride, (asp->vecs_wide + 1) * 128, 4);
	if (asp->bot_zap_loc)
		l2fetch(asp->bot_zap_loc + input_offset - in_row_stride, in_row_stride, (asp->vecs_wide) * 128, 2);
	uint8_t * out_ptr =asp->out_ptr_base  + batch_idx * ibp->tout.batch_stride  + d32_idx * ibp->tout.d32_stride;


	if(asp->top_zap_loc != NULL){
		uint8_t * rowptr = asp->top_zap_loc + input_offset;
		row_zap_2rows(
			rowptr - in_row_stride,		// output: row up
			rowptr, rowptr + in_row_stride,	// input: 2 rows
			asp->row_zap_nvecs, asp->row_zap_skip);
	}
	if (asp->lr_zap_loc) {
		unsigned char *ptrlb = asp->lr_zap_loc + input_offset + asp->in_wpad_left * 32;
		unsigned char *ptrrb = asp->lr_zap_loc + input_offset + (asp->in_wpad_left+asp->in_width-2)*32;
		l2fetch((void *)((size_t)ptrlb&-128), in_row_stride, 256, asp->lr_zap_height);
		l2fetch((void *)((size_t)ptrrb&-128), in_row_stride, 256, asp->lr_zap_height);
	}
	if(asp->bot_zap_loc != NULL){
		uint8_t * rowptr = asp->bot_zap_loc + input_offset;
		row_zap_2rows(
			rowptr + in_row_stride,			//output : row below
			rowptr - in_row_stride, rowptr,	// input: 2 rows
			asp->row_zap_nvecs, asp->row_zap_skip);
	}
	if( asp->lr_zap_loc != NULL){
		uint8_t * rowptr = asp->lr_zap_loc + input_offset;
		avgpool_zap_lr( rowptr, // pointer to start of row, including width-before padding
			asp->lr_zap_height,		// rows to process
			asp->in_width,		// actual width
			asp->in_wpad_left,  // width_before padding
			in_row_stride);
	}

	avgpool_slice_hvx_3x3_stride1(
			out_ptr, in_ptr,
			in_row_stride,
			asp->out_next_row,
			asp->vecs_wide,
			ibp->outshape.height,
			asp->out_shift);
}



static int avgpool_check(struct nn_node *self, struct nn_graph *nn)
{
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

static int avgpool_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct integral_buffer_plan * ibp = (struct integral_buffer_plan *) self->opaque;
	if( ibp!= NULL){
		if (ibp->edgepad_scale_alloc) nn_free( ibp->edgepad_scale_alloc);
		nn_free( ibp);
		self->opaque= NULL;
	}
	return node_free_common(self,nn);
}


static int avgpool_earlywork_register(struct nn_node *self, struct nn_graph *nn, struct nn_early_work *work)
{
	struct integral_buffer_plan *ibp = self->opaque;
	if (ibp == NULL) return errlog(nn,"not set up yet");
	ibp->misc = work;
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedAvgPool_8_d32 = {
	.execute = avgpool_execute,
	.check = avgpool_check,
	.ctor = node_alloc_common,
	.dtor = avgpool_dtor,
	.earlywork_register = avgpool_earlywork_register,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT | NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedAvgPool_8_d32_ref = {
	.execute = avgpool_execute,
	.check = avgpool_check,
	.ctor = node_alloc_common,
	.dtor = avgpool_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT | NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

