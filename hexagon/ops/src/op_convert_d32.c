
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
#include "hvx_inlines.h"
#include "quantize.h"


#define NUM_D32_THREADS 2


static int get_option( struct nn_graph * nn, const struct tensor * tens_in, int default_val, char const *option_desc, int maxval)
{
	if( tens_in == NULL) return default_val;
	int newval = tensor_get_int32( tens_in,  0 );
	if( newval < 0 || newval > maxval){
		logmsg(nn,2,"convert_d32: value %d out of range (0..%d) for %s; using default of %d",
				newval, maxval, option_desc, default_val );
		return default_val;
	}else{
		return newval;
	}
}
struct conv_to_d32_runstate {
	struct shape op_shape;
	struct tensor_addressing tout;
	uint8_t const * in_data;
	int32_t in_batch_stride;
	int32_t in_height_stride;

	// batch splitting across threads
	int njobs;				// = batches, or 2x batches
	volatile int next_job;	// = next job to do
	int h0,h1;				// two height extents (h1=0 means only one).

	nn_sem_t done_sem;
};

static void hvx_conv_d32_work_func(  struct nn_graph * nn, void * rstpv );


// TODO:
// Identify some special cases that can be done with vmemcpy_2d, especially w=1 cases,
// and depth=32.
//

static int convert_to_d32_execute( struct nn_node *self, struct nn_graph *nn )
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int b_in = in_tensor->shape.batches;
	int h_in = in_tensor->shape.height;
	int w_in = in_tensor->shape.width;
	int d_in = in_tensor->shape.depth;


	// process optional padding
	int d_pad_before = 0;		// defaults
	int w_pad_left = 4;
	int w_pad_right_min = 0;
	int h_pad_top = 4;

	if( self->n_inputs >=2){
		d_pad_before = get_option( nn, self->inputs[1], d_pad_before, "depth padding", MAX_PADDING_DEPTH );
		if( self->n_inputs >=3 )
			w_pad_left = get_option( nn, self->inputs[2], w_pad_left, "width padding(left)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=4 )
			w_pad_right_min = get_option( nn, self->inputs[3], w_pad_right_min, "width padding (min right)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=5 )
			h_pad_top = get_option( nn, self->inputs[4], h_pad_top, "height padding", MAX_PADDING_HEIGHT );
	}
	if( d_pad_before!= 0 ){
		logmsg(nn,0,"depth_pad_before = %d: ignored",  d_pad_before );
	}
	d_pad_before = 0;

	int wtotal = (w_pad_left + w_in + w_pad_right_min + 3)&~3;
	int w_pad_right = wtotal - (w_pad_left + w_in);
	if( w_pad_right > MAX_PADDING_WIDTH) w_pad_right -= 4;

	int d_pad_after = (-(d_in+d_pad_before))&31;
	int h_pad_bottom = h_pad_top;


	logmsg(nn,2,"h: %d|%d|%d w: %d|%d|%d d: %d|%d|%d avail=%d",
			h_pad_top,h_in,h_pad_bottom,
			w_pad_left,w_in,w_pad_right,
			d_pad_before,d_in,d_pad_after,
			out_tensor->max_size);
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		b_in,
		h_in,h_pad_top,h_pad_bottom,
		w_in,w_pad_left,w_pad_right,
		d_in,d_pad_before,d_pad_after,
		NN_TYPE_QUINT8) != 0) {
		logmsg(nn,2,"h: %d|%d|%d w: %d|%d|%d d: %d|%d|%d avail=%d",
			h_pad_top,h_in,h_pad_bottom,
			w_pad_left,w_in,w_pad_right,
			d_pad_before,d_in,d_pad_after,
			out_tensor->max_size);
		return errlog(nn,"out prepare fail (tensor %p)", out_tensor);
	}
	struct conv_to_d32_runstate runstate;
	runstate.tout = tensor_addressing_d32( out_tensor );
	runstate.op_shape = in_tensor->shape;
	runstate.in_data = (uint8_t const*) in_tensor->data;
	runstate.in_height_stride = w_in * d_in;
	runstate.in_batch_stride = w_in * d_in * h_in;

	// figure out how many jobs.
	int njobs = runstate.op_shape.batches;
	int h1 = 0;
	if( njobs < 4 && h_in >= njobs*2){	// split height in two
		njobs *= 2;
		h1= h_in>>1;
	}
	runstate.njobs = njobs;
	runstate.h0 = h_in - h1;
	runstate.h1 = h1;
	runstate.next_job = 0;
	int nthreads = min_i32( njobs, NUM_D32_THREADS);

	nn_sem_init( &runstate.done_sem, 0 );
	for( int i = 0; i < nthreads; i++)
		nn_os_work_for_vector(nn,hvx_conv_d32_work_func, &runstate);
	nn_sem_wait_n_times( &runstate.done_sem, nthreads);

	return 0;
}



static void hvx_to_d32_core_1( struct conv_to_d32_runstate *rstp,
		uint8_t const * inptr, 	uint8_t * outptr, int height) __attribute__((noinline));
static void hvx_to_d32_core_2( struct conv_to_d32_runstate *rstp,
		uint8_t const * inptr, 	uint8_t * outptr, int height) __attribute__((noinline));
static void hvx_to_d32_core_4( struct conv_to_d32_runstate *rstp,
		uint8_t const * inptr, 	uint8_t * outptr, int height) __attribute__((noinline));


//
// do the operation with hvx ops, general case
//
		
static void
hvx_conv_d32_work_func(  struct nn_graph * nn, void * rstpv )
{
	struct  conv_to_d32_runstate * rstp = (struct  conv_to_d32_runstate *)rstpv;
	int in_batch_stride = rstp->in_batch_stride;
	int out_d32_stride = rstp->tout.d32_stride;
	int in_height_stride = rstp->in_height_stride;
	int out_batch_stride = rstp->tout.batch_stride;

	int nd32 = rstp->tout.nd32;		// # of d32 slices
	int nd32_4s = nd32>>2;			// full groups of 4
	int nd32_remain = nd32& 3;

	// get base pointers
	uint8_t const * inp0 = rstp->in_data;
	uint8_t * outp0 = rstp->tout.data;
	//
	/// inp0, outp0 are for even jobs; inp1, outp1 for odd.
	// h0,h1 are the heights.
	// if the height is split, inp1 and outp1 are based on the lower half;
	// if not split, we advance them one batch and double the batch strides
	// (and make h0=h1 = height)
	//
	uint8_t const * inp1;
	uint8_t  * outp1;
	
	int h0 = rstp->h0;
	int h1 = rstp->h1;
	if( h1 != 0 ){			// height is split
		inp1 = inp0 + h0*in_height_stride;
		outp1 = outp0 + h0*rstp->tout.height_stride;
	}else{
		inp1 = inp0 + in_batch_stride;
		outp1 = outp0 + out_batch_stride;
		in_batch_stride *= 2;
		out_batch_stride *= 2;
		h1 = h0;
	}
	int njobs = rstp->njobs;
	int ijobno;
	// run through the jobs...
	
	while( ijobno = __sync_fetch_and_add( &rstp->next_job,1),   ijobno < njobs){
		int oddjob = ijobno & 1;
		uint8_t const * inp = oddjob? inp1:inp0;
		uint8_t  * outp = oddjob? outp1:outp0;
		int height = oddjob ? h1: h0;
		inp += (ijobno>>1)*in_batch_stride;
		l2fetch( inp, 128, 128, (height*in_height_stride+127)/128u);

		outp += (ijobno>>1)*out_batch_stride;

		
		for( int i = 0; i < nd32_4s; i++ ){	
			hvx_to_d32_core_4( rstp, inp, outp, height);
			inp += 32*4;
			outp += 4*out_d32_stride;
		}

		switch( nd32_remain )
		{
		  case 3:
			if (nd32 > 3 ){		// instead of 2+1, back up one and do 4.
				hvx_to_d32_core_4( rstp, inp-32, outp-out_d32_stride, height);
				break;
			}
			// if nd32 is 3, we'll do 2 and then 1.
			hvx_to_d32_core_2( rstp, inp, outp, height);
			inp += 32*2;
			outp += 2*out_d32_stride;
			/* no break */
		  case 1:
			hvx_to_d32_core_1( rstp, inp, outp, height);
			break;
		  case 2:
			hvx_to_d32_core_2( rstp, inp, outp, height);
			break;
		  case 0:
		  default:
			break;
		}
	} // job loop
	nn_sem_post( &rstp->done_sem );
}

//// Inner loop for hvx-based Convert-to-d32
// this function processes height x width x depth
// converting flat to d32; depth is either 32,64, or 128
// This is actually called via one of 3 functions
//     hvx_to_core_1
//     hvx_to_core_2
//     hvx_to_core_4
//
// (according to nd32). So if the actual depth is 172
// you'd use     hvx_to_core_4 to process first 128 (4*32) and then
//               hvx_to_core_2 to process the remaining 44 (32+12)
//
 
//

static inline void __attribute__((always_inline))
hvx_to_d32_core_inline(
		struct conv_to_d32_runstate *rstp,
		uint8_t const * inptr,  		// input
		uint8_t * outptr,				// out (32-byte aligned)
		int height,						// rows to convert
		int nd32						// 1, 2 or 4
	)
{
	int width = rstp->op_shape.width;
	int in_width_stride = rstp->op_shape.depth;
	int in_height_stride = rstp->in_height_stride;
	int out_d32_stride = rstp->tout.d32_stride;
	int out_height_stride = rstp->tout.height_stride;
	

	int w_off_bytes = (size_t)outptr & (128-32);	// 0,32,64,96
	int w_left_pad = w_off_bytes >> 5;				// 0,1,2,3

	outptr -= w_off_bytes;					// align output pointer
	int wpl = w_left_pad + width;
	int remnant = width;
	if( w_left_pad != 0)
		remnant = max_i32(0, wpl-4); // # w units not in first vector
	int nvecs = remnant >> 2;					// # of 'middle' loops
	remnant &= 3;

	HVX_Vector v0,v1,v2,v3;
	v0 = Q6_V_vzero();
	v1 = Q6_V_vzero();
	v2 = Q6_V_vzero();
	v3 = Q6_V_vzero();
	
	int is_aligned = (( in_width_stride | (size_t)inptr)&127)==0;
	
	for( int irow = 0; irow < height; irow++)
	{
		uint8_t const *r0p = inptr+ irow*in_height_stride;
		uint8_t * __restrict w0p = outptr + irow*out_height_stride;
		uint8_t * __restrict w2p = w0p;
		if( nd32 > 2)
			w2p += 2*out_d32_stride;

		// conditionally load at left edge, when w_left_pad != 0
		if( w_left_pad !=0){
			switch(w_left_pad){
				case 1:
					v1 = q6op_V_vldu_A((HVX_Vector const*)r0p);
					r0p += in_width_stride;
					if( wpl==2) break;
					/* no break */
				case 2:
					v2 = q6op_V_vldu_A((HVX_Vector const*)r0p);
					r0p += in_width_stride;
					if( wpl==3) break;
					/* no break */
				default:
				case 3:
					v3 = q6op_V_vldu_A((HVX_Vector const*)r0p);
					r0p += in_width_stride;
					break;
			}
			// store first output vector (only shuffling v1,v2,v3)
			HVX_VectorPair sh13 = Q6_W_vshuff_VVR( v3,v1,64);
			HVX_VectorPair sh01 = Q6_W_vshuff_VVR( Q6_V_lo_W(sh13), Q6_V_vror_VR(v2,64),32);
			HVX_Vector vo0 = Q6_V_lo_W(sh01);
			HVX_Vector vo1 = Q6_V_hi_W(sh01);
			// write to output
			*(HVX_Vector *)w0p  = vo0;
			if( nd32 > 1){
				*(HVX_Vector *)(w0p + out_d32_stride)  = vo1;
				if( nd32 > 2 ){
					HVX_VectorPair sh23 = Q6_W_vshuff_VVR( Q6_V_hi_W(sh13), v2,32);
					*(HVX_Vector *)w2p  = Q6_V_lo_W(sh23);
					*(HVX_Vector *)(w2p + out_d32_stride)  = Q6_V_hi_W(sh23);
					w2p += 128;
				}
			}
			w0p += 128;
		}
		if( nd32 < 4 || !is_aligned){
			// loop over the rest...
			for( int j = 0; j < nvecs; j++){
				v0 = q6op_V_vldu_A((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				v1 = q6op_V_vldu_A((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				v2 = q6op_V_vldu_A((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				v3 = q6op_V_vldu_A((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				HVX_VectorPair sh02 = Q6_W_vshuff_VVR( v2,v0,64);
				HVX_VectorPair sh13 = Q6_W_vshuff_VVR( v3,v1,64);
				HVX_VectorPair sh01 = Q6_W_vshuff_VVR( Q6_V_lo_W(sh13), Q6_V_lo_W(sh02),32);
				HVX_Vector vo0 = Q6_V_lo_W(sh01);
				HVX_Vector vo1 = Q6_V_hi_W(sh01);
				// write to output
				*(HVX_Vector *)w0p  = vo0;
				if( nd32 > 1){
					*(HVX_Vector *)(w0p + out_d32_stride)  = vo1;
					if( nd32 > 2 ){
						HVX_VectorPair sh23 = Q6_W_vshuff_VVR( Q6_V_hi_W(sh13), Q6_V_hi_W(sh02),32);
						*(HVX_Vector *)w2p  = Q6_V_lo_W(sh23);
						*(HVX_Vector *)(w2p + out_d32_stride)  = Q6_V_hi_W(sh23);
						w2p += 128;
					}
				}
				w0p += 128;
			}
		}else{
			// special case: nd32 = 4 and aligned
			for( int j = 0; j < nvecs; j++){
				v0 = *((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				v1 = *((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				v2 = *((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				v3 = *((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				HVX_VectorPair sh02 = Q6_W_vshuff_VVR( v2,v0,64);
				HVX_VectorPair sh13 = Q6_W_vshuff_VVR( v3,v1,64);
				HVX_VectorPair sh01 = Q6_W_vshuff_VVR( Q6_V_lo_W(sh13), Q6_V_lo_W(sh02),32);
				HVX_VectorPair sh23 = Q6_W_vshuff_VVR( Q6_V_hi_W(sh13), Q6_V_hi_W(sh02),32);
				// write to output
				*(HVX_Vector *)w0p  = Q6_V_lo_W(sh01);
				*(HVX_Vector *)(w0p + out_d32_stride)  = Q6_V_hi_W(sh01);
				*(HVX_Vector *)w2p  = Q6_V_lo_W(sh23);
				*(HVX_Vector *)(w2p + out_d32_stride)  = Q6_V_hi_W(sh23);
				w0p += 128;
				w2p += 128;
			}
		}
		if( remnant){
		    v0 = q6op_V_vldu_A((HVX_Vector const*)r0p);
			r0p += in_width_stride;
			if( remnant > 1){
				v1 = q6op_V_vldu_A((HVX_Vector const*)r0p);
				r0p += in_width_stride;
				if( remnant > 2){
					v2 = q6op_V_vldu_A((HVX_Vector const*)r0p);
					r0p += in_width_stride;
				}
			}
			HVX_VectorPair sh02 = Q6_W_vshuff_VVR( v2,v0,64);
			HVX_VectorPair sh01 = Q6_W_vshuff_VVR( v1, Q6_V_lo_W(sh02),32);
			HVX_Vector vo0 = Q6_V_lo_W(sh01);
			HVX_Vector vo1 = Q6_V_hi_W(sh01);
			// write to output
			*(HVX_Vector *)w0p  = vo0;
			if( nd32 > 1){
				*(HVX_Vector *)(w0p + out_d32_stride)  = vo1;
				if( nd32 > 2 ){
					HVX_VectorPair sh23 = Q6_W_vshuff_VVR( Q6_V_vror_VR(v1,64), Q6_V_hi_W(sh02),32);
					*(HVX_Vector *)w2p  = Q6_V_lo_W(sh23);
					*(HVX_Vector *)(w2p + out_d32_stride)  = Q6_V_hi_W(sh23);
				}
			}

		}
	}
}

static void
hvx_to_d32_core_1(
		struct conv_to_d32_runstate *rstp,
		uint8_t const * inptr,  		// input
		uint8_t * outptr,				// out (32-byte aligned)
		int height						// rows to convert
	)
{
	//printf( "[%d,%d] x 32 of %d;  %p->%p\n", 
	//	height, (int)rstp->op_shape.width, (int)rstp->op_shape.depth, inptr, outptr );
	hvx_to_d32_core_inline( rstp,inptr,outptr,height,1);

}
static void
hvx_to_d32_core_2(
		struct conv_to_d32_runstate *rstp,
		uint8_t const * inptr,  		// input
		uint8_t * outptr,				// out (32-byte aligned)
		int height						// rows to convert
	)
{
	//printf( "[%d,%d] x 2*32 of %d;  %p->%p\n", 
	//	height, (int)rstp->op_shape.width, (int)rstp->op_shape.depth, inptr, outptr );
	hvx_to_d32_core_inline( rstp,inptr,outptr,height,2);

}
static void
hvx_to_d32_core_4(
		struct conv_to_d32_runstate *rstp,
		uint8_t const * inptr,  		// input
		uint8_t * outptr,				// out (32-byte aligned)
		int height						// rows to convert
	)
{
	//printf( "[%d,%d] x 4*32 of %d;  %p->%p\n", 
	//	height, (int)rstp->op_shape.width, (int)rstp->op_shape.depth, inptr, outptr );
	hvx_to_d32_core_inline( rstp,inptr,outptr,height,4);
}


////
//// Convert_from_d32
//// 6 different strategies for this. In all cases the work is split into 'jobs' by batches,
//  and sometimes split in two vertically within each batch; so that we have things to do across threads.
// Below, 'entire plane' refers to a height extent of a single batch (if we are splitting; if not, the whole height).
//
/// (1a) general case: repack to an intermediate format, in which depth is padded out
//      to a multiple of 128, and width to a multiple of 4 ( and includes any left padding%4);
//      this done a a few rows at a time and the intermediates are copied to output using memcpy-2d ops
//  (1b) In cases where the width is a multiple of 4, width_before_pad%4 = 0, depth is a multiple of 128,
//      we can do the same thing but directly to the output buffer (but see #2)
//  (2) there is an asm routine which can do any depth multiple of 32, provided left_padding%4==0.
//      This is a superset of (1b); case (1b) is currently unused.
/// (3a) special case when depth=32, we can do the entire plane with a memcpy_2d
//  (3b) special case when width=1 and depth is a multiple of 32 (or < 32): we can do the entire plane with
//       memcpy_2d.
//  (4) when width=1, depth >32 and not a multiple of 32, we can do a loop of memcpy_2d operations.
//
// The order of checking cases is:
//    - if width = 1, we always have (3b) or (4)
//    - otherwise if depth = depth_total = 32, we always have (3a)
//    - otherwise if the conditions for (2) are met, then use (2)
//    - otherwise (1a)
//
enum convert_from_d32_strategy {
	FROM_D32_no_strategy = 0,		// used as 'strategy not  valid'
	FROM_D32_using_general_loop, // with or without intermediate buffer (1a and 1b)
	FROM_D32_using_asm,	     	// (2)
	FROM_D32_using_memcpy_2d,	// (3a) and (3b) above
	FROM_D32_using_memcpy_3d,	// (4) loop of memcpy_2d (one per depth-slice).
	FROM_D32_using_packzilla	// (5) using 'packzilla' strategy
};
// 'info' entries for packzilla strategy
// (see comment above setup_packzilla_info() ).
// if vcollect_valid=1, the 'vcollect' vectors contain the
// actual vrdelta vectors; if = 0, they containg map tables which
// need conversion to vrdelta vectors.
struct packzilla_from_d32_info {
	uint16_t nd_left, nd_middle, nd_right;
	uint16_t nvec_middle;
	uint16_t vcollect_valid;
	// vrdelta for left edge (if nd_left >0). Not vec aligned.
	uint8_t __attribute__((aligned(4))) vcollect_left[128] ;
	// vrdelta for others. Not vec aligned
	uint8_t __attribute__((aligned(4))) vcollect_middle[128];
};


struct conv_from_d32_info{
	struct shape opshape;		// tensor shape
	struct tensor_format in_format;	// for checking if strategy is still ok
	void * in_ptr;					// for checking if strategy is still ok
	struct tensor_addressing tin;
	uint8_t * out_ptr;			// base pointer of output.

	enum convert_from_d32_strategy strategy_code;

	// The tbuf_* apply to FROM_D32_using_general_loop; however, tbuf_rows=0 for all other strategies

	// when temp-buf is not needed, tbuf_rows is 0, and the others reflect the output geometry.
	int tbuf_rows;		// rows in temp buffer (0 if not used)
	int tbuf_width;		// width of tbuffer. Multiple of 4, >= (width_pad_before %4) + width
	int tbuf_depth;		// depth of tbuf; this is in_depth rounded up to a multiple of 128.
	int tbuf_height_stride;	// tbuf_width*tbuf_depth;



	// memcpy_2d parameters: these apply to FROM_D32_using_general_loop (when a temp buf is needed)
	// and also to FROM_D32_using_memcpy_2d, FROM_D32_using_memcpy_3d.
	//
	int copy2d_width;				// width of unit to copy
	int copy2d_height;				// height (multiplied by rows_processed, see below).
	int copy2d_src_stride;			// source stride for the copy
	int copy2d_dst_stride;			// dest stride for the copy
	int copy2d_loop;				// is a '3d' copy needed?
	// for FROM_D32_using_general_loop,
	//    copy2d_loop = 0 means the depth *or* the width are packed in the input, so  we can do a plane
	//                   with a single 2d copy, multiplty copy2d_height by rows;
	//    copy2d_loop = 1 meand we need a separate 2d copy for each row, advancing the pointers by row-stride each time
	// for FROM_D32_using_memcpy_2d:
	//     - single copy, multiply the  copy2d_height by number of rows
	// for FROM_D32_using_memcpy_3d: only dst_stride and src_stride are used; copy2d_height is the row count,
	// copy2d_width is either 32 (or <32 on the last slice).

	struct packzilla_from_d32_info pzilla;	// packzilla info

	int job_ht0, job_ht1;			// input height split in 2 for subjobs (not always; job_ht1 could be 0).
	int n_jobs;
};
struct conv_from_d32_runstate{
	struct conv_from_d32_info const * info;
	volatile int next_jobind;
	// when tmpbuf is used, this points to a pool of tmpbuf for all threads; each is tbuf_rows*tbuf_height_stride.
	uint8_t * tmpbuf_pool;
	volatile int tbuf_alloc;		// used to allocate to threads
	nn_sem_t done_sem;
};
static int setup_packzilla_info( struct conv_from_d32_info *info);

static int
plan_convert_from_d32(struct nn_graph * nn, struct conv_from_d32_info * info,
		struct tensor const *in_tensor, struct tensor *out_tensor)
{
	info->opshape = in_tensor->shape;
	info->in_format = in_tensor->format;
	info->in_ptr = in_tensor->data;
	info->tin = tensor_addressing_d32( in_tensor);

	int in_w_pad_before = info->in_format.width_pad[0];
	int depth = info->opshape.depth;
	int width = info->opshape.width;
	int height = info->opshape.height;
	int out_height_stride = width * depth;

	info->tbuf_rows = 0;		// flag 'no buffer' until otherwise determined.
	info->strategy_code = FROM_D32_no_strategy;	// for error returns.
	if( info->tin.d0 != 0){
		return errlog(nn,"unsupported: input tensor has depth_before !=0");
	}
	// prepare output
	if (tensor_out_prepare_normal_fromshape(out_tensor,&info->opshape,NN_TYPE_QUINT8)) {
		return errlog(nn,"can't prepare output bhwd=%d,%d,%d,%d out_size=%d",
			(int)info->opshape.batches,
			(int)info->opshape.height,
			(int)info->opshape.width,
			(int)info->opshape.depth,
			(int)out_tensor->max_size);
	}
	info->out_ptr = out_tensor->data;

	// how many jobs? each batch is a job, and we usually split height in two as well.
	int job_count = info->opshape.batches;
	int thread_height = height;			// height per thread
	if( height >= 4 || (job_count <4  && height >= 2)){	// split height
		thread_height  = (height+1)/2u;
		job_count *= 2;
	}
	info->n_jobs = job_count;
	info->job_ht0 = thread_height;
	info->job_ht1 = height - thread_height;

	// work out padding & alignment.
	// check for special cases

	in_w_pad_before &= 3;
	if( width == 1){		// special case : FROM_D32_using_memcpy_2d or FROM_D32_using_memcpy_3d
		if( depth <=32 || depth == 32 * info->tin.nd32_total){	// can be done as single 2d copy
			int per_row = min_i32(depth,32);			// width of the 2d memcpy; divides depth exactly.
			info->strategy_code = FROM_D32_using_memcpy_2d;
			info->copy2d_height = info->tin.nd32;		// will be multiplied by rows at runtime
			info->copy2d_width = per_row;
			info->copy2d_src_stride = (depth <= 32)? info->tin.height_stride: info->tin.d32_stride;
			info->copy2d_dst_stride = per_row;
		}else{
			// 'depth' does not run cleanly from row to row.
			// we will need 'nd32' 2d copies; each is 32 wide except for the last. The height of
			// each is the row count of the plane. Here we just set the strides.
			info->strategy_code = FROM_D32_using_memcpy_3d;
			info->copy2d_src_stride = info->tin.height_stride;
			info->copy2d_dst_stride = out_height_stride;;
		}
		return 0;
	}
	if( (depth & 31) == 0 ){
		if( depth == 32 && info->tin.nd32_total == 1 ){
			// special case: depth is 32, each row is already packed in the right format.
			// so it's a 2d memcpy
			int per_row = depth*width;
			info->strategy_code = FROM_D32_using_memcpy_2d;
			info->copy2d_height = 1;		// will be multiplied by rows at runtime
			info->copy2d_width = per_row;
			info->copy2d_src_stride = info->tin.height_stride;
			info->copy2d_dst_stride = per_row;
			return 0;
		}
		if( in_w_pad_before == 0 ){
			// use asm op
			info->strategy_code = FROM_D32_using_asm;
			return 0;
		}
	}
	if( depth <= 31){
		int res = setup_packzilla_info( info );
		if( res != 0) return res;
		info->strategy_code = FROM_D32_using_packzilla;
		return 0;
	}

	// OK, need to use general loop
#define FROM_D32_WD_CONTIG 1		// means that tbuf_depth = depth, so w*depth can be copied at once
#define FROM_D32_HW_CONTIG 2		// means tbuf_width = width, so all rows can be treated as a group.
					// if both are contiguous, we don't need the temp buffer, we can just go direct to out.

	info->strategy_code = FROM_D32_using_general_loop;
	int depth_padded = (depth+127)&~127;
	int width_padded = (width + in_w_pad_before + 3) & ~3;
	int flags = ((depth_padded == depth)? FROM_D32_WD_CONTIG:0) |  ((width_padded == width)? FROM_D32_HW_CONTIG:0);

	info->tbuf_width = width_padded;
	info->tbuf_depth = depth_padded;
	int tbuf_height_stride = width_padded*depth_padded;
	info->tbuf_height_stride = tbuf_height_stride;

// try to make the temp buf about this size.
#define TBUF_TARGET_SIZE (32*1024)

	if( flags == (FROM_D32_WD_CONTIG|FROM_D32_HW_CONTIG)){
		info->tbuf_rows = 0;			// don't need tbuf
	}else{
		// tbuf rows is max(1, min( thread_height, TBUF_TARGET_SIZE/tbuf_row_size))
		//
		int tbuf_rows = 1;
		unsigned tbuf_row_size = width_padded*depth_padded;
		if( thread_height > 1  && tbuf_row_size <= TBUF_TARGET_SIZE/2){	// make it larger
			if( thread_height *tbuf_row_size <= TBUF_TARGET_SIZE){
				tbuf_rows = thread_height;		// do all of it in one shot...
			}else{
				tbuf_rows = TBUF_TARGET_SIZE/(unsigned)tbuf_row_size;
			}
		}
		info->tbuf_rows = tbuf_rows;


		// strategize the 2d-copy done after each chunk.
		// if flags = FROM_D32_WD_CONTIG:
		//  the depths are packed together, but not the widths; so we can copy
		//  each row as a unit.
		//    width = depth*in_width,  height = rows_now
		// if flags = FROM_D32_HW_CONTIG:
		//  the width are packed together, but not the depth; so we can copy
		//  each w-unit as a 'row', and all rows at once.
		//    width = depth*in_width,  height = rows_now
		// if flags are zero, we need to loop across the rows in the buffer
		// and use a 2d memcpy for each one. This is the same as the last
		// case in terms of strides & width; also this case maps to the previous
		// when tbuf_rows =1, since we only will have the one row.
		// Also, when width = 1, we can always use the FROM_D32_WD_CONTIG case.
		//
		info->copy2d_loop = 0;

		if( tbuf_rows){
			if( flags == FROM_D32_WD_CONTIG || width == 1){
				info->copy2d_width = out_height_stride;
				info->copy2d_height =1;			// will get multiplied by copynow
				info->copy2d_src_stride  = info->tbuf_height_stride;
				info->copy2d_dst_stride = out_height_stride;
			}else{
				info->copy2d_width = depth;
				info->copy2d_height  = width;			// will get multiplied by copynow if copy2d_loop=0
				info->copy2d_src_stride  = depth_padded;
				info->copy2d_dst_stride = depth;
				if( flags == 0 && tbuf_rows > 1){		// need loop of copies.
					//@@ maybe transpose this case if that seems better (e.g. large width). Not so easy
					// to handle in the execution though.
					info->copy2d_loop = 1;
				}
			}
		}
	}
	return 0;
}

static void hvx_convert_from_d32_work_function( struct nn_graph *nn, void * rstpv);
static void convert_packzilla_vcollect_workfunc( struct nn_graph *nn, void *rstpv );

static inline void __attribute__((always_inline))
hvx_from_d32_core_inline(
		struct conv_from_d32_info const *info,
		uint8_t const * inptr,  		// input (vector aligned)
		uint8_t * outptr,				// out (vector_aligned)
		int height						// rows to convert
	);

static inline int convert_from_d32_check_valid_plan(struct conv_from_d32_info const *info, struct tensor const * in_tensor)
{
	if( info->strategy_code != FROM_D32_no_strategy
		&& shape_matches( &in_tensor->shape, &info->opshape )
		&& format_matches( &in_tensor->format, &info->in_format)
		&& in_tensor->data == info->in_ptr )
		return 1;
	return 0;
}


static int convert_from_d32_execute(struct nn_node *self, struct nn_graph *nn)
{

	struct tensor const * in_tensor = self->inputs[0];
	struct tensor * out_tensor = self->outputs[0];
	struct conv_from_d32_info *info = (struct conv_from_d32_info *)self->opaque;

	if (!convert_from_d32_check_valid_plan(info, in_tensor)){
		int k = plan_convert_from_d32( nn, info, in_tensor, out_tensor);
		if( k!= 0) return k;
	}

	logmsg(nn,3, "convert_from_d32 [%d, %d:%d:%d, %d:%d:%d, %d:%d:%d]",
			(int)info->opshape.batches,
			(int)in_tensor->format.height_pad[0], (int)info->opshape.height, (int)in_tensor->format.height_pad[1],
			(int)in_tensor->format.width_pad[0], (int)info->opshape.width,  (int)in_tensor->format.width_pad[1],
			 (int)in_tensor->format.depth_pad[0], (int)info->opshape.depth, (int)in_tensor->format.depth_pad[1] );

	logmsg(nn,4,"tbuf_rows = %d; width,depth,stride = %d,%d,%d  job_ht = %d,%d\n",
			info->tbuf_rows, info->tbuf_width, info->tbuf_depth, info->tbuf_height_stride,
			info->job_ht0, info->job_ht1);

	struct conv_from_d32_runstate rst;
	rst.info = info;
	rst.tbuf_alloc = 0;
	rst.next_jobind = 0;
	nn_sem_init(&rst.done_sem,0);

	if( info->strategy_code == FROM_D32_using_packzilla && info->pzilla.vcollect_valid==0){
		// need to convert to vrdelta codes via hvx
		/*
		static char const fmt[] = "%02X%c";
		printf("vleft:\n");
		for(int i =0; i < 128; i++) 	printf(fmt, info->pzilla.vcollect_left[i], ((i&31)==31)?'\n':' ');
		printf("vmiddle:\n");
		for(int i =0; i < 128; i++) 	printf(fmt, info->pzilla.vcollect_middle[i], ((i&31)==31)?'\n':' ');
		printf("\n");
		*/
		nn_os_work_for_vector(nn,convert_packzilla_vcollect_workfunc, &rst );
		nn_sem_wait( &rst.done_sem);
		/*
		printf("vleft:\n");
		for(int i =0; i < 128; i++) 	printf(fmt, info->pzilla.vcollect_left[i], ((i&31)==31)?'\n':' ');
		printf("vmiddle:\n");
		for(int i =0; i < 128; i++) 	printf(fmt, info->pzilla.vcollect_middle[i], ((i&31)==31)?'\n':' ');
		printf("\n");
		*/
	}


	// determine threads
	int nthreads = min_i32(info->n_jobs, 2);
	// allocate scratch, if needed
	if( info->tbuf_rows){
		int tbuf_size = info->tbuf_rows * info->tbuf_height_stride * nthreads;
		nn_scratch_reset(nn);
		nn_scratch_grow(nn, tbuf_size);
		void * p = nn_scratch_alloc(nn, tbuf_size);
		if( p == NULL)
			return errlog(nn,"can't alloc %d scratch", tbuf_size);
		rst.tmpbuf_pool = (uint8_t*)p;
	}
	// spawn the threads
	for( int i= 0; i< nthreads; i++){
		nn_os_work_for_vector(nn,hvx_convert_from_d32_work_function, &rst );
	}
	nn_sem_wait_n_times( &rst.done_sem, nthreads);
	return 0;
}

static void packzilla_run_func(struct conv_from_d32_info const *info, uint8_t *outp, uint8_t const *inp, int height);
// work function for hvx convert_from_d32
static void
hvx_convert_from_d32_work_function( struct nn_graph *nn, void * rstpv)
{
	struct conv_from_d32_runstate *rstp = (struct conv_from_d32_runstate *)rstpv;
	struct conv_from_d32_info const *info = rstp->info;


	enum convert_from_d32_strategy strategy_code = info->strategy_code;

	unsigned tbuf_rows = info->tbuf_rows;
	uint8_t *tbuf_ptr = NULL;


	if( tbuf_rows > 0){	 // get a buffer address
		int idx = __sync_fetch_and_add( &rstp->tbuf_alloc, 1);
		tbuf_ptr = rstp->tmpbuf_pool + idx * tbuf_rows * info->tbuf_height_stride;
	}
	int in_batch_stride = info->tin.batch_stride;
	int out_height_stride = info->opshape.width * info->opshape.depth;
	int out_batch_stride = info->opshape.height * out_height_stride;

	// base pointers
	uint8_t const *inp_0 = info->tin.data;
	uint8_t *outp_0 = info->out_ptr;
	int height_0 = info->job_ht0;

	int in_w_left_pad = (((size_t)inp_0 ) & 127)>>5;
	int tbuf_read_offs = in_w_left_pad * info->tbuf_depth;

	if( strategy_code == FROM_D32_using_general_loop )
		inp_0 -= in_w_left_pad *32;			// align pointer


	// 'odd job' base pointers.
	uint8_t const * inp_1;
	uint8_t * outp_1;
	int height_1 = info->job_ht1;
	// if the height is split vertically, offset inp1,outp1 from inp0, outp0
	if( height_1 > 0){
		inp_1 = inp_0 + height_0* info->tin.height_stride;
		outp_1 = outp_0 + height_0 * out_height_stride;
	}else{
		// otherwise, fake it; offset the 'odd' pointers one batch and double
		// the batch strides.
		inp_1 = inp_0 + in_batch_stride;
		outp_1 = outp_0 + out_batch_stride;
		in_batch_stride <<= 1;
		out_batch_stride <<= 1;
		height_1 = height_0;
	}
	int pf_stride = info->tin.d32_stride;
	// don't prefetch if it has large gaps in d32's
	int pf_nd32 = (info->tin.nd32_total < 2*info->tin.nd32)? info->tin.nd32_total :0;
	int pf_width = info->opshape.width*32;

	// now we can process both cases the same way
	int njobs = info->n_jobs;
	int job_idx;


	while( job_idx = __sync_fetch_and_add( &rstp->next_jobind, 1), job_idx < njobs ){
		uint8_t const *inptr = inp_0;
		uint8_t *outptr = outp_0;
		int job_height = height_0;
		if( job_idx & 1){
			inptr = inp_1;
			outptr = outp_1;
			job_height = height_1;
		}
		inptr += (job_idx>>1)*in_batch_stride;
		l2fetch( inptr, pf_stride, pf_width, pf_nd32*job_height);

		outptr += (job_idx>>1)*out_batch_stride;

		// now we have the in, out pointers and job height;
		// the rest depends on strategy.

		switch( strategy_code){
		 case FROM_D32_no_strategy:
		 default:
			 goto  done_thread;		// should not happen
		 case FROM_D32_using_memcpy_2d:
			 // cases which can be done ine one 2d-memcpy
			 vmemcpy_2d_general_asm( info->copy2d_width, info->copy2d_height * job_height,
					 outptr, info->copy2d_dst_stride,
					 inptr, info->copy2d_src_stride );
			 break;
		 case FROM_D32_using_memcpy_3d:
			 // cases which can be done in one 2d-memcpy per depth_slice.
		 	 {
		 		 int dleft = info->opshape.depth;
		 		 int nd32 = info->tin.nd32;
		 		 int copy2d_dst_stride = info->copy2d_dst_stride;
		 		 int copy2d_src_stride = info->copy2d_src_stride;
		 		 int in_bump = info->tin.d32_stride;
		 		 int out_bump = 32;
		 		 for(int i = 0; i < nd32; i++){
					 vmemcpy_2d_general_asm( min_i32(32,dleft), job_height,	// width height
							 outptr + out_bump *i, copy2d_dst_stride,
							 inptr  + in_bump*i, copy2d_src_stride );
					 dleft -=32;
		 		 }
		 	 }
		 	 break;
		 case FROM_D32_using_asm:
		 	 {
		 		 from_d32_asm( inptr, info->tin.d32_stride,
		 				 outptr, info->opshape.width, job_height, info->opshape.depth);
		 	 }
			 break;
		 case FROM_D32_using_general_loop:
		    {
				uint8_t * tptr = outptr;
				int rows_per = job_height;
				if( tbuf_rows>0){		// need a temp buf
					tptr = tbuf_ptr;
					rows_per = tbuf_rows;
				}
				//
				// run it..
				int rows_to_go = job_height;
				do{
					int rows_now = min_i32(rows_to_go, rows_per);
					// process the transposing...
					hvx_from_d32_core_inline( info, inptr, tptr, rows_now);

					if( tbuf_rows == 0) break;		// all done in one call.

					// need to copy from tptr to actual output, 'rows_now' height units.
					// start reading data from here...
					uint8_t const * tptr_rd = tptr +tbuf_read_offs;

					if( info->copy2d_loop ==0){		// do it all in one 2d-memcpy
						 vmemcpy_2d_general_asm( info->copy2d_width, info->copy2d_height * rows_now,
								 outptr, info->copy2d_dst_stride,
								 tptr_rd, info->copy2d_src_stride );
					}else {	// gapped in h and w dims
						int copy2d_width = info->copy2d_width;
						int copy2d_height = info->copy2d_height;
						int copy2d_src_stride = info->copy2d_src_stride;
						int copy2d_dst_stride = info->copy2d_dst_stride;
						int copy2d_src_bump = info->tbuf_height_stride;
						int copy2d_dst_bump = out_height_stride;
						for( int k = 0; k < rows_now; k++){
							 vmemcpy_2d_general_asm( copy2d_width, copy2d_height,
									 outptr+k*copy2d_dst_bump, copy2d_dst_stride,
									 tptr_rd + k*copy2d_src_bump, copy2d_src_stride );
						}
					}
					inptr += rows_per * info->tin.height_stride;
					outptr += rows_per * out_height_stride;
					rows_to_go -= rows_now;
				}while(rows_to_go > 0);
		    }
			break;
		 case FROM_D32_using_packzilla:
			 packzilla_run_func( info, outptr, inptr, job_height);
			 break;
		} // end switch
	} // end while(job)
 done_thread:
	nn_sem_post(&rstp->done_sem);
}
//////////////////////////////////////////////////////////////////////////////////////////////////
// HVX code for convert_from_d32
//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Routine reads nd32 rows at a time (1,2 or 4), transposes, and and writes the data to an output area
// which is vector aligned. If the output depth is a multiple of 128, and width and width_before_padding
// are both multiples of 4, then the output area is the final result; in other cases the output area will be a temp
// buffer, and vmemcpy_2d will be used to move from the output area to the actual output, in decent-sized
// chunks (at least one row; more if the row size is small).
//
// When a temp area is needed, it is based on a width padded out to a multiple of 4, and a depth padded
// out to a multiple of 128. When  width_pad_before %4 != 0, the temp area will likewise have 'left_padding'.
//
//
// for instance, if the w = 7, depth = 90, and we are doing 2 rows at one, the temp area will be
// of shape [2, 8, 128] i.e. 2*8 vectors; this is copied to the [2,7,90] area of the output using
// two 2d memcpys, each 7 groups of 90.
// For threading, the operation is broken up into batches and (if height is large enough) into
// two height extents.
//
//
//     for each chunk of '4' nd4:
//      repeat tbuf_width/4 times in w direction
//       - read 4 depth units, from rows spaced by tin_d32_stride;
//       - transpose each group of 4 vectors, write the 4 results evenly spaced
//       to output (spacing is tbuf_depth).
//     if there is a 'remnant', do the same thing but only read 1,2 or 3.
//
//  repeat 'height' times, input spacing is tin_height_stride,
//  output spacing is tbuf_h_stride.
//
//

static inline void __attribute__((always_inline))
hvx_from_d32_core_inline(
		struct conv_from_d32_info const *info,
		uint8_t const * inptr,  		// input (vector aligned)
		uint8_t * outptr,				// out (vector_aligned)
		int height						// rows to convert
	)
{

	int nd32 = info->tin.nd32;
	int in_d32_stride = info->tin.d32_stride;
	int in_height_stride = info->tin.height_stride;
	int tout_height_stride = info->tbuf_height_stride;
	int tbuf_depth = info->tbuf_depth;

	int w_loops= info->tbuf_width >>2;		//# of loops in w dimension

	int nd32_4loops = (nd32+1)>>2;	// number of loops; 4 at a time; last can be 3

	// height loop
	for (int iht = 0; iht < height; iht++){
		uint8_t const * inpD = inptr + in_height_stride * iht;
		uint8_t *outpD = outptr + tout_height_stride * iht;
		// nd32 loops. we first process groups of 4; the 3rd row is read by
		// a separate pointer, and if the group is 1 short(i.e. last group 3) we adjust
		// that pointer to re-read the same input. There's a separate loop to process
		// a remnant of 1 or two groups.
		int nd32_remain = nd32;

		for( int id32 = 0; id32< nd32_4loops; id32++){
			uint8_t *outpW = outpD;
			uint8_t const *inpW = inpD;
			uint8_t const * inpW3 = inpD + 3* in_d32_stride;
			if( nd32_remain < 4) inpW3 -= in_d32_stride;	// re-read last row

			for(int iw= 0; iw < w_loops; iw++ ){
				HVX_Vector v0 = *(HVX_Vector const *)inpW;
				HVX_Vector v1 = *(HVX_Vector const *)(inpW + in_d32_stride );
				HVX_Vector v2 = *(HVX_Vector const *)(inpW + 2*in_d32_stride );
				HVX_Vector v3 = *(HVX_Vector const *)(inpW3);
				HVX_VectorPair sh02 = Q6_W_vshuff_VVR( v2,v0,64);
				HVX_VectorPair sh13 = Q6_W_vshuff_VVR( v3,v1,64);
				HVX_VectorPair sh01 = Q6_W_vshuff_VVR( Q6_V_lo_W(sh13), Q6_V_lo_W(sh02),32);
				HVX_VectorPair sh23 = Q6_W_vshuff_VVR( Q6_V_hi_W(sh13), Q6_V_hi_W(sh02),32);
				// write to output
				*(HVX_Vector *)outpW  = Q6_V_lo_W(sh01);
				*(HVX_Vector *)(outpW + tbuf_depth)  = Q6_V_hi_W(sh01);
				outpW += 2*tbuf_depth;
				*(HVX_Vector *)outpW  = Q6_V_lo_W(sh23);
				*(HVX_Vector *)(outpW + tbuf_depth)  = Q6_V_hi_W(sh23);
				outpW += 2*tbuf_depth;	// next output location (4 W units per loop)
				inpW += 128;		// next vector in each
				inpW3 += 128;
			}
			inpD += 4*in_d32_stride;
			outpD += 128;
			nd32_remain -= 4;
		}
		// we now have nd32_remain = -1,0,1 or 2.
		// i.e. there may be 1 or 2 passes left to do.
		if( nd32_remain> 0 ){
			uint8_t *outpW = outpD;
			uint8_t const *inpW = inpD;
			uint8_t const * inpW1 = inpD;
			if( nd32_remain > 1) inpW1 += in_d32_stride;	// otherwise re-read last row

			for(int iw= 0; iw < w_loops; iw++ ){
				HVX_Vector v0 = *(HVX_Vector const *)inpW;
				HVX_Vector v1 = *(HVX_Vector const *)(inpW1);
				HVX_VectorPair sh01 = Q6_W_vshuff_VVR( v1, v0,32);
				*(HVX_Vector *)outpW  = Q6_V_lo_W(sh01);
				*(HVX_Vector *)(outpW + tbuf_depth)  = Q6_V_hi_W(sh01);
				outpW += 2*tbuf_depth;
				*(HVX_Vector *)outpW  = Q6_V_vror_VR(Q6_V_lo_W(sh01),64);
				*(HVX_Vector *)(outpW + tbuf_depth)  = Q6_V_vror_VR(Q6_V_hi_W(sh01),64);
				outpW += 2*tbuf_depth;	// next output location (4 W units per loop)
				inpW += 128;		// next vector in each
				inpW1 += 128;
			}
		}
	}
}

/////////////////////// convert-from-32 'packzilla' strategy for use when d <32.
// This just goes through an enire HxW extent, collecting the useful bytes from each vector into a 'accumulator'
// vector, storing each one (using aligned store) as soon as it is full.
//   Associated with the accumulator is a count 'acc_room' of how many unused bytes are in it; this is
//  always 1..128, and the unused bytes are at the start (zero) end of the vector, followed by
//     (128-acc_room) valid bytes (so the last byte is always the most recently stuffed.)
//    We can 'insert' n bytes to this register, which are supplied at the zero end of another vector;
//    if acc_room > n, this just involves shifting in new bytes using accum = valign( newv, accum, n) and
//       then acc_room -= n;  when acc_room <=n, we 'dump' a full vector to memory, and wind up with
//       acc_room increased by (128-n).
//
//   In the inner loop, most input vectors are processed as follows:
//       (1) read the vector, which contains 4*d useful bytes
//       (2) use a precomputed vrdelta to pack the useful bytes to the first 4*d lanes.
//       (3) 'insert' 4*d bytes in the accum, possibly storing out a vector.
//  This is done across each row and from row to row, and is complicated by some things:
//        -- generally at the end of each row there may be a short vector with only 1,2, or 3 width units.
//           These are processed with the same vrdelta but a shorter byte count.
//        -- likewise at the start of each row there may be a short vector; this one requires a different
//           byte count but also a different vrdelta, in order to skip over padding bytes
//
//        -- It general, we will find that the first output byte may go to non-aligned address
//           This is dealt with as follows:
//                (1) the vector stores in the loop (when dumping acc) are done as:
//                         * outp_curr = vec; outp_curr = outp_next;  outp_next ++
//                (2) if the initial address is aligned, we init outp_curr = out, outp_next = out +128(bytes),
//                    acc_room=128; but if it has an offset of k, we set outp_next to out  + (128-k)bytes, i.e.
//                    the first aligned address; and point outp_curr at a temp area which is aligned; and init
//                    acc_room to 128-k, so that the first partial vector goes to the temp area and then the rest
//                    are in their proper place.
//           At the end of the operation we generally have a partial store of 'accum' bytes, and we may also
//           need to move some bytes from the temp area with a partial store. In some cases it could be that
//           outp_curr still points to the temp area, which means we haven't done any store and the bytes in accum
//           need trimming on both ends for the store.
//
//     This is how the width left/right is done:
//         - if the width_before_padding is not a multiple of 4, *and* the width spreads across at least 2
//           vectors, then nd_left is the number of valid bytes in the first vector, and vcollect_left is the
//           vrdelta control to gather them. In other cases nd_left is 0.
//         - nvec_middle is the # of middle vectors, >= 0. It never includes the last vector containing useful
//           bytes, and it only includes the first one when nd_left=0.
//         - nd_middle is the # of bytes in each 'middle' (i.e 4*d), and vcollect_middle is vrdelta control.
//         - nd_right is the # of useful bytes in the last vector, (either 1,2,3 or 4)*d. The last vector
//           also uses vcollect_middle.
//           In the case where only one vector contains useful bytes: nd_left = nvec_middle=0, nd_right = w*d,
//           and vcollect_left is used to process the only input vec.
//
//  Since the vrdelta need to be designed by hvx code, the following 'caching' technique is used:
//  - setup code writes the *mapping* vector to vcollect_left and vcollect_middle, and sets vcollect_valid to 0;
//  - before using the strategy, we check to see if vcollect_valid is 0, and if so, hvx code is used to make
//    them to vrdelta codes, and vcollect_valid is set to 1. So if subsequent runs can use the same strategy
//    this step is skipped.
//
//   the 'mapping' form of the collect vectors is simply the following in each byte lane:
//     - a value from 0..127 indicating the desired source lane for the given lane; or
//     - a value >= 128 when the output is a 'don't care'.
//
//   The operation of 'packing' the bytes into 4*d (or k*d) consecutive bytes is always possible in a vrdelta -
//   it can be shown that vrdelta will support any mapping in which the 'n' outputs you care about are consecutive,
//   and are sourced from any 'n' distinct input lanes that have the same ordering as the corresponding outputs.
//   This is a sufficient (but not necessary) condition for vrdelta support.

// this function sets up the "packzilla_from_d32_info" for a given situation; assuming
//  that info->opshape and info->in_format are set up.

static int
setup_packzilla_info( struct conv_from_d32_info *info)
{
	int wid = info->opshape.width;
	int d = info->opshape.depth;						// must be 1..31 or we shouldn't be here.
	int wpad = info->in_format.width_pad[0] & 3;		// width padding
	int nd_middle = 4*d;
	int nvec = (wpad +wid + 3)>>2;					// total # of vectors to process
	if( d <1  || d >32 || nvec < 1) return -1;		// should never happen

	int nd_left = 0;
	int nd_right;
	if( nvec == 1){		// special case
		nvec = 0;
		nd_right = wid*d;					// all is on the 'right' vector
	}else{
		if( wpad != 0){
			nd_left = (4-wpad)*d;						// break off the first vector as 'left'
			nvec--;
		}
		nvec--;		// account for right
		nd_right = ((wpad +wid)&3)*d;
		if( nd_right == 0){				// right can't be empty.
			nd_right = nd_middle;
		}
	}
	info->pzilla.nd_middle = nd_middle;		// this is always 4*d even if no actual middle.
	info->pzilla.nd_left = nd_left;
	info->pzilla.nd_right = nd_right;
	info->pzilla.nvec_middle = nvec;
	// set up the 'collect'; first fill -1
	uint8_t * wpl = info->pzilla.vcollect_left;
	uint8_t * wpm = info->pzilla.vcollect_middle;
	for( int i =0; i < 32;i++ ){
		((int32_t*)wpl)[i] = -1;
		((int32_t*)wpm)[i] = -1;
	}
	// fill in mapping vec. I.e. if d= 3 and wpad = 1, we want
	//   vcollect_left =   {  32,33,34, 64,65,66,  96, 97, 98, 0xFF ... 0xFF }
	//   vcollect_middle = {  0, 1, 2, 32,33,34, 64,65,66,  96, 97, 98, 0xFF ... 0xFF }

	for( int i = 0; i < 4; i++ ){		// 4 copies...
		for( int j = 0; j < d; j++){
			*wpm++= i*32+j;
			if( i >= wpad ) *wpl++ = i*32+j;
		}
	}
	info->pzilla.vcollect_valid = 0;			// indicate they need transform to vector.
	return 0;
}

//
// hvx work function to convert the 'vcollect_left' and 'vcollect_middle' to rdelta controls.
// This is done in-place, and we then change vcollect_valid from 0 to 1, so it only needs
// doing the first time the plan is used.
static void
convert_packzilla_vcollect_workfunc(  struct nn_graph *nn, void *rstpv )
{
	struct conv_from_d32_runstate *rstp= (struct conv_from_d32_runstate *)rstpv;
	struct conv_from_d32_info *info = (struct conv_from_d32_info *)rstp->info;
	if( info->pzilla.vcollect_valid == 0){
		for( int i = 0; i < 2; i++){
			HVX_Vector * p = (HVX_Vector*)( (i==0)? info->pzilla.vcollect_left: info->pzilla.vcollect_middle);
			HVX_Vector vmap = q6op_V_vldu_A( p );	// get mapping as unaligned load
			HVX_Vector vctl = design_for_delta( vmap, vmap, 1);	// find reverse delta control
			q6op_vstu_AV( p, vctl);
		}
		info->pzilla.vcollect_valid = 1;
	}
	nn_sem_post( &rstp->done_sem);
}
//
// This is the 'run' function for packzilla.
// - do 'height' rows starting at inp, store to 'outp' (which could be misaligned).
// This only is used when d <= 32, so there's no 'd32' loop.
static void
packzilla_run_func(struct conv_from_d32_info const *info, uint8_t *outp, uint8_t const *inp, int height)
{
	HVX_Vector temp_for_misaligned;

	// get input pointer
	uint8_t const * rdp0 = inp;

	rdp0 = (uint8_t const*)( (size_t)rdp0 & ~(size_t)127);

	HVX_Vector * outp_curr, *outp_next;
	unsigned out_offs = (size_t)outp & 127u;
	int acc_room = 128-out_offs;
	// first store goes to output if aligned, or to temp_for_misaligned if not.
	if( out_offs == 0 ){
		outp_curr = (HVX_Vector*)outp;
	}else{
		outp_curr = &temp_for_misaligned;
	}
	outp_next = (HVX_Vector*)(outp+acc_room);	// aligned, second write,

	// get the params that control packing across the loop
	int nd_left = info->pzilla.nd_left;
	int nd_middle = info->pzilla.nd_middle;
	int nvec_middle = info->pzilla.nvec_middle;
	int nd_right = info->pzilla.nd_right;
	int in_height_stride = info->tin.height_stride;
	HVX_Vector vcollect_left = q6op_V_vldu_A( (HVX_Vector const*)&info->pzilla.vcollect_left);
	HVX_Vector vcollect_middle = q6op_V_vldu_A( (HVX_Vector const*)&info->pzilla.vcollect_middle);
	HVX_Vector accum = Q6_V_vzero();

	//printf("nd_left, middle,right= %d %d %d  nvec = %d height = %d ->%p [%p %p]\n",
	//		nd_left, nd_middle, nd_right, nvec_middle, height, outp, outp_curr, outp_next);

	// if nd_left is not zero, bump nvecs, since we do it in the same loop;
	// making sure that the first iteration packs nd_left bytes and the rest pack nd_middle.
	// Also the first vector is always packed (outside loop) with vcollect_left, the rest are
	// packed with vcollect_middle.
	//
	if (nd_left!=0) nvec_middle++;
	else nd_left = nd_middle;
	//
    // if (num_bytes < acc_room) {
    //      shift vnew into last 'num_bytes' of acc'; acc_room -= nbytes }
    // else {
    //      [the previous valid content of acc, and the first (acc_room) bytes of vnew ]--> *curr_ptr;
    //      acc_room += 128-num_bytes
    //      'acc' is left with the remaining bytes (if any) end-justified
    //       curr_ptr = next_ptr++
    //  }
#if 0
	// I can't get the compiler to do these ops without conditional branches, but it can
	// be done with asm()
	// turns out the branches are faster...
#define INSERT_BYTES(VPK,NUM){\
	int n_insert = (NUM); HVX_Vector vtmp, vnew=(VPK);\
    asm("{ %[vTmp] = valign(%[vNew],%[vAcc],%[accRoom]);"    /* speculative.construct of stored data */\
           "p2=cmp.gt(%[accRoom],%[nBytes]);"                /* if accRoom <= nBytes, need store */\
           "%[accRoom]=sub(%[accRoom],%[nBytes])}\n\t"       /* update of accRoom -= nBytes */\
        "{ %[vAcc] = valign(%[vNew],%[vAcc],%[nBytes]);"     /* construct next acc */\
           "if(!p2) vmem(%[curPtr]+#0)=%[vTmp];"             /*    store */\
           "if(!p2) %[curPtr]=%[nextPtr];"                   /*    update store ptr */\
           "if(!p2) %[accRoom]=add(%[accRoom],%[k128]) }\n\t"/*    complete update of accRoom +=128 */\
        "{  if(!p2) %[nextPtr]=add(%[nextPtr],%[k128])}"     /*    update next store ptr */\
        /* %[vStM] does not appear in the code */\
        /* note, the compiler assumes that 'input' regs are preserved by the code, except when it decides */\
        /* to allocate an output in the same place (which is not possible here) */\
        : [accRoom]"+l"(acc_room),  /* input & output must be in r0...r7 */\
          [curPtr] "+r"(outp_curr), /* input & output */\
          [nextPtr]"+r"(outp_next), /* input & output */\
          [vAcc]   "+v"(accum),     /* input & output */\
          [vTmp]  "=&v"(vtmp),      /* used as temp; dummy output. may not alias an input.*/\
          [vStM]   "=m"(*outp_curr) /* dummy output to indicate we store this    */\
         :[nBytes]  "l"(n_insert),  /* input only; must be in r0...r7  */\
          [k128]    "r"(128),       /* input only; contains 128        */\
          [vNew]    "v"(vnew)       /* input only (expected to alias vLD in loop)  */\
         : "p2");                   /* p2 clobbered */\
       }
// this is like INSERT_BYTES but also does VLOAD = vrdelta(*LOADP++, VDELTA)
#define INSERT_BYTES_LOAD_NEXT(VPK,NUM,VLOAD,LOADP,VRDELTA){\
	int n_insert = (NUM); HVX_Vector vtmp,vtmp2,vloaded,vnew=(VPK),vdeltak=(VRDELTA);\
    asm("{ %[vTmp] = valign(%[vNew],%[vAcc],%[accRoom]);"    /* speculative.construct of stored data */\
           "p2=cmp.gt(%[accRoom],%[nBytes]);"                /* if accRoom <= nBytes, need store */\
           "%[accRoom]=sub(%[accRoom],%[nBytes]);"           /* update of accRoom -= nBytes */\
           "%[vLtmp]=vmem(%[rdPtr]++#1);}\n\t"               /* load next vector */\
        "{ %[vAcc] = valign(%[vNew],%[vAcc],%[nBytes]);"     /* construct next acc */\
           "if(!p2) vmem(%[curPtr]+#0)=%[vTmp];"             /*    store */\
           "if(!p2) %[curPtr]=%[nextPtr];"                   /*    update store ptr */\
           "if(!p2) %[accRoom]=add(%[accRoom],%[k128]) }\n\t"/*    complete update of accRoom +=128 */\
        "{  if(!p2) %[nextPtr]=add(%[nextPtr],%[k128]);"     /*    update next store ptr */\
            "%[vLd]=vrdelta(%[vLtmp],%[vDelt]);}"            /* pack bytes in loaded vec */\
        /* %[vStM] %[vStM] do not appear in the code */\
        /* note, the compiler assumes that 'input' regs are preserved by the code, except when it decides */\
        /* to allocate an output in the same place. Here we expect input vNew to be in same reg as output */\
        /* vLd, when code is in a loop, to avoid a move.*/\
        : [accRoom]"+l"(acc_room),  /* input & output must be in r0...r7 */\
          [curPtr] "+r"(outp_curr), /* input & output */\
          [nextPtr]"+r"(outp_next), /* input & output */\
          [rdPtr]  "+r"(LOADP),  /* input & output */\
          [vAcc]   "+v"(accum),     /* input & output */\
          [vLtmp] "=&v"(vtmp2),     /* used as temp; dummy output. may not alias an input */\
          [vLd]    "=v"(vloaded),   /* output of loaded vec. May alias an input. */\
          [vTmp]  "=&v"(vtmp),      /* used as temp; dummy output. may not alias an input.*/\
          [vStM]   "=m"(*outp_curr) /* dummy output to indicate we store this    */\
         :[nBytes]  "l"(n_insert),  /* input only; must be in r0...r7  */\
          [k128]    "r"(128),       /* input only; contains 128        */\
          [vNew]    "v"(vnew),      /* input only (expected to alias vLD in loop)  */\
          [vDelt]   "v"(vdeltak),   /* input only  */\
          [vLdM]    "m"(*LOADP)     /* dummy memory op to indicate we read this value  */\
         : "p2");                   /* p2 clobbered */\
       (VLOAD)=vloaded;}
#else
#define INSERT_BYTES(VPK,NUM)	{\
		int n_insert = (NUM);\
		HVX_Vector tmp = Q6_V_valign_VVR( VPK, accum, min_i32(n_insert, acc_room));\
		if( unlikely(n_insert >= acc_room)){\
			*outp_curr = tmp; outp_curr = outp_next++;\
			tmp = Q6_V_vror_VR( VPK,n_insert);\
			acc_room +=128;\
		}\
		acc_room -= n_insert; \
		accum = tmp;\
	}
#define INSERT_BYTES_LOAD_NEXT(VPK,NUM,VLOAD,LOADP,VRDELTA)	{ \
		HVX_Vector vlt = *LOADP++; INSERT_BYTES(VPK,NUM);\
		(VLOAD)= Q6_V_vrdelta_VV(vlt,(VRDELTA));}
#endif
	// OK, here is the height loop
	//
	for(int ih = 0; ih < height; ih++){
		HVX_Vector const * rdp = (HVX_Vector const *)rdp0;
		HVX_Vector vin = *rdp++;
		HVX_Vector vpk = Q6_V_vrdelta_VV( vin,vcollect_left );	// get nd_left bytes squeezed down
		// width loop : first iteration (if any) packs nd_left bytes; the rest
		// pack nd_middle. Each iteration loads vpk = vrdelta( *rdp++, vcollect_middle) for next.
		int nd_inloop = nd_left;
		for( int i = 0; i < nvec_middle;i++){
			INSERT_BYTES_LOAD_NEXT(vpk,nd_inloop, vpk, rdp, vcollect_middle )
			nd_inloop = nd_middle;
		}
		{ // last
			INSERT_BYTES(vpk,nd_right )
		}
		rdp0 += in_height_stride;
	}
	// storing out any at the end?
	if( acc_room < 128){	// is 1..127
		accum = Q6_V_vror_VR( accum,acc_room);	// align it properly;
		HVX_VectorPred wmask = Q6_Q_vsetq_R( -acc_room);	// only write these.
		if( outp_curr == &temp_for_misaligned){  		// no store has yet been done.
			wmask = Q6_Q_and_QQn( wmask, Q6_Q_vsetq_R(out_offs));	// need left mask
			outp_curr = (HVX_Vector*)(outp-out_offs);
			out_offs = 0;	// suppress the copy out of temp_for_misaligned
		}
		q6op_vstcc_QAV(wmask, outp_curr, accum );
	}
	// do we need to copy a partial vector, stored to temp_for_misaligned?
	if(out_offs != 0){
		outp_curr = (HVX_Vector*)(outp-out_offs);
		q6op_vstcc_QnAV(Q6_Q_vsetq_R(out_offs), outp_curr, temp_for_misaligned );
	}
}



// Convert_to_d32:
// input 0: 'flat 'u8' tensor
//  input 1:  (optional) scalar int: depth padding start - default 0  (0..31)
//  input 2:  (optional) scalar int: width padding start - default 4  (0..MAX_PADDING_WIDTH)
//  input 3:  (optional) scalar int: width padding end (min) - default 0  (0..MAX_PADDING_WIDTH)
//    The 'end' padding will be adjusted up so that the width total is a multiple of 4. If it exceeds
//    MAX_PADDING_WIDTH as a result, it will then be adjusted down by 4.
//  input 4:  (optional): scalar int: top/bottom padding for height  default 4 (0..MAX_PADDING_HEIGHT)
//
// output : d32 u8 tensor

static int convert_from_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	if( self->opaque != NULL) nn_free( self->opaque);
	void * info = nn_calloc(1, sizeof(struct conv_from_d32_info));
	if ( info == NULL) return errlog(nn,"calloc");
	self->opaque = info;
	return 0;
}


struct nn_node_ops nn_ops_for_Convert_from_d32 = {
	.execute = convert_from_d32_execute,
	.check = convert_from_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.flags = NN_NODE_FLAG_D32_INPUT,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_Convert_to_d32 = {
	.execute = convert_to_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(1,5),
	.n_outputs = NN_IOCOUNT(1),
};


/////////////////////////////////////////////////////// To/From d32 16-bit ////////////////////
// This is a placeholder; should eventually use the same strategies as 8-bit d32
///////////////////////////////////////////////////////////////////////////////////////////////

static int convert_from_d32_16b(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int b_in = in_tensor->shape.batches;
	int h_in = in_tensor->shape.height;
	int w_in = in_tensor->shape.width;
	int d_in = in_tensor->shape.depth;

	// propagate the type; if it's not one of the two allowed, use NN_TYPE_QUINT16
	int out_type = (in_tensor->format.type == NN_TYPE_QINT16)? NN_TYPE_QINT16 : NN_TYPE_QUINT16;

	if (tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, out_type)) {
		return errlog(nn, "can't prepare output bhwd=%d,%d,%d,%d out_size=%d",
			b_in, h_in, w_in, d_in, out_tensor->max_size);
	}

	struct tensor_addressing tin = nn_tensor_addressing_d32_16b(in_tensor);

	uint8_t *pout_row = (uint8_t*)out_tensor->data;
	unsigned elbytes= sizeof(uint16_t);
	unsigned bytes_per_row = w_in*d_in * elbytes;

	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );
	for (int b = 0; b < b_in; b++) {
		for (int h = 0; h < h_in; h++) {
			uint8_t const * p_in = tin.data + b*tin.batch_stride + h*tin.height_stride;
			for(int id32 = 0; id32 < tin.nd32; id32++){
				int dn = min_i32( 32, d_in-32*id32);	// depths to copy (1..32)

				nn_mcmanager_vmemcpy_2d( nn, &mcman,
						dn*elbytes,	w_in,					// width, height to copy
						pout_row+(32*elbytes)*id32, d_in*elbytes,		// dest, dest stride
						p_in,  32*elbytes);						// src, src_stride
				p_in += tin.d32_stride;
			}
			pout_row += bytes_per_row;
		} //h
	} //b
	nn_mcmanager_wait( nn, &mcman );
	return 0;
}


static int convert_to_d32_16b( struct nn_node *self, struct nn_graph *nn )
{
	const struct tensor *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	int b_in = in_tensor->shape.batches;
	int h_in = in_tensor->shape.height;
	int w_in = in_tensor->shape.width;
	int d_in = in_tensor->shape.depth;


	// process optional padding
	int d_pad_before = 0;		// defaults
	int w_pad_left = 4;
	int w_pad_right_min = 0;
	int h_pad_top = 4;

	// note, input #1 is legacy 'depth_pad_before, is ignored
	if( self->n_inputs >=3){
		w_pad_left = get_option( nn, self->inputs[2], w_pad_left, "width padding(left)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=4 )
			w_pad_right_min = get_option( nn, self->inputs[3], w_pad_right_min, "width padding (min right)", MAX_PADDING_WIDTH );
		if( self->n_inputs >=5 )
			h_pad_top = get_option( nn, self->inputs[4], h_pad_top, "height padding", MAX_PADDING_HEIGHT );
	}
	// find wtotal, rounded up to even...
	int wtotal = (w_pad_left + w_in + w_pad_right_min + 3)&~3;
	// total padding must be >= 3
	int k = w_in+4 - wtotal;
	if( k >=2 ) wtotal += k&~1;	// add 2 or 4 as needed.

	//if( wtotal < w_in+3) wtotal += 2;

	int w_pad_right = wtotal - (w_pad_left + w_in);

	if( w_pad_right > MAX_PADDING_WIDTH) w_pad_right -= 2;

	int d_pad_after = (-(d_in+d_pad_before))&31;
	int h_pad_bottom = h_pad_top;

	// propagate the type; if it's not one of the two allowed, use NN_TYPE_QUINT16
	int out_type = (in_tensor->format.type == NN_TYPE_QINT16)? NN_TYPE_QINT16 : NN_TYPE_QUINT16;

	logmsg(nn,2,"h: %d|%d|%d w: %d|%d|%d d: %d|%d|%d avail=%d",
			h_pad_top,h_in,h_pad_bottom,
			w_pad_left,w_in,w_pad_right,
			d_pad_before,d_in,d_pad_after,
			out_tensor->max_size);
	if (tensor_out_prepare_padded_d32(
		out_tensor,
		b_in,
		h_in,h_pad_top,h_pad_bottom,
		w_in,w_pad_left,w_pad_right,
		d_in,d_pad_before,d_pad_after,
		out_type) != 0) {
		logmsg(nn,2,"h: %d|%d|%d w: %d|%d|%d d: %d|%d|%d avail=%d",
			h_pad_top,h_in,h_pad_bottom,
			w_pad_left,w_in,w_pad_right,
			d_pad_before,d_in,d_pad_after,
			out_tensor->max_size);
		return errlog(nn,"out prepare fail (tensor %p)", out_tensor);
	}

	struct tensor_addressing tout = nn_tensor_addressing_d32_16b(out_tensor);

	uint8_t const *pin_row = (uint8_t*)in_tensor->data;
	unsigned elbytes= sizeof(uint16_t);
	unsigned bytes_per_row = w_in*d_in * elbytes;


	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );
	for (int b = 0; b < b_in; b++) {
		for (int h = 0; h < h_in; h++) {
			uint8_t  * p_out = tout.data + b*tout.batch_stride + h*tout.height_stride;
			for(int id32 = 0; id32 < tout.nd32; id32++){
				int dn = min_i32( 32, d_in-32*id32);	// depths to copy (1..32)

				if (dn  < 32) {
					// fill depth padding with actual 0
					nn_mcmanager_vmemset32_2d(nn, &mcman, p_out + elbytes * dn, 0, (32-dn)*elbytes, w_in, 32 * elbytes);
				}
				nn_mcmanager_vmemcpy_2d( nn, &mcman,
						dn*elbytes,	w_in,					// width, height to copy
						p_out,  32*elbytes,						// dst, dst_stride
						pin_row+(32*elbytes)*id32, d_in*elbytes);		// src, src stride
				p_out += tout.d32_stride;
			}
			pin_row += bytes_per_row;
		} //h
	} //b
	nn_mcmanager_wait( nn, &mcman );
	return 0;
}
// input 0: d32 'i16' tensor
//
// output : flat 'i16 tensor

struct nn_node_ops nn_ops_for_Convert_from_d32_16b = {
	.execute = convert_from_d32_16b,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT,
	.n_inputs = NN_IOCOUNT(1),
	.n_outputs = NN_IOCOUNT(1),
};


// Convert_to_d32_16b
// input 0: 'flat 'b16' tensor
//  input 1:  (optional) scalar int: depth padding start -(ignored, always 0)
//  input 2:  (optional) scalar int: width padding start - default 2  (0..MAX_PADDING_WIDTH)
//  input 3:  (optional) scalar int: width padding end (min) - default 0  (0..MAX_PADDING_WIDTH)
//    The 'end' padding will be adjusted up so that the width total is a multiple of 2 and L+R padding is at least 3
//  input 4:  (optional): scalar int: top/bottom padding for height  default 4 (0..MAX_PADDING_HEIGHT)
//
// output : d32 16 tensor

struct nn_node_ops nn_ops_for_Convert_to_d32_16b = {
	.execute = convert_to_d32_16b,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_OUTPUT,
	.n_inputs = NN_IOCOUNT_RANGE(1,5),
	.n_outputs = NN_IOCOUNT(1),
};
