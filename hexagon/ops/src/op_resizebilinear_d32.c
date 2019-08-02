/*
 * Copyright (c) 2017-2019, The Linux Foundation. All rights reserved.
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
#include "nn_bufferpool.h"
#define LRN_MAXTHREADS 2

#if defined(__hexagon__)
#define min(a, b) (((a)<(b)) ? (a) : (b))
#endif

//#define TEST_PERFORMANCE
/*
 * resize_bilinear:
 * inputs:
 *   0 -  Input tensor (d32)
 *   1   new_dims tensor (with height and width, int32)
 *   2,3 : input min,max
 *  outputs:
 *   0 : output tensor
 *   1,2: min,max
 */
// This is used to make a table for h. zoom
struct hcontroltab {	// one entry per output column
	uint32_t offs;	// offset to sample the input at (in bytes, from the first 'proper' input pixel in intermed buf)
	int32_t frac;	// fractional sample point ; i.e. we want in[offs/128] *(1-frac) + in[offs/128+1]*frac
					// has 15 fractional bits; dup'd to both halves.
	uint64_t: 0;	// make 64-aligned for faster access.
};
// This is used to hold the precalculated values for the
//  'resize plan' so we don't have to do it each time
struct resize_plan {
	int32_t in_height, in_width;
	int32_t out_height, out_width;
	int64_t vscale;		// in_height/out_height, with 32 fractional bits
	// this must be the last entry
	struct hcontroltab hcontrol[1];	// resized to 'out_width, rounded up to mult. of 4
};

struct bilin_runstate;
typedef void (*run_all_fp)(struct nn_graph *nn, void *rstpv);
typedef void (*run_slice_fp)( uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height );

struct bilin_runstate {
	struct tensor_addressing tin;
	struct tensor_addressing tout;

	int depth;
	int batches;

	int in_ht;		// input height, width
	int in_width;
	int out_ht;		// output size
	int out_width;
	int align_corners;

	run_all_fp run_all_func;
	run_slice_fp run_slice_func;	// needed when run_all_func = bilin_interp_work
	int with_hvx;
	struct resize_plan *planp;
	struct buffer_pool intermed_bufs;

	nn_sem_t done_sem;
	int jobs;		// total # of 'subsections' to run
	volatile int next_job;	 // index of next subsection
};

static void bilin_interp_work(struct nn_graph *nn, void *rstpv);
static void do_bilin_interp_x_single_slice(uint8_t const * in_ptr, int in_row_stride,
		int in_width, int in_height, uint8_t * out_ptr, int out_row_stride,
		int out_width, int out_height) __attribute__((unused));

static void do_bilin_interp_x2_single_slice(uint8_t const * in_ptr, int in_row_stride,
		int in_width, int in_height, uint8_t * out_ptr, int out_row_stride,
		int out_width, int out_height)__attribute__((unused));

void do_bilin_interp_x2_single_slice_HVX(uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height);
static int obtain_scaling_plan( struct nn_node *self, struct bilin_runstate  *rstp);
static void do_bilin_interp_general_all_HVX( struct nn_graph *nn, void *rstpv );


static int resize_bilinear_d32_execute(struct nn_node *self, struct nn_graph *nn)
{

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *newdims_tensor = self->inputs[1];
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];

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
	runstate.run_all_func = bilin_interp_work;
	runstate.run_slice_func = do_bilin_interp_x2_single_slice_HVX;
	runstate.with_hvx = 1;

	runstate.align_corners = 0;
	if (self->n_inputs == 5)
		runstate.align_corners = *(int32_t *)(self->inputs[4]->data) != 0;
	//
	// two back-ends:
	//  (1) a special case, which requires 'width_pad_before' to be even, and 'x2' in both directions;
	//  (2) a general case which doesn't have those restrictions
	//
	int use_hvx_general = 0;

	if( !runstate.align_corners && new_height == height*2 && new_width == width *2 && (width_pad_before&1) == 0 ){
		runstate.run_all_func = bilin_interp_work;
		logmsg(nn, 2, "resizebilinear using x2 mode");
		if((width_pad_before&2) != 0 ){	// in padding is 2...
			runstate.in_width += 2;
			runstate.tin.data -= 64;	// include 2 in pixels in 'gutter'
			runstate.out_width += 4;
			runstate.tout.data -= 128;	// put 4 out pixels in 'gutter'.
		}
	}
	else if (new_height == height && new_width == width) {
		struct nn_memcpy_manager mcman;
		nn_mcmanager_init(nn, &mcman);
		nn_mcmanager_tensor_copy(nn, &mcman, out_tensor, in_tensor);
		tensor_copy(out_min_tensor, in_min_tensor);
		tensor_copy(out_max_tensor, in_max_tensor);
		nn_mcmanager_wait(nn, &mcman);
		goto done;
	}
	else{
		runstate.run_all_func = do_bilin_interp_general_all_HVX;
		use_hvx_general = 1;
	}
	nn_scratch_reset(nn);

	int n_threads = min_i32(2, runstate.jobs);
	nn_sem_init( &runstate.done_sem,0);

	if( use_hvx_general){ // need some setup for this
		if( obtain_scaling_plan(self,&runstate)!= 0){
			return errlog(nn,"didn't get scaling plan");
		}
		// the intermediate area is the full-padded input width *256, multiplied by * of threads.
		unsigned intermed_size = runstate.tin.d32_stride * 4;
		void * mem = nn_scratch_alloc(nn, intermed_size * n_threads);
		if( mem == 0) return errlog(nn, "didn't get temp mem");
		bufpool_init( &runstate.intermed_bufs, n_threads, mem, intermed_size);
	}


	for(int i =0; i < n_threads; i++)
		nn_os_work_for_vector( nn, runstate.run_all_func, &runstate );

	// copy the min and max through
	tensor_copy( out_min_tensor, in_min_tensor );
	tensor_copy( out_max_tensor, in_max_tensor );
	nn_sem_wait_n_times( &runstate.done_sem, n_threads);

done:
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("resizebilinear_d32  cycles = %d (elements = %d) w=(%d->%d) h=(%d->%d)\n",
		(end_time-start_time),	(int)tensor_element_count(out_tensor), width, new_width, height, new_height);
#endif

	logmsg(nn,2,"resizebilinear_d32 %p done",self);
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
	uint32_t in_width = rstp->in_width;
	uint32_t out_width = rstp->out_width;
	uint32_t out_height = rstp->out_ht;
	int in_row_stride = rstp->tin.height_stride;
	int out_row_stride = rstp->tout.height_stride;

	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode, nd32);

	while( job_idx = __sync_fetch_and_add(&rstp->next_job, 1), job_idx < rstp->jobs) {
		int d32_idx = batchslice_decode_update( &bsdecode, job_idx);
		int batch_idx = bsdecode.ibatch;

		// ok, get pointers;
		uint8_t const *in_ptr = rstp->tin.data  + batch_idx * rstp->tin.batch_stride  + d32_idx * rstp->tin.d32_stride;
		l2fetch( in_ptr, pf_stride, pf_width, pf_height);

		uint8_t * out_ptr =     rstp->tout.data + batch_idx * rstp->tout.batch_stride + d32_idx * rstp->tout.d32_stride;

		(*run_slice_func)(in_ptr, in_row_stride, in_width, pf_height, out_ptr, out_row_stride, out_width, out_height);

	}
	nn_sem_post( &rstp->done_sem);
}


// 'reference' single slice.
// Note, pointers here are x32 aligned but not generally 128 aligned
//
static void
do_bilin_interp_x_single_slice( uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height )
{
	uint64_t xstep = (((uint64_t)in_width << 32) + (out_width>>1)) / out_width;
	uint64_t ystep = (((uint64_t)in_height << 32) + (out_height>>1)) / out_height;

	uint64_t ycoord = (1<<23);		// rounding bias (31..24 will be fraction; 32 up are integer)

	for (int irow = 0; irow < out_height; irow++, ycoord += ystep) {
		int y0 = (ycoord >> 32);
		int y1 = min(y0 + 1, in_height - 1);
		y0 *= in_row_stride;
		y1 *= in_row_stride;
		int yfrac0 = (uint32_t)ycoord >> 24;
		uint64_t xcoord = (1<<23);
		for (int icol = 0; icol < out_width; icol++, xcoord += xstep) {
			int x0 = xcoord >> 32;
			int x1 = min(x0 + 1, in_width - 1);
			int xfrac0 = (uint32_t)xcoord >> 24;

			for (int d = 0; d < 32; d++) {
				unsigned char v00 = in_ptr[y0 + x0 * 32 + d];
				unsigned char v01 = in_ptr[y0 + x1 * 32 + d];
				unsigned char v10 = in_ptr[y1 + x0 * 32 + d];
				unsigned char v11 = in_ptr[y1 + x1 * 32 + d];
				unsigned short v0 = (v00 + v00 * (255-xfrac0) + v01 * xfrac0);
				unsigned short v1 = (v10 + v10 * (255 - xfrac0) + v11 * xfrac0);
				out_ptr[irow*out_row_stride + icol * 32 + d] = min(255, (v0 + v0 * (255 - yfrac0) + v1 * yfrac0 + (1<<15)) >> 16);
			}
		}
	}
}


static void
do_bilin_interp_x2_single_slice( uint8_t const * in_ptr, int in_row_stride, int in_width, int in_height, uint8_t * out_ptr, int out_row_stride, int out_width, int out_height )
{
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
	// make klockwork happy
	if (prev_outp) memcpy(outp_last, prev_outp, in_width*2 * 32 );	// last row same as previous)
}


///////////////////////////////////////////////////
/////// Handle general case with HVX
///////////////////////////////////////////////////
// The key to doing this efficiently is to do the 'h' scaling on a signal in which 1 vector = 1 'pixel'
// (i.e. one unit of 'width' per vector) to avoid having to do adaptive extractions.
//
// The approach here is to do v-scaling first, and generate a 16-bit intermediate; but also do two work
// units at once, so that we can interleave the two results into one vector, and so each vector holds
// 32 depth units from a single  [h,w] from the two work units (a 'work unit' is a single d32 slice
// within a single batch).
//
// The vertical interp is done by combining two rows (from each of the two work units) using two
// interpolation weights which add up to 128 (so that the 16-bit results are in range 0 .. 0x7F80).
// This operation includes odd width-padding lanes in the input, so that the intermediate result
// is always a multiple of 4 vectors.
// The interpolation pass works on pairs of vectors from this buffer, combining them using h interpolation.
// We use a pre-built table for the h scaling; for each output it gives the offset into the row, and the
// fractional value (as a 16-bit scale factor).
//  This table accounts for left & right edge situation, and is itself padded out to a multiple of 4 output
// width, so that the h-scaling loop doesn't have to do anything special at the right edge.
// Each time the h-scaling loop processes four entries from this table, it generates 4x32x2 16-bit results,
// which are packed into two vectors of 4x32 u8 results; which are then written to the two output areas.
//


// make "resize_plan" to scale according to sizes in rstp;
// if we already have one (in opaque ) use that; otherwise
// make a new one and store it at opaque.
// In either case it's put in the runstate.
// 0 return if OK.
//
static int
obtain_scaling_plan(
		struct nn_node *self,
		struct bilin_runstate  *rstp)
{
	int ht_in = rstp->in_ht;
	int wid_in = rstp->in_width;
	int ht_out = rstp->out_ht;
	int wid_out = rstp->out_width;
	if( ht_in < 1 || wid_in < 1 || ht_out < 1 || wid_out <1 ) return -1;

	int htable_needed = (wid_out+3)&~3;	// length needed for h scaling table.
	struct resize_plan * planp =  (struct resize_plan *)self->opaque;
	if( planp != NULL){
		if(  ht_in == planp->in_height && ht_out == planp->out_height
			&&	wid_in == planp->in_width && wid_out == planp->out_width ){
			rstp->planp = planp;	// reuse exising
			return 0;
		}
		// is existing alloc large enough?
		// free it if not
		if( htable_needed  > ((planp->out_width+3)&~3)){
			nn_free((void*)planp);
			planp = NULL;
			self->opaque = NULL;
		}
	}
	if( planp == NULL){
		planp = (struct resize_plan *) nn_malloc(
				sizeof(struct resize_plan) + sizeof(struct hcontroltab)*( htable_needed-1));
		if( planp == NULL) return -1;
		self->opaque = (void*)planp;
	}
	rstp->planp = planp;
	// fill in the table
	planp->in_height = ht_in;
	planp->in_width =  wid_in;
	planp->out_height= ht_out;
	planp->out_width = wid_out;
	if (rstp->align_corners && ht_out > 1) {
		ht_in--;
		ht_out--;
	}
	planp->vscale = (((uint64_t)ht_in <<32) + (ht_out>>1)) / (uint64_t)ht_out;

	// scale for h table.
	int win_max = wid_in - 1;
	int wid_out_pi = wid_out;
	if (rstp->align_corners && wid_out > 1) {
		wid_in--;
		wid_out_pi--;
	}
	int64_t add_per = (((uint64_t)wid_in <<32) + (wid_out_pi >>1)) / (uint64_t)wid_out_pi;
	struct hcontroltab * outrecs = planp->hcontrol;

	int64_t acc =  (1<<16);	// we take a 15-bit fraction, this is the rounding bias
	for( int i = 0; i < wid_out; i++){
		int intpart = (acc>>32);		// integer part
		unsigned fpart = (uint32_t)acc >> 17;	// fractional, 0..32767
		if( intpart >= win_max){		// don't go beyond wid_in-1;
			intpart = win_max;
			fpart = 0;
		}
		//printf("-- %3d %5d %d\n", i, intpart, fpart);
		outrecs->offs  = 128* intpart;	// convert to vector offset
		outrecs->frac  = Q6_R_combine_RlRl( fpart, fpart);
		outrecs++;
		acc += add_per;
	}

	// pad out to multiple of 4.
	while( (wid_out&3)!= 0){
		outrecs->offs = win_max * 128;
		outrecs->frac = 0;
		outrecs++;
		wid_out++;
	}
	return 0;
}

//
// vertical scaling loop
//
static inline void hvx_bilin_v_scaling(
		HVX_Vector *outrow,			// output (writes nvec*4 vectors
		HVX_Vector const * inrowA,	// input (reads nvec vectors) - first work unit
		HVX_Vector const * inrowB,	// input (reads nvec vectors) - second work unit
		uint32_t h_stride,			// bytes to next input row
		int beta,					// value in range 0..127 to interpolate with
		int nvec )					//
{
	beta &= 127;
#if __HEXAGON_ARCH__ < 65
	if( beta == 0){			// 128 won't work as a weight
		h_stride = 0;		// so use the same row twice
		beta = 64;
	}
#endif
	int rfac = 0x00800080 + 0xFF00FF * beta;  // beta:(128-beta):beta:(128-beta)
	for( int i = 0; i < nvec; i++){
		HVX_Vector vA0 = inrowA[i];
		HVX_Vector vB0 = inrowB[i];
		HVX_Vector vA1 = ((HVX_Vector const*)( (char const *)inrowA + h_stride))[i];
		HVX_Vector vB1 = ((HVX_Vector const*)( (char const *)inrowB + h_stride))[i];
#if __HEXAGON_ARCH__ >= 65
		HVX_VectorPair SumsA = Q6_Wh_vmpa_WubRub( Q6_W_vcombine_VV( vA1, vA0), rfac );
		HVX_VectorPair SumsB = Q6_Wh_vmpa_WubRub( Q6_W_vcombine_VV( vB1, vB0), rfac );
#else
		HVX_VectorPair SumsA = Q6_Wh_vmpa_WubRb( Q6_W_vcombine_VV( vA1, vA0), rfac );
		HVX_VectorPair SumsB = Q6_Wh_vmpa_WubRb( Q6_W_vcombine_VV( vB1, vB0), rfac );
#endif
		HVX_VectorPair shuf10 = Q6_W_vshuff_VVR( Q6_V_lo_W(SumsB), Q6_V_lo_W(SumsA), -2);
		HVX_VectorPair shuf11 = Q6_W_vshuff_VVR( Q6_V_hi_W(SumsB), Q6_V_hi_W(SumsA), -2);
		HVX_VectorPair shuf20 = Q6_W_vshuff_VVR( Q6_V_lo_W(shuf11), Q6_V_lo_W(shuf10), -4);
		HVX_VectorPair shuf21 = Q6_W_vshuff_VVR( Q6_V_hi_W(shuf11), Q6_V_hi_W(shuf10), -4);
		outrow[0] = Q6_V_lo_W(shuf20);
		outrow[1] = Q6_V_hi_W(shuf20);
		outrow[2] = Q6_V_lo_W(shuf21);
		outrow[3] = Q6_V_hi_W(shuf21);
		outrow += 4;
	}
}

static inline HVX_Vector
h_scale_single(HVX_Vector const *inrow, struct hcontroltab hct )
{
	uint32_t offs = hct.offs;
	int32_t frac = hct.frac;
	HVX_Vector const *inpos = (HVX_Vector const *)((char const*)inrow + offs);
	HVX_Vector vxL = inpos[0];		// elements are 0..0x7f80
	HVX_Vector vxD = Q6_Vh_vsub_VhVh(inpos[1],vxL);		// elements are +/- 0x7f80
	return Q6_Vh_vadd_VhVh( vxL, Q6_Vh_vmpy_VhRh_s1_rnd_sat(vxD,frac));	// 0 .. 7f80 again
}

//
// This operation processes one h-scaling row according to the information in table 'hct'
// 'inrow' should point to the intermediate buffer generated by hvx_bilin_v_scaling, and
// should skip any vectors which arose from 'left padding' on the input.
//
static inline void hvx_bilin_h_scaling(
		HVX_Vector  * outrowA,				// output - first work unit (writes nvec vectors)
		HVX_Vector  * outrowB,				// output - second work unit (writes nvec vectors)
		HVX_Vector const *inrow,			// input  - reads nvec vectors
		struct hcontroltab const * hct,		// h scaling table (reads nvec *4 entries)
		int nvec							// width of operation
		)
{
	// four scaling ops per loop
	for( int i = 0; i < nvec; i++){
		HVX_Vector v0i = h_scale_single( inrow, hct[0]);
		HVX_Vector v2i = h_scale_single( inrow, hct[2]);
		HVX_Vector v1i = h_scale_single( inrow, hct[1]);
		HVX_Vector v3i = h_scale_single( inrow, hct[3]);
		hct += 4;
		// 4 vectors; pack down to 2
		HVX_Vector out02 = Q6_Vub_vasr_VhVhR_rnd_sat( v2i, v0i, 7);
		HVX_Vector out13 = Q6_Vub_vasr_VhVhR_rnd_sat( v3i, v1i, 7);
		// deal twice to get the right order
		HVX_VectorPair dealt = Q6_W_vdeal_VVR( out13, out02, -1 );
		dealt = Q6_W_vdeal_VVR( Q6_V_hi_W(dealt), Q6_V_lo_W(dealt),-1);
		*outrowA ++= Q6_V_lo_W(dealt);
		*outrowB ++= Q6_V_hi_W(dealt);
	}
}

// assumes the following fields are added to the runstate:
// struct resize_plan *planp ( already built)
//  HVX_Vector * intermed;	// work area large enough for intermed
//


static void
do_bilin_interp_general_all_HVX( struct nn_graph *nn, void *rstpv )
{
	struct bilin_runstate *rstp = (struct bilin_runstate *)rstpv;
	struct resize_plan *planp = rstp->planp;

	int bufind;
	HVX_Vector *intermed_buf = bufpool_take( &rstp->intermed_bufs, &bufind);

	int hlim = rstp->in_ht-1;
	int ijob;
	int nd32 = rstp->tin.nd32;
	int id32, ibat;
	// take jobs 2 at once

	int32_t in_h_stride = rstp->tin.height_stride;
	int32_t out_h_stride = rstp->tout.height_stride;

	uint8_t const * in_base_ptr0 = rstp->tin.data;
	uint8_t  * out_base_ptr0 = rstp->tout.data;

	// account for input left pad; in_base_ptr0 may not be aligned
	// out_base_ptr0 is assumed to be aligned.

	int in_left_pad= ((size_t)in_base_ptr0>>5)&3;	//0,1,2,3
	in_base_ptr0 -= 32*in_left_pad;
	int nvecs_vscale = (in_left_pad + rstp->in_width + 3)>>2;
	int nvecs_hscale = (rstp->out_width+ 3)>>2;

	// pair-stride to be used on the last d32 slice of a batch, if nd32 is odd
	// (and it never happens when it's even).
	int in_wrap_stride = rstp->tin.batch_stride - (nd32-1)*rstp->tin.d32_stride;
	int out_wrap_stride = rstp->tout.batch_stride - (nd32-1)*rstp->tout.d32_stride;

	// take jobs 2 at once;
	// assume ijob = id32 + nd32 * ibat
	//

	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode, nd32);

	while( ijob = __sync_fetch_and_add( &rstp->next_job, 2), ijob < rstp->jobs ){
		// convert to slice #, batch;
		id32 = batchslice_decode_update( &bsdecode, ijob);
		ibat = bsdecode.ibatch;
		// determine the strides across the two work units
		// first, assume the next one is the next depth slice
		// (always true when nd32 is even).
		int32_t in_pair_stride = rstp->tin.d32_stride;
		int32_t out_pair_stride = rstp->tout.d32_stride;
		if( id32+1 >= nd32){	/// not the next depth slice!
			if( ibat +1 < rstp->batches ){	// wrap to next batch
				in_pair_stride = in_wrap_stride;
				out_pair_stride = out_wrap_stride;
			}else{
				in_pair_stride = 0;		// one odd one at the end; do it on top of itself.
				out_pair_stride = 0;
			}
		}
		//printf("id32 = %d; ib  = %d; ips = %d\n", id32, ibat, (int)in_pair_stride);
		// get the base pointers for the first unit of the two
		uint8_t const *in_base_ptr = in_base_ptr0 + ibat * rstp->tin.batch_stride + id32 * rstp->tin.d32_stride;
		uint8_t  *out_base_ptr = out_base_ptr0 + ibat * rstp->tout.batch_stride + id32 * rstp->tout.d32_stride;

		int64_t vacc = (1<<24);	// rounding bias for 7-bit fraction
		for( int orow = 0; orow < rstp->out_ht; orow++ ){
			int iypos = vacc>>32;
			int yfrac = (uint32_t)vacc >> 25;	// 7-bit frac
			int yfrac2 = -1;				// yfrac for second op.
			vacc += planp->vscale;
			if( iypos >= hlim) {		// don't exceed hlim
				iypos = hlim;
				yfrac = 0;
			}else if( (int)(vacc>>32) == iypos){		// next is at same int. part
				// can do two at once. @@@ (currently not actually doing this)
				if(0)if( orow+1 < rstp->out_ht){
					yfrac2 = (uint32_t)vacc >> 25;
					vacc += planp->vscale;
				}
			}
			// if yfrac2 >= 0 here, we can vscale two rows from the same pair of input
			// rows.. this will reduce loop overhead and re-reading of data. (@@@ TODO @@@)

			// note that in_row_ptr advances according to the scale ratio.
			uint8_t const * in_row_ptr = in_base_ptr + in_h_stride * iypos;
			uint8_t * out_row_ptr = out_base_ptr + out_h_stride * orow;

			hvx_bilin_v_scaling(
					intermed_buf,									// output
					(HVX_Vector const*)in_row_ptr,                // input - first work unit
					(HVX_Vector const *)(in_row_ptr + in_pair_stride),	// input  - second work unit
					in_h_stride,			// bytes to next input row
					yfrac,					// value in range 0..127 to interpolate with
					nvecs_vscale );
			hvx_bilin_h_scaling(
					(HVX_Vector *)out_row_ptr,					// output - first work unit
					(HVX_Vector *)(out_row_ptr + out_pair_stride),	// output - second work unit
					intermed_buf + in_left_pad,			// input
					planp->hcontrol,				// h scaling table
					nvecs_hscale);					// width of operation
		}
	}
	bufpool_release( &rstp->intermed_bufs, bufind);
	nn_sem_post( &rstp->done_sem);
}



struct nn_node_ops nn_ops_for_QuantizedResizeBilinear_8_d32 = {
	.execute = resize_bilinear_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
