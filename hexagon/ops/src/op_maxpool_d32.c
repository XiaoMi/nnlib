
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
 * This contains implementations for quantized max pooling node
 */


#include <nn_graph.h>
#include <string.h>
#include "quantize.h"
#include "nn_hmaxpool_d32.h"
#include "hvx_inlines.h"


// NOTE: MAXPOOL_MAX_THREADS does NOTHING for graphs where batch=1

#ifdef HEXAGON_V66
#define MAXPOOL_MAX_THREADS 4
#else
#define MAXPOOL_MAX_THREADS 2
#endif

/*
 * A note about strategy
 * In Depth32, we should have some sufficient padding around H and W dimensions
 *
 * So we should be able to zap borders and compute pretty liberally
 * In particular, if we don't care about output values, we don't even care about X and Y position
 * relative to the border, since we will just compute padding in the same format.
 */


//
// Certain common cases of size and stride are handled by asm routines tailored to those cases.
// General cases handled as below:
//  => we do H and then V separately.
//      The intermediate is in a 'rolling' buffer.
//  => there are a series of functions which do the 'H' maxpool; each does 2 rows of input and writes
//      two intermediate buffers. Function is chosen based on H window size and stride. See nn_hmaxpool_d32.h
//       Left/right padding needs to be 'zapped' to zero in the input buffer where applicable.
//  => The vertical maxpool from the intermediate to the output buffer is relatively easy. We zero-pad the intermediate
//      buffer on top and bottom as needed.
// The size of the rolling buffer is defined by:
//     - must be even # of rows (because rows are loaded in even/odd pairs, and we don't want the pair
//       to be split across the wrap).
//     - must be at least win_ht+1 rows high
//     - initially we may have zero-padding rows at the top; this must be an even number. So, if
//       win_ht = 4, and top_pad = 1, we need to have 2 rows of top padding, and then we need to load
//       3 rows to get the initial output row; so thus the window height must be 6 in this case
//       (we will load 4 rows initially,  into rows 2,3,4,5 of the buffer).
// It is not required that the buffer height >= stride_height. E.g. you have win_ht=3, stride_height=9, we
// will get along fine with a 4-row buffer, the 'out_credit' will make it work (in this case, 'reader_bump' will be 1).
// In such a case we're doing horiz. processing on a lot of rows that are not used, but...
//
// There is an 'out_credit' variable which is used as follows:
//
//  initially:
//      - set out_credit to 'initial_out_credit' (precalculated; <0)
//    then repeat until all output rows generated:
//      (1) read 2 rows into buffer; add 2 to out_credit; repeat until
//         out_credit >= 1
//      (2) generate an output row; subtract 'stride_h' from out_credit.
//        repeat until out_credit <= 0 (or until we've generated all rows)
// - If we run out of input rows in step (1), "load" zero rows in instead. This can
//  be done 1 at a time instead of 2.
//
//
struct rollbuf_desc {
	int buf_rows;		// height of the rolling buffer (even)
	int pad_rows;		// padding rows for 'top of frame' (even)
	int row_bytes;		// bytes per row
	int total_bytes;	// total size of buffer in bytes
	int reader_bump;	//  this is row_bytes * (stride_ht % buf_rows)
	int16_t initial_out_credit;	// init 'out_credit' to this

	struct hmaxpool_func_info hmax_funcinfo;	// info for the hmaxpool operator
	hmaxpool_funcp hmax_funcp;					// the function itself

};

struct rollbuf_state {
	uint8_t * bufbase;		// buffer base pointer
	uint8_t * bufend;		// bufbase + total_bytes (for wrap)
	int row_bytes;			// bytes/row
	uint8_t * load_rowp;	// points to next row to load
	uint8_t * read_rowp; // points to initial row for next output window.
	int rows_loaded;		// current # rows loaded from input (includes bottom padding where applicable)
	int rows_output;		// rows generated
	int out_credit;			// this is advanced by 'n' when we load n rows; when > 0, we can
};
//
// prototype for calling asm function which does one of several fixed window/stride combos.
//
typedef int (*maxpool_slice_fp)(
		uint8_t * out_data,
		const uint8_t *in_data,
		int32_t in_next_row,
		int32_t out_next_row,
		int32_t out_vecs_wide,
		int32_t out_lines,
		int32_t out_lalign );
//
// Function for reduce->1x1 of one slice; can use same function prototype as maxpool_slice_fp,
// except that
//      - out_next_row is unused;
//      - out_vecs_wide -> becomes the valid input width ( =window_width)
//      - out_lines -> becomes 'input_height'			( = window_height)
//  	- out_lalign is one of 0,32,64,96 and represents left-padding to ignore
static int
do_maxpool_out1x1_slice(
		uint8_t * out,		// a single vector stored here
		uint8_t const * in,
		int32_t in_next_row,
		int32_t out_next_row,		// unused
		int32_t input_width,		// valid input width
		int32_t input_height,		// input height
		int32_t input_skipbytes );	// # bytes to skip at left (left padding, {0..3}*32)


struct maxpool_runstate {
	struct tensor_addressing tin;	// input addressing
	struct tensor_addressing tout;	// output addressing;

	const uint8_t *indata_base;
	uint8_t *outdata_base;
	int32_t out_vectors_wide;
	int32_t out_lines;
	int32_t out_lalign;
	int32_t top_padding_rows;
	int32_t prefetch_rows;
	int32_t in_height;
	uint16_t window_height, stride_height;

	//...for zapping
	// for zapping, the jobs are 0..batches-1
	// for the actual run, it's 0 .. batches*nd32-1
	uint8_t *zap_in_base;
	int32_t zap_height;
	int32_t zap_left_w;
	int32_t zap_w_skip;
	int32_t zap_top;
	int32_t zap_bot;

	volatile int jobno;
	int numjobs;
	nn_sem_t donesem;
	maxpool_slice_fp slice_funcp;

	//
	// some cases use a 'rolling buffer'; the layout is described by the field in rollbuf
	// below. When not in used, rollbuf.total_byes = 0.
	//
	struct rollbuf_desc rollbuf;
	uint8_t * rollbuf_pool;		// pointer to rollbuf.total_bytes * no_of_threads of scratch
	volatile int pool_alloc;	// used to allocate rollbuf_pool to threads.
};
// setup 'rollbuf_state' at start of each slice
// assumes bufbase,bufend already set up.
static inline void init_rollbuf_state( struct rollbuf_state *rbs, struct maxpool_runstate const *rstp)
{
	int row_bytes = rstp->rollbuf.row_bytes;
	uint8_t * buf = rbs->bufbase;
	int top_pad_bytes = row_bytes * rstp->top_padding_rows;	// actually needs clearing
	int top_pad_start = row_bytes * rstp->rollbuf.pad_rows - top_pad_bytes;
	rbs->row_bytes = row_bytes;
	rbs->load_rowp = (uint8_t *)buf + top_pad_start + top_pad_bytes;
	rbs->read_rowp = (uint8_t *)buf + top_pad_start;
	rbs->rows_loaded = 0;
	rbs->rows_output = 0;
	rbs->out_credit = rstp->rollbuf.initial_out_credit;
	if ( top_pad_bytes > 0 ){
		vmemset_asm( rbs->read_rowp, 0, top_pad_bytes );
	}
}


// worker thread for when we have a 'slice' function in assembler.
// or when do_maxpool_out1x1_slice is used.

static void maxpool_execute_worker(struct nn_graph *nn, void *rstpv)
{
	struct maxpool_runstate *rstp = rstpv;
	uint8_t *outdata0 = rstp->outdata_base;
	const uint8_t *indata0 = rstp->indata_base;
	int in_next_row = rstp->tin.height_stride;
	int in_next_d32 = rstp->tin.d32_stride;
	int prefetch_rows = rstp->prefetch_rows;
	int out_next_row = rstp->tout.height_stride;
	int out_vectors_wide = rstp->out_vectors_wide;
	int out_lines = rstp->out_lines;
	int out_lalign = rstp->out_lalign;
	int njobs = rstp->numjobs;

	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode, rstp->tin.nd32);

	int ijob;
	while(  (ijob = __sync_fetch_and_add( &rstp->jobno,1)), ijob < njobs){
		int id32 = batchslice_decode_update( &bsdecode, ijob);
		int ibat = bsdecode.ibatch;
		uint8_t const * indata = indata0 + ibat * rstp->tin.batch_stride + id32 * rstp->tin.d32_stride;
		uint8_t * outdata = outdata0 + ibat * rstp->tout.batch_stride + id32 * rstp->tout.d32_stride;
		l2fetch(indata,in_next_row,in_next_d32,prefetch_rows);
		rstp->slice_funcp(outdata,indata,in_next_row,out_next_row,out_vectors_wide,out_lines,out_lalign);
	}
	nn_sem_post(&rstp->donesem);
}
//
// worker function when 'rolling buffer' is in use.
static void maxpool_execute_rollbuf_worker(struct nn_graph *nn, void *rstpv)
{
	struct maxpool_runstate *rstp = rstpv;
	uint8_t *outdata0 = rstp->outdata_base;
	const uint8_t *indata0 = rstp->indata_base;
	int in_next_row = rstp->tin.height_stride;
	int in_next_d32 = rstp->tin.d32_stride;
	int prefetch_rows = rstp->prefetch_rows;
	int out_next_row = rstp->tout.height_stride;
	int out_vectors_wide = rstp->out_vectors_wide;
	int out_lines = rstp->out_lines;
	int in_height = rstp->in_height;
	int window_height = rstp->window_height;
	int stride_height = rstp->stride_height;

	if( out_vectors_wide <=0) return;  // make the compiler truly believe out_vectors_wide > 0

	int njobs = rstp->numjobs;
	int threadno = __sync_fetch_and_add( & rstp->pool_alloc,1);
	//stride
	// set up the buffer pointers in rollbuf_state
	struct rollbuf_state rbstate;
	int rollbufsize = rstp->rollbuf.total_bytes;
	rbstate.bufbase = rstp->rollbuf_pool + rollbufsize * threadno;
	rbstate.bufend = rbstate.bufbase + rollbufsize;

	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode, rstp->tin.nd32);

	int ijob;
	while(  (ijob = __sync_fetch_and_add( &rstp->jobno,1)), ijob < njobs){
		int id32 = batchslice_decode_update( &bsdecode, ijob);
		int ibat = bsdecode.ibatch;
		uint8_t const * indata = indata0 + ibat * rstp->tin.batch_stride + id32 * rstp->tin.d32_stride;
		uint8_t * outdata = outdata0 + ibat * rstp->tout.batch_stride + id32 * rstp->tout.d32_stride;
		l2fetch(indata,in_next_row,in_next_d32,prefetch_rows);

		// setup up rolling buffer state (includes clearing any top-padding rows)
		init_rollbuf_state( &rbstate, rstp);
		int window_height1 = window_height;

		while( rbstate.rows_output < out_lines){
			while( rbstate.out_credit < 1){		// need to process rows into rolling buffer
				uint8_t * load_rowp_next;
				if( rbstate.rows_loaded < in_height){	// load with hmaxpool function
//printf("ocred = %d; rows_loaded = %d (of %d) -->%p\n", rbstate.out_credit, rbstate.rows_loaded , in_height,rbstate.load_rowp);
					// load 2 rows
					uint8_t const * in0 = indata;
					uint8_t const * in1 = indata + in_next_row;
					uint8_t *out0 = rbstate.load_rowp;
					uint8_t *out1 = out0 + rbstate.row_bytes;
					load_rowp_next = rbstate.load_rowp + 2*rbstate.row_bytes;
					rbstate.rows_loaded += 2;
					rbstate.out_credit += 2;
					if( unlikely( rbstate.rows_loaded > in_height)){	// last row of odd height
						out1 = out0;
						in1 = in0;
						rbstate.out_credit --;
						load_rowp_next = rbstate.load_rowp + rbstate.row_bytes;
					}
					rstp->rollbuf.hmax_funcp(&rstp->rollbuf.hmax_funcinfo, out0, out1, in0, in1);
					indata += 2*in_next_row;
					rbstate.load_rowp = (load_rowp_next < rbstate.bufend) ? load_rowp_next: rbstate.bufbase;
				}else{
					// Have loaded all rows and need more padding rows to support next output row.
					// in principle: "load" one row of zeros, and add 1 to out_credit.
					// in practice we can just subtract 1 from window_height1, and +1 to out_credit
					// better still, subtract n from window_height1 and +n to out_credit, where n = 1-out_credit
					int n = 1 - rbstate.out_credit;
					rbstate.out_credit = 1 ;
					window_height1 -= n;
					if( window_height1 < 1 ) return;	// should never happen
					break;
				}
			}
			// we have enough rows in the buffer to process at least one output row. So do that.
			// Two different loops, according to whether the needed rows are 'wrapped over' in the buffer.
			uint8_t * rdr0 = rbstate.read_rowp;
			uint8_t const * rdr_end = rdr0 + rbstate.row_bytes * window_height1;
			// do the whole thing without wrapping?
//printf("ocred = %d; row %d out from %p wrapped = %d\n",rbstate.out_credit,rbstate.rows_output,  rdr0, rdr_end > rbstate.bufend);
//	rdr0[0],rdr0[rbstate.row_bytes],rdr0[2*rbstate.row_bytes]);

			if( rdr_end <= rbstate.bufend){	// not wrapped...
				for( int i  =0; i < out_vectors_wide; i++){
					HVX_Vector const * rdp = (HVX_Vector const *)rdr0 + i;
					HVX_Vector v = *rdp;
					for( int k = 1; k < window_height1; k++){
						rdp = (HVX_Vector const*)( (char const*) rdp + rbstate.row_bytes);
						v = Q6_Vub_vmax_VubVub( v, *rdp);
					}
					((HVX_Vector*)outdata)[i] = v;
				}
			}else{
				for( int i  =0; i < out_vectors_wide; i++){
					uint8_t const * rdp = rdr0 + i*128;
					HVX_Vector v = *(HVX_Vector const*)rdp;
					for( int k = 1; k < window_height1; k++){
						rdp += rbstate.row_bytes;
						if ( rdp >= rbstate.bufend ) rdp -= rollbufsize;
						v = Q6_Vub_vmax_VubVub( v, *(HVX_Vector const*)rdp);
					}
					((HVX_Vector*)outdata)[i] = v;
				}
			}
			rdr0 += rstp->rollbuf.reader_bump;	// advance according to stride_height.
			if( rdr0 >= rbstate.bufend ) rdr0 -= rollbufsize;	// .. and wrap
			rbstate.read_rowp = rdr0;
			rbstate.rows_output++;
			outdata += out_next_row;
			rbstate.out_credit -= stride_height;
		}

	}
	nn_sem_post(&rstp->donesem);
}

static void maxpool_zap_worker(struct nn_graph *nn, void *rstpv)
{
	struct maxpool_runstate *rstp = rstpv;
	uint8_t *indata0 = rstp->zap_in_base;
	int32_t in_next_row = rstp->tin.height_stride;
	int32_t in_next_d32 = rstp->tin.d32_stride;
	int32_t in_height = rstp->zap_height;
	int32_t zap_top = rstp->zap_top;
	int32_t zap_bot = rstp->zap_bot;
	int32_t zap_left_w = rstp->zap_left_w;
	int32_t zap_w_skip = rstp->zap_w_skip;
	int32_t zap_r_woff = zap_w_skip+zap_left_w;
	int32_t zap_r_amt = (-zap_r_woff) & 3;
	int32_t zap_d32_iters = rstp->tin.nd32;
	int32_t in_next_batch = rstp->tin.batch_stride;
	int ibat;
	while(  (ibat = __sync_fetch_and_add( &rstp->jobno,1)), ibat < rstp->numjobs ){
		uint8_t *indata = indata0 + ibat * in_next_batch;
		if( zap_top > 0)
			vmemset_asm(indata,0,in_next_row*zap_top);
		if( zap_bot > 0)
			vmemset_asm(indata+in_next_row*(in_height+zap_top),0,in_next_row*zap_bot);
		padzap_part(indata+in_next_row*zap_top,0,in_next_d32,zap_d32_iters,in_next_row,in_height+2,zap_left_w);
		if (zap_r_amt) {
			padzap_part(indata+in_next_row*zap_top+32*(zap_w_skip+zap_left_w),0,in_next_d32,zap_d32_iters,in_next_row,in_height+1,zap_r_amt);
		}
	}
	nn_sem_post(&rstp->donesem);
}

static int maxpool_d32_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *window_tensor = self->inputs[3];
	const struct tensor *stride_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t window_width = window_tensor->shape.width;
	int32_t window_height = window_tensor->shape.height;

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	int32_t out_batches = in_batches;
	int32_t required_w_before, required_h_before;
	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,window_width,stride_width,self->padding, &required_w_before);
	int32_t out_height = nn_pad_compute_outsize_and_padbefore(in_height,window_height,stride_height,self->padding, &required_h_before);
	int32_t out_depth = in_depth;

	int32_t out_left_pad = 4;
	int32_t out_right_pad = (-out_width) & 3;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = 4;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;

	const uint8_t *inptr;
	uint8_t *outptr;

	int32_t b,h,w,d,out_h,out_w,win_h,win_w;

	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail");
	}
	tensor_set_float(out_min,0,tensor_get_float(in_min_tensor,0));
	tensor_set_float(out_max,0,tensor_get_float(in_max_tensor,0));

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	for (b = 0; b < out_batches; b++) {
	for (out_h = 0; out_h < out_height; out_h++) {
	for (out_w = 0; out_w < out_width; out_w++) {
	for (d = 0; d < out_depth; d++) {
		int32_t max = 0;
		outptr = tensor_location_d32(out_tensor,b,out_h,out_w,d);
		for (win_h = 0; win_h < window_height; win_h++) {
		for (win_w = 0; win_w < window_width; win_w++) {
			h = out_h * stride_height - required_h_before + win_h;
			w = out_w * stride_width - required_w_before + win_w;
			if (h < 0) continue;
			if (w < 0) continue;
			if (h >= in_height) continue;
			if (w >= in_width) continue;
			inptr = tensor_location_d32(in_tensor,b,h,w,d);
			if (max < *inptr) max = *inptr;
		}}
		*outptr = max;
	}}}}
	return 0;
}

static void maxpool_earlywork_v(struct nn_graph *nn, void *vinfo)
{
	struct nn_node *self = vinfo;
	struct nn_early_work *work = self->opaque;
	if (work == NULL) return;
	if (work->vtcm_addr != nn->vtcm_ptr) return;
	if (work->src_addr == NULL) return;
	if (work->dst_addr == NULL) return;
	if (work->bytes == 0) return;
	nn_graph_memcpy(nn,work->dst_addr,work->src_addr,work->bytes);
	work->valid = 1;
}
static int maxpool_earlywork(struct nn_graph *nn, void *vinfo)
{
	maxpool_earlywork_v(nn,vinfo);
	return 0;
}

static int maxpool_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	const struct tensor *window_tensor = self->inputs[3];
	const struct tensor *stride_tensor = self->inputs[4];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t in_left_pad = in_tensor->format.width_pad[0];
	int32_t in_right_pad = in_tensor->format.width_pad[1];
	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];
	int32_t in_top_pad = in_tensor->format.height_pad[0];
	int32_t in_bottom_pad = in_tensor->format.height_pad[1];

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	int32_t window_width = window_tensor->shape.width;
	int32_t window_height = window_tensor->shape.height;

	int32_t out_batches = in_batches;

	int32_t required_w_before, required_w_after;
	int32_t required_h_before, required_h_after;
	if(in_depth_before_pad != 0){
		// we can't handle this
		logmsg(nn,0,"maxpool, depth_before = %d; use reference", (int)in_depth_before_pad);
		return maxpool_d32_ref(self,nn);
	}
	struct maxpool_runstate rstt;
	rstt.tin =  tensor_addressing_d32( in_tensor );

	int32_t out_width = nn_pad_compute_outsize_and_pad(in_width,window_width,stride_width,self->padding,
			& required_w_before, &required_w_after);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height,window_height,stride_height,self->padding,
			&required_h_before, &required_h_after);
	int32_t out_depth = in_depth;

	int maxpool_to_1x1 = 0;
	if( out_height <= 0 || out_width <= 0) return errlog(nn,"invalid maxpool geometry");
	if( out_height == 1 ){
		// out_height = 1:	we can eliminate req. for top/bottom padding.
		int in_needed = min_i32(window_height - required_h_before, in_height);
		in_height = in_needed;
		window_height = in_needed;
		stride_height = 1;
		required_h_before  = required_h_after = 0;
	}
	if( out_width == 1 ){
		int in_needed = min_i32(window_width - required_w_before, in_width);
		in_width = in_needed;
		window_width = in_needed;
		stride_width = 1;
		required_w_before  = required_w_after = 0;
		if( out_height == 1)
			maxpool_to_1x1 = 1;
	}
	rstt.in_height = in_height;
	rstt.window_height = window_height;
	rstt.stride_height = stride_height;



	// we need an extra bottom padding row if the right padding is not sufficient for the filter.
	// This is true even when using rolling buffer.
	int extra_bottom_padding_row = (in_right_pad < required_w_after)?1:0;

	if( in_left_pad < required_w_before || in_left_pad + in_right_pad < required_w_after ){
		return errlog(nn,"insufficient width padding");
	}
	int32_t out_left_pad = 4;
	int32_t out_right_pad = (-out_width) & 3;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = 4;
	int32_t out_depth_before_pad = in_depth_before_pad;
	int32_t out_depth_after_pad = in_depth_after_pad;

	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	int32_t out_vectors_wide = out_width_total*32/128u;

	int (*f)(uint8_t *, const uint8_t *, int32_t, int32_t, int32_t, int32_t, int32_t);
	int in_w_offset = -in_left_pad;
	int out_w_offset = -out_left_pad;

	if ((window_tensor->shape.batches != 1)
		|| (window_tensor->shape.depth != 1)
		|| (stride_tensor->shape.depth != 1)
		|| (stride_tensor->shape.batches != 1)) {
		return maxpool_d32_ref(self,nn);
	}
	int input_full_height = in_height+required_h_before+required_h_after;

	rstt.rollbuf.total_bytes = 0;	// indicates no rolling buffer.
	int use_rollbuf = 0;
	f = NULL;
	// see if we have an asm function that will handle the current case
	// some of the asm routines only need 4 rows prefetched; others need the whole thing
	int prefetch_rows = 4;
	// only use asm if there is sufficient top/bottom padding for it
	if( maxpool_to_1x1){
		f = do_maxpool_out1x1_slice;
		prefetch_rows = input_full_height;
	}else if(  in_top_pad >= required_h_before && in_bottom_pad >= required_h_after + extra_bottom_padding_row){
		// only 2x2 stride 2x2,  3x3 stride 1x1, 3x3 stride 2x2
		if( window_width == window_height && stride_width == stride_height ){
			if( window_width == 2){
				if( stride_width == 2)
					f = maxpool_slice_hvx_2x2_stride2;
			}else if( window_width == 3){
				if( stride_width == 1){
					f = maxpool_slice_hvx_3x3_stride1;
				}else if( stride_width == 2){
					f = maxpool_slice_hvx_3x3_stride2;
					prefetch_rows = 5;
				}
			}
		}
	}
	// if we did not pick an asm func, use rolling buffer
	//
	if(f==NULL){		// design 'rolling buffer' for the general case
		// select a function to use for the horizontal maxpool op.
		// will also store some info in stt.rollbuf.hmax_funcinfo


		rstt.rollbuf.hmax_funcp = hmaxpool_select_function( &rstt.rollbuf.hmax_funcinfo,
				window_width, stride_width, out_width, (in_left_pad-required_w_before) & 3);
		int rollbuf_row_bytes = rstt.rollbuf.hmax_funcinfo.outvecs * 128;

		int top_pad_rows  = (required_h_before+1)&~1u;	// top padding rows (must accommodate filter and be even )
		// Buffer height is at least window_height+1; round up to even is done below
		int rollbuf_rows = window_height+1;

		// Try to use a larger rolling buffer, when reasonable, since
		// it runs faster when it's not wrapped around the end.
		//
		{
			// size will be expanded to approx. this, if the minimum size is less
			// than this.
			unsigned rollbuf_size_goal = 32*1024;
			if( rollbuf_rows*rollbuf_row_bytes < rollbuf_size_goal ){	// it could be larger...
				int rollbuf_max = input_full_height; // this is the biggest we could use
				if( rollbuf_max*rollbuf_row_bytes <= rollbuf_size_goal){
					rollbuf_rows = rollbuf_max;			// big enough for the whole input
				}else{		// find the # of rows (will not decrease)
					rollbuf_rows = rollbuf_size_goal/rollbuf_row_bytes;
				}
			}
		}
		rollbuf_rows = (rollbuf_rows+1)&~1u;	// round up to even
		rstt.rollbuf.buf_rows = rollbuf_rows;
		rstt.rollbuf.pad_rows = top_pad_rows;
		rstt.rollbuf.row_bytes = rollbuf_row_bytes;
		rstt.rollbuf.total_bytes = rollbuf_rows * rollbuf_row_bytes;
		// amount by which reader bumps its pointer for each output row (modulo buffer size).
		rstt.rollbuf.reader_bump = (stride_height % rollbuf_rows)* rollbuf_row_bytes;
		// find initial 'out_credit': we want out_credit=1 after loading window_height - required_h_before
		// rows (even if that number > input_height):
		rstt.rollbuf.initial_out_credit = 1 - (window_height-required_h_before);
		use_rollbuf = 1;
		prefetch_rows = in_height;

		/*printf("rollbuf = %d rows; width = %d bytes; %d are top-pad; initial out credit = %d\n",
			rollbuf_rows, rollbuf_row_bytes, top_pad_rows, rstt.rollbuf.initial_out_credit); */
	}

	/*
	 * For VALID padding we want to start after the IN_LEFT_OFFSET
	 * For stride == 1, out_lalign == 32 for SAME.
	 * For stride == 2, out_lalign == 64 for SAME.
	 */
	nn_sem_init(&rstt.donesem,0);
	int n_threads;

	prefetch_rows  = (prefetch_rows>input_full_height)? input_full_height: prefetch_rows;

	// zapping
	if( self->padding != NN_PAD_VALID  && !maxpool_to_1x1 ){
		int zap_top_rows =  use_rollbuf? 0: required_h_before;	// no top/bottom zap with rollbuf
		rstt.zap_in_base = rstt.tin.data
			- zap_top_rows*rstt.tin.height_stride
			- in_left_pad * 32;
		rstt.zap_height = in_height;
		rstt.zap_w_skip = in_width;
		rstt.zap_top = zap_top_rows;
		rstt.zap_bot = (use_rollbuf? 0: required_h_after) + extra_bottom_padding_row;
		rstt.zap_left_w = in_left_pad;
		if( rstt.zap_top > in_top_pad || rstt.zap_bot > in_bottom_pad){
			return errlog(nn,"insufficient top or bottom padding");
		}

		n_threads = min_i32( MAXPOOL_MAX_THREADS, in_batches);
		rstt.jobno = 0;
		rstt.numjobs = in_batches;

		for( int i =0; i < n_threads; i++ )
			nn_os_work_for_vector(nn,maxpool_zap_worker,&rstt);
		nn_sem_wait_n_times( &rstt.donesem, n_threads );
	}
	int32_t out_lalign = 0;
	if( maxpool_to_1x1){
		out_vectors_wide = in_width;		// special case
		in_w_offset = -(in_left_pad&3);
		out_w_offset = 0;
		out_lalign = (in_left_pad&3)*32;
	}else if( use_rollbuf){
		// adjust it down to the start of the first vector containing the first usable
		// width unit (including any padding the filter needs on the left).
		in_w_offset = ((in_left_pad-required_w_before)&~3u) - in_left_pad;
		out_w_offset = 0;
	}else{
		if (required_w_before == 0) {
			out_vectors_wide -= out_left_pad/4;
			in_w_offset = 0;
			out_w_offset = 0;
		} else if (stride_width == 2) {
			/* EJP: this is a bit clunky. Works for 3x3 but kind of in a funny way. */
			required_w_before += 2;
		}
		out_lalign = required_w_before*32;
	}

	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail");
	}
	tensor_set_float(out_min,0,tensor_get_float(in_min_tensor,0));
	tensor_set_float(out_max,0,tensor_get_float(in_max_tensor,0));

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	rstt.tout = tensor_addressing_d32( out_tensor );

	rstt.indata_base = rstt.tin.data + in_w_offset*32 - (use_rollbuf? 0 : required_h_before * rstt.tin.height_stride);
	rstt.top_padding_rows = required_h_before;
	rstt.outdata_base = rstt.tout.data + out_w_offset*32;
	rstt.out_vectors_wide = out_vectors_wide;
	rstt.out_lines = maxpool_to_1x1? in_height: out_height;
	rstt.out_lalign = out_lalign;
	rstt.prefetch_rows = prefetch_rows;
	rstt.slice_funcp = f;

	int njobs = rstt.tin.nd32 * in_batches;
	n_threads = min_i32( MAXPOOL_MAX_THREADS, njobs );
	rstt.numjobs = njobs;
	rstt.jobno = 0;
	rstt.pool_alloc = 0;

	if( use_rollbuf){
		if (nn_scratch_grow(nn, rstt.rollbuf.total_bytes * n_threads)){
			return errlog(nn, "scratch too small" );
		}
		nn_scratch_reset(nn);
		rstt.rollbuf_pool = nn_scratch_alloc(nn, rstt.rollbuf.total_bytes * n_threads);
		if( rstt.rollbuf_pool == NULL){
			return errlog(nn, "failed to get %d*%d scratch", n_threads, rstt.rollbuf.total_bytes);
		}
	}
/*
if( use_rollbuf){
	printf("in_tensor: (%d,%d,%d,%d),  data = %p, top_pad = %d, left_pad = %d, row_stride = %d, d32_stride = %d,"
		"tin.data =%p, base=%p rhb=%d, rwb=%d, in_w_offs = %d; h_in_off=%d\n",
		(int)in_tensor->shape.batches,
		(int)in_tensor->shape.height,
		(int)in_tensor->shape.width,
		(int)in_tensor->shape.depth,
		in_tensor->data, in_tensor->format.height_pad[0], in_tensor->format.width_pad[0],
		(int)rstt.tin.height_stride, (int)rstt.tin.d32_stride,
		rstt.tin.data,
		rstt.indata_base , (int)required_h_before, (int)required_w_before,in_w_offset,
		rstt.rollbuf.hmax_funcinfo.inoffs);
	}
*/
	for( int i=0; i < n_threads; i++ ) {
		nn_os_work_for_vector(nn,use_rollbuf? maxpool_execute_rollbuf_worker: maxpool_execute_worker, &rstt);
	}
	nn_os_vector_call(nn,maxpool_earlywork,self);
	nn_sem_wait_n_times( &rstt.donesem, n_threads );

	return 0;
}


//
// find the max (for each of 32 depths) of an entire d32 slice
// Result is a single vector (with the same 32 results stored 4 times).
//
static
int do_maxpool_out1x1_slice(
		uint8_t * out,		// a single vector stored here.
		uint8_t const * in,
		int32_t in_next_row,
		int32_t out_next_row,		// unused
		int32_t input_width,		// valid input width
		int32_t input_height,		// input height
		int32_t input_skipbytes )	// # bytes to skip at left (left padding, {0..3}*32)
{

	HVX_Vector max_all = Q6_V_vzero();

	HVX_VectorPred qleft_not = Q6_Q_vsetq_R(input_skipbytes);
	HVX_VectorPred qright = q6op_Q_vsetq2_R(input_skipbytes+32*input_width);

	// # of inner vec loops; 1 less than the # of vectors to read.
	// Possibly zero.
	//
	int nvloop = (input_skipbytes+32*input_width-32)>>7;
	if( nvloop >= 0 && input_height >= 1 ){
		for(int irow = 0; irow < input_height; irow++){
			HVX_Vector const *vpin = (HVX_Vector const*)(in + in_next_row *irow);
			HVX_Vector vin = *vpin++;
			// max & left masking
			HVX_Vector maxnew = Q6_V_vmux_QVV( qleft_not, max_all, Q6_Vub_vmax_VubVub(vin,max_all));
			// 0 or more repeats of...
			for( int i = 0; i < nvloop; i++){
				max_all = maxnew;
				vin = *vpin++;
				maxnew = Q6_Vub_vmax_VubVub(vin, max_all);
			}
			// apply right masking
			max_all = Q6_V_vmux_QVV( qright, maxnew, max_all);
		}// for irow
		// reduce 4 to 1 in the vector.
		for( int i  = 0; i < 2; i++){
			HVX_VectorPair dealt = Q6_W_vdeal_VVR( max_all, max_all, -32 );
			max_all =  Q6_Vub_vmax_VubVub(Q6_V_hi_W(dealt),Q6_V_lo_W(dealt));
		}
	}
	*(HVX_Vector *)out = max_all;
	return 0;
}


static int maxpool_dtor(struct nn_node *self, struct nn_graph *nn)
{
	self->opaque = NULL;
	return node_free_common(self,nn);
}

static int maxpool_earlywork_register(struct nn_node *self, struct nn_graph *nn, struct nn_early_work *work)
{
	self->opaque = work;
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedMaxPool_8_d32 = {
	.execute = maxpool_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = maxpool_dtor,
	.earlywork_register = maxpool_earlywork_register,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT | NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION | NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedMaxPool_8_d32_ref = {
	.execute = maxpool_d32_ref,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT | NN_NODE_FLAG_OUTPUT_ACCEPTS_PREPARATION | NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

