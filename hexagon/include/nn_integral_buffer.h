
/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
#ifndef NN_INTEGRAL_BUFFER_H
#define NN_INTEGRAL_BUFFER_H 1

#include <stddef.h>
#include <nn_graph.h>
#include <nn_graph_padding.h>
#include <quantize.h>
#include "hvx_inlines.h"

//
// datatypes and routines for managing an 'integral buffer'
// (common to avgpool_d32 and l2pool_d32)
//
//
//   The integral buffer has the following layout:
//     - each 'pixel' is a vector of 32xi32. We always process one depth-slice
//       of one batch at a time.
//     - there is a zero row and zero col, which are all zero. The rows to
//       the right and below integrate the quantity measured, which is either
//       pixel values (x256) for avgpool, or squares of pixels for l2pool.
//       Counting the zero row/col, the size of the 'core' buffer is in_height rows
//       and in_width cols, and this represents the input.
//	   - There are generally padding rows above and below, and padding columns
//       to left and right of the 'core', to support overlapping windows. These
//       values are calculated from the nearby core values.
//     - There is always space allocated for >=3 padding cols on the left and on the right;
//       when the input is processed into the integral buffer, this allows us to treat
//       partial-vector left & right width padding as part of the image. This 'overshoot'
//       data will either be unused, or will be overwritten by padding before being used.
//	   - the integral buffer is managed as a 'rolling' buffer; if the input height is large
//       relative to the window, the same set of buffer rows are used over and over again
//
//  There are some cases where the input size, window size, stride, may be adjusted
//  to simplify the problem without changing the result:
//
//   - in some cases, where right/bottom padding is 0, it may be that some right edge input cols
//     or bottom rows are not actually used (when strides >1 ); in these cases the
//     input size is reduced to lower the cost of building the integral buffer.
//   - in all case when the output dimension = 1 in either direction, we can reduce the
//     window size to match the input size (possibly reduced to eliminate unused input) and
//     use padding=0 on both sides, and get the same result.
//   - in some cases, when the window is larger than the input, we can reduce the padding
//     on both sides by 'k', and reduce the window by 2*k and get the same result. (This only applies
//     when all of the outputs encompass the same set of inputs, and therefore will be the same).
//   - result = 1 in both directions is a special case that should be handled with a different
//     execution engine (i.e. without building an integral buffer).
//

#ifdef NN_INTEGRAL_BUF_FOR_L2POOL
#define INTEGRAL_BUF_FUNC(x)  l2pool_##x
#else
#define INTEGRAL_BUF_FUNC(x)  avgpool_##x
#endif

//
// the 'struct integral_buffer_plan' is intended to be kept as persistent data, attached
// to a node; most of it depends on input dimensions and strides which are expected to be the same
// from run to run. There is a 'fast setup' path which updates certain fields (e.g data pointers)
// for cases when this assumption proves to be true.
//
// Below is the list of fields which are set up by setup_integral_buffer_plan() even when
//   the new size matches the old:
//            tin			 						- input addressing info
//            inshape.batches, outshape.batches		- both set to current input batches
//            inshape.depth, inshape.depth			- both set to current input depth
//
// fields which are *not* set in setup_integral_buffer_plan():
//
//			tout		 - should be set up after creating the output tensor layout based on plan.outshape.
//
// 		zero_code		 - this is the 'zero_code' for the signal, only used in l2pool
//
// Also:
//    recip_mant,recip_shift are set up when the plan is computed, based on the window size;
//                           they will identical to initial_recip_mant, but they could be
//                           changed later due to dynamic scaling.
//
//     edgepad_scale_alloc	Initially should be NULL
//     edgepad_scale_size	Initially should be 0
//
//
//    hvx_specialized_handler_code
//    hvx_specialized_handler
//        - not managed by this code, except they are cleared when a new plan is made.
//			When setup_integral_buffer_plan returns 1, look at the
//          dimensions, strides, etc, and decide if a special case handler can be used;
//          then set these as need. when it returns 0, the old value will still valid.
//
//
enum integral_buffer_pad_flags {
		intbuf_PAD_T=(1<<0),
		intbuf_PAD_B=(1<<1),
		intbuf_PAD_L=(1<<2),
		intbuf_PAD_R=(1<<3)
};

struct integral_buffer_plan {
	// The input size, window, stride actually requested, in order
	// to compare in the next run to see if a recalc is needed. The original values
	// are kept since any of these may get adjusted during setup.
	// To force a recalc, set prev_in_ht to zero.

	uint32_t prev_in_ht, prev_in_wid;	// used to check input shape unchanged.
	uint32_t prev_win_ht, prev_win_wid;	// used to check window shape unchanged
	uint32_t prev_str_ht, prev_str_wid;	// used to check stride unchanged.

	struct shape inshape;			// input shape, possibly trimmed down.
	struct shape outshape;			// output shape
	int16_t window_ht, window_wid;		// window size
	int16_t stride_ht, stride_wid;		//
	int16_t wpad_top, wpad_bottom;		// pad needed for window (both >= 0 )
	int16_t wpad_left, wpad_right;		// pad needed for window (both >= 0 )
	uint8_t wpad_flags;					// for faster runtime tests.
	//
	// these are the number of 'infeasible padding rows' in each direction
	//  infeas_pad_ht = max(0,  window_ht - in_ht-1 )
	int16_t infeas_pad_ht, infeas_pad_wid;

	// these point to the scale values for adding edge padding
	// if there are 'infeasible' padding rows (or cols) the first 'infeas' entries in
	// the edgepad_scales tables (if present) will be valid and compensated for bottom/right edge.
	int16_t * edgepad_scales_h;		// points to max( wpad_top, wpad_bot) values
	int16_t * edgepad_scales_w;		// points to max( wpad_left, wpad_right) values
	//
	// This is storage for the edgepad_scales. If the number required does not fit in
	// 8, then we allocate in
	//
	int16_t edgepad_scale_storage[8];
	void * edgepad_scale_alloc;		// if not null, points to allocated storage
	int edgepad_scale_size;			// if edgepad_scale_alloc != NULL, # of int16's allocated

	// note; top & left padding determine the placement of the window for output 0.
	// relative to the input.

	struct tensor_addressing tin;		// input tensor addressing
	struct tensor_addressing tout;		// output tensor addressing (*not* set in setup)
	int input_wpad_left;				// the left padding of the input tensor.

	// integral buffer layout.
	// note,  there is no pointer; each thread will have a different buffer.
	// all of the '*_bytes' and '*_offs' fields are multiples of 128.
	// If 'ibuf_is_rolled', then ibuf_row0 is odd and ibuf_rows is even (to facilitate loading
	// input rows in pairs).
	//
	int32_t ibuf_row_bytes;			// row length in bytes (includes left/right padding; also is row pitch
	int32_t ibuf_proper_row_bytes;	// row length actually used for integral image (w/o padding for overshoot)
	int32_t ibuf_total_bytes;		// = ibuf_rows * ibuf_row_bytes
	int16_t ibuf_rows;				// the number of rows (in truncated buffer, not whole thing)
	int16_t ibuf_row0;				// top padding rows (>= wpad_top)
	int32_t ibuf_col0_offs;			// offset in bytes to 'col 0' on each row
	int32_t ibuf_proper_offs;		// this is col0_off - 128*wpad_left; where the 'proper' row starts.
	int16_t ibuf_is_rolled;			// true if the buffer re-uses rows during an op.
	int16_t ibuf_initial_load;		// # of rows to load initially

	int16_t initial_recip_mant;		// reciprocal (mantissa) based on actual window size
	int16_t initial_recip_shift;	// reciprocal (shift)    based on actual window size
	int16_t recip_mant;				// reciprocal (mantissa)
	int16_t recip_shift;			// reciprocal (shift)
	int16_t	zero_code;				// only used in l2pool

	// for keeping track of whether we have special case, and details.
	int  hvx_specialized_handler_code;
	void *hvx_specialized_handler;
	void *misc;				// misc pointer
};
//
// set up the plan
//  return value:
//   <0:   error (and errlog has been called)
//    0:   OK (and reused old plan)
//    1:   OK (and built new plan)

int
setup_integral_buffer_plan( struct nn_node *self, struct nn_graph * nn,
		struct integral_buffer_plan * ibp,
		struct tensor const * in_tensor,
		struct tensor const * window_tensor,
		struct tensor const * stride_tensor);

// this can be used to force or encourage the 'unrolled' state of the integral buffer,
// after calling setup_integral_buffer_plan. if the full buffer size <=
// 'max_bytes', the buffer layout will be modified to 'unrolled'.
//
static inline void
integral_buffer_force_unrolled(struct integral_buffer_plan * ibp , uint32_t max_bytes )
{
	if( ibp->ibuf_is_rolled ){
		int top_pad = ibp->wpad_top;
		int ibuf_fullrows = top_pad + 1 + ibp->inshape.height + ibp->wpad_bottom;
		uint32_t new_size = ibuf_fullrows * ibp->ibuf_row_bytes;
		if( new_size <= max_bytes){
			ibp->ibuf_rows = ibuf_fullrows;
			ibp->ibuf_row0 = top_pad;
			ibp->ibuf_total_bytes = new_size;
			ibp->ibuf_is_rolled = 0;
		}
	}
}

//
// This struct represents the 'vertical' progress of an operation
// Will need one per thread, with different buffers; all point
// to the same integral_buffer_plan
//
struct integ_buff_vscan {
	int32_t * buffer_base;							// the base of the buffer
	int32_t * buffer_end;							// = buffer_base + ibuf_total_bytes
	int32_t ibuf_total_bytes;						// copy of ibp->ibuf_total_bytes

	// 'producer' process
	int producer_nrows;							// the number of rows loaded by producer so far
	int32_t * producer_prevrow;					// most recent row loaded by producer (points to 'col 0')
	int row_offset;								// producer_nrows - row_offset, if >0, is # output rows
												// available (assuming stride_h = 1)
	uint8_t const * producer_in_row;			// producer's current input row (vec aligned)
	int16_t producer_widvecs;							// producer's width in full vectors
	int16_t producer_w0;								// 0..3; width padding on left to suppress when loading.

	// these relate to bottom-padding; they are only valid
	// once the bottom-padding process starts, (producer_nrows > input height)
	// which is only applicable when bottom padding is requireds. These are pointers to
	// start of 'proper' row (not to col0).
	// The bottom padding can't be generated all at once after we load the last row, since
	// it may overwrite rows that are still needed. So it is treated as an extension of the loading process.
	//
	// Each time a padding row is generated, botpad_next and botpad_refrow get moved down one row (with wrapping).
	// Exception: when some bottom padding rows are infeasible, the initial 'botpad_refrow' is to row 0,
	// and it's only moved when all infeasible rows are generated.
	//
	int16_t bot_pad_count;				// # of rows generated so far
	int16_t input_height;
	int32_t const * row_H;				// this is where 'row H' is stored. row (in_ht + ibuf_row0)% ibuf_rows
	int32_t * botpad_next;				// points to next row to be generated ( bot_pad_count+1 rows below row_H)
	int32_t * botpad_refrow;			// points to the next row's reference row; win_h rows above botpad_next (or 0).


	// 'consumer' process
	int32_t const * consumer_top;				// pointer to 'top' row used by consumer
	int32_t const * consumer_bot;				// 'bottom' row used by consumer
	int32_t consumer_row_stride;				// this is ibuf_row_bytes * stride_h
	int consumer_rows_generated;				// # of generated output rows.
	uint8_t * consumer_out_row;					// consumer's current row ptr
	int16_t consumer_width;						// output width (counting only valid lanes)
	int16_t consumer_w0;						// # of width units to skip on left (0..3)
};

// set up the state in integ_buff_vscan
// ** First vscan->buffer_base must be pointed to the buffer area, of size
//  ibp->ibuf_total_bytes
//
//
static inline void
setup_integ_buffer_vscan( struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp, int batch_idx, int d32_idx)
{
	char * bufp = (char*) vscan->buffer_base;
	int32_t ibuf_total_bytes = ibp->ibuf_total_bytes;
	vscan->ibuf_total_bytes = ibuf_total_bytes;
	vscan->buffer_end = (int32_t*)(bufp+ ibuf_total_bytes );

	//
	// set up the producer's read pointer
	//
	uint8_t const *prod_rd_ptr = ibp->tin.data +  batch_idx * ibp->tin.batch_stride + d32_idx * ibp->tin.d32_stride;
	int in_wid = ibp->inshape.width;
	int in_ht = ibp->inshape.height;

	// l2 prefetch
	l2fetch( prod_rd_ptr, ibp->tin.height_stride, in_wid*32, in_ht);

	// pointer is 32-aligned but maybe not 128 aligned...
	int producer_w0 = ((size_t)prod_rd_ptr >> 5) & 3;		// # of width units to suppress on the left.
	// if producer_w0 > 0, adj pointer to the left and expand width
	vscan->producer_in_row = (uint8_t const*)( (size_t)prod_rd_ptr & ~127);
	vscan->producer_widvecs = ( producer_w0+ in_wid+3) >>2;	// # of full vecs.
	vscan->producer_w0 = producer_w0;
	vscan->producer_nrows = 0;

	int win_h = ibp->window_ht;
	int wpad_top = ibp->wpad_top;
	uint32_t rowbytes = ibp->ibuf_row_bytes;
	char * buf00 = bufp + ibp->ibuf_row0 * rowbytes + ibp->ibuf_col0_offs;

	// producer_prevrow points to row 0, column 0 and advances down rows. It is not adjusted for
	// 'producer_w0'.
	// it is assumed that the entire 0 row will be zeroed before loading starts
	vscan->producer_prevrow = (int32_t *) buf00;
	// this indicates skew: after 'n' rows are loaded, n-row_offset output rows will be available
	//  (assuming stride_h =1). This is always >= 0.
	vscan->row_offset = (win_h-1)-wpad_top;

	vscan->input_height = in_ht;
	vscan->bot_pad_count = 0;


	//
	// set up the consumer's write pointer
	//
	uint8_t *cons_wr_ptr = ibp->tout.data +  batch_idx * ibp->tout.batch_stride + d32_idx * ibp->tout.d32_stride;
	// pointer is 32-aligned but maybe not 128 aligned...
	int cons_w0 = ((size_t)cons_wr_ptr >> 5) & 3;		// # of width units to pad on left 0..3
	vscan->consumer_w0 = cons_w0;
	vscan->consumer_width = ibp->outshape.width;
	vscan->consumer_out_row = (uint8_t *)( (size_t)cons_wr_ptr & ~127);

	// consumer top: point to (0,0) but offset by top_pad, left_pad
	// consumer bot: win_h rows below that.
	char *cons_top = buf00 - (rowbytes*wpad_top + 128*ibp->wpad_left);
	vscan->consumer_top = (int32_t const *) cons_top;
	vscan->consumer_bot = (int32_t const *) ( cons_top+ win_h*rowbytes );
	vscan->consumer_rows_generated = 0;

	// amount to bump consumer_top, consumer_bot after each output row
	// (after bumping, check if >= buffer_end, and subtract ibuf_total_bytes if so).
	vscan->consumer_row_stride = rowbytes * ibp->stride_ht;
}
//
// call after setup_integ_buffer_vscan to clear the first
// row of buffer
//
static inline void
vscan_clear_initial( struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp)
{
	char * ptr = (char *)vscan->buffer_base
		+ ibp->ibuf_row0 * ibp->ibuf_row_bytes + ibp->ibuf_proper_offs;
	memset( ptr, 0, ibp->ibuf_proper_row_bytes );
}

// HVX version...
static inline void
vscan_clear_initial_hvx( struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp)
{
	HVX_Vector * ptr = (HVX_Vector*) ( (char *)vscan->buffer_base
		+ ibp->ibuf_row0 * ibp->ibuf_row_bytes + ibp->ibuf_proper_offs);
	int nvec = (unsigned)ibp->ibuf_proper_row_bytes / 128u;
	for(int i = 0; i < nvec ; i++ ){
		ptr[i] = Q6_V_vzero();
	}
}


static inline void
vscan_bump_consumer_pointers(struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp)
{
	char * new_top_p = (char *) vscan->consumer_top + vscan->consumer_row_stride;
	char * new_bot_p = (char *) vscan->consumer_bot + vscan->consumer_row_stride;
	if( new_top_p >= (char*)vscan->buffer_end ) new_top_p -= vscan->ibuf_total_bytes;
	if( new_bot_p >= (char*)vscan->buffer_end ) new_bot_p -= vscan->ibuf_total_bytes;
	vscan->consumer_top  = (int32_t const *) new_top_p;
	vscan->consumer_bot  = (int32_t const *) new_bot_p;
	vscan->consumer_out_row += ibp->tout.height_stride;
}

#define PTR_ADD_BYTES( T, p, n) ((T*)( (char *)(p) + (n)))
#define PTR_SUB_BYTES( T, p, n) ((T*)( (char *)(p) - (n)))

#ifndef NN_INTEGRAL_BUF_FOR_L2POOL
static inline void
avgpool_inner_load1_loop( int32_t * row0_ptr, int32_t const * ref_row_ptr, const uint8_t * inrow0, int wid_proc , int parm)
{
	for(int id = 0; id < 32; id ++){
		int32_t hsum0=0;	// sum of pixels on 0 row
		int32_t prevsum= 0;	// sum from above.
		for( int j = 0; j < wid_proc; j++){
			row0_ptr[32*j +id]= hsum0*256 + prevsum;
			prevsum = ref_row_ptr[32*(j+1) + id];
			int pix0 = inrow0[32*j+id];
			hsum0 += pix0;
		}
		row0_ptr[32*wid_proc+id] = hsum0*256+prevsum;
	}
}


static inline void
avgpool_inner_load2_loop( int32_t * row0_ptr,
		int32_t *row1_ptr,
		int32_t const * ref_row_ptr,
		const uint8_t * inrow0,
		const uint8_t * inrow1,
		int wid_proc,
		int parm )
{
	for(int id = 0; id < 32; id ++){
		int32_t hsum0=0;	// sum of pixels on 0 row
		int32_t hsum1=0;	// sum of pixels on 0 and 1 rows
		int32_t prevsum= 0;	// sum from above.
		for( int j = 0; j < wid_proc; j++){
			row0_ptr[32*j+id]= hsum0*256 + prevsum;
			row1_ptr[32*j+id]= hsum1*256 +prevsum;
			prevsum = ref_row_ptr[32*(j+1)+id];
			int pix0 = inrow0[32*j+id];
			int pix1 = inrow1[32*j+id];
			hsum0 += pix0;
			hsum1 += (pix1+pix0);
		}
		row0_ptr[32*wid_proc+id] = hsum0*256+prevsum;
		row1_ptr[32*wid_proc+id] = hsum1*256+prevsum;
	}
}




#else
static inline void
l2pool_inner_load1_loop( int32_t * row0_ptr, int32_t const * ref_row_ptr, const uint8_t * inrow0, int wid_proc , int zcode){
	for(int id = 0; id < 32; id ++){
		int32_t hsum0=0;	// sum of pixels^2 on 0 row
		int32_t prevsum= 0;	// sum from above.
		for( int j = 0; j < wid_proc; j++){
			row0_ptr[32*j +id]= hsum0 + prevsum;
			prevsum = ref_row_ptr[32*(j+1) + id];
			int pix0 = inrow0[32*j+id]-zcode;
			hsum0 += pix0*pix0;
		}
		row0_ptr[32*wid_proc+id] = hsum0+prevsum;
	}
}
static inline void
l2pool_inner_load2_loop( int32_t * row0_ptr,
		int32_t *row1_ptr,
		int32_t const * ref_row_ptr,
		const uint8_t * inrow0,
		const uint8_t * inrow1,
		int wid_proc,
		int zcode )
{
	for(int id = 0; id < 32; id ++){
		int32_t hsum0=0;	// sum of pixels^2 on 0 row
		int32_t hsum1=0;	// sum of pixels^2 on 0 and 1 rows
		int32_t prevsum= 0;	// sum from above.
		for( int j = 0; j < wid_proc; j++){
			row0_ptr[32*j+id]= hsum0 + prevsum;
			row1_ptr[32*j+id]= hsum1 +prevsum;
			prevsum = ref_row_ptr[32*(j+1)+id];
			int pix0 = inrow0[32*j+id]-zcode;
			int pix1 = inrow1[32*j+id]-zcode;
			pix0 *= pix0;
			pix1 *= pix1;
			hsum0 += pix0;
			hsum1 += (pix1+pix0);
		}
		row0_ptr[32*wid_proc+id] = hsum0+prevsum;
		row1_ptr[32*wid_proc+id] = hsum1+prevsum;
	}
}
#endif

#ifdef __hexagon__
#define FAKEDEP_VV( vec,vec2)	asm ("/*%0 %1*/": "=v"(vec), "=v"(vec2): "0"(vec), "1"(vec2))
#else
#define FAKEDEP_VV( vec,vec2)
#endif


///////////////////////////////////////////////////////////////////////////
// avgpool loading loops
#ifndef NN_INTEGRAL_BUF_FOR_L2POOL
//
// inner loop, hvx, 1 rows at once
// - reads nvec vectors from inrow0,
// - reads nvec*4+1 vectors from ref_row_ptr;
// - writes nvec*4+1 vector to each of row0_ptr
//  The first output on each of those rows is 0.
//
// All of the pointers are vec-aligned. The first
// input vector may contain 0..3 'dead' lanes;
// this is encoded in in_w0 (e.g. in_w0=1 means the first
//  32 input bytes are padding; they will be forced to 0
// when processed)
//
static inline void
avgpool_inner_load1_loop_hvx( int32_t * row0_ptr,
		int32_t const * ref_row_ptr,
		const uint8_t * inrow0,
		int nvec,
		int in_w0,
		int parm )
{
	HVX_Vector const *refrow = (HVX_Vector const*)ref_row_ptr;
	HVX_Vector const *vinp0 = (HVX_Vector const*) inrow0;

	HVX_Vector * voutp0 = (HVX_Vector *)row0_ptr;
	HVX_Vector rowBefore = Q6_V_vzero();
	HVX_Vector Above0 = Q6_V_vzero();

	// pixels are multiplied by 16 when promoting
	// from u8 to i16; and then again when promoting
	// from i16 to i32.
	//
	uint32_t const_NNNN = 0x10101010;
	uint32_t const_0N0N = 0x00100010;

	// mask to suppress 0,1,2 or 3 initial width units
	HVX_VectorPred startmask = Q6_Q_vsetq_R( 32*in_w0);

	HVX_Vector extAB,extCD;
	{ // unpeel half loop...
		HVX_Vector vin0 = *vinp0++;

		vin0 = q6op_V_vand_QnV( startmask, vin0 );

		HVX_VectorPair shuf1 = Q6_W_vshuff_VVR( Q6_V_vzero(), vin0,-1);

		// extAB,extCD are the values of AB and CD multiplied by 16.
		// sumAB,sumCD are the same thing but added across both rows
		// all are ordered A0 A1 .. A31  B0 B1 .. B31

		extAB = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(shuf1), const_0N0N );
		extCD = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(shuf1), const_0N0N );
	}
	for( int j = 0; j < nvec-1; j++ ){
		//A0 A1  .. A31  B0 B1 ... B31
		//C0 C1  .. C31  D0 D0 ... D31
		//... shuffle those to get
		//A0 C0 A1    ....     A31 C31
		//B0 D0 B1    ....     B31 D31

		HVX_VectorPair vshuf1 = Q6_W_vshuff_VVR(  extCD, extAB, -2 );

		HVX_Vector rowA    = Q6_Vw_vdmpyacc_VwVhRb(  rowBefore, Q6_V_lo_W(vshuf1), const_0N0N );
		HVX_Vector rowB    = Q6_Vw_vdmpy_VhRb(  Q6_V_hi_W(vshuf1), const_0N0N );
		HVX_Vector rowAC   = Q6_Vw_vdmpyacc_VwVhRb( rowBefore, Q6_V_lo_W(vshuf1), const_NNNN );
		HVX_Vector rowABCD = Q6_Vw_vdmpyacc_VwVhRb( rowAC, Q6_V_hi_W(vshuf1), const_NNNN );
		HVX_Vector rowAB   = Q6_Vw_vadd_VwVw( rowA, rowB);		// prefix + A+ B
		HVX_Vector rowABC  = Q6_Vw_vadd_VwVw( rowAC, rowB);		// prefix + A+ B +C

		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp0 += 4;
		refrow += 4;
		rowBefore = rowABCD;
		//------------- split...
		HVX_Vector vin0 = *vinp0++;
		HVX_VectorPair shuf1 = Q6_W_vshuff_VVR( Q6_V_vzero(), vin0,-1);

		extAB = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(shuf1), const_0N0N );
		extCD = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(shuf1), const_0N0N );
	}// end loop
	// unpeel last half loop
	// This 'fake asm' prevents the compiler from seeing the below as
	// being the same as the first 75% of the loop; if it sees that it can
	// roll them back together with a 'middle exit' and then we don't get a loop0.
	FAKEDEP_VV( extAB,extCD);
	{

		HVX_VectorPair vshuf1 = Q6_W_vshuff_VVR(  extCD, extAB, -2 );

		HVX_Vector rowA    = Q6_Vw_vdmpyacc_VwVhRb(  rowBefore, Q6_V_lo_W(vshuf1), const_0N0N );
		HVX_Vector rowB    = Q6_Vw_vdmpy_VhRb(  Q6_V_hi_W(vshuf1), const_0N0N );
		HVX_Vector rowAC   = Q6_Vw_vdmpyacc_VwVhRb( rowBefore, Q6_V_lo_W(vshuf1), const_NNNN );
		HVX_Vector rowABCD = Q6_Vw_vdmpyacc_VwVhRb( rowAC, Q6_V_hi_W(vshuf1), const_NNNN );
		HVX_Vector rowAB   = Q6_Vw_vadd_VwVw( rowA, rowB);		// prefix + A+ B
		HVX_Vector rowABC  = Q6_Vw_vadd_VwVw( rowAC, rowB);		// prefix + A+ B +C

		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp0 += 4;
		refrow += 4;
		rowBefore = rowABCD;

	}
	voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
}

//
// avgpool inner loop, hvx, 2 rows at once
// - reads nvec vectors from each of inrow0, inrow1
// - reads nvec*4+1 vectors from ref_row_ptr;
// - writes nvec*4+1 vector to each of row0_ptr, row1_ptr
//  The first output on each of those rows is 0.
//
// All of the pointers are vec-aligned. The first
// input vector may contain 0..3 'dead' lanes;
// this is encoded in in_w0 (e.g. in_w0=1 means the first
//  32 input bytes are padding; they will be forced to 0
// when processed)
//
static inline void
avgpool_inner_load2_loop_hvx( int32_t * row0_ptr,
		int32_t *row1_ptr,
		int32_t const * ref_row_ptr,
		const uint8_t * inrow0,
		const uint8_t * inrow1,
		int nvec,
		int in_w0,
		int parm )
{
	HVX_Vector const *refrow = (HVX_Vector const*)ref_row_ptr;
	HVX_Vector const *vinp0 = (HVX_Vector const*) inrow0;
	HVX_Vector const *vinp1 = (HVX_Vector const*) inrow1;

	HVX_Vector * voutp0 = (HVX_Vector *)row0_ptr;
	HVX_Vector * voutp1 = (HVX_Vector *)row1_ptr;

	HVX_Vector rowBefore = Q6_V_vzero();
	HVX_Vector row2Before = Q6_V_vzero();

	HVX_Vector Above0 = Q6_V_vzero();

	// pixels are multiplied by 16 when promoting
	// from u8 to i16; and then again when promoting
	// from i16 to i32.
	//
	uint32_t const_NNNN = 0x10101010;
	uint32_t const_0N0N = 0x00100010;

	// mask to suppress 0,1,2 or 3 initial width units
	HVX_VectorPred startmask = Q6_Q_vsetq_R( 32*in_w0);

	HVX_Vector extAB,sumAB, extCD,sumCD;
	// ... start unpeel of 1/2 loop
	{
		HVX_Vector vin0 = *vinp0++;
		HVX_Vector vin1 = *vinp1++;

		vin0 = q6op_V_vand_QnV( startmask, vin0 );
		vin1 = q6op_V_vand_QnV( startmask, vin1 );

		HVX_VectorPair shuf1 = Q6_W_vshuff_VVR( vin1, vin0,-1);

		// extAB,extCD are the values of AB and CD multiplied by 16.
		// sumAB,sumCD are the same thing but added across both rows
		// all are ordered A0 A1 .. A31  B0 B1 .. B31
		extAB = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(shuf1), const_0N0N );
		sumAB = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(shuf1), const_NNNN );
		extCD = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(shuf1), const_0N0N );
		sumCD = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(shuf1), const_NNNN );
	}
	for( int j = 0; j < nvec-1; j++){

		//A0 A1  .. A31  B0 B1 ... B31
		//C0 C1  .. C31  D0 D0 ... D31
		//... shuffle those to get
		//A0 C0 A1    ....     A31 C31
		//B0 D0 B1    ....     B31 D31

		HVX_VectorPair vshuf1 = Q6_W_vshuff_VVR(  extCD, extAB, -2 );
		HVX_VectorPair vshuf2 = Q6_W_vshuff_VVR(  sumCD, sumAB, -2 );

		HVX_Vector rowA    = Q6_Vw_vdmpyacc_VwVhRb(  rowBefore, Q6_V_lo_W(vshuf1), const_0N0N );
		HVX_Vector rowB    = Q6_Vw_vdmpy_VhRb(  Q6_V_hi_W(vshuf1), const_0N0N );
		HVX_Vector rowAC   = Q6_Vw_vdmpyacc_VwVhRb( rowBefore, Q6_V_lo_W(vshuf1), const_NNNN );
		HVX_Vector rowABCD = Q6_Vw_vdmpyacc_VwVhRb( rowAC, Q6_V_hi_W(vshuf1), const_NNNN );
		HVX_Vector rowAB   = Q6_Vw_vadd_VwVw( rowA, rowB);		// prefix + A + B
		HVX_Vector rowABC  = Q6_Vw_vadd_VwVw( rowAC, rowB);		// prefix + A+ B +C

		HVX_Vector row2A    = Q6_Vw_vdmpyacc_VwVhRb(  row2Before, Q6_V_lo_W(vshuf2), const_0N0N );
		HVX_Vector row2B    = Q6_Vw_vdmpy_VhRb(  Q6_V_hi_W(vshuf2), const_0N0N );
		HVX_Vector row2AC   = Q6_Vw_vdmpyacc_VwVhRb( row2Before, Q6_V_lo_W(vshuf2), const_NNNN );
		HVX_Vector row2ABCD = Q6_Vw_vdmpyacc_VwVhRb( row2AC, Q6_V_hi_W(vshuf2), const_NNNN );
		HVX_Vector row2AB   = Q6_Vw_vadd_VwVw( row2A, row2B);		// prefix + A + B
		HVX_Vector row2ABC  = Q6_Vw_vadd_VwVw( row2AC, row2B);		// prefix + A+ B +C

		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp1[0] = Q6_Vw_vadd_VwVw(Above0,row2Before);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		voutp1[1] = Q6_Vw_vadd_VwVw(AboveA,row2A);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp1[2] = Q6_Vw_vadd_VwVw(AboveB,row2AB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp1[3] = Q6_Vw_vadd_VwVw(AboveC,row2ABC);
		voutp0 += 4;
		voutp1 += 4;
		refrow += 4;
		rowBefore = rowABCD;
		row2Before = row2ABCD;

		//--------/ middle ...
		HVX_Vector vin0 = *vinp0++;
		HVX_Vector vin1 = *vinp1++;

		HVX_VectorPair shuf1 = Q6_W_vshuff_VVR( vin1, vin0,-1);

		extAB = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(shuf1), const_0N0N );
		sumAB = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(shuf1), const_NNNN );
		extCD = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(shuf1), const_0N0N );
		sumCD = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(shuf1), const_NNNN );
	} // end of loop
	// This 'fake asm' prevents the compiler from seeing the below as
	// being the same as the first 75% of the loop; if it sees that it can
	// roll them back together with a 'middle exit' and then we don't get a loop0.
	FAKEDEP_VV( extAB,sumAB);
	FAKEDEP_VV( extCD,sumCD);
	{  // >> unpeel

		HVX_VectorPair vshuf1 = Q6_W_vshuff_VVR(  extCD, extAB, -2 );
		HVX_VectorPair vshuf2 = Q6_W_vshuff_VVR(  sumCD, sumAB, -2 );

		HVX_Vector rowA    = Q6_Vw_vdmpyacc_VwVhRb(  rowBefore, Q6_V_lo_W(vshuf1), const_0N0N );
		HVX_Vector rowB    = Q6_Vw_vdmpy_VhRb(  Q6_V_hi_W(vshuf1), const_0N0N );
		HVX_Vector rowAC   = Q6_Vw_vdmpyacc_VwVhRb( rowBefore, Q6_V_lo_W(vshuf1), const_NNNN );
		HVX_Vector rowABCD = Q6_Vw_vdmpyacc_VwVhRb( rowAC, Q6_V_hi_W(vshuf1), const_NNNN );
		HVX_Vector rowAB   = Q6_Vw_vadd_VwVw( rowA, rowB);		// prefix + A + B
		HVX_Vector rowABC  = Q6_Vw_vadd_VwVw( rowAC, rowB);		// prefix + A+ B +C

		HVX_Vector row2A    = Q6_Vw_vdmpyacc_VwVhRb(  row2Before, Q6_V_lo_W(vshuf2), const_0N0N );
		HVX_Vector row2B    = Q6_Vw_vdmpy_VhRb(  Q6_V_hi_W(vshuf2), const_0N0N );
		HVX_Vector row2AC   = Q6_Vw_vdmpyacc_VwVhRb( row2Before, Q6_V_lo_W(vshuf2), const_NNNN );
		HVX_Vector row2ABCD = Q6_Vw_vdmpyacc_VwVhRb( row2AC, Q6_V_hi_W(vshuf2), const_NNNN );
		HVX_Vector row2AB   = Q6_Vw_vadd_VwVw( row2A, row2B);		// prefix + A + B
		HVX_Vector row2ABC  = Q6_Vw_vadd_VwVw( row2AC, row2B);		// prefix + A+ B +C

		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp1[0] = Q6_Vw_vadd_VwVw(Above0,row2Before);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		voutp1[1] = Q6_Vw_vadd_VwVw(AboveA,row2A);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp1[2] = Q6_Vw_vadd_VwVw(AboveB,row2AB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp1[3] = Q6_Vw_vadd_VwVw(AboveC,row2ABC);
		voutp0 += 4;
		voutp1 += 4;
		refrow += 4;
		rowBefore = rowABCD;
		row2Before = row2ABCD;
	} // << end unpeel
	voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
	voutp1[0] = Q6_Vw_vadd_VwVw(Above0,row2Before);
}
#else
////////////////////////// L2pool loading loops

//
// l2pool inner loop, hvx, 1 rows at once
// - reads nvec vectors from inrow0,
// - reads nvec*4+1 vectors from ref_row_ptr;
// - writes nvec*4+1 vector to each of row0_ptr
//  The first output on each of those rows is 0.
//
// All of the pointers are vec-aligned. The first
// input vector may contain 0..3 'dead' lanes;
// this is encoded in in_w0 (e.g. in_w0=1 means the first
//  32 input bytes are padding; they will be forced to 0
// when processed)

static inline void
l2pool_inner_load1_loop_hvx( int32_t * row0_ptr,
		int32_t const * ref_row_ptr,
		const uint8_t * inrow0,
		int nvec,
		int in_w0,
		int parm )
{
	HVX_Vector inzero = q6op_Vb_vsplat_R( parm);
	HVX_Vector const *refrow = (HVX_Vector const*)ref_row_ptr;
	HVX_Vector const *vinp0 = (HVX_Vector const*) inrow0;

	HVX_Vector * voutp0 = (HVX_Vector *)row0_ptr;
	HVX_Vector rowBefore = Q6_V_vzero();
	HVX_Vector Above0 = Q6_V_vzero();

	// mask to suppress 0,1,2 or 3 initial width units
	HVX_VectorPred startmask = Q6_Q_vsetq_R( 32*in_w0);

	HVX_Vector difAC,difBD;
	{ // unpeel half loop...
		HVX_Vector vin0 = *vinp0++;
		// replace 'width padding' values with inzero
		vin0 = Q6_V_vmux_QVV (startmask, inzero, vin0);
		// shuffle...
		// A0 C0 A1 ... A31 C31 B0 D0   .. D31
		vin0 = Q6_Vb_vshuff_Vb( vin0);
		// subtract inzero from it, promote to int16 in the process
		HVX_VectorPair x16 = Q6_Wh_vsub_VubVub( vin0, inzero);
		// A0 A1 .. A31 B0 ..B31
		// C0 C1 .. C31 D0..D31
		// Reorder
		//
		x16 = Q6_W_vshuff_VVR(  Q6_V_hi_W(x16), Q6_V_lo_W(x16),-2);
		// A0 C0 A1 C1 ..  B31
		// B0 D0 ..        D31
		difAC = Q6_V_lo_W(x16);
		difBD = Q6_V_hi_W(x16);

	}
	for( int j = 0; j < nvec-1; j++ ){
		// square it
		HVX_VectorPair sqA_C = Q6_Ww_vmpy_VhVh(difAC,difAC);
		HVX_VectorPair sqAB_CD = Q6_Ww_vmpyacc_WwVhVh(sqA_C, difBD,difBD);
		// now we have 4 of A0 .. A31, B0 .. B31

		HVX_Vector rowA = Q6_Vw_vadd_VwVw( rowBefore, Q6_V_lo_W(sqA_C));	// before + A
		HVX_Vector rowAB = Q6_Vw_vadd_VwVw( rowBefore, Q6_V_lo_W(sqAB_CD));	// before + A+B
		HVX_Vector rowABC = Q6_Vw_vadd_VwVw( rowAB, Q6_V_hi_W(sqA_C));	// before + A+B+C
		HVX_Vector rowABCD = Q6_Vw_vadd_VwVw( rowAB, Q6_V_hi_W(sqAB_CD));	// before + A+B+C+D

		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp0 += 4;
		refrow += 4;
		rowBefore = rowABCD;
		//------------- split...
		HVX_Vector vin0 = *vinp0++;
		vin0 = Q6_Vb_vshuff_Vb( vin0);
		HVX_VectorPair x16 = Q6_Wh_vsub_VubVub( vin0, inzero);
		x16 = Q6_W_vshuff_VVR(  Q6_V_hi_W(x16), Q6_V_lo_W(x16),-2);
		difAC = Q6_V_lo_W(x16);
		difBD = Q6_V_hi_W(x16);

	}// end loop
	// unpeel last half loop
	// This 'fake asm' prevents the compiler from seeing the below as
	// being the same as the first 75% of the loop; if it sees that it can
	// roll them back together with a 'middle exit' and then we don't get a loop0.
	FAKEDEP_VV( difAC,difBD);
	{
		HVX_VectorPair sqA_C = Q6_Ww_vmpy_VhVh(difAC,difAC);
		HVX_VectorPair sqAB_CD = Q6_Ww_vmpyacc_WwVhVh(sqA_C, difBD,difBD);
		HVX_Vector rowA = Q6_Vw_vadd_VwVw( rowBefore, Q6_V_lo_W(sqA_C));	// before + A
		HVX_Vector rowAB = Q6_Vw_vadd_VwVw( rowBefore, Q6_V_lo_W(sqAB_CD));	// before + A+B
		HVX_Vector rowABC = Q6_Vw_vadd_VwVw( rowAB, Q6_V_hi_W(sqA_C));	// before + A+B+C
		HVX_Vector rowABCD = Q6_Vw_vadd_VwVw( rowAB, Q6_V_hi_W(sqAB_CD));	// before + A+B+C+D
		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp0 += 4;
		refrow += 4;
		rowBefore = rowABCD;

	}
	voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
}


//
// l2pool inner loop, hvx, 2 rows at once
// - reads nvec vectors from each of inrow0, inrow1
// - reads nvec*4+1 vectors from ref_row_ptr;
// - writes nvec*4+1 vector to each of row0_ptr, row1_ptr
//  The first output on each of those rows is 0.
//
// All of the pointers are vec-aligned. The first
// input vector may contain 0..3 'dead' lanes;
// this is encoded in in_w0 (e.g. in_w0=1 means the first
//  32 input bytes are padding; they will be forced to 0
// when processed)
//

static inline void
l2pool_inner_load2_loop_hvx( int32_t * row0_ptr,
		int32_t *row1_ptr,
		int32_t const * ref_row_ptr,
		const uint8_t * inrow0,
		const uint8_t * inrow1,
		int nvec,
		int in_w0,
		int parm )
{

	HVX_Vector inzero = q6op_Vb_vsplat_R( parm);
	HVX_Vector const *refrow = (HVX_Vector const*)ref_row_ptr;
	HVX_Vector const *vinp0 = (HVX_Vector const*) inrow0;
	HVX_Vector const *vinp1 = (HVX_Vector const*) inrow1;

	HVX_Vector * voutp0 = (HVX_Vector *)row0_ptr;
	HVX_Vector * voutp1 = (HVX_Vector *)row1_ptr;

	HVX_Vector rowBefore = Q6_V_vzero();
	HVX_Vector row2Before = Q6_V_vzero();

	HVX_Vector Above0 = Q6_V_vzero();


	// mask to suppress 0,1,2 or 3 initial width units
	HVX_VectorPred startmask = Q6_Q_vsetq_R( 32*in_w0);

	HVX_Vector difAC_0, difBD_0, difAC_1, difBD_1;

	{ // unpeel half loop...
		// read two...
		HVX_Vector vin0 = *vinp0++;
		HVX_Vector vin1 = *vinp1++;
		// replace 'width padding' values with inzero
		vin0 = Q6_V_vmux_QVV (startmask, inzero, vin0);
		vin1 = Q6_V_vmux_QVV (startmask, inzero, vin1);
		HVX_VectorPair inshuf = Q6_W_vshuff_VVR( vin1, vin0, -1);

		// shuffle...  (ABCD from vin0, abcd from vin1)
		// A0 a0 ... A31 a31 B0 b0   .. b31
		// C0 c0 ... C31 c31 D0 d0   ...d31

		// subtract inzero from it, promote to int16 in the process
		HVX_VectorPair x16ab = Q6_Wh_vsub_VubVub( Q6_V_lo_W(inshuf), inzero);
		HVX_VectorPair x16cd = Q6_Wh_vsub_VubVub( Q6_V_hi_W(inshuf), inzero);
		// x16ab: A0  .. A31 B0 .. B31
		//        a0  .. a31 b0 .. b31
		// x16cd: C0 ..  C31 D0 .. D31
		//        c0 ..  c31 d0 .. d31
		// Reorder
		HVX_VectorPair x16_0 = Q6_W_vshuff_VVR(  Q6_V_lo_W(x16cd), Q6_V_lo_W(x16ab),-2);
		// A0 C0  ... A31 C31
		// B0 D0  ... B31 D31
		HVX_VectorPair x16_1 = Q6_W_vshuff_VVR(  Q6_V_hi_W(x16cd), Q6_V_hi_W(x16ab),-2);
		// a0 c0  ... a31 c31
		// b0 d0  ... b31 d31
		// now we can do the squares of A,B,C,D
		difAC_0 = Q6_V_lo_W(x16_0);
		difBD_0 = Q6_V_hi_W(x16_0);
		difAC_1 = Q6_V_lo_W(x16_1);
		difBD_1 = Q6_V_hi_W(x16_1);
	}
	for( int j = 0; j < nvec-1; j++){
		// find all the squares; for the lower row, add the upper row's squares.
		HVX_VectorPair sqA0_C0 = Q6_Ww_vmpy_VhVh(difAC_0,difAC_0);	//A^2, C^2
		HVX_VectorPair sqB0_D0 = Q6_Ww_vmpy_VhVh(difBD_0,difBD_0);	//B^2, D^2
		HVX_VectorPair sqA1_C1 = Q6_Ww_vmpyacc_WwVhVh(sqA0_C0, difAC_1,difAC_1);	// +a^2, c^2
		HVX_VectorPair sqB1_D1 = Q6_Ww_vmpyacc_WwVhVh(sqB0_D0, difBD_1,difBD_1);	// b^2, d^2

		// add across
		HVX_Vector rowA = Q6_Vw_vadd_VwVw(rowBefore, Q6_V_lo_W(sqA0_C0) );
		HVX_Vector rowAB = Q6_Vw_vadd_VwVw(rowA, Q6_V_lo_W(sqB0_D0) );
		HVX_Vector rowABC = Q6_Vw_vadd_VwVw(rowAB, Q6_V_hi_W(sqA0_C0) );
		HVX_Vector rowABCD = Q6_Vw_vadd_VwVw(rowABC, Q6_V_hi_W(sqB0_D0) );

		HVX_Vector row2A = Q6_Vw_vadd_VwVw(row2Before, Q6_V_lo_W(sqA1_C1) );
		HVX_Vector row2AB = Q6_Vw_vadd_VwVw(row2A, Q6_V_lo_W(sqB1_D1) );
		HVX_Vector row2ABC = Q6_Vw_vadd_VwVw(row2AB, Q6_V_hi_W(sqA1_C1) );
		HVX_Vector row2ABCD = Q6_Vw_vadd_VwVw(row2ABC, Q6_V_hi_W(sqB1_D1) );


		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp1[0] = Q6_Vw_vadd_VwVw(Above0,row2Before);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		voutp1[1] = Q6_Vw_vadd_VwVw(AboveA,row2A);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp1[2] = Q6_Vw_vadd_VwVw(AboveB,row2AB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp1[3] = Q6_Vw_vadd_VwVw(AboveC,row2ABC);
		voutp0 += 4;
		voutp1 += 4;
		refrow += 4;
		rowBefore = rowABCD;
		row2Before = row2ABCD;

		//--------/ middle ...
		HVX_Vector vin0 = *vinp0++;
		HVX_Vector vin1 = *vinp1++;
		HVX_VectorPair inshuf = Q6_W_vshuff_VVR( vin1, vin0, -1);
		HVX_VectorPair x16ab = Q6_Wh_vsub_VubVub( Q6_V_lo_W(inshuf), inzero);
		HVX_VectorPair x16cd = Q6_Wh_vsub_VubVub( Q6_V_hi_W(inshuf), inzero);
		HVX_VectorPair x16_0 = Q6_W_vshuff_VVR(  Q6_V_lo_W(x16cd), Q6_V_lo_W(x16ab),-2);
		HVX_VectorPair x16_1 = Q6_W_vshuff_VVR(  Q6_V_hi_W(x16cd), Q6_V_hi_W(x16ab),-2);
		difAC_0 = Q6_V_lo_W(x16_0);
		difBD_0 = Q6_V_hi_W(x16_0);
		difAC_1 = Q6_V_lo_W(x16_1);
		difBD_1 = Q6_V_hi_W(x16_1);
	} // end of loop
	// This 'fake asm' prevents the compiler from seeing the below as
	// being the same as the first 75% of the loop; if it sees that it can
	// roll them back together with a 'middle exit' and then we don't get a loop0.
	FAKEDEP_VV( difAC_0,difBD_0);
	FAKEDEP_VV( difAC_1,difBD_1);
	{  // >> unpeel
		HVX_VectorPair sqA0_C0 = Q6_Ww_vmpy_VhVh(difAC_0,difAC_0);	//A^2, C^2
		HVX_VectorPair sqB0_D0 = Q6_Ww_vmpy_VhVh(difBD_0,difBD_0);	//B^2, D^2
		HVX_VectorPair sqA1_C1 = Q6_Ww_vmpyacc_WwVhVh(sqA0_C0, difAC_1,difAC_1);	// +a^2, c^2
		HVX_VectorPair sqB1_D1 = Q6_Ww_vmpyacc_WwVhVh(sqB0_D0, difBD_1,difBD_1);	// b^2, d^2
		HVX_Vector rowA = Q6_Vw_vadd_VwVw(rowBefore, Q6_V_lo_W(sqA0_C0) );
		HVX_Vector rowAB = Q6_Vw_vadd_VwVw(rowA, Q6_V_lo_W(sqB0_D0) );
		HVX_Vector rowABC = Q6_Vw_vadd_VwVw(rowAB, Q6_V_hi_W(sqA0_C0) );
		HVX_Vector rowABCD = Q6_Vw_vadd_VwVw(rowABC, Q6_V_hi_W(sqB0_D0) );

		HVX_Vector row2A = Q6_Vw_vadd_VwVw(row2Before, Q6_V_lo_W(sqA1_C1) );
		HVX_Vector row2AB = Q6_Vw_vadd_VwVw(row2A, Q6_V_lo_W(sqB1_D1) );
		HVX_Vector row2ABC = Q6_Vw_vadd_VwVw(row2AB, Q6_V_hi_W(sqA1_C1) );
		HVX_Vector row2ABCD = Q6_Vw_vadd_VwVw(row2ABC, Q6_V_hi_W(sqB1_D1) );
		HVX_Vector AboveA = refrow[1];
		HVX_Vector AboveB = refrow[2];
		HVX_Vector AboveC = refrow[3];
		voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
		voutp1[0] = Q6_Vw_vadd_VwVw(Above0,row2Before);
		voutp0[1] = Q6_Vw_vadd_VwVw(AboveA,rowA);
		voutp1[1] = Q6_Vw_vadd_VwVw(AboveA,row2A);
		Above0 = refrow[4];
		voutp0[2] = Q6_Vw_vadd_VwVw(AboveB,rowAB);
		voutp1[2] = Q6_Vw_vadd_VwVw(AboveB,row2AB);
		voutp0[3] = Q6_Vw_vadd_VwVw(AboveC,rowABC);
		voutp1[3] = Q6_Vw_vadd_VwVw(AboveC,row2ABC);
		voutp0 += 4;
		voutp1 += 4;
		refrow += 4;
		rowBefore = rowABCD;
		row2Before = row2ABCD;
	} // << end unpeel
	voutp0[0] = Q6_Vw_vadd_VwVw(Above0,rowBefore);
	voutp1[0] = Q6_Vw_vadd_VwVw(Above0,row2Before);
}

#endif //  NN_INTEGRAL_BUF_FOR_L2POOL
///////////////////////////////////////////////////////////////////////////
// multiply a 32-bit value by a signed 16-bit quantity with 15 fractional bits
//
static inline int32_t mul_frac_i32xi16( int32_t val, int16_t scl15)
{
	int64_t prod = (int64_t)val * scl15;
	return (prod + 0x4000) >> 15;
}

//
// load 'n' rows of u8 data into the integral buffer
// This is the 'scalar version'; unlike the vector version, it
// won't clear lanes to the left of the 0 column when producer_w0 > 0
// Note that all values are x 256 before summation, to gain
// fractional bits we use in the edge-padding calcs.
// It is a requirement that nrows > 0, and also must be is even,
// unless the input height is odd, and we are loading all of the remaining rows.
//
// We also assume that the first call is done with nrows= ibuf_initial_load;
// so that on the first call, we know that we can do the 'upper padding' rows.
//
//

static inline void
INTEGRAL_BUF_FUNC(producer_load_rows)( struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp, int nrows )
{
	int ibuf_row_bytes = ibp->ibuf_row_bytes;
	int wpad_flags = ibp->wpad_flags;

	int32_t * ref_row_ptr = vscan->producer_prevrow;		// previous row is a reference row

	int row_cnt = vscan->producer_nrows;				// # of rows so far
	int prod_w0 = vscan->producer_w0;

	int sigzero = 0;
#ifdef NN_INTEGRAL_BUF_FOR_L2POOL
	sigzero = ibp->zero_code;
#endif

	// # of width units to process (not counting 0 fill)
	int wid_proc = 4* vscan->producer_widvecs - prod_w0;

	// need input row pointer(s)
	uint8_t const * inrow0 = vscan->producer_in_row;
	int inrow_stride = ibp->tin.height_stride;
	vscan->producer_in_row = inrow0 + nrows * inrow_stride;
	int new_nrows = row_cnt + nrows;
	vscan->producer_nrows = new_nrows;

	inrow0 += 32*prod_w0;			// skip padding cols
	int nrows_remain = nrows;
	while( nrows_remain > 0){
		int32_t * row0_ptr= PTR_ADD_BYTES(int32_t, ref_row_ptr, ibuf_row_bytes);
		// wrap?
		if( row0_ptr >= vscan->buffer_end)
				row0_ptr = PTR_SUB_BYTES( int32_t, row0_ptr, vscan->ibuf_total_bytes);
		// now go do it...

		if( nrows_remain == 1){
			INTEGRAL_BUF_FUNC(inner_load1_loop)( row0_ptr, ref_row_ptr,  inrow0, wid_proc, sigzero );
			ref_row_ptr = row0_ptr;
			inrow0 += inrow_stride;	// dead code.
			break;
		}else{
			int32_t * row1_ptr= PTR_ADD_BYTES(int32_t, row0_ptr, ibuf_row_bytes);
			uint8_t const * inrow1 = inrow0 + inrow_stride;
			INTEGRAL_BUF_FUNC(inner_load2_loop)( row0_ptr, row1_ptr, ref_row_ptr,  inrow0, inrow1, wid_proc, sigzero );
			ref_row_ptr = row1_ptr;
			inrow0 += 2*inrow_stride;
			nrows_remain -= 2;
		}
	}
	// store the 'next' "producer_prevrow"
	vscan->producer_prevrow = ref_row_ptr;

	// everything else in here is for adding
	// edge padding. if there's no edge padding, we are done.
	if( wpad_flags == 0)
		return;

	// LEFT/RIGHT PADDING
	// apply to all rows which were just loaded

	if( (wpad_flags & (intbuf_PAD_L|intbuf_PAD_R) )!= 0){
		int infeas_cols = ibp->infeas_pad_wid;
		int win_w = ibp->window_wid;

		// set up some things for moving pointers up a row at a time:
		int32_t const * wrap_thr = PTR_ADD_BYTES ( const int32_t, vscan->buffer_base, ibuf_row_bytes);
		int32_t wrap_around_adj = ibuf_row_bytes - ibp->ibuf_total_bytes;
		// now, if ptr >= wrap_thr, subtract ibuf_row_bytes; else sub wrap_around_adj.

		if( (wpad_flags & intbuf_PAD_L)!= 0){
			int pad_l = ibp->wpad_left;
			int p = 1;			// col index to start
			// apply left padding to all of the rows we loaded.
			int32_t * p0 = ref_row_ptr; /// 'col 0' of the row we just loaded.
			if( infeas_cols > 0){	// some are infeasible
				int wid = min_i32(pad_l, infeas_cols);	// clear this many cols
				uint8_t * pclr = (uint8_t *)p0 - 128*wid;	// starting here
				for( int i = 0; i < nrows; i++){
					memset(pclr ,0, 128*wid);
					pclr -= ( pclr >= (uint8_t *) wrap_thr) ? ibuf_row_bytes: wrap_around_adj;
				}
				p = wid+1;		// this many are done already
			}
			// do remaining ops col by col.
			for (; p <= pad_l; p++){
				// p/(w-p)
				int scale = ibp->edgepad_scales_w[p-1];
				int32_t *p_pad = p0 - 32*p;
				for(int i =0; i < nrows; i++){
					int32_t const * p_ref = p_pad + win_w*32;
					for( int id = 0; id <32; id++){
						int32_t delt =  p_ref[id];
						p_pad[id] =  mul_frac_i32xi16( delt, scale);
					}
					p_pad = PTR_SUB_BYTES( int32_t, p_pad, (p_pad >= wrap_thr)?ibuf_row_bytes: wrap_around_adj);
				}
			}

		}
		if( (wpad_flags & intbuf_PAD_R)!= 0){
			int pad_r = ibp->wpad_right;
			int pmin = infeas_cols +1;
			// apply right padding to all of the rows we loaded.
			int32_t * pW0 = ref_row_ptr + 32*ibp->inshape.width; /// 'col [inwid]' of the row we just loaded.
			// do remaining ops col by col.
			for (int pi= 1; pi <= pad_r; pi++){
				// pi is the column to pad; p is the eff. index
				int p = max_i32(pi, pmin);

				// p/(w-p)
				int wmp = win_w - p;
				int scale = ibp->edgepad_scales_w[pi-1];
				int32_t const * pW = pW0;
				for(int i =0; i < nrows; i++){
					int32_t *p_pad = (int32_t*)pW + 32*pi;		// based on pi
					int32_t const * p_ref = pW - wmp*32;	// based on p.
					for( int id = 0; id <32; id++){
						int32_t allsum = pW[id];
						int32_t delt = allsum- p_ref[id];	// incr from [wid-(w-p)] to [wid]
						p_pad[id] = allsum - mul_frac_i32xi16(delt,scale);
					}
					pW = PTR_SUB_BYTES( int32_t, pW, (pW >= wrap_thr)?ibuf_row_bytes: wrap_around_adj);
				}
			}
		} // if right
	} // if left or right

	// TOP PADDING
	// do all of it, if we just loaded row 0.

	if( row_cnt == 0 && (wpad_flags & intbuf_PAD_T)!= 0 ){
		// apply top padding; we can do all of it; given the constraint
		// that the first load op must load 'initial_buf_load'. For
		// this operation, buffer wrapping not an issue; all the rows are
		// where we expect them,

		int pad_top = ibp->wpad_top;
		int win_h = ibp->window_ht;
		int p = 1;
		int inf_rows = ibp->infeas_pad_ht;
		// point to row 0 (start of the 'proper' integral buf, not col0)
		int32_t *row0  = PTR_ADD_BYTES( int32_t, vscan->buffer_base,
				ibp->ibuf_row0 * ibuf_row_bytes + ibp->ibuf_proper_offs);
		int32_t *padrow = row0;
		if( inf_rows > 0){	// need some 0 fill
			int prows = min_i32( pad_top, inf_rows);	// this many
			padrow = PTR_SUB_BYTES( int32_t, row0, ibuf_row_bytes*prows);	// move to start
			int zfill = ibuf_row_bytes *(prows-1) + ibp->ibuf_proper_row_bytes;
			memset( padrow, 0, zfill);
			p = prows+1;		// p starts here
		}
		int nwid= (unsigned)ibp->ibuf_proper_row_bytes/sizeof(int32_t);
		// find pointer to first ref row.
		int32_t const *refrow = PTR_ADD_BYTES( const int32_t,  row0, (win_h-p) * ibuf_row_bytes );
		for( ; p  <= pad_top; p++){
			padrow = PTR_SUB_BYTES( int32_t, padrow, ibuf_row_bytes);	// move up one row
			// find fraction p/(w-p)
			int scale = ibp->edgepad_scales_h[p-1];
			for(int i =0; i < nwid; i++){
				int32_t delt = refrow[i];
				padrow[i] = mul_frac_i32xi16(delt,scale);
			}
			refrow = PTR_SUB_BYTES( const int32_t, refrow, ibuf_row_bytes);	// move up one row
		}
	}
	//
	// Bottom padding
	// did we just load all of the rows? set up for bottom  padding if so.
	//
	if( new_nrows == vscan->input_height && (wpad_flags & intbuf_PAD_B)!= 0 ){
		int win_h = ibp->window_ht;
		int in_ht = vscan->input_height;
		// store position of last row (row in_ht) in integral buffer. This is ref_row_ptr,
		// but we want the start of the 'proper' row.
		int32_t * row_H = PTR_ADD_BYTES(int32_t, ref_row_ptr, ibp->ibuf_proper_offs - ibp->ibuf_col0_offs);
		vscan->row_H = row_H;
		// advance one row (with wrap) for first padding row
		int32_t * nextpad = PTR_ADD_BYTES(int32_t, row_H, ibuf_row_bytes );
		if( nextpad >= vscan->buffer_end)
			nextpad = PTR_SUB_BYTES( int32_t, nextpad, (int)vscan->ibuf_total_bytes);
		vscan->botpad_next = nextpad;
		// reference row:
		// back up from row H to row H-(win_h-1), but don't go below 0.
		int backup = min_i32( win_h-1, in_ht);
		int32_t * next_refrow = PTR_SUB_BYTES(int32_t, row_H, backup*ibuf_row_bytes );
		// check for wrap
		if( next_refrow < vscan->buffer_base)
			next_refrow = PTR_ADD_BYTES( int32_t, next_refrow, (int)vscan->ibuf_total_bytes);
		vscan->botpad_refrow = next_refrow;
	}
}



static inline void
INTEGRAL_BUF_FUNC(producer_load_rows_hvx)(
	struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp, int nrows )
{
	int ibuf_row_bytes = ibp->ibuf_row_bytes;
	int wpad_flags = ibp->wpad_flags;

	int32_t * ref_row_ptr = vscan->producer_prevrow;		// previous row is a reference row

	int row_cnt = vscan->producer_nrows;				// # of rows so far
	int prod_w0 = vscan->producer_w0;

	int sigzero = 0;
#ifdef NN_INTEGRAL_BUF_FOR_L2POOL
	sigzero = ibp->zero_code;
#endif

	int widvecs = vscan->producer_widvecs;

	// need input row pointer(s)
	uint8_t const * inrow0 = vscan->producer_in_row;
	int inrow_stride = ibp->tin.height_stride;
	vscan->producer_in_row = inrow0 + nrows * inrow_stride;
	int new_nrows = row_cnt + nrows;
	vscan->producer_nrows = new_nrows;

	int nrows_remain = nrows;
	// if the left padding is not a multiple of 4: we will process 1..3 'dummy'
	// width units at the start of each row; the loop will force them to zero
	// but we need to compensate our write pointer to the left by that many vectors,
	// so that the results are properly placed; then move it back after loading,
	int left_margin_adj = 128*(prod_w0&3);
	ref_row_ptr = PTR_SUB_BYTES( int32_t, ref_row_ptr, left_margin_adj);

	while( nrows_remain > 0){
		int32_t * row0_ptr= PTR_ADD_BYTES(int32_t, ref_row_ptr, ibuf_row_bytes);
		// wrap?
		if( row0_ptr >= vscan->buffer_end)
				row0_ptr = PTR_SUB_BYTES( int32_t, row0_ptr, vscan->ibuf_total_bytes);
		// now go do it...

		if( nrows_remain == 1){
			INTEGRAL_BUF_FUNC(inner_load1_loop_hvx)( row0_ptr, ref_row_ptr,  inrow0,
				widvecs, prod_w0, sigzero );
			ref_row_ptr = row0_ptr;
			inrow0 += inrow_stride;	// dead code.
			break;
		}else{
			int32_t * row1_ptr= PTR_ADD_BYTES(int32_t, row0_ptr, ibuf_row_bytes);
			uint8_t const * inrow1 = inrow0 + inrow_stride;
			INTEGRAL_BUF_FUNC(inner_load2_loop_hvx)( row0_ptr, row1_ptr, ref_row_ptr,  inrow0, inrow1,
				widvecs, prod_w0, sigzero );
			ref_row_ptr = row1_ptr;
			inrow0 += 2*inrow_stride;
			nrows_remain -= 2;
		}
	}

	ref_row_ptr = PTR_ADD_BYTES( int32_t, ref_row_ptr, left_margin_adj);
	// store the 'next' "producer_prevrow"
	vscan->producer_prevrow = ref_row_ptr;

	// everything else in here is for adding
	// edge padding. if there's no edge padding, we are done.
	if( wpad_flags == 0)
		return;

	// LEFT/RIGHT PADDING
	// apply to all rows which were just loaded

	if( (wpad_flags & (intbuf_PAD_L|intbuf_PAD_R) )!= 0){
		int infeas_cols = ibp->infeas_pad_wid;
		int win_w = ibp->window_wid;

		// set up some things for moving pointers up a row at a time:
		HVX_Vector const * wrap_thr = PTR_ADD_BYTES ( const HVX_Vector, vscan->buffer_base, ibuf_row_bytes);
		int32_t wrap_around_adj = ibuf_row_bytes - ibp->ibuf_total_bytes;
		// now, if ptr >= wrap_thr, subtract ibuf_row_bytes; else sub wrap_around_adj.

		if( (wpad_flags & intbuf_PAD_L)!= 0){
			int pad_l = ibp->wpad_left;
			// apply left padding to all of the rows we loaded.
			HVX_Vector * p0 = (HVX_Vector *)ref_row_ptr; /// 'col 0' of the row we just loaded.

			for( int p = 1; p <= pad_l; p++ ){
				HVX_Vector *p_pad = p0 - p;		// point to the column to fill...
				if( p <= infeas_cols ){			// zero fill
					for(int i = 0; i < nrows; i++ ){
						p_pad[0] = Q6_V_vzero();
						p_pad = PTR_SUB_BYTES( HVX_Vector, p_pad, (p_pad >= wrap_thr)?ibuf_row_bytes: wrap_around_adj);
					}
				}else{
					int scale = ibp->edgepad_scales_w[p-1];
					HVX_Vector vscale = q6op_Vh_vsplat_R( scale);	// need it in odd h lanes
					for(int i = 0; i < nrows; i++ ){
						p_pad[0] = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( p_pad[win_w], vscale );
						p_pad = PTR_SUB_BYTES( HVX_Vector, p_pad, (p_pad >= wrap_thr)?ibuf_row_bytes: wrap_around_adj);
					}
				}
			}
		}
		if( (wpad_flags & intbuf_PAD_R)!= 0){
			int pad_r = ibp->wpad_right;
			int in_wid = ibp->inshape.width;
			// apply right padding to all of the rows we loaded.
			HVX_Vector * pW0 = (HVX_Vector*)ref_row_ptr + in_wid; /// 'col [inwid]' of the row we just loaded.
			// do remaining ops col by col.
			for (int p = 1; p <= pad_r; p++){
				int scale = ibp->edgepad_scales_w[p-1];
				HVX_Vector vscale = q6op_Vh_vsplat_R( scale);	// need it in odd h lanes
				int ref_delt = - min_i32( in_wid, win_w - p);	// offset from pW to 'ref' row (<0)
				HVX_Vector * pW = pW0;
				for(  int i = 0; i < nrows; i++ ){
					HVX_Vector allsum = pW[0];
					HVX_Vector delt = Q6_Vw_vsub_VwVw( allsum, pW[ref_delt]);
					delt = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( delt, vscale );
					pW[p] = Q6_Vw_vsub_VwVw( allsum, delt );
					pW = PTR_SUB_BYTES( HVX_Vector, pW, (pW >= wrap_thr)?ibuf_row_bytes: wrap_around_adj);
				}
			}
		} // if right
	} // if left or right

	// TOP PADDING
	// do all of it, if we just loaded row 0.

	if( row_cnt == 0 && (wpad_flags & intbuf_PAD_T)!= 0 ){
		// apply top padding; we can do all of it; given the constraint
		// that the first load op must load 'initial_buf_load'. For
		// this operation, buffer wrapping not an issue; all the rows are
		// where we expect them,

		int nwid= (unsigned)ibp->ibuf_proper_row_bytes/sizeof(HVX_Vector);

		int pad_top = ibp->wpad_top;
		int win_h = ibp->window_ht;
		int inf_rows = ibp->infeas_pad_ht;

		// point to row 0 (start of the 'proper' integral buf, not col0)
		int32_t *row0  = PTR_ADD_BYTES( int32_t, vscan->buffer_base,
				ibp->ibuf_row0 * ibuf_row_bytes + ibp->ibuf_proper_offs);
		HVX_Vector * padrow = (HVX_Vector *)row0;

		// the reference for the first 'feasible' padding row
		HVX_Vector const *refrow = PTR_ADD_BYTES( const HVX_Vector,  row0, (win_h-inf_rows-1) * ibuf_row_bytes );

		// row[-p] = scale * row[win_h-p]
		// (where p <= infeas_h ,use 0)
		//
		for( int p = 1; p <= pad_top; p++ ){
			// move up one row
			padrow = PTR_SUB_BYTES( HVX_Vector, padrow, ibuf_row_bytes);	// move up one row
			if( p <= inf_rows ){			// 'infeasible' row, filled with 0
				for( int j = 0; j < nwid; j++ ){
					padrow[j] = Q6_V_vzero();
				}
			}else{
				int scale = ibp->edgepad_scales_h[p-1];
				HVX_Vector vscale = q6op_Vh_vsplat_R( scale);	// need it in odd h lanes
				for( int j = 0; j < nwid; j++ ){
					HVX_Vector delt = refrow[j];
					padrow[j] = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( delt, vscale );
				}
				refrow = PTR_SUB_BYTES( HVX_Vector, refrow, ibuf_row_bytes);	// move up one row
			}
		}
	}
	//
	// Bottom padding
	// did we just load all of the rows? set up for bottom  padding if so.
	//
	if( new_nrows == vscan->input_height && (wpad_flags & intbuf_PAD_B)!= 0 ){
		int win_h = ibp->window_ht;
		int in_ht = vscan->input_height;
		// store position of last row (row in_ht) in integral buffer. This is ref_row_ptr,
		// but we want the start of the 'proper' row.
		int32_t * row_H = PTR_ADD_BYTES(int32_t, ref_row_ptr, ibp->ibuf_proper_offs - ibp->ibuf_col0_offs);
		vscan->row_H = row_H;
		// advance one row (with wrap) for first padding row
		int32_t * nextpad = PTR_ADD_BYTES(int32_t, row_H, ibuf_row_bytes );
		if( nextpad >= vscan->buffer_end)
			nextpad = PTR_SUB_BYTES( int32_t, nextpad, (int)vscan->ibuf_total_bytes);
		vscan->botpad_next = nextpad;
		// reference row:
		// back up from row H to row H-(win_h-1), but don't go below 0.
		int backup = min_i32( win_h-1, in_ht);
		int32_t * next_refrow = PTR_SUB_BYTES(int32_t, row_H, backup*ibuf_row_bytes );
		// check for wrap
		if( next_refrow < vscan->buffer_base)
			next_refrow = PTR_ADD_BYTES( int32_t, next_refrow, (int)vscan->ibuf_total_bytes);
		vscan->botpad_refrow = next_refrow;
	}
}


//
// add 'maxrows' bottom padding rows, or all of them, whichever comes first.
//
// IMPORTANT: 'producer_nrows' is advanced by 'maxrows', to reflect the additional rows;
// even if the number of padding rows generated is less. In fact, if all padding has already
// been generated the function has no other effect than to add maxrows to producer_nrows.
//
// This makes 'producer_nrows' sometimes greater than the reality, but it
// prevents a possible infinite loop where the consumer is waiting for a specific
// count in order to finish, which is greater than the bottom padding can provide.
// That should never happen if the computations are done right, but at least this
// way it will finish.
//
static inline void
producer_add_bottom_padding( struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp, int maxrows )
{
	int rows_added = vscan->bot_pad_count;			// we have so far...
	int stop_at_rows = min_i32( rows_added+ maxrows,ibp->wpad_bottom );	//stop when this many
	int ibuf_row_bytes = ibp->ibuf_row_bytes;

	int infeas_h = ibp->infeas_pad_ht;

	// set up pointers
	int32_t const * refp = vscan->botpad_refrow;
	int32_t const * rowH = vscan->row_H;
	int32_t * wrtp = vscan->botpad_next;
	int nwid= (unsigned)ibp->ibuf_proper_row_bytes/sizeof(int32_t);

	while( rows_added < stop_at_rows){
		int scale = ibp->edgepad_scales_h[rows_added];
		for(int i = 0; i < nwid; i++ ){
			int32_t fullsum = rowH[i];
			int32_t delt = fullsum - refp[i];
			wrtp[i] = fullsum - mul_frac_i32xi16(delt,scale);
		}
		++rows_added;
		// bump the rows
		// the 'ref' row is only bumped when we have completed a 'feasible' row.
		// (in other cases it points to row 0 and we need it to stay there).
		//
		if( rows_added > infeas_h){
			refp = PTR_ADD_BYTES( const int32_t, refp, ibuf_row_bytes);
		}
		wrtp = PTR_ADD_BYTES( int32_t, wrtp, ibuf_row_bytes);
		if( wrtp >= vscan->buffer_end)
			wrtp = PTR_SUB_BYTES( int32_t, wrtp, vscan->ibuf_total_bytes);
		if( refp >= vscan->buffer_end)
			refp = PTR_SUB_BYTES( const int32_t , refp, vscan->ibuf_total_bytes);

	}
	vscan->botpad_refrow = (int32_t*) refp;
	vscan->botpad_next = wrtp;
	vscan->producer_nrows += maxrows;
	vscan->bot_pad_count = rows_added;
}


static inline void
producer_add_bottom_padding_hvx( struct integ_buff_vscan *vscan, struct integral_buffer_plan const * ibp, int maxrows )
{
	int rows_added = vscan->bot_pad_count;			// we have so far...
	int stop_at_rows = min_i32( rows_added+ maxrows,ibp->wpad_bottom );	//stop when this many
	int ibuf_row_bytes = ibp->ibuf_row_bytes;

	int infeas_h = ibp->infeas_pad_ht;

	// set up pointers
	HVX_Vector const * refp = (HVX_Vector const *)vscan->botpad_refrow;
	HVX_Vector const * rowH = (HVX_Vector const *)vscan->row_H;
	HVX_Vector * wrtp = (HVX_Vector  *)vscan->botpad_next;
	int nwid= (unsigned)ibp->ibuf_proper_row_bytes/sizeof(HVX_Vector);

	while( rows_added < stop_at_rows){
		int scale = ibp->edgepad_scales_h[rows_added];
		HVX_Vector vscale = q6op_Vh_vsplat_R( scale);	// need it in odd h lanes
		HVX_Vector const *p_rowH = rowH;
		HVX_Vector const *p_ref = refp;
		HVX_Vector  *p_wrt = wrtp;

		// do 2 at once...
		for(int i = 0; i < nwid>>1; i++ ){
			HVX_Vector fullsum0 = p_rowH[0];
			HVX_Vector delt0 = Q6_Vw_vsub_VwVw(fullsum0,p_ref[0]);
			HVX_Vector fullsum1 = p_rowH[1];
			HVX_Vector delt1 = Q6_Vw_vsub_VwVw(fullsum1,p_ref[1]);
			delt0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( delt0, vscale);
			delt1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( delt1, vscale);
			p_wrt[0] = Q6_Vw_vsub_VwVw(fullsum0, delt0);
			p_wrt[1] = Q6_Vw_vsub_VwVw(fullsum1, delt1);
			p_rowH +=2;
			p_ref+=2;
			p_wrt+=2;
		}
		if(nwid&1){
			HVX_Vector fullsum = p_rowH[0];
			HVX_Vector delt = Q6_Vw_vsub_VwVw(fullsum,p_ref[0]);
			delt = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( delt, vscale);
			p_wrt[0] = Q6_Vw_vsub_VwVw(fullsum, delt);
		}
		++rows_added;
		// bump the rows
		// the 'ref' row is only bumped when we have completed a 'feasible' row.
		// (in other cases it points to row 0 and we need it to stay there).
		//
		if( rows_added > infeas_h){
			refp = PTR_ADD_BYTES( const HVX_Vector, refp, ibuf_row_bytes);
		}
		wrtp = PTR_ADD_BYTES( HVX_Vector, wrtp, ibuf_row_bytes);
		if( wrtp >= (HVX_Vector *)vscan->buffer_end)
			wrtp = PTR_SUB_BYTES( HVX_Vector, wrtp, vscan->ibuf_total_bytes);
		if( refp >= (HVX_Vector const *)vscan->buffer_end)
			refp = PTR_SUB_BYTES( const HVX_Vector , refp, vscan->ibuf_total_bytes);

	}
	vscan->botpad_refrow = (int32_t*) refp;
	vscan->botpad_next = (int32_t*) wrtp;
	vscan->producer_nrows += maxrows;
	vscan->bot_pad_count = rows_added;
}


#endif //NN_INTEGRAL_BUFFER_H
