
/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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

#ifndef HVX_FUNNEL_REDUCE_H_
#define HVX_FUNNEL_REDUCE_H_

#include "hvx_inlines.h"

//
// This is a framework for efficiently doing a series of 'lateral' reductions of HVX vectors.
// For instance, suppose you have an array of rows x cols of int32's and you want the largest
// value in each row (so rows x int32 outputs). It is very efficient to reduce each row to 32 values
// vector 'max' operations, but less inefficient to do the horizontal reduction to one int32.
// The 'funnel reduction' uses a vector register to pipeline the lateral reduction over several
// rows of process, and uses another register to accumulate values from adjacent rows so that
// 32 values (a full vector) are stored at once. When all rows are processed, the pipeline must be correctly
// flushed. The functions in this header take care of the details of the funnel reduction and
// make it fairly easy to set up for an arbitrary reduction.

//
// The reduction operation must meet the following constraints
// (usually it will be a min, max, add, or 'or' operation):
//  * it must work in units of ELWID bytes (which must be a power of 2 1 .. 32
//  * it must be commutative and associative
//  * you must supply a vector function vec_red_func which takes two vectors, and
//    applies the reduction identically in each lane of ELWID bytes, producing a
//    vector result (most useful functions are declared in this header).
//  * you may also supply a function vec_output_func which can be used to modify
//    the result before it is stored; this accepts a single vector and returns
//    a vector. Usually this will be funnelreduce_output_nop() which just returns
//    its input. This function should perform the same operation on each lane of
//    ELWID bytes.
//
//
// **************
// * IMPORTANT  *
// **************
// In order to get good code generation:
//  - The "struct funnelreduce_state" variable must be declared as local variable and in the same
//    function as the calls to funnelreduce_init, funnelreduce_insert_value funnelreduce_flush_pipe
//    (or they must be effectively in the same function via inlining). See examples below.
//  - The "struct funnelreduce_state" variable must not be visible to other code in any way, and should go out
//    of scope after the call to funnelreduce_flush_pipe.
//  - The two function-pointer parameters to funnelreduce_init MUST be inline functions. Do not, for
//    instance, pass one of two different functions depending on a runtime variable - it will work but
//    code efficiency will be *catastrophically* lower.
//  - Normally the 'opaque' pointer to funnelreduce_init is NULL; this can be used to pass data to
//    the reduce or output functions, e.g. to obtain different behaviour in the same function.
//    This should be passed by a struct which is is declared in the same scope as the "struct funnelreduce_state"
//    variable and is likewise not visible to any code beyond the reduction process. See Example 4.
//
// When properly used, the function pointers passed to funnelreduce_init will not exist at runtime; instead, the
// functions will be directly expanded inside of funnelreduce_insert_value and funnelreduce_flush_pipe.
//
// Use of functions passed to funnelreduce_init:
//   static:
//     'reduce' and 'output' functions are both expanded once each in funnelreduce_insert_value and funnelreduce_flush_pipe
//   dynamic:
//     'reduce' function is invoked once per call to funnelreduce_insert_value, and log2(vec_bytes/ELWID) times in
//         funnelreduce_flush_pipe
//     'output' function is invoked once for each vector stored to output (including once or twice in funnelreduce_flush_pipe).
//
// The output 'end' pointer passed to funnelreduce_flush_pipe must be greater than the pointer passed to funnelreduce_init; it
// points to the first location which is *not* to be written in the output buffer.
// The difference (in bytes) must be at most ELWID*n, and at least ELWID*n-(n-1) bytes, where 'n' is the number of
// calls to funnelreduce_insert_value. Usually the difference wil be ELWID*n, unless you are reducing more than one
// row at a time; example (2) shows such a case where ELWID=4 and the difference is sometimes ELWID*n and sometimes
// ELWID*n-2 (the latter when 'rows' is odd).
//
/* ***** EXAMPLES ****
// Example (1)
// find the max i16 value in rows x cols;  data is aligned and cols is a multiple of 64
//
void example_max16_in_rows( int16_t const * data, int rows, int cols, int16_t * outp )
{
	struct funnelreduce_state frstate;
	funnelreduce_init( &frstate, outp, 2, funnelreduce_max_Vh, funnelreduce_output_nop ,NULL);
	int vecloops = (cols/64u)-1;
	for( int irow = 0; irow < rows; irow++){
		HVX_Vector const * rowp = (HVX_Vector const *)( data + irow*cols);
		HVX_Vector mx = *rowp++;
		for( int j = 0; j < vecloops; j++){
			mx = Q6_Vh_vmax_VhVh( mx, *rowp++);
		}
		// 'mx' is 64 values which, after horizontal reduction, gives the max of the row.
		funnelreduce_insert_value( &frstate, mx);
	}
	// flush the pipe
	funnelreduce_flush_pipe( & frstate, outp + rows);
}

// Example (2)
// Same as before, except
//   - any value of col supported, and no alignment requirements
//   - we do two rows at once to reduce the outer-loop overhead.
// Because we are doing 2 rows at once we use ELWID=4 (each 'element' is
// two int16's, one for each of the rows.
// Note: when rows is odd the function will only store the first 2 bytes of the
// final pair.
// Processing two rows at once means the per-row overhead is done half as often.
// (note, the code size of example 1 is mostly in the 'flush_pipe'; the extra code to
// to do two rows at once is quite minor).
//
void example2_max16_in_rows( int16_t const * data, int rows, int cols, int16_t * outp )
{
	struct funnelreduce_state frstate;
	funnelreduce_init( &frstate, outp, 4, funnelreduce_max_Vh, funnelreduce_output_nop ,NULL);
	int vec_per_row = (cols-1)/64u;	// # of inner loops per row
	HVX_VectorPred last_mask = q6op_Q_vsetq2_R( cols*2);

	for( int irow = 0; irow < rows; irow+=2){
		// get two row pointers.
		// if we are on the last 'odd' row, read it twice.
		HVX_Vector const * rowp0 = (HVX_Vector const*)( data + irow*cols);
		HVX_Vector const * rowp1 = (irow < rows-1)? (HVX_Vector const*)( data + (irow+1)*cols): rowp0;
		HVX_Vector vin0 = q6op_V_vldu_A( rowp0);
		HVX_Vector vin1 = q6op_V_vldu_A( rowp1);
		rowp0 += 64;
		rowp1 += 64;

		HVX_Vector mx0 = q6op_Vh_vsplat_R( 0x8000);		// init max-so-far
		HVX_Vector mx1 = q6op_Vh_vsplat_R( 0x8000);
		for( int j= 0; j < vec_per_row; j++){
			mx0 = Q6_Vh_vmax_VhVh( mx0, vin0);
			mx1 = Q6_Vh_vmax_VhVh( mx1, vin1);
			vin0 = q6op_V_vldu_A( rowp0);
			vin1 = q6op_V_vldu_A( rowp1);
			rowp0 += 64;
			rowp1 += 64;
		}
		// For the last vector column: avoid including the 'extra' lanes
		vin0 = Q6_V_vmux_QVV( last_mask, vin0, mx0);
		vin1 = Q6_V_vmux_QVV( last_mask, vin1, mx1);
		mx0 = Q6_Vh_vmax_VhVh( mx0, vin0);
		mx1 = Q6_Vh_vmax_VhVh( mx1, vin1);
		// now we have the mx for even row and odd row.
		HVX_VectorPair vtx = Q6_Wh_vshuffoe_VhVh( mx1, mx0);
		// same data, but the 'mx0' values are in even lanes and
		// mx1 values in odd lanes
		HVX_Vector maxall = Q6_Vh_vmax_VhVh( Q6_V_hi_W(vtx), Q6_V_lo_W(vtx));
		// now we need to reduce the 32 { min0, min1) pairs across..
		funnelreduce_insert_value( &frstate, maxall);
	}
	// flush the pipe.
	// The 'outp+rows' value is used to decide which is the last value
	// to store, so even if rows is odd (and we've inserted an extra value)
	// the total # of int16's stored is limited to 'rows'.
	funnelreduce_flush_pipe( & frstate, outp + rows);
}

// Example (3)
// Find the min and max i16 on each row;
// results are stored interleaved: outp[2*i] is min for row i, outp[2*i+1] is max
//   - any value of col supported, and no alignment requirements
//
// In order to reduce min and max together, we put min in even lanes and ~max
// (one's complement) in odd lanes, and reduce that using min. The ~max needs
// corecting to max at the end,  which needs an output function. This function
// will be used just prior to storing any value.
//
static inline HVX_Vector __attribute__((always_inline))
example2_out_func( HVX_Vector vin, void const * unused){
	return Q6_V_vxor_VV( vin, Q6_V_vsplat_R(0xFFFF0000));
}
//
void example3_minmax16_in_rows( int16_t const * data, int rows, int cols, int16_t * outp )
{
	struct funnelreduce_state frstate;
	// reduce in units of 4 bytes, using 16-bit 'min', and using s special output function.
	funnelreduce_init( &frstate, outp, 4, funnelreduce_min_Vh, example2_out_func ,NULL);
	int vec_per_row = (cols-1)/64u;	// # of inner loops per row
	HVX_VectorPred last_mask = q6op_Q_vsetq2_R( cols*2);

	for( int irow = 0; irow < rows; irow++){
		// get row pointers.
		// if we are on the last 'odd' row, read it twice.
		HVX_Vector const * rowp = (HVX_Vector const*)( data + irow*cols);
		HVX_Vector vin = q6op_V_vldu_A( rowp);
		rowp += 64;

		HVX_Vector mx = q6op_Vh_vsplat_R( 0x8000);		// init max-so-far
		HVX_Vector mn = q6op_Vh_vsplat_R( 0x7fff);		// min-so-far
		for( int j= 0; j < vec_per_row; j++){
			mn = Q6_Vh_vmin_VhVh( mn, vin);
			mx = Q6_Vh_vmax_VhVh( mx, vin);
			vin = q6op_V_vldu_A( rowp);
			rowp += 64;
		}
		// For the last vector column: avoid including the 'extra' lanes
		mn = Q6_Vh_vmin_VhVh( mn, Q6_V_vmux_QVV( last_mask, vin, mn));
		mx = Q6_Vh_vmax_VhVh( mx, Q6_V_vmux_QVV( last_mask, vin, mx));

		// now, take 1's complement of max, and interleave with min...
		HVX_VectorPair vtx = Q6_Wh_vshuffoe_VhVh( Q6_V_vnot_V(mx), mn);
		// Now 'mn' values are in even lanes and ~max in odd
		HVX_Vector minall = Q6_Vh_vmin_VhVh( Q6_V_hi_W(vtx), Q6_V_lo_W(vtx));
		// now we need to reduce the 32 { min0, min1) pairs across..
		funnelreduce_insert_value( &frstate, minall);
	}
	// flush the pipe (2 values per row)
	funnelreduce_flush_pipe( & frstate, outp + 2*rows);
}

// Example (4)
// This shows the use of 'opaque' to pass data to the output function.
// This finds the min (or max, according to find_max parameter) of each row
// of int32's. The rows must be a multiple of 32 wide, and vector aligned.
// This works by doing a 'min' reduction, but all incoming data, and all output
// results, are xor'd with a 'constant' vector: that vector is set to all 0's
// to get a 'min' function, and to all 0xFFFFFFFF to get a 'max' function.
//
// redconsts.xor_value will be kept in a vector register if 'redconst' is declared
// locally as shown, and is not visible to any other code.
//

struct example_4_const {
    HVX_Vector xor_value;
};
// output function xor's the data with 'xor_value'
static inline HVX_Vector __attribute__((always_inline))
example4_out_func( HVX_Vector vin, void const * ptr){
    HVX_Vector xor_value = ((struct example_4_const const*)ptr)->xor_value;
	return Q6_V_vxor_VV( vin, xor_value);
}

void example4_max_or_min_32_in_rows( int32_t const * data, int rows, int cols, int16_t * outp, int find_max )
{
    // create a struct example_4_data containing all 0's for min, all 1's for max
    struct example_4_const redconsts;
    redconsts.xor_value = Q6_V_vsplat_R( find_max ? 0 : -1 );

	struct funnelreduce_state frstate;
    // reduce using min, but we invert all incoming words for max.
	funnelreduce_init( &frstate, outp, 4, funnelreduce_min_Vh, example4_out_func , & redconsts);

	int vecloops = (cols/32u)-1;
	for( int irow = 0; irow < rows; irow++){
		HVX_Vector const * rowp = (HVX_Vector const *)( data + irow*cols);
		HVX_Vector mn = Q6_V_vxor_VV(*rowp++, redconsts.xor_value);
		for( int j = 0; j < vecloops; j++){
            HVX_Vector vnew = Q6_V_vxor_VV(*rowp++, redconsts.xor_value);
			mn = Q6_Vh_vmin_VhVh( mn, vnew);
		}
		// 'mn' is 32 values which, after horizontal reduction, gives the min of the row.
        // for 'max' function, reduction using 'vmin' will give the 1's complement of the max.
		funnelreduce_insert_value( &frstate, mn);
	}
	// flush the pipe
	funnelreduce_flush_pipe( & frstate, outp + rows);
}

****(end of examples)**** */

// struct which defines the variables used in the loop
// fields [*] will be constant at compile time (assuming ELWID is)
//
struct funnelreduce_state {
	int elwid;			// element width [*]
	int pervec;			// elemehts/vector [*]
	int funnel_pipe;	// pipeline depth of funnel [*]
	void const * opaque;	// 'user' poiner [*]
	HVX_Vector (*vec_red_funcp)( HVX_Vector, HVX_Vector, void const *); //[*]
	HVX_Vector (*vec_output_funcp)( HVX_Vector, void const *); //[*[
	uint8_t * outp;
	int shifter_room;	// # of shifts until shifter is full
	HVX_Vector funnel;
	HVX_Vector shifter;
};


// The only output function provided in the header is 'nop'

static inline HVX_Vector __attribute__((always_inline))
funnelreduce_output_nop( HVX_Vector vin, void const * opaque){
	return vin;
}

// various reductions. We need to provide wrapper functions since
// it's not possible to take the address of Q6_Vw_vadd_VwVw
// (and also we need to have the 'opaque' parameter)

// 'or' reduction
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_bitwise_or( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_V_vor_VV(vin0,vin1);
}

// 32-bit reductions: add, max, min

static inline HVX_Vector __attribute__((always_inline))
funnelreduce_sum_Vw( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vw_vadd_VwVw(vin0,vin1);
}
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_max_Vw( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vw_vmax_VwVw(vin0,vin1);
}
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_min_Vw( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vw_vmin_VwVw(vin0,vin1);
}

// 16-bit reductions : add, max, min (signed and unsigned)
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_sum_Vh( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vh_vadd_VhVh(vin0,vin1);
}

static inline HVX_Vector __attribute__((always_inline))
funnelreduce_max_Vh( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vh_vmax_VhVh(vin0,vin1);
}
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_min_Vh( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vh_vmin_VhVh(vin0,vin1);
}

static inline HVX_Vector __attribute__((always_inline))
funnelreduce_max_Vuh( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vuh_vmax_VuhVuh(vin0,vin1);
}
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_min_Vuh( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vuh_vmin_VuhVuh(vin0,vin1);
}


// 8-bit reductions (unsigned):  max, min
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_max_Vub( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vub_vmax_VubVub(vin0,vin1);
}
static inline HVX_Vector __attribute__((always_inline))
funnelreduce_min_Vub( HVX_Vector vin0, HVX_Vector vin1, void const * opaque){
	return Q6_Vub_vmin_VubVub(vin0,vin1);
}

static inline void __attribute__((always_inline))
funnelreduce_init( struct funnelreduce_state *statep,
		void * outp,
		int ELWID,		// element width (must be a constant power of 2)
		HVX_Vector (*vec_red_func)( HVX_Vector, HVX_Vector, void const *),
		HVX_Vector (*vec_output_func)( HVX_Vector, void const *),
		void const * opaque)
{
	int el_per_v = sizeof(HVX_Vector)/ELWID;
	statep->pervec = el_per_v;
	statep->elwid = ELWID;
	statep->funnel_pipe = __builtin_ctz( el_per_v);
	statep->opaque = opaque;
	statep->vec_red_funcp = vec_red_func;
	statep->vec_output_funcp = vec_output_func;
	statep->outp = (void*)outp;
	statep->shifter_room = el_per_v + __builtin_ctz( el_per_v);
}

static inline void __attribute__((always_inline))
funnelreduce_insert_value( struct funnelreduce_state *statep, HVX_Vector vin )
{
	HVX_VectorPair shuffled = Q6_W_vdeal_VVR( vin, statep->funnel, -statep->elwid);
	statep->funnel = (*statep->vec_red_funcp)( Q6_V_hi_W(shuffled), Q6_V_lo_W(shuffled), statep->opaque);
	HVX_Vector newshift = Q6_V_valign_VVR(Q6_V_hi_W(shuffled), statep->shifter, statep->elwid );
	statep->shifter = newshift;
	int new_room = statep->shifter_room-1;
	if( new_room <= 0 ){		// store the shifter out
		HVX_Vector outv =  (*statep->vec_output_funcp)( newshift, statep->opaque);
		q6op_vstu_AV( (HVX_Vector*)statep->outp, outv);
		statep->outp += 128;
		new_room = statep->pervec;
	}
	statep->shifter_room = new_room;
}
static inline void __attribute__((always_inline))
funnelreduce_flush_pipe( struct funnelreduce_state *statep, void * outp_end )
{
	int in_funnel = statep->funnel_pipe;
	int sh_room = statep->shifter_room;
	uint8_t * outp = statep->outp;
	HVX_Vector shifter = statep->shifter;
	while( outp < (uint8_t*)outp_end ){
		if( in_funnel > 0 ){
			HVX_VectorPair shuffled = Q6_W_vdeal_VVR( statep->funnel, statep->funnel, -statep->elwid);
			statep->funnel = (*statep->vec_red_funcp)( Q6_V_hi_W(shuffled), Q6_V_lo_W(shuffled), statep->opaque);
			shifter = Q6_V_valign_VVR(Q6_V_hi_W(shuffled), shifter, statep->elwid );
			in_funnel--;
			sh_room--;
		}else{
			shifter = Q6_V_vror_VR( shifter, sh_room * statep->elwid);
			sh_room = 0;
		}
		if( sh_room <=0){		// store something
			uint8_t *outp_next = outp + sizeof(HVX_Vector);
			HVX_Vector outv =  (*statep->vec_output_funcp)( shifter, statep->opaque);
			if( outp_next <= (uint8_t*)outp_end){
				// full vector store; may or may not be the last
				q6op_vstu_AV( (HVX_Vector*)outp, outv);
				outp = outp_next;
				sh_room = statep->pervec;
			}else{
				// partial store - always last
				int nbytes = (uint8_t *)outp_end - outp;
				q6op_vstu_variable_ARV( (HVX_Vector*)outp, nbytes,  outv);
				return;
			}
		}
	}
}


#endif /* HVX_FUNNEL_REDUCE_H_ */
