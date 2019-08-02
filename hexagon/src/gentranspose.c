/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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
//
// General Transpose facility
// ================================
// This module analyzes and efficiently implements arbitrary 
// transpose operations on 'flat' tensors. Since depth to/from space
// and batch to/from space can also be expressed as transpose, in terms
// of their actual effect on the data, it can
// be used for that (so any hvx routines written to handle certain classes
// could end being applies to those other nodes too).
//
//  This starts off by making a table like this, from inner to outer
//  dims of the output:
//
//    n      out_stride       in_stride
//   elbytes     1                1			<- represents copying element
//   dout     elbytes*dout       ...
//   wout     elbytes*dout*wout  ...
//   hout      ...               ...
//   bout      ...               ...
//  
//  .. in which each row shows the size, and output and input strides in bytes.
//  The first row always is (elbytes, 1,1 ) even if elbytes=1 and represents the
//  "loop" to copy 1 element; in other rows the out_stride is the product of the
// previous row's n and out_stride (i.e. they are contiguous in output).
//
//  Then, the table is reduced, by:
//    (a) any row (other than first) which has n=1 is removed.
//    (b) if any row with n=n1 is contiguous in input with thr next row having
//       n=n2 (i.e. the next row's in_stride = current_row.in_stride*n1), the
//       two are merged to one: the new row has n = n1*n2 and has the same strides as
//       the first of the two rows.
//
//
//  At this point, the table is a canonical representation of the work to be done.
//  The first row will always be {n,1,1), where n is the # of bytes to be moved in
//  the smallest unit. The table will have one row, or at least 3; if it has one
//  row, it's a simple copy of n bytes, this case is not further considered.
//
//  If the last row has equal input and output strides, it is a common outer
//  dimension, well suited for slicing the work across threads; it is removed
//  from the table at this point, and set aside as n_outer (if not, then n_outer=1).
//
//  We can then start looking for hvx routines that are capable of the specific case.
// 
//  For a  general solution, the first 3 rows always look like this:
//     D   1     1
//     W   D     wsi
//     H  W*D    hsi
//  We can use a 2-vector memcpy to handle 'W' rows of 'D', and iterate that operation H
//   times (and into any outer loops). if H is larger than W, we can swap those rows
//   to make the process more efficient (so it will copy H rows of D, W times).
/// This is a good approach when D is large enough, but for smaller D it will be
//  inefficient. If D is either 1,2,4, or 8, we have specific routines to support each
//  of those values in a '2d-strided' copy;  both dimensions having arbitrary strides.
//  In other cases - where D is not one of those values, but is a small multiple, e.g. 
//  D = 3*2 or 5*1; we create an extra row; so if the first row is { 6,1,1} we change
//  it to {2,1,1} and add an extra row {3,2,2} representing a loop to copy 3 2-byte values;
//  This extra row may be moved down the table, assuming it's smaller than W and H, so that
//  it doesn't add too much loop overhead. And then the 16-bit copy loop is used to handle W,H
//   and this is iterated 3 times according to the new loop level.
// 
// In any case, any chosen strategy handles a certain number of rows of the table and we need to
// iterate over the rest - including the removed 'outer' dim.
// If no outer loop was removed, and we have table row(s) which are mot handled by the strategy,
// we can remove one as the outer dim for thread splitting. The best choice is one with the smaller
// source stride, to keep the jobs grouped in source addresses (and scattered in dest) if possible.
//
// A batch-to-space (or from-space) operation can be expressed as a transpose over (up to) 6
// dimensions; however the 'depth' dimension will always combine with the top row of the table,
// and the batch dimension will always become the outer dimension. To support this, we allow
// the table to be up to 7 rows when initially built, but when the reduction is complete (and
// outer 'batch' loop removed) the table can't exceed 5 rows. This sets the upper limit on the
// number of loops needed to execute the strategy.

#include <stdlib.h>
#include <stdint.h>

#include "nn_graph.h"
#include "nn_gentranspose.h"
#include "nn_bulk_transpose.h"
#include "hvx_inlines.h"
#include "quantize.h"

#define TRANSPOSE_NUM_THREADS 2

// all of these need to be capable of being called with NULL nn
//  (handle all in the calling hvx thread)
// or non-null nn (launch vector threads, as needed.
static int
__attribute__((unused,noinline))
transpose_execute_NULL( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
				 uint8_t * output, uint8_t const *input);
static int
__attribute__((unused,noinline))
transpose_execute_MEMCPY( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
				 uint8_t * output, uint8_t const *input);
static int
__attribute__((unused,noinline))
transpose_execute_3DMEMCPY_GEN( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
				 uint8_t * output, uint8_t const *input);
static int
__attribute__((unused,noinline))
transpose_execute_SCALAR( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
				 uint8_t * output, uint8_t const *input);

static int
__attribute__((unused,noinline))
transpose_execute_HVXFUNC( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
				 uint8_t * output, uint8_t const *input);

// run threads with HVX code for specific cases (used by  transpose_execute_HVXFUNC)
static void transpose_thread_k64_deal6( struct nn_graph *nn , void *rstpv );
static void transpose_thread_A8xB8x16( struct nn_graph *nn , void *rstpv );
static void transpose_thread_1vec_shuffle4( struct nn_graph *nn , void *rstpv );
static void transpose_thread_deal2( struct nn_graph *nn , void *rstpv );

static void transpose_thread_bulktranspose( struct nn_graph *nn , void *rstpv );

typedef void (*nn_stride_scalar_copy_fp)(uint8_t * outp, uint8_t const *inp, int h, int w, int hsi, int wsi, int hso, int wso);


// swap rows i0, i1 of table
static inline void swap_rows( struct nn_transpose_desc * tdp, int i0, int i1 )
{
	struct nn_transpose_tabrow * r0 = &tdp->table[i0];
	struct nn_transpose_tabrow * r1 = &tdp->table[i1];

	struct nn_transpose_tabrow t0 = *r0;
	struct nn_transpose_tabrow t1 = *r1;
	*r0 = t1;
	*r1 = t0;
}
// if i0 < i1, move row i1 to i0, and rows i0...i1-1 are shifted up to i0+1 ... i1
static inline void roll_rows( struct nn_transpose_desc * tdp, int i0, int i1 )
{
	struct nn_transpose_tabrow * r0 = &tdp->table[i0];
	struct nn_transpose_tabrow * r1 = &tdp->table[i1];
	if( r1 > r0){
		struct nn_transpose_tabrow tmp = *r1;
		do{
			r1--;
			r1[1] = r1[0];
		}while( r1 > r0);
		*r1 = tmp;
	}
}

// find first table row,  row0 <= i < nrows, with instride = needed_stride.
// return row index or -1 if not found.
static inline int
find_row_with_instride( struct nn_transpose_desc * tdp, int needed_stride, int row0, int nrows)
{
	for( int i =row0; i < nrows; i++){
		if( tdp->table[i].in_stride == needed_stride ) return i;
	}
	return -1;
}

// this does 2 things:
// (1) check that the permutation is valid
// (2) generate the output shape.

int nn_transpose_check(int32_t const* perm_arr, int permn,
		                   struct shape const *in_shape,
		                   struct shape * out_shape )
{
	if( permn < 0 || permn > 4) return 1;

	*out_shape = *in_shape;		// start by cloning it
	unsigned dimset = 0;
	for( int i =0 ; i < permn; i++){
		int k = perm_arr[i];
		if( (unsigned)k >= (unsigned)permn) return -1;		// is ok
		unsigned m = 1 << k;
		if( dimset & m) return -1;		// seen it before
		dimset |= m;
		out_shape->dimension[4-permn+i] = in_shape->dimension[4-permn+k];
	}
	return 0;
}
static inline int is_power2( unsigned x){
	return Q6_R_popcount_P(x)==1;
}
//
// analyze.
// We assume
// (1) the perm_arr and permn have been checked by nn_transpose_check;
// (2) ndims is >= 1, <= NN_TRANSPOSE_MAX_ORIG_DIMS and all of the dims are >=1
//    (some of this is checked anyway for store safety).
//

int nn_transpose_analyze_direct( struct nn_transpose_desc * tdp,
                  int elementsize,
                  int32_t const*perm_arr, int permn,
                  uint32_t const *dims,		// [ndim], outer to inner
                  int ndim )
{
	tdp->execute_fp = transpose_execute_NULL;
	tdp->is_batch_scalable = 0;

	// start by ignoring any outer dims == 1.
	// but not if they might be involved in the permutation.
	
	if( permn < 0 || permn > ndim) return -1;	// => ndim >=0
	while( ndim > permn && dims[0] <=1){
		++dims; --ndim;
	}
	// maybe ndims = 0 now (a scalar).
	if( ndim >NN_TRANSPOSE_MAX_ORIG_DIMS ) return -1;		// protect memory
	int nrows = ndim+1;		// dimensions in the table

	tdp->buffer_needed = 0;
	//
	// start filling in the table
	//
	tdp->table[0].n = elementsize;		// row 0 represents copying an element.
	tdp->table[0].in_stride = 1;
	tdp->table[0].out_stride =1;

	unsigned full_tensor_size = elementsize;

	if( ndim > 0 ){
		// start by filling in 'tmp' column of table
		// according to the input strides, from inner to outer.
		// we also fill in n, instride for
		// the sake of outer dims which won't be covered by the permutation.
		unsigned stride = elementsize;
		for( int i = 0; i < ndim;i++){
			struct nn_transpose_tabrow * rowp = &tdp->table[i+1];
			unsigned n = dims[ndim-1-i];		// read these backwards, inner to outer.
			rowp->n = n;
			rowp->in_stride = rowp->out_stride = rowp->tmp= stride;
			stride *= n;
		}
		full_tensor_size = stride;
		// now, fill out the table  the permutation. It will
		// be in *output* order; we can only read ->tmp while doing
		// this (to get in strides).
		stride = elementsize;
		for( int i = 0; i < permn;i++){
			struct nn_transpose_tabrow * rowp = &tdp->table[i+1];
			int k = perm_arr[permn-1-i];			// traverse inner to outer
			unsigned n = dims[ndim-permn+k];	// size of corresponding source dim
			rowp->out_stride = stride;
			rowp->n  = n;
			rowp->in_stride = tdp->table[permn-k].tmp;	// in stride of source dim.
			stride *= n;
		}
		// ok, now the table has its nrows = 1+ndims rows.
		// can simplify it as follows:
		// - remove any row where n = 1  (except first row)
		// - If a row has (n1, outstride, instride) is followed by a row with (n2, outstride*n1, instride*n1)
		//   then we can change n1 to n1*n2, and delete the second row.
		//   (we only need to check instride, outstride is always contiguous)
		//
		//   Start at the end and work back.
		//
		// first delete any trailing with n=1.
		while( nrows> 1 && tdp->table[nrows-1].n == 1 )
			--nrows;
		// now go through them starting at the second last.
		for( int irow = nrows-2; irow >= 0 ; irow--){
			struct nn_transpose_tabrow * rowp = &tdp->table[irow];
			// there are are always >=2 rows starting at rowp.
			unsigned n = rowp[0].n;
			if( n == 1 && irow >0){		//delete this row
				rowp[0] = rowp[1];	// replace with next outer.
				// fall through to row deletion logic
			}else if( rowp[1].in_stride == n*rowp[0].in_stride ){
				// can combine these rows. This is ok to do on row 0.
				rowp[0].n = n*rowp[1].n;
				// fall through to row deletion logic
			}else{
				continue;	// nothing to do here
			}
			// we need to delete rowp[1], and move any following rows up.
			nrows--;
			int nmove = nrows-1-irow;		// # of rows to move.
			if( nmove > 0){
				for( int k = 0; k < nmove;k++)
					rowp[1+k] = rowp[2+k];
			}
		}
	}
	// now, we should have a table of nrows >=1, and the first row will be { k, 1 , 1 }
	// where k is a multiple of elementsize. None of the rows have n=1 except maybe row 0.

	// try to remove an outer 'batch' dimension, if we can.
	if( nrows >= 2 && tdp->table[nrows-1].in_stride == tdp->table[nrows-1].out_stride ){
		unsigned n_outer = tdp->table[nrows-1].n;
		tdp->n_outer = n_outer;
		tdp->outer_in_stride =tdp->outer_out_stride = tdp->outer_size =  tdp->table[nrows-1].out_stride;
		--nrows;
	}else{
		tdp->n_outer = 1;
		tdp->outer_in_stride = tdp->outer_out_stride = 0;
		tdp->outer_size= full_tensor_size;
	}
	// it should be impossible to have exactly 2 rows under any circumstances, because even
	// a simple full transpose will reduce to 3 rows (see comment a little ways down).

	if( nrows == 2) return -__LINE__;

	tdp->n_dims = nrows;
	// not allowed to have more than 5 rows at this point (this is enough to support batch/space operations; that's
	// a 6-dim transpose, but the 'D' dim will crush into the element row, and there will be an outer 'B' dim that
	// we just removed).
	//
	if( nrows >5) return -1;

	// now, choose a strategy. we can also reorder the rows in the table --  provided we don't move
	// the first one -- to e.g. give higher counts to outer loops. We can also insert dims to make some
	// cases work better.

	if( nrows== 1){				// easy. memcpy of outer_size.
		tdp->execute_fp = transpose_execute_MEMCPY;
		return 0;
	}
	// fill in any extra rows of the 5  with [1,0,0]
	// (less work for 'execute' to do).
	for( int i = nrows; i < 5; i++ ){
		struct nn_transpose_tabrow * rowp = &tdp->table[i];
		rowp->n = 1;
		rowp->out_stride = rowp->in_stride = 0;
	}
	//
	// First 3 rows will now look like this:
	//      d      1         1
	//      w      d       inst1
	//      h     w*d      inst2
	// .. where 'd' could be fairly large, or could be 1 byte
	// this can be done as a 2d memcpy w rows of d, out_stride= d, in_stride = inst1
	// and then repeat over the h dimension and any others.
	// we also have a set of 2d scalar copies, which can do [wxh] strided copy of d-size elements,
	// if d = 1,2,4 or 8. If d is something like 12, we can use the d=4 scalar copy and
	// add another loop level for 3 iters; this can be nested toward the outside.
	//
	/*for(int i = 0; i < nrows; i++ ){
		printf("  %4d: %5d %5d %5d\n",
			i, (int)tdp->table[i].n,(int)tdp->table[i].out_stride, (int)tdp->table[i].in_stride);
	}*/
	// here is where we check for special cases that HVX stuff is coded for.
	if( nrows == 3){
		int dsize= tdp->table[0].n;
		int w = tdp->table[1].n;
		int h = tdp->table[2].n;
		if( is_power2(dsize)){
			// [b,A*8,B*8,16] -> [b,B*8,A*8,d]
			// d is a power of 2 <=64; h*d is a multiple of 64.
			if( dsize == 16 && ((h|w)&7) ==0 ){
				tdp->execute_fp = transpose_execute_HVXFUNC;
				tdp->funcp = transpose_thread_A8xB8x16;
				return 0;
			}
			// [b,w,6,d] -> [b,6,w,d]
			// d is a power of 2 <=64; w*d is a multiple of 64.
			if( dsize <= 64 && h == 6 && ((w*dsize)&63)==0 ){
				tdp->execute_fp = transpose_execute_HVXFUNC;
				tdp->funcp = transpose_thread_k64_deal6;
				return 0;
			}
			// [b,4,h,d] -> [b,h,4,d]
			// where d is a power of 2 <= 64,
			// and h*d <= 128.
			// (for [b,4,6,16] cases which are eligible for this or _k64_deal6, that one
			// seems to be faster)
			if(dsize <= 64 && w == 4 && h*dsize <= 128){
				tdp->execute_fp = transpose_execute_HVXFUNC;
				tdp->funcp = transpose_thread_1vec_shuffle4;
				return 0;
			}
			// [b,w,2,d] -> [b,2,w,d]
			// d is a power of 2 <=64; w*d is >=128
			//
			if(dsize <= 64 && h == 2 && w*dsize >= 128){
				tdp->execute_fp = transpose_execute_HVXFUNC;
				tdp->funcp = transpose_thread_deal2;
				return 0;
			}
		}
	}

	{
		int dsize= tdp->table[0].n;
		int w = tdp->table[1].n;
		int h = tdp->table[2].n;
		if( tdp->table[1].out_stride != dsize || tdp->table[2].out_stride != w*dsize )
			return - __LINE__;

		// bulk transpose?
		// requires: d is 1,2,4,8, or 16
		// prefers:  d*w >= 48
		//           d*eff_h >= 48  (where eff_h is the n from the row with in_stride = d)
		//  tdp->outer_size >= 2K
		//
		if(  dsize <=16  &&  is_power2(dsize) && dsize*w >= 48  && tdp->outer_size >= 2048){
			// find the row, in range 2..nrows-1, which has in_stride ==d
			int xrow = find_row_with_instride( tdp, dsize, 2,nrows);
			if( xrow < 0) return -__LINE__;
			if( dsize * tdp->table[xrow].n >= 48){
				//printf("going with dsize =%d, w = %d, h = %d, outer_size = %d, nrows = %d\n", dsize, w, (int)tdp->table[xrow].n,
				//		tdp->outer_size, nrows);
				// OK, decision is made, we will do this..
				// if xrow is not row 2, roll it down to row 2.
				if( xrow > 2){
					roll_rows( tdp, 2, xrow);
				}
				// now: if the n_outer is 1, and nrows >= 4, and outer_size >= 32768,
				// we can drop the last loop and make it an 'outer' loop (with different
				// input and output strides; but bulk transpose can handle that). This will
				// allow it to be done across multiple threads.
				//
				if( tdp->n_outer ==1 && nrows >= 4 && tdp->outer_size >= 32768){
					// if nrows = 5, we can drop either of the two extra;
					//if both n's are small, but row 3 is at least 2 more than
					// row 4, swap them so we can split the larger one.
					if( nrows == 5
					      && tdp->table[3].n > tdp->table[4].n+1
						  && tdp->table[3].n <= 6 ){
						swap_rows( tdp, 3,4 );
					}

					tdp->n_dims = --nrows;
					struct nn_transpose_tabrow * xrowp = &tdp->table[nrows];
					tdp->n_outer = xrowp->n;
					tdp->outer_in_stride = xrowp->in_stride;
					tdp->outer_out_stride = xrowp->out_stride;
					tdp->outer_size /= tdp->n_outer;
					xrowp->n = 1;		// 'extra' rows must always have n=1.
				}
				// we need to have a work area for this. just find the largest work area
				// based on dsize...
				int work_area_size = 128*128 >> __builtin_ctz(dsize);	// 16K for d=1, 1K for d=16
				tdp->buffer_needed = work_area_size * TRANSPOSE_NUM_THREADS;
				tdp->buf_per_thread = work_area_size;

				tdp->execute_fp = transpose_execute_HVXFUNC;
				tdp->funcp = transpose_thread_bulktranspose;
				return 0;
			}
		}



		// find the gcf( dsize, 8)
		int log2_gcf = Q6_R_ct0_R(dsize|8);	//  0...3
		int gcf = 1<<log2_gcf;

		// use vmemcpy if any of
		// (1) dsize >= 32
		// (2) scalar copy would need at least 8 iteration to traverse d
		// (3) scalar copy needs >1 iterations and table is already full.
		//
		if( dsize >= 32 || dsize >= 8*gcf || 	// enough to chew for vmemcpy_2d
				(dsize >gcf && nrows == 5) ){	// full house & can't do as scalar.
			// invert the h,w if h much larger.
			if( h > w*2 )
				swap_rows( tdp, 1, 2);	// do the other way
			tdp->execute_fp = transpose_execute_3DMEMCPY_GEN;
			return 0;
		}
		// ok it's small. factor d into dsize = dn*gcf, convert the inner to size gcf; if dn >1 then make another level of looping
		// like this :     dn     gcf  gcf
		// and insert that amongst the first few rows so the n's descend.
		// We use a scalar strided copy to 2d copy 'gcf' , and if dsize > gcf, another loop level to move through that.
		//
		if( dsize > gcf){
			unsigned dn = dsize >> log2_gcf;
			// reduce inner copy to just 'gcf' bytes
			tdp->table[0].n = gcf;
			// invert the h,w if h much larger.
			if( h > w*2 )
				swap_rows( tdp, 1, 2);	// do the other way
			// now insert  a new row [ dn, gcf, gcf which interleaves the scalars.
			tdp->n_dims = ++nrows;
			// move table rows down until one of the following:
			// (a) the next one to move would be row 0
			// (b) the next to move is 1 or 2 but dn is < its 'n'.
			int irw = nrows-1;		// candidate location (currently, new added one)
			struct nn_transpose_tabrow * rowp = &tdp->table[irw];
			while( irw > 1 && ( irw >3 || dn >= rowp[-1].n  )){
				rowp[0] = rowp[-1];	// move this down
				--irw;
				--rowp;
			}	// put it there
			rowp->n = dn;
			rowp->in_stride = rowp->out_stride = gcf;
		}else{
			// when dsize= gcf, we have the possibility of a write-adjacent gather copy
			// (i.e. when first rows are [d 1 1], [w d *], but we'll still swap w,h rows
			// if h is much larger
			if( h >= w*4 )
				swap_rows( tdp, 1, 2);	// do the other way
		}
		static const nn_stride_scalar_copy_fp funcptrs[4] =
		{ strided_copy_2d_1b, strided_copy_2d_2b, strided_copy_2d_4b, strided_copy_2d_8b };

		tdp->execute_fp = transpose_execute_SCALAR;
		tdp->funcp = funcptrs[log2_gcf];
		return 0;
	}
}


// 'One step' entry point for transpose op
// The elementsize, perm_arr, permn, dims, ndim are exactly as per
//     nn_transpose_analyze_direct
// You can optionally pass in a scratch buffer via scratch, scratch_bytes.
// If the operation needs a scratch buffer, and the supplied buffer is
// too small, one will be allocated and freed.
//
// Note: nn may be NULL, this causes the execute to take
// place in the current thread, which must support hvx.
//
int nn_transpose_operation( struct nn_graph *nn,
		void *outp,			// output, vec-aligned
		void const *inp,	// input, vec-aligned
        int elementsize,
        int32_t const*perm_arr, int permn,
        uint32_t const *dims, int ndim,
        void * scratch,		// or NULL; must be vector aligned
        uint32_t scratch_bytes )	// size of scratch buffer.
{
	// analyze..
	struct nn_transpose_desc tdesc;
	int res = nn_transpose_analyze_direct( &tdesc,
           elementsize, perm_arr, permn, dims, ndim );
	if( res != 0 ) return res;
	void *sbuf = scratch;
	if( tdesc.buffer_needed != 0 ){
		if( sbuf == NULL || scratch_bytes < tdesc.buffer_needed){
			sbuf = nn_memalign( 128, tdesc.buffer_needed);
			if( sbuf == NULL){
				return errlog(nn,"can't alloc %u bytes", (unsigned)tdesc.buffer_needed);
			}
		}
	}
	// execute..
	res = nn_transpose_execute( nn, &tdesc, sbuf, outp, inp );
	if( sbuf != scratch)
		nn_free( sbuf);
	return res;
}

//
// special cases:
//
// Any operation that can be stated as [B,H,W,D] -> [B,W,H,D] (where D includes elementsize)
// will wind up as a 3-row case
//
//     D       1      1
//     W       D     H*D
//     H      W*D     D
// + outer loops  =B
//
// The hvx currently in s2d can do that, in multiple batches per call,
// provided that W = 4, D is a power of 2,  and W*D is <= 128
//
// We also want to handle the case where D=16,W=6,H a multiple of 4, efficiently.
// (this involves reading one 8 rows at a time - each being 96-byte unaligned;
//  then big 8x6 ->6x8 transpose in the register - and then store out 6 rows, 8 wide.
// if H is an odd multiple of 4, odd rows need to be stored out with unaligned stores and the
// last 4 cols (1/2 wide) are messy.
//

static int transpose_execute_NULL( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
		uint8_t * output, uint8_t const *input)
{
	return 0;
}

static int transpose_execute_MEMCPY( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
		uint8_t * output, uint8_t const *input)
{
	int cpylen = tdp->table[0].n;
	if( nn != NULL){
		if( cpylen < 256 ){
			memcpy( output, input, cpylen );
		}else{
			struct nn_memcpy_manager  mcman;
			nn_mcmanager_init(nn, &mcman );
			nn_mcmanager_vmemcpy( nn, &mcman, output, input, cpylen );
			nn_mcmanager_wait( nn, &mcman );
		}
	}else{
		vmemcpy_asm( output, input, cpylen );
	}
	return 0;
}					

static int transpose_execute_3DMEMCPY_GEN( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
		uint8_t * output, uint8_t const *input)
{
	unsigned nB= tdp->table[3].n;
	unsigned out_stride_B = tdp->table[3].out_stride;
	unsigned in_stride_B = tdp->table[3].in_stride;

	unsigned nC= tdp->table[2].n;
	unsigned out_stride_C = tdp->table[2].out_stride;
	unsigned in_stride_C = tdp->table[2].in_stride;
	
	int cpwid = tdp->table[0].n;		// width of copies
	int cpht = tdp->table[1].n;			// height of copies
	int cp_instride = tdp->table[1].in_stride;
	int cp_outstride = tdp->table[1].out_stride;

	struct nn_memcpy_manager  mcman;
	if( nn != NULL)
		nn_mcmanager_init(nn, &mcman );

	for( int iout = 0; iout <(int)tdp->n_outer; iout++){
		for( int iA = 0; iA < (int)tdp->table[4].n; iA++){
			uint8_t *op_A  = output + iout * tdp->outer_out_stride + iA * tdp->table[4].out_stride;
			uint8_t const *ip_A  = input + iout * tdp->outer_in_stride + iA * tdp->table[4].in_stride;
			for( int iB = 0; iB < (int)nB; iB++){
				if( nn != NULL){
					for( int iC = 0; iC < (int)nC; iC++){
						uint8_t *op = op_A + iB *out_stride_B + iC * out_stride_C;
						uint8_t const *ip = ip_A + iB *in_stride_B + iC * in_stride_C;
						nn_mcmanager_vmemcpy_2d( nn, &mcman,
							 cpwid, cpht,          // width, height
							 op, cp_outstride,      // outp, out_stride
							 ip, cp_instride);      // inp, in_stride
					}
				}else{
					for( int iC = 0; iC < (int)nC; iC++){
						uint8_t *op = op_A + iB *out_stride_B + iC * out_stride_C;
						uint8_t const *ip = ip_A + iB *in_stride_B + iC * in_stride_C;
						vmemcpy_2d_general_asm(
							 cpwid, cpht,          // width, height
							 op, cp_outstride,      // outp, out_stride
							 ip, cp_instride);      // inp, in_stride
					}
				}
			}
		}
	}
	
	if( nn != NULL)
		nn_mcmanager_wait( nn, &mcman );
	
	return 0;
}


static int transpose_execute_SCALAR( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
		uint8_t * output, uint8_t const *input)
{
	// 'scalar 2d copy' in effect does a 3-dimension copy with the constraint that the
	// size of the inner dimension is fixed (at 1,2,4,or 8).
	// So when used, the first row of the table is known to match that size. the function call
	// takes care of two more rows; 3 loops handle outer,A,B.

	unsigned nB= tdp->table[3].n;
	unsigned out_stride_B = tdp->table[3].out_stride;
	unsigned in_stride_B = tdp->table[3].in_stride;

	// get the dimensions for the function call...
	int cpwid = tdp->table[1].n;		// width of copies
	int cpht = tdp->table[2].n;						// height of copies
	int cp_winstride = tdp->table[1].in_stride;
	int cp_woutstride = tdp->table[1].out_stride;
	int cp_hinstride = tdp->table[2].in_stride;
	int cp_houtstride = tdp->table[2].out_stride;

	nn_stride_scalar_copy_fp copy_func = (nn_stride_scalar_copy_fp)tdp->funcp;

	for( int iout = 0; iout <(int)tdp->n_outer; iout++){
		for( int iA = 0; iA < (int)tdp->table[4].n; iA++){
			uint8_t *op_A  = output + iout * tdp->outer_out_stride + iA * tdp->table[4].out_stride;
			uint8_t const *ip_A  = input + iout *  tdp->outer_in_stride + iA * tdp->table[4].in_stride;
			for( int iB = 0; iB < (int)nB; iB++){
				uint8_t *op = op_A + iB *out_stride_B;
				uint8_t const *ip = ip_A + iB *in_stride_B;
				(*copy_func)( op, ip, cpht, cpwid,
						cp_hinstride, cp_winstride,
						cp_houtstride, cp_woutstride );
			}
		}
	}
	return 0;
}



// strided 2d scalar copy
#define STRIDED_COPY_2D( FNAME,TYP)\
void FNAME( uint8_t * outp, uint8_t const *inp, int h, int w, int hsi, int wsi, int hso, int wso )\
{\
    __builtin_assume(w >= 2);              \
    __builtin_assume(h >= 2);              \
	int h_half = h>>1;                     \
	for( int ih = 0; ih < h_half; ih++ ){  \
		TYP x0 = *(TYP const*)inp;         \
		TYP x1 = *(TYP const*)(inp+hsi);   \
		uint8_t const * inpw = inp + wsi;  \
		uint8_t const * outpw = outp;      \
		for( int iw = 0; iw < w-1; iw++){  \
			*(TYP*)outpw = x0;             \
			*(TYP*)(outpw+hso) = x1;       \
			x0 = *(TYP const*)inpw;        \
			x1 = *(TYP const*)(inpw+hsi);  \
			outpw += wso;                  \
			inpw += wsi;                   \
		}                                  \
		*(TYP*)outpw = x0;                 \
		*(TYP*)(outpw+hso) = x1;           \
		inp += hsi*2;                      \
		outp += hso*2;                     \
	}                                      \
	if( h&1){ 	/* single row  */          \
		TYP x0 = *(TYP const*)inp;         \
		uint8_t const * inpw = inp + wsi;  \
		uint8_t const * outpw = outp;      \
		for( int iw = 0; iw < w-1; iw++){  \
			*(TYP*)outpw = x0;             \
			x0 = *(TYP const*)inpw;        \
			outpw += wso;                  \
			inpw += wsi;                   \
		}                                  \
		*(TYP*)outpw = x0;                 \
	}                                      \
}

STRIDED_COPY_2D( strided_copy_2d_1b, uint8_t)
STRIDED_COPY_2D( strided_copy_2d_2b, uint16_t)
STRIDED_COPY_2D( strided_copy_2d_4b, uint32_t)
STRIDED_COPY_2D( strided_copy_2d_8b, uint64_t)


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// HVX code for special cases

// when we run hvx the handler sets up a 'runstate' object in its frame,
// and passes a reference to all the run threads.
// jobs are divided up according to 'n_outer'; each thread takes 'n_outer_chunk'
// at a time by using a __sync_fetch_and_add on the job_index variable.
//
struct transpose_hvx_runstate {
	struct nn_transpose_desc const * tdp;
	uint8_t const* input;
	uint8_t * output;
	uint8_t * work_area_base;		// where applicable
	uint32_t work_area_perthread;	// where applicable
	int n_outer_chunk;				// number to do in one pass >= 1
	volatile int job_index;			// used to get next chunk index
	volatile int thrd_index;		// used to divide up work area (where applicable)
	nn_sem_t done_sem;				// only used when nn!=NULL
};


// find n / ( round_down_to_power_of_2(d))
// rounding up. n>=0, d>0.
unsigned approx_divide( unsigned n, unsigned d)
{
	int rsh = 31-Q6_R_cl0_R(d);
	return ((int)(n-1) >> rsh) + 1;
}
///////////////////////////////////////
// execute function for generic hvx code.
// tdp->funcp must contain the pointer to the run thread.
//
static int transpose_execute_HVXFUNC( struct nn_graph *nn, struct nn_transpose_desc const * tdp, void *buffer,
		uint8_t * output, uint8_t const *input)
{
	struct transpose_hvx_runstate runstate;
	runstate.tdp = tdp;
	runstate.input = input;
	runstate.output = output;
	runstate.job_index = 0;
	runstate.thrd_index = 0;
	runstate.work_area_perthread = tdp->buf_per_thread;
	runstate.work_area_base = buffer;

	int nouter = tdp->n_outer;
	// each chunk is about 32k, or if total < 32k*threads, divide by threads,
	int outersize = tdp->outer_size;
	int nchunk;
	if( outersize*nouter >= 32768*TRANSPOSE_NUM_THREADS){
		nchunk = approx_divide(32768,outersize);
	}else{
		nchunk = (nouter+ (TRANSPOSE_NUM_THREADS-1))/(unsigned)TRANSPOSE_NUM_THREADS;
	}
	runstate.n_outer_chunk =  nchunk;
	void( *work_fp)( struct nn_graph *, void*) = (void(*)( struct nn_graph *, void*)) tdp->funcp;

	if( nn == NULL){		// call directly
		(*work_fp)( NULL, &runstate);
		return 0;
	}
	nn_sem_init(&runstate.done_sem, 0);

	int nthreads = min_i32( nchunk,TRANSPOSE_NUM_THREADS );
	for( int i = 0; i < nthreads; i++){
		nn_os_work_for_vector( nn,work_fp, & runstate );
	}
	nn_sem_wait_n_times( &runstate.done_sem, nthreads);
	return 0;
}
///////////////////////////////////////////////////////////////////////////
// HVX code for 'bulk 'transpose'.
// in the below, 'D' is a power of 2, in range 1..16
// Simpler case: generic 3-row table
//     D      1        1
//     W      D     H*D
//     H      W*D     D
// (+ 'outer batch loop' = B,H*W*D,H*W*D)
//
// More complex 4-row cases (all reduce to):
//     D      1                       1
//     W      D                    H*D*N
//     H     W*D*N                   D
//     N     D*W                    H*D
// (note that the strategy will reorder rows so that row 3 has an input stride of D;
// this is why the W*D*N, W*D are out of order).
//
// For 5-row cases, there are 8 distinct cases, each with 2 'outer' dims.
// In all cases the output stride on 2nd row, and input stride in 3rd row, are 'D'.
// That's all we really need to know here; we just allow perform_bulk_transpose_2 to handle
// the first 4 rows of the table; 5th row is handled in 'middle_count' loop here,
// and there is generally an outer loop which is shared over threads.).
// If there's no true 'batch' outer loop (with common strides), and the table has >= 4 rows,
// the strategy may choose to make the table last row an outer loop, so we respect differing
// outer_in_stride, outer_out_stride here.
//
static void transpose_thread_bulktranspose( struct nn_graph *nn , void *rstpv ){
	struct transpose_hvx_runstate *rstp = (struct transpose_hvx_runstate*) rstpv;
	struct nn_transpose_desc const * tdp = rstp->tdp;

	int thrdno = __sync_fetch_and_add( &rstp->thrd_index,1);
	uint8_t * work_area = rstp->work_area_base + thrdno * rstp->work_area_perthread;

	struct bulk_transpose_parms btxp = {
		.in_dims = {
				tdp->table[3].n,		// 'batches' (=1 when nrows ==3)
				tdp->table[1].n,		// input 'h' (output w)
				tdp->table[2].n,		// input 'w' (output h)
				tdp->table[0].n,		// depth (bytes)
		},
		.in_h_stride = tdp->table[1].in_stride,
		.out_h_stride = tdp->table[2].out_stride,
		.in_b_stride = tdp->table[3].in_stride,
		.out_b_stride = tdp->table[3].out_stride,
	};
	uint8_t const * inp0 = rstp->input;
	uint8_t * outp0 = rstp->output;


	// sometimes we'll need a 'middle' loop
	int middle_count = tdp->table[4].n;  // =1 when nrows < 5

	int njobs = tdp->n_outer;
	int outer_chunk = rstp->n_outer_chunk;
	int job0;
	int res = 0;
	while( job0 = __sync_fetch_and_add( &rstp->job_index,outer_chunk),   job0 < njobs ){
		int job1 = min_i32( job0 + outer_chunk, njobs);
		for(int ijob = job0; ijob < job1; ijob++){
			uint8_t const * inp = inp0 + ijob * tdp->outer_in_stride;
			uint8_t * outp = outp0 + ijob * tdp->outer_out_stride;
			for(int imid =0; imid < middle_count; imid++ ){
				res = perform_bulk_transpose_2( outp, inp, work_area, &btxp, 0);
				if(res != 0) goto quit;
				inp += tdp->table[4].in_stride;
				outp += tdp->table[4].out_stride;
			}

		}
	}
 quit:
	if(nn!=NULL)
		nn_sem_post( &rstp->done_sem);
}

//////////////////////////////////////////////
// special case for transposing  [n,28,6,16] - > [n,6,28,16]
//  or  [n,112,6,4]->[n,6,112,4]
//
// or in general  transpose [n,W/d,6,d] to [n,6,W/d,d]
//   where d = 1,2, .... 64
//  and W = 64*k
//
// Each chunk of input is (k/2)* 6 vectors;
// Each group of 6 contains 192*i32;
//    - deal that 6-ways to 6 vectors (6 output rows)
//    - store the 6 with a stride of W = k/2 vectors.
//
// Each whole unit is W*6 bytes long, so is vector aligned, but the odd
// row stores are misaligned when k is odd (W is not a multiple of 128);
// in that case we need to a last chunk of 3 input vectors with 6 half-row
// stores.
//
//  3 row table looks like this:
//     0:    d     1       1
//     1    W/d    d      6*d
//     2     6     W       d
//
//         n_outer 6*W    6*W  <- row removed to 'outer'
//
// Input and output pointers are both multiples of 6*W from start of tensor, so
// must be aligned.
//
// (this could be generalized to work with more complex transposes, having more than 3 table rows,
// where the input stride on row 1 is a multiple of 6*d. We would need to add the outer-loop stuff).
//
static void
transpose_thread_k64_deal6( struct nn_graph *nn , void *rstpv )
{

	struct transpose_hvx_runstate *rstp = (struct transpose_hvx_runstate*) rstpv;
	struct nn_transpose_desc const * tdp = rstp->tdp;

	int d = tdp->table[0].n;							// element size
	int out_row_stride = tdp->table[2].out_stride;		// W
	int k = out_row_stride  >> 6;	// value of k

	int group_stride = 6*out_row_stride;

	int full_ops = k>>1;			// # of full_vector loops
	struct hvx_shufdeal3_consts d3const = hvx_shufdeal3_get_consts( d );	// get consts for deal3
	HVX_VectorPred q_lo = Q6_Q_vsetq_R(64);

	int njobs = tdp->n_outer;
	int outer_chunk = rstp->n_outer_chunk;
	int job0;
	while( job0 = __sync_fetch_and_add( &rstp->job_index,outer_chunk),   job0 < njobs ){
		int job1 = min_i32( job0 + outer_chunk, njobs);

		// do jobs job0 ... job1-1

		uint8_t * output = rstp->output + group_stride * job0;
		uint8_t const * input = rstp->input + group_stride * job0;

		for( int i = 0; i < job1-job0; i++){
			uint8_t * wptr = output;
			HVX_Vector const * rdp=(HVX_Vector const *)input;	// always vec aligned

			for(int i = 0; i < full_ops; i++ ){

				HVX_Vector_x3 out_0 = hvx_deal3( d3const, rdp[0], rdp[1], rdp[2] );
				rdp += 3;
				HVX_Vector_x3 out_1 = hvx_deal3( d3const, rdp[0], rdp[1], rdp[2] );
				rdp += 3;

				HVX_VectorPair shuf = Q6_W_vdeal_VVR( out_1.val[0], out_0.val[0], -d3const.elementsize );
				*(HVX_Vector *)wptr = Q6_V_lo_W( shuf );									// even row 0 aligned
				q6op_vstu_AV( (HVX_Vector *)(wptr+ 3*out_row_stride), Q6_V_hi_W( shuf ));	// odd row  3 may be misaligned

				shuf = Q6_W_vdeal_VVR( out_1.val[1], out_0.val[1], -d3const.elementsize );
				q6op_vstu_AV( (HVX_Vector *)(wptr+ out_row_stride), Q6_V_lo_W( shuf ));		// odd row 1
				*(HVX_Vector *)(wptr+ 4*out_row_stride) = Q6_V_hi_W( shuf );				// even row 4

				shuf = Q6_W_vdeal_VVR( out_1.val[2], out_0.val[2], -d3const.elementsize );
				*(HVX_Vector *)(wptr+2*out_row_stride)= Q6_V_lo_W( shuf );					// even row 2
				q6op_vstu_AV( (HVX_Vector *)(wptr+ 5*out_row_stride), Q6_V_hi_W( shuf ));	// odd row 5
				wptr += 128;
			}
			if( (k & 1)!= 0){
				// last half chunk when k odd.
				HVX_Vector_x3 out_0 = hvx_deal3( d3const, rdp[0], rdp[1], rdp[2] );

				// deal them with themselves (each half of the results will be the same; we store the
				// lower half in even rows and upper half in odd rows).
				HVX_VectorPair shuf = Q6_W_vdeal_VVR( out_0.val[0], out_0.val[0], -d3const.elementsize );
				q6op_vstcc_QAV ( q_lo, (HVX_Vector*)wptr,                    Q6_V_lo_W(shuf));
				q6op_vstcc_QnAV( q_lo, (HVX_Vector*)(wptr+3*out_row_stride), Q6_V_hi_W(shuf));

				shuf = Q6_W_vdeal_VVR( out_0.val[1], out_0.val[1], -d3const.elementsize );
				q6op_vstcc_QnAV( q_lo, (HVX_Vector*)(wptr+out_row_stride),   Q6_V_lo_W(shuf));
				q6op_vstcc_QAV ( q_lo, (HVX_Vector*)(wptr+4*out_row_stride), Q6_V_hi_W(shuf));

				shuf = Q6_W_vdeal_VVR( out_0.val[2], out_0.val[2], -d3const.elementsize );
				q6op_vstcc_QAV ( q_lo, (HVX_Vector*)(wptr+2*out_row_stride), Q6_V_lo_W(shuf));
				q6op_vstcc_QnAV( q_lo, (HVX_Vector*)(wptr+5*out_row_stride), Q6_V_hi_W(shuf));
			}
			output += group_stride;
			input += group_stride;
		}
	}
	if(nn!=NULL)
		nn_sem_post(&rstp->done_sem);
}


//
//  cases [ b, 4, n, d] -> [ b, n, 4, d]
// where d is a power of 2 <= 64,
// and n*d <= 128. So, each read is one vector; each store is 1..5 vectors
//
//
//  3 row table looks like this:
//     0:    d     1       1
//     1     4     d      n*d
//     2     n   4*d       d
//
//        n_outer 4*n*d    4*n*d  <- row removed to 'outer'
//
static void
transpose_thread_1vec_shuffle4( struct nn_graph *nn , void *rstpv )
{
	struct transpose_hvx_runstate *rstp = (struct transpose_hvx_runstate*) rstpv;
	struct nn_transpose_desc const * tdp = rstp->tdp;

	int d = tdp->table[0].n;							// element size, power of 2 <= 64
	int in_stride =  tdp->table[1].in_stride;  // = w_in*d_in_stride; out stride is 4 times this
	int group_stride = 4* in_stride;

	int nd = in_stride;	// size of 'input row'; 4 of these shuffled to 1 output.

	int njobs = tdp->n_outer;
	int outer_chunk = rstp->n_outer_chunk;
	int job0;
	while( job0 = __sync_fetch_and_add( &rstp->job_index,outer_chunk),   job0 < njobs ){
		int job1 = min_i32( job0 + outer_chunk, njobs);

		// do jobs job0 ... job1-1

		uint8_t * output = rstp->output + group_stride * job0;
		uint8_t const * input = rstp->input + group_stride * job0;

		if(nd <= 32){
			// each store is always 1 or 2 vectors
			for( int ih = 0; ih < job1-job0; ih++ ){
				// read 4 vectors using unaligned reads
				HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_Vector v1 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_Vector v2 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_Vector v3 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_VectorPair sh02 = Q6_W_vshuff_VVR( v2, v0, -d );
				HVX_VectorPair sh13 = Q6_W_vshuff_VVR( v3, v1, -d );
				HVX_VectorPair sh01 = Q6_W_vshuff_VVR( Q6_V_lo_W(sh13), Q6_V_lo_W(sh02), -d );
				v0 = Q6_V_lo_W(sh01);	// all the results are here.
				size_t op = (size_t) output;
				HVX_Vector vo0 = Q6_V_vlalign_VVR( v0,v0,op);
				HVX_VectorPred q0 = Q6_Q_vsetq_R( op );
				op &= 127;
				int extent = op + 4*in_stride;		// <255;
		#if __HEXAGON_ARCH__ >= 62
				HVX_VectorPred q1 = Q6_Q_vsetq2_R( extent );
				if( extent > 128 ){
		#else
				HVX_VectorPred q1 = Q6_Q_vsetq_R( extent );
				if( extent >= 128 ){
		#endif
					q6op_vstcc_QAV( q1, (HVX_Vector*)(output+128), vo0);
					q1 = Q6_Q_or_QQn(q1,q1);		// force to 1's
				}
				q0 = Q6_Q_or_QQn(q0,q1);
				q6op_vstcc_QnAV( q0, (HVX_Vector*)output, vo0);
				output += group_stride;
			}
		}else{	// nd >32; each store is at least 2 vectors; up to 5.
			for( int ih = 0; ih < job1-job0; ih++ ){
				// read 4 vectors using unaligned reads
				HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_Vector v1 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_Vector v2 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_Vector v3 = q6op_V_vldu_A( (HVX_Vector const*)input ); input += in_stride;
				HVX_VectorPair sh02 = Q6_W_vshuff_VVR( v2, v0, -d );
				HVX_VectorPair sh13 = Q6_W_vshuff_VVR( v3, v1, -d );
				HVX_VectorPair sh01 = Q6_W_vshuff_VVR( Q6_V_lo_W(sh13), Q6_V_lo_W(sh02), -d );
				HVX_VectorPair sh23 = Q6_W_vshuff_VVR( Q6_V_hi_W(sh13), Q6_V_hi_W(sh02), -d );
				v0 = Q6_V_lo_W(sh01);
				v1 = Q6_V_hi_W(sh01);
				v2 = Q6_V_lo_W(sh23);
				v3 = Q6_V_hi_W(sh23);

				size_t op = (size_t) output;
				HVX_Vector vo0 = Q6_V_vlalign_VVR( v0,v0,op);
				// first store
				HVX_VectorPred q0 = Q6_Q_vsetq_R( op );
				q6op_vstcc_QnAV( q0, (HVX_Vector*)output, vo0);
				op &= 127;
				// we now need (0..3) full vectors and a final one which
				// is partly or completely empty.

				int extent = op + 4*in_stride;		// extent of output; > 128; <= 635
				HVX_VectorPred q1 = Q6_Q_vsetq_R(  extent );
				int nvec = extent>>7;					// 1..4; number of full vecs, +1

				// get the other results; we need up to 4 of them
				HVX_Vector vo1 = Q6_V_vlalign_VVR( v1,v0,op);
				HVX_Vector vo2 = Q6_V_vlalign_VVR( v2,v1,op);
				HVX_Vector vo3 = Q6_V_vlalign_VVR( v3,v2,op);
				HVX_Vector vo4 = Q6_V_vlalign_VVR( v3,v3,op);

				// point to where the partial store goes.
				uint8_t *olast = output + nvec*128;
				if( nvec >= 3 ){						// if 3 or 4
					((HVX_Vector*)output)[1] = vo1;		// store 1st 2 and copy the next down
					((HVX_Vector*)output)[2] = vo2;
					vo1 = vo3;
					vo2 = vo4;
				}
				if ( (nvec &1)==0 ){				// if 2 or 4
					((HVX_Vector*)olast)[-1] = vo1;	// store 1 more, copy last down
					vo1 = vo2;
				}
				// finally v01 is last one
				q6op_vstcc_QAV( q1, (HVX_Vector*)olast, vo1);
				output += group_stride;
			}
		}
	}
	if(nn!=NULL)
		nn_sem_post(&rstp->done_sem);
}

//
//  cases [ b, n, 2, d] -> [ b, 2, n, d]
// where d is a power of 2 <= 64,
// and n*d >=128. Done with unaligned reads; deal to two;
// if n*d is not a multiple of 128, there's an extra overlapped
// op at the end
//
//
//  3 row table looks like this:
//     0:    d     1       1
//     1     n     d      2*d
//     2     2   n*d       d
//
//        n_outer 2*n*d    2*n*d  <- row removed to 'outer'
//
static void
transpose_thread_deal2( struct nn_graph *nn , void *rstpv )
{
	struct transpose_hvx_runstate *rstp = (struct transpose_hvx_runstate*) rstpv;
	struct nn_transpose_desc const * tdp = rstp->tdp;

	int d = tdp->table[0].n;							// element size, power of 2 <= 64
	int out_stride =  tdp->table[2].out_stride;  // = n*d; in stride is 2 times this
	int group_stride = 2* out_stride;

	int nv = out_stride /128u;			// # of full vector ops (>=1)
	int last_adj = (-out_stride) & 127;
	int njobs = tdp->n_outer;
	int outer_chunk = rstp->n_outer_chunk;
	uint8_t const * input0 = rstp->input;
	int job0;
	while( job0 = __sync_fetch_and_add( &rstp->job_index,outer_chunk),   job0 < njobs ){
		int job1 = min_i32( job0 + outer_chunk, njobs);

		// do jobs job0 ... job1-1
		uint8_t const * input = input0 + group_stride * job0;
		unsigned npf = ((job1-job0)*group_stride+127)/128u;
		l2fetch(input,128,128,npf);
		// TODO : probably should have another loop for the fully-aligned case.
		uint8_t * output = rstp->output + group_stride * job0;
		for( int ih = 0; ih < job1-job0; ih++ ){
			uint8_t const * inp = input;
			uint8_t  * outp = output;
			for( int iv = 0; iv < nv; iv++){
				// read 2 vectors contiguously;
				HVX_Vector v0 = q6op_V_vldu_A((HVX_Vector const*) inp);
				HVX_Vector v1 = q6op_V_vldu_A((HVX_Vector const*)(inp+128));
				HVX_VectorPair dealt = Q6_W_vdeal_VVR( v1, v0, -d );
				q6op_vstu_AV( (HVX_Vector *) outp, Q6_V_lo_W(dealt));
				q6op_vstu_AV( (HVX_Vector *) (outp + out_stride), Q6_V_hi_W(dealt));
				inp += 256;
				outp += 128;
			}
			if( last_adj >0 ){
				inp -= last_adj*2;
				outp -= last_adj;
				HVX_Vector v0 = q6op_V_vldu_A((HVX_Vector const*) inp);
				HVX_Vector v1 = q6op_V_vldu_A((HVX_Vector const*)(inp+128));
				HVX_VectorPair dealt = Q6_W_vdeal_VVR( v1, v0, -d );
				q6op_vstu_AV( (HVX_Vector *) outp, Q6_V_lo_W(dealt));
				q6op_vstu_AV( (HVX_Vector *) (outp + out_stride), Q6_V_hi_W(dealt));

			}
			input += group_stride;
			output += group_stride;
		}
	}
	if(nn!=NULL)
		nn_sem_post(&rstp->done_sem);
}

// apply two transposes across 4 vectors (0<->2, 1<->3)
#define TRANSPOSE_4( VIN0, VIN1, VIN2, VIN3,  VOUT0, VOUT1, VOUT2, VOUT3 , NN) \
		HVX_Vector VOUT0,VOUT1,VOUT2,VOUT3;\
		{\
			HVX_VectorPair t02 = Q6_W_vdeal_VVR( VIN2, VIN0, NN );\
			HVX_VectorPair t13 = Q6_W_vdeal_VVR( VIN3, VIN1, NN );\
			VOUT0 = Q6_V_lo_W( t02 ); VOUT2 = Q6_V_hi_W( t02 );\
			VOUT1 = Q6_V_lo_W( t13 ); VOUT3 = Q6_V_hi_W( t13 );\
		}
			
//
// This is for use when transposing [batches, A*8, B*8, 16] -> [batches, B*8, A*8,16]
//
// So, in each batch, the input is in A*8 rows of 'B' vectors, and the
// output is in B*8 rows of A vector.
//  process for each batch is:
//   for iA in 0..A-1:
//     for iB in 0..B-1:
//        - get 8 vectors starting at row iA*8, col iB, row spacing=1 row (B vecs)
//        - transpose internally, 8x8 units of 16 bytes
//        - store 8 vectors starting at row iB*8, col iA, row spacing=1 row (A vecs)
//
//  3 row table looks like this:
//     0:    16    1       1
//     1     A*8  16      B*128
//     2     B*8  A*128    16
//
//     n_outer A*B*1K   A*B*1K  <- row removed to 'outer'
//
//
static void
transpose_thread_A8xB8x16( struct nn_graph *nn , void *rstpv )
{

	struct transpose_hvx_runstate *rstp = (struct transpose_hvx_runstate*) rstpv;
	struct nn_transpose_desc const * tdp = rstp->tdp;

	int kA = tdp->table[1].n /8u;
	int kB = tdp->table[2].n /8u;

	// this eliminates <=0 tests in inner loops
	if(kA <=0 || kB <=0) { if(nn!=NULL)nn_sem_post(&rstp->done_sem); return; }
	
	int in_stride = kB*128;
	int out_stride = kA*128;

	int group_stride = 8*kB*out_stride;

	int njobs = tdp->n_outer;
	int outer_chunk = rstp->n_outer_chunk;
	int job0;

	
	while( job0 = __sync_fetch_and_add( &rstp->job_index,outer_chunk),   job0 < njobs ){
		int job1 = min_i32( job0 + outer_chunk, njobs);
		// do jobs job0 ... job1-1

		uint8_t * output = rstp->output + group_stride * job0;
		uint8_t const * input = rstp->input + group_stride * job0;

		for( int i = 0; i < job1-job0; i++){
			l2fetch( input, in_stride + (1<<16), in_stride, kA*8);	// fetched in columns
			for( int iB = 0; iB < kB ;iB++){ 
			  for( int iA = 0; iA< kA ;iA++ ){
				uint8_t const * inp = input + iA*8*in_stride  +128*iB;
				uint8_t * outp = output + iB*8*out_stride + 128*iA;
				HVX_Vector va0 = *(HVX_Vector const*)inp;
				HVX_Vector va1 = *(HVX_Vector const*)(inp + in_stride );  inp += 2* in_stride;
				HVX_Vector va2 = *(HVX_Vector const*)inp;
				HVX_Vector va3 = *(HVX_Vector const*)(inp + in_stride );  inp += 2* in_stride;
				TRANSPOSE_4( va0, va2, va1, va3,    vb0, vb2, vb1, vb3,   16 );
				TRANSPOSE_4( vb0, vb1, vb2, vb3,    vc0, vc1, vc2, vc3,   2*16 );
				
				HVX_Vector va4 = *(HVX_Vector const*)inp;
				HVX_Vector va5 = *(HVX_Vector const*)(inp + in_stride );  inp += 2* in_stride;
				HVX_Vector va6 = *(HVX_Vector const*)inp;
				HVX_Vector va7 = *(HVX_Vector const*)(inp + in_stride );  inp += 2* in_stride;
				TRANSPOSE_4( va4, va6, va5, va7,    vb4, vb6, vb5, vb7,   16 );
				TRANSPOSE_4( vb4, vb5, vb6, vb7,    vc4, vc5, vc6, vc7,   2*16 );

				TRANSPOSE_4( vc0, vc2, vc4, vc6,    vd0, vd2, vd4, vd6,   4*16 );
				TRANSPOSE_4( vc1, vc3, vc5, vc7,    vd1, vd3, vd5, vd7,   4*16 );
				
				*(HVX_Vector *)outp = vd0;
				*(HVX_Vector *)(outp + out_stride ) = vd1;	outp += 2*out_stride;
				*(HVX_Vector *)outp = vd2;
				*(HVX_Vector *)(outp + out_stride ) = vd3;	outp += 2*out_stride;
				*(HVX_Vector *)outp = vd4;
				*(HVX_Vector *)(outp + out_stride ) = vd5;	outp += 2*out_stride;
				*(HVX_Vector *)outp = vd6;
				*(HVX_Vector *)(outp + out_stride ) = vd7;	outp += 2*out_stride;
			  }
			}
			input += group_stride;
			output += group_stride;
		}
	}
	if(nn!=NULL)
		nn_sem_post(&rstp->done_sem);
}
