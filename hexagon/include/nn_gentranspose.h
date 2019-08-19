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
#ifndef NN_GENTRANSPOSE_H
#define NN_GENTRANSPOSE_H 1

#include <stdlib.h>
#include <stdint.h>

#include "nn_graph.h"


/*
 *  This is for a generalized flat-tensor 4d tranpose, which can be used a 'transpose' node, but can
 *  also be used for Depth to/from Space. and Batch to/from Space, since those can be expressed as
 *  transposes. Batch/Space is a special case, requiring 6 dimensions, including the innermost 'depth'
 *  and outermost 'batch').
 *
 */
/// For a normal transpose, there are three steps
//  STEP 1:
//    res =  nn_transpose_check(int32_t const*perm_arr, int permn,
//                   struct shape const *in_shape,
//                   struct shape * out_shape );
//
//				perm_arr:  array of permn ints, containg 0..permn-1 in some order
//				permn: # of dims transposed, must be 0..4 (must be >=2 to have any effect)
//				in_shape: input shape
//				out_shape: output shape, filled in according to transpose. Can't be in_shape.
//       This function checks the validity of the transpose and fills in outshape.
//			if permn == 4, the perm_arr indicates, for each output dim, the index of which
//			input dim is used. with 0,1,2,3 meaning b,h,w,d
//			if permn < 4, dims starting with b are skipped (and unaffected.
//			so if permn =3,  0,1,2 mean h,w,d; if permn = 2, the only non-nop operation is [1,0]
//			which means w,d are transposed.
//		Example: to transpose h & w, you can use permn = 4, perm_arr = [0,2,1,3], or
//		        permn = 3, perm_arr = [1,0,2].
//
// STEP 2:
//    res = nn_transpose_analyze( struct nn_transpose_desc * tdp,
//                  int elementsize,
//                  int32_t const*perm_arr, int permn,
//                   struct shape const *in_shape)
//
//     (this assumes (1) has been called to check the transpose is valid; it also
//      assumes that in_shape has all >= 1 dims, and elementsize is  >= 1).
//
//  This operation analyzes the transpose and figures out a strategy for it, leaving that in
//   tdp (which is assumed to be previously garbage).
//
//
// OR:
//  For the 6-dimension transpose in batch/space, you can skip steps 1 and 2 and instead use this
//   backdoor:
//
//    res = nn_transpose_analyze_direct( struct nn_transpose_desc * tdp,
//                  int elementsize,
//                  int32_t const*perm_arr, int permn,
//                  uint32_t const *dims, int ndim )
//		 Here, 'nd' is the number of dims of the source shape, 2..6; 'dims' is a pointer to nd dimensions (from outer to inner).
//		 describing the source shape. The rest are the same as for nn_transpose_analyze. in this case permn  can
//		be as large as ndim. (and caller must ensure the perm_arr is valid).
//		Not all 6-d transforms are actually supported: if the transform is 6-d, it must simplify to no more
//		than 6 dimensions, after adding an extra inner dimension to account for 'elementsize'; and if it still has
//		6 dimensions, there must be a common outer dimension. These rules still allow for all Batch/Space operations.
//
// NOTE that the above steps can be done just once and you can keep the struct nn_transpose_desc for
// executing the op multiple times.
//
//
// STEP 3:
//    - look at tdp->buffer_needed to see if the operation needs an intermediate.
// 		if >= 0, allocate that (vector aligned) and pass as buffer param to nn_transpose_execute
// Then do:
//	 res = nn_transpose_execute( struct nn_graph *nn, struct nn_transpose_desc const * tdp,
//			void * buffer,
//				uint8_t const *input, uint8_t * output)
//
//
//  The strategy can be re-used for the same operation (same input size, same perm), but must be
//  recalculated when these change. the struct nn_transpose_desc contains a flag
//   'is_batch_scalable' which, if set, means that the operation is scaleable if only the batch
//   size changes; if this is set you can  call nn_transpose_rescale_for_batches( &tdp, new_batches)
//   to scale for a new batch size (if the nn_transpose_analyze_direct entry point is used, 'batches'
//   refers to the first input dimension).
//
//
// this struct is used in the table rows.
struct nn_transpose_tabrow {
	uint32_t n;				// size of dimension
	uint32_t out_stride;		// output stride in bytes
	uint32_t in_stride;		// input stride in bytes
	uint32_t tmp;
} __attribute__((aligned(8))) ;

#define NN_TRANSPOSE_MAX_ORIG_DIMS  6		/* max # of dims in the input shape spec */
struct nn_transpose_desc {
	uint8_t is_batch_scalable;
	int n_dims;				// # of valid rows in the table,1..5.
	int n_outer;			// size of 'outer' dimension, >=1 (see below)
	int outer_in_stride;		// stride of 'outer dimension, if n_outer >= 2 (see below)
	int outer_out_stride;		// stride of 'outer dimension, if n_outer >= 2 (see below)
	unsigned outer_size;	// this is total size of each outer op, or the whole thing if n_outer = 1.
	int (*execute_fp)( struct nn_graph *nn, struct nn_transpose_desc const * td, void *,
				uint8_t *, uint8_t const *);
				
	void *funcp;			// general purpose function pointer.
	// table of dimensions (inner to outer)
	// row 0 is initialized to { n = elementsize, in_stride = out_stride = 1 }
	// and represents the inner element copy. So we need 6 more rows to handle
	// the n_tensor case.
	// After simplification:
	//   * the first row is always {  n0, 1, 1 } even if n0 =1;
	//     so table[0].n is the size of the element we are working with.
	//   * if n_dim = 0, the operation is a copy of n0 elements.
	//   * if n_outer >1, then the last row in the table had n = n_outer,
	//    and had in_stride = out_stride, and was thus an 'outer_batch'
	//    dimension, and was removed. outer_stride is the stride for this
	//    (and is only valid if n_outer>1). The last dim is not removed
	//    if it is the only one.
	//  In this case outer_in_stride = outer_out_stride = outer_size.
	//  There may be cases where a row is removed that doesn't have outstride = instride;
	//  in order to give us something to split up over threads.
	//  in such a case outer_in_stride != outer_out_stride, and outer_size
	//  represents the amount of data per outer; it may not be contiguous in either
	//  in or out.
	//
	struct nn_transpose_tabrow table[1+NN_TRANSPOSE_MAX_ORIG_DIMS];

	unsigned buf_per_thread;		// size of per-thread buffer needed (where applicable)

	// *** code using gentranspose should not access fields above here ***

	unsigned buffer_needed;		// size of buffer needed for execute; or 0.
};

int nn_transpose_check(int32_t const*perm_arr, int permn,
		                   struct shape const *in_shape,
		                   struct shape * out_shape );
int nn_transpose_analyze_direct( struct nn_transpose_desc * tdp,
                  int elementsize,
                  int32_t const*perm_arr, int permn,
                  uint32_t const *dims, int ndim );

// nn_transpose_analyze is a special case of nn_transpose_analyze_direct for ndim=4.

static inline int nn_transpose_analyze( struct nn_transpose_desc * td,
		int elementsize,
		int32_t const*perm_arr, int permn,
		struct shape const *in_shape){
	return nn_transpose_analyze_direct( td, elementsize, perm_arr, permn, &in_shape->dimension[0], 4 );
}
// this is a dummy until batch scaling is implemented
static inline int nn_transpose_rescale_for_batches(struct nn_transpose_desc const * td, uint32_t new_batches )
{
	return -1;
}

// ********
// * NOTE *
// ********
// nn_transpose_execute actually does the operation. You can call in *execute*
// phase, from a scalar thread with nn_graph pointer, and it will spawn hvx threads
// to do the work.
// OR, you can can call from a hvx thread, with NULL pointer for nn, and it will do
// all the work in the calling thread. This is intended for use, e.g. in prepare
// Same applies to nn_transpose_operation (which calls nn_transpose_execute indirectly)

static inline int nn_transpose_execute( struct nn_graph *nn, struct nn_transpose_desc const * td, void *buffer,
				 uint8_t * output, uint8_t const *input)
{
	return (*td->execute_fp)( nn,td,buffer, output, input );
}

// 'One step' entry point for transpose op
// The elementsize, perm_arr, permn, dims, ndim are exactly as per
//     nn_transpose_analyze_direct
// You can optionally pass in a scratch buffer via scratch, scratch_bytes.
// If the operation needs a scratch buffer, and the supplied buffer is
// too small, one will be allocated and freed.
// See **NOTE** above.
//
int nn_transpose_operation( struct nn_graph *nn,
		void *outp,			// output, vec-aligned
		void const *inp,	// input, vec-aligned
        int elementsize,
        int32_t const*perm_arr, int permn,
        uint32_t const *dims, int ndim,
        void * scratch,		// or NULL; must be vector aligned
        uint32_t scratch_bytes );	// size of scratch buffer.


//
// These are exported symbols in case they are useful elsewhere
// (lookin' at you, StridedSlice...)
// These just copy over h & width dims, with arbitrary input & output strides (in bytes) on each;
//   - elementsize is 1,2,4,or 8;
//   - the pointers and all 4 strides must be multiples of the elementsize.
// IMPORTANT: these assume h and w are both >= 2.
//
void strided_copy_2d_1b( uint8_t * outp, uint8_t const *inp, int h, int w, int hsi, int wsi, int hso, int wso);
void strided_copy_2d_2b( uint8_t * outp, uint8_t const *inp, int h, int w, int hsi, int wsi, int hso, int wso);
void strided_copy_2d_4b( uint8_t * outp, uint8_t const *inp, int h, int w, int hsi, int wsi, int hso, int wso);
void strided_copy_2d_8b( uint8_t * outp, uint8_t const *inp, int h, int w, int hsi, int wsi, int hso, int wso);

#endif // NN_GENTRANSPOSE_H
