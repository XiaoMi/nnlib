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
#include <quantize.h>
#include <stdio.h>
#include "hvx_inlines.h"
//
// 'gather' operator.
// This has 2-4 inputs:
//   input 0: index tensor; of int32's
//   input 1: table tensor; of ELTYPE
//   input 2: (optional) dimension select (see below).
//   input 3: (optional): actual rank of 'index' (see below).
//
//   output 0: output tensor (of ELTYPE)
//
// In the 'quint8' version, we add min/max inputs  and outputs, after input 1 and output 0.
//   (i.e. the optional inputs move to 4 and 5)
//
// (A) When there is no dimension select or rank input:
//
// - the shape of the table tensor is checked to find the first dimension (starting from
//   batches) which is > 1. This is the 'table index' dimension, and its size is TabN.
//    E.g. if the shape is  (1,64,4,32) then TabN is 64 and the table index dim is 1 (height).
// - the output shape is found by taking the trailing dims of the index tensor, and appending
//   the remaining dims of the table tensor (after index dim).
//   E.g.   in0  = (1,1, 5, 10)   in1 =   (1,64,4,32)   -> out = (5, 10, 4, 32 )
// - The 'dropped' dimensions from the index tensor must all be 1.
//
// The output tensor is of the same datatype as the table tensor, and is constructed as follows:
//     - each element in the index tensor is an index, normally 0.. TabN-1.
//     - a 'lookup' is done on the table tensor, on the index dimension; in the example
//       each lookup gives a result of shape (4,32).
//     - We arrange all these results in the buffer.
//
// Some simple cases:
//    - if the table is of shape   (1,1,1,TabN),
//       the result is the same shape as the index tensor, and is just an element-by element replacement
//       via lookup in the 1d table.
//    - if the table is of shape  (TabN,h,w,d), then the index tensor can only be 1-d,
//        i.e. (1,1,1,N), and the output will be be (N,h,w,d), formed by selecting N 'batches'
//       from the table tensor and concatenating them.
//
// (B) if the dimension select input is given, it is a single integer in range 0..3
//    and specifies the dimension of 'table' to be used as the table index. This must have size >=1.
//     (note that setting dimsel to < 0 is the same as omitting it)
//      Examples: in0: (1,1,2,5), in1 (1,8,64,20)  dimsel = 2  -> (8,2,5,20)
//      		  in0: (1,1,1,5), in1 (1,8,64,20)  dimsel = 2  -> (1,  8,5,20)
//      		  in0: (1,1,1,5), in1 (1,8,64,20)  dimsel = 3  -> (1,  8,64,5)
//                in0: (1,1,1,1), in1 (1,8,64,20)  dimsel  =2   ->(1,1,   8,20)
// (in the last example, the index input is considered to be rank 0)
//
//       note that:
//          -  leading 1's in the table shape are dropped (table is considered to be (8,64,20) in examples)
//          -  leading 1's in the index tensor are dropped.
//          -  The index tensor shape replaces the 'index dimension' in the tables shape.
//     By defining a dimension select, you can force use of a table dimension which is size 1;
//    in this case the index input must be all 0 (or will be ignored, if range-clipping is used).
//
// (C) if the 'index_rank' is given, this must be 0..4 and overrides auto-detecting the rank
//     of the index tensor. Any leading dimensions (the first '4-index_rank') must be 1.
//    Setting index_rank to <0 is the same as omitting it.
//
//     index_rank can be used to force leading '1' dims to be retained on the index tensor shape,
//     -- which only matters when dimsel is used
//
//      Examples:
//        in0: (1,1,1,9), in1: (1,8,64,32) dimsel = 2, ind_rank = 1  -> (1, 8,9,32)    (index is [9])
//        in0: (1,1,1,9), in1: (1,8,64,32) dimsel = 2, ind_rank = 2  -> (8,1, 9,32)    (index is [1,9])
//        in0: (1,1,1,1), in1: (1,8,64,32) dimsel = 3, ind_rank = 1  -> (1, 8,64, 1)   (index is [9])
//        in0: (1,1,1,1), in1: (1,8,64,32) dimsel = 3, ind_rank = 0  -> (1,1, 8,64)       (index is [])
//
//

/////////////////////
// This file also has implementation for 'embedded lookup':
//
// This has 2-4 inputs:
//   input 0: index tensor; of int32's
//   input 1: table tensor; of ELTYPE
//   input 2: (optional) table_structure:  list of ints giving the partition structure of the table.
//   input 3: (optional) partition_strategy  ( single int; 0= mod, 1 = div)
//
//   output 0: output tensor (of ELTYPE)
//
// When the 'table_structure' parameter is absent, this is processed exactly as
// 'lookup'. The 'table index' dimension of the table is the first dim >=1.
//
// For the full embedded lookup, however, we need at least two dimension of the table:
//   - the first one is a partition index
//   - the second one is used for the lookup index
//     e.g. we could have a table of shape [5,32,48]
//      - this is 5 separate tables, each of 32 of [48]; 5 partitions.
//     If the 'index' tensor is [1,7,2,5], then the result shape is
//            [7,2,5,48]
//        the operation selects one of the 5 partitions, according to
//        the last dimension of the index shape.
//    This can generalize in two ways:
//       (1) if the last dimension of the index tensor is larger than the number
//          of partitions, there are two ways of mapping the larger index to the smaller,
//           called 'div' and 'mod', selected by 'partition_strategy' (described below).
//       (2) The table tensor shape could be uneven in the second dimension; the lookup tables
//           could be of different sizes.
//           e.g. for 5 partitions it could be [5,*,48] where the size of the 2nd dimension
//           (the table lookup) depends on the first index. This is handled by storing the table as [total,48]
//           where 'total' is the sum of the 5 partitions, and the 'table structure' gives a map.
//           For equal partitions, e.g. [5,32,48], we can store it as [1,5,32,48] or as [1,1,5*32,48];
//           but both require a table_structure.
//
//   The table_structure is an array (1,1,1,NTS) of at least 3 integers:
//      -first one is 0..3, and it must match the index of the first dimension >= 1 in the table.
//      - second once is the number of partitions, TPN >= 2; TPN < 64K.
//      - the remaining values are table sizes for each of the partitions, each must be >=1 .
//        There must be either one value (indicating all of the partitions are the same size)
//        or TPN values (indicating all of the various sizes). All table sizes must be >=2.
//      The table input dimension (indicated by first entry) must have one of the following values:
//         (1) the total size of of all partitions as computed from the array; or
//         (2) exactly TPN; this is only allowed if there is a single 'size' entry (NTS=3), and in this
//             case the next dimension must match this size entry.
//		    Some examples:
//               table_shape = (1,5,32,128)       structure = (1,5,32)
//                         table is 5 partitions of 32 each; the elements are [128]
//               table_shape = (1,1,5*32,128)       structure = (2,5,32)
//                         same as previous, in effect.
//               table_shape = (1,1,5*32,128)       structure = (2,5,32,32,32,32,32)
//                         also same as previous.
//               table_shape = (1,320,6, 128),		structure = (1,5,64,64,32,32,128)
//                    table has 5 partitions of different sizes, adding up to 320, which
//                    are packed in the 2nd dimension. Each element is (6,128).
//
//     TPN  =  number of partitions in the table
//     IPN  = number of partitions in the index (size of last dimension).
//     if IPN > TPN, we need a way to select a table partition (idxT) based on the index partition (idxI)
//     The strategies are below, set by 'strategy' input; default is mod:
//     0 'mod':    idxT= idxI % TPN
//     1 'div':    idxT = idxI / TPN  (this formula applies only when IPN is a multiple of TPN).
//         when IPN is not a multiple of TPN, the 'div' strategy assigns ceil(IPN/TPN) indices
//         to the initial table partitions, until the remainder is evenly split, e.g.
//         IPN = 12, TPN = 5:
//                 indices 0,1,2    use partition 0    (9 remain for 4 partitions)
//                         3,4,5    use partition 1	   (6 remain for 3 partitions)
//                         6,7      use partition 2
//                         8,9      use partition 3
//                         10,11    use partition 4
//   The output shape for this operation is the input shape, with any 'trailing' dimensions of the table
//   shape appended, and 1's dropped on the left as needed. The total dims must be <= 4, e.g if the elements in the table are (6,128), the
//   index shape can be (1,1,12,20) , giving a result  (12,20,6,128); but the index shape can't be (1,12,3,4).
//
//

///////////////////////////////////////////////////////////
// this struct summarizes the gather op:
//  -outshape is the tensor shape
//  - we can do the op as:
//      for( i = 0; i < size_tab_outer; i++ ){		// usually 1
//          for ( j = 0;  j< size_index; j++ ){
//              idx = in_tensor[j];						// get index (flat addressing)
//              memcpy( out, table[opitch * i + size_tab_inner*idx,   tab_size_inner*elbytes )
//             out += tab_size_inner*elbytes
//    ... where 'opitch' = tab_size_inner * table_size*elbytes
// The 'for  j' loop is done by a single call to one of the copy funcs, selected according to
//  tab_size_inner*elbytes (but it can also be broken into threads, to split 'size_index'
//
// OUT-OF-RANGE indices:
//   handled according to the 'padding' field:
//     NN_PAD_NA:    same as 'NN_PAD_VALID'
//     NN_PAD_SAME :  error if the value is < 0 or >= TabN
//     NN PAD_VALID:   values are 'clipped' to range 0 .. TabN-1
//   Others are undefined.
//

struct gather_opparms {
	struct shape outshape;
	int index_dim;			// 0..3 dimension index for table
	int table_size;			// size of the table (>=1)  at index dim
	int size_tab_inner;		// size of table to the right of index dim (in elements)
	int size_tab_outer;		// size of table to the left of index dim (in elements)
	int size_index;			// total size of tensor array
	int elbytes;
};
// we choose a 'table copy funcp' based on the element size for lookup,
// this is specialized for various values of 'elbytes'.
// The generic one uses memcpy.
//
// for each there is a 'limit' and 'check' version.
// - both normally return 0.
// - if any indices are out of range the 'check' version will return 1+i where
///  is the position of the out of range index.
//
typedef int ( *table_copy_funcp)(
		uint8_t * out,		// output pointer [num] of elbytes
		int32_t const *indices, 	// input pointer (indices) [ num ] of int32,
		uint8_t const *tbl,			// table pointer [ TabN ] of elbytes
		int TabN, int elbytes, int num );

static table_copy_funcp
select_copy_func( int elsize, int check_range);struct gather_runstate;

// for two threads, the work is split in two across 'size_index'
// unless that is small and rather smaller than size_tab_outer, in which case
// the work is split on size_tab_outer. If both are 1,
// it's done in 1 thread.

struct gather_thrinfo {
	struct gather_runstate *rstp;
	int begin_index, end_index;	// range of 'index' to do
	int begin_outer, end_outer;	// range of 'tab_outer to do.
	int out_of_range_index;
};
struct gather_runstate {
	 struct gather_opparms opp;
	 uint8_t const * table_base;
	 int32_t const * index_base;
	 uint8_t * output_base;
	 table_copy_funcp copy_fp;
	 nn_sem_t done_sem;
	 struct gather_thrinfo thrinfo[2];
};

// 'opaque' is pointed to one of these.
struct gather_table_variant_token {
	int16_t element_bytes;			// e.g 4 for float
	int16_t element_typecode;		// e.g. NN_TYPE_FLOAT
	int16_t is_quant;		// 1 if it has the extra inputs/outputs
	// 'table' is diverted to 'gather' if it has no extra inputs.
	int16_t is_gather;
};

// this checks input parameters, and sets up everything in "gather_opparms" except elbytes.

static int
analyze_gather_op( struct nn_graph * nn, struct tensor const *index_tensor, struct tensor const * table_tensor,
		struct gather_opparms *oparms,
		int dimension_index, int index_rank )
{

	// determine index shape
	int irank = shape_apparent_rank( &index_tensor->shape);
	if( index_rank >= 0 ){		// specified rank must be >= apparent rank
		if(index_rank > 4 || index_rank <irank){
			return errlog(nn,"improper rank %d given for apparent rank=%d", index_rank, irank);
		}
		irank = index_rank;
	}
	// determine table rank and index dim
	// Note: as we establish valid limits on the ranges, the tests should be fully
	// comprehensible to static code analysis.
	//
	int trank = shape_apparent_rank(&table_tensor->shape);
	if( trank < 1) trank = 1;		// is now 1..4
	int index_dim  = 4- trank;		// default if no 'dimension_index'; 0..3
	if( dimension_index >= 0 ){
		if( dimension_index > 3){
			return errlog(nn, "dimension_index = %d : must be -1 ...3", dimension_index);
		}
		index_dim = dimension_index;
		// expand trank to include specified dim, if needed
		if( 4-index_dim > trank)
			trank = 4-index_dim;		// still in range 1..4
	}
	// When we have a table (1,1,3,5), then trank=2 and the default index_dim is 2.
	// if you select dimension_index= 0 or 1, then technically the table's rank
	// has been increased to 4 or 3 (resp), i.e. to (4-dimension_index)
	// The computations below are such that if (4-dimension_index)>trank,
	// it will still work as dei

	int tabsize = table_tensor->shape.dimension[index_dim];
	//
	// Now allowing tabsize == 1
	// In this situation, index must be 0 (unless we are clipping, in which case it will be ignored)
	//
	if( 0 && tabsize <= 1){
		return errlog(nn,"can't lookup on dimension %d of table tensor (size=%d)", index_dim, tabsize);
	}
	// proven:
	//    irank is 0..4
	//    trank is 1..4
	//     index_dim  is 0..3
	//    index_dim + trank_full >=4

	// work out the output shape
	int out_rank = irank + trank -1;		// guaranteed >= 0 since trank >= 1
	if(out_rank > 4 ){
		return errlog(nn,"index_rank = %d, table_rank = %d; output_rank = %d is too large", irank, trank, out_rank);
	}
	oparms->index_dim = index_dim;
	oparms->table_size = tabsize;

	// construct the output shape and consolidate dims
	int dpos = 3;
	int size_tab_inner = 1;
	int size_tab_outer = 1;
	int size_index = 1;		// total size of the index

	// from right to left: first the table dims after the index dim
	// (number = 3-index_dim;  i.e. 0..3)
	for( int i = 3; i > index_dim; i-- ){
		if (dpos < 0){
			return errlog(nn,"attempted to access dimension of %d", dpos);
		}
		int n = table_tensor->shape.dimension[i];
		oparms->outshape.dimension[dpos] = n;
		size_tab_inner *= n;
		dpos--;
	}
	// now the index dims
	// number = irank; 0..4
	for( int i = 3; i >= 4-irank; i-- ){
		if (dpos < 0){
			return errlog(nn,"attempted to access dimension of %d", dpos);
		}
		int n = index_tensor->shape.dimension[i];
		oparms->outshape.dimension[dpos] = n;
		size_index *= n;
		dpos--;
	}
	//
	// now the rest of the table, if any
	// number = index_dim_+ trank -4; 0..3
	//
	for( int i = index_dim-1; i >= 4-trank; i-- ){
		if (dpos < 0){
			return errlog(nn,"attempted to access dimension of %d", dpos);
		}
		int n = table_tensor->shape.dimension[i];
		oparms->outshape.dimension[dpos] = n;
		size_tab_outer *= n;
		dpos--;
	}
	// total filled = 3-index_dim + irank + (index_dim+trank-4)
	//    = irank+trank-1  (0..4)
	// Fill the rest in
	while( dpos >= 0){
		oparms->outshape.dimension[dpos] = 1;
		dpos--;
	}
	oparms->size_tab_inner = size_tab_inner;
	oparms->size_tab_outer = size_tab_outer;
	oparms->size_index = size_index;
	/*
	printf( "index shape = %d %d %d %d ; rank = %d\n",
			(int)index_tensor->shape.batches, (int)index_tensor->shape.height, (int)index_tensor->shape.width, (int)index_tensor->shape.depth,
			irank);
	printf( "table shape = %d %d %d %d ; rank = %d; dim @ %d (= %d)\n",
			(int)table_tensor->shape.batches, (int)table_tensor->shape.height, (int)table_tensor->shape.width, (int)table_tensor->shape.depth,
			trank, index_dim, tabsize);
	printf( "output shape = %d %d %d %d ; rank = %d\n",
			(int)oparms->outshape.batches, (int)oparms->outshape.height, (int)oparms->outshape.width, (int)oparms->outshape.depth,
			out_rank);
	*/
	return 0;

}

static void gather_worker_thread( struct nn_graph *nn, void * p);

static int
gather_execute( struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * index_tensor = self->inputs[0];
	struct tensor const * table_tensor = self->inputs[1];
	struct tensor const *const* scalar_tensors = &self->inputs[2];
	struct tensor * out_tensor = self->outputs[0];

	int n_scalar_inputs = self->n_inputs - 2;

	logmsg(nn,2,"gather node %p execute",self);

	struct gather_table_variant_token  * var_token = (struct gather_table_variant_token *)self->opaque;

	int elbytes = var_token->element_bytes;
	int data_type = var_token->element_typecode;

	if( var_token->is_quant){
		// copy min/max through
		tensor_copy( self->outputs[1], self->inputs[2]);
		tensor_copy( self->outputs[2], self->inputs[3]);
		// correct the scalar input pointer and count...
		n_scalar_inputs -= 2;
		scalar_tensors += 2;
	}
	// extract scalar parms where available
	int index_dim = -1;
	int index_rank = -1;
	if( n_scalar_inputs >= 1){
		index_dim = tensor_get_int32( scalar_tensors[0], 0);
		if( n_scalar_inputs >= 2 )
			index_rank = tensor_get_int32( scalar_tensors[1], 0);
	}

	struct gather_runstate rstate;
	rstate.opp.elbytes = elbytes;

	if( analyze_gather_op( nn, index_tensor, table_tensor, &rstate.opp, index_dim, index_rank)!= 0)
		return -1;
	// create the output tensor
	if( tensor_out_prepare_normal_fromshape( out_tensor, &rstate.opp.outshape, data_type) != 0 ){
		return errlog(nn, "output too small");
	}
	int check_range = self->padding == NN_PAD_SAME;
	int inner_size_bytes  = rstate.opp.size_tab_inner * elbytes;
	// select a copy func
	table_copy_funcp copy_func = select_copy_func( inner_size_bytes, check_range);
	rstate.copy_fp = copy_func;

	// set up for threads
	int size_index =  rstate.opp.size_index;		// # of lookups to do
	int size_tab_outer = rstate.opp.size_tab_outer;

#if 1
	rstate.index_base = (int32_t *) index_tensor->data;
	rstate.table_base = (uint8_t *) table_tensor->data;
	rstate.output_base = (uint8_t *) out_tensor->data;
	rstate.thrinfo[0].rstp = &rstate;
	rstate.thrinfo[0].begin_index = 0;
	rstate.thrinfo[0].end_index = size_index;
	rstate.thrinfo[0].begin_outer = 0;
	rstate.thrinfo[0].end_outer = size_tab_outer;
	int nthreads = 1;
	if( size_index > 1 || size_tab_outer > 1 ){	// two threads
		nthreads = 2;
		rstate.thrinfo[1] = rstate.thrinfo[0];
		if( size_index < 8 && size_tab_outer >= 2*size_index){
			// split on outer
			int n = (size_tab_outer+1)>>1;
			rstate.thrinfo[0].end_outer = n;
			rstate.thrinfo[1].begin_outer = n;
		}else{
			// split on index
			int n = (size_index+1)>>1;
			rstate.thrinfo[0].end_index = n;
			rstate.thrinfo[1].begin_index = n;
		}
	}
	nn_sem_init( &rstate.done_sem,0);
	for( int i = 0; i < nthreads; i++){
		nn_os_work_for_vector(nn, gather_worker_thread, &rstate.thrinfo[i]);
	}
	nn_sem_wait_n_times( &rstate.done_sem, nthreads);

	for( int i = 0; i < nthreads; i++){
		int k = rstate.thrinfo[i].out_of_range_index;
		if( k >= 0){
			int32_t val = ((int32_t *) index_tensor->data)[k];
			return errlog(nn, "out-of-range value %d found at offset %d; table size is %d",
					val, k, rstate.opp.table_size );
		}
	}
#else

	uint8_t * outp = (uint8_t *) out_tensor->data;
	uint8_t const * tabp = (uint8_t *) table_tensor->data;
	int32_t const * indp = (int32_t *) index_tensor->data;
	int Ntab = rstate.opp.table_size;	// size of lookup dim of table
	int table_outer_stride = inner_size_bytes * Ntab;	// table size per outer loop
	int output_outer_stride = inner_size_bytes * size_index;	// output size per outer loop
	//printf("inner_size_bytes= %d; Ntab= %d; size_index = %d; size_tab_outer = %d\n",
	//		inner_size_bytes, Ntab, size_index, size_tab_outer );

	for( int i =0; i < size_tab_outer; i++){
		int k = (*copy_func)( outp + i*output_outer_stride,
				indp, tabp + i*table_outer_stride,
				Ntab, inner_size_bytes, size_index );
		if( k!= 0){
			--k;
			int32_t val = indp[k];
			return errlog(nn, "out-of-range value %d found at offset %d; table size is %d",
					val, k, Ntab );
		}
	}
#endif
	logmsg(nn,2,"gather node %p complete",self);
	return 0;
}

static void gather_worker_thread( struct nn_graph *nn, void * p)
{
	struct gather_thrinfo * thrp = (struct gather_thrinfo *)p;
	struct gather_runstate  * rstp = thrp->rstp;
	int inner_size_bytes  = rstp->opp.size_tab_inner * rstp->opp.elbytes;
	int size_index =  rstp->opp.size_index;		// # of lookups to do
	int begin_outer = thrp->begin_outer;
	int begin_inner = thrp->begin_index;
	int end_inner = thrp->end_index;
	int end_outer = thrp->end_outer;
	uint8_t * outp = rstp->output_base;
	uint8_t const * tabp = rstp->table_base;
	int32_t const * indp = rstp->index_base;
	int Ntab = rstp->opp.table_size;	// size of lookup dim of table
	int table_outer_stride = inner_size_bytes * Ntab;	// table size per outer loop
	int output_outer_stride = inner_size_bytes * size_index;	// output size per outer loop
	//printf("inner_size_bytes= %d; Ntab= %d; size_index = %d; size_tab_outer = %d\n",
	//		inner_size_bytes, Ntab, size_index, size_tab_outer );
	table_copy_funcp copy_func = rstp->copy_fp;
	// offset for start of inner slice
	indp += begin_inner;
	outp += begin_inner * inner_size_bytes;
	int count_inner = end_inner - begin_inner;

	//logmsg(nn,0, "outer range %d.. %d; inner %d..%d", begin_outer, end_outer, begin_inner, end_inner);
	int oor_index = -1;
	for( int i = begin_outer; i < end_outer; i++){
		int k = (*copy_func)( outp + i*output_outer_stride,
				indp, tabp + i*table_outer_stride,
				Ntab, inner_size_bytes, count_inner );
		if( k> 0){	// k-1 is index of bad table index.
			--k;
			oor_index = begin_inner + k;
			break;
		}
	}
	thrp->out_of_range_index = oor_index;
	nn_sem_post( &rstp->done_sem);
}

//////////////////////// table //////////////////////////////////////////////////
//
// this struct contains parms of the 'table' op
//
#define MAX_TABLE_PARTITIONS 65536		// max # of partitions in table

struct table_pdesc	// partition descriptor
{
	int32_t partn_size;			// size of this partition, in entries
	uint8_t const *partn_data;		// pointer to where the data is.
};
//
// the size of the 'index' tensor is outer_index_size * index_partn  * sizeof(int32_t)
// the size of the output tensor is outer_index_size * index_partn  * inner_table_size * elbytes

struct table_opparms {
	struct shape outshape;
	int index_partn;		// number of partitions in index
	int table_partn;		// number of partitions in table
	struct table_pdesc * pdesc;	// pointers to partition descs [0..table_partn]
	int outer_index_size;	// product of 'other' dims of index (left of the last one)
	int inner_table_size;	// product of all dims to right of lookup index, in table
	int elbytes;
	int table_entry_bytes;				// inner_table_size *  elbytes.
	int output_partition_stride;	// index_partn * inner_table_size *  elbytes.
	int div_break;			// used in 'div' mapping
	int div_d;
	int div_offs;
};
struct table_runstate {
	struct table_opparms opp;
};

// analyze the 'table' op and
// fill out the table_opparms. This allocates scratch for the
// 'table_pdesc'.
// This is only used when there is a 'layout_tensor'.
// NOTE: oparms->elbytes must be filled in before calling.

static int
analyze_table_op( struct nn_graph * nn, struct tensor const *index_tensor,
		struct tensor const * table_tensor,
		struct tensor const * layout_tensor,
		struct table_opparms *oparms )
{
	// shape of layout must be (1,1,1,n_layout)
	int n_layout = layout_tensor->shape.depth;
	if(  shape_apparent_rank( & layout_tensor->shape)!= 1
			|| n_layout < 3 || n_layout > MAX_TABLE_PARTITIONS+2){
		return errlog(nn,"bad shape for layout tensor");
	}
	int32_t const * layout_arr = (int32_t const*)layout_tensor->data;
	int table_idx_dim = layout_arr[0];		// must be 0..3
	int n_table_partn = layout_arr[1];		// must be 2.. MAX_TABLE_PARTITIONS
	// table_idx_dim must be 0..3 and must be consistent with apparent rank of the table tensor
	// (that's a "brown M&M's" thing).
	//
	if( table_idx_dim > 3 || table_idx_dim < 0 || table_idx_dim + shape_apparent_rank( & table_tensor->shape) != 4
		|| n_table_partn < 2 || n_table_partn > MAX_TABLE_PARTITIONS
		|| (n_layout != 3 && n_layout != n_table_partn+2) ){
		return errlog(nn,"bad layout_tensor format");
	}
	// allocate the partition desc table
	//
	struct table_pdesc * pdescs = (struct table_pdesc*)nn_scratch_alloc( nn, n_table_partn *sizeof(struct table_pdesc) );
	if( pdescs == NULL){
		return errlog (nn,"scratch alloc");
	}
	oparms->pdesc = pdescs;
	oparms->table_partn = n_table_partn;

	// .. allow for cases where n_layout = 3:
	uint64_t total_table_size;
	if( n_layout == 3){
		uint32_t tsize = layout_arr[2];
		uint32_t ttable_size = table_tensor->shape.dimension[table_idx_dim];
		uint32_t nextdim_size = (table_idx_dim>=3)?0:
				table_tensor->shape.dimension[table_idx_dim+1];
		total_table_size = tsize * (uint64_t)n_table_partn;
		if( tsize < 2 ||
			(ttable_size != total_table_size  &&
					(ttable_size !=n_table_partn  || nextdim_size != tsize))
			){
			return errlog(nn,"tensor_layout doesn't match table tensor shape");
		}
		if( ttable_size == n_table_partn ){
			table_idx_dim++;			// table shape is (npart,size ..)
		}
		// fill out the sizes
		for(int i =0; i < n_table_partn; i++){
			pdescs[i].partn_size = tsize;
		}
	} else{
		// add them all up..
		total_table_size = 0;
		uint32_t min_dim = 0x7fffffff;
		for( int i =0; i < n_table_partn; i++){
			uint32_t tsize = layout_arr[2+i];
			total_table_size += tsize;
			min_dim = min_u32( tsize, min_dim);
			pdescs[i].partn_size = tsize;
		}
		// sum must match the 1st dim of table.
		if( total_table_size != table_tensor->shape.dimension[table_idx_dim]
		       || min_dim < 2 ){
			return errlog( nn, "sum of %d partition sizes is %d (not %d) and min_dim = %d",
					(int)total_table_size, (int)table_tensor->shape.dimension[table_idx_dim],
					(int) min_dim);
		}
	}
	// determine the output shape.
	// and find the size products
	int append_dims = 3-table_idx_dim;	// number of dims to append to index shape
	if( append_dims < 0 || shape_apparent_rank( & index_tensor->shape) + append_dims > 4 ){
		return errlog(nn,"can't append %d dims to index tensor", append_dims);
	}
	uint32_t outer_isize = 1;
	for( int i= append_dims; i < 3; i++){
		uint32_t n = index_tensor->shape.dimension[i];
		outer_isize *= n;
		oparms->outshape.dimension[i-append_dims] = n;
	}
	uint32_t index_lastdim = 1;
	if( append_dims < 4){
		index_lastdim = index_tensor->shape.dimension[3];
		oparms->outshape.dimension[3-append_dims] = index_lastdim;
	}

	uint32_t inner_tsize = 1;
	for( int i = table_idx_dim+1; i < 4; i++){
		uint32_t n = table_tensor->shape.dimension[i];
		inner_tsize *= n;
		oparms->outshape.dimension[i] = n;
	}
	uint32_t table_entry_bytes = inner_tsize*oparms->elbytes;

	oparms->index_partn = index_lastdim;
	oparms->outer_index_size = outer_isize;
	oparms->inner_table_size = inner_tsize;
	oparms->table_entry_bytes = table_entry_bytes;
	oparms->output_partition_stride = table_entry_bytes * index_lastdim;

	// fill in the pointers in the partition descs
	//
	uint8_t const * tables = (uint8_t const *)table_tensor->data;
	for( int i =0; i < n_table_partn; i++){
		pdescs[i].partn_data = tables;
		tables += pdescs[i].partn_size * table_entry_bytes;
	}

	// one more thing to do:
	//  if index_partn > table_partn, we need to fill in the mapping parms
	// which are used as follows:
	//    if(  idx < oparms->div_break ){
	//          opart = idx /oparms->div_d;
	//    }else{
	//          opart = (idx -oparms->div_break)  /(oparms->div_d-1) + opparms->div_offs
	//	  }
	// div_d = 1 if, and only if, no mapping is needed.
	//
	oparms->div_break = index_lastdim;		// assume no break;
	oparms->div_d = 1;
	if( index_lastdim > n_table_partn ){	// will be used
		int q = index_lastdim / n_table_partn;
		int rem = index_lastdim - q*n_table_partn;
		if( rem != 0 ){
			q++;			// round the quotient up (guaranteed >= 2)
			oparms->div_break = q*rem;
			oparms->div_offs = rem;
		}
		oparms->div_d = q;
	}
	// thats' about it!

	return 0;
}

static int
__attribute__((unused))
table_execute( struct nn_node *self, struct nn_graph *nn)
{
	struct gather_table_variant_token  * var_token = (struct gather_table_variant_token *)self->opaque;
	if( var_token->is_gather){
		// non-partitioned 'table' is handled via 'gather'.
		return gather_execute( self, nn);
	}

	struct tensor const * index_tensor = self->inputs[0];
	struct tensor const * table_tensor = self->inputs[1];
	struct tensor const *const* optional_tensors = &self->inputs[2];
	struct tensor * out_tensor = self->outputs[0];

	int n_optional_inputs = self->n_inputs - 2;

	logmsg(nn,2,"table node %p execute",self);

	int elbytes = var_token->element_bytes;
	int data_type = var_token->element_typecode;

	if( var_token->is_quant){
		// copy min/max through
		tensor_copy( self->outputs[1], self->inputs[2]);
		tensor_copy( self->outputs[2], self->inputs[3]);
		// correct the scalar input pointer and count...
		n_optional_inputs -= 2;
		optional_tensors += 2;
	}


	// we definitely have a layout_tensor (or would not be here).
	struct tensor const * layout_tensor = optional_tensors[0];

	struct table_runstate rstate;
	rstate.opp.elbytes = elbytes;
	// work out all the details

	if( analyze_table_op( nn, index_tensor, table_tensor, layout_tensor, &rstate.opp)!= 0)
		return -1;

	// extract scalar parms where available
	int partn_strategy = 0;
	if( n_optional_inputs >= 2){
		partn_strategy = tensor_get_int32( optional_tensors[1], 0);
	}


	// create the output tensor
	if( tensor_out_prepare_normal_fromshape( out_tensor, &rstate.opp.outshape, data_type) != 0 ){
		return errlog(nn, "output too small");
	}
	///@@int check_range = self->padding == NN_PAD_SAME;
	int inner_size_bytes  = rstate.opp.table_entry_bytes;
	// select a copy func

	if( rstate.opp.index_partn <= rstate.opp.table_partn ){
		partn_strategy  = 0;	// use mod, it's all the same...
	}

	uint8_t *outp = (uint8_t*)out_tensor->data;
	int32_t const *idxp = (int32_t const*)index_tensor->data;
	//
	// loop through the partitions
	// @@ unfortunately we can't use the 'perform_lookup' functions for this;
	// unless we add both an index stride and an outer stride, which will
	// make them pretty inefficient for 'gather'.
	// TODO 'padding' is ignored here (always limits to range)

	int index_partn = rstate.opp.index_partn;

	//
	for(int ipart = 0; ipart < index_partn; ipart ++ ){
		int tpart;
		if( partn_strategy == 0){	// mod
			tpart = ipart;
			if( tpart >= rstate.opp.table_partn){
				tpart = tpart % rstate.opp.table_partn;
			}
		}else{
			int d = rstate.opp.div_d;
			int delt = ipart-rstate.opp.div_break;
			if( delt < 0){
				tpart = ipart/d;
			}else{
				tpart = delt/(d-1) + rstate.opp.div_offs;
			}
		}
		uint8_t *outp_part = outp + ipart*inner_size_bytes;

		// get the table for this partition
		int tsize = rstate.opp.pdesc[tpart].partn_size;
		uint8_t const * tbl_data = rstate.opp.pdesc[tpart].partn_data;

		for( int iout = 0; iout < rstate.opp.outer_index_size; iout++){
			int idx = idxp[ ipart+ iout * index_partn]; // read the value.
			idx = min_i32( tsize-1, max_i32(idx,0));		// clip it
			memcpy( outp_part, tbl_data + inner_size_bytes * idx,  inner_size_bytes);
			outp_part += rstate.opp.output_partition_stride;
		}
	}

	logmsg(nn,2,"table node %p complete",self);
	return 0;

}

///////////////////////////////////////////////////////////////////////
// generic copy routine.
///////////////////////////////////////////////////////////////////////
//  out :   num * elbytes  'blobs'   (output area)
//  indices:  num * int32   indexes,  each 0 .. TabN-1
//  tbl:     TabN * elbytes  'blobs'   (table input)
//  TabN	 -size of table
//  elbytes:   - size of blob
//  num:      - size of index
//
// functions with _limit suffix clip the index to range and return 0
// functions with _check suffix check the range, return i+1 if index at 'i' is out of range
// There are specialized versions for elbytes = 1,2,4,8,12,16,32.
//
static int perform_lookup_N_limit( uint8_t * out, int32_t const *indices, uint8_t const *tbl,
		int TabN, int elbytes, int num )
{
	for(int i =0; i < num; i++){
		int idx = indices[i];
		idx = min_i32( TabN-1, max_i32(idx,0));
		memcpy( out, tbl + idx * elbytes, elbytes);
		out += elbytes;
	}
	return 0;
}
static int perform_lookup_N_check( uint8_t * out, int32_t const *indices, uint8_t const *tbl,
		int TabN, int elbytes, int num )
{
	for(int i =0; i < num; i++){
		int idx = indices[i];
		if( (unsigned)idx >= (unsigned)TabN) return i+1;
		memcpy( out, tbl + idx * elbytes, elbytes);
		out += elbytes;
	}
	return 0;
}

// these are for use with HVX, and only when elbytes >=128
static int perform_lookup_bigN_limit( uint8_t * out, int32_t const *indices, uint8_t const *tbl,
		int TabN, int elbytes, int num )
{
	int nvcopy = (elbytes-1)>>7;		// number of copy ops,-1 (>=0)
	int bump1 = ((elbytes-1)&127)+1;	// amount to bump ptrs after first copy

	for(int i =0; i < num; i++){
		int idx = indices[i];
		idx = min_i32( TabN-1, max_i32(idx,0));
		uint8_t const *rptr  = tbl + idx * elbytes;
		uint8_t * wptr = out;
		q6op_vstu_AV( (HVX_Vector*)wptr, q6op_V_vldu_A((HVX_Vector const*)rptr));
		wptr += bump1;
		rptr += bump1;
		for( int k = 0; k <nvcopy; k++){
			q6op_vstu_AV( (HVX_Vector*)wptr, q6op_V_vldu_A((HVX_Vector const*)rptr));
			wptr += 128;
			rptr += 128;
		}
		out += elbytes;
	}
	return 0;
}
static int perform_lookup_bigN_check( uint8_t * out, int32_t const *indices, uint8_t const *tbl,
		int TabN, int elbytes, int num )
{
	int nvcopy = (elbytes-1)>>7;		// number of copy ops,-1 (>=0)
	int bump1 = ((elbytes-1)&127)+1;	// amount to bump ptrs after first copy

	for(int i =0; i < num; i++){
		int idx = indices[i];
		if( (unsigned)idx >= (unsigned)TabN) return i+1;
		uint8_t const *rptr  = tbl + idx * elbytes;
		uint8_t * wptr = out;
		q6op_vstu_AV( (HVX_Vector*)wptr, q6op_V_vldu_A((HVX_Vector const*)rptr));
		wptr += bump1;
		rptr += bump1;
		for( int k = 0; k <nvcopy; k++){
			q6op_vstu_AV( (HVX_Vector*)wptr, q6op_V_vldu_A((HVX_Vector const*)rptr));
			wptr += 128;
			rptr += 128;
		}
		out += elbytes;
	}
	return 0;
}
//
// specializations for a given data size
//
#define MAKE_LOOKUP_FUNC( NCODE,DTYPE)\
static int perform_lookup_##NCODE##_limit( uint8_t * out, int32_t const *indices, uint8_t const *tbl,\
   int TabN, int elbytes, int num )\
{\
	DTYPE *outp = (DTYPE*)out;\
	DTYPE const * tblp = (DTYPE const *)tbl;\
	int nh = num&~1;\
	for(int i =0; i < nh; i+=2){\
		int idx0 = indices[i];\
		int idx1 = indices[i+1];\
		idx0 = min_i32( TabN-1, max_i32(idx0,0));\
		idx1 = min_i32( TabN-1, max_i32(idx1,0));\
		outp[0] = tblp[idx0];\
		outp[1] = tblp[idx1];\
		outp += 2;\
	}\
	if(num&1){\
		int idx = indices[nh];\
		idx = min_i32( TabN-1, max_i32(idx,0));\
		outp[0] = tblp[idx];\
	}\
	return 0;\
}\
static int perform_lookup_##NCODE##_check( uint8_t * out, int32_t const *indices, uint8_t const *tbl,\
    int TabN, int elbytes, int num )\
{\
	DTYPE *outp = (DTYPE*)out;\
	DTYPE const * tblp = (DTYPE const *)tbl;\
	int nh = num&~1;\
	for(int i =0; i < nh; i+=2){\
		int idx0 = indices[i];\
		int idx1 = indices[i+1];\
		if( (unsigned)idx0 >= (unsigned)TabN) return i+1;\
		if( (unsigned)idx1 >= (unsigned)TabN) return i+2;\
		outp[0] = tblp[idx0];\
		outp[1] = tblp[idx1];\
		outp += 2;\
	}\
	if(num&1){\
		int idx = indices[nh];\
		if( (unsigned)idx >= (unsigned)TabN) return nh+1;\
		outp[0]= tblp[idx];\
	}\
	return 0;\
}


struct blob12 { uint32_t vals[3]; };
struct blob16 { uint64_t vals[2]; };
struct blob32 { uint64_t vals[4]; };

MAKE_LOOKUP_FUNC( 1,uint8_t)
MAKE_LOOKUP_FUNC( 2,uint16_t)
MAKE_LOOKUP_FUNC( 4,uint32_t)
MAKE_LOOKUP_FUNC( 8,uint64_t)
MAKE_LOOKUP_FUNC( 12,struct blob12)
MAKE_LOOKUP_FUNC( 16, struct blob16)
MAKE_LOOKUP_FUNC( 32,struct blob32)

// select a copy function, based on the element
// size and 'check_range'
//
static table_copy_funcp
select_copy_func( int elsize, int check_range)
{
	check_range = (check_range!=0)?1:0;

	static const table_copy_funcp pow2funcs[12] = {
			perform_lookup_1_limit, perform_lookup_1_check,
			perform_lookup_2_limit, perform_lookup_2_check,
			perform_lookup_4_limit, perform_lookup_4_check,
			perform_lookup_8_limit, perform_lookup_8_check,
			perform_lookup_16_limit, perform_lookup_16_check,
			perform_lookup_32_limit, perform_lookup_32_check,
	};
	if( elsize <= 32 ){
		int l2elsize = floor_log2(elsize|1);
		if( elsize== (1<<l2elsize)){
			return pow2funcs[ l2elsize*2 + check_range];
		}
		if(elsize == 12){
			return check_range? perform_lookup_12_check : perform_lookup_12_limit;
		}
	}
	// if we don't have a function for it, use generic
	if(elsize >=128)
		return check_range? perform_lookup_bigN_check : perform_lookup_bigN_limit;

	return check_range? perform_lookup_N_check : perform_lookup_N_limit;
}
const struct gather_table_variant_token
variant_token_for_Gather_f = {
		.element_bytes = sizeof(float),
		.element_typecode = NN_TYPE_FLOAT,
		.is_quant = 0,
		.is_gather = 1
};
const struct gather_table_variant_token
variant_token_for_Gather_int32 = {
		.element_bytes = sizeof(int32_t),
		.element_typecode = NN_TYPE_INT32,
		.is_quant = 0,
		.is_gather = 1
};

const struct gather_table_variant_token
variant_token_for_Gather_8 = {
		.element_bytes = sizeof(uint8_t),
		.element_typecode = NN_TYPE_QUINT8,
		.is_quant = 1,
		.is_gather = 1
};

// for 'table', the variant token is typically
// one of these; but if it has no extra
// inputs, the token is instead the
// corresponding 'gather' token.
// and the 'is_gather!=0' will cause it
// to immediately divert to gather
// at runtime. So table doesn't need to
// handle that case.
//

const struct gather_table_variant_token
variant_token_for_Table_f = {
		.element_bytes = sizeof(float),
		.element_typecode = NN_TYPE_FLOAT,
		.is_quant = 0,
		.is_gather = 0
};
const struct gather_table_variant_token
variant_token_for_Table_int32 = {
		.element_bytes = sizeof(int32_t),
		.element_typecode = NN_TYPE_INT32,
		.is_quant = 0,
		.is_gather = 0
};

const struct gather_table_variant_token
variant_token_for_Table_8 = {
		.element_bytes = sizeof(uint8_t),
		.element_typecode = NN_TYPE_QUINT8,
		.is_quant = 1,
		.is_gather = 0
};
//
// the 'check' functions set self->opaque to the variant token
// according to node_type.
// Also, for 'table', if the node has no extra inputs,
// the 'gather' variant is used instead
//
static int gather_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking gather node %p",self);

	int n_in = self->n_inputs;

	switch( self->node_type){
	 case OP_Gather_f:
		 self->opaque = (void*) & variant_token_for_Gather_f;
		 break;
	 case OP_Gather_int32:
		 self->opaque = (void*) & variant_token_for_Gather_int32;
		 break;
	 case OP_Table_f:
		 self->opaque = (void*)((n_in > 2) ? & variant_token_for_Table_f : &variant_token_for_Gather_f);
		 break;
	 case OP_Table_int32:
		 self->opaque = (void*)((n_in > 2) ? & variant_token_for_Table_int32 : &variant_token_for_Gather_int32);
		 break;
	 default:
		 return errlog(nn, "bad node_type = %d", self->node_type);
	}

	logmsg(nn,2,"gather %p check OK",self);
	return 0;
}
// this has 2 additional inputs and outputs
static int gather_q8_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking gather_8 node %p",self);

	if( self->node_type == OP_Gather_8
			|| self->n_inputs == 4 ){
		self->opaque = (void*)&variant_token_for_Gather_8;
	}else{
		self->opaque = (void*)&variant_token_for_Table_8;
	}
	logmsg(nn,2,"gather_8 %p check OK",self);
	return 0;
}


static int gather_dtor(struct nn_node *self, struct nn_graph *nn)
{
	self->opaque = NULL;
	return node_free_common(self,nn);
}
struct nn_node_ops nn_ops_for_Gather_f = {
	.execute = gather_execute,
	.check = gather_check,
	.ctor = node_alloc_common,
	.dtor = gather_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(2,4),
	.n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_Gather_int32 = {
	.execute = gather_execute,
	.check = gather_check,
	.ctor = node_alloc_common,
	.dtor = gather_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(2,4),
	.n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_Gather_8 = {
	.execute = gather_execute,
	.check = gather_q8_check,
	.ctor = node_alloc_common,
	.dtor = gather_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(4,6),
	.n_outputs = NN_IOCOUNT(3),
};



struct nn_node_ops nn_ops_for_Table_f = {
	.execute = table_execute,
	.check = gather_check,
	.ctor = node_alloc_common,
	.dtor = gather_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(2,4),
	.n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_Table_int32 = {
	.execute = table_execute,
	.check = gather_check,
	.ctor = node_alloc_common,
	.dtor = gather_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(2,4),
	.n_outputs = NN_IOCOUNT(1),
};
struct nn_node_ops nn_ops_for_Table_8 = {
	.execute = table_execute,
	.check = gather_q8_check,
	.ctor = node_alloc_common,
	.dtor = gather_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(4,6),
	.n_outputs = NN_IOCOUNT(3),
};
