
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
 */

#include <nn_graph.h>
#include <string.h>
#include <stdio.h>
#include "quantize.h"

struct dim_limits {
	union {
		struct {
			int b_start, b_end;
			int h_start, h_end;
			int w_start, w_end;
			int d_start, d_end;
		};
		int dimlims[4][2];
	};
};

static void init_dim_limits( struct dim_limits * dlim, struct tensor const *tens )
{
	dlim->b_start = 0;   dlim->b_end = tens->shape.batches;
	dlim->h_start = 0;   dlim->h_end = tens->shape.height;
	dlim->w_start = 0;   dlim->w_end = tens->shape.width;
	dlim->d_start = 0;   dlim->d_end = tens->shape.depth;
}

// analyze the optional inputs and set up the dimension limits.
//  (call init_dim_limits first; the values in dlim are used to establish valid limits).
//
static void set_optional_dim_limits( struct nn_graph *nn,
		struct tensor const **tens_opts,
		int n_opts,
		struct dim_limits * dlim)
{
	if( n_opts >= 1 && tens_opts[0] != NULL){
		int dimno = tensor_get_int32( tens_opts[0], 0);
		if( dimno < 0 || dimno > 3) return;
		int dim_extent_lo = dlim->dimlims[dimno][0];
		int dim_extent_hi = dlim->dimlims[dimno][1];

		int dlim_lo = 0;	// default for 2nd optional
		if( n_opts >= 2 && tens_opts[1] != NULL){
			dlim_lo = tensor_get_int32( tens_opts[1], 0);
			dlim_lo = max_i32( dim_extent_lo, min_i32(dim_extent_hi-1, dlim_lo));
		}
		int dlim_hi = dlim_lo + 1;	// default for 3rd optional
		if( n_opts >= 3 && tens_opts[2] != NULL){
			dlim_hi = tensor_get_int32( tens_opts[2], 0);
			dlim_hi = max_i32( dlim_lo+1, min_i32(dim_extent_hi, dlim_hi));
		}
		logmsg(nn,1,"reporting only on %c dimension, range limit to [%d...%d]",
				"BHWD"[dimno], dlim_lo, dlim_hi-1 );
		dlim->dimlims[dimno][0] = dlim_lo;
		dlim->dimlims[dimno][1] = dlim_hi;
	}
}

static inline int pprint_generic(struct nn_node *self, struct nn_graph *nn,
		int elsize, void (printer_f)( struct nn_graph *, int,int,int,int, void const *))
{
	const struct tensor *in_tensor = self->inputs[0];
	struct dim_limits dlim;
	init_dim_limits(&dlim, in_tensor);
	int w,x,y,z;

	char const *in_data = (char const *)in_tensor->data;
	logmsg(nn,2,"pprinting node %p id %x",self,self->node_id);
	logmsg(nn,1,"bhwd = %d,%d,%d,%d", in_tensor->shape.batches,
			in_tensor->shape.height, in_tensor->shape.width, in_tensor->shape.depth );
	// pass optional parms to limit dimension extent.
	set_optional_dim_limits( nn, &self->inputs[1], self->n_inputs-1, &dlim);

	int depth = in_tensor->shape.depth;
	int row_stride = in_tensor->shape.width * in_tensor->shape.depth * elsize;
	int height = in_tensor->shape.height;

	for (w = dlim.b_start; w < dlim.b_end; w++) {
	 for (y = dlim.h_start; y < dlim.h_end; y++) {
	  char const * row = in_data + row_stride * (y + height*w);
	  for (x = dlim.w_start; x < dlim.w_end; x++){
	   for (z = dlim.d_start; z < dlim.d_end; z++){
		   (*printer_f)( nn, w,y,x,z, row + elsize * ( z + depth*x));
	   } //z
	  }//x
	 }// y
	}// w
	return 0;
}
//
// 'plug in' printer funcs
//
static inline void
print_one_8( struct nn_graph *nn, int w,int y,int x,int z, void const *p)
{
	logmsg(nn,1,"[%d,%d,%d,%d]: 0x%02x" ,w,y,x,z,*(uint8_t const *)p);
}
static inline void
print_one_32( struct nn_graph *nn, int w,int y,int x,int z, void const *p)
{
	logmsg(nn,1,"[%d,%d,%d,%d]: 0x%08x" ,w,y,x,z, (unsigned)*(uint32_t const *)p);
}

static inline void
print_one_f( struct nn_graph *nn, int w,int y,int x,int z, void const *p)
{
	logmsg(nn,1,"[%d,%d,%d,%d]: %f" ,w,y,x,z,*(float const *)p);
}

//
// 'instantiate
//
static int
pprint_8_execute( struct nn_node *self, struct nn_graph *nn)
{
	return pprint_generic( self,nn, sizeof(uint8_t), print_one_8);
}
static int
pprint_32_execute( struct nn_node *self, struct nn_graph *nn)
{
	return pprint_generic( self,nn, sizeof(uint32_t), print_one_32);
}
static int
pprint_f_execute( struct nn_node *self, struct nn_graph *nn)
{
	return pprint_generic( self,nn, sizeof(float), print_one_f);
}

//
// pprint a d32 array.
// if 'with_padding' is true, the padding is also printed
// (can including -ve h,w,b dimensions)
//

static  int pprint_d32_execute(struct nn_node *self, struct nn_graph *nn )
{
	const struct tensor *in_tensor = self->inputs[0];
	struct dim_limits dlim;
	init_dim_limits(&dlim, in_tensor);
	int w,x,y,z;
	int with_padding = (self->node_type == OP_PPrintWithPadding_8_d32)? 1: 0;


	uint8_t const *in_data = (uint8_t const *)in_tensor->data;
	logmsg(nn,2,"pprinting node %p id %x",self,self->node_id);
	logmsg(nn,1,"bhwd = %d,%d[%d]%d,%d[%d]%d,%d[%d]%d", in_tensor->shape.batches,
			in_tensor->format.height_pad[0],in_tensor->shape.height, in_tensor->format.height_pad[1],
			in_tensor->format.width_pad[0],	in_tensor->shape.width, in_tensor->format.width_pad[1],
			in_tensor->format.depth_pad[0],	in_tensor->shape.depth, in_tensor->format.depth_pad[1] );

	if( with_padding){
		// expand ranges to include padding
		dlim.h_start = -in_tensor->format.height_pad[0];
		dlim.h_end  += in_tensor->format.height_pad[1];
		dlim.w_start = -in_tensor->format.width_pad[0];
		dlim.w_end  += in_tensor->format.width_pad[1];
		dlim.d_start = -in_tensor->format.depth_pad[0];
		dlim.d_end  += in_tensor->format.depth_pad[1];

	}
	// pass optional parms to limit dimension extent.
	set_optional_dim_limits( nn, &self->inputs[1], self->n_inputs-1, &dlim);

	int batch_stride = tensor_batch_stride_d32( in_tensor);
	int row_stride = tensor_row_stride_d32( in_tensor);
	int d32_stride = tensor_d32_stride_d32( in_tensor);
	int h_pad_top = in_tensor->format.height_pad[0];
	int w_pad_left = in_tensor->format.width_pad[0];
	int d_pad_before = in_tensor->format.depth_pad[0];

	// addressing is done to support reading padding with out-of-range indices
	// (including negative indices, into 'left' padding)

	for (w = dlim.b_start; w < dlim.b_end; w++){
		for (y = dlim.h_start; y < dlim.h_end; y++){
			for (x = dlim.w_start; x < dlim.w_end; x++){
				uint8_t const * d_start = in_data +
						w*batch_stride + (y + h_pad_top)* row_stride
						+ (x + w_pad_left) * 32;
				for (z = dlim.d_start; z < dlim.d_end; z++){
				   unsigned zz = z + d_pad_before;		// 0 ..dtotal-1
				   uint8_t const * pos = d_start + (zz%32) + (zz/32)*d32_stride;
				   print_one_8( nn, w,y,x,z, pos);
				}
			} //x
	   } //y
	} //w
	return 0;
}


//
// inputs:
///  0 - a tensor
//  The rest of the inputs are optional integer scalars
//  which will limit the report along a specified dimension.
//    e.g.  [3] -> same as [3,0] , same as [3,0,1]:
//                 means print only d=0.
//          [2,5] same as [2,5,6]  means print only w=5
//          [1,0,3] means print only for h= 0,1,2
//
//   1 - (optional) int value 0..3 specify dimension to limit.
//        default is -1, meaning no limiting
//   2 - (optional) int value indicating index to start report on
//       specified dimension. default is 0
//   3 - (optional) int value indicating end of extend of report
//       on specified dimension (first index to exclude). Default
//       is 1 more than start index (only reports start index).
// If the range implied by inputs 2,3 exceeds the range of the array,
//    or implies an empty range, it is corrected to a non-empty valid range.
//




struct nn_node_ops nn_ops_for_PPrint_8 = {
	.execute = pprint_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,4),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_PPrint_32 = {
	.execute = pprint_32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,4),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_PPrint_f = {
	.execute = pprint_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,4),
	.n_outputs = NN_IOCOUNT(0),
};


// print d32, with and without padding.

struct nn_node_ops nn_ops_for_PPrint_8_d32 = {
	.execute = pprint_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,4),
	.n_outputs = NN_IOCOUNT(0),
};

struct nn_node_ops nn_ops_for_PPrintWithPadding_8_d32 = {
	.execute = pprint_d32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(1,4),
	.n_outputs = NN_IOCOUNT(0),
};
