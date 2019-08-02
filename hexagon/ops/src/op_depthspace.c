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
#include <quantize.h>
#include <math.h>
#include "hvx_inlines.h"

#include "nn_gentranspose.h"


struct blksiz {
	uint32_t blocksize_h;
	uint32_t blocksize_w;
	int errflag;
};

struct s2d_info {
	uint8_t elementsize;
	uint8_t dtype;
	uint8_t is_q;	// 1 if it has min/max
	uint8_t is_strategy_valid;
	struct shape inshape;
	struct shape outshape;
	struct blksiz block_size;
	struct nn_transpose_desc txdesc;
};



// NOTE: the 'blocksize' input tensor can be a scalar int32
// or it can be two values [blksizeH, blksizeW]
// (i.e. if the shape is [1,1,1,2])
//

static struct blksiz get_blksiz_from_tensor( struct tensor const * t);

//static void hvx_s2d_run_thread( struct nn_graph * nn, void * rstpv );

//
// space-to-depth:
// equivalent to:
//    (1)    input is    [b, hi, wi, di]       where hi = ho*N, wi=wo*N
//    (2)   relabel as   [b, ho, N , wo, N*di]
//    (3)   transpose to [b, ho, wo, N, N* di]
//    (4)   relabel as   [b, ho,  wo, N*N*do]
//
// DepthToSpace uses the general-transpose facility to execute the operation; a strategy
// is chosen via nn_transpose_analyze_direct, and this only needs to change if the
// input shape or block size change from the previous run.
//

static int depthspace_s2d_setup_strategy( struct nn_node *self, struct nn_graph *nn, struct blksiz const *bsp, int elementsize)
{
	struct s2d_info * info = (struct s2d_info*)self->opaque;
	const struct tensor *in_tensor = self->inputs[0];
	info->inshape = in_tensor->shape;
	int32_t block_size_h = bsp->blocksize_h;
	int32_t block_size_w = bsp->blocksize_w;

	uint32_t block_size_prod = mulu32_sat( block_size_h,block_size_w);

	int32_t out_height = info->inshape.height / (unsigned)block_size_h;
	int32_t out_width = info->inshape.width / (unsigned)block_size_w;
	int32_t out_depth = info->inshape.depth * block_size_prod;

	if (info->inshape.width != out_width* block_size_w ) return errlog(nn,"width must be multiple of block size");
	if (info->inshape.height  != out_height * block_size_h) return errlog(nn,"height must be multiple of block size");

	// fill in the output shape etc

	info->outshape.batches= info->inshape.batches;
	info->outshape.height = out_height;
	info->outshape.width = out_width;
	info->outshape.depth = out_depth;
	info->block_size.blocksize_h = block_size_h;
	info->block_size.blocksize_w = block_size_w;

	// make a transpose strategy
	// 'tx_dims' are the input shape for the purpose of expressing the operation as a transpose;
	// we want to exchange the two middle axes (bock_size_h and out_width).
	//
	uint32_t tx_dims[4] = { info->inshape.batches* out_height, block_size_h, out_width, block_size_w * info->inshape.depth};
	int32_t perm[3] = {1,0,2};
	int res = nn_transpose_analyze_direct( &info->txdesc, elementsize,
			perm, 3,
			tx_dims, 4 );
	if( res!=0) return errlog(nn, "failed to make transpose strategy");

	if( info->txdesc.buffer_needed > nn->scratch_size){
		if( nn_scratch_grow(nn, info->txdesc.buffer_needed)!=0)
			return errlog(nn,"couldn't grow scratch for transpose\n");
	}
	info->is_strategy_valid = 1;
	return 0;
}


static int depthspace_s2d_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *block_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];

	struct blksiz bs =  get_blksiz_from_tensor( block_tensor);
	if( bs.errflag ) return errlog(nn,"bad block size");
	
	struct s2d_info * info = (struct s2d_info*)self->opaque;
	int elementsize = info->elementsize;
	int dtype = info->dtype;

	// check old strategy; make a new one if it's not applicable.
	if( !info->is_strategy_valid
		|| !shape_matches( & in_tensor->shape, &info->inshape)
		|| bs.blocksize_h != info->block_size.blocksize_h
		|| bs.blocksize_w != info->block_size.blocksize_w){
		int res = depthspace_s2d_setup_strategy( self,nn, &bs, elementsize);
		if( res != 0) return -1;
	}

	// size the tensors
	if (tensor_out_prepare_normal_fromshape(out_tensor,&info->outshape,dtype)!=0){
		return errlog(nn,"failed to prepare output");
	}
	if( info->is_q){
		tensor_copy(self->outputs[1],self->inputs[2]);
		tensor_copy(self->outputs[2],self->inputs[3]);
	}
	if( dtype == NN_TYPE_QUINT16 && in_tensor->format.type == NN_TYPE_QINT16)
		out_tensor->format.type = NN_TYPE_QINT16;
	// execute it

	nn_transpose_execute( nn, &info->txdesc, nn->scratch, out_tensor->data, in_tensor->data );
	return 0;
}



///////////////////////////////////////////////////////////////////////
//
// depth-to-space:
// equivalent to:
//    (1)    input is    [b, hi, wi, di]       where di = Nh*Nw*do
//    (2)   relabel as   [b, hi, wi, Nh, Nw*do]
//    (3)   transpose to [b, hi, Nh, wi, Nw* do]
//    (4)   relabel as   [b, hi*Nh,  wi*Nw,  do]
//
//
// We can also crop the height/width on both sides.
//
// (notes on how to do this using hvx '2d memcpy')
//
// Without cropping, this can be be viewed as a generic 'strided copy'
//
//    #          in_stride                   out_stride
//    batches    hi*wi*di                      ho*wo*do = hi*wi*di
//    hi            wi*di                      Nh*wo*do
//    wi               di                      Nw*do
//    Nh             di/Nh = Nw*do             wo*do  = wi*Nw*do
//    Nw*do              1                       1
//
// using '2d memcpy' we can do this as
//    - core operation is to copy Nw*do wide, and 'batches*hi' high, with source
//       pitch of wi*di and dest pitch of Nh*wo*do.
//    - repeat Nh times * wi times.
//
//
// With cropping, this is more complicated: the strided copy is non-uniform, but the strides are
//  constant as below (note ho <= Nh*hi, wo <= Nw*wo,  di == Nh*Nw*do); the Nw*do loop has been broken
// into two adjacent loops of length Nw and do.
//
//    #          in_stride                   out_stride
//    batches    hi*wi*di                      ho*wo*do
//    hi [1]        wi*di                       Nh*wo*do
//    wi             di                         Nw*do
//    Nh             di/Nh = Nw*do               wo*do
//    Nw[2]          do                           do
//    do             1                            1
//
// Notes:
//   [1] the 'hi' loop may be reduced in length to hi-1 or hi-2, depending on
//       cropping and on which iteration of the 'Nh' loop we are at
//   [2] the Nw loop may be reduced in length  depending on cropping
//        and on which iteration of the wi loop we are in.
//
// So, this can be done as a series of 2d copies, with width = Nw*do, height = hi-{0,1,2}
//  input row stride = wi*di, output row stride = Nh*wo*do
//   -  This is repeated over Nh, and wi, and batches;
//   - in first/last index of wi, the 'width' of the 2d copy may be reduced due to w cropping.
//

//
// (below is a reference implementation using memcpy)
//


static int depthspace_d2s_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *block_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	struct s2d_info * info = (struct s2d_info*)self->opaque;

	struct blksiz bs =  get_blksiz_from_tensor( block_tensor);
	if( bs.errflag ) return errlog(nn,"bad block size");
	uint32_t block_size_h =  bs.blocksize_h;
	uint32_t block_size_w =  bs.blocksize_w;

	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_batches = in_tensor->shape.batches;
	const char *in = in_tensor->data;
	char *out_base = out_tensor->data;
	int top_crop = 0;
	int bottom_crop = 0;
	int left_crop = 0;
	int right_crop = 0;

	int elementsize = info->elementsize;
	uint32_t block_size_prod = mulu32_sat(block_size_h,block_size_w);

	uint32_t out_depth = (block_size_prod > 1 )? ( in_depth/block_size_prod): in_depth;

	if ( block_size_prod == 0 || block_size_prod > in_depth ||
		 out_depth * block_size_prod != in_depth ) return errlog(nn,"depth must be multiple of square of block size");

	int32_t copy_size = out_depth * elementsize * block_size_w;

	// check for cropping
	int32_t const * crop_info = NULL;
	int n_crop_in = (info->is_q)? 4:2;
	int any_cropping = 0;

	if( self->n_inputs > n_crop_in){
		struct tensor const *crop_tensor = self->inputs[n_crop_in];
		if( crop_tensor->shape.depth != 4 || crop_tensor->data_size != 4*sizeof(int32_t))
			return errlog(nn,"bad crop_tensor shape");
		crop_info = (int32_t const*) crop_tensor->data;
		for( int i = 0; i < 4 ; i++){
			int32_t c = crop_info[i];
			if( c < 0 )
				return errlog(nn,"bad crop tensor element %d at index %d", (int)c, i);
			if( c > 0) any_cropping = 1;
		}
		top_crop = crop_info[0];
		bottom_crop = crop_info[1];
		left_crop = crop_info[2];
		right_crop = crop_info[3];
	}

	int32_t out_batches = in_batches;
	int32_t out_height = in_height * block_size_h;
	int32_t out_width = in_width * block_size_w;
	if( any_cropping){
		out_height -= (top_crop + bottom_crop);
		out_width -= (left_crop + right_crop);
		if( out_height <= 0 || out_width <= 0) return errlog(nn, "cropping leaves no output");
	}
	int dtype = info->dtype;
	if (tensor_out_prepare_normal(out_tensor,out_batches,out_height,out_width,out_depth,dtype)!=0){
		return errlog(nn,"failed to prepare output");
	}
	if( dtype == NN_TYPE_QUINT16 && in_tensor->format.type == NN_TYPE_QINT16)
		out_tensor->format.type = NN_TYPE_QINT16;
	if( info->is_q){
		tensor_copy(self->outputs[1],self->inputs[2]);
		tensor_copy(self->outputs[2],self->inputs[3]);
	}
	// allow for top_crop or left_crop >= blocksize: effectively, trim the input,
	// and reduce the cropping by the block_size*trim; so that the remaining output
	// crop needed is always 0..blocksize-1.
	// it's impossible for the input dims to be trimmed to < 1 if the output cropping
	// leaves any result.
	//
	int in_height_trimmed = in_height;			// input height after trimming
	int in_width_trimmed = in_width;			// input width after trimming
	if( any_cropping){
		int trim_hw_offset = 0;						// # of elements to offset pointer for trimming.

		if( top_crop >= block_size_h){
			int crop_in = top_crop/(unsigned)block_size_h;
			in_height_trimmed  -= crop_in;
			top_crop -= crop_in * block_size_h;   // i.e. top_crop % block_size_h
			trim_hw_offset += crop_in * in_width*in_depth;
		}
		if( bottom_crop >= block_size_h){
			int crop_in = bottom_crop/(unsigned)block_size_h;
			in_height_trimmed  -= crop_in;
			bottom_crop -= crop_in * block_size_h;
		}
		if( left_crop >= block_size_w){
			int crop_in = left_crop/(unsigned)block_size_w;
			in_width_trimmed  -= crop_in;
			left_crop -= crop_in * block_size_w;
			trim_hw_offset += crop_in * in_depth;
		}
		if( right_crop >= block_size_w){
			int crop_in = right_crop/(unsigned)block_size_w;
			in_width_trimmed  -= crop_in;
			right_crop -= crop_in * block_size_w;
		}
		// offset for the upper-left 'input trim' if any
		in += elementsize * trim_hw_offset;
	}

	// # of bytes to copy at w = 0, allowing for left crop (and sometimes right, when input_width=1)
	int copy0_size = (block_size_w-left_crop - ((in_width_trimmed>1)?0:right_crop) )* out_depth*elementsize;
	// # of bytes to copy at w = in_width_trimmed-1
	// (only used if in_width_trimmed > 1)
	int copylast_size = (block_size_w-right_crop) *  out_depth*elementsize;
	int in_next_w = in_depth * elementsize;
#if 0
	// batch loop
	for (int b = 0; b < in_batches; b++) {
		// input height loop
		int ho = 0;		// output h index
		for (int h = 0; h < in_height_trimmed; h++) {
			// height block loop
			// normally 0 <= hb < blocksize; but skip top crop when h=0; skip bottom crop
			//
			int hb_init = (h==0)?top_crop:0;
			int hb_limit = min_i32( block_size_h, out_height+hb_init-ho);
			for(int  hb = hb_init; hb < hb_limit; hb++, ho++){

				char *outp = out_base + ( ho+ b*out_height) * out_width * out_depth * elementsize;
				char const *inp = in + (h + b*in_height)*in_width*in_next_w + copy_size*hb;

				// w = 0 (possibly cropped)
				//
				memcpy( outp, inp + left_crop * out_depth*elementsize, copy0_size);
				outp += copy0_size;
				inp += in_next_w;
				// all the rest of the w
				if (in_width_trimmed > 1){
					for( int w = 1; w < in_width_trimmed-1; w++ ){
						memcpy( outp, inp, copy_size );
						outp += copy_size;
						inp += in_next_w;
					}
					// w = in_width_trimmed-1
					memcpy(outp, inp, copylast_size);
				}
			}
		}
	}
#else
	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );
	int odsize = out_depth * elementsize;
	int owsize = out_width * odsize;
	// '2d vector memcpy' version
	// This is adapted directly from the commented-out  code above, and is not very good:
	//   - when block_size_h = 2, hbn will always be 1 or 2, it would be better to merge
	//     the copy0_size and copylast_size ops with the main ops when they are the same size
	//  but for sufficiently large w it should be pretty good.
	//

	// batch loop
	for (int b = 0; b < in_batches; b++) {
		// input height loop
		int ho = 0;		// output h index
		for (int h = 0; h < in_height_trimmed; h++) {
			// height block loop
			// normally 0 <= hb < blocksize; but skip top crop when h=0; skip bottom crop
			//
			int hb_init = (h==0)?top_crop:0;
			int hb_limit = min_i32( block_size_h, out_height+hb_init-ho);

			char const *inp0 = in + (h + b*in_height)*in_width*in_next_w + copy_size*hb_init;
			char *outp0 = out_base + ( ho+ b*out_height) * owsize;

			// do the 'copy0_size' copies, for w=0, as a 2d-memcopy across the hb loop (possibly cropped)
			// (do the first output on 'hbn' output rows)
			int hbn = hb_limit-hb_init;	 // # of h iterations here.
			nn_mcmanager_vmemcpy_2d( nn, &mcman,
					copy0_size, hbn,					// width, height of region
					outp0,  owsize,						// dest pointer, dest stride
					inp0 + left_crop * odsize, copy_size);		// src pointer, src stride

			if( in_width_trimmed >1 ){
				char *outp = outp0 + copy0_size;		// start at second w operation
				char const *inp = inp0 + in_next_w;
				// if more than 2, do the intermediate ones as 'hbn' 2d memcpy,
				// each generating one output row.
				if( in_width_trimmed >2){
					for( int hb =0; hb < hbn; hb++){
						nn_mcmanager_vmemcpy_2d( nn, &mcman,
								copy_size, in_width_trimmed-2,					// width, height of region
								outp + owsize*hb,  copy_size,	// dest pointer, dest stride
								inp + copy_size*hb, in_next_w);		// src pointer, src stride
					}
					outp += copy_size * (in_width_trimmed-2);
					inp += in_next_w * (in_width_trimmed-2);
				}
				// do the last output on 'hbn' output rows
				nn_mcmanager_vmemcpy_2d( nn, &mcman,
						copylast_size, hbn,					// width, height of region
						outp,  owsize,	// dest pointer, dest stride
						inp, copy_size);		// src pointer, src stride
			}
			ho += hbn;		// update output pos
		}
	}
	nn_mcmanager_wait( nn, &mcman );
#endif

	return 0;
}

// block_tensor is either [blksize] or [blksize_h, blksize_w]
// This function does
//   - basic check on shape of tensor
//   - verifies value(s) >=1
//   - maps a single value to two values
//
static
struct blksiz get_blksiz_from_tensor( struct tensor const * block_tensor)
{
	struct blksiz result;
	result.errflag = 1;
	int nd = block_tensor->shape.depth;
	if( nd >=1 && nd <= 2 && nd* sizeof(int32_t) == block_tensor->data_size){	// size is OK
		int32_t const * ptr = (int32_t const*)block_tensor->data;
		int bh = ptr[0];
		int bw = ptr[nd-1];
		if( bh >= 1 && bw >= 1){
			result.blocksize_h = bh;
			result.blocksize_w = bw;
			result.errflag = 0;
		}
	}
	return result;
}


static int depthspace_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking depthspace node %p",self);
	void * infop = nn_calloc(1,sizeof(struct s2d_info));
	if( infop == NULL){
		return errlog(nn,"alloc failed");
	}
	self->opaque = infop;
	struct s2d_info * info = (struct s2d_info*)infop;
	int elsize,dtype,isq;
	switch( self->node_type){
	 case OP_DepthToSpace_f:
	 case OP_SpaceToDepth_f:
		elsize = sizeof(float);
		dtype = NN_TYPE_FLOAT;
		isq = 0;
		break;
	 case OP_DepthToSpace_8:
	 case OP_SpaceToDepth_8:
		elsize = sizeof(uint8_t);
		dtype = NN_TYPE_QUINT8;
		isq = 1;
		break;
	 case OP_DepthToSpace_16:
	 case OP_SpaceToDepth_16:
		elsize = sizeof(uint16_t);
		dtype = NN_TYPE_QUINT16;
		isq = 1;
		break;
	 default:
		return errlog(nn,"bad node_type %d", (int)self->node_type);
	}
	info->elementsize = elsize;
	info->dtype = dtype;
	info->is_q = isq;

	logmsg(nn,2,"depthspace %p check OK",self);
	return 0;
}


// DepthToSpace can have 2 or 3 inputs

struct nn_node_ops nn_ops_for_DepthToSpace_f = {
	.execute = depthspace_d2s_execute,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(2,3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_SpaceToDepth_f = {
	.execute = depthspace_s2d_execute,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};
// DepthToSpace_8 can have 4 or 5 inputs

struct nn_node_ops nn_ops_for_DepthToSpace_8 = {
	.execute = depthspace_d2s_execute,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
};

struct nn_node_ops nn_ops_for_SpaceToDepth_8 = {
	.execute = depthspace_s2d_execute,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
};
struct nn_node_ops nn_ops_for_DepthToSpace_16 = {
	.execute = depthspace_d2s_execute,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
};
struct nn_node_ops nn_ops_for_SpaceToDepth_16 = {
	.execute = depthspace_s2d_execute,
	.check = depthspace_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_OUTPUT_USES_INPUT_RANGE,
};

