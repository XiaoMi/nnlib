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
#include <stdio.h>

//
// QuantizedPad_8_d32 node
// Currently this is constrained to support only B,H,W padding (not d).
//
//
//
//   4 inputs:
//        0 : input tensor, qu8
//        1 : scalar float (input min)
//        2 : scalar float (input max)
//        3 : 'padding' tensor, int32 [1,1,n,2] where n = 1..4  (see input #1 of Pad_f)
//       4 : (optional) scalar float, value to pad (default 0.0).
//    3 output:
//        0: output tensor, qu8
//        1 : scalar float (output min)
//        2 : scalar float (output max)
//
// QuantizedPad_u16_d32
// QuantizedPad_16_d32
// Currently this is constrained to support only B,H,W padding (not d).
//  We need different types for u16 and s16, in order to figure out
//  how to generate the 'pad' value
//
// Strategy is agnostic to elementsize
//
//
// QuantizedPadForConv_8_d32
// This does a QuantizedPad8 operation, except:
//  (1) the only possible fill value is (quantized 0.0).
//  (2) the padding is determined by examining the input shape, the supplied filter and stride shape,
//      and the 'padding_mode' (which is any convolution padding mode, other than VALID).
//    The padding on h,w which produces the proper result (when a 'valid' convolution is done to the padded data)
//    is selected.
//   5 inputs:
//        0 : input tensor, qu8
//        1 : scalar float (input min)
//        2 : scalar float (input max)
//        3 : weights tensor (only the first 2 dims are examined; filt_height, filt_wid)
//        4 : stride tensor (only the h and w dims are examined: stride_ht, stride_wid)
//    3 output:
//        0: output tensor, qu8
//        1 : scalar float (output min)  (these are copied from input)
//        2 : scalar float (output max)
//
// There should be no need for a non-d32 version of this; it is only generated in prepare. When we find
// a d32 supernode with a filter size that may not work due to limitations of padding, we insert a
// QuantizedPadForConv_8_d32 ahead of it, and change it to the same conv with VALID padding.
//
//  QuantizedPadForConv_u16_d32:
//    Same thing but works on u16 data.

//
struct pad_s32_info {
	// these are set when the field is allocated in check()
	uint8_t elbytes;	// 1 or 2
	uint8_t dtype;		// NN_TYPE_UINT8 or NN_TYPE_QUINT8 or NN_TYPE_QINT8
	uint8_t is_padforconv;
	// fields to check if the strategy is valid.
	uint8_t strategy_valid;
	struct tensor in_tensor;	// copy of input tensor
	float in_min,in_max;	// input range
	float pad_val_f;
	uint32_t pad_amount[4][2];	// values from the pad tensor.

	// if it's a PadForConv, the next four fields store the last-seen filter
	// size and stride; and the pad_amount[] are the previously calculated padding.
	uint32_t prev_filt_h, prev_filt_w;
	uint32_t prev_stride_h, prev_stride_w;

	struct shape out_shape;
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	uint32_t pad_val;			// 8 or 16 bits, replicated to fill 32.

	unsigned obatch_core_len; 	// amount to fill each output batch which is padding.

	// H,W padding and the core copy are described 'per batch';

	int h_add_before_offset;	// offset to skip top padding in output
	int h_add_before_fill_len;	// 0, or amount to fill in top padding
	int h_add_after_offset;		// additional offset to reach bottom padding
	int h_add_after_fill_len;	// 0, or amount to fill in bottom padding.

	int core_copy_width;		// width of the 'core copy' in bytes
	int core_copy_height;	// height in rows (really, total # nd32 rows)
	int core_copy_dstoff;	// offset in bytes to leave room for padding on left

	// w_before uses core_copy_height, and is done to the output start address
	int w_add_before_width;	// 32 * amount of left-padding added
	// w_after uses core_copy_height, and is done to the output start address + core_copy_dstoff + core_copy_width
	int w_add_after_width;	// 32 * amount of right-padding added
};
struct pad_d32_runstate {
	struct pad_s32_info * info;
	nn_sem_t done_sem;
};
static int
pad32_setup_scaling( struct nn_graph * nn,struct nn_node *self, struct pad_s32_info *info );

//
// is strategy ok?
//  0 - no
//  1 - yes
//  < 0 - error
//
static int
pad32_check_strategy(struct nn_node *self, struct nn_graph *nn )
{
	struct pad_s32_info *info = (struct pad_s32_info*) self->opaque;
	if( info->strategy_valid==0)
		return 0;
	struct tensor const  *in_tensor = self->inputs[0];
	struct tensor const *in_min_tensor = self->inputs[1];
	struct tensor const *in_max_tensor = self->inputs[2];
	if( memcmp( in_tensor,&info->in_tensor,sizeof(struct tensor))!=0) return 0;

	float req_padval = 0.0f;

	if( !info->is_padforconv){
		struct tensor const * pads_tensor = self->inputs[3];
		//
		// check pad values
		//
		int rank = pads_tensor->shape.width;
		if( rank <1 || rank > 4 || pads_tensor->shape.depth!=2
				|| pads_tensor->data_size < sizeof(int32_t)*2*rank){
			return 0;
		}
		if( memcmp( pads_tensor->data,info->pad_amount[4-rank],sizeof(int32_t)*2*rank)!=0)
			return 0;
		for(int i=0; i < 4-rank; i++){
			if( info->pad_amount[i][0] !=0 || info->pad_amount[i][1] != 0){
				return 0;
			}
		}
		if(self->n_inputs >=5) req_padval = tensor_get_float( self->inputs[4],0);
	}else{
		// check filt size and stride are the same
		struct tensor const * filt_tensor = self->inputs[3];
		struct tensor const * stride_tensor = self->inputs[4];
		if( info->prev_filt_h != filt_tensor->shape.filt_height
			|| info->prev_filt_w != filt_tensor->shape.filt_width
			|| info->prev_stride_h != stride_tensor->shape.height
			|| info->prev_stride_w != stride_tensor->shape.width )
			return 0;
	}
	// all OK except maybe for scaling...
	// check that
	if( tensor_get_float(in_min_tensor,0) != info->in_min
	  || tensor_get_float(in_max_tensor,0) != info->in_max
	  || req_padval != info->pad_val_f )
	{
		return pad32_setup_scaling(nn,self,info);
	}
	return 1;
}
static int
pad32_build_strategy(struct nn_node *self, struct nn_graph *nn )
{
	struct pad_s32_info *info = (struct pad_s32_info*) self->opaque;
	int is_16 = info->elbytes==2;
	struct tensor const  *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];

	info->strategy_valid = 0;
	info->in_tensor = *in_tensor;

	if(!info->is_padforconv ){
		struct tensor const * pads_tensor = self->inputs[3];
		int rank = pads_tensor->shape.width;
		if( rank <1 || rank > 4 || pads_tensor->shape.depth!=2
				|| pads_tensor->data_size < sizeof(int32_t)*2*rank){
			return errlog(nn,"bad shape for dims tensor");
		}
		{
			int32_t const * pads_arr = (int32_t const*)pads_tensor->data;
			uint32_t *fillp = info->pad_amount[4-rank];
			if( rank < 4)
				memset( info->pad_amount, 0, sizeof(info->pad_amount[0])*(4-rank));
			for( int i=0;i < rank*2;i++){
				int p = pads_arr[i];
				if( p < 0) return errlog( nn,"negative value in pads tensor");
				fillp[i] = p;
			}
		}
	}else{
		// PadForConv
		// compute padding from filter shape, stride, self->padding.
		struct tensor const * filt_tensor = self->inputs[3];
		struct tensor const * stride_tensor = self->inputs[4];
		unsigned filt_h = filt_tensor->shape.filt_height;
		unsigned filt_w = filt_tensor->shape.filt_width;
		unsigned stride_h = stride_tensor->shape.height;
		unsigned stride_w = stride_tensor->shape.width;

		info->prev_filt_h = filt_h;
		info->prev_filt_w = filt_w;
		info->prev_stride_h = stride_h;
		info->prev_stride_w = stride_w;

		// do the padding calc
		int32_t h_pad_before, h_pad_after;
		int32_t w_pad_before, w_pad_after;

		int out_h = nn_pad_compute_outsize_and_pad( info->in_tensor.shape.height, filt_h, stride_h, self->padding, &h_pad_before, &h_pad_after );
		int out_w = nn_pad_compute_outsize_and_pad( info->in_tensor.shape.width, filt_w, stride_w, self->padding, &w_pad_before, &w_pad_after );
		if( out_h <1 || out_w <1) return errlog(nn,"invalid size/filter/padding detected in PadForConv");

		//
		// note that h_pad_after, w_pad_after may be <0; this means that the operation doesn't use all the input (usually due to a large stride)
		// and we treat those as 0 here.
		if ( h_pad_after < 0) h_pad_after = 0;
		if ( w_pad_after < 0) w_pad_after = 0;

		info->pad_amount[1][0] = h_pad_before;
		info->pad_amount[1][1] = h_pad_after;
		info->pad_amount[2][0] = w_pad_before;
		info->pad_amount[2][1] = w_pad_after;

		logmsg(nn,3,"PadForConv: input is (%d,%d) - filter (%d,%d) stride (%d,%d) pad mode = %d; pad by h=%d:%d w = %d:%d",
				(int) info->in_tensor.shape.height, (int) info->in_tensor.shape.width,
				filt_h, filt_w, stride_h, stride_w, (int)self->padding,
				(int)h_pad_before, (int)h_pad_after,(int)w_pad_before, (int)w_pad_after);

	}
	int elbytes = info->elbytes;

	// Work out the padded shape.
	// Be vigilant against overflow.
	unsigned total = elbytes;
	for( int i = 0; i < 4; i++){
		unsigned pad0 = info->pad_amount[i][0];
		unsigned pad1 = info->pad_amount[i][1];
		uint64_t padt = (uint64_t)pad0 + (uint64_t)pad1;
		uint64_t newdim = padt + (uint64_t)info->in_tensor.shape.dimension[i];
		if( newdim > 0x7fffffffu) return errlog(nn,"overflow in pad dimension %d", i);
		info->out_shape.dimension[i]= (uint32_t)newdim;
		total = mulu32_sat( total, (uint32_t)newdim);
		if( i == 3 && (unsigned)padt != 0)
			return errlog(nn,"can't pad in depth dimension");
	}
	if( total >=0x80000000u){
		return errlog(nn,"insanely large padded size (>=2GB)");
	}
	logmsg(nn,3,"pad_d32: elbytes=%d in_shape = (%d %d %d %d) padding = (%d:%d %d:%d %d:%d %d:%d) out=(%d %d %d %d)",
			elbytes, (int)info->in_tensor.shape.batches,(int)info->in_tensor.shape.height,
			(int)info->in_tensor.shape.width, (int)info->in_tensor.shape.depth,
			(int)info->pad_amount[0][0], (int)info->pad_amount[0][1],
			(int)info->pad_amount[1][0], (int)info->pad_amount[1][1],
			(int)info->pad_amount[2][0], (int)info->pad_amount[2][1],
			(int)info->pad_amount[3][0], (int)info->pad_amount[3][1],
			(int)info->out_shape.batches, (int)info->out_shape.height,
			(int)info->out_shape.width, (int)info->out_shape.depth);

	// set up output array
	int w_out_before_pad = 4;
	int h_padding  = 4;
	int w_out_after_pad = (-(info->out_shape.width + w_out_before_pad))&3;
	int d_out_after_pad = (-info->out_shape.depth)& 31;

	int res = tensor_out_prepare_padded_d32( out_tensor,
			info->out_shape.batches,
			info->out_shape.height, h_padding, h_padding,
			info->out_shape.width, w_out_before_pad, w_out_after_pad,
			info->out_shape.depth, 0, d_out_after_pad, info->dtype );
	if( res != 0){
		return errlog(nn,"output too small");
	}
	info->tin = (is_16?nn_tensor_addressing_d32_16b:nn_tensor_addressing_d32)( &info->in_tensor);
	info->tout = (is_16?nn_tensor_addressing_d32_16b:nn_tensor_addressing_d32)( out_tensor);

	int wbytes = 32*elbytes;
	// figure out how many bytes we need to zero in a batch, starting from the first 'real' byte
	// (thus skipping all the top padding, and the left padding on first d32 slice, and right on last d32 slice).
	{
		int out_nd32 = info->tout.nd32;
		int out_d32_stride = info->tout.d32_stride;
		int od32_rows = info->out_shape.height * out_nd32;
		int core_width = info->out_shape.width * wbytes;
		info->obatch_core_len =  core_width + (od32_rows-1)*out_d32_stride;
		int h_add_before = info->pad_amount[1][0];
		int od32_hbefore = out_nd32 * h_add_before;
		info->h_add_before_offset = out_d32_stride * od32_hbefore;
		info->h_add_before_fill_len = 0;
		if( h_add_before > 0){
			info->h_add_before_fill_len = core_width + (od32_hbefore -1)*out_d32_stride;
		}

		int h_add_after = info->pad_amount[1][1];
		int od32_hafter = out_nd32 * h_add_after;
		// h_add_after_offset is where the bottom padding rows are, relative to adding h_add_before_offset
		info->h_add_after_offset = out_d32_stride * out_nd32 * info->in_tensor.shape.height;
		info->h_add_after_fill_len = 0;
		if( h_add_after > 0){
			info->h_add_after_fill_len = core_width + (od32_hafter -1)*out_d32_stride;
		}

		info->core_copy_width = info->in_tensor.shape.width*wbytes;
		info->core_copy_height = out_nd32 * info->in_tensor.shape.height;
		info->core_copy_dstoff = info->pad_amount[2][0]*wbytes;		// offset into dest
		info->w_add_before_width = info->pad_amount[2][0]*wbytes;
		info->w_add_after_width = info->pad_amount[2][1]*wbytes;
		/*
		printf("out_d32_stride = %d; out_nd32= %d od32_rows = %d core_wid = %d\n",
				out_d32_stride, out_nd32, od32_rows, core_width);
		printf("h_add_after_offs = %d, h_add_after_fill_len=%d\n",
				(int)info->h_add_after_offset,(int)info->h_add_after_fill_len);
		printf("obatch_core_len = %d h_add_before_offs = %d, h_add_before_fill_len=%d\n",
				(int)info->obatch_core_len, (int)info->h_add_before_offset,(int)info->h_add_before_fill_len);
		printf("core_copy_ width = %d height=%d  dstoff = %d\n",
			(int)info->core_copy_width, (int)info->core_copy_height,(int)info->core_copy_dstoff);
		printf("w_add_before_after = %d %d\n", (int)info->w_add_before_width, (int)info->w_add_after_width );
		*/
	}
	int k = pad32_setup_scaling(nn,self,info);
	if(k ==0) info->strategy_valid = 1;
	return k;
}

static int
pad32_setup_scaling( struct nn_graph * nn,struct nn_node *self, struct pad_s32_info *info )
{
	float pad_val_f = 0.0f;
	float in_min = tensor_get_float(self->inputs[1],0);
	float in_max = tensor_get_float(self->inputs[2],0);

	if( !info->is_padforconv && self->n_inputs >= 5){
		pad_val_f =  tensor_get_float( self->inputs[4],0);
		// if there's an input[5] and it's non zero, it is disallowed
		// for the pad to be outside the range at all.
		if (self->n_inputs >= 6 && tensor_get_int32(self->inputs[5],0)!=0) {
			if (pad_val_f < in_min || pad_val_f > in_max) {
				return errlog(nn, "Pad value (%f) is outside the range of inputs (%f,%f)", pad_val_f, in_min, in_max);
			}
		}
	}

	info->in_min = in_min;
	info->in_max = in_max;
	info->pad_val_f = pad_val_f;
	int is_s16 = info->dtype == NN_TYPE_QINT16;

	uint32_t padding;
	if( fabsf(pad_val_f) != INFINITY ){
		float range = in_max-in_min;
		if( info->elbytes == 1){	// u8 scaling
			int qval = roundf_i32((pad_val_f-in_min)*255.0f/range);
			if( qval < -4 || qval > 259){
				logmsg(nn,0,"NOTE: pad val %f quantizes to %d", pad_val_f,qval);
			}
			padding = Q6_R_vsplatb_R( saturate_u8(qval));
		}else{
			// we assume for s16, that max = -min. So, 0.0 converts
			// to 0x8000 and will be changed to 0 later.
			int qval = roundf_i32((pad_val_f-in_min)*65536.0f/range);
			int delta = is_s16 ? 32768:0;
			if( qval < -1024 || qval > (65535+1024)){
				logmsg(nn,0,"NOTE: pad val %f quantizes to %d", pad_val_f,qval-delta);
			}
			qval = saturate_u16( qval)^delta;
			padding = Q6_R_combine_RlRl( qval,qval);
		}
	}else{
		// +/- inf - map to max or min.
		padding = (pad_val_f < 0)? 0 : 0xFFFFFFFFu;
		if( is_s16 )
			padding ^= 0x80008000u;	// convert to signed range
	}
	info->pad_val = padding;
	return 0;
}

static void pad32_execute(struct nn_graph * nn, struct pad_s32_info * info);

static int pad_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	int k = pad32_check_strategy(self,nn);
	if( k <= 0){
		if (k == 0){
			k = pad32_build_strategy( self, nn);
		}
		if( k < 0) return k;
	}
	struct pad_s32_info * info= (struct pad_s32_info *)self->opaque;
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	pad32_execute( nn, info);
	tensor_set_single_float(out_min_tensor, info->in_min);
	tensor_set_single_float(out_max_tensor, info->in_max);

	return 0;

}

// do the whole thing using mcmanager ops, so it will be done
// by vector threads
//
static void
pad32_execute(struct nn_graph * nn, struct pad_s32_info * info)
{
	struct nn_memcpy_manager  mcman;
	nn_mcmanager_init(nn, &mcman );

	unsigned pad_val = info->pad_val;

	int b_add_before = info->pad_amount[0][0];	// batches pad before
	int obatches = info->out_shape.batches;
	int ibatches = info->in_tensor.shape.batches;

	uint8_t *obat_ptr0 = info->tout.data;
	uint8_t const* ibat_ptr0 = info->tin.data ;

	int core_copy_width = info->core_copy_width;
	int core_copy_height = info->core_copy_height;
	int out_d32_stride = info->tout.d32_stride;


	for( int obat = 0; obat < obatches; obat++ ){
		uint8_t *obat_ptr = obat_ptr0 + info->tout.batch_stride * obat;
		int ibat = obat-b_add_before;
		if( (unsigned)ibat >= (unsigned)ibatches){	// is a padding batch
			nn_mcmanager_vmemset32(nn, &mcman, obat_ptr, pad_val,info->obatch_core_len );
		}else{
			int h_before_len = info->h_add_before_fill_len;
			if( h_before_len > 0){
				nn_mcmanager_vmemset32(nn, &mcman, obat_ptr, pad_val, h_before_len );
				obat_ptr += info->h_add_before_offset;
			}
			int h_after_len = info->h_add_after_fill_len;
			if( h_after_len > 0 ){
				nn_mcmanager_vmemset32(nn, &mcman, obat_ptr + info->h_add_after_offset, pad_val, h_after_len );
			}

			// core 2d copy
			uint8_t const * ibat_ptr = ibat_ptr0 + info->tin.batch_stride * ibat;
			uint8_t *obat_copy_at = obat_ptr + info->core_copy_dstoff;	// dest rectangle

			//printf("core copy: [%d x %d] @ %p stride = %d\n", core_copy_height, core_copy_width, obat_copy_at, out_d32_stride);

			nn_mcmanager_vmemcpy_2d(nn, &mcman,
					core_copy_width,	// width of copy
					core_copy_height,	// height
					obat_copy_at,			// dest rectangle
					out_d32_stride,			// stride
					ibat_ptr,				// source rectangle
					info->tin.d32_stride);	// stride

			int w_add_bef_wid = info->w_add_before_width;
			if( w_add_bef_wid > 0){
				nn_mcmanager_vmemset32_2d( nn, &mcman,
						obat_ptr, pad_val, w_add_bef_wid, core_copy_height, out_d32_stride);
			}
			int w_add_aft_wid = info->w_add_after_width;
			if( w_add_aft_wid > 0){
				//printf("wafter copy: [%d x %d] @ %p stride = %d\n", core_copy_height,w_add_aft_wid, obat_copy_at + core_copy_width, out_d32_stride);
				nn_mcmanager_vmemset32_2d( nn, &mcman,
						obat_copy_at + core_copy_width, pad_val, w_add_aft_wid, core_copy_height,out_d32_stride );
			}
		}
	}
	nn_mcmanager_wait(nn,&mcman);
}

static int pad_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	struct pad_s32_info *info = (struct pad_s32_info *)nn_calloc(1,sizeof(struct pad_s32_info));
	if( info == NULL) return errlog(nn,"calloc failed");
	self->opaque = info;
	int dtype,elbytes;
	int is_pad_for_conv =0;

	switch( self->node_type){
		case OP_QuantizedPadForConv_8_d32:
			is_pad_for_conv = 1;
			/* no break */
		case OP_QuantizedPad_8_d32:
			dtype = NN_TYPE_QUINT8;
			elbytes=1;
			break;
		case OP_QuantizedPadForConv_u16_d32:
			is_pad_for_conv = 1;
			/* no break */
		case OP_QuantizedPad_u16_d32:
			dtype = NN_TYPE_QUINT16;
			elbytes=2;
			break;
		case OP_QuantizedPad_16_d32:
			dtype = NN_TYPE_QINT16;
			elbytes=2;
			break;
		default:
			return errlog(nn,"bad node_type %d",(int)self->node_type);
	}
	info->is_padforconv = is_pad_for_conv;
	info->dtype = dtype;
	info->elbytes = elbytes;
	return 0;
}



struct nn_node_ops nn_ops_for_QuantizedPad_8_d32 = {
	.execute = pad_d32_execute,
	.check = pad_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,6),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedPad_16_d32 = {
	.execute = pad_d32_execute,
	.check = pad_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,6),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};
struct nn_node_ops nn_ops_for_QuantizedPad_u16_d32 = {
	.execute = pad_d32_execute,
	.check = pad_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,6),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};


struct nn_node_ops nn_ops_for_QuantizedPadForConv_8_d32 = {
	.execute = pad_d32_execute,
	.check = pad_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedPadForConv_u16_d32 = {
	.execute = pad_d32_execute,
	.check = pad_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};
