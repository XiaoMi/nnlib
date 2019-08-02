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
#include <math.h>
#include <quantize.h>
#include "nn_oper16.h"
//
// This contains conversions to/from 'i16 flat tensors'.
// These tensors have 'float' min and max, but it's assumed that min = -max;
// i.e. there is no zero offset, 'min' is the value represented by -0x8000,
// and max is the value represented by 0x7fff + 1
////////////////////////////////////////////////////////////////////////////
//  Dequantize_16:
///     input 0:  i16 flat tensor, any shape
//      input 1,2: scalar float, min, max
//       output 0: float tensor, same shape as input 0.
//
static int dequant_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor  * out_tensor = self->outputs[0];
	int is_u16 = self->node_type == OP_Dequantize_u16;
	
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_FLOAT) != 0){
		return errlog(nn,"output too small");
	}
	float minval = tensor_get_float( self->inputs[1], 0 );
	float maxval = tensor_get_float( self->inputs[2], 0 );
	float scale = (maxval-minval) * (float)( 1.0/65536.);
	int zero = 0x8000;
	uint32_t xor = 0x80008000;
	if( is_u16){
		xor = 0;
		zero = saturate_u16( 0.5f + -minval/scale);
	}
	
	float * outp = (float*) out_tensor->data;
	int32_t const *inp = (int32_t const*)in_tensor->data;		// read in pairs
	int count = tensor_element_count( in_tensor);
	
	int npair =count>>1;
	for( int i =0; i < npair; i++ ){
		uint32_t xin = *inp++ ^ xor;
		outp[0] = scale * (float)((uint16_t)xin - zero);
		outp[1] = scale * (float)((uint16_t)(xin>>16) - zero);
		outp += 2;
	}
	if( count & 1 ){
		uint32_t xin = *inp ^ xor;
		outp[0] = scale * (float)((uint16_t)xin - zero);
	}
	return 0;
}


struct nn_node_ops nn_ops_for_Dequantize_16 = {
	.execute = dequant_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),

};
struct nn_node_ops nn_ops_for_Dequantize_u16 = {
	.execute = dequant_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(1),
};
///////////////////////////////////////////////////////
//  Quantize_16:
///     input 0:  float flat tensor, any shape
//      input 1,2: scalar float, min, max
//      output 0: int16_t tensor, same shape as input 0.
//      output 1,2: scalar float, min, max
//  The output 'max' will be max(in_max, -in_min)
//  and the output min will be -out_max.
//  
//
static int quant_x16_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor  * out_tensor = self->outputs[0];
	int is_u16 = (self->node_type!=OP_Quantize_16);
	
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, (is_u16? NN_TYPE_QUINT16: NN_TYPE_QINT16)) != 0){
		return errlog(nn,"output too small");
	}
	float in_minval = tensor_get_float( self->inputs[1], 0 );
	float in_maxval = tensor_get_float( self->inputs[2], 0 );
	float out_minval, out_maxval, out_scale;
	int out_offset = 0;	// amount to add before saturating to i16
	int xor_val = 0;	// xor with this after saturating
	if( !is_u16){
		out_maxval = fmaxf( -in_minval, in_maxval);
		out_maxval = fmaxf(out_maxval, 1e-5f);
	
		out_maxval = flt_round_up_4eps(out_maxval);	// round up to a value with 2 zeros at the bottom.
		out_minval = -out_maxval;
	}else{
		out_minval = fminf(in_minval, 0.0f);
		out_maxval = fmaxf(in_maxval, 1e-5f);
		if( out_minval < 0.0f){
			if( out_maxval < out_minval * (float)(-1./65535.)){
				out_maxval = out_minval * (float)(-1./65535.);
			}else{
				float rmn = out_minval, rmx = out_maxval;
				adjust_minmax_for_zero_16b( &rmn, &rmx);
				out_minval = rmn; out_maxval = rmx;
			}
		}
	}
	out_scale = 65536.0f/(out_maxval-out_minval);
	if( is_u16){
		int out_zero = saturate_u16( (int)(0.5f -out_minval * out_scale));
		out_offset = out_zero-32768;
		xor_val = 32768;
	}
	tensor_set_single_float( self->outputs[1], out_minval );
	tensor_set_single_float( self->outputs[2], out_maxval );
	
	
	float const * inp = (float const *) in_tensor->data;
	int16_t *outp = (int16_t *)out_tensor->data;
	int count = tensor_element_count( in_tensor);
	
	for( int i = 0; i < count; i++ ){
		int ival = roundf_i32( out_scale * inp[i]) + out_offset;
		outp[i] =  saturate_i16(ival) ^ xor_val;
	}
	return 0;
}

struct nn_node_ops nn_ops_for_Quantize_16 = {
	.execute = quant_x16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_Quantize_u16 = {
	.execute = quant_x16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
///////////////////////////////////////////////////////
//  Convert_8_16:
///     input 0: u8 flat tensor, any shape
//      input 1,2: scalar float, min, max
//      output 0: int16_t tensor, same shape as input 0.
//      output 1,2: scalar float, min, max
//
//   The conversion is done via
//      out = (in-in_zero)*k
//  ... where k is something in the range 128.. 256 (according to in_zero)
//   such that no overflow is possible.
// (this implementation is free of rounding errors, but it can produce an
//  output range which is a bit larger than needed).
//
//
static int cvt_8_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor  * out_tensor = self->outputs[0];
	
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QINT16) != 0){
		return errlog(nn,"output too small");
	}
	float in_minval = tensor_get_float( self->inputs[1], 0 );
	float in_maxval = tensor_get_float( self->inputs[2], 0 );
	
	int zerolev = saturate_u8( roundf_i32( in_minval * -255.0f/(in_maxval-in_minval) ));
	// find multiplier 'k'
	int k;
	if (zerolev <128 ){		// k defined by upper limit, k = 128..255
		k = 32767u/(255-zerolev);
	}else{					// k defined by lower limit k = 128..256
		k = 32768u/zerolev;
	}
	// now work out the output range...
	float outmax = 32768.0f*(in_maxval-in_minval)/(255.0f* (float)k); 
	
	tensor_set_single_float( self->outputs[1], -outmax );
	tensor_set_single_float( self->outputs[2], outmax );

	uint8_t const * inp = (uint8_t const *) in_tensor->data;
	int16_t *outp = (int16_t *)out_tensor->data;
	int count = tensor_element_count( in_tensor);
	
	for(int i =0; i < count; i++ ){
		outp[i] = k * (inp[i]-zerolev);
	}
	return 0;
}	
///////////////////////////////////////////////////////
//  Convert_8_16u:
///     input 0: u8 flat tensor, any shape
//      input 1,2: scalar float, min, max
//      output 0: qu16 tensor, same shape as input 0.
//      output 1,2: scalar float, min, max
//
//   The conversion is done via
//      out = in[i] * 257  (i.e. replicate each byte to upper & lower)
// So the output range is almost identical to the input range.
//
static int cvt_8_u16_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor  * out_tensor = self->outputs[0];

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT16) != 0){
		return errlog(nn,"output too small");
	}
	float in_minval = tensor_get_float( self->inputs[1], 0 );
	float in_maxval = tensor_get_float( self->inputs[2], 0 );

	float out_minval = in_minval;
	// adjust upper range; in_maxval is for code 255->65535; we want the value
	// for code 65536.
	float out_maxval = in_maxval + (in_maxval-in_minval)*(float)(1./65536.0);

	tensor_set_single_float( self->outputs[1], out_minval );
	tensor_set_single_float( self->outputs[2], out_maxval );

	// do 8 at once
	uint64_t const * inp = (uint64_t const *) in_tensor->data;
	uint64_t *outp = (uint64_t *)out_tensor->data;
	int count = tensor_element_count( in_tensor);
	count = (count+7)/8u;
	if( count > 0){
		uint64_t inval0 = *inp++;
		uint64_t v16_0 = Q6_P_vzxtbh_R((uint32_t)inval0);	// zero extend
		uint64_t v16_1 = Q6_P_vzxtbh_R((uint32_t)(inval0>>32));

		for(int i =0; i < count-1; i++ ){
			uint64_t inval = *inp++;
			outp[0] = Q6_P_shuffeb_PP( v16_0,v16_0);		// replicate the bytes
			outp[1] = Q6_P_shuffeb_PP( v16_1,v16_1);
			v16_0 = Q6_P_vzxtbh_R((uint32_t)inval);
			v16_1 = Q6_P_vzxtbh_R((uint32_t)(inval>>32));
			outp += 2;
		}
		outp[0] = Q6_P_shuffeb_PP( v16_0,v16_0);		// replicate the bytes
		outp[1] = Q6_P_shuffeb_PP( v16_1,v16_1);
	}
	return 0;
}


struct nn_node_ops nn_ops_for_Convert_8_16 = {
	.execute = cvt_8_16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_Convert_8_u16 = {
	.execute = cvt_8_u16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};
///////////////////////////////////////////////
//
//
// Converting 16 to 8
//
///////////////////////////////////////////////////////
//  Convert_16_8:
//  Convert_u16_8:
//     input 0: s16 (or u16) flat tensor, any shape
//      input 1,2: scalar float, min, max
//      optional inputs:
//      input 3,4: scalar float, min, max (actual data range)
//
//      output 0: qu8 tensor, same shape as input 0.
//      output 1,2: scalar float, min, max
// Inputs 3,4 are the actual range you
// want at the output. This range will be corrected to
// have a proper zero, and may be enlarged if needed, to
// allow the output step to be at least 2x as large as the input step)
// If there are only 3 inputs the 1,2 inputs are used at the requested
// range, resulting in a 'generic' conversion, which will in general
// be more lossy than a range based on the data
//
// datapath is:
//   (1) read value; if it's u16, xor with 0x8000
//   (2) multiply by a scale value, which is at most 0x4000
//       using a fractional multiply
//   (3) add an offset which accounts for input & output zeros.
//   (4) >>rsh with rounding, and truncate to u8; rsh = 0..7
//    Max 'through' gain is 0.5 (*0x4000 >> 0); min (with decent
//    scaling precision) is about (1/2048) (*0x400>>7) (which is
//    lower than we really need).
//
//
//
static int cvt_x16_8_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor  * out_tensor = self->outputs[0];

	float in_minval = tensor_get_float( self->inputs[1], 0 );
	float in_maxval = tensor_get_float( self->inputs[2], 0 );
	float req_minval = in_minval;
	float req_maxval = in_maxval;
	if( self->n_inputs >=5){
		req_minval = tensor_get_float( self->inputs[3], 0 );
		req_maxval = tensor_get_float( self->inputs[4], 0 );
	}

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8) != 0){
		return errlog(nn,"output too small");
	}
	int is_u16 = self->node_type == OP_Convert_u16_8;

	// work out the output range...
	float out_minval, out_maxval;
	float out_qstep, out_recip_qstep;

	quantize_adjust_range(&out_minval, &out_maxval, &out_qstep, &out_recip_qstep, req_minval,req_maxval);
	int out_zero = saturate_u8( roundf_i32(-out_minval*out_recip_qstep));
	// find input step
	float in_qstep = (in_maxval-in_minval)* (float)(1./65536);
	float conv_ratio;
	if( out_qstep < in_qstep*2.0f){
		out_qstep = in_qstep*2.0f;
		conv_ratio = 0.5f;
		out_minval = out_qstep * -out_zero;
		out_maxval = out_qstep *(255-out_zero);
	}else{
		conv_ratio = in_qstep/out_qstep;
	}
	// find the input zero...
	int in_zero = 0x8000;
	int in_xor = 0;
	if( is_u16 ){
		in_xor = 0x8000;
		in_zero = saturate_u16( 0.5f -in_minval/in_qstep);
	}
	tensor_set_single_float( self->outputs[1], out_minval );
	tensor_set_single_float( self->outputs[2], out_maxval );
	// now set scale_fac and rsh
	int rsh = max_i32(7,-1-flt_getexp(conv_ratio*0.99993896f));
	int scale_fac,offset;
	while(1){
		scale_fac = roundf_i32( flt_ldexp(conv_ratio,rsh+15));

		// offset to add after scaling, before shift...
		int inzero_scaled = ((in_zero-0x8000)*scale_fac + (1<<14))>>15;
		offset = (out_zero<<rsh) - inzero_scaled;
		// need to check if offset fits in 16 bits; if not,
		// decrease rsh by 1 and redo
		//
		if( saturate_i16(offset)==offset) break;
		// is is not possible for rsh to be 0 here.
		--rsh;
	}
	uint16_t const * inp = (uint16_t const *) in_tensor->data;
	uint8_t *outp = (uint8_t *)out_tensor->data;
	int count = tensor_element_count( in_tensor);
	int rbias = (1<<rsh)>>1;

	for(int i = 0; i < count;i++){
		int inval = (int16_t)( inp[i]^in_xor);
		int prod = (inval * scale_fac + (1<<14))>>15;
		outp[i] = saturate_u8( (prod + offset + rbias)>>rsh );
	}
	return 0;
}
static int cvt_x16_8_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking cvt_16_8 node %p",self);
	if( (self->n_inputs != 3 && self->n_inputs != 5)
			|| self->n_outputs != 3 ) return errlog(nn,"wrong number of inputs or outputs");
	logmsg(nn,2,"cvt_16_8 node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Convert_16_8 = {
	.execute = cvt_x16_8_execute,
	.check = cvt_x16_8_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(3,5),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_Convert_u16_8 = {
	.execute = cvt_x16_8_execute,
	.check = cvt_x16_8_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(3,5),
	.n_outputs = NN_IOCOUNT(3),
};
///////////////////////
// 16 -> 16 conversions
// (input may be u16 or s16; output may be u16 or s16; you can also change the range
// as needed).
//
static inline int32_t
mul_i32xi16_rsh15_rnd_sat( int32_t a, int16_t b)
{
	int64_t p = Q6_P_mpy_RR(a,b);
	return Q6_R_sat_P(Q6_P_asrrnd_PI( p,15));
}

struct cvt_x16_x16_runparms {
	uint16_t *outp;
	uint16_t const *inp;
	int numel;
	uint32_t inout_offs;
	int32_t gain;
	int32_t offset;
	int is_s16s16;

	nn_sem_t done_sem;
};

static void cvt_x16_x16_worker( struct nn_graph *nn, void * rstpv);

static int cvt_x16_x16_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor  * out_tensor = self->outputs[0];

	float in_minval = tensor_get_float( self->inputs[1], 0 );
	float in_maxval = tensor_get_float( self->inputs[2], 0 );
	float req_minval = fminf(0.0f, tensor_get_float( self->inputs[3], 0 ));
	float req_maxval = fmaxf( 1e-5f, tensor_get_float( self->inputs[4], 0 ));

	float in_qstep = (in_maxval-in_minval)*(float)(1./65536.);
	float out_qstep, out_minval, out_maxval;
	int out_zero, in_zero;
	float through_gain;

	int in_u16 = 0;
	int out_u16 = 0;
	int s16s16 = 0;
	switch( self->node_type){
	 case OP_Requantize_u16_u16:
		 in_u16 = 1;
		 out_u16 = 1;
		 break;
	 case OP_Convert_u16_16:
		 in_u16 = 1;
		 break;
	 case OP_Convert_16_u16:
		 out_u16 = 1;
		 break;
	 case OP_Requantize_16_16:
		 s16s16 = 1;
		 break;
	 default:
		 break;
	}
	if( !out_u16){		// output is s16
		out_qstep = fmaxf( req_maxval, -req_minval)*(float)(1./32768.);
		float min_ratio = in_u16 ? (float)(1./8192.):(float)(1./32768.);
		out_qstep = fmaxf( in_qstep*min_ratio, out_qstep);
		out_zero = 0x8000;
	}else{
		if(req_minval<0.0f){
			if( req_maxval < req_minval * (float)(-1./65535.)){
				req_maxval = req_minval * (float)(-1./65535.);
			}else{
				float rmn = req_minval, rmx = req_maxval;
				adjust_minmax_for_zero_16b( &rmn, &rmx);
				req_minval = rmn; req_maxval = rmx;
			}
		}
		// req_minval, req_maxval should be a proper u16 output range now.
		out_qstep = (req_maxval-req_minval)*(float)(1./65536.);
		out_zero = saturate_u16( 0.5f - req_minval/out_qstep);
		out_qstep = fmaxf( out_qstep, in_qstep * (float)(1./8192.));// enforce max gain
	}
	// find the min and max output based on out_qstep and out_zero

	out_minval = out_qstep * (float)(-out_zero);
	out_maxval = out_qstep * (float)(65536-out_zero);
	// find the input zero
	in_zero = 0x8000;
	if( in_u16){
		in_zero =  saturate_u16( 0.5f - in_minval/in_qstep);
	}
	//
	// conversion parameters
	//
	through_gain = in_qstep/out_qstep;
	int32_t gain = min_u32(0x7fffffff,roundf_u32(  through_gain * ( s16s16?  (float)(1<<16) : (float)(1<<18))));

	struct cvt_x16_x16_runparms runparms;

	if( !s16s16){
		runparms.inout_offs = ((unsigned)out_u16<<31)|(in_u16 <<15);
		// for the u16->u16 case, we want
		// out[i] = out_zero + ( in[i] - in_zero )*gain*2^-18
		//        = in[i] * gain * 2^18  + (out_zero - in_zero*gain*2^-18)
		// and nn_do_scaleoff_16to16 does
		//    out[i] = (gain * 2^-18) * in[i] + [  (offset-gain)*2^-3 + 32768]
		//
		// so we need to find 'offset' such that  [(offset-gain)*2^-3 + 32768] = (out_zero - in_zero*gain*2^-18)
		//
		//  => offset = 8*out_zero - 2^18 + g - g*(in_zero* 2^-15).
		//  or
		//     offset = 8*(out_zero - 2^15) - g * ( in_zero-2^15)/2^15
		// ...and this will work out for cases mixed with s16, which have zero=2^15
		//
		int32_t off_out= 8*(out_zero-32768);
		int32_t off_in = mul_i32xi16_rsh15_rnd_sat( gain, in_zero-32768);
		int32_t offset= Q6_R_add_RR_sat( off_in, off_out);
		uint32_t sat_test = (uint32_t)off_in + (uint32_t)off_out;
		if( (uint32_t)offset != sat_test ) logmsg(nn,0,"?? saturated offset %d +  %d\n", (int)off_in, (int)off_out);
		runparms.offset = offset;
	}
	if ( ! nn_tensor_out_prepare_normal_fromshape(out_tensor, &in_tensor->shape, out_u16? NN_TYPE_QUINT16: NN_TYPE_QINT16)){
		return errlog(nn,"output to small");
	}
	int n = tensor_element_count( in_tensor);
	int use_hvx = n >= 64;
	if( !use_hvx){
		if( s16s16){
			nn_do_scale_s16( (int16_t *)out_tensor->data, (int16_t const *)in_tensor->data, gain,  n);
		}else{
			nn_do_scaleoff_16to16( (uint16_t *)out_tensor->data, (uint16_t const *)in_tensor->data,
					runparms.inout_offs, gain, runparms.offset,  n);
		}
	}else{
		runparms.gain = gain;
		runparms.is_s16s16 = s16s16;
		runparms.inp = (uint16_t const *)in_tensor->data;
		runparms.outp = (uint16_t *)out_tensor->data;
		runparms.numel = n;
		nn_sem_init( & runparms.done_sem, 0);
		nn_os_work_for_vector( nn, cvt_x16_x16_worker, &runparms);
	}
	tensor_set_single_float( self->outputs[1], out_minval );
	tensor_set_single_float( self->outputs[2], out_maxval );
	if( use_hvx){
		nn_sem_wait( &runparms.done_sem);
	}
	return 0;
}

static void
cvt_x16_x16_worker( struct nn_graph *nn, void * rstpv)
{
	struct cvt_x16_x16_runparms * rstp = (struct cvt_x16_x16_runparms *)rstpv;
	int n = rstp->numel;
	int32_t gain = rstp->gain;
	if( rstp->is_s16s16){
		nn_do_scale_s16_hvx( (int16_t *)rstp->outp, (int16_t const *)rstp->inp, gain,  n);
	}else{
		nn_do_scaleoff_16to16_hvx( rstp->outp, rstp->inp,rstp->inout_offs, gain, rstp->offset,  n);
	}
	nn_sem_post( & rstp->done_sem);
}


// all of the x16->x16 ops have this interface:
//  input 0: input tensor (quint16 or qint16)
//  input 1,2: input range (must comply with input format)
//  input 3,4: req. output range (will be adjusted to comply with output format)
//
//  output 0 : output tensor (quint16 or qint16)
//  output 1,2: output range (as input 3,4, or adjusted as needed).
//
// The output range may be further expanded if the combination of input & output
// ranges implies an excessively high through-gain (>2^15 for 16_16; >2^13 for other combinations).
//

struct nn_node_ops nn_ops_for_Requantize_u16_u16 = {
	.execute = cvt_x16_x16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Convert_u16_16 = {
	.execute = cvt_x16_x16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_Convert_16_u16 = {
	.execute = cvt_x16_x16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Requantize_16_16 = {
	.execute = cvt_x16_x16_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(3),
};
