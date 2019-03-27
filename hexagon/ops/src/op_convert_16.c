/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
	
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_FLOAT) != 0){
		return errlog(nn,"output too small");
	}
	float minval = tensor_get_float( self->inputs[1], 0 );
	float scale = minval * (float)( -1.0/32768.);
	
	float * outp = (float*) out_tensor->data;
	int32_t const *inp = (int32_t const*)in_tensor->data;		// read in pairs
	int count = tensor_element_count( in_tensor);
	
	int npair =count>>1;
	for( int i =0; i < npair; i++ ){
		int32_t xin = *inp++;
		outp[0] = scale * (float)(int16_t)xin;
		outp[1] = scale * (float)(xin>>16);
		outp += 2;
	}
	if( count & 1 ){
		outp[0] = scale * (float)(int16_t)inp[0];
	}
	return 0;
}

static int dequant_16_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking dequant_16 node %p",self);
	int k = node_check_inputs_outputs_n( self, nn, "dequant_16", 3, 1);
	if( k != 0 ) return k;
	logmsg(nn,2,"dequant_16 node %p check OK",self);
	return 0;
}
struct nn_node_ops nn_ops_for_Dequantize_16 = {
	.execute = dequant_16_execute,
	.check = dequant_16_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
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
static int quant_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor  * out_tensor = self->outputs[0];
	
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QINT16) != 0){
		return errlog(nn,"output too small");
	}
	float in_minval = tensor_get_float( self->inputs[1], 0 );
	float in_maxval = tensor_get_float( self->inputs[2], 0 );
	float out_maxval = fmaxf( -in_minval, in_maxval);
	
	out_maxval = flt_round_up_4eps(out_maxval);	// round up to a value with 2 zeros at the bottom.

	tensor_set_single_float( self->outputs[1], -out_maxval );
	tensor_set_single_float( self->outputs[2], out_maxval );
	
	float scale = 32768.0f/ out_maxval;
	
	float const * inp = (float const *) in_tensor->data;
	int16_t *outp = (int16_t *)out_tensor->data;
	int count = tensor_element_count( in_tensor);
	
	for( int i = 0; i < count; i++ ){
		int ival = roundf_i32( scale * inp[i]);
		outp[i] =  min_i32(32767, max_i32(ival,-32768));
		//outp[i] = saturate_i16( ival);	// <- crashes compiler
	}
	return 0;
}
static int quant_16_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking quant_16 node %p",self);
	int k = node_check_inputs_outputs_n( self, nn, "dequant_16", 3, 3);
	if( k != 0 ) return k;
	logmsg(nn,2,"quant_16 node %p check OK",self);
	return 0;
}
struct nn_node_ops nn_ops_for_Quantize_16 = {
	.execute = quant_16_execute,
	.check = quant_16_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
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
static int cvt_8_16_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking cvt_8_16 node %p",self);
	int k = node_check_inputs_outputs_n( self, nn, "cvt_8_16", 3, 3);
	if( k != 0 ) return k;
	logmsg(nn,2,"cvt_8_16 node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Convert_8_16 = {
	.execute = cvt_8_16_execute,
	.check = cvt_8_16_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
