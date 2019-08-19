/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (mulject to the limitations in the
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
#include <math.h>
#include "nn_broadcast.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
//#define DEBUG_PRINT_PERFORMANCE

struct mul_info {
	int a_offset;
	int b_offset;
};

struct tdata {
	struct nn_node *self;
	const struct tensor *a_tensor;
	const struct tensor *b_tensor;
	struct tensor *out_tensor;
	int opt_flag;
	struct mul_info *info;
	nn_sem_t donesem;
};
// mul vector to vector
#define OPERATOR_MUL(X,Y) ((X)*(Y))
BROADCAST_STRIDE_11_FUNC( mul_int32_stride_11, int32_t, int32_t, OPERATOR_MUL)
// mul vector by scalar
BROADCAST_STRIDE_10_FUNC( mul_int32_stride_10, int32_t, int32_t, OPERATOR_MUL)


static const struct elementwise_funcs Mul_int32_funcs = {
	.op_stride_11 = mul_int32_stride_11,
	.op_stride_10 = mul_int32_stride_10,
	.op_rev_stride_01 = mul_int32_stride_10,
	.in_elbytes = 4,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};

static int mul_int32_execute(struct nn_node *self, struct nn_graph *nn)
{
	return nn_elementwise_with_broadcast( self, nn, &Mul_int32_funcs,NULL, NULL, NULL );
}


#if 0
static void qmul_thread_process(struct nn_graph *nn, void *vtdata) {

	struct tdata *td = vtdata;
	const struct tensor *a_tensor = td->a_tensor;
	const struct tensor *b_tensor = td->b_tensor;
	struct tensor *out_tensor = td->out_tensor;
	const uint8_t *a_data = a_tensor->data;
	const uint8_t *b_data = b_tensor->data;
	int32_t *out_data = out_tensor->data;
	struct mul_info *info = td->info;
	int elements, a_const_value, b_const_value;
	struct hvx_info opt_info;
	uint8_t *a_data_pad;
	uint8_t *b_data_pad;

	/* Look for patterns to use HVX intrinsics version of the code and broadcast/prepare the data */
	td->opt_flag = check_prepare_hvx_opt(nn, a_tensor, b_tensor, out_tensor, a_data, b_data, &opt_info);

	a_data_pad = opt_info.a_data_pad;
	b_data_pad = opt_info.b_data_pad;
	elements = opt_info.elements;
	elements = opt_info.elements;
	a_const_value = opt_info.a_const_value;
	b_const_value = opt_info.b_const_value;
	if(td->opt_flag == 1 ) {
		uint32_t tsize = elements*sizeof(int32_t);
		if( tsize > out_tensor->max_size){
			errlog(nn,"output too small");
			td->opt_flag = 2;
		}else{
			out_tensor->data_size = elements*sizeof(int32_t);
			/* Intrinsic version of qmul */
			l2fetch(a_data_pad, 128 , 128 , 1);
			l2fetch(b_data_pad, 128 , 128 , 1);
			qmul_asm(a_data_pad, b_data_pad, out_data, info, elements, a_const_value, b_const_value);
		}
	}
	nn_sem_post(&td->donesem);
}

static int mul_q8_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	struct mul_info info;
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	const struct tensor *a_min_tensor = self->inputs[2];
	const struct tensor *a_max_tensor = self->inputs[3];
	const struct tensor *b_min_tensor = self->inputs[4];
	const struct tensor *b_max_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float a_min_float = tensor_get_float(a_min_tensor,0);
	float a_max_float = tensor_get_float(a_max_tensor,0);
	float b_min_float = tensor_get_float(b_min_tensor,0);
	float b_max_float = tensor_get_float(b_max_tensor,0);
	float a_level_size = (a_max_float - a_min_float)/255;
	float b_level_size = (b_max_float - b_min_float)/255;
	float out_level_size = a_level_size * b_level_size;
	float out_max = 2147483648.0f/*0x1.0p31f*/ * out_level_size;
	float out_min = -out_max;
	int retval;

#ifdef DEBUG_PRINT_PERFORMANCE
	int start_time, end_time;//, t1, t2, t3, t4, t5;
	start_time =  nn_os_get_cycles(nn);
#endif

	info.a_offset = quantize_uint8(0.0f,a_min_float,a_max_float);
	info.b_offset = quantize_uint8(0.0f,b_min_float,b_max_float);

	tensor_set_single_float( out_min_tensor,out_min);
	tensor_set_single_float( out_max_tensor,out_max);

	struct tdata td = {
		.self = self,
		.a_tensor = a_tensor,
		.b_tensor = b_tensor,
		.out_tensor = out_tensor,
		.opt_flag = 0,
		.info = &info,
	};

	nn_sem_init(&td.donesem,0);
	nn_os_work_for_vector(nn, qmul_thread_process, &td);
	nn_sem_wait(&td.donesem);

	if (td.opt_flag == 2){
		return -1;
	}

	if(td.opt_flag == 1) {
		retval = 0;
	} else {
		int32_t offsets[3] = { info.a_offset, info.b_offset, info.a_offset };
		retval = nn_elementwise_with_broadcast( self, nn, &QuantizedMul_funcs, offsets );
	}

#ifdef DEBUG_PRINT_PERFORMANCE
		end_time =  nn_os_get_cycles(nn);
		printf("opt=%d, qmul HVX cycles = %d\n", td.opt_flag, end_time-start_time);
#endif
	return retval;
}
#endif


////////////// 8x8->8 Quantized Mul ///////////////////////////////////////////////////
// this is intended as a fallback for broadcast cases which can't convert to d32
// (it will handle all cases, but uses no hvx code)
// It constructs an intermediate of u16 products which are the lower 16 bits of the
//  product (a-aoffs)*(b-boffs); in the process it finds the 32-bit range of the same
// product. Second pass is to determine output scaling, then quantize back.
//
//
struct qmul_8x8to8_info {
	uint16_t * tmp_buf;		// nn_elementwise_with_broadcast sets this to point to scratch result.
	int a_offset, b_offset;	// the zero-offsets of the inputs
	volatile int32_t minlevel,maxlevel;	// these keep the min/max overall.
};
// mul vector by vector
static void  qmul_8x8to8_stride_11( void *out, void const *in1, void const *in2, int n, void *opaque)
{
	struct qmul_8x8to8_info *info = (struct qmul_8x8to8_info*)opaque;
	uint16_t * op = (uint16_t*)out;
	uint8_t const * inp1 = (uint8_t const *)in1;
	uint8_t const * inp2 = (uint8_t const *)in2;
	int a_offs = info->a_offset;
	int b_offs = info->b_offset;
	int min_here = 0, max_here = 0;

	if( n > 0){
		int32_t s = (*inp1++ - a_offs) * (*inp2++ - b_offs);
		min_here = min_i32(min_here,s);
		max_here = max_i32(max_here,s);
		for(int i = 0; i < n-1; i++){
			*op++ = (uint16_t)s;
			s = (*inp1++ - a_offs) * (*inp2++ - b_offs);
			min_here = min_i32(min_here,s);
			max_here = max_i32(max_here,s);
		}
		*op =(uint16_t) s;
	}
	nn_atomic_min( &info->minlevel, min_here);
	nn_atomic_max( &info->maxlevel, max_here);
}
// mul vector by scalar
//
//  we do (a[i]-a_offs)*(b[0]-b_offs)
//   = a[i]*bval - delt
//   where bval = b[0]-b_offs
//	     delt = bval*a_offs
// The min/max can be done by keeping track of the range of b[i].
//

static void  qmul_8x8to8_vec_by_scalar( struct qmul_8x8to8_info *info,
		uint16_t *outp, uint8_t const *inp, int bval, int delt, int n )
{

	unsigned mina = 255;
	unsigned maxa = 0;
	if( bval==0){
		// this implies delt = 0, all outputs are zero.
		memset( outp, 0, n*sizeof(uint16_t));
		return;
	}
	for( int i =0; i < n; i++){
		unsigned aval = inp[i];
		mina = min_u32( aval, mina);
		maxa = max_u32( aval, maxa);
		outp[i] = aval * bval - delt;
	}
	int min_here = mina*bval -delt;
	int max_here = maxa*bval -delt;
	// if bval < 0, min/max are in the wrong order

	if( bval < 0){ int t = min_here; min_here = max_here; max_here = t; };

	nn_atomic_min( &info->minlevel, min_here);
	nn_atomic_max( &info->maxlevel, max_here);
}

// mul vector by scalar
//

static void  qmul_8x8to8_stride_10( void *out, void const *in1, void const *in2, int n, void *opaque)
{
	struct qmul_8x8to8_info *info = (struct qmul_8x8to8_info*)opaque;
	uint16_t * op = (uint16_t*)out;
	uint8_t const * inp1 = (uint8_t const *)in1;

	int bval = *(uint8_t const *)in2 - info->b_offset;		// bin-b_offs
	int delt = info->a_offset*bval;
	qmul_8x8to8_vec_by_scalar( info, op, inp1, bval, delt, n );
}
// mul vector by scalar, where 'a' is the scalar.
// in1 points to 'b' (vector); in2 points to 'a' (scalar)
static void  qmul_8x8to8_rev_stride_01( void *out, void const *in1, void const *in2, int n, void *opaque)
{
	struct qmul_8x8to8_info *info = (struct qmul_8x8to8_info*)opaque;
	uint16_t * op = (uint16_t*)out;
	uint8_t const * inp1 = (uint8_t const *)in1;

	int bval = *(uint8_t const *)in2 - info->a_offset;		// ain-a_offs
	int delt = info->b_offset*bval;
	qmul_8x8to8_vec_by_scalar( info, op, inp1, bval, delt, n );
}
static const struct elementwise_funcs QuantizedMul_8x8to8_funcs = {
	.op_stride_11 = qmul_8x8to8_stride_11,
	.op_stride_10 = qmul_8x8to8_stride_10,
	.op_rev_stride_01 = qmul_8x8to8_rev_stride_01,
	.in_elbytes = 1,
	.out_elbytes = 1,
	.out_typecode =  NN_TYPE_QUINT8,
	.scratch_elbytes = 2			// -> causes scratch array of 2-byte elements to be used as output.
};

static int mul_8x8to8_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *a_tensor = self->inputs[0];
	const struct tensor *b_tensor = self->inputs[1];
	const struct tensor *a_min_tensor = self->inputs[2];
	const struct tensor *a_max_tensor = self->inputs[3];
	const struct tensor *b_min_tensor = self->inputs[4];
	const struct tensor *b_max_tensor = self->inputs[5];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float a_min_float = tensor_get_float(a_min_tensor,0);
	float a_max_float = tensor_get_float(a_max_tensor,0);
	float b_min_float = tensor_get_float(b_min_tensor,0);
	float b_max_float = tensor_get_float(b_max_tensor,0);
	float a_level_size = flt_div_255(a_max_float - a_min_float);
	float b_level_size = flt_div_255(b_max_float - b_min_float);
	float prod_level_size = a_level_size * b_level_size;

	int a_offset = saturate_u8( roundf_i32(-a_min_float/a_level_size));
	int b_offset = saturate_u8( roundf_i32(-b_min_float/b_level_size));

#ifdef DEBUG_PRINT_PERFORMANCE
	int start_time, end_time;//, t1, t2, t3, t4, t5;
	start_time =  nn_os_get_cycles(nn);
#endif

	float out_min, out_max;

	// look for broadcast one-to-many cases.
	static const struct shape shp1111 = { .batches=1, .height=1, .width=1, .depth=1 };
	int a_is_scalar = 0;
	if( shape_matches( &b_tensor->shape, & shp1111) || ( a_is_scalar=1, shape_matches( &a_tensor->shape, & shp1111))){
		const struct tensor *full_tensor;
		int sc_val;
		float sc_flt;
		if( !a_is_scalar){		// b is the scalar
			full_tensor = a_tensor;
			out_min = a_min_float;
			out_max = a_max_float;
			sc_val = *(uint8_t const*)b_tensor->data - b_offset;
			sc_flt = sc_val * b_level_size;
		}else{
			full_tensor = b_tensor;
			out_min = b_min_float;
			out_max = b_max_float;
			sc_val = *(uint8_t const*)a_tensor->data - a_offset;
			sc_flt = sc_val * a_level_size;
		}

		if( sc_val > 0){		// copy the tensor as-is, and scale the output range.
			out_min *= sc_flt;
			out_max *= sc_flt;
			if( nn_tensor_copy(out_tensor,full_tensor)!= 0) return errlog(nn,"output too small");
		}else{
			// TODO: we should have nn_tensor_copy_scaled (non-d32) for this.
			if( nn_tensor_out_prepare_normal_fromshape(out_tensor, &full_tensor->shape,NN_TYPE_QUINT8)!=0){
				return errlog(nn,"output too small");
			}
			if( sc_val == 0){		// multiply all by 0.
				out_min = 0.0f;
				out_max = 1.0f;
				memset( out_tensor->data, 0, out_tensor->data_size);
			}else{		// -ve scale; flip endpoints, and invert the data.
				float tmp_max = out_min * sc_flt;	 // sc_flt < 0
				out_min = out_max * sc_flt;
				out_max = tmp_max;
				// now, copy with 1's complement
				uint64_t const *srcp = (uint64_t const *)full_tensor->data;
				uint64_t  *dstp = (uint64_t  *)out_tensor->data;
				int n64 = (out_tensor->data_size + 7)/8u;
				for( int i = 0; i < n64; i++) dstp[i] = ~srcp[i];
			}
		}

	}else{
		struct qmul_8x8to8_info info;
		info.tmp_buf = NULL;
		info.a_offset = a_offset;
		info.b_offset = b_offset;
		info.minlevel = 0;
		info.maxlevel = 0;
		nn_scratch_reset(nn);
		/*printf( "(%d,%d,%d,%d) z= %d * (%d,%d,%d,%d)  z = %d\n",
				(int)a_tensor->shape.batches,
				(int)a_tensor->shape.height,
				(int)a_tensor->shape.width,
				(int)a_tensor->shape.depth,   info.a_offset,
				(int)b_tensor->shape.batches,
				(int)b_tensor->shape.height,
				(int)b_tensor->shape.width,
				(int)b_tensor->shape.depth,  info.b_offset );*/

		int res = nn_elementwise_with_broadcast( self, nn, &QuantizedMul_8x8to8_funcs,NULL, NULL, &info );
		if( res != 0) return res;
		// nn_elementwise_with_broadcast has prepared the output tensor, but has formed
		// the result as a 16-bit intermediate in the scratch area now pointed to by info.tmp_buf.
		// We now choose the output range, and quantize the results.
		//
		if (info.maxlevel == info.minlevel) info.maxlevel += 4;	// can't both be 0.

		// actual range of product; adjust for accurate zero code.
		float out_min0 = info.minlevel * prod_level_size;
		float out_max0 = info.maxlevel * prod_level_size;
		adjust_minmax_for_zero( & out_min0, &out_max0);
		out_min = out_min0;
		out_max = out_max0;


		float output_level_size = flt_div_255( out_max- out_min);
		int out_zero = saturate_u8( roundf_i32(-out_min/output_level_size));

		float scaleby;
		if( prod_level_size * (float)(1.0/16.0) < output_level_size){	// which should almost always be true...
			scaleby = prod_level_size / output_level_size;
		}else{
			// scaling wants scale *up* by >16. Unlikely but possible. Instead, force scaling
			// to exactly 16.0 and increase the output range to make that correct. zero is unchanged.
			scaleby = 16.0f;		// bump range to avoid scaleby > 16
			output_level_size = prod_level_size* (float)(1.0/16.0);
			out_min = -out_zero * output_level_size;
			out_max = (255-out_zero) * output_level_size;
		}
		int nvals = out_tensor->data_size;		// # of values to convert
		// the values in the temp area are uint16, but they are really the 16 lsbs of products which could range over -65025 .. 65025.
		// However, for any given (a_offset,b_offset), the total range of possible values max-min is <= 65025 ( and can be as small as 32640 )
		// we have p = (a-a0)*(b-b0), with all of a,b,a0,b0 constrained to 0..255
		//  the minimum p is a0*b0 - 255*min(a0,b0)						// ranging -65025 .. 0
		//     maximum  p is a0*b0 + 255*max(0, 255-(a0+b0))			// ranging  0 .. 65025
		// to 'decode' these, we
		//    - subtract pmin  mod 16 bits; uint16 result is a true representation of prod-pmin, which is 0..65025
		//    - add pmin, as int
		// (we can use the min calculated during the operation,  instead of the formula)
		//int pmin = info.a_offset * info.b_offset - 255 * min_i32( info.a_offset,info.b_offset );
		int pmin = info.minlevel;

		// one other trick:
		//  instead of doing  p = (uint16_t)(buf[i]-pmin) + pmin
		//    and then result = (p*scale+offs) >> rsh,
		//  we can do p0 = (uint16_t)(buf[i]-pmin)
		//     and then  result = (p0*scale + offs) >> rsh;
		//  if we just roll the '+pmin*scale' into offs.

		int scale,offs,rsh;
		// find scale and rsh; we want scale to be +/-16383 at most.
		int expp = flt_getexp( scaleby);		// at most 5 (for scaleby  =16)
		rsh = min_i32(14-expp,23);							// at least 9, at most 23
		scale = roundf_i32( flt_ldexp(scaleby,rsh));	// will be at most 16384, usually >=8K
		// offs introduces the output zero-offset and a rounding bias.
		offs = (2*out_zero+1)<<(rsh-1);
		// compensate for p actually being prod-pmin
		offs += pmin*scale;

		uint16_t const *rp = info.tmp_buf;
		uint8_t * outp = (uint8_t*)out_tensor->data;
		for( int i =0; i < nvals; i++){
			uint16_t p = (uint16_t)( rp[i]- pmin);		// find product - pmin
			p= (p*scale + offs) >> rsh;
			outp[i] = saturate_u8(p);
		}
	}
	tensor_set_single_float( out_min_tensor,out_min);
	tensor_set_single_float( out_max_tensor,out_max);


#ifdef DEBUG_PRINT_PERFORMANCE
		end_time =  nn_os_get_cycles(nn);
		printf("qmul cycles = %d\n", end_time-start_time);
#endif
	return 0;
}

static int mul_q8_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"mul node %p",self);

	// if 8x8to8 mul, we need scratch for 2x the max output
    ///if( self->node_type == OP_QuantizedMul_8x8to8 || self->node_type == OP_QuantizedMul_8x8to8_ref){
	if(1){
		// find max output size.
		struct output const *od0 = &self->output_defs[0];
		unsigned nmax = mulu32_x4_sat( od0->max_sizes[0], od0->max_sizes[1], od0->max_sizes[2], od0->max_sizes[3]);
		if( nmax ==0 || nmax > (1u<<25))
			return errlog(nn,"can't get plausible max output size for mul output");
		if( nn_scratch_grow(nn, nmax * sizeof(uint16_t)) != 0 ){
			return errlog(nn, "can't get %u*2 scratch", nmax);
		}

	}
	logmsg(nn,2,"mul %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Mul_int32 = {
	.execute = mul_int32_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(1),
};

#if 0
struct nn_node_ops nn_ops_for_QuantizedMul_8x8to32 = {
	.execute = mul_q8_execute_hvx,
	.check = mul_q8_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_CLS_QUANTMUL8TO32,
};

struct nn_node_ops nn_ops_for_QuantizedMul_8x8to32_ref = {
	.execute = mul_q8_execute,
	.check = mul_q8_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};
#endif

struct nn_node_ops nn_ops_for_QuantizedMul_8x8to8 = {
	.execute = mul_8x8to8_execute,
	.check = mul_q8_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_QuantizedMul_8x8to8_ref = {
	.execute = mul_8x8to8_execute,
	.check = mul_q8_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};
