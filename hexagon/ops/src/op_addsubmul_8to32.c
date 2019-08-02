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
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#include <nn_broadcast.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_inlines.h"



//
// this file contains the 'reference' and hvx code for:
// QuantizedAdd_8p8to32[_ref]
// QuantizedSub_8p8to32[_ref]
// (also, below: QuantizedMul_8x8to32[_ref])
//
// The core operations for these is as follows:
//  (1) find out = ka *ina[i] + kb * inb[i] - offs;
//
// .. where ina,inb are u8, and ka,kb are 16-bit signed
//  and 'offs' is ka*ina_zero + kb*inb-zero.
//
// for add: ka, kb, are in proportion to the 'a' and 'b' input steps,
// and we scale them as large as possible to keep them in 16-bit signed.
// for sub, it's the same thing but we change the sign of kb.
//

struct addsub_832_scaling {
	int16_t ka, kb;
	int32_t offs;
};


//
// general add/sub function (vector+vector)
//
static void addsub_832_stride_11( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_832_scaling const * info = (struct addsub_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inp1 = (uint8_t const *)in1;
	uint8_t const * inp2 = (uint8_t const *)in2;
	int ka = info->ka;
	int kb = info->kb;
	int offs = info->offs;
	if( n > 0){
		int32_t s = ka* (*inp1++) + kb * (*inp2++) - offs;
		for(int i = 0; i < n-1; i++){
			*op++ = s;
			s = ka* (*inp1++) + kb * (*inp2++) - offs;
		}
		*op = s;
	}
}
// this is the hvx version of addsub_832_stride_11.
static void __attribute__((unused))
addsub_832_stride_11_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_832_scaling const * info = (struct addsub_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inp1 = (uint8_t const *)in1;
	uint8_t const * inp2 = (uint8_t const *)in2;
	int ka = info->ka;
	int kb = info->kb;
	int offs = info->offs;
	// this is the number of vectors which have at least 65
	// elements in them.
	int nvecs = (n+63)/128u;
	ka = Q6_R_combine_RlRl( ka, ka);
	kb = Q6_R_combine_RlRl( kb, kb);
	HVX_Vector offsw = Q6_V_vsplat_R( -offs);

	int nremain = n;
	for( int i =0; i< nvecs;i++){
		HVX_Vector va = q6op_V_vldu_A( (HVX_Vector const*)inp1); inp1 += 128;
		HVX_Vector vb = q6op_V_vldu_A( (HVX_Vector const*)inp2); inp2 += 128;
		// shuffle them across 2 vecs {a0,b0,a1,b1...   a127,b127}
		HVX_VectorPair vabshuf = Q6_W_vshuff_VVR( vb, va,-1);
		// get the values, zero-extended to 16,
		// A from even lanes->lo, B from odd ->hi
		HVX_VectorPair vab0 = Q6_Wb_vshuffoe_VbVb(Q6_V_vzero(), Q6_V_lo_W(vabshuf) );
		HVX_VectorPair vab64 = Q6_Wb_vshuffoe_VbVb(Q6_V_vzero(), Q6_V_hi_W(vabshuf) );

		// do the sums of products
		HVX_VectorPair sum_0 = Q6_Ww_vmpyacc_WwVhRh_sat(
						Q6_Ww_vmpy_VhRh( Q6_V_lo_W(vab0), ka),
						Q6_V_hi_W(vab0), kb );
		HVX_VectorPair sum_64 = Q6_Ww_vmpyacc_WwVhRh_sat(
						Q6_Ww_vmpy_VhRh( Q6_V_lo_W(vab64), ka),
						Q6_V_hi_W(vab64), kb );
		HVX_Vector res_0 = Q6_Vw_vadd_VwVw( Q6_V_lo_W(sum_0), offsw);
		HVX_Vector res_1 = Q6_Vw_vadd_VwVw( Q6_V_hi_W(sum_0), offsw);
		HVX_Vector res_64 = Q6_Vw_vadd_VwVw( Q6_V_lo_W(sum_64), offsw);
		HVX_Vector res_65 = Q6_Vw_vadd_VwVw( Q6_V_hi_W(sum_64), offsw);

		HVX_VectorPair out2 = Q6_W_vshuff_VVR(res_1,res_0,-4);
		// nremain is at least 65. So store the first two vecs..
		q6op_vstu_AV( (HVX_Vector*)op, Q6_V_lo_W(out2));	op += 32;
		q6op_vstu_AV( (HVX_Vector*)op, Q6_V_hi_W(out2));	op += 32;
		out2 = Q6_W_vshuff_VVR(res_65,res_64,-4);
		if( nremain >= 128){
			q6op_vstu_AV( (HVX_Vector*)op, Q6_V_lo_W(out2));	op += 32;
			q6op_vstu_AV( (HVX_Vector*)op, Q6_V_hi_W(out2));	op += 32;
			nremain -= 128;
		}else{	// must be last iteration; nremain is 65..127; do the last 1..63
			hvx_store_vec_x2_unaligned( op,Q6_V_lo_W(out2),Q6_V_hi_W(out2), (nremain-64)*4 );
			return;
		}
	}
	if( nremain > 0){	// have an odd half-vector (1..64) at the end
		HVX_Vector va = q6op_V_vldu_A( (HVX_Vector const*)inp1);
		HVX_Vector vb = q6op_V_vldu_A( (HVX_Vector const*)inp2);
		HVX_VectorPair vabshuf = Q6_W_vshuff_VVR( vb, va,-1);
		HVX_VectorPair vab0 = Q6_Wb_vshuffoe_VbVb(Q6_V_vzero(), Q6_V_lo_W(vabshuf) );

		// do the sums of products
		HVX_VectorPair sum = Q6_Ww_vmpyacc_WwVhRh_sat( Q6_Ww_vmpy_VhRh( Q6_V_lo_W(vab0), ka),Q6_V_hi_W(vab0), kb );
		HVX_Vector res_0 = Q6_Vw_vadd_VwVw( Q6_V_lo_W(sum), offsw);
		HVX_Vector res_1 = Q6_Vw_vadd_VwVw( Q6_V_hi_W(sum), offsw);
		HVX_VectorPair out2 = Q6_W_vshuff_VVR(res_1,res_0,-4);
		hvx_store_vec_x2_unaligned( op,Q6_V_lo_W(out2),Q6_V_hi_W(out2), nremain*4 );
	}
}

////////////////////////////////////////////

// used to do the degenerate case *outp = *in1++ * k + delt
// when one input is a scalar
//
static void addsub_832_vec_and_scalar( int32_t *out, uint8_t const * in, int n, int k, int delt)
{
	if( n > 0){
		int32_t s = k*(*in++) + delt;
		for(int i = 0; i < n-1; i++){
			*out++ = s;
			s =k*(*in++) + delt;
		}
		*out = s;
	}
}

//
// used when 'b' is invariant (in2 pointer, to b, doesn't move)

static void addsub_832_stride_10( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_832_scaling const * info = (struct addsub_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpa = (uint8_t const *)in1;
	uint8_t const * inpb = (uint8_t const *)in2;
	int ka = info->ka;
	int delt = info->kb * inpb[0] - info->offs;
	addsub_832_vec_and_scalar( op, inpa, n, ka, delt);
}
//
// used when 'a' is invariant (in2 pointer, to a, doesn't move)

static void addsub_832_rev_stride_01( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_832_scaling const * info = (struct addsub_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpb = (uint8_t const *)in1;
	uint8_t const * inpa = (uint8_t const *)in2;
	int kb = info->kb;
	int delt = info->ka * inpa[0] - info->offs;
	addsub_832_vec_and_scalar( op, inpb, n, kb, delt);
}
/////////////////////////////////////////////
// HVX Version of *outp = *in1++ * k + delt
// when one input is a scalar
//
static void __attribute__((noinline))
addsub_832_vec_and_scalar_hvx( int32_t *out, uint8_t const * in, int n, int k, int delt)
{
	HVX_Vector const * rdp = (HVX_Vector const *)in;
	HVX_Vector  * op = (HVX_Vector  *)out;
	k = Q6_R_combine_RlRl(k,k);
	HVX_Vector vdelt = Q6_V_vsplat_R(delt);
	HVX_VectorPair wdelt = Q6_W_vcombine_VV( vdelt,vdelt);

	int nvec = n/sizeof(HVX_Vector);	// # of full loops
	int nremain = n %sizeof(HVX_Vector);
	for(int i = 0; i < nvec; i++){
		// zero extend and re-order across 2 vecs...
		HVX_Vector vin = q6op_V_vldu_A( rdp);	rdp ++;
		// zero extend and re-order across 2 vecs...
		HVX_VectorPair in_h = Q6_W_vshuff_VVR( Q6_V_vzero(), vin, -1);
		// sums of products...
		HVX_VectorPair prod0 = Q6_Ww_vmpyacc_WwVhRh_sat( wdelt, Q6_V_lo_W(in_h), k);
		HVX_VectorPair prod64 = Q6_Ww_vmpyacc_WwVhRh_sat( wdelt, Q6_V_hi_W(in_h), k);
		// put them into sequential order
		HVX_VectorPair out0 = Q6_W_vshuff_VVR(Q6_V_hi_W(prod0),Q6_V_lo_W(prod0),-4);
		HVX_VectorPair out64 = Q6_W_vshuff_VVR(Q6_V_hi_W(prod64),Q6_V_lo_W(prod64),-4);
		q6op_vstu_AV(op, Q6_V_lo_W(out0) );  op ++;
		q6op_vstu_AV(op, Q6_V_hi_W(out0) );  op ++;
		q6op_vstu_AV(op, Q6_V_lo_W(out64) );  op ++;
		q6op_vstu_AV(op, Q6_V_hi_W(out64) );  op ++;
	}
	if( nremain > 0){
		HVX_Vector vin = q6op_V_vldu_A( rdp);	rdp ++;
		// zero extend and re-order across 2 vecs...
		HVX_VectorPair in_h = Q6_W_vshuff_VVR( Q6_V_vzero(), vin, -1);
		// sums of products...
		HVX_VectorPair prod0 = Q6_Ww_vmpyacc_WwVhRh_sat( wdelt, Q6_V_lo_W(in_h), k);
		// put them into sequential order
		HVX_VectorPair out0 = Q6_W_vshuff_VVR(Q6_V_hi_W(prod0),Q6_V_lo_W(prod0),-4);
		if( nremain >= 64 ){
			q6op_vstu_AV(op, Q6_V_lo_W(out0) );  op ++;
			q6op_vstu_AV(op, Q6_V_hi_W(out0) );  op ++;
			HVX_VectorPair prod64 = Q6_Ww_vmpyacc_WwVhRh_sat( wdelt, Q6_V_hi_W(in_h), k);
			out0 = Q6_W_vshuff_VVR(Q6_V_hi_W(prod64),Q6_V_lo_W(prod64),-4);
			nremain -= 64;
		}
		if( nremain >0 ){
			hvx_store_vec_x2_unaligned( op,Q6_V_lo_W(out0),Q6_V_hi_W(out0), nremain*4 );
		}
	}
}

//
// used when 'b' is invariant (in2 pointer, to b, doesn't move)

static void addsub_832_stride_10_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_832_scaling const * info = (struct addsub_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpa = (uint8_t const *)in1;
	uint8_t const * inpb = (uint8_t const *)in2;
	int ka = info->ka;
	int delt = info->kb * inpb[0] - info->offs;
	addsub_832_vec_and_scalar_hvx( op, inpa, n, ka, delt);
}
//
// used when 'a' is invariant (in2 pointer, to a, doesn't move)

static void addsub_832_rev_stride_01_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct addsub_832_scaling const * info = (struct addsub_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpb = (uint8_t const *)in1;
	uint8_t const * inpa = (uint8_t const *)in2;
	int kb = info->kb;
	int delt = info->ka * inpa[0] - info->offs;
	addsub_832_vec_and_scalar_hvx( op, inpb, n, kb, delt);
}
/////////////
static const struct elementwise_funcs AddSub_i832_funcs = {
	.op_stride_11 = addsub_832_stride_11,
	.op_stride_10 = addsub_832_stride_10,
	.op_rev_stride_01 = addsub_832_rev_stride_01,
	.in_elbytes = 1,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};
// same with HVX
static const struct elementwise_funcs AddSub_i832_hvx_funcs = {
	.op_stride_11 = addsub_832_stride_11,
	.op_stride_10 = addsub_832_stride_10,
	.op_rev_stride_01 = addsub_832_rev_stride_01,

	.op_hvx_stride_11 = addsub_832_stride_11_hvx,
	.op_hvx_stride_10 = addsub_832_stride_10_hvx,
	.op_hvx_rev_stride_01 = addsub_832_rev_stride_01_hvx,
	.minlen_hvx = 16,		// use scalar if inner-loop n < 16.

	.in_elbytes = 1,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};

// the execute function for QuantizedSub_8p8to32[_ref], QuantizedAdd_8p8to32[_ref]
//
static int
addsub_832_execute( struct nn_node * self, struct nn_graph * nn)
{
	struct tensor const * a_min_tensor = self->inputs[2];
	struct tensor const * a_max_tensor = self->inputs[3];
	struct tensor const * b_min_tensor = self->inputs[4];
	struct tensor const * b_max_tensor = self->inputs[5];

	struct tensor * out_min_tensor = self->outputs[1];
	struct tensor * out_max_tensor = self->outputs[2];

	float a_in_min = tensor_get_float( a_min_tensor, 0);
	float a_in_max = tensor_get_float( a_max_tensor, 0);
	float b_in_min = tensor_get_float( b_min_tensor, 0);
	float b_in_max = tensor_get_float( b_max_tensor, 0);
	// get quantization of input ranges

	float a_in_step, b_in_step;

	int a_in_zero = get_qu8_level_size_zero(a_in_min, a_in_max, &a_in_step );
	int b_in_zero = get_qu8_level_size_zero(b_in_min, b_in_max, &b_in_step );


	// whichever step is largest will scale by 0x7F80; the other in proportion.
	struct addsub_832_scaling sc;
	float largest_step;
	if( fabsf(a_in_step) > fabsf(b_in_step)){
		largest_step = a_in_step;
		sc.ka = 0x7F80;
		sc.kb = roundf_i32( b_in_step * (float)0x7F80 / fmaxf(1e-10f,fabsf(a_in_step)) );
	}else{
		largest_step = b_in_step;
		sc.ka = roundf_i32( a_in_step * (float)0x7F80 / fmaxf(1e-10f,fabsf(b_in_step)) );
		sc.kb = 0x7F80;
	}
	if(largest_step <= 0.0f) return errlog(nn,"bad input scale");


	int node_type = self->node_type;
	if( node_type == OP_QuantizedSub_8p8to32 || node_type == OP_QuantizedSub_8p8to32_ref ){
		sc.kb = -sc.kb;
	}

	sc.offs = sc.ka * a_in_zero + sc.kb * b_in_zero;
	// determine output range: out max is 2^31 / 0x7f80 * largest_step

	float out_max = (float) ( (double)(1u <<31) / (double)0x7f80) * largest_step;

	/*
	printf(" %d: %f  +  %d: %f --> %f;  %d,%d, %d\n",
			a_in_zero, a_in_step, b_in_zero, b_in_step, out_max, sc.ka, sc.kb, (int)sc.offs);
	*/
	if( tensor_set_single_float( out_min_tensor, -out_max)!= 0
		|| tensor_set_single_float( out_max_tensor, out_max)!= 0) {
		return errlog(nn,"out min/max tensor too small");
	}
	struct elementwise_funcs const *ew_funcs =
			( node_type == OP_QuantizedAdd_8p8to32_ref ||
			  node_type == OP_QuantizedSub_8p8to32_ref)? &AddSub_i832_funcs : &AddSub_i832_hvx_funcs;

	return nn_elementwise_with_broadcast( self, nn, ew_funcs,NULL, NULL, &sc );

}
//=================================================================
//=========================== Multiply ============================
//=================================================================

struct mul_832_scaling {
	int16_t zeroa, zerob;
};
//
// general multiply function (vector+vector)
//
static void mul_832_stride_11( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_832_scaling const * info = (struct mul_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inp1 = (uint8_t const *)in1;
	uint8_t const * inp2 = (uint8_t const *)in2;
	int za = info->zeroa;
	int zb = info->zerob;
	if( n > 0){
		int32_t s = (*inp1++ - za)*(*inp2++ - zb)*128;
		for(int i = 0; i < n-1; i++){
			*op++ = s;
			s = (*inp1++ - za)*(*inp2++ - zb)*128;
		}
		*op = s;
	}
}

// this is the hvx version of mul_832_stride_11.
static void __attribute__((unused))
mul_832_stride_11_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{

	struct mul_832_scaling const * info = (struct mul_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inp1 = (uint8_t const *)in1;
	uint8_t const * inp2 = (uint8_t const *)in2;
	int za = info->zeroa;
	int zb = info->zerob;
	// this is the number of vectors which have at least 65
	// elements in them.
	int nvecs = (n+63)/128u;
	HVX_Vector vZa = q6op_Vb_vsplat_R(za);
	HVX_Vector vZb = q6op_Vb_vsplat_R(zb);


	int nremain = n;
	for( int i =0; i< nvecs;i++){
		HVX_Vector va = q6op_V_vldu_A( (HVX_Vector const*)inp1); inp1 += 128;
		HVX_Vector vb = q6op_V_vldu_A( (HVX_Vector const*)inp2); inp2 += 128;
		// shuffle each together with its zero
		HVX_VectorPair ashufz = Q6_W_vshuff_VVR( vZa, va, -1);
		HVX_VectorPair bshufz = Q6_W_vshuff_VVR( vZb, vb, -1);

		// find 16(a[i]-az) and 8*(b[i]-bz)
		HVX_Vector diffa0 = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(ashufz), 0xF010F010);
		HVX_Vector diffb0 = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(bshufz), 0xF808F808);
		HVX_Vector diffa64 = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(ashufz), 0xF010F010);
		HVX_Vector diffb64 = Q6_Vh_vdmpy_VubRb( Q6_V_hi_W(bshufz), 0xF808F808);

		// Just find the products now...

		HVX_VectorPair prod0 = Q6_Ww_vmpy_VhVh( diffa0,diffb0);
		HVX_VectorPair prod64 = Q6_Ww_vmpy_VhVh( diffa64,diffb64);
		// put the firsy 64 in order...
		HVX_VectorPair out0 = Q6_W_vshuff_VVR( Q6_V_hi_W(prod0),Q6_V_lo_W(prod0),-4 );
		// nremain is at least 65. So store the first two vecs..
		q6op_vstu_AV( (HVX_Vector*)op, Q6_V_lo_W(out0));	op += 32;
		q6op_vstu_AV( (HVX_Vector*)op, Q6_V_hi_W(out0));	op += 32;
		out0 = Q6_W_vshuff_VVR( Q6_V_hi_W(prod64),Q6_V_lo_W(prod64),-4);
		if( nremain >= 128){
			q6op_vstu_AV( (HVX_Vector*)op, Q6_V_lo_W(out0));	op += 32;
			q6op_vstu_AV( (HVX_Vector*)op, Q6_V_hi_W(out0));	op += 32;
			nremain -= 128;
		}else{	// must be last iteration; nremain is 65..127; do the last 1..63
			hvx_store_vec_x2_unaligned( op,Q6_V_lo_W(out0),Q6_V_hi_W(out0), (nremain-64)*4 );
			return;
		}
	}
	if( nremain > 0){	// have an odd half-vector (1..64) at the end
		HVX_Vector va = q6op_V_vldu_A( (HVX_Vector const*)inp1);
		HVX_Vector vb = q6op_V_vldu_A( (HVX_Vector const*)inp2);

		HVX_VectorPair ashufz = Q6_W_vshuff_VVR( vZa, va, -1);
		HVX_VectorPair bshufz = Q6_W_vshuff_VVR( vZb, vb, -1);
		HVX_Vector diffa0 = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(ashufz), 0xF010F010);
		HVX_Vector diffb0 = Q6_Vh_vdmpy_VubRb( Q6_V_lo_W(bshufz), 0xF808F808);
		HVX_VectorPair prod0 = Q6_Ww_vmpy_VhVh( diffa0,diffb0);

		HVX_VectorPair out0 = Q6_W_vshuff_VVR( Q6_V_hi_W(prod0),Q6_V_lo_W(prod0),-4 );
		hvx_store_vec_x2_unaligned( op,Q6_V_lo_W(out0),Q6_V_hi_W(out0), nremain*4 );
	}
}
//
// used when 'b' is invariant (in2 pointer, to b, doesn't move)
//
static void mul_832_stride_10( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_832_scaling const * info = (struct mul_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpa = (uint8_t const *)in1;
	uint8_t const * inpb = (uint8_t const *)in2;
	int za = info->zeroa;
	int b = (inpb[0] - info->zerob)*128;
	if( b == 0){
		memset( out, 0, sizeof(int32_t)*n);
	}else{
		int delt = -za * b;
		addsub_832_vec_and_scalar( op, inpa, n, b, delt);
	}
}
// (hvx variant)
static void mul_832_stride_10_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_832_scaling const * info = (struct mul_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpa = (uint8_t const *)in1;
	uint8_t const * inpb = (uint8_t const *)in2;
	int za = info->zeroa;
	int b = (inpb[0] - info->zerob)*128;
	if( b == 0){
		vmemset_asm( out, 0, sizeof(int32_t)*n);
	}else{
		int delt = -za * b;
		addsub_832_vec_and_scalar_hvx( op, inpa, n, b, delt);
	}
}

//
// used when 'a' is invariant (in2 pointer, to a, doesn't move)
//
static void mul_832_rev_stride_01( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_832_scaling const * info = (struct mul_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpa = (uint8_t const *)in2;
	uint8_t const * inpb = (uint8_t const *)in1;
	int zb = info->zerob;
	int a = (inpa[0] - info->zeroa)*128;
	if( a == 0){
		memset( out, 0, sizeof(int32_t)*n);
	}else{
		int delt = -zb * a;
		addsub_832_vec_and_scalar( op, inpb, n, a, delt);
	}
}
// (hvx variant)
static void mul_832_rev_stride_01_hvx( void *out, void const *in1, void const *in2, int n, void *infov)
{
	struct mul_832_scaling const * info = (struct mul_832_scaling const*)infov;
	int32_t * op = (int32_t*)out;
	uint8_t const * inpa = (uint8_t const *)in2;
	uint8_t const * inpb = (uint8_t const *)in1;
	int zb = info->zerob;
	int a = (inpa[0] - info->zeroa)*128;
	if( a == 0){
		vmemset_asm( out, 0, sizeof(int32_t)*n);
	}else{
		int delt = -zb * a;
		addsub_832_vec_and_scalar_hvx( op, inpb, n, a, delt);
	}
}

static const struct elementwise_funcs Mul_i832_funcs = {
	.op_stride_11 = mul_832_stride_11,
	.op_stride_10 = mul_832_stride_10,
	.op_rev_stride_01 = mul_832_rev_stride_01,
	.in_elbytes = 1,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};
static const struct elementwise_funcs Mul_i832_hvx_funcs = {
	.op_stride_11 = mul_832_stride_11,
	.op_stride_10 = mul_832_stride_10,
	.op_rev_stride_01 = mul_832_rev_stride_01,

	.op_hvx_stride_11 = mul_832_stride_11_hvx,
	.op_hvx_stride_10 = mul_832_stride_10_hvx,
	.op_hvx_rev_stride_01 = mul_832_rev_stride_01_hvx,
	.minlen_hvx = 16,		// use scalar if inner-loop n < 16.

	.in_elbytes = 1,
	.out_elbytes = 4,
	.out_typecode =  NN_TYPE_INT32
};

// the execute function for QuantizedMul_8x8to32[_ref]
//
// The calculation is just (qa[i]-azero)*(qb[i]-bzero)<<7
// when 'qb' is a constant, it expands
//  to qa[i] * [(qb-qzero)*128] + (-azero*[(qb-qzero)*128])
// .. which can be done in the same 'vec+scalar' ops used for add/sub.
//
static int
mul_832_execute( struct nn_node * self, struct nn_graph * nn)
{
	struct tensor const * a_min_tensor = self->inputs[2];
	struct tensor const * a_max_tensor = self->inputs[3];
	struct tensor const * b_min_tensor = self->inputs[4];
	struct tensor const * b_max_tensor = self->inputs[5];

	struct tensor * out_min_tensor = self->outputs[1];
	struct tensor * out_max_tensor = self->outputs[2];

	float a_in_min = tensor_get_float( a_min_tensor, 0);
	float a_in_max = tensor_get_float( a_max_tensor, 0);
	float b_in_min = tensor_get_float( b_min_tensor, 0);
	float b_in_max = tensor_get_float( b_max_tensor, 0);
	// get quantization of input ranges

	float a_in_step, b_in_step;

	struct mul_832_scaling sc;
	sc.zeroa = get_qu8_level_size_zero(a_in_min, a_in_max, &a_in_step );
	sc.zerob = get_qu8_level_size_zero(b_in_min, b_in_max, &b_in_step );

	float out_max = a_in_step*b_in_step * (float)( 1u << (31-7));

	int node_type = self->node_type;
	if( tensor_set_single_float( out_min_tensor, -out_max)!= 0
		|| tensor_set_single_float( out_max_tensor, out_max)!= 0) {
		return errlog(nn,"out min/max tensor too small");
	}
	struct elementwise_funcs const *ew_funcs =
			( node_type == OP_QuantizedMul_8x8to32_ref) ? &Mul_i832_funcs : &Mul_i832_hvx_funcs;

	return nn_elementwise_with_broadcast( self, nn, ew_funcs,NULL, NULL, &sc );

}


struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to32 = {
	.execute = addsub_832_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedSub_8p8to32 = {
	.execute = addsub_832_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedAdd_8p8to32_ref= {
	.execute = addsub_832_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedSub_8p8to32_ref= {
	.execute = addsub_832_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};


struct nn_node_ops nn_ops_for_QuantizedMul_8x8to32 = {
	.execute = mul_832_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
    .flags = NN_NODE_FLAG_CLS_QUANTMUL8TO32,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedMul_8x8to32_ref= {
	.execute = mul_832_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(6),
	.n_outputs = NN_IOCOUNT(3),
};
