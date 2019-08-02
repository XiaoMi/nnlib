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
#include "nn_string_map.h"
#include "hvx_inlines.h"

//
// QuantizedTanh_16
// QuantizedSigmoid_16
//
// Inputs:
//   0: array of int16
//   1: scalar,float, min of input
//   2 :scalar,float, max of input   (note: it is assumed max = -min)
// Outputs:
//   0: array of int16; tanh of first input
//   1: output min, -1.0f
//   2: output max, +1.0f
//
struct mult_shift {
	int mult;
	int shift;
};
#define MAX_THREADS 2
#define NOMINAL_JOB_SIZE 2048		// number of elements per 'job'


struct tanh_sigmoid_16_runstate {
	int16_t const * inp;
	int16_t  * outp;
	struct mult_shift inscale;
	int is_sigmoid;
	uint32_t total_length;
	int total_jobs;				// # jobs on this run (>=1)
	int num_threads;			// <= total_jobs
	int n_per_job;				// elements per job on this run (multiple of 128)
	volatile int curr_job;		// used to distribute jobs
	nn_sem_t done_sem;
};

static int
set_input_scale( struct nn_graph *nn,
		struct mult_shift *dst,		// result goes here
		int is_sigmoid,
		float minval )
{
	int scale = 0;
	int shift = 0;
	if( minval < -1.e-5f){
		float fscale= (-3./8)* minval;		// what we need to mul by
		if( is_sigmoid ) fscale *= 0.5f;
		// use a margin when extracting exponent; so result won't round to 32768
		int expo = flt_getexp( fscale * 1.0000305f);	// exponent (with margin)
		shift = 15-expo;						// amount to right shift after mul
		if( shift > 0 && shift <= 18){				// check reasonable range...
			shift = min_i32(shift,15);				// for small values, reduce mag of scale
			scale = roundf_i32(flt_ldexp( fscale, shift) );
		}
	}
	if( scale == 0 ){
		return errlog(nn,"bad range for input: min = %f",  minval);
	}
	dst->mult = scale;
	dst->shift = shift;
	return 0;
}

static void tanh_sigmoid_work_func( struct nn_graph *nn, void *rstpv);

static int tahn_sigmoid_16_execute(struct nn_node *self, struct nn_graph *nn)
{
	struct tensor const * in_tensor = self->inputs[0];
	struct tensor const * in_min_tensor = self->inputs[1];
	struct tensor  * out_tensor = self->outputs[0];
	struct tanh_sigmoid_16_runstate runstate;

	int is_sigmoid = ( self->node_type == OP_QuantizedSigmoid_16);
	char const *nm = (is_sigmoid? "sigmoid_16": "tanh_16");

	int k =  set_input_scale( nn, &runstate.inscale, is_sigmoid, tensor_get_float(in_min_tensor,0));
	if ( k != 0) return k;

	logmsg(nn,2,"begin %s node at %p", nm, self);
	k = tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QINT16);
	if( k != 0) return errlog(nn,"output too small");

	runstate.is_sigmoid = is_sigmoid;
	runstate.inp = (int16_t const*)in_tensor->data;
	runstate.outp = (int16_t *)out_tensor->data;
	//
	// find size of workload; determin #jobs, job size, #threads
	//
	uint32_t elements  = tensor_element_count( in_tensor);
	runstate.total_length = elements;
	int njobs = (elements + (NOMINAL_JOB_SIZE>>2))/NOMINAL_JOB_SIZE;

	if( njobs <= 1){		// single job
		njobs = 1;
		runstate.n_per_job = elements;
	}else{
		// divide work into 'njobs' jobs, rounding up to a multiple of 128
		// it's maybe possible that the rounding-up will mean that some jobs are empty - let the thread
		// deal with that.
		runstate.n_per_job = 128* ( (elements + (128u*njobs)-1)/(128u*njobs) );
	}
	runstate.num_threads = min_i32( njobs,MAX_THREADS);
	runstate.total_jobs = njobs;
	runstate.curr_job = 0;
	logmsg(nn,4,"tanh/sigmoid_16: sc = *%d >> %d; %u done in %d groups of %u",
			runstate.inscale.mult, runstate.inscale.shift,
			(unsigned)elements, njobs, (unsigned)runstate.n_per_job);

	nn_sem_init( &runstate.done_sem, 0);
	for( int i = 0; i < runstate.num_threads; i++ ){
		nn_os_work_for_vector( nn, tanh_sigmoid_work_func, &runstate);
	}

	tensor_set_single_float( self->outputs[1], -1.0f);
	tensor_set_single_float( self->outputs[2], 1.0f);

	nn_sem_wait_n_times( & runstate.done_sem, runstate.num_threads);

	logmsg(nn,2,"end %s node at %p", nm, self);

	return 0;

}
////////////////////////////////////////////////////////////////////
static inline HVX_Vector
tanh_hvx_prescale( HVX_Vector xin, int scaleby, int rsh )	// rsh must be 0..15
{
	xin = Q6_Vh_vabs_Vh( xin );
	HVX_VectorPair prod = Q6_Wuw_vmpy_VuhRuh( xin, Q6_R_combine_RlRl(scaleby,scaleby));
	HVX_Vector x1 = Q6_Vuh_vasr_VwVwR_sat( Q6_V_hi_W(prod), Q6_V_lo_W(prod), rsh);
	x1 = Q6_V_vnot_V(x1);			// invert...
	HVX_VectorPair x1sq = Q6_Wuw_vmpy_VuhVuh( x1,x1);	// square that
	return Q6_Vh_vshuffo_VhVh(  Q6_V_hi_W(x1sq), Q6_V_lo_W(x1sq)); // >>16
}
// This evaluates the polynomial , in xres = x2 & 0xFFF: (xres is considered to
// have 12 fractional bits; c3,c2,c1,c0 have 23,20,17,15 resp.)
// This also applies the sign of xin.
static inline HVX_Vector
tanh_hvx_cubic( HVX_Vector x2, HVX_Vector xin, HVX_Vector c3, HVX_Vector c2, HVX_Vector c1, HVX_Vector c0 )
{
	HVX_Vector xres = Q6_V_vand_VV( x2, Q6_V_vsplat_R(0x0FFF0FFF));
	HVX_Vector resx2 = Q6_Vh_vadd_VhVh( xres,xres);
	HVX_Vector tmp = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vmpy_VhVh_s1_rnd_sat(c3, xres ), c2 );	 // 20 bit intermediate
	tmp = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vmpy_VhVh_s1_rnd_sat(tmp, xres ), c1 );					// 17 bit intermediate
	HVX_Vector result = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vmpy_VhVh_s1_rnd_sat(tmp, resx2 ), c0 );	// 15 bit result

	HVX_Vector xsign = Q6_Vh_vasr_VhR( xin, 15);
	// that's the result! if input < 0, need to ~ and add 1.
	result = Q6_V_vxor_VV( result, xsign);
	return Q6_Vh_vsub_VhVh_sat( result, xsign );
}
static inline
__attribute__((unused))
HVX_Vector tanh_hvx_i16( HVX_Vector xin, int scaleby, int rsh )
{
	HVX_Vector x2 = tanh_hvx_prescale( xin, scaleby, rsh);	// do 'prescale'; find x2

	// extract the table select code: x2 >> 12
	HVX_Vector tablesel = Q6_Vuh_vlsr_VuhR( x2,12);
	HVX_Vector table = *(HVX_Vector const *) tanh_16_cubic_lookup;
	// extract the 'residue' : 12 lsbs
	// get the coefficients from table
#if  __HEXAGON_ARCH__ >= 62
	HVX_Vector c3 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,0 ));
	HVX_Vector c2 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,1 ));
	HVX_Vector c1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,2 ));
	HVX_Vector c0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,3 ));
#else
	// need to break out the table values in two parts and
	// make a second tablesel with +32
	HVX_Vector tablesel_2 = Q6_V_vor_VV( tablesel, Q6_V_vsplat_R(0x00200020));	// +2 to table sel
	HVX_Vector tab1 = Q6_V_vror_VR( table, 64);	// 32,34 .. 62 are section 1
	HVX_Vector c3 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(tablesel, table,0 ));
	HVX_Vector c2 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(tablesel, tab1,0 ));
	HVX_Vector c1 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(tablesel_2, table,2 ));
	HVX_Vector c0 = Q6_V_lo_W(Q6_Wh_vlut16_VbVhR(tablesel_2, tab1,2 ));

#endif
	return tanh_hvx_cubic( x2, xin, c3,c2,c1,c0);
}

//
// same thing but two at once, so that the lut16's are fully used.
// Each can have a different 'rsh' value.
static inline
HVX_VectorPair tanh_hvx_i16_x2( HVX_Vector xin0, HVX_Vector xin1, int scaleby, int rsh0, int rsh1 )
{

	HVX_Vector x2_0 = tanh_hvx_prescale( xin0, scaleby, rsh0);	// do 'prescale'; find x2
	HVX_Vector x2_1 = tanh_hvx_prescale( xin1, scaleby, rsh1);	// do 'prescale'; find x2


	HVX_Vector table = *(HVX_Vector const *) tanh_16_cubic_lookup;

	// combine lookup table indices [bits 15..12 of each uh lane] to a single register
	HVX_Vector tablesel = Q6_Vb_vshuffo_VbVb( x2_1, x2_0);		// upper bytes
	tablesel = Q6_Vuh_vlsr_VuhR( tablesel,4);					// >>4

	// get the coefficients from table
#if __HEXAGON_ARCH__ >= 62
	HVX_VectorPair c3 = Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,0 );
	HVX_VectorPair c2 = Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,1 );
	HVX_VectorPair c1 = Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,2 );
	HVX_VectorPair c0 = Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,3 );
#else
	// need to 'clean up' the table sel
	tablesel = Q6_V_vand_VV( tablesel, Q6_V_vsplat_R( 0x0F0F0F0F));
	// and make a second one with +32
	HVX_Vector tablesel_2 = Q6_V_vor_VV( tablesel, Q6_V_vsplat_R(0x20202020));	// +2 to table sel
	// need to break out the table values to two
	HVX_Vector tab1 = Q6_V_vror_VR( table, 64);	// 32,34 .. 62 are section 1
	HVX_VectorPair c3 = Q6_Wh_vlut16_VbVhR(tablesel, table,0 );
	HVX_VectorPair c2 = Q6_Wh_vlut16_VbVhR(tablesel, tab1,0 );
	HVX_VectorPair c1 = Q6_Wh_vlut16_VbVhR(tablesel_2, table,2 );
	HVX_VectorPair c0 = Q6_Wh_vlut16_VbVhR(tablesel_2, tab1,2 );
#endif

	HVX_Vector y_0 = tanh_hvx_cubic( x2_0, xin0,
				Q6_V_lo_W(c3),Q6_V_lo_W(c2),Q6_V_lo_W(c1),Q6_V_lo_W(c0));
	HVX_Vector y_1 = tanh_hvx_cubic( x2_1, xin1,
				Q6_V_hi_W(c3),Q6_V_hi_W(c2),Q6_V_hi_W(c1),Q6_V_hi_W(c0));
	return Q6_W_vcombine_VV( y_1, y_0);
}
////////////////////////////////////////////////////////////////////
static void
tanh_sigmoid_work_func( struct nn_graph *nn, void *rstpv)
{
	struct tanh_sigmoid_16_runstate *rstp = (struct tanh_sigmoid_16_runstate *)rstpv;

	int njobs = rstp->total_jobs;
	unsigned nperj = rstp->n_per_job;
	unsigned total_length = rstp->total_length;
	int inscale = rstp->inscale.mult;
	int inrsh = rstp->inscale.shift;

	//
	// hack to get tanh and sigmoid in the same processing loop with little
	// overhead:
	//  tanh:   q_is_sigmoid = all 0,  vone = (don't care)
	//  sigmoid:   q_is_sigmoid = all 1;  vone = splat_h ( 0x7fff)
	//
	//  when q_is_sigmoid, result is vavg( tanh_result, vone )
	//

	int is_sigmoid = rstp->is_sigmoid;
	HVX_Vector vone = Q6_V_vsplat_R( is_sigmoid?0x7fff7fff: 0);
	HVX_VectorPred q_is_sigmoid = Q6_Q_vcmp_gt_VhVh( vone, Q6_V_vzero());

	// TODO prefetch

	int ijob;
	while( ijob = __sync_fetch_and_add( &rstp->curr_job, 1),   ijob < njobs){
		unsigned offset = ijob * nperj;			// start offset within the whole tensor
		if( offset >= total_length) break;
		int nrun = min_i32( total_length-offset,nperj);		// do this many

		int16_t const * in_ptr = rstp->inp + offset;
		int16_t * out_ptr = rstp->outp + offset;

		int nv64 = (nrun + 63)>>6;		// # of 'vectors' (rounded up)
		int nv128 = nv64>>1;			// # of full 128


		for( int i = 0 ; i< nv128; i++){
			HVX_Vector in0 = *(HVX_Vector const *)in_ptr;
			HVX_Vector in1 = *(HVX_Vector const *)(in_ptr+64);
			HVX_VectorPair v_tanh = tanh_hvx_i16_x2(in0, in1, inscale, inrsh, inrsh );
			HVX_Vector out0 = Q6_V_lo_W(v_tanh);
			HVX_Vector out1 = Q6_V_hi_W(v_tanh);
			out0 = Q6_V_vmux_QVV( q_is_sigmoid,  Q6_Vh_vavg_VhVh_rnd(out0,vone),out0);
			out1 = Q6_V_vmux_QVV( q_is_sigmoid,  Q6_Vh_vavg_VhVh_rnd(out1,vone),out1);
			*(HVX_Vector *)out_ptr = out0;
			*(HVX_Vector *)(out_ptr+64) = out1;
			in_ptr += 128;
			out_ptr += 128;
		}
		if( nv64 & 1){		// odd group of 64 at  the end
			HVX_Vector in0 = *(HVX_Vector const *)in_ptr;
			HVX_Vector out0 = tanh_hvx_i16(in0,inscale, inrsh);
			out0 = Q6_V_vmux_QVV( q_is_sigmoid,  Q6_Vh_vavg_VhVh_rnd(out0,vone),out0);
			*(HVX_Vector *)out_ptr = out0;

		}

	}
	nn_sem_post( &rstp->done_sem);
}


/*
static int tahn_sigmoid_16_check(struct nn_node *self, struct nn_graph *nn)
{
	char const *nm = op_type_to_string_alt(self->node_type, "unknown");

	// must have 3 inputs and 3 outputs
	return node_check_inputs_outputs_n( self, nn, nm, 3, 3 );
}
*/

struct nn_node_ops nn_ops_for_QuantizedTanh_16 = {
	.execute = tahn_sigmoid_16_execute,
	.check = NULL,//tahn_sigmoid_16_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedSigmoid_16 = {
	.execute = tahn_sigmoid_16_execute,
	.check = NULL, //tahn_sigmoid_16_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
};

