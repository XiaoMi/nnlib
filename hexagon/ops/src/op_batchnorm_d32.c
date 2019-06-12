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
//#define TEST_PERFORMANCE
#include "hvx_inlines.h"

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
/*
 *
 * This contains implementations for batchnorm_d32
 *  Inputs are:
 *  *  main input, (b,w,h,d)		 		- d32 format;  + associated min/max
 *  * scale input (1,1,1,d) or (1,1,1,1)	- flat qu8 format   + associated min/max
 *  (3) bias input  (1,1,1,d) or (1,1,1,1)  - flat qu8 format   + associated min/max
 *  (4) output min  (1,1,1,1) float  (may be -inf)
 *  (5) output max  (1,1,1,1) float  (may be +inf)
 *
 *  Output is d32, shape (b,w,h,d)
 *
 *  The operation is is to multiply the input by 'scale' then add 'bias', and requantize
 *  the result; if applicable it will be clipped to out_max, out_min.
 *
 *  Overview:
 *    Operation is (conceptually)
 *         (1) (in[i]-inz)* (scale[d]-scalez) * 128, 32 bits, and approx +/- 2^23 worst case
 *         (2) add 'bias', prescaled to the units of result (1)
 *         (3) Add a result_offset
 *         (4) mul by a 16-bit scale factor result_scale (i.e. with 15 fractional bits)
 *         (5) pack/sat to u8 with a pre-determined right shift.
 *
 *    Actual operation is:
 *         (1)   in[i] * (scalex[d]*result_scale/32k)			 (scalex[d]  = [scale[d]-scalez]*128
 *         (2)  add  ((doff[d]+result_offset) * result_scale/32k)
 *         (3) pack/sat to u8 with a pre-determined right shift.
 *        doff[d] is  BiasAdapt*(BiasQ[d]-Biaz0)   (see below)
 *                   -  scx[d]*InputZero
 *         .. so the multiplication by 'result_scale' is thus factored into the
 *         scalex[d] and doff[d] which are common to all elements at depth d.
 *
 *	For this operation, we can keep track of the min/max values in each 8-bit unsigned lane, in order to determine the
 *	overall ranging (the same operation is applied to the min/max results, up to step (2)
 *
 *
 *     Operation is:
 *        Res = Data[] * Scale[] + Bias[]
 *        Res = Data_step * (  DataQ[] - Data0) * Scale_step *  ( ScaleQ[] - Scale0 ) + BiasStep*(BiasQ[]-Bias0)
 *        Res = DS128_step * (  (DataQ[]- Data0) * ( ScaleQ[] - Scale0)*128)       + BiasStep*(BiasQ[]-Bias0)
 *        Res = DS128_step * [  (DataQ[]- Data0) * ( ScaleQ[] - Scale0)*128)       + BiasAdapt*(BiasQ[]-Bias0)]
 *
 *             .. where DS128_step = Data_step*Scale_step/128
 *                      BiasAdapt = BiasStep/DS128_step
 *
 *       Now, with Scx[] = ( ScaleQ[] - Scale0)*128:
 *       Res/DS128_step   =  (DataQ[]- Data0) * Scx[])       + BiasAdapt*(BiasQ[]-Bias0)
 *       Res/DS128_step   =  DataQ[] * Scx[]     + BiasAdapt*(BiasQ[]-Bias0)- (Data0*Scx[])
 *
 *     But  Res[] = Res_step * (ResQ[] - Res0)
 *          Res[]/DS128_step =    (ResQ[] - Res0) * Res_step/ DS128_Step
 *
 *       let ResAdapt = DS128_Step/Res_step
 *
 *          ResQ[] / Res_Adapt  = Res[]/DS128_step  +  Res0/Res_Adapt
 *
 *
 *     => ResQ[] /Res_Adapt =      DataQ[] * Scx[]     + BiasAdapt*(BiasQ[]-Bias0)- (Data0*Scx[]) + Res0/Res_Adapt
 *
 *     And    DataQ[] * Scx[]  is the result of the multiplication   (it's actually DataQ[b,h,w,d] * Scx[d])
 *     .. and the step (2) result adds
 *            BiasX[d] =    BiasAdapt*(BiasQ[]-Bias0)- (Data0*Scx[]) + Res0/Res_Adapt
 *
 *     ..  where only the last term depends on the output scaling. So, after a 'failed' pass, we can subtract that from the
 *     min/max to find the range of Res[]/DS128_step, which gives the actual range of the result.
 *
 *            And once we have   ResQ[] / Res_Adapt, we just multiply by ResAdapt and clip to range.
 *
 * Note that Res0/Res_Adapt = -out_min/DS128_Step
 *
 */

struct batchnorm_scaling {
	uint8_t data_zero;
	uint8_t scale_zero;
	uint8_t bias_zero;
	uint8_t bias32_flag;
	float data_step;
	float scale_step;
	float ds128_step;		// data_step * scale_step/128

	// scaling u8 bias to get a value to add to the scale*data prod:
	// subtract bias_zero, << bias_lsh, mul by bias_scale, and >> the 32-bit result by bias_rsh
	// (for bias32 mode it's (bias_in*bias_scale/32k), and then >> by bias_rsh (which may be < 0)
	int16_t bias_scale;
	int16_t bias_lsh;		// apply << lsh before *bias_scale (range is 0..6)
	int16_t bias_rsh;		// apply >>  after *bias_scale;

	// 'late' scaling parms (depend on output range)
	int16_t result_scale;	// this is ResAdapt x 2^(15 + result_rsh)
	int16_t result_rsh;		// shift converting w->ub
	int16_t result_rsh1, result_rsh2;		// done in 2 parts; fist is 0..15; second is is 1..7
	int32_t result_offset;	// this is Res0/Res_Adapt
	float result_step_comp;	// this is result_step/(2^result_rsh)
	float result_zero_comp;	// result_0 * 2^result_rsh
};

// set the scaling parameters for a given input range (all that can
// be done without knowing the output range).
// returns 0 if ok
//
static inline int set_scale_from_input(
		struct batchnorm_scaling * scp,
		float data_min, float data_max,
		float scale_min, float scale_max,
		float bias_min, float bias_max,
		int bias32 )	// is bias 32-bits?
{
	float data_range = data_max-data_min;
	float data_step = flt_div_255( data_range );
	float scale_range = scale_max-scale_min;
	float scale_step = flt_div_255( scale_range );
	float bias_range = bias_max-bias_min;
	float bias_step;
	float ds128_step = data_step * scale_step * (float)(1./128.);
	scp->data_zero = saturate_u8( roundf_i32( -data_min/ data_step));
	scp->scale_zero = saturate_u8( roundf_i32( -scale_min/ scale_step));
	scp->data_step = data_step;
	scp->scale_step = scale_step;
	scp->ds128_step = ds128_step;
	scp->bias32_flag = bias32;
	if( !bias32){
		bias_step = flt_div_255( bias_range );
		scp->bias_zero = saturate_u8( roundf_i32( -bias_min/ bias_step));
	}else{
		bias_step = bias_range * (float)(1./(65536.0*65536.0));
		scp->bias_zero = 0;
	}

	/*
		printf("input scaling: in is %f ... %f, step = %f zero= %d; "
		 "scale is %f ... %f, step = %f zero= %d; "
		 "bias is %f ... %f, step = %f zero= %d\n",
		 data_min, data_max, data_step, scp->data_zero,
		 scale_min, scale_max, scale_step, scp->scale_zero,
		 bias_min, bias_max, bias_step, scp->bias_zero ); */
	// bias_adapt is 128*bias_step/(data_step*scale_step)
	//    = 32640 * bias_range/(data_range * scale_range)
	// tends to be about 30K
	// Bias scaling is done as follows:
	//        scale_bias[d]      =   ((bias[d]-bias_zero)<<shL) * bias_scale >> shR
	//   where bias_scale * 2^(shL-shR)  ~=~ bias_adapt
	//     .. bias_scale is hopefully 16K..3K
	// So if bias_adapt is <= 32K, we use shL = 0, and shR >> after allowing bias_scale to be bigger;
	// If  > 32K,  we use shL in 0..6 to keep bias_scale in range and shR = 0.
	// We can't support bias_adapt > 2^21
	//   i.e. bias_range/(data_range * scale_range) can't exceed 64.2

	float bias_adapt = bias_step/ds128_step;
	int ba_exp = flt_getexp( bias_adapt * 1.00001831f)-15;	// exponent (with margin)
	if( !bias32){
		if( ba_exp > 6) return -1;
		ba_exp = max_i32(ba_exp,-31);	// absurdly small #, limit to avoid overshift
		scp->bias_lsh = max_i32( ba_exp, 0);
		scp->bias_rsh = max_i32( -ba_exp, 0);
	}else{
		if( ba_exp > 1) return -1;	// avoid rsh < -16
		ba_exp = max_i32(ba_exp,-46);	// avoid rsh > 31
		scp->bias_lsh = 0;
		scp->bias_rsh = -(15+ba_exp);	// may be -16 .. 31 (but only <0 when bias range is underused)
	}
	// scale by -ba_exp,and round to frac, will be 16k..32k range (unless absurdly small)
	scp->bias_scale = saturate_i16( roundf_i32( flt_ldexp(bias_adapt,-ba_exp)));


	/*printf("ds128_step = %g;  bias_adapt  = %g  (%d <<%d) >> %d\n" ,ds128_step, bias_adapt,
			scp->bias_scale, scp->bias_lsh, scp->bias_rsh );*/

	return 0;
}

// set the scaling parameters for a proposed output range
// returns 0 if ok
//
static inline int set_scale_for_output(
		struct batchnorm_scaling * scp,
		float out_min, float out_max )
{
	// res_adapt is ds128_step/out_step
	float res_scale = 255.0f  / (out_max-out_min);
	float res_adapt =  res_scale *scp->ds128_step;

	int ra_exp = - flt_getexp( res_adapt * 1.00001831f);		// exponent (with margin)
	// the exponent is the right-shift we need to do.
	if( ra_exp < 1|| ra_exp > 29) return -1;				// not sane
	ra_exp = min_i32( ra_exp, 22);					// will generate a subnormal result_scale if this clips

	int32_t res_offs = 0;		// output offset for out_min
	float res_zero = 0.0f;
	if( out_min != 0.0f){
		float rbias = -out_min/scp->ds128_step;
		res_offs = roundf_i32( rbias);
		res_zero = -out_min*res_scale;
	}
	scp->result_offset = res_offs;
	// round scale to nearest
	scp->result_rsh = ra_exp;
	scp->result_scale = saturate_i16( roundf_i32( flt_ldexp(res_adapt,15+ra_exp)));
	// this is used to convert min/max values to 'app' values.
	scp->result_step_comp = flt_ldexp(flt_div_255(out_max-out_min), -ra_exp);
	scp->result_zero_comp = flt_ldexp(res_zero, ra_exp);
	// break ra_exp up into rsh1 (for 32->16) and rsh2 (for 16->8) to accommodate limits in the
	// hvx shift instructions. they must be in range 0..15 and 1..7 respectively
	// (the rounding is done on the second one). since ra_shift is 1..22, this is always
	// possible.
	int rash2 = max_i32( 1, ra_exp-15);		// 1..7
	scp->result_rsh1 = ra_exp  - rash2;		// 0..15
	scp->result_rsh2 = rash2;

	/*printf("Output: %f .. %f  step = %f  zero = %.2f  res_adapt = %g  = %d/32k >> (%d+%d); res0_off = %d  result_step/z_comp = %g/%f\n",
			out_min, out_max, 1.0f/res_scale, res_zero, res_adapt, scp->result_scale, scp->result_rsh1, scp->result_rsh2, (int)scp->result_offset,
			scp->result_step_comp, scp->result_zero_comp);*/


	return 0;
}

// persistent info, attached to 'opaque' by check function
struct batchnorm_d32_info {
	float out_min,out_max;						// current out min/max
	float out_min_specified, out_max_specified;	// original specified values
	int min_max_precalc;		// bit 0: min precalc; bit 1: max precalc
	int run_yet;				// any previous runs? will estimate output range when no.
};
struct batchnorm_d32_runstate {
	struct batchnorm_scaling scl;		// scaling info
	struct shape shp;					// the shape of the operation
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	struct batchnorm_d32_info *info;	// pointer to 'info' (persistent data)
	int16_t left_mask;					// of width units on left which are padding (0..3)
	int16_t active_width;				// leftmask + width
	int16_t scale_broadcast;			// 0 if scale shape is (1,1,1,d), 1 if (1,1,1,1)
	int16_t bias_broadcast;				// same for bias shape
	uint8_t const * scale_data;
	uint8_t const * bias_data;

	int n_threads;						// number of threads running.
	int njobs;							// d32 units * batches
	volatile int next_job;				// next to be handled
	volatile int next_minmax;			// for dividing up next_minmax
	nn_sem_t done_sem;					// when threads are done

	HVX_Vector * scale_bias_data;		// preprocessed scale/bias (in scratch)
	HVX_Vector * minmax_result;			// one per thread, for storing min/max
};

static void fill_depth_consts( struct nn_graph * nn, void * runstv);
static void batchnorm_worker( struct nn_graph * nn, void * runstv);

static HVX_Vector batchnorm_run_single_plane(		// TODO: inline this
		struct batchnorm_d32_runstate const * rstp,
		uint8_t const *inp,
		uint8_t *outp,
		HVX_Vector const * sclbias_data,
		HVX_Vector prev_minmax);

//
// when using 'bias32', the range of the bias could be far less than implied by the minmax.
// This is used to get the proper range on the first run.
// it is permitted to set min > 0 or max < 0, when that is the case.
//
static void get_bias32_endpoints( int32_t const * data, int npts, float * biasmin, float * biasmax)
{
	float range = *biasmax - *biasmin;	// initially these are the 'endpoints'.
	int32_t minq = data[0];
	int32_t maxq = minq;
	for( int i = 1 ; i < npts; i++){
		int32_t k = data[i];
		minq = min_i32(minq, k);
		maxq = max_i32(maxq, k);
	}
	float scale = range * 0x1.0p-32f;
	*biasmin  = (float)minq * scale;
	*biasmax  = (float)maxq * scale;
}



#define BATCHNORM_MAX_THREADS 2

static int batchnorm_d32_execute(struct nn_node *self, struct nn_graph *nn)
{

	logmsg(nn,2,"batchnorm_d32 execute. self=%p ",self);
#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *scale_tensor = self->inputs[1];
	const struct tensor *bias_tensor = self->inputs[6];

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	struct batchnorm_d32_info * info = (struct batchnorm_d32_info *) self->opaque;

	struct batchnorm_d32_runstate runstt;
	runstt.info = info;
	runstt.shp = in_tensor->shape;

	int depth = runstt.shp.depth;
	int bias32 = (self->node_type == OP_QuantizedBatchNorm_8x8p32to8_d32);

	if( in_tensor->format.depth_pad[0] != 0 ) return errlog(nn,"can't process depth_pad_before > 0");
	int tmp = scale_tensor->shape.depth;
	if( scale_tensor->shape.batches_height != (1ull<<32)+1 ||
			scale_tensor->shape.width != 1 || (tmp != 1 && tmp != depth))
		return errlog(nn, "bad scale shape");
	runstt.scale_broadcast = (tmp== depth)?0:1;
	runstt.scale_data = (uint8_t const*)scale_tensor->data;

	tmp = bias_tensor->shape.depth;
	if( bias_tensor->shape.batches_height != (1ull<<32)+1 ||
			bias_tensor->shape.width != 1 || (tmp != 1 && tmp != depth))
		return errlog(nn, "bad bias shape");
	runstt.bias_broadcast = (tmp== depth)?0:1;
	runstt.bias_data = (uint8_t const*)bias_tensor->data;

	// load all the scalar inputs.
	float scalar_in[6];
	for( int i =0; i< 6;i++){
		const struct tensor * scl_tensor = self->inputs[((i<4)?2:3) + i];	// 2,3,4,5,7,8
		if( scl_tensor->shape.batches_height != (1ull<<32)+1
		|| scl_tensor->shape.width_depth != (1ull<<32)+1 )
			return errlog(nn,"bad shape for scalar input");
		scalar_in[i] = tensor_get_float( scl_tensor, 0);
	}

	runstt.tin = tensor_addressing_d32( in_tensor);

	if ( tensor_out_prepare_d32_sameas(out_tensor, in_tensor) != 0 ){
		return errlog(nn,"output too small");
	}
	runstt.tout = tensor_addressing_d32( out_tensor);

	// adjust for left-padding		runstt.next_job = -1;			// -1 = set up

	{
		int pleft = in_tensor->format.width_pad[0] & 3;
		runstt.left_mask = pleft;
		runstt.active_width = pleft + runstt.shp.width;
		// vector align the data pointers
		runstt.tin.data -= pleft * 32;
		runstt.tout.data -= pleft * 32;
	}


	int res = set_scale_from_input( &runstt.scl,
			scalar_in[0], scalar_in[1],		// input min/max
			scalar_in[2], scalar_in[3],		// scale min/max
			scalar_in[4], scalar_in[5], bias32);		// bias min/max

	if( res != 0) return  errlog(nn, "failed to set scaling");

	// if this is the first run, take a guess at the output range
	// we use 20% of the calculated max range, which is unlikely to be sufficient but
	// it's better to underestimate.
	if( !info->run_yet){
		float range_factor = 0.20f;
		int min_max_precalc = info->min_max_precalc;
		if( (min_max_precalc!= 3) && bias32){
			// get actual range, for bias32;likely to be a fair bit smaller than min/max. This function
			// modifies scalar_in[4] and scalar_in[5].
			get_bias32_endpoints( (int32_t const *)bias_tensor->data, bias_tensor->shape.depth, &scalar_in[4], &scalar_in[5]);
		}
		float calculated_min, calculated_max;
		if( (min_max_precalc &1) != 0 ){
			calculated_min = info->out_min_specified;
		}else{
			float min_out = fminf( scalar_in[1]*scalar_in[2], scalar_in[0]*scalar_in[3]) + scalar_in[4];
			calculated_min = range_factor *fminf(0.0f, min_out);
		}
		if( (min_max_precalc &2) != 0 ){
			calculated_max = info->out_max_specified;
		}else{
			float max_out = fmaxf( scalar_in[1]*scalar_in[3], scalar_in[0]*scalar_in[2]) + scalar_in[5];
			calculated_max = fmaxf(range_factor * max_out, calculated_min+1e-4f);
		}
		info->out_min = calculated_min;
		info->out_max = calculated_max;
		adjust_minmax_for_zero_with_constraints( &info->out_min, & info->out_max,min_max_precalc);
		info->run_yet = 1;
	}

	runstt.njobs = runstt.shp.batches * runstt.tin.nd32;
	int n_threads = min_i32( runstt.njobs, BATCHNORM_MAX_THREADS);
	runstt.n_threads = n_threads;

	// allocate scratch;
	nn_scratch_reset(nn);
	int vecs_for_work = ((depth+31)>>5)*2;		// 2 vecs per depth unit
	uint8_t * scr  = (uint8_t *)nn_scratch_alloc( nn, 128*(vecs_for_work + n_threads));
	if( scr == NULL)
		return errlog(nn,"did not get scratch, %d bytes", (int)(128*(vecs_for_work + n_threads)));
	runstt.minmax_result  = (HVX_Vector *)scr;
	runstt.scale_bias_data  = (HVX_Vector *)(scr+ 128*n_threads);

	nn_sem_init(&runstt.done_sem,0);

	// we only need to build the scale/range data the first time, since
	// it's not affected by output ranging.
	// @@ this will need to launch a thread, when it's vectorized
	//
	nn_os_work_for_vector( nn, fill_depth_consts,(void*)&runstt);
	nn_sem_wait(&runstt.done_sem );

	int need_range_check = info->min_max_precalc != 3;

	int ran_again = 0;
	int need_rerun;
	do{		// loop back to here if we need to re-range

		res = set_scale_for_output( &runstt.scl, info->out_min, info->out_max);
		if( res != 0)
			return errlog(nn,"failed to set output scaling");

		// reset the indicies that the worker threads use to allocate buffer & jobs
		runstt.next_job = 0;
		runstt.next_minmax = 0;

		for( int i= 0; i < n_threads; i++ ){
			nn_os_work_for_vector( nn, batchnorm_worker,(void*)&runstt);
		}
		//
		// while we are waiting... find the bounds of the 32-bit result that will
		// round to 0 and FF
		// e.g. if result_rsh = 16, the value must be >= -0x8000 and <= 0xFF7FFF
		// (but we allow <= 0xFF8000 because we're generous... :-)
		int32_t lo_limit, hi_limit;
		{
			int rsh = runstt.scl.result_rsh;
			int bias = (1<<rsh)>>1;
			lo_limit = -bias;
			hi_limit = (0xFF<<rsh)+bias;
		}
		// and set the current outputs
		tensor_set_single_float( out_min_tensor, info->out_min);
		tensor_set_single_float( out_max_tensor, info->out_max);

		// wait for threads

		nn_sem_wait_n_times( &runstt.done_sem, n_threads);
		need_rerun = 0;
		if( need_range_check){
			// first reduce across the threads
			// each thread stores a vector { ~min, XX, max, xx, xx, ...}
			// so reduce both with 'max'
			int32_t *mmp = (int32_t*)runstt.minmax_result;
			int minall = mmp[0];
			int maxall = mmp[2];
			for( int i =1; i < n_threads;i++){
				mmp += 32;		// next vector
				minall = max_i32(minall, mmp[0]);
				maxall = max_i32(maxall, mmp[2]);
			}
			minall = ~minall;

			// are these out of bounds?
			int mm_precalc = info->min_max_precalc;
			float actual_max = ( (float)maxall - runstt.scl.result_zero_comp) * runstt.scl.result_step_comp;
			float actual_min = ( (float)minall - runstt.scl.result_zero_comp) * runstt.scl.result_step_comp;

			/*printf("i32 range is %d .. %d (against %d,%d); in floats this is %f .. %f\n",
					minall, maxall, (int)lo_limit, (int)hi_limit, actual_min, actual_max );*/

			float rangeexp = fmaxf(0.0f, 0.25f*( actual_max - actual_min));

			if( (mm_precalc & 2) == 0 ){		// max is negotiable
				if( maxall > hi_limit ){			// need to set new max
					info->out_max = fmaxf(0.0f, actual_max) + rangeexp;
					need_rerun = 1;
				}
			}
			if( (mm_precalc & 1) == 0 ){			// min is negotiable
				if( minall < lo_limit ){			// need to set new min
					info->out_min = fminf( 0.0f, actual_min) - rangeexp;
					need_rerun = 1;
				}
			}
			if( need_rerun){
				ran_again =1;
				// reset any 'fixed' points to avoid drift
				if( (mm_precalc&1)!=0){
					info->out_min = info->out_min_specified;
				}
				if( (mm_precalc&2)!=0){
					info->out_max = info->out_max_specified;
				}
				adjust_minmax_for_zero_with_constraints( &info->out_min, & info->out_max,mm_precalc);
				// @@in principle we can clear 'need_range_check here
			}
		}
	}while( need_rerun);
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("batchnorm_d32 cycles = %d (elements = %d) ran_again = %d\n",
			(end_time-start_time),
			(int)tensor_element_count(out_tensor), ran_again);
#endif
	logmsg(nn,2,"batchnorm_d32 %p done", self);
	return 0;
}


static void
batchnorm_worker( struct nn_graph * nn, void * runstv)
{
	struct batchnorm_d32_runstate * rstp = (struct batchnorm_d32_runstate *) runstv;

	uint32_t pf_height = rstp->shp.height;
	uint32_t pf_width = ((rstp->active_width+3) & ~3u)*32;
	uint32_t pf_stride = (rstp->tin.height_stride);

	// go through 'jobs'...
	HVX_Vector all_minmax = Q6_V_vsplat_R( -(1<<31));

	int nd32 = rstp->tin.nd32;	// # of d32 units
	int ijob;
	while(  ijob = __sync_fetch_and_add( &rstp->next_job, 1),  ijob < rstp->njobs)
	{
		int ibatch= 0;
		int id32 =  ijob;
		if( id32 >= nd32){
			ibatch = ijob/nd32;
			id32 -= ibatch*nd32;
		}
		uint8_t * input_ptr = rstp->tin.data + id32*rstp->tin.d32_stride + ibatch * rstp->tin.batch_stride;
		l2pref( input_ptr, pf_height, pf_width, pf_stride);

		uint8_t * output_ptr = rstp->tout.data + id32*rstp->tout.d32_stride + ibatch * rstp->tout.batch_stride;
		HVX_Vector const * scalebias_data = rstp->scale_bias_data + 2*id32;	// 2 vecs per slot

		all_minmax= batchnorm_run_single_plane( rstp, input_ptr, output_ptr, scalebias_data, all_minmax );
	}
	// store minmax
	// get 'minmax' buffer from the pool using 'next_minmax'
	{
		int i = __sync_fetch_and_add( &rstp->next_minmax,1);
		rstp->minmax_result[i] = all_minmax;
	}
	nn_sem_post( &rstp->done_sem);
}

//
// fill in the depth/scale tables:
// for each d32 of the depth, we will have
// - one vectors of 'Scx', and one of bias
//  The 'scx' vector is (in h)
//    [ Scx0 Sx2 .. Scx30  ]  (repeat)   [ scx1 Sx3 ...Scx31  ]  (repeat)
// and the bias is (in w )
//   [ d0  d4  ...   d28 ] [d1 .. d29] [ d2, d6 .. d30]  (d3, d7 .. d31]
//
// The 'scx' values are (scaleq[d]-scalez) * 128
// The 'd' values are BiasAdapt*(biasq[d]-biasz) - data0 *scx
//  where BiasAdapt *x is done as  (x << bias_lsh)*bias_scale >> bias_rsh
//
// If the depth is not a multiple of 32, the 'extra' slots in the last
// set must all be zero.
//
//
static void fill_depth_consts( struct nn_graph * nn, void * runstv)
{
	struct batchnorm_d32_runstate * rstp = (struct batchnorm_d32_runstate *) runstv;

	struct batchnorm_scaling const * scp = &rstp->scl;
	HVX_Vector * out= rstp->scale_bias_data;
	uint8_t const * in_scale = rstp->scale_data;
	uint8_t const * in_bias = rstp->bias_data;
	int scale_is_d1 = rstp->scale_broadcast;
	int bias_is_d1 = rstp->bias_broadcast;
	int depth = rstp->shp.depth;


	int scale_zero = scp->scale_zero;
	int bias_zero = scp->bias_zero;
	int data_zero = scp->data_zero;
	int brsh = scp->bias_rsh;
	int blsh = scp->bias_lsh;
	int bround = (1<<brsh)>>1;

#if 1	// vector version
	int bias32 = scp->bias32_flag;
	int bias_scl = Q6_R_combine_RlRl(scp->bias_scale,scp->bias_scale);
	int minus_dzero  = Q6_R_combine_RlRl(-data_zero,-data_zero);

	HVX_Vector vbround = Q6_V_vsplat_R(bround);
	HVX_Vector vscale0 = q6op_Vb_vsplat_R(scale_zero);
	HVX_Vector vscaleq = q6op_Vb_vsplat_R(in_scale[0]);
	HVX_Vector vbias0 = q6op_Vb_vsplat_R(bias_zero);
	int32_t bias0 = *(int32_t const*)in_bias;

	if( !bias32){
		HVX_Vector vbiasq = q6op_Vb_vsplat_R( bias0);

		// do up to 64 at once
		for(int dpos = 0; dpos < depth; dpos += 64){
			int dremain = min_i32(64, depth-dpos);	// # of valid slots here, 1..64
			if(!scale_is_d1){
				vscaleq = q6op_V_vldu_A( (HVX_Vector const*)in_scale);
				in_scale += 64;
			}
			if( !bias_is_d1){
				vbiasq = q6op_V_vldu_A( (HVX_Vector const*)in_bias );
				in_bias += 64;
			}
			// mask things off and subtract
			HVX_VectorPred mask = Q6_Q_vsetq_R(dremain);
			HVX_VectorPair scdiff = Q6_Wh_vsub_VubVub( q6op_V_vand_QV(mask,vscaleq), q6op_V_vand_QV(mask,vscale0));
			HVX_VectorPair bdiff = Q6_Wh_vsub_VubVub( q6op_V_vand_QV(mask,vbiasq), q6op_V_vand_QV(mask,vbias0));

			// reorg the scale values...
			// they are { 0,2 .. 30,  32,34 .. 62, xxx } and { 1,3, ..31,   33,35, ..63,xxx }
			// get all the good ones in one vector
			//// { 0..30, 32 .. 62, 1..31, 33..63}
			HVX_Vector scxall = Q6_V_lo_W( Q6_W_vshuff_VVR( Q6_V_hi_W(scdiff), Q6_V_lo_W(scdiff), 64));
			// now x 128
			scxall = Q6_Vh_vasl_VhR( scxall, 7);

			// now bias values
			HVX_Vector biash = Q6_V_lo_W( Q6_W_vshuff_VVR(  Q6_V_hi_W(bdiff), Q6_V_lo_W(bdiff), 64));// { 0..30, 32 .. 62, 1..31, 33..63}
			// apply bias scale
			biash = Q6_Vh_vasl_VhR( biash, blsh);
			HVX_VectorPair biasw = Q6_Ww_vmpy_VhRh( biash, bias_scl);
			// that has { 0,4..28  32,36 .. 60.  1,5.. 29   33..61} in low vec, and +2 in high
			// do right-shift
			biasw = Q6_W_vcombine_VV(
					Q6_Vw_vasr_VwR(   Q6_Vw_vadd_VwVw( Q6_V_hi_W(biasw), vbround), brsh),
					Q6_Vw_vasr_VwR(   Q6_Vw_vadd_VwVw( Q6_V_lo_W(biasw), vbround), brsh));
			// multiply scx by -data0 and add that
			biasw = Q6_Ww_vmpyacc_WwVhRh_sat( biasw,scxall, minus_dzero );

			// deal those out so we have { 0,4,..28, 1.. 29, 2..30, 3..31} in lo, +32 in high


			HVX_VectorPair bias_dealt = Q6_W_vdeal_VVR( Q6_V_hi_W(biasw), Q6_V_lo_W(biasw), -32 );
			// similar on scale... want  {0,2 .. 30;  0, 2 .. 30;  1, 3 .. 31;  1, 3 ..31} in 1st
			HVX_VectorPair scale_dealt = Q6_W_vdeal_VVR( scxall, scxall, 32 );
			out[0] = Q6_V_lo_W(scale_dealt);
			out[1] = Q6_V_lo_W(bias_dealt);
			if( dremain > 32){
				out[2] = Q6_V_hi_W(scale_dealt);
				out[3] = Q6_V_hi_W(bias_dealt);
				out += 4;
			}
		}
	}else{
		HVX_Vector vbiasq0 = Q6_V_vsplat_R(bias0);
		HVX_Vector vbiasq1 = vbiasq0;
		HVX_Vector vbias_scl = Q6_V_vsplat_R(bias_scl <<16);	// need in a vreg for 32-bit scaling

		// do up to 64 at once
		for(int dpos = 0; dpos < depth; dpos += 64){
			int dremain = min_i32(64, depth-dpos);	// # of valid slots here, 1..64
			if(!scale_is_d1){
				vscaleq = q6op_V_vldu_A( (HVX_Vector const*)in_scale);
				in_scale += 64;
			}
			if( !bias_is_d1){		// read 64 * 32-bit bias
				vbiasq0 = *( (HVX_Vector const*)in_bias );
				in_bias += 128;
				vbiasq1 = *( (HVX_Vector const*)in_bias );
				in_bias += 128;
			}
			if( dremain < 64){		// clear excess bias elements.
				HVX_VectorPred mask32 = Q6_Q_vsetq_R(dremain*4);
				if( dremain < 32 ){
					vbiasq1 = Q6_V_vzero();
					vbiasq0 = q6op_V_vand_QV( mask32, vbiasq0);
				}else{
					vbiasq1 = q6op_V_vand_QV( mask32, vbiasq1);
				}
			}
			// mask things off and subtract
			HVX_VectorPred mask = Q6_Q_vsetq_R(dremain);
			HVX_VectorPair scdiff = Q6_Wh_vsub_VubVub( q6op_V_vand_QV(mask,vscaleq), q6op_V_vand_QV(mask,vscale0));

			// reorg the scale values...
			// they are { 0,2 .. 30,  32,34 .. 62, xxx } and { 1,3, ..31,   33,35, ..63,xxx }
			// get all the good ones in one vector
			//// { 0..30, 32 .. 62, 1..31, 33..63}
			HVX_Vector scxall = Q6_V_lo_W( Q6_W_vshuff_VVR( Q6_V_hi_W(scdiff), Q6_V_lo_W(scdiff), 64));
			// now x 128
			scxall = Q6_Vh_vasl_VhR( scxall, 7);


			// deal bias twice, over the 2 vectors
			HVX_VectorPair biasdealt = Q6_W_vdeal_VVR( vbiasq1, vbiasq0, -4);
			biasdealt = Q6_W_vdeal_VVR( Q6_V_hi_W(biasdealt), Q6_V_lo_W(biasdealt), -4);
			// that has { 0,4..28  32,36 .. 60.  1,5.. 29   33..61} in low vec, and +2 in high

			// apply bias scale
			HVX_Vector biasw0 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( Q6_V_lo_W(biasdealt), vbias_scl);
			HVX_Vector biasw1 = Q6_Vw_vmpyo_VwVh_s1_rnd_sat( Q6_V_hi_W(biasdealt), vbias_scl);
			if( brsh >= 0){
				biasw0 = Q6_Vw_vasr_VwR(   Q6_Vw_vadd_VwVw( biasw0, vbround), brsh);
				biasw1 = Q6_Vw_vasr_VwR(   Q6_Vw_vadd_VwVw( biasw1, vbround), brsh);
			}else{
				biasw0 = Q6_Vw_vasl_VwR(   biasw0, -brsh);
				biasw1 = Q6_Vw_vasl_VwR(   biasw1, -brsh);
			}
			HVX_VectorPair biasw = Q6_W_vcombine_VV( biasw1, biasw0);
			// multiply scx by -data0 and add that
			biasw = Q6_Ww_vmpyacc_WwVhRh_sat( biasw,scxall, minus_dzero );

			// deal those out so we have { 0,4,..28, 1.. 29, 2..30, 3..31} in lo, +32 in high


			HVX_VectorPair bias_dealt = Q6_W_vdeal_VVR( Q6_V_hi_W(biasw), Q6_V_lo_W(biasw), -32 );
			// similar on scale... want  {0,2 .. 30;  0, 2 .. 30;  1, 3 .. 31;  1, 3 ..31} in 1st
			HVX_VectorPair scale_dealt = Q6_W_vdeal_VVR( scxall, scxall, 32 );
			out[0] = Q6_V_lo_W(scale_dealt);
			out[1] = Q6_V_lo_W(bias_dealt);
			if( dremain > 32){
				out[2] = Q6_V_hi_W(scale_dealt);
				out[3] = Q6_V_hi_W(bias_dealt);
				out += 4;
			}
		}

	}

#else		// scalar version

	int bfac = scp->bias_scale << blsh;
	int nd32 = (depth+31)>>5;
	int bias32 = scp->bias32_flag;


	int scale_bump = scale_is_d1 ? 0:1;
	int bias_bump = bias_is_d1 ? 0: (bias32?4:1);

	struct outrec {
		int16_t svals[4*16];	// one vector
		int32_t dvals[4*8];		// one vector
	};
	struct outrec *outp = (struct outrec *)out;

	// preclear the last unit, if it's not a multiple of 32.

	if( depth< nd32*32){
		struct outrec * outp_last = outp + (nd32-1);
		memset( outp_last, 0, 2*128);
	}
	int bias_overflow = 0;
	for(int id32 = 0; id32 < nd32; id32++){
		int dnow = min_i32( depth-id32*32,32);
		for(int i = 0; i < dnow; i++){
			int scx = ( *in_scale - scale_zero)<<7;
			int bias;
			if( !bias32){
				bias = ((*in_bias - bias_zero) * bfac + bround) >> brsh;
			}else{
				bias = *(int32_t const *)in_bias;
				bias = ((int64_t)bias * bfac  + 0x4000) >> 15;	// scale it
				if( brsh >= 0){
					bias = (bias + bround) >> brsh;
				}else{
					int64_t biasx = (int64_t)bias << (-brsh);
					bias = (int)biasx;
					if( bias != biasx) bias_overflow = 1;
				}
			}
			bias = bias - scx * data_zero;
			int16_t * scale_dest = & outp->svals[((i&1)?32:0) + (i>>1)];
			int32_t * bias_dest = &outp->dvals[ (i>>2) + ((i&3)*8)];
			scale_dest[0] = scx;
			scale_dest[16] = scx;
			bias_dest[0] = bias;

			in_scale += scale_bump;
			in_bias += bias_bump;
		}
		outp ++;
	}
	if( bias_overflow) logmsg(nn,0,"bias overflow in batchnorm");
#endif

	nn_sem_post( &rstp->done_sem);
}

//
// process a single 'depth-slice-plane'
// The min/max is accumulated on the supplied 'prev_minmax'
// and the result is returned; the minmax vector
// has this in 'w' slots: {  ~min, xx, max , xxx, .... xxx }
//
static HVX_Vector
batchnorm_run_single_plane(
		struct batchnorm_d32_runstate const * rstp,
		uint8_t const *inp,
		uint8_t *outp,
		HVX_Vector const * sclbias_data,		// 2 vecs for the current depth slice
		HVX_Vector prev_minmax
		)
{
	HVX_Vector vresult_offset = Q6_V_vsplat_R(rstp->scl.result_offset);
	int rows_high = rstp->shp.height;
	int in_next_row = rstp->tin.height_stride;
	int out_next_row = rstp->tout.height_stride;
	int vecs_wide = (rstp->active_width +3)>>2;

	int32_t res_scale = Q6_R_combine_RlRl(rstp->scl.result_scale, rstp->scl.result_scale);
	HVX_Vector vrscale = Q6_V_vsplat_R(res_scale);

	// get the scx values from the table, scale them by res_scale, break out to odd/even
	HVX_Vector scx0 = Q6_Vh_vmpy_VhRh_s1_rnd_sat(sclbias_data[0], res_scale );
	HVX_Vector scx1;
	{
		HVX_VectorPair stmp = Q6_W_vdeal_VVR( scx0,scx0, 64);
		scx0 = Q6_V_lo_W(stmp);
		scx1 = Q6_V_hi_W(stmp);
	}
	// get precomp 'bias' from table; offset by result offset, scale
	HVX_Vector tmp =  Q6_Vw_vadd_VwVw( sclbias_data[1], vresult_offset );
	tmp = Q6_Vw_vmpyo_VwVh_s1_rnd_sat(tmp, vrscale);
	// break into 4 parts
	HVX_VectorPair doff02, doff13;
	{
		HVX_VectorPair stmp = Q6_W_vdeal_VVR( tmp,tmp, 32);
		doff02 = Q6_W_vdeal_VVR( Q6_V_lo_W(stmp),Q6_V_lo_W(stmp), 64);
		doff13 = Q6_W_vdeal_VVR( Q6_V_hi_W(stmp),Q6_V_hi_W(stmp), 64);
	}
	// we find min/max of the input bytes
	// We can start with the 'zero' of the input.
	HVX_Vector all_max = q6op_Vb_vsplat_R( rstp->scl.data_zero);
	HVX_Vector all_min = all_max;
	//

	int res_rsh1 = rstp->scl.result_rsh1;
	int res_rsh2 = rstp->scl.result_rsh2;

	// we need to exclude left/right width padding from each row. We don't need
	// to exclude depth padding, since the precalc scale/bias effectively puts those to zero.
	// Note that if nvecs_wide = 1, the inner loop is run 0 times and both masks are applied
	// on the single vector.

	HVX_VectorPred qleft = Q6_Q_vsetq_R( rstp->left_mask*32);	// mask for 0,32,64, or 96 left cols
	HVX_VectorPred qright = q6op_Q_vsetq2_R( rstp->active_width*32);	// mask for right side


	for( int irow = 0; irow < rows_high; irow++){
		HVX_Vector const *invp = (HVX_Vector const *)(inp + in_next_row * irow);
		HVX_Vector *outvp = (HVX_Vector *)(outp + out_next_row * irow);


		// this loop is explicitly unpeeled to allow the width masking to be done outside.

		HVX_Vector inv = invp[0];
		HVX_Vector all_maxp = all_max;		// "previous"
		HVX_Vector all_minp = all_min;
		all_max = Q6_Vub_vmax_VubVub( inv, all_max);
		all_min = Q6_Vub_vmin_VubVub( inv, all_min);
		all_max = Q6_V_vmux_QVV(qleft,all_maxp, all_max);		// left side gating
		all_min = Q6_V_vmux_QVV(qleft,all_minp, all_min);

		// zero extend to h
		HVX_Vector inv0 = Q6_Vb_vshuffe_VbVb( Q6_V_vzero(), inv);
		HVX_Vector inv1 = Q6_Vb_vshuffo_VbVb( Q6_V_vzero(), inv);

		// invariant: 'all_max' includes columns up to 'j'; all_maxp is the value prior to that.
		for(int j = 0; j < vecs_wide-1; j++){
			// mul even reg by scx0, and odd by scx1; add the applicable bias values
			HVX_VectorPair res_w02 = Q6_Ww_vmpyacc_WwVhVh( doff02, inv0, scx0);
			HVX_VectorPair res_w13 = Q6_Ww_vmpyacc_WwVhVh( doff13, inv1, scx1);
			HVX_Vector result_h02 = Q6_Vh_vasr_VwVwR_sat( Q6_V_hi_W(res_w02), Q6_V_lo_W(res_w02), res_rsh1 );
			HVX_Vector result_h13 = Q6_Vh_vasr_VwVwR_sat( Q6_V_hi_W(res_w13), Q6_V_lo_W(res_w13), res_rsh1 );
			HVX_Vector result = Q6_Vub_vasr_VhVhR_rnd_sat( result_h13, result_h02, res_rsh2);

			inv = invp[j+1];
			*outvp++ = result;

			all_maxp = all_max;		// save prev
			all_minp = all_min;
			all_max = Q6_Vub_vmax_VubVub( inv, all_max);
			all_min = Q6_Vub_vmin_VubVub( inv, all_min);
			// zero extend to h
			inv0 = Q6_Vb_vshuffe_VbVb( Q6_V_vzero(), inv);
			inv1 = Q6_Vb_vshuffo_VbVb( Q6_V_vzero(), inv);
		}
		// right side gating
		all_max = Q6_V_vmux_QVV(qright,all_max, all_maxp);	// keep where q = 1
		all_min = Q6_V_vmux_QVV(qright,all_min, all_minp);	// keep where q = 1
		// finish the row
		HVX_VectorPair res_w02 = Q6_Ww_vmpyacc_WwVhVh( doff02, inv0, scx0);
		HVX_VectorPair res_w13 = Q6_Ww_vmpyacc_WwVhVh( doff13, inv1, scx1);
		HVX_Vector result_h02 = Q6_Vh_vasr_VwVwR_sat( Q6_V_hi_W(res_w02), Q6_V_lo_W(res_w02), res_rsh1 );
		HVX_Vector result_h13 = Q6_Vh_vasr_VwVwR_sat( Q6_V_hi_W(res_w13), Q6_V_lo_W(res_w13), res_rsh1 );
		HVX_Vector result = Q6_Vub_vasr_VhVhR_rnd_sat( result_h13, result_h02, res_rsh2);
		*outvp++ = result;
	}
	// all_min, all_max are min/max input in this depth slice
	HVX_VectorPair mm = Q6_W_vdeal_VVR( all_max, Q6_V_vnot_V(all_min), 64);	// ~min in 1st half, max in 2nd
	all_min = Q6_Vub_vmax_VubVub( Q6_V_hi_W(mm), Q6_V_lo_W(mm));				// ~min, ~min ,max,max
	HVX_Vector inv = Q6_Vub_vmax_VubVub( all_min, Q6_V_vror_VR(all_min, 32 ));	// ~min, XXX, max, xxx
	// flip the ~min back to min
	HVX_Vector halfmask = Q6_V_vand_QR(Q6_Q_vsetq_R(64),-1);
	inv = Q6_V_vxor_VV( inv, halfmask);

	// now run that through the process, up to the 32-bit

	HVX_Vector inv0 = Q6_Vb_vshuffe_VbVb( Q6_V_vzero(), inv);
	HVX_Vector inv1 = Q6_Vb_vshuffo_VbVb( Q6_V_vzero(), inv);
	HVX_VectorPair res_w02 = Q6_Ww_vmpyacc_WwVhVh( doff02, inv0, scx0);
	HVX_VectorPair res_w13 = Q6_Ww_vmpyacc_WwVhVh( doff13, inv1, scx1);

	// 1st quadrant is based on min input, third on max; however, due to -ve scales, the
	// min/max could be reversed within any given lane. So reduce min & max separately across
	// all 4 results...

	HVX_Vector minw = Q6_Vw_vmin_VwVw(
			Q6_Vw_vmin_VwVw(Q6_V_hi_W(res_w02), Q6_V_lo_W(res_w02)),
			Q6_Vw_vmin_VwVw(Q6_V_hi_W(res_w13), Q6_V_lo_W(res_w13)) );

	HVX_Vector maxw = Q6_Vw_vmax_VwVw(
			Q6_Vw_vmax_VwVw(Q6_V_hi_W(res_w02), Q6_V_lo_W(res_w02)),
			Q6_Vw_vmax_VwVw(Q6_V_hi_W(res_w13), Q6_V_lo_W(res_w13)) );

	// tranpose/combine so that ~min is in low half, max in high
	HVX_VectorPair mmtx = Q6_W_vdeal_VVR(  maxw, Q6_V_vnot_V(minw), 64);
	HVX_Vector minmaxw = Q6_Vw_vmax_VwVw( Q6_V_hi_W(mmtx), Q6_V_lo_W(mmtx) );

	// lanes 0 .. 7 are ~min, lanes 16..23 are max
	for( int i =0; i < 3; i++){
		HVX_VectorPair mmdeal = Q6_W_vdeal_VVR( minmaxw, minmaxw,-4);
		minmaxw = Q6_Vw_vmax_VwVw(  Q6_V_lo_W(mmdeal), Q6_V_hi_W(mmdeal));
	}
	// lane 0 is ~min, lane 2 is max
	// combine with previous and return
	return Q6_Vw_vmax_VwVw( minmaxw, prev_minmax);
}


//
// inputs:
//    (0) data input, d32 format, shape (b,h,w,d)
//    (1) scale input,  flat qu8 format, shape (1,1,1,d) or (1,1,1,1)
//    (2)  data_min
//    (3)  data_max
//    (4)  scale_min
//    (5)  scale_mix
//    (6) bias input, flat qu8 format, shape (1,1,1,d) or (1,1,1,1)
//    (7)  bias min
//    (8)  bias max
//    (9) output min		(either specified as a clip level, or -inf)
//    (10) output max       (either specified as a clip level, or inf)
// outputs:
//    (0) output data (b,h,w,d)
//    (1)   output_min
//    (2)   output_max
//
//
static int
batchnorm_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking batchnorm_d32 node %p",self);

	////////////////
	struct batchnorm_d32_info *info;

	if ((info = (struct batchnorm_d32_info *)nn_calloc(1,sizeof(struct batchnorm_d32_info))) == NULL) {
		return errlog(nn,"calloc");
	}
	self->opaque = (void*) info;

	info->out_max =0.5f;
	const struct tensor *out_min_tensor =  self->inputs[9];
	const struct tensor *out_max_tensor =  self->inputs[10];

	int min_max_precalc = 0;
	// if both min,max are speciified, we must leave  (min,max) as a proper range.
	// (this is only an issue when min < 0 )
	//
	{
		float val = tensor_get_float(out_max_tensor,0);
		if( val < INFINITY){
			min_max_precalc |= 2;
			info->out_max = info->out_max_specified = fmaxf(0.0f,val);
		}
	}

	{
		float val = tensor_get_float(out_min_tensor,0);
		if( val > -INFINITY){
			min_max_precalc |= 1;
			val  = fminf(0.0f,val);
			info->out_min = info->out_min_specified = val;
			if( min_max_precalc == 3){	// both ends were set
				info->out_max = fmaxf(val + 1e-6f, info->out_max);
				adjust_minmax_for_zero_with_constraints( &info->out_min, & info->out_max,min_max_precalc);
			}
		}
	}
	info->min_max_precalc = min_max_precalc;
	// if both are preset, no need to guess at range on first run.
	if( min_max_precalc == 3)
		info->run_yet = 1;

	logmsg(nn,2,"@ %p:  min_preset = %d (%f); max_preset = %d (%f)", self,
			min_max_precalc&1, info->out_min,
			(min_max_precalc>>1)&1, info->out_max );
	logmsg(nn,2,"batchnorm_d32 node %p check OK",self);
	return 0;
}

static int batchnorm_d32_dtor(struct nn_node *self, struct nn_graph *nn)
{
	if (self->opaque) nn_free(self->opaque);
	self->opaque = NULL;
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_QuantizedBatchNorm_8x8p8to8_d32 = {
	.execute = batchnorm_d32_execute,
	.check = batchnorm_d32_check,
	.ctor = node_alloc_common,
	.dtor = batchnorm_d32_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

//
// version with 32-bit bias
//
struct nn_node_ops nn_ops_for_QuantizedBatchNorm_8x8p32to8_d32 = {
	.execute = batchnorm_d32_execute,
	.check = batchnorm_d32_check,
	.ctor = node_alloc_common,
	.dtor = batchnorm_d32_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};


// These are dummy ops which act as place-holders during graph prep.

static int batchnorm_8_execute(struct nn_node *self, struct nn_graph *nn)
{
	return errlog(nn,"batchnorm_8 not implemented");
}

struct nn_node_ops nn_ops_for_QuantizedBatchNorm_8x8p8to8 = {
	.execute = batchnorm_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};

//
// version with 32-bit bias
//
struct nn_node_ops nn_ops_for_QuantizedBatchNorm_8x8p32to8 = {
	.execute = batchnorm_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};

