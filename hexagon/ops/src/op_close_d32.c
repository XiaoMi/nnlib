
/*
 * Copyright (c) 2017, The Linux Foundation. All rights reserved.
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

/*
 * This operator checks to see if a d32 tensor (with min and max)
 * is 'close enough' to a supplied reference (float) tensor.
 * it is intended to be used in situations like this, where a single
 * quantized op is being tested; and the reference op is given the same
 * input (dequantized to match the quantized input), so  the results
 * can be expected to be very close to the reference result:
 *
 *   (Here ||| and == represent qu8_d32,min,max):
 *
 *
 *     (test constant)                        (test_constant)
 *           |                                     |
 *           |                                     |
 *      [QuantizeForTest_d32]                  [QuantizeForTest_d32]
 *        |||           |                        |||            |
 *        |||           |                        |||            |
 *        |||           |        +---------------|||------------+
 *        |||           |        |               |||
 *        |||        (reference op )             |||
 *        |||             |                      |||
 *        |||             +---------+            |||
 *        |||                       |            |||
 *        |||     +++============================+++
 *        |||     |||               |
 *       ( op. being tested)        |
 *        |||                       |
 *        |||     +-----------------+
 *        |||     |
 *       (Close_d32)
 */

#define MAX_TO_SHOW 100

////////////////////////////////////////////////////////
//
// stats on the error.
// All stats are measured in the quantized domain.

struct err_stats {
	int ncount;					// # of points
	float sum_delta;			// sum of errors
	float sum2_delta;			// sum of error-squared (for rms)
	int ncount_excerr;			// count of nonzero excess error
	float max_excerr;			// largest 'excess error'
	int pos_largerr;			// position of largest error

	// stats of the ref vs. test, for correlation
	// points where test = 0 or 255 are not included on the grounds
	// that they could be affected by saturation.
	// we want to correlate y = test result vs x= ref, but
	// since we expect them to be about the same, it's better (from a numerical
	// precision standpoint) to record stats for s = test+ref, d=test-ref, and
	// sort it out after.
	int ncount_notsat;
	float regr_sum_s;		// sum of test+ref
	float regr_sum_d;		// sum of test-ref
	float regr_sum_s2;		// sum of (test+ref)^2
	float regr_sum_d2;		// sum of (test-ref)^2
	float regr_sum_sd;		// sum of (test+ref)*((test-ref)
};

// clear the stats
//
static inline void errstats_clear( struct err_stats * esp )
{
	esp->ncount = 0;
	esp->sum_delta = 0.0f;
	esp->sum2_delta = 0.0f;
	esp->ncount_excerr = 0;
	esp->max_excerr = 0.0f;
	esp->pos_largerr = -1;

	esp->ncount_notsat = 0;
	esp->regr_sum_s = 0.0f;
	esp->regr_sum_d = 0.0f;
	esp->regr_sum_s2 = 0.0f;
	esp->regr_sum_d2 = 0.0f;
	esp->regr_sum_sd = 0.0f;
}

// accumulate 'sto' from 'sfrom' and then clear sfrom
//
static inline void errstats_dump_and_clear( struct err_stats * sto, struct err_stats *sfrom )
{
	sto->ncount += sfrom->ncount;
	sto->sum_delta += sfrom->sum_delta;
	sto->sum2_delta += sfrom->sum2_delta;
	sto->ncount_excerr += sfrom->ncount_excerr;
	if( sfrom->max_excerr > sto->max_excerr ){
		sto->max_excerr = sfrom->max_excerr;
		sto->pos_largerr = sfrom->pos_largerr;
	}

	sto->ncount_notsat += sfrom->ncount_notsat;
	sto->regr_sum_s += sfrom->regr_sum_s;
	sto->regr_sum_d += sfrom->regr_sum_d;
	sto->regr_sum_s2 += sfrom->regr_sum_s2;
	sto->regr_sum_d2 += sfrom->regr_sum_d2;
	sto->regr_sum_sd += sfrom->regr_sum_sd;
	errstats_clear(sfrom);
}

//
// add a point.
// 'ixtext' is the 'test' value, 0..255
// 'xref' is the reference value, converted to the same units
// 'pos' is the index in the data (for recording position of first error)
//
static inline void errstats_add_point( struct err_stats * esp, int ixtest, float xref, int pos )
{
	float xtest = ixtest;
	float xrefi = roundf(xref);
	float delt = xtest - xref;		// the error...
	// 'excess' error: how much the error is in excess of the error
	// in reference, when the reference is rounded to integer.
	//
	float excess_error = fabsf(delt) - fabsf(xrefi-xref);

	esp->ncount ++;
	esp->sum_delta += delt;
	esp->sum2_delta += delt*delt;
	if( excess_error > 0.0f ){
		esp->ncount_excerr ++;
		if( excess_error > esp->max_excerr ){
			esp->max_excerr = excess_error;
			esp->pos_largerr = pos;
		}
	}
	// the regression is not done on codes 0 or 255, on the assumption that they
	// might be distorted by saturation
	if( ixtest >= 1 && ixtest < 255 ){
		float sm = xtest+xref;
		esp->ncount_notsat ++;
		esp->regr_sum_s += sm;
		esp->regr_sum_d += delt;
		esp->regr_sum_s2 += sm*sm;
		esp->regr_sum_d2 += delt*delt;
		esp->regr_sum_sd += sm*delt;
	}
}
//
// find mean and root-mean-square of the delta.
//
static inline void errstats_find_mean_rms(  struct err_stats const * esp, float * mean_out, float * rms_out )
{
	float pop = esp->ncount;
	*mean_out = esp->sum_delta / pop;

	*rms_out = sqrtf( (pop * esp->sum2_delta - esp->sum_delta * esp->sum_delta) ) /pop;

}
//
// find correlation between x = ref and y = tst.
//

static int errstats_find_correlation( struct nn_graph *nn, int level, struct err_stats const * esp)
{
	if( esp-> ncount_notsat < 6 ) return -1;

	float pop = esp->ncount_notsat;

	float mean_s = esp->regr_sum_s/pop;
	float mean_d = esp->regr_sum_d/pop;

	// this is sum{ ( s - s_mean)^2} /pop
	// and  sum{ ( d - d_mean)^2} /pop
	// and sum{  (s-s_mean)*(d-d_mean)) /pop
	float var_s2 = (pop * esp->regr_sum_s2 - esp->regr_sum_s * esp->regr_sum_s )/(pop*pop);
	float var_d2 = (pop * esp->regr_sum_d2 - esp->regr_sum_d * esp->regr_sum_d )/(pop*pop);
	float var_sd = (pop * esp->regr_sum_sd - esp->regr_sum_s * esp->regr_sum_d )/(pop*pop);

	float evx=1.0f, evy=0.0f;
	float eig0 = var_s2, eig1= 0.0f;
	// these values form a matrix
	//  [ var_s2   var_sd ]
	//  [ var_sd   var_d2 ]
	// .. find its eigenvalues and eigenvevc
	if( var_d2 >  0.0f) {	// it could be 0...
		float mtrace = var_s2 + var_d2;
		float d = 0.5f*(var_s2 - var_d2);
		float eghdiff = hypotf(d, var_sd );
		float r = fabsf(d) + eghdiff;
		float dd = hypotf( r, var_sd );
		if( dd> 0.0f){
			// normally the two expressions below would be divided by dd to get a unit
			// vector; we don't need that here
			evx = r; /* divide /dd; to get unit vector */
			evy = var_sd;  /* divide /dd to get unit vector */
			if( d < 0){
				float xt = fabsf(evy);
				evy = copysignf(evx,evy);
				evx = xt;
			}
		}
		eig0 = 0.5f * mtrace + eghdiff;	// the big one
		eig1 = fmaxf(0.0f, 0.5f * mtrace - eghdiff);	// the little one
	}
	// (evx, evy) is the direction of largest correlation (x+y) vs (y-x);
	// we expect it to be about (1,0); we can correct this to a 'scale)
	// by rotating 45 degrees...
	// first ensure it makes sense.
	//
	if( evx <= 0.0 || fabsf(evy) > 0.5f * evx || eig0 < 16.0f || eig1 > 0.1f*eig0){
		float dd = hypot( evx, evy );
		logmsg(nn, level, "correlation of %d points: results unclear, eigs(%.2f,%.2f), vec = (%.6f, %.6f)",
				esp-> ncount_notsat, eig0,eig1, evx/dd, evy/dd);
		return 1;
	}

	// find a fit: test = ref * scale + offs

	float scale = (evx + evy)/(evx-evy);		// ref->scale gain
	// the fit passes through (mean_x, mean_y)
	float meanx = (mean_s - mean_d)*0.5f;
	float meany = (mean_s + mean_d)*0.5f;
	//
	float offs = meany - scale * meanx;

	logmsg(nn,level, "correlation of %d points: sdev = %.4f across, and %.1f along axis gain = %.6f",
			esp-> ncount_notsat, sqrtf(eig1), sqrtf(eig0), scale );
	logmsg(nn,level, " offsets:  0->%.4f,  128-> %.4f, 255->%.4f", offs, offs + 128.0f *scale, offs + 255.0f*scale );


	return 0;
}


////////////////////////////////////////////////////////


static int close_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
#ifdef TIMING_MODE
	return 0;
#endif
	int i;
	const struct tensor *tensor_in = self->inputs[0];
	const struct tensor *tensor_in_min = self->inputs[1];
	const struct tensor *tensor_in_max = self->inputs[2];
	const struct tensor *tensor_ref = self->inputs[3];
	logmsg(nn,2,"close_d32 execute. self=%p ",self);


	float max_exc_err = 0.2f;		// the maximum excess error allowed
	float max_exc_err_frac = 0.05;	// max fraction of points with nonzero excess error.

	if( self->n_inputs >= 5 ){	// optional parms?
		if( self->inputs[4] != NULL){
			max_exc_err = tensor_get_float(self->inputs[4],0);
		}
		if( self->n_inputs >= 6 &&  self->inputs[5] != NULL ){
			max_exc_err_frac = tensor_get_float(self->inputs[5],0);
		}
	}

	for(i = 0; i < 4; i++){
		if(tensor_in->shape.dimension[i] != tensor_ref->shape.dimension[i] ){
			return errlog(nn,
				"close_d32: shape %d:%d:%d:%d does not match reference %d:%d:%d:%d",
				  tensor_in->shape.batches, tensor_in->shape.height,
				  tensor_in->shape.width, tensor_in->shape.depth,
				  tensor_ref->shape.batches, tensor_ref->shape.height,
				  tensor_ref->shape.width, tensor_ref->shape.depth);

		}
	}
	float dut_min_float = tensor_get_float(tensor_in_min,0);
	float dut_max_float = tensor_get_float(tensor_in_max,0);

	int ib, ih,iw,id;
	int batches = tensor_in->shape.batches;
	int height = tensor_in->shape.height;
	int width = tensor_in->shape.width;
	int depth = tensor_in->shape.depth;

	// check range is sane...

	float qscale = 255.0f/(dut_max_float-dut_min_float);
	// check constraints (also catch NaN and inf)
	if ( !( (dut_max_float >= 0.0f)		 // catches max <0  and NaN
	      && (dut_min_float <= 0.0f)	// catches min > 0 and NaN
	      && qscale > 0.0f 				// to catch inf
	      && qscale <= 255e+6f ) ) {	// range must be >= 1e-6
		return errlog(nn,"invalid input range: %f .. %f", dut_min_float, dut_max_float);
	}
	//
	// check if 'zero' value is clean. We've already established it's in range 0...255 by the
	// above constraints.
	//
	float inzero = -qscale * dut_min_float;
	float inzero_round = roundf(inzero);
	if( fabsf(inzero-inzero_round)>1e-3f ){
		logmsg(nn,0,"**** close_d32: input range %f .. %f  has zero at %.8f ****", dut_min_float, dut_max_float, inzero);
	}
	/// ---------------------
	/// measure the stats
	// Use 3 levels to reduce precision loss in the accumulation
	struct err_stats errstatA, errstatB, errstatC;
	errstats_clear( & errstatA );
	errstats_clear( & errstatB );
	errstats_clear( & errstatC );
	float const * refp = (float const *)tensor_ref->data;

	for( ib = 0; ib < batches; ib++ ){
		for (ih = 0; ih < height; ih++ ){
			int ix_bh = (ib*height + ih) * (width*depth);
			for( iw = 0; iw < width; iw++ ){
				for( id = 0; id < depth ; id ++ ){
					int idx = ix_bh + iw*depth + id;

					uint8_t tdata = *tensor_location_d32( tensor_in, ib, ih, iw, id );
					float refdata = (refp[idx] - dut_min_float)*qscale;
					errstats_add_point( &errstatA, tdata, refdata, idx );
					if( errstatA.ncount >= 100){
						errstats_dump_and_clear( &errstatB, &errstatA );
						if( errstatB.ncount >= 100 )
							errstats_dump_and_clear( &errstatC, &errstatB );
					}
				}
			}
		}
	}
	// collect all the errors to errstatC.

	errstats_dump_and_clear( &errstatB, &errstatA );
	errstats_dump_and_clear( &errstatC, &errstatB );

	float mean_error,rms_error;
	errstats_find_mean_rms( &errstatC, &mean_error,&rms_error);

	int force_report = max_exc_err < 0.0f;
	max_exc_err = fabsf(max_exc_err);

	int acceptable =
			( errstatC.max_excerr <= max_exc_err )
		&& ( errstatC.ncount_excerr  <=  max_exc_err_frac * errstatC.ncount );

	int loglev = (acceptable && !force_report)? 2: 0;


	logmsg(nn,loglev,"Out of %d points: mean err = %.3f, rms err = %.3f; largest excess err = %.3f (%d are nonzero)",
			errstatC.ncount, mean_error, rms_error, errstatC.max_excerr, errstatC.ncount_excerr );
	if( errstatC.ncount_notsat >= 16){
		errstats_find_correlation( nn , loglev, &errstatC);
	}
	if( ! acceptable ){
		int count = batches * height * width * depth;
		int ipos;
		logmsg(nn,0,"Exceeds error limits. Values below are in quantized units");
		logmsg(nn,0,"\t\tActual\t\tExpected\tDiff");
		int nlogged = 0;
		// only log points with excess err > 0; only log up to MAX_TO_SHOW
		// (but always log the worst one)

		for (ipos = 0; ipos < count; ipos++) {
			int b,h,w,d,k;
			d  = ipos % depth; k = ipos/ depth;
			w = k % width;	k= k/width;
			h = k % height;
			b = k / height;
			// find excess error
			uint8_t tdata = *tensor_location_d32( tensor_in, b, h, w, d );
			float refdat = refp[ipos];
			float refdatq = (refdat - dut_min_float)*qscale;
			float excerr = fabs( tdata-refdatq) - fabs( roundf(refdatq)-refdatq);
			if( excerr > 0.0f){
				char const * flag = ( ipos == errstatC.pos_largerr)? " <====": "";

				if( nlogged < MAX_TO_SHOW || ipos == errstatC.pos_largerr){
					logmsg(nn,0,"%d[%d,%d,%d,%d])\t%d\t%f\t%f%s",ipos,b,h,w,d, tdata,refdatq,tdata-refdatq, flag);
					nlogged ++;
				}else if( nlogged < MAX_TO_SHOW+10){
					logmsg(nn,0, "[Stopped after %d errors]", nlogged);
					nlogged = MAX_TO_SHOW +10;
				}
				if( ipos >= errstatC.pos_largerr && nlogged >= MAX_TO_SHOW+10) break;
			}
		}
		return errlog(nn, "test failed");
	}

	logmsg(nn,2,"close_d32 node %p OK",self);
	return 0;
}


//
//  input0:   test tensor, d32 quantized
//  input1:   float scalar: min of range
//  input2:   float scalar: max of range
//  input3:   reference tensor, float
// OPTIONAL (may be missing or null):
//   input4:  max acceptable 'excess error' - default 0.2
//   input5:  max accetable frac of outputs which have nonzero excess error; default = 0.05
//
// If the max acceptable error is negative, its abs value will be used, and stats will
// be reported even if the test passes.
//
//  No outputs

static int close_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking close_d32 node %p",self);
	int k = node_check_inputs_range( self, nn, "close_d32", 4, -6);
	if( k == 0 )k = node_check_outputs_n( self, nn, "close_d32", 0);
	if( k!=0)
		return k;
	logmsg(nn,2,"close_d32 node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_Close_d32 = {
	.execute = close_d32_execute,
	.check = close_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};
