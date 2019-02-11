/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
#include "errstats.h"

// clear the stats
//
void nn_errstats_clear( struct err_stats * esp )
{
	errstats_clear_inline( esp );
}

// accumulate 'sto' from 'sfrom' and then clear sfrom
//
void
nn_errstats_dump_and_clear( struct err_stats * sto, struct err_stats *sfrom )
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
	errstats_clear_inline(sfrom);
}

//
// add a point.
// 'ixtext' is the 'test' value, 0..255 (or -32768..32767)
// 'xref' is the reference value, converted to the same units
// 'pos' is the index in the data (for recording position of first error)
// 'isclip' is true if the value is considered to be at the limit of the
//  range (it will be excluded from the regression).
// 
void nn_errstats_add_point( struct err_stats * esp, 
	int ixtest, float xref, int pos, 
	int isclip)
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
	// the regression is not done on at the min or max, on the grounds that they
	// might be distorted by saturation
	if(!isclip){
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
void 
nn_errstats_find_mean_rms(  struct err_stats const * esp, float * mean_out, float * rms_out )
{
	float pop = esp->ncount;
	*mean_out = esp->sum_delta / pop;

	*rms_out = sqrtf( (pop * esp->sum2_delta - esp->sum_delta * esp->sum_delta) ) /pop;

}
//
// find correlation between x = ref and y = tst.
// Report at three different 'x' values.
//
int nn_errstats_find_correlation(
 struct nn_graph *nn, int level, struct err_stats const * esp,
 int xa, int xb, int xc)
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
		float dd = hypotf( evx, evy );
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
	logmsg(nn,level, " offsets:  %d->%.4f,  %d-> %.4f, %d->%.4f",
		xa, offs + (float)xa*scale,
		xb, offs + (float)xb*scale,
		xc, offs + (float)xc*scale);

	return 0;
}
