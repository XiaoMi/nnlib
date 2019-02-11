#ifndef ERRSTATS_H_
#define ERRSTATS_H_ 1
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
#include <nn_graph.h>
#include <string.h>
#include <math.h>
//
// stats on the error - for 'check' nodes.
// All stats are measured in the quantized domain.

struct err_stats {
	int ncount;					// # of points
	float sum_delta;			// sum of errors
	float sum2_delta;			// sum of error-squared (for rms)
	int ncount_excerr;			// count of nonzero excess error
	float max_excerr;			// largest 'excess error'
	int pos_largerr;			// position of largest error

	// stats of the ref vs. test, for correlation
	// points where 'test' is at min or max are not included on the grounds
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
void nn_errstats_clear( struct err_stats * esp );

static inline void errstats_clear( struct err_stats * esp )
{
	nn_errstats_clear( esp );
}

static inline void errstats_clear_inline( struct err_stats * esp )
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
void nn_errstats_dump_and_clear( struct err_stats * sto, struct err_stats *sfrom );

static inline void errstats_dump_and_clear( struct err_stats * sto, struct err_stats *sfrom )
{
	nn_errstats_dump_and_clear( sto, sfrom );
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
	int isclip);
	

static inline void errstats_add_point( struct err_stats * esp, 
	int ixtest, float xref, int pos, 
	int isclip)
{
	return nn_errstats_add_point( esp, ixtest, xref, pos, isclip );
}
//
// find mean and root-mean-square of the delta.
//
void nn_errstats_find_mean_rms(  struct err_stats const * esp, float * mean_out, float * rms_out );

static inline void errstats_find_mean_rms(  struct err_stats const * esp, float * mean_out, float * rms_out )
{
	return nn_errstats_find_mean_rms( esp, mean_out, rms_out );
}
//
// find correlation between x = ref and y = tst.
// Report at three different 'x' values.
int nn_errstats_find_correlation(
 struct nn_graph *nn, int level, struct err_stats const * esp,
 int xa, int xb, int xc);


static inline int errstats_find_correlation(
  struct nn_graph *nn, int level, struct err_stats const * esp,
  int xa, int xb, int xc)
{
	 return nn_errstats_find_correlation(nn,level,esp,xa,xb,xc);
}


#endif // ERRSTATS_H_