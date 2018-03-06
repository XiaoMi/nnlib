
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
#include <quantize.h>
//
// This adjusts a min .. max range
// (by decreasing min, or increasing max) until
// the 'zero point' (the value which encodes zero)
// is an exact integer.
//  if 0 encodes 'mn', and 255 encodes 'mx', then zero is encoded by
//     z = -255*mn/(mx-mn)
//
//
// If this is not an integer, we want to either adjust mx up (to mx+b)
//  or mn down (to mn-a), in order to make z an integer, using
//  the smallest possible adjustment.
//
// (we require mn < mx, and discussion below assumes mn <= 0, mx > 0).
//
// Sensitivity of z to adjustments a,b:
//   dz/da = -dz/d(mn) = 255*mx/(mx-mn)^2        (this is > 0)
//   dz/db =  dz/d(mx) = 255*mn/(mx-mn)^2		 (this is <=0)
// (note that these are in the ratio mx:mn)
//
// if z = zi + zf (integer and fraction), we have the choice
//   of (a) increasing it by (1-zf), using a ~=~ (1-zf)/(dz/da)
//   or (b) decreasing it by zf,  by b ~=~ -zf / (dz/db)
//
//  to pick the one which chooses the smallest move:
//     - choose (b) if the mag of (1-zf)/mx  is more than mag of -zf/mn;   else (a)
//  ie - choose (b) if f (zf-1)*mn is more  than zf*mx;   else (a).
//
// in cases where zi <= 0, or zi >=254, we choose (a) or (b) respectively, thus
// avoiding division by 0 when calculating the new endpoint;
// this also means that cases (0 < mn < mx) and (mn < mx < 0)
// are handled properly (in these cases, zi will be <=0 or >=255 resp, and
// the adjustment is always made to the endpoint which is closest to 0).
//
//
// return value:
//  -1 : test failed for max > min (includes either being NaN)
//   0 : ok, no adjustment made; zero point already an integer +/- 2^-14
//   1 : adjusted min downward
//   2 : adjusted max upward
//
// note: if you want min <=0 and max >=0, enforce that before calling
// (and if you decrease min to 0.0, you don't need to call this since
// the condition is met by that).
//

int adjust_minmax_for_zero( float *min_p, float *max_p )
{
	float mn = *min_p;
	float mx = *max_p;
	float dif = mx-mn;
	if( !(dif >= 1e-6f)) return -1;	// check valid min,max_p
	if( mn == 0.0f) return 0;		// common case
	float z = (-255.0f)*mn / dif;		// current 'zero point'
	float zi = floorf(z);
	float zf = z - zi;
	// if within 2^-14 of an integer, call it close enough
	if( zf <= 6.1035156e-05f || zf >= 0.999938965f)
		return 0;
	// choose which end to move
	// if zi <= 0  or >= 254, the decision is based on that only (to
	// avoid divide by 0) otherwise choose based on zf.
	//
	if( zi > 0.0f && ( zi > 253.0f || (zf-1.0f)*mn >= zf*mx )) {
		// move max, change z to zi
		*max_p = mn - 255.0f*mn/zi;
		return 2;
	}else{
		// move min; change z to zi+1
		*min_p = mx*(zi+1.0f)/(zi-254.0f);
		return 1;
	}
}

//
// This is like adjust_minmax_for_zero except that it accepts a 'constraint'
// parameter which indicates which endpoints are (nominally) fixed:
//
//  constraint =
//       0          same as adjust_minmax_for_zero
//       1          'min' is fixed, max is free
//       2          'max' is fixed; min is free
//       3          both are fixed.
//
// If only one endpoint is fixed:
//     (1) decide which endpoint is to be moved - as per adjust_minmax_for_zero
//         but skew the decision towards moving the 'free' endpoint outward (increasing
//         the range).  So, this is what will usually happen 
//      (2)If the decision nonetheless falls to move the 'fixed' endpoint,
//         it will be moved, but it will be moved as little as possible (moving inward
//         if needed). 
// If both endpoints are 'fixed': We adjust both endpoints by the same amount;
// unless the range is skewed by 3:1 or more
// in which case we adjust the endpoint which is closest to 0.
//
// An example of a case where the skewed decision still goes to the 'fixed' end:
//   fixed min:
//    min = -1.3    max = 10.6    => z = 27.85714
// In this case, it will adjust min to -1.3074 (for z = 28)
//  rather than changing max to 10.9778 (to get z = 29)
//

int adjust_minmax_for_zero_with_constraints( float *min_p, float *max_p , int constraint )
{
	float mn = *min_p;
	float mx = *max_p;
	float dif = mx-mn;
	if( !(dif >= 1e-6f)) return -1;	// check valid min,max_p
	if( mn == 0.0f) return 0;		// common case
	float z = (-255.0f)*mn / dif;		// current 'zero point'
	float zi = floorf(z);
	float zf = z - zi;
	// if within 2^-14 of an integer, call it close enough
	if( zf <= 6.1035156e-05f || zf >= 0.999938965f)
		return 0;
	// choose which end to move
	// Avoid divide by 0.
	//
	
	float mnk = (zf-1.0f) * mn;
	float mxk =  zf *mx;
	float zirnd = (zf >=0.5f)? (zi+1.0f): zi;
	
	switch( constraint &3 ){
	 default:
	 case 0:
		// move whichever requires least change.
		if( zi >= 1.0f && ( zi >= 254.0f || mnk >= mxk ))
			goto move_max_out;
		goto move_min_out;
	 case 1:		// min is fixed; max is free
		// skew decision to 'move max out'
		if( zi >= 1.0f && (z >= 192.25f ||  mnk * 8.0f > mxk) ) 
			goto move_max_out;
		goto move_min_nearest;
	 case 2:
		// skew decision to 'move min out'
		if( z >= 63.75f && ( zi >= 254.0f || mnk > mxk * 8.0f )) 
			goto move_max_nearest;
		goto move_min_out;
	 case 3:
		// move single endpoint if range is skewed at least 3:1; otherwise both.
		if( z <= 63.75f ) goto move_min_nearest;
		if( z >= 192.25f) goto move_max_nearest; 
		// shift zero to zirnd, by moving both ends.
		float adj = (z-zirnd)*dif * (float)(1./255.);
		*min_p = mn + adj;
		*max_p = mx + adj;
		return 3;
	}
	/* NOTREACHED*/
	
   move_max_nearest:   // move max, change z to nearest
     zi = zirnd;
   move_max_out:  		// move max, change z to zi
	*max_p = mn - 255.0f*mn/zi;
	return 2;

   move_min_out:   // move min; change z to zi+1
	 zirnd = zi+1.0f;
   move_min_nearest:  	// move min, change z to nearest
	*min_p = mx*zirnd/(zirnd-255.0f);
	return 1;
}
