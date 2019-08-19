/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
#include <stdlib.h>
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_inlines.h"
#include "hvx_mathops.h"
#include "hvx_funnel_reduce.h"


/* HeatmapMaxKP_f operator */

// input 0:  [ N, hwh, hmw, n_heatmap ] of floats - heatmaps   (hmh>=2, hmw>=2 )
// input 1:  [ 1, 1, N, 4] of floats  (bounding boxes).
// input 2:  scalar int is_NCHW
//
// output 0:  [1,1,N,n_heatmap]  of floats (peak vals)
// output 1:  [1,N,n_heatmap,2]  of floats (peak posns)
//
// input 1 can also be [N,1,1,4] and then outputs are [N,1,1,n_heatmap] and [N,1,h_heatmap,2]
//
/* QuantizedHeatmapMaxKP_8 operator */

// input 0:  [ N, hwh, hmw, n_heatmap ] of qu8 - heatmaps   (hmh>=2, hmw>=2 )
// input 1:  min for heatmaps
// input 2:  max for heatmaps
// input 3:  [ 1, 1, N, 4] of quint16  (bounding boxes).
// input 4:  scalar int is_NCHW
//
// output 0:  [1,1,N,n_heatmap]  of qu8 (peak vals)
// output 1:  min for peak vals
// output 2:  max for peak vals (may be > input max)
// output 3:  [1,N,n_heatmap,2]  quint16 (peak posns) (same quant as input 3)
//
// input 3 can also be [N,1,1,4] and then outputs 0,3 are [N,1,1,n_heatmap] and [N,1,h_heatmap,2]
//

// ======
// Some adjustable parameters for the peak locator
// These should match those used in the test generator
//
// for floating point, the determinant of the Hessian is OK if
// it's >= DETER_TOL_ABS + DETER_TOL_REL * hxx*hxy
// (and if hxx*hxy>=0, which is basically guaranteed by the peak criterion).
//
// So DETER_TOL_ABS is in signal-units-squared, and DETER_TOL_REL
// is unitless.
#define DETER_TOL_ABS 1e-5f
#define DETER_TOL_REL 1e-5f

// This is the max correction in X or Y direction, in x/y units
// This needs to be >=0.25 and < 2.0f
#define MAX_XY_CORR 1.5f
//
// This is the determinant tolerance for fixed-point.
// abs tolerance is in 'quantized-unit-squared', with 12 fractional bits.
#define DETER_TOL_FXP_ABS  20      // about 5e-3
// relative tolerance is expressed with 15 fractional bits
#define DETER_TOL_FXP_REL  33     // about 1e-3

//
//  the peak offset  vector 'x' is estimated by solving the matrix equation
//      A * x = G
// where G is the gradient estimate and A = -Hessian
// The 'extrapolated' peak function value is then found with Taylor series
//
//  fpeak = f0  +  x^t * G   +  (1/2) x^t * H * x
//
//  .. where f0 is the function value at the (sampled) peak value.
// However, if A*x=G, the above simplifies to just using half of the linear term:
//
//	fpeak = f0 + (1/2)x^t * G
//
// and the original result is only different from this if the 'x' result has been clipped
// according to MAX_XY_CORR.
// USE_SIMPLER_PEAK_EST is set to 1 to use the simpler form regardless of whether
// x has been clipped (it leads to slightly lower fpeak in the clipped case, but
// clipping is unusual)
//
#define USE_SIMPLER_PEAK_EST 0	// must be 0 or 1


struct heatmap_fltpeak {
	union{
		int index;			// float version stores index
		struct {
			uint16_t index_x, index_y;	// u8 version stores x,y
		};
	};
};

struct solve_peak_results {
	float xoff, yoff;
	float pkest;
};

struct heatmap_maxpk_info {
	int strategy_valid;				// valid if this is 1 and the input shapes & pointers match.
	struct shape in_hm_shape;		// previous heatmap input shape
	struct shape in_bb_shape;		// previous bbox input shape
	struct shape peaks_shape;		// shape of the 'peaks' output
	struct shape xyout_shape;		// shape of the 'xy' output
	void const *box_ptr;			// pointer to the rectangle defs
	void const * hm_input;			// pointer to the heatmap inputs

	int hm_ht, hm_wid;			// size of heatmap (each >= 2)
	int hm_area;				// = hnm_ht * hm_wid
	int batches;				// number of batches  (>=1)
	int heatmaps;				// number of heatmaps per batch (>=1)

	uint32_t wid_divide_fac;	// used to to do integer-divide by 'hm_wid'
	int wid_divide_shift;

	int hm_batch_stride;		// batch stride of input, in bytes
	int hm_data_stride;			// = elsize (if NCHW) or elsize*heatmaps (if NHWC)
	int hm_hmap_stride;			// = elsize*hm_area (if NCHW) or elsize (if NHWC)

	void * out_peak_ptr;	// output to peaks array
	void * out_xy_ptr;		// output to xy array

	float in_min, in_max;	// input ramge (quantized variant only)
	float in_qstep;			// input quant step (quantized variant only)
	struct heatmap_fltpeak *fltpk_arr;	// area in scratch [ heatmaps]
	uint8_t *winbuf_ptr;	// for 3x3 window samples (quantized variant only - vec aligned)
							// each group of 3 vectors is 32 windows.
	uint16_t *pk_store;		// scratch array to accumulate all the peaks, so we can range them
							// (quantized variant only)
};


struct heatmap_maxpk_runstate {
	struct heatmap_maxpk_info * info;

	int current_batch_ind;

	volatile int curr_heatmap;	// next group of 128 to do (always a multiple of 128)

	volatile int32_t peak_min, peak_max;

	nn_sem_t done_sem;
};


static int
check_heatmap_maxpk_strategy( struct nn_graph *nn, struct nn_node *self, int is_qu8)
{
	struct heatmap_maxpk_info * info = (struct heatmap_maxpk_info*)self->opaque;

	struct tensor const *heatmap_tensor= self->inputs[0];
	struct tensor const *bbox_tensor = self->inputs[ is_qu8? 3:1];
	struct tensor const *is_NCHW_tensor = self->inputs[is_qu8? 4:2];
	struct tensor *out_peak_tensor = self->outputs[0];
	struct tensor *out_xy_tensor = self->outputs[is_qu8?3:1];

	int is_NCHW = tensor_get_int32(is_NCHW_tensor,0 );

	info->hm_input = heatmap_tensor->data;
	info->box_ptr = bbox_tensor->data;

	if( info->strategy_valid
			&& shape_matches( &heatmap_tensor->shape, &info->in_hm_shape )
			&& shape_matches( &bbox_tensor->shape, &info->in_bb_shape )){
		return 0;		// already done
	}
	info->strategy_valid = 0;
	info->in_hm_shape = heatmap_tensor->shape;
	info->in_bb_shape = bbox_tensor->shape;

	int batches = info->in_hm_shape.batches;
	int hm_ht = info->in_hm_shape.height;
	int hm_wid = info->in_hm_shape.width;
	int heatmaps = info->in_hm_shape.depth;
	if( is_NCHW){
		hm_ht = hm_wid;
		hm_wid = heatmaps;
		heatmaps = info->in_hm_shape.height;
	}
	// since we store these indices in 16 bits...
	if( is_qu8 && max_i32( hm_ht, hm_wid)>65536) {
		return errlog(nn,"heatmap size may not exceed 65536");
	}

	if( hm_ht <2 || hm_wid < 2) return errlog(nn,"heatmap must be at least 2x2");
	int hm_area = hm_wid * hm_ht;

	info->batches = batches;
	info->hm_ht= hm_ht;
	info->hm_wid= hm_wid;
	info->hm_area = hm_area;
	info->heatmaps = heatmaps;

	info->peaks_shape.batches = 1;
	info->peaks_shape.height = 1;
	info->peaks_shape.width = batches;
	info->peaks_shape.depth = heatmaps;

	info->xyout_shape.batches = 1;
	info->xyout_shape.height = batches;
	info->xyout_shape.width = heatmaps;
	info->xyout_shape.depth = 2;

	{
		// check the bbox shape
		// must be [1,1,batches,4] or [batches,1,1,4]
		int k1 = info->in_bb_shape.batches;
		int k2 = info->in_bb_shape.width;
		if( info->in_bb_shape.depth !=4 || info->in_bb_shape.height != 1
			|| min_i32( k1,k2)!=1 || max_i32(k1,k2)!= batches) {
			return errlog(nn,"bbox input shape mismatch");
		}
		if( k1 > 1){	// N,1,1,4 shape: use [batches,1,2,heatmaps] shape.
			info->peaks_shape.batches = batches;
			info->peaks_shape.width = 1;
			info->xyout_shape.batches = batches;
			info->xyout_shape.height = 1;
		}
	}

	// set strides for reading heatmap
	{
		int elbytes = is_qu8? 1: sizeof(float);
		info->hm_batch_stride = elbytes * hm_area * heatmaps;
		if( is_NCHW){
			info->hm_data_stride = elbytes;
			info->hm_hmap_stride = elbytes * hm_area;
		}else{
			info->hm_data_stride = elbytes * heatmaps;
			info->hm_hmap_stride = elbytes;
		}
	}

	if( nn_tensor_out_prepare_normal_fromshape( out_peak_tensor, &info->peaks_shape, is_qu8? NN_TYPE_QUINT8:NN_TYPE_FLOAT)!=0
	  ||  nn_tensor_out_prepare_normal_fromshape( out_xy_tensor, &info->xyout_shape, is_qu8? NN_TYPE_QUINT16:NN_TYPE_FLOAT)!=0){
		return errlog(nn,"output too small");
	}
	info->out_peak_ptr = out_peak_tensor->data;
	info->out_xy_ptr = out_xy_tensor->data;

	nn_scratch_reset(nn);
	info->fltpk_arr = NULL;

	void *tmp = nn_scratch_alloc(nn, sizeof(struct heatmap_fltpeak)*heatmaps );
	if( tmp == NULL) return errlog(nn,"scratch alloc");
	info->fltpk_arr = tmp;

	// work out how to integer divide by hm_wid
	// This is done exactly by (k*wid_divide_fac)>>(32+wid_divide_shift)
	// (where the multiply is u32*u32->u64)
	// When wid is a power of 2, wid_divide_fac is 0x80000000.
	// So wid_divide_shift >=0, since hm_wid >= 2
	//
	{
		uint32_t fac;
		int shift;
		int log2w = Q6_R_ct0_R(hm_wid);
		int wx = hm_wid >> log2w;		// always odd
		if( wx == 1){
			fac = 0x80000000u;			// is a power of 2
			shift = log2w-1;
		}else{
			static const uint32_t divtab[8] = {
					0xaaaaaaab,		// 3
					0xcccccccd,		// 5
					0x92492493,		// 7
					0xe38e38e4,		// 9
					0xba2e8ba3,		// 11
					0x9d89d89e,		// 13
					0x88888889,		// 15
					0xF0F0F0F1,		// 17
			};
			shift = 31-Q6_R_cl0_R(wx);
			if( wx <= 17){
				int idx = max_i32((wx-3)>>1,0);
				fac = divtab[idx];
			}else{
				fac = ((uint64_t)0x100000000ull << shift)/wx + 1;
			}
			shift += log2w;
			// test it
			unsigned p1 = Q6_R_mpyu_RR( hm_area, fac )>>shift;
			unsigned p2 = Q6_R_mpyu_RR( (hm_area-1), fac )>>shift;
			if( p1 != hm_ht || p2 != hm_ht-1) return errlog(nn,"div calc failed");
		}
		info->wid_divide_fac = fac;
		info->wid_divide_shift = shift;
	}


	if( is_qu8){
		// storage for 3x3 windows over a whole batch.
		// each occupies 4-byte slot over 3 vectors. Total # of vectors is thus
		// heatmaps/32 (rounded up) times 3.
		// (each group of 3 vectors in the array contains 32 3x3 windows).
		// Make a full group of 12 vecs for each 128 heat maps
		int winbuf_size = ((heatmaps+127)/128u)*3*4*128;  // this many bytes per row
		tmp = nn_scratch_alloc(nn, winbuf_size);
		if( tmp == NULL) return errlog(nn,"scratch alloc");
		info->winbuf_ptr = tmp;

		int pkbuf_size =heatmaps * batches* sizeof(uint16_t);
		tmp = nn_scratch_alloc(nn, pkbuf_size);
		if( tmp == NULL) return errlog(nn, "scratch alloc");
		info->pk_store = tmp;
	}

	info->strategy_valid = 1;
	return 0;
}
static void find_peak_position_floats(struct nn_graph *nn, struct heatmap_maxpk_info * info , int ibatch);
static void process_heatmap_float_one_batch(struct nn_graph *nn, struct heatmap_maxpk_info * info , int ibatch);
static struct solve_peak_results  solve_peak_float( const float window[3][3]);

static int
heatmap_maxpk_execute( struct nn_node *self, struct nn_graph *nn )
{
	int is_qu8 = 0;
	if( check_heatmap_maxpk_strategy(nn,self,is_qu8)!= 0 ) return -1;

	struct heatmap_maxpk_info * info = (struct heatmap_maxpk_info*)self->opaque;

	for( int ibatch = 0; ibatch < info->batches; ibatch++){
		find_peak_position_floats( nn, info, ibatch);
		process_heatmap_float_one_batch( nn, info, ibatch);
	}
	return 0;
}

static void
process_heatmap_float_one_batch(struct nn_graph *nn, struct heatmap_maxpk_info * info , int ibatch)
{
	int heatmaps = info->heatmaps;

	struct heatmap_fltpeak const *fptab = info->fltpk_arr;
	// where the peak values go in the output
	float * outp_peak= (float*)info->out_peak_ptr + heatmaps * ibatch;
	// where the x,y pairs go
	float * outp_xy = (float*)info->out_xy_ptr + heatmaps*2 * ibatch;

	int hm_area = info->hm_area;
	int hm_width = info->hm_wid;
	int hm_height = info->hm_ht;

	int x_stride = info->hm_data_stride / sizeof(float);
	int y_stride = x_stride * hm_width;

	float window[3][3];

	// get the window parms
	float const *boxp = (float const *)info->box_ptr + 4*ibatch;
	float x_bb_0 = boxp[0];
	float y_bb_0 =  boxp[1];
	float x_bb_len = fmaxf( boxp[2]-x_bb_0,1.0f);
	float y_bb_len = fmaxf( boxp[3]-y_bb_0,1.0f);


	float x_bb_scale = x_bb_len /(float)(hm_width);
	float y_bb_scale = y_bb_len /(float)(hm_height);

	uint32_t div_fac = info->wid_divide_fac;
	int div_rsh = info->wid_divide_shift;

	for(int ihm =0; ihm < heatmaps; ihm++){
		int ipk = fptab[ihm].index;
		if( (unsigned)ipk >= (unsigned)hm_area){
			errlog(nn,"peak index out of range!");
			continue;
		}
		//int ipy = ipk/(unsigned)hm_width;
		int ipy = Q6_R_mpyu_RR(ipk,div_fac)>>div_rsh;	// divide by using precomputed scale
		int ipx = ipk-ipy * hm_width;
		// extract 3x3 at that position. wrap edges if needed
		// this is the start of the heatmap.
		float const * readp = (float const*) (
				(uint8_t const*)info->hm_input + ibatch * info->hm_batch_stride + ihm * info->hm_hmap_stride
				);
		// offset for peak position
		readp += x_stride * ipk;
		float const * rpup = readp-y_stride;	// row above
		float const * rpdn = readp+y_stride;	// row below
		if( ipy ==0) rpup = rpdn;
		else if( ipy+1 >= hm_height)rpdn = rpup;

		int off_left = (ipx>0)? -x_stride: x_stride;
		int off_right = (ipx+1 < hm_width)? x_stride: -x_stride;
		window[0][0] = rpup[off_left];
		window[0][1] = rpup[0];
		window[0][2] = rpup[off_right];
		window[1][0] = readp[off_left];	// get middle row
		window[1][1] = readp[0];
		window[1][2] = readp[off_right];
		window[2][0] = rpdn[off_left];
		window[2][1] = rpdn[0];
		window[2][2] = rpdn[off_right];

		struct solve_peak_results peakres = solve_peak_float( window);
		// rescale the values to the window
		float x_in_bbox = x_bb_0 + x_bb_scale* ( (float)ipx + 0.5f + peakres.xoff );
		float y_in_bbox = y_bb_0 + y_bb_scale* ( (float)ipy + 0.5f + peakres.yoff );

		outp_peak[ihm] = peakres.pkest;
		outp_xy[2*ihm] = x_in_bbox;
		outp_xy[2*ihm+1] = y_in_bbox;
	}

}
//
// window[3][3] is a local sampling around the peak
// This function attempts to estimate the position and value of the true
// peak.
//   this is done by solving
//       H*x = -G
//  where H is the hessian (which should be negative definite, around a peak)
//  and G is the gradient.
//
// This is actually done a bit differently; we find A=-H (which is +ve definite), and 'B' is
// found as twice the gradient, so that we solve
//    A*x = B/2
// (the solution will be clipped to a range, without changing its direction,
//  if it is too large).
//
// The value of the function at the peak is found using a 2nd order Taylor series:
//
//     Fmax = window[0][0] +  grad .dot. x   + (1/2)  x^T * H  * x
//
//   which is done here as
//     Fmax = window[0][0] +  (1/2)[   B .dot. x   -  x^T * A  * x ]
//
//   and can be factored as
//     Fmax = window[0][0] +  (1/2)[   B   -  A * x ].dot x
//
//  and then further simplified (since A*x=B/2) to
//     Fmax = window[0][0] +  (1/4)B.dot x
//
// The last simplification can only be done when the x solution was not clipped.
//
// Also, when solving A*x = B/2, this code refuses to invert cases
// where the det(A) < 0. Since the middle element is the peak,
// the diagonal of A should both be >=0; negative determinants should only
// occur in strange diagonal-ridge situations where the offset estimate
// will be unstable anyway.
//
//#define USE_LARGER_GRADHESS 1
static struct solve_peak_results  solve_peak_float( const float window[3][3])
{
	float tolA = DETER_TOL_ABS;
	float tolR = DETER_TOL_REL;
	float maxcorr = MAX_XY_CORR;
	struct solve_peak_results result;
#ifdef USE_LARGER_GRADHESS
	float vf0  = 0.25f*( window[0][0] + 2.0f*window[1][0] + window[2][0]);
	float vf1_x2  = 0.5f*( window[0][1] + window[2][1])+ window[1][1];
	float vf2  = 0.25f*( window[0][2] + 2.0f*window[1][2] + window[2][2]);

	float hf0  = 0.25f*( window[0][0] + 2.0f*window[0][1] + window[0][2]);
	float hf1_x2  = 0.5f*( window[1][0] + window[1][2])+ window[1][1] ;
	float hf2  = 0.25f*( window[2][0] + 2.0f*window[2][1] + window[2][2]);
	float B_0 = vf2 - vf0;		// x gradient * 2
	float B_1 = hf2 - hf0;		// y gradient * 2
	float A_00 = vf1_x2- (  vf0 + vf2);	// - d2F/dx2
	float A_11 = hf1_x2- (  hf0 + hf2);	// - d2F/dy2

#else
	float B_0 = window[1][2] - window[1][0];		// x gradient * 2
	float B_1 = window[2][1] - window[0][1];		// y gradient * 2

	float A_00 = 2.0f*window[1][1]- (  window[1][2] + window[1][0]);	// - d2F/dx2
	float A_11 = 2.0f*window[1][1]- (  window[2][1] + window[0][1]);	// - d2F/dy2
#endif
	float A_01 = (-0.25f)*((window[0][0] - window[0][2]) - ( window[2][0] - window[2][2]));
	float A_10 = A_01;

	// now we need to solve
	//
	//  [ A_00  A_01]  [ dx ]   = [ B_0 / 2]
	//  [ A_10  A_11]  [ dy ]   = [ B_1 / 2]

	float xp1 = A_00*A_11;
	float xp2 = A_01*A_10;
	float deter= xp1-xp2;

	// this matrix should be positive definite; a negative determinant is
	// a bad case (e.g. saddle surface), and that is considered non-invertable.
	//
	if( xp1 <=0.0f || deter < tolA + tolR*xp1){	// non-invertable
		result.pkest = window[1][1];
		result.xoff = 0.0f;
		result.yoff = 0.0f;
	}else{
		// below is the (dx,dy) solution, but still need multiply by 2/determinant.

		float dxV = A_11* B_0 - A_01 * B_1;
		float dyV = A_00* B_1 - A_10 * B_0;
		// check for 'too large'
		float maxV = fmaxf( fabsf(dxV), fabsf(dyV));
		float dx, dy, peakadj;
		if( maxV <= (maxcorr*2.0f)*deter  ){	// ok;=> max <= maxcorr
			float invdet = 0.5f/deter;
			dx = dxV * invdet;
			dy = dyV * invdet;
			// we can use the simpler form of the peak estimate here
			peakadj = 0.25f*( dx * B_0 + dy * B_1);
		}else{
			// use a smaller invdet
			float invdet = maxcorr/maxV;
			dx = dxV * invdet;
			dy = dyV * invdet;
#if USE_SIMPLER_PEAK_EST==0
			// estimate the peak:
			//  window[1][1] + 1st order corr + 2nd_order_corr
			//
			peakadj = 0.5f*(
					  dx*(B_0 - (dx*A_00 + dy*A_01))
					+ dy*(B_1 - (dx*A_10 + dy*A_11))
				);
#else
			peakadj = 0.25f*( dx * B_0 + dy * B_1);
#endif
		}

		result.xoff = dx;
		result.yoff = dy;
		result.pkest = window[1][1] + peakadj;
	}
	return result;
}

// for an entire batch, find the peak position within each heatmap.
// these are stored to the info->fltpk_arr array.
// for NCHW mode, each heatmap is contiguous, so we do one at a time;
// for NHWC mode, they are all interleaved; so we do 4 in a pass to
// get better caching.
//

static void
find_peak_position_floats(struct nn_graph *nn, struct heatmap_maxpk_info * info , int ibatch)
{
	float const *batchp = (float const*) ( (uint8_t const*)info->hm_input + ibatch * info->hm_batch_stride);
	int hm_area = info->hm_area;
	int hm_data_stride = info->hm_data_stride;	// these are in bytes
	int hm_hmap_stride = info->hm_hmap_stride;
	int heatmaps = info->heatmaps;

	struct heatmap_fltpeak *fpa = info->fltpk_arr;
	if( hm_data_stride == sizeof(float)){	// each is contiguous
		hm_hmap_stride /= sizeof(float);

		for( int i = 0; i < heatmaps; i++){
			float const * rp = batchp + i*hm_hmap_stride;
			int besti = 0;
			float peakval = rp[0];
			for(int k= 1; k < hm_area;k++){
				float v = rp[k];
				if( v > peakval) besti = k;
				peakval = fmaxf( v, peakval);
			}
			//fpa[i].pkval = peakval;
			fpa[i].index = besti;
		}
	}else{
		hm_data_stride /= sizeof(float);		// they are all interleaved
		// do 4 at a time so as not to thrash the cache too much.
		int heatmaps4 = heatmaps & ~3;
		for( int i = 0; i < heatmaps4; i+= 4){
			float const * rp = batchp + i;
			int best0 = 0,  best1 =0,  best2 = 0, best3 = 0;
			float peakval0 = rp[0];
			float peakval1 = rp[1];
			float peakval2 = rp[2];
			float peakval3 = rp[3];
			for( int k =1; k < hm_area; k++){
				rp += hm_data_stride;
				float v0 = rp[0]; if ( v0 > peakval0) best0 = k; peakval0 = fmaxf(peakval0,v0);
				float v1 = rp[1]; if ( v1 > peakval1) best1 = k; peakval1 = fmaxf(peakval1,v1);
				float v2 = rp[2]; if ( v2 > peakval2) best2 = k; peakval2 = fmaxf(peakval2,v2);
				float v3 = rp[3]; if ( v3 > peakval3) best3 = k; peakval3 = fmaxf(peakval3,v3);
			}
			fpa[i].index = best0;
			fpa[i+1].index = best1;
			fpa[i+2].index = best2;
			fpa[i+3].index = best3;
		}
		for( int i = heatmaps4; i < heatmaps; i++){
			float const * rp = batchp + i;
			int best0 = 0;
			float peakval0 = rp[0];
			for( int k =1; k < hm_area; k++){
				rp += hm_data_stride;
				float v0 = rp[0]; if ( v0 > peakval0) best0 = k; peakval0 = fmaxf(peakval0,v0);
			}
			fpa[i].index = best0;
		}
	}
}

////////// 8-bit implementation
static void __attribute__((unused)) run_pkfind_hvx_test_cases( struct nn_graph *nn, void *parm);

static void find_peak_position_u8(struct nn_graph *nn,  void * rstpv);
static void process_heatmap_u8_one_batch(struct nn_graph *nn, void * rstpv);
static void collect_windows_u8_one_batch(struct nn_graph *nn, struct heatmap_maxpk_info * info , int ibatch );
static void hvx_solve_peak_offset( HVX_Vector * win0, HVX_Vector *win1);

static int
heatmap_maxpk_execute_8( struct nn_node *self, struct nn_graph *nn )
{
	int is_qu8 = 1;
	if( check_heatmap_maxpk_strategy(nn,self,is_qu8 )!= 0 ) return -1;

	struct tensor const *in_min_tensor = self->inputs[1];
	struct tensor const *in_max_tensor = self->inputs[2];
	struct tensor * out_min_tensor = self->outputs[1];
	struct tensor * out_max_tensor = self->outputs[2];

	struct heatmap_maxpk_info * info = (struct heatmap_maxpk_info*)self->opaque;
	info->in_min = tensor_get_float(in_min_tensor,0);
	info->in_max = tensor_get_float(in_max_tensor,0);
	int qzero = get_qu8_level_size_zero( info->in_min, info->in_max, & info->in_qstep);


	struct heatmap_maxpk_runstate rst;
	rst.info = info;
	nn_sem_init( &rst.done_sem, 0);

	int n_threads = info->heatmaps > 128? 2: 1;

	//nn_os_work_for_vector(nn, run_pkfind_hvx_test_cases, NULL);
	// init min/max to the value corresponding to zero
	rst.peak_max = qzero*64;
	rst.peak_min = qzero*64;

	for( int ibatch = 0; ibatch < info->batches; ibatch++){
		rst.current_batch_ind = ibatch;

		rst.curr_heatmap = 0;
		nn_os_work_for_vector(nn,  find_peak_position_u8, &rst );
		nn_sem_wait( & rst.done_sem);

		collect_windows_u8_one_batch( nn, info, ibatch );

		rst.curr_heatmap = 0;

		for( int i = 0; i < n_threads; i++)
			nn_os_work_for_vector(nn,  process_heatmap_u8_one_batch, &rst );

		nn_sem_wait_n_times(&rst.done_sem, n_threads);
	}
	// now we need to look at the range of our peak values; and convert from
	// current form (input units with 6 extra fractional bits; possibly >= 256*64)
	// to an appropriate scale according to the actual range seen.

	int pkmin = rst.peak_min-qzero*64;	// same units as input with 6 fractional bits (<=0)
	int pkmax = rst.peak_max-qzero*64;	// >=0
	pkmax = max_i32( pkmax, pkmin+ 128);	// avoid a too-tiny range
	// convert to application units
	float out_minval = (float)pkmin * ( info->in_qstep * (float)(1./64));
	float out_maxval = (float)pkmax * ( info->in_qstep * (float)(1./64));
	// adjust range
	adjust_minmax_for_zero( &out_minval, &out_maxval);
	// get new z & q
	float out_qstep;
	int out_zero = get_qu8_level_size_zero( out_minval, out_maxval, & out_qstep);

	logmsg(nn,3,"in_q=%f; zero=%d; integer range = (%d..%d) out_range = %f..%f (q=%f,z=%d)",
			info->in_qstep, qzero, pkmin+qzero*64,pkmax+qzero*64, out_minval, out_maxval, out_qstep,out_zero);
	// the transform is done as out[i] = in[i]*cvt_scale + cvtoff, where
	float cvt_scale = info->in_qstep/(64.0f*out_qstep);
	float cvt_off = (float)out_zero - (float)(qzero*64) * cvt_scale + 0.5f;
	// ** TODO ** vectorize this

	int n = info->heatmaps * info->batches;
	uint16_t  const * inp = info->pk_store;
	uint8_t *outp = (uint8_t *)info->out_peak_ptr;

	for(int i = 0; i < n; i++){
		float newval = (float)inp[i] * cvt_scale + cvt_off;
		outp[i] = saturate_u8( (int)newval);
	}
	if( tensor_set_single_float( out_min_tensor, out_minval)!=0
		|| tensor_set_single_float( out_max_tensor, out_maxval)!=0 ){
		return errlog(nn,"no room for min/max out");
	}

	return 0;
}

// define this to enable checking each fixed-point solution against the
// float calc, for debug
//#define FLOAT_CHECK 1


// this is called to read the values from the 3x3 window tables, do the peak estimation,
// and finish the outputs. It works in units of 128 heatmaps; it can be called in multiple
// threads if there are > 128.
static void
process_heatmap_u8_one_batch(struct nn_graph *nn, void *rstpv )
{
	struct heatmap_maxpk_runstate * rstp = (struct heatmap_maxpk_runstate *)rstpv;
	struct heatmap_maxpk_info const * info = rstp->info;
	int ibatch = rstp->current_batch_ind;

	struct heatmap_fltpeak const *fptab = info->fltpk_arr;
	int heatmaps = info->heatmaps;
	int hm_width = info->hm_wid;
	int hm_height = info->hm_ht;

	// where the peak values go (not to output, but to a temp buffer)
	uint16_t * outp_peak= info->pk_store + heatmaps * ibatch;
	// where the x,y pairs go
	uint16_t * outp_xy = (uint16_t*)info->out_xy_ptr + heatmaps*2 * ibatch;


	// get the window parms
	uint16_t const *boxp = (uint16_t const *)info->box_ptr + 4*ibatch;
	int x_bb_0 = boxp[0];
	int y_bb_0 =  boxp[1];
	int x_bb_len = max_i32( boxp[2]-x_bb_0,1);
	int y_bb_len = max_i32( boxp[3]-y_bb_0,1);
	// factored-out scaling factors - this includes (1/256) since the
	// dx,dy offsets have 8 fractional bits.
	//
	float x_bb_scale = (float)(1./256.) * (float)x_bb_len /(float)(hm_width);
	float y_bb_scale = (float)(1./256.) * (float)y_bb_len /(float)(hm_height);


	int local_pk_min = 0x7fffffff;
	int local_pk_max = -0x7fffffff;


	int ihm0;
	// grab a unit of 128 to do
	while( ihm0 = __sync_fetch_and_add( &rstp->curr_heatmap, 128), ihm0 < heatmaps){
		int n_heatmaps = min_i32( 128, heatmaps-ihm0);	// number to do here.. 1..128
		int16_t*  winpos=  (int16_t*) ( info->winbuf_ptr + 12*ihm0);	// point to the current group of (up to) 12 vectors
		int vloops = (n_heatmaps+63)/64u;	// # of vector loops

#ifdef FLOAT_CHECK
		uint8_t copy_wins[12*128];
		memcpy( copy_wins, winpos,vloops*6*128);
#endif
		int hm_remain = n_heatmaps;
		HVX_Vector * vwinpos = (HVX_Vector*)winpos;
		// process 1,2,3 or 4 groups of 3 vectors.
		// Each loop handles 1 or 2 groups. The results are stored back into the first 2 vectors of each group.
		for( int i =0; i < vloops; i++){
			hvx_solve_peak_offset(
					vwinpos, // point to first group of 3 vectors
					(hm_remain<=32)? vwinpos : (vwinpos+3));	// point to next group (or same, <= 32 remain)
			vwinpos += 6;
		}
		// store all the results out.
		Q6_dcfetch_A(winpos);
		Q6_dcfetch_A(winpos+64);

#ifdef FLOAT_CHECK
		//// check with float reference
		for(int ii = 0; ii < n_heatmaps; ii++){
			float win[3][3];
			uint8_t const *wptr = &copy_wins[ii*4 + 256*(ii>>5)*256];
			int16_t const * res_ptr = winpos + ii*2 + (ii>>5)*128; // {0,2, ..60,62, 192,194,..}
			for(int i =0; i < 3; i++){
				win[i][0] = wptr[0];
				win[i][1] = wptr[1];
				win[i][2] = wptr[2];
				wptr += 128;
			}
			struct solve_peak_results pkres =  solve_peak_float(win);
			printf("%4d:  fxp = %9.5f %9.5f %9.5f   flt= fxp = %9.5f %9.5f %9.5f\n",
					ii+ihm0, res_ptr[0]*(float)(1/256.), res_ptr[1]*(float)(1/256.), res_ptr[64]*(float)(1/64.),
					pkres.xoff, pkres.yoff, pkres.pkest );
		}
#endif

		for( int ii = 0; ii  < n_heatmaps; ii++){
			int ihm = ihm0 + ii;
			// point to the (x,y) solution in the vector memory...
			int16_t const * res_ptr = winpos + ii*2 + (ii>>5)*128; // {0,2, ..60,62, 192,194,..}

#if 1
			// rescale the values to the window
			int32_t ipxy = fptab[ihm].index;			// load two 16-bit values (index_x, index_y)
			int64_t ipxy_w = Q6_P_vmpyhsu_RR_sat( 0x01000100, ipxy);	// multiply uint16 by 256 and extend

			int32_t dxy = *(int32_t const*)res_ptr;		// get dx and dy correction
			dxy = Q6_R_vaddh_RR( dxy, 0x00800080);		// add 128 to each ( +0.5)
			int64_t pxy = Q6_P_vaddw_PP(
					ipxy_w, Q6_P_vsxthw_R( dxy) );		// sign extend, add to ipx,ipy
			int x_in_bbox = x_bb_0 + roundf_i32(x_bb_scale* (float)( (int32_t) pxy));
			int y_in_bbox = y_bb_0 + roundf_i32(y_bb_scale* (float)( (int32_t) (pxy>>32)));
#else	// non-vector version
			// rescale the values to the window
			int ipy = fptab[ihm].index_y;
			int ipx = fptab[ihm].index_x;
			ipx = ipx*256 + res_ptr[0] + (1<<7);		// convert to 8 fractional bits; add dx; add 0.5
			ipy = ipy*256 + res_ptr[1] + (1<<7);		// same for y
			int x_in_bbox = x_bb_0 + roundf_i32(x_bb_scale* (float)ipx);
			int y_in_bbox = y_bb_0 + roundf_i32(y_bb_scale* (float)ipy);
#endif

			outp_xy[2*ihm] = saturate_u16(x_in_bbox);
			outp_xy[2*ihm+1] = saturate_u16( y_in_bbox);

			// get the 'pkest' from memory; it has 6 fractional bits, and is just copied to another buffer
			// (later to be rescaled to output)
			int pk_est = res_ptr[64];
			local_pk_min = min_i32( local_pk_min, pk_est);
			local_pk_max = max_i32( local_pk_max, pk_est);
			outp_peak[ihm] = pk_est;
		}
	}
	// update the overall min/max

	nn_atomic_min( &rstp->peak_min, local_pk_min);
	nn_atomic_max( &rstp->peak_max, local_pk_max);

	nn_sem_post( &rstp->done_sem);
}
static void reduce_to_row_col( struct heatmap_maxpk_info const * info, uint32_t * datap, int n);
//
// collect all the 3x3 windows for one batch
//
static void
collect_windows_u8_one_batch(struct nn_graph *nn, struct heatmap_maxpk_info * info , int ibatch )
{
	struct heatmap_fltpeak const *fptab = info->fltpk_arr;
	int hmcount = info->heatmaps;
	int hm_width = info->hm_wid;
	int hm_height = info->hm_ht;
	int x_stride = info->hm_data_stride;
	int y_stride = x_stride * hm_width;

	for( int ihm0 = 0; ihm0 < hmcount; ihm0 += 32){  // groups of 32...
		int hlimit = min_i32(hmcount,ihm0+32);
		uint32_t *winpos = (uint32_t*)(info->winbuf_ptr + (4*3)*ihm0);	// which group of 3 vectors...
		for(int ihm = ihm0; ihm < hlimit; ihm++){
			int ipy = fptab[ihm].index_y;
			int ipx = fptab[ihm].index_x;
			// extract 3x3 at that position. wrap edges if needed
			// this is the start of the heatmap.
			uint8_t const * readp =  (
					(uint8_t const*)info->hm_input + ibatch * info->hm_batch_stride + ihm * info->hm_hmap_stride
					);
			// offset for peak position
			readp += x_stride *ipx + y_stride * ipy;
			uint8_t const * rpup = readp-y_stride;	// row above
			uint8_t const * rpdn = readp+y_stride;	// row below
			if( ipy ==0) rpup = rpdn;
			else if( ipy+1 >= hm_height)rpdn = rpup;

			int off_left = (ipx>0)? -x_stride: x_stride;
			int off_right = (ipx+1 < hm_width)? x_stride: -x_stride;
			// pack 3 bytes into each word:  0:right:middle:left
#define COMBINE_3(PTR) (Q6_R_combine_RlRl( (PTR)[off_right], ((PTR)[0]<<8) | (PTR)[off_left]))
			winpos[0*32] = COMBINE_3(rpup);
			winpos[1*32] = COMBINE_3(readp);
			winpos[2*32] = COMBINE_3(rpdn);
			winpos ++;	// next window
#undef COMBINE_3
		}
	}
}

// find the largest peak position in each heatmap
// and store the results in the table.
static void
find_peak_position_u8(struct nn_graph *nn, void * rstpv)
{
	struct heatmap_maxpk_runstate * rstp = (struct heatmap_maxpk_runstate *)rstpv;
	struct heatmap_maxpk_info const * info = rstp->info;
	int ibatch = rstp->current_batch_ind;

	int hm_area = info->hm_area;
	int hm_data_stride = info->hm_data_stride;	// these are in bytes
	int hm_hmap_stride = info->hm_hmap_stride;
	int heatmaps = info->heatmaps;

	uint8_t const *batchp = ( (uint8_t const*)info->hm_input + ibatch * info->hm_batch_stride);

	struct heatmap_fltpeak *fpa = info->fltpk_arr;

	if( hm_data_stride ==1){
		hvx_argmin_or_max_in_rows( batchp, heatmaps, hm_area,hm_hmap_stride, (int32_t*)fpa , 1 );
	}else{
		hvx_argmin_or_max_in_cols( batchp, hm_area, heatmaps, hm_data_stride, (int32_t*)fpa , 1 );
	}
	// convert all the indices to row/col format
	reduce_to_row_col(info, (uint32_t*)fpa , heatmaps );

	nn_sem_post(& rstp->done_sem);
}

//
// 'data' points to 'n' uint32 values
// reduce all of them to  x,y such that  x + y*width == (old value)
//  and put the 'x,y' back instead.
//  It is guaranteed that x,y both fit in u16.
// use HVX shifts or mul, with precalc factors.
//
static void
reduce_to_row_col( struct heatmap_maxpk_info const * info, uint32_t * datap, int n)
{
	int vecloops = (n-1)/32u;
	HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector *)datap);
	HVX_Vector vout_last = Q6_V_vzero();
	uint32_t *datap_end = datap + n;

	int width = info->hm_wid;
	uint32_t mulfac = info->wid_divide_fac;
	int rshift = info->wid_divide_shift;

	if( mulfac == 0x80000000){		// is a power of 2
		int log2w = rshift+1;
		HVX_Vector vmask = Q6_V_vsplat_R(width-1);
		HVX_Vector vx = Q6_V_vand_VV( v0, vmask);
		HVX_Vector vy = Q6_Vuw_vlsr_VuwR(v0, log2w );
		for( int i=0; i < vecloops; i++){
			HVX_Vector vout = Q6_Vh_vshuffe_VhVh( vy,vx);
			v0 = q6op_V_vldu_A( (HVX_Vector *)(datap+32));
			q6op_vstu_AV( (HVX_Vector*)datap, vout);
			vx = Q6_V_vand_VV( v0, vmask);
			vy = Q6_Vuw_vlsr_VuwR(v0, log2w );
			datap += 32;
		}
		vout_last = Q6_Vh_vshuffe_VhVh( vy,vx);
	}else {
		// not a power of 2
		// To get the quotient, we need to multiply the values by mulfac using
		// u32*u32 -> u64, keep only the upper 32 bits, and then >> rsh
		// The product must be exact.
		HVX_Vector prodhi = find_u64_prod_VuwRuw( v0, mulfac).val[1];
		HVX_Vector vy = Q6_Vuw_vlsr_VuwR(prodhi, rshift );		// this is the quotient...
		HVX_Vector vx = Q6_Vw_vsub_VwVw( v0, Q6_V_lo_W( Q6_Wuw_vmpy_VuhRuh(vy,width)));
		for( int i=0; i < vecloops; i++){
			HVX_Vector vout = Q6_Vh_vshuffe_VhVh( vy,vx);
			v0 = q6op_V_vldu_A( (HVX_Vector *)(datap+32));
			q6op_vstu_AV( (HVX_Vector*)datap, vout);
			HVX_Vector prodhi = find_u64_prod_VuwRuw( v0, mulfac).val[1];
			vy = Q6_Vuw_vlsr_VuwR(prodhi, rshift );		// this is the quotient...
			vx = Q6_Vw_vsub_VwVw( v0, Q6_V_lo_W( Q6_Wuw_vmpy_VuhRuh(vy,width)));
			datap += 32;
		}
		vout_last = Q6_Vh_vshuffe_VhVh( vy,vx);
	}
	int nbytes = (uint8_t*)datap_end - (uint8_t*)datap;
	q6op_vstu_variable_ARV( (HVX_Vector*)datap, nbytes,  vout_last);
}



static inline HVX_VectorPair hvx_Ww_dotproduct_VhVh_VhVh(
	HVX_Vector a, HVX_Vector b, HVX_Vector c, HVX_Vector d )
{
	return Q6_Ww_vmpyacc_WwVhVh(Q6_Ww_vmpy_VhVh(a,b), c,d);
}


// input is 2 of { xp1.w, xp2,w, det.w}
// where xp1 is a*d, xp2 is b*c for an [ a b; c d ] matrix
// and det = xp1-xp2
// Return 32 w which are >= 0 if the matrix is usable.
// requires
//   det >= tol_A + tol_R * xp1
// All of the inputs have 12 fractional bits.
//

static inline HVX_Vector check_deter_condition( HVX_Vector xp1, HVX_Vector xp2, HVX_Vector det)
{
	int tolA_i = DETER_TOL_FXP_ABS;			// with 12 fractional bits
	int Rscale = DETER_TOL_FXP_REL;			// with 15 fractional bits

	HVX_Vector diff = Q6_Vw_vsub_VwVw( det,Q6_V_vsplat_R(tolA_i+1) ); 	// subtract tolA+1

	HVX_Vector R_times_xp1 = Q6_Vw_vmpyo_VwVh_s1_sat( xp1, Q6_V_vsplat_R(Rscale<<16));
	// subtract scaled xp1
	diff = Q6_Vw_vsub_VwVw( diff, R_times_xp1);
	return diff;
}

// find max( deter,  maxv/max_corr) in each W lane
// (where 'max_corr' is a compile-time constant, and at least 0.25.
//
static inline HVX_Vector find_eff_determinant( HVX_Vector deter, HVX_Vector maxv, float max_corr )
{
	HVX_Vector scaled;
	if( max_corr >=1.0000){
		int inv_i = (int)(0.5f + 32768.0f/max_corr);
		if( inv_i == 32768){
			scaled = maxv;
		}else{
			scaled = Q6_Vw_vmpyo_VwVh_s1_sat( maxv, Q6_V_vsplat_R(inv_i<<16));
		}
	}else if( max_corr >= 0.5f){
		int inv_i = (int)(0.5f + 16384.0f/max_corr);
		if( inv_i == 32768){
			scaled = maxv;
		}else{
			scaled = Q6_Vw_vmpyo_VwVh_s1_sat( maxv, Q6_V_vsplat_R(inv_i<<16));
		}
		scaled = Q6_Vw_vadd_VwVw_sat( scaled,scaled);
	}else{
		int inv_i = (int)(0.5f + 8192.0f/max_corr);
		if( inv_i == 32768){
			scaled = maxv;
		}else{
			scaled = Q6_Vw_vmpyo_VwVh_s1_sat( maxv, Q6_V_vsplat_R(inv_i<<16));
		}
		scaled = Q6_Vw_vadd_VwVw_sat( scaled,scaled);
		scaled = Q6_Vw_vadd_VwVw_sat( scaled,scaled);
	}
	return Q6_Vw_vmax_VwVw( scaled, deter);
}


//
// This processes 2 groups of 3  vectors; each group contains 32 3x3 windows of u8. The
// result is 2x32 of { deltx, delty, peakest}. The 'peakest' is in the same units as the
// input u8, but with 8 fractional bits (and may be greater than 255.0)
//
static inline void __attribute__((unused)) hvx_solve_peak_offset( HVX_Vector * win0, HVX_Vector *win1)
{
	const float max_corr = MAX_XY_CORR; // >=0.25, <= 1.5


	HVX_Vector vup0 = win0[0];
	HVX_Vector vmid0 = win0[1];
	HVX_Vector vdn0 = win0[2];
	HVX_Vector vup1 = win1[0];
	HVX_Vector vmid1 = win1[1];
	HVX_Vector vdn1 = win1[2];


	// find the gradient estimate with 6 fractional bits
	// 32*(f+) - 32*(f-)
	// range is +/-8160
	//
	HVX_Vector gradx_0 = Q6_Vw_vrmpy_VubRb( vmid0, 0x002000E0);
	HVX_Vector upmid0 = Q6_Vw_vrmpy_VubRb( vup0,0x0000C000);  // f[0,1]*-64
	HVX_Vector dnmid0 = Q6_Vw_vrmpy_VubRb( vdn0,0x0000C000);  // f[2,1]*-64

	HVX_Vector gradx_1 = Q6_Vw_vrmpy_VubRb( vmid1, 0x002000E0);
	HVX_Vector upmid1 = Q6_Vw_vrmpy_VubRb( vup1,0x0000C000);
	HVX_Vector dnmid1 = Q6_Vw_vrmpy_VubRb( vdn1,0x0000C000);

	// pack 0/1 into h lanes
	HVX_Vector gradx = Q6_Vh_vshuffe_VhVh(gradx_1, gradx_0 );
	HVX_Vector upmid = Q6_Vh_vshuffe_VhVh( upmid1, upmid0);
	HVX_Vector dnmid = Q6_Vh_vshuffe_VhVh( dnmid1, dnmid0);
	HVX_Vector grady = Q6_Vh_vnavg_VhVh( upmid, dnmid);// 32*f[2,1] - 32*f[0,1]
	//
	// find the hessian estimates with 6 fractional bits
	// (these are actually the -ve hessian, so we get a +ve definite matrix)
	//
	//  -64*(f-) 128*(f0) - 64*f1
	// range is +/-32640 for the hxx/hxy, and +/- 8160 for the hxy
	//
	HVX_Vector mid128_0 = Q6_Vuw_vrmpy_VubRub( vmid0, 0x00008000);	// middle * (+128)
	HVX_Vector mid128_1 = Q6_Vuw_vrmpy_VubRub( vmid1, 0x00008000);	// middle * (+128)
	HVX_Vector mid128 = Q6_Vh_vshuffe_VhVh( mid128_1, mid128_0);

	HVX_Vector hess_xx_0 = Q6_Vw_vrmpyacc_VwVubRb(mid128_0,  vmid0, 0x00C000C0);
	HVX_Vector hess_xx_1 = Q6_Vw_vrmpyacc_VwVubRb(mid128_1,  vmid1, 0x00C000C0);

	HVX_Vector hess_xx = Q6_Vh_vshuffe_VhVh( hess_xx_1, hess_xx_0);
	HVX_Vector hess_yy = Q6_Vh_vadd_VhVh( mid128, Q6_Vh_vadd_VhVh( dnmid, upmid));

	HVX_Vector hess_xy_neg_0 = Q6_Vw_vrmpyacc_VwVubRb(
			Q6_Vw_vrmpy_VubRb( vup0, 0x00F00010),
						vdn0,0x001000F0);
	HVX_Vector hess_xy_neg_1 = Q6_Vw_vrmpyacc_VwVubRb(
			Q6_Vw_vrmpy_VubRb( vup1, 0x00F00010),
						vdn1,0x001000F0);
	HVX_Vector hess_xy_neg= Q6_Vh_vshuffe_VhVh( hess_xy_neg_1, hess_xy_neg_0);


	/*printf("grad: %d %d  hess: %d %d %d\n",
			vextract_i16(gradx,0), vextract_i16( grady,0),
			vextract_i16(hess_xx,0), -vextract_i16( hess_xy_neg,0), vextract_i16(hess_yy,0) );
	*/

	// We now have 64 of all 5. Check the determinants....
	HVX_VectorPair xp1 = Q6_Ww_vmpy_VhVh( hess_xx, hess_yy);
	HVX_VectorPair xp2 = Q6_Ww_vmpy_VhVh( hess_xy_neg, hess_xy_neg);
	HVX_VectorPair det_w = Q6_Ww_vsub_WwWw( xp1, xp2 );


	// check matrix condition
	// Result is < 0 if the matrix is bad
	HVX_Vector mcond_0 = check_deter_condition( Q6_V_lo_W(xp1), Q6_V_lo_W(xp2), Q6_V_lo_W(det_w));
	HVX_Vector mcond_1 = check_deter_condition( Q6_V_hi_W(xp1), Q6_V_hi_W(xp2), Q6_V_hi_W(det_w));
	HVX_Vector mcond = Q6_Vh_vshuffo_VhVh( mcond_1, mcond_0);	// keep sign bits only

	HVX_VectorPred bad_lane = Q6_Q_vcmp_gt_VhVh( Q6_V_vzero(), mcond);

	/*printf("mcond= %d based on xp1=%d det = %d\n", vextract_i16(mcond,0), vextract_i32(Q6_V_lo_W(det_w),0),
			vextract_i32(Q6_V_lo_W(xp1),0));*/

	// force the gradient to 0 in lanes which have bad condition. This is sufficient
	// to get a (0,0) solution for dx, and suppress adjustment of the peak estimate.

	gradx= q6op_V_vand_QnV( bad_lane, gradx);
	grady= q6op_V_vand_QnV( bad_lane, grady);

	// find the matrix solution, prior to /det
	// 	float dxV = A_11* B_0 - A_01 * B_1;
	//  float dyV = A_00* B_1 - A_10 * B_0;

	HVX_VectorPair dxV = hvx_Ww_dotproduct_VhVh_VhVh( hess_yy, gradx, hess_xy_neg, grady);
	HVX_VectorPair dyV = hvx_Ww_dotproduct_VhVh_VhVh( hess_xx, grady, hess_xy_neg, gradx);

	HVX_Vector max_DV_0 = Q6_Vw_vmax_VwVw(  Q6_Vw_vabs_Vw( Q6_V_lo_W(dxV)), Q6_Vw_vabs_Vw( Q6_V_lo_W(dyV)));
	HVX_Vector max_DV_1 = Q6_Vw_vmax_VwVw(  Q6_Vw_vabs_Vw( Q6_V_hi_W(dxV)), Q6_Vw_vabs_Vw( Q6_V_hi_W(dyV)));
	//
	// Now find  max( deter, max_DV/maxcorr)
	// and that's what we divide by to complete the solution (thus, the largest of dx, dy won't exceed maxcorr).
	// This has 12 fractional bits.
	//
	HVX_Vector eff_det_w_0 = find_eff_determinant( Q6_V_lo_W(det_w), max_DV_0, max_corr);
	HVX_Vector eff_det_w_1 = find_eff_determinant( Q6_V_hi_W(det_w), max_DV_1, max_corr);

	/*printf("DxV,DyV = %d %d   eff_det = %d\n",
			vextract_i32( Q6_V_lo_W(dxV),0), vextract_i32( Q6_V_lo_W(dyV),0), vextract_i32(eff_det_w_0,0));
	*/
	// find the magnitude of the determinants so we can invert
	//
	HVX_Vector nshift_0 = Q6_Vw_vnormamt_Vw( eff_det_w_0);
	HVX_Vector nshift_1 = Q6_Vw_vnormamt_Vw( eff_det_w_1);

	// normalize determinants, and pack to h. Now they have nshift-4 fractional bits.
	HVX_Vector det_h = Q6_Vh_vshuffo_VhVh(
			Q6_Vw_vasl_VwVw(eff_det_w_1, nshift_1 ),
			Q6_Vw_vasl_VwVw(eff_det_w_0, nshift_0 ) );

	// reciprocal has 33-nshift fractional bits.
	HVX_Vector inv_det_h = hvx_recip16_inline(det_h, 5 );

	//printf("det_shift = %d  det_h = %d inv_det_h = %d\n", vextract_i16(nshift_0,0),  vextract_i16(det_h,0), vextract_i16(inv_det_h,0));

	// the dx/dy solution will be in 16 bits, and will have 13 fractional bits.
	// first, position dvX, dvY as needed and pack to 16 bits, so we can just fractional mul by the reciprocal
	// of the determinant. Since the final result has mag <=1.5, it will be +/-12K, and thus the packed
	// dV will fit in +/-24K.
	// if we << DV by  (nshift-17), the result has nshift-5 fractional bits; multiply by recip with (33-nshift)
	// fractional bits, and deduct 15 for fractional mul; result will be 13 fractional bits.
	//  We actually << by nshift-16 and then >>1 in the packing. Note that nshift-16 can be +ve or negative.
	//
	nshift_0 = Q6_Vw_vsub_VwVw( nshift_0, Q6_V_vsplat_R(16));
	nshift_1 = Q6_Vw_vsub_VwVw( nshift_1, Q6_V_vsplat_R(16));

#if __HEXAGON_ARCH__>= 62
	HVX_Vector dxV_h = Q6_Vh_vasr_VwVwR_rnd_sat(
			Q6_Vw_vasl_VwVw( Q6_V_hi_W(dxV), nshift_1),
			Q6_Vw_vasl_VwVw( Q6_V_lo_W(dxV), nshift_0), 1);
	HVX_Vector dyV_h = Q6_Vh_vasr_VwVwR_rnd_sat(
			Q6_Vw_vasl_VwVw( Q6_V_hi_W(dyV), nshift_1),
			Q6_Vw_vasl_VwVw( Q6_V_lo_W(dyV), nshift_0), 1);
#else
	// v60 doesn't have bidirectional shift. So, need to do it the hard way...
	// shift left by |n|, or right by |n|, according to sign.
	//
	HVX_Vector nshift_0_abs = Q6_Vw_vabs_Vw( nshift_0);
	HVX_Vector nshift_1_abs = Q6_Vw_vabs_Vw( nshift_1);
	HVX_VectorPred nshift_0_isneg = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(),nshift_0);
	HVX_VectorPred nshift_1_isneg = Q6_Q_vcmp_gt_VwVw(Q6_V_vzero(),nshift_1);

	HVX_Vector dxV_h = Q6_Vh_vasr_VwVwR_rnd_sat(
			Q6_V_vmux_QVV( nshift_1_isneg,
					Q6_Vw_vasr_VwVw( Q6_V_hi_W(dxV), nshift_1_abs),
					Q6_Vw_vasl_VwVw( Q6_V_hi_W(dxV), nshift_1_abs)),
			Q6_V_vmux_QVV( nshift_0_isneg,
					Q6_Vw_vasr_VwVw( Q6_V_lo_W(dxV), nshift_0_abs),
					Q6_Vw_vasl_VwVw( Q6_V_lo_W(dxV), nshift_0_abs)), 1);

	HVX_Vector dyV_h = Q6_Vh_vasr_VwVwR_rnd_sat(
			Q6_V_vmux_QVV( nshift_1_isneg,
					Q6_Vw_vasr_VwVw( Q6_V_hi_W(dyV), nshift_1_abs),
					Q6_Vw_vasl_VwVw( Q6_V_hi_W(dyV), nshift_1_abs)),
			Q6_V_vmux_QVV( nshift_0_isneg,
					Q6_Vw_vasr_VwVw( Q6_V_lo_W(dyV), nshift_0_abs),
					Q6_Vw_vasl_VwVw( Q6_V_lo_W(dyV), nshift_0_abs)), 1);


#endif

	HVX_Vector dx = Q6_Vh_vmpy_VhVh_s1_rnd_sat( dxV_h, inv_det_h);		// dx result
	HVX_Vector dy = Q6_Vh_vmpy_VhVh_s1_rnd_sat( dyV_h, inv_det_h);		// dy result

#if USE_SIMPLER_PEAK_EST==0
	//
	// now: peak correction.
	// 			peakadj =
	//   dx*(B_0 - (1/2)(dx*A_00 + dy*A_01))
	// + dy*(B_1 - (1/2)(dx*A_10 + dy*A_11))
	// We want this with 6 fractional bits.
	//
	HVX_Vector hess_xy = Q6_Vh_vsub_VhVh_sat( Q6_V_vzero(), hess_xy_neg);

	// (1/2)(...)  with 6+13+1 = 20 fractional bits
	HVX_VectorPair tmpx = hvx_Ww_dotproduct_VhVh_VhVh( dx, hess_xx, dy, hess_xy);
	HVX_VectorPair tmpy = hvx_Ww_dotproduct_VhVh_VhVh( dx, hess_xy, dy, hess_yy);

	// round it down to 5 fractional bits...
	HVX_Vector tmpx_h = q6op_Vh_vasr_WwR_rnd_sat( tmpx,14);
	HVX_Vector tmpy_h = q6op_Vh_vasr_WwR_rnd_sat( tmpy,14);

	// subtract from the gradient...
	tmpx_h = Q6_Vh_vsub_VhVh_sat( gradx, tmpx_h);
	tmpy_h = Q6_Vh_vsub_VhVh_sat( grady, tmpy_h);

	// peak adjust with 6+13 = 19 fractional bits
	HVX_VectorPair peakadjw = hvx_Ww_dotproduct_VhVh_VhVh( dx, tmpx_h, dy, tmpy_h);

							// and round it down to 7 fractional bits
	HVX_Vector peakadj = q6op_Vh_vasr_WwR_rnd_sat( peakadjw,12);
#else
	// dot gradient * dx: 6+13= 19 fractional bits; 20 because we want to /2.
	HVX_VectorPair peakadjw = hvx_Ww_dotproduct_VhVh_VhVh( dx, gradx, dy, grady);
	// and round it down to 7 fractional bits
	HVX_Vector peakadj = q6op_Vh_vasr_WwR_rnd_sat( peakadjw,13);
#endif
	// add it to the middle sample (scaled to 6 fractional bits)
	HVX_Vector peakest = Q6_Vh_vavg_VhVh_rnd( peakadj, mid128 );

	// reduce dx,dy from 13 to 8 fractional bits.
	dx = Q6_Vh_vmpy_VhRh_s1_rnd_sat( dx, 0x04000400);
	dy = Q6_Vh_vmpy_VhRh_s1_rnd_sat( dy, 0x04000400);

	// unpack odd/even and store estimates.
	HVX_VectorPair pkest_w = Q6_Ww_vsxt_Vh(peakest);
	win0[0] = Q6_Vh_vshuffe_VhVh( dy, dx);		// [x:y] for the first set
	win0[1] = Q6_V_lo_W(pkest_w);

	win1[0] = Q6_Vh_vshuffo_VhVh( dy, dx);		// [x:y] for the second set
	win1[1] = Q6_V_hi_W(pkest_w);
}

#if 0
// test cases for 8-bit peak finder
static const
struct pkfind_testcase {
	uint8_t input[3][3];
	int16_t dx,dy;	// expected dx/dy with 13 fractional bits
	int16_t pk;		// expected peak val in u8 units with 6 more frac bits
} pkfind_testcases[6] = {
	{
		{{ 59,209,191}, { 90, 222, 185}, { 24, 138, 84 }},
		2675, -3495, 15189
	},
	{
		{{ 158,142,56}, { 201, 216, 160}, { 78, 123, 98 }},
		-2784, -975, 14083
	},
	{
		{{187, 184,170}, { 197,209, 212}, { 33, 62, 81 }},
		1906, -2725, 14081
	},
	{	// reversal in X of previous case
		{{170, 184,187}, { 212,209, 197}, { 81, 62, 33 }},
		-1906, -2725, 14081
	},
	{
		{{143,178,27}, { 145,200,69}, { 46, 119, 7 }},
		-1954, -2711, 13403
	},
	// this one is fairly badly-conditioned but the result isn't too bad
	{
		{{245,185,7}, { 201,248,178}, { 54, 209, 245 }},
		2165, 3240, 15927
	},
};

static void
run_pkfind_hvx_test_cases( struct nn_graph *nn, void *parm)
{
	union {
		HVX_Vector as_v[6];
		uint8_t as_u8[6][128];
		int16_t as_i16[6][64];
	} uu;
	int ncases = 6;
	// even test cases go in vectors 0,1,2; odd in 3,4,5
	for( int i = 0; i < ncases; i++){
		struct pkfind_testcase const * tcp = &pkfind_testcases[i];
		uint8_t *base_p = &uu.as_u8[(i&1)?3:0][(i>>1)*4];
		for( int j = 0; j <3; j++){
			base_p[128*j+0] = tcp->input[j][0];
			base_p[128*j+1] = tcp->input[j][1];
			base_p[128*j+2] = tcp->input[j][2];
		}
	}
	hvx_solve_peak_offset( &uu.as_v[0], &uu.as_v[3]);

	// check them
	for( int i = 0; i < ncases; i++){
		struct pkfind_testcase const * tcp = &pkfind_testcases[i];
		int16_t const * result_p = &uu.as_i16[(i&1)?3:0][(i>>1)*2];
		int16_t res_x = result_p[0];	// dx, dy from first vector
		int16_t res_y = result_p[1];
		int16_t pkval = result_p[64];	// pkval from second

		int16_t ok_dx = (tcp->dx+16)>>5;	// convert these from 13 to 8 frac. bits
		int16_t ok_dy = (tcp->dy+16)>>5;
		printf(" case %2d: expected (%5d, %5d, %6d);  got( %5d, %5d, %5d)\n",
			i, ok_dx, ok_dy, tcp->pk, res_x, res_y, pkval);

	}
}
#endif



static int
heatmap_maxpk_check( struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"heatmap_maxp 0x%08X check OK",(unsigned)self->node_id);
	if( self->opaque != NULL){
		nn_free(self->opaque);
	}
	self->opaque = nn_calloc( 1, sizeof(struct heatmap_maxpk_info ) );
	if( self->opaque ==NULL)
		return errlog(nn,"can't allocate info struct");
	return 0;
}

struct nn_node_ops nn_ops_for_HeatmapMaxKP_f = {
	.execute = heatmap_maxpk_execute,
	.check = heatmap_maxpk_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(2),
};

struct nn_node_ops nn_ops_for_QuantizedHeatmapMaxKP_8 = {
	.execute = heatmap_maxpk_execute_8,
	.check = heatmap_maxpk_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(5),
	.n_outputs = NN_IOCOUNT(4),
};


