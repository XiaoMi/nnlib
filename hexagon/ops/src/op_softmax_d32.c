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
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <quantize.h>
#include <math.h>
#include "hvx_inlines.h"
#include "hvx_mathops.h"

//#define TEST_PERFORMANCE
//
// softmax d32
//
// softmax on qu8 inputs
// defined (over a set of inputs x[0..n-1]) as
//
//       t[i] = exp( beta * x[i])
//       y[i] = t[i]/sum(t[i])
//
// All results are in range 0..1, and have the same ordering as the input values.
// - sum of all outputs is 1.0 (ideally, mathematically..)
// - if one value is distinctly greater than all the others, it will tend to be close to 1.0
//    and the others close to 0.0
//   The amount of margin needed for 'distinctly' depends on beta.
//
// Note that adding a constant amount to all inputs has no effect; and
// scaling the inputs by some factor is equivalent to changing beta; so we  can
// assume for the discussion that x[i] are 0..255  and beta has been thus scaled.
//
// Also, it can be done as
//
//       t[i] = pow2( beta2 * x[i])
//       y[i] = t[i]/sum(t[i])
//
// .. where 'beta2' is beta * (input_range/255.) /ln(2)
//
// When doing the summation, we need to consider that the range of 'pow2' results could be considerable.
// The method below is used for vectorized fixed-point:
//
// (1)  for each i (across depth dimension):
//         - find the largest value xmax across the range
// (2)  for each i across depth dimension:
//         (2a) find xx = beta2*(xmax-x) as a fixed-point value with 16 fractional bits.
//         (2b) find m = pow2(-frac(xx)), in range 1.0 .. 0.5, with 15 fractional bits
//         (2c)  find mp = (m<<16)>>(accshift+int(xx))   (>> by 31 at most, which gives 0).
//		   (2d)  find m[i]= mp >> (16-accshift), keep for step[3]. This is m >>int(xx).
//         (2e)  find the sum of all the m[i].
// (3) find a reciprocal of sum(m[i]) :    recip*sum(m[i]) = (255 << rsh)
// (4)  for each i across depth dimension:
//         (4a)  output = m[i]*recip >> rsh
//
// This operation is repeated on slices of (1,1,4,d), so each slice does 4 separate ops.
// (so, for instance, the 'reciprocal' is vectorized, but only 4 distinct values are
// in the vector).
//
// The 'max' in (a) and sum(m(i])) in (2d) need to be masked to eliminate 'depth padding' lanes
//
//  The operation in step (2d) uses reducing shift, so 'accshift' must be in range 1..16.
//  this means depth must be <= 65536.
//
struct softmax_scaling_parms
{
	int betamul;
	int beta_rsh;
	int accshift;		// ceiling(log2(depth))
};
static inline int
set_scaling_parms( struct softmax_scaling_parms * sparms, float inmin, float inmax, float beta, int depth)
{
	// we will use pow2( scalep2 * x[i])  instead of
	// exp( beta * dequant(x[i]))
	// So  scalep2 = beta * (range/255) / ln(2)
	//
	//
	float scalep2 = (inmax-inmin)* beta * (float)(1.0 / ( 255.0 * 0.69314718055994));
	// find betamul, rsh such that
	//   betamul  =   scalep2 * 2^(rsh+8)
	// and rsh in range 0..31, betamul as large as possible but < 32768
	// (this requires scalep2 < 256, which is a very reasonable constraint)
	// Also, if scalep2 is so small that rsh gets >=31, the beta is effectively 0.
	int scexp = flt_getexp( scalep2);
	int rsh = min_i32(31,7-scexp);
	if( rsh < 0)
		return -1;
	sparms->beta_rsh = rsh;
	sparms->betamul = saturate_i16( roundf_i32( scalep2 * flt_power2(rsh+8)));
	int accshift = max_i32(2,32-Q6_R_cl0_R(depth-1));			// ceiling(log2(depth))
	if( accshift >16)
 		return -1; // depth must be <= 65536
	sparms->accshift = accshift;
	return 0;
}
// MAX_THREADS can only be 1 or 2.
// if 2, we split by h
//
#define MAX_THREADS 2
struct softmax_run_state;
typedef void (*operate_func_fp)( struct softmax_run_state*,uint8_t *, uint8_t const *, int, int16_t *);


struct softmax_run_state {
	struct shape shape;			// shape of operation
	int eff_width;				// width when 'rounded out' to 4 boundaries at each end (d32 version only)
	struct tensor_addressing tin;		// (d32 version only - except 'data')
	struct tensor_addressing tout;		// (d32 version only - except 'data')
	nn_sem_t done_sem;
	struct softmax_scaling_parms  sparms;

	// 'flat version >>>
	int log2_K;			// for 'flat' version only: the smallest value such
						// that depth << log2_K is a multiple of 128; range is 0..7
	int total_dunits;	// total depth units for the whole thing
	int job_dunits;		// dunits done per work unit (last one could be short)
	// <<<
	operate_func_fp operate_func;
	struct softmax_thrinfo {
		struct softmax_run_state *stt; 	// all point to containin struct
		int h0;							// start at this h dimension
		int hcount;						// run this many h
		int16_t * workbuf;
	} thrinfo[MAX_THREADS];
};

static void softmax_d32_run_thread( struct nn_graph * nn, void * tinfov);
static void softmax_flat_run_thread( struct nn_graph * nn, void * tinfov);

static void softmax_operate_ref_function(
		struct softmax_run_state *rstp,
		uint8_t *data_out,						// vec aligned
		uint8_t const *data_in,					// vec aligned
		int height,
		int16_t *workbuf );
static void softmax_operate_hvx_function(
		struct softmax_run_state *rstp,
		uint8_t *data_out,						// vec aligned
		uint8_t const *data_in,					// vec aligned
		int height,
		int16_t *workbuf );
static void softmax_operate_hvx_flat_function(
		struct softmax_run_state *rstp,
		uint8_t *data_out,						// x4 aligned
		uint8_t const *data_in,					// x4 aligned
		int height,
		int16_t *workbuf );
static void softmax_operate_hvx_flat_notx4_function(
		struct softmax_run_state *rstp,
		uint8_t *data_out,						// not aligned
		uint8_t const *data_in,					// not aligned
		int height,
		int16_t *workbuf );


static int softmax_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float beta = (self->n_inputs >= 4)? tensor_get_float(self->inputs[3],0) : 1.0f;

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif


	logmsg(nn,2,"softmax_d32 execute. self=%p ",self);

	struct softmax_run_state rstt;

	if( set_scaling_parms( &rstt.sparms, in_min, in_max, beta, in_tensor->shape.depth) != 0){
		return errlog(nn,"softmax_32: range = %f ... %f, beta = %f: beta too large", in_min, in_max, beta);
	}

	// set up the runstate for input tensor
	rstt.shape = in_tensor->shape;
	rstt.tin = tensor_addressing_d32(in_tensor);

	// set up the output tensor

	int b = in_tensor->shape.batches;
	int h = in_tensor->shape.height;
	int h_pad_before = in_tensor->format.height_pad[0];
	int h_pad_after = in_tensor->format.height_pad[1];
	int w = in_tensor->shape.width;
	int w_pad_before = in_tensor->format.width_pad[0];
	int w_pad_after = in_tensor->format.width_pad[1];
	int d = in_tensor->shape.depth;
	int d_pad_before = in_tensor->format.depth_pad[0];
	int d_pad_after = in_tensor->format.depth_pad[1];
//printf("shape = %d %d %d %d\n", b,h,w,d);
	int use_hvx = 0;
	// currently the hvx code needs d_pad_before = 0.
	if( d_pad_before == 0 && self->node_type == OP_QuantizedSoftmax_8_d32){
		use_hvx = 1;
	}
//use_hvx = 0;//@@@
	if( tensor_out_prepare_padded_d32( out_tensor, b,
			h,  h_pad_before,  h_pad_after,
			w,  w_pad_before,  w_pad_after,
			d,  d_pad_before,  d_pad_after,
			NN_TYPE_QUINT8) != 0 ){
		return errlog(nn,"out too small");
	}

	// set up the rest of the run state
	rstt.operate_func = use_hvx? softmax_operate_hvx_function: softmax_operate_ref_function;

	rstt.tout = tensor_addressing_d32(out_tensor);

	// size of work buffer is full depth * width *uint16
	// first, round out width to *4 boundaries
	int eff_width = (w_pad_before+ w +3)&~3;	// rounded up
	int w_offs = w_pad_before;
	if( w_offs >= 4){	// can reduce on the left by mutiple of 4
		int w_excess = w_offs & ~3;
		w_offs -= w_excess;
		eff_width -= w_excess;
	}
	rstt.tin.data -= w_offs *32;		// round these down to vector boundary
	rstt.tout.data -= w_offs *32;
	rstt.eff_width = eff_width;

	nn_sem_init( &rstt.done_sem, 0);

	int nthreads = 1;
	int h_thread_0 = h;	// rows done by first thread
#if MAX_THREADS >= 2
	if( h >= 3){
		nthreads = 2;
		h_thread_0 = (h+1)>>1;
	}
#endif
	int work_buf_per_thread = rstt.tin.nd32 * 32 * eff_width * sizeof(int16_t);
	int16_t * work_buffer = (int16_t*)nn->scratch;
	if( work_buf_per_thread * nthreads > nn->scratch_size ){
		if(work_buf_per_thread > nn->scratch_size ){
			return errlog(nn, "softmax_d32: scratch too small");
		}
		nthreads = 1;
		h_thread_0 = h;
	}
	rstt.thrinfo[0].stt = &rstt;
	rstt.thrinfo[0].workbuf = work_buffer;
	rstt.thrinfo[0].h0 = 0;
	rstt.thrinfo[0].hcount = h_thread_0;
	nn_os_work_for_vector(nn, softmax_d32_run_thread , &rstt.thrinfo[0]);
	if( nthreads > 1){
		rstt.thrinfo[1].stt = &rstt;
		rstt.thrinfo[1].workbuf = work_buffer + work_buf_per_thread;
		rstt.thrinfo[1].h0 = h_thread_0;
		rstt.thrinfo[1].hcount = h-h_thread_0;
		nn_os_work_for_vector(nn, softmax_d32_run_thread , &rstt.thrinfo[1]);
	}

	tensor_set_single_float( out_min_tensor, 0.0f);
	tensor_set_single_float( out_max_tensor, 1.0f);

	for( int i =0; i < nthreads; i++){
		nn_sem_wait( & rstt.done_sem);
	}
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("softmax_d32 %s cycles = %d (elements = %d)\n",
			use_hvx?"hvx": "ref",(end_time-start_time),
			(int)tensor_element_count(in_tensor));
#endif
	logmsg(nn,2,"softmax_d32 done. self=%p ",self);

	return 0;
}

static void
softmax_d32_run_thread( struct nn_graph * nn, void * tinfov)
{
	struct softmax_thrinfo *thrinfop = (struct softmax_thrinfo *)tinfov;
	struct softmax_run_state *rstp = (struct softmax_run_state *)thrinfop->stt;

	int  batch_index;
	int16_t * wbuf= thrinfop->workbuf;
	int ht = thrinfop->hcount;

	int pfheight = ht * rstp->tin.nd32;	// these vars are so that the ..
	int pfwid = rstp->eff_width * 32;	// .. prefetch 'shape' value can be hoisted..
	int pfstride = rstp->tin.d32_stride; // .. out of the loop

	operate_func_fp funcp = rstp->operate_func;
	// offset pointers vertically
	uint8_t const *in_ptr = rstp->tin.data + thrinfop->h0 * rstp->tin.height_stride;
	uint8_t *out_ptr = rstp->tout.data + thrinfop->h0 * rstp->tout.height_stride;

	for( batch_index = 0; batch_index < rstp->shape.batches; batch_index++){
		l2pref( in_ptr, pfheight, pfwid, pfstride );
		(*funcp)( rstp, out_ptr, in_ptr, ht, wbuf);
		in_ptr += rstp->tin.batch_stride;
		out_ptr += rstp->tout.batch_stride;
	}
	nn_sem_post( & rstp->done_sem);
}



//
// 'frac' is an integer in range 0..0x7FFF, representing 0..1.0; returns
// pow2(-x) with 15 frac posns (in range 0x7fff .. 0x4000)


static inline int ref_powm2_op( unsigned frac )
{
	int section = (frac >> 11)& 0x0F;
	int c0 = lut_Log2_and_Pow2[3*64+2*section];
	int c1 = lut_Log2_and_Pow2[4*64+2*section];
	int c2 = lut_Log2_and_Pow2[5*64+2*section];
	int x = frac & 0x7FF;	// bottom 11 bits
	int k = Q6_R_vaddh_RR_sat( c1, Q6_R_vmpyh_RR_s1_rnd_sat( x, c2 ));
	return (int16_t)Q6_R_vaddh_RR_sat( c0, Q6_R_vmpyh_RR_s1_rnd_sat( x, k ));
}
//
// 'reference' op
// This is designed to give the same result as the hvx version.
// ** This is no longer true, since I don't have bit-accurate
// scalar references for new pow2 and recip operations. It will
// be close, though **
//
static void
softmax_operate_ref_function(
		struct softmax_run_state *rstp,
		uint8_t *data_out,						// vec aligned
		uint8_t const *data_in,					// vec aligned
		int height,				// # of rows processed (could be < rstp->shape.height)
		int16_t *workbuf )		// vec aligned, width*depth_total*int16
{
	int width = rstp->eff_width;
	int in_height_stride = rstp->tin.height_stride;
	int in_d32_stride = rstp->tin.d32_stride;
	int out_height_stride = rstp->tout.height_stride;
	int out_d32_stride = rstp->tout.d32_stride;
	int d0 = rstp->tin.d0;
	int depth = rstp->shape.depth;
	int nd32 = rstp->tin.nd32;
	int accshift = rstp->sparms.accshift;
	int bscale = rstp->sparms.betamul;
	int brsh =  rstp->sparms.beta_rsh;

	if(  width <= 0 || nd32 < 0) return; // (eliminates all the nd32 > 0 tests below)

	for( int ih = 0; ih < height; ih++){
		for(int iw = 0; iw < width; iw++ ){
			uint8_t const *pin0 = data_in + ih*in_height_stride + iw *32;
			uint8_t *pout = data_out + ih*out_height_stride + iw *32;


			// find the max of all the input values
			uint8_t xmax = 0;
			int dstart = d0;
			int dk = d0+depth;
			uint8_t const *pin = pin0;
			for( int id32 = 0; id32 < nd32; id32++){
				int dend = min_i32(32,dk);
				for( int j = dstart; j < dend; j++ )
					xmax = max_i32( xmax, pin[j]);
				dstart = 0;
				pin += in_d32_stride;
				dk -= 32;
			}

			// now, find all the pow2(-(xmax-x)*beta2) values...
			int16_t *ptmp = workbuf;
			int msum = 0;
			dstart = d0;
			dk = d0+depth;
			pin = pin0;
			for( int id32 = 0; id32 < nd32; id32++){
				int dend = min_i32(32,dk);
				for( int j = dstart; j < dend; j++ ){
					int xscaled = (xmax - pin[j])*256 * bscale >> brsh;
					int xfrac = (uint16_t)xscaled;	// lower 16 bits
					int xexp = xscaled >> 16;		// upper 16 bits
					int mval;
					int rshn = xexp + accshift;		// shift used below.
					if( rshn > 31){
						mval = 0;
					}else{
						// find mant = 2*^-xfrac
						int mant = ref_powm2_op( xfrac>>1);	// mant with 15 fractional bits
						mval = (mant<<16) >> rshn;  // mant*2^-xexp with (31-accshift) frac bits
						msum += mval;
						// >> to 15 with round/sat, to keep for later. Note: *2 will not overflow
						mval = (mval*2 >> (16-accshift));	// 16 frac bits
						mval = saturate_i16(  (mval+1)>>1);	// round it, sat to 15
					}
					*ptmp ++ = mval;
				}
				dstart = 0;
				pin += in_d32_stride;
				dk -= 32;
			}

			// find reciprocal
			// we want recip_m*msum = (255 << rsh)
			// with recip_m being of reasonable magnitude...

			int nx = Q6_R_cl0_R( msum);	//should be at most 17, at least 7

			int m_norm = (unsigned)(msum << (nx-1)) >> 16;	// value in range 0x4000 .. 0x7fff
			// divide (255<<6) by m_norm; result will be roughly in range 16320 .. 32640
			//int recip_m = ref_fracdivide_10bit_Rh_RhRh( 255 << 6, m_norm);
			// (don't have a bit-accurate scalar ref for new recip16, so use this):
			int recip_m = ((255u<<21) + (m_norm>>1))/(unsigned)m_norm;	// 16320..32640

			//int rsh = 38-max_i32(nx,7);			// 21..31

			int rsh = min_i32(31,22+accshift-nx);	// 21 ..31

			// large values of m_norm occur when you have a large depth and all values roughly
			// the same. In this case rsh could be quite large (e.g. a value of nx = 7
			// can occur with >512 accs of 32767; in this case rsh = 31 and all values will actually
			// become zero in the output (which is actually ok since the 'proper' outputs
			// are all  < 1/512, and we are rounding to 8 bits)

			// in the hvx the rsh will be done in 3 stages:
			//   (1) >> by rsh-18, variable per lane (this is in range 3..13)
			//   (2) >> by 12 and conv/sat to h
			//   (3) >> by  6 and round/sat to ub

			// output results
			ptmp = workbuf;

			dstart = d0;
			dk = d0+depth;
			int rbias = (1<<rsh)>>1;
			for( int id32 = 0; id32 < nd32; id32++){
				int dend = min_i32(32,dk);
				for( int j = dstart; j < dend; j++ ){
					int y = (*ptmp++ * recip_m + rbias) >> rsh;
					pout[j] = saturate_u8(y);
				}
				dstart = 0;
				pout += out_d32_stride;
				dk -= 32;
			}
		}
	}
}

static int softmax_d2_flat_exec(struct nn_node *self, struct nn_graph *nn);
static int softmax_larged_flat_exec(struct nn_node *self, struct nn_graph *nn);

// timing as of 11-Dec-2018
// pmus on SDM845
//
//  64 x d= 1000  1500  3000 3700 7100
//  normal   43    58    94  110   214  (kc/whole op)
//  larged   51    56    74   83   130
// .. based on this, break-even is around d=1380
//

#define MIN_DEPTH_FOR_LARGED 1408
#define MIN_SIZE_FOR_LARGED (1408*4)

static int softmax_flat_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	int d = in_tensor->shape.depth;
	if( d <=2){
		if(d==2) return softmax_d2_flat_exec(self,nn);
		if(d==0) return errlog(nn,"d=0 not supported");
	}else if( d >=MIN_DEPTH_FOR_LARGED && in_tensor->data_size >= MIN_SIZE_FOR_LARGED){
		return softmax_larged_flat_exec(self,nn);
	}
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float beta = (self->n_inputs >= 4)? tensor_get_float(self->inputs[3],0) : 1.0f;

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif


	logmsg(nn,2,"softmax_8 execute. self=%p ",self);

	struct softmax_run_state rstt;

	if( d > 1 && set_scaling_parms( &rstt.sparms, in_min, in_max, beta, in_tensor->shape.depth) != 0){
		return errlog(nn,"softmax_8: range = %f ... %f, beta = %f: beta too large", in_min, in_max, beta);
	}

	// set up the runstate for input tensor
	rstt.shape = in_tensor->shape;
	rstt.tin.data = in_tensor->data;

	if( tensor_out_prepare_normal_fromshape( out_tensor, &rstt.shape, NN_TYPE_QUINT8) != 0 ){
		return errlog(nn,"out too small");
	}
	tensor_set_single_float( out_min_tensor, 0.0f);
	tensor_set_single_float( out_max_tensor, 1.0f);
	// handle d=1 case
	if( d == 1 ){
		memset( out_tensor->data, 0xFF, out_tensor->data_size);
		return 0;
	}
	rstt.tout.data = out_tensor->data;
	int depth = rstt.shape.depth;
	int gcf = (depth|128) & ~((depth|128)-1);
	// gcf is GCF (depth,128); power of 2 in range 1..128
	// if gcf < 4, we need to use another run loop which uses K=1.
	rstt.log2_K = 7-Q6_R_ct0_R(gcf);	// 0..5  (if gcf>=4)
	rstt.operate_func=softmax_operate_hvx_flat_function;
	if( gcf < 4){
		gcf=1;
		rstt.log2_K = 0;
		rstt.operate_func=softmax_operate_hvx_flat_notx4_function;
	}
	int depth_units = rstt.shape.batches * rstt.shape.height * rstt.shape.width;
	// work buffer size:
	// - find the largest # of vectors needed to contain a depth, at any possible alignment;
	// - multiply by 4 vectors (we are storing 16-bit intermediates for 2 rows at once)
	// note that (128-gcf), 0..127, is an upper bound on where a depth unit can start within a vector.
	// this is oversize for gcf=1 but that's ok
	unsigned vecs_across = ( depth +127+ 128-gcf) >> 7;
	int work_buf_per_thread = vecs_across * 4*128;

	rstt.total_dunits = depth_units;

	nn_sem_init( &rstt.done_sem, 0);

	int nthreads = 1;
	int n_thread_0 = depth_units;		// units done by first thread
	int job_size_limit = 128*1024;
	//
	// 'job_dunits' is the job size in units of depth; each run thread
	// does 'job_dunits' at  a time.
	// we'd like to have job_dunits be a multiple of 2*gcf, so that
	// each one splits efficiently into operations of 2*n rows with
	// aligned stride. Also, the prefetch for a job unit mustn't
	// be too huge. If we can't have 2*gcf - because there are not
	// that many, or because it's too much to prefetch- then we
	// use '1' as the minimum.
	// note: if the total dunits is 2*gcf, we are better to have one
	// job do that many than have two do half each, because each of the
	// two will actually be running at 1/2 efficiency.
	//
	// Thus we have various, possibly conflicting goals in setting job_dunits
	//  (a) job_dunits should be a multiple of 2*gcf; if that's too large,
	//    then make it anything;
	//  (b) job_dunits should be such that job_dunits*depth is large as possible
	//    while <= job_size_limit
	//  (c) If possible job_dunits should be small enough so that there's work
	//   for two threads. But don't break (a) just to meet (c).
	//
	int job_dunits = 2*gcf;	 // 2, ..256
	if (depth_units == 1){		// common case.
		job_dunits = 1;
	}else{
		if( depth_units < job_dunits || job_dunits * depth > job_size_limit)
			job_dunits = 1;
		unsigned job_dunits_bytes = job_dunits * depth;
		if( job_size_limit >=2*job_dunits_bytes){
			// most we can do in prefetch limit
			unsigned mul = job_size_limit/job_dunits_bytes;	// >= 2
			// if that's more than half the work, do half the work instead.
			if( job_dunits*2*mul > depth_units ){
				if( depth_units >= 4*job_dunits ){
					mul = depth_units/(2u*(unsigned)job_dunits);
				}else{
					mul =1;
				}
			}
			job_dunits *= mul;
		}
	}
	rstt.job_dunits = job_dunits;
#if MAX_THREADS >= 2
	if( depth_units > job_dunits){
		nthreads = 2;
		unsigned fulljobs = depth_units/(unsigned)job_dunits;
		unsigned cut = (fulljobs+1)>>1;
		n_thread_0 = cut * job_dunits;
	}
#endif
	logmsg(nn,2,"thread 0 does %d of %d; thread 1 does %d; chunk size is %d",
		n_thread_0, depth_units, (depth_units-n_thread_0), job_dunits);

	int16_t * work_buffer = (int16_t*)nn->scratch;
	if( work_buf_per_thread * nthreads > nn->scratch_size ){
		if(work_buf_per_thread > nn->scratch_size ){
			return errlog(nn, "softmax_flat: scratch too small");
		}
		nthreads = 1;
		n_thread_0 = depth_units;
	}
	rstt.thrinfo[0].stt = &rstt;
	rstt.thrinfo[0].workbuf = work_buffer;
	rstt.thrinfo[0].h0 = 0;
	rstt.thrinfo[0].hcount = n_thread_0;
	nn_os_work_for_vector(nn, softmax_flat_run_thread , &rstt.thrinfo[0]);
	if( nthreads > 1){
		rstt.thrinfo[1].stt = &rstt;
		rstt.thrinfo[1].workbuf = work_buffer + work_buf_per_thread;
		rstt.thrinfo[1].h0 = n_thread_0;
		rstt.thrinfo[1].hcount = depth_units-n_thread_0;
		nn_os_work_for_vector(nn, softmax_flat_run_thread , &rstt.thrinfo[1]);
	}


	for( int i =0; i < nthreads; i++){
		nn_sem_wait( & rstt.done_sem);
	}
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("softmax_flat cycles = %d (elements = %d)\n",
			(end_time-start_time),
			(int)tensor_element_count(in_tensor));
#endif
	logmsg(nn,2,"softmax_8 done. self=%p ",self);

	return 0;
}


static void
softmax_flat_run_thread( struct nn_graph * nn, void * tinfov)
{
	struct softmax_thrinfo *thrinfop = (struct softmax_thrinfo *)tinfov;
	struct softmax_run_state *rstp = (struct softmax_run_state *)thrinfop->stt;


	int depth = rstp->shape.depth;
	int dep0 = thrinfop->h0;
	int dunits_remain = thrinfop->hcount;
	operate_func_fp operate_func = rstp->operate_func;
	int16_t * workbuf = thrinfop->workbuf;

	// get pointer to the extent we are working on
	uint8_t const * inptr = rstp->tin.data + dep0*depth;
	uint8_t * outptr = rstp->tout.data  + dep0*depth;

	// do it in chunks of 'nchunk' depth units (with possible remnant at the end)
	//
	int nchunk = rstp->job_dunits;
	int log2k = rstp->log2_K;
	int K = 1<<log2k;
	while( dunits_remain > 0 ){
		int dunits_now = min_i32( nchunk, dunits_remain);		// do this many depths  1 .. nchunk
		int pfheight = (dunits_now*depth+127)>>7;
		l2pref( inptr, pfheight, 128, 128 );

		int slcops = min_i32( dunits_now, K);			// number of slices to do
		int dunits_now_pad = dunits_now + K-1;				// used to find slice sizes
		//printf("chunk of %d at %p: slcops =%d\n", dunits_now, inptr,slcops);
		for( int ik = 0; ik  < slcops; ik++ ){
			// find the slice height
			int slice_ht = (dunits_now_pad-ik)>>log2k;	// # of rows, starting at ik, with stride of K
			//printf("  in = %p; rows= %d; pitch = %d<<%d\n",inptr + depth*ik, slice_ht,  depth , log2k );
			(*operate_func)(rstp, outptr + depth*ik, inptr + depth*ik, slice_ht, workbuf);
		}
		dunits_remain -= dunits_now;
		inptr += depth * nchunk;			// move to next chunk
		outptr += depth * nchunk;
	}

	nn_sem_post( & rstp->done_sem);
}

#if __HEXAGON_ARCH__ >= 65
// ====================================================
// FUNCTION: (2**-~x)     y(0) = 0.5,  y(0.5) = 0.7071, y(1) = 1
// Order:2; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 15
// Peak Error: 6.9827e-05  Rms Error: 2.4918e-05   Mean Error: 1.2308e-07
//      32769   22652    4300
//      32876   21816    5119
//      33371   19857    6087
//      34675   16404    7230

static inline HVX_Vector
pow2_v65_poly( HVX_Vector vx )
{
    // input is 0..0xffff representing 0.0  .. 1.0
	vx = Q6_V_vnot_V(vx);	// poly built for this
    HVX_Vector p;
    p = Q6_Vh_vlut4_VuhPh( vx, 0x1C3E17C713FF10CCull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat( p, vx, 0x40144D915538587Cull );
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat( p, vx, 0x8773825B806C8001ull );
    return p;  // signed result, 15 fractional bits
}
#if 0

// ====================================================
// FUNCTION: (2**-x)-1/2     y(0) = 0.5,  y(0.5) = 0.2071, y(1) = 0
// Order:3; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 15
// Peak Error: 4.0413e-05  Rms Error: 1.1650e-05   Mean Error: 1.6034e-06
//      32769  -45423   15704   -1676
//      32760  -45319   15300   -1401
//      32703  -44987   14640   -1180
//      32538  -44333   13772    -987

static inline HVX_Vector
pow2_v65_poly( HVX_Vector vx )
{
    // input is 0..0xffff representing 0.0  .. 1.0
    HVX_Vector p;
    p = Q6_Vh_vlut4_VuhPh( vx, 0xFC25FB64FA87F974ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat( p, vx, 0x35CC39303BC43D58ull );
    p = Q6_Vh_vmps_VhVhVuhPuh_sat( p, vx, 0xAD2DAFBBB107B16Full );
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat( p, vx, 0x7F1A7FBF7FF88001ull );
	p = Q6_Vh_vadd_VhVh_sat( p,Q6_Vh_vsplat_R( 0x4000));
    return p;  // signed result, 15 fractional bits
}
#endif
#endif


////////////////////////////// HVX CODE ///////////////////////////////////
// these are used to create fake dependencies so that the compiler won't hoist
// invariants out of loops (and run out of vector registers).
//       FAKE_DEP_RR(a,b) is the same as 'a' but appears to depend on b; by using a variable
//  that changes in the loop as 'b', we can force values depending on a (such as a vsplat) to
// be done on every loop.
//
#if defined(__hexagon__)
#define FAKE_DEP_RR( VAL1, VAL2) ({ typeof(VAL1) t__1 = (VAL1); typeof(VAL2) t__2 = (VAL2);\
   asm ("/* %0 = FAKE_DEP_RR(%1, %2) */" : "=r"(t__1): "0"(t__1), "r"(t__2)); t__1; })
#define FAKE_DEP_VR( VAL1, VAL2) ({ HVX_Vector t__1 = (VAL1); typeof(VAL2) t__2 = (VAL2);\
   asm ("/* %0 = FAKE_DEP_VR(%1, %2) */" : "=v"(t__1): "0"(t__1), "r"(t__2)); t__1; })
#else
#define FAKE_DEP_RR( VAL1, VAL2) (VAL1)
#define FAKE_DEP_VR( VAL1, VAL2) (VAL1)
#endif
// the 'pow2( scl*x)' portion of the operation:
//  - vin is 128 input bytes
//  - vinmax is 'maxin' values (these are replicated within each 32-bit section)
//  - betamul and beta_rsh are the beta scaling parms.
//
// return is 128 16-bit values, pow2( -(xmax-x)*beta)
//  and 128 32-bit values, which are the same thing <<(16-accshift)
//

struct pow2func_result
{
	HVX_Vector exp16[2];	// exponents in 16-bit form, 15 fractional bits
	HVX_Vector exp32[4];	// exponents in 32-bit form, 31-accshift fractional bits
};

static inline
struct pow2func_result
hvx_pow2_func( HVX_Vector vin, HVX_Vector vinmax, int betamul, int beta_rsh,  int accshift, int dummy_invariant)
{
	betamul = Q6_R_combine_RlRl( betamul, betamul);

	// set up constants. This will be hoisted only out of loops where dummy_invariant
	// is invariant.
#if __HEXAGON_ARCH__ >= 65
	HVX_Vector vconsth_001F = Q6_Vh_vsplat_R( FAKE_DEP_RR(0x001F,dummy_invariant));
#else
	HVX_Vector vconsth_001F,vconsth_0FFF;
	{
		HVX_Vector tmp = Q6_V_vsplat_R( FAKE_DEP_RR(0x0FFF001F,dummy_invariant));
		HVX_VectorPair tshuf = Q6_Wh_vshuffoe_VhVh( tmp,tmp );
		vconsth_001F = Q6_V_lo_W( tshuf );
		vconsth_0FFF = Q6_V_hi_W( tshuf );
	}
#endif

	// find vinmax-vin
	// (saturation is not needed, but won't hurt)
	//
	HVX_Vector vdiff = Q6_Vub_vsub_VubVub_sat( vinmax, vin );
	// now we need to multiply this unsigned value by 256, and by the 16-bit 'betamul', and get a 32-bit
	// result.
	HVX_VectorPair vinh = Q6_Wb_vshuffoe_VbVb(  vdiff, Q6_V_vzero());	// *256 (u16 now)
	HVX_VectorPair vprod0 = Q6_Wuw_vmpy_VuhRuh( Q6_V_lo_W(vinh), betamul);
	HVX_VectorPair vprod1 = Q6_Wuw_vmpy_VuhRuh( Q6_V_hi_W(vinh), betamul);

	//now shift each right by beta_rsh

	HVX_Vector vprsh_00 = Q6_Vw_vasr_VwR( Q6_V_lo_W(vprod0), beta_rsh);
	HVX_Vector vprsh_10 = Q6_Vw_vasr_VwR( Q6_V_hi_W(vprod0), beta_rsh);
	HVX_Vector vprsh_01 = Q6_Vw_vasr_VwR( Q6_V_lo_W(vprod1), beta_rsh);
	HVX_Vector vprsh_11 = Q6_Vw_vasr_VwR( Q6_V_hi_W(vprod1), beta_rsh);

	// the integer part is in the upper 16 bits, and frac in lower
	HVX_VectorPair shtmp = Q6_Wh_vshuffoe_VhVh( vprsh_10, vprsh_00);
	HVX_Vector frac0 = Q6_V_lo_W(shtmp);
	HVX_Vector int0 = Q6_V_hi_W(shtmp);
	shtmp = Q6_Wh_vshuffoe_VhVh( vprsh_11, vprsh_01);
	HVX_Vector frac1 = Q6_V_lo_W(shtmp);
	HVX_Vector int1 = Q6_V_hi_W(shtmp);

#if __HEXAGON_ARCH__ >= 65
	HVX_Vector pow2_0 = pow2_v65_poly(frac0);
	HVX_Vector pow2_1 = pow2_v65_poly(frac1);
#else
	// now, we need to pow2(-x) on the fractional parts.
	// start by constructing a table lookup index from bits 15..12 of each
	// value (need to move to the lsbs of each byte)
	HVX_Vector tindex = Q6_Vb_vshuffo_VbVb( frac1, frac0);	// get upper bytes
	HVX_Vector const *lut_ptr = (HVX_Vector const *) lut_Log2_and_Pow2  + 3;
#if __HEXAGON_ARCH__ < 62
	tindex = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(tindex,4), q6op_Vb_vsplat_R(0x0F));
	HVX_VectorPair co2 =  Q6_Wh_vlut16_VbVhR( tindex, lut_ptr[2], 0);
	HVX_VectorPair co1 =  Q6_Wh_vlut16_VbVhR( tindex, lut_ptr[1], 0);
	HVX_VectorPair co0 =  Q6_Wh_vlut16_VbVhR( tindex, lut_ptr[0], 0);
#else
	tindex = Q6_Vuh_vlsr_VuhR(tindex,4);
	HVX_VectorPair co2 =  Q6_Wh_vlut16_VbVhR_nomatch( tindex, lut_ptr[2], 0);
	HVX_VectorPair co1 =  Q6_Wh_vlut16_VbVhR_nomatch( tindex, lut_ptr[1], 0);
	HVX_VectorPair co0 =  Q6_Wh_vlut16_VbVhR_nomatch( tindex, lut_ptr[0], 0);
#endif
	// the 'x' for the 2nd order poly is bits 10..1 of the frac, >>1
	frac0 = Q6_Vh_vavg_VhVh( Q6_V_vand_VV( frac0,vconsth_0FFF), Q6_V_vzero());
	frac1 = Q6_Vh_vavg_VhVh( Q6_V_vand_VV( frac1,vconsth_0FFF), Q6_V_vzero());
	// evaluate the poly in all 128 lanes
	HVX_Vector tmp = Q6_Vh_vadd_VhVh_sat( Q6_V_lo_W(co1), Q6_Vh_vmpy_VhVh_s1_rnd_sat( Q6_V_lo_W(co2), frac0));
	HVX_Vector pow2_0 = Q6_Vh_vadd_VhVh_sat( Q6_V_lo_W(co0), Q6_Vh_vmpy_VhVh_s1_rnd_sat( tmp, frac0));
	tmp = Q6_Vh_vadd_VhVh_sat( Q6_V_hi_W(co1), Q6_Vh_vmpy_VhVh_s1_rnd_sat( Q6_V_hi_W(co2), frac1));
	HVX_Vector pow2_1 = Q6_Vh_vadd_VhVh_sat( Q6_V_hi_W(co0), Q6_Vh_vmpy_VhVh_s1_rnd_sat( tmp, frac1));
#endif
	// To make the 32 bit form: multiply by 1<<(16-accshift)
	// (this will not overflow to sign)

	HVX_VectorPair m32_02 = Q6_Wuw_vmpy_VuhRuh( pow2_0, 0x00010001 <<(16-accshift));
	HVX_VectorPair m32_13 = Q6_Wuw_vmpy_VuhRuh( pow2_1, 0x00010001 <<(16-accshift));

	// clip the shift amount to a max of 31
	int0 = Q6_Vh_vmin_VhVh( int0, vconsth_001F );
	int1 = Q6_Vh_vmin_VhVh( int1, vconsth_001F );
	// right shift by int0,int1 amounts.
	HVX_Vector pow32_0 = Q6_Vw_vasr_VwVw( Q6_V_lo_W(m32_02), int0 );
	HVX_Vector pow32_1 = Q6_Vw_vasr_VwVw( Q6_V_lo_W(m32_13), int1 );
	// for the odd lanes, we need to get the shift from the upper half
	int0 = Q6_Vh_vshuffo_VhVh( int0, int0);
	int1 = Q6_Vh_vshuffo_VhVh( int1, int1);
	HVX_Vector pow32_2 = Q6_Vw_vasr_VwVw( Q6_V_hi_W(m32_02), int0 );
	HVX_Vector pow32_3 = Q6_Vw_vasr_VwVw( Q6_V_hi_W(m32_13), int1 );


	struct pow2func_result result;
	// make the 16 bit result by packing those down.
	result.exp16[0] = Q6_Vh_vasr_VwVwR_rnd_sat( pow32_2, pow32_0, (16-accshift));
	result.exp16[1] = Q6_Vh_vasr_VwVwR_rnd_sat( pow32_3, pow32_1, (16-accshift));

	result.exp32[0] = pow32_0;
	result.exp32[1] = pow32_1;
	result.exp32[2] = pow32_2;
	result.exp32[3] = pow32_3;

	return result;
}


//
// 'final scaling' operation
// m0,m1 contain the output from hvx_pow2_func ( i128 x i16, to be scaled)
// recip_mant contains the 'mantissa' of the reciprocal (  4  x {32 x i16} }
// recip_mant contains the shift for the reciprocal (  4  x {16 x i32} }
// (within each quadrant of the recip_mant and recip_sh, all va are the same.
//  And the upper half of each i32 in recip_sh is garbage, since it's only
// needed for a w shift count).
//
// return is 128 x u8, the scaled outputs
//
// 'recip_sh' is in the range 3..13, so >> by recip_sh
// can also be done by Q6_Vw_vmpye_VwVuh( x, 65536 >> recip_sh)
// Since this is used in a loop where recip_sh is invariant, and since those loops
// are otherwise highly shift-bound, we do half of those shifts with the mpye.
//
static inline HVX_Vector
final_scaling_op( HVX_Vector m0, HVX_Vector m1 , HVX_Vector recip_mant,  HVX_Vector recip_sh )
{
	// this operation will be hoisted out of loops where recip_sh is invariant
	HVX_Vector sh_factor = Q6_Vw_vasr_VwVw(  Q6_V_vsplat_R(0x10000), recip_sh);	// 0x10000 >> recip_sh

	HVX_VectorPair prod0 = Q6_Ww_vmpy_VhVh( m0, recip_mant);	// full 32 bit products
	HVX_VectorPair prod1 = Q6_Ww_vmpy_VhVh( m1, recip_mant);

	// now do the variable shift, and a fixed shift of 12, and saturate to 8.

	HVX_Vector scaled0 = Q6_Vh_vasr_VwVwR_sat(
			Q6_Vw_vmpye_VwVuh( Q6_V_hi_W(prod0), sh_factor ),	// >> recip_sh
			Q6_Vw_vasr_VwVw( Q6_V_lo_W(prod0), recip_sh ), 12);
	HVX_Vector scaled1 = Q6_Vh_vasr_VwVwR_sat(
			Q6_Vw_vmpye_VwVuh( Q6_V_hi_W(prod1), sh_factor ),
			Q6_Vw_vasr_VwVw( Q6_V_lo_W(prod1), recip_sh ), 12);

	// combine those to a u8 with >>6
	return Q6_Vub_vasr_VhVhR_rnd_sat( scaled1, scaled0, 6);
}

static void
softmax_operate_hvx_function(
		struct softmax_run_state *rstp,
		uint8_t *data_out,						// vec aligned
		uint8_t const *data_in,					// vec aligned
		int height,				// # of rows processed (could be < rstp->shape.height)
		int16_t *workbuf )
{
	int wvec = rstp->eff_width >>2;				// number of vecs across.
	int in_height_stride = rstp->tin.height_stride;
	int in_d32_stride = rstp->tin.d32_stride;
	int out_height_stride = rstp->tout.height_stride;
	int out_d32_stride = rstp->tout.d32_stride;
	int d0 = rstp->tin.d0;
	int depth = rstp->shape.depth;
	int nd32 = rstp->tin.nd32;
	// currently this supports only depth padding at the end (d0 assumed to be 0)
	HVX_Vector final_mask_ub;
	{
		int  dlenx = ((d0 + depth-1)&31)+  1;	// 1..32; number in last vector
		HVX_VectorPred qfinalmask =  hvx_make_d32_mask( dlenx);	  // 0..dlenx-1 in each slice
		final_mask_ub = Q6_V_vand_QR( qfinalmask, -1);	// transform to vector
	}
	// adapt mask to even and odd h lane patterns
	//            h0   h1
	//  00 00     0    0		So h0 is cmp.uh val > 0)
	//  FF 00     1    0           h1 is cmp.h( val < 0)
	//  FF FF     1    1
	//
	HVX_VectorPred qfinalmask_h0 = Q6_Q_vcmp_gt_VuhVuh( final_mask_ub, Q6_V_vzero());
	HVX_VectorPred qfinalmask_h1 = Q6_Q_vcmp_gt_VhVh(  Q6_V_vzero(),final_mask_ub);

	int bscale = rstp->sparms.betamul;
	int brsh =  rstp->sparms.beta_rsh;
	int accshift = rstp->sparms.accshift;

	if(  wvec <= 0 || nd32 < 0) return; // (eliminates all the nd32 > 0 tests below)

	for( int ih = 0; ih < height; ih++){
		for(int iw = 0; iw < wvec; iw++ ){
			uint8_t const *pin0 = data_in + ih*in_height_stride +iw *128;
			uint8_t *pout = data_out + ih*out_height_stride + iw *128;

			// find max of all input values at each w pos.
			HVX_Vector xmax = Q6_V_vzero();
			HVX_Vector xmnext = *(HVX_Vector const *)pin0;

			for( int i = 1; i < nd32; i++){
				xmax = Q6_Vub_vmax_VubVub( xmax, xmnext);
				xmnext = *(HVX_Vector const *)( pin0 + i*in_d32_stride);
			}
			// trim padding bytes from last depth slice
			xmnext = Q6_V_vand_VV( xmnext, final_mask_ub);
			xmax = Q6_Vub_vmax_VubVub( xmax,xmnext);

			// ok, now propagate them laterally within each group of 32
			xmax = reduce_in_quadrants_vmax_Vub(xmax);

			// now find all the pow2( -beta*(maxx-x)) and store them at
			// 'workbuf', also sum them.
			HVX_Vector msum = Q6_V_vzero();
			HVX_Vector *wbuf = (HVX_Vector*)workbuf;
			struct pow2func_result p2res;
			p2res.exp32[0] = Q6_V_vzero();
			p2res.exp32[1] = Q6_V_vzero();
			p2res.exp32[2] = Q6_V_vzero();
			p2res.exp32[3] = Q6_V_vzero();

			// accumulation of exp32 vals to msum is done at the top of the loop,
			// so we can mask the final acc.

			for( int  i = 0; i < nd32; i++ ){
				msum = Q6_Vw_vadd_VwVw( msum, Q6_Vw_vadd_VwVw( p2res.exp32[0],p2res.exp32[2]));
				msum = Q6_Vw_vadd_VwVw( msum, Q6_Vw_vadd_VwVw( p2res.exp32[1],p2res.exp32[3]));
				HVX_Vector x = *(HVX_Vector const *)( pin0 + i*in_d32_stride);
				p2res = hvx_pow2_func( x, xmax, bscale, brsh, accshift, 0);
				wbuf[0] = p2res.exp16[0];		// save for later
				wbuf[1] = p2res.exp16[1];
				wbuf += 2;
			}
			// add the last set..
			// but first do right side depth masking, while adding m0 & m1
			// We shuffle the 0/2 and 1/3 together to use the h masks.
			{
				HVX_VectorPair sh02 = Q6_Wh_vshuffoe_VhVh( p2res.exp32[2], p2res.exp32[0]);
				HVX_VectorPair sh13 = Q6_Wh_vshuffoe_VhVh( p2res.exp32[3], p2res.exp32[1]);
				sh02 = Q6_Wh_vshuffoe_VhVh(
						q6op_V_vand_QV( qfinalmask_h0, Q6_V_hi_W(sh02)),
						q6op_V_vand_QV( qfinalmask_h0, Q6_V_lo_W(sh02)));
				sh13 = Q6_Wh_vshuffoe_VhVh(
						q6op_V_vand_QV( qfinalmask_h1, Q6_V_hi_W(sh13)),
						q6op_V_vand_QV( qfinalmask_h1, Q6_V_lo_W(sh13)));
				sh02 = Q6_Ww_vadd_WwWw( sh02,sh13);
				msum = Q6_Vw_vadd_VwVw( msum, Q6_Vw_vadd_VwVw( Q6_V_hi_W(sh02),Q6_V_lo_W(sh02) ));
			}
			// OK, now have 4 { 8 x i32 } sums. Need h sums within those groups
			//
			msum = reduce_in_quadrants_vadd_Vw( msum );

			// now, we really only have 4 values in the 'msum' vector, but 8 copies
			// of each. We need to find reciprocals.
			HVX_Vector normsh =  Q6_Vw_vnormamt_Vw(msum);		// at most 10; and >= 0
			msum = Q6_Vw_vasl_VwVw( msum, normsh);				// each is now '01' in the upper 2 bits
			msum = Q6_Vh_vshuffo_VhVh(msum,msum);				// discard 16 lower bits; dup the upper
			// (255/256) / msum
			HVX_Vector recip_mant = hvx_recip16_inline( msum ,10 );
			recip_mant = Q6_Vh_vmpy_VhRh_s1_rnd_sat( recip_mant, 0x10001 * (255<<7));

			// the variable rsh will be 19-(normsh+(16-a)) = (3+accshift)-normsh

			HVX_Vector recip_sh = Q6_Vuh_vsub_VuhVuh_sat( q6op_Vh_vsplat_R(accshift+3), normsh );
			// (note that this is only valid in even h lanes, but that's good enough)

			// now we can do the last operation

			wbuf = (HVX_Vector*)workbuf;
			for( int  i = 0; i < nd32; i++ ){
				HVX_Vector m0 = wbuf[0];	// get the unscaled values
				HVX_Vector m1 = wbuf[1];
				HVX_Vector yout = final_scaling_op( m0, m1, recip_mant, recip_sh );
				*(HVX_Vector *)( pout + i*out_d32_stride) = yout;
				wbuf += 2;
			}
		} // for iw
	} // for ih
}

//
// This is a softmax operation for u8 'flat' tensors
// It does 'height' rows starting at the given pointers
// with the given stride (same at in and out; each 'row' is "depth" bytes, which
// can be any value.
// The pointers may be misaligned, but must have a common misalignment;
// also, the stride must be aligned.
//  So if we are working with depth = 96, for instance, we will need to call
// this four times (so stride = 96*4 = 3 vectors in each call) and each call will
// process 1/4 of the 'rows'.
//  **NOTE** this requires the depth to be a multiple of 4; this lets us use
// the same condition masks for bytes & words.
//
static void __attribute__((noinline))
softmax_operate_hvx_flat_function(
	struct softmax_run_state *rstp,
	uint8_t *data_out,						// may not be aligned better than x4
	uint8_t const *data_in,					// mat not be aligned better than x4
	int height,				// # of rows processed (could be < rstp->shape.height)
	int16_t *workbuf )
{
	// find horizontal vector situation
	int depth = rstp->shape.depth;
	unsigned left_align = (size_t)data_in & 127;
	unsigned padded_len = left_align + depth;
	int vecs_wide = (padded_len-1)/128u;			// # total # vecs, -1.
	int last_vector_n = padded_len - 128*vecs_wide;	//	# in last vector (1..128)
	uint32_t stride = depth << rstp->log2_K;	// this is a multiple of 128

	int accshift = rstp->sparms.accshift;


	HVX_VectorPred qL_not = Q6_Q_vsetq_R( left_align );
	HVX_VectorPred qR = q6op_Q_vsetq2_R( last_vector_n );
	HVX_VectorPred qR2;
	{
		// if vecs_wide = 0, we need to modify qR according to qL_not
		HVX_Vector ltmp = Q6_V_vand_QR( qL_not,-1);
		qR2 = Q6_Q_and_QQn( qR, Q6_Q_vand_VR( ltmp, (vecs_wide==0)?-1:0));
	}

	int bscale = rstp->sparms.betamul;
	int brsh =  rstp->sparms.beta_rsh;


	// work on 2 rows at once (horizontal reductions can be done just
	// as fast on 2 as one)

	int height_remain = height;
	for( int iht2 = 0; iht2 < ((height+1)>>1); iht2++, height_remain -= 2){

		//qL_not = Q6_Q_vsetq_R( FAKE_DEP_RR(left_align,height_remain) );

		// (if on last odd row, do the same thing twice)
		unsigned stridex = (height_remain > 1)? stride: 0;
		// first, just find the max across all the depth.
		HVX_Vector const * rp0 = (HVX_Vector const*)data_in;
		HVX_Vector const * rp1 = (HVX_Vector const*)(data_in+stridex);

		HVX_Vector v0max = Q6_V_vzero();
		HVX_Vector v1max = Q6_V_vzero();
		HVX_Vector v0a = q6op_V_vand_QnV( qL_not, *rp0++ );		// initial masking
		HVX_Vector v1a = q6op_V_vand_QnV( qL_not, *rp1++ );		// initial masking
		for(int i = 0; i < vecs_wide; i++){
			v0max = Q6_Vub_vmax_VubVub( v0max, v0a);			// 'max' previous va
			v1max = Q6_Vub_vmax_VubVub( v1max, v1a);
			v0a = * rp0++;							// get new one
			v1a = * rp1++;
		}
		v0a =  q6op_V_vand_QV( qR, v0a );	// apply final masking,
		v0max = Q6_Vub_vmax_VubVub( v0max, v0a);
		v1a =  q6op_V_vand_QV( qR, v1a );	// apply final masking,
		v1max = Q6_Vub_vmax_VubVub( v1max, v1a);
		// now reduce the max across
		HVX_VectorPair shuf = Q6_Wb_vshuffoe_VbVb( v1max, v0max);
		v0max = Q6_Vub_vmax_VubVub( Q6_V_hi_W(shuf),Q6_V_lo_W(shuf));
		// 64 row 0 results in even lanes; 64 row 1 results in odd lanes
		//
		for( int i =0; i < 6; i ++){
			shuf = Q6_W_vdeal_VVR( v0max, v0max, -2 );
			v0max = Q6_Vub_vmax_VubVub( Q6_V_hi_W(shuf),Q6_V_lo_W(shuf));
		}
		// now, all the even lanes are the row 0 max; odd lanes are row 1 max. untangle...
		shuf= Q6_Wb_vshuffoe_VbVb( v0max, v0max);
		v0max = Q6_V_lo_W(shuf);		// all lanes are row0-max
		v1max = Q6_V_hi_W(shuf);		// all lanes are row1-max

		// now find all the pow2( -beta*(maxx-x)) and store them at
		// 'workbuf', also sum them.
		HVX_Vector *wbuf = (HVX_Vector*)workbuf;

		rp0 = (HVX_Vector const*)data_in;
		rp1 = (HVX_Vector const*)(data_in+stridex);

		//
		// do the first vector on each row
		//
		HVX_Vector v0 = *rp0++;
		HVX_Vector v1 = *rp1++;

		struct pow2func_result p2res0 = hvx_pow2_func( v0, v0max, bscale, brsh, accshift, height_remain);
		struct pow2func_result p2res1 = hvx_pow2_func( v1, v1max, bscale, brsh, accshift, height_remain);

		wbuf[0] = p2res0.exp16[0];
		wbuf[1] = p2res0.exp16[1];
		wbuf[2] = p2res1.exp16[0];
		wbuf[3] = p2res1.exp16[1];
		wbuf += 4;

		// initial sums, with left masking.

		HVX_Vector msum_0 = Q6_Vw_vadd_VwVw(
			Q6_Vw_vadd_VwVw( p2res0.exp32[0], p2res0.exp32[2]),
			Q6_Vw_vadd_VwVw( p2res0.exp32[1], p2res0.exp32[3]));
		HVX_Vector msum_1 = Q6_Vw_vadd_VwVw(
			Q6_Vw_vadd_VwVw( p2res1.exp32[0], p2res1.exp32[2]),
			Q6_Vw_vadd_VwVw( p2res1.exp32[1], p2res1.exp32[3]));
		msum_0 = q6op_V_vand_QnV( qL_not, msum_0);
		msum_1 = q6op_V_vand_QnV( qL_not, msum_1);

		// now process over the other vectors (this loop runs 0 iterations if
		// there's only 1 vector)
		for( int i= 0; i < vecs_wide; i++ ){
			// get new values
			v0 =  *rp0++;
			v1 =  *rp1++;
			p2res0 = hvx_pow2_func( v0, v0max, bscale, brsh, accshift, height_remain);
			p2res1 = hvx_pow2_func( v1, v1max, bscale, brsh, accshift, height_remain);
			// accumulate them
			HVX_Vector tmp = Q6_Vw_vadd_VwVw( p2res0.exp32[0],p2res0.exp32[2]);
			msum_0 = Q6_Vw_vadd_VwVw( msum_0, p2res0.exp32[1]);
			msum_0 = Q6_Vw_vadd_VwVw( msum_0, p2res0.exp32[3]);
			msum_0 = Q6_Vw_vadd_VwVw( msum_0, tmp);
			tmp = Q6_Vw_vadd_VwVw( p2res1.exp32[0],p2res1.exp32[2]);
			msum_1 = Q6_Vw_vadd_VwVw( msum_1, p2res1.exp32[1]);
			msum_1 = Q6_Vw_vadd_VwVw( msum_1, p2res1.exp32[3]);
			msum_1 = Q6_Vw_vadd_VwVw( msum_1, tmp);
			wbuf[0] = p2res0.exp16[0];
			wbuf[1] = p2res0.exp16[1];
			wbuf[2] = p2res1.exp16[0];
			wbuf[3] = p2res1.exp16[1];
			wbuf += 4;
		}
		// we are all done this pass except we
		// need to remove some from sums according to right-masking.

		msum_0 = Q6_Vw_condnac_QnVwVw( qR, msum_0, p2res0.exp32[0]);
		msum_0 = Q6_Vw_condnac_QnVwVw( qR, msum_0, p2res0.exp32[2]);
		msum_0 = Q6_Vw_condnac_QnVwVw( qR, msum_0, p2res0.exp32[1]);
		msum_0 = Q6_Vw_condnac_QnVwVw( qR, msum_0, p2res0.exp32[3]);

		msum_1 = Q6_Vw_condnac_QnVwVw( qR, msum_1, p2res1.exp32[0]);
		msum_1 = Q6_Vw_condnac_QnVwVw( qR, msum_1, p2res1.exp32[2]);
		msum_1 = Q6_Vw_condnac_QnVwVw( qR, msum_1, p2res1.exp32[1]);
		msum_1 = Q6_Vw_condnac_QnVwVw( qR, msum_1, p2res1.exp32[3]);

		// OK, the msums are now complete, but each one is 32 x i32 sum and we need to sum
		// them horizontally. Start by combining two into 1.
		shuf = Q6_W_vdeal_VVR( msum_1, msum_0, 4 );
		msum_0 = Q6_Vw_vadd_VwVw( Q6_V_hi_W(shuf), Q6_V_lo_W(shuf) );
		// now reduce 16 pairs to 1 (duplicating in the process)
		for( int i =0; i < 4; i ++){
			shuf = Q6_W_vdeal_VVR( msum_0, msum_0, -8 );
			msum_0 = Q6_Vw_vadd_VwVw( Q6_V_hi_W(shuf),Q6_V_lo_W(shuf));
		}
		// each pair of i32 lanes is the same { msum_0, msum_1}
		// find reciprocal
		HVX_Vector normsh =  Q6_Vw_vnormamt_Vw(msum_0);		// at most 16; very likely >=6
		msum_0 = Q6_Vw_vasl_VwVw( msum_0, normsh);				// each is now '01' in the upper 2 bits
		msum_0 = Q6_Vh_vshuffo_VhVh(msum_0,msum_0);				// discard 16 lower bits; dup the upper
		//(255/256)/msum
		HVX_Vector recip_mant = hvx_recip16_inline( msum_0 ,10 );
		recip_mant = Q6_Vh_vmpy_VhRh_s1_rnd_sat( recip_mant, 0x10001 * (255<<7));
		// the variable rsh will be accshift+3 - normsh
		HVX_Vector v_accshift_3 = q6op_Vh_vsplat_R( FAKE_DEP_RR(accshift+3,height_remain));
		HVX_Vector recip_sh = Q6_Vuh_vsub_VuhVuh_sat( v_accshift_3, normsh );

		// (note that this is only valid in even h lanes, but that's good enough)

		// deinterleave the results to 'row 0' and 'row1' values - each will dup'd across
		// the whole vector.
		shuf = Q6_W_vdeal_VVR( recip_mant, recip_mant, 4);
		HVX_Vector recip_mant_0 = Q6_V_lo_W(shuf);
		HVX_Vector recip_mant_1 = Q6_V_hi_W(shuf);
		shuf = Q6_W_vdeal_VVR( recip_sh, recip_sh, 4);
		HVX_Vector recip_sh_0 = Q6_V_lo_W(shuf);
		HVX_Vector recip_sh_1 = Q6_V_hi_W(shuf);

		// now we can do the last pass
		HVX_Vector *wp0 = (HVX_Vector *)data_out;
		HVX_Vector *wp1 = (HVX_Vector *)(data_out + stridex);
		wbuf = (HVX_Vector*)workbuf;

		HVX_Vector m0_0 = wbuf[0];	// get the unscaled values
		HVX_Vector m1_0 = wbuf[1];
		HVX_Vector m0_1 = wbuf[2];
		HVX_Vector m1_1 = wbuf[3];
		wbuf += 4;
		HVX_Vector yout0 = final_scaling_op( m0_0, m1_0, recip_mant_0, recip_sh_0 );
		HVX_Vector yout1 = final_scaling_op( m0_1, m1_1, recip_mant_1, recip_sh_1 );

		HVX_Vector qX = qL_not;
		HVX_Vector q0 = Q6_Q_vand_VR( yout0, 0);
		// The store condition is : ~qX on first vector, ~q0 on any 'middle'
		// and qR on the last one (if there's only one vector, it's qR)
		//
		// final output loop has two cases, one for single row and one for double,
		// so we don't do extra stores.
		if( stridex != 0){
			for( int  i = 0; i < vecs_wide; i++ ){
				q6op_vstcc_QnAV(qX, wp0, yout0);			wp0 ++;
				q6op_vstcc_QnAV(qX, wp1, yout1);			wp1 ++;
				qX = q0;					// force to 0
				m0_0 = wbuf[0];	// get the unscaled values
				m1_0 = wbuf[1];
				m0_1 = wbuf[2];
				m1_1 = wbuf[3];
				wbuf += 4;
				yout0 = final_scaling_op( m0_0, m1_0, recip_mant_0, recip_sh_0 );
				yout1 = final_scaling_op( m0_1, m1_1, recip_mant_1, recip_sh_1 );
			}
			q6op_vstcc_QAV( qR2, wp0, yout0);
			q6op_vstcc_QAV( qR2, wp1, yout1);
		}else{
			for( int  i = 0; i < vecs_wide; i++ ){
				q6op_vstcc_QnAV(qX, wp0, yout0);			wp0 ++;
				qX = q0;					// force to 0
				m0_0 = wbuf[0];	// get the unscaled values
				m1_0 = wbuf[1];
				wbuf += 4;
				yout0 = final_scaling_op( m0_0, m1_0, recip_mant_0, recip_sh_0 );
			}
			q6op_vstcc_QAV( qR2, wp0, yout0);
		}

		// move to the next pair of rows.

		data_in += stride*2;
		data_out += stride*2;
	}
}

//
// This is a less aggressive, less efficient 'flat' operation which can handle any depth
// which is *not* a multiple of 4. This one leans to reducing code size while getting
// reasonable performance.
// It uses unaligned reads, does two rows at once (when possible) and interleaves the
// two rows by elements for most of the path. Knowing that the depth is not a multiple
// of 4 simplifies a few masking decisions.
//
static void __attribute__((noinline,unused))
softmax_operate_hvx_flat_notx4_function(
	struct softmax_run_state *rstp,
	uint8_t *data_out,						// no alignment constraint
	uint8_t const *data_in,					// no alignment constraint
	int height,				// # of rows processed (could be < rstp->shape.height)
	int16_t *workbuf )
{
	int depth = rstp->shape.depth;
	uint32_t stride = depth;

	int accshift = rstp->sparms.accshift;
	int bscale = rstp->sparms.betamul;
	int brsh =  rstp->sparms.beta_rsh;
	// find righ edge masks for even/odd words out of 64
	// These must work for end cases : depth = 64n+1  and 64n+63
	// in the first case, qR_w1 must all 0; in the second, qR_w0 must be all 1.
	HVX_VectorPred qR_w0 = q6op_Q_vsetq2_R((2*depth+2) & ~3);
	HVX_VectorPred qR_w1 = Q6_Q_vsetq_R(2*depth & ~3);

	// work on 2 rows at once

	int height_remain = height;
	for( int iht2 = 0; iht2 < ((height+1)>>1); iht2++, height_remain -= 2)
	{
		int stridex = ( height_remain ==1)? 0: stride;	// just do 1 row
		uint8_t const * inp0 = data_in + 2*iht2*stride;
		uint8_t const * inp1 = inp0 + stridex;
		uint8_t const * inpA0 = inp0;
		uint8_t const * inpA1 = inp1;

		// unaligned loads, read across two rows to find max
		// Each loop processes a full 128 from each row;
		// even lanes of vmax are for row 0, odd for row1.
		HVX_Vector vmax = Q6_V_vzero();
		for(int i= 0; i < (depth/128u); i++ ){
			HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector const*)inpA0);	inpA0 += 128;
			HVX_Vector v1 = q6op_V_vldu_A( (HVX_Vector const*)inpA1);	inpA1 += 128;
			HVX_VectorPair shuf = Q6_Wb_vshuffoe_VbVb( v1,v0);
			vmax = Q6_Vub_vmax_VubVub( vmax, Q6_V_lo_W(shuf));
			vmax = Q6_Vub_vmax_VubVub( vmax, Q6_V_hi_W(shuf));
		}
		{	// there is always an extra...
			HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector const*)inpA0);
			HVX_Vector v1 = q6op_V_vldu_A( (HVX_Vector const*)inpA1);
			HVX_VectorPair shuf = Q6_W_vshuff_VVR( v1,v0,-1);		// properly ordered shuffle
			HVX_Vector v_extra = Q6_V_lo_W(shuf);
			if( depth & 64 ){	// at least 64
				vmax = Q6_Vub_vmax_VubVub( vmax, v_extra );		// include all 64 of these pairs
				v_extra = Q6_V_hi_W(shuf);						// and some of these
			}
			v_extra = Q6_V_valign_VVR( v_extra, Q6_V_vzero(), depth*2);	// include the 'good' bytes from v_extra
			vmax = Q6_Vub_vmax_VubVub( vmax, v_extra );
		}

		// ok now we have the 'max' values in both rows;
		// reduce across 64 lane pairs
		for( int k = 0; k < 6; k++){
			HVX_VectorPair mxshuf = Q6_W_vdeal_VVR( vmax, vmax, -2);
			vmax = Q6_Vub_vmax_VubVub( Q6_V_hi_W(mxshuf), Q6_V_lo_W(mxshuf));
		}

		//
		// now find all the pow2( -beta*(maxx-x)) and store them at
		// 'workbuf', also sum them.
		// We interleave 2 rows to bytes as we go.
		HVX_Vector *wbuf = (HVX_Vector*)workbuf;

		// 64 elements per loop, from each row.
		HVX_Vector vin01;		// input vector (row 0 in even, row 1 in odd bytes)
		HVX_Vector vin01_next = Q6_V_vzero();
		HVX_Vector msum_0 = Q6_V_vzero();		// 32 bit sums
		HVX_Vector msum_1 = Q6_V_vzero();
		struct pow2func_result p2res;

		// each loop handles 64 elements from row 0, and 64 from row 1.
		for(int i =0; i < (depth+63)/64u; i ++ ){
			vin01 = vin01_next;
			if( (i&1)== 0){
				HVX_Vector v0 = q6op_V_vldu_A( (HVX_Vector const*)inp0);
				HVX_Vector v1 = q6op_V_vldu_A( (HVX_Vector const*)inp1);
				HVX_VectorPair shuf = Q6_W_vshuff_VVR( v1,v0,-1);
				vin01 = Q6_V_lo_W(shuf);
				vin01_next = Q6_V_hi_W(shuf);
			}
			inp0 += 64;
			inp1 += 64;
			p2res = hvx_pow2_func( vin01, vmax, bscale, brsh, accshift, height_remain);
			wbuf[0] = p2res.exp16[0];		// these are 64 row 0 exps
			wbuf[1] = p2res.exp16[1];		// 64 row 1 exps
			msum_0 = Q6_Vw_vadd_VwVw( msum_0, p2res.exp32[0]);	// 32 row 0 exps, even lanes
			msum_0 = Q6_Vw_vadd_VwVw( msum_0, p2res.exp32[2]);	// odd lanes
			msum_1 = Q6_Vw_vadd_VwVw( msum_1, p2res.exp32[1]);	// row 1 exps, even lanes
			msum_1 = Q6_Vw_vadd_VwVw( msum_1, p2res.exp32[3]);	// odd lanes
			wbuf += 2;
		}
		// since the depth is not a multiple of 64, we need to correct the sums by deducting
		// invalid lanes from last iteration.
		//
		msum_0 = Q6_Vw_condnac_QnVwVw( qR_w0, msum_0, p2res.exp32[0]);
		msum_0 = Q6_Vw_condnac_QnVwVw( qR_w1, msum_0, p2res.exp32[2]);
		msum_1 = Q6_Vw_condnac_QnVwVw( qR_w0, msum_1, p2res.exp32[1]);
		msum_1 = Q6_Vw_condnac_QnVwVw( qR_w1, msum_1, p2res.exp32[3]);
		//
		// now reduce the sums across 32 lanes.
		//
		HVX_VectorPair shuf = Q6_W_vdeal_VVR( msum_1, msum_0, 4);	// align & combine
		msum_0 = Q6_Vw_vadd_VwVw( Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));	// 16 even/odd pairs now
		for( int k = 0; k < 4; k++){
			shuf = Q6_W_vdeal_VVR( msum_0, msum_0, -8);
			msum_0 = Q6_Vw_vadd_VwVw( Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));
		}
		// even w lanes are row 0 sum; odd lanes are row 1
		HVX_Vector normsh =  Q6_Vw_vnormamt_Vw(msum_0);			// at most 16; very likely >=6
		msum_0 = Q6_Vw_vasl_VwVw( msum_0, normsh);				// each is now '01' in the upper 2 bits
		msum_0 = Q6_Vh_vshuffo_VhVh(msum_0,msum_0);				// discard 16 lower bits; dup the upper
		//(255/256)/msum
		HVX_Vector recip_mant = hvx_recip16_inline( msum_0 ,10 );
		recip_mant = Q6_Vh_vmpy_VhRh_s1_rnd_sat( recip_mant, 0x10001 * (255<<7));
		// the variable rsh will be accshift+3 - normsh
		HVX_Vector v_accshift_3 = q6op_Vh_vsplat_R( FAKE_DEP_RR(accshift+3,height_remain));
		HVX_Vector recip_sh = Q6_Vuh_vsub_VuhVuh_sat( v_accshift_3, normsh );
		// now, we have recip_mant and recip_sh. But the arrangement is: h lanes 0,1, 4,5  8,9 .. 60,61
		// are for row 0, and 2,3, 6,7 .. 62,63 are for row 1 (and 'recip_sh' is only ok in the even lanes).
		// We do just one row at a time, so separate these.
		//
		HVX_VectorPair recipm = Q6_W_vshuff_VVR( recip_mant, recip_mant, 4 );	// one for row 0, one for row 1
		HVX_VectorPair recips = Q6_W_vshuff_VVR( recip_sh, recip_sh, 4 );	// one for row 0, one for row 1
		recip_mant = Q6_V_lo_W( recipm );
		recip_sh = Q6_V_lo_W( recips );

		wbuf = (HVX_Vector*)workbuf;
		while(1){	// loop once (stridex==0) or twice (stridex!=0)

			int dremain = depth;
			HVX_Vector yout;
			HVX_Vector *voutp = (HVX_Vector*)data_out;
			for(int i =0 ; i < (depth+127)/128u; i++, dremain-= 128) {
				// get two workbuf vecs containing values for this row; deal out even/odd values
				// so the bytes come out in the right order.
				HVX_VectorPair dealt = Q6_W_vdeal_VVR( wbuf[2], wbuf[0], -2);  wbuf += 4;
				yout = final_scaling_op( Q6_V_lo_W(dealt), Q6_V_hi_W(dealt), recip_mant, recip_sh );
				if( dremain >= 128 ){
					q6op_vstu_AV( voutp, yout );
					voutp++;
				}
			}
			dremain = depth & 127;		// restore this
			// always have 'dremain' left, 1..127
			q6op_vstu_variable_ARV( voutp, dremain, yout);
			data_out += stride;

			if( stridex != 0 ){
				recip_mant = Q6_V_hi_W(recipm);	// get the row 1 recip
				recip_sh = Q6_V_hi_W( recips );	// and shift
				wbuf = (HVX_Vector*)workbuf + 1;	// odd vectors for row 1
				stridex = 0;					// don't run again
			}else{
				break;
			}
		}// even/odd output loop
	} // for iht
}
///////////////////////////////////////////////////////////
/// code for depth=2 flat mode ////////////////////////////
///////////////////////////////////////////////////////////

/////////////////// depth = 2 special case /////////////////////////////////
//
// for depth=2, softmax is similar to sigmoid of the difference:
//   inputs = d0, d1
//   outputs = (1 + tanh(beta*(d0-d1)/2)), (1 - tanh(beta*(d0-d1)/2))
//
//  It's done like this:
//    (1) find d0-d1   (from quantized values, range -255...255)
//    (2) using abs(d0-d1), do a table lookup. result is 128..255
//        Table depends on beta & scaling; the result is correct for
//        output 0 if (d0-d1)>=0.
//    (3) find the second result by subtracting the first from 255;
//    (4) if (d0-d1)<0, swap the results.
//     - result is correct for output range 0..1.0
//  Unfortunately when both inputs are the same, one output will be 128
//   and the other will be 127.
//
//
// this struct maintains the lookup table from call to call.
// The same struct is also used for 'large-depth' case, but there are independent entries
// for that: fullscale_largedepth and table_largedepth
//
struct softmax_lookup_cache{
	float d2_fullscale;				// this is beta * (inmax-inmin); -1 if invalid.
	float largedepth_fullscale;		// this is beta * (inmax-inmin); -1 if invalid.
	HVX_Vector * largedepth_tables;	// points to 6 vectors, or NULL.
	uint8_t d2_lookup [256];	// lookup table (not vec aligned; pre-shuffled.
};

struct softmax_d2_runstate {
	struct softmax_lookup_cache *cache;		// points to the cached table
	uint8_t const * input;
	uint8_t *output;
	unsigned total_vecs;		// total amount of work in vectors
	unsigned vecs_per_chunk;	// amount to do per work unit
	volatile unsigned next_work;	// which vector offset to do nexy
	nn_sem_t done_sem;
};
static void softmax_d2_flat_worker( struct nn_graph *nn, void * rstpv );
static void fill_d2_lookup_table(struct softmax_lookup_cache * cachep, float fullscale );

static int
softmax_d2_flat_exec( struct nn_node * self,  struct nn_graph * nn)
{
	struct tensor const *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct softmax_lookup_cache *cache = (struct softmax_lookup_cache*)self->opaque;
	if( cache == NULL){
		cache = (struct softmax_lookup_cache*)nn_calloc( 1, sizeof(struct softmax_lookup_cache));
		if( cache == NULL)return errlog(nn,"calloc");
		self->opaque = (void*)cache;
		cache->largedepth_fullscale = -1.0f;
		cache->d2_fullscale = -1.0f;
	}
#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif
	float in_min = tensor_get_float( self->inputs[1],0);
	float in_max = tensor_get_float( self->inputs[2],0);
	float beta = (self->n_inputs <4)? 1.0f : tensor_get_float( self->inputs[3],0);
	float fullscale = beta*(in_max-in_min);
	if( cache->d2_fullscale != fullscale ){	// rebuild lookup table if needed
		fill_d2_lookup_table( cache, fullscale);
	}
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!= 0){
		return errlog(nn,"output too small");
	}
	struct softmax_d2_runstate runstate;
	runstate.cache = cache;
	runstate.input = in_tensor->data;
	runstate.output = out_tensor->data;

	unsigned nelements = tensor_element_count( in_tensor);
	unsigned nvecs = (nelements+127)/128u;
	unsigned vecs_per_chunk;
	int n_threads;

	// The run thread only works 2 vectors at a time, in chunks
	// of vecs_per_chunk which is always even and <= nvecs. If nvecs is odd,
	// the last operation will be of odd # vecs, and it will be adjusted
	// to include one more vec at the front (so it will be an even #  of vecs,
	// and overlap a previous work item by 1 vec). If the total # of vecs is 1,
	// however, this won't work; so we move the input and output to scratch and
	// make it a single 2-vector operation.
	//
	if( nelements <=128){	// special case
		uint8_t * tmp = (uint8_t*)nn->scratch;
		nvecs = 2;
		vecs_per_chunk = 2;
		memcpy( tmp, runstate.input, nelements );
		runstate.input = tmp;
		runstate.output = tmp+256;
		n_threads = 1;
	}else{
		vecs_per_chunk = 256;
		if ( nvecs < vecs_per_chunk * 4* MAX_THREADS ){
			// not so much work, try to chop it up better.
			if( nvecs <= 33 || (MAX_THREADS==2)){	// small case;
				// chop in half if >= 16 or if odd #
				// (when chopping in half, must round up to even).
				vecs_per_chunk = (nvecs>=16 || (nvecs&1)!=0)? (((nvecs+3)>>1)&~1u) : nvecs;
			}else{
				// divide nvecs by maxthreads, round up to even.
				vecs_per_chunk = 2*((nvecs + (2*MAX_THREADS-1))/(2u*MAX_THREADS));
			}
		}
		vecs_per_chunk = min_u32( nvecs&~1u, vecs_per_chunk);
		if( MAX_THREADS==2){
			n_threads = (vecs_per_chunk < nvecs) ? 2: 1;
		}else{
			if(vecs_per_chunk*MAX_THREADS <= nvecs ){
				n_threads = MAX_THREADS;
			}else{
				n_threads = (nvecs+(vecs_per_chunk-1))/vecs_per_chunk;
			}
		}
	}
	runstate.total_vecs = nvecs;
	runstate.vecs_per_chunk = vecs_per_chunk;
	runstate.next_work = 0;
	nn_sem_init( &runstate.done_sem, 0);

	for( int i = 0; i < n_threads; i++){
		nn_os_work_for_vector( nn, softmax_d2_flat_worker, &runstate);
	}
	int err = 0;
	if(  tensor_set_single_float(self->outputs[1],0.0f) || tensor_set_single_float(self->outputs[2],1.0f)){
		err = 1;
	}
	nn_sem_wait_n_times( &runstate.done_sem, n_threads);
	if( err ) return errlog(nn,"failed to set output value");
	if( nelements <= 128){
		memcpy( out_tensor->data, runstate.output, nelements);
	}
#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("softmax_flat_d2 cycles = %d (elements = %d)\n",
			(end_time-start_time),
			(int)tensor_element_count(in_tensor));
#endif
	return 0;
}
// expf approx, same domain as 'proper' expf, and
// accurate enough for softmax with 8-bit output.
//
static inline float expf_approx(float x){
	int xfrac = roundf_i32(x*(float)(16384.0/0.69314718056) );
	// now we want 2^xfrac (which has 14 fractional bits).
	float xf  = (xfrac & 0x3FFF) * (float)(1./16384.);
	// approx for 2^x
	// good to about 11 bits
	// 4th order would be:
	// 1.          0.69304423  0.24130777  0.0521633   0.0134847

	float exp0 = 1.0f + xf*(0.69051585f + xf*(0.23793659f + xf*0.07154756f));
	return flt_power2(xfrac>>14)*exp0;
}


// fill in the table for a given 'fullscale' value.
//
//   tanh(x/2) = (1-exp(-x))/(1+exp(-x))
//
//   =>( 1 + tanh(x/2))/2 =  1/(1+exp(-x))
//
//  we want (1 + tanh(k*d/2))/2 = 1/(1+exp(-k*d))
//
//  .. and then multiply by 255 and round.
//   'd' is the difference in range 0..255; k = fullscale/255.
// Also
//    - first output is always going to be 127.5 -> 128;
//    - as soon as any result is 255, all the rest will be too, so we can stop.
//      finding exps.
//    - values are stored in 'shuffle' order for vlut.

static void
fill_d2_lookup_table(struct softmax_lookup_cache * cachep, float fullscale )
{
	float k = flt_div_255( fullscale);

	uint8_t *wp = &cachep->d2_lookup[0];
	wp[0] = 128;
	int i;
	int vali = 0;
	int windex= 2;	// write index
	for( i = 1; i < 256;i++){
		if (vali < 255){
			float val = 255.0f/(1.0f + expf_approx(-k*(float)i));
			vali = roundf_i32(val);
		}
		wp[windex] = vali;
		windex += 2;
		if( (windex & 0x7e)== 0){	// reached 128,129,256,or 257 @ i==63,127,191,255
			windex -= (windex&1)?1:127;   // ->1, 128,129,(don't-care)
		}
	}
	cachep->d2_fullscale = fullscale;
}

//
// inner-loop function for the depth=2 softmax.
// works on 128 pairs at a time.
// tbl0, tbl1 are the lookup table.
//
static inline HVX_VectorPair
process_softmax_128xd2( HVX_Vector v0, HVX_Vector v1, HVX_Vector tbl0, HVX_Vector tbl1)
{
	// start by finding all the diffs even-odd, as 16-bit signed #'s  1*even +(-1)*odd
	HVX_Vector delt0 = Q6_Vh_vdmpy_VubRb( v0, 0xFF01FF01);
	HVX_Vector delt1 = Q6_Vh_vdmpy_VubRb( v1, 0xFF01FF01);
	// take the abs values of these and pack them into bytes in 1 vector.
	//
	HVX_Vector dabs = Q6_Vb_vshuffe_VbVb( Q6_Vh_vabs_Vh(delt1),Q6_Vh_vabs_Vh(delt0));
	// take the signs all in one vector
	HVX_Vector dsign = Q6_Vb_vshuffo_VbVb( delt1, delt0);
	// and find opposite sign
	HVX_Vector dsign_not = Q6_V_vnot_V(dsign);
	//
	// u8->u8 lookup on dabs
	HVX_Vector vout =  q6op_Vb_vlut32_VbVbI( dabs,tbl0, 0 );
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, dabs, tbl0,1);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, dabs, tbl0,2);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, dabs, tbl0,3);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, dabs, tbl1,4);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, dabs, tbl1,5);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, dabs, tbl1,6);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, dabs, tbl1,7);

	// find the '0' and '1' outputs based on original sign
	HVX_Vector vout0 = Q6_V_vxor_VV(vout,dsign);	// 255-x
	HVX_Vector vout1 = Q6_V_vxor_VV(vout,dsign_not);	// 255-vout1
	// and then combine them
	return Q6_Wb_vshuffoe_VbVb( vout1, vout0);
}
// worker thread
// This only works in even # of vectors - i.e. vchunk must be even;
// if vtotal is odd, the last chunk will be odd and in this case
// we move the start point back one to make it even. So, if the total
// # of vectors in the job is odd, we must have vchunk < vtotal so
// the last one will be even. If the total # is 1 vector, it needs
// to be set up as vchunk=vtotal=2 and pointing to another area.
//
static void
softmax_d2_flat_worker( struct nn_graph *nn, void * rstpv )
{
	struct softmax_d2_runstate * rstp = (struct softmax_d2_runstate*)rstpv;

	struct softmax_lookup_cache const *cachep = rstp->cache;
	HVX_Vector vlut0 = q6op_V_vldu_A( (HVX_Vector const*)&cachep->d2_lookup[0]);
	HVX_Vector vlut1 = q6op_V_vldu_A( (HVX_Vector const*)&cachep->d2_lookup[128]);
	uint8_t const *inp = rstp->input;
	uint8_t * outp = rstp->output;

	unsigned vchunk = rstp->vecs_per_chunk;
	unsigned vtotal = rstp->total_vecs;
	unsigned vstart;
	while( vstart = __sync_fetch_and_add( &rstp->next_work, vchunk), vstart < vtotal){
		unsigned vnum = min_u32( vchunk, vtotal-vstart);
		if( (vnum&1) ){		// odd #? pad it out at the front.
			vnum++;
			if( vstart == 0 ) return;
			--vstart;
		}
		HVX_Vector const *vpin = (HVX_Vector const*)(inp + vstart*128);
		l2fetch( vpin, 128,128,vnum);
		HVX_Vector *vpout = (HVX_Vector *)(outp + vstart*128);
		int nvpair = vnum >>1;
		for(int i = 0; i < nvpair; i++){
			HVX_VectorPair result = process_softmax_128xd2( vpin[0],vpin[1], vlut0,vlut1);
			vpout[0] = Q6_V_lo_W(result);
			vpout[1] = Q6_V_hi_W(result);
			vpin += 2;
			vpout += 2;
		}
	}
	nn_sem_post( &rstp->done_sem);
}

///////////////////////////////////////////////////////////////////////////////////////////////
/// Histogram-based algo (for large depth)
///////////////////////////////////////////////////////////////////////////////////////////////

static int fill_largedepth_lookup( struct nn_graph * nn, struct softmax_lookup_cache * cache, float fullscale);
static void softmax_larged_flat_worker( struct nn_graph * nn, void *rstpv);
struct softmax_largedepth_runstate {
	HVX_Vector const *exptab;		// points to the exponent table
	uint8_t const * input;
	uint8_t *output;
	int depth;					// the depth
	unsigned total_batches;		// total amount of work in depth units
	unsigned batches_per_chunk;	// amount to do per work unit
	volatile unsigned next_work;	// which batch offset to do next

	uint8_t *histo_scratch;		// scratch area for histos (across all threads)
	uint32_t histo_scratch_size;	// size of it, per thread
	volatile int thrdindex;		// used to divvy up the scratch.

	nn_sem_t done_sem;
};


static inline void
softmax_larged_batchchunk(uint8_t const * in_data, uint8_t *out_data,	int depth,int batches,uint16_t *histo_store,
		uint16_t *max_h_store,	HVX_Vector const * exp_tables);

static int
softmax_larged_flat_exec( struct nn_node * self,  struct nn_graph * nn)
{
	struct tensor const *in_tensor = self->inputs[0];
	struct tensor *out_tensor = self->outputs[0];
	struct softmax_lookup_cache *cache = (struct softmax_lookup_cache*)self->opaque;
	if( cache == NULL){
		cache = (struct softmax_lookup_cache*)nn_calloc( 1, sizeof(struct softmax_lookup_cache));
		if( cache == NULL)return errlog(nn,"calloc");
		self->opaque = (void*)cache;
		cache->largedepth_fullscale = -1.0f;
		cache->d2_fullscale = -1.0f;
	}
#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif
	float in_min = tensor_get_float( self->inputs[1],0);
	float in_max = tensor_get_float( self->inputs[2],0);
	float beta = (self->n_inputs <4)? 1.0f : tensor_get_float( self->inputs[3],0);
	float fullscale = beta*(in_max-in_min);
	if( cache->d2_fullscale != fullscale ){	// rebuild lookup table if needed
		if( fill_largedepth_lookup(nn, cache, fullscale)!=0) return -1;
	}
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!= 0){
		return errlog(nn,"output too small");
	}
	int depth = in_tensor->shape.depth;
	int batches = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width;
	// support of depth = 65536 is possible
	// because the histogram will saturate; if all the values are the same you'll get one
	// bin with n = 65535 which will produce the correct final output. Larger depths
	// could result in overflow in the dot product, however.
	if( depth > 65536) return errlog(nn,"depth > 65536 not supported in softmax");

	// preload the precomputed vector table
	l2fetch( cache->largedepth_tables, 128, 128, 18);

	struct softmax_largedepth_runstate runstate;
	runstate.exptab = cache->largedepth_tables;
	runstate.input = in_tensor->data;
	runstate.output = out_tensor->data;
	runstate.depth = depth;
	runstate.total_batches = batches;

	unsigned batches_per_chunk;
	int n_threads;
	// how many batches per chunk, and how many threads?
	batches_per_chunk = depth < 4096?32: 16;
	if( batches < batches_per_chunk * 4*MAX_THREADS ){	// smaller #, try to chop it up better
		// not so much work, try to chop it up better.
		if( batches <= 32 ){	// small case; divide by 2
			batches_per_chunk = (batches+1)>>1;
		}else{		// divide by 4
			batches_per_chunk = (batches+3)>>2;
		}
	}
	batches_per_chunk = min_u32(batches, batches_per_chunk);
	runstate.batches_per_chunk = batches_per_chunk;

	if( batches > (MAX_THREADS-1)*batches_per_chunk ){
		n_threads = MAX_THREADS;
	}else {
		n_threads = (batches + (batches_per_chunk-1))/(unsigned)batches_per_chunk;
		if(MAX_THREADS==2) n_threads = 1;
	}
	// how much scratch do we need?
	// in vectors, it's
	//   4* batches_per_chunk for the histograms;
	//   batches_per_chunk/64 (rounded up) for the 'histo_max' buffers.
	unsigned scratchv_per_thread = 4*batches_per_chunk + (batches_per_chunk+63)/64u;
	if( nn->scratch_size  < n_threads * 128*scratchv_per_thread )
		return errlog(nn, "need %d bytes scratch * %d threads",scratchv_per_thread*128,n_threads );
	runstate.histo_scratch_size = 128*scratchv_per_thread;
	runstate.histo_scratch = nn->scratch;
	runstate.thrdindex = 0;

	runstate.next_work = 0;

	//printf("%d batches of %d; in chunks of %d = %d threads\n", batches, depth,batches_per_chunk, n_threads );

	nn_sem_init( &runstate.done_sem, 0);

	for( int i = 0; i < n_threads; i++){
		nn_os_work_for_vector( nn, softmax_larged_flat_worker, &runstate);
	}
	int err = 0;
	if(  tensor_set_single_float(self->outputs[1],0.0f) || tensor_set_single_float(self->outputs[2],1.0f)){
		err = 1;
	}
	nn_sem_wait_n_times( &runstate.done_sem, n_threads);
	if( err ) return errlog(nn,"failed to set output value");

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("softmax_flat_larged cycles = %d (elements = %d)\n",
			(end_time-start_time),
			(int)tensor_element_count(in_tensor));
#endif
	return 0;
}

//
// work function for the large-depth flat softmax
//
static void
softmax_larged_flat_worker( struct nn_graph * nn, void *rstpv)
{
	struct softmax_largedepth_runstate * rstp = (struct softmax_largedepth_runstate*)rstpv;
	// get a thread id
	int thrid= __sync_fetch_and_add( &rstp->thrdindex, 1);
	// get our scratch partition
	uint8_t *our_scratch = rstp->histo_scratch + thrid * rstp->histo_scratch_size;
	int batches_per_chunk = rstp->batches_per_chunk;
	HVX_Vector const * exptab = rstp->exptab;

	// divvy it up
	uint16_t * histo_store = (uint16_t*)our_scratch;		// for histograms
	uint16_t * histo_max_store =  histo_store + 256*batches_per_chunk;	// for 'max bin' table
	int depth = rstp->depth;
	uint8_t const * in_base = rstp->input;
	uint8_t  * out_base = rstp->output;
	int batches_total = rstp->total_batches;
	int curr_batch_idx;

	while( curr_batch_idx = __sync_fetch_and_add( &rstp->next_work,batches_per_chunk),curr_batch_idx <  batches_total)
	{
		int batches_now = min_i32(batches_per_chunk,batches_total- curr_batch_idx );
		// ok, do that many batches
		uint8_t const *inp = in_base + depth * curr_batch_idx;
		l2fetch(inp, 128,128,(depth*batches_now+127)/128u);

		uint8_t *outp = out_base + depth * curr_batch_idx;
		//printf("%d batches %p->%p\n", batches_now, inp,outp);
		softmax_larged_batchchunk(inp, outp, depth, batches_now, histo_store, histo_max_store, exptab );
	}

	nn_sem_post( &rstp->done_sem);
}

// this function is given a pointer to nhisto histograms,
// each 256 x uint16 vec aligned
// and for each one it finds the index (0..255) of the largest nonzero bin.
// If all are zero the result will be zero.
// The output is 'result' which is a uint16 array; note that this must be vec-aligned, and
// will be garbage-filled out to a vector boundary.
//
static inline void __attribute__((unused))
find_largest_histo_bin(
		uint16_t const * histos,		// must be vec-aligned; [nhisto * 256]
		uint16_t * result,				// output [nhisto]; vec aligned
		int nhisto)
{
	HVX_Vector indices_01 = *(HVX_Vector const*)const_Count128;	// [0, 1, ... 127]
	indices_01 = Q6_Vb_vshuff_Vb( indices_01);						//[0,64,1,65, ...63,127]
	HVX_Vector indices_23 = Q6_V_vor_VV( indices_01, q6op_Vb_vsplat_R(0x80));	// [128,192,129 ... 191, 255]

	HVX_Vector funnel = Q6_V_vzero();		// reduction funnel
	HVX_Vector shifter = Q6_V_vzero();		// accumulates reduced values.

	int scount = 64+6;	// # of loops until we need a vector store of results.
	uint16_t * endp = result + nhisto;		// not done until result >= this.

	HVX_Vector const *hptr =(HVX_Vector const*)histos;
	for( int i =0; i < nhisto; i++ ){
		HVX_Vector h0 = hptr[0];
		HVX_Vector h1 = hptr[1];
		// pack 64+64 into one vector, maintaining non-zeroness
		HVX_VectorPair shuf = Q6_Wb_vshuffoe_VbVb( h1,h0);
		HVX_Vector h01 = Q6_V_vor_VV( Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));
		// and the next two
		shuf = Q6_Wb_vshuffoe_VbVb( hptr[3],hptr[2]);
		HVX_Vector h23 = Q6_V_vor_VV( Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));
		// make a vector which has the 'index' value where those are nonzero, and zero where they
		// are zero.
		h01 = q6op_V_vand_QnV( Q6_Q_vcmp_eq_VbVb( h01, Q6_V_vzero()), indices_01 );
		h23 = q6op_V_vand_QnV( Q6_Q_vcmp_eq_VbVb( h23, Q6_V_vzero()), indices_23 );
		// now we just want to find the maximum of all 256 of those values.
		h01 = Q6_Vub_vmax_VubVub( h01,h23);
		// deal with funnel, reduce across
		HVX_VectorPair dealt = Q6_W_vdeal_VVR( h01, funnel,-1);
		funnel = Q6_Vub_vmax_VubVub(Q6_V_hi_W(dealt), Q6_V_lo_W(dealt));
		// there is now a fully reduced value in position 1 (it is the 'i-6' result).
		// shift it into the shifter, also get the garbage byte @ 0
		shifter = Q6_V_valign_VVI( funnel, shifter,2);
		if( --scount ==0){
			// we have 64 values ready to store. They are in the odd bytes, with garbage in even...
			// so fix that first
			HVX_Vector t = Q6_Vb_vshuffo_VbVb(  Q6_V_vzero(),shifter);
			*(HVX_Vector *)result = t;
			result += 64;
			scount = 64;
		}
		hptr += 4;	// next histo
	}
	// flush the pipe...
	// This runs at least 6 and usually 7 loops, and does at least one and sometimes two stores.
	// E.g. when nhisto = 66, we have at this point results 0..59 in the shifter, and 60..65 in the funnel (scount = 4)
	// so we will end up doing	one store on 4th iteration to finish the first 64, and another on 7th to
	// store the last 2 (that one will use a valign(zero, shifter, 2*62) to position the results).
	int in_funnel = 6;		// # of partially-reduced results in funnel (including 'pre-garbage' if nhisto < 6).
	while( result < endp){
		if( in_funnel > 0 ){
			HVX_VectorPair dealt = Q6_W_vdeal_VVR( funnel, funnel,-1);
			funnel = Q6_Vub_vmax_VubVub(Q6_V_hi_W(dealt), Q6_V_lo_W(dealt));
			shifter = Q6_V_valign_VVI( funnel, shifter,2);
			--in_funnel;
		}else{
			//-> everything valid is in the shifter; just 'fast-forward'
			shifter = Q6_V_valign_VVR( Q6_V_vzero(), shifter, 2*scount);
			scount =1;	// and can store next
		}
		if( --scount == 0){
			HVX_Vector t = Q6_Vb_vshuffo_VbVb(  Q6_V_vzero(),shifter);
			*(HVX_Vector *)result = t;
			result += 64;
			scount = 64;
		}
	}
}
// this will fill in the 'largedepth_tables' for a given 'fullscale' value.
//
// This is two tables of the value y[i] = exp( step*(255-i) )
//   where i = 0..255
//   and step = fullscale/255
// note that y[255] = 1.0
// The first table is y16 = y*(255/256) * (2^15), rounded and saturated to 0..0x7fff.
// The second table is y32 = y*(2^32), rounded and saturated to u32; however,
//  the lower 16 and upper bits are stored separately (and the two are interleaved,and the
// first 3/4 of the table repeated, as below)
//
// These are stored in 18 vectors as below:
//             #0       y16[0..63]
//             #1       y16[64..127]
//             #2       y16[128..191]
//             #3       y16[192..255]]
//            #4,#5     low 16 bits of y32[0..63]; hi 16 bits of y32[0..63]
//            #6,#7     low 16 bits of y32[64..127]; hi 16 bits of y32[64..127]
//            #8,#9     low 16 bits of y32[128..191]; hi 16 bits of y32[128..191]
//            #10,#11    low 16 bits of y32[192..255]; hi 16 bits of y32[192..255]
//            #12,#13    Same as #4,#5
//            #14,#16    Same as #6,#7
//            #16,#17    Same as #8,#9
//
static int
fill_largedepth_lookup( struct nn_graph * nn, struct softmax_lookup_cache * cache, float fullscale)
{
	uint16_t * tptr = (uint16_t*)cache->largedepth_tables;
	if( tptr== NULL){
		tptr = nn_memalign(128,18*128 );
		if( tptr == NULL) return errlog(nn,"alloc failed");
		cache->largedepth_tables = (HVX_Vector*)tptr;
	}
	float step  = flt_div_255( fullscale);
	float exp2_m1 = expm1f( step*2.0 );		// exp(2*step)-1
	float exp1_m1 = expm1f( step );			// exp(step)-1

	for( int ipos = 0; ipos < 256; ipos += 4){			// groups of 4
		float y0 = expf( step*(ipos-255));	// y[ipos]		// y0 = first of 4
		float y1 = y0 + y0*exp1_m1;			// find the next y
		uint16_t * py32 = &tptr[ 4*64 + (ipos>>6)* 128 + (ipos&63) ];	// point to where y32[ipos].lo goes
		uint16_t * py16 = &tptr[ ipos ]; // and y16
		for( int k = 0; k < 4; k++){
			int yi16 = roundf_i32( y0 * (float)(255*128));
			uint32_t yu32 = (y0>=1.0f)? 0xFFFFFFFFu : (uint32_t)(0.5f + y0 *4294967296.0f);
			py32[k] = (uint16_t)yu32;			// lo part here
			py32[k+64] = (uint16_t)(yu32>>16);	// hi part in next vector
			py16[k] = yi16;
			float tmp = y0 + y0*exp2_m1;
			y0 = y1; y1 = tmp; 	// y0 <-y1; y1 <-y2
		}
	}
	// entry 255 is always 0xFFFFFFFF and 0x7f80
	tptr[10*64+63] = 0xFFFF;
	tptr[11*64+63] = 0xFFFF;
	tptr[255] = 0x7F80;
	memcpy( &tptr[12*64], &tptr[4*64], 3*64*sizeof(uint16_t));
	cache->largedepth_fullscale = fullscale;
	return 0;
}

// LARGE DEPTH ALGORITHM for softmax;
// used for depth <=65536 but should be >256 (best min value needs to be determined)
//
// steps are:
//   (1) find an exact histogram of the 8-bit inputs in the batch
//   (2) find the largest non-empty bin in the histogram (maxh)
//   (3) extract 256 adjacent entries from the histogram, ending at bin 'maxh'.
//       This needs (255-max) zero pads on the left.
//   (4) We have a precalc table of 256 'exp' values, based on the current scale, normalized so
//   (5) Take the full 'dot product' of (3) and (4). This is the sum of the exponentials,
//     relative to maxh. It will always be < 2^48 and >= 0xFFFFFFFF.
//   (6) find 1/dot_product, expressed as a normalized fraction with 15 frac bits, and a shift
//   (7) we also have a table as in 4, but the exps are 16-bit with 15 fract bits, and prescaled
//        by 255/256; i.e. [255] is 0x7F80. The first 64 entries from this table are assumed to be
//        too small to matter, and are omitted.
//       multiply all of these by the value from (6) and convert the result to u8.
//   (8) The 192 results from (7) are a lookup table which is used in this batch to map input codes to
//       output codes. First, (maxh-191) is subtracted from each input; then they are used to index
//       the table.
//
/// The operation is worked across several batches at once; this makes it more efficient to do the
// horizontal reductions in (2) and (5), and the reciprocal in (6).
//
//
// PROTOTYPE CODE
// do a series of 'flat' softmax using the 'high depth' algo
//
static inline void
softmax_larged_batchchunk(
		uint8_t const * in_data,				// any alignment allowed; read [batches*depth]
		uint8_t *out_data,					// any alignment allowed; store [batches *depth]
		int depth,							//  must be >=128 (larger is better)
		int batches,						// # of batches >=1
		uint16_t *histo_store,				// room for 'batches' histos, vec aligned [256*batches]
		uint16_t *max_h_store,				// rooom for batches * u16, and vec padding
		HVX_Vector const * exp_tables)		// points to exponent table built by fill_largedepth_lookup
{
	// first find all the histograms and the max values
    histogram_flat_asm( histo_store, in_data, depth, batches, depth );
    find_largest_histo_bin( histo_store, max_h_store, batches );

    // hist_store is an array [batches] [4] of vector; this is recycled in two ways:
    //  (1) vectors 0,1,2,3 of the first batch are used as working space when doing the horizontal
    //     sum reductions
    //  (2) the reciprocals are stored, 8 per vector, with a stride of 16 bytes,  starting in the third vector of
    //     the first histo. Each is a pair  { recip, recip} - two identical 16-bit values.
    //

    HVX_Vector *ppprev= (HVX_Vector *)&histo_store[0];	// reused for reduction tree
    int32_t * recip_store = (int32_t *) &histo_store[3*64];

	// The first loop finds the a dot product in each batch, and its reciprocal;
	// to reduce the overhead here, there is an 8->1 reduction tree, and
	// on batches 7,15,23.. it finds 8 reciprocals in parallel and stores them.
	//
	// When 'batches' is not a multiple of 8, there is a 'batch_offset' which is used
	// in the last round to make the reduction tree work properly. For instance
	// if batches = 11, batch_offset will be 0 for batches 0..7, and will be 5
	// for batches 8,9,10, causing them to be seen as batches 13,14,15 for the packing
	//  tree.
	//
	int batch_offs_thr = batches & ~7;	// apply last_batch_offs when ibat >= batch_offs_thr
	int last_batch_offs = batch_offs_thr+8-batches;


	// The dot product is calculated losslessly using 32-bit upper and 32-bit
	// lower partial sums, each 32 bits; when these are combined the lower 16
	// bits are discarded; the result is guaranteed to be at least 0xFFFF and
	// and at most (depth*0xFFFFFFFF)>>16, i.e. it will fit in u32 provided
	// depth <=65536. cl0 is used to find the magnitude of sum>>16 ( it will have 0..16 leading zeros)
	// and then the 15 bits starting with the leading '1' are used to find the
	// reciprocal.
	//

    for( int ibat = 0; ibat <batches; ibat++){
    	HVX_Vector sop_hi, sop_lo;		// dot product, prior to horizontal reduction
    	{
			int maxh = max_h_store[ibat];			// 0..255

			// the histogram is 4 vectors containing 16-bit counts for codes 0..255
			// Need to find the dot product of this, with 256 32-bit codes in the exp table.
			// The two tables are first rotated relative to each other to line up 'maxh' of the
			// histogram with 255 in the exp table:
			//   - using vlalign, histogram is rotated up by 0..63 to place 'maxh' bin at location
			//      63,127,191 or 255;
			//   - exp table is rotated down by 0,64,128 or 192, by adjusting the
			//    point where we start reading it (it contains the 4 lo/hi
			//    pairs, followed by a repeat of the first 3 pairs, so each 2-vector
			//    bump rotates it down by 64).
			//
			//   read exptab at +4,+6,+8,+10 according to 2 msbs of maxh being 3,2,1,0
			HVX_Vector const * exptab_p = exp_tables + 4 + 6-2*((maxh>>6)&3);
			int vlalign_ctl = 2*~maxh;		// used to rotate histogram up by 0..63 locations

			// point to the histogram...
			HVX_Vector const  * hist = (HVX_Vector const  *) &histo_store[ibat*256];
			HVX_Vector v0 = hist[0];
			HVX_Vector v3 = hist[3];
			HVX_Vector v1 = hist[1];
			HVX_Vector v2 = hist[2];

			HVX_Vector vh0 = Q6_V_vlalign_VVR( v0,v3, vlalign_ctl);
			HVX_VectorPair wsop_lo = Q6_Wuw_vmpy_VuhVuh( vh0, exptab_p[0]);
			HVX_VectorPair wsop_hi = Q6_Wuw_vmpy_VuhVuh( vh0, exptab_p[1]);

			HVX_Vector vh1 = Q6_V_vlalign_VVR( v1,v0, vlalign_ctl);
			wsop_lo = Q6_Wuw_vmpyacc_WuwVuhVuh(wsop_lo, vh1, exptab_p[2]);
			wsop_hi = Q6_Wuw_vmpyacc_WuwVuhVuh(wsop_hi, vh1, exptab_p[3]);

			HVX_Vector vh2 = Q6_V_vlalign_VVR( v2,v1, vlalign_ctl);
			wsop_lo = Q6_Wuw_vmpyacc_WuwVuhVuh(wsop_lo, vh2, exptab_p[4]);
			wsop_hi = Q6_Wuw_vmpyacc_WuwVuhVuh(wsop_hi, vh2, exptab_p[5]);

			HVX_Vector vh3 = Q6_V_vlalign_VVR( v3,v2, vlalign_ctl);
			wsop_lo = Q6_Wuw_vmpyacc_WuwVuhVuh(wsop_lo, vh3, exptab_p[6]);
			wsop_hi = Q6_Wuw_vmpyacc_WuwVuhVuh(wsop_hi, vh3, exptab_p[7]);
			// add across even/odd
			sop_hi = Q6_Vw_vadd_VwVw(  Q6_V_hi_W(wsop_hi), Q6_V_lo_W(wsop_hi));
			sop_lo = Q6_Vw_vadd_VwVw(  Q6_V_hi_W(wsop_lo), Q6_V_lo_W(wsop_lo));
    	}
		// determine the batch index for the reduction tree
		// (this is adjusted so that of batches=11, the eff_batch will be 0..7, then 13,14,15)
		//
		int batch_offs = (ibat >=batch_offs_thr )? last_batch_offs:0;
		int eff_batch = ibat + batch_offs;

    	HVX_Vector ppsum;	// dot product, reduced and >>16, packed 8 per vector, but still lo:hi
    	{

			// We now have 2 partial products, each 32 x 32-bit, which need 'lateral' sums.
			// We assume that the depth <= 32K, which means that each partial prod will be <= 0x7FFF8000
			// even after adding
			HVX_VectorPair shuf = Q6_W_vshuff_VVR( sop_hi, sop_lo,4);	// interleave hi/lo
			ppsum = Q6_Vw_vadd_VwVw(  Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));			// and add them...

			// We now have 16 pairs { lo, hi}
			//
			// Reduce across batcheds in a group of 8
			// eff_batch%8 = 0,2,4,6:  store at pprev[0]
			//                1,  5:    combine with pprev[0] and store at pprev[1]
			//                  3:      combine with pprev[0], and with pprev[1] , store at pprev[2]
			//                  7:      combine with pprev[0], then pprev[1], pprev[3] for all 8.
			//
			if( (eff_batch & 1)==0){
				ppprev[0] = ppsum;
				continue;
			}
			shuf = Q6_W_vshuff_VVR( ppsum, ppprev[0], 8);
			ppsum= Q6_Vw_vadd_VwVw(  Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));	// add.
			if( (eff_batch & 2)==0){
				ppprev[1] = ppsum;
				continue;
			}
			shuf = Q6_W_vshuff_VVR( ppsum, ppprev[1], 16);
			ppsum= Q6_Vw_vadd_VwVw(  Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));	// add.
			if( (eff_batch & 4)==0){
				ppprev[2] = ppsum;
				continue;
			}
			shuf = Q6_W_vshuff_VVR( ppsum, ppprev[2], 32);
			ppsum= Q6_Vw_vadd_VwVw(  Q6_V_hi_W(shuf), Q6_V_lo_W(shuf));	// add.

			// in the last set of batches, the values will be offset by batch_offs
			// so we need to shift them down by 8*batch_offs.
			//
			// Now we have  8 pairs {lo,hi} followed by 8 pairs {lo,hi} and those need
			// adding across.. combine this with the batch_offs correction
			// This results in the same 8 pairs occurring in both halves.
			//
			ppsum = Q6_Vw_vadd_VwVw( Q6_V_vror_VR(ppsum,64 + 8*batch_offs),
							Q6_V_vror_VR(ppsum,8*batch_offs));
    	}
    	HVX_Vector dotprod;		// sop>>16
    	{
			// now we have full sums for 8 batches, each being 2 partial sums
			// find the full 64-bit sums...
			// In fact we know they fit in 48 bits, since the sum of the hist <= 64K.
			// So, in the process of doing this, we discard the lower & upper 16 bits
			// and just keep the middle 32.

			HVX_Vector losum = ppsum;		// lo sum in w lanes 0,2 ...
			HVX_Vector hisum = Q6_V_vror_VR( ppsum, 4);	// hi sum in those lanes
			losum = Q6_Vh_vshuffo_VhVh( Q6_V_vzero(), losum);	// losum >> 16
			dotprod = Q6_Vw_vadd_VwVw( hisum, losum );
    	}
    	// dotprod is valid in w lanes 0,2,4,..14; and repeats on lanes 16,18 ..30
    	//
		// find the reciprocals of those 8 values; put them in recip_store
    	{
			HVX_Vector shcount = Q6_Vuw_vcl0_Vuw(dotprod);				// expected to be 0..16
			HVX_Vector normsum = Q6_Vw_vasl_VwVw( dotprod, shcount);	// normalized to upper bit of each lane
			// the 'hvx_recip16_inline' needs a value in range 0x4000.. 0x7fff; so use a 16-bit 'unsigned average'
			// to >>1.
			HVX_Vector zero_in_odd_h_lanes = shcount;
			HVX_Vector recip_mant = hvx_recip16_inline( Q6_Vuh_vavg_VuhVuh(normsum,zero_in_odd_h_lanes),10);

			// now:
			// zero extend to w  (currently, recip is in the odd 'h' lane)
			//  << by 'shcount'
			//  >>15 with rounding and pack/sat to h
			//
			// it is possible to have shcount = 16 : if there is one large input, and all
			// the others are small and 'fall off the earth' (possible when input_range * beta > 22.5 or so)
			// In that case, though, 'recip' will always be 0x4000, and the procedure will give
			// us a final scale of 0x7FFF which will work.
			recip_mant = Q6_Vh_vshuffo_VhVh( zero_in_odd_h_lanes, recip_mant );		// zero extend to 32 bits
			recip_mant = Q6_Vw_vasl_VwVw( recip_mant, shcount );				// << shcount
			recip_mant = Q6_Vh_vasr_VwVwR_rnd_sat( recip_mant, recip_mant, 15);	// >> 15
			//
			// now each of w lanes 0,2, ... 14 has a duplicate pair { recip, recip }
			// Since the pattern is the same in the second half, we can do this:
			recip_mant = Q6_Vh_vshuff_Vh( recip_mant);
			// and we have the same situation but in w lanes 0,4,8 ... 28
			// store to the next 'recip_store' vector location
			// (eff_batch-7 is always a multiple of 8)
			*(HVX_Vector*)&recip_store[ 4 * (eff_batch-7)] = recip_mant;
    	}
    }
    // (this prefetch appears to make it run slower. Batches should be sized so that
    // data will remain in cache from histogram calc).
	//l2fetch(in_data, 128,128,(depth*batches+127)/128u);

    HVX_VectorPred q_zero = Q6_Q_vsetq_R(0);

    for( int ibat = 0; ibat  <batches; ibat++){
    	// recip is are stored in first vector of every 8th histo; each has 8 values with a stride of 16 bytes.
    	// each value is already replicated to both 16-bit halves of the 32  bits

		int maxh = max_h_store[ibat];			// 0..255
		int recip = recip_store[ibat*4];

		HVX_Vector sc0 = Q6_Vh_vmpy_VhRh_s1_rnd_sat( exp_tables[1], recip);
		HVX_Vector sc1 = Q6_Vh_vmpy_VhRh_s1_rnd_sat( exp_tables[2], recip);
		HVX_Vector sc2 = Q6_Vh_vmpy_VhRh_s1_rnd_sat( exp_tables[3], recip);
		// ordering we want is: first vec: 0,64,1,65, ...   63,127
		//                      second vec: 128,x,129 , ...  191,x
		HVX_Vector tbl0 = Q6_Vub_vasr_VhVhR_rnd_sat( sc1, sc0, 7);
		HVX_Vector tbl1 = Q6_Vub_vasr_VhVhR_rnd_sat( Q6_V_vzero(), sc2, 7);

		HVX_Vector voffs = q6op_Vb_vsplat_R( maxh-191);

		uint8_t const *inp = in_data + depth * ibat;
		uint8_t const *outp = out_data + depth * ibat;
		// # we assume that in and out have a common alignment
		unsigned algn = (size_t)inp & 127;
		// # of vecs in loop (includes partial vecs at start, end)
		int full_vecs = (algn+depth+127)/128u;
		HVX_Vector qstore_not = Q6_Q_vsetq_R(algn);
		HVX_Vector qstore_last = q6op_Q_vsetq2_R(algn+depth);
		// manually unpeeled loop... note that full_vecs >= 1 since depth >= 128
		{
			HVX_Vector  vin = *(HVX_Vector const*) inp;
			vin = Q6_Vb_vsub_VbVb( vin, voffs);
			inp += 128;
			HVX_Vector vout = q6op_Vb_vlut32_VbVbI( vin, tbl0,0);
			vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl0,1);
			vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl0,2);


			for( int i = 0; i < full_vecs-1; i++){
				HVX_Vector vin_next = *(HVX_Vector const*) inp;
				inp += 128;

				vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl0,3);
				vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl1,4);
				vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl1,5);
				q6op_vstcc_QnAV_nt( qstore_not,(HVX_Vector*)outp, vout);
				qstore_not = q_zero;
				outp += 128;

				vin = Q6_Vb_vsub_VbVb( vin_next, voffs);
				vout = q6op_Vb_vlut32_VbVbI( vin, tbl0,0);
				vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl0,1);
				vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl0,2);
			}
			vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl0,3);
			vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl1,4);
			vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, tbl1,5);
			q6op_vstcc_QAV( qstore_last,(HVX_Vector*)outp, vout);
			//outp += 128;
		}
    }

}


#if 0
/////////////////////////////////////////////
// function to test recip operation
/////////////////////////////////////////////
static void __attribute__((unused,noinline))
test_recip_function()
{
	static volatile int yet = 0;
	if( __sync_fetch_and_add( &yet, 1) != 0 )
		return;

	void *tbuf = nn_memalign(128, 16384*2);	// space for results
	HVX_Vector vi = *(HVX_Vector const*)const_Count128;
	vi = Q6_Vh_vasr_VhR( vi, 9 );		// { 0, 1, .. 63 } in h slots
	vi = Q6_Vh_vadd_VhVh( vi, q6op_Vh_vsplat_R(0x4000));	// 0x4000 .. 0x403f
	int prec = -17;
	for( int i = 0; i < 256; i++ ){	//all 16 inputs in 0x4000 .. 0x7fff
		HVX_Vector r = hvx_recip16_inline( vi, prec );
		((HVX_Vector*)tbuf)[i] = r;
		vi = Q6_Vh_vadd_VhVh( vi, q6op_Vh_vsplat_R(64));
	}
	// now check them...
	float maxerr =0.0f, maxexerr = 0.0f;
	double errsum = 0.0, e2sum = 0.0;
	for( int i= 0; i < 16384; i++){
		int dut_res = ((int16_t const*)tbuf)[i];
		float ref_res = 536870912.0f/(float)(i+16384);
		//printf(" %7d: %7d %.2f\n", i+16384, dut_res, ref_res);
		float err = dut_res - ref_res;
		float minerr = fabsf(roundf_i32( ref_res)-ref_res);
		maxexerr = fmaxf(maxexerr,  fabsf(err) - minerr );
		float relerr= err/ref_res;
		maxerr = fmaxf( maxerr, fabsf(relerr));
		errsum += err;
		e2sum += relerr*relerr;
	}
	float over_pop = 1.0f/16384.;

	printf("---- prec = %d-----------\n",prec);
	printf( "max excess err     =  %.4f q's\n", maxexerr );
	printf( "average error      =  %.4f q's\n",  (float)errsum * over_pop);
	printf( "max relative error =  %.5f * 1e-5\n", maxerr * 1e5f);
	printf( "rms relative error =  %.5f * 1e-5\n", sqrtf((float)e2sum * over_pop)*1e5f );
}
#endif


static int softmax_d32_free(struct nn_node *self, struct nn_graph *nn)
{
	if( self->opaque !=NULL){
		struct softmax_lookup_cache * cachep = (struct softmax_lookup_cache*)self->opaque;
		if( cachep->largedepth_tables != NULL) nn_free( cachep->largedepth_tables);
		nn_free( self->opaque);
		self->opaque = NULL;
	}
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8_d32 = {
	.execute =  softmax_d32_execute,
	.check =  NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(3,4),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8_d32_ref = {
	.execute = softmax_d32_execute,
	.check =  NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(3,4),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};


struct nn_node_ops nn_ops_for_QuantizedSoftmax_8 = {
	.execute =  softmax_flat_execute,
	.check =  NULL,
	.ctor = node_alloc_common,
	.dtor = softmax_d32_free,
	.n_inputs = NN_IOCOUNT_RANGE(3,4),
	.n_outputs = NN_IOCOUNT(3),
	.flags = 0
};

