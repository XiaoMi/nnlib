/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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
#if defined(__hexagon__)
#include "hvx_inlines.h"
#include "hvx_mathops.h"
#endif

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
//         (2c)  m >>= int(xx)   (ensuring it's zero if int(xx) >= 16)
//         (2d)  keep the m[i] for step (3); and find the sum of all the m[i]
// (3) find a reciprocal of sum(m[i]) :    recip*sum(m[i]) = (255 << rsh)
// (4)  for each i across depth dimension:
//         (4a)  output = m[i]*recip >> rsh
//
// This operation is repeated on slices of (1,1,4,d), so each slice does 4 separate ops.
// (so, for instance, the 'reciprocal' is vectorized, but only 4 distinct values are
// in the vector).
//
// The 'max' in (a) and sum(m(i])) in (2d) need to be masked to eliminate 'depth padding' lanes

struct softmax_scaling_parms
{
	int betamul;
	int beta_rsh;
};
static inline int
set_scaling_parms( struct softmax_scaling_parms * sparms, float inmin, float inmax, float beta)
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
	int eff_width;				// width when 'rounded out' to 4 boundaries at each end
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	nn_sem_t done_sem;
	struct softmax_scaling_parms  sparms;
	operate_func_fp operate_func;
	struct softmax_thrinfo {
		struct softmax_run_state *stt; 	// all point to containin struct
		int h0;							// start at this h dimension
		int hcount;						// run this many h
		int16_t * workbuf;
	} thrinfo[MAX_THREADS];
};

static void softmax_d32_run_thread( struct nn_graph * nn, void * tinfov);

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

	if( set_scaling_parms( &rstt.sparms, in_min, in_max, beta) != 0){
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
	int nd32 = rstt.tin.nd32;
//printf("shape = %d %d %d %d\n", b,h,w,d);
	int use_hvx = 0;
	// currently the hvx code needs d_pad_before = 0.
	if( d_pad_before == 0 && self->node_type == OP_QuantizedSoftmax_8_d32){
		use_hvx = 1;
	}

	int d_pad_after = nd32*32 - (d_pad_before + d);
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
//
//
static void
softmax_operate_ref_function(
		struct softmax_run_state *rstp,
		uint8_t *data_out,						// vec aligned
		uint8_t const *data_in,					// vec aligned
		int height,				// # of rows processed (could be < rstp->shape.height)
		int16_t *workbuf )		// vec aligned, width*depth_total*int16

/*		int height,
		int width,								// must be mul of 4
		int in_height_stride, in_d32_stride,	// vec aligned
		int out_height_stride, out_d32_stride,	// vec aligned
		int depth,d0,							// depth slice */
{
	int width = rstp->eff_width;
	int in_height_stride = rstp->tin.height_stride;
	int in_d32_stride = rstp->tin.d32_stride;
	int out_height_stride = rstp->tout.height_stride;
	int out_d32_stride = rstp->tout.d32_stride;
	int d0 = rstp->tin.d0;
	int depth = rstp->shape.depth;
	int nd32 = rstp->tin.nd32;

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
					if( xexp >= 16){
						mval = 0;
					}else{
						// find mant = 2*^-xfrac
						int mant = ref_powm2_op( xfrac>>1);
						mval = mant >> xexp;
						msum += mval;
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
			int recip_m = ref_fracdivide_10bit_Rh_RhRh( 255 << 6, m_norm);
			int rsh = 38-max_i32(nx,7);			// 21..31
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
////////////////////////////// HVX CODE ///////////////////////////////////

//
// the 'pow2( scl*x)' portion of the operation:
//  - vin is 128 input bytes
//  - vinmax is 'maxin' values (these are replicated within each 32-bit section)
//  - betamul and beta_rsh are the beta scaling parms.
//
// return is 128 16-bit values, pow2( -(xmax-x)*beta)
//

static inline HVX_VectorPair hvx_pow2_func( HVX_Vector vin, HVX_Vector vinmax, int betamul, int beta_rsh)
{
	betamul = Q6_R_combine_RlRl( betamul, betamul);
	HVX_Vector vconsth_0FFF = q6op_Vh_vsplat_R(0x0FFF);
	HVX_Vector vconsth_000F = q6op_Vh_vsplat_R(0x000F);
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

	// clip the shift amount to a max of 15
	int0 = Q6_Vh_vmin_VhVh( int0, vconsth_000F );
	int1 = Q6_Vh_vmin_VhVh( int1, vconsth_000F );
	// now >> the pow2 values by that amount
	pow2_0 = Q6_Vh_vasr_VhVh( pow2_0, int0);
	pow2_1 = Q6_Vh_vasr_VhVh( pow2_1, int1);
	// that's the result. All are <= 32767
	//
	return Q6_W_vcombine_VV( pow2_1,pow2_0);
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
static inline HVX_Vector
final_scaling_op( HVX_Vector m0, HVX_Vector m1 , HVX_Vector recip_mant,  HVX_Vector recip_sh )
{
	HVX_VectorPair prod0 = Q6_Ww_vmpy_VhVh( m0, recip_mant);	// full 32 bit products
	HVX_VectorPair prod1 = Q6_Ww_vmpy_VhVh( m1, recip_mant);

	// now do the variable shift, and a fixed shift of 12, and saturate to 8.

	HVX_Vector scaled0 = Q6_Vh_vasr_VwVwR_sat(
			Q6_Vw_vasr_VwVw( Q6_V_hi_W(prod0), recip_sh ),
			Q6_Vw_vasr_VwVw( Q6_V_lo_W(prod0), recip_sh ), 12);
	HVX_Vector scaled1 = Q6_Vh_vasr_VwVwR_sat(
			Q6_Vw_vasr_VwVw( Q6_V_hi_W(prod1), recip_sh ),
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
			HVX_Vector m0 = Q6_V_vzero();
			HVX_Vector m1 = Q6_V_vzero();

			// accumulation of mvals to msum is done at the top of the loop,
			// so we can mask the final acc.

			for( int  i = 0; i < nd32; i++ ){
				// add m0,m1  to 'msum'; we can safely add them together as uh
				// since all are < 32768
				HVX_Vector tsum = Q6_Vh_vadd_VhVh( m0, m1);  //  4 x { 16 x u16 }
				HVX_VectorPair tsumx = Q6_Wh_vshuffoe_VhVh( Q6_V_vzero(), tsum ); // zero extend
				// add in pairs, we now have 4 x { 8 x i32 }
				tsum = Q6_Vw_vadd_VwVw( Q6_V_lo_W(tsumx), Q6_V_hi_W(tsumx));
				msum = Q6_Vw_vadd_VwVw( msum, tsum );		// 4 x { 8 x i32 } on the whole thing.

				HVX_Vector x = *(HVX_Vector const *)( pin0 + i*in_d32_stride);
				HVX_VectorPair mvals = hvx_pow2_func( x, xmax, bscale, brsh);
				m0 = Q6_V_lo_W(mvals);	// even lanes
				m1 = Q6_V_hi_W(mvals);	// odd lanes
				wbuf[0] = m0;		// save for later
				wbuf[1] = m1;
				wbuf += 2;
			}
			// add the last set..
			// but first do right side depth masking, while adding m0 & m1
			{
				HVX_Vector tsum = q6op_V_vand_QV( qfinalmask_h0, m0);
				tsum =  Q6_Vh_condacc_QVhVh( qfinalmask_h1, tsum,m1); //  4 x { 16 x u16 }
				HVX_VectorPair tsumx = Q6_Wh_vshuffoe_VhVh( Q6_V_vzero(), tsum ); // zero extend
				// add in pairs, we now have 4 x { 8 x i32 }
				tsum = Q6_Vw_vadd_VwVw( Q6_V_lo_W(tsumx), Q6_V_hi_W(tsumx));
				msum = Q6_Vw_vadd_VwVw( msum, tsum );		// 4 x { 8 x i32 } on the whole thing.

			}
			// OK, now have 4 { 8 x i32 } sums. Need h sums within those groups
			//
			msum = reduce_in_quadrants_vadd_Vw( msum );

			// now, we really only have 4 values in the 'msum' vector, but 8 copies
			// of each. We need to find reciprocals.
			HVX_Vector normsh =  Q6_Vw_vnormamt_Vw(msum);		// at most 16; very likely <=6
			msum = Q6_Vw_vasl_VwVw( msum, normsh);				// each is now '01' in the upper 2 bits
			msum = Q6_Vh_vshuffo_VhVh(msum,msum);				// discard 16 lower bits; dup the upper
			// this is the value we will multiply by:
			HVX_Vector recip_mant = hvx_fracdivide_10bit_Vh_VhVh( q6op_Vh_vsplat_R(255<<6), msum);

			// the variable rsh will be 19-max_i32(normsh,6); in range 3 .. 13
			// done as 13-max(0, normsh-6)
			normsh = Q6_Vuh_vsub_VuhVuh_sat( normsh, q6op_Vh_vsplat_R(6));	// max(0, normsh-6)
			HVX_Vector recip_sh = Q6_Vh_vsub_VhVh( q6op_Vh_vsplat_R(13), normsh );
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

static int softmax_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	int k;
	logmsg(nn,2,"Checking softmax_d32 node %p",self);

	k = node_check_inputs_range( self,nn, "softmax_d32", 3,4);
	if( k==0) k = node_check_outputs_n( self,nn, "softmax_d32", 3);
	if( k!= 0) return k;

	logmsg(nn,2,"tanh_d32 node %p check OK",self);
	return 0;
}


struct nn_node_ops nn_ops_for_QuantizedSoftmax_8_d32 = {
	.execute =  softmax_d32_execute,
	.check =  softmax_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8_d32_ref = {
	.execute = softmax_d32_execute,
	.check =  softmax_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};




