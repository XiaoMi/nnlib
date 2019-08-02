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

//#define TEST_PERFORMANCE
//
// This contains an implementation of 'tanh' (and sigmoid) for 'd32' quantized
// tensors.
// The process in the inner loop is as follows:
// (1) get uint8 value, subtract zero offset; result is  -255 ..255
// (2) multiply by k, which is < 32k range; result fits in 24 bits
// (3) >>rsh and saturate to i16
//     this result is the 'application' result, scaled by 1.25 and then
//     expressed with 13 fractional bits, i.e. it's +/-3.2  in
//     'application' units (for tanh; for sigmoid it will have 12).
// (4) apply a 'core' operation which maps this to tanh (range is roughly +/-32k)
// (5) average with the midrange so it's 0..midrange
// (6) >>7, round, saturate to 0..255

//
//   'k' and 'rsh' are chosen to get the result of (3) with 12 fractional bits,
//  and to have k >= 16k (when possible; if the input range is tiny ( < .0128)
//   k will be < 16k, and rsh will be 15).
// The upper limit on range is about 800; this is the largest range that can
// be done with rsh = 0. Values above this can be safely mapped to the 'full' range
// since, based on the quantization step of about 3.1, it's equivalent to a
// sign function (i.e. with rsh = 0, k = 32767, codes above and below the zero
// point will map to +/- full-range results from step 3).
//
// Note that with a full output range of 255 steps, tanh(x) rounds to +/- 1.0 for |x| > 3.116,
// so the input range of +/- 3.2 does not lose anything.
//
//
// For tanh, output range is -1.0 ... 1.007874 (128/127) so that
//    the zero code is 127, and the max output code will be 254).
// for sigmoid, output range is 0.0 .. 1.0
//
//  The op in step 4 is done with segmented poly; for tanh it's scaled
//   to +/- 254*128, and for sigmoid it's scaled to +/- 255*128, so there
// are two different midranges applied at (5).
//
static const int16_t tanh_lut[64];
static const int16_t sigmoid_lut[64];

// this is attached to  the node, and caches the lut for a given min/max
struct lookup_table_cache {
	float minval, maxval;
	uint8_t lut[256];		// note: not vector aligned
};


struct tanh_scaling_parms {
	int16_t in_zero;		// the input 'zero point
	int16_t scalek;			// amount to multiply by
	int16_t rsh;			// amount to >>
	int16_t midrange;		// for step (5)
	int16_t const *polylut;	// for (4)
};

static inline void
set_scaling_parms( struct tanh_scaling_parms * sprm, float minval, float maxval, int is_sigmoid)
{
	float step = flt_div_255(maxval-minval);

	sprm->in_zero = saturate_u8( roundf_i32( -255.0f * minval/( maxval-minval)));
	float gain_needed = step * (float)(1.25*8192.0);	// this is gain in steps (2) and (3) combined
	if( is_sigmoid) gain_needed *= 0.5f;
	int gexp = flt_getexp( gain_needed);
	int rsh= 15-max_i32(gexp,0);
	//printf(" %f.. %f - zero = %d, gain = %g, rsh = %d\n", minval,maxval, sprm->in_zero, gain_needed,rsh);

	float kval;
	if( rsh < 0){		// super-wide input range; map to largest we can handle
		kval = 32767;
		rsh = 0;
	}else{
		kval = saturate_i16(roundf_i32( gain_needed *(float)( 1 <<rsh)));
	}
	sprm->scalek = kval;
	sprm->rsh = rsh;
	sprm->midrange = is_sigmoid ? 255*128: 254*128;
	sprm->polylut = is_sigmoid ?   sigmoid_lut:  tanh_lut;
}
struct tanh_run_state;
typedef void (*operate_func_fp)( struct tanh_run_state*,uint8_t *, uint8_t const *, int,int);

//
// 'flat' tensors are also processed by this code; each 'work unit'
// is TANH_FLATMODE_WORKUNIT_VECS in length. The last one may be short.
//
#define TANH_FLATMODE_WORKUNIT_VECS 64


struct tanh_run_state {
	struct shape shape;			// shape of operation
	struct tensor_addressing tin;
	struct tensor_addressing tout;
	uint32_t flat_total_bytes;		// for 'flat' tensors, total # of bytes
	volatile int next_job_index;
	int n_jobs;
	nn_sem_t done_sem;
	// when running hvx, the first thread builds the table, sets this
	// flag and then posts the table_built_sem. The main thread will
	// not launch other threads until the sem is posted.
	volatile int table_built;
	nn_sem_t table_built_sem;

	struct tanh_scaling_parms  sparms;
	uint8_t *runlut;		// where the pre-built LUT is (if hvx; null otherwise). Not vector aligned.
	operate_func_fp operate_func;
};

static void tanh_d32_run_thread( struct nn_graph * nn, void * rstpv);
static void tanh_flat_run_thread( struct nn_graph * nn, void * rstpv);
static  void  operate_ref_function(
		struct tanh_run_state * rstp,
		uint8_t *outp, uint8_t const * inp,	// 32-aligned
		 int depth, int d0);
static void operate_hvx_function(
				struct tanh_run_state * rstp,
				uint8_t *outp, uint8_t const * inp,	// 32-aligned
				 int depth, int d0);
static void make_tanh_lookup_table(  struct tanh_run_state * rstp );

static int tanh_d32_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);

#ifdef TEST_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif
	char const * opname = hexagon_nn_op_names[ self->node_type];
	int is_sigmoid= 0;
	int use_hvx = 1;
	int is_d32 = 1;
	switch( self->node_type){
	 case OP_QuantizedTanh_8_d32:
		break;
	 case OP_QuantizedTanh_8_d32_ref:
		use_hvx = 0;
		break;
	 case OP_QuantizedSigmoid_8_d32:
	    is_sigmoid= 1;
		break;
	 case OP_QuantizedSigmoid_8_d32_ref:
	    is_sigmoid= 1;
		use_hvx = 0;
		break;
	 case OP_QuantizedTanh_8:
		is_d32 = 0;
		break;
	 case OP_QuantizedSigmoid_8:
	    is_sigmoid= 1;
		is_d32 = 0;
		break;

	 default:
		 return errlog(nn,"bad node_type= %d", self->node_type);
	}
	logmsg(nn,2,"%s execute. self=%p ",opname,self);

	struct tanh_run_state rstt;
	// get cache
	// this is only non-null if use_hvx = 1
	struct lookup_table_cache *cachep = (struct lookup_table_cache *)self->opaque;
	// avoid the scaling calc if the cache is ok
	int cached_table_ok = use_hvx && (in_min == cachep->minval && in_max == cachep->maxval );
	if( !cached_table_ok)
		set_scaling_parms( &rstt.sparms, in_min, in_max, is_sigmoid );

	// set up the runstate for input tensor
	rstt.shape = in_tensor->shape;

	int njobs;
	if( is_d32){

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
		njobs = b*nd32;

		int d_pad_after = nd32*32 - (d_pad_before + d);
		if( tensor_out_prepare_padded_d32( out_tensor, b,
				h,  h_pad_before,  h_pad_after,
				w,  w_pad_before,  w_pad_after,
				d,  d_pad_before,  d_pad_after,
				NN_TYPE_QUINT8) != 0 ){
			return errlog(nn,"out too small");
		}
		rstt.tout = tensor_addressing_d32(out_tensor);

	}else{	// FLAT tensors.
		rstt.flat_total_bytes = tensor_element_count(in_tensor);
		nn_tensor_out_prepare_normal_fromshape(out_tensor,&rstt.shape, NN_TYPE_QUINT8);
		njobs = (rstt.flat_total_bytes  + TANH_FLATMODE_WORKUNIT_VECS*128-1)/(unsigned)(TANH_FLATMODE_WORKUNIT_VECS*128);
		rstt.tin.data = in_tensor->data;
		rstt.tout.data = out_tensor->data;
	}
	// set up the rest of the run state
	rstt.operate_func = use_hvx? operate_hvx_function: operate_ref_function;
	rstt.runlut = use_hvx? (uint8_t *) &cachep->lut :NULL;

	rstt.next_job_index = 0;
	rstt.n_jobs = njobs;
	nn_sem_init( &rstt.done_sem, 0);

	int table_built = 1;
	if( use_hvx && !cached_table_ok){
		// rebuild table if use_hvx and if min, max have changed
		cachep->minval = in_min;
		cachep->maxval = in_max;
		table_built = 0;
		nn_sem_init( &rstt.table_built_sem,0);
	}
	rstt.table_built = table_built;
	int nthreads = min_i32( rstt.n_jobs, 3);

	void (*thread_run_fp)( struct nn_graph * nn, void * rstpv) =  is_d32? tanh_d32_run_thread:tanh_flat_run_thread;

	// launch first thread
	nn_os_work_for_vector(nn, thread_run_fp , &rstt);
	if( !table_built){
		// hold off until table built.
		nn_sem_wait( & rstt.table_built_sem);
	}
	// launch the rest
	for( int i =1; i < nthreads; i++){
		nn_os_work_for_vector(nn,thread_run_fp  , &rstt);
	}

	float out_min, out_max;
	if( is_sigmoid){
		out_min = 0.0f;
		out_max = 1.0f;
	}else{
		out_min = -1.0f;	// - 127/127
		out_max = (float)(128./127.);
	}
	tensor_set_single_float( out_min_tensor, out_min);
	tensor_set_single_float( out_max_tensor, out_max);

	nn_sem_wait_n_times( & rstt.done_sem, nthreads);

#ifdef TEST_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsigmoid/tanh_d32 %s cycles = %d (elements = %d)\n",
			use_hvx?"hvx": "ref",(end_time-start_time),
			(int)tensor_element_count(in_tensor));
#endif
	logmsg(nn,2,"%s done. self=%p ",opname,self);

	return 0;
}

static void
tanh_d32_run_thread( struct nn_graph * nn, void * rstpv)
{
	struct tanh_run_state *rstp = (struct tanh_run_state *)rstpv;

	int job_index;
	int d = rstp->shape.depth;
	int nd32 = rstp->tin.nd32;
	int d0 = rstp->tin.d0;

	int pfheight = rstp->shape.height;	// these vars are so that the ..
	int pfwid = rstp->shape.width*32;	// .. prefetch 'shape' value can be hoisted..
	int pfstride = rstp->tin.height_stride; // .. out of the loop

	if( rstp->table_built == 0){	// need to build lut
		make_tanh_lookup_table(rstp);
		rstp->table_built = 1;
		nn_sem_post( & rstp->table_built_sem);
		/*for( int i = 0; i < 256; i++){
			int pos = (i&128)| ((i & 63)<<1) | ((i>>6)&1);
			printf(" %3d->%3d\n", i, rstp->runlut[pos]);
		}*/
	}
	operate_func_fp funcp = rstp->operate_func;
	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode,nd32);
	int njobs = rstp->n_jobs;

	while( job_index = __sync_fetch_and_add( &rstp->next_job_index, 1), job_index < njobs ){
		int id32 = batchslice_decode_update( &bsdecode, job_index);
		int ib = bsdecode.ibatch;
		uint8_t const * in_ptr = rstp->tin.data + ib*rstp->tin.batch_stride + id32*rstp->tin.d32_stride;
		l2pref( in_ptr, pfheight, pfwid, pfstride );
		uint8_t * out_ptr = rstp->tout.data + ib*rstp->tout.batch_stride + id32*rstp->tout.d32_stride;
		// range of depth to process here...
		int d_start = id32==0? d0: 0;
		int dn = min_i32(d0 + d - id32*32, 32)-d_start;

		(*funcp)( rstp, out_ptr, in_ptr, dn, d_start);
	}
	nn_sem_post( & rstp->done_sem);
}

static inline HVX_Vector
hvx_table_lookup_u8( HVX_Vector vin, HVX_Vector lut0, HVX_Vector lut1)
{
	HVX_Vector vout =  q6op_Vb_vlut32_VbVbI( vin,lut0, 0 );
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, lut0,1);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, lut0,2);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, lut0,3);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, lut1,4);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, lut1,5);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, lut1,6);
	vout = q6op_Vb_vlut32or_VbVbVbI( vout, vin, lut1,7);
	return vout;
}

//
// flat mode is like d32 mode except
//  - no 'reference' operator
//  - each 'work unit' is TANH_FLATMODE_WORKUNIT_VECS vectors (the last one can be smaller,
//   and may not be a full vector.
//

static void
tanh_flat_run_thread( struct nn_graph * nn, void * rstpv)
{
	struct tanh_run_state *rstp = (struct tanh_run_state *)rstpv;

	int job_index;

	if( rstp->table_built == 0){	// need to build lut
		make_tanh_lookup_table(rstp);
		rstp->table_built = 1;
		nn_sem_post( & rstp->table_built_sem);
	}
	uint32_t total_len = rstp->flat_total_bytes;
	uint8_t const  *inp0 = rstp->tin.data;
	uint8_t *outp0 = rstp->tout.data;

	HVX_Vector vlut0 = q6op_V_vldu_A( (HVX_Vector const *)rstp->runlut);
	HVX_Vector vlut1 = q6op_V_vldu_A( (HVX_Vector const *)(rstp->runlut +128));
	int njobs = rstp->n_jobs;

	while( job_index = __sync_fetch_and_add( &rstp->next_job_index, 1), job_index < njobs ){
		uint32_t offs = job_index * (TANH_FLATMODE_WORKUNIT_VECS*128);
		uint8_t const  *inp = inp0+offs;
		int32_t bytes = min_i32( (TANH_FLATMODE_WORKUNIT_VECS*128), total_len - offs);
		int nvecs = (bytes+127)>>7;		// # of vectors to process (>=1)
		if( nvecs < 1 ) break;			// shouldn't happen
		l2pref( inp, /*height*/nvecs , /*width*/ 128, /*stride*/128 );

		HVX_VectorPred qlast = q6op_Q_vsetq2_R(bytes);	// mask for last operation
		HVX_Vector const *pvin = (HVX_Vector const *)inp;
		HVX_Vector  *pvout = (HVX_Vector  *)(outp0 + offs);
		HVX_Vector v = *pvin++;
		HVX_Vector vout = hvx_table_lookup_u8( v, vlut0, vlut1);
		for( int i  =0 ; i <nvecs-1; i++){
			*pvout++ = vout;
			v = *pvin++;
			vout = hvx_table_lookup_u8( v, vlut0, vlut1);
		}
		q6op_vstcc_QAV(qlast,pvout,vout);
	}
	nn_sem_post( & rstp->done_sem);
}

// series of 2nd ord poly funcs for tan(x), x= 0..3.2 in 16 intervals.
// result is scaled to range 0..254*128 for tanh, and
// 0..255*128 for sigmoid (other than scaling, they are the same).
//
// the layout is:
//   slot 0   (0,2, .. 30) contains  2nd order terms
//   slot 1   (32,34.. 62) contains  1st order terms
//   slot 2   (1,3,    31) contains  0 order terms
//
// it's important that the last segment is able to reach 32386
// so that for extreme input values we round to 0,254.
// for sigmoid, it needs to reach 32514  to get 0..255
// These fits are constrained at the endpoints so they get to 32404
// and 32532 resp, which are ideal for tanh(3.2)
//
// these are scaled for use as
//   out = c[0] + (x*4)*( c[1] +  x*c[2])
//  where x is a residual 11 bit offset (0..2047)
// and the multiplies are 16-bit fractional muls (/32768)
//
//
static const int16_t tanh_lut[64] __attribute__((aligned(128))) = {
    -7994,      0, -21652,   6418, -29653,  12354, -31481,  17462,
   -28797,  21590, -23876,  24761, -18511,  27104, -13703,  28784,
    -9826,  29965,  -6894,  30782,  -4766,  31342,  -3262,  31723,
    -2217,  31981,  -1500,  32155,  -1011,  32272,   -677,  32351,
    26172,      0,  25098,      0,  22283,      0,  18480,      0,
    14485,      0,  10862,      0,   7879,      0,   5579,      0,
     3883,      0,   2671,      0,   1822,      0,   1236,      0,
      835,      0,    563,      0,    378,      0,    254,      0
};
static const int16_t sigmoid_lut[64] __attribute__((aligned(128))) = {
    -8026,      0, -21738,   6443, -29770,  12403, -31605,  17531,
   -28910,  21675, -23970,  24859, -18584,  27210, -13757,  28898,
    -9865,  30083,  -6921,  30904,  -4785,  31466,  -3275,  31848,
    -2226,  32107,  -1506,  32282,  -1015,  32399,   -680,  32479,
    26275,      0,  25197,      0,  22371,      0,  18553,      0,
    14542,      0,  10905,      0,   7910,      0,   5601,      0,
     3899,      0,   2681,      0,   1829,      0,   1240,      0,
      838,      0,    565,      0,    380,      0,    255,      0
};



static  void
operate_ref_function(
		struct tanh_run_state * rstp,
		uint8_t *outp, uint8_t const * inp,	// 32-aligned
		 int depth, int d0)
{

	int zval = rstp->sparms.in_zero;
	int kval = rstp->sparms.scalek;
	int rsh = rstp->sparms.rsh;
	int midpoint = rstp->sparms.midrange;
	int rbias = (1<<rsh)>>1;
	int16_t const * lut = rstp->sparms.polylut;

	int width = rstp->shape.width;
	int height = rstp->shape.height;
	int in_row_stride = rstp->tin.height_stride;
	int out_row_stride = rstp->tout.height_stride;

	if( depth == 32){
		depth= width*32;
		width = 1 + 0*sigmoid_lut[0];
	}
	inp += d0;
	outp += d0;
	for(int ih = 0; ih < height; ih++){
		for( int iw = 0; iw < width; iw++){
			uint8_t const *pin = inp + iw*32;
			uint8_t  *pout = outp + iw*32;
			for(int i= 0; i < depth; i++){
				int d = ( pin[i]-zval);
				int p = kval * (d < 0? -d:d);
				p = saturate_i16( (p +rbias)>>rsh);		// 0 .. 32767
				int tindex = (p>>11) & 15;				// 4-bit table index
				int tfrac = p & 0x7FF;					// resid.x
				int16_t const * tptr = &lut[2*tindex];
				// 0,1,2 order are at: +1, +32, +0
				int val =  ((tptr[0]*tfrac + 16384)>>15) + tptr[32];
				val = ((val*tfrac*4 + 16384) >> 15) + tptr[1];
				if( d < 0) val = ~val;
				val = (val + midpoint+1)>>1;	// offset the range to 0.. midpoint
				val = (val+64)>>7;
				pout[i] = saturate_u8(val);
			}
		}
		outp += out_row_stride;
		inp += in_row_stride;
	}
}

#if 1 // hard hat zone
//
// some inlines for the hvx version
//
// read in a vector of values, then:
//    -subtract offset to get d (16 bit)
//    -multiply abs(d) by kgain, and then >>rsh with rounding and saturation.
//     This will be in range 0..32767.
//  There are three return vectors: two for the result and one  for 'dsign' (0x00 or 0xFF)
//
struct first_stage_result {
	HVX_Vector x0,x1;
	HVX_Vector dsign;
};

static inline struct first_stage_result
tanh_first_stage(  HVX_Vector vin, int offs, int kgain, int rsh )
{
	kgain = Q6_R_combine_RlRl( kgain , kgain);
	HVX_Vector voffs = q6op_Vb_vsplat_R(offs);
	// find abs(in-offs)
	HVX_Vector absd = Q6_Vub_vabsdiff_VubVub( vin, voffs);
	// this is the 'sign'...
	HVX_VectorPred dneg = Q6_Q_vcmp_gt_VubVub( voffs, vin);

	// do the products...
	HVX_Vector absd_0 = Q6_Vb_vshuffe_VbVb( Q6_V_vzero(), absd);
	HVX_Vector absd_1 = Q6_Vb_vshuffo_VbVb( Q6_V_vzero(), absd);

	HVX_VectorPair prod0 = Q6_Ww_vmpy_VhRh( absd_0, kgain);
	HVX_VectorPair prod1 = Q6_Ww_vmpy_VhRh( absd_1, kgain);
	// >> and saturate. These are guaranteed to be in range 0...32767
	struct first_stage_result result;
	result.x0 = q6op_Vh_vasr_WwR_rnd_sat( prod0, rsh);
	result.x1 = q6op_Vh_vasr_WwR_rnd_sat( prod1, rsh);
	result.dsign = Q6_V_vand_QR( dneg, -1);	// signs from d
	return result;
}

// second stage.
//
static inline HVX_Vector
tanh_second_stage( struct first_stage_result fstg , HVX_Vector tbl, int midrange)
{
	HVX_Vector vmid = q6op_Vh_vsplat_R(midrange);
	// obtain table index, by extracting hi bytes from the
	// x0,x1, and then >> 3.
	HVX_Vector tind = Q6_Vb_vshuffo_VbVb( fstg.x1, fstg.x0);

#if __HEXAGON_ARCH__ >= 62
	tind = Q6_Vub_vlsr_VubR( tind, 3);	// >>3 bytes

	// use all three as they sit in the one vector
	HVX_VectorPair coeff2 = Q6_Wh_vlut16_VbVhR_nomatch( tind, tbl,0);
	HVX_VectorPair coeff1 = Q6_Wh_vlut16_VbVhR_nomatch( tind, tbl,1);
	HVX_VectorPair coeff0 = Q6_Wh_vlut16_VbVhR_nomatch( tind, tbl,2);
#else
	tind = Q6_Vuh_vlsr_VuhR( tind, 3);
	tind = Q6_V_vand_VV( tind, Q6_V_vsplat_R( 0x0F0F0F0F));

	HVX_Vector tbl1= Q6_V_vror_VR( tbl,64);	// second section
	HVX_Vector tbl2= Q6_Vh_vshuffo_VhVh( tbl, tbl);	// third section
	HVX_VectorPair coeff2 = Q6_Wh_vlut16_VbVhR( tind, tbl,0);
	HVX_VectorPair coeff1 = Q6_Wh_vlut16_VbVhR( tind, tbl1,0);
	HVX_VectorPair coeff0 = Q6_Wh_vlut16_VbVhR( tind, tbl2,0);
#endif
	// extract the lower 11 bits of the values
	HVX_Vector tx0 = Q6_V_vand_VV( fstg.x0, Q6_V_vsplat_R( 0x07FF07FF));
	HVX_Vector tx1 = Q6_V_vand_VV( fstg.x1, Q6_V_vsplat_R( 0x07FF07FF));
	// mul tx by the 2nd order
	HVX_Vector poly_0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(tx0, Q6_V_lo_W(coeff2));
	HVX_Vector poly_1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(tx1, Q6_V_hi_W(coeff2));
	// add the 1st order term..
	HVX_VectorPair poly = Q6_Wh_vadd_WhWh( coeff1, Q6_W_vcombine_VV( poly_1, poly_0));
	// we now need to mul by tx, but we need to  << x by 2
	poly_0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat( Q6_Vh_vasl_VhR(tx0,2), Q6_V_lo_W(poly));
	poly_1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat( Q6_Vh_vasl_VhR(tx1,2), Q6_V_hi_W(poly));
	// now add the 0th order term
	poly = Q6_Wh_vadd_WhWh_sat( coeff0, Q6_W_vcombine_VV( poly_1, poly_0));
	// now we need to change the sign if d < 0
	// this is done with a ~
	HVX_VectorPair dsign = Q6_Wb_vshuffoe_VbVb( fstg.dsign, fstg.dsign);

	HVX_Vector tanh_0 = Q6_V_vxor_VV( Q6_V_lo_W(dsign),Q6_V_lo_W(poly) );
	HVX_Vector tanh_1 = Q6_V_vxor_VV( Q6_V_hi_W(dsign),Q6_V_hi_W(poly) );
	// ok, that's signed tanh result. Need to average it with midpoint
	tanh_0 = Q6_Vh_vavg_VhVh_rnd( tanh_0, vmid);
	tanh_1 = Q6_Vh_vavg_VhVh_rnd( tanh_1, vmid);
	// scale to range 0..255
	return Q6_Vub_vasr_VhVhR_rnd_sat( tanh_1, tanh_0,7);
}

//
// This function builds a full lookup table for the operation defined
// in the runstate.
//
static void
make_tanh_lookup_table(  struct tanh_run_state * rstp )
{
	// for input we need a vector { 0, 64, 1, 65, ... 63, 127}
	// and another which is 128 greater.
	// If we take const_Count128 =  { 0, 1, .. 127}
	// or as int16's,  { 0x100, 0x302, 0x504, .. 0x7f7e}
	// and average it with 0x7f00 we get { 0x4000, 0x4101, .. 0x7f3f}
	// which is the first input.
	uint8_t * pout = rstp->runlut;	// 256 bytes, unaligned

	HVX_Vector count128 = *(HVX_Vector const *)const_Count128;
	HVX_Vector in0 = Q6_Vh_vavg_VhVh(count128, q6op_Vh_vsplat_R(0x7f00));
	HVX_Vector in1 = Q6_V_vxor_VV( in0,q6op_Vb_vsplat_R(0x80) );
	int offs = rstp->sparms.in_zero;
	int kgain = rstp->sparms.scalek;
	int rsh = rstp->sparms.rsh;
	int midrange = rstp->sparms.midrange;
	HVX_Vector vtbl = *(HVX_Vector const *)rstp->sparms.polylut;

	struct first_stage_result fstg0 = tanh_first_stage( in0, offs, kgain, rsh);
	struct first_stage_result fstg1 = tanh_first_stage( in1, offs, kgain, rsh);

	HVX_Vector vlut0 =  tanh_second_stage( fstg0, vtbl, midrange);
	HVX_Vector vlut1  = tanh_second_stage( fstg1, vtbl, midrange);
	q6op_vstu_AV( (HVX_Vector *)pout, vlut0);
	q6op_vstu_AV( (HVX_Vector *)(pout+128), vlut1 );

}

//
// The 'operate' function is just a brute_force u8->u8 mapping
//
// If the height >= 2, and all of the depth fits in the first half,
// we do two rows at once to make more efficient use of the vectors.
//

static void operate_hvx_function(
				struct tanh_run_state * rstp,
				uint8_t *outp, uint8_t const * inp,	// 32-aligned
				 int depth, int d0)
{
	int ht = rstp->shape.height;
	int wid = rstp->shape.width;
	int in_height_stride = rstp->tin.height_stride;
	int out_height_stride = rstp->tout.height_stride;
	HVX_Vector lut0,lut1;

	lut0 = q6op_V_vldu_A( (HVX_Vector const *)rstp->runlut);
	lut1 = q6op_V_vldu_A( (HVX_Vector const *)(rstp->runlut +128));

	// offset for width padding
	int wpad = (int)(size_t)outp & 127;
	wid += wpad >> 5;
	outp -= wpad;		// now aligned
	inp -= wpad;		// hope it's aligned
	int nvec_wide = (wid + 3)>>2;	// # of vecs in h loop

	// if all the active 'depth' elements are in the first half,
	// and ht >= 2, do two rows at once, packing to one vector, for more efficiency.
	if( ht >= 2 && (depth+d0) <=16){
		int nhpair = ht>>1;
		for( int ih = 0; ih < nhpair; ih++){
			HVX_Vector const * vinp0 =  (HVX_Vector const *) inp;
			HVX_Vector const * vinp1 =  (HVX_Vector const *)( inp + in_height_stride);
			HVX_Vector * voutp0 =  (HVX_Vector  *) outp;
			HVX_Vector * voutp1 =  (HVX_Vector  *)( outp + out_height_stride);
			for(int iw = 0; iw <nvec_wide; iw++){
				// shuffle vectors from two rows together, keeping 1st 16 from each
				// depth slot.
				HVX_Vector vin = Q6_V_lo_W( Q6_W_vshuff_VVR(vinp1[iw], vinp0[iw],16));
				HVX_Vector vout = hvx_table_lookup_u8( vin, lut0, lut1);
				voutp0[iw] = vout;	// first result
				voutp1[iw] = Q6_V_vror_VR( vout, 16);	//second result
			}
			inp += in_height_stride *2;
			outp += out_height_stride *2;
		}
		ht = ht & 1;	// maybe one odd row left
	}

	for(int ih = 0; ih < ht; ih++){
		HVX_Vector const * vinp =  (HVX_Vector const *)( inp + in_height_stride * ih);
		HVX_Vector * voutp =  (HVX_Vector  *)( outp + out_height_stride * ih);
		for(int iw = 0; iw <nvec_wide; iw++){
			HVX_Vector vin = vinp[iw];
			voutp[iw]= hvx_table_lookup_u8( vin, lut0, lut1);
		}
	}
}

#endif


static int tanh_d32_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking tanh_d32 node %p",self);
	// allocate space for the table cache
	void * t = nn_calloc(1, sizeof(struct lookup_table_cache ));
	if( t == NULL)return errlog(nn,"can't alloc %d bytes",(int)sizeof(struct lookup_table_cache ) );
	self->opaque = t;
	((struct lookup_table_cache*)t)->maxval = -999.0f;	// anything impossible, to invalidate

	logmsg(nn,2,"tanh_d32 node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedTanh_8_d32 = {
	.execute = tanh_d32_execute,
	.check = tanh_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedTanh_8_d32_ref = {
	.execute = tanh_d32_execute,
	.check = NULL, //tanh_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedSigmoid_8_d32 = {
	.execute = tanh_d32_execute,
	.check = tanh_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_QuantizedSigmoid_8_d32_ref = {
	.execute = tanh_d32_execute,
	.check = NULL,//tanh_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

///
/// flat mode
///
struct nn_node_ops nn_ops_for_QuantizedTanh_8 = {
	.execute = tanh_d32_execute,
	.check = tanh_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3)
};

struct nn_node_ops nn_ops_for_QuantizedSigmoid_8= {
	.execute = tanh_d32_execute,
	.check = tanh_d32_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(3),
	.n_outputs = NN_IOCOUNT(3)
};



