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
#include "hvx_inlines.h"

//
// QuantizedLstmInput_16x16to16
//
// This reads two 'flat' tensors, both i16
//  the first has shape [b,h,w,din1], the second [b,h,w,din2]
//  The output is an i16 tensor [b,h,w,dout]  dout = din2
//
// The calculation is:
//     out = Tanh( a ) * Logistic( b )  + Logistic(c) * d
// ... where:
//     a,b,c are all extracted from the first input, at specified depth positions
//     d is from the second input.
//
//
//
//  Inputs:
//    input 0:   array [1,1,1,n] of int32: configuration
//    input 1:   first input array [b,h,w,din1]   (i16 tensor)
//    input 2:   second input array: [b,w,din2]   (i16 tensor)
//    input 3,4:  min of first, min of second input
//    input 5,6:  max of first, max of second input.
//  Outputs:
//    output 0:  output [b,h,w,dout]   (i16 tensor)
//	  output 1,2:  output min/max (same as second input).
//
// (the inputs are arranged as in 'concat' node, to allow future expansion of number of inputs).
//
// It is assumed that the ranges will be symmetric; the 'min' value corresponds to a -32768 output
//  and max = -min correponds to +32768 (not 32767).
//
//
//  configuration is an array of int32:
//       +0     - output depth.
//       +1     - mode (only '0' currently supported).
//   then one pair of ints per source: the mode=0 has three sources a,b,c
//              - input select for source
//              - start offset for source in depth
//
// So, in general we could have more than two tensor inputs; and each of the sources could be extracted from a specific
// depth slice in a specific input. However, the final 'd' input must come from the last actual input, and the
// a,b,c, inputs must come from inputs other than the first.
//
//  Restrictions:
//   - currently:
//      - only mode=0 is supported (so configuration lengthy must be 8)
//      - all depths must be a multiple of 128.
//		- all 'start offsets' must be a multiple of 64.
//		- no restrictions on [b,h,w]
//		- the number of temnsor inputs must be 2 - so all of the 'input select' must be 0.
//   - all min/max inputs must be connected to consts; the 'configuration' input
//     must be a const. The scaling is worked out in first execution.
//
//////////
//
// Datapath:
//   - inputs a,b,c are scaled as needed to adapt to +/- 8 range,
//      using a negative scale so *1.0 is possible
//   - the first  product Tanh( a ) * Logistic( b ) is scaled before adding to second,
//     (according to the range ratio).
//   - the final add saturates to i16 range.
////////////////////////////////////////////////////////////////////////////////////////////
//
// QuantizedLstmOutput_16x16to8
//
// This reads two 'flat' tensors, both i16
//  the first has shape [b,h,w,din1], the second [b,h,w,din2]
//  The output is an i16 tensor [b,h,w,dout]  dout = din2
//
// The calculation is:
//     out = Logistic( a ) * Tanh( d )
// ... where:
//     a is  extracted from the first input, at specified depth position
//     d is from the second input.
//
//  Inputs:
//    input 0:   array [1,1,1,n] of int32: configuration
//    input 1:   first input array [b,h,w,din1]   (i16 tensor)
//    input 2:   second input array: [b,w,din2]   (i16 tensor)
//	  input 3,4:  min/max of first input array
//	  input 5,6:  min/max of second input array
//  Outputs:
//    output 0:  output [b,h,w,dout]   (u8 tensor)
//	  output 1,2:  output min/max [ will be -1.0, 0.9922 ]
//
//
// It is assumed that the input  ranges will be symmetric; the 'min' value corresponds to a -32768 output
//  and max = -min corresponds to +32768 (not 32767).
//
//
//  configuration is an array of int32:
//       +0     - output depth.
//       +1     - mode (only '0' currently supported)
//       +2     - start offset for 'a' in first input
//
//  Restrictions:
//   - currently:
//      - only mode=0 is supported
//      - output depth must be a multiple of 128.
//		- start offset must be a multiple of 64.
//		- no restrictions on [b,h,w]
//   - all min/max inputs must be connected to consts; the 'configuration' input
//     must be a const. The scaling is worked out in first execution.
//
//
//
struct mult_shift {
	int mult;
	int shift;
};
//
struct lstminput_info {
	uint8_t any_run_yet;
	uint8_t is_direct;			// true if out width == 128 and all input slices aligned.
	struct shape output_shape;
	int total_depths;		// b*h*w
	int input1_depth;
	int input2_depth;
	int input_slice_posns[3];

	// multiplier to adapt input1
	// the operation (inval * input1_scale.mult)>> input1_scale.shift
	// should map inval = 5.333333 to 0x10000.
	// And input1_scale should be >=16384, < 32768.
	// e.g. if the range is +/- 8.0, we want this to be scale = 0x6000, rsh = 13.
	struct mult_shift input1_scale;

	// for LstmOutput:
	// same thing for the second input.
	struct mult_shift input2_scale;

	// for LstmInput:
	// op is done as  f(a)*f(b) * preadd_gain   +   f(c) * d
	// .. where 'preadd_gain' is the reciprocal of the range of input d
	//  (since f(a),f(b), f(c) all have range +/-1 ); preadd_gain
	// is the reciprocal of d's range.
	// The scaling is actually dones as, effectively:
	// op is done as  ( f(a)*f(b) * preadd_mult   +   f(c) * d * (2^preadd_shift) ) >> preadd_shift
	// preadd_shift must be range 0..14 due to datapath constraints, so normally it will be 14
	// and preadd_gain will be  < 16K.
	struct mult_shift preadd_scale;

	// for LstmOutput:

	// output range
	float out_min, out_max;
};
#ifdef HEXAGON_V66
#define LSTMINPUT_MAX_THREADS 4
#define LSTMOUTPUT_MAX_THREADS 4
#else
#define LSTMINPUT_MAX_THREADS 2
#define LSTMOUTPUT_MAX_THREADS 2
#endif
// process in units of this many 'batches' (referring to b*h*w)
// must be a power of 2 and at least 8  (so that BATCH_SLICE*outwid is a multiple of 128).
#define LSTM_BATCH_SLICE 32

//
// LstmInput is processed in one of two modes: The actual processing is done 128 at a time (two
// adjacent vectors, at each input and at the output).
//  (1) if the output depth is 128 (and thus the input2 depth is also 128, and the input1 depth
//     is a multiple of 64, and all of the 'offsets' to input1 are multiples of 64, then 'is_direct'
//     will be set; in this mode, we do batches of 128, one oer loop, using the 'abc_stride' to advance the a,b,c
//      pointers between loops, and storing the results directly to output.
//  (2) in other cases, we first copy the a,b,c, inputs to 'split buffers' in which each of these are contiguous.
//     The output depth is always a multiple of 16.
//     Example: suppose the output depth is 80, and we are processing in groups of 16 depths. So each group generates
//     16*80 = 1280 elements = 10 loops = 10*2 vectors.
//     For this case, before running the 10 loops, we need to copy each input slice (which is 16 sections of 80xi16) to a 1280xi16 section
//     of the split buffer; when all three of these are done we can process the 10 computation loops. The 'in2' input
//     is packed as units of 80xi16, but contiguously, and each group of those is 16x80xi16 = 20x64xi16 so these are aligned reads. Likewise
//     the output writes are aligned; if the last output group is short (not a full group of 16) then we may need to calculate
//     it to a temp area to avoid over-running the output.


//
// 'runstate' contains information for one run, and is typically allocated on the stack
//
struct lstminput_runstate {
	struct lstminput_info const * info;
	int total_depths;			// b*w*h;
	int out_depth;
	//int nwide;					// = out_depth/128
	int n_jobs;					// each of <= LSTM_BATCH_SLICE depth units
	volatile int curr_job;
	volatile int thread_count;	// for allocating the temp area to threads.
	int16_t const *a_ptr;
	int16_t const *b_ptr;
	int16_t const *c_ptr;
	int16_t const *d_ptr;
	int16_t *optr;
	int32_t abc_stride;			// bytes
	HVX_Vector * tmp_area;		// pointer to temp area
	unsigned tmp_splitbuf_size;	// size of slice buffer (0 if not used)
	unsigned tmp_area_perthread;// total size of tmp area.

	nn_sem_t done_sem;
};


struct lstmoutput_runstate {
	struct lstminput_info const * info;
	int total_depths;			// b*w*h;
	int out_depth;
	//int nwide;					// = out_width/128

	volatile int thread_count;	// used to divide up work area
	int n_jobs;					// each of <= LSTM_BATCH_SLICE depth units
	volatile int curr_job;
	int16_t const *a_ptr;
	int16_t const *b_ptr;
	uint8_t *optr;
	int32_t a_stride;			// bytes
	uint8_t * split_buffers;	//	points to 'num_threads' split buffers (null if not needed)
	uint32_t split_buffer_size;	// size of each split buffer
	nn_sem_t done_sem;
};



static int lstminput_process_ref(struct nn_graph * nn , struct lstminput_runstate * rstp);
static void lstm_input_process_hvx( struct nn_graph * nn , void * rstpv);
static int set_input_scale( struct nn_graph *nn, struct mult_shift *dst, float minval, int input_num );
//
// setup the scaling etc based on input ranges
// This node assumes symmetric ranges, only looks at min.
//
static int
lstminput_setup_info( struct nn_node *self, struct nn_graph * nn)
{
	struct lstminput_info * info = (struct lstminput_info *)self->opaque;
	const struct tensor *input1_tensor = self->inputs[1];
	const struct tensor *input2_tensor = self->inputs[2];
	// set shapes
	struct shape tshape = input1_tensor->shape;
	info->input1_depth = tshape.depth;
	tshape.depth = info->input2_depth = input2_tensor->shape.depth;
	// tshape should now match input 2 shape
	if( ! shape_matches( &tshape, &input2_tensor->shape)){
		return errlog(nn,"inputs are not compatible shapes");
	}
	info->output_shape = tshape;
	// analyze the 'config' tensor.
	// Currently we support only mode == 0 and only 2 tensor inputs;
	// so the config must be [ *, 0,  0,*, 0,*,0,*]

	struct tensor const * config_tensor = self->inputs[0];
	if ( config_tensor->shape.depth < 8){
		return errlog(nn,"config shape is too small");
	}
	info->total_depths = info->output_shape.batches
		* info->output_shape.height * info->output_shape.width;
	int output_depth = tensor_get_int32( config_tensor,0);

	// 'direct' is the fully aligned case:
	// out_depth =128, in1_depth multiple of 64; all positions are multiple of 64.
	//
	int is_direct = (output_depth==128) && ((info->input1_depth & 63)==0);
	info->output_shape.depth = output_depth;

	for( int i = 0; i < 4; i++ ){
		int val = tensor_get_int32(config_tensor, (i==0)?1: (2*i));	// 1,2,4,6 must be 0
		if( val != 0)
			return errlog(nn,"improper or unsupported configuration");
	}

	if( output_depth <= 0 || (output_depth&15)!=0 || output_depth != info->input2_depth ){
		return errlog(nn,"bad output depth in config");
	}
	// get the slice positions, check for sane. Don't care if they overlap.
	for(int i=0; i < 3; i++ ){
		int spos =  tensor_get_int32( config_tensor,2*i+3);
		if( spos < 0 || spos + output_depth > info->input1_depth)
				return errlog(nn,"bad input slice: %d..%d in %d",
					spos, spos+output_depth-1, info->input1_depth);
		if( (spos & 63)!= 0 ) is_direct = 0;
		info->input_slice_posns[i] = spos;
	}
	info->is_direct = is_direct;
	///////// scaling //////////////////

	// get first input range
	struct tensor const * first_min_tensor = self->inputs[3];
	float firstmin = tensor_get_float( first_min_tensor, 0 );
	int sres = set_input_scale( nn, &info->input1_scale, firstmin, 1);
	if( sres != 0) return sres;

	// set up 'preadd_gain': to adapt a +/-1 range to 2nd input range.
	//
	struct tensor const * second_min_tensor = self->inputs[4];
	float secondmin = tensor_get_float( second_min_tensor, 0 );
	int preadd_gain = 0;
	int preadd_rsh = 0;
	float effgain=0.0f;
	if(secondmin < -1e-5){
		float gain_needed = -1.0f/secondmin;
		int expo = flt_getexp( gain_needed * 1.0000305f);	// exponent (with margin)
		preadd_rsh = min_i32(14, 15-expo);
		// is ok ?
		if( preadd_rsh >= 1 ){			// lower limit on rsh
			preadd_gain =  roundf_i32( flt_ldexp( gain_needed, preadd_rsh ));
			effgain = flt_ldexp(preadd_gain, -preadd_rsh);	// effective gain
		}
	}
	if( preadd_gain ==0 ){
		return errlog(nn,"bad range for input 2: min = %f" , secondmin);
	}else if( preadd_gain < 8192 ){
		// if preadd_gain gets too small, we could lose accuracy. It will still
		// be fine, e.g. if the ratio is a power of 2.
		float tmp = effgain*secondmin;	// will be close to -1.0 if accurate
		if( fabsf(tmp+1.0f) > (float)(1/8192.)){
			logmsg(nn,0,"may have precision loss due to input 2 range: %f .. %f", secondmin, -secondmin);
		}
	}
	info->preadd_scale.mult = preadd_gain;
	info->preadd_scale.shift = preadd_rsh;

	info->out_min = secondmin;
	info->out_max = -secondmin;
	info->any_run_yet = 1;
	return 0;
}
// used on second run - returns 1
// if depths are the same as on previous run
// and adapts to any change in shape.
// This is used for lstminput and lstmoutput nodes.
//
static int
lstm_check_shape_on_rerun( struct nn_node *self, struct nn_graph * nn, struct lstminput_info * info )
{
	const struct tensor *input1_tensor = self->inputs[1];
	const struct tensor *input2_tensor = self->inputs[2];

	if( input2_tensor->shape.depth != info->input2_depth) return 0;
	if( input1_tensor->shape.depth != info->input1_depth) return 0;

	// allow any change as long as the depths are the same as before.
	// and the inputs have the same (b,h,w)

	info->output_shape = input2_tensor->shape;
	if( input1_tensor->shape.batches_height != info->output_shape.batches_height
		|| input1_tensor->shape.width != info-> output_shape.width )
		return 0;

	info->total_depths = info->output_shape.batches
		* info->output_shape.height * info->output_shape.width;
	return 1;
}

static int __attribute__((noinline))
set_input_scale( struct nn_graph *nn,
		struct mult_shift *dst,		// result goes here
		float minval,
		int input_num )		// for error message
{
	int scale = 0;
	int shift = 0;
	if( minval < -1.e-5f){
		float fscale= (-3./8)* minval;		// what we need to mul by
		// use a margin when extracting exponent; so result won't round to 32768
		int expo = flt_getexp( fscale * 1.0000305f);	// exponent (with margin)
		shift = 15-expo;						// amount to right shift after mul
		if( shift > 0 && shift <= 17){				// check reasonable range...
			shift = min_i32(shift,14);				// for small values, reduce mag of scale
			scale = roundf_i32(flt_ldexp( fscale, shift) );
		}
	}
	if( scale == 0 ){
		return errlog(nn,"bad range for input %d: min = %f" ,input_num, minval);
	}
	dst->mult = scale;
	dst->shift = shift;
	return 0;
}

static int
lstminput_execute( struct nn_node *self, struct nn_graph * nn )
{

	struct lstminput_info * info = (struct lstminput_info *)self->opaque;
	if( !info->any_run_yet || !lstm_check_shape_on_rerun(self,nn,info)){
		int k = lstminput_setup_info(self,nn);
		if( k!= 0) return k;
	}

	struct tensor const * input1_tensor = self->inputs[1];
	struct tensor const * input2_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];

	if( tensor_out_prepare_normal_fromshape( out_tensor, &info->output_shape, NN_TYPE_QINT16)!= 0){
		return errlog(nn,"output too small");
	}

	struct lstminput_runstate runstate;
	runstate.info = info;
	runstate.total_depths = info->total_depths;
	runstate.n_jobs = (runstate.total_depths  + (LSTM_BATCH_SLICE-1))/(unsigned)LSTM_BATCH_SLICE;
	runstate.curr_job = 0;
	runstate.thread_count = 0;
	runstate.out_depth = info->output_shape.depth;
	int16_t const *dptr1 = (int16_t const*)input1_tensor->data;
	runstate.a_ptr =  dptr1 + info->input_slice_posns[0];
	runstate.b_ptr =  dptr1 + info->input_slice_posns[1];
	runstate.c_ptr =  dptr1 + info->input_slice_posns[2];
	runstate.d_ptr = (int16_t const*)input2_tensor->data;
	runstate.optr = (int16_t *)out_tensor->data;
	runstate.abc_stride = info->input1_depth * sizeof(int16_t);

	tensor_set_single_float( self->outputs[1],info->out_min);
	tensor_set_single_float( self->outputs[2],info->out_max);

	nn_scratch_reset(nn);
	int nthreads = min_i32( LSTMINPUT_MAX_THREADS, runstate.n_jobs);
	// Each thread needs a temp area, divided into three parts:
	//  (1) slice buffer a :  LSTM_BATCH_SLICE*out_depth * sizeof(int16)
	//  (2) slice buffer b :  same size as (a)
	//  (3) temp buffer (for use between passes); 2x size of slice buffer.
	// if we are 'direct' processing, the slice buffers are not used and their
	// size is set to 0.
	int slice_buffer_size =sizeof(int16_t)*LSTM_BATCH_SLICE*runstate.out_depth;

	int tmp_per_thread = 4*slice_buffer_size;
	if( info->is_direct){
		slice_buffer_size = 0;
		tmp_per_thread >>= 1;
	}
	runstate.tmp_splitbuf_size = slice_buffer_size;
	runstate.tmp_area_perthread  = tmp_per_thread;

	runstate.tmp_area = nn_scratch_alloc(nn,tmp_per_thread*nthreads);
	if( runstate.tmp_area == NULL){
		return errlog(nn,"failed to get scratch");
	}

	// all cases are done via hvx now
	if(0){
		lstminput_process_ref( nn, & runstate);
	}else{
		nn_sem_init( &runstate.done_sem, 0);
		for( int i = 0; i < nthreads; i++)
			nn_os_work_for_vector( nn, lstm_input_process_hvx, &runstate);
		nn_sem_wait_n_times( & runstate.done_sem, nthreads);
	}

	return 0;
}

#define FIXED_POINT_REF	// use fixed-point model in reference
//
// tanh table: map x2 -> tanh(x)
// where x2 = (1-x1)^2,  x1 = min( 1, abs(x)/(16/3))
//  so x2 in 0..1.0, in 16 cubic segments.
//  subtable  0 : 0,2,   ... 30    c3
//  subtable  1 : 32,34, ... 62    c2
//  subtable  2 : 1,3,   ... 31    c1
//  subtable  3 : 33,35, ... 63    c0
//
/*
// *** moved to hvx_constants.c ***
 int16_t tanh_16_cubic_lookup[64] __attribute__( (aligned(128),unused)) =
{
	     -1125,     -58,    -742,    -117,   -1192,    -250,   -1725,    -475,
	     -2424,    -837,   -3266,   -1392,   -4178,   -2214,   -5006,   -3385,
	     -5405,   -4987,   -4952,   -7078,   -3158,   -9658,     262,  -12622,
	      4941,  -15726,    9594,  -18595,   12789,  -20788,   13087,  -21927,
	       -36,   32766,    -390,   32746,    -672,   32702,   -1115,   32613,
	     -1760,   32453,   -2667,   32179,   -3894,   31735,   -5465,   31044,
	     -7354,   30007,   -9406,   28509,  -11296,   26426,  -12509,   23647,
	    -12443,   20101,  -10586,   15800,   -6953,   10858,   -2101,    5494,
}; */

///
// This the same function, in the same format, but fitted to 2nd order
// for lower precision required for 8-bit output. So, all c2 are 0.
// Furthermore, the 2nd order terms are constrained to be equal in groups of 4;
// so on V65 the the 4-way table lookup can be used to get c2 as a function
// of the upper 2 bits of x2.
static const int16_t tanh_16_lookup_quad[64] __attribute__( (aligned(128),unused)) =
{
         0,     -69,       0,    -170,       0,    -324,       0,    -678,
         0,    -605,       0,   -1217,       0,   -2195,       0,   -3648,
         0,   -4862,       0,   -7116,       0,   -9962,       0,  -12965,
         0,  -16616,       0,  -19276,       0,  -21037,       0,  -21319,
       -40,   32766,     -40,   32748,     -40,   32704,     -40,   32622,
     -4195,   32451,   -4195,   32169,   -4195,   31734,   -4195,   31054,
     -9234,   30011,   -9234,   28506,   -9234,   26439,   -9234,   23660,
     -4554,   20130,   -4554,   15833,   -4554,   10872,   -4554,    5471,
};
#if __HEXAGON_ARCH__ >= 65
// this is the 2nd order term from the table above - 4 values packed into uint64.
#define TANH_LOOKUP_C2 (\
	(uint16_t)(-40)  +  ((uint16_t)(-4195)) * (1ULL<<16)\
	+ (uint16_t)(-9234) * (1ULL<<32)  + (uint16_t)(-4554) * (1ULL<<48) )
#endif

#ifdef FIXED_POINT_REF
#if 0
static int32_t
find_tanh_16_pair( int32_t x)
{
	float x0 = (int16_t)x  * (float)(-1/4096.);
	float x1 = (int16_t)(x>>16)  * (float)(-1/4096.);

	int y0 = saturate_i16( roundf_i32( 32768.0f * tanhf(x0)));
	int y1 = saturate_i16( roundf_i32( 32768.0f * tanhf(x1)));
	return Q6_R_combine_RlRl(y1,y0);
}
#endif

static int32_t
find_tanh_16_pair( int32_t xx, int gain, int shift)
{
	int32_t result = 0;
	//int rnd = (1<<shift)>>1;
	for( int i  =0; i < 2; i++){
		int x = (int16_t)xx;	// extract first, then second
		xx >>= 16;
		int x1 = (((x>=0)?x:-x) * gain)>>shift;	// get 0..5.3333 in range 0..0xFFFF
		int y;
		if( 0 &&  x1 > 0xFFFF){		// if saturated to 0xFFFF, these would map to 0x7ffe
			y = 0x7FFF;
		}else{
			if( x1 > 0xFFFF) x1 = 0xFFFF;
			int x2 = ((x1^0xFFFF)*(x1^0xFFFF))>>16;
			int xr = x2 & 0xFFF;
			int segno = (x2 >> 12)&15;
			int c0 = tanh_16_cubic_lookup[2*segno+33];
			int c1 = tanh_16_cubic_lookup[2*segno+1];
			int c2 = tanh_16_cubic_lookup[2*segno+32];
			int c3 = tanh_16_cubic_lookup[2*segno+0];
			int tmp = Q6_R_vmpyh_RR_s1_rnd_sat( c3, xr) + c2;
			tmp = Q6_R_vmpyh_RR_s1_rnd_sat( tmp, xr) + c1;
			y = Q6_R_vmpyh_RR_s1_rnd_sat(tmp, xr*2)+ c0;
		}
		if( x < 0) y = -y;
		result = Q6_R_combine_RlRh(y,result);

	}
	return result;
}
#endif

static inline float rnd15( float val)
{
	return (float)(1/32768.0f)* roundf_i32( 32768.0f * val);
}
static inline float __attribute__((unused))
ref_tanh( float x)
{
	return rnd15( tanhf(x));
}
static inline float  __attribute__((unused))
ref_logistic( float x)
{
	//int ix = saturate_i16(roundf_i32( x * (-2048.0f)));
	//int iy = (int16_t)find_tanh_16_pair(ix);
	//return (float)(1./65536.)*iy + 0.5f;

	return rnd15(( tanhf(x*0.5)+1.0f)*0.5f);
}
//
// temporary reference implementation.
//
static int
lstminput_process_ref(struct nn_graph * nn , struct lstminput_runstate * rstp)
{
	struct lstminput_info const * info = rstp->info;

	int16_t const * input_a = rstp->a_ptr;
	int16_t const * input_b = rstp->b_ptr;
	int16_t const * input_c = rstp->c_ptr;
	int16_t const * input_d = rstp->d_ptr;
	int16_t * output = rstp->optr;

	int abc_stride = rstp->abc_stride / sizeof(int16_t);
	int d_stride = rstp->out_depth;
	int out_stride = rstp->out_depth;
	int nwide = rstp->out_depth;

	int nd = info->total_depths;

#ifndef FIXED_POINT_REF
	// (1) not even trying to be efficient
	float inscale = flt_ldexp( info->input1_scale.mult, -info->input1_scale.shift)*(float)( 8.0/(32768.0*3));
	float preadd_scale = flt_ldexp(info->preadd_scale.mult, 15-info->preadd_scale.shift);

	for(int ii= 0; ii < nd; ii++ ){
		for( int i = 0; i < nwide; i++ ){
			float xa = input_a[i]*inscale;
			float xb = input_b[i]*inscale;
			float xc = input_c[i]*inscale;
			float xd = input_d[i];		// 'as-is' units
			float p1 = ref_tanh(xa) * ref_logistic(xb);
			float p2 = ref_logistic(xc) * xd;
			float sum = p1*preadd_scale + p2;
			output[i] = saturate_i16( roundf_i32( sum ));
		}
		input_a += abc_stride;
		input_b += abc_stride;
		input_c += abc_stride;
		input_d += d_stride;
		output += out_stride;
	}
#else
	// (2) model the fixed-point ops the hvx will use.
	int a_scale = info->input1_scale.mult;
	int a_rsh = info->input1_scale.shift;
	//int a_scale_half = Q6_R_combine_RlRl(a_scale>>1,a_scale>>1);
	//a_scale = Q6_R_combine_RlRl(a_scale,a_scale);
	int preadd_scale = info->preadd_scale.mult;
	preadd_scale = Q6_R_combine_RlRl( preadd_scale,preadd_scale);
	int final_rsh = info->preadd_scale.shift+1;		// 2..15
	int cd_gain = 0x00010001 << info->preadd_scale.shift;

	for(int ii= 0; ii < nd; ii++ ){
		for( int i = 0; i < nwide; i+=2 ){
			int32_t xa = *(int32_t const*)&input_a[i];
			int32_t xb = *(int32_t const*)&input_b[i];
			int32_t xc = *(int32_t const*)&input_c[i];
			int32_t xd = *(int32_t const*)&input_d[i];

			int32_t tanh_a = find_tanh_16_pair(xa,a_scale,a_rsh);
			int32_t tanh_b = find_tanh_16_pair(xb,a_scale,a_rsh+1);
			int32_t tanh_c = find_tanh_16_pair(xc,a_scale,a_rsh+1);
			int32_t abprod = Q6_R_vmpyh_RR_s1_rnd_sat(tanh_a, tanh_b);
			int32_t cdprod = Q6_R_vmpyh_RR_s1_rnd_sat(tanh_c,xd);

			int64_t  prod = Q6_P_vmpyh_RR_sat( tanh_a, preadd_scale);	//
			prod = Q6_P_vmpyhacc_RR_sat( prod, abprod, preadd_scale);
			prod = Q6_P_vmpyhacc_RR_sat( prod, xd, cd_gain);
			prod = Q6_P_vmpyhacc_RR_sat( prod, cdprod, cd_gain);
			// >>preadd_rsh and round and sat.
			prod = Q6_P_combine_RR(		// # << 2 with sat
					Q6_R_asl_RR_sat( (int)(prod>>32), 16-final_rsh),
					Q6_R_asl_RR_sat( (int)(prod),  16-final_rsh ) );
			int32_t res = Q6_R_vrndwh_P_sat( prod);
			*(int32_t*)&output[i] = res;
		}
		input_a += abc_stride;
		input_b += abc_stride;
		input_c += abc_stride;
		input_d += d_stride;
		output += out_stride;
	}
#endif

	return 0;
}
/////////////////////////////////////// LstmOutput ///////////////////
//
// setup the scaling etc based on input ranges
// This node assumes symmetric ranges, only looks at min.
//
static int
lstmoutput_setup_info( struct nn_node *self, struct nn_graph * nn)
{
	struct lstminput_info * info = (struct lstminput_info *)self->opaque;
	const struct tensor *input1_tensor = self->inputs[1];
	const struct tensor *input2_tensor = self->inputs[2];
	// set shapes
	struct shape tshape = input1_tensor->shape;
	info->input1_depth = tshape.depth;
	tshape.depth = info->input2_depth = input2_tensor->shape.depth;
	// tshape should now match input 2 shape
	if( ! shape_matches( &tshape, &input2_tensor->shape)){
		return errlog(nn,"inputs are not compatible shapes");
	}
	info->output_shape = tshape;
	// analyze the 'config' tensor.
	// Currently we support only mode == 0

	struct tensor const * config_tensor = self->inputs[0];
	if ( config_tensor->shape.depth < 3){
		return errlog(nn,"config shape is too small");
	}
	info->total_depths = info->output_shape.batches
		* info->output_shape.height * info->output_shape.width;
	int output_depth = tensor_get_int32( config_tensor,0);
	info->output_shape.depth = output_depth;

	int mode = tensor_get_int32(config_tensor,1);
	if( mode != 0 ){
		return errlog(nn,"only mode 0 supported");
	}

	if( output_depth <= 0 || (output_depth&15)!=0 || output_depth != info->input2_depth ){
		return errlog(nn,"bad output depth in config");
	}
	// get the slice position, check for sane
		int spos =  tensor_get_int32( config_tensor,2);
	if( spos < 0 || spos + output_depth > info->input1_depth)
		return errlog(nn,"bad input slice: %d..%d in %d",
					spos, spos+output_depth-1, info->input1_depth);
	info->input_slice_posns[0] = spos;
	info->is_direct = (output_depth == 128) && ((spos&63)==0);


	///////// scaling //////////////////

	// get first input range
	{
		struct tensor const * first_min_tensor = self->inputs[3];
		float firstmin = tensor_get_float( first_min_tensor, 0 );
		int sres = set_input_scale( nn, &info->input1_scale, firstmin, 1);
		if( sres != 0) return sres;
	}
	{	// second
		struct tensor const * second_min_tensor = self->inputs[5];
		float secondmin = tensor_get_float( second_min_tensor, 0 );
		int sres = set_input_scale( nn, &info->input2_scale, secondmin, 2);
		if( sres != 0) return sres;
	}
	info->out_min = -1.0f;
	info->out_max = (float)(127.0/128.0);
	info->any_run_yet = 1;
	return 0;
}
static void lstmoutput_process_ref(struct nn_graph * nn , struct lstmoutput_runstate * rstp);
static void lstmoutput_process_hvx( struct nn_graph * nn , void * rstpv);

static int
lstmoutput_execute( struct nn_node *self, struct nn_graph * nn )
{

	struct lstminput_info * info = (struct lstminput_info *)self->opaque;

	if( !info->any_run_yet || !lstm_check_shape_on_rerun(self,nn,info)){
		int k = lstmoutput_setup_info(self,nn);
		if( k!= 0) return k;
	}
	struct tensor const * input1_tensor = self->inputs[1];
	struct tensor const * input2_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];

	struct lstmoutput_runstate runstate;
	runstate.info = info;
	runstate.total_depths = info->total_depths;
	runstate.n_jobs = (runstate.total_depths  + (LSTM_BATCH_SLICE-1))/(unsigned)LSTM_BATCH_SLICE;
	runstate.thread_count = 0;
	runstate.curr_job = 0;
	runstate.out_depth = info->output_shape.depth;
	runstate.a_ptr = (int16_t const*)input1_tensor->data + info->input_slice_posns[0];
	runstate.b_ptr = (int16_t const*)input2_tensor->data;
	runstate.optr = (uint8_t *)out_tensor->data;
	runstate.a_stride = info->input1_depth * sizeof(int16_t);
	runstate.split_buffers = NULL;

	int nthreads = min_i32( LSTMOUTPUT_MAX_THREADS, runstate.n_jobs);

	nn_scratch_reset(nn);
	if( !info->is_direct){
		// need to allocate one split buffer:
		// size is LSTM_BATCH_SLICE * out_depth * 2 bytes
		unsigned split_buffer_size  = LSTM_BATCH_SLICE * runstate.out_depth * sizeof(int16_t);
		runstate.split_buffer_size = split_buffer_size;
		runstate.split_buffers = nn_scratch_alloc( nn, split_buffer_size * nthreads);
		if( runstate.split_buffers == 0){
			return errlog(nn,"can't alloc %d*%d of scratch", nthreads, split_buffer_size);
		}
	}

	if( tensor_out_prepare_normal_fromshape( out_tensor, &info->output_shape, NN_TYPE_QUINT8)!= 0){
		return errlog(nn,"output too small");
	}
	tensor_set_single_float( self->outputs[1],info->out_min);
	tensor_set_single_float( self->outputs[2],info->out_max);

	if(0){	// all cases handled by hvx now
		lstmoutput_process_ref( nn, & runstate);
	}else{
		nn_sem_init( & runstate.done_sem,0);
		for( int i = 0; i < nthreads; i++)
			nn_os_work_for_vector( nn, lstmoutput_process_hvx, &runstate);
		nn_sem_wait_n_times( & runstate.done_sem, nthreads);
	}

	return 0;
}

static void
lstmoutput_process_ref( struct nn_graph * nn , struct lstmoutput_runstate * rstp)
{
	struct lstminput_info const * info = rstp->info;


	int16_t const * input_a = rstp->a_ptr;
	int16_t const * input_b = rstp->b_ptr;
	uint8_t * output = rstp->optr;

	int a_stride = rstp->a_stride/sizeof(int16_t);
	int b_stride = rstp->out_depth;
	int dout = rstp->out_depth;

	int nd = rstp->total_depths;
	int nwide = rstp->out_depth;

#ifndef FIXED_POINT_REF

	// (1) not even trying to be efficient
	float inscale1 = flt_ldexp( info->input1_scale.mult, -info->input1_scale.shift)*(float)( 8.0/(32768.0*3));
	float inscale2 = flt_ldexp( info->input2_scale.mult, -info->input2_scale.shift)*(float)( 8.0/(32768.0*3));

	for(int ii= 0; ii < nd; ii++ ){
		for( int i = 0; i < nwide; i++ ){
			float xa = input_a[i]*inscale1;
			float xb = input_b[i]*inscale2;
			float result = ref_logistic(xa) * ref_tanh(xb);	// -1..   .. 1.0
			output[i] = saturate_u8( roundf_i32( result * 128.0f + 128.0f ));
		}
		input_a += a_stride;
		input_b += b_stride;
		output += dout;
	}
#else
	// (2) model the fixed-point ops the hvx will use.
	int a_scale = info->input1_scale.mult;
	int a_rsh = info->input1_scale.shift;
	int b_scale = info->input2_scale.mult;
	int b_rsh = info->input2_scale.shift;
	for(int ii= 0; ii < nd; ii++ ){
		for( int i = 0; i < nwide; i+=2 ){
			int32_t xa = *(int32_t const*)&input_a[i];
			int32_t xb = *(int32_t const*)&input_b[i];

			int32_t tanh_a = find_tanh_16_pair(xa,a_scale,a_rsh+1);
			int32_t tanh_b = find_tanh_16_pair(xb,b_scale,b_rsh);
			int32_t abprod = Q6_R_vmpyh_RR_s1_rnd_sat(tanh_a, tanh_b);
			int32_t res = Q6_R_vavgh_RR_rnd( tanh_b, abprod );
			// now >>8 with round/sat to u8
			res= Q6_R_vaddh_RR_sat( res, 0x00800080);	// rounding bias
			int resu8 = Q6_R_vtrunohb_P(res) ^ 0x8080;			// extract odd bytes; cvt to unsigned

			*(uint16_t*) &output[i] = resu8;
		}
		input_a += a_stride;
		input_b += b_stride;
		output += dout;
	}
#endif
}


// HVX code.
/////////////////////////////////////////////
//
// tanh_hvx_i16 finds 64xtahn operations on 64 i16 inputs.
// Inputs are i16; and will be scaled by *scaleby >>rsh
// See step (3) for scaling info.
//
// Procedure is:
//  (1) take abs value of xin, so it's 0..0x8000  >=0
//  (2) unsigned mul by 'scaleby'
//  (3) >> rsh, no rounding, saturate to 0..0xFFFF  this is 'x1'
//   **  The scaling should be set up so that a value of 5.33333 (16/3) maps to
//       x1 = 0x10000.
//  (4) x1 = (x2 ^ 0xFFFF)^2 >> 16   (0..0xFFFF)
//  (5) using the upper 4 bits of x1 to select coefficients from a table, and
//      the lower 12 bits as 'xres', evaluate a polynomial. Result is 0..32766.
//  (5a) - maybe - if x1 saturated, use 32767 here (this is a pain to do).
//  (6) re-apply the sign of xin
//
// There is another version of this function which does two
// at once; that is more efficient since the lut32 can be fully utilized.
//
// --> this function converts xin to x2.

static void __attribute__((unused))
show_16( HVX_Vector v, char const *str){
	return;
	union {
		HVX_Vector v;
		int16_t i16[64];
	} u = { v };
	printf("%s = %d == 0x%X\n", str, u.i16[0], u.i16[0] & 0xFFFF);

}
static inline int16_t * __attribute__((unused,always_inline))
add_ptr_int16_bytes ( int16_t * p, int bytes){
	return (int16_t*)( (char *)p + bytes);
}
static inline int16_t const * __attribute__((unused,always_inline))
add_ptrk_int16_bytes ( int16_t const * p, int bytes){
	return (int16_t const *)( (char const *)p + bytes);
}
static inline HVX_Vector * __attribute__((unused,always_inline))
add_ptr_VEC_bytes ( HVX_Vector * p, int bytes){
	return (HVX_Vector*)( (char *)p + bytes);
}

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
// same thing but quadratic
static inline HVX_Vector
tanh_hvx_quad( HVX_Vector x2, HVX_Vector xin,HVX_Vector c2, HVX_Vector c1, HVX_Vector c0 )
{
	HVX_Vector xres = Q6_V_vand_VV( x2, Q6_V_vsplat_R(0x0FFF0FFF));
	HVX_Vector resx2 = Q6_Vh_vadd_VhVh( xres,xres);
	HVX_Vector tmp = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vmpy_VhVh_s1_rnd_sat(c2, xres ), c1 );		// 17 bit intermediate
	HVX_Vector result = Q6_Vh_vadd_VhVh_sat( Q6_Vh_vmpy_VhVh_s1_rnd_sat(tmp, resx2 ), c0 );	// 15 bit result

	HVX_Vector xsign = Q6_Vh_vasr_VhR( xin, 15);
	// that's the result! if input < 0, need to ~ and add 1.
	result = Q6_V_vxor_VV( result, xsign);
	return Q6_Vh_vsub_VhVh_sat( result, xsign );
}


//
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

//
// two at once using quadratic
// (lower accuracy; for 8-bit output)
static inline
HVX_VectorPair tanh_hvx_i16_quad_x2( HVX_Vector xin0, HVX_Vector xin1, int scaleby, int rsh0, int rsh1 )
{

	HVX_Vector x2_0 = tanh_hvx_prescale( xin0, scaleby, rsh0);	// do 'prescale'; find x2
	HVX_Vector x2_1 = tanh_hvx_prescale( xin1, scaleby, rsh1);	// do 'prescale'; find x2


	HVX_Vector table = *(HVX_Vector const *) tanh_16_lookup_quad;

	// combine lookup table indices [bits 15..12 of each uh lane] to a single register
	HVX_Vector tablesel = Q6_Vb_vshuffo_VbVb( x2_1, x2_0);		// upper bytes
	tablesel = Q6_Vuh_vlsr_VuhR( tablesel,4);					// >>4

	// get the coefficients from table
#if __HEXAGON_ARCH__ >= 62
	#if __HEXAGON_ARCH__ >= 65
		HVX_Vector c2_0 = Q6_Vh_vlut4_VuhPh(x2_0,TANH_LOOKUP_C2);
		HVX_Vector c2_1 = Q6_Vh_vlut4_VuhPh(x2_1,TANH_LOOKUP_C2);
	#else
		HVX_VectorPair c2 = Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,1 );
		HVX_Vector c2_0 = Q6_V_lo_W(c2);
		HVX_Vector c2_1 = Q6_V_hi_W(c2);
	#endif
	HVX_VectorPair c1 = Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,2 );
	HVX_VectorPair c0 = Q6_Wh_vlut16_VbVhR_nomatch(tablesel, table,3 );
#else
	// need to 'clean up' the table sel
	tablesel = Q6_V_vand_VV( tablesel, Q6_V_vsplat_R( 0x0F0F0F0F));
	// need to break out the table values to 3
	HVX_Vector tab1 = Q6_V_vror_VR( table, 64);	// 32,34 .. 62 are section 1
	HVX_Vector tab2 = Q6_Vh_vshuffo_VhVh( table, table);	// 1,3 .. 31 are section 2
	HVX_Vector tab3 = Q6_Vh_vshuffo_VhVh( tab1, tab1);	// 33,35 .. 63 are section 3

	HVX_VectorPair c2 = Q6_Wh_vlut16_VbVhR(tablesel, tab1,0 );
	HVX_VectorPair c1 = Q6_Wh_vlut16_VbVhR(tablesel, tab2,0 );
	HVX_VectorPair c0 = Q6_Wh_vlut16_VbVhR(tablesel, tab3,0 );
	HVX_Vector c2_0 = Q6_V_lo_W(c2);
	HVX_Vector c2_1 = Q6_V_hi_W(c2);

#endif

	HVX_Vector y_0 = tanh_hvx_quad( x2_0, xin0, c2_0,Q6_V_lo_W(c1),Q6_V_lo_W(c0));
	HVX_Vector y_1 = tanh_hvx_quad( x2_1, xin1, c2_1,Q6_V_hi_W(c1),Q6_V_hi_W(c0));
	return Q6_W_vcombine_VV( y_1, y_0);
}
//  This does an 'average' into the intermediate, reducing its size by 2
// Seems to speed up the entire node by about 4%; also reduces intermed buffer size.
// Need to check rounding error and also see if "preadd_mult*2" could overflow.
//#define FASTER_LSTMIN 1
//
// core operation for hvx
//
//
static void
lstm_input_process_hvx( struct nn_graph * nn , void * rstpv)
{
	struct lstminput_runstate * rstp = ( struct lstminput_runstate *)rstpv;
	struct lstminput_info const * info = rstp->info;
	int my_thread_id = __sync_fetch_and_add( &rstp->thread_count, 1);

	int out_depth = rstp->out_depth;
	// strides are all in bytes
	int abc_stride = rstp->abc_stride;
	int abc_split_stride = abc_stride;
	int out_stride = 2*128;
	int d_stride = 2*128;

	int splitcopywid = out_depth * sizeof(int16_t);

	HVX_Vector *work_area = add_ptr_VEC_bytes(rstp->tmp_area, rstp->tmp_area_perthread*my_thread_id);
	int splitbuf_len = rstp->tmp_splitbuf_size;
	int16_t * splitbuf_a = NULL;
	int16_t * splitbuf_b = NULL;
	if( splitbuf_len > 0){
		// out_depth !=128, so we are copying a,b,c into 'splitbuf' to get them
		// all contiguous before running the hvx loops.
		splitbuf_a = (int16_t *)work_area;
		splitbuf_b = add_ptr_int16_bytes(splitbuf_a, splitbuf_len);
		work_area = add_ptr_VEC_bytes( work_area, 2 * splitbuf_len );

		abc_stride = 2*128;
	}
	// this is the 'usual' n_loops
	int n_loops_full = (LSTM_BATCH_SLICE * out_depth )/ 128u;

	// this is the pitch per batch-slice, in a 'flat' tensor, in bytes
	int job_stride = out_depth * LSTM_BATCH_SLICE * sizeof(int16_t);	// always mult. of 256

	// Doing all this in one loop, 128 ops in parallel, is too many registers for the
	// compiler to cope with. But we need to do 128 at once to get the full efficiency
	// of the vluts16s. So instead, the first loop will find tahn(a) and tanh(a)*tanh(b/2), and store them in a temp area.

	int inscale = info->input1_scale.mult;
	int in_rsh = info->input1_scale.shift;
	int  preadd_mult =  info->preadd_scale.mult;
#ifdef FASTER_LSTMIN
	preadd_mult *= 2;
#endif
	int preadd_gain = Q6_R_combine_RlRl( preadd_mult, preadd_mult);
	int final_rsh = info->preadd_scale.shift  +1;
	int cd_gain = 0x00010001 << (final_rsh-1);
	int n_jobs = rstp->n_jobs;

	unsigned last_copy_len = 0;
	int16_t * lastcopy_dest = NULL;

	int ijobno;
	while(  ijobno = __sync_fetch_and_add( &rstp->curr_job, 1),   ijobno < n_jobs){
		int base_depth = ijobno * LSTM_BATCH_SLICE;
		// # of groups to process
		int n_groups = min_i32(LSTM_BATCH_SLICE, rstp->total_depths - base_depth );

		int n_loops = n_loops_full;
		if( unlikely( n_groups < LSTM_BATCH_SLICE) ){	// partial, at end...
			if( n_groups <= 0) break;
			n_loops = (n_groups * out_depth+127)/128u;
			// if the # of output vectors generated is > the the number actually needed,
			// redirect the output to splitbuf_b and copy a prefix back after. Here
			// we just find the length.
			//if( n_loops*2 > (n_groups*out_depth+63)/64u) {
			if(n_loops > (n_groups*out_depth+63)/128u ){		// equivalent test
				last_copy_len = n_groups*out_depth*2;			// copy this many
			}

		}

		// get a,b input pointers, to the particular depth position we are starting at
		int16_t const * in_a = add_ptrk_int16_bytes( rstp->a_ptr,  base_depth * abc_split_stride);
		int16_t const * in_b = add_ptrk_int16_bytes( rstp->b_ptr,  base_depth * abc_split_stride);

		if( splitbuf_a != NULL){		// need to copy slices to split buffers
			vmemcpy_2d_general_asm(
					splitcopywid,		// width of copy
					n_groups,			// height,
					splitbuf_a,			// dest buffer
				 splitcopywid,			// stride
				 in_a,					// source buffer
				 abc_split_stride);				// stride;
			vmemcpy_2d_general_asm(
					splitcopywid,		// width of copy
					n_groups,			// height,
					splitbuf_b,			// dest buffer
				 splitcopywid,			// stride
				 in_b,					// source buffer
				 abc_split_stride);				// stride;
			in_a = (int16_t const*)splitbuf_a;
			in_b = (int16_t const*)splitbuf_b;
		}


		HVX_Vector *workp = work_area;
		for( int i = 0; i <n_loops; i++){
			HVX_Vector a_in0 = *(HVX_Vector const *)in_a;
			HVX_Vector a_in1 = *(HVX_Vector const *)(in_a+64);
			HVX_Vector b_in0 = *(HVX_Vector const *)in_b;
			HVX_Vector b_in1 = *(HVX_Vector const *)(in_b+64);

			HVX_VectorPair a_tanh = tanh_hvx_i16_x2( a_in0, a_in1, inscale, in_rsh, in_rsh );
			HVX_VectorPair b_tanh = tanh_hvx_i16_x2( b_in0, b_in1, inscale, in_rsh+1, in_rsh+1 );

			// tanh(a0), tanh(a0)*tanh(b0/2), tanh(a1), tanh(a1)*tanh(b1/2)
			HVX_Vector tanha_0 = Q6_V_lo_W( a_tanh);
			HVX_Vector tanha_1 = Q6_V_hi_W( a_tanh);
#ifdef FASTER_LSTMIN
			HVX_Vector p0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(tanha_0, Q6_V_lo_W( b_tanh));
			HVX_Vector p1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(tanha_1, Q6_V_hi_W( b_tanh));
/* Attempt at unbaised rounding. this takes too long to do (no net speedup).
			HVX_Vector hash0 = Q6_V_vxor_VV( p0, tanha_0);
			HVX_Vector hash1 = Q6_V_vxor_VV( p1, tanha_1);
		// (hash>>1) & 1
			hash0 = Q6_V_vand_VV(Q6_Vh_vavg_VhVh( hash0, Q6_V_vzero()), Q6_V_vsplat_R(0x00010001));
			hash1 = Q6_V_vand_VV(Q6_Vh_vavg_VhVh( hash1, Q6_V_vzero()), Q6_V_vsplat_R(0x00010001));
			p0 = Q6_Vh_vadd_VhVh_sat( p0, hash0);
			p1 = Q6_Vh_vadd_VhVh_sat( p1, hash1);
*/
			workp[0] = Q6_Vh_vavg_VhVh( p0, tanha_0);
			workp[1] =  Q6_Vh_vavg_VhVh( p1,tanha_1 );
			workp += 2;
#else
			workp[0] = tanha_0;
			workp[1] = Q6_Vh_vmpy_VhVh_s1_rnd_sat(tanha_0, Q6_V_lo_W( b_tanh));
			workp[2] = tanha_1;
			workp[3] = Q6_Vh_vmpy_VhVh_s1_rnd_sat(tanha_1, Q6_V_hi_W( b_tanh));

			workp += 4;
#endif
			in_a = add_ptrk_int16_bytes(in_a,  abc_stride);
			in_b = add_ptrk_int16_bytes(in_b,  abc_stride);
		}

		// get c,d input pointers, to the particular depth position we are starting at
		int16_t const * in_c = add_ptrk_int16_bytes( rstp->c_ptr,  base_depth * abc_split_stride);
		int16_t const * in_d = add_ptrk_int16_bytes( rstp->d_ptr,  ijobno * job_stride);
		// and output pointer
		int16_t * out = add_ptr_int16_bytes( rstp->optr, ijobno * job_stride);

		if( splitbuf_a != NULL){		// need to copy slices of c to split buffer a
			vmemcpy_2d_general_asm(
					splitcopywid,		// width of copy
					n_groups,			// height,
					splitbuf_a,			// dest buffer
				 splitcopywid,			// stride
				 in_c,					// source buffer
				 abc_split_stride);				// stride;
			in_c = (int16_t const*)splitbuf_a;
			if( last_copy_len != 0){
				lastcopy_dest = out;		// redirect to splitbuf_b
				out = splitbuf_b;
			}
		}
		workp = work_area;

		for( int i = 0; i <n_loops; i++){
			HVX_Vector d_in0 = *(HVX_Vector const *)in_d;
			HVX_Vector d_in1 = *(HVX_Vector const *)(in_d+64);

			HVX_VectorPair sum0 = Q6_Ww_vmpy_VhRh( d_in0, cd_gain);
			HVX_VectorPair sum1 = Q6_Ww_vmpy_VhRh( d_in1, cd_gain);

			HVX_Vector c_in0 = *(HVX_Vector const *)in_c;
			HVX_Vector c_in1 = *(HVX_Vector const *)(in_c+64);
			HVX_VectorPair c_tanh = tanh_hvx_i16_x2( c_in0, c_in1, inscale, in_rsh+1, in_rsh+1 );

			HVX_Vector cd_prod_0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(d_in0, Q6_V_lo_W(c_tanh) );
			HVX_Vector cd_prod_1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(d_in1, Q6_V_hi_W(c_tanh) );

			sum0 = Q6_Ww_vmpyacc_WwVhRh_sat( sum0, cd_prod_0, cd_gain);
			sum1 = Q6_Ww_vmpyacc_WwVhRh_sat( sum1, cd_prod_1, cd_gain);

			// and the values from the first pass
			sum0 = Q6_Ww_vmpyacc_WwVhRh_sat( sum0, workp[0], preadd_gain);
#ifdef FASTER_LSTMIN
			workp -= 2;
#else
			sum0 = Q6_Ww_vmpyacc_WwVhRh_sat( sum0, workp[1], preadd_gain);
			sum1 = Q6_Ww_vmpyacc_WwVhRh_sat( sum1, workp[2], preadd_gain);
#endif
			sum1 = Q6_Ww_vmpyacc_WwVhRh_sat( sum1, workp[3], preadd_gain);
			HVX_Vector result_0 = Q6_Vh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(sum0),  Q6_V_lo_W(sum0), final_rsh);
			HVX_Vector result_1 = Q6_Vh_vasr_VwVwR_rnd_sat( Q6_V_hi_W(sum1),  Q6_V_lo_W(sum1), final_rsh);

			*(HVX_Vector *)out = result_0;
			*(HVX_Vector *)(out+64) = result_1;

			workp += 4;
			in_c = add_ptrk_int16_bytes( in_c , abc_stride);
			in_d = add_ptrk_int16_bytes( in_d , d_stride);
			out = add_ptr_int16_bytes(  out,  out_stride);
		}
	}
	// if last_copy_len > 0, it means the last operation was longer than it should be,
	// (as a result of needing to be a multiple of 2 vectors); so it was sent to splitbuf_a;
	// 'last_copy_dest' is the address of where it needs to go to.
	if( last_copy_len> 0){
		last_copy_len = (last_copy_len+127) & ~127u;
		vmemcpy_asm( lastcopy_dest, splitbuf_b, last_copy_len);
	}
	nn_sem_post( & rstp->done_sem);
}


static void
lstmoutput_process_hvx( struct nn_graph * nn , void * rstpv)
{

	struct lstmoutput_runstate * rstp = (struct lstmoutput_runstate *)rstpv;
	struct lstminput_info const *info = rstp->info;

	uint8_t *split_buffer = rstp->split_buffers;

	int out_depth = rstp->out_depth;
	int splitcopywid = 2*out_depth;
	int a_stride = rstp->a_stride;		// these are in *bytes*
	int b_stride = 2*128;
	int a_split_stride = a_stride;

	if( split_buffer != NULL){
		int my_thread_id = __sync_fetch_and_add( &rstp->thread_count, 1);
		split_buffer += my_thread_id * rstp->split_buffer_size;
		a_split_stride = a_stride;		// stride in input buffer
		a_stride = 2*128;
	}
	// this is the pitch per batch-slice, in a 'flat' input tensor, in elements
	int job_stride_elements = out_depth * LSTM_BATCH_SLICE;	// always mult. of 128.
	int out_stride = 128;

	int in1_scale = info->input1_scale.mult;
	int in1_rsh = info->input1_scale.shift;
	int in2_scale = info->input2_scale.mult;
	int in2_rsh = info->input2_scale.shift;
	int n_jobs = rstp->n_jobs;
	// this is the 'usual' n_loops
	int n_loops_full = (LSTM_BATCH_SLICE * out_depth )/ 128u;


	int ijobno;

	while(  ijobno = __sync_fetch_and_add( &rstp->curr_job, 1),   ijobno < n_jobs){
		int base_depth = ijobno * LSTM_BATCH_SLICE;
		// # of groups to process
		int n_groups = min_i32(LSTM_BATCH_SLICE, rstp->total_depths - base_depth );

		int16_t const * in_a = add_ptrk_int16_bytes( rstp->a_ptr, a_split_stride* base_depth );
		int16_t const * in_b = add_ptrk_int16_bytes( rstp->b_ptr, job_stride_elements*2* ijobno );
		uint8_t *out = rstp->optr +job_stride_elements* ijobno;
		//
		// # of loops is n_groups * (out_depth/128); for a full group this is always
		// exact. For the last group, it may not be; in that case do enough loops to
		// cover it; we may write extra data, but only within the same vector as valid data.

		int n_loops = n_loops_full;
		if( unlikely( n_groups < LSTM_BATCH_SLICE) ){	// partial, at end...
			if( n_groups <= 0) break;
			n_loops = (n_groups * out_depth+127)/128u;
		}

		if( split_buffer != NULL){
			// copy data to split buffer
			vmemcpy_2d_general_asm(
					splitcopywid,		// width of copy
					n_groups,			// height,
				 split_buffer,			// dest buffer
				 splitcopywid,			// stride
				 in_a,					// source buffer
				 a_split_stride);				// stride;
			in_a = (int16_t const*)split_buffer;
		}

		for( int i = 0; i <n_loops; i++){
			HVX_Vector a_in0 = *(HVX_Vector const *)in_a;
			HVX_Vector a_in1 = *(HVX_Vector const *)(in_a+64);
			HVX_Vector b_in0 = *(HVX_Vector const *)in_b;
			HVX_Vector b_in1 = *(HVX_Vector const *)(in_b+64);

			HVX_VectorPair a_tanh = tanh_hvx_i16_quad_x2( a_in0, a_in1, in1_scale, in1_rsh+1, in1_rsh+1 );
			HVX_VectorPair b_tanh = tanh_hvx_i16_quad_x2( b_in0, b_in1, in2_scale, in2_rsh, in2_rsh );
			HVX_Vector abprod_0 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_V_lo_W( a_tanh), Q6_V_lo_W( b_tanh));
			HVX_Vector abprod_1 = Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_V_hi_W( a_tanh), Q6_V_hi_W( b_tanh));

			// find Logistic(a) * Tanh(b) as a full-range i16 value
			HVX_Vector res0 = Q6_Vh_vavg_VhVh_rnd( Q6_V_lo_W(b_tanh), abprod_0);
			HVX_Vector res1 = Q6_Vh_vavg_VhVh_rnd( Q6_V_hi_W(b_tanh), abprod_1);
			// add 0x0080 rounding bias
			res0 = Q6_Vh_vadd_VhVh_sat( res0, q6op_Vh_vsplat_R(0x0080));
			res1 = Q6_Vh_vadd_VhVh_sat( res1, q6op_Vh_vsplat_R(0x0080));
			// now collect the odd bytes... and xor with 0x80
			HVX_Vector res_bytes = Q6_Vb_vpacko_VhVh( res1,res0);
			*(HVX_Vector *)out = Q6_V_vxor_VV( res_bytes,q6op_Vb_vsplat_R(0x80) );

			in_a = add_ptrk_int16_bytes( in_a, a_stride);
			in_b = add_ptrk_int16_bytes( in_b, b_stride);
			out  += out_stride;
		}
	}
	nn_sem_post ( &rstp->done_sem);
}
/////////////////// WORK IN PROGRESS ////////// <<<<<<<



static int lstminput_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking lstm_input node %p",self);
	void * info = nn_calloc( 1, sizeof(struct lstminput_info));
	if( info == 0 ){
		return errlog(nn,"alloc failed");
	}
	self->opaque = info;
	logmsg(nn,2,"lstm_input node %p check OK",self);
	return 0;
}

static int lstmoutput_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking lstm_output node %p",self);
	void * info = nn_calloc( 1, sizeof(struct lstminput_info));
	if( info == 0 ){
		return errlog(nn,"alloc failed");
	}
	self->opaque = info;
	logmsg(nn,2,"lstm_output node %p check OK",self);
	return 0;
}

static int lstm_free(struct nn_node *self, struct nn_graph *nn)
{
	if( self->opaque != NULL){
		nn_free( self->opaque);
		self->opaque = NULL;
	}
	return node_free_common( self, nn);
}



struct nn_node_ops nn_ops_for_QuantizedLstmInput_16x16to16 = {
	.execute = lstminput_execute,
	.check = lstminput_check,
	.ctor = node_alloc_common,
	.dtor = lstm_free,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedLstmOutput_16x16to8 = {
	.execute = lstmoutput_execute,
	.check = lstmoutput_check,
	.ctor = node_alloc_common,
	.dtor = lstm_free,
	.n_inputs = NN_IOCOUNT(7),
	.n_outputs = NN_IOCOUNT(3),
};

