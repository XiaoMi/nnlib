
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code for matrix multiply op
 */

#include <nn_graph.h>
#include <string.h>
#include <stdlib.h>
#include <quantize.h>
#include "hvx_inlines.h"
#include "op_supernode_procweights.h"
#ifndef __hexagon__
#include <malloc.h>
#endif
#include "nn_const_prep_share.h"

#ifdef HEXAGON_V66
#define MATMUL_MAX_THREADS 4
#else
#define MATMUL_MAX_THREADS 2
#endif

/* 8x8 matrix multiply + 32 bias -> 16 bits */

// inputs:
//     0: data, tensor qu8 [b,h,w,d_in]
//     1: matrix, tensor qu8 [1,1,d_in, d_out]		 (assumed to be const)
//     2:  scalar float, min for input data
//     3:  scalar float, max for input data
//	   4:  scalar float, min for matrix input    (assumed to be const)
//	   5:  scalar float, max for matrix input    (assumed to be const)
//     6:  bias tensor, qint32, [1,1,1,d_out]    (assumed to be const)
//     7:  scalar float, min for bias tensor	 (assumed to be const)
//     8:  scalar float, max for bias tensor	 (assumed to be const and -min)
//     9:  scalar float, min for output          (assumed to be const)
//    10:  scalar float, max for output          (assumed to be const and -max)
//
// In the current form, all inputs are assumed to be const (at least, unchanging) except
// for the primary input and input range.
// The bias range and output range are assumed to be symmetric.
//

// define a type for sharing const prep across instances of this node
typedef struct nn_cpshare_base matmul_16_cpshare;	// use the base class
static const struct nn_cpshare_typedesc
matmul_16_cpshare_typedesc = { sizeof(matmul_16_cpshare) };

// persistent info for the op
struct matmul_16_info {
	int in_depth, out_depth;
	int16_t run_yet;					// 0 until first run.
	int16_t using_hvx;					// can use hvx

	// when not using hvx, these are the same as in_depth, out_depth.
	int in_depth_padded;				// input depth padded to multiple of 32
	int out_depth_padded;				// output depth padded to multiple of 32

	// pointer to a (possibly) shared structure containing the b_repacked and b_sums
	matmul_16_cpshare * cpshare_ptr;
	int32_t * b_sums_alloc;			// only used when non-hvx (so we can free b_sums)
	// this is only valid when using_hvx: is repack/reorder of the weights.
	uint8_t * b_repacked;				// vec aligned
	// this is valid either way...
	int32_t * b_sums;					// sum of b over d_in, for d_out in [0..out_depth_padded-1]
	int32_t b_repacked_outer_stride;

	// B input step and zero
	float b_in_step;
	int b_in_zero;

	// bias input step
	float bias_in_step;
};


struct matmul_16_runstate
{
	struct matmul_16_info  const *info;
	struct shape out_shape;
	struct tensor const *a_tensor;
	uint8_t const * b_ptr;
	int16_t * out_ptr;

	int32_t total_batches;		// this is in_b * in_h * inw (# of mat * vector products)
	int32_t batch_chunks;		// # of chunks  'total_batches' divides into
	int32_t odepth_chunks;		// # of jobs 'output_depth' divides into
	int32_t batch_chunk_for_depth_prefech;		// a thread which processes this batch chunk
												// will prefetch the weigths for next odeph.

	int32_t total_jobs;					// this is batch_chunks * odepth_chunks
	volatile int32_t current_job;		// used for threading.
	nn_sem_t done_sem;

	// A input step and zero
	float a_in_step;
	int a_in_zero;

	float output_scale;			// amount to scale SOP->output, as a float
	float bias_scale;			// amount to scale bias->SOP, as a a float

	float out_min, out_max;		// output range

	// output scaling (from SOP to 16-bit output) done as
	//  (1) multiply by a 32-bit amount (with >31 inherent) and rounding
	//  (2) trim to i16 range.
	// This means that 'output_scale' cannot be > 1.0 (well, it can be up to (65537/65536), since we
	// can handle that as 'unity gain' using output_scfac = 0x7fffffff with no error in the 16-bit result).
	// this number is output_scale * 2^31, rounded to nearest and limited to < 2^31.
	int32_t output_scfac;
	// this points to the per-output offset buffer (in scratch); this is in SOP units,
	// and combines the bias (scaled) and the b_sums (*-a_in_zero). Also (d_in *a_in_zero*b_in_zero).
	int32_t * offset_buffer;
};

// to split across threads:
//  - output depth is divided into chunks of 64
//  - output batches are divided into chunks of 16
//
#define MATMUL_ODEPTH_CHUNK 64		// must be power of 2  >= 64
#define MATMUL_OBATCH_CHUNK 16		// must be power of 2 >= 4

static int
matmul_16_setup_info( struct nn_node * self, struct nn_graph *nn)
{
	struct matmul_16_info * info = (struct matmul_16_info *)self->opaque;
	struct tensor const * b_tensor = self->inputs[1];
	struct tensor const * b_min_tensor = self->inputs[4];
	struct tensor const * b_max_tensor = self->inputs[5];
	struct tensor const * bias_tensor = self->inputs[6];

	if( b_tensor->shape.batches != 1 || b_tensor->shape.height != 1){
		return errlog(nn,"matmul: bad B shape");
	}
	int d_in = b_tensor->shape.width;
	int d_out = b_tensor->shape.depth;

	if( bias_tensor->shape.depth != d_out){
		return errlog(nn,"bias depth mismatch");
	}

	info->in_depth = d_in;
	info->out_depth = d_out;
	info->using_hvx = 0;

	int d_in_padded = d_in;
	int d_out_padded = d_out;
	// can use hvx if input depth a multiple of 8 and output depth a multiple of 64
	if( (d_in & 7) == 0  && (d_out&63) == 0 ){
		info->using_hvx = 1;
		d_in_padded = (d_in + 31) & ~31;	// b_repacked is based on this padding.
	}
	info->in_depth_padded = d_in_padded;
	info->out_depth_padded = d_out_padded;

	// find B input range
	float b_in_min = tensor_get_float( b_min_tensor, 0);
	float b_in_max = tensor_get_float( b_max_tensor, 0);

	float b_in_step = flt_div_255(b_in_max-b_in_min);
	int b_in_zero = saturate_u8( roundf_i32( b_in_min *-255.0f/(b_in_max-b_in_min)));

	info->b_in_step = b_in_step;
	info->b_in_zero = b_in_zero;

	{
		struct tensor const * bias_min_tensor = self->inputs[7];
		struct tensor const * bias_max_tensor = self->inputs[8];

		info->bias_in_step = ( tensor_get_float(bias_max_tensor,0)- tensor_get_float(bias_min_tensor,0) ) * (float)(0x1.0p-32);
	}
	if( info->b_sums_alloc != NULL){
		nn_free(info->b_sums_alloc);
		info->b_sums_alloc = NULL;
	}
	if( info->cpshare_ptr != NULL ){
		nn_cpshare_decref( nn, info->cpshare_ptr );
		info->cpshare_ptr = NULL;
	}
	//printf("b_zero = %d; b_step = %f;  bias_step = %f\n", b_in_zero, b_in_step, info->bias_in_step);

	// convert B input as needed
	//

	if( info->using_hvx){
		// try to get shared data from the const
		struct nn_node * wt_nodep = nn_cpshare_get_const_node( nn, self, 1 );
		if( wt_nodep == NULL ) return errlog(nn,"can't get weights const");
		matmul_16_cpshare * cpshare = 
              (matmul_16_cpshare*) nn_cpshare_get_existing( nn, &matmul_16_cpshare_typedesc, wt_nodep );
		if( cpshare == NULL ){	// need to build it.
			cpshare = (matmul_16_cpshare*)nn_cpshare_new( nn, &matmul_16_cpshare_typedesc);
			if( cpshare == NULL ) return errlog(nn,"alloc failed");
			
			int32_t *bsums_p=NULL;
			uint8_t * repacked = nn_memalign( 128, d_in_padded * d_out_padded);
			if(repacked!=NULL) bsums_p = nn_memalign(128, sizeof(int32_t)*d_out_padded);
			if( bsums_p == NULL){
				if( repacked !=NULL ) nn_free( repacked);
				nn_cpshare_decref( nn, cpshare );
				return errlog(nn,"alloc failed");
			}
			// call a function which will reorder the weights and find the sums.
			struct repack_filter_parms rpfp;
			rpfp.filt_tensor = b_tensor;
			rpfp.zero_offset = b_in_zero;
			rpfp.signed_mode_sel = 0;
			rpfp.out_data = repacked;
			rpfp.gemsumb = bsums_p;

			nn_sem_init( &rpfp.done_sem, 0);

			nn_os_work_for_vector( nn,repack_filter_for_d32, &rpfp);
			nn_sem_wait( &rpfp.done_sem);
			cpshare-> ptr_w = repacked;
			cpshare-> ptr_sumb = bsums_p;
			nn_cpshare_attach( nn, wt_nodep, cpshare );
		}
		info->b_repacked_outer_stride = d_in_padded* 32;
		info->cpshare_ptr = cpshare;	// retain for later.
		info->b_sums = cpshare->ptr_sumb;
		info->b_repacked = cpshare->ptr_w;
	}else{
		int32_t *bsums_p = nn_memalign(128, sizeof(int32_t)*d_out_padded);
		if( bsums_p == NULL){
			return errlog(nn,"alloc failed");
		}
		info->b_sums = info->b_sums_alloc = bsums_p;
		// find all the sums
		uint8_t const * bp = (uint8_t const *)b_tensor->data;
		for( int i = 0; i < d_out; i++){
			int sum = 0;
			for( int j = 0; j < d_in; j++){
				sum += bp[i + d_out*j];
			}
			bsums_p[i] = sum;
		}
	}

	info->run_yet = 1;
	return 0;
}

static int
matmul_16_scale_for_run( struct nn_node * self, struct nn_graph *nn, struct matmul_16_runstate *rstp)
{
	struct matmul_16_info const * info = (struct matmul_16_info const *)self->opaque;
	struct tensor const *a_tensor = self->inputs[0];
	struct tensor const *b_tensor = self->inputs[1];
	struct tensor const *a_min_tensor = self->inputs[2];
	struct tensor const *a_max_tensor = self->inputs[3];
	struct tensor const *out_min_tensor = self->inputs[9];
	struct tensor const *out_max_tensor = self->inputs[10];


	rstp->out_shape = a_tensor->shape;
	rstp->a_tensor = a_tensor;
	rstp->b_ptr = (uint8_t const*) b_tensor->data;

	if(rstp->out_shape.depth != info->in_depth){
		return errlog(nn,"shape mismatch: A depth != B width");
	}
	rstp->out_shape.depth = info->out_depth;

	// find input scaling
	float a_in_min = tensor_get_float( a_min_tensor, 0);
	float a_in_max = tensor_get_float( a_max_tensor, 0);

	float a_in_step = flt_div_255(a_in_max-a_in_min);
	int a_in_zero = saturate_u8( roundf_i32( a_in_min *-255.0f/(a_in_max-a_in_min)));
	//printf("a_zero = %d; a_step = %f\n", a_in_zero, a_in_step );

	rstp->a_in_step = a_in_step;
	rstp->a_in_zero = a_in_zero;
	float SOP_step = a_in_step * info->b_in_step;

	float out_min = tensor_get_float( out_min_tensor, 0);
	float out_max = tensor_get_float( out_max_tensor, 0);
	float output_step = (out_max-out_min) * (float)(0x1.0p-16);
	if( !flt_isfinite(output_step)){
		return errlog(nn,"inf or Nan in output range");
	}

	rstp->out_max = output_step * 32768.0f;
	rstp->out_min = -rstp->out_max;

	//printf("out_range = %f .. %f\n", rstp->out_min, rstp->out_max);

	// find scale from SOP to output
	float output_scale = SOP_step/output_step;
	rstp->output_scale = output_scale;
	// find scale from bias to SOP
	rstp->bias_scale = info->bias_in_step/SOP_step;

	//
	// find the quantized scale factor
	// note that scfac in range 0x80000000 .. 0x80008000 all have the same
	// effect in the 16-bit result as 0x7FFFFFFF, after clipping. So we can
	// go a little higher than 1.0.
	//
	uint32_t scfac = roundf_u32( output_scale * (float)0x1.0p31);
	if( output_scale > 1.5f || scfac > ((1u<<31)+0x8000) || scfac < 0x4000){
		return errlog(nn,"can't scale output by %f (out limit is +/-%f, needs to be >=+/-%f)",
				output_scale, output_step*32768.0f,output_step*32768.0f*output_scale);
	}
	rstp->output_scfac = min_u32( scfac, 0x7FFFFFFF);
	//printf("out_scfac = %d;  bias_scale = %f\n",(int) rstp->output_scfac, rstp->bias_scale);
	return 0;
}
//
// make the 'offset buffer', [out_depth_padded], which is sum of
//      bias[i] (from input) scaled by bias_scale
//      -a_in_zero * b_sum[i]
//      a_in_zero * b_in_zero * in_depth_padded

static int
make_offset_buf( struct nn_graph * nn, struct matmul_16_runstate * rstp, struct tensor const *bias_tensor )
{
	struct matmul_16_info const * info = rstp->info;
	int dout_padded = info->out_depth_padded;
	int dout = info->out_depth;				// could be less than padded
	int32_t *offs_buf = rstp->offset_buffer;

	// First the bias, scaled by bias_scale
	int32_t const *bias_in = (int32_t const*)bias_tensor->data;
	// convert the scaling w/22 fractional bits
	uint32_t scaling = roundf_u32( rstp->bias_scale * (float)(0x1.0p22) );
	// special case when result == (1<<22)
	if( scaling == (1<<22)){
		memcpy( offs_buf, bias_in, dout*sizeof(int32_t));
	}else{
		if( scaling >= (1u<<31)){
			return errlog(nn,"can't scale bias");
		}
		for( int i =0; i < dout; i++){
			int64_t p = (int64_t)bias_in[i] * (int32_t)scaling;
			offs_buf[i] = (int32_t)( (p  + (1<<21)) >> 22 );
		}
	}
	if( dout_padded > dout){
		memset( &offs_buf[dout],0, (dout_padded-dout)*sizeof(int32_t));
	}

	// now do the SOP biases
	int a_zero = rstp->a_in_zero;
	if( a_zero > 0){		// if 0, we don't need to do this
		int Nb_zero = info->b_in_zero * info->in_depth_padded;
		int32_t const * bsum = info->b_sums;
		for( int i = 0; i < dout_padded; i++){
			offs_buf[i] += a_zero*( Nb_zero - bsum[i]);
		}
	}
	return 0;
}

static void matmul_16_work_ref( struct nn_graph * nn, void * rstpv);
static void matmul_16_work_hvx( struct nn_graph * nn, void * rstpv);

static int matmul_16_execute( struct nn_node * self, struct nn_graph * nn)
{
	struct tensor const * bias_tensor = self->inputs[6];
	struct tensor * out_tensor = self->outputs[0];
	struct tensor * out_min_tensor = self->outputs[1];
	struct tensor * out_max_tensor = self->outputs[2];

	struct matmul_16_info * info = (struct matmul_16_info *)self->opaque;
	if( !info->run_yet){			// need initial setup
		if( matmul_16_setup_info(self,nn)!=0){
			return -1;
		}
	}
	struct matmul_16_runstate runstate;
	runstate.info = info;


	if ( matmul_16_scale_for_run( self,nn, &runstate)!= 0 ){
		return -1;
	}
	// allocate the depth offset buffer
	nn_scratch_reset(nn);
	int32_t * offs_buf = nn_scratch_alloc( nn, info->out_depth_padded * sizeof(int32_t));
	if( offs_buf == NULL){
		return errlog(nn,"scratch alloc failed");
	}
	runstate.offset_buffer = offs_buf;
	if( make_offset_buf( nn, &runstate , bias_tensor) != 0 ) return -1;
	runstate.total_batches = runstate.out_shape.batches * runstate.out_shape.height * runstate.out_shape.width;

	// The operation is:
	//    [b,h,w,din] * [ 1,1,din,dout]  -> [ 1,1,b*h*w,dout]
	//    except for the QuantizedMatMulDims_8x8p32to16 variant which has -> [b,h,w,dout]
	//
	if( self->node_type != OP_QuantizedMatMulDims_8x8p32to16){
		runstate.out_shape.batches = 1;
		runstate.out_shape.height = 1;
		runstate.out_shape.width = runstate.total_batches;
	}
	//
	// setup the output
	//
	if( tensor_out_prepare_normal_fromshape( out_tensor, &runstate.out_shape, NN_TYPE_QINT16)!= 0){
		return errlog(nn,"output too small");
	}
	// set up the 'jobs'
	// We chop the batches up into units of MATMUL_OBATCH_CHUNK,
	// and the output depth into units of MATMUL_ODEPTH_CHUNK, resulting in a 'grid'
	// of jobs. These are processed across batches first, and then across output depth,
	// so as to keep the same set of weights in cache.
	// Initial set of weights are prefetched now for the first MATMUL_ODEPTH_CHUNK;
	// the next set are prefetched  by thread which reaches the batch-chunk numbered batch_chunk_for_depth_prefech.
	//
	
	runstate.batch_chunks = (runstate.total_batches + (MATMUL_OBATCH_CHUNK-1))/(unsigned)MATMUL_OBATCH_CHUNK;
	runstate.odepth_chunks = (info->out_depth  + (MATMUL_ODEPTH_CHUNK-1))/(unsigned)MATMUL_ODEPTH_CHUNK;
	runstate.total_jobs = runstate.batch_chunks *  runstate.odepth_chunks;
	runstate.out_ptr = (int16_t *) out_tensor->data;

	int nthreads = min_i32(MATMUL_MAX_THREADS, runstate.total_jobs);

	runstate.batch_chunk_for_depth_prefech = max_i32(0, runstate.odepth_chunks - nthreads);

	// prefetch the first set of weights
	if( info->using_hvx)
		l2fetch( info->b_repacked, 128, 128, (info->b_repacked_outer_stride * (MATMUL_ODEPTH_CHUNK/32)) / 128u );

	runstate.current_job = 0;
	nn_sem_init( & runstate.done_sem, 0);

	void (*run_func)( struct nn_graph *, void *)  = 
		info->using_hvx ? matmul_16_work_hvx: matmul_16_work_ref;
	
	for(int i = 0;  i < nthreads; i++){
		nn_os_work_for_vector( nn, run_func, &runstate);
	}
	tensor_set_single_float( out_min_tensor, runstate.out_min );
	tensor_set_single_float( out_max_tensor, runstate.out_max );

	nn_sem_wait_n_times( &runstate.done_sem, nthreads);

	return 0;
}

static void
matmul_16_work_ref( struct nn_graph * nn, void * rstpv)
{
	struct matmul_16_runstate * rstp = (struct matmul_16_runstate*)rstpv;
	struct matmul_16_info const * info = rstp->info;

	int d_in = info->in_depth;
	int d_out = info->out_depth;
	int jobno;
	int total_jobs = rstp->total_jobs;
	unsigned batch_chunks = rstp->batch_chunks;
	unsigned d_per_job = MATMUL_ODEPTH_CHUNK;
	unsigned b_per_job = MATMUL_OBATCH_CHUNK;
	int b_zero = info->b_in_zero;
	int32_t oscale = rstp->output_scfac;

	uint8_t const * inptr0 = (uint8_t const*)rstp->a_tensor->data;
	uint8_t const * bptr0 = (uint8_t const*)rstp->b_ptr;
	int16_t * outptr0 = (int16_t *)rstp->out_ptr;


	while(   jobno = __sync_fetch_and_add( &rstp->current_job,1),  jobno < total_jobs){
		int depth_idx = jobno/batch_chunks;
		int batch_idx = jobno - depth_idx * batch_chunks;
		int dstart = depth_idx * d_per_job;
		int dno = min_i32( d_per_job, d_out-dstart);	// # of dout to do in this run
		int bstart = batch_idx * b_per_job;
		int bno = min_i32( b_per_job, rstp->total_batches-bstart);	// # of batch to do in this run
		if( dno <= 0 ) continue;		// ??
		
		int32_t const * offs_buf = &rstp->offset_buffer[dstart];
		for( int batchno = bstart; batchno < bstart + bno; batchno ++ ){
			uint8_t const * inptr = inptr0 + d_in * batchno;		// get 'a' pointer
			uint8_t const * bptr = bptr0 + dstart;				// get 'b' pointer
			int16_t * outptr = outptr0 + dstart + d_out * batchno;	// get output pointer

			// first one - include sum of A
			int asum_bzero;
			{
				int asum = 0;
				int sum = offs_buf[0];
				for( int j = 0; j < d_in; j++ ){
					int aval = inptr[j];
					asum += aval;
					sum += aval*bptr[d_out*j];
				}
//printf("asum= %d\n", asum);
				asum_bzero = b_zero*asum;
				sum -= asum_bzero;
				sum = ((int64_t)sum * oscale + (1u<<30)) >> 31;
				*outptr++= saturate_i16( sum );
			}
			// the rest of the outputs can re-use the asum
			for( int i = 1; i < dno; i++){
				bptr ++;			// next set of 'b'
				int sum = offs_buf[i];
				for( int j = 0; j < d_in; j++ ){
					int aval = inptr[j];
					sum += aval*bptr[d_out*j];
				}
				sum -= asum_bzero;
				sum = ((int64_t)sum * oscale + (1u<<30)) >> 31;
				*outptr++= saturate_i16( sum );
			}
		}
	}
	nn_sem_post( &rstp->done_sem);
}

static void matmul_hvx_loop_4batch( struct matmul_16_runstate * rstp,
		uint8_t const * a_ptr,	uint8_t const * b_slice,
		int32_t const * depth_offsets,	int16_t * out_ptr);
static void matmul_hvx_loop_2batch( struct matmul_16_runstate * rstp,
		uint8_t const * a_ptr,	uint8_t const * b_slice,
		int32_t const * depth_offsets,	int16_t * out_ptr,
		int nbatch 	);


static void
matmul_16_work_hvx( struct nn_graph * nn, void * rstpv)
{
	struct matmul_16_runstate * rstp = (struct matmul_16_runstate*)rstpv;
	struct matmul_16_info const * info = rstp->info;
	int b_repacked_outer_stride = info->b_repacked_outer_stride;

	int d_in = info->in_depth;
	int d_out = info->out_depth;
	int jobno;
	int total_jobs = rstp->total_jobs;
	unsigned batch_chunks = rstp->batch_chunks;
	unsigned d_per_job = MATMUL_ODEPTH_CHUNK;
	unsigned b_per_job = MATMUL_OBATCH_CHUNK;

	uint8_t const * inptr0 = (uint8_t const*)rstp->a_tensor->data;
	uint8_t const * bptr0 = (uint8_t const*)info->b_repacked;
	int16_t * outptr0 = (int16_t *)rstp->out_ptr;

	unsigned b_depth_chunk_bytes = b_repacked_outer_stride * (MATMUL_ODEPTH_CHUNK/32);

#if  MATMUL_ODEPTH_CHUNK!=64
#error "this needs rework"
#endif

	// "bsdecode" is used to decompose depth_idx = jobno/batch_chunks; batch_idx = jobno% batch_chunks
	// without a divide
	batchslice_decode bsdecode;
	batchslice_decode_init( &bsdecode, batch_chunks);

	while(   jobno = __sync_fetch_and_add( &rstp->current_job,1),  jobno < total_jobs){
		int batch_idx = batchslice_decode_update( &bsdecode, jobno);	// faster count: batch chunk indx
		int depth_idx = bsdecode.ibatch;								// slower: depth chunk index
		int dstart = depth_idx * d_per_job;
		int dno = min_i32( d_per_job, d_out-dstart);	// # of dout to do in this run
		int bstart = batch_idx * b_per_job;
		int bno = min_i32( b_per_job, rstp->total_batches-bstart);	// # of batch to do in this run
		//printf("batches %d..%d; depth %d..%d\n", bstart, bstart+ bno-1, dstart, dstart+dno-1);
		if( dno != 64 ) continue;		// ??

		int batchno = bstart;
		uint8_t const * inptr = inptr0 + d_in * batchno;		// get 'a' pointer
		uint8_t const * bptr = bptr0 + b_repacked_outer_stride*(dstart>>5);// get 'b' pointer
		//  prefetch the current activations
		l2fetch( inptr, d_in, d_in, bno );

		// prefetch the weights for next output depth??
		if( batch_idx == rstp->batch_chunk_for_depth_prefech && dstart + dno < d_out ){
			l2fetch( bptr + b_depth_chunk_bytes, 128, 128, b_depth_chunk_bytes/ 128u );
		}
		
		int32_t const * offs_buf = &rstp->offset_buffer[dstart];
		
		int16_t * outptr = outptr0 + dstart + d_out * batchno;	// get output pointer
		while( bno >= 4 ){	// we can do 4 at once
			matmul_hvx_loop_4batch( rstp, inptr, bptr, offs_buf, outptr );
			inptr += d_in * 4;
			outptr += d_out * 4;
			bno -= 4;
		}
		// and then 2 at once (or 1 )
		while (bno > 0){
			matmul_hvx_loop_2batch( rstp, inptr, bptr, offs_buf, outptr, bno );
			inptr += d_in * 2;
			outptr += d_out * 2;
			bno -= 2;
		}
	}
	nn_sem_post( &rstp->done_sem);
}

// add the 8 pixels in pixels0 together; acc to LS word of accs
// add the 8 pixels in pixels1 together; acc to MS word of accs
// return new acc.
static inline
uint64_t acc_two_8pix( uint64_t accs,  uint64_t pixels0, uint64_t pixels1)
{
	// rearrange the pixels into 32-bit lane, for vraddub operarion
	uint64_t inputA = Q6_P_combine_RR( (uint32_t) pixels1, (uint32_t) pixels0); 
	uint64_t inputB = Q6_P_combine_RR( (uint32_t) (pixels1>>32), (uint32_t)( pixels0>>32)); 
	return Q6_P_vraddubacc_PP( accs, inputA, inputB);
}

//
// mul the 8 values in 'avals' by the 64x8 matrix in wts0_0 wts0_1 wts1_0 wts1_1
// and add the 64 32-bit results to 'acc'.

static inline
HVX_Vector_x2
accum_8x64( HVX_Vector_x2 acc, 
	HVX_Vector wts0_0, HVX_Vector wts0_1, HVX_Vector wts1_0, HVX_Vector wts1_1,
		uint64_t avals )
{
	HVX_Vector acc0 = Q6_Vuw_vrmpyacc_VuwVubRub( acc.val[0], wts0_0, (uint32_t)avals);		// 32 outputs
	HVX_Vector acc1 = Q6_Vuw_vrmpyacc_VuwVubRub( acc.val[1], wts0_1, (uint32_t)avals);		// next 32 outputs
	acc0 = Q6_Vuw_vrmpyacc_VuwVubRub( acc0, wts1_0, (uint32_t)(avals>>32));		// 32 outputs
	acc1 = Q6_Vuw_vrmpyacc_VuwVubRub( acc1, wts1_1, (uint32_t)(avals>>32));		// next 32 outputs
	HVX_Vector_x2 res  = {{ acc0,acc1}};
	return res;
}

//
//   multiply the a_sum by b_zero, subtract from the two 32-bit accs;
//   then scale by final_scale.
static inline
HVX_Vector_x2
sub_asum_and_scale( HVX_Vector_x2 acc, int b_zero, int32_t a_sum, HVX_Vector final_scale )
{
	HVX_Vector corr = Q6_V_vsplat_R( b_zero * a_sum);
	HVX_Vector acc0 = Q6_Vw_vsub_VwVw(acc.val[0], corr);
	HVX_Vector acc1 = Q6_Vw_vsub_VwVw(acc.val[1], corr);
	acc0 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( acc0, final_scale );
	acc1 = q6op_Vw_vmpy_VwVw_s1_rnd_sat( acc1, final_scale );
	HVX_Vector_x2 res  = {{ acc0,acc1}};
	return res;
}


// This function is what's in the main hvx loop, so we can can unroll it 4 x easily

struct matmul_loopstate {
	HVX_Vector_x2 acc0,acc1,acc2,acc3;		// output sums 64 xi32 in 4 batches
	uint64_t sum01,sum23;		// pixel sums in 32-bit regs
};

static inline void
__attribute__((always_inline))
matmul_loop_core(
		struct matmul_loopstate *loopstatep,
		uint8_t const *a_ptr,
		int in_depth,
		uint8_t const *b_slice,
		int b_repacked_outer_stride,
		int prefetch_sel )		 // 1 on first unroll, 2 on second, 0 on others.
{
	uint64_t a0_8vals = *(uint64_t const *)a_ptr;
	// weights for 1st 4 inputs, across 64 outputs
	HVX_Vector wts0_0 = *(HVX_Vector const *)b_slice;
	HVX_Vector wts0_1 = *(HVX_Vector const *)(b_slice + b_repacked_outer_stride);
	b_slice += 128;
	// weights for 2nd 4 inputs, across 64 outputs
	HVX_Vector wts1_0 = *(HVX_Vector const *)b_slice;
	HVX_Vector wts1_1 = *(HVX_Vector const *)(b_slice + b_repacked_outer_stride);

	loopstatep->acc0 =  accum_8x64( loopstatep->acc0, wts0_0, wts0_1, wts1_0, wts1_1, a0_8vals );
	// next batch
	uint64_t a1_8vals = *(uint64_t const *)(a_ptr + in_depth);

	if( prefetch_sel ==1) Q6_dcfetch_A( a_ptr + 64);
	if( prefetch_sel ==2) Q6_dcfetch_A( a_ptr + 2*in_depth+ 64-8);

	loopstatep->acc1 =  accum_8x64( loopstatep->acc1, wts0_0, wts0_1, wts1_0, wts1_1, a1_8vals );
	loopstatep->sum01 = acc_two_8pix( loopstatep->sum01, a0_8vals, a1_8vals);

	if( prefetch_sel ==1) Q6_dcfetch_A( a_ptr + in_depth+ 64);
	if( prefetch_sel ==2) Q6_dcfetch_A( a_ptr + 3*in_depth+ 64-8);

	// next batch
	uint64_t a2_8vals = *(uint64_t const *)(a_ptr + 2*in_depth);
	loopstatep->acc2 =  accum_8x64( loopstatep->acc2, wts0_0, wts0_1, wts1_0, wts1_1, a2_8vals );
	// next batch
	uint64_t a3_8vals = *(uint64_t const *)(a_ptr + 3*in_depth);
	loopstatep->acc3 =  accum_8x64( loopstatep->acc3, wts0_0, wts0_1, wts1_0, wts1_1, a3_8vals );
	loopstatep->sum23 = acc_two_8pix( loopstatep->sum23, a2_8vals, a3_8vals);
}

//
// hvx inner loop: perform the operation on 64 output depths (starting at a point which is a multiple of 64)
// and process 4 adjacent batches, This operation requires the 'A' input depth to
// be a multiple of 8 so we can read the 'A' values using 64-bit ops; and it requires the output depth to be a multiple of 64.
//
// Note that the B data has been rearranged as
//      [ dout_hi ] [ din_hi ] [ dout_lo=32] [din_lo=4]
// where 'din_hi' dimension is in_depth/4, padded up to a multiple of 8;
// and dout_hi is output depth dimension divided by 32.
// So, to process 64 output depths at once, we need two adjacent pointers to that upper index.
//
//
static void
matmul_hvx_loop_4batch(
		struct matmul_16_runstate * rstp,
		uint8_t const * a_ptr,			// points to 'A' side (multiple of 8)
		uint8_t const * b_slice,		// points to processed 'B' weights (vector aligned, offset according to output depth start)
		int32_t const * depth_offsets,	// points to 64 'depth' offset values (vector aligned).
		int16_t * out_ptr)				// output, vector aligned.
{
	struct matmul_16_info const * info = rstp->info;
	int in_depth = info->in_depth;					// input depth and a_batch-pitch  (must be a multiple of 8).
	int out_depth = info->out_depth;				// used to determine output pitch when writing results.
	int b_repacked_outer_stride = info->b_repacked_outer_stride;

	Q6_dcfetch_A( a_ptr );
	Q6_dcfetch_A( a_ptr+ in_depth);
	Q6_dcfetch_A( a_ptr+ 2*in_depth );
	Q6_dcfetch_A( a_ptr+3*in_depth );

	struct matmul_loopstate loopstate;

	loopstate.acc0.val[0] =  *(HVX_Vector const*)&depth_offsets[0];		// initialize the accums
	loopstate.acc0.val[1] =	*(HVX_Vector const*)&depth_offsets[32];

	int bzero = info->b_in_zero;
	int nloops = in_depth>>5;			// do 32 in-depths per loop

	HVX_Vector final_scale = Q6_V_vsplat_R(rstp->output_scfac);
	// set  up accs for four adjacent batches, 64 results per batch

	loopstate.acc1 = loopstate.acc0;
	loopstate.acc2 = loopstate.acc0;
	loopstate.acc3 = loopstate.acc0;

	loopstate.sum01 = 0;  // pixel sums in batches 0,1
	loopstate.sum23 = 0;		// 2,3

	// unroll the inner op 4 times, to amortize the prefetch.
	// the prefetches are distributed through the op, the compiler handles them badly if they are all together.
	for( int i =0 ; i < nloops; i++){
		matmul_loop_core( &loopstate, a_ptr, in_depth, b_slice, b_repacked_outer_stride, 1);
		b_slice += 256;
		a_ptr += 8;
		matmul_loop_core( &loopstate, a_ptr, in_depth, b_slice, b_repacked_outer_stride, 2 );
		b_slice += 256;
		a_ptr += 8;
		matmul_loop_core( &loopstate, a_ptr, in_depth, b_slice, b_repacked_outer_stride, 0 );
		b_slice += 256;
		a_ptr += 8;
		matmul_loop_core( &loopstate, a_ptr, in_depth, b_slice, b_repacked_outer_stride, 0 );
		b_slice += 256;
		a_ptr += 8;
	}
	// depth not a multiple of 32?
	for( int i = 0; i < ((in_depth>>3)&3); i++){
		matmul_loop_core( &loopstate, a_ptr, in_depth, b_slice, b_repacked_outer_stride,0 );
		b_slice += 256;
		a_ptr += 8;
	}
	// roll in the bzero * asum
	
	HVX_Vector_x2  res0 = sub_asum_and_scale( loopstate.acc0, bzero, (uint32_t)loopstate.sum01, final_scale );
	HVX_Vector_x2  res1 = sub_asum_and_scale( loopstate.acc1, bzero, (uint32_t)(loopstate.sum01>>32), final_scale );
	HVX_Vector_x2  res2 = sub_asum_and_scale( loopstate.acc2, bzero, (uint32_t)loopstate.sum23, final_scale );
	HVX_Vector_x2  res3 = sub_asum_and_scale( loopstate.acc3, bzero, (uint32_t)(loopstate.sum23>>32), final_scale );

	// now we need to saturate to i16, and then pack the 64 results from each
	// batch into one register.
	HVX_Vector res_01_lo = Q6_Vh_vsat_VwVw( res1.val[0], res0.val[0]);	// first 32 from each of 2 batches, interleaved
	HVX_Vector res_01_hi = Q6_Vh_vsat_VwVw( res1.val[1], res0.val[1]);	// remaining 32 from  each of 2 batches
	HVX_VectorPair final_01 = Q6_W_vdeal_VVR( res_01_hi, res_01_lo, -2);	// deal them out to proper order

	HVX_Vector res_23_lo = Q6_Vh_vsat_VwVw( res3.val[0], res2.val[0]);	// first 32 from each of 2 batches, interleaved
	HVX_Vector res_23_hi = Q6_Vh_vsat_VwVw( res3.val[1], res2.val[1]);	// remaining 32 from  each of 2 batches
	HVX_VectorPair final_23 = Q6_W_vdeal_VVR( res_23_hi, res_23_lo, -2);	// deal them out

	*(HVX_Vector *)out_ptr = Q6_V_lo_W(final_01);
	out_ptr += out_depth;
	*(HVX_Vector *)out_ptr = Q6_V_hi_W(final_01);
	out_ptr += out_depth;
	*(HVX_Vector *)out_ptr = Q6_V_lo_W(final_23);
	out_ptr += out_depth;
	*(HVX_Vector *)out_ptr = Q6_V_hi_W(final_23);
}
// same but only does 1 or 2 batches.
// when nbatch = 1, it will run two batches with the same data and only save one result).
// when nbatch >=2 it will process 2 batches.
static void
matmul_hvx_loop_2batch(
		struct matmul_16_runstate * rstp,
		uint8_t const * a_ptr,			// points to 'A' side (multiple of 8)
		uint8_t const * b_slice,		// points to processed 'B' weights (vector aligned, offset according to output depth start)
		int32_t const * depth_offsets,	// points to 64 'depth' offset values (vector aligned).
		int16_t * out_ptr,				// output, vector aligned.
		int nbatch 	)				// must be 1 or >=2
{
	struct matmul_16_info const * info = rstp->info;
	int in_depth = info->in_depth;					// input depth and a_batch-pitch  (must be a multiple of 8).
	int out_depth = info->out_depth;				// used to determine output pitch when writing results.
	int b_repacked_outer_stride = info->b_repacked_outer_stride;


	HVX_Vector_x2 acc0 = {{ *(HVX_Vector const*)&depth_offsets[0],		// initialize the accums
						*(HVX_Vector const*)&depth_offsets[32] }};

	int bzero = info->b_in_zero;
	int nloops = in_depth>>3;			// do 8 in-depths per loop

	HVX_Vector final_scale = Q6_V_vsplat_R(rstp->output_scfac);

	Q6_dcfetch_A( a_ptr );
	Q6_dcfetch_A( a_ptr+ in_depth);

	// set  up accs for four adjacent batches, 64 results per batch

	HVX_Vector_x2 acc1 = acc0;

	uint64_t sum01 = 0;  // pixel sums in batches 0,1
	int a_ptr_offs = (nbatch >= 2)? in_depth: 0;

	for( int i =0 ; i < nloops; i++){
		uint64_t a0_8vals = *(uint64_t const *)a_ptr;
		HVX_Vector wts0_0 = *(HVX_Vector const *)b_slice;
		HVX_Vector wts0_1 = *(HVX_Vector const *)(b_slice + b_repacked_outer_stride);
		b_slice += 128;
		HVX_Vector wts1_0 = *(HVX_Vector const *)b_slice;
		HVX_Vector wts1_1 = *(HVX_Vector const *)(b_slice + b_repacked_outer_stride);
		b_slice += 128;
		acc0 =  accum_8x64( acc0, wts0_0, wts0_1, wts1_0, wts1_1, a0_8vals );
		Q6_dcfetch_A( a_ptr+64 );

		// next batch (or same if nbatch = 1)
		uint64_t a1_8vals = *(uint64_t const *)(a_ptr + a_ptr_offs);
		acc1 =  accum_8x64( acc1, wts0_0, wts0_1, wts1_0, wts1_1, a1_8vals );
		Q6_dcfetch_A( a_ptr+ in_depth+64);
		sum01 = acc_two_8pix( sum01, a0_8vals, a1_8vals);
		a_ptr += 8;
	}
	// roll in the bzero * asum
	HVX_Vector_x2  res0 = sub_asum_and_scale( acc0, bzero, (uint32_t)sum01, final_scale );
	HVX_Vector_x2  res1 = sub_asum_and_scale( acc1, bzero, (uint32_t)(sum01>>32), final_scale );

	// now we need to saturate to i16, and then pack the 64 results from each
	// batch into one register.
	HVX_Vector res_01_lo = Q6_Vh_vsat_VwVw( res1.val[0], res0.val[0]);	// first 32 from each of 2 batches, interleaved
	HVX_Vector res_01_hi = Q6_Vh_vsat_VwVw( res1.val[1], res0.val[1]);	// remaining 32 from  each of 2 batches
	HVX_VectorPair final_01 = Q6_W_vdeal_VVR( res_01_hi, res_01_lo, -2);	// deal them out to proper order

	*(HVX_Vector *)out_ptr = Q6_V_lo_W(final_01);
	if( nbatch >= 2 ){
		out_ptr += out_depth;
		*(HVX_Vector *)out_ptr = Q6_V_hi_W(final_01);
	}
}


static int matmul_16_check(struct nn_node *self, struct nn_graph *nn)
{
	/*  11 inputs,  3 output */
	void * infop = nn_calloc( 1, sizeof( struct matmul_16_info));
	if( infop == NULL){
		return errlog( nn,"can't alloc %u bytes", (unsigned)sizeof(struct matmul_16_info));
	}
	self->opaque = infop;
	return 0;
}
static int matmul_16_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct matmul_16_info * info = (struct matmul_16_info *)self->opaque;
	if( info != NULL){
		if( info->cpshare_ptr != NULL )
			nn_cpshare_decref( nn, info->cpshare_ptr );
		if( info->b_sums_alloc != NULL){ nn_free( info->b_sums_alloc);  }
		nn_free( self->opaque);
		self->opaque = NULL;
	}
	return node_free_common(self, nn);
}


struct nn_node_ops nn_ops_for_QuantizedMatMul_8x8p32to16 = {
	.execute = matmul_16_execute,
	.check = matmul_16_check,
	.ctor = node_alloc_common,
	.dtor = matmul_16_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_QuantizedMatMul_8x8p32to16_ref = {
	.execute = matmul_16_execute,
	.check = matmul_16_check,
	.ctor = node_alloc_common,
	.dtor = matmul_16_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};

// This is like QuantizedMatMul_8x8p32to16, but it does
// not flatten dims, e.g. [b,h,w,din] * [ 1,1,din,dout] -> [b,h,w,dout]
struct nn_node_ops nn_ops_for_QuantizedMatMulDims_8x8p32to16 = {
	.execute = matmul_16_execute,
	.check = matmul_16_check,
	.ctor = node_alloc_common,
	.dtor = matmul_16_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};
