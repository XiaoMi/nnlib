
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
 * This contains the code for matmul
 */
#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef __hexagon__
#include "hexagon_types.h"
#else
#include <malloc.h>
#endif
#include "hvx_hexagon_protos.h"
#include "op_supernode_procweights.h"
#include "nn_bufferpool.h"
#include "nn_const_prep_share.h"

#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif

/*  --> 32 bits, biasadd,x multiply relu, quantizedown to 8 bits  */

/*
 * Input and output have ordering BD
 * Filter has ordering DB (B is # of filters)
 */

#ifdef __hexagon__
#include <hexagon_protos.h>
#endif

/*
 * This structure has values that change with different work items
 */

struct workitem {
	int (*execute)(struct workitem *, struct nn_node *node, struct nn_graph *nn);	// exec function
	struct nn_node *self;		// This node
	struct superfc_info *info;	// same as self->opaque
	nn_sem_t *donesem;	// semaphore to post completion
        int32_t  wait_before;   // wait on the semphores or not
	int32_t  threads;       // how many threads will work on this.

	/* Main matmulal work items */
	size_t indata_offset;		// Input data.  Offset from info->input_base.
	const uint8_t *weights;	// Filter data.  Could be from input tensor or temp buf
	const int32_t *biases;	// Bias data, in product space (added to in * filt products)
	uint8_t *output;	// Output location in output tensor
	uint8_t *output_t;	// Output location in output tensor (or temp buf, if misaligned out depth)
	int32_t *suma_buf;	// Output data.  This is batch long
	int32_t start_batch;	// batch to start working on
	int32_t stop_batch;	// batch too far to work on
	int32_t num_batches;	// Number of batches to skip each iteration
	int32_t *minmax_buf;	// min/max values
	int32_t weight_chunks;	// How many d32 chunks of weights to do
	int32_t weight_batch_size; // How big is a d32 chunk of weights
	int32_t start_chunk;    // the first absolute position of the weight chunks to use
	int32_t actual_odepth;		// actual output depth (may be < weight_chunks*32, due to output depth padding)
	const uint8_t * *ptr_in_batches; //list of pointers to the input batches
	uint8_t * *ptr_out_batches; //list of pointers to the output batches

	/* Information for prefetching */
	const uint8_t *pf_inp;	// where to start prefetch, NULL if no pf needed
	uint32_t pf_width;	// width to fetch
	uint32_t pf_stride;	// Distance between rows
	uint32_t pf_height;	// number of rows;


	int32_t join_iters;	// how many times to decrement done semaphore
};

struct superfc_cpshare_type;
/*
 * Pointers and values that are common to everything in the node
 * Also values that are constant across all work items
 */
struct superfc_info {
	uint8_t *weights;	// weights, padded and adjusted as necessary (not owned memory)
	int32_t *biasbuf;	// int32 bias buffer, including min offsets and gemsumb
	int32_t *minmax_buf;	// pointer to min/max values, enough storage per thread...
	nn_sem_t *semaphores;	// pointer to preallocated array of semaphores
	struct workitem *work_items;	// All the work items to execute at execute time
	int n_work_items;		// how many work items?
	int workitems_alloc;	//	bytes allocated for work items
	float out_minval;	// Minimum output value, either specified or guessed
	float out_maxval;	// maximum output value, either specified or guessed
	int minval_precalculated;	// Is the minval precalculated?
	int maxval_precalculated;	// Is the maxval precalculated?
	float out_minval_spec;		// exact value specified (when not precalculated)
	float out_maxval_spec;		// exact value specified (when not precalculated)
	int32_t minval;			// Minimum value (in prod space) actually observed
	int32_t maxval;			// Maximum value (in prod space) actually observed
	int32_t weight_batch_size;	// How big is a weight batch (32 filters)
	int32_t n_weight_batches;	// Number of weight batches we can try and fit at once into vtcm
	int32_t needs_retry;		// Do we need to try this op over again?
	int32_t strategy_valid;		// Do we believe the strategy is currently valid?
	int32_t weights_arranged;	// Have the weights been rearranged yet?
	float in_max_float;	// maximum input float value
	float in_min_float;	// minimum input float value
	float weights_level_size;	// how large in float is one increment in the weights?
	int weights_offset;	// where is 0 in weight number space?
	struct shape inshape;		// actual input tensor shape (to check for recalc - see [1])

	int32_t batches;	// total # batches b*h*w
	int32_t in_depth;	// input depth to compute (set in 'check')
	int32_t in_depth_pad;	// input depth to compute padded out to neartest 32
	int32_t out_depth;	// total depth to compute (set in 'check')
	int32_t out_depth_pad;	// amount to not store i.e. actual output depth pad = out_depth + nearest 32
	int32_t *suma_buf;	// GEMSUMA (if needed)
	int32_t recip_val;	// reciprocal for product space --> output space
	int32_t recip_shamt;	// amount to shift before recip mpy
	int in_offset;		// amount to normalize inputs by.  Needed?
	int filt_offset;	// amount to normalize filter values by. Needed?
	int32_t recursion_depth;// how far have we recursed?

	const uint8_t *input_base;	// first row (including in-use left padding, in-use top padding).
	const uint8_t *weights_base;
	int32_t max_valid_val;	// maximum value that results in a value not above max_out
	int32_t min_valid_val;	// minimum value that results in a value not below min_out
	float prod_level_size;	// in_level_size * filt_level_size
        float output_level_size; // output level size
        uint8_t * out_temp_buf; // temporary output buffer aligned
	/* Information for GEMSUMA / integral calculations */
	uint8_t *need_initialize_suma;
	struct superfc_cpshare_type * cpshare;	// pointer to  object shared with other instances.
	int32_t *gemsumb;	// GEMSUMB value, if we want to calculate it at preparation time (not owned memory)
	uint64_t cycles;	// Cycle accumulator for children
};



// Note: typically weight shape is (1,1,fd,dout), input shape is (b,1,1,fd)  -> (1,1,b,dout)
// but in general: weight shape is (fh,fw,fd,dout) input shape is (b,h,w,d)
//   ... which is done as matrix product  (bin, din) * (din,dout) -> (1,1,bin,dout)
//   ... where din = fh*fw*fd, and
//            bin = b*h*w*d/din  must divide exactly.
// The in_depth and batches reflect the size of the computation (they are din and bin) but
// the 'inshape' is the actual input tensor shape, and is checked to see if we need to recalc.
//


static int setup_initial_output_range( struct nn_graph *nn, struct superfc_info *info, float,float,float,float);

#define roundup(a, p2)       (((a)+(p2)-1)&~((p2)-1))

static inline int superfc_n_weight_batches(int batch_size)
{
	int slices_per_256 = (384*1024/batch_size);
	if (slices_per_256 > 0) return slices_per_256;
	else return 1;
}

static void superfc_statistics(struct nn_graph *nn, struct superfc_info *node, struct nn_node *self)
{
	int d,dd;
	const uint8_t *in;
	const uint8_t *in_d;
	uint32_t word;
	uint32_t word_count = 0;
	uint32_t zero_word_count = 0;
		in = node->input_base ;
		for (d = 0; d < node->in_depth/32; d++) {
			in_d = in + d*32;
			for (dd = 0; dd < 32; dd += 4) {
				word = *((const uint32_t *)(in_d+dd));
				if (word == 0) zero_word_count++;
				word_count++;
			}
		}
	//logmsg(nn,0,"superfc %x input %d words %d zero_words",self->node_id,word_count,zero_word_count);
}

//static int __attribute__((unused)) superfc_execute_ref(struct nn_node *self, struct nn_graph *nn) 
static int superfc_execute_ref(struct nn_node *self, struct nn_graph *nn) 
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *bias_tensor = self->inputs[6];
	const struct tensor *bias_min_tensor = self->inputs[7];
	const struct tensor *bias_max_tensor = self->inputs[8];
	const struct tensor *out_min_tensor = self->inputs[9];
	const struct tensor *out_max_tensor = self->inputs[10];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	int32_t in_batches = in_tensor->shape.batches * in_tensor->shape.height * in_tensor->shape.width;
	int32_t in_depth = in_tensor->shape.depth;

	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_depth = filt_tensor->shape.filt_height * filt_tensor->shape.filt_width * filt_tensor->shape.filt_depth;

	// in general given  (b,h,w,d) * ( fh,fw,fd,dout)
	// we do the matrix product ( bin, din) * ( din,dout)  -> (1,1,bin,dout)
	//  where din = fh*fw*fd
	//     and din must divide   b*h*w*d
	//      ( b*h*w*d = bin*din)
	// In these terms, currently filt_depth is din, in_depth is d, in_batches is b*h*w
	//
	if(filt_depth != in_depth){
		uint32_t in_size = in_batches * in_depth;
		uint32_t in_bat = in_size/(uint32_t)filt_depth;
		if( in_bat * filt_depth != in_size)
			return errlog(nn,"weight shape does not match input shape");
		in_depth = filt_depth;
		in_batches = in_bat;
	}
	int32_t out_batches = in_batches;
	int32_t out_depth = filt_batches;

	int32_t batch;
	int32_t filt_z;
	int32_t out_z;

	uint8_t *in = in_tensor->data;
	uint8_t *filt = filt_tensor->data;
	int32_t *bias = bias_tensor->data;
	uint8_t *out = out_tensor->data;

	uint8_t *instripe;
	uint8_t *filtstripe;
	int32_t *outstripe;

	int32_t in_element;
	int32_t filt_element;
	int32_t sum;
	int32_t minsum = 0;
	int32_t maxsum = 0;

	uint32_t out_elements = out_batches*out_depth;
	size_t out_size = out_elements;
	/* FIXME: if you pad depth you should adjust tmp_out_size here!!! */


	int32_t *biasbuf = nn->scratch;
	int32_t *tmp_out = biasbuf + out_depth;

	float in_max_float = tensor_get_float(max_in_tensor,0);
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	float bias_min_float = tensor_get_float(bias_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);

	/*
	 * output min/max is computed this way:
	 * Compute the size of each grade for each input: (max-min)/(2**bits)
	 * Multiply the grade sizes for the output grade size.
	 * output min/max == INT_MIN / INT_MAX * output grade size
	 */

	float in_level_size, filt_level_size;
	int32_t input_offset = get_qu8_level_size_zero(in_min_float,in_max_float, & in_level_size);
	int32_t filt_offset = get_qu8_level_size_zero(filt_min_float,filt_max_float, &filt_level_size);

	float bias_level_size = (bias_max_float - bias_min_float) *(float)(1.0/4294967296.0);
	float out_level_size = in_level_size * filt_level_size;

	float bias_mpy_amt = (bias_level_size / out_level_size);
	//int bias_adder = (bias_max_float / out_level_size);

	//float conv_out_max_val = ((float)(INT32_MAX)) * out_level_size;
	//float conv_out_min_val = 0.0f;

	/* input_offset is 0.0f quantized to in min/max */
	/* filt_offset is 0.0f quantized to filt min/max */

	int32_t bias_offset = 0;

	int i;

	logmsg(nn,2,"superfc execute. node=%p id=%x",self,self->node_id);
	logmsg(nn,2,"superfc input %dx%d",in_batches,in_depth);
	logmsg(nn,2,"superfc filt %dx%d",filt_batches,filt_depth);
	logmsg(nn,2,"superfc padding %d",self->padding);
	logmsg(nn,2,"expected out shape %dx%d",out_batches,out_depth);
	if (in_depth != filt_depth) return errlog(nn,"oops, depth != depth");

	if (out_min->max_size < sizeof(float)) return errlog(nn,"min too small");
	if (out_max->max_size < sizeof(float)) return errlog(nn,"max too small");

	// normally the shape is (1,1,out_batches, out_depth)
	// When indicated by NN_NODE_FLAG_KEEP_BATCH, out is (out_batches, 1,1, depth).
	int keep_batch =  0;	// need to find another way to do this. (self->flags & NN_NODE_FLAG_KEEP_BATCH);

	if( tensor_out_prepare_normal( out_tensor,
			(keep_batch?out_batches:1), 1, (keep_batch?1:out_batches), out_depth,
			 NN_TYPE_QUINT8) ){
		return errlog(nn,"output too small, %d < %d",out_tensor->max_size,out_size);
	}

	//printf("astep = %f; wstep= %f; bias_step= %g; bias_mpy = %g; offs = %d\n", in_level_size, filt_level_size, bias_level_size, bias_mpy_amt, (int)bias_offset);

	/* 
	 * This *could* be changed to fixed point and vectorized, but it shouldn't
	 * impact performance that much, just traversing depth once. 
	 */
	for (i = 0; i < out_depth; i++) {
		int32_t biasval = bias[i];
		biasbuf[i] = fast_roundf((biasval - bias_offset) * bias_mpy_amt );
	}

	/* BEGIN CONV2D. results in tmp_out buffer, also maxsum is updated */

	for (batch = 0; batch < out_batches; batch++) {
	    outstripe = tmp_out+out_depth*batch;
	    for (out_z = 0; out_z < out_depth; out_z++) {
	        sum = biasbuf[out_z];
		instripe = in+in_depth*batch;
	        filtstripe = filt+out_z;
	        for (filt_z = 0; filt_z < filt_depth; filt_z++) {
	              in_element = instripe[filt_z] - input_offset;
	              filt_element = filtstripe[filt_z*out_depth] - filt_offset;
	              sum += in_element*filt_element;
	//logmsg(nn,9,"[%d %d %d %d]: sum += %d*%d --> %d",
	//	batch,out_y,out_x,out_z,in_element,filt_element,sum);
	        }
	        if (sum < minsum) minsum = sum;
	        if (sum > maxsum) maxsum = sum;
	        outstripe[out_z] = sum;
	    }
	}

	// determine out range
	float out_min_float = tensor_get_float(out_min_tensor,0);
	float out_max_float = tensor_get_float(out_max_tensor,0);
	float final_out_min_val = out_min_float;
	float final_out_max_val = out_max_float;

	int constraint = 3;
	if( final_out_min_val == (float)-INFINITY){
		final_out_min_val =  minsum * out_level_size;
		constraint &=~1;
	}
	if( final_out_max_val == (float)INFINITY){
		final_out_max_val = fmaxf(final_out_min_val+1e-5f, maxsum * out_level_size);
		constraint &=~2;
	}
	adjust_minmax_for_zero_with_constraints( & final_out_min_val, & final_out_max_val, constraint );

	float outq_level_size = flt_div_255(final_out_max_val-final_out_min_val);
	// need to to mul 32-bit values by this
	float outscale = out_level_size/outq_level_size;
	int32_t outadd = fast_roundf(-final_out_min_val/out_level_size);

	int32_t scalef31 = fast_roundf( outscale * (float)(1u<<31) );

	/*
	int shamt = 16-__builtin_clz(maxsum);
	if (shamt < 0) shamt = 0;
	maxsum >>= shamt;
	fixed_recip_level_size = 0x00FF0000U/maxsum;	// chosen to align at bit 16
	*/
	/* Now go back through, add bias, clip to positive and requantize. */

	for (i = 0; i < out_batches*out_depth; i++) {
		int32_t val = tmp_out[i] + outadd;
		int32_t out_i = ((int64_t)val * scalef31 + (1u<<30)) >> 31;
		*out++ = saturate_u8(out_i);
	}

	//final_out_max_val = maxsum * out_level_size;
	//final_out_min_val = 0.0f;

	tensor_set_shape(out_min,1,1,1,1);
	tensor_set_float(out_min,0,final_out_min_val);
	out_min->data_size = sizeof(float);
	tensor_set_shape(out_max,1,1,1,1);
	tensor_set_float(out_max,0,final_out_max_val);
	out_max->data_size = sizeof(float);
	logmsg(nn,2,"superfc execute (ref) done! %dx%dx%dx%d",
		out_batches,1,1,out_depth);
	return 0;
}

static inline void __attribute__((always_inline))
l2fetch_linear( uint8_t const * addr, unsigned len){
	unsigned misalign = (size_t)addr & 127;
	int vec_count = (len + misalign + 127)>>7;		// exact # of vectors needed
	l2fetch( addr-misalign, 128,128, vec_count);
	//printf("prefetch: %p..%p  %p ... %p\n",
	//		addr, addr+len,  addr-misalign, addr-misalign + vec_count*128);
}

/*
 *   foreach 32 weigts of output depth
 *       prefetch next filters
 *   foreach batch -- moved outside
 *   l2fetch base line
 *       gvsumb (if needed?)
 *       gvconv (w/ gemsuma)
 * set output info
 * post sem
 */

#if 0
static inline void debug_pprint_vector(const struct superfc_info *ndata) {}

static inline void debug_value_range(
	struct nn_graph *nn,
	const uint8_t *out,
	int out_width,
	int n_lines,
	int out_next_row) {}
#endif

//in next row is same as in depth 
//next _d32 is just 32 as width = height = 1
static void superfc_execute_hvx_matmul(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	struct nn_node *self = work->self;
	struct superfc_info *info = self->opaque;

	int32_t start_batch = work->start_batch;
	int32_t stop_batch = work->stop_batch;
	int32_t in_depth = info->in_depth;
        int32_t num_batches = stop_batch - start_batch;
        
	int32_t recip_val = info->recip_val;
	int32_t recip_shamt = info->recip_shamt;
	int32_t weight_chunks = work->weight_chunks;
	int32_t weight_batch_size = info->weight_batch_size;
	int32_t out_depth_pad = info->out_depth_pad;
	int32_t out_depth = info->out_depth;

	const uint8_t *weights = work->weights;
	const int32_t *biasbuf = work->biases;
	const uint8_t *input = info->input_base + work->indata_offset;
 	uint8_t *output = work->output_t;

	uint64_t start_cycles;
	union {
		HVX_Vector vec[2];
		int32_t words[64];
	} minmax;

	int w, i, num_batches2, batch, b;

	uint8_t const * pf_addr = input;
	uint8_t const * pf_end = input + in_depth * num_batches;	// first byte we don't read
	unsigned pf_ahead = max_i32( in_depth *2, 1024 );			// size to read ahead
	unsigned pf_len = min_i32(pf_ahead, pf_end-pf_addr);		// first fetch size

	l2fetch_linear(pf_addr, pf_len);
	pf_addr += pf_len;


	num_batches2 = (num_batches+1)&~1;

	start_cycles = nn_os_get_cycles(nn);
#if 1
	logmsg(nn,2,"start/stop: %d/%d output %p input %p filt %p bias %p in_depth %d recip_shamt %d recip_val 0x%08x minmax_buf %p n_batches=%d suma_buf=%p weight_chunks=%d weight_batch_size=%d cycles=%lld",
			start_batch,
			stop_batch,
			output,
			input,
			weights,
			biasbuf,
			in_depth,
			recip_shamt,
			recip_val,
			minmax.words,
			stop_batch - start_batch ,
			work->suma_buf,
			weight_chunks,
			weight_batch_size,
			start_cycles);
#endif
	minmax.vec[1] = Q6_V_vsplat_R(0x7FFFFFFF);
	minmax.vec[0] = Q6_V_vnot_V(minmax.vec[1]);

	if(start_batch >= stop_batch) goto done;
    /*-------------------------------------------------------------*/
    /*              V60 Implementations                            */
    /*-------------------------------------------------------------*/

        logmsg(nn,1,"batch = %d -> %d weight chunks = %d",start_batch, stop_batch, weight_chunks);

		const uint8_t ** work_ptr_in_batches = work->ptr_in_batches;
		uint8_t ** work_ptr_out_batches = work->ptr_out_batches;
		int32_t * work_suma_buf = work->suma_buf;
		// build the tables of pointers in scratch; and do suma if not already done.
		{
			//global suma goes here (shared across threads, so don't reorder stores)
        	volatile int32_t *suma = info->suma_buf + start_batch;
			volatile uint8_t * need_init_p = info->need_initialize_suma;

			for(batch = 0; batch < num_batches2; batch+=2) {
				int pf_advance = in_depth*2;	// normally advance 2 batches
				if(pf_addr >= pf_end){			// nothing left to read
					pf_addr = input;			// .. so start at the beginning for next w loop
					pf_advance = pf_ahead;
				}
				pf_len = min_i32(pf_advance, pf_end-pf_addr);
				// @@ we should probably wait here for previous prefetch to complete...
				l2fetch_linear( pf_addr,  pf_len);
				pf_addr += pf_len;

				// compute SUMA and fil local array
				for(i=0; i < 2; i++) {
					b = batch+i;
					union  {
						int sum;
						HVX_Vector vtemp;
					}usum = {0};

					if(b < num_batches) {
						// need_initialize_suma is shared across threads
						// Note - there's a possibility here that two threads will do the same suma
						// calculation, and then both store the same result to global arr suma
						// and then both clearing the 'need_init'. Not really a problem; as long as we only
						// do one store to suma[b] (an 'intermediate' result store could break it), and clear
						// need_init_p[] *after* that.
						//logmsg(nn,2,"wc = %d init suma %d",w,info->need_initialize_suma[work->start_batch+b]);
						if (need_init_p[start_batch+b]) {
							if(info->filt_offset != 0){
								   fcsuma_asm( input + b*in_depth,
										 in_depth,
										 &usum.vtemp);
								  //for(j=0; j < in_depth; j++) logmsg(nn,0,"%d]=%02x",j,input[(batch+i)*in_depth+j]);
							}
							suma[b] = info->filt_offset * usum.sum;                       //populate global suma array
							logmsg(nn,4,"suma[%d] = %d",start_batch+b,suma[b]);
							need_init_p[start_batch+b] = 0;
						}
						work_ptr_in_batches[b] = input + b*in_depth;
						work_suma_buf[b] =suma[b];                                 //local array
						work_ptr_out_batches[b] = output + b*out_depth_pad;
					} else {
						// for last unpaired batch when num_batches is odd:
						// need valid input pointer (re-use previous) and NULL output pointer.
						// Use the matching suma to avoid bad min/max.
						work_ptr_in_batches[b] = work_ptr_in_batches[b-1];
						work_suma_buf[b] = work_suma_buf[b-1];
						work_ptr_out_batches[b] = NULL;
					}
				}//i
			}//end for batches
        }

             //for (b = 0; b < num_batches2; b++) logmsg(nn,2,"in/out ptrs for %d: in: %p out: %p",b,work->ptr_in_batches[b],work->ptr_out_batches[b]);
        int even_batches = num_batches & (~1);
        int odd_batch = num_batches & (1);
        if(odd_batch) logmsg(nn,3,"running odd batch code %d + %d", even_batches, odd_batch);
	for (w = 0; w < weight_chunks; w++) {

	     // matrix multiply 2 batches per output depth 32 slice
#if 0
             logmsg(nn,2,"fclb_asm(%p,%p,%p,%d,%d,%p,0x%x,%p,%p,%d)",
                                  work->ptr_in_batches,
                                  weights + w*32*info->in_depth_pad,
                                  work->ptr_out_batches,
                                  in_depth,
                                  num_batches2,
                                  minmax.words,
                                  recip_val, //reciprocal of max for quantization
                                  biasbuf + 32*w,
                                  suma,
                                  32*w
             );
#endif
             if(even_batches) fullconnlayerbatch_asm(
                                  work_ptr_in_batches,
                                  weights + w*32*info->in_depth_pad,
                                  work_ptr_out_batches,
                                  in_depth,
                                  even_batches,
                                  minmax.words,
                                  recip_val, //reciprocal of max for quantization
                                  biasbuf + 32*w,
                                  work_suma_buf,
                                  32*w,
                                  recip_shamt
             );
             if(odd_batch) fullconnlayerbatch1_asm(
                                  work_ptr_in_batches[even_batches],
                                  weights + w*32*info->in_depth_pad,
                                  work_ptr_out_batches[even_batches],
                                  in_depth,
                                  odd_batch,
                                  minmax.words,
                                  recip_val, //reciprocal of max for quantization
                                  biasbuf + 32*w,
                                  work_suma_buf+even_batches,
                                  32*w,
                                  recip_shamt
             );
        } // end for weights
        //only if the output is not multiple of 32: copy from temp to output
        uint8_t *outp2 = work->output;
        if( outp2 != work->output_t)
        {
               int length = work->actual_odepth;
               logmsg(nn,3,"copy after, start pos = %ld length = %d", work->start_chunk*32,length);
               vmemcpy_2d_general_asm(  length, num_batches,        // width, height
                      outp2,  out_depth,                            // output ptr, out stride
                      work->output_t, out_depth_pad );              // input pointer, input stride
        }
	gvrmaxmin(minmax.words);
	nn_atomic_min(&info->minval,minmax.words[32]);
	nn_atomic_max(&info->maxval,minmax.words[ 0]);
	logmsg(nn,2,"min: %d max: %d",info->minval,info->maxval);

	start_cycles = nn_os_get_cycles(nn) - start_cycles;
	nn_atomic_add64(&info->cycles,start_cycles);
	//asm volatile ("":::"memory");
	//logmsg(nn,2,"min=%d(%d) max=%d(%d) cycles=%lld",minmax.words[32],info->minval,minmax.words[0],info->maxval,start_cycles);
	//debug_value_range(nn,work->output+start_batch*out_depth,1,stop_batch-start_batch,out_depth);
	//logmsg(nn,0,"posting to %p",work->donesem);
done:
	nn_sem_post(work->donesem);
}

static void superfc_execute_hvx_work(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	superfc_execute_hvx_matmul(nn, work);
        return;
}

static inline int superfc_reset_work_items(
	struct nn_node *self,
	struct nn_graph *nn,
	struct superfc_info *info)
{
	if (info->work_items) nn_free(info->work_items);
	info->work_items = NULL;
	info->n_work_items = 0;
	info->workitems_alloc = 0;
	return 0;
}
// (reset record count; keep the buffer)
static inline int superfc_softreset_work_items(
	struct nn_node *self,
	struct nn_graph *nn,
	struct superfc_info *info)
{
	info->n_work_items = 0;
	return 0;
}

static inline int superfc_add_work_item(
	struct nn_node *self, 
	struct nn_graph *nn, 
	struct superfc_info *info,
	const struct workitem work /* BY VAL */)
{
	struct workitem *witems = info->work_items;
	int new_work_items = info->n_work_items+1;
	int new_work_size = new_work_items*sizeof(work);

	if( new_work_size > info->workitems_alloc)
	{
		// reallocate (or first alloc when witems=NULL)
		// round up to multiple of 512; at least 1536
		new_work_size = (new_work_size <= 1536)? 1536 : ( (new_work_size+511) & ~511);
		struct workitem *new_data;
		if ((new_data=nn_realloc(witems,new_work_size)) == NULL) {
			return errlog(nn,"realloc fails");
		}
		info->workitems_alloc = new_work_size;
		info->work_items = witems = new_data;
	}

	witems[new_work_items-1] = work;
	info->n_work_items = new_work_items;
	return 0;
}

int superfc_execute_workitem_prefetch(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	logmsg(nn,1,"prefetching @ %p: %d(%d)x%d",work->pf_inp,work->pf_width,work->pf_stride,work->pf_height);
	if(work->pf_inp!=NULL) l2fetch(work->pf_inp,work->pf_stride,work->pf_width,work->pf_height);
	return 0;
}

// This workitem does the following:
//     (1) wait for 'conv_done', 'threads' times  (but only if conv_done is not NULL)
//     (2) Start a vector thread which:
//          (a) runs the copy, then
//          (b) does post(do_conv) 'work.threads' times
//
//   (step (2) is skipped if work->copy_out is null)
//
// This is used between groups of executions where weights need switching. The subsequent
// matmul workitems all take( do_conv) before reading the weights.
// The 'wait for conv_done' is used when this is not the first copy, in order to enure
//  the previous matmuls are finished.

static inline __attribute__((unused)) void superfc_add_l2fetch(
	struct nn_node *self,
	struct nn_graph *nn,
	struct superfc_info *info,
	const uint8_t *input,
	int width,
	int height)
{
	struct workitem work;
	work.pf_inp = input,
	work.pf_width = width;
	work.pf_stride = width;
	work.pf_height = height;
	work.execute = superfc_execute_workitem_prefetch;
	superfc_add_work_item(self,nn,info,work);
}

static inline void superfc_weights(
        struct nn_node *self,
        struct nn_graph *nn,
        struct superfc_info *info,
        const uint8_t *weights,
        uint32_t weight_chunk_size,
        uint32_t weight_chunks)
{
        return superfc_add_l2fetch(self,nn,info,weights,weight_chunk_size/32,weight_chunks*32);
}

//
// This is like to_next_power_of_two(x), but it finds the first power of 2^(1/4) which
// is > x, and then returns the one after that. So y/x > 1.189 but <= 1.414
// The binary mantissa will always be one of four specific values.
//     1.0000 1.1892 1.4142 1.6718
//
static inline float
__attribute__((unused))
round_up_quarter_octave( float x)
{
	float xk = x * 1.18920708f;
	float mant = fabsf( flt_getfrac( xk) );		 // 0.5 .. 0.999999
	// slice against  r/2, r^2/2, r^3/2  and round up to the one at the end of the interval.
	if( mant >= 0.7071067691f){		/* r^2/2 */
		mant = (mant >= 0.8408964276f) ? 1.0f: 0.8408964276f;		// r^3/2,  r^4/2, r^3/2
	}else{
		mant = (mant >= 0.594603539f) ? 0.7071067691f: 0.594603539f;			// r/2, r^2/2, r/2
	}
	float res = flt_ldexp( mant, flt_getexp(xk));
	return copysignf( res, x);
}

int superfc_execute_workitem_check_for_retry(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	struct superfc_info *info = node->opaque;
	float newval;
	float extreme_out;
	int recalc = 0;
	if (info->minval_precalculated && info->maxval_precalculated) return 0;
	if (unlikely(!info->maxval_precalculated && (info->maxval > 255))) {
		/* Mark as needing retry and set new max value */
		info->needs_retry = 1;
		extreme_out = info->output_level_size*(info->maxval + 0.5f)+ info->out_minval;
		newval = round_up_quarter_octave( fmaxf(extreme_out, 0x1.0p-4f));
		logmsg(nn,1,"max too small, recalculating %d > %d / %f > %f... picking %f",
			info->maxval,info->max_valid_val,extreme_out,info->out_maxval,newval);
		info->out_maxval = newval;
		recalc = 1;
	}
	if (unlikely(!info->minval_precalculated && (info->minval < 0))) {
		/* Mark as needing retry and set new min value */
		info->needs_retry = 1;
		extreme_out = info->output_level_size*(info->minval - 0.5f)+ info->out_minval;
		newval = round_up_quarter_octave( fminf(extreme_out, -0x1.0p-8f));
		logmsg(nn,1,"min too large, recalculating %d < %d / %f < %f... picking %f",
			info->minval,info->min_valid_val,extreme_out,info->out_minval,newval);
		info->out_minval = newval;
		recalc = 1;
	}
	// if endpoints moved, adjust to get a valid zero.
	// TODO: this should also be done if one of the endpoints is 'fixed',
	// (using adjust_minmax_for_zero_with_constraint); but that will, in some cases, want
	// to move the 'fixed' endpoint by a small amount, and so there should also be a mechanism
	// in place to ensure that the 'fixed' endpoint is always moved to the preset value before the
	// adjustment (so it can't "drift" after repeated corrections).
	// In cases where the 'fixed' endpoint is zero, this is moot; a range with a zero endpoint
	// never needs adjustment.
	//
	if( recalc && !info->maxval_precalculated && !info->minval_precalculated ){
		adjust_minmax_for_zero( &info->out_minval, &info->out_maxval);
		logmsg(nn,2,"corrected range: %f ... %f", info->out_minval, info->out_maxval);
	}


	//logmsg(nn,1,"Checking workitem, maxval=%x minval=%x max_valid_val=%x needs_retry=%d",info->maxval,info->minval,info->max_valid_val,info->needs_retry);
	return 0;
}

int superfc_execute_workitem_vector_dispatch(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	//logmsg(nn,0,"hvx launch work semaphore %p",work->donesem);
	nn_os_work_for_vector(nn,superfc_execute_hvx_work,work);
	return 0;
}

int superfc_execute_workitem_vector_join(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	nn_sem_wait_n_times( work->donesem, NUM_THREADS);
	return 0;
}

int superfc_execute_workitem_join_some(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	nn_sem_wait_n_times( work->donesem, work->join_iters);
	return 0;
}

//theres no zapping in this, a function in execute everything pads out the input batch data

//
// This is the 'class' for sharing weights with other instance of the same node.
// The NN_CPSHARE_HEADER includes ptr_w (points to prepared weights) and ptr_sumb (to gemsumb).
//
// This node requires the sumb to be done as  ( filt_offset* in_depth_total - sum(filt)), so
// can only share with other nodes that see the same filter offset; thus we store that in the type.
//
struct superfc_cpshare_type {
     NN_CPSHARE_HEADER		// must have this
     int filter_offset;
};

/*
 * We are computing (a-a_offset)*(b-b_offset)
 * If a_offset != 0, we need to compute a_offset * sum(b-b_offset)
 * We can sum the elements of the filter (offsetted) and multiply by a_offset
 * This we can do ahead of time.
 */

/*
 *
 * This rearrangement is equivalent to:
 *     initial tensor:  [ depth, batches ]
 *    pad dims up to    [ d_total, b_total ]	(multiple of 4 and 32 )
 *    reshape       [d_hi, d_lo, b_hi, b_lo]    (where d_lo = 4, d_hi = 32)
 *    transpose     [b_hi, d_hi, b_lo, d_lo]
 *    ... so the last 2 indices select a location within a vector.
 *    At the padding stage, edges are filled with 'filt_offset'.
 *  The 'gsumb' values are gsumb[b] =  sum{d} ( filt_offset-in[d,b])
 *  The d_total is in fact a multiple of 32, so d_hi is a multiples of 8;
 *  this makes it equivalent to the convolution packing  situation for filt_h=filt_w =1.
 *  so we can use repack_filter_for_d32 - but we need to tweak the sumb after.
 *
 */
void superfc_rearrange_and_sum_for_d32(
   struct nn_graph *nn,
  const uint8_t* in_data,
  int filt_depth,   //in_depth
  int filt_batches, //out_depth
  uint8_t *out_data,
  int filt_depth_total,   //in_depth_pad
  int filt_batches_total, //out_dept_pad
  int filt_offset,
  int32_t * gsumb) {

	// set up  a call to repack_filter_for_d32 - first need a 'tensor'
	struct tensor faketens = {
			.data = (void*)in_data,
			.data_size = filt_depth*filt_batches,
			.max_size = filt_depth*filt_batches,
			.shape = { .filt_height = 1, .filt_width = 1, .filt_depth = filt_depth, .filt_batches= filt_batches},
	};
	struct repack_filter_parms rpfparms = {
			.out_data = out_data,
			.filt_tensor = &faketens,
			.zero_offset= filt_offset,
			.signed_mode_sel = 0,
			.gemsumb = gsumb
	};
	nn_sem_init( & rpfparms.done_sem, 0);
	// call it directly.
	repack_filter_for_d32( nn, &rpfparms);

	// repack_filter_for_d32 found the filter sums across depth, including padding; we need to
	// subtract each from filt_offset*filt_depth_total.
	int32_t adj = filt_offset * filt_depth_total;
	HVX_Vector vadj = Q6_V_vsplat_R( adj);
	HVX_Vector *gsumv = (HVX_Vector*)gsumb;
	for( int i = 0; i < filt_batches_total/32u; i++ ){
		gsumv[i] = Q6_Vw_vsub_VwVw( vadj, gsumv[i]);
	}
	// flush last vector store...
#if defined(__hexagon__)
	asm volatile( "/*  */");
#endif
	((volatile int32_t *)gsumb)[filt_batches_total-1];

#if 0
    int w,x,y,z,sumw;

    for (w = 0; w < filt_batches; w++) {
      z = w & 31;
      x = w & ~31;
      sumw = filt_depth*(int)filt_offset;
      for (y = 0; y < filt_depth; y+=1) {
           out_data[32* (y&~3)+filt_depth_total*x+4*z+(y&3)] = in_data[filt_batches*y+x+z];
           sumw -= in_data[filt_batches*y+w];
      }
      gsumb[w] = sumw;
      for (y = filt_depth; y < filt_depth_total; y+=1) {
           out_data[32* (y&~3)+filt_depth_total*x+4*z+(y&3)] = filt_offset;
      }
    }
    for (w = filt_batches; w < filt_batches_total; w++) {
      gsumb[w] = 0;
      z = w & 31;
      x = w & ~31;
      for (y = 0; y < filt_depth_total; y+=1) {
           out_data[32* (y&~3)+filt_depth_total*x+4*z+(y&3)] = filt_offset;
      }
    }
#endif
  return;
}

/* 
 * EJP: FIXME:
 * When we have large activations and small weights, it's better to chop up the activation
 * into pieces and bring through the entire set of weights rather than run through the 
 * activation multiple times with a fixed set of weights.
 * 
 * Find minimum reasonable weights L2$ footprint.  2x weight slices for cache-only system
 * 0 for memcpy into vtcm
 * 1x weight slices for vmemcpy to vtcm?
 * 
 * Find activation total size
 * If activation size + weights L2$ footprint < L2$ size, we're good
 * Find total number+size of weight slices
 * While doesn't fit (based on L2$ footprint of weights & activation slice size):
 *   Increase slicing factor that minimizes traffic
 *   If we try and slice up the activation to fit better, we bring in 2x the weights
 *   If we try and slice up the weights to fit better, we bring in 2x the activation
 *   We have to bring in the whole activation + whole weights once
 *   Activation might be hot in the cache, weights probably not
 *   Early in the graph, activations are big and weights are comparatively small
 *   Later in ght graph, activations are small and weights are comparatively big
 * 
 *   Since we have to bring in all the weights at least once, if the whole activation
 *   fits + 2 slices of weights that's probably the best option.  If not, we want to
 *   find some combination of slices of weights and slices of activations that fits nicely.
 */


//Fill in scaling info for superfc. TODO: Add per channel scaling support
static int fill_info_minmax_basics(
	struct nn_graph *nn,
	struct nn_node *self,
	struct superfc_info *info)
{
	/* Pull out the inputs we need */
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];

	/* Get min/max values for input, weights, and bias data */
	float in_min_float = tensor_get_float(min_in_tensor,0);
	//float in_max_float = fmaxf(tensor_get_float(max_in_tensor,0),in_min_float+0.00001f);
	float in_max_float =tensor_get_float(max_in_tensor,0);

    if( in_min_float > 0.0f || in_max_float < 0.0f || in_min_float >= in_max_float )
        return errlog(nn, "SuperFC: invalid input min/max");

	/* find zero offset,level for each input */
	float in_level_size;
	int32_t input_offset = get_qu8_level_size_zero(in_min_float,in_max_float, &in_level_size);

	/* filter level (already compensated for any scaling done) */
	float filt_level_size = info->weights_level_size;

	/* The product level size is the product of the input and filter level size */
	float prod_level_size = in_level_size * filt_level_size;

	/* final scaling is to multiply by prod_level_size / output_level_size */
	float output_level_size;
    get_qu8_level_size_zero(info->out_minval,info->out_maxval, &output_level_size);
    info->output_level_size = output_level_size;

    float final_scaling = prod_level_size / output_level_size;
    int recip_shamt = (final_scaling <= 1.0f)? 0: flt_getexp(final_scaling);

	// find final_scaling with 31-recip_shamt frac bits now.
	// Will be <= 0x7FFFFF80, except in border case where final_scaling = 1.0
	//This rounding will be lossless unless final_scaling < (1/128).
	unsigned recip_val = roundf_u32( flt_ldexp( final_scaling, (31-recip_shamt)));
	float final_scaling_inv = output_level_size/(prod_level_size);
        recip_val = (recip_val < 0x7fffffffu)? recip_val :0x7FFFFFFFu;
	info->prod_level_size = prod_level_size;

    // find range of pre-scaled values which don't constitute overflow; allows for rounding to 0
    // or to 255.
	info->min_valid_val = -0.49f*final_scaling_inv;
	info->max_valid_val = 255.49f*final_scaling_inv;

	info->in_max_float = in_max_float;
	info->in_min_float = in_min_float;

	info->in_offset = input_offset;

	info->recip_val = recip_val;
	info->recip_shamt = recip_shamt;

	return 0;
}

static int fill_bias_buf(
	struct nn_graph *nn,
	struct nn_node *self,
	struct superfc_info *info,
	int bias32,
	int32_t extra)
{
	const struct tensor *bias_tensor = self->inputs[6];
	const struct tensor *bias_min_tensor = self->inputs[7];
	const struct tensor *bias_max_tensor = self->inputs[8];
	float bias_min_float = tensor_get_float(bias_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);
	int32_t bias_offset = bias32 ? 0 : quantize_uint(0.0f,bias_min_float,bias_max_float);
	float bias_denom = bias32 ? 0x1.0p32 : 255.0f;
	float bias_level_size = (bias_max_float - bias_min_float) / bias_denom;
	const uint8_t *bias8_ptr = bias_tensor->data;
	const int32_t *bias32_ptr = bias_tensor->data;
	float bias_to_prod_ratio = (bias_level_size / info->prod_level_size);
	float min_out_prod_offset = -info->out_minval / info->prod_level_size;
	int32_t bias_depth = bias_tensor->shape.depth;
	int i;
	int32_t biasval;
	float bias_fval;
	float minout_bias_fval;
	int32_t gemsumb_val;
	int32_t final;
	logmsg(nn,2,"bias_depth=%ld out_depth_pad=%ld out_minval=%f in_offset=%d bias_offset=%d bias_levelsize=%2.12f prod_level_size=%2.12f ratio=%2.12f",bias_depth,info->out_depth_pad,info->out_minval,info->in_offset,bias_offset,bias_level_size,info->prod_level_size,bias_to_prod_ratio);
	for (i = 0; i < info->out_depth_pad; i++) {
		if (i >= bias_depth) biasval = bias_offset;
		else if (bias32) biasval = bias32_ptr[i];
		else biasval = bias8_ptr[i];
		bias_fval = (biasval - bias_offset) * bias_to_prod_ratio;
		minout_bias_fval = bias_fval + min_out_prod_offset;
		gemsumb_val = info->gemsumb[i];
		final = gemsumb_val * info->in_offset + fast_roundf(minout_bias_fval) + extra;
		logmsg(nn,2,"i=%d biasval%d=%d fval=%f minout_fval=%f(%d) gemsumb_val=%d extra=%d final=%d",
			i,bias32?32:8,biasval,bias_fval,minout_bias_fval,fast_roundf(minout_bias_fval),gemsumb_val,extra,final);
		info->biasbuf[i] = final;
	}
	return 0;
}


/*
 * TODO: have multiple strategies, and go through them to find one that matches.
 * Common code will be refactored into helper functions
 * Scenarios:
 * * Tiny enough to just do reference / scalar version
 * * Input in D32, large output depth (currently implemented)
 * * Input in normal format, want D32, large output depth
 * * Input in normal format, want D32, short output depth
 * * Input in D32, short output depth
 * * Short input depth, large output depth
 */

/*
 * What do we have to do before calling the work function?
 *
 * Some things are cacheable:
 * * Biasbuf number range conversion
 * * Adding min_out value (converted) into biasbuf
 * * Adding gemsumb values (if needed) into biasbuf
 * * Adding in_offset * weight_offset * N into biasbuf (if needed)
 * * Partitioning scheme
 * * Reciprocal for range conversion
 * * Min and Max output information
 * * Strategy for partitioning
 * * Work function
 * * output shape/padding
 *
 * * Ensure that cached value is still accurate given these inputs (if not, recalc)
 * * Any preproc (im2col) or padding adjustments
 * * Add work_for_vector items
 * * Wait for workers to complete
 * * Any fixup / postproc 
 * 
 * * The work function is passed to work_for_vector.  It can be different for different architectures.
 * * The strategy for partitioning could be a set of functions.
 */
int superfc_recalculate_strategy(struct nn_node *self, struct nn_graph *nn)
{
	/* Pull out the inputs we need */
	const struct tensor *in_tensor = self->inputs[0];
	//const struct tensor *filt_tensor = self->inputs[1];
	//const struct tensor *min_in_tensor = self->inputs[2];
	//const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	//
	//const struct tensor *bias_tensor = self->inputs[6];
	//const struct tensor *bias_min_tensor = self->inputs[7];
	//const struct tensor *bias_max_tensor = self->inputs[8];
	/* Find the output tensors */
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	/* Structures with auxillary information */
	struct superfc_info *info = self->opaque;
	/* 
	 * Find the dimensions of the input data, 
	 * both dimensions of data as well as padding
	 */
	info->inshape = in_tensor->shape;

	int32_t in_elements = tensor_element_count( in_tensor);
	int32_t in_depth = info->in_depth;
	int32_t in_batches = (uint32_t)in_elements/in_depth;

	if( in_depth * in_batches != in_elements)
		return errlog(nn,"input size not divided by weights depth");
	//printf("(%d x %d) * ( %d * %d) -> ( %d * %d)\n",
	//	(int)in_batches, (int)in_depth, (int)in_depth, (int)info->out_depth, (int)in_batches, (int)info->out_depth);

	int32_t in_depth_total = in_depth;
	in_depth_total = (in_depth_total + 31) & (~31);

	info->batches= in_batches;

	/*  output dimensions */
	int32_t out_batches = in_batches;
	int32_t out_depth = info->out_depth;

	/* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */

	// total number of inputs on each row actually needed (including the required_w before and after).
	// this is normally the same as required_w_before + in_width + required_w_after
	//  but it may be a bit less, when required_w_after = 0 and stride_width >1
	//
	/* Set up pointers to bulk data */
	//uint8_t *in = in_tensor->data;
	//uint8_t *filt = filt_tensor->data;
	//uint8_t *bias = bias_tensor->data;
	//uint8_t *out = out_tensor->data;
	//int32_t *bias32_ptr = bias_tensor->data;

	/* 
	 * Set output padding values to sensible defaults.
	 * this should output an exact number, will require a width and a mask
         * depth mask = out_depth & 31
	 */
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
        info->out_depth = out_depth;
        info->out_depth_pad = out_depth_total;


	/* How much storage for each frame in the batch? */
	int32_t input_batch_size = in_depth;
	int32_t output_batch_size = out_depth;

        /* used as a tempoary buffer to deal with non % 32 output depts */
        if(out_depth != out_depth_total) {
           info->out_temp_buf = (uint8_t *) nn_scratch_alloc(nn, out_batches * out_depth_total * sizeof(uint8_t));
           if( info->out_temp_buf == NULL) return errlog( nn, "scratch alloc failed");
        } else {
           info->out_temp_buf = NULL;
        }

	int out_depth_iters = out_depth_total/32;

	/* Grab some precomputed weight size information */
	//int n_weight_batches = info->n_weight_batches;
	int weight_batch_size = info->weight_batch_size;

	//int i;
	//int d,d2,b,t;

	struct workitem work = {0}; // klocwork
	struct workitem waitwork;
	//int workidx = 0;
	//int32_t tmpval32;

	logmsg(nn,1,"SuperFC %x: Recalculating Strategy...",self->node_id);
	//logmsg(nn,0,"Weight batch size: %d. Per 256KB: %d",weight_batch_size,256*1024/weight_batch_size);

	/* Some sanity checks... */
	if (out_depth_iters <= 0) return errlog(nn,"no out depth to iterate?");
	if ((out_depth_total % 32) != 0) return errlog(nn,"depth math fail");
	if (in_depth < 32) return errlog(nn,"in_depth must be at least 32");
	if ((in_depth & 15) != 0) return errlog(nn,"in_depth not a multiple of 16, not supported");
	info->in_depth_pad = in_depth_total;
	info->in_depth = in_depth;

        logmsg(nn,2,"in_depth = %ld in_depth_pad = %ld",in_depth,in_depth_total);


	//logmsg(nn,0,"maxsum=0x%x recip_val=0x%x shamt=%d",maxsum,recip_val,recip_shamt);
	//if (recip_val & 0x80000000U) logmsg(nn,0,"***** reciprocal is negative if signed, ??problem??");


	/* Compute reciprocal and shift amount and associated scaling info */
	if(fill_info_minmax_basics(nn,self,info)) return -1;

	logmsg(nn,2,"out_maxval=%f in_level_size=%f filt_level_size=%f prod_level_size=%2.12f max_valid ~= %d",
		info->out_maxval,
		info->prod_level_size/info->weights_level_size,
		info->weights_level_size,
		info->prod_level_size,
		info->max_valid_val);

	/*
	 * Update info / superfc info values
	 * These values are stored in structures for use during normal execution
	 */

	info->need_initialize_suma = (uint8_t *) nn_scratch_alloc(nn, sizeof(uint8_t)*(in_batches+1));
	if( info->need_initialize_suma==NULL) return errlog(nn,"scratch alloc failed");
	memset( info->need_initialize_suma,1,sizeof(uint8_t)*(in_batches+1) );
	// input_base0, in_height:  this omits any top/bottom padding rows that we don't need

	//info->input_base = in;
	info->weights_base = info->weights;


	//fill_info_dim_basics(nn,self,info);

	/*
	 * Recompute bias values
	 * We need to incorporate the input bias values, min offset, and gemsumb values into the bias buffer
	 * The bias buffer is added in the "product" (in_stepsize * filt_stepsize) number system
	 */

	int32_t bias_extra = 0; //in_depth_total*input_offset*filt_offset;
	logmsg(nn,2,"in_depth_total=%d input_offset=%d filt_offset=%d bias_extra=%d",in_depth_total,info->in_offset,info->weights_offset,bias_extra);
	fill_bias_buf(nn,self,info,1,bias_extra);

	/*
	 * Prepare output tensors
	 */
	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail");
	}
	tensor_set_float(out_min,0,info->out_minval);
	tensor_set_float(out_max,0,info->out_maxval);

	// When indicated by NN_NODE_FLAG_KEEP_BATCH, out is (out_batches, 1,1, depth).
	int keep_batch = 0; // need to find another way to do this. (self->flags & NN_NODE_FLAG_KEEP_BATCH);

	if (tensor_out_prepare_normal(out_tensor,
			(keep_batch? out_batches:1), 1, (keep_batch?1: out_batches), out_depth, NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail");
	}

	/* Where does our output data start? */
	uint8_t *out_data_start = out_tensor->data;

	/*
	 * Preparing the work list
	 * We create a work list, which is just a list of items for the main thread to do
	 * We need to put in all the things we need to do to execute the node:
	 * * Padding zapping
	 * * GEMSUMA computation
	 * * Actual matmul
	 * * l2fetch / memcpy of values
	 * * Waiting on semaphores for things to finish
	 * The work list is currently executed by the main thread
	 * This means that vector work needs to be passed to nn_os_work_for_vector
	 */

	superfc_softreset_work_items(self,nn,info);

	/*
	 * Set up work items
	 */

	/* FIXME for v66: copy work has to actually work. */
	waitwork.execute = superfc_execute_workitem_join_some;

	work.info = info;
	work.self = self;
	//work.indata_offset = 0;
	//work.output  = out;
	//work.weights = filtdst;
	work.execute = superfc_execute_workitem_vector_dispatch;	// FIXME, pick the right function

	/* 
            split n batches across M threads, 2 per thread
            do this for each set of inner weight batches 
	 */

	/* Determine how we want to slice the work */
	int32_t inner_act_batches;
	int32_t inner_weight_chunks;

	inner_act_batches = (out_batches+NUM_THREADS-1)/(NUM_THREADS);
	/*
	 *  NOTE: calculation of outer_weight_chunks is duplicated in 'superfc_check' function
	 *  to determine scratch usage.
	 */
	/* Ignore batch / weight slice factor for now ... ignore that used by activations*/
        inner_weight_chunks = (384*1024 + weight_batch_size - 1)/ weight_batch_size;
         if(inner_weight_chunks > out_depth_iters) inner_weight_chunks = out_depth_iters;
	int32_t outer_weight_chunks = (out_depth_total/32 + inner_weight_chunks - 1) / inner_weight_chunks;

	logmsg(nn,1,"batch/weight chks: inner=%d,%d,%d out_depth_chunks=%d",
		inner_act_batches,inner_weight_chunks, outer_weight_chunks,
		out_depth_total/32);

	superfc_weights(self,nn,info,info->weights,weight_batch_size,inner_weight_chunks);
    /*-------------------------------------------------------------*/
    /*  Setup parameters and allocate scratch for SUMA computation */
    /*-------------------------------------------------------------*/
    // sumabuf is allocated in scratch, for all batches

	info->suma_buf = (int32_t *) nn_scratch_alloc(nn,  (in_batches+1) * sizeof(int32_t));
	if( info->suma_buf == NULL ) return errlog(nn,"scratch alloc failed");

	int32_t semaphore_count = 0;
	int ow,b;

	for (ow = 0; ow < outer_weight_chunks; ow++) {
	/* l2fetch first weight chunk */
	    logmsg(nn,1,"adding l2fetch: %p %d %d",info->weights,weight_batch_size,inner_weight_chunks);
	    int start_weights = ow * inner_weight_chunks;
            work.start_chunk = start_weights;
            int actual_odepth = min_i32( out_depth- start_weights*32,  32*inner_weight_chunks) ;
            work.actual_odepth = actual_odepth;
	    int needs_next_outer_weights = (ow != (outer_weight_chunks-1));

	    int now_chunks = (actual_odepth+31)/32u;
	    work.weights = info->weights_base + start_weights*weight_batch_size;
            logmsg(nn,3,"now_chunks=%d",now_chunks);
	    work.weight_chunks = now_chunks;
	    work.biases = info->biasbuf + start_weights*32;
	    if (needs_next_outer_weights) {
		int32_t next_weight_chunks = inner_weight_chunks;
		int32_t max_next_weight_chunks = out_depth_total/32-start_weights-now_chunks;
		next_weight_chunks = Q6_R_min_RR(max_next_weight_chunks,next_weight_chunks);
		work.pf_inp = work.weights + now_chunks*weight_batch_size;
		work.pf_width = weight_batch_size;
		work.pf_stride = work.pf_width;
		work.pf_height = next_weight_chunks;

		logmsg(nn,1,"ow=%d set up weight pf ptr=%p width=%d height=%d",
				ow,work.pf_inp,work.pf_width,work.pf_height);
	    } else {
		work.pf_inp = NULL;
		work.pf_width = 0;
		work.pf_stride = 0;
		work.pf_height = 0;
	    }
            superfc_add_l2fetch( self, nn, info,work.pf_inp,work.pf_width, work.pf_height);

	    for (b = 0; b < out_batches;  b+=inner_act_batches) {

		//int pf_outer_act = (or == 0);

		int start_batch =  b;
		int n_batches = Q6_R_min_RR(out_batches-start_batch,inner_act_batches);
                logmsg(nn,2,"batch = %d n_batches = %d",b, n_batches);

		/* FILL OUT NORMAL WORK INFORMATION */
		work.indata_offset= b*input_batch_size;
		work.output = work.output_t = out_data_start + b*output_batch_size + start_weights*32;
		if( info->out_temp_buf != NULL){
			work.output_t = info->out_temp_buf + b*out_depth_total + start_weights*32;
		}
		work.start_batch = start_batch;
		work.stop_batch = work.start_batch + n_batches;
		int nb_round_even = (n_batches+1)&~1;

		work.ptr_in_batches = (const uint8_t **) nn_scratch_alloc(nn, 2*nb_round_even * sizeof(uint8_t *));
		work.ptr_out_batches = (uint8_t **) work.ptr_in_batches + nb_round_even;
		work.suma_buf = (int32_t *) nn_scratch_alloc(nn, nb_round_even * sizeof(int32_t));

		if( work.ptr_in_batches == NULL || work.suma_buf == NULL) return errlog(nn,"scratch alloc failed");

		work.donesem = &info->semaphores[0];
		superfc_add_work_item(self,nn,info,work);
		semaphore_count++;
            }//b
        }//ow
	logmsg(nn,2,"semaphore_count / join_iters=%d",semaphore_count);

	// 'semaphore_count is the # of convolves in the whole operation; number
	// of donesem posts; this is used in the final join.

	waitwork.join_iters = semaphore_count;
	waitwork.donesem = &info->semaphores[0];
	superfc_add_work_item(self,nn,info,waitwork);
	/* Add work to check the output min/max and see if we need to adjust and try again */
	work.execute = superfc_execute_workitem_check_for_retry;
	superfc_add_work_item(self,nn,info,work);

	/*
	 * Sometimes we want to collect some statistics...
	 */
	if (0) superfc_statistics(nn,info,self);

	/*
	 * We've calculated the strategy, mark that the work is done. Hopefully it sticks!
	 */
	logmsg(nn,3,"superfc actual scratch use = %u * 128", nn->scratch_nextalloc/128u);
	info->needs_retry = 0;
	info->strategy_valid = 1;
	return 0;
}

/*
 * What do we have to do before calling the work function?
 *
 * Some things are cacheable:
 * * Biasbuf number range conversion
 * * Adding min_out value (converted) into biasbuf
 * * Adding gemsumb values (if needed) into biasbuf
 * * Adding in_offset * weight_offset * N into biasbuf (if needed)
 * * Partitioning scheme
 * * Reciprocal for range conversion
 * * Min and Max output information
 * * Strategy for partitioning
 * * Work function
 * * output shape/padding
 *
 * * Ensure that cached value is still accurate given these inputs (if not, recalc)
 * * Add work_for_vector items
 * * Wait for workers to complete
 * * Any fixup / postproc 
 * 
 * * The work function is passed to work_for_vector.  It can be different for different architectures.
 * * The strategy for partitioning could be a set of functions.
 */

int superfc_execute_strategy(struct nn_node *self, struct nn_graph *nn)
{
	int i;
	int err = 0;
	struct superfc_info *info = self->opaque;
	int n_work_items = info->n_work_items;
	info->cycles = 0;
	info->minval = 0;
	info->maxval = 0;
	for (i = 0; i < n_work_items; i++) {
		//Q6_dcfetch_A(&info->work_items[i+1]);
		struct workitem *p = &info->work_items[i];
		err |= p->execute(p,self,nn);
	}
	return err;
}

static inline int superfc_strategy_valid(
	struct nn_node *self,
	struct nn_graph *nn,
	struct superfc_info *info)
{
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	if (info->needs_retry) return 0;
	if (!info->strategy_valid) return 0;
	if (tensor_get_float(in_min_tensor,0) != info->in_min_float) return 0;
	if (tensor_get_float(in_max_tensor,0) != info->in_max_float) return 0;

	// check shape
	if( !shape_matches( &self->inputs[0]->shape, &info->inshape)) return 0;

	return 1;
}

static int superfc_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	struct superfc_info *nodeinfo = self->opaque;
	struct tensor *out = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	unsigned long long int total_time;
	//logmsg(nn,0,"NEW superfc id=%d",self->node_id);
	//logmsg(nn,0,"FIXME: memset out for debug... remove");
	//memset(out->data,0xAB,out->max_size);
	nn_scratch_reset(nn);			// if we recurse, reset scratch ptr

	// set this on each run.
	nodeinfo->input_base = self->inputs[0]->data;

	if (nodeinfo->strategy_valid && nodeinfo->need_initialize_suma) {
		int32_t in_batches = nodeinfo->batches;
		memset( nodeinfo->need_initialize_suma, 1, in_batches+1);
	}
	if (likely(superfc_strategy_valid(self,nn,nodeinfo))) {
		if (superfc_execute_strategy(self,nn) != 0) {
			return errlog(nn,"execute strategy failed");
		}
	} else {
		if (superfc_recalculate_strategy(self,nn) != 0) {
			return errlog(nn,"recalc strategy failed");
		}
		if (superfc_execute_strategy(self,nn) != 0) {
			return errlog(nn,"execute strategy fail after recalc");
		}
	}
	/* Replay if self-calculated min/max are insufficient */
	if (nodeinfo->needs_retry) {
		nodeinfo->recursion_depth++;
		if (nodeinfo->recursion_depth < 5) {
			return superfc_execute_hvx(self,nn);
		} else {
			logmsg(nn,0,"Extreme recursion detected, problem finding min/max?");
		}
	}

	nodeinfo->recursion_depth = 0;
	tensor_set_float(out_min,0,nodeinfo->out_minval);
	tensor_set_float(out_max,0,nodeinfo->out_maxval);
	/* Record cycles (divide by # of vector worker threads somehow?) */
	total_time = nodeinfo->cycles;
	record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER0,total_time);
	logmsg(nn,2,"out tensor info: bd=%d,%d paddings=(%d,%d)x(%d,%d)",
		out->shape.batches,out->shape.depth,
		out->format.depth_pad[0],out->format.depth_pad[1]);
	logmsg(nn,2,"SuperFC execute done!");
	return 0;
}

int superfc_check_ref(struct nn_node *self, struct nn_graph *nn)
{
	struct superfc_info *info = self->opaque;
	//if (self->n_inputs != 11) return errlog(nn,"superfc wrong # inputs... now need min/max with inf for self-detecting");
	//if (self->n_outputs != 3) return errlog(nn,"superfc wrong # outputs");
	float specified_minval = tensor_get_float(self->inputs[ 9],0);
	float specified_maxval = tensor_get_float(self->inputs[10],0);

	if (info != NULL) {
		/* Already set up, invalidate strategy and return */
		info->strategy_valid = 0;
		logmsg(nn,0,"info was already set up?");
		return 0;
	}
	if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
		return errlog(nn,"couldn't allocate info");
	}

	info->strategy_valid = 0;	/* Redundant w/ calloc */
	self->opaque = info;

	if(setup_initial_output_range(nn, info, specified_minval, specified_maxval, 0.0f, 0.5f)) return -1;

	return 0;
}

// this allocates a number of things and leaves pointers in the info struct
// if any of the allocs fail (other than for info itself)
// the strategy is to leave opaque pointing to the info
// struct, and the dtor will free any of the non-null pointers there.

int superfc_check(struct nn_node *self, struct nn_graph *nn)
{
	struct superfc_info *info = self->opaque;
	//if (self->n_inputs != 11) return errlog(nn,"superfc wrong # inputs... now need min/max with inf for self-detecting");
	//if (self->n_outputs != 3) return errlog(nn,"superfc wrong # outputs");
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *filt_min_tensor = self->inputs[4];
	const struct tensor *filt_max_tensor = self->inputs[5];
	float filt_max_float = tensor_get_float(filt_max_tensor,0);
	float filt_min_float = tensor_get_float(filt_min_tensor,0);
	float filt_level_size;
	int32_t filt_offset = get_qu8_level_size_zero(filt_min_float,filt_max_float, &filt_level_size);

	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_batches_roundup = (filt_batches + 31) & ~31;  //out_depth

	int32_t filt_depth = filt_tensor->shape.filt_height * filt_tensor->shape.filt_width  * filt_tensor->shape.filt_depth;

	int32_t filt_depth_roundup = (filt_depth + 31) & ~31;      //in depth
	uint32_t filt_elements = filt_depth_roundup;
	uint32_t weights_size = filt_elements * filt_batches_roundup;
	uint32_t out_depth = filt_batches_roundup;
	int32_t weight_batch_size = filt_depth_roundup * 32;
	int32_t n_weight_batches = superfc_n_weight_batches(weight_batch_size);
	float specified_minval = tensor_get_float(self->inputs[ 9],0);
	float specified_maxval = tensor_get_float(self->inputs[10],0);
	int i;

        logmsg(nn,2,"preset max and min %f %f",specified_minval, specified_maxval);
        logmsg(nn,2,"filt_offset = %ld",filt_offset);
  

	int weights_align = 128;
	if ((filt_elements % 32) != 0) return errlog(nn,"FIXME: < 32 depth");
	if ((filt_batches_roundup % 32) != 0) return errlog(nn,"FIXME: < 32 filts");
	if (info != NULL) {
		/* Already set up, invalidate strategy and return */
		info->strategy_valid = 0;
		logmsg(nn,0,"info was already set up?");
		return 0;
	}
	if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
		return errlog(nn,"couldn't allocate info");
	}
	self->opaque = info;

	// allocate objects.
	// We don't allocate weights & gemsumb here, since that's done within the shared weights mechanism.

	if ((info->minmax_buf = nn_memalign(128,NUM_THREADS*n_weight_batches*64*sizeof(int))) == NULL) {
		return errlog(nn,"malloc/memalign");
	}

	if ((info->biasbuf = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return errlog(nn,"alloc biasbuf");
	}
	if ((info->semaphores = nn_calloc(3+n_weight_batches,sizeof(nn_sem_t))) == NULL) {
		return errlog(nn,"alloc semaphores");
	}
	for (i = 0; i < n_weight_batches+3; i++) {
		nn_sem_init(&info->semaphores[i],0);
	}

	if ((filt_depth_roundup * filt_batches_roundup) % 128) return errlog(nn,"filt dims too odd");

	//
	// try to get existing prepared weights using using superfc_cpshare_type;
	// if we can't, make one and attach it to the const.
	//
	{
		// this is a type descriptor used by the cpshare mechanism; its address is effectively the type id
		// defining the node identity, so it must be statically allocated but local to this op.
		//
		static const struct nn_cpshare_typedesc superfc_cp_typedesc = { sizeof(struct superfc_cpshare_type) };

		struct nn_node  *const_node = nn_cpshare_get_const_node( nn, self, 1 );	// get the actual const node
		struct superfc_cpshare_type * cpshare= NULL;

		if( const_node != NULL){
			cpshare = (struct superfc_cpshare_type*) nn_cpshare_get_existing( nn, &superfc_cp_typedesc, const_node );
		}
		// if cpshare is not NULL, there's already an object of our type. We can only use it if it has
		// the right filt_offset; try again if no.

		while( cpshare != NULL && cpshare->filter_offset != filt_offset ){
			cpshare = (struct superfc_cpshare_type*)
						nn_cpshare_get_another_existing( nn, &superfc_cp_typedesc, cpshare);
		}
		// if cpshare is null, we need to make one, allocate the memory, do the conversion to packed,
		// and attach the memory to the object,
		if( cpshare == NULL ){
			// There is no suitable attached object. So make a new one.
			cpshare = (struct superfc_cpshare_type*)nn_cpshare_new( nn, &superfc_cp_typedesc);
			//that makes a new object, cleared to 0, with the header filled in. NULL return is fatal.
			if( cpshare == NULL){
				return errlog(nn,"failed to make object for weight sharing");
			}
			cpshare->filter_offset = filt_offset;	 // record this
			// allocate the weights
			uint8_t * mem_weights = nn_memalign(weights_align,weights_size);
			if( mem_weights == NULL) return errlog(nn,"alloc weights");
			int32_t *mem_gemsumb = nn_memalign(128,out_depth*sizeof(int32_t));
			if( mem_gemsumb == NULL ) return errlog(nn,"alloc gemsumb");

			// do the weight packing and gemsumb calc

			superfc_rearrange_and_sum_for_d32( nn,
				filt_tensor->data,
				filt_depth,
				filt_batches,
				mem_weights,
				filt_depth_roundup,
				filt_batches_roundup,
				filt_offset,
				mem_gemsumb);
			// store pointers in shared object
			cpshare->ptr_w = mem_weights;
			cpshare->ptr_sumb = mem_gemsumb;
			// now attach this to the const.
			nn_cpshare_attach( nn, const_node, cpshare );
			// the 'attach' call will attempt to attach the new object to the Const; if there's not
			// one already, it will be attached and the reference count bumped.
		}
		// now we have an owned reference to cpshare.
		info->cpshare = cpshare;	// make sure it gets freed ... by us or another dtor
		info->weights = cpshare->ptr_w;		// copy out the pointers to info
		info->gemsumb = cpshare->ptr_sumb;
	}


	info->in_depth = filt_depth;
	info->out_depth = filt_batches;

	info->strategy_valid = 0;	/* Redundant w/ calloc */
	info->weight_batch_size = weight_batch_size;
	info->n_weight_batches = n_weight_batches;


	info->weights_offset = filt_offset;
	info->filt_offset = filt_offset;
	info->weights_level_size = filt_level_size;

	if(setup_initial_output_range( nn, info, specified_minval, specified_maxval, 0.0f, 0.5f)) return -1;

	// figure out scratch requirements
	// components are ( each rounded up to vector):
	//
	//   need_init_suma:  1 bytes * (batches + 1)
	//   suma_buf:        4 bytes * (batches + 1)
	//    per work unit:
	//          pointers:  2 * (4bytes)*nbatches
	//          suma:     	   (4bytes)*nbatches
	//   If needed, output buffer: filt_depth_batches*batches
	//

	// find output max batches (respecting 'rank')
	unsigned max_batches;
	{
		struct output const * odef = &self->output_defs[0];
		int r = min_i32(4, odef->rank);
		max_batches = 1;
		for( int i = 0; i < r; i++){
			max_batches = mulu32_sat( max_batches, odef->max_sizes[i]);
			if( max_batches >= 0x80000000u) return errlog(nn,"bad output spec");
		}
		max_batches /= filt_batches;
	}

	/// ** this is duplicated from the strategy calc
	int out_depth_iters = filt_batches_roundup/32u;
    int inner_weight_chunks = (384*1024 + weight_batch_size - 1)/ weight_batch_size;
    if(inner_weight_chunks > out_depth_iters) inner_weight_chunks = out_depth_iters;
    int32_t outer_weight_chunks = (out_depth_iters + inner_weight_chunks - 1) / inner_weight_chunks;

	//
	// find size summed across batches assuming they are cut in NUM_THREADS parts.
    // this is a loose upper bound, hopefully.
	unsigned scratch_vecs_per_ow = 3 * ( ( max_batches*4)/128u + NUM_THREADS *3);
	unsigned need_scratch_vecs = outer_weight_chunks * scratch_vecs_per_ow;
	// room for need_init_suma
	need_scratch_vecs += ( max_batches + 1+127)/128u;
	// room for suma
	need_scratch_vecs +=  ( (max_batches + 1)*sizeof(int32_t)+127)/128u;

	// space for output buffer
	if( filt_batches_roundup != filt_batches ){
		need_scratch_vecs += (max_batches*filt_batches_roundup+127)/128u;
	}
	logmsg(nn,3,"superfc needs %u * 128 bytes scratch based on owc=%d, max_batches=%u",
			need_scratch_vecs,(int)outer_weight_chunks,max_batches);
	if( nn_scratch_grow( nn, need_scratch_vecs*128)!= 0 ){
		return errlog(nn,"can't grow to %u bytes scratch for superfc",need_scratch_vecs*128 );
	}
	return 0;
}


// this sets up:
//   info->out_minval, info->out_minval_spec, and info->minval_precalculated
// .. and the same for 'maxval'.
//
// This will always ensure that that (out_minval, out_maxval is a 'proper' range,
//  e.g -1.0 .. 1.0 may be corrected to -1.0 .. 1.00787
// The correction will be done on 'non-specified' endpoint where mathematically possible.
// The original spec is saved in out_minval_spec, out_maxval_spec; this is so, if it is necessary
// to tweak a 'fixed' endpoint, it may be restored to its original spec before the range is adjusted again.
//
// It is assumed that minval_default <=0, maxval_default >=1/128
// But the range need not be 'proper'.
//
static int
setup_initial_output_range( struct nn_graph *nn, struct superfc_info *info,
	float specified_minval,		// range specified by inputs
	float specified_maxval,
	float minval_default,			// use when specified_minval = -INF
	float maxval_default )			// use when specified_maxval = INF
{
	// enforce sanity:  min <= 0.0 <= max
	// and max > min + 1/128
	//
	if( specified_minval > 0.0f || specified_maxval < 0.0f || specified_minval >= specified_maxval )
	    return errlog(nn, "SuperFC: invalid input min/max");
	//specified_minval = fminf( specified_minval, 0.0f);
    //specified_maxval = fmaxf( specified_maxval, 0.f);
    //specified_maxval = fmaxf( fmaxf( specified_maxval, 0.f),
    //							specified_minval + 0x1.0p-7f);

	info->out_minval_spec = specified_minval;
	info->out_maxval_spec = specified_maxval;

	int mnp = (specified_minval == -INFINITY)?0:1;		// is min precalc
	int mxp = (specified_maxval == INFINITY)?0:1;		// is max precalc

	info->out_minval = mnp ? specified_minval : minval_default;
	info->out_maxval = mxp ? specified_maxval : maxval_default;

	info->minval_precalculated = mnp;
	info->maxval_precalculated = mxp;

	int corr_code = 2*mxp + mnp;
	// corr_code:
	//    bit 0 -> out_min is 'fixed'
	//    bit 1 -> out_max is 'fixed';
	// only need if minval != 0
	//
	if( info->out_minval < 0.0f ){
		adjust_minmax_for_zero_with_constraints( &info->out_minval, &info->out_maxval, corr_code);
	}
	return 0;
}

static int superfc_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct superfc_info *info = self->opaque;
	if (info != NULL) {
		superfc_reset_work_items(self,nn,info);
		// note that weights and gemsumb are not freed here since they are managed via info->cpshare

		if( info->cpshare != NULL) nn_cpshare_decref( nn, info->cpshare );	// decref shared object

		// tolerate a partially constructed info
		//
		if( info->semaphores != NULL ) nn_free(info->semaphores);
		if( info->biasbuf != NULL ) nn_free(info->biasbuf);
		if( info->minmax_buf != NULL ) nn_free(info->minmax_buf);
		nn_free(info);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}

static int superfc_dtor_ref(struct nn_node *self, struct nn_graph *nn)
{
	struct superfc_info *info = self->opaque;
	if (info != NULL) {
		superfc_reset_work_items(self,nn,info);
		nn_free(info);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}

struct nn_node_ops nn_ops_for_SuperFC_8x8p32to8_ref = {
	.execute = superfc_execute_ref,
	.check = superfc_check_ref,
	.ctor = node_alloc_common,
	.dtor = superfc_dtor_ref,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
};
struct nn_node_ops nn_ops_for_SuperFC_8x8p32to8 = {
	.execute = superfc_execute_hvx,
	.check = superfc_check,
	.ctor = node_alloc_common,
	.dtor = superfc_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
	.flags = 0,
};
struct nn_node_ops nn_ops_for_SuperFC_8x8p32to8_d32 = {
	.execute = superfc_execute_hvx,
	.check = superfc_check,
	.ctor = node_alloc_common,
	.dtor = superfc_dtor,
	.n_inputs = NN_IOCOUNT(11),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
