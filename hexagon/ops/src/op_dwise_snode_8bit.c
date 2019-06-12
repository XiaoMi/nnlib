
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
 * This contains the code for convolution depthwise for 5x5 filters
 */
/*
 * FIXME: temporary minmax buf should be on stack
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

#include "nn_bufferpool.h"

#ifdef HEXAGON_V66
#define NUM_THREADS 4
#else
#define NUM_THREADS 2
#endif

#define MAX_FILT_HEIGHT 7
/* 
 * Size of VTCM to reserve for circular buffer
 * Rest of VTCM will possibly be used for weights 
 * 
 * Note: at last calculation, max required circbuf size is 160K.
 * For Inception V3, max slice of weights is 128K.  
 * 
 * 0 will disable this
 */
#define VTCM_CIRCBUF_SIZE (0)

/* 8x8 convolution --> 32 bits, biasadd, relu, quantizedown to 8 bits  */

/*
 * Input and output have ordering BHWD
 * Filter has ordering HWDB (B is # of filters)
 */

#ifdef __hexagon__
#include <hexagon_protos.h>
#endif

/*
 * This structure has values that change with different work items
 */

struct workitem {
	int (*execute)(struct workitem *, struct nn_node *node, struct nn_graph *nn);	// exec function
	void (*new_execute)(struct nn_graph *nn, void *opaque); // work_for_vector-compatible exec function, passed work item
	struct nn_node *self;		// This node
	struct supernode_info_new *info;	// same as self->opaque
	nn_sem_t *donesem;	// semaphore to post completion
	nn_sem_t *do_conv;      // semaphore to trigger convolution
	// nn_sem_t *copy_done;    // semaphore to wait on completion of weights copy
	nn_sem_t *conv_done;    // semaphore to wait on completion of convolution
	int32_t  wait_before;   // wait on the semphores or not
	int32_t  threads;       // how many threads will work on this.
	int32_t batch_index;	// current batch index (input supernode)
	/* Main convolutional work items */
	const uint8_t *input;	// Input data.  Could be from input tensor or temp buf
	const uint8_t *weights;	// Filter data.  Could be from input tensor or temp buf
	const int32_t *biases;	// Bias data, in product space (added to in * filt products)
	const uint32_t *recip;  // Vector per channel reciprocal quantization
	uint8_t *output;	// Output data.  Could be output tensor or temp buf
	int32_t *suma_buf;	// Output data.  Could be output tensor or temp buf
	int32_t start_line;	// Row to start working on
	int32_t stop_line;	// Row too far to work on, could be out_height
	int32_t skip_lines;	// Number of rows to skip each iteration
	int32_t num_lines;	// Number of rows to skip each iteration
	int32_t *minmax_buf;	// min/max values
	uint8_t *circ_buf;      // temp buf used by v65 code
	int32_t weight_chunks;	// How many d32 chunks of weights to do
	int32_t weight_batch_size; // size of each weight chunk
        int32_t start_chunk;    // the first absolute poisiton of the weight chunks to use
	int32_t last_chunk;     // is this the last weight chunk in entire operation (v65 only)
        int32_t vtcm_chunks;    // num weight chunks in vtcm
        int32_t fix_overcompute;// deal with oadding soecifically for tuples of 4
	/* Information about input pad zapping */
	uint8_t *zap_top;	// pointer to top zap
	uint8_t *zap_bot;	// pointer to bottom zap
	uint8_t *zap_left;	// pointer to first left zap, NOT necessarily a whole vector
	uint8_t *zap_right;	// pointer to right zap, until end of vector
	int32_t zap_top_size;	// amount to zap on top;
	int32_t zap_bot_size;	// amount to zap on bottom;
	int32_t zap_rl_depths;	// number of right/left zaps per row
	int32_t zap_left_size;	// width to zap on the left
	int32_t zap_right_size;	// width to zap on the right
	int32_t zap_batches;	// batches to zap
	int32_t nonzap_width;	// width to copy into
	int32_t zap_height;	// height to zap
	uint8_t zap_value;	// value to zap with
	int32_t zap_jobs;
	/* Information for prefetching */
	const uint8_t *pf_inp;	// where to start prefetch, NULL if no pf needed
	uint32_t pf_width;	// width to fetch
	uint32_t pf_stride;	// Distance between rows
	uint32_t pf_height;	// number of rows;
	/* Information for GEMSUMA / integral image calculations */
	uint8_t need_initialize_suma;
	int16_t suma_progress_index;	// selects progress variable for suma.
	int32_t suma_progress_need;		// v60 convolution needs this suma progress.
//	const uint8_t *suma_in;	// input data, NULL if no suma needed
//	int32_t *suma_int_tmp;	// Temporary storage for output of integral --> suma
//	int32_t *suma_output;	// output data
//	int32_t *suma_scratch;	// scratch storage
//	int32_t suma_num_lines;	// Number of final output lines
	/* Memcpy work items */
	const uint8_t *copy_in;	// copy in location, NULL if no copy in needed
	uint8_t *copy_out;	// copy out location
	uint32_t copy_size;	// amount to copy
	int32_t join_iters;	// how many times to decrement done semaphore
	int32_t my_idx;		// sometimes we want to know our index.
	nn_checkpoint_t *zap_checkpoint;
	int32_t next_startup_offset;
};

struct weight_slice_info {
	nn_checkpoint_t checkpoint;
	const uint8_t *copy_in;
	uint8_t *copy_out;
	uint32_t copy_size;
	int batch_start_offset;
	uint32_t wakeup_items;
};

/*
 * Pointers and values that are common to everything in the node
 * Also values that are constant across all work items
 */
struct supernode_info_new {
	uint8_t *weights;	// weights, padded and adjusted as necessary
	int32_t *biasbuf;	// int32 bias buffer, including min offsets and gemsumb
	int32_t *minmax_buf;	// pointer to min/max values, enough storage per thread...
        uint32_t *recip;        //local quantization per channel
	nn_sem_t *semaphores;	// pointer to preallocated array of semaphores
	struct workitem *work_items;	// All the work items to execute at execute time
	nn_os_workitem_t *work_list;	// compiled work list
	int n_work_items;		// how many work items?
	int batch_start_idx;		// pointer during more-distributed execution
	int workitems_alloc;	//	bytes allocated for work items
	float out_minval;	// Minimum output value, either specified or guessed
	float out_maxval;	// maximum output value, either specified or guessed
	uint8_t outrange_firstguess;	// is the output range a baseless guess
	uint8_t minval_precalculated;	// Is the minval precalculated?
	uint8_t maxval_precalculated;	// Is the maxval precalculated?
	uint8_t minmax_precalc_flags;	// bit 0 = min_precalc, bit 1 = max_precalc
	float out_minval_spec;		// exact value specified (when not precalculated)
	float out_maxval_spec;		// exact value specified (when not precalculated)
	// Note: minval,maxval are in *output* units (without saturation) for supernode,
	// to support channel and weight scaling more easily.
	// For Depthwise and shortin, they are in product space.
	int32_t minval;			// Minimum value actually observed
	int32_t maxval;			// Maximum value actually observed
	int32_t weight_batch_size;	// How big is a weight batch (32 filters)
	int32_t n_weight_batches;	// Number of weight batches we can try and fit at once into vtcm
        int32_t depth_multiplier;       // Used in depthwise ops to see how many output depth per input depthes there are
	uint8_t is_dwise;		// is depthwise?
	uint8_t needs_retry;		// Do we need to try this op over again?
	uint8_t strategy_valid;		// Do we believe the strategy is currently valid?
	uint8_t UNUSED_weights_arranged;	// Have the weights been rearranged yet?
	float in_max_float;	// maximum input float value
	float in_min_float;	// minimum input float value
	float weights_level_size;	// how large in float is one increment in the weights?
	int weights_offset;	// where is 0 in weight number space?
	struct shape in_shape;		// previous actual shape
	int32_t in_height;	// height of the input
	int32_t in_width;	// input width to compute
	int32_t in_next_row;	// distance from one row to the next
	int32_t in_depth;	// input depth to compute
        int32_t in_depth_after; // amount of depth padding in total in depth
	int32_t in_next_d32;	// distance from one depth32 slice to the next on the same row
	int32_t in_left_skip; 	// number of width elements to throw away on the left side output
	int32_t in_right_padpad;// number of width elements to add onto the padded data in circ buffer
	int32_t in_next_batch;	// stride from one batch to the next
	int32_t out_width;		// output width to compute, should be width/stride
	int32_t out_width_processed;		// output width, incl any left_pad %4, but no right pad
	int32_t out_next_row;	// distance from one row to the next 
	int32_t out_next_batch; // distance from one batch to the next
	int32_t out_depth_total;	// total depth of output
	int32_t out_depth_valid;	// total depth to compute
	int32_t out_next_d32;	// distance from one depth32 slice to the next on the same row
	int32_t out_height;	// number of output lines to compute
	int32_t out_left_junk; 	// number of width elements to throw away on the left side output
	int32_t skip_col;       // skip an iteration and flush data out
	int32_t filt_width;	// filter width
	int32_t filt_height;	// filter height
	int32_t stride_width;	// stride in the width dimension
	int32_t stride_height;	// stride in the height dimension (== width usually)
        dwconv2dbbb_t dwfunc;   // pointer to the processing function in use.
	const uint8_t *suma_in;	// input pointer to start SUMA work... should be in workitem...
	int32_t suma_width;	// elements of a SUMA row
	int32_t next_suma_off;	// bytes of a SUMA row
	int32_t *suma_buf;	// GEMSUMA (if needed)
	int32_t suma_start;	// where to start in suma buffer
	int32_t integral_off;   //index into integral buffer used by gvsuma
	int32_t recip_val;	// reciprocal for product space --> output space
	int32_t recip_shamt;	// amount to shift before recip mpy
	int recip_shamt_must_be_zero;	// set in nodes which don't support recip_shamt (shortinconv; v66 conv)
	int32_t num_accs;       // number of accumulators used in main computation
	int in_offset;		// amount to normalize inputs by.  Needed?
	int filt_offset;	// amount to normalize filter values by. Needed?
	int32_t recursion_depth;// how far have we recursed?
	const uint8_t *input_base0;	// first row (including all left padding, in-use top padding)
	const uint8_t *input_base;	// first row (including in-use left padding, in-use top padding).
	const uint8_t *weights_base;
	const uint8_t * raw_input; //ptr to the input tensor for use when copied into temp


	// 'k' is the channel scaling factors channel_scale[d]/wt_scale[d]; channel_scale
	// is the optional 'channel_scale' external factors (<=1.0) and wt_scale is the internal
	// weight scaling factor (127/256 <= wt_scale <= 1.0)
	// Both are vector aligned and dimensioned as [out_depth_total].
	// Intiially the channel_scale are loaded into k_factor, and the weight_scale into k_factor_recip
	// (as fixed-point integer), and then both are corrected.
	float * k_factor;
	float * k_factor_recip;
	uint8_t has_channel_scale;
	uint8_t has_weight_scale;
	float max_k_factor;		// the  largest 'k_factor' (and >=1.0).

	// min_valid_val, max_valid_val apply only when min/max is in product space.
	int32_t max_valid_val;	// maximum value that results in a value not above max_out
	int32_t min_valid_val;	// minimum value that results in a value not below min_out
	float prod_level_size;	// in_level_size * filt_level_size
	float output_level_size;
	int32_t *gemsumb;	// GEMSUMB value, if we want to calculate it at preparation time
	uint8_t use_v65;	// Should we use V65 mode?
	uint8_t use_v66;	// Should we use V66 mode?
	uint8_t use_signed_weights;		// weights converted to signed?
	uint8_t use_vtcm;       //flag to use vtcm or not for weights
	uint64_t cycles;	// Cycle accumulator for children
	struct nn_os_bufstack_t bufstack;	// stack of temporary buffers
	struct buffer_pool bufpool_suma_scratch;// pool of 'suma_scratch' buffers
	struct weight_slice_info *conv_slices;	// checkpoints and info for when conv is complete
	struct weight_slice_info startup_info;
	int errors;
	nn_checkpoint_t alldone_checkpoint;	// checkpoint for all convs completing
	nn_sem_t alldone_sem;
	void *prepared_vtcm_addr;
	const struct nn_node *earlywork_pred;	// A node before this one that might be able to take early work
	struct nn_early_work my_earlywork;	// Information about early work
	struct nn_early_work *next_earlywork;	// Work requested by future node
};


static void setup_initial_output_range( struct supernode_info_new *info, float,float,float,float);

static inline int supernode_execute_some_strategy(struct nn_node *self, struct nn_graph *nn, int start, int n_work_items)
{
	struct supernode_info_new *info = self->opaque;
	int err = 0;
	logmsg(nn,3,"adding %d items @ %d",n_work_items,start);
	nn_os_worklist_for_vector(nn,&info->work_list[start],n_work_items);
	return err;
}


#define roundup(a, p2)       (((a)+(p2)-1)&~((p2)-1))

static inline void supernode_do_memcpy(struct nn_graph *nn, uint8_t *dst, const uint8_t *src, uint32_t size)
{
	return nn_graph_memcpy(nn,dst,src,size);
}

/*
 * Managing the padding and zapping amounts is getting insane!
 * Break it down some... maybe even with some arch-specific functions
 */

/*
 * A note on padding
 * Our input tensor has some amount of LEFT, RIGHT, TOP, and BOTTOM padding
 * Additionally, there is at least one vector of space BEFORE 
 * 
 * The total stride from one row to the next is depth * (input_width + PAD_L + PAD_R) 
 * The input should have:
 * PAD_R >= 4
 * (input_width+PAD_L+PAD_R) % 4 == 0
 * 
 * If we have insufficient input LEFT padding, we should have at 
 * least 4 columns (4 * depth32) of RIGHT padding.  To generate the LEFT
 * padding, we move the pointer back by one 4*depth32 vector.  This moves 4 
 * values from the RIGHT padding to the LEFT padding.
 *
 * If we have insufficient input RIGHT padding, check to see if RIGHT+LEFT padding is OK.
 * If so, we can just read a little extra into the padding of the next row.
 *
 * Padding should be considered to have GARBAGE values on input, so you might need to zero them.
 *
 * You also need to pick output padding.
 * We want to have whatever PAD_LEFT is convenient (reducing the PAD_L by 1 for 3x3 filter, for example).
 * We want to ensure the total output stride % 4 == 0 (128B) and (hopefully) that the PAD_R is >= 4.
 * 
 * We could support misaligned stores on output, which could be useful if we
 * want to force no padding on output.
 * 
 * For stride=2, we have a problem if parity(wanted_padding) != parity(actual_padding).  
 * There should be enough wiggle room in the asm implementation to support starting one d32 over.
 */

/*
 * In the hopes of simplification:
 * We have been guaranteed that we are sufficiently padded and aligned.  No im2col here!
 * Padding and alignment have been factored in so that we don't care about padding type.
 * 
 * Although I'm slightly concerned about strided accesses and exact left padding...
 */



/*
 * // foreach batch -- moved outside
 *   l2fetch base line
 *   foreach 32 weigts of output depth
 *     foreach slice
 *       prefetch next slice
 *       prefetch next filters
 *       gvsumb (if needed?)
 *       gvconv (w/ gemsuma)
 *  // moved outside? unpad if needed
 * set output info
 * post sem
 */

static inline void debug_pprint_vector(const struct supernode_info_new *ndata) {}

static inline int supernode_reset_work_items(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info)
{
	if (info->work_items) nn_free(info->work_items);
	if (info->work_list) nn_free(info->work_list);
	info->work_items = NULL;
	info->work_list = NULL;
	info->n_work_items = 0;
	info->workitems_alloc = 0;
	return 0;
}
// (reset record count; keep the buffer)
static inline int supernode_softreset_work_items(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info)
{
	info->n_work_items = 0;
	return 0;
}

static inline int supernode_add_work_item(
	struct nn_node *self, 
	struct nn_graph *nn, 
	struct supernode_info_new *info,
	const struct workitem work /* BY VAL */)
{
	struct workitem *witems = info->work_items;
	int this_index = info->n_work_items;
	int new_work_items = this_index+1;

	/* Round up to multiple of 16 work items */
	if (new_work_items > info->workitems_alloc)
	{
		int new_work_alloc = (new_work_items + 15) & ~15;
		struct workitem *new_data;
		if ((new_data=nn_realloc(witems,new_work_alloc*sizeof(*new_data))) == NULL) {
			return errlog(nn,"realloc fails");
		}
		info->workitems_alloc = new_work_alloc;
		info->work_items = witems = new_data;
#if 1
		nn_os_workitem_t *new_list;
		if ((new_list=nn_realloc(info->work_list,new_work_alloc*sizeof(*new_list))) == NULL) {
			return errlog(nn,"realloc fails");
		}
		info->work_list = new_list;
#endif
	}
	witems[this_index] = work;
	witems[this_index].my_idx = this_index;
	info->n_work_items = new_work_items;
	
	return this_index;
}

static inline void supernode_compile_worklist(
	struct nn_graph *nn,
	struct supernode_info_new *info,
	struct nn_node *self)
{
	int i;
	for (i = 0; i < info->n_work_items; i++) {
		info->work_list[i].f = info->work_items[i].new_execute;
		info->work_list[i].arg = &info->work_items[i];
	};
}

static int supernode_execute_workitem_prefetch(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	logmsg(nn,1,"prefetching @ %p: %d(%d)x%d",work->pf_inp,work->pf_width,work->pf_stride,work->pf_height);
	if(work->pf_inp!=NULL) l2fetch(work->pf_inp,work->pf_stride,work->pf_width,work->pf_height);
	return 0;
}

#if 1
/* EJP: used in dwise supernode currently */
static void supernode_hvx_copy_wait(struct nn_graph *nn, void * vinfo)
{
	/* Copy or prefetch */
#ifdef V65
	struct workitem *work = vinfo;
    nn_graph_hvx_blockcopy(work->copy_out, work->copy_in, work->copy_size);

    nn_sem_post_n_times( work->do_conv,  work->threads);

#endif
	return;
}
#endif

// This workitem does the following:
//     (1) wait for 'conv_done', 'threads' times  (but only if conv_done is not NULL)
//     (2) Start a vector thread which:
//          (a) runs the copy, then
//          (b) does post(do_conv) 'work.threads' times
//
//   (step (2) is skipped if work->copy_out is null)
//
// This is used between groups of executions where weights need switching. The subsequent
// convolution workitems all take( do_conv) before reading the weights.
// The 'wait for conv_done' is used when this is not the first copy, in order to enure
//  the previous convolutions are finished.

#if 1
/* EJP: used in dwise supernode currently */
static int supernode_execute_workitem_copy(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	nn_sem_t * conv_done = work->conv_done;
	if( conv_done != NULL){
		nn_sem_wait_n_times( conv_done, work->threads);
	}
	if (work->copy_out == NULL) return 0;
	nn_os_work_for_vector(nn,supernode_hvx_copy_wait,work);
	return  0;
}

//
// add a "weight_copy" workitem.
// if wait_before !=0, there will be a sem_wait off semaphores[2] prior
// to triggering the copy.
// FIXME: EJP: need to change to have this triggered by checkpoint
//
static inline __attribute__((unused)) int supernode_add_weight_copy(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info,
	int wait_before,
	const uint8_t *input,
	int weight_batch_size,
	int weight_chunks, int threads)
{
	struct workitem work;
	work.info = info;
	work.copy_in = input,
	work.copy_out = nn->vtcm_ptr;
	work.execute = supernode_execute_workitem_copy;
//	work.weight_chunks = weight_chunks;
        work.copy_size = weight_chunks*weight_batch_size;
	work.threads = threads;
	work.do_conv = &info->semaphores[1];		// post this after, * threads
	work.conv_done = wait_before ?  &info->semaphores[2]	// wait for this before, * threads
			: NULL;					// no wait
//	work.copy_done = &info->semaphores[3];
	return supernode_add_work_item(self,nn,info,work);
}
#endif
//
// Add an l2fetch work item
// This needs to be triggered from checkpoint
static inline __attribute__((unused)) int supernode_add_l2fetch(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info,
	const uint8_t *input,
	int width,
	int height)
{
	struct workitem work;
	work.pf_inp = input,
	work.pf_width = width;
	work.pf_stride = width;
	work.pf_height = height;
	work.execute = supernode_execute_workitem_prefetch;
	return supernode_add_work_item(self,nn,info,work);
}


// Find y, the smallest power of 2 such that abs(y) >= abs(x)
// (and y having the same sign as x).
// x should be !=0 and not denormal.
//
static inline float
__attribute__((unused))
to_next_power_of_two( float x)
{
	// round the 32-bit code up to the next value in which the mantissa is all zero.
	union {
		float f;
		uint32_t u32;
	} uu = { x };
	uint32_t m_mask = (1<<23)-1;
	uu.u32 =  ( uu.u32 + m_mask ) & ~m_mask;
	return uu.f;
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

// needs to wait for all workers to be done
// Maybe change to just be called by worker thread
static int supernode_execute_workitem_check_for_retry(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	struct supernode_info_new *info = node->opaque;
	float newval;
	float extreme_out;

	logmsg(nn,1,"output range is %d .. %d\n", (int)info->minval , (int)info->maxval);
	int recalc = 0;
	int precalc_flags = info->minmax_precalc_flags;

	if( precalc_flags == 3)
		return 0;		// nothing to move

	int min_allowed, max_allowed;
	float minmax_level_size;
	min_allowed = 0;
	max_allowed = 255;
	minmax_level_size = info->output_level_size;
	if (unlikely((precalc_flags&2)==0 && (info->maxval >max_allowed))) {
		/* Mark as needing retry and set new max value */
		info->needs_retry = 1;
		extreme_out = minmax_level_size*(info->maxval + 0.5f)+ info->out_minval;
		newval = round_up_quarter_octave( fmaxf(extreme_out, 0x1.0p-4f));
		logmsg(nn,1,"max too small, recalculating %d > %d / %f > %f... picking %f",
			info->maxval,info->max_valid_val,extreme_out,info->out_maxval,newval);
		info->out_maxval = newval;
		recalc = 1;
	}
	if (unlikely((precalc_flags&1)==0 && (info->minval < min_allowed))) {
		/* Mark as needing retry and set new min value */
		info->needs_retry = 1;
		extreme_out = minmax_level_size*(info->minval - 0.5f)+ info->out_minval;
		newval = round_up_quarter_octave( fminf(extreme_out, -0x1.0p-8f));
		logmsg(nn,1,"min too large, recalculating %d < %d / %f < %f... picking %f",
			info->minval,info->min_valid_val,extreme_out,info->out_minval,newval);
		info->out_minval = newval;
		recalc = 1;
	}
	// if endpoints moved, adjust to get a valid zero.
	if( recalc ){
		if( precalc_flags & 1) info->out_minval = info->out_minval_spec;
		if( precalc_flags & 2) info->out_maxval = info->out_maxval_spec;
		adjust_minmax_for_zero_with_constraints( & info->out_minval, &info->out_maxval, precalc_flags);
	}

	return 0;
}

static void supernode_execute_zap_right(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	const struct supernode_info_new *info = work->info;
	uint8_t *rowstart = work->zap_right;
	uint8_t val = work->zap_value;
	uint32_t size = work->zap_right_size;
	uint32_t in_next_d32 = info->in_next_d32;
	uint32_t in_next_row = info->in_next_row;
	int32_t batch_stride = info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn,2,"zapping right: %d*%d bytes @ %p rl_depths=%d next_d32=%d next_row=%d val=%d height=%d",work->zap_batches,size*32,rowstart,work->zap_rl_depths,in_next_d32,in_next_row,val,work->zap_height);
	for (i = 0; i < n_batches; i++) {
		padzap_part(rowstart,val,in_next_d32,work->zap_rl_depths,in_next_row,work->zap_height,size);
		rowstart += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn,work->self,work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void supernode_execute_zap_left(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	const struct supernode_info_new *info = work->info;
	uint8_t *rowstart = work->zap_left;
	uint8_t val = work->zap_value;
	uint32_t in_next_d32 = info->in_next_d32;
	uint32_t in_next_row = info->in_next_row;
	int32_t batch_stride = info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn,2,"zapping left: %d*%d bytes @ %p rl_depths=%d next_d32=%d next_row=%d val=%d height=%d",work->zap_batches,32*work->zap_left_size,rowstart,work->zap_rl_depths,in_next_d32,in_next_row,val,work->zap_height);
	for (i = 0; i < n_batches; i++) {
		padzap_part(rowstart,val,in_next_d32,work->zap_rl_depths,in_next_row,work->zap_height+2,work->zap_left_size);
		rowstart += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn,work->self,work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void __attribute__((unused)) supernode_execute_zap_toptop(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	uint8_t *start = work->zap_top - work->info->in_next_row;
	int32_t batch_stride = work->info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn,2,"zapping toptop: %d*%d bytes @ %p val=%d",work->zap_batches,work->zap_top_size,work->zap_top,work->zap_value);
	for (i = 0; i < n_batches; i++) {
		vmemset_nt_asm(start,work->zap_value,work->info->in_next_row);
		start += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn,work->self,work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void supernode_execute_zap_top(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	uint8_t *start = work->zap_top;
	int32_t batch_stride = work->info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn,2,"zapping top: %d*%d bytes @ %p val=%d",work->zap_batches,work->zap_top_size,work->zap_top,work->zap_value);
	for (i = 0; i < n_batches; i++) {
		vmemset_nt_asm(start,work->zap_value,work->zap_top_size);
		start += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn,work->self,work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void supernode_execute_zap_bot(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	uint8_t *start = work->zap_bot;
	int32_t batch_stride = work->info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn,2,"zapping bot: %d*%d bytes @ %p val=%d",work->zap_batches,work->zap_bot_size,work->zap_bot,work->zap_value);
	for (i = 0; i < n_batches; i++) {
		vmemset_nt_asm(start,work->zap_value,work->zap_bot_size);
		start += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn,work->self,work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}
/*
 * EJP: FIXME: Change semaphores to checkpoint
 * This workitem would start on a vector thread, add work to the queue, and checkpoint should trigger what's next
 */
static int supernode_execute_workitem_zap(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
//int supernode_execute_workitem_zap(struct nn_graph *nn, void *vwork)
{
	//struct workitem *work = vwork;
	int sems_down = 0;
	long right_ptr_l = (long)work->zap_right;
	nn_sem_t donesem;
	work->donesem = &donesem;
	work->zap_checkpoint = NULL;
	nn_sem_init(&donesem,0);
	logmsg(nn,2,"zapping start");
	if (right_ptr_l % 128) {
		nn_os_work_for_vector(nn,supernode_execute_zap_right,work);
		sems_down++;
	}
	if (work->zap_top_size > 0) {
		//printf("zapping %d (%d) at top: %p\n", (int)(work->zap_top_size/work->info->in_next_row), (int)work->zap_top_size, work->zap_top);
		nn_os_work_for_vector(nn,supernode_execute_zap_top,work);
		sems_down++;
	}
	if (work->zap_bot_size > 0) {
		//printf("zapping %d (%d) at bottom: %p\n", (int)(work->zap_bot_size/work->info->in_next_row), (int)work->zap_bot_size, work->zap_bot);
		nn_os_work_for_vector(nn,supernode_execute_zap_bot,work);
		sems_down++;
	}
		nn_os_work_for_vector(nn,supernode_execute_zap_left,work);
		sems_down++;
#ifndef V66
	if(1) if (work->zap_top_size > 0) {
		nn_os_work_for_vector(nn,supernode_execute_zap_toptop,work);
		sems_down++;
	}
#endif
	nn_sem_wait_n_times(&donesem, sems_down);
	debug_pprint_vector(work->info);
	work->donesem = NULL; // make klockwork happy: don't leave pointer to stack var
	logmsg(nn,2,"zapping complete");
	return 0;
}

static inline void supernode_handle_earlywork(struct nn_graph *nn, struct nn_node *self, struct supernode_info_new *info)
{
	struct nn_early_work *work = info->next_earlywork;
	if (work == NULL) return;
	if (work->vtcm_addr != nn->vtcm_ptr) return;
	if (work->src_addr == NULL) return;
	if (work->dst_addr == NULL) return;
	if (work->bytes == 0) return;
	logmsg(nn,2,"Doing early work copy: %d bytes %p <-- %p",work->bytes,work->dst_addr,work->src_addr);
	supernode_do_memcpy(nn,work->dst_addr,work->src_addr,work->bytes);
	work->valid = 1;
}

static void note_alldone_checkpoint_arrival(struct nn_graph *nn, struct nn_node *self, void *opaque)
{
	struct workitem *work = opaque;
	struct supernode_info_new *info = work->info;
	logmsg(nn,2,"Saw all done checkpoint complete @ node %p work item %p",self,work);
	if (work->next_startup_offset == 0) {
		logmsg(nn,2,"Last batch, should return info=%p",info);
		// Move early work before the alldone sem post
		supernode_handle_earlywork(nn,self,info);
		nn_sem_post(&info->alldone_sem);
	} else {
		logmsg(nn,4,"Enqueue next batch startup distance %d",work->next_startup_offset);
		supernode_execute_some_strategy(self,nn,info->batch_start_idx + work->next_startup_offset,1);
	}
}

static void note_startup_arrival(struct nn_graph *nn, struct nn_node *self, void *opaque)
{
	struct supernode_info_new *info = self->opaque;
	struct weight_slice_info *myslice = opaque;
	logmsg(nn,2,"Saw startup checkpoint complete @ node %p batch start offset=%d n=%d",self,
		myslice->batch_start_offset,myslice->wakeup_items);
	supernode_execute_some_strategy(self,nn,info->batch_start_idx+myslice->batch_start_offset,myslice->wakeup_items);
	//nn_checkpoint_arrival(nn,self,&info->alldone_checkpoint);
}

static inline void supernode_do_startup_memcpy(
	struct nn_graph *nn,
	struct nn_node *self,
	struct supernode_info_new *info,
	uint8_t *dst,
	const uint8_t *src,
	uint32_t size)
{
	if (info->my_earlywork.valid) {
		/* Skip over early work, predecessor did it */
		logmsg(nn,2,"Yay, early work was valid! Skipping copy...");
		info->my_earlywork.valid = 0;
	} else {
		supernode_do_memcpy(nn,dst,src,size);
	}
}


/*
 * EJP: FIXME: Change semaphores to checkpoint
 * This workitem would start on a vector thread, add work to the queue, and checkpoint should trigger what's next
 */
//int supernode_execute_workitem_startwork(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
static void supernode_execute_workitem_startwork(struct nn_graph *nn, void *vwork) //struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	struct workitem *work = vwork;
	struct nn_node *node = work->self;
	struct supernode_info_new *info = work->info;
	int jobs = 0;
	long right_ptr_l = (long)work->zap_right;
	int work_idx = work->my_idx;
	logmsg(nn,3,"starting work idx=%d",work_idx);
	//nn_sem_init(&info->alldone_sem,0);
	work->donesem = NULL;
	work->zap_checkpoint = &info->startup_info.checkpoint;
	info->batch_start_idx = work_idx + 1;
	work->wait_before = 0;
	// We set up our checkpoint for the number of zaps to do plus one arrival for ourselves.
	nn_checkpoint_init(&info->startup_info.checkpoint,work->zap_jobs+1,note_startup_arrival,&info->startup_info);
	nn_checkpoint_init(&info->alldone_checkpoint,work->join_iters,note_alldone_checkpoint_arrival,work);
#ifdef V66
	// V66 memcpy is in background, do it first to overlap with zapping
	if (work->copy_size > 0) {
		supernode_do_startup_memcpy(nn,node,info,work->copy_out,work->copy_in,work->copy_size);
	}
#endif
	if (work->pf_height && work->pf_inp) {
		l2fetch(work->pf_inp,work->pf_stride,work->pf_width,work->pf_height);
	}
	if (work->zap_right_size && (right_ptr_l % 128)) {
		nn_os_work_for_vector(nn,supernode_execute_zap_right,work);
		jobs++;
	}
	if (work->zap_top_size > 0) {
		//printf("zapping %d (%d) at top: %p\n", (int)(work->zap_top_size/work->info->in_next_row), (int)work->zap_top_size, work->zap_top);
		nn_os_work_for_vector(nn,supernode_execute_zap_top,work);
		jobs++;
	}
	if (work->zap_bot_size > 0) {
		//printf("zapping %d (%d) at bottom: %p\n", (int)(work->zap_bot_size/work->info->in_next_row), (int)work->zap_bot_size, work->zap_bot);
		nn_os_work_for_vector(nn,supernode_execute_zap_bot,work);
		jobs++;
	}
#if 0 //def V66
	if (work->zap_left_size > 0) {
		nn_os_work_for_vector(nn,supernode_execute_zap_left,work);
		sems_down++;
	}
#else
	if (work->zap_left_size > 0) {
		nn_os_work_for_vector(nn,supernode_execute_zap_left,work);
		jobs++;
	}
#endif
#ifndef V66
	if(1) if (work->zap_top_size > 0) {
		nn_os_work_for_vector(nn,supernode_execute_zap_toptop,work);
		jobs++;
	}
#endif
#ifndef V66
	// Pre-V66 memcpy work is in foreground, do it here to do it in parallel
	if (work->copy_size > 0) {
		supernode_do_startup_memcpy(nn,node,info,work->copy_out,work->copy_in,work->copy_size);
	}
#endif
	// our own mark of checkpoint completion, mainly for pre-V66 memcpy
	nn_checkpoint_arrival(nn,node,&info->startup_info.checkpoint);
	if (unlikely(jobs != work->zap_jobs)) errlog(nn,"consistency failure in startup zap work");
	//nn_sem_wait(&info->alldone_sem);
	//logmsg(nn,0,"v60 startup complete");
	return;
}

static int supernode_count_zap_jobs(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	int zaps = 0;
	long right_ptr_l = (long)work->zap_right;
	if (work->zap_right_size && ((right_ptr_l % 128) != 0)) zaps++;
	if (work->zap_top_size > 0) zaps++;
	if (work->zap_bot_size > 0) zaps++;
	if (work->zap_left_size > 0) zaps++; /* Zap Left always? */
#ifndef V66
	if (work->zap_top_size > 0) zaps++; /* zap toptop */
#endif
	return zaps;
}

/*
 * We are computing (a-a_offset)*(b-b_offset)
 * If a_offset != 0, we need to compute a_offset * sum(b-b_offset)
 * We can sum the elements of the filter (offsetted) and multiply by a_offset
 * This we can do ahead of time.
 */

static inline __attribute__((unused)) int supernode_add_padding_zap(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info,
	struct workitem zapwork /* BY VAL */,
	int32_t input_batch_offset,
	int32_t top_zap,
	int32_t left_zap)
{
	zapwork.zap_top += input_batch_offset;
	zapwork.zap_bot += input_batch_offset;
	zapwork.zap_left += input_batch_offset;
	zapwork.zap_right += input_batch_offset;
	zapwork.execute = supernode_execute_workitem_zap;
	return supernode_add_work_item(self,nn,info,zapwork);
}

static inline __attribute__((unused)) int supernode_add_batch_startup(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info,
	struct workitem startwork /* BY VAL */,
	int32_t input_batch_offset,
	int32_t top_zap,
	int32_t left_zap,
	int32_t zap_batches)
{
	startwork.next_startup_offset = 0;
	startwork.zap_top += input_batch_offset;
	startwork.zap_bot += input_batch_offset;
	startwork.zap_left += input_batch_offset;
	startwork.zap_right += input_batch_offset;
	startwork.zap_batches = zap_batches;
	startwork.new_execute = supernode_execute_workitem_startwork;
	int n_zaps = supernode_count_zap_jobs(&startwork,self,nn);
	startwork.zap_jobs = n_zaps;
	return supernode_add_work_item(self,nn,info,startwork);
}

static inline int32_t __attribute__((unused)) weights_min_footprint(int b, int h, int w, int d)
{
#ifdef V65
	return b*h*w*d;
#elif V66
	return 0;
#else
	return b*h*w*d*2;
#endif
}

static inline __attribute__((unused)) uint32_t propose_bw(
	int32_t weight_slice_factor,
	int32_t height_slice_factor,
	int32_t weight_d32_size,
	int32_t weight_d32_slices,
	int32_t size_per_line,
	int32_t height_total)
{
	int alines = (height_total)/height_slice_factor;
	int wslices = (weight_d32_slices)/weight_slice_factor;
	if ((weight_slice_factor > 1) && (wslices < 2)) return ~0U;
	if ((height_slice_factor > 1) && (alines < 4)) return ~0U;
	/* We will read the activations in for every chunk of weights,
	 * and read the weights in for every chunk of activations */
	return height_slice_factor*weight_d32_size*weight_d32_slices
		+ weight_slice_factor*size_per_line*height_total;
}

static int fill_info_minmax_basics(
	struct nn_graph *nn,
	struct nn_node *self,
	struct supernode_info_new *info)
{
	/* Pull out the inputs we need */
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	//const struct tensor *bias_min_tensor = self->inputs[8];
	//const struct tensor *bias_max_tensor = self->inputs[9];

	/* Get min/max values for input, weights, and bias data */
	float in_min_float = tensor_get_float(min_in_tensor,0);
	float in_max_float = tensor_get_float(max_in_tensor,0);

	if( !flt_isfinite(in_min_float) || ! flt_isfinite(in_max_float)){
		return errlog(nn,"input range to supernode, not finite");
	}
	in_max_float = fmaxf(in_max_float,in_min_float+0.00001f);
	//float filt_min_float = tensor_get_float(min_filt_tensor,0);
	//float filt_max_float = fmaxf(tensor_get_float(max_filt_tensor,0),filt_min_float+0.00001f);
	//float bias_min_float = tensor_get_float(bias_min_tensor,0);
	//float bias_max_float = tensor_get_float(bias_max_tensor,0);

	/* find zero offset, level size for input */
	float in_level_size;
	int32_t input_offset = get_qu8_level_size_zero(in_min_float,in_max_float, & in_level_size);

	// filter level (already compensated for any scaling done)
	float filt_level_size = info->weights_level_size;

	/* The product level size is the product of the input and filter level size */
	float prod_level_size = in_level_size * filt_level_size;

	//
	// final scaling is to multiply by prod_level_size / output_level_size.
	//
	float output_level_size;
	/*int32_t output_offset =*/ get_qu8_level_size_zero(info->out_minval,info->out_maxval, & output_level_size);
	// output level size, adjusted for weight scaling
	info->output_level_size = output_level_size;
	float max_k = info->max_k_factor;
	if( max_k == 0.0f){
		logmsg(nn,0,"max_k was 0.0!!");// shouldn't happen - maybe if some dwise/shortin not updated?
		max_k = 1.0f;
	}
	if( max_k < 1.0f || max_k > 2.02f){
		return errlog(nn,"bad maxk = %.7g", max_k);
	}
	// for cases with per-channel scaling, max_k sets the highest scale, and determines the
	// recip_shamt.

	float final_scaling = max_k * prod_level_size/output_level_size;

	if( info->outrange_firstguess){
		float max_scale = info->recip_shamt_must_be_zero ? 0.875f: 1.75f;
		if( final_scaling > max_scale ){ // do not want to go with a large scale on the first guess
			output_level_size = max_k * prod_level_size /max_scale;	// final scaling will be max_scale;
			int need_adjust = 0;
			switch(info->minmax_precalc_flags){
			 case 0:
			 default: // should not happen; out_range_firstguess not set unless 0,1 or 2.
				 info->out_minval = -63.0f * output_level_size;
				 info->out_maxval = 192.0f * output_level_size;
				 break;
			 case 1:		// only min is specified
				 info->out_maxval = info->out_minval + 255.0f * output_level_size;
				 need_adjust = (info->out_minval < 0.0f);
				 break;
			 case 2:		// only max is specified
				 info->out_minval = info->out_maxval - 255.0f * output_level_size;
				 need_adjust = 1;
				 break;
			}
			final_scaling = max_scale;
			if( need_adjust){
				// adjust zero and recalc all the things
				adjust_minmax_for_zero_with_constraints( & info->out_minval, & info->out_maxval, info->minmax_precalc_flags);
				output_level_size= flt_div_255( info->out_maxval - info->out_minval);
				final_scaling = max_k * prod_level_size/output_level_size;
			}
			info->output_level_size = output_level_size;
		}
		logmsg(nn,2,"first guess changed to %f .. %f scaling = %f\n", info->out_minval, info->out_maxval, final_scaling);
		info->outrange_firstguess = 0;
	}


	// if it's >1.0 we need recip_shift > 0
	int recip_shamt = (final_scaling <= 1.0f)? 0: flt_getexp(final_scaling);
	unsigned recip_val;
	float final_scaling_inv;

	if( recip_shamt > 0 && info->recip_shamt_must_be_zero ){
		// oops, node doesn't support gain >=1.0 ... We need to expand the output range
		// by a factor of 'final_scaling'; and then force scaling to 1.0 (or to
		// recip_val = 0x7fffffff, which is as close as we can get).
		// *important*: when info->recip_shamt_must_be_zero, caller must be
		// prepared for change in output range (bias should be found *after*)
		//
		// note: if we move one endpoint, and the other is fixed (and non-zero),
		// the adjust_minmax_for_zero_with_constraints may expand the range a bit more
		// and then we need to use recip_val smaller than 0x7fffffff; 'need_correction'
		// is set when this occurs.
		int need_correction = 0;
		if( final_scaling > 1.0f){ // because it could be exactly 1.0
			int flags = info->minmax_precalc_flags;
			if( flags== 3 ){
				logmsg(nn,0,"can't support this fixed output range; need gain=%f", final_scaling);
			}else{
				// TODO: Maybe this should be unified with the 'first_guess' adjustment
				if(flags == 0 ){	// unconstrained
					info->out_minval *= final_scaling;
					info->out_maxval *= final_scaling;
				}else {
					float new_range = (info->out_maxval-info->out_minval)*final_scaling;
					if( flags==1){		// min constrained, max not.
						info->out_minval = info->out_minval_spec;
						info->out_maxval = info->out_minval_spec + new_range;
					}else{ // max constrained, min not
						info->out_minval = info->out_maxval_spec - new_range;
						info->out_maxval = info->out_maxval_spec;
					}
					adjust_minmax_for_zero_with_constraints( & info->out_minval, & info->out_maxval, flags);
					float new_adj_range = info->out_maxval-info->out_minval;
					if( new_adj_range > new_range){
						// ok then, we need to adjust scale for the range we expanded because of the scale...
						final_scaling = new_range/new_adj_range;  // < 1.0
						need_correction = 1;
					}
				}
			}
		}
		recip_shamt  = 0;
		if( !need_correction){
			recip_val= 0x7FFFFFFF;
			final_scaling = final_scaling_inv = 1.0f;
		}else{
			recip_val = roundf_u32(  final_scaling * (float)(1u<<31));
			final_scaling_inv = 1.0f/final_scaling;
		}

	}else{
		// find final_scaling with 31-recip_shamt frac bits now.
		// Will be <= 0x7FFFFF80, except in border case where final_scaling = 1.0
		//This rounding will be lossless unless final_scaling < (1/128).
		recip_val = roundf_u32( flt_ldexp( final_scaling, (31-recip_shamt)));
		final_scaling_inv = output_level_size/(prod_level_size*max_k);
	}
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

    // all of that is valid for the cases where we don't support any per-channel
    // scaling. Now take care of the per-channel...
    // calculation is gain = k * prod_level_size/output_level_size, and scale with 32-recip_shamt
    // fractional bits.

    if( info->recip != NULL){
    	if(!info->has_channel_scale && !info->has_weight_scale){	// all the k are 1
    		// all k are 1; just use the single value we found already
    		memset_uint32( info->recip, recip_val, info->out_depth_valid);
    	}else{
    		float common_scale = flt_ldexp( prod_level_size/output_level_size, (31-recip_shamt));
    		int odv = info->out_depth_valid;

    		for(int i = 0; i < odv; i++){
    			unsigned rval = roundf_u32( common_scale * info->k_factor[i] );
    			info->recip[i] = (rval < 0x7fffffffu)? rval :0x7FFFFFFFu;
    		}
    	}
    }

	return 0;
}

/*
 * EJP: XXX: FIXME: We allocate enough for rounded up computation depth, but here we write full info->out_depth
 */
static int fill_bias_buf(
	struct nn_graph *nn,
	struct nn_node *self,
	struct supernode_info_new *info,
	int bias32,
	int32_t extra)
{
	const struct tensor *bias_tensor = self->inputs[7];
	const struct tensor *bias_min_tensor = self->inputs[8];
	const struct tensor *bias_max_tensor = self->inputs[9];
	float bias_min_float = tensor_get_float(bias_min_tensor,0);
	float bias_max_float = tensor_get_float(bias_max_tensor,0);
	int32_t bias_offset = bias32 ? 0 : quantize_uint(0.0f,bias_min_float,bias_max_float);
	float bias_denom = bias32 ? 0x1.0p32 : 255.0f;
	float bias_level_size = (bias_max_float - bias_min_float) / bias_denom;
	const uint8_t *bias8_ptr = bias_tensor->data;
	const int32_t *bias32_ptr = bias_tensor->data;
	float prod_level_size = info->prod_level_size;
	float bias_to_prod_ratio = (bias_level_size / prod_level_size);
	float min_out_prod_offset = -info->out_minval / prod_level_size;
	int32_t bias_depth = bias_tensor->shape.depth;
	int i;
	int32_t biasval;
	float bias_fval;
	float minout_bias_fval;
	int32_t gemsumb_val;
	int32_t final;
	logmsg(nn,3,"in_offset=%d bias_levelsize=%f prod_level_size=%f ratio=%f",info->in_offset,bias_level_size,info->prod_level_size,bias_to_prod_ratio);

	int per_chan_scaling = info->has_channel_scale || info->has_weight_scale;

	for (i = 0; i < info->out_depth_valid; i++) {
		if (i >= bias_depth) biasval = bias_offset;
		else if (bias32) biasval = bias32_ptr[i];
		else biasval = bias8_ptr[i];
		bias_fval = (biasval - bias_offset) * bias_to_prod_ratio;
		minout_bias_fval = bias_fval + min_out_prod_offset;
		if(per_chan_scaling){
			minout_bias_fval *= info->k_factor_recip[i];
		}
		gemsumb_val = info->gemsumb[i];
		final = -gemsumb_val * info->in_offset + fast_roundf(minout_bias_fval) + extra;
		logmsg(nn,3,"i=%d biasval%d=%d fval=%f minout_fval=%f gemsumb_val=%d extra=%d final=%d",
			i,bias32?32:8,biasval,bias_fval,minout_bias_fval,gemsumb_val,extra,final);
		info->biasbuf[i] = final;
	}
	return 0;
}


static void supernode_vector_kickoff(struct nn_graph *nn, void *vself)
{
	struct nn_node *self = vself;
	logmsg(nn,2,"vector kickoff!");
	supernode_execute_some_strategy(self,nn,0,1);
}

static int supernode_execute_strategy(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info_new *info = self->opaque;
	info->cycles = 0;
	info->minval = 1<<24;		// so we need the 'true' min output (when it's > 0)
	info->maxval = -(1<<24);
	if (0) {
		return supernode_execute_some_strategy(self,nn,0,info->n_work_items);
	} else {
		nn_sem_init(&info->alldone_sem,0);
		nn_os_work_for_vector(nn,supernode_vector_kickoff,self);
		nn_sem_wait(&info->alldone_sem);
		supernode_execute_workitem_check_for_retry(NULL,self,nn);
		return 0;
	}
}

static inline int supernode_strategy_valid(
	struct nn_node *self,
	struct nn_graph *nn,
	struct supernode_info_new *info)
{
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	if (info->needs_retry) return 0;
	if (!info->strategy_valid) return 0;
	if (tensor_get_float(in_min_tensor,0) != info->in_min_float) return 0;
	if (tensor_get_float(in_max_tensor,0) != info->in_max_float) return 0;
	if (nn->vtcm_ptr != info->prepared_vtcm_addr) return 0;

	if( ! shape_matches( &info->in_shape, &self->inputs[0]->shape)){
		return 0;
	}
	/*
	 * FIXME: check input max/min/shape
	 */
	return 1;
}

// check if 'channelscale' present on input 13.
// if so, the *fp is set to point to the floats; if not, it's set to NULL.
// An input with a single value of 1.0 is considered the same as 'absent'.
// The size of the dimension is checked (error return if it doesn't match).
// (this is intended to be used by shortin and depthwise, if & when they support ChannelScale)
static int __attribute__((unused))
check_channelscale_present(struct nn_graph *nn, struct nn_node *self, int filt_batches, float const **fp){

	*fp = NULL;
	if( self->n_inputs >= 13){	// looks like we do...
		const struct tensor *cscale_tensor = self->inputs[12];
		int n = cscale_tensor->data_size/sizeof(float);
		if( n == 1 && tensor_get_float(cscale_tensor,0) == 1.0f){
			// we don't really have channel-scaling; ignore a 1.0 input
		}else{
			if( n != filt_batches){
				return errlog(nn,"expected size %d vector for channel_scale, got %d", (int)filt_batches, n );
			}
			*fp = (float const*) cscale_tensor->data;
		}
	}
	return 0;
}


//
// load the channel scales from float tensor into k_factor
// As coded this allow values in range 1/32 .. 1.0.
// any 'padded' values are set to  of 1.0
// it also sets info->has_channel_scale = 1
//
// if channel_scale_flts is NULL, it sets info->has_channelscale  0 and does nothing.
//
// (this is intended to be used by shortin and depthwise, if & when they support ChannelScale)
//

static int __attribute__((unused))
load_channel_scales(struct nn_graph *nn,struct supernode_info_new *info,
		float const * channel_scale_flts, int out_depth)
{
	if( channel_scale_flts == NULL){
		info->has_channel_scale = 0;
		return 0;
	}

	float * outp = info->k_factor;
	info->has_channel_scale = 1;

	int out_depth_roundup = (out_depth + 31) & ~31;
	for( int i =0; i < out_depth; i++){
		float scval = channel_scale_flts[i];
		if( !( scval <= 1.0f && scval >= (float)(1./32.))){
			return errlog(nn,"bad channel scale[%d]= %.8f",i,scval);
		}
		outp[i] = scval;
	}
	if( out_depth_roundup > out_depth){
		memset_float( &outp[out_depth], 1.0f, out_depth_roundup - out_depth );
	}
	return 0;
}

//
// This fills in info->k_factor and info->k_factor_recip
// (this is intended to be used by shortin and depthwise, if & when they support ChannelScale)
//
//
// Requires:
//   (1) if info->has_channel_scale = 1, the channels scales are loaded
//       into info->k_factor (result from calling load_channel_scales)
//   (2) if info->has_weight_scale, the weight scales have been
//       loaded into info->k_factor_recip (but as fixed-point numbers
//        with 32 fractional bits, not as floats). Otherwise no assumption
//        is made about contents of k_factor_recip.
//
//  The arrays are filled as follows:
//     k_factor[i] = chanscale[i]/weight_scale[i]
//     k_factor_recip[i] = weight_scale[i]/channel_scale[i];
//  Also mak_k_factor is set to the largest k_factor encountered
//  (or 1.0, if all are < 1).
//
static int __attribute__((unused))
find_k_kinv(struct nn_graph *nn, struct supernode_info_new *info, int filt_batches_roundup)
{
	int has_channel_scale = info->has_channel_scale;
	int has_weight_scale= info->has_weight_scale;
	float * k_wrp = info->k_factor;				// output pointer
	float * kinv_wrp = info->k_factor_recip;
	if( !has_weight_scale){
		if( !has_channel_scale){
			memset_float( k_wrp, 1.0f, filt_batches_roundup );
			memset_float( kinv_wrp, 1.0f, filt_batches_roundup );
		}else{
			// channel scales (all <=1.0) but no weight scale
			// Just leave the k alone, find reciprocals -> k_factor_recip.
			float const *chanscale_rdp = info->k_factor;	// read channel scales
			for( int i =0; i < filt_batches_roundup; i++){
				float chsc = chanscale_rdp[i];
				kinv_wrp[i] = (chsc == 1.0f)? 1.0f : ( 1.0f/chsc);
			}
		}
		info->max_k_factor = 1.0f;
		return 0;
	}
	float const *chanscale_rdp = info->k_factor;	// read channel scales
	int32_t const *wscale_rp = (int32_t const*) info->k_factor_recip;
	float max_k = 1.0f;
	for( int i = 0; i < filt_batches_roundup; i++){
		float cscale = has_channel_scale? chanscale_rdp[i]: 1.0f;
		float wsf = (float)wscale_rp[i] * ( float)( 1.0/ (1u<<31));	//
		float k = cscale;
		float kinv = wsf;
		if( cscale!=1.0f) kinv = wsf/cscale;
		if( wsf != 1.0f) k = cscale/wsf;
		max_k = fmaxf( max_k, k);
		k_wrp[i] = k;
		kinv_wrp[i] = kinv;
	}
	info->max_k_factor = max_k;
	return 0;
}

// this sets up:
//   info->out_minval, info->out_minval_spec, and info->minval_precalculated
// .. and the same for 'maxval'.
//  It also sets minmax_precalc_flags which is both flags in one value.
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
static void
setup_initial_output_range( struct supernode_info_new *info,
	float specified_minval,		// range specified by inputs
	float specified_maxval,
	float minval_default,			// use when specified_minval = -INF
	float maxval_default )			// use when specified_maxval = INF
{
	// enforce sanity:  min <= 0.0 <= max
	// and max > min + 1/128
	//
	specified_minval = fminf( specified_minval, 0.0f);
	specified_maxval = fmaxf( fmaxf( specified_maxval, 0.f),
					 specified_minval + 0x1.0p-7f);

	info->out_minval_spec = specified_minval;
	info->out_maxval_spec = specified_maxval;

	int mnp = (specified_minval == -INFINITY)?0:1;		// is min precalc
	int mxp = (specified_maxval == INFINITY)?0:1;		// is max precalc

	info->out_minval = mnp ? specified_minval : minval_default;
	info->out_maxval = mxp ? specified_maxval : maxval_default;

	info->minval_precalculated = mnp;
	info->maxval_precalculated = mxp;

	int corr_code = info->minmax_precalc_flags = 2*mxp + mnp;
	// unless both min and max are specified, this is considered a 'first guess'
	info->outrange_firstguess = corr_code !=3 ;

	// corr_code:
	//    bit 0 -> out_min is 'fixed'
	//    bit 1 -> out_max is 'fixed';
	// only need if minval != 0
	//
	if( info->out_minval < 0.0f ){
		adjust_minmax_for_zero_with_constraints( &info->out_minval, &info->out_maxval, corr_code);
	}
}


static void dwise_supernode_execute_conv_work(struct nn_graph *nn, void *vinfo)
{
    struct workitem *work = vinfo;
    struct nn_node *self = work->self;
    struct supernode_info_new *info = self->opaque;

    int32_t start_line = work->start_line;
    int32_t stop_line = work->stop_line;
    int32_t in_next_row = info->in_next_row;
    int32_t in_next_d32 = info->in_next_d32;
    int32_t out_next_row = info->out_next_row;
    int32_t depth = info->out_depth_valid;
    int32_t out_width = info->out_width;
    int32_t filt_height = info->filt_height;
    int32_t stride_height = info->stride_height;

    const uint8_t *input = work->input + start_line*stride_height*in_next_row;
    uint8_t *output = work->output + start_line*out_next_row;
    const uint8_t *weights = (const uint8_t *)work->weights;
    const int32_t *biasbuf = work->biases;

    int32_t recip_val = info->recip_val;
    int32_t recip_shamt = info->recip_shamt;
    int32_t filt_offset = info->filt_offset;

    uint64_t start_cycles;
    uint64_t my_cycles;
    union {
    HVX_Vector vec[2];
        int32_t words[64];
    } minmax;
    HVX_Vector scratch_buf[MAX_FILT_HEIGHT+1];

    start_cycles = nn_os_get_cycles(nn);

    logmsg(nn,1,"DWSUPER: input=%p weights=%p output=%p in_next_row=%d out_next_row=%d in_next_d32=%d "
        "out_next_d32=%d out_depth=%d out_width=%d n_lines=%d filt_height=%d minmax_buf=%p "
        "recip_val=0x%x biasbuf=%p stride_height=%d recip_shamt=%d in_left_skip=%d filt_offset=%d",
        input,weights,output,in_next_row,out_next_row,in_next_d32,info->out_next_d32,
        depth,out_width,
        stop_line-start_line,filt_height,minmax.words,
        recip_val,biasbuf,stride_height,recip_shamt,
        info->in_left_skip,filt_offset);

    minmax.vec[1] = Q6_V_vsplat_R(0x7FFFFFFF);
    minmax.vec[0] = Q6_V_vnot_V(minmax.vec[1]);
    int32_t  pf_offset = Q6_R_max_RR(filt_height-stride_height, 0);

    if ((info->stride_width != 1)&& (info->stride_width != 2))  {
        errlog(nn,"sorry, horizontal stride currently only 1 or 2... %d",info->stride_width);
	    goto done;
    }
    int out_row; 

    for(out_row = start_line; out_row < stop_line; out_row++) {
        wait_for_l2fetch(); 

        if (out_row < (stop_line-1)) {
            l2fetch_v(input+(stride_height+pf_offset)*in_next_row, in_next_row, in_next_row, filt_height-pf_offset);
        }
        (*info->dwfunc)(
                input,
                weights,
                output,
                info->in_next_row,
                info->out_next_row,
                info->in_next_d32,
                info->out_next_d32,
                depth,
                info->out_width,
                1,                  //n_lines,
                info->filt_height,
                info->filt_offset,
                biasbuf,
                minmax.words,
                info->recip_val,
                info->recip_shamt,			//correct 32bit mpy ,can be shift of less 
                info->stride_height,
                scratch_buf); 

        input += in_next_row*stride_height;
        output+= out_next_row;
    }
    gvrmaxmin(minmax.words);
    my_cycles = nn_os_get_cycles(nn) - start_cycles;
    nn_atomic_min(&info->minval,minmax.words[32]);
    nn_atomic_max(&info->maxval,minmax.words[ 0]);
	nn_atomic_add64(&info->cycles,my_cycles);
	logmsg(nn,2,"min=%d(%d) max=%d(%d) cycles=%lld",minmax.words[32],info->minval,minmax.words[0],info->maxval,my_cycles);
done:
	nn_checkpoint_arrival(nn,self,&info->alldone_checkpoint);
}

static void dwise_supernode_execute_hvx_work(struct nn_graph *nn, void * vinfo)
{
    struct workitem *work = vinfo;
    struct nn_node *self = work->self;
    struct supernode_info_new *info = self->opaque;

    // initial prefetch 
    l2fetch_v( work->input + work->start_line*info->stride_height*info->in_next_row,
               info->in_next_row, info->in_next_row, info->filt_height );

    dwise_supernode_execute_conv_work(nn, work);
}

#if 0
/*
  push the work on the vector queue
 */
//int dwise_supernode_execute_workitem_vector_dispatch(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
static void dwise_supernode_execute_workitem_vector_dispatch(struct nn_graph *nn, void *vinfo)
{
	struct workitem *work = vinfo;
        nn_os_work_for_vector(nn, dwise_supernode_execute_hvx_work, work);
        return 0;
}
#endif

/*
  shuffle the weight so they are serialized and pad 3taps to 4
 */ 
static float dwise_rearrange_weights(
	uint8_t *out_weights,
	const uint8_t *in_weights,
	int32_t filt_height,
	int32_t filt_width,
	int32_t filt_depth,
	int32_t filt_depth_roundup,
	int32_t depth_multiplier,
	int zero_val)
{
	const int32_t in_filt_width = filt_width;
	const int32_t out_filt_width = (filt_width + 3)&(~3);
        int b,h,w,v,od,id,in_idx,out_idx,val;

        for (b = 0; b < depth_multiplier; b++) {
            for (od = 0; od < filt_depth_roundup; od += 32) {
                for (id = 0; id < 32; id++) {
                    for (h = 0; h < filt_height; h++) {
                        for (w = 0; w < out_filt_width; w+=4) {
                            for (v = 0; v < 4; v++) {
                                if((w+v) < in_filt_width) {
                                        in_idx = h*in_filt_width*filt_depth*depth_multiplier
                                               + (w + v)*filt_depth*depth_multiplier
                                               + (od+id)*depth_multiplier
                                               + b;
                                        if ((od+id) < filt_depth) val = in_weights[in_idx] ;
                                        else val = zero_val;
                                } else {
                                        val = 0;
                                }
                                out_idx = b*filt_height*out_filt_width*filt_depth
                                        + od*filt_height*out_filt_width
                                        + (h*out_filt_width+w)*32
                                        + id*4 + v;
                                out_weights[out_idx] = val;
                            }
                        }
                    }
                }
            }
        }
        return(1.f);
}

/*
 * perform the sum of weights for each output depth position and subtract constant 
 */
void dwise_sumb(
        int32_t *filt_sum,
        uint8_t *out_weights,
        int32_t filt_height,
        int32_t filt_width,
        int32_t filt_depth,
        int32_t out_depth,
        int32_t depth_multiplier)
{
        const int32_t out_filt_width = (filt_width+3)&(~3);
        int b,h,w,v,od,id,out_idx;
        int32_t sum;

        for (b = 0; b < depth_multiplier; b++) {
                for (od = 0; od < out_depth; od += 32) {
                        for (id = 0; id < 32; id++) {
                                sum = 0;
                                for (h = 0; h < filt_height; h++) {
                                        for (w = 0; w < out_filt_width; w+=4) {
                                             out_idx = b*filt_height*out_filt_width*filt_depth
                                                       + od*filt_height*out_filt_width
                                                       + (h*out_filt_width+w)*32
                                                       + id*4 ;
                                             for(v=0; v < 4; v++) {
                                                sum += out_weights[out_idx+v] ;
                                             }
                                        }
                               }
                               filt_sum[b*out_depth+od+id] = sum;
                        }
                }
        }
        return;
}


/*
  compare how the actual max and min compaes with the preidcted max and min if too small increase
  it until it fits. 
 */
static int __attribute__((unused))
dwise_supernode_execute_workitem_check_for_retry(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{

	/*
	 * - info->minval, info->maxval are intermediate values;
  	 * Conversion to 'application' values is:
	 *  x_app = x_intermed * info->prod_level_size  + info->out_minval
	 */

        struct supernode_info_new *info = node->opaque;

        int needs_retry = 0;
        int fixed_flags = info->minmax_precalc_flags;				// for adjust_minmax_for_zero_with_constraints
        float ominval = info->out_minval;	// for interpreting minval/maxval


        if ((fixed_flags&2)==0) {	// max not fixed
            logmsg(nn,1,"in check for retry max %d ", info->maxval);
            float app_maxval = info->maxval * info->prod_level_size + ominval;

            if (app_maxval > info->out_maxval) {
            	// need to increase it
            	info->out_maxval = round_up_quarter_octave( fmaxf(app_maxval, 0x1.0p-4f));
        		logmsg(nn,2,"maxval retry out_minval=%f out_maxval=%f maxval=%d (%f).",
        				info->out_minval,info->out_maxval,info->maxval,app_maxval);
                needs_retry = 1;
            }
        }
        if ((fixed_flags&1)==0) { // min not fixed
        	if (info->minval < 0) {		// need to move min
        		float app_minval = info->minval * info->prod_level_size + ominval;
            	info->out_minval = round_up_quarter_octave( fminf(app_minval, -0x1.0p-8f));
        		logmsg(nn,2,"minval retry out_minval=%f out_maxval=%f minval=%d (%f).",
        				info->out_minval,info->out_maxval,info->minval,app_minval);
                needs_retry = 1;
        	}
        }
        // correct the endpoints for a proper zero
        if (needs_retry){
    		// restore 'fixed' endpoints to original spec (in case they were tweaked by previous endpoint corrs)
    		if (fixed_flags & 1) info->out_minval = info->out_minval_spec;
    		if (fixed_flags & 2) info->out_maxval = info->out_maxval_spec;
        	adjust_minmax_for_zero_with_constraints( &info->out_minval, &info->out_maxval, fixed_flags);
        	info->needs_retry = 1;
        }

        logmsg(nn,2,"Checking workitem, maxval=%x minval=%x new range %f .. %f needs_retry=%d",
        		info->maxval,info->minval, info->out_minval, info->out_maxval, info->needs_retry);
        return 0;
}

/*
   generate the strategy of thow the dwise conv is peroftrmed generating the schedule to be replayed
 */
static int dwise_supernode_recalculate_strategy(struct nn_node *self, struct nn_graph *nn) //, void *vinfo)
{
	/* Pad Zap */
	//struct nn_node *self = vinfo;
	struct supernode_info_new *info = self->opaque;
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	//const struct tensor *min_in_tensor = self->inputs[2];
	//const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];

	info->in_shape = in_tensor->shape;

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_left_pad = in_tensor->format.width_pad[0];
	int32_t in_right_pad = in_tensor->format.width_pad[1];
	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];
	int32_t in_top_pad = in_tensor->format.height_pad[0];
	int32_t in_bottom_pad = in_tensor->format.height_pad[1];

	//int32_t depth_multiplier = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	//int32_t filt_depth = filt_tensor->shape.filt_depth;
	//int32_t filt_depth_roundup = ((filt_depth + 31) & ~31);

	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	/* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */
	int32_t required_w_before, required_h_before, required_h_after;

	int32_t out_width = nn_pad_compute_outsize_and_padbefore(in_width,filt_width,stride_width,self->padding, &required_w_before);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height,filt_height,stride_height,self->padding, &required_h_before, &required_h_after );

	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;

	int32_t input_batch_size = in_height_total * in_width_total * in_depth_total;

	int32_t out_batches = in_batches;
	int32_t out_depth = in_depth_total*info->depth_multiplier;


	int32_t out_left_pad; //poss pad 1,3,4
	if(self->padding == NN_PAD_VALID) out_left_pad = in_left_pad/stride_width;	/* dwise onv pads same for VALID */	/* FIXME: adjust for stride */
	else out_left_pad = (in_left_pad - required_w_before)/stride_width; /* dwise 3x3 conv moves over for SAME*/
	logmsg(nn,1,"in left pad=%d required_w_before=%d stride_width=%d out_left_pad=%d",in_left_pad,required_w_before,stride_width,out_left_pad);
	int32_t out_right_pad = (-(out_width + out_left_pad)) & 3;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = in_depth_before_pad;
	int32_t out_depth_after_pad = in_depth_after_pad;

	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	//int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;

	/*
	 * Set up work items
	 */
	struct workitem work = {0};
	//struct workitem waitwork = work;
	struct workitem startwork = work;

	info->minval = 0;
	info->maxval = 0;

	info->in_width = in_width_total;
	info->in_depth = in_depth_total;
	info->in_next_d32 = in_width_total * 32;
	info->in_next_row = in_width_total * in_depth_total;

	info->out_width = out_width_total;
	info->out_depth_total = out_depth_total;
	info->out_depth_valid = out_depth_total;
	info->out_height = out_height;
	info->out_next_d32 = out_width_total * 32;
	info->out_next_row = out_width_total * out_depth_total;

	info->stride_height = stride_height;
	info->stride_width = stride_width;
	info->filt_height = filt_height;
	info->filt_width = filt_width;
        if(info->stride_width == 1) {
            if(info->filt_width == 5) {
              info->dwfunc = &dwconv2dbbb_s1_5xN_asm;
            } else if(info->filt_width == 3) {
              info->dwfunc = &dwconv2dbbb_s1_3xN_asm;
            } else errlog(nn,"sorry, filter width currently only 3 or 5... is %d", info->filt_width);
        } else if(info->stride_width == 2) {
            if(info->filt_width == 5) {
              info->dwfunc = &dwconv2dbbb_s2_5xN_asm;
            } else if(info->filt_width == 3) {
              info->dwfunc = &dwconv2dbbb_s2_3xN_asm;
            } else errlog(nn,"sorry, filter width currently only 3 or 5... is %d", info->filt_width);
        } else errlog(nn,"sorry, horizontal stride currently only 1 or 2... is %d",info->stride_width);

        logmsg(nn,2," strides = (%d,%d)",stride_height,stride_width);
	// find input range, output scaling and limits
	// Note: may expand the output range

	if( fill_info_minmax_basics(nn,self,info) !=0 ) return -1;
	logmsg(nn,1,"out_maxval=%f out_minval=%f in_max_float=%f in_min_float=%f in_level_size=%f filt_level_size=%f prod_level_size=%f max_valid_val=%d",
			info->out_maxval,info->out_minval,info->in_max_float,info->in_min_float,info->prod_level_size/info->weights_level_size,
			info->weights_level_size,info->prod_level_size,info->max_valid_val);

	supernode_softreset_work_items(self,nn,info);

        int ib, ob;

	int bias32 = (self->node_type == OP_DepthwiseSupernode_8x8p32to8_d32);
	fill_bias_buf(nn,self,info,bias32,0);

	logmsg(nn,1,"max_valid_val=%x recip_shamt=%d recip_val=%x",info->max_valid_val,info->recip_shamt,info->recip_val);

	logmsg(nn,2,"Out tensor: %d x %d|%d|%d x %d|%d|%d x %d|%d|%d",
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad);
	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height,out_top_pad,out_bottom_pad,
		out_width,out_left_pad,out_right_pad,
		out_depth,out_depth_before_pad,out_depth_after_pad,
		NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"output tensor prep fail (%p).  data_size(%d)>max_size(%d)",
		       out_tensor, out_tensor->data_size, out_tensor->max_size);
	}

	info->in_left_skip = in_left_pad - (out_left_pad * stride_width + required_w_before);

	if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
		return errlog(nn,"max out prep fail");
	}
	tensor_set_float(out_min,0,info->out_minval);
	tensor_set_float(out_max,0,info->out_maxval);

	int32_t batchstart_idx = 0;
	info->startup_info.batch_start_offset = 0;

	int32_t inner_batches;
	int32_t outer_batches;
	int32_t weights_fit;
#ifdef V66
	weights_fit = ((info->weight_batch_size) <= nn->vtcm_size);
#else
	weights_fit = 1;
#endif
	if (likely(weights_fit)) {
		outer_batches = 1;
		inner_batches = in_batches;
	} else {
		outer_batches = in_batches;
		inner_batches = 1;
	}

	info->in_height = in_height + required_h_before + required_h_after;
	info->weights_base = info->weights;
	info->in_next_batch = input_batch_size;
  for (ob = 0; ob < outer_batches; ob++) {
	//info->input_base = in + (in_top_pad - required_h_before)*info->in_next_row;
	//info->input_base = tensor_location_bhw_d32(in_tensor,b, -required_h_before,-in_left_pad);
	work.weights = info->weights;

	/*
	 * Preparing the work list
	 * We create a work list, which is just a list of items for the main thread to do
	 * We need to put in all the things we need to do to execute the node:
	 * * Padding zapping
	 * * Actual convolution
	 * * l2fetch / memcpy of values
	 * * Waiting on semaphores for things to finish
	 * The work list is currently executed by the main thread
	 * This means that vector work needs to be passed to nn_os_work_for_vector
	 */

	//waitwork.execute = supernode_execute_workitem_join_some;

	/* add_batch_startup will add the offsets appropriately, so here always use the base pointer */
	startwork.info = info;
	startwork.self = self;

	startwork.zap_left = tensor_location_bhw_d32(in_tensor,0,-required_h_before,-in_left_pad);
	startwork.zap_right = startwork.zap_left + (in_left_pad + in_width)*32;
	startwork.zap_left_size = in_left_pad;
	startwork.zap_right_size = in_right_pad;
	startwork.zap_top = tensor_location_bhw_d32(in_tensor,0,-required_h_before,-in_left_pad);
	startwork.zap_top_size = info->in_next_row * required_h_before;
	startwork.zap_bot = tensor_location_bhw_d32(in_tensor,0,in_height,-in_left_pad); 
	startwork.zap_bot_size = info->in_next_row * (required_h_after+1); //add extra row along bottom for corner case
	startwork.zap_rl_depths = in_depth_total / 32;
	startwork.zap_height = required_h_before+in_height+required_h_after;
	startwork.zap_value = info->in_offset;

	logmsg(nn,1,"dwise supernode zapping pad");

#ifdef V66
	if (weights_fit) {
		startwork.copy_in = info->weights_base;
		startwork.copy_out = nn->vtcm_ptr;
		startwork.copy_size = info->weight_batch_size * info->n_weight_batches;
		work.weights = nn->vtcm_ptr;
	} else {
		startwork.pf_inp = info->weights;
		startwork.pf_width = startwork.pf_stride = info->weight_batch_size * info->n_weight_batches;
		startwork.pf_height = 1;
	}
#else
	startwork.pf_inp = info->weights;
	startwork.pf_width = startwork.pf_stride = info->weight_batch_size * info->n_weight_batches;
	startwork.pf_height = 1;
#endif

	//supernode_add_padding_zap(self,nn,info,zapwork,0,required_h_before,required_w_before);

	work.info = info;
	work.self = self;
	//work.execute = dwise_supernode_execute_workitem_vector_dispatch; 
	work.new_execute = dwise_supernode_execute_hvx_work;
	work.biases = info->biasbuf;
	//work.weights = supernode_filtbuf_location(nn,info,0,info->weights,info->weight_batch_size*(info->out_depth/32));

	int32_t inner_act_rows = (out_height + NUM_THREADS - 1)/NUM_THREADS;
	int32_t outer_act_iters = (out_height + inner_act_rows - 1)/ inner_act_rows;
	int or;

	/* If we have a large # of batches, make each work item a larger piece of work */
	if (inner_batches > NUM_THREADS*4) {
		inner_act_rows = out_height;
		outer_act_iters = 1;
	}

	info->startup_info.wakeup_items = outer_act_iters*inner_batches;
	startwork.join_iters = outer_act_iters*inner_batches;
	if (ob > 0) info->work_items[batchstart_idx].next_startup_offset = info->n_work_items - batchstart_idx - 1;
	if (ob > 0) logmsg(nn,2,"next startup offset @ %d == %d",batchstart_idx,info->work_items[batchstart_idx].next_startup_offset);
	int NOW_BATCHES=inner_batches;

	batchstart_idx = supernode_add_batch_startup(self,nn,info,startwork,ob*inner_batches*input_batch_size,required_h_before,required_w_before,NOW_BATCHES);
	logmsg(nn,2,"Adding batch startup: off=%d batchstart_idx=%d",ob*inner_batches*input_batch_size,batchstart_idx);
     for (ib = 0; ib < inner_batches; ib++) {
        int b = ob*inner_batches+ib;
	for(or = 0; or < outer_act_iters; or++) {
		int start_row = or * inner_act_rows;
		int n_rows = Q6_R_min_RR(out_height-start_row,inner_act_rows);

		work.start_line = start_row;
		work.stop_line = start_row + n_rows;

		work.donesem = NULL;
		work.output = tensor_location_bhw_d32(out_tensor,b,0,0);
		work.input = tensor_location_bhw_d32(in_tensor,b,-required_h_before,-required_w_before);

		logmsg(nn,2,"Adding work item: start_row=%d",start_row);
		supernode_add_work_item(self,nn,info,work);
	}
     }
  } // batch iter
	//work.execute = dwise_supernode_execute_workitem_check_for_retry;
	//supernode_add_work_item(self,nn,info,work);
	/*
	 * We've calculated the strategy, mark that the work is done. Hopefully it sticks!
	 */
	supernode_compile_worklist(nn,info,self);
	info->needs_retry = 0;
	info->strategy_valid = 1;
	return 0;
}

/*
  do some checks and execute the schedule
 */
static int dwise_supernode_execute(struct nn_node *self, struct nn_graph *nn)
{
	/* Check 3x3, non expanding */
        struct supernode_info_new *nodeinfo = self->opaque;
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *in_tensor = self->inputs[0];
	//int32_t depth_multiplier = filt_tensor->shape.filt_batches;
	//int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	int32_t filt_depth_roundup = (filt_depth + 31) & ~31;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t in_depth_before_pad = in_tensor->format.depth_pad[0];
	int32_t in_depth_after_pad = in_tensor->format.depth_pad[1];
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_left_pad = in_tensor->format.width_pad[0];

	if (filt_width > 5) return errlog(nn,"only filt width > 5 depthwise conv supported for now...");
	if (in_depth_total != filt_depth_roundup) return errlog(nn,"filter depth must match input depth (%d != %d)",in_depth_total,filt_depth_roundup);
	if (in_depth_total < 32) return errlog(nn,"(padded) input depth must be >= 32 (%d)",in_depth_total);
	if (nodeinfo->depth_multiplier != 1) logmsg(nn,1,">1 depth expansion supported but format is not in stadard order - needs shuffling");
	if (in_left_pad < 1) return errlog(nn,"Need at least 1 left pad");//EJP for SAME, valid needs no pad

        struct tensor *out = self->outputs[0];
        struct tensor *out_min = self->outputs[1];
        struct tensor *out_max = self->outputs[2];
        unsigned long long int total_time;
        int loopcount = 0;
        while(1){
			if (likely(supernode_strategy_valid(self,nn,nodeinfo))) {
					if (supernode_execute_strategy(self,nn) != 0) {
							return errlog(nn,"execute strategy failed");
					}
			} else {
					if (dwise_supernode_recalculate_strategy(self,nn) != 0) {
							return errlog(nn,"recalc strategy failed");
					}
					if (supernode_execute_strategy(self,nn) != 0) {
							return errlog(nn,"execute strategy fail after recalc");
					}
			}
			/* Replay if self-calculated min/max are insufficient */
			if (!nodeinfo->needs_retry)
				break;
			if( ++loopcount > 4){
				return errlog(nn,"can't find range for depthwise");
			}
        }
        tensor_set_float(out_min,0,nodeinfo->out_minval);
        tensor_set_float(out_max,0,nodeinfo->out_maxval);
        /* Record cycles (divide by # of vector worker threads somehow?) */
        total_time = nodeinfo->cycles;
        record_usertime(nn,self,NN_GRAPH_PERFEVENT_USER0,total_time);

        logmsg(nn,2,"out tensor info: bhwd=%d,%d,%d,%d paddings=(%d,%d)x(%d,%d)x(%d,%d)",
                out->shape.batches,out->shape.height,out->shape.width,out->shape.depth,
                out->format.height_pad[0],out->format.height_pad[1],
                out->format.width_pad[0],out->format.width_pad[1],
                out->format.depth_pad[0],out->format.depth_pad[1]);

	logmsg(nn,2,"dwise supernode done executing work");
	return 0;
}

/* 
   at prepare time, alocate the memory and set up the  dwise part of theis graph
 */
static int dwise_supernode_check(struct nn_node *self, struct nn_graph *nn)
{
	// ctor checks that n_inputs = 12 or 13 (13th is ChannelScale)
	// and n_outputs = 3
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_filt_tensor = self->inputs[4];
	const struct tensor *max_filt_tensor = self->inputs[5];
	int32_t depth_multiplier = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_width_roundup = (filt_width + 3) &(~3);
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	int32_t filt_depth_roundup = ((filt_depth + 31) & ~31);
	int32_t out_depth = depth_multiplier * filt_depth_roundup;
	int weights_size = filt_height * filt_width_roundup * depth_multiplier * filt_depth_roundup;
	uint8_t *filt = filt_tensor->data;
	float filt_max_float = tensor_get_float(max_filt_tensor,0);
	float filt_min_float = tensor_get_float(min_filt_tensor,0);
	int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
        float specified_minval = tensor_get_float(self->inputs[10],0);
        float specified_maxval = tensor_get_float(self->inputs[11],0);

	struct supernode_info_new *info;
	float weights_scale;
	logmsg(nn,2,"weights: (%d,%d,%d,%d-->%d)",depth_multiplier,filt_height,filt_width,filt_depth,filt_depth_roundup);
	logmsg(nn,2,"weights_size: %d filt_depth: %d depth_mpy: %d out_depth: %d",weights_size,filt_depth,depth_multiplier,out_depth);
	/* Fill out info->weights */
	if (filt_width > 5) return errlog(nn,"Oops: implement depthwise support for filt width > 5");
	if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
		return errlog(nn,"calloc");
	}
	info->is_dwise = 1;

	if ((info->weights = nn_memalign(128,weights_size)) == NULL) {
		nn_free(info);
		return errlog(nn,"memalign");
	}
	info->weight_batch_size = weights_size;
	info->n_weight_batches = 1;
	if ((info->biasbuf = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		nn_free(info->weights);
		nn_free(info);
		return errlog(nn,"memalign");
	}
	if ((info->gemsumb = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
                nn_free(info->biasbuf);
                nn_free(info->weights);
		nn_free(info);
		return errlog(nn,"memalign");
	}

	self->opaque = info;
	info->filt_offset = filt_offset;
	/* Rearrange weights */
	//logmsg(nn,1,"rearrange weights %p to %p [hdb=%d,%d,%d]",filt,info->weights,filt_height,filt_depth,filt_batches);
	weights_scale = dwise_rearrange_weights(info->weights,
                                                filt,filt_height,filt_width,
                                                filt_depth,filt_depth_roundup,
                                                depth_multiplier,filt_offset);
	info->weights_offset = filt_offset;
	// NOTE: currently (3/7/19) dwise_convert_weights_to_signed does nothing and returns 1.0

	info->weights_level_size =  (filt_max_float - filt_min_float) / (255.0f * weights_scale);
 	logmsg(nn,1,"weights_scale=%f  weights_level_size=%f",weights_scale,info->weights_level_size);

        dwise_sumb(info->gemsumb,
                   info->weights,
                   filt_height,
                   filt_width,
                   filt_depth,
                   filt_depth_roundup,
                   depth_multiplier);

	info->max_k_factor = 1.0f;
	info->strategy_valid = 0;
        info->depth_multiplier = depth_multiplier;
	setup_initial_output_range( info, specified_minval, specified_maxval, -0.125f, 0.125f);

	logmsg(nn,1,"during prepare: out_minval=%f out_maxval=%f",info->out_minval,info->out_maxval);
	return 0;
}

/*
   tear down this node when we are done
 */

static int dwise_supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
	struct supernode_info_new *info = self->opaque;
	if (info) {
		supernode_reset_work_items(self,nn,info);
		nn_free(info->gemsumb);
		nn_free(info->biasbuf);
		nn_free(info->weights);
		nn_free(info);
	}
	self->opaque = NULL;
	return node_free_common(self,nn);
}

/*
  define the depthwise node, setup, execute, teardown
 */ 

struct nn_node_ops nn_ops_for_DepthwiseSupernode_8x8p8to8_d32_unused = {
	.execute = dwise_supernode_execute,
	.check = dwise_supernode_check,
	.ctor = node_alloc_common,
	.dtor = dwise_supernode_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};

struct nn_node_ops nn_ops_for_DepthwiseSupernode_8x8p32to8_d32_unused = {
	.execute = dwise_supernode_execute,
	.check = dwise_supernode_check,
	.ctor = node_alloc_common,
	.dtor = dwise_supernode_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
/* --------------------- end of depthwise stuff ---------------------  */
