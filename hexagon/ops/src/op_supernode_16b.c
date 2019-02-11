
/*
 * Copyright (c) 2016-2018, The Linux Foundation. All rights reserved.
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
 * This contains the code for convolution
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#ifdef __hexagon__
#include "hexagon_types.h"
#else
#include <malloc.h>
#endif
#include "hvx_hexagon_protos.h"
#include "nn_bufferpool.h"
#define DLO 2

#define NUM_THREADS 2

#define ENABLE_FASTSUMA_1x1 0

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

struct workitem {
	int(*execute)(struct workitem *, struct nn_node *node, struct nn_graph *nn);	// exec function
	void(*new_execute)(struct nn_graph *nn, void *opaque); // work_for_vector-compatible exec function, passed work item
	struct nn_node *self;		// This node
	struct sn16b_info *info;	// same as self->opaque
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
	uint16_t zap_value;	// value to zap with
	int32_t zap_jobs;
	int32_t zap_oedelta; // e & o plane

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

typedef struct sn16b_info {
	uint8_t *weights;	// weights, padded and adjusted as necessary
	int32_t *biasbuf;	// int32 bias buffer, including min offsets and gemsumb
	int32_t *minmax_buf;	// pointer to min/max values, enough storage per thread...
	nn_sem_t *semaphores;	// pointer to preallocated array of semaphores
	struct workitem *work_items;	// All the work items to execute at execute time
	nn_os_workitem_t *work_list;	// compiled work list
	int n_work_items;		// how many work items?
	int batch_start_idx;		// pointer during more-distributed execution
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
	const uint8_t *suma_in;	// input pointer to start SUMA work... should be in workitem...
	int32_t suma_width;	// elements of a SUMA row
	int32_t next_suma_off;	// bytes of a SUMA row
	int32_t *suma_buf;	// GEMSUMA (if needed)
	int32_t suma_start;	// where to start in suma buffer
	int32_t integral_off;   //index into integral buffer used by gvsuma
	int32_t recip_val;	// reciprocal for product space --> output space
	int32_t recip_shamt;	// amount to shift before recip mpy
	int32_t circ_buf_size;  //size pf the circular buffer used in v65 conv
	int32_t num_accs;       // number of accumulators used in main computation
	int in_offset;		// amount to normalize inputs by.  Needed?
	int filt_offset;	// amount to normalize filter values by. Needed?
	int32_t recursion_depth;// how far have we recursed?
	const uint8_t *input_base0;	// first row (including all left padding, in-use top padding)
	const uint8_t *input_base;	// first row (including in-use left padding, in-use top padding).
	const uint8_t *weights_base;
	const uint8_t * raw_input; //ptr to the input tensor for use when copied into temp
	int32_t max_valid_val;	// maximum value that results in a value not above max_out
	int32_t min_valid_val;	// minimum value that results in a value not below min_out
	float prod_level_size;	// in_level_size * filt_level_size
	int32_t *gemsumb;	// GEMSUMB value, if we want to calculate it at preparation time
	int32_t use_v65;	// Should we use V65 mode?
	int32_t use_v66;	// Should we use V66 mode?
	int32_t use_vtcm;       //flag to use vtcm or not for weights
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

	/// old stuff
	//int16_t *weights;		// weights, padded and adjusted as necessary
	//int32_t *gemsumb;		// sum of weights along input depth filters
	int32_t *filt_corr;		// sum of lo weights for correction factor
	//int32_t *biasbuf;		// int32 bias buffer, including min offsets and gemsumb
	//int32_t minmax_buf[2];	// pointer to min/max values, enough storage per thread...
	//int32_t *gemsumb;	// GEMSUMB value, if we want to calculate it at preparation time
	//float out_minval;	// Minimum output value, either specified or guessed
	//float out_maxval;	// maximum output value, either specified or guessed
	//int minval_precalculated;	// Is the minval precalculated?
	//int maxval_precalculated;	// Is the maxval precalculated?
	//float out_minval_spec;		// exact value specified (when not precalculated)
	//float out_maxval_spec;		// exact value specified (when not precalculated)

	//const uint8_t *suma_in;	// input pointer to start SUMA work... should be in workitem...
	//int32_t suma_width;	// elements of a SUMA row
	//int32_t next_suma_off;	// bytes of a SUMA row
	//int32_t *suma_buf;	// GEMSUMA (if needed)
	//int32_t suma_start;	// where to start in suma buffer
	//int32_t integral_off;   //index into integral buffer used by gvsuma

	//int32_t minval;			// Minimum value (in prod space) actually observed
	//int32_t maxval;			// Maximum value (in prod space) actually observed
	//int32_t use_vtcm;       //flag to use vtcm or not for weights

	int32_t use_2planes;	// high and low byte in different planes
	int32_t use_usmodel;	// use unsigned (activation) * signed(weight) model, up to 1 bit loss in percision
	int32_t is_u16;
	float filt_level_size;
	float bias_level_size;
	//int32_t filt_offset;
	int32_t bias_offset;
	//int32_t in_offset;

	struct nn_node *self;
	nn_sem_t donesem;

	int fsplit;
	uint8_t * din_e;
	uint8_t * din_o;
	uint8_t * dout_e;
	uint8_t * dout_o;
	size_t in_oedelta;
	size_t out_oedelta;
} sn16b_info;


void gvint_16b(
	const uint8_t * in_bufe,     //input activations - aligned to 8bytes
	const uint8_t * in_bufo,     //input activations - aligned to 8bytes
	int32_t *out,
	int32_t next_d32,
	int32_t next_row,
	int32_t integral_width,
	int32_t in_depth,
	int32_t out_height,
	int32_t *scratch_128xW
)
{
	int in_height = out_height;
	const uint8_t *ine = in_bufe;
	const uint8_t *ino = in_bufo;
	int *out_sum = (int *)out;
	int *tmp_buf = (int *)scratch_128xW;
	int i, j, k, l;
	int64_t sum, sumo;

	for (j = 0; j < in_height; j++)
	{
		memset(tmp_buf, 0, sizeof(int) * 8);
		for (i = 0; i < integral_width - 8; i++)
		{
			sum = 0;
			for (l = 0; l < in_depth / 32; l++) {
				for (k = 0; k < 32; k++) {
					uint8_t inlb = ine[j*next_row + l * next_d32 + 32 * i + k];
					uint8_t inhb = ino[j*next_row + l * next_d32 + 32 * i + k];
					sum += inhb * 256 + inlb;
				}
			}
			tmp_buf[i + 8] = (int)sum;
		}

		sum = 0;
		for (i = 0; i < integral_width; i++)
		{
			sum += tmp_buf[i];
			if (j == 0) sumo = sum; else sumo = sum + out_sum[(j - 1)*integral_width + i];
			out_sum[integral_width*j + i] = (int)sumo;
		}
	}
}

void gvsuma_16b(
	const int32_t *integral,
	int32_t       *suma_out,
	int32_t        in_width,
	int32_t        next_output_width,
	int32_t        stride_height,
	int32_t        filt_width,
	int32_t        filt_height,
	int32_t        out_height,
	uint32_t       filt_offset)
{
	int i, j, sum;

	for (j = 0; j < out_height; j++) {
		for (i = 0; i < in_width; i++) {
			sum = integral[j*stride_height              *in_width + i] +
				integral[(j*stride_height + filt_height)*in_width + i + filt_width] -
				integral[(j*stride_height + filt_height)*in_width + i] -
				integral[j*stride_height              *in_width + i + filt_width];
			suma_out[j*next_output_width / 4 + i] = - (((int64_t)sum*filt_offset)>>16);
		}
	}
}
void gvconv2dbbb_16b_i(
	const int16_t *input,
	const int16_t *filt,
	int16_t *output,
	int32_t in_width,
	int32_t out_width,
	int32_t out_next_row,
	int32_t stride_height_width,
	int32_t in_depth,
	int32_t filt_width,
	int32_t filt_height,
	int32_t out_height,
	const int32_t *bias,
	int64_t *minmax,
	uint32_t recip_val
)
{
	assert(DLO == 2);
#define DN 32
#define LOG2DN 5
	int32_t stride_width = stride_height_width & 0x0ffff;
	int32_t stride_height = stride_height_width >> 16;
	int32_t in_next_row = in_width * in_depth;
	int32_t filt_x, filt_y;
	int32_t h, w;

#define WBLOCK 2
	int next_outputs = filt_height * (in_depth >> 5) * in_width * 32 - stride_width * 32 * WBLOCK;

	HVX_Vector sBias = *(HVX_Vector *)bias;
	HVX_Vector sZero = Q6_V_vzero();
	HVX_Vector s0ffff = Q6_V_vsplat_R(0x0000ffff);
	HVX_Vector sRecip = Q6_V_vsplat_R(recip_val);
	HVX_Vector sMaxval = sZero;
	HVX_Vector sMinval = sZero;
	HVX_Vector sSum0L, sSum0H, sSum1L, sSum1H;
	for (h = 0; h < out_height; h++) {
		const int16_t * ptr_x0 = (const int16_t *)(input + h * in_next_row * stride_height);
		HVX_Vector * ptr_z = (HVX_Vector *)(output + h * out_next_row);
		for (w = 0; w < out_width; w+= WBLOCK) {
			sSum0L = sSum1L = Q6_V_vand_VV(sBias, s0ffff);
			sSum0H = sSum1H = Q6_Vw_vasr_VwR(sBias, 16);

			HVX_Vector *pFilt = (HVX_Vector *)filt;
			for (filt_y = 0; filt_y < filt_height * (in_depth >> 5); filt_y++) {
				const uint64_t * ptr_x1 = (const uint64_t *)ptr_x0;
				ptr_x0 += in_width*32;		  // move down rows ...
				for (filt_x = 0; filt_x < filt_width*8; filt_x++) {
					HVX_Vector sW0 = *pFilt++;
					HVX_Vector sW1 = *pFilt++;

					int64_t x13x10 = ptr_x1[stride_width*32/(sizeof(int64_t)/sizeof(int16_t))];
					HVX_Vector sM110 = Q6_Vw_vdmpy_VhRh_sat(sW0, (int32_t)x13x10);
					HVX_Vector sM132 = Q6_Vw_vdmpy_VhRh_sat(sW1, x13x10>>32);
					HVX_Vector sM110L = Q6_V_vand_VV(sM110, s0ffff);
					HVX_Vector sM110H = Q6_Vw_vasr_VwR(sM110, 16);
					HVX_Vector sM132L = Q6_V_vand_VV(sM132, s0ffff);
					HVX_Vector sM132H = Q6_Vw_vasr_VwR(sM132, 16);
					HVX_Vector sM130L = Q6_Vw_vadd_VwVw(sM132L, sM110L);
					HVX_Vector sM130H = Q6_Vw_vadd_VwVw(sM132H, sM110H);
					sSum1L = Q6_Vw_vadd_VwVw(sSum1L, sM130L);
					sSum1H = Q6_Vw_vadd_VwVw(sSum1H, sM130H);

					int64_t x03x00 = *ptr_x1++;
					HVX_Vector sM010 = Q6_Vw_vdmpy_VhRh_sat(sW0, (int32_t)x03x00);
					HVX_Vector sM032 = Q6_Vw_vdmpy_VhRh_sat(sW1, x03x00>>32);
					HVX_Vector sM010L = Q6_V_vand_VV(sM010, s0ffff);
					HVX_Vector sM010H = Q6_Vw_vasr_VwR(sM010, 16);
					HVX_Vector sM032L = Q6_V_vand_VV(sM032, s0ffff);
					HVX_Vector sM032H = Q6_Vw_vasr_VwR(sM032, 16);
					HVX_Vector sM030L = Q6_Vw_vadd_VwVw(sM032L, sM010L);
					HVX_Vector sM030H = Q6_Vw_vadd_VwVw(sM032H, sM010H);
					sSum0L = Q6_Vw_vadd_VwVw(sSum0L, sM030L);
					sSum0H = Q6_Vw_vadd_VwVw(sSum0H, sM030H);
				}
			}
			ptr_x0 -= next_outputs;

			HVX_Vector sSum0Lt = Q6_Vuw_vlsr_VuwR(sSum0L, 16);
			HVX_Vector sSum0 = Q6_Vw_vadd_VwVw(sSum0Lt, sSum0H);
			sMaxval = Q6_Vw_vmax_VwVw(sMaxval, sSum0);
			sMinval = Q6_Vw_vmin_VwVw(sMinval, sSum0);
			HVX_Vector sRes0t = Q6_Vw_vmpye_VwVuh(sSum0, sRecip);
			HVX_Vector sRes0 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sRes0t, sSum0, sRecip);

			HVX_Vector sSum1Lt = Q6_Vuw_vlsr_VuwR(sSum1L, 16);
			HVX_Vector sSum1 = Q6_Vw_vadd_VwVw(sSum1Lt, sSum1H);
			sMaxval = Q6_Vw_vmax_VwVw(sMaxval, sSum1);
			sMinval = Q6_Vw_vmin_VwVw(sMinval, sSum1);
			HVX_Vector sRes1t = Q6_Vw_vmpye_VwVuh(sSum1, sRecip);
			HVX_Vector sRes1 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sRes1t, sSum1, sRecip);

			*ptr_z++ = Q6_Vh_vpack_VwVw_sat(sRes1, sRes0);
		}
	}
	for (int i = 0; i < 5; i++) {
		HVX_Vector sMaxvalt = Q6_V_valign_VVR(sZero, sMaxval, (i + 1) * 4);
		HVX_Vector sMinvalt = Q6_V_valign_VVR(sZero, sMinval, (i + 1) * 4);
		sMaxval = Q6_Vw_vmax_VwVw(sMaxval, sMaxvalt);
		sMinval = Q6_Vw_vmin_VwVw(sMinval, sMinvalt);
	}
	int64_t curmaxval = ((int64_t)*(int32_t*)&sMaxval)<<16;
	int64_t curminval = ((int64_t)*(int32_t*)&sMinval)<<16;
	if (curminval < minmax[1]) minmax[1] = curminval;
	if (curmaxval > minmax[0]) minmax[0] = curmaxval;
}

/*
   GVCONV2DB2B2B2
	   - generalized vector convolution - 2dimensional -hword * hword => hword
	   - meax and min of the accumulations are tracked.

   Notes
   -----
   Does multiple of 4 outputs in parallel. This code takes the pointer to where the valid
   data begins.

   Weights are reformed by transpoing so are sequential in memory.

   <------------------------------------------ filt_height = 3 * indepth/32 --------------->
   <------------------------- filt width = 3 ------------------------------------><---><--->
   <---- 32 depth 4 each----><---- 32 depth 4 each----><---- 32 depth 4 each-----><><><><><>

   Data is stored in distributed form

   <----------------4 wide line depth 32 format ---------->
   <-32 lo-byte-><-32 lo-byte-><-32 lo-byte-><-32 lo-byte->
   <-32 hi-byte-><-32 hi-byte-><-32 hi-byte-><-32 hi-byte->

   This c code is bit accurate to the assembly and uses same inputs.
*/
void gvconv2db2b2b2_d32_asm(
	uint8_t * in_bufe,     //input activations - aligned to 8bytes
	uint8_t * in_bufo,     //input activations - aligned to 8bytes
	int8_t * weights,     //8bit unsigned weights - aligned to 128bytes
	uint8_t * out_bufe,    //quantized output -
	uint8_t * out_bufo,    //quantized output -
	int next_in_width,    //total physical width in depths
	int next_out_width,   //total size of full line out_depth* total out_width
	int out_width,       //amount of work to do
	int stride_h_w,      //striding y | x
	int in_depth,        //input depth total
	int filt_width,      //2d filter
	int filt_height,     //-sizes
	int out_height,      //number of lines to compute
	const int32_t * bias_add,  //activation bias for accumulators
	int32_t * ptr_minmax,//ptr for tracking min and max
	int32_t recip,       //1/max for quantization
	int recip_shift,
	int out_align,
	int skip_col
)
#if defined(V66)
;
#else
{
	int out_y, out_x, out_x4, i, h;
	int index_x, index_y, write;
	int stride_height = 0xffff & (stride_h_w >> 16);
	int stride_width = 0xffff & (stride_h_w >> 0);
	int64_t lsum;
	int num_depth32 = in_depth / 32;
	int64_t outval;
	int32_t max = -0x7fffffff, min = 0x7fffffff;
	int out_z, in_z, filt_y, filt_x, filt_z;
	int8_t w0, w1;
	uint8_t d0, d1, y0, y1;
	uint8_t * obufe_ptr, *obufo_ptr;
	int32_t sum, sums;
	int64_t s11;
	uint8_t bufe[256], bufo[256];

	//recip_shift  = 31- (recip_shift-16);
	int8_t * weights0 = weights;                                      //even bytes
	int8_t * weights1 = weights + filt_width * filt_height*in_depth * 32; //odd bytes

	for (out_y = 0; out_y < out_height; out_y++)
	{
		if (out_align == 0) write = 1; else write = 0;
		obufe_ptr = out_bufe + next_out_width * out_y;
		obufo_ptr = out_bufo + next_out_width * out_y;
		for (out_x4 = 0; out_x4 < out_width; out_x4 += 4) //will overrun into out of sample data
		{
			for (h = 0; h < 4; h++) {
				out_x = out_x4 + h;
				for (out_z = 0; out_z < 32; out_z++)
				{
					s11 = 0x80;
					//2d filter
					for (filt_y = 0; filt_y < filt_height*num_depth32; filt_y++)
					{
						for (filt_x = 0; filt_x < filt_width; filt_x++)
						{
							//perform the filtering across 4 accumulators and long the depth
							for (in_z = 0; in_z < 32; in_z++)
							{
								//addressing scheme for transposed weight array
								filt_z = 32 * (filt_width*filt_y + filt_x) + in_z;
								w0 = weights0[128 * (filt_z / 4) + filt_z % 4 + 4 * out_z];
								//printf("s10=%02x\n",w0);

								index_x = stride_width * out_x + filt_x;
								index_y = stride_height * out_y*num_depth32 + filt_y;
								d1 = in_bufo[32 * (next_in_width*(index_y)+index_x) + in_z];

								s11 += (d1 * w0); //hi * lo
							}//in_z
						}//filt x
					}//filt_y
					//2d filter
					for (filt_y = 0; filt_y < filt_height*num_depth32; filt_y++)
					{
						for (filt_x = 0; filt_x < filt_width; filt_x++)
						{
							//perform the filtering across 4 accumulators and long the depth
							for (in_z = 0; in_z < 32; in_z++)
							{
								//addressing scheme for transposed weight array
								filt_z = 32 * (filt_width*filt_y + filt_x) + in_z;
								w1 = weights1[128 * (filt_z / 4) + filt_z % 4 + 4 * out_z];

								index_x = stride_width * out_x + filt_x;
								index_y = stride_height * out_y*num_depth32 + filt_y;
								d0 = in_bufe[32 * (next_in_width*(index_y)+index_x) + in_z];

								s11 += (d0 * w1); //lo * hi
							}//in_z
						}//filt x
					}//filt_y
					s11 >>= 8;
					s11 += bias_add[out_z];

					//2d filter
					for (filt_y = 0; filt_y < filt_height*num_depth32; filt_y++)
					{
						for (filt_x = 0; filt_x < filt_width; filt_x++)
						{
							//perform the filtering across 4 accumulators and long the depth
							for (in_z = 0; in_z < 32; in_z++)
							{
								//addressing scheme for transposed weight array
								filt_z = 32 * (filt_width*filt_y + filt_x) + in_z;
								w1 = weights1[128 * (filt_z / 4) + filt_z % 4 + 4 * out_z];

								index_x = stride_width * out_x + filt_x;
								index_y = stride_height * out_y*num_depth32 + filt_y;
								d1 = in_bufo[32 * (next_in_width*(index_y)+index_x) + in_z];

								s11 += (d1 * w1); //hi * hi
							}//in_z
						}//filt x
					}//filt_y
					//printf(" %16llx \n", s11);

					//monitor max and min and quantize the accumulations
					sum = (int32_t)s11;
					if (sum > max) max = sum;
					if (sum < min) min = sum;
					sums = sum << recip_shift;
					lsum = (int64_t)sums * (int64_t)recip + 0x40000000LL;
					outval = lsum >> 31; //39;
					//printf("%16llx -> %08lx -> %08lx -> %16llx\n",s11,sum,sums,outval);

					if (outval < 0) outval = 0;
					if (outval > 65535) outval = 65535;
					y0 = (uint8_t)(0xff & (outval >> 0));
					y1 = (uint8_t)(0xff & (outval >> 8));

					bufe[128 + (32 * out_x + out_z) % 128] = y0;
					bufo[128 + (32 * out_x + out_z) % 128] = y1;
				}//out_z
			}//end h
			if (write == 1) for (i = 0; i < 128; i++) {
				*obufe_ptr++ = bufe[128 - out_align + i];
				*obufo_ptr++ = bufo[128 - out_align + i];
			}
			write = 1;
			for (i = 0; i < 128; i++) {
				bufe[i] = bufe[128 + i];
				bufo[i] = bufo[128 + i];
			}
		}//out_x4
		if (skip_col) for (i = 0; i < 128; i++) {
			*obufe_ptr++ = bufe[(128 - out_align + i) % 256];
			*obufo_ptr++ = bufo[(128 - out_align + i) % 256];
		}
	}//out_y
	for (i = 0; i < 32; i++) {
		ptr_minmax[i] = max_i32(ptr_minmax[i],((int32_t)max));
		ptr_minmax[32 + i] = min_i32(ptr_minmax[32 + i], ((int32_t)min));
	}
	return;
}
#endif
void gvconv2db2b2b2u_d32_asm(
	const uint8_t * in_bufe,     //input activations - aligned to 8bytes
	const uint8_t * in_bufo,     //input activations - aligned to 8bytes
	uint8_t * out_bufe,    //quantized output - 
	uint8_t * out_bufo,    //quantized output - 
	const uint8_t * weights,     //8bit unsigned weights - aligned to 128bytes
	int in_width,    //total physical width in depths
	int next_out_width,   //total size of full line out_depth* total out_width
	int out_width,       //amount of work to do 
	int stride_h_w,      //striding y | x
	int in_depth,        //input depth total
	int filt_width,      //2d filter
	int filt_height,     //-sizes
	int out_height,      //number of lines to compute
	const int32_t * biasbuf,  //activation bias for accumulators
	const int32_t *suma,
	int32_t next_suma_row,
	int32_t * ptr_minmax,//ptr for tracking min and max
	int32_t recip,       //1/max for quantization
	int recip_shift,
	int out_align,
	int skip_col
);

/*
  convert between 16bit depth 32 format to distributed 16bit format
 */

void d32_16_to_88_cn(
	uint8_t * ine8_ptr,
	uint8_t * ino8_ptr,
	uint16_t * in16_ptr,
	int width,
	int height,
	int depth
) {
	int i, j, k;
	uint16_t d0;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			for (k = 0; k < depth; k++) {
				d0 = in16_ptr[i*depth*width + (k / 32) * 32 * width + 32 * j + (k % 32)];
				ine8_ptr[i*depth*width + (k / 32) * 32 * width + 32 * j + (k % 32)] = (d0 >> 0) & 0x00ff;
				ino8_ptr[i*depth*width + (k / 32) * 32 * width + 32 * j + (k % 32)] = (d0 >> 8) & 0x00ff;
			}
		}
	}
	return;
}

void d32_88_to_16_cn(
	uint16_t * in16_ptr,
	uint8_t * ine8_ptr,
	uint8_t * ino8_ptr,
	int width,
	int height,
	int depth
) {
	int i, j, k;
	uint16_t dlo, dhi;

	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			for (k = 0; k < depth; k++) {
				dlo = ine8_ptr[i*depth*width + (k / 32) * 32 * width + 32 * j + (k % 32)];
				dhi = ino8_ptr[i*depth*width + (k / 32) * 32 * width + 32 * j + (k % 32)];
				in16_ptr[i*depth*width + (k / 32) * 32 * width + 32 * j + (k % 32)] = (dhi << 8) | (0x00ff & dlo);
			}
		}
	}
	return;
}


// Find y, the smallest power of 2 such that abs(y) >= abs(x)
// (and y having the same sign as x).
// x should be !=0 and not denormal.
//
static inline float
to_next_power_of_two(float x)
{
	// round the 32-bit code up to the next value in which the mantissa is all zero.
	union {
		float f;
		uint32_t u32;
	} uu = { x };
	uint32_t m_mask = (1 << 23) - 1;
	uu.u32 = (uu.u32 + m_mask) & ~m_mask;
	return uu.f;
}

/*
  zaps the border of the activations
 */

void zap_d32(
	uint8_t* in_data_d32_pad, int in_height, int in_width, int in_depth, int in_offset,
	int pad_left, int pad_right, int pad_top, int pad_bot)
{
	int in_y, k;
	int in_width_pad = pad_left + in_width + pad_right;
	uint8_t *iptr;

	iptr = in_data_d32_pad - pad_top * in_depth*in_width_pad;
	for (in_y = 0; in_y < pad_top; in_y++)
		for (k = 0; k < in_depth / 32; k++) {
			vmemset_asm(iptr, in_offset, 32 * in_width_pad);
			iptr += 32 * in_width_pad;
		}
	for (in_y = 0; in_y < in_height; in_y++)
	{
		for (k = 0; k < in_depth / 32; k++)
		{
			vmemset_asm(iptr, in_offset, 32 * pad_left);
			iptr += 32 * (pad_left + in_width);
			if (pad_right > 0)vmemset_asm(iptr, in_offset, 32 * pad_right);
			iptr += 32 * pad_right;
		}
	}
	for (in_y = 0; in_y < pad_bot; in_y++)
		for (k = 0; k < in_depth / 32; k++) {
			vmemset_asm(iptr, in_offset, 32 * in_width_pad);
			iptr += 32 * in_width_pad;
		}
	return;
}

static int supernode_16b_execute(struct nn_graph *nn, void *tdata)
{
	struct nn_node *self = tdata;
	sn16b_info *info = self->opaque;
	struct tensor *tensor_out = self->outputs[0];
	struct tensor *tensor_out_min = self->outputs[1];
	struct tensor *tensor_out_max = self->outputs[2];
	const struct tensor *tensor_in = self->inputs[0];
	const struct tensor *tensor_filt = self->inputs[1];
	const struct tensor *tensor_in_min = self->inputs[2];
	const struct tensor *tensor_in_max = self->inputs[3];
	//const struct tensor *tensor_flt_min = self->inputs[4];
	//const struct tensor *tensor_flt_max = self->inputs[5];
	const struct tensor *tensor_stride = self->inputs[6];
	const struct tensor *tensor_bias = self->inputs[7];
	//const struct tensor *tensor_bias_min = self->inputs[8];
	//const struct tensor *tensor_bias_max = self->inputs[9];
	// TODO: check if max val is specified before searching
	//float specified_minval = tensor_get_float(self->inputs[10], 0);
	//float specified_maxval = tensor_get_float(self->inputs[11], 0);

	/*
	 * Find the dimensions of the input data,
	 * both dimensions of data as well as padding
	 */
	int32_t in_batches = tensor_in->shape.batches;
	int32_t in_width = tensor_in->shape.width;
	int32_t in_height = tensor_in->shape.height;
	int32_t in_depth = tensor_in->shape.depth;
	int32_t in_left_pad = tensor_in->format.width_pad[0];
	int32_t in_right_pad = tensor_in->format.width_pad[1];
	int32_t in_depth_before_pad = tensor_in->format.depth_pad[0];
	int32_t in_depth_after_pad = tensor_in->format.depth_pad[1];
	int32_t in_top_pad = tensor_in->format.height_pad[0];
	int32_t in_bottom_pad = tensor_in->format.height_pad[1];
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;

	/* Find the dimensions of the filter weights.  filt_batches == out_depth */
	int32_t filt_batches = tensor_filt->shape.filt_batches;
	int32_t filt_height = tensor_filt->shape.filt_height;
	int32_t filt_width = tensor_filt->shape.filt_width;

	/* Find the stride values */
	int32_t stride_width = tensor_stride->shape.width;
	int32_t stride_height = tensor_stride->shape.height;

	/* Calculate output dimensions */
	int32_t out_batches = in_batches;
	int32_t out_depth = filt_batches;

	/* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */
	int32_t required_w_before, required_w_after;
	int32_t required_h_before, required_h_after;
	int32_t out_width = nn_pad_compute_outsize_and_pad(in_width, filt_width, stride_width, self->padding,
		&required_w_before, &required_w_after);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height, filt_height, stride_height, self->padding,
		&required_h_before, &required_h_after);

	/*
	 * Set output padding values to sensible defaults.
	 */
	int32_t out_right_pad = ((-out_width) & 3);
	int32_t out_left_pad = 4;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;

	// allocate the quantized tensor
	if (tensor_out_prepare_padded_d32(tensor_out,
		out_batches,
		out_height, out_top_pad, out_bottom_pad,
		out_width, out_left_pad, out_right_pad,
		out_depth, out_depth_before_pad, out_depth_after_pad,
		NN_TYPE_QINT16) != 0) {
		errlog(nn, "failure preparing output, max_size=%d bhwd=%d,%d,%d,%d",
			tensor_out->max_size, out_batches, out_height, out_width, out_depth);
		goto error_sn16b;
	}


	// compute the bias
	float filt_level_size = info->filt_level_size;
	float bias_level_size = info->bias_level_size;
	float in_max_float = tensor_get_float(tensor_in_max, 0);
	float in_min_float = tensor_get_float(tensor_in_min, 0);
	int32_t in_off = fast_roundf(-in_min_float * 65535.0f / (in_max_float - in_min_float));
	info->in_offset = in_off;
	float in_level_size = (-in_min_float) * (float)(1./ 32768.0);
	if (info->is_u16) {
		in_level_size = (in_max_float - in_min_float) / 65535.0f;
	}
	float prod_level_size = in_level_size * filt_level_size;

	float bias_mpy_amt = (bias_level_size / prod_level_size);

	int16_t *bias = (int16_t *)tensor_bias->data;
	int32_t bias_offset = info->bias_offset;
	int64_t min_out_prod_offset = -info->out_minval / prod_level_size; // TODO: saturate?
	if (!((min_out_prod_offset >> 32) == 0 || (min_out_prod_offset >> 32) == -1)) {
		logmsg(nn, 0, "error overflow???");
	}

	float out_max_val = 0.0, out_min_val = 0.0;
	if (!info->use_2planes) {
	for (int i = 0; i < out_depth; i++) {
			int32_t biasval = fast_roundf((bias[i] - bias_offset) * bias_mpy_amt);
			if (info->is_u16) biasval = fast_roundf(((uint16_t)bias[i] - bias_offset) * bias_mpy_amt);
			info->biasbuf[i] = biasval;
	}

	// conv, find max/min
	int block_height = (out_height + 2)/ 3;
	int out_next_row = out_width_total*out_depth_total;
		int16_t *filt = (int16_t *)info->weights;
	int64_t minmax[2] = {1LL<<63, 0x7fffffffffffffff};
	uint64_t recip_val = 0;
	for (int h = 0; h < out_height; h+=block_height) {
		int16_t *input = tensor_location_bhw_16b_d32(tensor_in, 0, h*stride_height, 0);
		int16_t *output = tensor_location_bhw_16b_d32(tensor_out, 0, h, 0);
		for (int w = 0; w < in_depth; w+=32) {
			gvconv2dbbb_16b_i(
				input,
				filt+w*filt_width*filt_height*in_depth,
				output+w*out_width_total,
				in_width_total,
				out_width,
				out_next_row,
				Q6_R_combine_RlRl(stride_height, stride_width),
				in_depth,
				filt_width,
				filt_height,
				Q6_R_min_RR(block_height, out_height-h),
				(const int32_t *)info->biasbuf+w,
				minmax,
				recip_val
			);
		}
	}

		out_max_val = to_next_power_of_two(minmax[0] * prod_level_size);
		out_min_val = to_next_power_of_two(minmax[1] * prod_level_size);
		adjust_minmax_for_zero_16b(&out_min_val, &out_max_val);
		info->out_maxval = out_max_val;
		info->out_minval = out_min_val;
		uint64_t maxval = Q6_P_max_PP(out_max_val, -out_min_val) / prod_level_size;
		if (info->is_u16) {
			maxval = (out_max_val - out_min_val) / prod_level_size;
		}
		min_out_prod_offset = -info->out_minval / prod_level_size;

		recip_val = ((1ULL << 62) + (maxval / 2)) / (maxval);
		if (info->is_u16) recip_val = 0x7FFF800000000000ULL / maxval;  //65535 << (31+16)

		if (recip_val >> 32) {
			assert(0);
			errlog(nn, "supernode_16b need to scale the accum!!!");
			goto error_sn16b;
		}

	// conv with scaling
	minmax[0] = 1LL << 63;
	minmax[1] = 0x7fffffffffffffff;
	for (int h = 0; h < out_height; h+=block_height) {
		int16_t *input = tensor_location_bhw_16b_d32(tensor_in, 0, h*stride_height, 0);
		int16_t *output = tensor_location_bhw_16b_d32(tensor_out, 0, h, 0);
		for (int w = 0; w < out_depth; w += 32) {
			gvconv2dbbb_16b_i(
				input,
				filt+w*filt_width*filt_height*in_depth,
				output+w*out_width_total,
				in_width_total,
				out_width,
				out_next_row,
				Q6_R_combine_RlRl(stride_height, stride_width),
				in_depth,
				filt_width,
				filt_height,
				Q6_R_min_RR(block_height, out_height-h),
				(const int32_t *)info->biasbuf+w,
				minmax,
				recip_val
			);
		}
	}
	}
	else {
		// when performing the suma, vector width is at least 24. even with bottom padding (4) is not enough
		int32_t more_byte_suma = in_width_total < 24 ? 24 * in_depth_total : 0;
		uint8_t * din_e = nn_memalign(128, in_width_total*in_depth_total*in_height_total + more_byte_suma);
		uint8_t * din_o = nn_memalign(128, in_width_total*in_depth_total*in_height_total + more_byte_suma);
		uint8_t * dout_e = nn_memalign(128, out_width_total*out_depth_total*out_height_total);
		uint8_t * dout_o = nn_memalign(128, out_width_total*out_depth_total*out_height_total);
		int32_t * sumabuf = NULL;
		int32_t * scratch_128xW = NULL;

		for (int i = 0; i < out_depth_total; i++) {
			int64_t fsum = /*info->filt_corr[i]*/ -(int64_t)in_off * info->gemsumb[i];
			fsum += min_out_prod_offset;
			if (i < out_depth)
				fsum += (int64_t)(((uint16_t)bias[i] - bias_offset) * bias_mpy_amt+0.5f);
			if (info->use_usmodel == 0)
				fsum += (int64_t)filt_height*filt_width*in_depth_total*info->filt_offset*info->in_offset;
			info->biasbuf[i] = (fsum >> 16);
		}

		uint8_t *input_e = din_e + in_top_pad * in_depth_total*in_width_total;
		uint8_t *input_o = din_o + in_top_pad * in_depth_total*in_width_total;
		int block_height = (out_height + 2) / 3;
		int out_next_row = out_width_total * out_depth_total;
		d32_16_to_88_cn(din_e, din_o, tensor_in->data, in_width_total, in_height_total, in_depth_total);

		zap_d32(input_e, in_height, in_width, in_depth_total, (in_off >> 0) & 0xff,
			in_left_pad, in_right_pad, in_top_pad, in_bottom_pad);
		zap_d32(input_o, in_height, in_width, in_depth_total, (in_off >> 8) & 0xff,
			in_left_pad, in_right_pad, in_top_pad, in_bottom_pad);

		const int8_t *filt = (const int8_t *)info->weights;
		union {
			HVX_Vector vec[2];
			int32_t words[64];
		} minmax;
		minmax.vec[1] = Q6_V_vsplat_R(0x7FFFFFFF);
		minmax.vec[0] = Q6_V_vnot_V(minmax.vec[1]);

		int32_t recip_val = 0;
		int32_t recip_shift = 46;

		int i, out_left_junk, skip_col, output_align, out_width_amt, input_align;

		if (in_left_pad == 4 && required_w_before == 0)
		{
			out_left_junk = 0;
			input_align = 4;
			out_width_amt = out_width + out_right_pad;
		}
		else {
			out_left_junk = (in_left_pad - required_w_before) / stride_width;
			input_align = 0;
			out_width_amt = out_width + out_right_pad + 4;
		}
		output_align = (4 - out_left_junk) & 3;
		input_align += (in_left_pad - required_w_before) & (stride_width - 1);
		skip_col = (((out_width & 3) <= output_align) && (out_width & 3) != 0) ? 1 : 0;
		if (skip_col) out_width_amt -= 4;

		input_e = din_e + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * input_align;
		input_o = din_o + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * input_align;

		uint8_t *output_e = dout_e + out_next_row * out_top_pad + 32 * out_left_pad;
		uint8_t *output_o = dout_o + out_next_row * out_top_pad + 32 * out_left_pad;

		int suma_buf_rowstride = 0;
		int32_t required_w_total = filt_width + (out_width - 1)*stride_width;
		if (info->use_usmodel == 0) {
			// each row of the integral buffer will have (in i32 slots):
			//   (a) 8 zeros
			//   (b) one slot for each 'unused' left-padding element (if any); 0..3
			//   (c) one slot for each input element which is used by the convolution
			//       including req. left & right (i.e. required_w_total )
			//   (d) padding up to a multiple of 32.
			// The 'suma_start' points to the first element before section (c)
			// 'suma buf' has the same width, and the the first 'proper' output
			// will be at position 'suma_start'.
			//
			info->suma_start = 7 + ((in_left_pad - required_w_before) & 3);
			suma_buf_rowstride = (info->suma_start + 1 + required_w_total + (32-1)) & -32;
			info->suma_width = suma_buf_rowstride;
			// this is 4 rows for 'scratch_128xW' plus enough rows for the integral buffer
			int scratch_rows = 4 + (block_height - 1)*stride_height + filt_height + 1;
			int scratch_size = suma_buf_rowstride * scratch_rows + 32;

			uint32_t sumatmp_size = scratch_size * sizeof(int32_t);
			nn_scratch_reset(nn);
			scratch_128xW = nn_scratch_alloc(nn, sumatmp_size);

			info->next_suma_off = suma_buf_rowstride * sizeof(int32_t);
			int32_t sumabuf_batch_stride = suma_buf_rowstride * out_height;	// size of 1 plane in int32's
			int32_t suma_buf_size = sumabuf_batch_stride * in_batches;	// size of whole buffer in int32's

			sumabuf = nn_scratch_alloc(nn, suma_buf_size * sizeof(int32_t));
			if (sumabuf == NULL) {
				return errlog(nn, "failed to get %d bytes for sumabuf", (int)(suma_buf_size * sizeof(int32_t)));
			}
			input_e = din_e + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * (in_left_pad - required_w_before);
			input_o = din_o + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * (in_left_pad - required_w_before);
		}

		int32_t start_line = 0;
		int32_t stop_line = out_height;
		int32_t n_lines = Q6_R_min_RR(stop_line - start_line, block_height);
		int32_t n_in_rows = (n_lines - 1)*stride_height + filt_height;
		int K = filt_width * filt_height*in_depth_total;
		for (int h = 0; h < out_height; h += block_height) {
			int32_t next_n_lines = Q6_R_min_RR(stop_line - h - block_height, block_height);
			int32_t next_n_in_rows = (next_n_lines - 1)*stride_height + filt_height;

			if (info->use_usmodel == 0) {
				int32_t *integral_tmp = scratch_128xW + info->suma_width * 4;
				vmemset_asm(integral_tmp, 0, info->suma_width * sizeof(int32_t));
				gvint_16b(input_e-32*(info->suma_start-7),
					input_o-32*(info->suma_start-7),
					integral_tmp + info->suma_width,
					in_width_total*32,
					in_width_total*in_depth_total,
					info->suma_width,
					in_depth_total,
					n_in_rows,
					scratch_128xW
				);

				gvsuma_16b(
					integral_tmp,
					sumabuf,
					info->suma_width,
					info->next_suma_off,
					stride_height,
					filt_width,
					filt_height,
					n_lines,
					info->filt_offset
				);
			}
			for (int w = 0; w < out_depth; w += 32) {

				if (info->use_usmodel) {
					nn_graph_memcpy(nn, nn->vtcm_ptr, (const uint8_t *)filt + w * K * 2, 32 * K * 2);

					gvconv2db2b2b2_d32_asm(
						input_e,
						input_o,
						nn->vtcm_ptr, //filt + 2 * w* K,
						output_e + w * out_width_total,
						output_o + w * out_width_total,
						in_width_total,
						out_next_row,
						out_width,
						Q6_R_combine_RlRl(stride_height, stride_width),
						in_depth_total,
						filt_width,
						filt_height,
						n_lines,
						(const int32_t *)info->biasbuf + w,
						minmax.words,
						recip_val,
						47 - recip_shift,
						32 * output_align,
						skip_col
					);
				}
				else {
					gvconv2db2b2b2u_d32_asm(
						input_e,
						input_o,
						output_e + w * out_width_total,
						output_o + w * out_width_total,
						(const uint8_t*)filt + 2 * w* K,
						in_width_total,
						out_next_row,
						out_width,
						Q6_R_combine_RlRl(stride_height, stride_width),
						in_depth_total,
						filt_width,
						filt_height,
						n_lines,
						(const int32_t *)info->biasbuf + w,
						sumabuf + info->suma_start,
						info->next_suma_off,
						minmax.words,
						recip_val,
						47 - recip_shift,
						32 * output_align,
						skip_col
					);
				}
			}
			input_e += stride_height * block_height * in_width_total * in_depth_total;
			input_o += stride_height * block_height * in_width_total * in_depth_total;
			output_e += out_next_row * block_height;
			output_o += out_next_row * block_height;
			n_lines = next_n_lines;
			n_in_rows = next_n_in_rows;
		}
		for (i = 1; i < 32; i++) {
			if (minmax.words[i] > minmax.words[0]) minmax.words[0] = minmax.words[i];
			if (minmax.words[32 + i] < minmax.words[32]) minmax.words[32] = minmax.words[32 + i];
		}
		out_max_val = to_next_power_of_two(minmax.words[0] * prod_level_size);
		out_min_val = to_next_power_of_two(minmax.words[32] * prod_level_size);
		if (!info->maxval_precalculated && out_max_val*(1<<16) > info->out_maxval) {
			info->out_maxval = out_max_val;
		}
		else {
			out_max_val = info->out_maxval / (float)(1 << 16);
		}
		if (!info->minval_precalculated && out_min_val*(1 << 16) < info->out_minval) {
			info->out_minval = out_min_val;
		}
		else {
			out_min_val = info->out_minval / (float)(1 << 16);
		}

		adjust_minmax_for_zero_16b(&out_min_val, &out_max_val);
		info->out_maxval = out_max_val;
		info->out_minval = out_min_val;
		int32_t maxval = fast_roundf(((out_max_val > out_min_val) ? out_max_val : out_min_val) / prod_level_size);
		if (info->is_u16) {
			maxval = fast_roundf((out_max_val - out_min_val) / prod_level_size);
		}
		min_out_prod_offset = fast_roundf (-info->out_minval / prod_level_size);

		for (int i = 0; i < out_depth_total; i++) {
			int64_t fsum = /*info->filt_corr[i]*/ -(int64_t)in_off * info->gemsumb[i];
			if (i < out_depth)
				fsum += (int64_t)(((uint16_t)bias[i] - bias_offset) * bias_mpy_amt + 0.5f);
			if (info->use_usmodel == 0)
				fsum += (int64_t)filt_height*filt_width*in_depth_total*info->filt_offset*info->in_offset;
			info->biasbuf[i] = (fsum >> 16) + min_out_prod_offset;
		}

		recip_shift = 61 - Q6_R_normamt_R(maxval);
		if (recip_shift > 46) recip_shift = 46;

		int64_t denom = 0xffffLL << (recip_shift - 16);

		recip_val = (int32_t)(denom / maxval);
		input_e = din_e + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * input_align;
		input_o = din_o + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * input_align;
		if (info->use_usmodel == 0) {
			input_e = din_e + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * (in_left_pad - required_w_before);
			input_o = din_o + in_width_total * in_depth_total * (in_top_pad - required_h_before) + 32 * (in_left_pad - required_w_before);
		}

		output_e = dout_e + out_next_row * out_top_pad + 32 * out_left_pad;
		output_o = dout_o + out_next_row * out_top_pad + 32 * out_left_pad;

		n_lines = Q6_R_min_RR(stop_line - start_line, block_height);
		n_in_rows = (n_lines - 1)*stride_height + filt_height;
		for (int h = 0; h < out_height; h += block_height) {
			int32_t next_n_lines = Q6_R_min_RR(stop_line - h - block_height, block_height);
			int32_t next_n_in_rows = (next_n_lines - 1)*stride_height + filt_height;

			if (info->use_usmodel == 0) {
				int32_t *integral_tmp = scratch_128xW + info->suma_width * 4;
				vmemset_asm(integral_tmp, 0, info->suma_width * sizeof(int32_t));
				gvint_16b(input_e-32*(info->suma_start-7),
					input_o-32*(info->suma_start-7),
					integral_tmp + info->suma_width,
					in_width_total * 32,
					in_width_total*in_depth_total,
					info->suma_width,
					in_depth_total,
					n_in_rows,
					scratch_128xW
				);

				gvsuma_16b(
					integral_tmp,
					sumabuf,
					info->suma_width,
					info->next_suma_off,
					stride_height,
					filt_width,
					filt_height,
					n_lines,
					info->filt_offset);
			}
			for (int w = 0; w < out_depth; w += 32) {
				if (info->use_usmodel) {
					nn_graph_memcpy(nn, nn->vtcm_ptr, (const uint8_t *)filt + w * K * 2, 32 * K * 2);
					gvconv2db2b2b2_d32_asm(
						input_e,
						input_o,
						nn->vtcm_ptr, //filt + 2 * w* K,
						output_e + w * out_width_total,
						output_o + w * out_width_total,
						in_width_total,
						out_next_row,
						out_width,
						Q6_R_combine_RlRl(stride_height, stride_width),
						in_depth_total,
						filt_width,
						filt_height,
						n_lines,
						(const int32_t *)info->biasbuf + w,
						minmax.words,
						recip_val,
						47 - recip_shift,
						32 * output_align,
						skip_col
					);
				}
				else {
					gvconv2db2b2b2u_d32_asm(
						input_e,
						input_o,
						output_e + w * out_width_total,
						output_o + w * out_width_total,
						(const uint8_t*)filt + 2 * w* K,
						in_width_total,
						out_next_row,
						out_width,
						Q6_R_combine_RlRl(stride_height, stride_width),
						in_depth_total,
						filt_width,
						filt_height,
						n_lines,
						(const int32_t *)info->biasbuf + w,
						sumabuf + info->suma_start,
						info->next_suma_off,
						minmax.words,
						recip_val,
						47 - recip_shift,
						32 * output_align,
						skip_col
					);
				}
			}
			input_e += stride_height * block_height * in_width_total * in_depth_total;
			input_o += stride_height * block_height * in_width_total * in_depth_total;
			output_e += out_next_row * block_height;
			output_o += out_next_row * block_height;
			n_lines = next_n_lines;
			n_in_rows = next_n_in_rows;
		}
		d32_88_to_16_cn(tensor_out->data, dout_e, dout_o, out_width_total, out_height_total, out_depth_total);

		out_min_val = out_min_val * (1 << 16);
		out_max_val = out_max_val * (1 << 16);
		nn_free(din_e);
		nn_free(din_o);
		nn_free(dout_e);
		nn_free(dout_o);
	}
	if (info->is_u16) {
		tensor_set_single_float(tensor_out_min, out_min_val);
		tensor_set_single_float(tensor_out_max, out_max_val);
	}
	else {
		float out_max = fmaxf(-out_min_val, out_max_val);
		tensor_set_single_float(tensor_out_min, -out_max);
		tensor_set_single_float(tensor_out_max, out_max);
	}

	return 0;
error_sn16b:
	return -1;
}

static int supernode_16b_execute_spawn(struct nn_node *self, struct nn_graph *nn) {
	return nn_os_vector_call(nn, supernode_16b_execute, self);
}

static inline void supernode_gemsumb_unsigned(
	sn16b_info *info,
	uint16_t *filt,
	int32_t *sumb,
	int32_t filt_height,
	int32_t filt_width,
	int32_t filt_depth,
	int32_t filt_depth_total,
	int32_t filt_batches,
	int32_t filt_offset,
	int32_t out_depth)
{
	int32_t sum;
	int32_t tmp;
	int32_t h;
	int32_t w;
	int32_t d;
	int32_t b;
	for (b = 0; b < filt_batches; b++) {
		sum = 0;
		for (h = 0; h < filt_height; h++) {
			for (w = 0; w < filt_width; w++) {
				for (d = 0; d < filt_depth_total; d++) {
					if (d >= filt_depth) tmp = filt_offset;
					else tmp = filt[
						h*filt_width*filt_depth*filt_batches
							+ w * filt_depth*filt_batches
							+ d * filt_batches
							+ b];
					sum += tmp;
				}
			}
		}
		sumb[b] = sum;
	}
	for (b = filt_batches; b < out_depth; b++) {
		/* likely executes 0 times */
		sumb[b] = filt_height * filt_width*filt_depth_total*filt_offset;
	}
}

void supernode_rearrange_for_16b_d32(
	int16_t *out_data,
	const int16_t* in_data,
	int filt_height,
	int filt_width,
	int filt_depth,
	int filt_depth_total,
	int filt_batches,
	int filt_batches_total)
{
	int x, y, z, d, s, v, i;

	//  [ batch_hi ][ filt_height ] [ depth_hi ] [ filt_width ] [ depth_mid] [ batch_lo=32] [depth_lo]
	int16_t inval;
	for (x = 0; x < filt_batches_total; x += 32) // batch_hi
	{
		for (y = 0; y < filt_height; y++)
		{
			for (d = 0; d < filt_depth_total; d += 32) // depth_hi
			{
				for (z = 0; z < filt_width; z++)
				{
					for (v = 0; v < 32; v += DLO) // depth_med
					{
						for (s = 0; s < 32; s += 1) // batch_lo
						{
							for (i = 0; i < DLO; i += 1) // depth_lo
							{
								int in_d = d + v + i;
								int in_b = x + s;
								int in_idx = y * filt_width*filt_depth*filt_batches
									+ z * filt_depth*filt_batches
									+ in_d * filt_batches
									+ in_b;
								int out_idx = x * filt_height*filt_width*filt_depth_total
									+ y * filt_width*filt_depth_total * 32
									+ z * 32 * 32
									+ d * filt_width * 32
									+ v * 32
									+ DLO * s
									+ i;
								if ((in_d >= filt_depth) || (in_b >= filt_batches)) inval = 0;
								else inval = in_data[in_idx];
								out_data[out_idx] = inval;
							} //i
						} // s
					} // v
				} // z
			} //d
		} //y
	} //x
	return;
}

// recode a 16 bit signed as hi*256 + lo  where both are signed bytes.
// returns hi:lo in lower 16 bits of result, hi is sign-extended.

static inline int hi_lo_sep(int val)
{
	// clip to range: if upper needs to be range clipped, saturate the lower.
	// lower saturate value is -0x8080, which will not happen
	// val = max_i32(min_i32(val, 0x7f7f), -0x8080);
	val = min_i32(val, 0x7f7f);
	if( val & 0x80) val += 0x100;
	return val;
}


float supernode_rearrange_for_i16b_8b8b(
	int8_t *out_data,
	const uint16_t* in_data,
	int filt_height,
	int filt_width,
	int filt_depth,
	int filt_depth_total,
	int filt_batches,
	int filt_batches_total,
	int zero_val)
{
	int x, y, z, d, s, v, d0;
	int8_t hi, lo;
	//out_width = 32*filt_width*filt_height*32;
	//out_height = out_depth/32 * in_depth/32;
	float scaling = 1.0f / ((zero_val > (32768 - 8)) && (zero_val < (32768 + 8)) ? 1 : 2);

	for (x = 0; x < filt_batches_total; x += 32)
	{
		for (y = 0; y < filt_height; y++)
		{
			for (d = 0; d < filt_depth_total; d += 32)
			{
				for (z = 0; z < filt_width; z++)
				{
					for (s = 0; s < 32; s += 1) // batch
						for (v = 0; v < 32; v += 1) // filt_depth
						{
							int v4 = (v / 4) * 4;
							d0 = fast_roundf((in_data[x + (y*filt_width*filt_depth_total + z * filt_depth_total + d + v)*filt_batches_total + s] - zero_val) * scaling);
							int d1 = hi_lo_sep(d0);
							hi = d1 >>8;
							lo = d1 & 0xff;
							out_data[(2 * x + 0)*filt_height*filt_width*filt_depth_total + y * filt_width*filt_depth_total * 32 + z * 32 * 32 + d * filt_width * 32 + 32 * v4 + 4 * s + (v % 4)] = lo;
							out_data[(2 * x + 32)*filt_height*filt_width*filt_depth_total + y * filt_width*filt_depth_total * 32 + z * 32 * 32 + d * filt_width * 32 + 32 * v4 + 4 * s + (v % 4)] = hi;
						}//filt_width
				}//segment
			}//filt_depth
		}//filt_height
	}//filt_batches
	return scaling;
}

void supernode_rearrange_for_u16b_8b8h(
	uint8_t *out_data,
	const uint16_t* in_data,
	int filt_height,
	int filt_width,
	int filt_depth,
	int filt_depth_total,
	int filt_batches,
	int filt_batches_total,
	int offset)
{
	int x, y, z, d, s, v, d0;
	int8_t hi, lo;
	//out_width = 32*filt_width*filt_height*32;
	//out_height = out_depth/32 * in_depth/32;

	for (x = 0; x < filt_batches_total; x += 32)
	{
		for (y = 0; y < filt_height; y++)
		{
			for (d = 0; d < filt_depth_total; d += 32)
			{
				for (z = 0; z < filt_width; z++)
				{
					for (s = 0; s < 32; s += 1)
						for (v = 0; v < 32; v += 1)
						{
							int v4 = (v / 4) * 4;
							if (x + s < filt_batches && d + v < filt_depth)
								d0 = in_data[x + (y*filt_width*filt_depth + z * filt_depth + d + v)*filt_batches + s];
							else
								d0 = offset;
							hi = d0 >> 8;
							lo = d0 & 0xff;
							out_data[(2 * x + 0)*filt_height*filt_width*filt_depth_total + y * filt_width*filt_depth_total * 32 + z * 32 * 32 + d * filt_width * 32 + 32 * v4 + 4 * s + (v % 4)] = lo;
							out_data[(2 * x + 32)*filt_height*filt_width*filt_depth_total + y * filt_width*filt_depth_total * 32 + z * 32 * 32 + d * filt_width * 32 + 32 * v4 + 4 * s + (v % 4)] = hi;
						}//filt_width
				}//segment
			}//filt_depth
		}//filt_height
	}//filt_batches
}

static void
setup_initial_output_range(
	sn16b_info *info,
	float specified_minval,		// range specified by inputs
	float specified_maxval,
	float minval_default,			// use when specified_minval = -INF
	float maxval_default)			// use when specified_maxval = INF
{
	// enforce sanity:  min <= 0.0 <= max
	// and max > min + 1/128
	//
	specified_minval = fminf(specified_minval, 0.0f);
	specified_maxval = fmaxf(fmaxf(specified_maxval, 0.f),
		specified_minval + 0x1.0p-7f);

	info->out_minval_spec = specified_minval;
	info->out_maxval_spec = specified_maxval;

	int mnp = (specified_minval == -INFINITY) ? 0 : 1;		// is min precalc
	int mxp = (specified_maxval == INFINITY) ? 0 : 1;		// is max precalc

	info->out_minval = mnp ? specified_minval : minval_default;
	info->out_maxval = mxp ? specified_maxval : maxval_default;

	info->minval_precalculated = mnp;
	info->maxval_precalculated = mxp;

	if (info->out_minval < 0.0f) {
		//int corr_code = 2 * mxp + mnp;
		// corr_code:
		//    bit 0 -> out_min is 'fixed'
		//    bit 1 -> out_max is 'fixed';
		// only need if minval != 0
		// TODO:
		adjust_minmax_for_zero_16b(&info->out_minval, &info->out_maxval);
	}
}

static inline float supernode_convert_weights_to_signed(
	sn16b_info *info,
	uint16_t *src,
	int filt_height,
	int filt_width,
	int in_depth,
	int out_depth,
	int zero_val)
{
	int tmp;
	int i;
	float scaling = 1.0f / ((zero_val > (32768 - 8)) && (zero_val < (32768 + 8)) ? 1 : 2);
	for (i = 0; i < (filt_height * filt_width * in_depth * out_depth); i++) {
		tmp = src[i] - zero_val;
		tmp = fast_roundf(scaling * tmp);
		src[i] = min_i32(max_i32(tmp,-32768),32767);
	}
	return scaling;
}

//Sum the raw weights together for the in_offset correction
void filt_sumb_bias_cn(int32_t * filt_sum, int32_t * filt_corr, uint16_t * filt, int32_t zero_val, int out_depth, int K) {
	int i, j;
	int32_t sumw, s0;

	float scaling = 1.0f / ((zero_val > (32768 - 8)) && (zero_val < (32768 + 8)) ? 1 : 2);
	for (i = 0; i < out_depth; i++) {
		sumw = 0;
		s0 = 0x8000LL;
		for (j = 0; j < K; j++) {
			int16_t w0 = fast_roundf((filt[j*out_depth + i]- zero_val)*scaling);
			sumw += w0;
			w0 = w0 & 0xff;
			if (w0 & 0x80) w0 |= 0xff00;
			s0 += 128 * w0;                         //correction factor for loss of 4th loxlo product
		}
		filt_sum[i] = sumw;
		filt_corr[i] = s0;
	}
	return;
}

//=============================================================================
static inline int supernode_n_weight_batches(int batch_size, int vtcm_size)
{
	int slices_per_vtcm = (vtcm_size / batch_size);
	if (slices_per_vtcm > 0) return slices_per_vtcm;
	else return 1;
}

int supernode_16b_check(struct nn_node *self, struct nn_graph *nn) {
	logmsg(nn, 2, "Checking supernode_16b node %p", self);
	int k = node_check_inputs_range(self, nn, "supernode_16b", 12, -12);
	if (k == 0) {
		k = node_check_outputs_range(self, nn, "supernode_16b", 3, -3);
	}
	logmsg(nn, 2, "supernode_16b node %p OK", self);

	sn16b_info *info = self->opaque;
	if (info != NULL) {
		/* Already set up, invalidate strategy and return */
		logmsg(nn, 0, "info was already set up?");
		return 0;
	}
	if ((info = nn_calloc(1, sizeof(*info))) == NULL) {
		goto err_supernode_16b_check;
	}

	// code path:
	// int16  -> s*s             (u16 == 0)
	// uint16 -> u*u             (u16) => v60 model
	//        -> u*s             (u16 + usmodel) => need  this?
	//        -> u*s => 2planes  (2planes+u16+usmodel) => v65/v66model
	info->is_u16 = self->node_type == OP_Supernode_u16x16p16to16_d32;
	info->use_usmodel = info->is_u16 && 0; // unsigned * signed model
	info->use_2planes = info->is_u16 && 1; // uu model only

	// 2plane u*u  u*s
	//   1    v60  v65
	//   0     1   1

	// create weights
	const struct tensor *filt_tensor = self->inputs[1];
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_batches_roundup = (filt_batches + 31) & ~31;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	int32_t filt_depth_roundup = (filt_depth + 31) & ~31;
	uint32_t filt_elements = filt_width * filt_height * filt_depth_roundup;
	uint32_t weights_size = filt_elements * filt_batches_roundup * sizeof(int16_t);

	float weight_scalefac = 1.0;
	int32_t filt_offset = 0;
	float filt_min_float = tensor_get_float(self->inputs[4], 0);
	float filt_max_float = tensor_get_float(self->inputs[5], 0);
	float bias_min_float = tensor_get_float(self->inputs[8], 0);
	float bias_max_float = tensor_get_float(self->inputs[9], 0);
	if (info->is_u16) {
		filt_offset = fast_roundf(-filt_min_float / (filt_max_float - filt_min_float)*65535.0f);
		info->filt_offset = filt_offset;
	}

	if ((info->weights = nn_memalign(128, weights_size)) == NULL) {
		goto err_supernode_16b_check;
	}
	if ((info->gemsumb = nn_memalign(128, filt_batches_roundup * sizeof(int32_t))) == NULL) {
		goto err_supernode_16b_check;
	}
	if (info->use_2planes) {
		if ((info->filt_corr = nn_memalign(128, filt_batches_roundup * sizeof(int32_t))) == NULL) {
			goto err_supernode_16b_check;
		}
		if (info->use_usmodel) {
			weight_scalefac = supernode_rearrange_for_i16b_8b8b(
				(int8_t*)info->weights,
				filt_tensor->data,
				filt_height,
				filt_width,
				filt_depth,
				filt_depth_roundup,
				filt_batches,
				filt_batches_roundup,
				filt_offset);

			filt_sumb_bias_cn(
				info->gemsumb, info->filt_corr,
				filt_tensor->data, filt_offset, filt_batches_roundup, filt_width*filt_depth_roundup*filt_height);
		}
		else { // uu model
			supernode_rearrange_for_u16b_8b8h(
				info->weights,
				filt_tensor->data,
				filt_height,
				filt_width,
				filt_depth,
				filt_depth_roundup,
				filt_batches,
				filt_batches_roundup,
				info->filt_offset);
			supernode_gemsumb_unsigned(
				info,
				filt_tensor->data,
				info->gemsumb,
				filt_height,
				filt_width,
				filt_depth,
				filt_depth_roundup,
				filt_batches,
				filt_offset,
				filt_batches_roundup);
		}

	}
	else {
		// u16 = 0 or 1, usmodel && !use_2planes
		supernode_rearrange_for_16b_d32(
			(int16_t *)info->weights,
			filt_tensor->data,
			filt_height,
			filt_width,
			filt_depth,
			filt_depth_roundup,
			filt_batches,
			filt_batches_roundup);

		if (info->is_u16 && info->use_usmodel) {
			weight_scalefac = supernode_convert_weights_to_signed(
				info,
				(uint16_t*)info->weights,
				filt_height,
				filt_width,
				filt_depth_roundup,
				filt_batches_roundup,
				filt_offset);
		}
		else {
			supernode_gemsumb_unsigned(
				info,
				filt_tensor->data,
				info->gemsumb,
				filt_height,
				filt_width,
				filt_depth,
				filt_depth_roundup,
				filt_batches,
				filt_offset,
				filt_batches_roundup);
		}
	}


	if (info->is_u16) {

		if ((info->suma_buf = nn_memalign(128, filt_batches_roundup * sizeof(info->suma_buf[0]))) == NULL) {
			goto err_supernode_16b_check;
		}

		info->bias_offset = fast_roundf(-bias_min_float / (bias_max_float - bias_min_float) * 65535.0f);
		info->filt_level_size = (filt_max_float - filt_min_float) / (65535.0f * weight_scalefac);
		info->bias_level_size = (bias_max_float - bias_min_float) / 65535.0f;

	}
	else {
		info->filt_level_size = (-filt_min_float) * (float)(1. / 32768.0);
		info->bias_level_size = (-bias_min_float) * (float)(1. / 32768.0);
	}

	// bias buffer
	if ((info->biasbuf = nn_memalign(128, filt_batches_roundup * sizeof(int32_t))) == NULL) {
		goto err_supernode_16b_check;
	}

	self->opaque = info;

	float specified_minval = tensor_get_float(self->inputs[10], 0);
	float specified_maxval = tensor_get_float(self->inputs[11], 0);
	setup_initial_output_range(info, specified_minval, specified_maxval, 0.0f, 0.5f);

	return 0;

err_supernode_16b_check:
	if (info) {
		nn_free(info->gemsumb);
		nn_free(info->filt_corr);
		nn_free(info->gemsumb);
		nn_free(info->suma_buf);
		nn_free(info->weights);
	}
	nn_free(info);
	return errlog(nn, "supernode_16b_check error");
}

//=============================================================================
// using the supernode_new MT 
//=============================================================================
#define roundup(a, p2)       (((a)+(p2)-1)&~((p2)-1))

//=============================================================================
#if !defined(V66)
struct split_thread {
	struct split_state *pstate;
	uint32_t tid;
	uint16_t *input;
	uint8_t *oute;
	uint8_t *outo;
	uint32_t row;
};

struct split_state {
	nn_sem_t done_sem;
	uint32_t batch;
	uint32_t width;
	uint32_t depth;
	uint32_t batch_stride;
	struct split_thread tinfo[NUM_THREADS];
};

static void
split_worker_thread(struct nn_graph *nn, void *thrpv) {
	struct split_thread *pth = (struct split_thread *)thrpv;
	struct split_state *pst = pth->pstate;

	uint8_t *oute = pth->oute;
	uint8_t *outo = pth->outo;
	uint16_t *input = pth->input;
	uint16_t b_stride = pst->batch_stride;
	for (int b = 0; b < pst->batch; b++) { // TODO:interlock increment
		d32_16_to_88_cn(oute, outo, input, pst->width, pth->row, pst->depth);
		oute += b_stride;
		outo += b_stride;
		input += b_stride;
	}
	nn_sem_post(&pst->done_sem);
}

static int supernode_split_input(struct nn_node *self, struct nn_graph *nn) {
	const struct tensor *tensor_in = self->inputs[0];
	int32_t in_width = tensor_in->shape.width;
	int32_t in_height = tensor_in->shape.height;
	int32_t in_depth = tensor_in->shape.depth;
	int32_t in_left_pad = tensor_in->format.width_pad[0];
	int32_t in_right_pad = tensor_in->format.width_pad[1];
	int32_t in_depth_before_pad = tensor_in->format.depth_pad[0];
	int32_t in_depth_after_pad = tensor_in->format.depth_pad[1];
	int32_t in_top_pad = tensor_in->format.height_pad[0];
	int32_t in_bot_pad = tensor_in->format.height_pad[1];
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	uint16_t *input = tensor_in->data;
	sn16b_info *info = self->opaque;

	int n_threads = min_i32(NUM_THREADS, in_height);
	int row = (in_height + n_threads - 1) / n_threads;

	struct split_state runstate;
	runstate.batch = tensor_in->shape.batches;
	runstate.width = in_width_total;
	runstate.depth = in_depth_total;
	runstate.batch_stride = in_width_total * in_depth_total * (in_top_pad + in_height + in_bot_pad);
	nn_sem_init(&runstate.done_sem, 0);

	for (int i = 0; i < n_threads; i++) {
		int trow = min_i32(row, in_height - i * row);
		runstate.tinfo[i].pstate = &runstate;
		runstate.tinfo[i].tid = i;
		runstate.tinfo[i].row = trow;
		runstate.tinfo[i].input = input + (in_top_pad + i * row) * in_depth_total * in_width_total;
		runstate.tinfo[i].oute = info->din_e+(in_top_pad+i*row) * in_depth_total * in_width_total;
		runstate.tinfo[i].outo = info->din_o+(in_top_pad+i*row) * in_depth_total * in_width_total;
		nn_os_work_for_vector(nn, split_worker_thread, &runstate.tinfo[i]);
	}
	nn_sem_sub(&runstate.done_sem, n_threads);
	return 0;
}

//=============================================================================
struct desplit_thread {
	struct desplit_state *pstate;
	uint32_t tid;
	uint16_t *output;
	uint8_t *input_e;
	uint8_t *input_o;
	uint32_t row;
};

struct desplit_state {
	nn_sem_t done_sem;
	uint32_t batch;
	uint32_t depth;
	uint32_t width;
	uint32_t batch_stride;
	struct desplit_thread tinfo[NUM_THREADS];
};

static void
desplit_worker_thread(struct nn_graph *nn, void *thrpv) {
	struct desplit_thread *pth = (struct desplit_thread *)thrpv;
	struct desplit_state *pst = pth->pstate;
	uint8_t *input_e = pth->input_e;
	uint8_t *input_o = pth->input_o;
	uint16_t *output = pth->output;
	uint16_t b_stride = pst->batch_stride;
	for (int b = 0; b < pst->batch; b++) { // TODO:interlock increment
		d32_88_to_16_cn(output, input_e, input_o, pst->width, pth->row, pst->depth);
		input_e += b_stride;
		input_o += b_stride;
		output += b_stride;
	}
	nn_sem_post(&pst->done_sem);
}

int supernode_desplit_input(struct nn_node *self, struct nn_graph *nn) {
	struct tensor *tensor_out = self->outputs[0];
	uint16_t *output = tensor_out->data;
	sn16b_info *info = self->opaque;

	/*
	 * Set output padding values to sensible defaults.
	 */
	int32_t out_depth_total = info->out_depth_total;
	int32_t out_width_total = info->out_next_d32 >> 5;
	int32_t out_height = info->out_height;
	int32_t out_top_pad = tensor_out->format.height_pad[0];
	int32_t out_bot_pad = tensor_out->format.height_pad[1];

	struct desplit_state runstate;
	runstate.batch = tensor_out->shape.batches;
	runstate.depth = out_depth_total;
	runstate.width = out_width_total;
	runstate.batch_stride = out_depth_total * out_width_total * (out_top_pad+out_height+out_bot_pad);
	nn_sem_init(&runstate.done_sem, 0);

	int n_threads = min_i32(NUM_THREADS, out_height);
	int row = (out_height + n_threads - 1) / n_threads;

	for (int i = 0; i < n_threads; i++) {
		int trow = min_i32(row, out_height - i * row);
		runstate.tinfo[i].pstate = &runstate;
		runstate.tinfo[i].tid = i;
		runstate.tinfo[i].row = trow;
		runstate.tinfo[i].input_e = info->dout_e + (out_top_pad + i * row) * out_depth_total * out_width_total;
		runstate.tinfo[i].input_o = info->dout_o + (out_top_pad + i * row) * out_depth_total * out_width_total;
		runstate.tinfo[i].output = output + (out_top_pad + i * row) * out_depth_total * out_width_total;
		nn_os_work_for_vector(nn, desplit_worker_thread, &runstate.tinfo[i]);
	}
	nn_sem_sub(&runstate.done_sem, n_threads);
	return 0;
}

//=============================================================================
static inline int supernode_execute_some_strategy(struct nn_node *self, struct nn_graph *nn, int start, int n_work_items)
{
	sn16b_info *info = self->opaque;
	int err = 0;
	logmsg(nn, 3, "adding %d items @ %d", n_work_items, start);
	nn_os_worklist_for_vector(nn, &info->work_list[start], n_work_items);
	return err;
}

static inline void supernode_do_memcpy(struct nn_graph *nn, uint8_t *dst, const uint8_t *src, uint32_t size)
{
	return nn_graph_memcpy(nn, dst, src, size);
}

/* Choose VTCM address or in-place */
static inline uint8_t *supernode_filtbuf_location(
	struct nn_graph *nn,
	sn16b_info *info,
	int pingpong,
	const uint8_t *weights_in,
	const uint32_t inner_weight_size)
{
#if defined(V65) || defined(V66)
	if (info->use_v65 || info->use_v66) {
		if (/* nn->vtcm_ptr && */ (inner_weight_size <= (nn->vtcm_size - VTCM_CIRCBUF_SIZE))) {
			return nn->vtcm_ptr;
		}
		else {
			logmsg(nn, 1, "weights %ld dont fit @ %p", inner_weight_size, nn->vtcm_ptr);
		}
	}
	else {
		logmsg(nn, 1, "V65/V66 compilation, but not using v65 or v66, do not copy to vtcm");
	}
#endif
	/* For now, just in-place */
	return (uint8_t *)weights_in;
}

static void supernode_execute_hvx_conv_work(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	struct nn_node *self = work->self;
	sn16b_info *info = self->opaque;

	int32_t start_line = work->start_line;
	int32_t stop_line = work->stop_line;
	int32_t skip_lines = work->skip_lines;
	int32_t in_next_row = info->in_next_row;
	int32_t in_depth = info->in_depth;
	int32_t in_next_d32 = info->in_next_d32;
	int32_t out_width = info->out_width_processed;
	int32_t out_next_row = info->out_next_row;
	int32_t filt_width = info->filt_width;
	int32_t filt_height = info->filt_height;
	int32_t stride_width = info->stride_width;
	int32_t stride_height = info->stride_height;
	int32_t next_suma_row = info->next_suma_off;

	int32_t recip_val = info->recip_val;
	//int32_t recip_shamt = info->recip_shamt;
	int32_t out_left_junk = info->out_left_junk;
	int32_t skip_col = info->skip_col;
	int32_t weight_chunks = work->weight_chunks;
	int32_t out_next_d32 = info->out_next_d32;
	int32_t weight_batch_size = info->weight_batch_size;

	const uint8_t *weights = work->weights;
	const int32_t *biasbuf = work->biases;
	const uint8_t *input = work->input + start_line * in_next_row*stride_height;
	uint8_t *output = work->output + start_line * out_next_row;

#ifdef V65
	//int32_t use_v65 = info->use_v65;
#else
	//int32_t use_v65 = 0;
#endif
	uint64_t start_cycles;
	uint64_t my_cycles;
	union {
		HVX_Vector vec[2];
		int32_t words[64];
	} minmax;

	int w, out_row;

	//logmsg(nn,2,"min: %d max: %d",info->minval,info->maxval);

	start_cycles = nn_os_get_cycles(nn);
	//if (start_line >= stop_line) goto done;

#if 1
	logmsg(nn, 2, "start/stop/skip_lines: %d/%d/%d output %p input %p filt %p bias %p in_depth %d in_next_row %d in_next_d32 %d out_width %d out_next_row %d out_next_d32 %d filt_width %d filt_height %d stride_width %d stride_height %d recip_shamt %d recip_val 0x%08x minmax_buf %p out_left_junk=%d in_left_skip=%d dummy_zeros=%p n_lines=%d suma_buf_start=%p suma_start=%d suma_buf=%p next_suma_row=%d skip_col=%d weight_chunks=%d weight_batch_size=%d cycles=%lld",
		start_line,
		stop_line,
		skip_lines,
		output,
		input,
		weights,
		biasbuf,
		in_depth,
		in_next_row,
		in_next_d32,
		out_width,
		out_next_row,
		out_next_d32,
		filt_width,
		filt_height,
		stride_width,
		stride_height,
		0,
		recip_val,
		minmax.words,
		out_left_junk,
		info->in_left_skip,
		NULL,
		(stop_line - start_line + (skip_lines - 1)) / skip_lines,
		work->suma_buf,
		info->suma_start,
		work->suma_buf + info->suma_start,
		next_suma_row,
		skip_col,
		weight_chunks,
		weight_batch_size,
		start_cycles);
#endif

	nn_sem_t * done_sem = work->donesem;	// sem to post at the end

	minmax.vec[1] = Q6_V_vsplat_R(0x7FFFFFFF);
	minmax.vec[0] = Q6_V_vnot_V(minmax.vec[1]);
	{
		/*-------------------------------------------------------------*/
		/*              V60 Implementations                            */
		/*-------------------------------------------------------------*/
			//  The 'suma_scratch buffer' is used for the intermediate integral buffer;
			// only grab it if/when we need it.
		int suma_scratch_index;
		int32_t * suma_scratch = NULL;
		// actual suma goes here. Current vertical extent of current batch.
		int32_t *suma = work->suma_buf + start_line * next_suma_row / sizeof(int32_t);

		int32_t  proc_rows = work->num_lines;
		int32_t  pf_offset = Q6_R_max_RR(filt_height - stride_height, 0);

		int32_t n_lines = Q6_R_min_RR(stop_line - start_line, proc_rows);
		int32_t n_in_rows = (n_lines - 1)*stride_height + filt_height;

		// prefetch initial activations
		l2fetch(input, in_next_row, in_next_row, n_in_rows);
		for (out_row = start_line; out_row < stop_line; out_row += proc_rows) {

			int32_t next_n_lines = Q6_R_min_RR(stop_line - out_row - proc_rows, proc_rows);
			int32_t next_n_in_rows = (next_n_lines - 1)*stride_height + filt_height;

			const uint8_t *input_e = input;
			const uint8_t *input_o = input + info->in_oedelta;
			uint8_t *output_e = output;
			uint8_t *output_o = output + info->out_oedelta;

			if (work->need_initialize_suma) {
				/* OK do sumabuf.
				 */
				if (suma_scratch == NULL) {
					suma_scratch = bufpool_take(&info->bufpool_suma_scratch, &suma_scratch_index);
					if (suma_scratch == NULL) {
						errlog(nn, "failed to get suma scratch");
						goto done;
					}
				}
				// four lines of "scratch128xW; the rest is for integral
				int32_t *scratch_128xW = suma_scratch;
				int32_t *integral_tmp = scratch_128xW + info->suma_width * 4;

				// zero the first row of integral_tmp
				vmemset_asm(integral_tmp, 0, info->suma_width * sizeof(int32_t));

				gvint_16b(input_e - 32 * (info->suma_start - 7),
					input_o - 32 * (info->suma_start - 7),
					integral_tmp + info->suma_width,
					in_next_d32, //in_width_total*32,
					in_next_row, //in_width_total*in_depth_total,
					info->suma_width,
					in_depth, //in_depth_total,
					n_in_rows,
					scratch_128xW
				);
				gvsuma_16b(
					integral_tmp,
					suma,
					info->suma_width,
					info->next_suma_off,
					stride_height,
					filt_width,
					filt_height,
					n_lines,
					info->filt_offset
				);
			}
			else {
				//logmsg(nn,2,"waiting for progress...");
				//nn_progress_waitfor(suma_progress,suma_progress_need);
				// prefetch, if it was already done.
				// likely just been found.
				l2fetch(suma + info->suma_start, next_suma_row, info->in_width * sizeof(int32_t), n_lines);
				//logmsg(nn,2,"progress complete...");
			}

			// convolution
			for (w = 0; w < weight_chunks; w++) {

				// prefetch next batch of weights
				if (w < (weight_chunks - 1)) {
					if (out_row == 0)
						l2fetch(weights + (w + 1)*weight_batch_size, weight_batch_size / 32, weight_batch_size / 32, 32);
				}
				else {
					if (work->pf_inp && out_row == (stop_line - n_lines))
						l2fetch(weights + weight_chunks * weight_batch_size, weight_batch_size / 32, weight_batch_size / 32, 32);

					// prefetch activations
					if (next_n_lines > 0)
						l2fetch(input + (proc_rows*stride_height + pf_offset)*in_next_row, in_next_row, in_next_row, next_n_in_rows - pf_offset);
				}
				gvconv2db2b2b2u_d32_asm(
					input_e,
					input_o,
					output_e + w * out_next_d32,
					output_o + w * out_next_d32,
					weights + w * weight_batch_size,
					in_next_d32 >> 5,
					out_next_row,
					out_width,
					Q6_R_combine_RlRl(stride_height, stride_width),
					in_depth,
					filt_width,
					filt_height,
					n_lines,
					(const int32_t *)work->biases + w*32,
					suma + info->suma_start,
					info->next_suma_off,
					minmax.words,
					recip_val,
					47 - info->recip_shamt/*recip_shift*/,
					32 * 0,
					skip_col
				);
			}
			input += proc_rows * in_next_row*stride_height;
			output += proc_rows * out_next_row;
			suma += proc_rows * next_suma_row / sizeof(int32_t);
			n_lines = next_n_lines;
			n_in_rows = next_n_in_rows;
		}
		// release suma_scratch, if needed.
		if (suma_scratch != NULL)
			bufpool_release(&info->bufpool_suma_scratch, suma_scratch_index);
	}
	gvrmaxmin(minmax.words);
	nn_atomic_min(&info->minval, minmax.words[32]);
	nn_atomic_max(&info->maxval, minmax.words[0]);

	my_cycles = nn_os_get_cycles(nn) - start_cycles;
	nn_atomic_add64(&info->cycles, my_cycles);
	//asm volatile ("":::"memory");
	logmsg(nn, 2, "min=%d(%d) max=%d(%d) cycles=%lld", minmax.words[32], info->minval, minmax.words[0], info->maxval, my_cycles);
	//debug_value_range(nn, work->output + start_line * out_next_row, out_width, stop_line - start_line, out_next_row*skip_lines);
	//logmsg(nn,0,"posting to %p",work->donesem);
done:
	nn_checkpoint_arrival(nn, self, &info->conv_slices[work->start_chunk].checkpoint);
	if (done_sem != NULL) nn_sem_post(done_sem);
}
#endif

static inline int supernode_reset_work_items(
	struct nn_node *self,
	struct nn_graph *nn,
	sn16b_info *info)
{
	if (info->work_items) nn_free(info->work_items);
	if (info->work_list) nn_free(info->work_list);
	info->work_items = NULL;
	info->work_list = NULL;
	info->n_work_items = 0;
	info->workitems_alloc = 0;
	return 0;
}

#if !defined(V66)
// (reset record count; keep the buffer)
static inline int supernode_softreset_work_items(
	struct nn_node *self,
	struct nn_graph *nn,
	sn16b_info *info)
{
	info->n_work_items = 0;
	return 0;
}

static inline int supernode_add_work_item(
	struct nn_node *self,
	struct nn_graph *nn,
	sn16b_info *info,
	const struct workitem work /* BY VAL */)
{
	struct workitem *witems = info->work_items;
	int this_index = info->n_work_items;
	int new_work_items = this_index + 1;

	/* Round up to multiple of 16 work items */
	if (new_work_items > info->workitems_alloc)
	{
		int new_work_alloc = (new_work_items + 15) & ~15;
		struct workitem *new_data;
		if ((new_data = nn_realloc(witems, new_work_alloc * sizeof(*new_data))) == NULL) {
			return errlog(nn, "realloc fails");
		}
		info->workitems_alloc = new_work_alloc;
		info->work_items = witems = new_data;
#if 1
		nn_os_workitem_t *new_list;
		if ((new_list = nn_realloc(info->work_list, new_work_alloc * sizeof(*new_list))) == NULL) {
			return errlog(nn, "realloc fails");
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
	sn16b_info *info,
	struct nn_node *self)
{
	int i;
	for (i = 0; i < info->n_work_items; i++) {
		info->work_list[i].f = info->work_items[i].new_execute;
		info->work_list[i].arg = &info->work_items[i];
	};
}

// needs to wait for all workers to be done
// Maybe change to just be called by worker thread
static int supernode_execute_workitem_check_for_retry(struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	sn16b_info *info = node->opaque;
	float newval;
	float extreme_out;
	int recalc = 0;
	if (info->minval_precalculated && info->maxval_precalculated) return 0;
	if (unlikely(!info->maxval_precalculated && (info->maxval > info->max_valid_val))) {
		/* Mark as needing retry and set new max value */
		info->needs_retry = 1;
		extreme_out = info->maxval * info->prod_level_size * (1<<16) + info->out_minval;
		newval = to_next_power_of_two(fmaxf(extreme_out, 0x1.0p-4f));
		logmsg(nn, 1, "max too small, recalculating %d > %d / %f > %f... picking %f",
			info->maxval, info->max_valid_val, extreme_out, info->out_maxval, newval);
		info->out_maxval = newval;
		recalc = 1;
	}
	if (unlikely(!info->minval_precalculated && (info->minval < info->min_valid_val))) {
		/* Mark as needing retry and set new min value */
		info->needs_retry = 1;
		extreme_out = info->minval * info->prod_level_size * (1 << 16) + info->out_minval;
		newval = to_next_power_of_two(fminf(extreme_out, -0x1.0p-8f));
		logmsg(nn, 1, "min too large, recalculating %d < %d / %f < %f... picking %f",
			info->minval, info->min_valid_val, extreme_out, info->out_minval, newval);
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
	if (recalc && !info->maxval_precalculated && !info->minval_precalculated) {
		adjust_minmax_for_zero_16b(&info->out_minval, &info->out_maxval);
		logmsg(nn, 2, "corrected range: %f ... %f", info->out_minval, info->out_maxval);
	}


	//logmsg(nn,1,"Checking workitem, maxval=%x minval=%x max_valid_val=%x needs_retry=%d",info->maxval,info->minval,info->max_valid_val,info->needs_retry);
	return 0;
}

static void supernode_execute_zap_right(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	const sn16b_info *info = work->info;
	uint8_t *rowstart = work->zap_right;
	uint16_t val = work->zap_value;
	uint32_t size = work->zap_right_size;
	uint32_t in_next_d32 = info->in_next_d32;
	uint32_t in_next_row = info->in_next_row;
	int32_t batch_stride = info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn, 2, "zapping right: %d*%d bytes @ %p rl_depths=%d next_d32=%d next_row=%d val=%d height=%d", work->zap_batches, size * 32, rowstart, work->zap_rl_depths, in_next_d32, in_next_row, val, work->zap_height);
	for (i = 0; i < n_batches; i++) {
		padzap_part(rowstart, val, in_next_d32, work->zap_rl_depths, in_next_row, work->zap_height, size);
		padzap_part(rowstart+work->zap_oedelta, val>>8, in_next_d32, work->zap_rl_depths, in_next_row, work->zap_height, size);
		rowstart += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn, work->self, work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void supernode_execute_zap_left(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	const sn16b_info *info = work->info;
	uint8_t *rowstart = work->zap_left;
	uint16_t val = work->zap_value;
	uint32_t in_next_d32 = info->in_next_d32;
	uint32_t in_next_row = info->in_next_row;
	int32_t batch_stride = info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn, 2, "zapping left: %d*%d bytes @ %p rl_depths=%d next_d32=%d next_row=%d val=%d height=%d", work->zap_batches, 32 * work->zap_left_size, rowstart, work->zap_rl_depths, in_next_d32, in_next_row, val, work->zap_height);
	for (i = 0; i < n_batches; i++) {
		padzap_part(rowstart, val, in_next_d32, work->zap_rl_depths, in_next_row, work->zap_height + 2, work->zap_left_size);
		padzap_part(rowstart+work->zap_oedelta, val>>8, in_next_d32, work->zap_rl_depths, in_next_row, work->zap_height + 2, work->zap_left_size);
		rowstart += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn, work->self, work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void __attribute__((unused)) supernode_execute_zap_toptop(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	uint8_t *start = work->zap_top - work->info->in_next_row;
	int32_t batch_stride = work->info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn, 2, "zapping toptop: %d*%d bytes @ %p val=%d", work->zap_batches, work->zap_top_size, work->zap_top, work->zap_value);
	for (i = 0; i < n_batches; i++) {
		//vmemset_nt_asm(start, work->zap_value, work->info->in_next_row);
		//vmemset_nt_asm(start+work->zap_oedelta, work->zap_value >> 8, work->info->in_next_row);
		start += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn, work->self, work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void supernode_execute_zap_top(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	uint8_t *start = work->zap_top;
	int32_t batch_stride = work->info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn, 2, "zapping top: %d*%d bytes @ %p val=%d", work->zap_batches, work->zap_top_size, work->zap_top, work->zap_value);
	for (i = 0; i < n_batches; i++) {
		vmemset_nt_asm(start, work->zap_value, work->zap_top_size);
		vmemset_nt_asm(start + work->zap_oedelta, work->zap_value >> 8, work->zap_top_size);
		start += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn, work->self, work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}

static void supernode_execute_zap_bot(struct nn_graph *nn, void * vinfo)
{
	struct workitem *work = vinfo;
	uint8_t *start = work->zap_bot;
	int32_t batch_stride = work->info->in_next_batch;
	int32_t n_batches = work->zap_batches;
	int i;
	logmsg(nn, 2, "zapping bot: %d*%d bytes @ %p val=%d", work->zap_batches, work->zap_bot_size, work->zap_bot, work->zap_value);
	for (i = 0; i < n_batches; i++) {
		vmemset_nt_asm(start, work->zap_value, work->zap_bot_size);
		vmemset_nt_asm(start + work->zap_oedelta, work->zap_value >> 8, work->zap_bot_size);
		start += batch_stride;
	}
	if (work->zap_checkpoint) nn_checkpoint_arrival(nn, work->self, work->zap_checkpoint);
	if (work->donesem) nn_sem_post(work->donesem);
}


static inline void supernode_handle_earlywork(struct nn_graph *nn, struct nn_node *self, sn16b_info *info)
{
	struct nn_early_work *work = info->next_earlywork;
	if (work == NULL) return;
	if (work->vtcm_addr != nn->vtcm_ptr) return;
	if (work->src_addr == NULL) return;
	if (work->dst_addr == NULL) return;
	if (work->bytes == 0) return;
	logmsg(nn, 2, "Doing early work copy: %d bytes %p <-- %p", work->bytes, work->dst_addr, work->src_addr);
	supernode_do_memcpy(nn, work->dst_addr, work->src_addr, work->bytes);
	work->valid = 1;
}

#if 0
static inline void supernode_note_earlywork(
	struct nn_graph *nn,
	struct nn_node *self,
	sn16b_info *info,
	void *dst_addr,
	const void *src_addr,
	size_t bytes)
{
	struct nn_early_work *work = &info->my_earlywork;
	work->dst_addr = dst_addr;
	work->src_addr = src_addr;
	work->bytes = bytes;
	work->valid = 0;
	work->vtcm_addr = nn->vtcm_ptr;
}
#endif

static void note_alldone_checkpoint_arrival(struct nn_graph *nn, struct nn_node *self, void *opaque)
{
	struct workitem *work = opaque;
	sn16b_info *info = work->info;
	logmsg(nn, 2, "Saw all done checkpoint complete @ node %p work item %p", self, work);
	if (work->next_startup_offset == 0) {
		logmsg(nn, 2, "Last batch, should return info=%p", info);
		nn_sem_post(&info->alldone_sem);
		supernode_handle_earlywork(nn, self, info);
	}
	else {
		logmsg(nn, 4, "Enqueue next batch startup distance %d", work->next_startup_offset);
		supernode_execute_some_strategy(self, nn, info->batch_start_idx + work->next_startup_offset, 1);
	}
}

static void note_startup_arrival(struct nn_graph *nn, struct nn_node *self, void *opaque)
{
	sn16b_info *info = self->opaque;
	struct weight_slice_info *myslice = opaque;
	logmsg(nn, 2, "Saw startup checkpoint complete @ node %p batch start offset=%d n=%d", self,
		myslice->batch_start_offset, myslice->wakeup_items);
	supernode_execute_some_strategy(self, nn, info->batch_start_idx + myslice->batch_start_offset, myslice->wakeup_items);
	//nn_checkpoint_arrival(nn,self,&info->alldone_checkpoint);
}

static inline void supernode_do_startup_memcpy(
	struct nn_graph *nn,
	struct nn_node *self,
	sn16b_info *info,
	uint8_t *dst,
	const uint8_t *src,
	uint32_t size)
{
	if (info->my_earlywork.valid) {
		/* Skip over early work, predecessor did it */
		logmsg(nn, 2, "Yay, early work was valid! Skipping copy...");
		info->my_earlywork.valid = 0;
	}
	else {
		supernode_do_memcpy(nn, dst, src, size);
	}
}

static void supernode_execute_workitem_startwork(struct nn_graph *nn, void *vwork) //struct workitem *work, struct nn_node *node, struct nn_graph *nn)
{
	struct workitem *work = vwork;
	struct nn_node *node = work->self;
	sn16b_info *info = work->info;
	int jobs = 0;
	long right_ptr_l = (long)work->zap_right;
	int work_idx = work->my_idx;
	logmsg(nn, 3, "starting work idx=%d", work_idx);
	//nn_sem_init(&info->alldone_sem,0);
	work->donesem = NULL;
	work->zap_checkpoint = &info->startup_info.checkpoint;
	info->batch_start_idx = work_idx + 1;
	work->wait_before = 0;
	// We set up our checkpoint for the number of zaps to do plus one arrival for ourselves.
	nn_checkpoint_init(&info->startup_info.checkpoint, work->zap_jobs + 1, note_startup_arrival, &info->startup_info);
	nn_checkpoint_init(&info->alldone_checkpoint, work->join_iters, note_alldone_checkpoint_arrival, work);
#ifdef V66
	// V66 memcpy is in background, do it first to overlap with zapping
	if (work->copy_size > 0) {
		supernode_do_startup_memcpy(nn, node, info, work->copy_out, work->copy_in, work->copy_size);
	}
#endif
	if (work->pf_height && work->pf_inp) {
		l2fetch(work->pf_inp, work->pf_stride, work->pf_width, work->pf_height);
	}
	if (work->zap_right_size && (right_ptr_l % 128)) {
		nn_os_work_for_vector(nn, supernode_execute_zap_right, work);
		jobs++;
	}
	if (work->zap_top_size > 0) {
		//printf("zapping %d (%d) at top: %p\n", (int)(work->zap_top_size/work->info->in_next_row), (int)work->zap_top_size, work->zap_top);
		nn_os_work_for_vector(nn, supernode_execute_zap_top, work);
		jobs++;
	}
	if (work->zap_bot_size > 0) {
		//printf("zapping %d (%d) at bottom: %p\n", (int)(work->zap_bot_size/work->info->in_next_row), (int)work->zap_bot_size, work->zap_bot);
		nn_os_work_for_vector(nn, supernode_execute_zap_bot, work);
		jobs++;
	}
#if 0 //def V66
	if (work->zap_left_size > 0) {
		nn_os_work_for_vector(nn, supernode_execute_zap_left, work);
		sems_down++;
	}
#else
	if (work->zap_left_size > 0) {
		nn_os_work_for_vector(nn, supernode_execute_zap_left, work);
		jobs++;
	}
#endif
#ifndef V66
	if (1) if (work->zap_top_size > 0) {
		nn_os_work_for_vector(nn, supernode_execute_zap_toptop, work);
		jobs++;
	}
#endif
#ifndef V66
	// Pre-V66 memcpy work is in foreground, do it here to do it in parallel
	if (work->copy_size > 0) {
		supernode_do_startup_memcpy(nn, node, info, work->copy_out, work->copy_in, work->copy_size);
	}
#endif
	// our own mark of checkpoint completion, mainly for pre-V66 memcpy
	nn_checkpoint_arrival(nn, node, &info->startup_info.checkpoint);
	if (unlikely(jobs != work->zap_jobs)) errlog(nn, "consistency failure in startup zap work");
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


static inline __attribute__((unused)) int supernode_add_batch_startup(
	struct nn_node *self,
	struct nn_graph *nn,
	sn16b_info *info,
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
	int n_zaps = supernode_count_zap_jobs(&startwork, self, nn);
	startwork.zap_jobs = n_zaps;
	return supernode_add_work_item(self, nn, info, startwork);
}

static inline int32_t slice_for_cache(
	int32_t node_id,
	/* General stuff */
	struct nn_graph *nn,
	/* activation information */
	int32_t batches,
	int32_t d32_slices,
	int32_t width_total,
	int32_t out_width,
	int32_t out_height,
	int32_t stride_height,
	int32_t filt_width,
	int32_t filt_height,
	/* Weights information */
	int32_t weight_d32_size,
	int32_t weight_d32_slices,
	/* Results! */
	int32_t *inner_act_batches,
	int32_t *inner_act_rows,
	int32_t *inner_weight_chunks)
{
	int32_t size_per_line = 32 * d32_slices*width_total;
	//luc
	logmsg(nn, 1, "id=%d in_h=%d out_h=%d sz_per_line=%d w_d32_sz=%d w_d32_slices=%d act_tot=%d w_tot_sz=%d",
		node_id,
		out_height*stride_height + filt_height - stride_height,
		out_height,
		size_per_line,
		weight_d32_size,
		weight_d32_slices,
		size_per_line*batches*(out_height*stride_height + filt_height - stride_height),
		weight_d32_size * weight_d32_slices);

	int32_t nchunks = Q6_R_min_RR(Q6_R_max_RR(128 * 1024 / weight_d32_size, 1), weight_d32_slices);
	int32_t overhead = (filt_height - stride_height)*size_per_line;

	if (filt_width != 1 || filt_height != 1) {
		overhead += (filt_height + 2)*width_total * sizeof(int32_t);
	}

	int32_t datasize_for_outrow = stride_height * size_per_line + width_total * sizeof(int32_t) + 32 * out_width*nchunks;
	int32_t nlines0 = 32 * 1024 / (size_per_line*stride_height);
	int32_t nlines1 = (nn->vtcm_size - (nchunks + 1)*weight_d32_size - NUM_THREADS * overhead) / (NUM_THREADS*datasize_for_outrow);

	*inner_act_batches = batches;
	*inner_act_rows = Q6_R_max_RR(Q6_R_min_RR(nlines0, nlines1), 1);
	*inner_weight_chunks = nchunks;
	return 0;
}

static int fill_info_minmax_basics(
	struct nn_graph *nn,
	struct nn_node *self,
	sn16b_info *info)
{
	/* Pull out the inputs we need */
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	//const struct tensor *bias_min_tensor = self->inputs[8];
	//const struct tensor *bias_max_tensor = self->inputs[9];

	/* Get min/max values for input, weights, and bias data */
	float in_min_float = tensor_get_float(min_in_tensor, 0);
	float in_max_float = tensor_get_float(max_in_tensor, 0);

	if (!flt_isfinite(in_min_float) || !flt_isfinite(in_max_float)) {
		return errlog(nn, "input range to supernode, not finite");
	}
	in_max_float = fmaxf(in_max_float, in_min_float + 0.00001f);
	//float filt_min_float = tensor_get_float(min_filt_tensor,0);
	//float filt_max_float = fmaxf(tensor_get_float(max_filt_tensor,0),filt_min_float+0.00001f);
	//float bias_min_float = tensor_get_float(bias_min_tensor,0);
	//float bias_max_float = tensor_get_float(bias_max_tensor,0);

	/* find zero offset for each input */
	int32_t input_offset = quantize_uint16(0.0f, in_min_float, in_max_float);

	/* Find level size for each input */
	float in_level_size = (in_max_float - in_min_float) / 65535.0f;

	// filter level (already compensated for any scaling done)
	float filt_level_size = info->weights_level_size;

	//float bias_level_size = (bias_max_float - bias_min_float) / 65535.0f;

	/* The product level size is the product of the input and filter level size */
	float prod_level_size = in_level_size * filt_level_size;

	/* Calculate conversion ratio from bias to product space */
	//float bias_to_prod_ratio = (bias_level_size / prod_level_size);
	/* What is the value of the output minimum in the product space? */
	/* We need to add it to the products to move the smallest valid value to zero */
	//float min_out_prod_offset = -info->out_minval / prod_level_size;

	int32_t maxsum = fast_roundf((info->out_maxval - info->out_minval) / (float)(1<<16) / prod_level_size);
	//uint32_t recip_shamt = 0;
	//uint64_t recip_val_64 = 0x7F80000000ULL / maxsum;  //255 << 31

	//maxsum += 1; // remove.failed seed 774

	info->prod_level_size = prod_level_size;
	info->max_valid_val = fast_roundf((info->out_maxval - info->out_minval) / (float)(1 << 16) / prod_level_size);

	info->in_max_float = in_max_float;
	info->in_min_float = in_min_float;

	info->in_offset = input_offset;
#if 0
	while (recip_val_64 >= 0x80000000ULL) {
		recip_shamt++;
		recip_val_64 = 0x7F80000000ULL / (maxsum << recip_shamt);
	}
	info->recip_val = recip_val_64;
	info->recip_shamt = recip_shamt;
#endif
	int32_t recip_shift = 61 - Q6_R_normamt_R(maxsum);
	if (recip_shift > 46) recip_shift = 46;
	info->recip_shamt = recip_shift;
	int64_t denom = 0xffffLL << (recip_shift - 16);
	info->recip_val = (int32_t)(denom / maxsum);
	return 0;
}

/*
 * EJP: XXX: FIXME: We allocate enough for rounded up computation depth, but here we write full info->out_depth
 */
static int fill_bias_buf(
	struct nn_graph *nn,
	struct nn_node *self,
	sn16b_info *info,
	int bias32,
	int64_t extra)
{
	const struct tensor *bias_tensor = self->inputs[7];
	const struct tensor *bias_min_tensor = self->inputs[8];
	const struct tensor *bias_max_tensor = self->inputs[9];
	float bias_min_float = tensor_get_float(bias_min_tensor, 0);
	float bias_max_float = tensor_get_float(bias_max_tensor, 0);
	int32_t bias_offset = bias32 ? 0 : quantize_uint16(0.0f, bias_min_float, bias_max_float);
	float bias_denom = bias32 ? 0x1.0p32f : 65535.0f;
	float bias_level_size = (bias_max_float - bias_min_float) / bias_denom;
	const uint16_t *bias8_ptr = bias_tensor->data;
	const int32_t *bias32_ptr = bias_tensor->data;
	float bias_to_prod_ratio = (bias_level_size / info->prod_level_size);
	float min_out_prod_offset = -info->out_minval / info->prod_level_size;
	int32_t bias_depth = bias_tensor->shape.depth;
	int i;
	int32_t biasval;
	float bias_fval;
	float minout_bias_fval;
	int32_t gemsumb_val;
	int64_t final;
	logmsg(nn, 3, "in_offset=%d bias_levelsize=%f prod_level_size=%f ratio=%f", info->in_offset, bias_level_size, info->prod_level_size, bias_to_prod_ratio);
	for (i = 0; i < info->out_depth_valid; i++) {
		if (i >= bias_depth) biasval = bias_offset;
		else if (bias32) biasval = bias32_ptr[i];
		else biasval = bias8_ptr[i];
		bias_fval = (biasval - bias_offset) * bias_to_prod_ratio;
		minout_bias_fval = bias_fval + min_out_prod_offset;
		gemsumb_val = info->gemsumb[i];
		final = -(int64_t)gemsumb_val * info->in_offset + (int64_t)(minout_bias_fval+0.5f) + extra;
		logmsg(nn, 3, "i=%d biasval%d=%d fval=%f minout_fval=%f gemsumb_val=%d extra=%d final=%d",
			i, bias32 ? 32 : 8, biasval, bias_fval, minout_bias_fval, gemsumb_val, extra, final);
		info->biasbuf[i] = final >> 16;
	}
	return 0;
}

static void note_chunk_checkpoint_arrival(struct nn_graph *nn, struct nn_node *self, void *opaque)
{
	sn16b_info *info = self->opaque;
	struct weight_slice_info *myslice = opaque;
	//long idx = myslice - info->conv_slices;
#if 1
	logmsg(nn, 2, "chunk checkpoint did %d... copy %d bytes %p --> %p, wakeup %d @ offset %d alldone=%d/%d",
		myslice->checkpoint.required,
		myslice->copy_size, myslice->copy_in, myslice->copy_out,
		myslice->wakeup_items, myslice->batch_start_offset,
		info->alldone_checkpoint.count, info->alldone_checkpoint.required);
#endif
	//if (myslice->copy_size > 0) supernode_do_memcpy(nn, myslice->copy_out, myslice->copy_in, myslice->copy_size);
	supernode_execute_some_strategy(self, nn, info->batch_start_idx + myslice->batch_start_offset, myslice->wakeup_items);
	nn_checkpoint_arrival(nn, self, &info->alldone_checkpoint);
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
 * * Any preproc (im2col) or padding adjustments
 * * Add work_for_vector items
 * * Wait for workers to complete
 * * Any fixup / postproc
 *
 * * The work function is passed to work_for_vector.  It can be different for different architectures.
 * * The strategy for partitioning could be a set of functions.
 */
static int supernode_recalculate_strategy(struct nn_node *self, struct nn_graph *nn)
{
	/* Pull out the inputs we need */
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *min_in_tensor = self->inputs[2];
	const struct tensor *max_in_tensor = self->inputs[3];
	//const struct tensor *min_filt_tensor = self->inputs[4];
	//const struct tensor *max_filt_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	//const struct tensor *bias_tensor = self->inputs[7];
	//const struct tensor *bias_min_tensor = self->inputs[8];
	//const struct tensor *bias_max_tensor = self->inputs[9];
	/* Find the output tensors */
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	/* Structures with auxillary information */
	sn16b_info *info = self->opaque;
	/*
	 * Find the dimensions of the input data,
	 * both dimensions of data as well as padding
	 */
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
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;

	/* Find the dimensions of the filter weights.  filt_batches == out_depth */
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	//int32_t filt_depth = filt_tensor->shape.filt_depth;

	/* Find the stride values */
	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;

	/* Calculate output dimensions */
	int32_t out_batches = in_batches;
	int32_t out_depth = filt_batches;

	int32_t required_w_before, required_w_after;
	int32_t required_h_before, required_h_after;

	/* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */

	int32_t out_width = nn_pad_compute_outsize_and_pad(in_width, filt_width, stride_width, self->padding,
		&required_w_before, &required_w_after);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height, filt_height, stride_height, self->padding,
		&required_h_before, &required_h_after);

	// total number of inputs on each row actually needed (including the required_w before and after).
	// this is normally the same as required_w_before + in_width + required_w_after
	//  but it may be a bit less, when required_w_after = 0 and stride_width >1
	//
	int32_t required_w_total = filt_width + (out_width - 1)*stride_width;

	//logmsg(nn,1, "h: %ld(%ld)%ld  w %ld(%ld)%ld\n", required_h_before, out_height, required_h_after,
	//		required_w_before, out_width, required_w_after);
	/* Set up pointers to bulk data */
	uint8_t *in = info->din_e;
	//uint8_t *filt = filt_tensor->data;
	//uint8_t *bias = bias_tensor->data;
	uint8_t *out = info->dout_e;
	//int32_t *bias32_ptr = bias_tensor->data;

	/* Get min/max values for input, weights, and bias data */
	float in_max_float = tensor_get_float(max_in_tensor, 0);
	float in_min_float = tensor_get_float(min_in_tensor, 0);
	//float filt_max_float = tensor_get_float(max_filt_tensor,0);
	//float filt_min_float = tensor_get_float(min_filt_tensor,0);
	//float bias_min_float = tensor_get_float(bias_min_tensor,0);
	//float bias_max_float = tensor_get_float(bias_max_tensor,0);

	/* find zero offset for each input */
	int32_t input_offset = quantize_uint16(0.0f, in_min_float, in_max_float);
	int32_t filt_offset = info->filt_offset;
	//int32_t bias_offset = quantize_uint(0.0f,bias_min_float,bias_max_float);

	/* Find level size for each input */
	float in_level_size = (in_max_float - in_min_float) / 65535.0f;
	/* level size for weights, precomp for any scaling done */
	float filt_level_size = info->weights_level_size;
	//float bias_level_size = (bias_max_float - bias_min_float) / 65535.0f;

	/* The product level size is the product of the input and filter level size */
	float prod_level_size = in_level_size * filt_level_size;

	/* Calculate conversion ratio from bias to product space */
	//float bias_to_prod_ratio = (bias_level_size / prod_level_size);
	/* What is the value of the output minimum in the product space? */
	/* We need to add it to the products to move the smallest valid value to zero */
	//float min_out_prod_offset = -info->out_minval / prod_level_size;

	/*
	 * Set output padding values to sensible defaults.
	 * FUTURE WORK: find optimized padding values instead of default ones
	 */
	int32_t out_right_pad = ((-out_width) & 3);
	int32_t out_left_pad = 4;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;

		//int32_t filt_depth_total = in_depth_total;
		//int32_t filt_batches_total = out_depth_total;

		/* How much storage for each frame in the batch? */
	int32_t input_batch_size = in_height_total * in_width_total * in_depth_total;
	int32_t output_batch_size = out_height_total * out_width_total * out_depth_total;

	/*
	 * If we are striding, we need to ensure that the total left padding is compatible with
	 * the filter width / stride combination, otherwise we won't be hitting the right pixels exactly.
	 */

	 /* Where does our output data start? */
	uint8_t *out_data_start = out + out_top_pad * out_width_total * out_depth_total + out_left_pad * 32;

	/* What is the maximum value in product space? */
	/* Maybe it should be (info->out_maxval - info->out_minval) / prod_level_size... */
	int32_t maxsum = fast_roundf((info->out_maxval - info->out_minval) / prod_level_size);
	uint32_t recip_val = 0x7F80000000ULL / maxsum;  //255 << 31
	uint32_t recip_shamt = 0;

	int num_out_slices = (out_depth + 31) / 32;

	/* Grab some precomputed weight size information */
	//int n_weight_batches = info->n_weight_batches;
	int weight_batch_size = info->weight_batch_size;

	//int i;
	//int d,d2,b,t;

	//uint32_t maxval_leading_zeros;

	//int32_t input_skip_lines;

	struct workitem work = { 0 }; // klocwork
	//struct workitem waitwork = work;
	//struct workitem copywork = work;
	struct workitem startwork = work;
	//int workidx = 0;
	//int32_t tmpval32;


	logmsg(nn, 1, "Supernode %x: Recalculating Strategy...", self->node_id);
	//logmsg(nn,0,"Weight batch size: %d. Per %d: %d",nn->vtcm_size,weight_batch_size,nn->vtcm_size/weight_batch_size);

	/* Some sanity checks... */
	if (num_out_slices <= 0) return errlog(nn, "no out depth to iterate?");
	if (((in_width_total) % 4) != 0) return errlog(nn, "width fail");
	if ((in_depth_total % 32) != 0) return errlog(nn, "depth fail");
	if (((out_width_total) % 4) != 0) return errlog(nn, "width math fail");
	if ((out_depth_total % 32) != 0) return errlog(nn, "depth math fail");
	if (in_depth_before_pad != 0) return errlog(nn, "depth before pad not supported");


	// check all the padding requirements:
	//   (1) required_w_before  <= in_left_pad
	//   (2) required_w_after  <= in_right_pad+in_left_pad
	//     (because the  right side can use the left-padding of the following row).
	//      Condition (2) can fail when (1) passes, if you have an even-width filter and the right-side
	//      padding is 0.
	//  NOTE: if required_w_after > in_right_pad, it means that right-side
	//  output pixels are using left-padding on the next line, and the bottom right corner
	//  will need one extra row of bottom padding to support it.
	//
	// and
	//  (3) required_h_before <= in_top_pad
	//  (4) required_h_after + extra_bottom_row <= in_bottom_pad
	//   (where 'extra_bottom_row' is 1 if the right edge overlap mentioned above is needed.


	/* EJP: hopefully handle arbitrary in left pad... */
	if (in_left_pad != 4) logmsg(nn, 2, "Caution: left pad == %d", in_left_pad);

	// 1 if an extra bottom row is needed to supply RHS padding in the last input row.
	int32_t extra_bottom_row = 0;
	if (required_w_after > in_right_pad) {
		extra_bottom_row = 1;
		// However when the vertical stride > 1, and the required_h_after = 0, it's possible
		// that we don't really need an extra bottom row, since in some cases the
		// last input row is itself not used.
		// Check for that, to avoid unnecessary requirement for extra padding.
		//
		if (stride_height > 1 && required_h_after == 0) {
			// number of input rows needed to generate output (including padding)
			int h_needed = filt_height + (out_height - 1)*stride_height;
			if (h_needed < required_h_before + in_height) {
				// we have at least one superfluous bottom row, so don't need
				// to artificially add padding.
				extra_bottom_row = 0;
			}
		}
	}

	int32_t required_h_after_augmented = required_h_after + extra_bottom_row;

	if (required_w_before > in_left_pad || required_w_after > (in_right_pad + in_left_pad)) {
		return errlog(nn, "insufficient W padding: %d(%d)%d, filter needs %d()%d\n",
			(int)in_left_pad, (int)in_width, (int)in_right_pad, (int)required_w_before, (int)required_w_after);
	}
	if (required_h_before > in_top_pad || required_h_after_augmented > in_bottom_pad) {
		return errlog(nn, "insufficient H padding: %d(%d)%d, filter needs %d()%d+%d\n",
			(int)in_top_pad, (int)in_height, (int)in_bottom_pad,
			(int)required_h_before, (int)required_h_after, (int)extra_bottom_row);
	}


	//logmsg(nn,0,"maxsum=0x%x recip_val=0x%x shamt=%d",maxsum,recip_val,recip_shamt);
	if (recip_val & 0x80000000U) logmsg(nn, 1, "***** reciprocal is negative if signed, ??problem??");

	/* Compute reciprocal and shift amount */
	logmsg(nn, 2, "out_maxval=%f in_level_size=%f filt_level_size=%f prod_level_size=%f maxsum ~= %d",
		info->out_maxval,
		in_level_size,
		filt_level_size,
		prod_level_size,
		(int)maxsum);

	logmsg(nn, 2, "in: %p h/w/d: %d/%d/%d total h/w/d: %d/%d/%d first valid row: %p first valid data: %p",
		in,
		in_height, in_width, in_depth,
		in_height_total, in_width_total, in_depth_total,
		in + (in_depth_total*in_width_total*in_top_pad),
		in + (in_depth_total*in_width_total*in_top_pad) + (in_depth_total*in_left_pad));

	if (tensor_out_prepare_padded_d32(out_tensor,
		out_batches,
		out_height, out_top_pad, out_bottom_pad,
		out_width, out_left_pad, out_right_pad,
		out_depth, out_depth_before_pad, out_depth_after_pad,
		NN_TYPE_QINT16) != 0) {
		return errlog(nn, "output tensor prep fail");
	}

	/*
	 * Update info / supernode info values
	 * These values are stored in structures for use during normal execution
	 */

	info->prod_level_size = prod_level_size;
	info->max_valid_val = (int64_t)((info->out_maxval - info->out_minval) / prod_level_size) >> 16;
	info->min_valid_val = 0;

	info->in_max_float = in_max_float;
	info->in_min_float = in_min_float;

	info->recip_val = recip_val;
	info->recip_shamt = recip_shamt;

	info->in_offset = input_offset;

	info->in_next_d32 =
		tensor_location_16b_d32(in_tensor, 0, 0, 0, 32) -
		tensor_location_16b_d32(in_tensor, 0, 0, 0, 0);
	info->in_next_row =
		tensor_location_16b_d32(in_tensor, 0, 1, 0, 0) -
		tensor_location_16b_d32(in_tensor, 0, 0, 0, 0);
	info->in_next_batch =
		tensor_location_16b_d32(in_tensor, 1, 0, 0, 0) -
		tensor_location_16b_d32(in_tensor, 0, 0, 0, 0);
	info->in_width = info->in_next_d32 / 32;
	info->in_depth = info->in_next_row / info->in_width;

	info->out_width_processed = out_width;

	info->out_next_d32 =
		tensor_location_16b_d32(out_tensor, 0, 0, 0, 32) -
		tensor_location_16b_d32(out_tensor, 0, 0, 0, 0);
	info->out_next_row =
		tensor_location_16b_d32(out_tensor, 0, 1, 0, 0) -
		tensor_location_16b_d32(out_tensor, 0, 0, 0, 0);
	info->out_width = info->out_next_d32 / 32;
	info->out_depth_total = info->out_next_row / info->out_width;	// includes all padding. 
	info->out_depth_valid = out_depth + out_depth_after_pad;	// know how much we'll compute

	info->stride_height = stride_height;
	info->stride_width = stride_width;


	// input_base0, in_height:  this omits any top/bottom padding rows that we don't need

	info->input_base0 = in + (in_top_pad - required_h_before)*info->in_next_row;
	info->in_height = in_height + required_h_before + required_h_after;
	info->weights_base = info->weights;

	/*
	 * If we are striding, we need to ensure that the total left padding is compatible with
	 * the filter width / stride combination, otherwise we won't be hitting the right pixels exactly.
	 */
	 /* If we did normal convolution, how many junk values would we have on the left? */
	info->out_left_junk = (in_left_pad - required_w_before) / info->stride_width;
	/* Do we need to skip an element on the input so that we stride to the right starting point? (covers even/odd leftover from out_left_junk) */
	info->in_left_skip = in_left_pad - (info->out_left_junk * info->stride_width + required_w_before);
	if (info->in_left_skip < 0) return errlog(nn, "wrong in left skip");

	/* Is skip_col the same as in_left_skip? */
		//info->skip_col = ((out_width & 3) <= ((4-out_left_junk)&3) && (out_width & 3)!=0);
	info->skip_col = info->in_left_skip;

	info->out_height = out_tensor->shape.height;
	info->filt_width = filt_width;
	info->filt_height = filt_height;

	if (fill_info_minmax_basics(nn, self, info) != 0)
		return -1;
	//fill_info_dim_basics(nn,self,info);

	/*
	 * Recompute bias values
	 * We need to incorporate the input bias values, min offset, and gemsumb values into the bias buffer
	 * The bias buffer is added in the "product" (in_stepsize * filt_stepsize) number system
	 */

	int bias32 = (self->node_type == OP_Supernode_8x8p32to8_d32);
	int64_t bias_extra = (int64_t)info->in_depth*input_offset*filt_offset*filt_width*filt_height;
	//if (!info->use_v65 && filt_height == 1 && filt_width == 1 && ENABLE_FASTSUMA_1x1) bias_extra = info->in_depth*input_offset*filt_offset;
	logmsg(nn, 2, "in_depth_total=%d input_offset=%d filt_offset=%d bias_extra=%d", in_depth_total, input_offset, filt_offset, bias_extra);
	fill_bias_buf(nn, self, info, bias32, bias_extra);

#if 0
	/* Compute bias values */
	//supernode_biasbuf_recalc(nn,info);
	int bias32 = (self->node_type == OP_Supernode_8x8p32to8_d32);
	if (!bias32) {
		if (bias_max_float > 0x1p30f * prod_level_size) return errlog(nn, "bias mag too big");
		if (-bias_min_float > 0x1.0p30f * prod_level_size) return errlog(nn, "bias mag too big");
	}
	else {
		bias_offset = 0;
		bias_to_prod_ratio *= 0x1.0p-24f;
	}
	for (i = 0; i < info->out_depth; i++) {
		int32_t biasval = bias[i];
		if (bias32) biasval = bias32_ptr[i];
		float bias_fval = ((biasval - bias_offset) * bias_to_prod_ratio);
		bias_fval += min_out_prod_offset;
		if (i >= out_depth) bias_fval = 0.0f;
		/* If necessary, add GEMSUMB related values here */
		int32_t gemsumb_val = supernode_gemsumb(
			filt,
			filt_height,
			filt_width,
			filt_depth,
			filt_depth_total,
			filt_batches,
			input_offset,
			filt_offset,
			i);
		//gemsumb_val += filt_height*filt_width*in_depth*in_offset*filt_offset;
		logmsg(nn, 3, "gemsumb[%d]=%d bias=%f total=%d", i, gemsumb_val, bias_fval, (int32_t)(bias_fval + 0.5f - gemsumb_val));
		/* Add the minimum output value; 0 if followed by relu */
		//info->biasbuf[i] = -gemsumb_val - bias_fval + 0.5f;
		tmpval32 = fast_roundf(bias_fval);
		info->biasbuf[i] = tmpval32 - gemsumb_val;
		//logmsg(nn,1,"biasval @ %d: (%d - %d) * %f --> %d",i,biasval,bias_offset,bias_to_prod_ratio,info->biasbuf[i]);
		//logmsg(nn,0,"biasbuf[%d]=%d",i,info->biasbuf[i]);
	}
#endif

	/*
	 * Recompute weights
	 * The weights need to be arranged so that they are in a depth32 compatible format.
	 * We keep note that we've rearranged the weights so we only do this once.
	 * For architectures that support signed weights, we convert weights to signed
	 * FIXME: maybe we should move this to check time instead of recalculation time
	 */
	 //FIXME: RECALC / REARRANGE WEIGHTS
#if 0
	//if (filt_depth % 32) return errlog(nn,"FIXME: in depth mult 32");
	//if (filt_batches % 32) return errlog(nn,"FIXME: out depth mult 32");
	if (info->weights_arranged == 0) {
		supernode_rearrange_for_d32(
			info->weights, filt,
			filt_height,
			filt_width,
			filt_depth,
			filt_depth_total,
			filt_batches,
			filt_batches_total,
			filt_offset);
		supernode_convert_weights_to_signed(
			info->weights,
			filt_height,
			filt_width,
			filt_depth_total,
			filt_batches_total,
			filt_offset);
		supernode_cleaninv_weights(
			info->weights,
			filt_height*filt_width*filt_depth_total*filt_batches_total);
		info->weights_arranged = 1;
	}
#endif

	/*
	 * Prepare output tensors
	 */
	if (tensor_out_prepare_normal(out_min, 1, 1, 1, 1, NN_TYPE_FLOAT) != 0) {
		return errlog(nn, "min out prep fail");
	}
	if (tensor_out_prepare_normal(out_max, 1, 1, 1, 1, NN_TYPE_FLOAT) != 0) {
		return errlog(nn, "max out prep fail max_size=%d @ %p", out_max->max_size, &out_max->max_size);
	}
	tensor_set_float(out_min, 0, info->out_minval);
	tensor_set_float(out_max, 0, info->out_maxval);

	/*
	 * Overcompute Reduction
	 * We may have too much input padding.  If we can avoid computing it, that's great!
	 * We should make this very robust so we can have truly arbitrary padding
	 */

	 //#ifndef HEXAGON_V66
#if 1
	/*
	 * In V60-V65, we should just be able to move the pointer to
	 *   (b,-required_h_before,required_w_before,0)
	 * since there is no strict vector alignment requirement for the activation (for this node...)
	 */
	 // maybe info->in_width = in_width + required_w_before + required_w_after;
	info->out_width = out_tensor->shape.width + out_tensor->format.width_pad[1];
	info->out_left_junk = 0;
	if (info->use_v65) {
		int naccs = 8;		// default, multiples of 8
		//if(((out_width+7)%8) < (2+((out_width+5)%6))) { //too general for now
		if (info->out_width >= 11 && info->out_width <= 36) {
			// use multiples of 6?
			//big gains for iv3
			// use 6 for 11,12,17,18,35,36
			// 9,10,33,34 would be good too but currently asm needs last group to be 5 or 6
			int div6 = (info->out_width + 5) / 6u;	// # of groups of 6
			int pad6 = div6 * 6 - info->out_width;	// amount to pad to reach mul of 6
			if (pad6 <= 1 && (info->out_width <= 18 || info->out_width >= 35)) {
				naccs = 6;
				info->out_width = div6 * 6;
			}
		}
		info->num_accs = naccs;

		//v65 chops off what it doesn't want and puts it in a buffer
		info->in_left_skip = in_left_pad - required_w_before; //1,2,3 or 4
		int edge = (in_left_pad - required_w_before) & ~3;
		info->input_base = info->din_e + (in_top_pad-required_h_before)*info->in_next_row + (in_left_pad+edge-required_w_before)*32;
	}
	else {
		// this may not be vector aligned, but it is 32 aligned: it points to the first h-padding unit we need
		info->input_base = info->din_e + (in_top_pad-required_h_before)*info->in_next_row + (in_left_pad-required_w_before)*32;
		info->in_left_skip = 0;
	}
	info->skip_col = info->in_left_skip;
	//out_data_start = tensor_location_16b_d32(out_tensor, 0, 0, 0, 0);
	logmsg(nn, 2, "now out_width=%d out_left_junk=%d out_data_start=%p", info->out_width, info->out_left_junk, out_data_start);
	//	info->suma_start = 3 + in_left_pad - required_w_before;
#else

	info->suma_start = (required_w_before & (stride_width >> 1)) + 3;

	/* If we don't require any left padding, we can skip it */
	//return errlog(nn,"FIXME: left padding");
	/*
	 * EJP: move this (maybe) and refactor
	 *
	 * Try and skip over input left padding if possible
	 *  - Adjust input pointer right by a vector
	 *  - Decrease in width
	 *  - Now we really need to recompute out_width / out_left_junk / in_left_skip / pointers
	 * So this maybe should be moved up.
	 * Perhaps a better strategy would be to reconsider as:
	 *  - For this architecture, how do I adjust the input pointers/values?
	 *  - Based on those values, how do I set output pointers/values?
	 */
	if ((required_w_before == 0) && (in_left_pad == 4)) {
		info->input_base += 128;
		info->in_width -= 4;
		/* FIXME: this is wrong if we have 1x1 filters with 2x stride or something maybe? */
		info->out_width -= 4;
		info->out_left_junk = 0;
		info->in_left_skip = 0;
		info->skip_col = info->in_left_skip;
		out_data_start += 128;
		info->suma_start += 4;
	}
#endif

	/* Skip initial padding lines */
	//input_skip_lines = in_top_pad-required_h_before;
	//info->input_base += info->in_next_row*input_skip_lines;


	/*
	 * Preparing the work list
	 * We create a work list, which is just a list of items for the main thread to do
	 * We need to put in all the things we need to do to execute the node:
	 * * Padding zapping
	 * * GEMSUMA computation
	 * * Actual convolution
	 * * l2fetch / memcpy of values
	 * * Waiting on semaphores for things to finish
	 * The work list is currently executed by the main thread
	 * This means that vector work needs to be passed to nn_os_work_for_vector
	 */

	supernode_softreset_work_items(self, nn, info);

	/*
	 * Set up work items
	 */

#if 0
	 /* FIXME for v66: copy work has to actually work. */
	copywork.execute = supernode_execute_workitem_copy;
	copywork.info = info;
	copywork.self = self;
#endif

	//waitwork.execute = supernode_execute_workitem_join_some;

	work.info = info;
	work.self = self;
	//work.input = info->input_base;
	//work.output  = out;
	//work.weights = filtdst;
	work.stop_line = info->out_height;
	work.skip_lines = 1;
	//work.new_execute = supernode_execute_workitem_vector_dispatch;	// FIXME, pick the right function
	work.new_execute = supernode_execute_hvx_conv_work;	// FIXME, pick the right function

#if 0
	if (required_h_before < in_top_pad) work.zap_top_size = required_h_before + 1; //allow for integration
	else work.zap_top_size = required_h_before;

	work.zap_bot_size = required_h_after_augmented;
	work.zap_right = tensor_location_d32(in_tensor, 0, 0, in_width, 0);
	work.zap_left = tensor_location_d32(in_tensor, 0, 0, -in_left_pad, 0);
	work.zap_left_size = in_left_pad * 32;
	work.zap_right_size = in_right_pad * 32;  // DJH just keep it clean
	work.zap_height = in_height;

	/*
	 * If we have padding zap work to do, add it to the list
	 * Some extra zapping happens due to getting GEMSUMA to work, perhaps it could be optimized
	 * FIXME: maybe just set zap_right and zap_right_size and let it be on
	 * the next row if enough left pad exists.
	 */
	if ((out_right_pad > 0) && (work.zap_left_size == 0)) {
		if (out_right_pad > in_left_pad) return errlog(nn, "oops, not enough left pad");
		/* EJP: FIXME: this probably doesn't work if zap_right_size goes byond the vector size */
	}
#else
	startwork.self = self;
	startwork.info = info;
	// top/bottom zapping based on entire rows (all padding) so they use input_base0.
	startwork.zap_top = (uint8_t *)info->input_base0;
	startwork.zap_bot = (uint8_t *)info->input_base0
		+ (required_h_before + in_height)*info->in_next_row;

	startwork.zap_top_size = info->in_next_row * required_h_before;
	startwork.zap_bot_size = info->in_next_row * required_h_after_augmented;

	uint8_t *z0 = info->din_e + in_top_pad * in_width_total*in_depth_total + in_left_pad * 32; // first real pixel on first	
	startwork.zap_left = z0 - 32 * in_left_pad;					// start of left padding on first row
	startwork.zap_right = z0 + 32 * in_width;					// start of right padding on first row
	// do not zap beyond the right padding
	// EJP: are there other cases where we need to zap right pad?
	startwork.zap_right_size = Q6_R_min_RR(in_right_pad, required_w_after);
	startwork.zap_rl_depths = in_depth_total / 32;
	startwork.zap_oedelta = info->in_oedelta;

	// zap all of the left padding if
	// (a) any left padding is needed, or
	// (b) the req. for right padding exceeds the actual right padding (so left is needed for that)
	if (required_w_before != 0 || required_w_after > in_right_pad) {
		startwork.zap_left_size = in_left_pad;
	}
	else {
		startwork.zap_left_size = 0;
	}
	startwork.zap_height = in_height;
	startwork.zap_value = input_offset;
	logmsg(nn, 2, "padding=%d required_w_before: %d required_w_after: %d required_h_before: %d required_h_after: %d", self->padding, required_w_before, required_w_after, required_h_before, required_h_after);

	/*
	 * If we have padding zap work to do, add it to the list
	 * Some extra zapping happens due to getting GEMSUMA to work, perhaps it could be optimized
	 * FIXME: maybe just set zap_right and zap_right_size and let it be on
	 * the next row if enough left pad exists.
	 */
	 /* GLS: leaving this stuff in; it may be needed to avoid contaminating min/max in output width padding */
	if ((out_right_pad > 0) && (startwork.zap_left_size == 0)) {
		//if (out_right_pad > in_left_pad) return errlog(nn,"oops, not enough left pad");
		/* EJP: FIXME: this probably doesn't work if zap_right_size goes byond the vector size */
		startwork.zap_left_size = in_left_pad;
		startwork.zap_right_size = in_right_pad;
	}
	//#ifndef V66
#if 1
	/* EJP: this is extra padding for integral/suma, but maybe it's no longer needed? */
	if (startwork.zap_left_size == 0) {
		startwork.zap_left_size = in_left_pad;
	}
	if (0)if (startwork.zap_top_size == 0 && in_top_pad > 0) {
		startwork.zap_top_size = info->in_next_row;
		startwork.zap_top = (uint8_t *)info->input_base0 - info->in_next_row;
	}
#endif
#endif
	work.zap_rl_depths = in_depth_total / 32;
	work.zap_value = input_offset;

	/*
	 * Slicing was a good first attempt, but needs refactoring.
	 * If all the weights fit in cache easily, load them all and do all the depths for a chunk of activation.
	 * Else, if the activation fits in cache easily, do a d32 output at a time
	 * Else, find some number of weights and some amount of activation that both fit
	 *   Traffic ~= activation_chunk_size * weight_slices + weight_chunk_size * activation_slices
	 *   ... but maybe not quite so simple.  Want to maximize use of locality
	 *   Maybe need outer_activation, outer_weights, inner_activation, inner_weights nested loops
	 * What's the right optimization technique?
	 *  Weights have to be at least 32*h*w*in_depth
	 *  Probably need at least a few rows of activation
	 *  Weight chunks outer, activation inner --> more activation BW
	 *  Activation chunks outer, weight chunks inner --> more weight BW
	 *  Looking at the literature, it looks like it's a non-trivial problem?
	 * If we have a huge activation, we want to favor activation locality.
	 * If we have huge weights, we want to favor weight locality.
	 * So... our constraints are:
	 *  * Minimum effective weights size (d32 chunk size), granularity
	 *  * Minimum effective activation size (currently some rows, but we could subdivide further...), granularity
	 *  * Weights + Activation + prefetching room < cache_size
	 *  * Either Weights or Activations needs to have multiple chunks for multithreading!
	 * Given those things, try and find something that matches the activation/weight size ratio.
	 *   Start with minimum amounts of activation/weights
	 *   Add to activation/weight size to try and get closer to the ratio, see if it's still cache-friendly
	 *   If we can't add the thing that makes the ratio better, try and add a little more of the other one
	 * Once we have activation/weight chunk sizes, how should we proceed to the next chunk of work?
	 *   Probably the one that needs less bandwidth immediately.
	 *   We will need to iterate over all the weight and activation chunks, but the instantaneous BW requirements are lower if going to the next small thing.
	 * Need to be aware of threading...
	 *   Let's start out with always threading by dividing the activation, easier to handle suma/integral/zapping
	 * So.......
	 * for (outer_weights) {
	 *   for (outer_activations) {
	 *     for (inner_weights) {
	 *       for (inner_activations) {
	 *       }
	 *     }
	 *   }
	 * }
	 * If weights_fit: outer_weights = 1, outer_activations = n_act_chunks, inner_weights = all, inner_activations=chunksize/threads
	 * else if activations_fit: outer_weights = 1, outer_activations = 1, inner_weights = all, inner_activations=all/threads
	 * else outer_weights = n_weight_chunks, outer_activations = n_act_chunks, inner_weights = weight_chunksize, inner_activations=chunksize/threads
	 *
	 * TBD: should we consider batches just part of the activation size?
	 *   * Probably, but it's hard to index and iterate over.
	 *   * Maybe split above loop activations into outer_batches, outer_rows
	 */

	 /* Determine how we want to slice the work */
		 //int32_t weight_batch_size0 = (info->out_depth_slice == 64 && info->out_depth % 64) ? weight_batch_size : weight_batch_size*2;
	 //int32_t batch_slice_factor;
	 //int32_t height_slice_factor;
	 //int32_t weight_slice_factor;
	int32_t inner_act_batches;
	int32_t inner_act_rows;
	int32_t inner_weight_chunks;
	//luc
//	logmsg(nn,0,"opname=%s name=%s ",info_id2opname(self->node_id),info_id2name(self->node_id));

	slice_for_cache(
		self->node_id,
		nn,
		out_batches,
		in_depth_total / 32,
		info->in_width,
		//info->in_height,
		info->out_width,
		info->out_height,
		info->stride_height,
		info->filt_width,
		info->filt_height,
		weight_batch_size,
		num_out_slices,
		&inner_act_batches,
		&inner_act_rows,
		&inner_weight_chunks);

	work.num_lines = inner_act_rows;
	inner_act_rows = (out_height + NUM_THREADS - 1) / (NUM_THREADS);

	/* Ignore batch / weight slice factor for now ... */
	int32_t outer_act_batches = (out_batches + inner_act_batches - 1) / inner_act_batches;
	int32_t outer_act_iters = (out_height + inner_act_rows - 1) / inner_act_rows;
	int32_t outer_weight_chunks = (num_out_slices + inner_weight_chunks - 1) / inner_weight_chunks;

	// NOTE: outer_act_iters <= NUM_THREADS.

	logmsg(nn, 1, "batch/row/weight chks: inner=%d,%d,%d outer=%d,%d,%d out_height=%d out_depth_chunks=%d",
		inner_act_batches, work.num_lines, inner_weight_chunks,
		outer_act_batches, outer_act_iters, outer_weight_chunks,
		out_height,
		num_out_slices);

	/*-------------------------------------------------------------*/
	/*  Setup parameters and allocate scratch for SUMA computation */
	/*-------------------------------------------------------------*/
	int32_t scratch_size;		// in int32 units
	int32_t suma_buf_rowstride;	// in int32 units

	// sumabuf is allocated in scratch, for all batches
	int32_t *sumabuf_batch0 = NULL;
	int sumabuf_batch_stride = 0;	// in int32's

	if (!info->use_v65) {
		if (info->filt_width == 1 && info->filt_height == 1 && ENABLE_FASTSUMA_1x1) {
			// each row of the suma has
			//   (a) one slot for each 'unused' left-padding element (if any); 0..3
			//   (b) one slot for each input element which is used by the convolution
			//       including req. left & right (i.e. required_w_total )
			//   (c) padding to a multiple of 32.
			//  suma_start points to section (b).
			//
			info->suma_start = (in_left_pad - required_w_before) & 3;
			suma_buf_rowstride = roundup(info->suma_start + required_w_total, 32);
			scratch_size = 32;	 // not actually used; minimal size
		}
		else {
			// each row of the integral buffer will have (in i32 slots):
			//   (a) 8 zeros
			//   (b) one slot for each 'unused' left-padding element (if any); 0..3
			//   (c) one slot for each input element which is used by the convolution
			//       including req. left & right (i.e. required_w_total )
			//   (d) padding up to a multiple of 32.
			// The 'suma_start' points to the first element before section (c)
			// 'suma buf' has the same width, and the the first 'proper' output
			// will be at position 'suma_start'.
			//
			info->suma_start = 7 + ((in_left_pad - required_w_before) & 3);
			suma_buf_rowstride = roundup(info->suma_start + 1 + required_w_total, 32);
			info->suma_width = suma_buf_rowstride;
			// this is 4 rows for 'scratch_128xW' plus enough rows for the integral buffer
			int scratch_rows = 4 + (work.num_lines - 1)*info->stride_height + info->filt_height + 1;
			scratch_size = suma_buf_rowstride * scratch_rows + 32;
		}

		// allocate suma_scratch buffers,x NUM_THREADS
		uint32_t sumatmp_size = scratch_size * sizeof(int32_t);
		void * tbufp = nn_scratch_alloc(nn, sumatmp_size*NUM_THREADS);
		bufpool_init(&info->bufpool_suma_scratch, NUM_THREADS, tbufp, sumatmp_size);

		info->next_suma_off = suma_buf_rowstride * sizeof(int32_t);
		sumabuf_batch_stride = suma_buf_rowstride * info->out_height;	// size of 1 plane in int32's
		int32_t suma_buf_size = sumabuf_batch_stride * in_batches;	// size of whole buffer in int32's

		sumabuf_batch0 = nn_scratch_alloc(nn, suma_buf_size * sizeof(int32_t));
		if (sumabuf_batch0 == NULL) {
			return errlog(nn, "failed to get %d bytes for sumabuf", (int)(suma_buf_size * sizeof(int32_t)));
		}
	}

	int32_t semaphore_count = 0;
	int ow, ob, or , ib;


	int curr_suma_progress = 0;
	int batchstart_idx = 0;
	//if (hlines_per_slice <= NUM_THREADS) return errlog(nn,"OOPS: chopped too fine");
	for (ob = 0; ob < outer_act_batches; ob++) {
		for (ib = 0; ib < inner_act_batches; ib++) {
			int b = ob * inner_act_batches + ib;
			if (b >= in_batches)
				continue;
			work.suma_buf = sumabuf_batch0 + b * sumabuf_batch_stride;

#if 0
			/* l2fetch first weight chunk */
			logmsg(nn, 1, "adding l2fetch: %p %d %d", info->weights, weight_batch_size, inner_weight_chunks);
			// Add workitem: weights
			// EJP: merge into startwork
			supernode_weights(self, nn, info, b != 0, /*don't wait for previous in batch 0*/
				info->weights, weight_batch_size, info->use_v65 ? inner_weight_chunks : 1, outer_act_iters);
#endif
			/* Zap padding is back */
			//supernode_add_padding_zap(self,nn,info,zapwork,b*input_batch_size,required_h_before,required_w_before);
			curr_suma_progress++;	// advance per batch
	// Add workitem: padzap
	// EJP: make startwork
			info->startup_info.batch_start_offset = 0;
			if (info->use_v65) {
				uint32_t weights_total = inner_weight_chunks * weight_batch_size;
				if (unlikely(weights_total > (nn->vtcm_size - VTCM_CIRCBUF_SIZE))) return errlog(nn, "oops: v65 selected but weights too big");
				startwork.copy_in = info->weights;
				startwork.copy_out = nn->vtcm_ptr;
				startwork.copy_size = inner_weight_chunks * weight_batch_size;
				info->startup_info.wakeup_items = outer_act_iters;
				//if (b == 0) supernode_note_earlywork(nn,self,info,startwork.copy_out,startwork.copy_in,startwork.copy_size);
			}
			else {
				/* For V60, we need to finish the gemsuma before the rest can continue */
				startwork.pf_inp = info->weights;
				startwork.pf_width = startwork.pf_stride = weight_batch_size / 32;
				/* EJP: V60 code prefetches next chunk of weights automatically, just PF first chunk */
				//startwork.pf_height = 1 * 32;
				startwork.pf_height = inner_weight_chunks * 32;
				//info->startup_info.wakeup_items = outer_act_iters * outer_weight_chunks;
				info->startup_info.wakeup_items = outer_act_iters;
			}
			startwork.join_iters = outer_weight_chunks;
			if (b > 0) info->work_items[batchstart_idx].next_startup_offset = info->n_work_items - batchstart_idx - 1;
			int NOW_BATCHES = 1;
			batchstart_idx = supernode_add_batch_startup(self, nn, info, startwork, b*input_batch_size, required_h_before, required_w_before, NOW_BATCHES);

			for (ow = 0; ow < outer_weight_chunks; ow++) {

				int last_chunk_in_batch = (ow == (outer_weight_chunks - 1));
				int start_weights = ow * inner_weight_chunks;
				int now_chunks = Q6_R_min_RR(num_out_slices - start_weights, inner_weight_chunks);
				int32_t next_weight_chunks = inner_weight_chunks;
				int32_t max_next_weight_chunks = num_out_slices - start_weights - now_chunks;
				next_weight_chunks = Q6_R_min_RR(max_next_weight_chunks, next_weight_chunks);
				for (or = 0; or < outer_act_iters; or ++) {
					// work.suma_scratch = scratch[n_scratch++%2];

					int pf_outer_act = (or == (outer_act_iters - 1));
					//int pf_outer_act = (or == 0);
					int needs_next_outer_weights = pf_outer_act && (ow != (outer_weight_chunks - 1));

					int start_row = or *inner_act_rows;
					int n_rows = Q6_R_min_RR(out_height - start_row, inner_act_rows);


					/* FILL OUT NORMAL WORK INFORMATION */
					work.need_initialize_suma = (ow == 0);
					//work.suma_progress_need = curr_suma_progress;
					//work.suma_progress_index = or;
					const uint8_t *filtsrc = info->weights_base + start_weights * weight_batch_size;
					const uint8_t *filtdst = supernode_filtbuf_location(nn, info, ow, filtsrc, now_chunks*weight_batch_size);
					work.weights = filtdst;
					work.weight_chunks = now_chunks;
					work.input = info->input_base + b * input_batch_size;
					//work.output = tensor_location_d32(out_tensor, b, 0, 0, start_weights * 32);
					work.output = out_data_start+b*output_batch_size+start_weights*info->out_next_d32;
					work.biases = info->biasbuf + start_weights * 32;
					work.start_line = start_row;
					work.stop_line = work.start_line + n_rows;
					//work.do_conv = &info->semaphores[1];			// sem to wait for, before starting conv
					work.do_conv = NULL;
					work.conv_done = &info->semaphores[2];			// sem to post, after conv if not last_chunk (v65 only)
			//		work.copy_done = &info->semaphores[3];
					work.start_chunk = start_weights;
					work.last_chunk = last_chunk_in_batch && (b == in_batches - 1);

					if (needs_next_outer_weights) {
						work.pf_inp = filtsrc + now_chunks * weight_batch_size;
						work.pf_width = weight_batch_size;
						work.pf_stride = work.pf_width;
						work.pf_height = next_weight_chunks;

						logmsg(nn, 1, "or=%d ow=%d/%d set up weight pf ptr=%p width=%d height=%d",
							or , ow, outer_weight_chunks, work.pf_inp, work.pf_width, work.pf_height);
					}
					else {
						work.pf_inp = NULL;
						work.pf_width = 0;
						work.pf_stride = 0;
						work.pf_height = 0;
						logmsg(nn, 1, "or=%d ow=%d/%d no pf", or , ow, outer_weight_chunks);
					}

					work.donesem = &info->semaphores[0];
					semaphore_count++;
					// Add workitem: work
					supernode_add_work_item(self, nn, info, work);
				}//or
				memset(&info->conv_slices[start_weights], 0, sizeof(info->conv_slices[ow]));
				if ((ow == 0) && (!info->use_v65)) {
					/* For V60 code, wake up rest of items once first batch is done */
					/* This ensures ordering of gemsuma with other work items */
					info->conv_slices[ow].wakeup_items = (outer_weight_chunks - 1)*outer_act_iters;
					info->conv_slices[ow].batch_start_offset = (ow + 1)*outer_act_iters;
				}
				if (!last_chunk_in_batch && info->use_v65) {
					info->conv_slices[start_weights].copy_in = work.pf_inp;
					info->conv_slices[start_weights].copy_out = nn->vtcm_ptr;
					info->conv_slices[start_weights].copy_size = next_weight_chunks * weight_batch_size;
					info->conv_slices[start_weights].batch_start_offset = (ow + 1)*outer_act_iters;
					info->conv_slices[start_weights].wakeup_items = outer_act_iters;
					logmsg(nn, 1, "v65: Setting up copy %p-->%p %d bytes, %d items @ %d",
						info->conv_slices[start_weights].copy_in,
						info->conv_slices[start_weights].copy_out,
						info->conv_slices[start_weights].copy_size,
						info->conv_slices[start_weights].wakeup_items,
						info->conv_slices[start_weights].batch_start_offset);
				}
				nn_checkpoint_init(&info->conv_slices[start_weights].checkpoint, outer_act_iters, note_chunk_checkpoint_arrival, (void *)&info->conv_slices[start_weights]);
#if 0
				if (!last_chunk_in_batch && info->use_v65)
					// Add workitem: weights
					// Needs to be merged into work checkpoint work
					supernode_weights(self, nn, info, 1, /*wait for previous*/
						work.pf_inp, work.pf_width, work.pf_height, outer_act_iters);
#endif
			}//ow
		}//ib
	}//ob
//logmsg(nn,0,"semaphore_count / join_iters=%d",semaphore_count);

// 'semaphore_count is the # of convolves in the whole operation; number
// of donesem posts; this is used in the final join.
// However, for v65, the work thread only posts done_sem in units where last_chunk is set.
// (earlier ones are joined by the weight-copy ops).

//if( info->use_v65) semaphore_count = outer_act_iters;
//waitwork.join_iters = semaphore_count;
//waitwork.donesem = &info->semaphores[0];
// Add workitem: wait for shutdown
	//supernode_add_work_item(self,nn,info,waitwork);
#if 0
	/*
	 * If we have padding zap work to do, add it to the list
	 * Some extra zapping happens due to getting GEMSUMA to work, perhaps it could be optimized
	 * FIXME: maybe just set zap_right and zap_right_size and let it be on
	 * the next row if enough left pad exists.
	 */
	if ((out_right_pad > 0) && (work.zap_left_size == 0)) {
		if (out_right_pad > in_left_pad) return errlog(nn, "oops, not enogh left pad");
		work.zap_left = tensor_location_d32(in_tensor, b, start_row - required_h_before, -in_left_pad, 0);
		work.zap_left_size = in_left_pad;
	}


	/* EJP FIXME NOW: Set up work item for inner rows + inner batches */

			//l2fetch_inner_activation;
	if (ow == 0) {
		const uint8_t *filtsrc = info->weights_base + ow * weight_batch_size;
		filtdst = supernode_filtbuf_location(info, ow, filtsrc);
		logmsg(nn, 0, "copy first inner weight chunk... ow=%d ob=%d or=%d fsrc=%p fdst=%p",
			ow, ob, or , filtsrc, filtdst);
		// copy_first_inner_weight_chunk;
	}
	for (iw = 0; iw < inner_weight_chunks; iw++) {
		if ((iw != (inner_weight_chunks - 1)) || needs_next_outer_weights) {
			//logmsg(nn,0,"copy next inner weight chunk... ow=%d ob=%d or=%d iw=%d",ow,ob,or,iw);
			//copy_next_weight_chunk
		}
		for (ib = 0; ib < inner_act_batches; ib++) {
			for (ir = 0; ir < inner_act_rows; ir++) {
				//logmsg(nn,0,"maybe zap: ow=%d ob=%d or=%d iw=%d ib=%d ir=%d",ow,ob,or,iw,ib,ir);
				//logmsg(nn,0,"work item: ow=%d ob=%d or=%d iw=%d ib=%d ir=%d",ow,ob,or,iw,ib,ir);

				//if (weight_slice == 0) zap;
				//else disable_zap;
				//setup rest of work
				//supernode_add_work_item(self,nn,info,work);
			}
		}
	}
}
	}
	}
#endif
#if 0
	for (hslice = 0; hslice < height_slice_factor; hslice++) {
		const uint8_t *batch_input = info->input_base + b * input_batch_size;
		/*
		 * For the first slice, we want to zap the padding and l2fetch the initial data
		 */
		if (hslice == 0) supernode_add_l2fetch(
			self,
			nn,
			info,
			batch_input,
			in_width_total*in_depth_total,
			in_hlines_per_slice + required_h_before + required_h_after_augmented);
		if (hslice == 0) supernode_add_padding_zap(
			self,
			nn,
			info,
			zapwork,
			b*input_batch_size,
			required_h_before,
			required_w_before);
		if (hslice == 0) work.suma_buf = supernode_add_suma(self, nn, info, batch_input);
		for (d = 0; d < num_out_slices; d++) {
			//const uint8_t *filtsrc = info->weights_base + workidx*weight_batch_size;
			const uint8_t *filtsrc = info->weights_base + d * weight_batch_size;
			int workidx_mod_batches = workidx % n_weight_batches;
			filtdst = supernode_filtbuf_location(info, workidx_mod_batches, filtsrc);
			copywork.copy_in = filtsrc;
			copywork.copy_out = filtdst;
			copywork.copy_size = weight_batch_size;
			supernode_add_work_item(self, nn, info, copywork);
			/* FIXME: padding zap:
				Always zap right pad
				If required_w_before, zap left pad (4 w's)
				If required_h_before, zap top pad
				If required_h_after_augmented, zap bottom pad
			*/
			/* as we near the end of this activation slice, get the next one */
			if ((d == num_out_slices - 1) && ((hslice + 1) < height_slice_factor)) supernode_add_l2fetch(
				self,
				nn,
				info,
				batch_input + in_width_total * in_depth_total*in_hlines_per_slice*(hslice + 1),
				in_width_total*in_depth_total,
				in_hlines_per_slice + required_h_before + required_h_after_augmented);
			work.weights = filtdst;
			work.donesem = &info->semaphores[workidx_mod_batches];

			work.input = batch_input;
			work.output = out_data_start + b * output_batch_size + d * info->out_next_d32;
			work.biases = info->biasbuf + d * 32;
			/* EJP: FIXME: broken for batches, really */
			for (t = 0; t < NUM_THREADS; t++) {
				work.minmax_buf = info->minmax_buf + (NUM_THREADS*workidx_mod_batches + t) * 64;
				work.start_line = hslice * out_hlines_per_slice + t;
				work.stop_line = work.start_line + out_hlines_per_slice - t;
				if (work.stop_line > info->out_height) {
					work.stop_line = info->out_height;
				}
				supernode_add_work_item(self, nn, info, work);
			}
			waitwork.donesem = &info->semaphores[(workidx + 1) % n_weight_batches];
			if (workidx >= (n_weight_batches - 1)) supernode_add_work_item(self, nn, info, waitwork);
			if (workidx >= (n_weight_batches - 1)) logmsg(nn, 2, "add wait %d for %d", (workidx + 1) % n_weight_batches, workidx);
			workidx++;
		}
	}
	}
	for (i = workidx - (n_weight_batches - 1); i < workidx; i++) {
		if (i < 0) continue;
		logmsg(nn, 2, "end: add wait %d", i%n_weight_batches);
		waitwork.donesem = &info->semaphores[i%n_weight_batches];
		supernode_add_work_item(self, nn, info, waitwork);
	}
#endif
	/* Add work to check the output min/max and see if we need to adjust and try again */
	//work.execute = supernode_execute_workitem_check_for_retry;
	//supernode_add_work_item(self,nn,info,work);

	/*
	 * Sometimes we want to collect some statistics...
	 */
	//if (0) supernode_statistics(nn, info, self);

	/*
	 * We've calculated the strategy, mark that the work is done. Hopefully it sticks!
	 */
	supernode_compile_worklist(nn, info, self);
	info->prepared_vtcm_addr = nn->vtcm_ptr;
	info->needs_retry = 0;
	info->strategy_valid = 1;
	return 0;
}

static void supernode_vector_kickoff(struct nn_graph *nn, void *vself)
{
	struct nn_node *self = vself;
	logmsg(nn, 2, "vector kickoff!");
	supernode_execute_some_strategy(self, nn, 0, 1);
}

static int supernode_execute_strategy(struct nn_node *self, struct nn_graph *nn)
{
	struct sn16b_info *info = self->opaque;
	info->cycles = 0;
	info->minval = 0;
	info->maxval = 0;
	if (0) {
		return supernode_execute_some_strategy(self, nn, 0, info->n_work_items);
	}
	else {
		nn_sem_init(&info->alldone_sem, 0);
		nn_os_work_for_vector(nn, supernode_vector_kickoff, self);
		nn_sem_wait(&info->alldone_sem);
		supernode_execute_workitem_check_for_retry(NULL, self, nn);
		return 0;
	}
#if 0
	logmsg(nn, 2, "weights cksum: %08x", data_cksum(
		info->weights,
		info->filt_height
		* info->filt_width
		* info->in_depth
		* info->out_depth));
	for (i = 0; i < n_work_items; i++) {
		//Q6_dcfetch_A(&info->work_items[i+1]);
		struct workitem *p = &info->work_items[i];
		err |= p->execute(p, self, nn);
	}
	return err;
#endif
}

static inline int supernode_strategy_valid(
	struct nn_node *self,
	struct nn_graph *nn,
	struct sn16b_info *info)
{
	const struct tensor *in_min_tensor = self->inputs[2];
	const struct tensor *in_max_tensor = self->inputs[3];
	if (info->needs_retry) return 0;
	if (!info->strategy_valid) return 0;
	if (tensor_get_float(in_min_tensor, 0) != info->in_min_float) return 0;
	if (tensor_get_float(in_max_tensor, 0) != info->in_max_float) return 0;
	if (nn->vtcm_ptr != info->prepared_vtcm_addr) return 0;

	if (!shape_matches(&info->in_shape, &self->inputs[0]->shape)) {
		return 0;
	}
	/*
	 * FIXME: check input max/min/shape
	 */
	return 1;
}

static int get_circ_buf_size(struct nn_node *self, struct nn_graph *nn)
{
	return 0;
}

static int supernode_execute_16b_hvx(struct nn_node *self, struct nn_graph *nn)
{
	struct sn16b_info *nodeinfo = self->opaque;
	struct tensor *out = self->outputs[0];
	struct tensor *out_min = self->outputs[1];
	struct tensor *out_max = self->outputs[2];
	unsigned long long int total_time;
	//logmsg(nn,0,"NEW supernode id=%d",self->node_id);
	//logmsg(nn,0,"FIXME: memset out for debug... remove");
	//memset(out->data,0xAB,out->max_size);
	if (out->format.layout == NN_LAYOUT_D32_LOVINGLY_PREPARED) {
		logmsg(nn, 2, "Prepared output.");
		logmsg(nn, 2, "out tensor info: bhwd=%d,%d,%d,%d paddings=(%d,%d)x(%d,%d)x(%d,%d)",
			out->shape.batches, out->shape.height, out->shape.width, out->shape.depth,
			out->format.height_pad[0], out->format.height_pad[1],
			out->format.width_pad[0], out->format.width_pad[1],
			out->format.depth_pad[0], out->format.depth_pad[1]);
	}
	nn_scratch_reset(nn);			// if we recurse, reset scratch ptr
	l2fetch(nodeinfo, sizeof(*nodeinfo), sizeof(*nodeinfo), 1);
	if ((nodeinfo->work_items != NULL) && (nodeinfo->n_work_items > 0)) {
		l2fetch(nodeinfo->work_items, sizeof(struct workitem), sizeof(struct workitem), nodeinfo->n_work_items);
	}

	nodeinfo->circ_buf_size = (get_circ_buf_size(self, nn) + 127) & (~127); //used in v65 
#if defined(V65) && !defined(V66)
	int i;
	int total_size = nodeinfo->circ_buf_size * NUM_THREADS;
	char *buf;
	if ((total_size <= VTCM_CIRCBUF_SIZE) && (total_size <= nn->vtcm_size)) {
		buf = nn->vtcm_ptr;
		buf += (nn->vtcm_size - VTCM_CIRCBUF_SIZE);
	}
	else {
		buf = nn_scratch_alloc(nn, total_size);
	}
	if (buf == NULL) return errlog(nn, "scratch failed to alloc %d bytes for bufstack", (int)total_size);
	nn_os_bufstack_init(&nodeinfo->bufstack);
	for (i = 0; i < NUM_THREADS; i++) {
		nn_os_bufstack_push(&nodeinfo->bufstack, buf + i * nodeinfo->circ_buf_size);
	}
#endif
	//nn_scratch_grow(nn,nodeinfo->circ_buf_size * NUM_THREADS);
	if (!nodeinfo->fsplit) {
		if (supernode_split_input(self, nn)) {
			return errlog(nn, "error in split");
		}
		nodeinfo->fsplit = 1;
	}
	if (likely(supernode_strategy_valid(self, nn, nodeinfo))) {
		if (supernode_execute_strategy(self, nn) != 0) {
			return errlog(nn, "execute strategy failed");
		}
	}
	else {
#if 0
		if (nodeinfo->use_v66 == 1) {
			if (supernode_recalculate_strategy_v66(self, nn) != 0) {
				return errlog(nn, "recalc strategy for v66 failed");
			}
		}
		else 
#endif
		{
			if (supernode_recalculate_strategy(self, nn) != 0) {
				return errlog(nn, "recalc strategy failed");
			}
		}
		if (supernode_execute_strategy(self, nn) != 0) {
			return errlog(nn, "execute strategy fail after recalc");
		}
	}
	/* Replay if self-calculated min/max are insufficient */
	if (nodeinfo->needs_retry) {
		nodeinfo->recursion_depth++;
		if (nodeinfo->recursion_depth < 3) {
			return supernode_execute_16b_hvx(self, nn);
		}
		else {
			logmsg(nn, 0, "Extreme recursion detected, problem finding min/max?");
		}
	}
	nodeinfo->recursion_depth = 0;
	// desplit here
	if (nodeinfo->fsplit) {
		supernode_desplit_input(self, nn);
		nodeinfo->fsplit = 0;
	}
	tensor_set_float(out_min, 0, nodeinfo->out_minval);
	tensor_set_float(out_max, 0, nodeinfo->out_maxval);
	/* Record cycles (divide by # of vector worker threads somehow?) */
	total_time = nodeinfo->cycles;
	record_usertime(nn, self, NN_GRAPH_PERFEVENT_USER0, total_time);
	logmsg(nn, 2, "out tensor info: bhwd=%d,%d,%d,%d paddings=(%d,%d)x(%d,%d)x(%d,%d)",
		out->shape.batches, out->shape.height, out->shape.width, out->shape.depth,
		out->format.height_pad[0], out->format.height_pad[1],
		out->format.width_pad[0], out->format.width_pad[1],
		out->format.depth_pad[0], out->format.depth_pad[1]);
	logmsg(nn, 2, "Supernode execute done!");
	return 0;
}
#endif // V66

int supernode_u16b_check(struct nn_node *self, struct nn_graph *nn) {
	sn16b_info *info = self->opaque;
	if (self->n_inputs != 12) return errlog(nn, "supernode wrong # inputs... now need min/max with inf for self-detecting");
	if (self->n_outputs != 3) return errlog(nn, "supernode wrong # outputs");
	const struct tensor *filt_tensor = self->inputs[1];
	const struct tensor *filt_min_tensor = self->inputs[4];
	const struct tensor *filt_max_tensor = self->inputs[5];
	const struct tensor *stride_tensor = self->inputs[6];
	float filt_max_float = tensor_get_float(filt_max_tensor, 0);
	float filt_min_float = tensor_get_float(filt_min_tensor, 0);
	int32_t filt_offset = fast_roundf(-filt_min_float / (filt_max_float - filt_min_float)*65535.0f);
	int32_t filt_batches = filt_tensor->shape.filt_batches;
	int32_t filt_batches_roundup = (filt_batches + 31) & ~31;
	int32_t filt_height = filt_tensor->shape.filt_height;
	int32_t filt_width = filt_tensor->shape.filt_width;
	int32_t filt_depth = filt_tensor->shape.filt_depth;
	int32_t filt_depth_roundup = (filt_depth + 31) & ~31;
	uint32_t filt_elements = filt_width * filt_height * filt_depth_roundup;
	uint32_t weights_size = filt_elements * filt_batches_roundup;
	int32_t stride_width = stride_tensor->shape.width;
	int32_t stride_height = stride_tensor->shape.height;
	uint32_t out_depth = filt_batches_roundup;
	int32_t weight_batch_size = filt_width * filt_height * filt_depth_roundup * 32 * 2;
	int32_t n_weight_batches = supernode_n_weight_batches(weight_batch_size, nn->vtcm_size);
	int32_t total_weight_batches = filt_batches_roundup / 32;
	float specified_minval = tensor_get_float(self->inputs[10], 0);
	float specified_maxval = tensor_get_float(self->inputs[11], 0);
	int i;
	int use_v65 = 0;
	int use_v66 = 0;

	int circ_buf_est = 2 * (filt_height + stride_tensor->shape.height)*filt_depth*(self->output_defs[0].max_sizes[1] * stride_width + 8);
	nn_scratch_grow(nn, circ_buf_est*NUM_THREADS);

	if ((filt_elements % 32) != 0) return errlog(nn, "FIXME: < 32 depth");
	if ((filt_batches_roundup % 32) != 0) return errlog(nn, "FIXME: < 32 filts");
	if (info != NULL) {
		/* Already set up, invalidate strategy and return */
		info->strategy_valid = 0;
		logmsg(nn, 0, "info was already set up?");
		return 0;
	}
	if ((info = nn_calloc(1, sizeof(*info))) == NULL) {
		return errlog(nn, "couldn't allocate info");
	}
	if ((info->minmax_buf = nn_memalign(128, NUM_THREADS*n_weight_batches * 64 * sizeof(int))) == NULL) {
		nn_free(info);
		return errlog(nn, "malloc/memalign");
	}
	if ((info->weights = nn_memalign(128, weights_size * sizeof(info->weights))) == NULL) {
		nn_free(info->minmax_buf);
		nn_free(info);
		return errlog(nn, "alloc weights");
	}
	if ((info->biasbuf = nn_memalign(128, out_depth * sizeof(int32_t))) == NULL) {
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info);
		return errlog(nn, "alloc biasbuf");
	}
	if ((info->semaphores = nn_calloc(3 + n_weight_batches, sizeof(nn_sem_t))) == NULL) {
		nn_free(info->biasbuf);
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info);
		return errlog(nn, "alloc semaphores");
	}
	if ((info->gemsumb = nn_memalign(128, out_depth * sizeof(int32_t))) == NULL) {
		nn_free(info->biasbuf);
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info->semaphores);
		nn_free(info);
		return errlog(nn, "alloc gemsumb");
	}
	if ((info->conv_slices = nn_calloc(total_weight_batches, sizeof(*info->conv_slices))) == NULL) {
		nn_free(info->gemsumb);
		nn_free(info->biasbuf);
		nn_free(info->minmax_buf);
		nn_free(info->weights);
		nn_free(info->semaphores);
		nn_free(info);
	}

	info->use_v65 = use_v65;
	info->use_v66 = use_v66;
	info->num_accs = 6;
	for (i = 0; i < n_weight_batches + 3; i++) {
		nn_sem_init(&info->semaphores[i], 0);
	}
	float weight_scalefac = 1.0;	// how much weights were scaled (if at all);  0.5 .. 1.0

	supernode_gemsumb_unsigned(
		info,
		filt_tensor->data,
		info->gemsumb,
		filt_height,
		filt_width,
		filt_depth,
		filt_depth_roundup,
		filt_batches,
		filt_offset,
		out_depth);

	for (i = 0; i < out_depth; i++) logmsg(nn, 4, "gemsumb[%d]=%x", i, info->gemsumb[i]);
	if ((filt_width * filt_height * filt_depth_roundup * filt_batches_roundup) % 128) return errlog(nn, "filt dims too odd");

	supernode_rearrange_for_u16b_8b8h(
		info->weights,
		filt_tensor->data,
		filt_height,
		filt_width,
		filt_depth,
		filt_depth_roundup,
		filt_batches,
		filt_batches_roundup,
		filt_offset);

	info->strategy_valid = 0;	/* Redundant w/ calloc */
	self->opaque = info;
	info->weight_batch_size = weight_batch_size;
	info->n_weight_batches = n_weight_batches;
	info->in_right_padpad = 8 * stride_width; //tack on this to circular buffer to avoid bad max's


	float filt_level_size = (filt_max_float - filt_min_float) / (65535.0f* weight_scalefac);
	if (use_v65 || use_v66) {
		info->weights_offset = 0;
	}
	else {
		info->weights_offset = filt_offset;
	}
	info->filt_offset = filt_offset;
	info->weights_level_size = filt_level_size;

	logmsg(nn, 2, "stride_width=%d in_right_padpad=%d", stride_width, info->in_right_padpad);

	nn_sem_init(&info->alldone_sem, 0);
	setup_initial_output_range(info, specified_minval, specified_maxval, 0.0f, 0.5f);

	// additional
	// allocate in/out for din
#if 1
	const struct tensor *tensor_in = self->inputs[0];
	int32_t in_width = tensor_in->shape.width;
	int32_t in_height = tensor_in->shape.height;
	int32_t in_depth = tensor_in->shape.depth;
	int32_t in_left_pad = tensor_in->format.width_pad[0];
	int32_t in_right_pad = tensor_in->format.width_pad[1];
	int32_t in_depth_before_pad = tensor_in->format.depth_pad[0];
	int32_t in_depth_after_pad = tensor_in->format.depth_pad[1];
	int32_t in_top_pad = tensor_in->format.height_pad[0];
	int32_t in_bottom_pad = tensor_in->format.height_pad[1];
	int32_t in_width_total = in_width + in_left_pad + in_right_pad;
	int32_t in_depth_total = in_depth + in_depth_before_pad + in_depth_after_pad;
	int32_t in_height_total = in_height + in_top_pad + in_bottom_pad;	int32_t more_byte_suma = in_width_total < 24 ? 24 * in_depth_total : 0;

	/* Calculate output dimensions */
	const struct tensor *in_tensor = self->inputs[0];
	int32_t in_batches = in_tensor->shape.batches;
	int32_t out_batches = in_batches;

	/* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */
	int32_t required_w_before, required_w_after;
	int32_t required_h_before, required_h_after;
	int32_t out_width = nn_pad_compute_outsize_and_pad(in_width, filt_width, stride_width, self->padding,
		&required_w_before, &required_w_after);
	int32_t out_height = nn_pad_compute_outsize_and_pad(in_height, filt_height, stride_height, self->padding,
		&required_h_before, &required_h_after);

	/*
	 * Set output padding values to sensible defaults.
	 */

	int32_t out_right_pad = ((-out_width) & 3);
	int32_t out_left_pad = 4;
	int32_t out_top_pad = 4;
	int32_t out_bottom_pad = out_top_pad;
	int32_t out_depth_before_pad = 0;
	int32_t out_depth_after_pad = (-out_depth) & 31;
	int32_t out_depth_total = out_depth + out_depth_before_pad + out_depth_after_pad;
	int32_t out_width_total = out_width + out_left_pad + out_right_pad;
	int32_t out_height_total = out_height + out_top_pad + out_bottom_pad;
	info->in_oedelta = in_batches*in_width_total * in_depth_total*in_height_total + more_byte_suma;
	info->din_e = nn_memalign(128, info->in_oedelta *2);
	info->din_o = info->din_e + info->in_oedelta;
	info->out_oedelta = out_batches*out_width_total * out_depth_total*out_height_total;
	info->dout_e = nn_memalign(128, info->out_oedelta *2);
	info->dout_o = info->dout_e + info->out_oedelta;
#endif

	// TODO: remove me
	info->filt_offset = filt_offset;
	info->filt_level_size = (filt_max_float - filt_min_float) / (65535.0f * weight_scalefac);

	float bias_min_float = tensor_get_float(self->inputs[8], 0);
	float bias_max_float = tensor_get_float(self->inputs[9], 0);
	info->bias_offset = fast_roundf(-bias_min_float / (bias_max_float - bias_min_float) * 65535.0f);
	info->bias_level_size = (bias_max_float - bias_min_float) / 65535.0f;

	info->is_u16 = 1;
#ifdef V66
	info->use_usmodel = info->is_u16 && 1; // unsigned * signed model
#else
	info->use_usmodel = info->is_u16 && 0; // unsigned * signed model
#endif
	info->use_2planes = info->is_u16 && 1; // uu model only

	return 0;
}

static int supernode_16b_dtor(struct nn_node *self, struct nn_graph *nn)
{
	sn16b_info *info = self->opaque;
	if (info != NULL) {
		supernode_reset_work_items(self, nn, info);
		nn_free(info->weights);
		nn_free(info->din_e);
		nn_free(info->dout_e);
		nn_free(info);
	}
	self->opaque = NULL;
	return node_free_common(self, nn);
}


/*
	12 inputs:
		0: input data (qint16; d32 format in _d32 variants);  shape [b,hin,win,din]
		1: weights (qint16, flat)  shape [fh,fw,din,dout]
		2: input min
		3: input max
		4: weights min
		5: weights max
		6: stride tensor, shape [1,stride_h, stride_w, 1 ]
		7: bias tensor	(qint16, (or qi32??), according to node type);	shape [1,1,1,dout]
		8: bias min
		9: bias max
		10: output min (-inf for "auto")
		11: output max (+inf for "auto")
	3 outputs:
		0: output data (qint16; d32 format in _d32 variants); shape [b,hout,wout,dout]
		1: output min
		2: output max

	General convolve, add bias, truncate range op.
*/

struct nn_node_ops nn_ops_for_Supernode_16x16p16to16_d32 = {
	.execute = supernode_16b_execute_spawn,
	.check = supernode_16b_check,
	.ctor = node_alloc_common,
	.dtor = supernode_16b_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};

struct nn_node_ops nn_ops_for_Supernode_u16x16p16to16_d32 = {
#if defined(V66)
	.execute = supernode_16b_execute_spawn,
	.check = supernode_16b_check,
#else
	.execute = supernode_execute_16b_hvx,
	.check = supernode_u16b_check,
#endif
	.ctor = node_alloc_common,
	.dtor = supernode_16b_dtor,
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT
};
