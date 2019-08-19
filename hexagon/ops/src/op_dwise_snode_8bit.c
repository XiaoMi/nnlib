
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
#include <supernode_utils.h>
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

#define MAX_FILT_HEIGHT (7*2)
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
        "out_next_d32=%d out_depth=%d out_width_processed/total=%d/%d n_lines=%d filt_height=%d minmax_buf=%p "
        "recip_val=0x%x biasbuf=%p stride_height=%d recip_shamt=%d in_left_skip=%d filt_offset=%d",
        input,weights,output,in_next_row,out_next_row,in_next_d32,info->out_next_d32,
        depth,info->out_width_processed,out_width,
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
                info->out_width_processed,
                1,                  //n_lines,
                info->filt_height,
                info->filt_offset,
                biasbuf,
                minmax.words,
                info->recip,
                info->recip_shamt,			//correct 32bit mpy ,can be shift of less 
                info->stride_height,
                scratch_buf,
                (info->in_left_skip & 1)*8); 

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
        int32_t filt_offset,
        int32_t depth_multiplier)
{
        const int32_t out_filt_width = (filt_width+3)&(~3);
        int b,h,w,v,od,id,out_idx;
        int32_t sum;

        for (b = 0; b < depth_multiplier; b++) {
                for (od = 0; od < out_depth; od += 32) {
                        for (id = 0; id < 32; id++) {
                                sum = -filt_offset*filt_height*filt_width;
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
	int32_t out_depth = out_tensor->shape.depth;


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
        logmsg(nn,1," width proc / width total %d / %d",info->out_width_processed, info->out_width);
	info->out_depth_total = out_depth_total;
	info->out_depth_valid = out_depth_total;
	info->out_height = out_height;
	info->out_next_d32 = out_width_total * 32;
	info->out_next_row = out_width_total * out_depth_total;

	info->stride_height = stride_height;
	info->stride_width = stride_width;
	info->filt_height = filt_height;
	info->filt_width = filt_width;
		if (info->filt_height < 2) {
			return errlog(nn, "Only currently support filter height >= 2");
		} 
        info->opt_3x3 = 0;
        if(info->stride_width == 1) {
            if(info->filt_width == 7) {
              info->dwfunc = &dwconv2dbbb_s1_7xN_asm;
            } else if(info->filt_width == 5) {
              info->dwfunc = &dwconv2dbbb_s1_5xN_asm;
            } else if(info->filt_width == 3) {
              if(info->filt_height == 3 && !info->has_channel_scale) {
                info->dwfunc = &dwconv2dbbb_s1_3x3_asm;
                info->opt_3x3 = 1;
              } else {
                info->dwfunc = &dwconv2dbbb_s1_3xN_asm;
              }
            } else errlog(nn,"sorry, for stride 1, filter width currently only 3,5, 7... is %d", info->filt_width);
        } else if(info->stride_width == 2) {
            if(info->filt_width == 7) {
              info->dwfunc = &dwconv2dbbb_s2_7xN_asm;
            } else if(info->filt_width == 5) {
              info->dwfunc = &dwconv2dbbb_s2_5xN_asm;
            } else if(info->filt_width == 3) {
              if(info->filt_height == 3 && !info->has_channel_scale) {
                info->dwfunc = &dwconv2dbbb_s2_3x3_asm;
                info->opt_3x3 = 1;
                logmsg(nn,1," filt 3x3 x 2x2\n");
              } else {
                info->dwfunc = &dwconv2dbbb_s2_3xN_asm;
              }
            } else errlog(nn,"sorry, for stride 2,  filter width currently only 3, 5, 7... is %d", info->filt_width);
        } else errlog(nn,"sorry, horizontal stride currently only 1 or 2... is %d",info->stride_width);
        if(info->opt_3x3)
	  info->out_width_processed = out_width_total;
        else
	  info->out_width_processed = out_width;

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
                if(info->opt_3x3) {
		work.output = tensor_location_bhw_d32(out_tensor,b,0,-out_left_pad);
		work.input = tensor_location_bhw_d32(in_tensor,b,-required_h_before,-in_left_pad);
                } else {
		work.output = tensor_location_bhw_d32(out_tensor,b,0,0);
		work.input = tensor_location_bhw_d32(in_tensor,b,-required_h_before,-required_w_before);
                }

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

	if (filt_width > 7) return errlog(nn,"Oops: implement depthwise support for filt width > 7");
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
	int32_t out_depth_non_padded = filt_depth * depth_multiplier;
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
  
	if (filt_width > 7) return errlog(nn,"Oops: implement depthwise support for filt width > 7");
	if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
		return errlog(nn,"calloc");
	}
	info->is_dwise = 1;

	// check if we have a channel-scaling input (#12)
	float const *channel_scale_flts = NULL;
	if( check_channelscale_present(nn,self, out_depth_non_padded, &channel_scale_flts)!=0)
		return -1;

	if ((info->weights = nn_memalign(128,weights_size)) == NULL) {
		nn_free(info);
		return errlog(nn,"memalign for weight buffer failed");
	}
	info->weight_batch_size = weights_size;
	info->n_weight_batches = 1;
	if ((info->biasbuf = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"biasbuf");
	}
	if ((info->gemsumb = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"gemsumb");
	}
	if ((info->recip = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"recip");
	}
	if ((info->k_factor = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"k_factor");
	}
	if ((info->k_factor_recip = nn_memalign(128,out_depth*sizeof(int32_t))) == NULL) {
		return supernode_check_error_return(nn, info,"k_factor_recip");
	}
	// load channel-scale (if any)
	if( load_channel_scales(nn,info,channel_scale_flts,out_depth_non_padded) != 0 ){
		return supernode_check_error_return(nn, info,NULL);
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
                   filt_offset,
                   depth_multiplier);

	info->max_k_factor = 1.0f;
	//
	// set up the k_factor and k_factor_recip
	// In general, we are setting k_factor = k_factor/weight_scale
	//                            k_factor_inv = weight_scale/k_factor
	find_k_kinv( nn, info, out_depth);
	info->strategy_valid = 0;
    info->depth_multiplier = depth_multiplier;
	setup_initial_output_range( nn, info, specified_minval, specified_maxval, -0.125f, 0.125f);

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

struct nn_node_ops nn_ops_for_DepthwiseSupernode_8x8p8to8_d32 = {
	.execute = dwise_supernode_execute,
	.check = dwise_supernode_check,
	.ctor = node_alloc_common,
	.dtor = dwise_supernode_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};

struct nn_node_ops nn_ops_for_DepthwiseSupernode_8x8p32to8_d32 = {
	.execute = dwise_supernode_execute,
	.check = dwise_supernode_check,
	.ctor = node_alloc_common,
	.dtor = dwise_supernode_dtor,
	.n_inputs = NN_IOCOUNT_RANGE(12,13),
	.n_outputs = NN_IOCOUNT(3),
	.flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
};
/* --------------------- end of depthwise stuff ---------------------  */
