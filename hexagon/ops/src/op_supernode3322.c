/*
 * Copyright (c) 2018-2019, The Linux Foundation. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nn_graph.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif
#include "hvx_hexagon_protos.h"
#include <quantize.h>

#if __HEXAGON_ARCH__ < 66
#define NUM_THREADS     2
#else
#define NUM_THREADS     4
#endif

#define roundup(a, p2)       (((a)+(p2)-1)&~((p2)-1))

struct workitem {
    struct nn_node *self;               // This node
    struct supernode3322_info *info;    // same as self->opaque
    nn_sem_t *donesem;                  // semaphore to post completion

    /* Main convolutional work items    */
    uint8_t *input;                     // Input data.  Could be from input tensor or temp buf
    uint8_t *output;                    // Output data.  Could be output tensor or temp buf
    int32_t batch_index;                // current batch index (input supernode)
    int32_t start_line;                 //
    int32_t stop_line;                  //
    int32_t num_lines;                  // Number of rows to skip each iteration
};

/*
 * Pointers and values that are common to everything in the node
 * Also values that are constant across all work items
 */
struct supernode3322_info {
    uint8_t *weights;                   // weights, padded and adjusted as necessary
    int32_t *biasbuf;                   // int32 bias buffer, including min offsets and gemsumb
    nn_sem_t *semaphores;               // pointer to preallocated array of semaphores
    float out_minval;                   // Minimum output value, either specified or guessed
    float out_maxval;                   // maximum output value, either specified or guessed
    int minval_precalculated;           // Is the minval precalculated?
    int maxval_precalculated;           // Is the maxval precalculated?
    float out_minval_spec;              // exact value specified (when not precalculated)
    float out_maxval_spec;              // exact value specified (when not precalculated)
    int32_t minval;                     // Minimum value (in prod space) actually observed
    int32_t maxval;                     // Maximum value (in prod space) actually observed
    int32_t strategy_valid;             // Do we believe the strategy is currently valid?
    float in_max_float;                 // maximum input float value
    float in_min_float;                 // minimum input float value
    float weights_level_size;           // how large in float is one increment in the weights?
    int32_t weights_offset;             // where is 0 in weight number space?
    int32_t in_width;                   // input width of input
    int32_t in_height;                  // height of the input
    int32_t in_depth;                   // input depth of input 
    int32_t in_next_row;                // distance from one row to the next
    int32_t out_width;                  // output width to compute, should be width/stride
    int32_t out_height;                 // number of output lines to compute
    int32_t out_depth;                  // total depth to compute
    int32_t out_next_row;               // distance from one row to the next 
    int32_t required_w_before;          //
    int32_t required_w_after;           //
    int32_t required_h_before;          //
    int32_t filt_width;                 // filter width
    int32_t filt_height;                // filter height
    int32_t filt_batches;               // filter batches
    int32_t stride_width;               // stride in the width dimension
    int32_t stride_height;              // stride in the height dimension (== width usually)
    int32_t recip_val;                  // reciprocal for product space --> output space
    int32_t recip_shamt;                // amount to shift before recip mpy
    int32_t in_offset;                  // amount to normalize inputs by.  Needed?
    int32_t filt_offset;                // amount to normalize filter values by. Needed?
    const uint8_t *raw_input;           // ptr to the input tensor for use when copied into temp
//  uint8_t *input_base;
    int32_t max_valid_val;              // maximum value that results in a value not above max_out
    int32_t min_valid_val;              // minimum value that results in a value not below min_out
    float prod_level_size;              // in_level_size * filt_level_size
    int32_t *gemsumb;                   // GEMSUMB value, if we want to calculate it at preparation time
//  uint64_t cycles;                    // Cycle accumulator for children
};


static void setup_initial_output_range( struct supernode3322_info *info, float,float,float,float);

static void fill_info_minmax_basics(
    struct nn_graph *nn,
    struct nn_node *self,
    struct supernode3322_info *info)
{
    /* Pull out the inputs we need */
    const struct tensor *min_in_tensor = self->inputs[2];
    const struct tensor *max_in_tensor = self->inputs[3];

    /* Get min/max values for input, weights, and bias data */
    float in_min_float = tensor_get_float(min_in_tensor,0);
    float in_max_float = fmaxf(tensor_get_float(max_in_tensor,0), in_min_float + 1e-18f);

    /* find zero offset for each input */
    int32_t input_offset = quantize_uint8(0.0f,in_min_float,in_max_float);
    info->in_offset = input_offset;

    /* Find level size for each input */
    float in_level_size = (in_max_float - in_min_float) / 255;

    // filter level (already compensated for any scaling done)
    float filt_level_size = info->weights_level_size;

    /* The product level size is the product of the input and filter level size */
    float prod_level_size = in_level_size * filt_level_size;
    info->prod_level_size = prod_level_size;

    /* Calculate conversion ratio from bias to product space */
    //float bias_to_prod_ratio = (bias_level_size / prod_level_size);
    /* What is the value of the output minimum in the product space? */
    /* We need to add it to the products to move the smallest valid value to zero */
    uint64_t maxsum = fast_roundf((info->out_maxval-info->out_minval) / prod_level_size);
    uint32_t recip_shamt = 0;
    uint64_t recip_val_64 = 0x7F80000000ULL/maxsum;  //255 << 31

    /* Compute reciprocal and shift amount */
    maxsum += 1;
    while (recip_val_64 >= 0x80000000ULL) {
        recip_shamt++;
        recip_val_64 = 0x7F80000000ULL / (maxsum << recip_shamt);
    }
    info->recip_val = recip_val_64;
    info->recip_shamt = recip_shamt;

    logmsg(nn,2,"out_maxval=%f filt_level_size=%f prod_level_size=%f maxsum ~= %f",
        info->out_maxval,info->weights_level_size,info->prod_level_size,maxsum);
    return;
}

static int fill_bias_buf(
    struct nn_graph *nn,
    struct nn_node *self,
    struct supernode3322_info *info,
    int32_t extra,
    int    bias32)
{
    const struct tensor *bias_tensor = self->inputs[7];
    const struct tensor *bias_min_tensor = self->inputs[8];
    const struct tensor *bias_max_tensor = self->inputs[9];
    float bias_min_float = tensor_get_float(bias_min_tensor,0);
    float bias_max_float = tensor_get_float(bias_max_tensor,0);
    int32_t bias_offset = bias32 ? 0 : quantize_uint(0.0f,bias_min_float,bias_max_float);
    float bias_denom = bias32 ? 0x1.0p32 : 255.0f;
    float bias_level_size = (bias_max_float - bias_min_float) / bias_denom;
    float bias_to_prod_ratio = (bias_level_size / info->prod_level_size);
    float min_out_prod_offset = -info->out_minval / info->prod_level_size;
    int32_t bias_depth = bias_tensor->shape.depth;
    const uint8_t *bias8_ptr = bias_tensor->data;
    const int32_t *bias32_ptr = bias_tensor->data;
    int i;
    int32_t biasval;
    float bias_fval;
    float minout_bias_fval;
    int32_t gemsumb_val;
    int32_t final;

    for (i = 0; i < info->out_depth; i++) {
        if (i >= bias_depth) biasval = bias_offset;
        else if (bias32) biasval = bias32_ptr[i];
        else biasval = bias8_ptr[i];
        bias_fval = (biasval - bias_offset) * bias_to_prod_ratio;
        minout_bias_fval = bias_fval + min_out_prod_offset;
        gemsumb_val = info->gemsumb[i];
        final = extra - gemsumb_val * info->in_offset + fast_roundf(minout_bias_fval);
        info->biasbuf[i] = final;
        //logmsg(nn,3,"i=%d biasval%d=%d fval=%f minout_fval=%f gemsumb_val=%d extra=%d final=%d",
        //  i,bias32?32:8,biasval,bias_fval,minout_bias_fval,gemsumb_val,0,final);
	}

    logmsg(nn,3,"in_offset=%d bias_levelsize=%f prod_level_size=%f ratio=%f",info->in_offset,bias_level_size,info->prod_level_size,bias_to_prod_ratio);
    return 0;
}

static void prepare_indata(
    struct supernode3322_info *info,
    const uint8_t *in_data,
    int32_t ypos, 
    int32_t nlines,
    uint8_t *optr )
{
    int32_t in_width  = info->in_width;
    int32_t in_height = info->in_height;
    int32_t in_depth  = info->in_depth;
    int32_t in_width_depth = in_width*in_depth;
    int32_t in_batch_size = in_height*in_width_depth;

    int pad_top  = info->required_h_before;
    int pad_left = info->required_w_before;
    int pad_right= info->required_w_after;

    int32_t y, in_y;


    for (y = ypos; y < (ypos + nlines); y++) {
        in_y = y - pad_top;

        if (in_y < 0 || in_y >= in_height) {
            vmemset_asm(optr, info->in_offset, info->in_next_row);

        } else {
            int32_t offset = in_y * in_width_depth;
            const uint8_t *iptr = in_data + offset;
            load_indata_d2(iptr, in_width, info->in_next_row, pad_left, pad_right, info->in_offset, optr, in_batch_size-offset);
        }
        optr += info->in_next_row;
    }
}

static void prefetch_input(
    struct supernode3322_info *info, 
    const uint8_t *in_data, 
    int32_t ypos, 
    int32_t nlines )
{
    int32_t in_width  = info->in_width;
    int32_t in_height = info->in_height;
    int32_t in_depth  = info->in_depth;
    int32_t in_width_depth = in_width*in_depth;

    int pad_top  = info->required_h_before;

    int32_t start = ypos - pad_top;
    int32_t stop  = start + nlines;

    if (start < 0) {
        nlines += start;
        start = 0;
    }
    if (stop >= in_height) {
       nlines -= (stop - in_height);
    }
    if (nlines > 0) {
        l2fetch(in_data + start*in_width_depth, in_width_depth, in_width_depth, nlines); 
    }
}

static void supernode3322_execute_conv_slice(struct nn_graph *nn, void *vwork)
{
    struct workitem *work = vwork;
    struct supernode3322_info *info = work->info;
    
    int32_t in_width  = info->in_width;
    int32_t in_height = info->in_height;
    int32_t in_depth  = info->in_depth;
    int32_t in_width_depth = in_width*in_depth;

    int32_t in_next_row = info->in_next_row;
    int32_t out_next_row= info->out_next_row;

    int32_t filt_height = info->filt_height;
    int32_t start_line = work->start_line;
    int32_t stop_line  = work->stop_line;

    int32_t ibatch = work->batch_index;

    const uint8_t *raw_input = (const uint8_t *)info->raw_input + ibatch*in_height*in_width_depth;
    uint8_t       *input = work->input;
    uint8_t       *output= work->output + start_line*out_next_row;

    int32_t  proc_rows = work->num_lines;
    int32_t  pf_offset = filt_height-1;

    int32_t n_lines = Q6_R_min_RR(stop_line-start_line, proc_rows);
    int32_t n_in_rows = n_lines-1 + filt_height; 

    // prefetch initial activations
    prefetch_input(info, raw_input, start_line, n_in_rows);

    for (int out_row = start_line; out_row < stop_line; out_row += proc_rows) {
#ifndef V66
        wait_for_l2fetch();
#endif
        prepare_indata(info, raw_input, out_row, n_in_rows, (uint8_t *)input);

        int32_t next_n_lines = Q6_R_min_RR(stop_line-out_row-proc_rows, proc_rows);
        int32_t next_n_in_rows = (next_n_lines-1) + filt_height; 

        if (next_n_lines > 0) {
             prefetch_input(info, raw_input, out_row + proc_rows + pf_offset, next_n_in_rows-pf_offset);
        }

        conv3322bbb(input, info->weights, info->biasbuf, output,
                    info->out_width, n_lines, 
                    info->recip_val, info->recip_shamt,
                    info->filt_offset, in_next_row );

        output += proc_rows*out_next_row;
        n_lines   = next_n_lines;
        n_in_rows = next_n_in_rows;
    }
    nn_sem_post(work->donesem);
}

#define  MAX_SLICE_SIZE         10

static int supernode3322_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
    //logmsg(nn,0,"NEW supernode id=%d",self->node_id);
    //long long start, my_cycles;
    //start = nn_os_get_cycles(nn);

    /* Structures with auxillary information */
    struct supernode3322_info *info = self->opaque;
    l2fetch(info,sizeof(*info),sizeof(*info),1);

    /* Pull out the inputs we need */
    const struct tensor *in_tensor   = self->inputs[0];
    /* 
     * Find the dimensions of the input data, 
     * both dimensions of data as well as padding
     */
    int32_t in_batches = in_tensor->shape.batches;
    int32_t in_width   = in_tensor->shape.width;
    int32_t in_height  = in_tensor->shape.height;
    int32_t in_depth   = in_tensor->shape.depth;

    info->raw_input = in_tensor->data;
    info->in_width  = in_width;
    info->in_height = in_height;
    info->in_depth  = in_depth;

    int32_t filt_batches = info->filt_batches;
    int32_t filt_height  = info->filt_height;
    int32_t filt_width   = info->filt_width;
    int32_t stride_height= info->stride_height;
    int32_t stride_width = info->stride_width;

    /* Calculate output dimensions */
    int32_t out_batches = in_batches;
    int32_t out_depth = filt_batches;

    int32_t required_w_before, required_w_after, required_h_before, required_h_after;

    /* Find output size, amount of padding required in each direction by the padding type, filter size, and stride */
    int32_t out_width = nn_pad_compute_outsize_and_pad(in_width,filt_width,stride_width,self->padding,
                        &required_w_before, &required_w_after);
    int32_t out_height= nn_pad_compute_outsize_and_pad(in_height,filt_height,stride_height,self->padding,
                        &required_h_before, &required_h_after);

    int32_t in_width_total = required_w_before + in_width + required_w_after;
    info->in_next_row = roundup(in_width_total,64)*in_depth;

    info->out_width  = out_width;
    info->out_height = out_height;
    info->out_depth  = out_depth;
    info->out_next_row = out_width*out_depth;

    info->required_w_before = required_w_before;
    info->required_w_after  = required_w_after;
    info->required_h_before = required_h_before;

    //FIXME: Do we need to set out_minval/out_maxval again?
    info->out_minval = tensor_get_float(self->inputs[10],0);
    info->out_maxval = tensor_get_float(self->inputs[11],0);

    /* Find the output tensors */
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min    = self->outputs[1];
    struct tensor *out_max    = self->outputs[2];
    /*
     * Prepare output tensors
     */
    if (tensor_out_prepare_normal(out_tensor,out_batches,out_height,out_width,out_depth,NN_TYPE_QUINT8) != 0) {
        return errlog(nn,"output tensor prep fail");
    }
    if (tensor_out_prepare_normal(out_min,1,1,1,1,NN_TYPE_FLOAT) != 0) {
        return errlog(nn,"min out prep fail");
    }
    if (tensor_out_prepare_normal(out_max,1,1,1,1,NN_TYPE_FLOAT) != 0) {
        return errlog(nn,"max out prep fail");
    }
    tensor_set_float(out_min,0,info->out_minval);
    tensor_set_float(out_max,0,info->out_maxval);

    /* Calculate conversion ratio from bias to product space */
    fill_info_minmax_basics(nn,self,info);
  
    /* Recompute bias values */
    int bias32 = (self->node_type == OP_Supernode3322_8x8p32to8);
    int32_t extra = info->in_offset*info->filt_offset*filt_height*roundup(filt_width,4)*in_depth;
    fill_bias_buf(nn,self,info, extra, bias32);

    int32_t scratch_size_t = (MAX_SLICE_SIZE+2)*info->in_next_row + 128;
    uint8_t *input_buf = nn_scratch_alloc(nn,NUM_THREADS*scratch_size_t);

    int32_t inner_act_rows  = (out_height+NUM_THREADS-1)/(NUM_THREADS);
    int32_t outer_act_iters = (out_height + inner_act_rows - 1)/inner_act_rows;
    int32_t slice = MAX_SLICE_SIZE;

    int32_t out_batch_size = out_width*out_height*out_depth;

    nn_sem_t donesem;
    nn_sem_init(&donesem,0);
    struct workitem filtwork[NUM_THREADS];

    for(int ibatch = 0; ibatch < in_batches; ibatch++){
        uint8_t *output = (uint8_t *)out_tensor->data + ibatch*out_batch_size;
        for (int i = 0; i < outer_act_iters; i++) {
            filtwork[i].info = info;
            filtwork[i].self = self;
            filtwork[i].batch_index = ibatch;
            filtwork[i].start_line = i*inner_act_rows;
            filtwork[i].stop_line  = Q6_R_min_RR((i+1)*inner_act_rows, out_height);
            filtwork[i].num_lines = slice;
            filtwork[i].input  = input_buf + i*scratch_size_t;
            filtwork[i].output = output;
            filtwork[i].donesem = &donesem;
            nn_os_work_for_vector(nn,supernode3322_execute_conv_slice,&filtwork[i]);
        }
        nn_sem_wait_n_times(&donesem, outer_act_iters);
    }
    
    logmsg(nn,2,"Supernode execute done!");
    //my_cycles = nn_os_get_cycles(nn) - start;
    //printf("Supernode3322 cycle-count: %lld\n", my_cycles);
    return 0;
}

static void rearrange_weights_3wto4_get_sumb(
    const uint8_t *in_weights,
    int zero_val,
    uint8_t *out_weights,
    int32_t *sumb )
{
    const int32_t filt_width = 3;
    const int32_t filt_height= 3;

    const int32_t out_filt_width = 4;
    const int32_t filt_batches = 2;
    const int32_t filt_depth = 2;
    int b,h,w,d;
    int val;

    for (b = 0; b < filt_batches; b++) {
        sumb[b] = 0;
        for (h = 0; h < filt_height; h++) {
            for (w = 0; w < out_filt_width; w++) {
                for (d = 0; d < filt_depth; d++) {
                    int32_t in_idx = h*filt_width*filt_depth*filt_batches
                            + w*filt_depth*filt_batches
                            + d*filt_batches
                            + b;

                    int32_t out_idx = h*out_filt_width*filt_depth*filt_batches
                            + (b%2)*8
                            + w*filt_depth
                            + d;

                    if (w < filt_width) val = in_weights[in_idx];
                    else val = zero_val;
                    out_weights[out_idx] = val;

                    sumb[b] += val;
                }
            }
        }
    }
}

int supernode3322_check(struct nn_node *self, struct nn_graph *nn)
{
    struct supernode3322_info *info = self->opaque;

    // supernode may wind up with 13 inputs due to ChannelScale - but by
    // the time we get to here, it should be dealt with and will be just a scalar [1.0].
    // so we can ignore it.
    // n_inputs = 12 or 13; n_outputs = 3; checked in ctor
    const struct tensor *filt_tensor = self->inputs[1];
    const struct tensor *filt_min_tensor = self->inputs[4];
    const struct tensor *filt_max_tensor = self->inputs[5];
    const struct tensor *stride_tensor = self->inputs[6];
    const uint8_t *filt = filt_tensor->data;
    float filt_max_float = tensor_get_float(filt_max_tensor,0);
    float filt_min_float = tensor_get_float(filt_min_tensor,0);
    int32_t filt_offset = quantize_uint8(0.0f,filt_min_float,filt_max_float);
    int32_t filt_batches = filt_tensor->shape.filt_batches;
    int32_t filt_height = filt_tensor->shape.filt_height;
    int32_t filt_width = filt_tensor->shape.filt_width;
    int32_t filt_depth = filt_tensor->shape.filt_depth;
    int32_t stride_width  = stride_tensor->shape.width;
    int32_t stride_height = stride_tensor->shape.height;
    uint32_t out_depth = filt_batches;
    float specified_minval = tensor_get_float(self->inputs[10],0);
    float specified_maxval = tensor_get_float(self->inputs[11],0);
    uint32_t weights_size = roundup(filt_width,4)*filt_height*filt_depth*filt_batches;
    int32_t n_semaphores = 2;
    int i;
    // sanity check
    if (stride_width!=1 || stride_height!=1) return errlog(nn,"stride !=1");
    if (filt_width!=3 || filt_height!=3 || filt_batches!=2 || filt_depth!=2) return errlog(nn,"filt not 3x3x2x2");

    // scratch
	int32_t out_width_max  = self->output_defs[0].max_sizes[2];
	int32_t input_width  = (out_width_max +1) * stride_width  + filt_width;

    int32_t scratch_size = NUM_THREADS*(roundup(input_width,64)*(MAX_SLICE_SIZE+2)*filt_depth + 128);
    logmsg(nn,3,"scratch_size = %ld", scratch_size);

    nn_scratch_grow(nn, scratch_size);

    if (info != NULL) {
        /* Already set up, invalidate strategy and return */
        info->strategy_valid = 0;
        logmsg(nn,0,"info was already set up?");
        return 0;
    }
    if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
        return errlog(nn,"couldn't allocate info");
    }
    if ((info->weights = nn_memalign(8,weights_size)) == NULL) {
        nn_free(info);
        return errlog(nn,"alloc weights");
    }
    if ((info->biasbuf = nn_memalign(8,out_depth*sizeof(int32_t))) == NULL) {
        nn_free(info->weights);
        nn_free(info);
        return errlog(nn,"alloc biasbuf");
    }
    if ((info->semaphores = nn_calloc(n_semaphores,sizeof(nn_sem_t))) == NULL) {
        nn_free(info->biasbuf);
        nn_free(info->weights);
        nn_free(info);
        return errlog(nn,"alloc semaphores");
    }
    if ((info->gemsumb = nn_memalign(8,out_depth*sizeof(int32_t))) == NULL) {
        nn_free(info->biasbuf);
        nn_free(info->weights);
        nn_free(info->semaphores);
        nn_free(info);
        return errlog(nn,"alloc gemsumb");
    }

    for (i = 0; i < n_semaphores; i++) {
        nn_sem_init(&info->semaphores[i],0);
    }

    rearrange_weights_3wto4_get_sumb(filt, filt_offset, info->weights, info->gemsumb);

    for (i = 0; i < out_depth; i++) logmsg(nn,4,"gemsumb[%d]=%x",i,info->gemsumb[i]);

    //logmsg(nn,2,"weights cksum: %08x",data_cksum(info->weights,filt_height*filt_width*filt_depth_roundup*filt_batches_roundup));
    self->opaque = info;

    float filt_level_size = (filt_max_float - filt_min_float)/255.0f;
    info->filt_offset = filt_offset;
    info->weights_level_size = filt_level_size;
    info->filt_batches = filt_batches;
    info->filt_height  = filt_height;
    info->filt_width   = filt_width;
    info->stride_height= stride_height;
    info->stride_width = stride_width;

    setup_initial_output_range( info, specified_minval, specified_maxval, 0.0f, 0.5f);
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
static void
setup_initial_output_range( struct supernode3322_info *info,
    float specified_minval,        // range specified by inputs
    float specified_maxval,
    float minval_default,            // use when specified_minval = -INF
    float maxval_default )            // use when specified_maxval = INF
{
    // enforce sanity:  min <= 0.0 <= max
    // and max > min + 1/128
    //
    specified_minval = fminf( specified_minval, 0.0f);
    specified_maxval = fmaxf( fmaxf( specified_maxval, 0.f),
                              specified_minval + 0x1.0p-7f);

    info->out_minval_spec = specified_minval;
    info->out_maxval_spec = specified_maxval;

    int mnp = (specified_minval == -INFINITY)?0:1;        // is min precalc
    int mxp = (specified_maxval == INFINITY)?0:1;         // is max precalc

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
}

static int supernode_dtor(struct nn_node *self, struct nn_graph *nn)
{
    struct supernode3322_info *info = self->opaque;
    if (info != NULL) {
        nn_free(info->gemsumb);
        nn_free(info->semaphores);
        nn_free(info->biasbuf);
        nn_free(info->weights);
        nn_free(info);
    }
    self->opaque = NULL;
    return node_free_common(self,nn);
}
// supernode may wind up with 13 inputs due to ChannelScale - but by
// the time we get to here, it should be dealt with and will be just a scalar [1.0].
// so we can ignore it.

struct nn_node_ops nn_ops_for_Supernode3322_8x8p8to8 = {
    .execute = supernode3322_execute_hvx,
    .check = supernode3322_check,
    .ctor = node_alloc_common,
    .dtor = supernode_dtor,
    .n_inputs = NN_IOCOUNT_RANGE(12,13),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_Supernode3322_8x8p32to8 = {
    .execute = supernode3322_execute_hvx,
    .check = supernode3322_check,
    .ctor = node_alloc_common,
    .dtor = supernode_dtor,
    .n_inputs = NN_IOCOUNT_RANGE(12,13),
    .n_outputs = NN_IOCOUNT(3),
};
