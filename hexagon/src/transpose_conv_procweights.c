
/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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
#include <stdint.h>
#include "nn_graph.h"
#include "hvx_inlines.h"
#include "transpose_conv_procweights.h"
#include "nn_pad.h"
#include "nn_gentranspose.h" // for strided_copy_2d_1b etc
#include "data_utils.h"

static int pad_filter(
    struct nn_graph *nn,
    const uint8_t *filt_data,
    const uint32_t orig_num_filters,
    const uint32_t orig_filt_height,
    const uint32_t orig_filt_width,
    const uint32_t orig_filt_depth,
    void *padded_filt_data,
    const uint32_t padded_num_filters,
    const uint32_t padded_filt_height,
    const uint32_t padded_filt_width,
    const uint32_t padded_filt_depth,
    unsigned element_size,
    const uint32_t padval,
    const uint32_t pad_before){
    uint32_t req_padding_num_filters = padded_num_filters - orig_num_filters;
    uint32_t req_padding_filt_height = padded_filt_height - orig_filt_height;
    uint32_t req_padding_filt_width = padded_filt_width - orig_filt_width;
    uint32_t req_padding_filt_depth = padded_filt_depth - orig_filt_depth;
    uint32_t pre_pad_b=0, post_pad_b=0;
    uint32_t pre_pad_h=0, post_pad_h=0;
    uint32_t pre_pad_w=0, post_pad_w=0;
    uint32_t pre_pad_d=0, post_pad_d=0;
    if (pad_before) {
        pre_pad_b = req_padding_num_filters;
        pre_pad_h = req_padding_filt_height;
        pre_pad_w = req_padding_filt_width;
        pre_pad_d = req_padding_filt_depth;
    } else {
        post_pad_b = req_padding_num_filters;
        post_pad_h = req_padding_filt_height;
        post_pad_w = req_padding_filt_width;
        post_pad_d = req_padding_filt_depth;
    }
    do_pad(padded_filt_data, filt_data,
           orig_num_filters, orig_filt_height, orig_filt_width, orig_filt_depth,
           pre_pad_b, post_pad_b,
           pre_pad_h, post_pad_h,
           pre_pad_w, post_pad_w,
           pre_pad_d, post_pad_d,
           element_size,
           padval);
    return 0;
}



// given
//   filt_height * filt_width units,
//   each of size
//       num_filters * filt_depth_bytes,
// reverse the order of the units.
// I.e. reverse in the height and width dimensions.
// If filt_height * filt_width=1 it's just a copy
//
static void rotate_filter(struct nn_graph *nn, void const *in_data,
                          const uint32_t num_filters, const uint32_t filt_height, const uint32_t filt_width, const uint32_t filt_depth_bytes,
                          void *out_data)
{
    int32_t unit_size = num_filters * filt_depth_bytes;
    int32_t unit_count = filt_height * filt_width;

    vmemcpy_2d_general_asm(
        unit_size,                                                 // 'width' (copy chunk)
        unit_count,                                                 // 'height' (number of units)
        out_data,                                                 //  dest pointer
        unit_size,                                                 // dest stride
        (uint8_t const *)in_data + unit_size * (unit_count - 1), // src pointer (start from end)
        -unit_size);                                             // source stride
}

//
// Process a 'transposed convolution':
//
// (1) input is filter (Dout, Fh0, Fw0, Din ) and blocksize Bh,Bw
// (2) Pad the filter to a larger size with its zero value:
//      Pad Fh0 -> Fh, multiple of Bh;
//      and Fw0->Fw, multiple of Bw
//      in some cases (when tcfparms->pad_num_filters != 0) Dout is padded to a multiple of 32 at the same time
// (3) SpaceToBatch:
//   ( Dout,Fh,Fw,Din)
//      -> restate as (Dout, Fsh*Bh, Fsw*Bw, Din)          Fsh,Fsw = subfilter size
//      -> transpose to ( Bh*Bw * Dout, Fsh, Fsw, Din)
// (4) Transpose and rotate
//       -> transpose from (Bh * Bw * Dout,Fsh,Fsw,Din) to ( Fsh, Fsw, Din, Dout * Bh * Bw)
//       -> reverse along Fsh, Fsw dimensions (rotate 180 degrees spatially)
//        Result is stored in tcfparms->out_data
// note, tcfparms->elbytes is 1 or 2, to get 8 or 16-bit processing.

int process_transpose_conv_filter(struct nn_graph *nn, struct transpose_conv_filter_parms *tcfparms)
{
    int res = 0;
    const struct tensor *filt_tensor = tcfparms->filt_tensor;
    uint32_t groups = tcfparms->groups;
    uint32_t num_filters = filt_tensor->shape.batches;
    uint32_t filt_height = filt_tensor->shape.height;
    uint32_t filt_width = filt_tensor->shape.width;
    uint32_t filt_depth = filt_tensor->shape.depth;
    uint32_t block_h = tcfparms->block_h;
    uint32_t block_w = tcfparms->block_w;
    uint32_t chmul = tcfparms->chmul;

    int element_size = tcfparms->elbytes; // 1 or 2

    //Padded filt stuff
    uint32_t padded_num_filters = (tcfparms->pad_num_filters) ? roundup(num_filters, 32) : num_filters;
    uint32_t new_num_filters = padded_num_filters * block_h * block_w;
    uint32_t padded_filt_height = roundup(filt_height, block_h);
    uint32_t padded_filt_width = roundup(filt_width, block_w);
    uint32_t new_filt_height = padded_filt_height / block_h;
    uint32_t new_filt_width = padded_filt_width / block_w;
    uint32_t padded_filt_depth = roundup(filt_depth, groups);

    uint32_t pad_to_optimal_width = 0;
    uint32_t pad_to_optimal_height = 0;
    if (new_filt_width*chmul <= 3) pad_to_optimal_width = 3 - new_filt_width*chmul;
    else if (new_filt_width*chmul <= 5) pad_to_optimal_width = 5 - new_filt_width*chmul;
    else if (new_filt_width*chmul <= 7) pad_to_optimal_width = 7 - new_filt_width*chmul;

    if (new_filt_height < 2) pad_to_optimal_height = 2 - new_filt_height;
    uint32_t post_transform_padding = tcfparms->use_depthwise && (pad_to_optimal_height != 0 || pad_to_optimal_width != 0);

    uint32_t new_filt_size = new_num_filters * new_filt_height * new_filt_width * padded_filt_depth * element_size;
    uint32_t filter_buffer_size = new_filt_size;
    if (tcfparms->use_depthwise) {
        filter_buffer_size = new_num_filters * (new_filt_height + pad_to_optimal_height) * (new_filt_width*chmul + pad_to_optimal_width) * padded_filt_depth/chmul * element_size;
    }

    if (new_filt_size != tcfparms->data_size) {
        errlog(nn, "Calculated filter size %d doesn't match allocated filter size %d", new_filt_size, tcfparms->data_size);
    }

    int subfilt_1x1 = (new_filt_height == 1 && new_filt_width == 1);
    uint8_t *padded_filt_data = nn_memalign(sizeof(HVX_Vector), filter_buffer_size);
    uint8_t *s2b_data = nn_memalign(sizeof(HVX_Vector), filter_buffer_size);
    if (!padded_filt_data || !s2b_data)
    {
        res = -1;
        goto done;
    }
    pad_filter(nn, filt_tensor->data, num_filters, filt_height, filt_width, filt_depth, padded_filt_data, padded_num_filters, padded_filt_height, padded_filt_width, padded_filt_depth,
               element_size, tcfparms->zero_offset, 0);

    // 'space to batch' transpose:
    // padded_filt_data -> s2b_data
    {
        // weights are split: { num_filters, filt_height, filt_width, filt_depth } -> { num_filters, Fsh, block_h, Fsw, block_w, filt_depth }
        // weights are transposed: { num_filters, Fsh, block_h, Fsw, block_w, filt_depth } -> { block_h, block_w, num_filters, Fsh, Fsw, filt_depth }
        uint32_t dims_in[5] = {padded_num_filters * padded_filt_height / block_h, block_h, padded_filt_width / block_w, block_w, padded_filt_depth};
        int32_t perm[5] = {1, 3, 0, 2, 4};
        res = nn_transpose_operation(NULL, s2b_data, padded_filt_data,
                                     element_size,
                                     perm, 5,
                                     dims_in, 5,
                                     tcfparms->out_data, new_filt_size); // scratch-
        if (res != 0)
            goto done;
    }
    if(tcfparms->use_subkerneling) {
        // weights are transposed: { block_h, block_w, num_filters, Fsh, Fsw, filt_depth } -> { block_h, block_w, Fsh, Fsw, filt_depth, num_filters }
        uint32_t dims_in[3] = { block_h*block_w, padded_num_filters, new_filt_height*new_filt_width*padded_filt_depth };
        int32_t perm[3] = {0, 2, 1 };
        uint8_t * dest = padded_filt_data;
        void * scratch = tcfparms->out_data;
        // normally s2b_data ->padded_filt_data;
        // but if subfilt_height and subfilt_width are both 1, this is the last
        // step, so write it directly to output area
        if( subfilt_1x1){
            dest = tcfparms->out_data;
            scratch = padded_filt_data;
        }

        res =  nn_transpose_operation( NULL, dest, s2b_data,
                                       element_size,
                                       perm, 3,
                                       dims_in, 3,
                                       scratch, new_filt_size);
        if (res!=0) goto done;
    } else {
        // weights are transposed: { block_h * block_w * num_filters, Fsh, Fsw, filt_depth } -> { Fsh, Fsw, filt_depth, block_h * block_w * num_filters }
        uint32_t dims_in[4] = {block_h * block_w * padded_num_filters, new_filt_height, new_filt_width, padded_filt_depth};
        int32_t perm[4] = {1, 2, 3, 0};
        uint8_t *dest = padded_filt_data;
        void *scratch = tcfparms->out_data;

        // normally s2b_data ->padded_filt_data;
        // but if subfilt_height and subfilt_width are both 1, this is the last
        // step, so write it directly to output area
        if (subfilt_1x1 && groups == 1)
        {
            dest = tcfparms->out_data;
            scratch = padded_filt_data;
        }

        res = nn_transpose_operation(NULL, dest, s2b_data,
                                     element_size,
                                     perm, 4,
                                     dims_in, 4,
                                     scratch, new_filt_size);

        if (res != 0)
            goto done;
    }


    uint8_t *rotate_out_data = NULL;
    if (!subfilt_1x1) { //rotate filter 180 degrees
        if (tcfparms->use_subkerneling) {
            if (chmul > 1 || post_transform_padding) {
                rotate_out_data = s2b_data;
            }  else {
                rotate_out_data = tcfparms->out_data;
            }
            uint32_t subkernel_bytes = padded_num_filters * new_filt_height * new_filt_width * padded_filt_depth * element_size;
            for (int i = 0; i < block_h * block_w; i++)
            {
                uint8_t *transposed_slice = &padded_filt_data[i * subkernel_bytes];
                rotate_filter(nn,
                              transposed_slice, padded_num_filters, new_filt_height,
                              new_filt_width, padded_filt_depth* element_size,
                              &rotate_out_data[i * subkernel_bytes]);
            }
        } else {
            if (groups == 1 && chmul == 1 && !post_transform_padding) {
                rotate_out_data = tcfparms->out_data;
            } else {
                rotate_out_data = s2b_data;
            }
            rotate_filter(nn,
                          padded_filt_data, new_num_filters,
                          new_filt_height, new_filt_width, padded_filt_depth * element_size,
                          rotate_out_data);
        }
    } else {
        rotate_out_data = padded_filt_data;
    }

    // In channel Multiplier case we do axis shuffle along the input depth
    uint8_t *shuffle_out_data = NULL;
    if (chmul > 1) {
        if (post_transform_padding) {
            if (rotate_out_data == s2b_data) {
                shuffle_out_data = padded_filt_data;
            } else {
                shuffle_out_data = s2b_data;
            }
        } else {
            shuffle_out_data = tcfparms->out_data;
        }

        if (tcfparms->use_subkerneling) {
            // split { block_h, block_w, Fsh, Fsw, filt_depth, num_filters } to { block_h, block_w, Fsh, Fsw, chmul, filt_depth/chmul, num_filters }
            // transpose to { block_h, block_w, Fsh, Fsw, filt_depth/chmul, chmul, num_filters }
            uint32_t dims_in[4] = { block_h*block_w*new_filt_height*new_filt_width, padded_filt_depth / chmul, chmul, padded_num_filters};
            int32_t perm[4] = {0,2,1,3};

            res = nn_transpose_operation(NULL, shuffle_out_data, rotate_out_data,
                                         element_size,
                                         perm, 4,
                                         dims_in, 4,
                                         NULL, new_filt_size);

            if (res != 0)
                goto done;

        } else {
            // split { Fsh, Fsw, filt_depth, new_num_filters} to { Fsh, Fsw, chmul, filt_depth/chmul, new_num_filters}
            // transpose to { Fsh, Fsw, filt_depth/chmul, chmul, new_num_filters}
            uint32_t new_num_filters = block_h * block_w * padded_num_filters;
            uint32_t dims_in[4] = {new_filt_height * new_filt_width, padded_filt_depth / chmul, chmul, new_num_filters};
            int32_t perm[4] = {0,2,1,3};

            res = nn_transpose_operation(NULL, shuffle_out_data, rotate_out_data,
                                         element_size,
                                         perm, 4,
                                         dims_in, 4,
                                         NULL, new_filt_size);

            if (res != 0)
                goto done;
        }
    } else {
        shuffle_out_data = rotate_out_data;
    }

    // For depthwise case pad filter up to optimized heights of > 2 and widths of 3 or 5 or 7
    if (post_transform_padding) {
        if (tcfparms->use_subkerneling) {
            // pad from { block_h*block_w, Fsh, Fsw, filt_depth*num_filters}
            // to { block_h*block_w, Fsh+height_pad, Fsw+width_pad, filt_depth*num_filters}
            pad_filter(nn, shuffle_out_data, block_h * block_w, new_filt_height, new_filt_width*chmul,(padded_filt_depth/chmul)*padded_num_filters,
                       tcfparms->out_data, block_h * block_w, new_filt_height + pad_to_optimal_height, new_filt_width*chmul + pad_to_optimal_width, (padded_filt_depth/chmul)*padded_num_filters, element_size, tcfparms->zero_offset, 1);
        } else {
            // pad from { Fsh, Fsw, filt_depth, block_h*block_w*num_filters }
            // to { Fsh+height_pad, Fsw+width_pad, filt_depth, block_h*block_w*num_filters }
            pad_filter(nn, shuffle_out_data, new_filt_height, new_filt_width*chmul, (padded_filt_depth/chmul), block_h*block_w*padded_num_filters,
                       tcfparms->out_data, new_filt_height + pad_to_optimal_height, new_filt_width*chmul + pad_to_optimal_width, (padded_filt_depth/chmul), block_h*block_w*padded_num_filters, element_size, tcfparms->zero_offset, 1);
        }
    }

    if (groups > 1)    {
        uint32_t filter_split_height = new_filt_height * new_filt_width;
        uint32_t filter_split_size = new_filt_size / groups;
        uint32_t filt_split_buffer_size = roundf(filter_split_size / sizeof(HVX_Vector) + 0.5f) * sizeof(HVX_Vector);
        if(nn_split_data_hvx_aligned(nn, rotate_out_data, (uint8_t **)tcfparms->out_data, sizeof(uint8_t), filt_split_buffer_size, filter_split_height,
                padded_filt_depth, new_num_filters, groups, 1)) {
            return errlog(nn, "Failed to pre-split filter tensor!");
        }
    }
done:
    if (padded_filt_data != NULL)
        nn_free(padded_filt_data);
    if (s2b_data != NULL)
        nn_free(s2b_data);
    return res;
}
