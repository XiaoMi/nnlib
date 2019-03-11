
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

/*
* Strategy to process the weights for transposed convolution is as follows:
* Weights come in as dout, height, width, din
* Pad weights so that filt_height % stride_h == 0 and filt_width % stride_w == 0
* Batch to space (a transpose) -> stride_h * stride_w * dout, height / stride_h, width / stride_w, din
* We now have stride_h * stride_w subkernels
* Transpose each subkernel to height, width, din, dout
* Rotate 180 along spatial plane (effectively a vertical + horizontal flip)
*/

static int pad_filter(struct nn_graph *nn, const struct tensor * filt_tensor, uint8_t * padded_filt_data,
                    const uint32_t padded_num_filters, const uint32_t padded_filt_height, const uint32_t padded_filt_width, const uint32_t padded_filt_depth,
                    const uint32_t padval)
{
    uint32_t req_padding_num_filters = padded_num_filters - filt_tensor->shape.batches;
    uint32_t req_padding_filt_height = padded_filt_height - filt_tensor->shape.height;
    uint32_t req_padding_filt_width = padded_filt_width - filt_tensor->shape.width;
    uint32_t req_padding_filt_depth = padded_filt_depth - filt_tensor->shape.depth;
    do_pad(padded_filt_data, filt_tensor->data,
          filt_tensor->shape.batches, filt_tensor->shape.height, filt_tensor->shape.width, filt_tensor->shape.depth,
          0, req_padding_num_filters,
          0, req_padding_filt_height,
          0, req_padding_filt_width,
          0, req_padding_filt_depth,
          sizeof(uint8_t),
          padval
          );
    return 0;
}
static int space_to_batch(struct nn_graph *nn, const uint8_t *in_data,
                          const uint32_t num_filters, const uint32_t filt_height, const uint32_t filt_width, const uint32_t filt_depth,
                          const uint32_t block_h, uint32_t block_w, uint8_t *out_data)
{
    if (filt_height % block_h != 0 || filt_width % block_w != 0)
        return errlog(nn, "Transpose conv filter dims must be divisible by stride h/w");

    uint32_t out_idx = 0;

    for (uint32_t h_start = 0; h_start < block_h; h_start++)
    {
        for (uint32_t w_start = 0; w_start < block_w; w_start++)
        {
            for (uint32_t b = 0; b < num_filters; b++)
            {
                for (uint32_t h = h_start; h < filt_height; h += block_h)
                {
                    for (uint32_t w = w_start; w < filt_width; w += block_w)
                    {

                        uint32_t in_idx = b * (filt_height * filt_width * filt_depth) + h * (filt_width * filt_depth) + w * (filt_depth);
                        vmemcpy_asm(&out_data[out_idx], &in_data[in_idx], filt_depth);
                        out_idx += filt_depth;
                    }
                }
            }
        }
    }
    return 0;
}

//The transpose here can be thought of as a 2d transpose of  a matrix where the dims are num_filters * (filt_height * filt_width * filt_depth)
static int transpose_to_hwio(struct nn_graph *nn, uint8_t *in_data,
                             const uint32_t num_filters, const uint32_t filt_height, const uint32_t filt_width, const uint32_t filt_depth,
                             uint8_t *out_data)
{
    uint32_t outer_dim = num_filters;
    uint32_t inner_dim = filt_height * filt_width * filt_depth;
    for (int i = 0; i < outer_dim * inner_dim; i++)
    {
        int j = i / outer_dim;
        int k = i % outer_dim;
        out_data[i] = in_data[inner_dim * k + j];
    }
    return 0;
}

static int rotate_filter(struct nn_graph *nn, uint8_t *in_data,
                         const uint32_t num_filters, const uint32_t filt_height, const uint32_t filt_width, const uint32_t filt_depth,
                         uint8_t *out_data)
{
    uint32_t src_h = filt_height;
    for (uint32_t h = 0; h < filt_height; h++)
    {
        uint8_t *src = &in_data[src_h * (num_filters * filt_width * filt_depth) - num_filters * filt_depth];
        uint8_t *dst = &out_data[h * (num_filters * filt_width * filt_depth)];
        vmemcpy_2d_general_asm(num_filters * filt_depth,
                               filt_width,
                               dst,
                               (num_filters * filt_depth),
                               src,
                               -(num_filters * filt_depth));
        src_h--;
    }
    return 0;
}

int process_tranpose_conv_filter(struct nn_graph *nn, struct transpose_conv_filter_parms *tcfparms)
{
    const struct tensor *filt_tensor = tcfparms->filt_tensor;
    const struct tensor *strides_tensor = tcfparms->strides_tensor;
    uint32_t num_filters = filt_tensor->shape.batches;
    uint32_t filt_height = filt_tensor->shape.height;
    uint32_t filt_width = filt_tensor->shape.width;
    uint32_t filt_depth = filt_tensor->shape.depth;
    uint32_t block_h = strides_tensor->shape.height;
    uint32_t block_w = strides_tensor->shape.width;
    uint8_t pad_num_filters = (1 < block_h && block_h < 5 && 1 < block_w && block_w < 5) ? 1 : 0;
    //Padded filt stuff
    uint32_t padded_num_filters = (pad_num_filters) ? roundup(num_filters, 32) : num_filters;
    uint32_t padded_filt_height = roundup(filt_height, block_h);
    uint32_t padded_filt_width = roundup(filt_width, block_w);
    uint32_t padded_filt_depth = filt_depth;
    uint32_t padded_filt_size = padded_num_filters * padded_filt_height * padded_filt_width * padded_filt_depth;
    if (padded_filt_size != tcfparms->data_size) 
        return errlog(nn, "Calculated filter size %d doesn't match allocated filter size %d", padded_filt_size, tcfparms->data_size);
    uint8_t * padded_filt_data = nn_memalign(sizeof(HVX_Vector), padded_filt_size);
    uint8_t *s2b_data = nn_memalign(sizeof(HVX_Vector), padded_filt_size);
    pad_filter(nn, filt_tensor, padded_filt_data, padded_num_filters, padded_filt_height, padded_filt_width, padded_filt_depth, tcfparms->zero_offset);
    int res = 0;
    res = space_to_batch(nn,
                         padded_filt_data, padded_num_filters, padded_filt_height, padded_filt_width, padded_filt_depth,
                         block_h, block_w,
                         s2b_data);

    if (res)
    {
        nn_free(padded_filt_data);
        nn_free(s2b_data);
    }
    nn_free(padded_filt_data);
    uint8_t *processed_weights = tcfparms->out_data;
    uint8_t *transposed_weights_data = nn_memalign(128, padded_filt_size);
    uint32_t subkernel_size = padded_num_filters * padded_filt_height / block_h * padded_filt_width / block_w * padded_filt_depth;
    for (int i = 0; i < block_h * block_w; i++)
    {
        uint8_t *in_data = &s2b_data[i * subkernel_size];
        uint8_t *transposed_slice = &transposed_weights_data[i * subkernel_size];
        uint8_t *out_data = &processed_weights[i * subkernel_size];
        res = transpose_to_hwio(nn,
                                in_data, padded_num_filters, padded_filt_height / block_h, padded_filt_width / block_w, padded_filt_depth,
                                transposed_slice);
        if (padded_filt_height / block_h == 1 && padded_filt_width / block_w == 1)
            vmemcpy_asm(out_data, transposed_slice, subkernel_size);
        else
        { 
        res = rotate_filter(nn,
                            transposed_slice, padded_num_filters, padded_filt_height / block_h, filt_width / block_w, padded_filt_depth,
                            out_data);
        }
    }
    nn_free(s2b_data);
    nn_free(transposed_weights_data);
    return 0;
}