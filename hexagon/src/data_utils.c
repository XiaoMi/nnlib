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

#include <data_utils.h>
#include <stdint.h>
#include <nn_graph.h>

void free_splits(uint8_t **ptr, int len) {
    int i;
    for(i=len-1; i>=0; i--) {
        nn_free(ptr[i]);
    }
}

/*
 * Splits a 3-D array into numOut buffers, each of which are address-aligned and size-aligned to 128
 * On non-zero return (error), frees all previously allocated buffers
 * On zero return (sucess), caller must free the allocated buffers after finished using them
 */
int nn_split_data_hvx_aligned(struct nn_graph *nn, uint8_t *data, uint8_t **splits, uint32_t elsize, uint32_t split_buffer_size, uint32_t height,
        uint32_t width, uint32_t depth, uint32_t numOut, uint32_t axis)
{
    if (split_buffer_size % sizeof(HVX_Vector) != 0)
        return errlog(nn, "ERROR: split_data_hvx_aligned: split_buffer_size must be a multiple of %d", sizeof(HVX_Vector));

    uint32_t in_dims[3] = { height, width, depth };
    uint32_t out_dims[3] = { height, width, depth };
    if (out_dims[axis] % numOut != 0)
        return errlog(nn, "ERROR: split_data_hvx_aligned: Data is not divisible by numOut at the specified axis!");
    out_dims[axis] /= numOut;

    uint32_t copy_stride = elsize;
    uint32_t in_start_stride = elsize;
    uint32_t in_copy_stride = elsize;
    uint32_t out_start_stride = elsize;
    uint32_t out_copy_stride = elsize;

    int i;
    for (i = 0; i < 3; i++) {
        if (i < axis) {
            copy_stride *= in_dims[i];
        } else {
            in_start_stride *= out_dims[i];
            in_copy_stride *= in_dims[i];
            out_copy_stride *= out_dims[i];
        }
        out_start_stride *= out_dims[i];
    }

    uint8_t *in, *out;
    int j;
    for (i = 0; i < numOut; i++) {
        splits[i] = (uint8_t *) nn_memalign(sizeof(HVX_Vector), split_buffer_size);
        if(NULL == splits[i]) {
            free_splits(splits, i); //free previously allocated buffers
            return errlog(nn, "ERROR: split_data_hvx_aligned: Failed to allocate buffer of size %d for the %dth split", split_buffer_size, i);
        }
        in = data + i * in_start_stride;
        out = splits[i];
        for (j = 0; j < copy_stride; j++) {
            memcpy((void*)out, in, out_copy_stride);
            out += out_copy_stride;
            in += in_copy_stride;
        }
    }

    return 0;
}