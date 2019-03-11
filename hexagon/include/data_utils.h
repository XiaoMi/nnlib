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

#ifndef NN_DATA_UTILS_H
#define NN_DATA_UTILS_H 1

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains definitions for data manipulation routines 
 */

#include <stdint.h>

static inline int split_data(uint8_t *data, uint8_t *splits, uint32_t height,
        uint32_t width, uint32_t depth, uint32_t numOut, uint32_t axis)
{
    uint32_t in_dims[3] = { height, width, depth };
    uint32_t out_dims[3] = { height, width, depth };
    if (out_dims[axis] % numOut != 0)
        return -1;
    out_dims[axis] /= numOut;

    uint32_t copy_stride = 1;
    uint32_t in_start_stride = 1;
    uint32_t in_copy_stride = 1;
    uint32_t out_start_stride = 1;
    uint32_t out_copy_stride = 1;

    uint32_t i;
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
    uint32_t j;
    for (i = 0; i < numOut; i++) {
        in = data + i * in_start_stride;
        out = splits + i * out_start_stride;
        for (j = 0; j < copy_stride; j++) {
            memcpy((void*)out, in, out_copy_stride);
            out += out_copy_stride;
            in += in_copy_stride;
        }
    }

    return 0;
}

#endif
