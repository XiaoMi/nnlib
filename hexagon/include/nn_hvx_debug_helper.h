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
#ifndef HEXAGON_NN_NN_HVX_DEBUG_HELPER_H
#define HEXAGON_NN_NN_HVX_DEBUG_HELPER_H

#include "quantize.h"
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_inlines.h"

//ref code for printing HVX vectors for debug
//print unsigned byte, 32 bytes per line
static inline void printHVXu8(HVX_Vector vin, int size) {
    uint8_t * vec = (uint8_t *)nn_memalign(128, 128);
    HVX_Vector * vec1 = (HVX_Vector *)vec;
    *vec1 = vin;
    for( int i = 0; i < size; ++i ) {
        if(i%32 == 0) printf("\n");
        printf(" %hhu", vec[i]);//
    }
    printf("\n");
}

//print signed byte
static inline void printHVX8(HVX_Vector vin, int size) {
    uint8_t * vec = (uint8_t *)nn_memalign(128, 128);
    HVX_Vector * vec1 = (HVX_Vector *)vec;
    *vec1 = vin;
    for( int i = 0; i < size; ++i ) {
        if(i%32 == 0) printf("\n");
        printf(" %hhi", vec[i]);//
    }
    printf("\n");
}

//print words
//size is the number of words
static inline void printW(HVX_Vector vin, int size) {
    int32_t * vec = (int32_t *)nn_memalign(128, 128);
    HVX_Vector * vec1 = (HVX_Vector *)vec;
    *vec1 = vin;
    for( int i = 0; i < size; ++i ) {
        if(i%8 == 0) printf("\n");
        printf(" %ld", vec[i]);//
    }
    printf("\n");
}

#endif