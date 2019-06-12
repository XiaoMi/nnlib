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

#ifndef HEXAGON_NN_NN_OPER16_H
#define HEXAGON_NN_NN_OPER16_H

#include <stdint.h>

/* header for oper16.c
 * This contains misc ops on 16-bit tensors which can be shared across nodes.
 */
//
// this applies a given 32-bit gain (with 16 fractional bits) to an s16 array;
// the operation is
//   out[i] =  saturate_i16(  round( inp[i] * gain /65536) )
//
// There are special cases for gain = 0x10000 (copy), gain = 0 (fill 0)
// and gain-fits-in-i16 (faster path).
// The output buffer may be identical to the input, but otherwise may not
// overlap. input, output pointers must be 16-bit aligned.
//
//
void nn_do_scale_s16( int16_t * outp, int16_t const * inp, int32_t gain, int n);
// same, hvx
void nn_do_scale_s16_hvx( int16_t * outp, int16_t const * inp, int32_t gain, int n);


//
// This does scale and/or offset of u16 tensors, and can also be used to convert between u16/s16 ranges.
//
//    parameters:
//         uint32    inout_offset:   LS word must be 0x8000 if input is u16,  0 if input is s16
//                                   MS word must be 0x8000 if output is u16,  0 if output is s16
//        int32_t gain	   :   a signed 32-bit gain value
//        int32_t offset   :   a signed 32-bit offset term
//
// The operation is equivalent to the below (assuming saturation does not occur):
// For  u16->u16 (inout_offset = 0x80008000):
//     out[i] = (gain * 2^-18) * in[i] + [  (offset-gain)*2^-3 + 32768]
// For u16->s16 (inout_offs = 0x8000):
//     out[i] = (gain * 2^-18) * in[i] + [  (offset-gain)*2^-3]
// For s16->u16 ( (inout_offs = 0x80000000:)
//     out[i] = (gain * 2^-18) * in[i] + [  offset*2^-3 + 32768]
// For s16->s16 ( inout_offs = 0:)
//     out[i] = (gain * 2^-18) * in[i] + [  offset*2^-3]

void nn_do_scaleoff_16to16( uint16_t * outp, uint16_t const * inp, uint32_t inout_offs, int32_t gain, int32_t offset, int n);
// same, hvx
void nn_do_scaleoff_16to16_hvx( uint16_t * outp, uint16_t const * inp, uint32_t inout_offs, int32_t gain, int32_t offset, int n);

#endif //HEXAGON_NN_NN_OPER16_H
