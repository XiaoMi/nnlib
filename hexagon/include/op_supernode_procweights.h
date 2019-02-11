
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

#ifndef OP_SUPERNODE_PROCWEIGHTS_H_
#define OP_SUPERNODE_PROCWEIGHTS_H_

#include <stdint.h>
#include "nn_graph_types.h"
// This struct contains all the parms to repack the filter coeffs.
struct repack_filter_parms {
  uint8_t *out_data;            		// out area (aligned)
  struct tensor const * filt_tensor;  	// input tensor, data & shape
  int16_t zero_offset;      			// byte to fill when padding
  int16_t signed_mode_sel;				// process to 'signed' mode: 0 = no, 1= offset only, 2= offset & shift.
  int32_t * gemsumb;					// area to put the the coefficient sums (aligned; or NULL)
  float    coeffscale;					// output: parms were scaled by this much (if signed_mode != 0)
  int32_t * scalefac_vec;               //array of scale factor per channel

  nn_sem_t done_sem;
};



//
// This executes the repack operation defined in the struct.
//  if signed_mode_sel = 0:
//    - values are repacked to the output area, and
//      the gemsumb is found on the actual values; it is stored at gemsumb.
//  if signed_mode_sel = 1:
//    - values are packed to output area, with 'zero_offset' subtracted from all,
//    - if zero_offset is in range 121..136, we subtract the offset and the result is saturated to
//        -128 .. 127;
//    - otherwise, we subtract the zero offset and divide by 2, (rounding towards 0), result being -127..127.
//    - in either case the gemsumb is the sum of these signed values.
//
// 'coeffscale' is set to the value by which the coefficients were scaled (1.0 or 0.5)
//
// it is ok if gemsumb is NULL; it will do everything else and not store gemsumb.
//
// Note:
// for signed_mode_sel = 0:
//   - Any elements in the 'padding' added to depth & height dimensions will be = zero_offset;
//   - gemsumb reflects this (so, in 'padding' batches, it will be
//         filter_width * filter_height * filt_batches_padded * zero_offset).
// for signed_mode_sel= 1
//   - Any elements in the 'padding' added to depth & height dimensions will be 0;
//   - gemsumb reflects this (so, in 'padding' batches, it will be 0).
//
// when compiled for < V65, only signed_mode_sel = 0 is supported.
//
//

//-> the vrpfp parameter is really a struct repack_filter_parms *.
//-> operation posts vrpfp->done_sem when done.
//
void repack_filter_for_d32( struct nn_graph *nn, void *vrpfp );
#endif /* OP_SUPERNODE_PROCWEIGHTS_H_ */
