
/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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
 * This contains definitions for things used internally.
 */

#if 0
inline void prelu_1D_alpha(uint8_t *in_data, uint8_t *out_data, float *alpha, uint32_t quantized_zero,size_t bytes , size_t alpha_depth )
{

	uint32_t *alpha_frac   = pad_and_align(alpha, sizeof(float)*alpha_depth);
	uint32_t *alpha_offset = pad_and_align(alpha_frac, NUM_BYT_PERVECTOR);
	int i, j,idx;

	for (i = 0; i < bytes/alpha_depth; i++) {
		for (j = 0; j < alpha_depth; j++) {
			idx = i*alpha_depth +j;
			out_data[idx] = in_data[idx];
			if (in_data[idx] < quantized_zero) {
				alpha_frac[j] = (1<<16) * alpha[j];
				alpha_offset[j] = quantized_zero - ((quantized_zero * alpha_frac[j] + 0x08000) >> 16);
				out_data[idx] = ((in_data[idx] * alpha_frac[j]+0x08000) >> 16) + alpha_offset[j];
			}
		}
	}





}

#endif
	.text

	.global prelu_1d_alpha_hvx
	.p2align 6
prelu_1d_alpha_hvx:
	.falign
.Loop:
#if 0   //TBD
#endif
	jumpr r31


