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

/*======================================================================*/
/*     ASSUMPTIONS                                                      */
/*        output Data is 128byte aligned                                */
/*======================================================================*/
// NoteA: "nonlin_coef.h" header file used as a medium for passing macros computed by Python to intrinsic/ASM code.
// - macros consist of parameters like coef_scale, remez_order, sature_max, sature_min, num_bits_lut_integral_part, num_bits_lut_fractional_part, Q format use etc. 
// - macros also consist of defines to skip parts of code that are unused (for example skipping code based on cases of odd and even symmetry)
// NoteB: "nonlin_coef.h" header file has two different formats for polynomial-approximation coefficient-containing Look-Up Table (LUT):
// - lut_non_lin_cn (for use as reference under CNATURAL_NONLIN_DEFS) in "natural-C" format - row of increasing-poly-order-wise coeffs for 1st rng/interval then row for 2nd interval and so on.
// - lut_non_lin_asm[] (under CINTRINSIC_NONLIN_DEFS) in different format as table is put in "vectorized" format for lookup using HVX vectors in intrinsic/ASM code.
// NoteC: "nonlin_coef.h" header file has function prototypes (under CINTRINSIC_NONLIN_DEFS) to allow calls to intrinsic/ASM code functions by C test code
extern const signed short lut_non_lin_exp2_16[];
static inline void non_lin_i_exp2_16(signed short *ptr_y, signed short *ptr_x, int numelements)
{
	int	poly_n_order = N_ORDER_EXP2_16;
	NON_LIN_I_16(ptr_y, ptr_x, lut_non_lin_exp2_16, numelements, poly_n_order, RNGIDX_MASK_EXP2_16, RNGIDX_RSHIFT_EXP2_16, DELTAX_MASK_EXP2_16, DELTAX_LSHIFT_EXP2_16)
    return;
}
