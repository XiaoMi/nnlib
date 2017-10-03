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

#include "hexagon_types.h"
#include <assert.h>

#define LOG2VLEN 7

typedef long HEXAGON_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HEXAGON_Vect_UN*)(A))

#define min(_a, _b) (((_a)<(_b)) ? (_a) : (_b))

void resizebilinear_kernel2(
	const unsigned char *in,
	unsigned char       *out,
	int                  b_in,
	int                  newheight,
	int                  newwidth,
	int                  d_in
) {
	int const2w = 2;
	int h_in = newheight >> 1;
	int w_in = newwidth >> 1;

	HVX_Vector *pSrc0 = (HVX_Vector *)in;
	HVX_Vector *pDst0 = (HVX_Vector *)out;
	HVX_Vector *pSrc1 = (HVX_Vector *)(in + w_in * d_in);
	HVX_Vector *pDst1 = (HVX_Vector *)(out + newwidth * d_in);
	int negd_in = 128 - d_in;
	int ohstride = (newwidth * d_in) >> LOG2VLEN;
	int ihstride = (w_in * d_in) >> LOG2VLEN;

	int const1b = 0x1010101;
	HVX_VectorPred Q0 = Q6_Q_vsetq_R(negd_in);
	int lp0cnt = w_in * d_in;

	lp0cnt = lp0cnt >> LOG2VLEN;
	negd_in = negd_in - 128;

	if (d_in < 128) {
		if (w_in*d_in >= 128) {
			for (int b = 0; b < b_in; b++) {
				pSrc1 = pSrc0 + ihstride;
				for (int h = 0; h < h_in; h++) {
					if (h == h_in - 1) pSrc1 = pSrc0;
					HVX_Vector sL0cur = *pSrc0++;
					HVX_Vector sL1cur = *pSrc1++;
					HVX_Vector sL0nxt = Q6_V_vzero();
					HVX_Vector sL1nxt = Q6_V_vzero();
					for (int z = 0; z < lp0cnt - 1; z++) {
						sL0nxt = *pSrc0++;
						HVX_Vector sL0s1 = Q6_V_valign_VVR(sL0nxt, sL0cur, d_in);

						sL1nxt = *pSrc1++;
						HVX_Vector sL1s1 = Q6_V_valign_VVR(sL1nxt, sL1cur, d_in);

						HVX_VectorPair dL10cur = Q6_W_vcombine_VV(sL1cur, sL0cur);
						HVX_VectorPair dAcc = Q6_Wh_vmpa_WubRb(dL10cur, const1b);

						HVX_Vector sL0Odd = Q6_Vub_vavg_VubVub_rnd(sL0cur, sL0s1);

						HVX_VectorPair dL10s1 = Q6_W_vcombine_VV(sL1s1, sL0s1);
						dAcc = Q6_Wh_vmpaacc_WhWubRb(dAcc, dL10s1, const1b);

						HVX_VectorPair dOut0 = Q6_W_vshuff_VVR(sL0Odd, sL0cur, negd_in);
						HVX_Vector sL1Even = Q6_Vub_vavg_VubVub_rnd(sL0cur, sL1cur);

						HVX_Vector sL1Odd = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAcc), Q6_V_lo_W(dAcc), const2w);

						sL1cur = sL1nxt;
						sL0cur = sL0nxt;

						*pDst0++ = Q6_V_lo_W(dOut0);
						*pDst0++ = Q6_V_hi_W(dOut0);
						HVX_VectorPair dOut1 = Q6_W_vshuff_VVR(sL1Odd, sL1Even, negd_in);
						*pDst1++ = Q6_V_lo_W(dOut1);
						*pDst1++ = Q6_V_hi_W(dOut1);
					}
					HVX_Vector sL0s1 = Q6_V_valign_VVR(sL0cur, sL0cur, d_in);
					sL0s1 = Q6_V_vmux_QVV(Q0, sL0s1, sL0cur);

					HVX_Vector sL1s1 = Q6_V_valign_VVR(sL1cur, sL1cur, d_in);
					sL1s1 = Q6_V_vmux_QVV(Q0, sL1s1, sL1cur);

					HVX_VectorPair dL10cur = Q6_W_vcombine_VV(sL1cur, sL0cur);
					HVX_VectorPair dAcc = Q6_Wh_vmpa_WubRb(dL10cur, const1b);

					HVX_Vector sL0Odd = Q6_Vub_vavg_VubVub_rnd(sL0cur, sL0s1);

					HVX_VectorPair dL10s1 = Q6_W_vcombine_VV(sL1s1, sL0s1);
					dAcc = Q6_Wh_vmpaacc_WhWubRb(dAcc, dL10s1, const1b);

					HVX_VectorPair dOut0 = Q6_W_vshuff_VVR(sL0Odd, sL0cur, negd_in);
					HVX_Vector sL1Even = Q6_Vub_vavg_VubVub_rnd(sL0cur, sL1cur);

					HVX_Vector sL1Odd = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAcc), Q6_V_lo_W(dAcc), const2w);

					*pDst0++ = Q6_V_lo_W(dOut0);
					*pDst0++ = Q6_V_hi_W(dOut0);
					HVX_VectorPair dOut1 = Q6_W_vshuff_VVR(sL1Odd, sL1Even, negd_in);
					*pDst1++ = Q6_V_lo_W(dOut1);
					*pDst1++ = Q6_V_hi_W(dOut1);
					pDst0 = pDst1;
					pDst1 += ohstride;
				}
			}
		}
		else {
			Q0 = Q6_Q_vsetq_R(d_in*(w_in-1));
			for (int b = 0; b < b_in; b++) {
				for (int y = 0; y < h_in; y++) {
					int y1 = min(h_in-1, y + 1);
					HVX_Vector *pSrc0 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y*w_in*d_in);
					HVX_Vector *pSrc1 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y1*w_in*d_in);
					HVX_Vector *pDst0 = (HVX_Vector *)(out + b*newheight*newwidth*d_in + (2 * y + 0)*newwidth*d_in);
					HVX_Vector *pDst1 = (HVX_Vector *)(out + b*newheight*newwidth*d_in + (2 * y + 1)*newwidth*d_in);

					HVX_Vector sL0cur = vmemu(pSrc0); pSrc0++;
					HVX_Vector sL1cur = vmemu(pSrc1); pSrc1++;

					HVX_Vector sL0s1 = Q6_V_valign_VVR(sL0cur, sL0cur, d_in);
					sL0s1 = Q6_V_vmux_QVV(Q0, sL0s1, sL0cur);

					HVX_Vector sL1s1 = Q6_V_valign_VVR(sL1cur, sL1cur, d_in);
					sL1s1 = Q6_V_vmux_QVV(Q0, sL1s1, sL1cur);

					HVX_VectorPair dL10cur = Q6_W_vcombine_VV(sL1cur, sL0cur);
					HVX_VectorPair dAcc = Q6_Wh_vmpa_WubRb(dL10cur, const1b);

					HVX_Vector sL0Odd = Q6_Vub_vavg_VubVub_rnd(sL0cur, sL0s1);

					HVX_VectorPair dL10s1 = Q6_W_vcombine_VV(sL1s1, sL0s1);
					dAcc = Q6_Wh_vmpaacc_WhWubRb(dAcc, dL10s1, const1b);

					HVX_VectorPair dOut0 = Q6_W_vshuff_VVR(sL0Odd, sL0cur, negd_in);
					HVX_Vector sL1Even = Q6_Vub_vavg_VubVub_rnd(sL0cur, sL1cur);

					HVX_Vector sL1Odd = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAcc), Q6_V_lo_W(dAcc), const2w);

					vmemu(pDst0) = Q6_V_lo_W(dOut0); pDst0++;
					HVX_VectorPair dOut1 = Q6_W_vshuff_VVR(sL1Odd, sL1Even, negd_in);
					vmemu(pDst1) = Q6_V_lo_W(dOut1); pDst1++;
				}
			}
		}
	}
	else {
		assert(d_in % 128 == 0);

		for (int b = 0; b < b_in; b++) {
			pSrc1 = pSrc0 + ihstride;
			for (int y = 0; y < h_in; y++) {
				int y1 = min(h_in - 1, y + 1); // y-boundary
				for (int x = 0; x < w_in; x++) {
					int x1 = min(w_in-1,x+1); // x-boundary
					HVX_Vector *pSrc00 = (HVX_Vector *)(in+b*h_in*w_in*d_in+ y*w_in*d_in+ x*d_in);
					HVX_Vector *pSrc01 = (HVX_Vector *)(in+b*h_in*w_in*d_in+ y*w_in*d_in+x1*d_in);
					HVX_Vector *pSrc10 = (HVX_Vector *)(in+b*h_in*w_in*d_in+y1*w_in*d_in+ x*d_in);
					HVX_Vector *pSrc11 = (HVX_Vector *)(in+b*h_in*w_in*d_in+y1*w_in*d_in+x1*d_in);
					HVX_Vector *pDst00 = (HVX_Vector *)(out+b*newheight*newwidth*d_in+(2*y+0)*newwidth*d_in+(2*x+0)*d_in);
					HVX_Vector *pDst01 = (HVX_Vector *)(out+b*newheight*newwidth*d_in+(2*y+0)*newwidth*d_in+(2*x+1)*d_in);
					HVX_Vector *pDst10 = (HVX_Vector *)(out+b*newheight*newwidth*d_in+(2*y+1)*newwidth*d_in+(2*x+0)*d_in);
					HVX_Vector *pDst11 = (HVX_Vector *)(out+b*newheight*newwidth*d_in+(2*y+1)*newwidth*d_in+(2*x+1)*d_in);
					for (int d = 0; d < d_in; d+=128) {
						HVX_Vector sL00 = *pSrc00++;
						HVX_Vector sL01 = *pSrc01++;
						HVX_Vector sL10 = *pSrc10++;
						HVX_Vector sL11 = *pSrc11++;

						HVX_VectorPair dL10cur = Q6_W_vcombine_VV(sL10, sL00);
						HVX_VectorPair dAcc = Q6_Wh_vmpa_WubRb(dL10cur, const1b);

						HVX_Vector sL0Odd = Q6_Vub_vavg_VubVub_rnd(sL01, sL00);

						HVX_VectorPair dL10s1 = Q6_W_vcombine_VV(sL11, sL01);
						dAcc = Q6_Wh_vmpaacc_WhWubRb(dAcc, dL10s1, const1b);

						HVX_Vector sL1Even = Q6_Vub_vavg_VubVub_rnd(sL10, sL00);

						HVX_Vector sL1Odd = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAcc), Q6_V_lo_W(dAcc), const2w);

						*pDst00++ = sL00;
						*pDst01++ = sL0Odd;
						*pDst10++ = sL1Even;
						*pDst11++ = sL1Odd;
					}
				}
			}
		}
	}
}


