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


#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include "hexagon_types.h"
#endif
#include "hvx_hexagon_protos.h"
#include "float_mathops.h"
#include "hvx_inlines.h"
#include "nn_bufferpool.h"

//==================================================================================
#define NUM_THREADS 2

#if defined(__hexagon__)
typedef long HEXAGON_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HEXAGON_Vect_UN*)(A))
#endif

//==================================================================================
struct resizebilinear_plan{
	int32_t in_height, in_width;
	int32_t out_height, out_width;
	int64_t vscale;		// in_height/out_height, with 32 fractional bits
	uint8_t *xoff;
	struct edgeidx {
		uint8_t start;
		uint8_t end;
	} *xload;
	uint8_t pad[96];
	uint16_t xfrac[0];
};

struct bilin_runstate {
	const uint8_t *tin;
	uint8_t *tout;

	int32_t depth;
	int32_t batches;

	int32_t in_ht;		// input height, width
	int32_t in_wh;
	int32_t out_ht;		// output height, width
	int32_t out_wh;

	int32_t blockrow; // processing row
	int32_t inner_count;
	int32_t in_bstride;
	int32_t in_hstride;
	int32_t out_bstride;
	int32_t out_hstride;
	int32_t factor;
	int32_t align_corners;
	void (*resizebilinear_ptr)(
		const uint8_t  *in,
		uint8_t        *out,
		int32_t         h_in,
		int32_t         w_in,
		int32_t         d_in,
		int32_t         h_out,
		int32_t         w_out,
		int32_t         h_in_end,
		struct bilin_runstate *rtsp);

	struct resizebilinear_plan *planp;
	struct buffer_pool intermed_bufs;
	nn_sem_t done_sem;
	int32_t jobs;		// total # of 'subsections' to run
	volatile int32_t next_job;	 // index of next subsection
};

//==================================================================================
static void q6op_vmemu_partial(HVX_Vector *addr, HVX_Vector vreg, int nwrite) {
#if defined(__hexagon__)
#define alignedaddr addr
#else
	HVX_Vector *alignedaddr = (HVX_Vector *)((size_t)addr&-128);
#endif
	HVX_VectorPred Q1, Q2;
	int endpos = ((uint32_t)addr & 0x7f) + nwrite;
	Q1 = Q6_Q_vsetq_R((int32_t)addr);
	Q2 = q6op_Q_vsetq2_R(min_i32(128, endpos));
	Q1 = Q6_Q_xor_QQ(Q1, Q2);
	HVX_Vector vregror = Q6_V_vror_VR(vreg, -(int32_t)addr);
	q6op_vstcc_QAV(Q1, alignedaddr, vregror);
	Q1 = Q6_Q_vsetq_R(max_i32(0, endpos - 128));
	q6op_vstcc_QAV(Q1, alignedaddr + 1, vregror);
}

void resizebilinear_2x(
	const uint8_t  *in,
	uint8_t        *out,
	int32_t         h_in,
	int32_t         w_in,
	int32_t         d_in,
	int32_t         h_out,
	int32_t         w_out,
	int32_t         h_in_end,
	struct bilin_runstate *rtsp
) {
	const int32_t b_in = 1;
	int32_t const2w = 2;

	HVX_Vector *pSrc0 = (HVX_Vector *)in;
	HVX_Vector *pDst0 = (HVX_Vector *)out;
	HVX_Vector *pSrc1 = (HVX_Vector *)(in + w_in * d_in);
	HVX_Vector *pDst1 = (HVX_Vector *)(out + w_out * d_in);
	int32_t negd_in = 128 - d_in;
	int32_t ohstride = (w_out * d_in) >> 7;
	int32_t ihstride = (w_in * d_in) >> 7;
	int32_t nwrite = (w_in*d_in * 2) & 0x7f;

	int32_t const1b = 0x1010101;
	HVX_VectorPred Q0 = Q6_Q_vsetq_R(negd_in);
	int32_t lp0cnt = w_in * d_in;

	lp0cnt = lp0cnt >> 7;
	negd_in = negd_in - 128;

	if (d_in < 128) {
		if (w_in*d_in >= 128) {
			for (int32_t b = 0; b < b_in; b++) {
				pSrc1 = pSrc0 + ihstride;
				for (int32_t h = 0; h < h_in; h++) {
					if (h == h_in_end - 1) pSrc1 = pSrc0;
					HVX_Vector sL0cur = *pSrc0++;
					HVX_Vector sL1cur = *pSrc1++;
					HVX_Vector sL0nxt = Q6_V_vzero();
					HVX_Vector sL1nxt = Q6_V_vzero();
					for (int32_t z = 0; z < lp0cnt - 1; z++) {
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
					q6op_vmemu_partial(pDst0++, Q6_V_hi_W(dOut0), nwrite);
					HVX_VectorPair dOut1 = Q6_W_vshuff_VVR(sL1Odd, sL1Even, negd_in);
					*pDst1++ = Q6_V_lo_W(dOut1);
					q6op_vmemu_partial(pDst1++, Q6_V_hi_W(dOut1), nwrite);
					pDst0 = pDst1;
					pDst1 += ohstride;
				}
			}
		}
		else {
			Q0 = Q6_Q_vsetq_R(d_in*(w_in - 1));
			for (int32_t b = 0; b < b_in; b++) {
				for (int32_t y = 0; y < h_in; y++) {
					int32_t y1 = min_i32(h_in_end - 1, y + 1);
					HVX_Vector *pSrc0 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y*w_in*d_in);
					HVX_Vector *pSrc1 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y1*w_in*d_in);
					HVX_Vector *pDst0 = (HVX_Vector *)(out + b*h_out*w_out*d_in + (2 * y + 0)*w_out*d_in);
					HVX_Vector *pDst1 = (HVX_Vector *)(out + b*h_out*w_out*d_in + (2 * y + 1)*w_out*d_in);

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

					q6op_vmemu_partial(pDst0++, Q6_V_lo_W(dOut0), nwrite);
					HVX_VectorPair dOut1 = Q6_W_vshuff_VVR(sL1Odd, sL1Even, negd_in);
					q6op_vmemu_partial(pDst1++, Q6_V_lo_W(dOut1), nwrite);
				}
			}
		}
	}
	else {
		for (int32_t b = 0; b < b_in; b++) {
			pSrc1 = pSrc0 + ihstride;
			for (int32_t y = 0; y < h_in; y++) {
				int32_t y1 = min_i32(h_in_end - 1, y + 1); // y-boundary
				for (int32_t x = 0; x < w_in; x++) {
					int32_t x1 = min_i32(w_in - 1, x + 1); // x-boundary
					HVX_Vector *pSrc00 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y*w_in*d_in + x*d_in);
					HVX_Vector *pSrc01 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y*w_in*d_in + x1*d_in);
					HVX_Vector *pSrc10 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y1*w_in*d_in + x*d_in);
					HVX_Vector *pSrc11 = (HVX_Vector *)(in + b*h_in*w_in*d_in + y1*w_in*d_in + x1*d_in);
					HVX_Vector *pDst00 = (HVX_Vector *)(out + b*h_out*w_out*d_in + (2 * y + 0)*w_out*d_in + (2 * x + 0)*d_in);
					HVX_Vector *pDst01 = (HVX_Vector *)(out + b*h_out*w_out*d_in + (2 * y + 0)*w_out*d_in + (2 * x + 1)*d_in);
					HVX_Vector *pDst10 = (HVX_Vector *)(out + b*h_out*w_out*d_in + (2 * y + 1)*w_out*d_in + (2 * x + 0)*d_in);
					HVX_Vector *pDst11 = (HVX_Vector *)(out + b*h_out*w_out*d_in + (2 * y + 1)*w_out*d_in + (2 * x + 1)*d_in);
					for (int32_t d = 0; d < d_in; d += 128) {
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

void resizebilinear_4x(
	const uint8_t  *in,
	uint8_t        *out,
	int32_t         h_in,
	int32_t         w_in,
	int32_t         d_in,
	int32_t         h_out,
	int32_t         w_out,
	int32_t         h_in_end,
	struct bilin_runstate *rtsp
	) 
{
	HVX_Vector *pSrc0 = (HVX_Vector *)in;
	HVX_Vector *pSrc1 = (HVX_Vector *)(in + w_in * d_in);
	HVX_Vector *pDst0, *pDst1, *pDst2, *pDst3;
	int32_t wxd = w_in * d_in;
	int32_t lpwxd = (wxd+127) >> 7;
	int32_t wxdmod = wxd & 127; if (!wxdmod) wxdmod = 128;
	HVX_VectorPred Q1 = q6op_Q_vsetq2_R(wxd-d_in);
	wxd = (wxd * 4)&0x7f;

	for (int32_t h = 0; h < h_in; h++) {
		pDst0 = (HVX_Vector *)(out+(4*h+0)*w_out*d_in);
		pDst1 = (HVX_Vector *)(out+(4*h+1)*w_out*d_in);
		pDst2 = (HVX_Vector *)(out+(4*h+2)*w_out*d_in);
		pDst3 = (HVX_Vector *)(out+(4*h+3)*w_out*d_in);

		pSrc0 = (HVX_Vector *)(in + h*w_in * d_in);
		pSrc1 = (HVX_Vector *)(in + (h+1)*w_in*d_in);
		if (h == h_in_end - 1) 
			pSrc1 = pSrc0;
		HVX_Vector sL0cur = vmemu(pSrc0); pSrc0++; //TODO: unaligned load
		HVX_Vector sL1cur = vmemu(pSrc1); pSrc1++; //TODO: unaligned load
		HVX_Vector sL0nxt = Q6_V_vzero();
		HVX_Vector sL1nxt = Q6_V_vzero();
		for (int32_t z = 0; z < lpwxd - 1; z++) {
			sL0nxt = vmemu(pSrc0); pSrc0++; //TODO: unaligned load
			HVX_Vector sL0s1 = Q6_V_valign_VVR(sL0nxt, sL0cur, d_in);
			sL1nxt = vmemu(pSrc1); pSrc1++; //TODO: unaligned load
			HVX_Vector sL1s1 = Q6_V_valign_VVR(sL1nxt, sL1cur, d_in);

			// 0
			HVX_VectorPair dAC0 = Q6_Wuh_vmpy_VubRub(sL0cur, 0x04040404*1);
			HVX_Vector sAC0_L = Q6_V_lo_W(dAC0);
			HVX_Vector sAC0_H = Q6_V_hi_W(dAC0);

			HVX_VectorPair dBD0 = Q6_Wuh_vmpy_VubRub(sL0s1, 0x04040404*1);
			HVX_Vector sBD0_L = Q6_V_lo_W(dBD0);
			HVX_Vector sBD0_H = Q6_V_hi_W(dBD0);

			HVX_Vector sAC01_L = Q6_Vh_vmpyiacc_VhVhRb(sBD0_L, sAC0_L, 0x03030303);
			HVX_Vector sAC03_L = Q6_Vh_vmpyiacc_VhVhRb(sAC0_L, sBD0_L, 0x03030303);
			HVX_Vector sAC02_L = Q6_Vh_vadd_VhVh(sAC0_L, sAC0_L);
			HVX_Vector sAC02t_L = Q6_Vh_vadd_VhVh(sBD0_L, sBD0_L);
			sAC02_L = Q6_Vh_vadd_VhVh(sAC02_L, sAC02t_L);

			HVX_Vector sAC01_H = Q6_Vh_vmpyiacc_VhVhRb(sBD0_H, sAC0_H, 0x03030303);
			HVX_Vector sAC03_H = Q6_Vh_vmpyiacc_VhVhRb(sAC0_H, sBD0_H, 0x03030303);
			HVX_Vector sAC02_H = Q6_Vh_vadd_VhVh(sAC0_H, sAC0_H);
			HVX_Vector sAC02t_H = Q6_Vh_vadd_VhVh(sBD0_H, sBD0_H);
			sAC02_H = Q6_Vh_vadd_VhVh(sAC02_H, sAC02t_H);

			HVX_Vector sOut00 = sL0cur;
			HVX_Vector sOut01 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC01_H, sAC01_L, 4);
			HVX_Vector sOut02 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC02_H, sAC02_L, 4);
			HVX_Vector sOut03 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC03_H, sAC03_L, 4);

			HVX_VectorPair dOut020 = Q6_W_vshuff_VVR(sOut02, sOut00, -d_in);
			HVX_VectorPair dOut031 = Q6_W_vshuff_VVR(sOut03, sOut01, -d_in);
			HVX_VectorPair dOut0A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut031), Q6_V_lo_W(dOut020), -d_in);
			HVX_VectorPair dOut0B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut031), Q6_V_hi_W(dOut020), -d_in);
			vmemu(pDst0) = Q6_V_lo_W(dOut0A); pDst0++;
			vmemu(pDst0) = Q6_V_hi_W(dOut0A); pDst0++;
			vmemu(pDst0) = Q6_V_lo_W(dOut0B); pDst0++;
			vmemu(pDst0) = Q6_V_hi_W(dOut0B); pDst0++;

			// 1
			HVX_VectorPair dAC1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1cur, sL0cur), 0x01030103*1);
			HVX_VectorPair dBD1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1s1, sL0s1), 0x01030103*1);
			HVX_Vector sAC1_L = Q6_V_lo_W(dAC1);
			HVX_Vector sAC1_H = Q6_V_hi_W(dAC1);
			HVX_Vector sBD1_L = Q6_V_lo_W(dBD1);
			HVX_Vector sBD1_H = Q6_V_hi_W(dBD1);

			HVX_Vector sAC11_L = Q6_Vh_vmpyiacc_VhVhRb(sBD1_L, sAC1_L, 0x03030303);
			HVX_Vector sAC13_L = Q6_Vh_vmpyiacc_VhVhRb(sAC1_L, sBD1_L, 0x03030303);
			HVX_Vector sAC12_L = Q6_Vh_vadd_VhVh(sAC1_L, sAC1_L);
			HVX_Vector sAC12t_L = Q6_Vh_vadd_VhVh(sBD1_L, sBD1_L);
			sAC12_L = Q6_Vh_vadd_VhVh(sAC12_L, sAC12t_L);

			HVX_Vector sAC11_H = Q6_Vh_vmpyiacc_VhVhRb(sBD1_H, sAC1_H, 0x03030303);
			HVX_Vector sAC13_H = Q6_Vh_vmpyiacc_VhVhRb(sAC1_H, sBD1_H, 0x03030303);
			HVX_Vector sAC12_H = Q6_Vh_vadd_VhVh(sAC1_H, sAC1_H);
			HVX_Vector sAC12t_H = Q6_Vh_vadd_VhVh(sBD1_H, sBD1_H);
			sAC12_H = Q6_Vh_vadd_VhVh(sAC12_H, sAC12t_H);

			HVX_Vector sOut10 = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAC1), Q6_V_lo_W(dAC1), 2);
			HVX_Vector sOut11 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC11_H, sAC11_L, 4);
			HVX_Vector sOut12 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC12_H, sAC12_L, 4);
			HVX_Vector sOut13 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC13_H, sAC13_L, 4);

			HVX_VectorPair dOut120 = Q6_W_vshuff_VVR(sOut12, sOut10, -d_in);
			HVX_VectorPair dOut131 = Q6_W_vshuff_VVR(sOut13, sOut11, -d_in);
			HVX_VectorPair dOut1A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut131), Q6_V_lo_W(dOut120), -d_in);
			HVX_VectorPair dOut1B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut131), Q6_V_hi_W(dOut120), -d_in);
			vmemu(pDst1) = Q6_V_lo_W(dOut1A); pDst1++;
			vmemu(pDst1) = Q6_V_hi_W(dOut1A); pDst1++;
			vmemu(pDst1) = Q6_V_lo_W(dOut1B); pDst1++;
			vmemu(pDst1) = Q6_V_hi_W(dOut1B); pDst1++;

			// 2
			HVX_VectorPair dAC2 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1cur, sL0cur), 0x02020202*1);
			HVX_VectorPair dBD2 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1s1, sL0s1), 0x02020202*1);
			HVX_Vector sAC2_L = Q6_V_lo_W(dAC2);
			HVX_Vector sAC2_H = Q6_V_hi_W(dAC2);
			HVX_Vector sBD2_L = Q6_V_lo_W(dBD2);
			HVX_Vector sBD2_H = Q6_V_hi_W(dBD2);

			HVX_Vector sAC21_L = Q6_Vh_vmpyiacc_VhVhRb(sBD2_L, sAC2_L, 0x03030303);
			HVX_Vector sAC23_L = Q6_Vh_vmpyiacc_VhVhRb(sAC2_L, sBD2_L, 0x03030303);
			HVX_Vector sAC22_L = Q6_Vh_vadd_VhVh(sAC2_L, sAC2_L);
			HVX_Vector sAC22t_L = Q6_Vh_vadd_VhVh(sBD2_L, sBD2_L);
			sAC22_L = Q6_Vh_vadd_VhVh(sAC22_L, sAC22t_L);

			HVX_Vector sAC21_H = Q6_Vh_vmpyiacc_VhVhRb(sBD2_H, sAC2_H, 0x03030303);
			HVX_Vector sAC23_H = Q6_Vh_vmpyiacc_VhVhRb(sAC2_H, sBD2_H, 0x03030303);
			HVX_Vector sAC22_H = Q6_Vh_vadd_VhVh(sAC2_H, sAC2_H);
			HVX_Vector sAC22t_H = Q6_Vh_vadd_VhVh(sBD2_H, sBD2_H);
			sAC22_H = Q6_Vh_vadd_VhVh(sAC22_H, sAC22t_H);

			HVX_Vector sOut20 = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAC2), Q6_V_lo_W(dAC2), 2);
			HVX_Vector sOut21 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC21_H, sAC21_L, 4);
			HVX_Vector sOut22 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC22_H, sAC22_L, 4);
			HVX_Vector sOut23 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC23_H, sAC23_L, 4);

			HVX_VectorPair dOut220 = Q6_W_vshuff_VVR(sOut22, sOut20, -d_in);
			HVX_VectorPair dOut231 = Q6_W_vshuff_VVR(sOut23, sOut21, -d_in);
			HVX_VectorPair dOut2A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut231), Q6_V_lo_W(dOut220), -d_in);
			HVX_VectorPair dOut2B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut231), Q6_V_hi_W(dOut220), -d_in);
			vmemu(pDst2) = Q6_V_lo_W(dOut2A); pDst2++;
			vmemu(pDst2) = Q6_V_hi_W(dOut2A); pDst2++;
			vmemu(pDst2) = Q6_V_lo_W(dOut2B); pDst2++;
			vmemu(pDst2) = Q6_V_hi_W(dOut2B); pDst2++;

			// 3
			HVX_VectorPair dAC3 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1cur, sL0cur), 0x03010301*1);
			HVX_VectorPair dBD3 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1s1, sL0s1), 0x03010301*1);
			HVX_Vector sAC3_L = Q6_V_lo_W(dAC3);
			HVX_Vector sAC3_H = Q6_V_hi_W(dAC3);
			HVX_Vector sBD3_L = Q6_V_lo_W(dBD3);
			HVX_Vector sBD3_H = Q6_V_hi_W(dBD3);

			HVX_Vector sAC31_L = Q6_Vh_vmpyiacc_VhVhRb(sBD3_L, sAC3_L, 0x03030303);
			HVX_Vector sAC33_L = Q6_Vh_vmpyiacc_VhVhRb(sAC3_L, sBD3_L, 0x03030303);
			HVX_Vector sAC32_L = Q6_Vh_vadd_VhVh(sAC3_L, sAC3_L);
			HVX_Vector sAC32t_L = Q6_Vh_vadd_VhVh(sBD3_L, sBD3_L);
			sAC32_L = Q6_Vh_vadd_VhVh(sAC32_L, sAC32t_L);

			HVX_Vector sAC31_H = Q6_Vh_vmpyiacc_VhVhRb(sBD3_H, sAC3_H, 0x03030303);
			HVX_Vector sAC33_H = Q6_Vh_vmpyiacc_VhVhRb(sAC3_H, sBD3_H, 0x03030303);
			HVX_Vector sAC32_H = Q6_Vh_vadd_VhVh(sAC3_H, sAC3_H);
			HVX_Vector sAC32t_H = Q6_Vh_vadd_VhVh(sBD3_H, sBD3_H);
			sAC32_H = Q6_Vh_vadd_VhVh(sAC32_H, sAC32t_H);

			HVX_Vector sOut30 = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAC3), Q6_V_lo_W(dAC3), 2);
			HVX_Vector sOut31 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC31_H, sAC31_L, 4);
			HVX_Vector sOut32 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC32_H, sAC32_L, 4);
			HVX_Vector sOut33 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC33_H, sAC33_L, 4);

			HVX_VectorPair dOut320 = Q6_W_vshuff_VVR(sOut32, sOut30, -d_in);
			HVX_VectorPair dOut331 = Q6_W_vshuff_VVR(sOut33, sOut31, -d_in);
			HVX_VectorPair dOut3A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut331), Q6_V_lo_W(dOut320), -d_in);
			HVX_VectorPair dOut3B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut331), Q6_V_hi_W(dOut320), -d_in);
			vmemu(pDst3) = Q6_V_lo_W(dOut3A); pDst3++;
			vmemu(pDst3) = Q6_V_hi_W(dOut3A); pDst3++;
			vmemu(pDst3) = Q6_V_lo_W(dOut3B); pDst3++;
			vmemu(pDst3) = Q6_V_hi_W(dOut3B); pDst3++;

			sL0cur = sL0nxt;
			sL1cur = sL1nxt;
		}
		sL0nxt = vmemu(pSrc0); pSrc0++; //TODO: unaligned load
		HVX_Vector sL0s1 = Q6_V_valign_VVR(sL0nxt, sL0cur, d_in);
		sL1nxt = vmemu(pSrc1); pSrc1++; //TODO: unaligned load
		HVX_Vector sL1s1 = Q6_V_valign_VVR(sL1nxt, sL1cur, d_in);
		sL0s1 = Q6_V_vmux_QVV(Q1, sL0s1, sL0cur);
		sL1s1 = Q6_V_vmux_QVV(Q1, sL1s1, sL1cur);

		// 0
		HVX_VectorPair dAC0 = Q6_Wuh_vmpy_VubRub(sL0cur, 0x04040404 * 1);
		HVX_Vector sAC0_L = Q6_V_lo_W(dAC0);
		HVX_Vector sAC0_H = Q6_V_hi_W(dAC0);

		HVX_VectorPair dBD0 = Q6_Wuh_vmpy_VubRub(sL0s1, 0x04040404 * 1);
		HVX_Vector sBD0_L = Q6_V_lo_W(dBD0);
		HVX_Vector sBD0_H = Q6_V_hi_W(dBD0);

		HVX_Vector sAC01_L = Q6_Vh_vmpyiacc_VhVhRb(sBD0_L, sAC0_L, 0x03030303);
		HVX_Vector sAC03_L = Q6_Vh_vmpyiacc_VhVhRb(sAC0_L, sBD0_L, 0x03030303);
		HVX_Vector sAC02_L = Q6_Vh_vadd_VhVh(sAC0_L, sAC0_L);
		HVX_Vector sAC02t_L = Q6_Vh_vadd_VhVh(sBD0_L, sBD0_L);
		sAC02_L = Q6_Vh_vadd_VhVh(sAC02_L, sAC02t_L);

		HVX_Vector sAC01_H = Q6_Vh_vmpyiacc_VhVhRb(sBD0_H, sAC0_H, 0x03030303);
		HVX_Vector sAC03_H = Q6_Vh_vmpyiacc_VhVhRb(sAC0_H, sBD0_H, 0x03030303);
		HVX_Vector sAC02_H = Q6_Vh_vadd_VhVh(sAC0_H, sAC0_H);
		HVX_Vector sAC02t_H = Q6_Vh_vadd_VhVh(sBD0_H, sBD0_H);
		sAC02_H = Q6_Vh_vadd_VhVh(sAC02_H, sAC02t_H);

		HVX_Vector sOut00 = sL0cur;
		HVX_Vector sOut01 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC01_H, sAC01_L, 4);
		HVX_Vector sOut02 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC02_H, sAC02_L, 4);
		HVX_Vector sOut03 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC03_H, sAC03_L, 4);

		HVX_VectorPair dOut020 = Q6_W_vshuff_VVR(sOut02, sOut00, -d_in);
		HVX_VectorPair dOut031 = Q6_W_vshuff_VVR(sOut03, sOut01, -d_in);
		HVX_VectorPair dOut0A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut031), Q6_V_lo_W(dOut020), -d_in);
		HVX_VectorPair dOut0B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut031), Q6_V_hi_W(dOut020), -d_in);

		// 1
		HVX_VectorPair dAC1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1cur, sL0cur), 0x01030103 * 1);
		HVX_VectorPair dBD1 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1s1, sL0s1), 0x01030103 * 1);
		HVX_Vector sAC1_L = Q6_V_lo_W(dAC1);
		HVX_Vector sAC1_H = Q6_V_hi_W(dAC1);
		HVX_Vector sBD1_L = Q6_V_lo_W(dBD1);
		HVX_Vector sBD1_H = Q6_V_hi_W(dBD1);

		HVX_Vector sAC11_L = Q6_Vh_vmpyiacc_VhVhRb(sBD1_L, sAC1_L, 0x03030303);
		HVX_Vector sAC13_L = Q6_Vh_vmpyiacc_VhVhRb(sAC1_L, sBD1_L, 0x03030303);
		HVX_Vector sAC12_L = Q6_Vh_vadd_VhVh(sAC1_L, sAC1_L);
		HVX_Vector sAC12t_L = Q6_Vh_vadd_VhVh(sBD1_L, sBD1_L);
		sAC12_L = Q6_Vh_vadd_VhVh(sAC12_L, sAC12t_L);

		HVX_Vector sAC11_H = Q6_Vh_vmpyiacc_VhVhRb(sBD1_H, sAC1_H, 0x03030303);
		HVX_Vector sAC13_H = Q6_Vh_vmpyiacc_VhVhRb(sAC1_H, sBD1_H, 0x03030303);
		HVX_Vector sAC12_H = Q6_Vh_vadd_VhVh(sAC1_H, sAC1_H);
		HVX_Vector sAC12t_H = Q6_Vh_vadd_VhVh(sBD1_H, sBD1_H);
		sAC12_H = Q6_Vh_vadd_VhVh(sAC12_H, sAC12t_H);

		HVX_Vector sOut10 = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAC1), Q6_V_lo_W(dAC1), 2);
		HVX_Vector sOut11 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC11_H, sAC11_L, 4);
		HVX_Vector sOut12 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC12_H, sAC12_L, 4);
		HVX_Vector sOut13 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC13_H, sAC13_L, 4);

		HVX_VectorPair dOut120 = Q6_W_vshuff_VVR(sOut12, sOut10, -d_in);
		HVX_VectorPair dOut131 = Q6_W_vshuff_VVR(sOut13, sOut11, -d_in);
		HVX_VectorPair dOut1A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut131), Q6_V_lo_W(dOut120), -d_in);
		HVX_VectorPair dOut1B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut131), Q6_V_hi_W(dOut120), -d_in);

		// 2
		HVX_VectorPair dAC2 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1cur, sL0cur), 0x02020202 * 1);
		HVX_VectorPair dBD2 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1s1, sL0s1), 0x02020202 * 1);
		HVX_Vector sAC2_L = Q6_V_lo_W(dAC2);
		HVX_Vector sAC2_H = Q6_V_hi_W(dAC2);
		HVX_Vector sBD2_L = Q6_V_lo_W(dBD2);
		HVX_Vector sBD2_H = Q6_V_hi_W(dBD2);

		HVX_Vector sAC21_L = Q6_Vh_vmpyiacc_VhVhRb(sBD2_L, sAC2_L, 0x03030303);
		HVX_Vector sAC23_L = Q6_Vh_vmpyiacc_VhVhRb(sAC2_L, sBD2_L, 0x03030303);
		HVX_Vector sAC22_L = Q6_Vh_vadd_VhVh(sAC2_L, sAC2_L);
		HVX_Vector sAC22t_L = Q6_Vh_vadd_VhVh(sBD2_L, sBD2_L);
		sAC22_L = Q6_Vh_vadd_VhVh(sAC22_L, sAC22t_L);

		HVX_Vector sAC21_H = Q6_Vh_vmpyiacc_VhVhRb(sBD2_H, sAC2_H, 0x03030303);
		HVX_Vector sAC23_H = Q6_Vh_vmpyiacc_VhVhRb(sAC2_H, sBD2_H, 0x03030303);
		HVX_Vector sAC22_H = Q6_Vh_vadd_VhVh(sAC2_H, sAC2_H);
		HVX_Vector sAC22t_H = Q6_Vh_vadd_VhVh(sBD2_H, sBD2_H);
		sAC22_H = Q6_Vh_vadd_VhVh(sAC22_H, sAC22t_H);

		HVX_Vector sOut20 = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAC2), Q6_V_lo_W(dAC2), 2);
		HVX_Vector sOut21 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC21_H, sAC21_L, 4);
		HVX_Vector sOut22 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC22_H, sAC22_L, 4);
		HVX_Vector sOut23 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC23_H, sAC23_L, 4);

		HVX_VectorPair dOut220 = Q6_W_vshuff_VVR(sOut22, sOut20, -d_in);
		HVX_VectorPair dOut231 = Q6_W_vshuff_VVR(sOut23, sOut21, -d_in);
		HVX_VectorPair dOut2A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut231), Q6_V_lo_W(dOut220), -d_in);
		HVX_VectorPair dOut2B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut231), Q6_V_hi_W(dOut220), -d_in);

		// 3
		HVX_VectorPair dAC3 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1cur, sL0cur), 0x03010301 * 1);
		HVX_VectorPair dBD3 = Q6_Wh_vmpa_WubRb(Q6_W_vcombine_VV(sL1s1, sL0s1), 0x03010301 * 1);
		HVX_Vector sAC3_L = Q6_V_lo_W(dAC3);
		HVX_Vector sAC3_H = Q6_V_hi_W(dAC3);
		HVX_Vector sBD3_L = Q6_V_lo_W(dBD3);
		HVX_Vector sBD3_H = Q6_V_hi_W(dBD3);

		HVX_Vector sAC31_L = Q6_Vh_vmpyiacc_VhVhRb(sBD3_L, sAC3_L, 0x03030303);
		HVX_Vector sAC33_L = Q6_Vh_vmpyiacc_VhVhRb(sAC3_L, sBD3_L, 0x03030303);
		HVX_Vector sAC32_L = Q6_Vh_vadd_VhVh(sAC3_L, sAC3_L);
		HVX_Vector sAC32t_L = Q6_Vh_vadd_VhVh(sBD3_L, sBD3_L);
		sAC32_L = Q6_Vh_vadd_VhVh(sAC32_L, sAC32t_L);

		HVX_Vector sAC31_H = Q6_Vh_vmpyiacc_VhVhRb(sBD3_H, sAC3_H, 0x03030303);
		HVX_Vector sAC33_H = Q6_Vh_vmpyiacc_VhVhRb(sAC3_H, sBD3_H, 0x03030303);
		HVX_Vector sAC32_H = Q6_Vh_vadd_VhVh(sAC3_H, sAC3_H);
		HVX_Vector sAC32t_H = Q6_Vh_vadd_VhVh(sBD3_H, sBD3_H);
		sAC32_H = Q6_Vh_vadd_VhVh(sAC32_H, sAC32t_H);

		HVX_Vector sOut30 = Q6_Vub_vasr_VhVhR_rnd_sat(Q6_V_hi_W(dAC3), Q6_V_lo_W(dAC3), 2);
		HVX_Vector sOut31 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC31_H, sAC31_L, 4);
		HVX_Vector sOut32 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC32_H, sAC32_L, 4);
		HVX_Vector sOut33 = Q6_Vub_vasr_VhVhR_rnd_sat(sAC33_H, sAC33_L, 4);

		HVX_VectorPair dOut320 = Q6_W_vshuff_VVR(sOut32, sOut30, -d_in);
		HVX_VectorPair dOut331 = Q6_W_vshuff_VVR(sOut33, sOut31, -d_in);
		HVX_VectorPair dOut3A = Q6_W_vshuff_VVR(Q6_V_lo_W(dOut331), Q6_V_lo_W(dOut320), -d_in);
		HVX_VectorPair dOut3B = Q6_W_vshuff_VVR(Q6_V_hi_W(dOut331), Q6_V_hi_W(dOut320), -d_in);


		if (wxdmod <= 32) {
			q6op_vmemu_partial(pDst0++, Q6_V_lo_W(dOut0A), wxd);
			q6op_vmemu_partial(pDst1++, Q6_V_lo_W(dOut1A), wxd);
			q6op_vmemu_partial(pDst2++, Q6_V_lo_W(dOut2A), wxd);
			q6op_vmemu_partial(pDst3++, Q6_V_lo_W(dOut3A), wxd);
		}
		else if (wxdmod <= 64) {
			vmemu(pDst0) = Q6_V_lo_W(dOut0A); pDst0++;
			vmemu(pDst1) = Q6_V_lo_W(dOut1A); pDst1++;
			vmemu(pDst2) = Q6_V_lo_W(dOut2A); pDst2++;
			vmemu(pDst3) = Q6_V_lo_W(dOut3A); pDst3++;
			q6op_vmemu_partial(pDst0++, Q6_V_hi_W(dOut0A), wxd);
			q6op_vmemu_partial(pDst1++, Q6_V_hi_W(dOut1A), wxd);
			q6op_vmemu_partial(pDst2++, Q6_V_hi_W(dOut2A), wxd);
			q6op_vmemu_partial(pDst3++, Q6_V_hi_W(dOut3A), wxd);
		}
		else if (wxdmod <= 96) {
			vmemu(pDst0) = Q6_V_lo_W(dOut0A); pDst0++;
			vmemu(pDst0) = Q6_V_hi_W(dOut0A); pDst0++;
			vmemu(pDst1) = Q6_V_lo_W(dOut1A); pDst1++;
			vmemu(pDst1) = Q6_V_hi_W(dOut1A); pDst1++;
			vmemu(pDst2) = Q6_V_lo_W(dOut2A); pDst2++;
			vmemu(pDst2) = Q6_V_hi_W(dOut2A); pDst2++;
			vmemu(pDst3) = Q6_V_lo_W(dOut3A); pDst3++;
			vmemu(pDst3) = Q6_V_hi_W(dOut3A); pDst3++;
			q6op_vmemu_partial(pDst0++, Q6_V_lo_W(dOut0B), wxd);
			q6op_vmemu_partial(pDst1++, Q6_V_lo_W(dOut1B), wxd);
			q6op_vmemu_partial(pDst2++, Q6_V_lo_W(dOut2B), wxd);
			q6op_vmemu_partial(pDst3++, Q6_V_lo_W(dOut3B), wxd);
		}
		else {
			vmemu(pDst0) = Q6_V_lo_W(dOut0A); pDst0++;
			vmemu(pDst0) = Q6_V_hi_W(dOut0A); pDst0++;
			vmemu(pDst0) = Q6_V_lo_W(dOut0B); pDst0++;
			vmemu(pDst1) = Q6_V_lo_W(dOut1A); pDst1++;
			vmemu(pDst1) = Q6_V_hi_W(dOut1A); pDst1++;
			vmemu(pDst1) = Q6_V_lo_W(dOut1B); pDst1++;
			vmemu(pDst2) = Q6_V_lo_W(dOut2A); pDst2++;
			vmemu(pDst2) = Q6_V_hi_W(dOut2A); pDst2++;
			vmemu(pDst2) = Q6_V_lo_W(dOut2B); pDst2++;
			vmemu(pDst3) = Q6_V_lo_W(dOut3A); pDst3++;
			vmemu(pDst3) = Q6_V_hi_W(dOut3A); pDst3++;
			vmemu(pDst3) = Q6_V_lo_W(dOut3B); pDst3++;
			q6op_vmemu_partial(pDst0++, Q6_V_hi_W(dOut0B), wxd);
			q6op_vmemu_partial(pDst1++, Q6_V_hi_W(dOut1B), wxd);
			q6op_vmemu_partial(pDst2++, Q6_V_hi_W(dOut2B), wxd);
			q6op_vmemu_partial(pDst3++, Q6_V_hi_W(dOut3B), wxd);
		}
	}
}

//==================================================================================
void resizebilinear_hvx(
	const uint8_t  *in,
	uint8_t        *out,
	int32_t         h_in,
	int32_t         w_in,
	int32_t         d_in,
	int32_t         h_out,
	int32_t         w_out,
	int32_t         tid,
	struct bilin_runstate *rtsp
)
{
	int bufind;
	uint8_t *row0 = bufpool_take(&rtsp->intermed_bufs, &bufind);
	int32_t h_stride = 512*4*2;
	uint8_t *row1 = row0 + h_stride;

	int hlim = h_in - 1;
	int32_t hblock = (h_out + rtsp->inner_count / 2) / rtsp->inner_count;
	int32_t hend = min_i32((tid + 1)*hblock, h_out);
	int64_t vacc = (tid*hblock*rtsp->planp->vscale)+(1<<16);	// rounding bias for 15-bit fraction
	int iypos0prev = -1;
	HVX_Vector sOut = Q6_V_vzero();
	HVX_Vector *pOut = NULL;
	for (int32_t h = tid*hblock; h < hend; h++) {
		int iypos0 = vacc >> 32;
		int iypos1 = iypos0 + 1;
		int yfrac = (uint32_t)(vacc+ (1 << 16)) >> 17;	// 15-bit frac
		vacc += rtsp->planp->vscale;
		if (iypos0 >= hlim) {		// don't exceed hlim
			iypos0 = iypos1 = hlim;
			yfrac = 0;
		}

		if (iypos0 != iypos0prev) {
			int32_t wover3 = (w_in - 1) / 3;
			for (int32_t xiter = d_in, x = 0; x < 3; xiter += wover3 * d_in * 4 + d_in, x++) {
				HVX_Vector sX0 = vmemu(&in[iypos0*w_in*d_in + (wover3*x + 0)*d_in]);
				HVX_VectorPair dY0 = Q6_W_vshuff_VVR(sX0, sX0, -d_in);
				HVX_VectorPair dTL0 = Q6_W_vshuff_VVR(Q6_V_lo_W(dY0), Q6_V_lo_W(dY0), -d_in * 2);
				HVX_VectorPair dTL1 = Q6_W_vshuff_VVR(Q6_V_hi_W(dY0), Q6_V_hi_W(dY0), -d_in * 2);
				vmemu(&row0[xiter + 0 * 128]) = Q6_V_lo_W(dTL0);
				vmemu(&row0[xiter + 1 * 128]) = Q6_V_hi_W(dTL0);
				vmemu(&row0[xiter + 2 * 128]) = Q6_V_lo_W(dTL1);
				vmemu(&row0[xiter + 3 * 128]) = Q6_V_hi_W(dTL1);

				HVX_Vector sX1 = vmemu(&in[iypos0*w_in*d_in + (wover3*x + 1)*d_in]);
				HVX_VectorPair dY1 = Q6_W_vshuff_VVR(sX1, sX1, -d_in);
				HVX_VectorPair dTR0 = Q6_W_vshuff_VVR(Q6_V_lo_W(dY1), Q6_V_lo_W(dY1), -d_in * 2);
				HVX_VectorPair dTR1 = Q6_W_vshuff_VVR(Q6_V_hi_W(dY1), Q6_V_hi_W(dY1), -d_in * 2);
				vmemu(&row0[h_stride / 2 + xiter + 0 * 128]) = Q6_V_lo_W(dTR0);
				vmemu(&row0[h_stride / 2 + xiter + 1 * 128]) = Q6_V_hi_W(dTR0);
				vmemu(&row0[h_stride / 2 + xiter + 2 * 128]) = Q6_V_lo_W(dTR1);
				vmemu(&row0[h_stride / 2 + xiter + 3 * 128]) = Q6_V_hi_W(dTR1);

				HVX_Vector sX2 = vmemu(&in[iypos1*w_in*d_in + (wover3*x + 0)*d_in]);
				HVX_VectorPair dY2 = Q6_W_vshuff_VVR(sX2, sX2, -d_in);
				HVX_VectorPair dBL0 = Q6_W_vshuff_VVR(Q6_V_lo_W(dY2), Q6_V_lo_W(dY2), -d_in * 2);
				HVX_VectorPair dBL1 = Q6_W_vshuff_VVR(Q6_V_hi_W(dY2), Q6_V_hi_W(dY2), -d_in * 2);
				vmemu(&row1[xiter + 0 * 128]) = Q6_V_lo_W(dBL0);
				vmemu(&row1[xiter + 1 * 128]) = Q6_V_hi_W(dBL0);
				vmemu(&row1[xiter + 2 * 128]) = Q6_V_lo_W(dBL1);
				vmemu(&row1[xiter + 3 * 128]) = Q6_V_hi_W(dBL1);

				HVX_Vector sX3 = vmemu(&in[iypos1*w_in*d_in + (wover3*x + 1)*d_in]);
				HVX_VectorPair dY3 = Q6_W_vshuff_VVR(sX3, sX3, -d_in);
				HVX_VectorPair dBR0 = Q6_W_vshuff_VVR(Q6_V_lo_W(dY3), Q6_V_lo_W(dY3), -d_in * 2);
				HVX_VectorPair dBR1 = Q6_W_vshuff_VVR(Q6_V_hi_W(dY3), Q6_V_hi_W(dY3), -d_in * 2);
				vmemu(&row1[h_stride / 2 + xiter + 0 * 128]) = Q6_V_lo_W(dBR0);
				vmemu(&row1[h_stride / 2 + xiter + 1 * 128]) = Q6_V_hi_W(dBR0);
				vmemu(&row1[h_stride / 2 + xiter + 2 * 128]) = Q6_V_lo_W(dBR1);
				vmemu(&row1[h_stride / 2 + xiter + 3 * 128]) = Q6_V_hi_W(dBR1);
			}
			int32_t wover3x4 = wover3 * 4;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < d_in; j++) {
					row0[(wover3x4*i + i)*d_in + j] = row0[(wover3x4*i + i + 1)*d_in + j];
					row0[h_stride / 2 + (wover3x4*i + i)*d_in + j] = row0[h_stride / 2 + (wover3x4*i + i + 1)*d_in + j];
					row1[(wover3x4*i + i)*d_in + j] = row1[(wover3x4*i + i + 1)*d_in + j];
					row1[h_stride / 2 + (wover3x4*i + i)*d_in + j] = row1[h_stride / 2 + (wover3x4*i + i + 1)*d_in + j];
				}
			}
		}
		iypos0prev = iypos0;

		uint8_t *ptl = row0;
		uint8_t *pbl = row1;
		uint8_t *ptr = row0 + h_stride/2;
		uint8_t *pbr = row1 + h_stride/2;
		HVX_Vector *pTL = (HVX_Vector *)ptl;
		HVX_Vector *pBL = (HVX_Vector *)pbl;
		HVX_Vector *pTR = (HVX_Vector *)ptr;
		HVX_Vector *pBR = (HVX_Vector *)pbr;
		HVX_Vector *pFrac = (HVX_Vector *)&rtsp->planp->xfrac[0];
		pOut = (HVX_Vector *)&out[h*w_out*d_in];
		int yfrac1 = (yfrac << 16) | yfrac;	// 15-bit frac
		for (int32_t x = 0; x < w_out*d_in; x+=128) {
			HVX_Vector sTL = *pTL++;
			HVX_Vector sBL = *pBL++;
			HVX_Vector sTR = *pTR++;
			HVX_Vector sBR = *pBR++;
			HVX_Vector sFrac_L = *pFrac++;
			HVX_Vector sFrac_H = *pFrac++;
			HVX_VectorPair dFrac = Q6_W_vdeal_VVR(sFrac_H, sFrac_L, -1);

			HVX_VectorPair dTi = Q6_Wh_vmpa_WubWb(Q6_W_vcombine_VV(sTR, sTL), dFrac);
			HVX_VectorPair dTL = Q6_Wuh_vzxt_Vub(sTL);
			HVX_VectorPair dTop = Q6_Wuh_vadd_WuhWuh_sat(dTL, dTi);

			HVX_VectorPair dBi = Q6_Wh_vmpa_WubWb(Q6_W_vcombine_VV(sBR, sBL), dFrac);
			HVX_VectorPair dBL = Q6_Wuh_vzxt_Vub(sBL);
			HVX_VectorPair dBot = Q6_Wuh_vadd_WuhWuh_sat(dBL, dBi);
			HVX_VectorPair dDel = Q6_Wh_vsub_WhWh(dBot, dTop);
			HVX_Vector sInt_L = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_lo_W(dDel), yfrac1);
			HVX_Vector sOut_L = Q6_Vh_vadd_VhVh(sInt_L, Q6_V_lo_W(dTop));
			HVX_Vector sInt_H = Q6_Vh_vmpy_VhRh_s1_rnd_sat(Q6_V_hi_W(dDel), yfrac1);
			HVX_Vector sOut_H = Q6_Vh_vadd_VhVh(sInt_H, Q6_V_hi_W(dTop));
			sOut = Q6_Vub_vasr_VhVhR_rnd_sat(sOut_H, sOut_L, 7);
			if (x + 128 < w_out*d_in || h != hend - 1) {
				vmemu(pOut) = sOut;
				pOut++;
			}
		}
	}
	int lastchunk = (w_out * d_in) & 127;
	if (lastchunk == 0) lastchunk = 128;
	memcpy(pOut, &sOut, lastchunk);
	bufpool_release(&rtsp->intermed_bufs, bufind);
}

//==================================================================================
void resizebilinear_general(
	const uint8_t  *in,
	uint8_t        *out,
	int32_t         h_in,
	int32_t         w_in,
	int32_t         d_in,
	int32_t         h_out,
	int32_t         w_out,
	int32_t         h_in_end,
	struct bilin_runstate *rtsp
)
{
	int b_in = 1;
	float xscale, yscale;
	if (!rtsp->align_corners) {
		xscale = (float)w_in / w_out;
		yscale = (float)h_in / h_out;
	}
	else {
		xscale = (float)(w_in-1) / (w_out-1);
		yscale = (float)(h_in-1) / (h_out-1);
	}
	for (int32_t b = 0; b < b_in; b++) {
		const uint8_t *bstart = in + b*h_in*w_in*d_in;
		for (int32_t h = 0; h < h_out; h++) {
			float yfloat = h*yscale;
			float yfrac = yfloat - floorf(yfloat);
			float yint = yfloat - yfrac;
			for (int32_t w = 0; w < w_out; w++) {
				float xfloat = w*xscale;
				float xfrac = xfloat - floorf(xfloat);
				float xint = xfloat - xfrac;
				for (int32_t d = 0; d < d_in; d++) {
					uint8_t f00 = bstart[(int32_t)(min_i32(h_in_end - 1, (yint + 0))*w_in*d_in + min_i32(w_in - 1, (xint + 0))*d_in) + d];
					uint8_t f01 = bstart[(int32_t)(min_i32(h_in_end - 1, (yint + 0))*w_in*d_in + min_i32(w_in - 1, (xint + 1))*d_in) + d];
					uint8_t f10 = bstart[(int32_t)(min_i32(h_in_end - 1, (yint + 1))*w_in*d_in + min_i32(w_in - 1, (xint + 0))*d_in) + d];
					uint8_t f11 = bstart[(int32_t)(min_i32(h_in_end - 1, (yint + 1))*w_in*d_in + min_i32(w_in - 1, (xint + 1))*d_in) + d];
					float outfloat = bilinear_interpolate(f00, f01, f10, f11, xfrac, yfrac);
					out[d] = outfloat + 0.5f;
				}
				out += d_in;
			}
		}
	}
}

//==================================================================================
static void resizebilinear_work(struct nn_graph *nn, void *vinfo)
{
	struct bilin_runstate *rstp = (struct bilin_runstate *)vinfo;
	int32_t job_idx, hid, batch_idx, blockrow;

	const uint8_t *pin = rstp->tin;
	uint8_t *pout = rstp->tout;
	int32_t in_bstride = rstp->in_bstride;
	int32_t in_hstride = rstp->in_hstride;
	int32_t in_blockhstride = in_hstride * rstp->blockrow;
	int32_t out_bstride = rstp->out_bstride;
	int32_t out_blockhstride = rstp->out_hstride*rstp->blockrow*rstp->factor;

	batchslice_decode bsdecode;
	batchslice_decode_init(&bsdecode, rstp->inner_count);

	if (rstp->resizebilinear_ptr != resizebilinear_hvx) {
		while (job_idx = __sync_fetch_and_add(&rstp->next_job, 1), job_idx < rstp->jobs) {
			hid = batchslice_decode_update(&bsdecode, job_idx);
			batch_idx = bsdecode.ibatch;

			const uint8_t *cur_pin = pin + batch_idx * in_bstride + hid * in_blockhstride;
			blockrow = min_i32(rstp->in_ht - hid * rstp->blockrow, rstp->blockrow);
			l2fetch(cur_pin, in_hstride, in_hstride, blockrow);

			rstp->resizebilinear_ptr(
				cur_pin,
				pout + batch_idx * out_bstride + hid * out_blockhstride,
				blockrow,
				rstp->in_wh,
				rstp->depth,
				rstp->out_ht,
				rstp->out_wh,
				rstp->in_ht - hid * rstp->blockrow,
				rstp);
		}
	}
	else {
		while (job_idx = __sync_fetch_and_add(&rstp->next_job, 1), job_idx < rstp->jobs) {
			hid = batchslice_decode_update(&bsdecode, job_idx);
			batch_idx = bsdecode.ibatch;

			const uint8_t *cur_pin = pin + batch_idx * in_bstride + hid * in_blockhstride;
			blockrow = min_i32(rstp->in_ht - hid * rstp->blockrow, rstp->blockrow);
			l2fetch(cur_pin, in_hstride, in_hstride, blockrow);

			rstp->resizebilinear_ptr(
				pin + batch_idx * in_bstride,
				pout + batch_idx * out_bstride,
				rstp->in_ht,
				rstp->in_wh,
				rstp->depth,
				rstp->out_ht,
				rstp->out_wh,
				hid,
				rstp);
		}
	}
	nn_sem_post(&rstp->done_sem);
}

//==================================================================================
static int resizebilinear_f_execute(struct nn_node *self, struct nn_graph *nn)
{
	int32_t elementsize = sizeof(float);
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *newdim_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	const int32_t *newdims = newdim_tensor->data;
	const int32_t h_out = newdims[0];
	const int32_t w_out = newdims[1];
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t d_in = in_tensor->shape.depth;
	float xscale = (float)w_in / w_out;
	float yscale = (float)h_in / h_out;
	uint32_t close_h;
	uint32_t close_w;
	int32_t b, h, w, d;
	float *out = (float *)out_tensor->data;
	const float *in = (const float *)in_tensor->data;
	const float *bstart;
	const float *hstart;
	const float *wstart;
	uint32_t depth_bytes = d_in * elementsize;
	uint32_t total_bytes = b_in * h_out*w_out*depth_bytes;

	int32_t align_corners = 0;
	if (self->n_inputs == 3)
		align_corners = *(int32_t *)(self->inputs[2]->data);
	if (align_corners) {
		if (w_out <= 1 || h_out <= 1) return errlog(nn, "aligned_corners flag is no good with out width/height of 1 or less");
		xscale = (float)(w_in-1) / (w_out-1);
		yscale = (float)(h_in-1) / (h_out-1);
	}

	if (total_bytes > out_tensor->max_size) return errlog(nn, "out too small");
	logmsg(nn, 2, "%dx%dx%dx%d --> %dx%dx%dx%d", b_in, h_in, w_in, d_in, b_in, h_out, w_out, d_in);
	tensor_set_shape(out_tensor, b_in, h_out, w_out, d_in);
	out_tensor->data_size = total_bytes;

	for (b = 0; b < b_in; b++) {
		bstart = in + b * h_in*w_in*d_in;
		for (h = 0; h < h_out; h++) {
			float yfloat = h * yscale;
			float yfrac = yfloat - floorf(yfloat);
			float yint = yfloat - yfrac;
			close_h = h * yscale;
			hstart = bstart + close_h * w_in*d_in;
			for (w = 0; w < w_out; w++) {
				float xfloat = w * xscale;
				float xfrac = xfloat - floorf(xfloat);
				float xint = xfloat - xfrac;
				close_w = w * xscale;
				wstart = hstart + close_w * d_in;
				for (d = 0; d < d_in; d++) {
					float f00 = bstart[(int32_t)(min_i32(h_in - 1, (yint + 0))*w_in*d_in + min_i32(w_in - 1, (xint + 0))*d_in) + d];
					float f01 = bstart[(int32_t)(min_i32(h_in - 1, (yint + 0))*w_in*d_in + min_i32(w_in - 1, (xint + 1))*d_in) + d];
					float f10 = bstart[(int32_t)(min_i32(h_in - 1, (yint + 1))*w_in*d_in + min_i32(w_in - 1, (xint + 0))*d_in) + d];
					float f11 = bstart[(int32_t)(min_i32(h_in - 1, (yint + 1))*w_in*d_in + min_i32(w_in - 1, (xint + 1))*d_in) + d];
					out[d] = bilinear_interpolate(f00, f01, f10, f11, xfrac, yfrac);
				}
				out += d_in;
			}
		}
	}
	return 0;
}

//==================================================================================
static int
weight_generation(
	struct nn_node *self,
	struct bilin_runstate  *rstp
) {
	int ht_in = rstp->in_ht;
	int wid_in = rstp->in_wh;
	int ht_out = rstp->out_ht;
	int wid_out = rstp->out_wh;
	int depth = rstp->depth;
	int wxd_align = (wid_out*depth + 127) & -128;
	if (ht_in < 1 || wid_in < 1 || ht_out < 1 || wid_out < 1) return -1;
	if (depth < 1 || depth > 2) return -1;

	struct resizebilinear_plan * planp = (struct resizebilinear_plan *)self->opaque;
	if (planp != NULL) {
		planp = (struct resizebilinear_plan*)(((size_t)planp + 0x7f)&-128);
		if (ht_in == planp->in_height && ht_out == planp->out_height
			&&	wid_in == planp->in_width && wid_out == planp->out_width) {
			rstp->planp = planp;	// reuse exising
			return 0;
		}
		nn_free((void*)planp);
		planp = NULL;
		self->opaque = NULL;
	}
	if (planp == NULL) {
		planp = (struct resizebilinear_plan *) nn_malloc(
			sizeof(struct resizebilinear_plan) + wxd_align *(sizeof(planp->xoff[0])*2+sizeof(planp->xfrac[0])) + (wxd_align >>7)*sizeof(planp->xload[0]) + 0x80);
		if (planp == NULL) return -1;
		self->opaque = (void*)planp;
		planp = (struct resizebilinear_plan*)(((size_t)planp + 0x7f)&-128);
	}
	rstp->planp = planp;
	// fill in the table
	planp->in_height = ht_in;
	planp->in_width = wid_in;
	planp->out_height = ht_out;
	planp->out_width = wid_out;
	planp->xoff = (uint8_t*)(&planp->xfrac[0] + wxd_align);
	planp->xload = (struct edgeidx*)(planp->xoff + wxd_align *2);

	if (rstp->align_corners && ht_out > 1) {
		ht_in--;
		ht_out--;
	}
	planp->vscale = (((uint64_t)ht_in << 32) + (ht_out >> 1)) / (uint64_t)ht_out;

	// scale for h table.
	int win_max = wid_in - 1;
	int wid_out_pi = wid_out;
	if (rstp->align_corners && wid_out > 1) {
		wid_in--;
		wid_out_pi--;
	}
	int64_t add_per = (((uint64_t)wid_in << 32) + (wid_out_pi >> 1)) / (uint64_t)wid_out_pi;

	int64_t acc = (1 << 16);	// we take a 15-bit fraction, this is the rounding bias
	int32_t i, previdx = -1;
	for (i = 0; i < wid_out*depth; i+=depth) {
		int intpart = (acc >> 32);		// integer part
		unsigned fpart = (uint32_t)acc >> 25;	// fractional, 0..127
		if (intpart >= win_max) {		// don't go beyond wid_in-1;
			intpart = win_max;
			fpart = 0;
		}
		planp->xoff[2*i+0] = intpart*depth;
		planp->xoff[2*i+1] = min_i32((intpart+1)*depth, win_max*depth + depth-1);
		planp->xfrac[i] = (fpart << 8) + (127-fpart);
		if ((i & 127) == 128 - depth || i == wid_out*depth - depth) {
			int curridx = ((intpart + 1)*depth + depth - 1) >> 7;
			if (previdx < curridx) {
				planp->xload[i >> 7].start = ((previdx < curridx) << 1) + (curridx & 1);
			}
			else {
				planp->xload[i >> 7].start = 0;
			}
			previdx = curridx;
			if (curridx > previdx + 1) {
				return -1;
			}
		}
		if (depth == 2) {
			planp->xoff[2*(i+1) + 0] = intpart*depth + 1;
			planp->xoff[2*(i+1) + 1] = min_i32((intpart+1)*depth + 1, win_max*depth + depth-1);
			planp->xfrac[i+1] = (fpart << 8) + (127 - fpart);
		}
		acc += add_per;
	}

	return 0;
}

//==================================================================================
static int resizebilinear_qu8_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *newdim_tensor = self->inputs[1];
	struct tensor *out_tensor = self->outputs[0];
	const int32_t *newdims = newdim_tensor->data;
	const int32_t h_out = newdims[0];
	const int32_t w_out = newdims[1];
	const int32_t b_in = in_tensor->shape.batches;
	const int32_t h_in = in_tensor->shape.height;
	const int32_t w_in = in_tensor->shape.width;
	const int32_t d_in = in_tensor->shape.depth;
	uint32_t total_bytes = b_in*h_out*w_out*d_in*sizeof(uint8_t);

#ifdef TEST_PERFORMANCE
	uint64_t start_time, end_time;
	start_time = nn_os_get_cycles(nn);
#endif

	if (total_bytes > out_tensor->max_size) return errlog(nn, "out too small");
	logmsg(nn, 2, "%dx%dx%dx%d --> %dx%dx%dx%d", b_in, h_in, w_in, d_in, b_in, h_out, w_out, d_in);

	struct bilin_runstate runstate;
	runstate.align_corners = 0;
	if (self->n_inputs == 5)
		runstate.align_corners = *(int32_t *)(self->inputs[4]->data) != 0;

	if (!runstate.align_corners && w_out == 2*w_in && h_out == 2*h_in &&
		((d_in&(d_in-1)) == 0) &&
		((d_in >=128) || ((w_in&(w_in-1)) == 0 ))) //if d<128, w must be a power of 2
	{
		runstate.inner_count = min_i32(h_in, NUM_THREADS); // partioned on height
		runstate.factor = 2;
		runstate.resizebilinear_ptr = resizebilinear_2x;
	}
	else if (!runstate.align_corners && w_out == 4*w_in && h_out == 4*h_in && d_in < 32 && ((d_in & (d_in-1)) == 0)) {
		runstate.inner_count = min_i32(h_in, NUM_THREADS); // partioned on height
		runstate.factor = 4;
		runstate.resizebilinear_ptr = resizebilinear_4x;
	}
	else if (w_out == w_in && h_out == h_in) {
		struct nn_memcpy_manager  mcman;
		nn_mcmanager_init(nn, &mcman);
		nn_mcmanager_vmemcpy(nn, &mcman, out_tensor->data, in_tensor->data, in_tensor->data_size);
		tensor_set_shape(out_tensor, b_in, h_out, w_out, d_in);
		out_tensor->data_size = total_bytes;
		tensor_copy(self->outputs[1], self->inputs[2]);
		tensor_copy(self->outputs[2], self->inputs[3]);
		nn_mcmanager_wait(nn, &mcman);
		return 0;
	}
	else if (runstate.align_corners && w_in * 4 == w_out && (w_in - 1) % 3 == 0 && w_in*d_in < 128 * 3) {
		runstate.inner_count = min_i32(h_in, NUM_THREADS);
		runstate.factor = -2;
		runstate.resizebilinear_ptr = resizebilinear_hvx;
	}
	else {
		runstate.inner_count = 1; //min_i32(h_in, NUM_THREADS); // partioned on height?
		runstate.factor = -1;
		runstate.resizebilinear_ptr = resizebilinear_general;
	}
	runstate.tin = (const uint8_t *)in_tensor->data;
	runstate.tout = (uint8_t *)out_tensor->data;
	runstate.batches = b_in;
	runstate.depth = d_in;
	runstate.in_ht = h_in;
	runstate.in_wh = w_in;
	runstate.out_ht = h_out;
	runstate.out_wh = w_out;
	runstate.blockrow = (h_in + runstate.inner_count - 1) / runstate.inner_count;
	runstate.in_hstride = w_in * d_in;
	runstate.in_bstride = h_in * runstate.in_hstride;
	runstate.out_hstride = w_out * d_in;
	runstate.out_bstride = h_out * runstate.out_hstride;
	runstate.jobs = b_in * runstate.inner_count;
	runstate.next_job = 0;
	nn_sem_init(&runstate.done_sem, 0);
	nn_scratch_reset(nn);

	int32_t n_threads = min_i32(NUM_THREADS, runstate.jobs);
	if (runstate.resizebilinear_ptr == resizebilinear_hvx) {
		if (weight_generation(self, &runstate) != 0) {
			return errlog(nn, "didn't get scaling plan");
		}
		unsigned intermed_size = max_i32(8192, (w_out*d_in*2 + 127) & -128);
		void * mem = nn_scratch_alloc(nn, intermed_size * n_threads);
		if (mem == 0) return errlog(nn, "didn't get temp mem");
		bufpool_init(&runstate.intermed_bufs, n_threads, mem, intermed_size);
	}

	for (int32_t i = 0; i < n_threads; i++)
		nn_os_work_for_vector(nn, resizebilinear_work, &runstate);

	tensor_set_shape(out_tensor, b_in, h_out, w_out, d_in);
	out_tensor->data_size = total_bytes;
	tensor_copy(self->outputs[1], self->inputs[2]);
	tensor_copy(self->outputs[2], self->inputs[3]);

	nn_sem_wait_n_times(&runstate.done_sem, n_threads);

#ifdef TEST_PERFORMANCE
	end_time = nn_os_get_cycles(nn);
	printf("resizebilinear_nond32  cycles = %lld (elements = %ld) w=(%ld->%ld) h=(%ld->%ld)\n",
		(end_time - start_time), tensor_element_count(out_tensor), w_in, w_out, h_in, h_out);
#endif

	return 0;
}


//==================================================================================
struct nn_node_ops nn_ops_for_ResizeBilinear_f = {
	.execute = resizebilinear_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT_RANGE(2,3),
	.n_outputs = NN_IOCOUNT(1),
};

struct nn_node_ops nn_ops_for_QuantizedResizeBilinear_8 = {
	.execute = resizebilinear_qu8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT_RANGE(4,5),
	.n_outputs = NN_IOCOUNT(3),
};
