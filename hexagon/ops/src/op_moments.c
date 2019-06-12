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

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include "hvx_inlines.h"

#define NUM_DIMS 4
#define NUM_THREADS 2

#if defined(__hexagon__)
#include "hexagon_types.h"
#include "hvx_inlines.h"
typedef long HVX_Vect_UN __attribute__((__vector_size__(128)))__attribute__((aligned(4)));
#define vmemu(A) *((HVX_Vect_UN*)(A))
#endif
//==================================================================================
struct moments_runstate {
	const uint8_t *pin;
	uint8_t *pmout;
	int32_t *pvout;
	int32_t input_offset;
	int32_t rdims[5];

	nn_sem_t done_sem;
	int32_t jobs;		// total # of 'subsections' to run
	volatile int32_t next_job;	 // index of next subsection
};

//==================================================================================
int check_buffer(
	const struct tensor *in_tensor,
	const struct tensor *axes_tensor,
	const struct tensor *out_m_tensor,
	const struct tensor *out_v_tensor
)
{
	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t* axes = (int32_t*)axes_tensor->data;
	int32_t axes_size = axes_tensor->data_size / sizeof(int32_t);

	int32_t modified_shape_final[NUM_DIMS] = { in_batches, in_height, in_width, in_depth };
	int rflag = 0;
	for (int i = 0; i < axes_size; i++) {
		if (axes[i] < 0 || axes[i] >= NUM_DIMS)
			return -1;
		modified_shape_final[axes[i]] = 1; // if our axis is being reduced on, it's size is 1
		rflag |= 1 << axes[i];
	}

	int32_t modified_data_size = 1;
	for (int i = 0; i < NUM_DIMS; i++){
		modified_data_size *= modified_shape_final[i]; // set the final size
	}

	int32_t mtensor_size = out_m_tensor->shape.batches * out_m_tensor->shape.height * out_m_tensor->shape.width * out_m_tensor->shape.depth;
	int32_t vtensor_size = out_v_tensor->shape.batches * out_v_tensor->shape.height * out_v_tensor->shape.width * out_v_tensor->shape.depth;

	if (modified_data_size != mtensor_size ||
		modified_data_size !=  vtensor_size){
		return -1;
	}

	return rflag;
}

void rflag2rdim(int32_t rflag, int32_t *rdims, const struct shape *shp) {
	rdims[0] = rdims[1] = rdims[2] = rdims[3] = rdims[4] = 1;
	switch (rflag) {
		// b h w d
		// 1 2 4 8
	case 0:
		// no reduction
		break;
	case 1:
		rdims[4] = shp->depth*shp->width*shp->height;
		rdims[3] = shp->batches;
		break;
	case 2:
		rdims[4] = shp->depth*shp->width;
		rdims[3] = shp->height;
		rdims[2] = shp->batches;
		break;
	case 3:
		rdims[4] = shp->depth*shp->width;
		rdims[3] = shp->height*shp->batches;
		break;
	case 4:
		rdims[4] = shp->depth;
		rdims[3] = shp->width;
		rdims[2] = shp->height*shp->batches;
		break;
	case 5:
		rdims[4] = shp->depth;
		rdims[3] = shp->width;
		rdims[2] = shp->height;
		rdims[1] = shp->batches;
		break;
	case 6:
		rdims[4] = shp->depth;
		rdims[3] = shp->width*shp->height;
		rdims[2] = shp->batches;
		break;
	case 7:
		rdims[4] = shp->depth;
		rdims[3] = shp->width*shp->height*shp->batches;
		break;
	case 8:
		rdims[3] = shp->depth;
		rdims[2] = shp->width*shp->height*shp->batches;
		break;
	case 9:
		rdims[3] = shp->depth;
		rdims[2] = shp->width*shp->height;
		rdims[1] = shp->batches;
		break;
	case 10:
		rdims[3] = shp->depth;
		rdims[2] = shp->width;
		rdims[1] = shp->height;
		rdims[0] = shp->batches;
		break;
	case 11:
		rdims[3] = shp->depth;
		rdims[2] = shp->width;
		rdims[1] = shp->height*shp->batches;
		break;
	case 12:
		rdims[3] = shp->depth*shp->width;
		rdims[2] = shp->height*shp->batches;
		break;
	case 13:
		rdims[3] = shp->depth*shp->width;
		rdims[2] = shp->height;
		rdims[1] = shp->batches;
		break;
	case 14:
		rdims[3] = shp->depth*shp->width*shp->height;
		rdims[2] = shp->batches;
		break;
	case 15:
		rdims[3] = shp->depth*shp->width*shp->height*shp->batches;
		break;
	}
}

void set_output_range(struct nn_node *self) {
	float minval = tensor_get_float(self->inputs[2], 0);
	float maxval = tensor_get_float(self->inputs[3], 0);
	tensor_set_single_float(self->outputs[2], minval);
	tensor_set_single_float(self->outputs[3], maxval);
	float out_level_size = (maxval - minval) / 255;
	float out_max = 2147483648.0f/*0x1.0p31f*/ * out_level_size * out_level_size;
	float out_min = -out_max;
	tensor_set_single_float(self->outputs[4], out_min);
	tensor_set_single_float(self->outputs[5], out_max);
}

//
// SumX is sum of x, SumX2 is sum of square of x.
// SumX' is sum of x-mx, SumX2' is sum of square of x-mx
// Input (x) are pivoted close to zero mean (x-mx), and mx = approx(mean(x)).
// var (x) is the same as var(x-mx)
// var(x-mx) = (n*SumX2'-SumX'^2)/n^2
// var(x-mx) is approximate (SumX2 - mx*(2*SumX-n*mx)) / n
//
void reduction1_hvx(const uint8_t *in, uint8_t inoff, uint8_t* mout, int32_t *vout, int32_t *rdims) {
	int n_out = rdims[0];
	int r_out = rdims[1];
	int n_in = rdims[2];
	int r_in = rdims[3];
	int i_out, i_in, ir_out, cnt0 = 0, cnt1 = 0, i;
	uint32_t reduction = r_out * r_in;
	uint32_t recip_den = min_u32(0x7fffffff, 0x80000000U / reduction);
	int32_t stride = r_in * n_in;

	HVX_Vector sZero = Q6_V_vzero();
	HVX_Vector sRecipDen = Q6_V_vsplat_R(recip_den);
	HVX_VectorPred Q0 = q6op_Q_vsetq2_R(r_in);
	HVX_VectorPred Q1 = Q6_Q_vsetq_R(4);
	HVX_Vector sMaccInt, sVaccInt, sMFifo, sVFifo;

	// compute ceil(log(reduction))
	// do we need more than 16b of guard bit in variance acc
	int32_t clb = Q6_R_clb_R(reduction);
	clb = Q6_R_clb_R(reduction + (1 << (31 - clb)) - 1);
	clb = clb > 16 ? 0 : 16 - clb;
	recip_den <<= clb;
	HVX_Vector sRecipDenVar = Q6_V_vsplat_R(recip_den);
	HVX_Vector sReduction = Q6_V_vsplat_R(reduction);
	HVX_Vector sReductionx64 = Q6_V_vsplat_R(reduction*64);
	HVX_Vector sC80 = Q6_V_vsplat_R(0x80808080);
	HVX_Vector sC80last = Q6_V_vmux_QVV(Q0, sC80, sZero);

	sMaccInt = sVaccInt = sMFifo = sVFifo = sZero;
	for (i_out = 0; i_out < n_out; i_out++) {
		for (i_in = 0; i_in < n_in; i_in++, cnt0++) {
			HVX_Vector sMacc = sZero;
			HVX_Vector sVacc = sZero;
			const uint8_t* inp = in + i_in * r_in + i_out * (r_in*n_in*r_out);
			for (ir_out = 0; ir_out < r_out; ir_out++) {
				for (i = 0;i < r_in-127; i += 128) {
					HVX_Vector sIn = vmemu(&inp[i]);
					sMacc = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc, sIn, 0x1010101);
					HVX_Vector sInpi = Q6_V_vxor_VV(sIn, sC80);
					sVacc = Q6_Vw_vrmpyacc_VwVbVb(sVacc, sInpi, sInpi);
				}
				if (i < r_in) {
					HVX_Vector sIn = vmemu(&inp[i]);
					sIn = Q6_V_vmux_QVV(Q0, sIn, sZero);
					sMacc = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc, sIn, 0x1010101);
					HVX_Vector sInpi = Q6_V_vxor_VV(sIn, sC80last);
					sVacc = Q6_Vw_vrmpyacc_VwVbVb(sVacc, sInpi, sInpi);
				}
				inp += stride;
			}
			HVX_Vector sVaccComb = Q6_Vuw_vlsr_VuwR(sVacc, clb);
			HVX_VectorPair dVaccInt = Q6_W_vdeal_VVR(sVaccComb, sVaccInt, -4);
			sVaccInt = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dVaccInt), Q6_V_lo_W(dVaccInt));
			HVX_VectorPair dMaccInt = Q6_W_vdeal_VVR(sMacc, sMaccInt, -4);
			sMaccInt = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dMaccInt), Q6_V_lo_W(dMaccInt));

			if (cnt0 >= 5) {
				sVFifo = Q6_V_valign_VVI(sVaccInt, sVFifo, 4);
				sMFifo = Q6_V_valign_VVI(sMaccInt, sMFifo, 4);
				sVaccInt = Q6_V_vmux_QVV(Q1, sZero, sVaccInt);
				sMaccInt = Q6_V_vmux_QVV(Q1, sZero, sMaccInt);
				cnt1++;

				if ((cnt1 & 31) == 0) {
					HVX_Vector sMean = Q6_Vw_vmpye_VwVuh(sMFifo, sRecipDen);
					sMean = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sMean, sMFifo, sRecipDen);

					HVX_Vector sT0 = Q6_Vw_vadd_VwVw(sMFifo, Q6_Vw_vsub_VwVw(sMFifo, Q6_Vw_vmpyie_VwVuh(sReduction, sMean)));
					sT0 = Q6_Vuw_vlsr_VuwR(sT0, clb);
					HVX_Vector sNVacc = Q6_Vw_vsub_VwVw(sVFifo, Q6_Vw_vmpyie_VwVuh(sT0, sMean));
					//var + (256*(sumX-n*64))>>clb
					HVX_Vector sAdj = Q6_Vw_vsub_VwVw(sMFifo, sReductionx64);
					sNVacc = Q6_Vw_vaslacc_VwVwR(sNVacc, sAdj, 8-clb);
					HVX_Vector sVar = Q6_Vw_vmpye_VwVuh(sNVacc, sRecipDenVar);
					sVar = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sVar, sNVacc, sRecipDenVar);

					vmemu(vout) = sVar;
					sMean = Q6_Vb_vpacke_VhVh(sZero, sMean);
					sMean = Q6_Vb_vpacke_VhVh(sZero, sMean);
					q6op_vstu_variable_ARV(mout, 32, sMean);

					mout+=32;
					vout+=32;
				}
			}
		}
	}
	for (int i = 0; i < 5; i++) {
		HVX_VectorPair dVaccInt = Q6_W_vdeal_VVR(sZero, sVaccInt, -4);
		sVaccInt = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dVaccInt), Q6_V_lo_W(dVaccInt));

		HVX_VectorPair dMaccInt = Q6_W_vdeal_VVR(sZero, sMaccInt, -4);
		sMaccInt = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dMaccInt), Q6_V_lo_W(dMaccInt));
		if (i+cnt0 >= 5) {
			sVFifo = Q6_V_valign_VVI(sVaccInt, sVFifo, 4);
			sMFifo = Q6_V_valign_VVI(sMaccInt, sMFifo, 4);
			sVaccInt = Q6_V_vmux_QVV(Q1, sZero, sVaccInt);
			sMaccInt = Q6_V_vmux_QVV(Q1, sZero, sMaccInt);
			cnt1++;

			if ((cnt1 & 31) == 0 || cnt0 == cnt1) {
				int cnt = cnt1 & 31;
				sVFifo = Q6_V_vror_VR(sVFifo, -cnt * 4);
				sMFifo = Q6_V_vror_VR(sMFifo, -cnt * 4);

				HVX_Vector sMean = Q6_Vw_vmpye_VwVuh(sMFifo, sRecipDen);
				sMean = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sMean, sMFifo, sRecipDen);

				HVX_Vector sT0 = Q6_Vw_vadd_VwVw(sMFifo, Q6_Vw_vsub_VwVw(sMFifo, Q6_Vw_vmpyie_VwVuh(sReduction, sMean)));
				sT0 = Q6_Vuw_vlsr_VuwR(sT0, clb);
				HVX_Vector sNVacc = Q6_Vw_vsub_VwVw(sVFifo, Q6_Vw_vmpyie_VwVuh(sT0, sMean));
				//var + (256*(sumX-n*64))>>clb
				HVX_Vector sAdj = Q6_Vw_vsub_VwVw(sMFifo, sReductionx64);
				sNVacc = Q6_Vw_vaslacc_VwVwR(sNVacc, sAdj, 8-clb);
				HVX_Vector sVar = Q6_Vw_vmpye_VwVuh(sNVacc, sRecipDenVar);
				sVar = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sVar, sNVacc, sRecipDenVar);

				if ((cnt1 & 31) == 0)
					vmemu(vout) = sVar;
				else
					q6op_vstu_variable_ARV(vout, cnt1 * 4, sVar);
				sMean = Q6_Vb_vpacke_VhVh(sZero, sMean);
				sMean = Q6_Vb_vpacke_VhVh(sZero, sMean);
				q6op_vstu_variable_ARV(mout, 32, sMean);

				mout += 32;
				vout += 32;
			}
		}
	}
}

void reductionX_hvx(const uint8_t *in, uint8_t inoff, uint8_t* mout, int32_t *vout, int32_t *rdims) {
	int r_out = rdims[1];
	int n_in = rdims[2];
	int r_in = rdims[3];
	int n_vec = rdims[4];
	int i_in, ir_out, ivec;
	uint32_t reduction = r_out * r_in;

	uint32_t recip_den = min_u32(0x7fffffff, 0x80000000U / reduction);
	HVX_Vector sZero = Q6_V_vzero();
	HVX_Vector sRecipDen = Q6_V_vsplat_R(recip_den);

	// compute ceil(log(reduction))
	// do we need more than 16b of guard bit in variance acc
	int32_t clb = Q6_R_clb_R(reduction);
	clb = Q6_R_clb_R(reduction+(1<<(31-clb))-1);
	clb = clb > 16 ? 0 : 16 - clb;
	recip_den <<= clb;
	HVX_Vector sRecipDenVar = Q6_V_vsplat_R(recip_den);
	HVX_Vector sReduction = Q6_V_vsplat_R(reduction);

	for (i_in = 0; i_in < n_in; i_in++) {
		for (ivec = 0; ivec < n_vec; ivec += 128) {
			HVX_VectorPair dVacc20 = Q6_W_vcombine_VV(sZero, sZero);
			HVX_VectorPair dVacc31 = Q6_W_vcombine_VV(sZero, sZero);
			HVX_Vector sMacc0, sMacc1, sMacc2, sMacc3;
			HVX_VectorPred Q0 = Q6_Q_vcmp_eq_VwVw(sZero, sZero);
			if (ivec + 128 > n_vec) Q0 = q6op_Q_vsetq2_R(n_vec);
			sMacc0 = sMacc1 = sMacc2 = sMacc3 = sZero;

			if (clb) {
				for (ir_out = 0; ir_out < r_out; ir_out++) {
					const uint8_t* inp = in + ivec + i_in * (r_in*n_vec) + ir_out * (n_in*r_in*n_vec);
					const int32_t blk = 1 << 15;
					for (int sup_i = 0; sup_i < r_in; sup_i += blk) {
						HVX_VectorPair dBlkVacc20 = Q6_W_vcombine_VV(sZero, sZero);
						HVX_VectorPair dBlkVacc31 = Q6_W_vcombine_VV(sZero, sZero);
						int32_t blklen = min_i32(r_in - sup_i, blk);
						for (int i = sup_i; i < sup_i+blklen; i++) {
							HVX_Vector sIn = vmemu(&inp[i*n_vec]);
							sIn = Q6_V_vmux_QVV(Q0, sIn, sZero);

							sMacc0 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc0, sIn, 0x00000001);
							sMacc1 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc1, sIn, 0x00000100);
							sMacc2 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc2, sIn, 0x00010000);
							sMacc3 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc3, sIn, 0x01000000);

							HVX_VectorPair dInSq = Q6_Wuh_vmpy_VubVub(sIn, sIn);
							dBlkVacc20 = Q6_Wuw_vmpyacc_WuwVuhRuh(dBlkVacc20, Q6_V_lo_W(dInSq), 0x10001);
							dBlkVacc31 = Q6_Wuw_vmpyacc_WuwVuhRuh(dBlkVacc31, Q6_V_hi_W(dInSq), 0x10001);
						}
						HVX_Vector sVacc20_L = Q6_Vw_vasracc_VwVwR(Q6_V_lo_W(dVacc20), Q6_V_lo_W(dBlkVacc20), clb);
						HVX_Vector sVacc20_H = Q6_Vw_vasracc_VwVwR(Q6_V_hi_W(dVacc20), Q6_V_hi_W(dBlkVacc20), clb);
						HVX_Vector sVacc31_L = Q6_Vw_vasracc_VwVwR(Q6_V_lo_W(dVacc31), Q6_V_lo_W(dBlkVacc31), clb);
						HVX_Vector sVacc31_H = Q6_Vw_vasracc_VwVwR(Q6_V_hi_W(dVacc31), Q6_V_hi_W(dBlkVacc31), clb);
						dVacc20 = Q6_W_vcombine_VV(sVacc20_H, sVacc20_L);
						dVacc31 = Q6_W_vcombine_VV(sVacc31_H, sVacc31_L);
					}
				}
			}
			else {
				for (ir_out = 0; ir_out < r_out; ir_out++) {
					const uint8_t* inp = in + ivec + i_in * (r_in*n_vec) + ir_out * (n_in*r_in*n_vec);
					for (int i = 0;i < r_in; i++) {
						HVX_Vector sIn = vmemu(&inp[i*n_vec]);
						sIn = Q6_V_vmux_QVV(Q0, sIn, sZero);

						sMacc0 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc0, sIn, 0x00000001);
						sMacc1 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc1, sIn, 0x00000100);
						sMacc2 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc2, sIn, 0x00010000);
						sMacc3 = Q6_Vuw_vrmpyacc_VuwVubRub(sMacc3, sIn, 0x01000000);

						HVX_VectorPair dInSq = Q6_Wuh_vmpy_VubVub(sIn, sIn);
						dVacc20 = Q6_Wuw_vmpyacc_WuwVuhRuh(dVacc20, Q6_V_lo_W(dInSq), 0x10001);
						dVacc31 = Q6_Wuw_vmpyacc_WuwVuhRuh(dVacc31, Q6_V_hi_W(dInSq), 0x10001);
					}
				}
			}

			HVX_Vector sMean0 = Q6_Vw_vmpye_VwVuh(sMacc0, sRecipDen);
			sMean0 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sMean0, sMacc0, sRecipDen);
			HVX_Vector sMean1 = Q6_Vw_vmpye_VwVuh(sMacc1, sRecipDen);
			sMean1 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sMean1, sMacc1, sRecipDen);
			HVX_Vector sMean2 = Q6_Vw_vmpye_VwVuh(sMacc2, sRecipDen);
			sMean2 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sMean2, sMacc2, sRecipDen);
			HVX_Vector sMean3 = Q6_Vw_vmpye_VwVuh(sMacc3, sRecipDen);
			sMean3 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sMean3, sMacc3, sRecipDen);

			HVX_Vector sT0 = Q6_Vw_vadd_VwVw(sMacc0, Q6_Vw_vsub_VwVw(sMacc0, Q6_Vw_vmpyie_VwVuh(sReduction, sMean0)));
			HVX_Vector sT1 = Q6_Vw_vadd_VwVw(sMacc1, Q6_Vw_vsub_VwVw(sMacc1, Q6_Vw_vmpyie_VwVuh(sReduction, sMean1)));
			HVX_Vector sT2 = Q6_Vw_vadd_VwVw(sMacc2, Q6_Vw_vsub_VwVw(sMacc2, Q6_Vw_vmpyie_VwVuh(sReduction, sMean2)));
			HVX_Vector sT3 = Q6_Vw_vadd_VwVw(sMacc3, Q6_Vw_vsub_VwVw(sMacc3, Q6_Vw_vmpyie_VwVuh(sReduction, sMean3)));
			sT0 = Q6_Vuw_vlsr_VuwR(sT0, clb);
			sT1 = Q6_Vuw_vlsr_VuwR(sT1, clb);
			sT2 = Q6_Vuw_vlsr_VuwR(sT2, clb);
			sT3 = Q6_Vuw_vlsr_VuwR(sT3, clb);
			HVX_Vector sVacc20_L = Q6_Vw_vsub_VwVw(Q6_V_lo_W(dVacc20), Q6_Vw_vmpyie_VwVuh(sT0, sMean0));
			HVX_Vector sVacc20_H = Q6_Vw_vsub_VwVw(Q6_V_hi_W(dVacc20), Q6_Vw_vmpyie_VwVuh(sT2, sMean2));
			HVX_Vector sVacc31_L = Q6_Vw_vsub_VwVw(Q6_V_lo_W(dVacc31), Q6_Vw_vmpyie_VwVuh(sT1, sMean1));
			HVX_Vector sVacc31_H = Q6_Vw_vsub_VwVw(Q6_V_hi_W(dVacc31), Q6_Vw_vmpyie_VwVuh(sT3, sMean3));

			HVX_Vector sVar0 = Q6_Vw_vmpye_VwVuh(sVacc20_L, sRecipDenVar);
			sVar0 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sVar0, sVacc20_L, sRecipDenVar);
			HVX_Vector sVar2 = Q6_Vw_vmpye_VwVuh(sVacc20_H, sRecipDenVar);
			sVar2 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sVar2, sVacc20_H, sRecipDenVar);
			HVX_Vector sVar1 = Q6_Vw_vmpye_VwVuh(sVacc31_L, sRecipDenVar);
			sVar1 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sVar1, sVacc31_L, sRecipDenVar);
			HVX_Vector sVar3 = Q6_Vw_vmpye_VwVuh(sVacc31_H, sRecipDenVar);
			sVar3 = Q6_Vw_vmpyoacc_VwVwVh_s1_rnd_sat_shift(sVar3, sVacc31_H, sRecipDenVar);

			int inc = min_i32(128, n_vec - ivec);

			HVX_Vector sMean20 = Q6_Vh_vshuffe_VhVh(sMean2, sMean0);
			HVX_Vector sMean31 = Q6_Vh_vshuffe_VhVh(sMean3, sMean1);
			HVX_Vector sMean = Q6_Vb_vshuffe_VbVb(sMean31, sMean20);
			q6op_vstu_variable_ARV(mout, inc, sMean);

			HVX_VectorPair dVar20 = Q6_W_vshuff_VVR(sVar2, sVar0, -4);
			HVX_VectorPair dVar31 = Q6_W_vshuff_VVR(sVar3, sVar1, -4);
			HVX_VectorPair dVarL = Q6_W_vshuff_VVR(Q6_V_lo_W(dVar31), Q6_V_lo_W(dVar20), -4);
			HVX_VectorPair dVarH = Q6_W_vshuff_VVR(Q6_V_hi_W(dVar31), Q6_V_hi_W(dVar20), -4);


			if (inc <= 32) {
				q6op_vstu_variable_ARV(vout, inc*4, Q6_V_lo_W(dVarL));
			}
			else if (inc <= 64) {
				vmemu(vout) = Q6_V_lo_W(dVarL);
				q6op_vstu_variable_ARV(vout+32, (inc&31)*4?(inc&31)*4:128, Q6_V_hi_W(dVarL));
			}
			else if (inc <= 96) {
				vmemu(vout+ 0) = Q6_V_lo_W(dVarL);
				vmemu(vout+32) = Q6_V_hi_W(dVarL);
				q6op_vstu_variable_ARV(vout + 64, (inc&31)*4?(inc&31)*4:128, Q6_V_lo_W(dVarH));
			}
			else {
				vmemu(vout+ 0) = Q6_V_lo_W(dVarL);
				vmemu(vout+32) = Q6_V_hi_W(dVarL);
				vmemu(vout+64) = Q6_V_lo_W(dVarH);
				q6op_vstu_variable_ARV(vout + 96, (inc&31)*4?(inc&31)*4:128, Q6_V_hi_W(dVarH));
			}

			vout += inc;
			mout += inc;
		}
	}
}

static void
moments_work(struct nn_graph *nn, void *parg) {
	struct moments_runstate *pstate = (struct moments_runstate *)parg;

	int32_t jobid;

	if (pstate->rdims[4] == 1) {
		while (jobid = __sync_fetch_and_add(&pstate->next_job, 1), jobid < pstate->jobs) {
			const uint8_t *in = pstate->pin;
			uint8_t *mout = pstate->pmout;
			int32_t *vout = pstate->pvout;
			reduction1_hvx(in, pstate->input_offset, mout, vout, pstate->rdims);
		}
	}
	else {
		while (jobid = __sync_fetch_and_add(&pstate->next_job, 1), jobid < pstate->jobs) {
			const uint8_t *in = pstate->pin;
			uint8_t *mout = pstate->pmout;
			int32_t *vout = pstate->pvout;
			reductionX_hvx(in, pstate->input_offset, mout, vout, pstate->rdims);
		}
	}

	nn_sem_post(&pstate->done_sem);
}

static int moments_execute(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *axes_tensor = self->inputs[1];
	struct tensor *out_m_tensor = self->outputs[0];
	struct tensor *out_v_tensor = self->outputs[1];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t elemcount = in_batches * in_height * in_width * in_depth;

	int rflag = check_buffer(in_tensor, axes_tensor, out_m_tensor, out_v_tensor);
	if (rflag < 0) {
		return errlog(nn, "Output tensor does NOT match expected output tensor size. Or input axis error!!!");
	}

	if (tensor_out_prepare_normal_fromshape(out_m_tensor, &out_m_tensor->shape, NN_TYPE_UINT8) != 0 ||
		tensor_out_prepare_normal_fromshape(out_v_tensor, &out_v_tensor->shape, NN_TYPE_INT32) != 0
		)
	{
		return errlog(nn, "out too small");
	}

	struct moments_runstate runstate;
	rflag2rdim(rflag, runstate.rdims, &in_tensor->shape);

	float in_min_float = tensor_get_float(self->inputs[2], 0);
	float in_max_float = tensor_get_float(self->inputs[3], 0);
	int32_t input_offset = quantize_uint8(0.0f, in_min_float, in_max_float);

#if defined(DUMPDATA)
	{
		const uint8_t *in = (const uint8_t *)in_tensor->data;
		for (int b = 0; b < in_batches; b++) {
			printf("[");
			for (int h = 0; h < in_height; h++) {
				printf("[");
				for (int w = 0; w < in_width; w++) {
					printf("[");
					for (int d = 0; d < in_depth; d++) {
						printf("%d", in[d+w*in_depth+h*in_depth*in_width+b* in_depth*in_width*in_height]-input_offset);
						if (d != in_depth - 1) printf(",");
					}
					printf("]");
					if (w != in_width - 1) printf(",");
				}
				printf("]");
				if (h != in_height - 1) printf(",");
			}
			printf("]");
			if (b != in_batches - 1) printf(",");
		}
	}
#endif

	if (runstate.rdims[3] == 1) {
		memcpy(out_m_tensor->data, in_tensor->data, elemcount);
		memset(out_v_tensor->data, 0, elemcount * sizeof(int));
	}
	else {
		runstate.pin = (const uint8_t *)in_tensor->data;
		runstate.pmout = (uint8_t *)out_m_tensor->data;
		runstate.pvout = (int32_t *)out_v_tensor->data;
		runstate.input_offset = input_offset;
		//runstate.jobs = runstate.rdims[4] == 1 ? runstate.rdims[0] * runstate.rdims[2] : runstate.rdims[2] * runstate.rdims[4];
		runstate.jobs = 1;
		runstate.next_job = 0;

		nn_sem_init(&runstate.done_sem, 0);
		int n_threads = min_i32(NUM_THREADS, runstate.jobs);

		for (int i = 0; i < n_threads; i++)
			nn_os_work_for_vector(nn, moments_work, &runstate);
		nn_sem_wait_n_times(&runstate.done_sem, n_threads);
	}

	// set the output min/max
	set_output_range(self);

	return 0;
}

static int moments_execute_f(struct nn_node *self, struct nn_graph *nn) {
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *axes_tensor = self->inputs[1];
	struct tensor *out_m_tensor = self->outputs[0];
	struct tensor *out_v_tensor = self->outputs[1];

	int32_t in_batches = in_tensor->shape.batches;
	int32_t in_height = in_tensor->shape.height;
	int32_t in_width = in_tensor->shape.width;
	int32_t in_depth = in_tensor->shape.depth;
	int32_t elemcount = in_batches * in_height * in_width * in_depth;

	int32_t rflag = check_buffer(in_tensor, axes_tensor, out_m_tensor, out_v_tensor);
	if (rflag < 0) {
		return errlog(nn, "Output tensor does NOT match expected output tensor size. Or input axis error!!!");
	}

	if (tensor_out_prepare_normal_fromshape(out_m_tensor, &out_m_tensor->shape, NN_TYPE_FLOAT) != 0 ||
		tensor_out_prepare_normal_fromshape(out_v_tensor, &out_v_tensor->shape, NN_TYPE_FLOAT) != 0
		)
	{
		return errlog(nn, "out too small");
	}

	int32_t rdims[5] = { 0 };
	rflag2rdim(rflag, rdims, &in_tensor->shape);

	int32_t n_out = rdims[0];
	int32_t r_out = rdims[1];
	int32_t n_in = rdims[2];
	int32_t r_in = rdims[3];
	int32_t n_vec = rdims[4];
	float *in = in_tensor->data;
	float *outm = out_m_tensor->data;
	float *outv = out_v_tensor->data;
	if (r_in == 1) { /* no reduction, copy */
		memcpy(outm, in, elemcount*sizeof(in[0]));
		for (int32_t i = 0; i < elemcount; i++)
			outv[i] = 0.0f;
	}
	else if (n_vec == 1) {
		for (int32_t i_out = 0; i_out < n_out; i_out++) {
			for (int32_t i_in = 0; i_in < n_in; i_in++) {
				float sum = 0.0f;
				float sum2 = 0.0f;
				for (int32_t ir_out = 0; ir_out < r_out; ir_out++) {
					const float* inp = in + i_in * r_in + ir_out * (r_in*n_in) + i_out * (r_in*n_in*r_out);
					for (int32_t i = 0;i < r_in; i ++) {
						sum += inp[i];
						sum2 += inp[i] * inp[i];
					}
				}
				float out_mean = sum / (r_out*r_in);
				*outm++ = out_mean;
				*outv++ = sum2 / (r_out*r_in) - out_mean * out_mean;
			}
		}
	}
	else {
		for (int32_t i_in = 0; i_in < n_in; i_in++) {
			for (int32_t ivec = 0; ivec < n_vec; ivec++) {
				float sum = 0.0f;
				float sum2 = 0.0f;
				for (int32_t ir_out = 0; ir_out < r_out; ir_out++) {
					const float* inp = in + ivec + i_in * (r_in*n_vec) + ir_out * (n_in*r_in*n_vec);
					for (int32_t i = 0;i < r_in; i++) {
						sum += inp[i*n_vec];
						sum2 += inp[i*n_vec] * inp[i*n_vec];
					}
				}
				float out_mean = sum / (r_out*r_in);
				*outm++ = out_mean;
				*outv++ = sum2 / (r_out*r_in) - out_mean * out_mean;
			}
		}
	}

	return 0;
}

struct nn_node_ops nn_ops_for_Moments_8to32 = {
		.execute = moments_execute,
		.check = NULL,
		.ctor = node_alloc_common,
		.dtor = node_free_common,
		.n_inputs = NN_IOCOUNT(4),
		.n_outputs = NN_IOCOUNT(6),
};

struct nn_node_ops nn_ops_for_Moments_f = {
		.execute = moments_execute_f,
		.check = NULL,
		.ctor = node_alloc_common,
		.dtor = node_free_common,
		.n_inputs = NN_IOCOUNT(2),
		.n_outputs = NN_IOCOUNT(2),
};
