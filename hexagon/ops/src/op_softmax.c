
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
#include <nn_graph.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <hexagon_types.h>
#include <op_softmax.h>




/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains min and max (floating) ops
 */
//#define SOFTMAX_PERFORMANCE

// expf approx, same domain as 'proper' expf, and
// accurate enough for softmax with 8-bit output.
//
static inline float expf_approx(float x){
	int xfrac = roundf_i32(x*(float)(16384.0/0.69314718056) );
	// now we want 2^xfrac (which has 14 fractional bits).
	float xf  = (xfrac & 0x3FFF) * (float)(1./16384.);
	// approx for 2^x
	// good to about 11 bits
	// 4th order would be:
	// 1.          0.69304423  0.24130777  0.0521633   0.0134847

	float exp0 = 1.0f + xf*(0.69051585f + xf*(0.23793659f + xf*0.07154756f));
	return flt_power2(xfrac>>14)*exp0;
}

// softmax on qu8 inputs
// defined (over a set of inputs x[0..n-1]) as
//
//       t[i] = exp( beta * x[i])
//       y[i] = t[i]/sum(t[i])
//
// All results are in range 0..1, and have the same ordering as the input values.
// - sum of all outputs is 1.0 (ideally, mathematically..)
// - if one value is distinctly greater than all the others, it will tend to be close to 1.0
//    and the others close to 0.0
//   The amount of margin needed for 'distinctly' depends on beta.
//
// Note that adding a constant amount to all inputs has no effect; and
// scaling the inputs by some factor is equivalent to changing beta; so we  can
// assume for the discussion that x[i] are 0..255  and beta has been thus scaled.
//
// To keep t[i] values in a manageable range this is typically done as
//       t[i] = exp( beta * (x[i]  - K))
//       y[i] = t[i]/sum(t[i])
//
// .. where K is max(x[i]), so that all the exp() parameters are <= 0, and
//     at least one is zero.
//
// When using floats, expf(x) is good up to x = 79.0  (even allowing up to 16k
// accumulations of the largest value); and is fully precise down to x = -87.
// (but can only go down to x= -83 for reasons given below).
// So, whenever beta *255 < 162, which is pretty much always, we can use K = 83/beta, and avoid the need to
// find the actual maximum ('beta' here refers to the scaled beta, which is 'stepsize' in code below).
// We need to ensure that 255.0/sum(expf) doesn't overflow; that gives the lower limit of -83.
//
static int qsoftmax_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	int i,j;
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	int depth = in_tensor->shape.depth;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	float beta = (self->n_inputs < 4) ? 1.0f : tensor_get_float(self->inputs[3],0);
	float in_min = tensor_get_float(in_min_tensor,0);
	float in_max = tensor_get_float(in_max_tensor,0);
	float stepsize = beta*(in_max-in_min)/255.0f;
	const uint8_t *in = in_tensor->data;
	uint8_t *out = out_tensor->data;
	int maxval;
	int inval;
	float *temp_slice;
	float sum;
	float recip;
	logmsg(nn,2,"qsoftmax ref in=%dx%dx%dx%d beta=%f stepsize=%f",batches,height,width,depth,beta,stepsize);

#ifdef SOFTMAX_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif

	if ((temp_slice = nn_scratch_alloc(nn,depth*sizeof(float))) == NULL) {
		return errlog(nn,"can't alloc temp buffer");
	}
	if (tensor_out_prepare_normal(out_tensor,batches,height,width,depth,NN_TYPE_QUINT8) != 0) {
		return errlog(nn,"out too small");
	}
	if( tensor_set_single_float(out_min_tensor,0.0f) != 0
	    || tensor_set_single_float(out_max_tensor,1.0f) != 0){
		return errlog(nn,"can't prep min or max");
	}

	if( stepsize < 0.63529f){
		// stepsize * 255 < 162
		// so stepsize*255 - 83  <= 79, and we can do expf of the
		// whole range of (stepsize * uint8 -  83). Avoids the need for
		// a 'max' pass.
		// Note that 255./expf(-83) does not overflow (as when depth =1
		// and the only input is zero).
		//
		for (j = 0; j < batches*height*width; j++) {
			float sum = 0.0f;
			for (i = 0; i < depth; i++) {
				inval = in[j*depth+i];
				sum += (temp_slice[i] = expf_approx(stepsize*inval-83.0f));
			}
			recip = 255.0f/sum;
			for (i = 0; i < depth; i++) {
				out[j*depth+i] = saturate_u8(fast_roundf(recip*temp_slice[i]));
			}
		}
	}else{
		for (j = 0; j < batches*height*width; j++) {
			maxval = 0;
			sum = 0.0f;
			for (i = 0; i < depth; i++) {
				inval = in[j*depth+i];
				if (maxval < inval) maxval = inval;
			}
			for (i = 0; i < depth; i++) {
				inval = in[j*depth+i];
				sum += (temp_slice[i] = expf_approx(stepsize*(inval-maxval)));
			}
			/* best temp_slice is 1.0. */
			recip = 255.0f/sum;
			for (i = 0; i < depth; i++) {
				out[j*depth+i] = saturate_u8(fast_roundf(recip*temp_slice[i]));
			}
		}
	}
#ifdef SOFTMAX_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsoftmax ref cycles = %d\n",end_time-start_time);
#endif

	logmsg(nn,2,"qsoftmax ref done");
	return 0;
}

#if 0
/*
	b**x = exp(x * log(b)
	exp(x) = exp2(x * (1/log(2)))

	// After Quantization
	x_i = in_i - 255 => Note: x_i will always be negative and max value of 0
	S = (in_max - in_min)/255
	const_fact = S*(1/log(2))
	softmax(x_j) = exp(x_j*S)/sum_i(exp(x_i*S))
	softmax(x_j) = exp2(x_j*const_fact)/sum_i(exp2(x_i*const_fact))

	x_j * const_fact => can be split into mantissa and exponent for further simplification
*/
static int qsoftmax_execute_ref(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	int depth = in_tensor->shape.depth;
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	size_t elements = depth * batches * height * width;
	float *expf_out = nn->scratch;								// Note: size(expf_out) = (depth * 4) bytes
	float *outf = (float *)pad_and_align(expf_out, depth*4);	// Note: size(outf) = (depth * batches * height * width * 4) bytes
	float beta = (self->n_inputs < 4) ? 1.0f : tensor_get_float(self->inputs[3],0);

	const uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	int i;
	int j;
	float in_min = tensor_get_float(in_min_tensor,0) * beta;
	float in_max = tensor_get_float(in_max_tensor,0) * beta;
	float inval, outval, calcmaxx=-INFINITY, outmin=INFINITY;
	float sum, sum_recip;
	float const_fact = (in_max - in_min)/(255*0.69315); // 0.69315 - log(2)
	float mant2;
	int exp2;

	logmsg(nn,2,"qsoftmax ref execute. self=%p ",self);
	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!=0){
		return errlog(nn,"out too small");
	}

#ifdef SOFTMAX_PERFORMANCE
	int start_time, end_time;
	start_time =  nn_os_get_cycles(nn);
#endif
	for (j = 0; j < batches*height*width; j++) {


		sum = 0.0f;
		for (i = 0; i < depth; i++) {
			inval = (in_data[i] - 255) * const_fact;
			exp2 = inval;
			mant2 = inval - exp2;
			expf_out[i] = powf(2.0,exp2) * powf(2.0,mant2-1);
			sum = sum + expf_out[i];
		}

		sum_recip = 1.0f/sum;
		for (i = 0; i < depth; i++) {
			outval = expf_out[i] * sum_recip;
			//printf("outval=%f\n",outval);
			calcmaxx = fmaxf(calcmaxx,outval);
			outmin = fminf(outmin,outval);
			outf[i] = outval;
		}
		out_data += depth;
		in_data += depth;
		outf += depth;
	}

	/* Quantize output */
	out_data = out_tensor->data;
	outf = (float *)pad_and_align(expf_out, depth*4);
	quant_u8(out_data, outf, elements, outmin, calcmaxx);

	tensor_set_single_float(out_min_tensor,outmin);
	tensor_set_single_float(out_max_tensor,calcmaxx);

#ifdef SOFTMAX_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsoftmax ref cycles = %d\n",end_time-start_time);
#endif

	logmsg(nn,2,"qsoftmax %p done",self);
	return 0;
}

#if 0
struct tdata {
	struct nn_node *self;
	int whoami;
	void *in_data;
	void *out_data;
	float in_min;
	float in_max;
	float *outputmax;
	int bhw;
	int depth;
	nn_sem_t donesem;
};

static void qsoftmax_execute_hvx_td(struct nn_graph *nn, void *vtdata)
{
	/* Parameters */
	struct tdata *td = vtdata;
	const uint8_t *in_data = td->in_data;
	uint8_t *out_data = td->out_data;
	float in_min = td->in_min;
	float in_max = td->in_max;
	float *outputmax = td->outputmax;
	int bhw = td->bhw;
	int depth = td->depth;
	
	/* Scalar variables */
	float factor;
	float calcmax = -INFINITY;
	float const_fact = (in_max - in_min)/(255*0.69315); // 0.69315 - log(2)
	int const_fact_int = (const_fact) * 0x1.0p16f + 0.5;
	int sum_fix;
	int ival;
	int i;
	int j;
	int depth_pad = (depth+MAXPAD-1)&~(MAXPAD-1);
	uint8_t max_out_data = 0;//, max_out_data_hvx=0;
	
	/* Scratch space allocation - temporary */
	uint8_t *out_pad = nn->scratch;
	uint8_t *in_pad = (uint8_t *)pad_and_align(out_pad, bhw*depth_pad);
	float *outf = (float *)pad_and_align(in_pad, depth_pad);				// Note: size(outf) = (depth * batches * height * width * 4) bytes
	int8_t *exp2_shift = (int8_t *)pad_and_align(outf, bhw*depth*sizeof(float));	// along depth dimension
	float *sum_recip_buf = (float *)pad_and_align(exp2_shift, depth_pad);
	uint8_t *exp2_out = (uint8_t *)pad_and_align(sum_recip_buf, depth_pad);

#ifdef SOFTMAX_PERFORMANCE
	int start_time, end_time;//, t1, t2, t3, t4, t5;
	start_time = nn_os_get_cycles(nn);
#endif

	/* HVX intrinsics variables */
	HVX_Vector vin_data, vexp_data, v255, v_ivalh_o, v_ivalh_e, v_tmp, v_tmp2, vx00ff;
	HVX_VectorPair d_vtemp, d_vtemp2, d_v_ivalw_e, d_v_ivalw_o;
	int const_fact_Rh = Q6_R_combine_RlRl(const_fact_int, const_fact_int);
	v255 = Q6_V_vsplat_R(0xffffffff);
	vx00ff = Q6_V_vsplat_R(0x00ff00ff);
	HVX_Vector v0 = Q6_V_vsplat_R(0x00000000);
	HVX_Vector vmax_out_data = Q6_V_vsplat_R(0x00000000);
	HVX_Vector vsum;
	HVX_Vector *ptr_out_data, *ptr_data;
	HVX_Vector *ptr_exp2_in, *ptr_in_pad;

	//int start_time, end_time;//, t1, t2, t3, t4, t5;
	//start_time = nn_os_get_cycles(nn);

	for (j = 0; j < bhw; j++){
		/*
			b**x = exp(x * log(b)
			exp(x) = exp2(x * (1/log(2)))

			// After Quantization
			x_i = in_i - 255 => Note: x_i will always be negative and max value of 0
			S = (in_max - in_min)/255
			const_fact = S*(1/log(2))
			softmax(x_j) = exp(x_j*S)/sum_i(exp(x_i*S))
			softmax(x_j) = exp2(x_j*const_fact)/sum_i(exp2(x_i*const_fact))

			x_j * const_fact => can be split into mantissa and exponent for further simplification
		*/

		/* Compute Mantissa for exp2(x) */
		/* Quantize input for non-linear function exp2 */

		/* 1. Find integer and fractional part of x_i = ((255 - in_i) * const_fact) where const_fact = (in_max - in_min)/(255 * log(2))
		   exp2_in_i = integer_part(x_i)
		   in_pad_i = frac_part(x_i) */

		/* HVX Intrinsics code below implements the following C code
			for (i = 0; i < depth; i++) {
				ival = (255-in_data[i]) * const_fact_int;
				exp2_shift[i] = ival >> 16;
				in_pad[i] = 255 - ((ival & 0xff00)>>8);	// (mant2+1) * 255
			}
		*/

		ptr_data = (HVX_Vector *)in_pad;
		ptr_exp2_in = (HVX_Vector *)exp2_shift;
		ptr_in_pad = (HVX_Vector *)in_pad;
		// Copy input to aligned buffer
		vmemcpy_asm(in_pad,in_data,depth);
		for (i = 0; i < depth_pad; i+=128) {
			vin_data = *ptr_data++;
			d_vtemp = Q6_Wh_vsub_VubVub(v255,vin_data);
			d_v_ivalw_e = Q6_Ww_vmpy_VhRh(Q6_V_lo_W(d_vtemp),const_fact_Rh);
			d_v_ivalw_o = Q6_Ww_vmpy_VhRh(Q6_V_hi_W(d_vtemp),const_fact_Rh);
			v_ivalh_e = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(d_v_ivalw_e), Q6_V_lo_W(d_v_ivalw_e), 10);
			v_ivalh_o = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(d_v_ivalw_o), Q6_V_lo_W(d_v_ivalw_o), 10);
			v_tmp = Q6_Vub_vasr_VhVhR_sat(v_ivalh_o, v_ivalh_e, 6);
			*ptr_exp2_in++ = v_tmp;

			v_ivalh_e = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(d_v_ivalw_e), Q6_V_lo_W(d_v_ivalw_e), 8);
			v_ivalh_o = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(d_v_ivalw_o), Q6_V_lo_W(d_v_ivalw_o), 8);
			v_ivalh_e = Q6_V_vand_VV(v_ivalh_e, vx00ff);
			v_ivalh_o = Q6_V_vand_VV(v_ivalh_o, vx00ff);
			v_tmp = Q6_Vub_vasr_VhVhR_sat(v_ivalh_o, v_ivalh_e, 0);
			v_tmp = Q6_Vub_vsub_VubVub_sat(v255,v_tmp);
			*ptr_in_pad++ = v_tmp;
		}
		//t1 = nn_os_get_cycles(nn);

		/* Compute exp2(x) */
		// y = exp2(x), x is in range [-1 0], y = [0.5 1]
		non_lin_i_exp2_8((signed char *)exp2_out, (signed char *)in_pad, depth_pad);

		//t2 = nn_os_get_cycles(nn);

		/* Zero out out_pad buffer and copy valid exp2_out into out_pad buffer */
		/* This is important since we compute max and sum on out_pad buffer */
		vmemset_asm(out_pad,0,depth_pad);
		vmemcpy_asm(out_pad,exp2_out,depth);

		/* 3. Find sum_fix and max_out_data on out_pad_i = (exp2_out_i >> exp2_in_i) */
		/* Following C code is implemented using HVX intrinsics
			sum_fix = 0;
			for (i = 0; i < depth; i++) {
				ival = out_pad[i] >> exp2_shift[i];
				max_out_data = (ival > max_out_data) ? ival : max_out_data;
				out_data[i] = ival;
				sum_fix += ival;
			}
		*/

		ptr_out_data = (HVX_Vector *)out_pad;
		ptr_in_pad = (HVX_Vector *)in_pad;
		ptr_data = (HVX_Vector *)exp2_shift;
		vsum = Q6_V_vsplat_R(0x00000000);
		for(i=0;i<depth_pad;i+=128) {
			vin_data = *ptr_out_data;
			vexp_data = *ptr_data++;
			d_vtemp = Q6_Wuh_vzxt_Vub(vin_data);
			d_vtemp2 = Q6_Wuh_vzxt_Vub(vexp_data);
			v_tmp = Q6_Vh_vasr_VhVh(Q6_V_lo_W(d_vtemp),Q6_V_lo_W(d_vtemp2));
			v_tmp2 = Q6_Vh_vasr_VhVh(Q6_V_hi_W(d_vtemp),Q6_V_hi_W(d_vtemp2));
			d_vtemp = Q6_Ww_vadd_VhVh(v_tmp,v_tmp2);
			vsum = Q6_Vw_vadd_VwVw_sat(vsum,Q6_V_lo_W(d_vtemp));
			vsum = Q6_Vw_vadd_VwVw_sat(vsum,Q6_V_hi_W(d_vtemp));
			v_tmp = Q6_Vub_vasr_VhVhR_rnd_sat(v_tmp2,v_tmp, 0);
			*ptr_out_data++ = v_tmp;
			vmax_out_data = Q6_Vub_vmax_VubVub(vmax_out_data,v_tmp);
		}

		//t3 = nn_os_get_cycles(nn);

		/* Find scalar maximum (8-bit output) from vector registers */
		ival = 64;
		// ival = 64, 32, 16, 8, 4, 2, 1
		while(ival >= sizeof(uint8_t)) {
			v_tmp = Q6_V_vror_VR(vmax_out_data,ival);					// vmax_out_data will contain per-lane maximum values
			vmax_out_data = Q6_Vub_vmax_VubVub(vmax_out_data, v_tmp);	// v_tmp contains max value among 64 bytes
			ival >>= 1;
		}

		/* Find scalar sum (32-bit output) from vector registers */
		ival = 64;
		// ival = 64, 32, 16, 8, 4
		while(ival >= sizeof(uint32_t)) {
			v_tmp = Q6_V_vror_VR(vsum,ival);
			vsum = Q6_Vw_vadd_VwVw_sat(vsum,v_tmp);
			ival >>= 1;
		}
		max_out_data = Q6_R_vextract_VR(vmax_out_data,0) & 0xff;
		sum_fix =  Q6_R_vextract_VR(vsum,0);

		/* Find calcmax */
		sum_recip_buf[j] = 255.0/sum_fix;
		calcmax = max_out_data/(float)sum_fix;
		//t4 = nn_os_get_cycles(nn);

		out_pad += depth_pad;
		in_data += depth;
	}

	/* Convert quantized values based on calcmax */
	if(bhw != 1) {
		//out_data = out_tensor->data;
		out_pad = nn->scratch;
		for (j = 0; j < bhw; j++){

			/* Multiply exp2(x) with reciprocal of sum(exp2(x)) */
			/*
			factor = sum_recip_buf[j]/calcmax;
			for (i = 0; i < depth; i++) {
				ival = out_data[i];
				out_data[i] = out_data[i] * factor;
				//printf("2. ival=%d factor=%f out_data=%d\n",ival,sum_recip_buf[j]/calcmax,out_data[i]);
				printf("C ref: In: %x out:%x HVX=%x\n", ival, out_data[i], out_pad[i]);
			}
			*/

			factor = sum_recip_buf[j]/calcmax;
			int factor_int = (factor) * 0x1.0p14f + 0.5;
			int factor_int_Rh = Q6_R_combine_RlRl(factor_int, factor_int);
			ptr_out_data = (HVX_Vector *)out_pad;
			for (i = 0; i < depth_pad; i+=128) {
				vin_data = *ptr_out_data;
				d_vtemp = Q6_Wh_vsub_VubVub(vin_data,v0);
				d_v_ivalw_e = Q6_Ww_vmpy_VhRh(Q6_V_lo_W(d_vtemp),factor_int_Rh);
				d_v_ivalw_o = Q6_Ww_vmpy_VhRh(Q6_V_hi_W(d_vtemp),factor_int_Rh);
				v_ivalh_e = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(d_v_ivalw_e), Q6_V_lo_W(d_v_ivalw_e), 10);
				v_ivalh_o = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(d_v_ivalw_o), Q6_V_lo_W(d_v_ivalw_o), 10);
				v_tmp = Q6_Vub_vasr_VhVhR_sat(v_ivalh_o, v_ivalh_e, 4);
				*ptr_out_data++ = v_tmp;
			}

			out_pad += depth_pad;
		}
	}

	/* 6. Copy out_pad to out_data */
	/* Copy from aligned buffer to out_data */
	out_pad = nn->scratch;
	for (j = 0; j < bhw; j++){
		vmemcpy_asm(out_data,out_pad,depth);
		out_data += depth;
		out_pad += depth_pad;
	}
	//t5 = nn_os_get_cycles(nn);
	*outputmax = calcmax;
	nn_sem_post(&td->donesem);
}

static int qsoftmax_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *in_min_tensor = self->inputs[1];
	const struct tensor *in_max_tensor = self->inputs[2];
	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;
	int bhw = batches * height * width;
	uint8_t *in_data = in_tensor->data;
	uint8_t *out_data = out_tensor->data;
	float beta = (self->n_inputs < 4) ? 1.0f : tensor_get_float(self->inputs[3],0);
	float in_min = tensor_get_float(in_min_tensor,0) * beta;
	float in_max = tensor_get_float(in_max_tensor,0) * beta;
	float out_max = -INFINITY;
	
	logmsg(nn,2,"qsoftmax hvx execute. self=%p ",self);

	if( tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8)!=0){
		return errlog(nn,"out too small");
	}

	struct tdata td = {
		.self = self,
		.whoami = 0,
		.in_data = in_data,
		.out_data = out_data,
		.in_min = in_min,
		.in_max = in_max,
		.outputmax = &out_max,
		.bhw = bhw,
		.depth = depth,
	};

	nn_sem_init(&td.donesem,0);
	nn_os_work_for_vector(nn, qsoftmax_execute_hvx_td, &td);
	nn_sem_wait(&td.donesem);

	tensor_set_single_float(out_min_tensor, 0.0f);
	tensor_set_single_float(out_max_tensor,out_max);


#ifdef SOFTMAX_PERFORMANCE
	end_time =  nn_os_get_cycles(nn);
	printf("qsoftmax hvx cycles = %d\n", end_time-start_time);

	/*
	printf("qsoftmax hvx cycles = %d->%d->%d->%d->%d = %d\n",
			t1-start_time, t2-start_time, t3-start_time, t4-start_time, t5-start_time,
			end_time-start_time);
	*/
#endif

	logmsg(nn,2,"qsoftmax %p done",self);
	return 0;
}
#endif
#endif

static int qsoftmax_check(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"Checking softmax node %p",self);

	int k = node_check_inputs_range( self,nn, "softmax",3,4);	// 3 or 4 inputs
	if( k==0) k = node_check_outputs_n( self, nn, "softmax", 3);	// 3 outputs
	if( k!=0)
		return k;
	logmsg(nn,2,"softmax node %p check OK",self);
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8_ref = {
	.execute = qsoftmax_execute_ref,
	.check = qsoftmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

struct nn_node_ops nn_ops_for_QuantizedSoftmax_8 = {
	//.execute = qsoftmax_execute_hvx,
	.execute = qsoftmax_execute_ref,
	.check = qsoftmax_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
};

