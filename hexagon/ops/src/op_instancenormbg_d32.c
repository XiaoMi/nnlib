
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
/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains implementations for instance normalization
 * 
 * Instance normalization is like batch normalization, but we don't average over all the images.
 * Find per-channel mean and variance
 * out = (in - mean) / sqrt(variance + variance_epsilon)
 * 
 */

#include <math.h>
#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef __hexagon__
#include "hexagon_types.h"
#else
#include <malloc.h>
#endif
#include "hvx_hexagon_protos.h"
#include <qf16.h>
#include <vqf16.h>
#include <qftables.h>

#define MANT 0
#define EXP  1

#if 0
static void qprint(HVX_Vector *vmant, HVX_Vector * vexp)
{
      int i;
      float m ;
      short *mant = (short *) vmant;
      short *exp = (short *) vexp;
      printf(" qfloat \n");
      for(i=0; i < 32; i++)
      {
            m = ((float) mant[i]) / pow(2.0, 15+exp[i]);
            printf("%d) %04x*2 ^ %d <-> %2.32f\n",i,mant[i],exp[i], m);
      }
      return;
}
#endif

struct instancenormbg_info {
        nn_sem_t doneanalyze;    //
        nn_sem_t donequantize;   //
        float out_min;           // pointer to min values, enough storage per thread...
        float out_max;           // pointer to max values, enough storage per thread...
        uint8_t * in_data;       // pointer to left of top
        uint8_t * out_data;      // pointer to left of top
        struct workitem *work_items; // All the work items to execute at execute time
        int n_work_items;        // how many work items?
        float out_minval;        // Minimum output value, either specified or guessed
        float out_maxval;        // maximum output value, either specified or guessed
        float * depth_out_min;   //
        float * depth_out_max;   //
        int minval_precalculated;// Is the minval precalculated?
        int maxval_precalculated;// Is the maxval precalculated?
        float out_minval_spec;   // exact value specified (when not precalculated)
        float out_maxval_spec;   // exact value specified (when not precalculated)
        int32_t needs_retry;     // Do we need to try this op over again?
        int32_t strategy_valid;  // Do we believe the strategy is currently valid?
        short * rsd;             // 1/sqrt variance stdev
        short * mean;            // mu - mean of depth data
        float * gammaf_buf;      // float gamma value 
        float * betaf_buf;       // float beta added 
        short * gamma_buf;       // qfloat gamma value 
        short * beta_buf;        // qfloat beta added 
        int     bg_depth;        // depth32 of the beta and gamma
        float   epsilon;         // threshold for flat signal
        int     pad_val;         // value to zap the boarders with 
        int     sum_fixup;       // volume of pad
        int32_t height;          // height of the input
        int32_t width;           // input width to compute
        int32_t area;            // product of raw width * height
        int32_t left_skip;       // amount of left margin to avoid
        int32_t depth;           // input depth to compute
        int32_t depth_main;      // input depth before padding
        int32_t depth_iters;     // input depth multiples of 32 to compute
        int32_t next_d32;        // distance from one depth32 slice to the next on the same row
        int32_t next_row;        // distance from one row to the next
        int32_t batch_size;      // distance from one batch to the next
        int32_t recip_val;       // reciprocal for float -> quant output space
        int32_t recursion_depth; // how far have we recursed?
        int32_t max_valid_val;   // maximum value that results in a value not above max_out
        int32_t min_valid_val;   // minimum value that results in a value not below min_out
        uint64_t cycles;         // Cycle accumulator for children
};


/*
 *  method for finding mean & variance of a bunch of u8 values:
 *   - find their sum as u32
 *   - find the sums of square,in u64
 *   - variance is ( pop* ssq -   sum*sum) / (pop^2)
 *   - mean is sum/pop
 */

#if 0
static void renorm_ref(uint8_t * in_vec, 
                int width,
                int stride,
                int height,
                uint8_t * out_vec,
                short *qmean,
                short *qrsd)
{
      //y = (in * rsd_mant * 2^(rsd_exp - mean_exp)  - mean_mant)  * 2^mean_exp 
       int i,j, d;
            for(d=0; d < 32; d++) {
               struct qf16 qr, qu;
               qr.e = qrsd[d+64];
               if(qr.e < 0) printf(" qrsd > 1 %d %d\n",d,qr.e); 
               qu.e = qmean[d+64];
               if(qu.e > 0) printf(" mean < 1 %d %d\n",d,qu.e); 
            }
       for(j=0; j < height; j++) {
        for (i = 0; i < width; i+=32) {
            for(d=0; d < 32; d++) {
               struct qf16 qr, qu; //, in, pr;
               short res;
               qr.m = qrsd[d];
               qr.e = qrsd[d+64];
               qr.e += 5;
               qu.m = -qmean[d];
               qu.e = qmean[d+64];
               qu.e -= 2;

               res = -128*in_vec[stride*j + i + d];
               short p;
               qu.m = qu.m >>(15+qu.e);
               //p = (qr.m * res) >> (qr.e; p = p - qu.m >> (qu.e+15);
               p = ((qr.m * res + 0x00004000) >> 15);
               if(qr.e < 0)
                 p = p << (-qr.e);
               else
                 p = p >> qr.e;
               p = qu.m - p;
               res = p>>2;
               if(res > 255) res = 255;
               if(res < 0) res = 0;

               out_vec[stride*j + i + d] = res;
            }
         }
       }
      return;
}
#endif
#if !defined(__hexagon__)
static void getstats_ref(uint8_t * in_vec,
              int width, int stride, int height,
              short * qsum,
              short * qsum2,
              short * qmax,
              short * qmin)
{
       int h, w, d, tmp;
       int32_t sum[32] ;
       uint64_t sum2[32] ;
       int max[32];
       int min[32];
       struct qf16 s, s2, mn, mx;

                /* Try to keep data in fixed point as long as possible */
                for (d = 0; d < 32; d++) { sum[d] = sum2[d] = max[d] = 0; min[d] = 255;}
                for (h = 0; h < height; h++) {
                        for (w = 0; w < width; w+=32) {
                                for (d = 0; d < 32; d++) {
                                        tmp = in_vec[h*stride + w + d];

                                        sum[d] += tmp;         //23bits
                                        sum2[d] += (uint64_t) tmp*tmp;    //30bits

                                        if(max[d] < tmp) max[d] = tmp;
                                        if(min[d] > tmp) min[d] = tmp;
                                }

                        }
                }
                for (d = 0; d < 32; d++) {
                     i2q(sum[d], &s);
                     qsum[d   ] = qsum[d+32] = s.m;
                     qsum[d+64] = qsum[d+96] = s.e;
                     uli2q(sum2[d], &s2);             //convert 64bit unsigned accumulator to qf
                     qsum2[d   ] = qsum2[d+32] = s2.m;
                     qsum2[d+64] = qsum2[d+96] = s2.e;
                     i2q(max[d], &mx);
                     qmax[d   ] = qmax[d+32] = mx.m;
                     qmax[d+64] = qmax[d+96] = mx.e;
                     i2q(min[d], &mn);
                     qmin[d   ] = qmin[d+32] = mn.m;
                     qmin[d+64] = qmin[d+96] = mn.e;
                     printf("min[%d] = %d  ", d,min[d]); printf("max[%d] = %d\n", d,max[d]);
                }

  return;
}
#endif


static int instancenormbg_ref_execute(struct nn_node *self,struct nn_graph *nn)
{
        struct instancenormbg_info *info = self->opaque;
	const struct tensor *in_tensor = self->inputs[0];
        //const struct tensor *in_min_tensor = self->inputs[1]; not used
        //const struct tensor *in_max_tensor = self->inputs[2]; not used

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

        int32_t left_pad = in_tensor->format.width_pad[0];
        int32_t right_pad = in_tensor->format.width_pad[1];
        int32_t top_pad = in_tensor->format.height_pad[0];
        int32_t bottom_pad = in_tensor->format.height_pad[1];

        logmsg(nn, 2, "padding l-r-t-b %ld %ld %ld %ld", left_pad, right_pad, top_pad, bottom_pad);

	int32_t batches = in_tensor->shape.batches;
	int32_t width = in_tensor->shape.width;
	int32_t height = in_tensor->shape.height;
	int32_t depth_main = in_tensor->shape.depth;
        int32_t depth_before_pad = in_tensor->format.depth_pad[0];
        int32_t depth_after_pad = in_tensor->format.depth_pad[1];

        int32_t depth = depth_before_pad + depth_main + depth_after_pad;
        int32_t width_total = width + left_pad + right_pad;
        int32_t height_total = height + top_pad + bottom_pad;

        int64_t wh = width*height;

        int32_t depth_iters = depth/32;
        int32_t batch_size = depth*width_total*height_total;

        if (info->bg_depth != depth) return errlog(nn,"depths dont match");
        if (depth_iters <= 0) return errlog(nn,"no out depth to iterate?");
        if (((width_total) % 4) != 0) return errlog(nn,"width fail");
        if ((depth % 32) != 0) return errlog(nn,"depth fail");

	int scratch_needed = 2*sizeof(float)*depth*batches; 
	if( nn->scratch_size < scratch_needed ){
		return errlog(nn, "needed %d bytes of scratch", scratch_needed);
	}
        float * mean = (float *) nn->scratch;
        float * recipsd = mean + depth*batches;

        if (tensor_out_prepare_padded_d32(out_tensor,
                batches,
                height,top_pad,bottom_pad,
                width,left_pad,right_pad,
                depth_main,depth_before_pad,depth_after_pad,
                NN_TYPE_QUINT8) != 0) {
                return errlog(nn,"output tensor prep fail");
        }
        logmsg(nn,2," before depth after %d-%d-%d = %d", depth_before_pad,depth_main,depth_after_pad, depth);
        logmsg(nn,2," width height depth %d-%d-%d %d ", left_pad,width,right_pad, height);

	const uint8_t * in_data = tensor_location_d32(in_tensor,0,0,0,0);
	uint8_t * out_data = tensor_location_d32(out_tensor,0,0,0,0);

        int32_t next_d32 = width_total * 32;
        int32_t next_row = width_total * depth;

	//
	// first pass finds scaling for each depth index, each batch;
	// these are kept in 'mean' and  'invsqrt_variance'
	// in the process, finds full range of output.
	//
        float out_min = 999999.f;
        float out_max =-999999.f; 
        int b, h, w, i, d;
        int64_t tmp2, tmp;
        float ival, oval;
        int32_t sum[32] ;
        uint64_t sum2[32] ;
        int32_t max[32];
        int32_t min[32];
        int32_t w32 = 32*width; //next_d32 - 32*left_pad;
        logmsg(nn,2,"area = %lld", wh);
	for (b = 0; b < batches; b++)
        {
	    for (i = 0; i < depth_iters; i++) {

                /* Try to keep data in fixed point as long as possible */
                for (d = 0; d < 32; d++) { sum[d] = sum2[d] = max[d] = 0; min[d] = 255;}
                for (h = 0; h < height; h++) {
                     for (w = 0; w < w32; w+=32) {
                         for (d = 0; d < 32; d++) {
                             tmp = in_data[b*batch_size + h*next_row + i*next_d32 + w + d];

                             sum[d] += tmp;         //32bits
                             sum2[d] += (uint64_t) tmp*tmp;    //32+bits

                             if(max[d] < tmp) max[d] = tmp;
                             if(min[d] > tmp) min[d] = tmp;
                         }
                     }
                }
                for (d = 0; d < 32; d++) { 
                     float u, rsd, var, lmin,lmax;
                     tmp = sum[d];
                     u = (float) tmp / (float) wh;
                     tmp2 = sum2[d]*wh - tmp*tmp; 
                     var = (float) tmp2 ;
                     logmsg(nn,2,"depth %d) N*var %f mean %f ", d+32*i,var, u);
                     logmsg(nn,2,"epsilon*wh^2 = %f", info->epsilon*info->epsilon*(float) wh * (float) wh);

                     if (var < info->epsilon*info->epsilon*(float) wh * (float) wh ) 
                         rsd = 0.f;
                     else
                         rsd= (float) wh /sqrtf(var) ;

                     logmsg(nn,2,"depth %d)mean / 1/sd %f %f", d+32*i,u,rsd);
                     logmsg(nn,2,"gamma/beta =  %f %f", info->gammaf_buf[32*i+d], info->betaf_buf[32*i+d]);

                     rsd = rsd * info->gammaf_buf[32*i+d];
                     recipsd[b*depth + 32*i+d] = rsd;
                     u = u*rsd - info->betaf_buf[32*i+d];
                     mean[b*depth + 32*i+d] = u;
                     logmsg(nn,2,"depth %d) adjusted mean / rsd %f %f", d+32*i,u,rsd);

                     lmin = ((float)min[d])*rsd - u;
                     lmax = ((float)max[d])*rsd - u;
                     logmsg(nn,2,"depth %d) %d ->lmin %f %d ->lmax %f", d+32*i,min[d],lmin, max[d],lmax);
                     out_min = (lmin < out_min) ? lmin : out_min;
                     out_max = (lmax > out_max) ? lmax : out_max;

                     out_min = (lmax < out_min) ? lmax : out_min;
                     out_max = (lmin > out_max) ? lmin : out_max;
                }
	    }//depth iters
	} // b
        logmsg(nn,1,"global max = %f global min = %f", out_max, out_min); 

        float recip_val = 255.f/(out_max - out_min);
        float zero_val = recip_val*out_min;
        logmsg(nn,2,"recip val = %f", recip_val);
        for (b = 0; b < batches; b++)
        {
	    for (i = 0; i < depth_iters; i++) {
                for (d = 0; d < 32; d++) { 
                   recipsd[b*depth + 32*i+d] *= recip_val;
                   mean[b*depth + 32*i+d] =  mean[b*depth + 32*i+d] * recip_val + zero_val ;
                   logmsg(nn, 2,"final rsd %f mn %f",recipsd[b*depth + 32*i+d], mean[b*depth + 32*i+d]);
                }
                for (h = 0; h < height; h++) {
                    for (w = 0; w < w32; w+=32) {
                        for (d = 0; d < 32; d++) {
                            ival = in_data[b*batch_size + h*next_row + i*next_d32 + w + d];
                            oval = recipsd[b*depth + 32*i+d]*ival - mean[b*depth + 32*i+d] ;

                            if(oval < 0.f) oval = 0.f; 
                            if(oval > 255.f) oval = 255.f; 
                            out_data[b*batch_size + h*next_row + i*next_d32 + w + d] = (uint8_t) oval;
                                       
                        }//d
                    }//w
                }//h
            }//i
        }//b
	tensor_set_single_float( out_min_tensor, out_min );
	tensor_set_single_float( out_max_tensor, out_max );
	return 0;
}
/* ----------------------------------------------------------------------------------------- */
int instancenormbg_analyze(struct nn_graph *nn, void * vinfo, int depth_num, int batch_num)
{
        struct instancenormbg_info *info = vinfo;
        int wh = info->area;
        struct qf16 qwh, qwh_inv, qout_min, qout_max, fixup1, fixup2;
        float epsi = wh * info->epsilon;
        epsi = epsi * epsi;
        struct qf16 qepsi = f2q(epsi);
        int32_t depth_main = info->depth_main & 31;
        HVX_Vector qepsi_mant = Q6_V_vsplat_R(Q6_R_combine_RlRl(qepsi.m, qepsi.m));
        HVX_Vector qepsi_exp = Q6_V_vsplat_R(Q6_R_combine_RlRl(qepsi.e, qepsi.e));
        HVX_Vector qf1_mant, qf1_exp, qf2_mant, qf2_exp;

        HVX_Vector *in_vec = (HVX_Vector *) &info->in_data[depth_num*info->next_d32 + batch_num*info->batch_size];
      //each qfloat val is short + short and replicated for 32 depths so 1 vector can process 2 depth chunks at once
        HVX_Vector * qrsd  = (HVX_Vector *) &info->rsd[(batch_num * info->depth + 32*depth_num)*4];
        HVX_Vector * qmean = (HVX_Vector *) &info->mean[(batch_num * info->depth + 32*depth_num)*4];

        HVX_Vector * qgamma = (HVX_Vector *) &info->gamma_buf[(32*depth_num)*4];
        HVX_Vector * qbeta = (HVX_Vector *)  &info->beta_buf[(32*depth_num)*4];

        HVX_Vector qmax_mant, qmax_exp, qmin_mant, qmin_exp;
        HVX_Vector qmaxmax_mant, qmaxmax_exp, qminmin_mant, qminmin_exp;
        HVX_Vector qsum2_mant, qsum2_exp, qsum_mant, qsum_exp, qmean_mant, qmean_exp;
        HVX_Vector qisum_mant, qisum_exp, qrsd_mant, qrsd_exp, qvar_mant, qvar_exp;
        HVX_Vector zero = Q6_V_vsplat_R(0x00000000);
        HVX_Vector qsum[2];
        HVX_Vector qsum2[2];
        HVX_Vector qmax[2];
        HVX_Vector qmin[2];
        HVX_VectorPred q0, q1;

        //filter out padded depth positions 0/0 issue
        q0 = Q6_Q_vsetq_R(2*32);                  //^^^^^^^^^^^^^|_____________
        if(32*depth_num == (info->depth-32) && depth_main != 0) { //last depth
           q1 = Q6_Q_vsetq_R(2*depth_main+2*32);  //^^^^^^^^^^^^^^^^^^^|_______
           q1 = Q6_Q_and_QQn(q1, q0);             //_____________|^^^^^|_______
           q0 = Q6_Q_vsetq_R(2*depth_main);       //^^^^^|_____________________
           q0 = Q6_Q_or_QQ(q0, q1);               //^^^^^|_______|^^^^^|_______
           q0 = Q6_Q_not_Q(q0);                   //_____|^^^^^^^|_____|^^^^^^^
        } else {
           q0 = Q6_Q_and_QQn(q0, q0);             //___________________________
        }

        logmsg(nn,1," batch %d depth %d analyze ", batch_num, depth_num);
        logmsg(nn,1," depth_out_min: %p@%p depth_out_max: %p@%p",
                      info->depth_out_min, &info->depth_out_min,info->depth_out_max,&info->depth_out_max);

        qwh = f2q((float) wh);
        qwh_inv = f2q(1.f/(float) wh);

        logmsg(nn,1," next_d32 = %d %d next_row = %d height = %d",
                      info->next_d32, info->left_skip, info->next_row, info->height);
#if defined(__hexagon__)
        getstats_asm(in_vec,
                     info->next_d32-32*info->left_skip,
                     info->next_row,
                     info->height,
                     qsum,
                     qsum2,
                     qmax,
                     qmin);
#else
        getstats_ref((uint8_t *)in_vec,
                     info->next_d32-32*info->left_skip,
                     info->next_row,
                     info->height,
                     (short *)qsum,
                     (short *)qsum2,
                     (short *)qmax,
                     (short *)qmin);
#endif

        //subtract off the value of the padding
        i2q(info->pad_val * info->sum_fixup, &fixup2);
        i2q(info->sum_fixup, &fixup1);
        qf2_mant= Q6_V_vsplat_R(Q6_R_combine_RlRl(fixup2.m,fixup2.m));
        qf2_exp = Q6_V_vsplat_R(Q6_R_combine_RlRl(fixup2.e,fixup2.e));
        qf1_mant= Q6_V_vsplat_R(Q6_R_combine_RlRl(fixup1.m,fixup1.m));
        qf1_exp = Q6_V_vsplat_R(Q6_R_combine_RlRl(fixup1.e,fixup1.e));
        vsubqf(qsum2[MANT], qsum2[EXP], qf2_mant, qf2_exp, &qsum2_mant, &qsum2_exp);
        vsubqf(qsum[MANT], qsum[EXP], qf1_mant, qf1_exp, &qsum_mant, &qsum_exp);

        vsmpyqf(qsum_mant, qsum_exp, qwh_inv.m,  qwh_inv.e, &qmean_mant, &qmean_exp); //mean =sum / wh
        vsmpyqf(qsum2_mant, qsum2_exp, qwh.m, qwh.e,  &qvar_mant,  &qvar_exp);  //sum2*N
        vmpyqf(qsum_mant, qsum_exp, qsum_mant, qsum_exp, &qsum_mant, &qsum_exp); //sum * sum
        vsubqf(qvar_mant, qvar_exp, qsum_mant, qsum_exp, &qisum_mant, &qisum_exp);

        //if the N*sum(x^2) - sum(x)^2 < epsilon^2 * N^2 then clamp recip sd to 0
        q1 = vcmpgtqf(qepsi_mant, qepsi_exp, qisum_mant, qisum_exp);
        q1 = Q6_Q_or_QQ(q1, q0);                                             //eliminate out of sample depths
        visqrt64_asm(&qisum_mant, &qisum_exp, &qrsd_mant, &qrsd_exp, lut_isqrt_asm);    //1/sqrt(^)
        vsmpyqf(qrsd_mant,  qrsd_exp, qwh.m, qwh.e, &qrsd_mant,  &qrsd_exp);  //N/sqrt()

        qrsd_mant = Q6_V_vmux_QVV(q1, zero, qrsd_mant);
        qrsd_exp = Q6_V_vmux_QVV(q1, zero, qrsd_exp);

        vmpyqf(qgamma[MANT], qgamma[EXP], qrsd_mant, qrsd_exp, &qrsd_mant, &qrsd_exp); //rsigma  = gamma*rsigma
        vmpyqf(qmean_mant, qmean_exp, qrsd_mant,  qrsd_exp,  &qmean_mant, &qmean_exp);  //mean*= qrsd
        vsubqf(qmean_mant, qmean_exp, qbeta[MANT], qbeta[EXP], &qmean_mant, &qmean_exp); //mean*gamma*rsd - beta

        qrsd[MANT] = qrsd_mant;
        qrsd[EXP]  = qrsd_exp;
        qmean[MANT] = qmean_mant;
        qmean[EXP]  = qmean_exp;

        vmpyqf(qmax[MANT], qmax[EXP], qrsd_mant,  qrsd_exp,  &qmax_mant, &qmax_exp);    //*qrsd
        vsubqf(qmax_mant,  qmax_exp,  qmean_mant, qmean_exp, &qmax_mant, &qmax_exp);    //max-mean

        vmpyqf(qmin[MANT], qmin[EXP], qrsd_mant,  qrsd_exp,  &qmin_mant, &qmin_exp);    //*qrsd
        vsubqf(qmin_mant,  qmin_exp,  qmean_mant, qmean_exp, &qmin_mant, &qmin_exp);    //min-mean

        vminqf(qmax_mant, qmax_exp, qmin_mant, qmin_exp, &qminmin_mant, &qminmin_exp);
        vminrqf(qminmin_mant, qminmin_exp, &qminmin_mant, &qminmin_exp);
        qout_min.m = *((short*) &qminmin_mant);
        qout_min.e = *((short*) &qminmin_exp);
        q2f(qout_min.m, qout_min.e, &info->depth_out_min[depth_num]);

        vmaxqf(qmax_mant, qmax_exp, qmin_mant, qmin_exp, &qmaxmax_mant, &qmaxmax_exp);
        vmaxrqf(qmaxmax_mant, qmaxmax_exp, &qmaxmax_mant, &qmaxmax_exp);
        qout_max.m = *((short*) &qmaxmax_mant);
        qout_max.e = *((short*) &qmaxmax_exp);
        q2f(qout_max.m, qout_max.e, &info->depth_out_max[depth_num]);

        logmsg(nn,1," min %f max %f", info->depth_out_min[depth_num], info->depth_out_max[depth_num]);

        return 0;
}

int instancenormbg_requantize(struct nn_graph *nn, void * vinfo, int depth_num, int batch_num)
{
       const struct instancenormbg_info *info = vinfo;

       HVX_Vector *in_vec  = (HVX_Vector *) &info->in_data[depth_num*info->next_d32 + batch_num*info->batch_size];
       HVX_Vector *out_vec = (HVX_Vector *) &info->out_data[depth_num*info->next_d32 + batch_num*info->batch_size];
       HVX_Vector * qrsd  = (HVX_Vector *) &info->rsd[(batch_num * info->depth + 32*depth_num)*4];
       HVX_Vector * qmean = (HVX_Vector *) &info->mean[(batch_num * info->depth + 32*depth_num)*4];

       int next_width = info->next_d32/32;
       int height = info->height;
       int width = next_width - info->left_skip;

       struct qf16 qresize_amt ;
       struct qf16 qout_min;

       HVX_Vector qmin_mant, qmin_exp;
       HVX_Vector qmean_mant, qmean_exp;
       HVX_Vector qrsd_mant, qrsd_exp;

       logmsg(nn,1," batch %d depth %d requantize ", batch_num, depth_num);

       qresize_amt = f2q(info->recip_val);
       qout_min = f2q(info->out_min * info->recip_val);

       qmin_mant = Q6_V_vsplat_R(Q6_R_combine_RlRl(qout_min.m, qout_min.m));
       qmin_exp  = Q6_V_vsplat_R(Q6_R_combine_RlRl(qout_min.e, qout_min.e));

       qmean_mant = qmean[MANT];
       qmean_exp = qmean[EXP];
       qrsd_mant = qrsd[MANT];
       qrsd_exp = qrsd[EXP];

       vsmpyqf(qrsd_mant, qrsd_exp, qresize_amt.m, qresize_amt.e, &qrsd_mant, &qrsd_exp); //qrsd*255/(max-min)
       vsmpyqf(qmean_mant, qmean_exp, qresize_amt.m, qresize_amt.e, &qmean_mant, &qmean_exp); //qmean*255/(max-min)
       vaddqf (qmean_mant, qmean_exp, qmin_mant, qmin_exp, &qmean_mant, &qmean_exp);

       qmean[MANT] = qmean_mant;
       qmean[EXP]  = qmean_exp;
       qrsd[MANT] = qrsd_mant;
       qrsd[EXP]  = qrsd_exp;
#if 1
       //fastest asm 
       renorm_asm(in_vec,
                  32*width,
                  info->next_row,
                  height,
                  out_vec,
                  qmean,
                  qrsd);
#else
       renorm_ref((uint8_t *)in_vec,
                  32*width,
                  info->next_row,
                  height,
                  (uint8_t *)out_vec,
                  (short *)qmean,
                  (short *)qrsd);
#endif

       return 0;
}

/* ----------------------------------------------------------------------------------------- */
static int instancenormbg_hvx_execute(struct nn_graph *nn, void *vself)
{
        struct nn_node *self = vself;
        struct instancenormbg_info *info = self->opaque;
        const struct tensor *in_tensor = self->inputs[0];
        const struct tensor *epsilon_tensor = self->inputs[3];
        struct tensor *out_tensor = self->outputs[0];
        struct tensor *out_min_tensor = self->outputs[1];
        struct tensor *out_max_tensor = self->outputs[2];

        int32_t left_pad = in_tensor->format.width_pad[0];
        int32_t right_pad = in_tensor->format.width_pad[1];
        int32_t top_pad = in_tensor->format.height_pad[0];
        int32_t bottom_pad = in_tensor->format.height_pad[1];

        int32_t batches = in_tensor->shape.batches;
        int32_t width = in_tensor->shape.width;
        int32_t height = in_tensor->shape.height;
        int32_t depth_main = in_tensor->shape.depth;
        int32_t depth_before_pad = in_tensor->format.depth_pad[0];
        int32_t depth_after_pad = in_tensor->format.depth_pad[1];
        info->depth_main = depth_main;

        int32_t depth = depth_before_pad + depth_main + depth_after_pad;
        int32_t width_total = width + left_pad + right_pad;
        int32_t height_total = height + top_pad + bottom_pad;

        int32_t depth_iters = depth/32;
        int32_t batch_size = depth*width_total*height_total;
        int b, d;
        uint8_t *zap_bot, *zap_left, *zap_right;
        logmsg(nn,1,"hvx instance norm %x: ",self->node_id);

        info->area = width * height;
        if (depth_iters <= 0) return errlog(nn,"no out depth to iterate?");
        if (((width_total) % 4) != 0) return errlog(nn,"width fail");
        if ((depth % 32) != 0) return errlog(nn,"depth fail");
        logmsg(nn,2," before depth after %d-%d-%d %d %d", depth_before_pad,depth_main,depth_after_pad);

        // work area:
        //   sum:    struct integer_acc * [depth]
        //    mean, invsqrt_variance:   each float * [batches*depth]
        //   out_scale,out_offs: each int32_t[depth]

        int scratch_needed = 2*2*sizeof(short)*depth*batches + 32 + 2*sizeof(float)*depth_iters;
        if( nn->scratch_size < scratch_needed ){
                return errlog(nn, "needed %d bytes of scratch", scratch_needed);
        }
        info->mean = (short *) nn->scratch;
        info->rsd = info->mean + 4*depth*batches;
        info->depth_out_min = (float *) &info->rsd[4*depth*batches];
        info->depth_out_max = info->depth_out_min + depth_iters + 8;
        logmsg(nn,2,"info: %p mean: %p rsd: %p depth_out_min: %p@%p depth_out_max: %p@%p",
                     info,info->mean,info->rsd,info->depth_out_min,
                     &info->depth_out_min,info->depth_out_max,&info->depth_out_max);

        info->epsilon = tensor_get_float(epsilon_tensor,0);

        if (tensor_out_prepare_padded_d32(out_tensor,
                batches,
                height,top_pad,bottom_pad,
                width,left_pad,right_pad,
                depth_main,depth_before_pad,depth_after_pad,
                NN_TYPE_QUINT8) != 0) {
                return errlog(nn,"output tensor prep fail");
        }
        logmsg(nn,2," width height depth %d-%d-%d %d %d", left_pad,width,right_pad, height, depth);

        //if left pad is 4 skip over it
        if(left_pad == 4) {
                info->in_data = tensor_location_d32(in_tensor,0,0,0,0);
                info->out_data = tensor_location_d32(out_tensor,0,0,0,0);
                info->left_skip = 4;
                logmsg(nn, 2, "skipping left margin");
        } else {
                info->in_data = tensor_location_d32(in_tensor,0,0,-left_pad,0);
                info->out_data = tensor_location_d32(out_tensor,0,0,-left_pad,0);
                info->left_skip = 0;
        }

        info->width = width;
        info->depth = depth;
        info->next_d32 = width_total * 32;
        info->next_row = width_total * depth;
        info->depth_iters = depth_iters;
        info->batch_size = batch_size;

        if(left_pad == 4) {
          zap_left = NULL;
          zap_right = info->in_data+width*32;
        } else {
          zap_left = info->in_data;
          zap_right = info->in_data+(left_pad+width)*32;
        }

        //make activations height multiple of 2 valid
        if(height % 2) {
                info->height = height + 1;
                zap_bot = tensor_location_d32(in_tensor,0,height,-left_pad,0);
        } else {
                info->height = height;
                zap_bot = NULL;
        }
        logmsg(nn, 2, "padded height = %d depth = %d", info->height, info->depth);

        //
        // first pass finds scaling for each depth index, each batch;
        // these are kept in 'mean' and  'invsqrt_variance'
        // in the process, finds full range of output.
        //
        logmsg(nn, 1, "beginning hvx");
        info->out_min = 9999.f;
        info->out_max = -9999.f;
        for (b = 0; b < batches; b++)
        {
            info->sum_fixup = 0;
            logmsg(nn, 1, "batch number %d",b);
            if(zap_left != NULL) {
                logmsg(nn,2,"zapping left %p %d",zap_left, left_pad*32);
                //zapout(zap_left,  left_pad*32,  height*depth_iters, info->next_d32);
                padzap_part(zap_left,info->pad_val,info->next_d32,depth_iters,info->next_row,height,left_pad);
                zap_left += batch_size;
                info->sum_fixup += left_pad*info->pad_val*height;
            }
            if(right_pad > 0) {
                logmsg(nn,2,"zapping right %p %d",zap_right, right_pad*32);
                //zapout(zap_right, right_pad*32, height*depth_iters, info->next_d32);
                padzap_part(zap_right,info->pad_val,info->next_d32,depth_iters,info->next_row,height,right_pad);
                zap_right += batch_size;
                info->sum_fixup += right_pad*info->pad_val*height;
            }
            if(zap_bot != NULL) {
                logmsg(nn,2,"zapping bottom row %p %d",zap_bot, info->next_row);
                //memset(zap_bot,info->pad_val,info->next_row);
                vmemset_asm(zap_bot,info->pad_val,info->next_row);
                if(left_pad == 4)
                    info->sum_fixup += info->pad_val*((info->next_d32/32)-left_pad);
                else
                    info->sum_fixup += info->pad_val*(info->next_d32/32);
                zap_bot += batch_size;
            }
            logmsg(nn,1," done zapping");

            for (d = 0; d < depth_iters; d++) {
                instancenormbg_analyze(nn, info, d, b);
            }
            for(d=0; d < depth_iters; d++)
            {
                if(info->depth_out_min[d] < info->out_min ) { info->out_min = info->depth_out_min[d]; }
                if(info->depth_out_max[d] > info->out_max ) { info->out_max = info->depth_out_max[d]; }
            }
        } // b
        logmsg(nn,1,"global max = %f global min = %f", info->out_max, info->out_min);
        /* deal with the case of all flat, zero sd */
        if(info->out_max > 0) {
          info->recip_val = 255.f/(info->out_max - info->out_min);
        } else {
          logmsg(nn,1,"squashing output due to zero sd");
          info->out_max = 0.f;
          info->out_min = 0.f;
          info->recip_val = 0.f;
        }
        logmsg(nn,2,"recip val = %f", info->recip_val);

        for (b = 0; b < batches; b++) {
                for(d=0; d < depth_iters; d++)
                {
                        instancenormbg_requantize(nn, info, d, b);
                }
        }//b

        tensor_set_single_float( out_min_tensor, info->out_min );
        tensor_set_single_float( out_max_tensor, info->out_max );
        logmsg(nn,2,"done instance norm hvx");
        return 0;
}

static int instancenormbg_hvx_execute_wrapper(struct nn_node *self, struct nn_graph *nn)
{
        return nn_os_vector_call(nn,instancenormbg_hvx_execute,self);
}

/* ----------------------------------------------------------------------------------------- */
/*
   functional check, allocate data
 */

static int instancenormbg_hvx_check(struct nn_node *self, struct nn_graph *nn)
{
        //if (self->n_inputs !=10) return errlog(nn,"wrong # inputs there are %d should be 10", self->n_inputs);
        //if (self->n_outputs != 3) return errlog(nn,"wrong # outputs there are %d should be 3", self->n_inputs);

        struct instancenormbg_info *info;
        if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
                return errlog(nn,"calloc info");
        }
        self->opaque = info;

        const struct tensor *epsilon_tensor = self->inputs[3];
        float epsilon = tensor_get_float(epsilon_tensor,0);
        info->epsilon = epsilon;
        info->pad_val = 0x80;

        const struct tensor *gamma_tensor = self->inputs[4]; // expecting vector, length matching input's depth 
        const struct tensor *gamma_min_tensor = self->inputs[5];
        const struct tensor *gamma_max_tensor = self->inputs[6];
        float gmin = tensor_get_float(gamma_min_tensor,0);
        float gmax = tensor_get_float(gamma_max_tensor,0);
        float grecip = (gmax - gmin)/255.0;
        const uint8_t * gamma = gamma_tensor->data;

        const struct tensor *beta_tensor = self->inputs[7]; // expecting vector, length matching input's depth 
        const struct tensor *beta_min_tensor = self->inputs[8];
        const struct tensor *beta_max_tensor = self->inputs[9];
        float bmin = tensor_get_float(beta_min_tensor,0);
        float bmax = tensor_get_float(beta_max_tensor,0);
        float brecip = (bmax - bmin)/255.0;
        const uint8_t * beta = beta_tensor->data;
        struct qf16 gval, bval;
        int d, k;
        if (gamma_tensor->shape.depth != beta_tensor->shape.depth) return errlog(nn,"gamma beta depths don't match");
        info->bg_depth = (beta_tensor->shape.depth + 31) & ~31 ;
        logmsg(nn,2," beta and gamma detph = %d",info->bg_depth);

        logmsg(nn, 3, "generating float gamma and beta ");
        if ((info->gamma_buf = nn_memalign(128,4*sizeof(short)*info->bg_depth)) == NULL) {
                nn_free(info);
                return errlog(nn,"malloc/memalign");
        }
        if ((info->beta_buf = nn_memalign(128,4*sizeof(short)*info->bg_depth)) == NULL) {
                nn_free(info->gamma_buf);
                nn_free(info);
                return errlog(nn,"malloc/memalign");
        }
        for(d = 0; d < info->bg_depth; d+=32)
        {
            for(k=0; k < 32; k++) 
            {
              if( (d+k)< gamma_tensor->shape.depth ) {
                gval = f2q(gamma[d+k] * grecip + gmin); 
                bval = f2q(beta[d+k] * brecip + bmin);
              } else {
                gval = f2q(0.f); 
                bval = gval;
              }
              info->gamma_buf[4*d+ 0+k] = gval.m;
              info->gamma_buf[4*d+32+k] = gval.m;
              info->gamma_buf[4*d+64+k] = gval.e;
              info->gamma_buf[4*d+96+k] = gval.e;
              info->beta_buf[4*d+ 0+k] = bval.m;
              info->beta_buf[4*d+32+k] = bval.m;
              info->beta_buf[4*d+64+k] = bval.e;
              info->beta_buf[4*d+96+k] = bval.e;
            }
        }
        logmsg(nn, 2, "done check");

        return 0;
}

static int instancenormbg_ref_check(struct nn_node *self, struct nn_graph *nn)
{
	//if (self->n_inputs !=10) return errlog(nn,"wrong # inputs");
	//if (self->n_outputs != 3) return errlog(nn,"wrong # outputs");

        struct instancenormbg_info *info;
        if ((info = nn_calloc(1,sizeof(*info))) == NULL) {
                return errlog(nn,"calloc info");
        }
        self->opaque = info;

	const struct tensor *epsilon_tensor = self->inputs[3];
	float epsilon = tensor_get_float(epsilon_tensor,0);
        info->epsilon = epsilon;
        logmsg(nn,2,"epsilon = %f ",epsilon);

        const struct tensor *gamma_tensor = self->inputs[4]; // expecting vector, length matching input's depth 
        const struct tensor *gamma_min_tensor = self->inputs[5]; 
        const struct tensor *gamma_max_tensor = self->inputs[6]; 
	float gmin = tensor_get_float(gamma_min_tensor,0);
	float gmax = tensor_get_float(gamma_max_tensor,0);
        float grecip = (gmax - gmin)/255.f;
        const uint8_t * gamma = gamma_tensor->data;

        const struct tensor *beta_tensor = self->inputs[7]; // expecting vector, length matching input's depth 
        const struct tensor *beta_min_tensor = self->inputs[8]; 
        const struct tensor *beta_max_tensor = self->inputs[9];
	float bmin = tensor_get_float(beta_min_tensor,0);
	float bmax = tensor_get_float(beta_max_tensor,0);
        float brecip = (bmax - bmin)/255.f;
        const uint8_t * beta = beta_tensor->data;
        logmsg(nn,2,"grecip = %f gmin = %f",grecip, gmin);
        logmsg(nn,2,"brecip = %f bmin = %f",brecip, bmin);
        int d;
	if (gamma_tensor->shape.depth != beta_tensor->shape.depth) return errlog(nn,"gamma beta depths don't match");
        info->bg_depth = (beta_tensor->shape.depth + 31) & ~31 ;

        logmsg(nn, 3, "generating float gamma and beta ");
        if ((info->gammaf_buf = nn_memalign(128,sizeof(float)*info->bg_depth)) == NULL) {
                nn_free(info);
                return errlog(nn,"malloc/memalign");
        }
        if ((info->betaf_buf = nn_memalign(128,sizeof(float)*info->bg_depth)) == NULL) {
                nn_free(info->gammaf_buf);
                nn_free(info);
                return errlog(nn,"malloc/memalign");
        }
        for(d=0; d < gamma_tensor->shape.depth; d++)
        {
            info->gammaf_buf[d] = (float)gamma[d] * grecip + gmin; 
            info->betaf_buf[d] = (float)beta[d] * brecip + bmin; 
        }
        for(d= gamma_tensor->shape.depth; d < info->bg_depth; d++)
        {
            info->gammaf_buf[d] = 0.f; 
            info->betaf_buf[d] = 0.f; 
        }
        logmsg(nn, 2, "done check");

        return 0;
}
/*
         destructors
 */

static int instancenormbg_hvx_dtor(struct nn_node *self, struct nn_graph *nn)
{
        struct instancenormbg_info *info = self->opaque;
        if (info != NULL) {
                nn_free(info->gamma_buf);
                nn_free(info->beta_buf);
                nn_free(info);
        }
        self->opaque = NULL;
        return node_free_common(self,nn);
}
static int instancenormbg_ref_dtor(struct nn_node *self, struct nn_graph *nn)
{
        struct instancenormbg_info *info = self->opaque;
        if (info != NULL) {
                //nn_free(info->gammaf_buf);
                //nn_free(info->betaf_buf);
                nn_free(info);
        }
        self->opaque = NULL;
        return node_free_common(self,nn);
}

/*
         function definitions
 */

struct nn_node_ops nn_ops_for_QuantizedInstanceNormBG_8_ref = {
	.execute = instancenormbg_ref_execute,
	.check = instancenormbg_ref_check,
	.ctor = node_alloc_common,
        .dtor = instancenormbg_ref_dtor,
        .n_inputs = NN_IOCOUNT(10),
        .n_outputs = NN_IOCOUNT(3),
        .flags = NN_NODE_FLAG_D32_OUTPUT,
};
struct nn_node_ops nn_ops_for_QuantizedInstanceNormBG_8 = {
	.execute = instancenormbg_hvx_execute_wrapper,
	.check = instancenormbg_hvx_check,
	.ctor = node_alloc_common,
        .dtor = instancenormbg_hvx_dtor,
        .n_inputs = NN_IOCOUNT(10),
        .n_outputs = NN_IOCOUNT(3),
        .flags = NN_NODE_FLAG_D32_OUTPUT,
};
struct nn_node_ops nn_ops_for_QuantizedInstanceNormBG_8_d32 = {
	.execute = instancenormbg_hvx_execute_wrapper,
	.check = instancenormbg_hvx_check,
	.ctor = node_alloc_common,
        .dtor = instancenormbg_hvx_dtor,
        .n_inputs = NN_IOCOUNT(10),
        .n_outputs = NN_IOCOUNT(3),
        .flags = NN_NODE_FLAG_D32_OUTPUT,
};
struct nn_node_ops nn_ops_for_QuantizedInstanceNormBG_8_d32_ref = {
	.execute = instancenormbg_ref_execute,
	.check = instancenormbg_ref_check,
	.ctor = node_alloc_common,
        .dtor = instancenormbg_ref_dtor,
        .n_inputs = NN_IOCOUNT(10),
        .n_outputs = NN_IOCOUNT(3),
        .flags = NN_NODE_FLAG_D32_OUTPUT,
};
struct nn_node_ops nn_ops_for_QuantizedInstanceNormBG_f = {
	.execute = instancenormbg_ref_execute,
	.check = instancenormbg_ref_check,
	.ctor = node_alloc_common,
        .dtor = instancenormbg_ref_dtor,
        .n_inputs = NN_IOCOUNT(10),
        .n_outputs = NN_IOCOUNT(3),
        .flags = NN_NODE_FLAG_D32_OUTPUT,
};
