/*
 * Copyright (c) 2017-2018, The Linux Foundation. All rights reserved.
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
       Vector versions of qf16 functions, these are HVX intrinsic based and will
       inline and optimize.
 */
#ifndef VQF16
#define VQF16

#ifdef __hexagon__
#include <hexagon_types.h>
#endif
#include "hvx_hexagon_protos.h"
#include <stdarg.h>
/*======================================================================*/
#if 0
void s2q(short ptr_x, struct qf16 *qy)
{
  short x, u, mant, expnt, y, range;

        u = ptr_x;
        expnt = norm16(u);
        qy->m = u <<expnt;
        qy->e = expnt - 15;
}
#endif
/*======================================================================*/
static inline void vs2q(HVX_Vector * indata, HVX_Vector *a_mant0, HVX_Vector *a_exp0,
                                      HVX_Vector *a_mant1, HVX_Vector *a_exp1)
{
 HVX_VectorPair mant1_mant0;
 HVX_Vector     mant1,mant0;
 HVX_Vector     vc15 = Q6_V_vsplat_R(0x000f000f);
 HVX_Vector     d127d000, exp0, exp1;

                d127d000 = *indata; 

                mant1_mant0 = Q6_Wuh_vunpack_Vub(d127d000);

                mant0 = Q6_V_lo_W(mant1_mant0);
                mant1 = Q6_V_hi_W(mant1_mant0);
                exp0= Q6_Vh_vnormamt_Vh(mant0);
                exp1= Q6_Vh_vnormamt_Vh(mant1);

                *a_mant0 = Q6_Vh_vasl_VhVh(mant0, exp0);
                *a_mant1 = Q6_Vh_vasl_VhVh(mant1, exp1);

                *a_exp0 = Q6_Vh_vsub_VhVh(exp0, vc15);
                *a_exp1 = Q6_Vh_vsub_VhVh(exp1, vc15);

       return;
}
/*======================================================================*/
#if 0
void qround(short xm, short xe, unsigned char *z)
{
  short x, u, mant, expnt, y, range;

        short rnd = 2 << (14+qy.e);
        y = (qy.m + rnd) >> (15+qy.e);

        if(y > 255) z = 255; else if (y < 0) z = 0; else z = (unsigned char ) y  
}
#endif
static inline void vqround(HVX_Vector a_mant0, HVX_Vector a_exp0, HVX_Vector a_mant1, HVX_Vector a_exp1,
                    HVX_Vector * outdata)
{
 HVX_Vector     vc15 = Q6_V_vsplat_R(0x000f000f);
 HVX_Vector     vc1  = Q6_V_vsplat_R(0x00010001);
 HVX_Vector     d127d000, rnd0, rnd1, exp3, exp2;
 HVX_VectorPred q0, q1;

                a_exp0 = Q6_Vh_vadd_VhVh(vc15, a_exp0);
                a_exp1 = Q6_Vh_vadd_VhVh(vc15, a_exp1);
                q0 = Q6_Q_vcmp_gt_VhVh(a_exp0, vc15);   //> 15?
                q1 = Q6_Q_vcmp_gt_VhVh(a_exp1, vc15);   //> 15?
                a_exp0 = Q6_V_vmux_QVV(q0, vc15, a_exp0); //clip to 15
                a_exp1 = Q6_V_vmux_QVV(q1, vc15, a_exp1); //clip to 15

                exp2 = Q6_Vh_vsub_VhVh(a_exp0, vc1);
                exp3 = Q6_Vh_vsub_VhVh(a_exp1, vc1);
                
                rnd0 = Q6_Vh_vasl_VhVh(vc1, exp2);
                rnd1 = Q6_Vh_vasl_VhVh(vc1, exp3);
                a_mant0 = Q6_Vh_vadd_VhVh_sat(a_mant0, rnd0);
                a_mant1 = Q6_Vh_vadd_VhVh_sat(a_mant1, rnd1);
                a_mant0 = Q6_Vh_vasr_VhVh(a_mant0, a_exp0);
                a_mant1 = Q6_Vh_vasr_VhVh(a_mant1, a_exp1);

                d127d000 = Q6_Vub_vpack_VhVh_sat(a_mant1, a_mant0);
                *outdata = d127d000;
       return;
}
/*======================================================================*/
#if 0
struct qf16 addqf(struct qf16 a, struct qf16 b)
{           
       struct qf16 c;
        short big, delta, dx, dy, rx, ry;
        short x = a.m;
        short y = b.m;
        if(a.e > b.e) {
           dx = a.e-b.e;
           dy = 0;
           big = b.e ;
        } else {
           dx = 0;
           dy = b.e-a.e;
           big = a.e ;
        }
        x = x >> dx;
        y = y >> dy;

        x = (x+y+1)>>1;
        delta = norm16(x);
        x = x << delta;
        c.m = x;
        c.e = big+delta-1;
        return(c);
}
#endif
/*======================================================================*/
static inline void vaddqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector b_mant, HVX_Vector b_exp,
            HVX_Vector * cm, HVX_Vector * ce)
{
 HVX_VectorPred  q0, q1;
 HVX_VectorPair dy_dx;
 HVX_Vector     diff, big, delta;
 HVX_Vector     ones = Q6_V_vsplat_R(0x00010001);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(ones, ones);
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
                q0 = Q6_Q_vcmp_gt_VhVh(a_exp, b_exp);

                diff = Q6_Vuh_vabsdiff_VhVh(a_exp, b_exp);
                q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
                diff = Q6_V_vmux_QVV(q1, maxshift, diff);

                dy_dx = Q6_W_vswap_QVV(q0, diff, zero);
  
                big = Q6_V_vmux_QVV(q0, b_exp, a_exp);

                big = Q6_Vh_vsub_VhVh(big, ones);

                a_mant = Q6_Vh_vasr_VhVh(a_mant, Q6_V_lo_W(dy_dx));
                b_mant = Q6_Vh_vasr_VhVh(b_mant, Q6_V_hi_W(dy_dx));

                a_mant = Q6_Vh_vavg_VhVh_rnd(a_mant, b_mant);

                delta = Q6_Vh_vnormamt_Vh(a_mant);

                big = Q6_Vh_vadd_VhVh(big, delta);

                a_mant = Q6_Vh_vasl_VhVh(a_mant, delta);
                *cm = a_mant;
                *ce = big;
       return;
}
static inline void vsaddqf(HVX_Vector a_mant, HVX_Vector a_exp, short bm, short be,
            HVX_Vector * cm, HVX_Vector * ce)
{
 HVX_VectorPred  q0,q1;
 HVX_VectorPair dy_dx;
 HVX_Vector     diff, big, delta;
 HVX_Vector     ones = Q6_V_vsplat_R(0x00010001);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(ones, ones);
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
 int            bebe  =  Q6_R_combine_RlRl(be, be);
 int            bmbm  =  Q6_R_combine_RlRl(bm, bm);
 HVX_Vector     b_mant = Q6_V_vsplat_R(bmbm);
 HVX_Vector     b_exp = Q6_V_vsplat_R(bebe);

                q0 = Q6_Q_vcmp_gt_VhVh(a_exp, b_exp);

                diff = Q6_Vuh_vabsdiff_VhVh(a_exp, b_exp);
                q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
                diff = Q6_V_vmux_QVV(q1, maxshift, diff);
                dy_dx = Q6_W_vswap_QVV(q0, diff, zero);

                big = Q6_V_vmux_QVV(q0, b_exp, a_exp);

                big = Q6_Vh_vsub_VhVh(big, ones);

                a_mant = Q6_Vh_vasr_VhVh(a_mant, Q6_V_lo_W(dy_dx));
                b_mant = Q6_Vh_vasr_VhVh(b_mant, Q6_V_hi_W(dy_dx));

                a_mant = Q6_Vh_vavg_VhVh_rnd(a_mant, b_mant);

                delta = Q6_Vh_vnormamt_Vh(a_mant);

                big = Q6_Vh_vadd_VhVh(big, delta);
                //q0 = Q6_Q_vcmp_gt_VhVh(big, maxshift);          //if greater its basically 0
                //a_mant = Q6_V_vmux_QVV(q0, zero, a_mant);

                a_mant = Q6_Vh_vasl_VhVh(a_mant, delta);

                *cm = a_mant;
                *ce = big;
       return;
}

#if 0
struct qf16 subqf(struct qf16 a, struct qf16 b)
{           
       struct qf16 c;
        short big, delta, dx, dy, rx, ry;
        short x = a.m;
        short y = b.m;
        if(a.e > b.e) {
           dx = a.e-b.e;
           dy = 0;
           big = b.e ;
        } else {
           dx = 0;
           dy = b.e-a.e;
           big = a.e ;
        }
        x = x >> dx;
        y = y >> dy;

        x = (x-y+1)>>1;
        delta = norm16(x);
        x = x << delta;
        c.m = x;
        c.e = big+delta-1;
        return(c);
}
#endif
/*======================================================================*/
static inline void vsubqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector b_mant, HVX_Vector b_exp,
            HVX_Vector * cm, HVX_Vector * ce)
{
 HVX_VectorPred  q0, q1;
 HVX_VectorPair dy_dx;
 HVX_Vector     diff, big, delta;
 HVX_Vector     ones = Q6_V_vsplat_R(0x00010001);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(ones, ones);
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
 //HVX_Vector     neg1 =  Q6_V_vsplat_R(0xffffffff);
                q0 = Q6_Q_vcmp_gt_VhVh(a_exp, b_exp);

                diff = Q6_Vuh_vabsdiff_VhVh(a_exp, b_exp);
                q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
                diff = Q6_V_vmux_QVV(q1, maxshift, diff);
                dy_dx = Q6_W_vswap_QVV(q0, diff, zero);
  
                big = Q6_V_vmux_QVV(q0, b_exp, a_exp);

                big = Q6_Vh_vsub_VhVh(big, ones);

                a_mant = Q6_Vh_vasr_VhVh(a_mant, Q6_V_lo_W(dy_dx));
                b_mant = Q6_Vh_vasr_VhVh(b_mant, Q6_V_hi_W(dy_dx));

                a_mant = Q6_Vh_vnavg_VhVh(a_mant, b_mant);

                delta = Q6_Vh_vnormamt_Vh(a_mant);
                big = Q6_Vh_vadd_VhVh(big, delta);       //new exp

                *ce = big;

                //q1 = Q6_Q_vcmp_eq_VhVh(a_mant, neg1);
                //delta = Q6_V_vmux_QVV(q1, zero, delta); //if all ones dont shift left

                //q1 = Q6_Q_vcmp_eq_VhVh(a_mant, zero);
                //delta = Q6_V_vmux_QVV(q1, zero, delta); //if all ones dont shift left

                *cm = Q6_Vh_vasl_VhVh(a_mant, delta);
       return;
}

/*======================================================================*/
#if 0
struct qf16 mpyqf(struct qf16 a, struct qf16 b)
{
    struct qf16 c;
    int delta;
    c.e = a.e + b.e;
    c.m = mpyrsat(a.m, b.m);
    delta = norm16(c.m);
    c.m = c.m << delta;
    c.e += delta;
    return(c);
}
#endif
/*======================================================================*/
static inline void vsmpyqf(HVX_Vector a_mant, HVX_Vector a_exp, short bm, short be, HVX_Vector *c_mant, HVX_Vector *c_exp)
{
 HVX_Vector   delta;
 int bebe  =  Q6_R_combine_RlRl(be, be);
 int bmbm  =  Q6_R_combine_RlRl(bm, bm);
 HVX_Vector   b_exp = Q6_V_vsplat_R(bebe);

                a_mant = Q6_Vh_vmpy_VhRh_s1_rnd_sat(a_mant, bmbm);
                a_exp= Q6_Vh_vadd_VhVh(a_exp, b_exp);

                delta = Q6_Vh_vnormamt_Vh(a_mant);

                *c_mant = Q6_Vh_vasl_VhVh(a_mant, delta);
                *c_exp= Q6_Vh_vadd_VhVh(a_exp, delta);

                return;
}
/*======================================================================*/
#if 0
struct qf16 mpyqf(struct qf16 a, struct qf16 b)
{
    struct qf16 c;
    int delta;
    c.e = a.e + b.e;
    c.m = mpyrsat(a.m, b.m);
    delta = norm16(c.m);
    c.m = c.m << delta;
    c.e += delta;
    return(c);
}
#endif
/*======================================================================*/
static inline void vmpyqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector b_mant, HVX_Vector b_exp,
            HVX_Vector * cm, HVX_Vector * ce)
{
 HVX_Vector     delta;
 HVX_Vector     c_exp, c_mant;

                a_mant = Q6_Vh_vmpy_VhVh_s1_rnd_sat(a_mant, b_mant);
                a_exp= Q6_Vh_vadd_VhVh(a_exp, b_exp);

                delta = Q6_Vh_vnormamt_Vh(a_mant);

                c_mant = Q6_Vh_vasl_VhVh(a_mant, delta);
                c_exp= Q6_Vh_vadd_VhVh(a_exp, delta);

                *cm = c_mant;
                *ce = c_exp;
       return;
}

/* ================================================================ */
#if 0
struct qf16 maxqf(struct qf16 a, struct qf16 b)
{
        struct qf16 c;
        short dx, dy, big, delta;
        short x = a.m;
        short y = b.m;
        if(b.e > a.e) {
           dx = a.e-b.e;
           dy = 0;
        } else {
           dx = 0;
           dy = a.e-b.e;
        }
        x = x >> dx;
        y = y >> dy;
        if(x > y)  c = a;
        else       c = b;
        return(c);
}
#endif
/* ================================================================ */
static inline void vmaxqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector b_mant, HVX_Vector b_exp,
                   HVX_Vector  *c_mant, HVX_Vector *c_exp)
{
 HVX_VectorPred  q0, q1;
 HVX_VectorPair dy_dx;
 HVX_Vector     diff, x_mant, y_mant;
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(maxshift, maxshift);
       //flush 0 to true 0
       q0 = Q6_Q_vcmp_eq_VhVh(a_mant, zero);
       a_exp = Q6_V_vmux_QVV(q0, maxshift, a_exp);
       q1 = Q6_Q_vcmp_eq_VhVh(b_mant, zero);
       b_exp = Q6_V_vmux_QVV(q1, maxshift, b_exp);

       q0 = Q6_Q_vcmp_gt_VhVh(b_exp, a_exp);
       diff = Q6_Vuh_vabsdiff_VhVh(a_exp, b_exp);

       q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
       diff = Q6_V_vmux_QVV(q1, maxshift, diff);

       dy_dx = Q6_W_vswap_QVV(q0, zero, diff);

       x_mant = Q6_Vh_vasr_VhVh(a_mant, Q6_V_lo_W(dy_dx));

       y_mant = Q6_Vh_vasr_VhVh(b_mant, Q6_V_hi_W(dy_dx));

       q1 = Q6_Q_vcmp_gt_VhVh(x_mant, y_mant);

       *c_mant = Q6_V_vmux_QVV(q1, a_mant, b_mant);
       *c_exp = Q6_V_vmux_QVV(q1, a_exp, b_exp);
       return;
}
static inline HVX_VectorPred vcmpgtqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector b_mant, HVX_Vector b_exp)
{
 HVX_VectorPred  q0, q1;
 HVX_VectorPair dy_dx;
 HVX_Vector     diff, x_mant, y_mant;
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(maxshift, maxshift);

       //flush 0 to true 0
       q0 = Q6_Q_vcmp_eq_VhVh(a_mant, zero);
       a_exp = Q6_V_vmux_QVV(q0, maxshift, a_exp);
       q1 = Q6_Q_vcmp_eq_VhVh(b_mant, zero);
       b_exp = Q6_V_vmux_QVV(q1, maxshift, b_exp);
       
       q0 = Q6_Q_vcmp_gt_VhVh(b_exp, a_exp);
       diff = Q6_Vuh_vabsdiff_VhVh(a_exp, b_exp);

       q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
       diff = Q6_V_vmux_QVV(q1, maxshift, diff);

       dy_dx = Q6_W_vswap_QVV(q0, zero, diff);

       x_mant = Q6_Vh_vasr_VhVh(a_mant, Q6_V_lo_W(dy_dx));

       y_mant = Q6_Vh_vasr_VhVh(b_mant, Q6_V_hi_W(dy_dx));

       q1 = Q6_Q_vcmp_gt_VhVh(x_mant, y_mant);
       return(q1);
}
/* ================================================================ */
#if 0
struct qf16 minqf(struct qf16 a, struct qf16 b)
{
        struct qf16 c;
        short dx, dy, big, delta;
        short x = a.m;
        short y = b.m;
        if(b.e > a.e) {
           dx = 0;
           dy = b.e-a.e;
        } else {
           dx = a.e-b.e;
           dy = 0;
        }
        x = x >> dx;
        y = y >> dy;
        if(x < y)  c = a;
        else       c = b;
        return(c);
}
#endif
/* ================================================================ */
static inline void vminqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector b_mant, HVX_Vector b_exp,
                   HVX_Vector * c_mant, HVX_Vector * c_exp)
{
 HVX_VectorPred  q0, q1;
 HVX_VectorPair dy_dx;
 HVX_Vector     diff, x_mant, y_mant;
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(maxshift, maxshift);
                //flush 0 to true 0
                q0 = Q6_Q_vcmp_eq_VhVh(a_mant, zero);
                a_exp = Q6_V_vmux_QVV(q0, maxshift, a_exp);
                q1 = Q6_Q_vcmp_eq_VhVh(b_mant, zero);
                b_exp = Q6_V_vmux_QVV(q1, maxshift, b_exp);

                q0 = Q6_Q_vcmp_gt_VhVh(b_exp, a_exp);
                diff = Q6_Vuh_vabsdiff_VhVh(a_exp, b_exp);

                q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
                diff = Q6_V_vmux_QVV(q1, maxshift, diff);

                dy_dx = Q6_W_vswap_QVV(q0, zero, diff);
                x_mant = Q6_Vh_vasr_VhVh(a_mant, Q6_V_lo_W(dy_dx));
                y_mant = Q6_Vh_vasr_VhVh(b_mant, Q6_V_hi_W(dy_dx));
                q1 = Q6_Q_vcmp_gt_VhVh(y_mant, x_mant);
                *c_mant = Q6_V_vmux_QVV(q1, a_mant, b_mant);
                *c_exp = Q6_V_vmux_QVV(q1, a_exp, b_exp);
       return;
}
/* ================================================================ */
//   Reduction versions
/* ================================================================ */

static inline void vminrqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector * cm, HVX_Vector * ce)
{
 HVX_VectorPred  q0, q1;
 HVX_VectorPair dy_dx;
 HVX_VectorPair a_manto_a_mante;
 HVX_VectorPair a_expo_a_expe;
 HVX_Vector     a_manto,a_mante;
 HVX_Vector     a_expo,a_expe;
 HVX_Vector     diff, x_mant, y_mant;
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(maxshift, maxshift);

 int i, c4 =-2;
       //flush 0 to true 0
       q0 = Q6_Q_vcmp_eq_VhVh(a_mant, zero);
       a_exp = Q6_V_vmux_QVV(q0, maxshift, a_exp);

       for(i=0; i < 6; i++)
       { 
                a_manto_a_mante = Q6_W_vshuff_VVR(a_mant, a_mant, c4);
                a_expo_a_expe = Q6_W_vshuff_VVR(a_exp, a_exp, c4);

                a_mante = Q6_V_lo_W(a_manto_a_mante);
                a_manto = Q6_V_hi_W(a_manto_a_mante);
                a_expe = Q6_V_lo_W(a_expo_a_expe);
                a_expo = Q6_V_hi_W(a_expo_a_expe);

                q0 = Q6_Q_vcmp_gt_VhVh(a_expo, a_expe);

                diff = Q6_Vuh_vabsdiff_VhVh(a_expo, a_expe);
                q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
                diff = Q6_V_vmux_QVV(q1, maxshift, diff);

                dy_dx = Q6_W_vswap_QVV(q0, zero, diff);

                x_mant = Q6_Vh_vasr_VhVh(a_mante, Q6_V_lo_W(dy_dx));
                y_mant = Q6_Vh_vasr_VhVh(a_manto, Q6_V_hi_W(dy_dx));

                q1 = Q6_Q_vcmp_gt_VhVh(y_mant, x_mant);
                a_mant = Q6_V_vmux_QVV(q1, a_mante, a_manto);
                a_exp = Q6_V_vmux_QVV(q1, a_expe, a_expo);
                c4 = c4 + c4;
       }
       *cm = a_mant;
       *ce = a_exp;
       return;
}
static inline void vmaxrqf(HVX_Vector a_mant, HVX_Vector a_exp, HVX_Vector * cm, HVX_Vector * ce)
{
 HVX_VectorPred  q0, q1;
 HVX_VectorPair dy_dx;
 HVX_VectorPair a_manto_a_mante;
 HVX_Vector     a_manto,a_mante;
 HVX_VectorPair a_expo_a_expe;
 HVX_Vector     a_expo,a_expe;
 HVX_Vector     diff, x_mant, y_mant;
 HVX_Vector     maxshift = Q6_V_vsplat_R(0x000F000F);
 HVX_Vector     zero = Q6_Vh_vsub_VhVh(maxshift, maxshift);
 int i, c4 =-2;
       //flush 0 to true 0
       q0 = Q6_Q_vcmp_eq_VhVh(a_mant, zero);
       a_exp = Q6_V_vmux_QVV(q0, maxshift, a_exp);

       for(i=0; i < 6; i++)
       { 
                a_manto_a_mante = Q6_W_vshuff_VVR(a_mant, a_mant, c4);
                a_expo_a_expe = Q6_W_vshuff_VVR(a_exp, a_exp, c4);

                a_mante = Q6_V_lo_W(a_manto_a_mante);
                a_manto = Q6_V_hi_W(a_manto_a_mante);
                a_expe = Q6_V_lo_W(a_expo_a_expe);
                a_expo = Q6_V_hi_W(a_expo_a_expe);

                q0 = Q6_Q_vcmp_gt_VhVh(a_expo, a_expe);

                diff = Q6_Vuh_vabsdiff_VhVh(a_expe, a_expo);

                q1 = Q6_Q_vcmp_gt_VhVh(diff, maxshift);
                diff = Q6_V_vmux_QVV(q1, maxshift, diff);

                dy_dx = Q6_W_vswap_QVV(q0, zero, diff);

                x_mant = Q6_Vh_vasr_VhVh(a_mante, Q6_V_lo_W(dy_dx));
                y_mant = Q6_Vh_vasr_VhVh(a_manto, Q6_V_hi_W(dy_dx));

                q1 = Q6_Q_vcmp_gt_VhVh(x_mant, y_mant);
                a_mant = Q6_V_vmux_QVV(q1, a_mante, a_manto);
                a_exp = Q6_V_vmux_QVV(q1, a_expe, a_expo);
                c4 = c4 + c4;
       }
       *cm = a_mant;
       *ce = a_exp;
       return;
}

void visqrt64_asm(HVX_Vector * am, HVX_Vector * ae, HVX_Vector * cm, HVX_Vector * ce, const short * lut);

#endif

/*======================================================================*/
/*                       end of file                                    */
/*======================================================================*/
