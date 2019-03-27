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

 /* assorted functions for pseudo float scalar ops using type 
     struct qf16 { short m; short e; };
  */

#include <stdio.h>
#include <hexagon_protos.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif
#include <math.h>
#include <qftables.h>
//extern const short int lut_inv_cn[];
//extern const short lut_isqrt_cn[];
#include <qf16.h>

//saturating fractional multiply
short mpyrsat(short x, short y) { return(Q6_R_mpy_RlRl_s1_rnd_sat(x, y)>>16); }
short norm16(short x) { return(Q6_R_normamt_R((int)x)-16); }
short norm32(int x) { return(Q6_R_normamt_R(x)); }
short cl032(int x) { return(Q6_R_cl0_R(x)); }

void inverse16_cn(short ptr_x, short expin, struct qf16 *qy)
{
  short u, x, mant, expnt, y, range;

        u = ptr_x;
        if(u==0)u = 1;
        expnt = norm16(u);
        mant = u << expnt;

        range = (mant >> 11) & 0x7;
        range|= (mant >> 12) & 0x8;
        x = (mant<<1) & 0xffe;
        expnt = expnt + 15 + expin;

        y =                 lut_inv_cn[3*range+2];
        y = mpyrsat(y, x) + lut_inv_cn[3*range+1];
        y = mpyrsat(y, x) + lut_inv_cn[3*range+0];
        expnt = 13-expnt;
        qy->e = expnt;  
        qy->m = y;
  return;
}
void uli2q(uint64_t x, struct qf16 *qy)
{
  short expnt;
  uint32_t u;
  uint32_t xhi, xlo;

        xlo = (uint32_t) (x & 0x00000000ffffffffLL);
        xhi = (uint32_t) ((x>>32) & 0x00000000ffffffffLL);

        if(xhi == 0)
          u = xlo;
        else
          u = xhi;
       
        expnt = cl032(u);
        u = u << expnt;
        xlo = xlo >> (32-expnt);
        expnt = expnt - 32;

        if(xhi != 0) 
          expnt = expnt - 32;
        else
          xlo = 0;
        
        u = u | xlo;
        qy->m = u >> (1+16); 
        qy->e = expnt;
}
void i2q(int x, struct qf16 *qy)
{
  short expnt;
  int u;

        u = x;
        expnt = norm32(u);
        u = u << expnt;
        if(u==0) expnt = 31;
        qy->m = u >> 16; 
        qy->e = expnt - 31;
}
void s2q(short x, struct qf16 *qy)
{
  short u, expnt;

        u = x;
        expnt = norm16(u);
        qy->m = u <<expnt; 
        qy->e = expnt - 15;
}
void q2s(struct qf16 qy, short *y)
{
        *y = qy.m >> (15+qy.e); 
        return;
}
void qround(struct qf16 qy, short *y)
{
        short rnd = 1 << (15+qy.e);
        *y = (qy.m + rnd) >> (15+qy.e); 
        return;
}

void inverse32_cn(int ptr_x, struct qf16 *qy)
{
  short x, mant, expnt, y, range;
  int u;

        u = ptr_x;
        if(u==0)u = 1;
        expnt = norm32(u);
        u = u << expnt;
        mant = u >> 16; 
        expnt -= 16;

        range = (mant >> 11) & 0x7;
        range|= (mant >> 12) & 0x8;
        x = (mant<<1) & 0xffe;

        y =                 lut_inv_cn[3*range+2];
        y = mpyrsat(y, x) + lut_inv_cn[3*range+1];
        y = mpyrsat(y, x) + lut_inv_cn[3*range+0];
        expnt = 13-expnt;
        qy->e = expnt;  
        qy->m = y;
  return;
}

void isqrt_cn(short x, short expin, struct qf16 *qy)
{
  short mant, expnt, y, y2, range, isqrt2 = 0x5a83;

        if(x<=0) {
          x = -x; 
        }
        expnt = norm16(x);
        mant = x << expnt;
        expnt = expin + expnt; 

        range = (mant >> 10) & 0xf;
        x = (mant<<1) & 0x7ff;

        y =                 lut_isqrt_cn[range+2*16];
        y = mpyrsat(y, x) + lut_isqrt_cn[range+1*16];
        y = mpyrsat(y, x) + lut_isqrt_cn[range+0*16];
        y2 = mpyrsat(isqrt2, y);
        if((expnt & 1)) y = y2;
        expnt = -expnt;
        qy->e = (expnt>>1)-1;
        qy->m = y;
  return;
}

void printq(struct qf16 x)
{
        float v = (float) x.m;
        v = ldexpf( v, -(x.e+15));	// x.m/ 2^(x.e+15)
        printf(" qfp %04X . 2^ %d=  %2.9f\n", x.m, x.e, v);
        return;
}
void printme(short m, short e)
{
        float v = (float) m;
        v = ldexpf( v, -(e+15));	// m/2^(e+15)
        printf(" qfp =  %2.9f\n", v);
        return;
}

float q2f(short m, short e, float *z)
{
        float v = (float) m ;
        v = ldexpf( v, -(e+15));	// *v/2^(e+15)
        *z = v;
        return(v);
}

typedef union {
   float f;
   int   i;
} flint;

struct qf16 f2q(float a) {
        flint ia;
        ia.f = a;
        int exp;
        int mant;
        struct qf16 c;

        exp = 126 - ((ia.i >> 23) & 0xFF) ;
        mant = ((ia.i << 7) | 0x40000000) & 0x7FFFFF80;
        if (ia.i & 0x80000000) mant = -mant;

        mant = (mant + (1<<15)) >> 16;
        c.m = (short) mant;
        c.e = (short) exp;

        return(c);
 }



struct qf16 addqf(struct qf16 a, struct qf16 b)
{
       struct qf16 c;
        short big, delta, dx, dy;
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
        big = big -1;

        x = Q6_R_vavgh_RR_rnd(x,y);
        delta = norm16(x);
        x = x << delta;
        c.m = x;
        c.e = big+delta;
        return(c);
}

struct qf16 subqf(struct qf16 a, struct qf16 b)
{
        struct qf16 c;
        short big, delta, dx, dy;
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
        big = big -1;
        
        x = Q6_R_vnavgh_RR(x,y);
        delta = norm16(x);
        x = x << delta;
        c.m = x;
        c.e = big+delta;
        return(c);
}

struct qf16 maxqf(struct qf16 a, struct qf16 b)
{
        struct qf16 c;
        short dx, dy;
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
        if(x > y)  c = a;
        else       c = b;
        return(c);
}
struct qf16 minqf(struct qf16 a, struct qf16 b)
{
        struct qf16 c;
        short dx, dy;
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

/*======================================================================*/
/*                              End of File                             */
/*======================================================================*/
