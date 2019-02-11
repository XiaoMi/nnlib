

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

#if defined(__hexagon__)
#include "hexagon_types.h"

#define cl0(x)          (Q6_R_cl0_R(Q6_R_zxth_R((x)))-16)         
#define mpyrsat(x,y)    (Q6_R_asrh_R(Q6_R_mpy_RlRl_s1_rnd_sat((x),(y))))

#else
static short cl0(unsigned short x) {
    int l0;
    if(x == 0) return(0);
    for(l0=0; l0<16; l0++) { 
        if((x<<l0) & 0x08000) break;
    }
    return  l0;
}
static short sat_16(long long int x) {
  short y;
  if(x > 32767) y = 32767; else if(x <-32768) y =-32768; else y = (short) x;
  return(y);
}
static short mpyrsat(short x, short y)
{
    long long int a = (long long int)x;
    long long int b = (long long int)y;
    b = (a*b + 0x4000LL)>>15;
    return(sat_16(b));
}
#endif

//*===========================================================*/
const int16_t lut_sqrt_cn[] = {
    23170,    11582,    -2767,    
    23884,    11237,    -2533,    
    24576,    10920,    -2330,    
    25249,    10629,    -2153,    
    25905,    10360,    -1998,    
    26545,    10111,    -1860,    
    27170,    9879,     -1737,    
    27780,    9662,     -1627,    
    28378,    9458,     -1529,    
    28963,    9267,     -1440,    
    29537,    9087,     -1359,    
    30099,    8917,     -1286,    
    30652,    8757,     -1218,    
    31194,    8605,     -1157,    
    31727,    8460,     -1101,    
    32251,    8322,     -1049    
};

uint8_t sqrt_fx(uint16_t u)
{
    int16_t expnt, deltax, y, y2, range;
    int16_t isqrt2 = 23170;

    if(u==0) return 0;

    expnt = cl0(u);
    u = u << expnt;

    range = (u >> 11) & 15;
    deltax = u & 0x7ff;

    y =                      lut_sqrt_cn[3*range+2];
    y = mpyrsat(y, deltax) + lut_sqrt_cn[3*range+1];
    y = mpyrsat(y, deltax) + lut_sqrt_cn[3*range+0];

    y2 = mpyrsat(isqrt2, y);

    if(expnt & 1) y = y2;

    y >>= (expnt>>1);

    y = (y + (1<<6)) >> 7;
    if (y > 255) y = 255;

    return  (uint8_t)y;
}

