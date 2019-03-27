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
/*
 */
/*======================================================================*/
/*  FUNCTIONS      : l2fetch driver                                     */
/*                                                                      */
/*  DESCRIPTION                                                         */
/*                 Make l2fetch instruciton a little easier to use      */
/*                 Fetch a block of data height * width with stride     */
/*  ARCHITECTURE   : QDSP6V6  + HVX                                     */
/*======================================================================*/
/*  REVISION HISTORY:                                                   */
/*  =================                                                   */
/*                                                                      */
/*  Author              Date           Comments                         */
/*  -------------------------------------------------------------       */
/*  DJH                 03/07/16       created                          */
/*======================================================================*/
        .text
        .global l2pref
        .type  l2pref, @function
/*======================================================================*/
#define ptr        r0
#define height     r1
#define width      r2
#define stride     r3
#define prehprel   r3:2
#define preh       r3
#define prel       r2
/*======================================================================*/
                .balign 32
l2pref:
/*======================================================================*/
          {
              preh = zxth(stride)
              prel = combine(width.L, height.L)
          } {
              l2fetch(ptr, prehprel)    
          } {
              jumpr r31
          }
/*======================================================================*/
.L_end:
/*======================================================================*/
      .size l2pref, .L_end-l2pref
