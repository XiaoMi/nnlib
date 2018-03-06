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
 *  FUNCTIONS      : gvrmaxmin
 *                                          
 *  DESCRIPTION                            
 *    find max/min of one vector 
 *                                       
 *  ARCHITECTURE   : QDSP6V60  + HVX     
 *
 *  REVISION HISTORY:                                    
 *  =================                                   
 *                                                     
 *  Author              Date           Comments       
 *  ------------------------------------------------------------- 
 *
 *  CYCLE-COUNT:                              
 *                                              
 *  MEMORY                                       
 *     CODESIZE =    bytes                       
 *     STACK    =    bytes                         
 *     ASSUMPTIONS                                  
 *        
 *        
 *  C MODEL                                             
 */
/*=============================================================================*/
        .text
        .file "vrmaxmin_h.S"
        .global gvrmaxmin
        .balign 32
        .type  gvrmaxmin, @function
gvrmaxmin:
/*=============================================================================*/
#define ptr_max                r0     //data
#define nrot                   r1     //constant
/*=============================================================================*/
#define max                    v0     //
#define max_t                  v1     //
#define min                    v2     //
#define min_t                  v3     //
/*=============================================================================*/
       {   min = vmem(ptr_max+#1)                       //
           nrot = #128/2                                //
           loop0(.L_peak, #4)                           //
           nop                                          //
       } {   
           max.cur = vmem(ptr_max+#0)                   //
           max_t = vror(max,nrot)                       //[0, 0]
           nop; nop                                     //
       } 
.L_peak:
       {   min_t = vror(min,nrot)                       //[0, 1]
           max.w = vmax(max.w, max_t.w)                 //[0, 1]
       } {
           nrot = lsr(nrot,#1)                          //[0, 2]
           min.w = vmin(min.w, min_t.w)                 //[0, 2]
       } {
           max_t = vror(max,nrot)                       //[1, 0]
       }:endloop0 

       {   min_t = vror(min,nrot)                       //[1, 1]
           max.w = vmax(max.w, max_t.w)                 //[1, 1]
           vmem(ptr_max+#0) = max.new                   //[E, 1]
       } {
           min.w = vmin(min.w, min_t.w)                 //[1, 2]
           vmem(ptr_max+#1) = min.new                   //[E, 2]
       }
/*=============================================================================*/
       {   jumpr r31                                    //
       } 
.L_end:
/*=============================================================================*/
      .size gvrmaxmin, .L_end-gvrmaxmin
/*=============================================================================*/
