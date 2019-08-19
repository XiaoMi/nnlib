
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
#ifndef NN_GRAPH_MEMCPY_H
#define NN_GRAPH_MEMCPY_H 1

#define WEIGHT_COPY_BLOCK (1024*16)

static inline void nn_graph_hvx_blockcopy(void *out0,const void *in0, int size)
{
    int b;
    uint8_t const *in = (uint8_t const *)in0;
    uint8_t *out = (uint8_t *)out0;
    int block = Q6_R_min_RR(size, WEIGHT_COPY_BLOCK);
    l2fetch(in, 128, 128, block>>7);
    for (b = 0; b < size; b+= WEIGHT_COPY_BLOCK) {
        int next_block = Q6_R_min_RR(size-b-block, WEIGHT_COPY_BLOCK);
        wait_for_l2fetch();
        if (next_block > 0) l2fetch(in + block, 128, 128, next_block>>7);
        vmemcpy_128(out, in, block);
        in  += block;
        out += block;
        block = next_block;
    }
}

// this assumes src,dst, size all multiples of 128, and size >= 128

#ifndef NN_GRAPH_MEMCPY_BOUNDARY
#define NN_GRAPH_MEMCPY_BOUNDARY (1<<18)
#endif

static inline void nn_graph_memcpy(struct nn_graph *nn, void *dst, const void *src, uint32_t size)
{
#if __HEXAGON_ARCH__ > 65
	logmsg(nn,2,"copy %d bytes %p --> %p",size,src,dst);

#if NN_GRAPH_MEMCPY_BOUNDARY
    {
        uint8_t const *src1 = (uint8_t const *)src;
        uint8_t *dst1 = (uint8_t *)dst;
        while(1) {
            // biggest m allowed by the addresses; in range 127 .. (NN_GRAPH_MEMCPY_BOUNDARY-1).
            unsigned max_m =  (NN_GRAPH_MEMCPY_BOUNDARY-1) - Q6_R_max_RR(
                 (size_t)src1 & (NN_GRAPH_MEMCPY_BOUNDARY-1), (size_t)dst1 & (NN_GRAPH_MEMCPY_BOUNDARY-1));
            // m we will use
            unsigned copym = Q6_R_minu_RR( max_m, size-1);
            unsigned copycount;
            // do the copy of copym+1 bytes
            // copycount = copym+1 is piggybacked in the M0=copym packet
            asm volatile (
               "{%0 = add(%3,#1); M0 = %3;}\n\t memcpy(%1,%2,M0)"
                 :"=&r"(copycount)
                 :"r"(dst1),"r"(src1),"r"(copym)
                 :"m0");
            if( __builtin_expect( copycount >= size,1) ) break;	// all done
            size -= copycount;
            src1 += copycount;
            dst1 += copycount;
        }
    }
#else
        asm volatile (
                "M0 = %2; memcpy(%0,%1,M0)"
        ::"r"(dst),"r"(src),"r"(size-1)
        :"m0");
#endif
#else
	logmsg(nn,2,"copy %d bytes %p --> %p",size,src,dst);
	nn_graph_hvx_blockcopy(dst,src,size);
	logmsg(nn,2,"memcpy complete",size,src,dst);
#endif
}

#endif
