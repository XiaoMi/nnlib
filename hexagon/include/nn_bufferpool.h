/*
 * Copyright (c) 2018, The Linux Foundation. All rights reserved.
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
#ifndef NN_BUFFERPOOL_H
#define NN_BUFFERPOOL_H 1

#include <stdlib.h>
#include <stdint.h>
#include "nn_graph_builtin.h"


// 'struct bufferpool':
// this manages a set of N equal-sized scratch buffers ( N<=32), which are taken from the pool
// and put back at low overhead, using a bit map
//
// (1) initialize
//
//  struct buffer_pool bp;
//  bufpool_init( &bp, int nbufs, void *memp, uint32_t bufsize );
//
///     initialize the pool, supplying 'bufs' buffers, each of 'bufsize' bytes;
//        'memp' is the pointer to the memory area where they are located
//        nbufs must be in range 1..32.
//   The buffers are at memp + i*bufsize, for i  =0..nbufs-1
//   bufpool functions do not use the memory at all; we just store the base and size,
//   and bufpool_take computes the buffer address as above.
//
//  (2) 'take' a buffer
//
//    int bufind;
//   void *ptr = bufpool_take( &bp, &bufind);
//
//       if successful, returns a non-null pointer, and sets bufind to 0..nbufs-1.
//       if all are taken, returns NULL and sets bufind to -1.
//
//  (3) 'release' a buffer
//		int result = bufpool_release( &bp, bufind);
//
//    normally returns 0; returns -1 if bufind is not in range 0..nbufs-1, or if
//    the buffer is already free.
//
// bufpool_take and bufpool_release are both thread-safe, via locked ops on bp->freeset.
//
//
// NOTE: there is no allocation or deallocation of the actual memory; this
// is intended to be used to distribute a pool of scratch buffers across threads.
//
//

struct buffer_pool {
	uint32_t freeset;	// each bit =1 if the buffer is free
	uint32_t allset; 	// set to the initial state of 'freeset' - lowest 'n' bits are 1
	void * membase;
	uint32_t bufsize;
};


static inline void
bufpool_init( struct buffer_pool *bp, int nbufs, void * memp, uint32_t bufsize )
{
	uint32_t m = 1u << (nbufs-1);	// m.s. bit needed (1 << by 0..31)
	m |= (m-1);		// fill in the lower bits;
	bp->freeset = bp->allset = m;
	bp->membase = memp;
	bp->bufsize = bufsize;
}
//
// take a buffer from the pool.
// if there is one, returns the pointer, and the index of the buffer via 'bufind_p';
// if no buffers, returns NULL and *bufind_p = -1
// The buffer index is used to release the buffer later.
//
static inline void *
bufpool_take( struct buffer_pool *bp, int * bufind_p )
{
    uint32_t m = bp->allset;
    uint32_t prev_bufset = bp->freeset;
    uint32_t bufset0;
    do{
        if( (prev_bufset & m)== 0 ){		// no valid nonzero bits
            *bufind_p = -1;
            return NULL;
        }
        // at least one of the lowest N bits is 1.
        // find the same value with the least-sig '1' removed...
        uint32_t upd_bufset = (prev_bufset-1)& prev_bufset;
        bufset0 = prev_bufset;
        // try to swap that in.
        prev_bufset = __sync_val_compare_and_swap( &bp->freeset, prev_bufset, upd_bufset );
    }while( __builtin_expect(prev_bufset != bufset0,0));

    int bufn = __builtin_ctz( prev_bufset );	// index of the 1 we removed
    *bufind_p = bufn;
    return (void*)( (uint8_t*) bp->membase + bufn* bp->bufsize );
}

//
// release a buffer, via the index from bufpool_take.
// if the index is out of range, or the buffer is already free, this has no effect
// and returns -1; otherwise it sets the 'free' bit and returns 0.
//
static inline int
bufpool_release( struct buffer_pool *bp, int bufind )
{
    if( bufind >= 0 && bufind < 32 ){
        uint32_t m = (1u<<bufind);
        if( (bp->allset & m)!=0 ){	// is valid...
            uint32_t x = __sync_fetch_and_or( &bp->freeset, m );
            // x = previous freeset; check if bit was 0 before.
            if( (x&m) == 0) return 0;
        }
    }
    return -1;
}

#endif // NN_BUFFERPOOL_H
