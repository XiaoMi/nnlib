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
/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 */

#ifndef NN_PQUEUE_H
#define NN_PQUEUE_H
#include <nn_graph.h>

#define PARENT_IDX(IDX) ((IDX-1)/2)
#define LEFT_CHILD_IDX(IDX) (2*IDX+1)
#define RIGHT_CHILD_IDX(IDX) (2*IDX+2)

struct nn_pqueue
{
    unsigned size;
    unsigned capacity;
    void **data;
    int (*compar)(const void *a, const void *b);
};

void nn_pqueue_init(struct nn_graph *nn, struct nn_pqueue *q, int (*compar)(const void *a, const void *b), unsigned capacity, void *data);
void nn_pqueue_deinit(struct nn_graph *nn, struct nn_pqueue *q);
void nn_pqueue_enqueue(struct nn_graph *nn, struct nn_pqueue *q, const void *data);
void *nn_pqueue_dequeue(struct nn_graph *nn, struct nn_pqueue *q);
static inline void __attribute__((always_inline)) *nn_pqueue_peek(struct nn_graph *nn, struct nn_pqueue *q, unsigned k)
{
    unsigned cur_idx = 0;
    int bit_idx = 0;
    while (1)
    {
        if (1 << (bit_idx + 1) > k)
            break;
        bit_idx++;
    }
    bit_idx--;
    for(; bit_idx >= 0; bit_idx--)
    {
        int mask = (1 << bit_idx);
        if (k & mask)
        {
            cur_idx = RIGHT_CHILD_IDX(cur_idx);
        }
        else
        {
            cur_idx = LEFT_CHILD_IDX(cur_idx);
        }
        
    }
    return q->data[cur_idx];
}
void nn_pqueue_clear(struct nn_graph *nn, struct nn_pqueue *q);
void nn_pqueue_vclear(struct nn_graph *nn, struct nn_pqueue *q);

#endif
