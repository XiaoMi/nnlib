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
 * This contains an implementation of a (soon to be) thread safe priority queue
 * Might need to make this double ended? Or just have the caller keep track of the min, to avoid further insertions
 * Assumes that the caller has allocated memory for the heap to be stored (since you might want to use scratch for these)
 */

#include "nn_pqueue.h"

static void nn_pqueue_heapify(struct nn_graph *nn, struct nn_pqueue *q, unsigned index);

void nn_pqueue_init(struct nn_graph *nn, struct nn_pqueue *q, int (*compar)(const void *a, const void *b), unsigned capacity, void *data)
{
    q->compar = compar;
    q->size = 0;
    q->capacity = capacity;
    q->data = data;
}

void nn_pqueue_clear (struct nn_graph *nn, struct nn_pqueue *q)
{
    //Clear the heap. Maybe the caller should do the memseting, since we don't know about the data type
    q->size = 0;
    memset(q->data, 0, q->capacity);
}

void nn_pqueue_vclear (struct nn_graph *nn, struct nn_pqueue *q)
{
    //Clear the heap. Maybe the caller should do the memseting, since we don't know about the data type
    q->size = 0;
    vmemset_asm(q->data, 0, q->capacity);
}

void nn_pqueue_deinit(struct nn_graph *nn, struct nn_pqueue *q)
{
    //If we used malloc or something to allocate the heap, we'd free that here
    if (NULL == q)
        return;
}

void nn_pqueue_enqueue(struct nn_graph *nn, struct nn_pqueue *q, const void *data)
{
    unsigned i = 0;
    void *tmp = NULL;
    if (q->size >= q->capacity)
    {
        q->data[q->size - 1] = (void *)data;
        i = q->size - 1;
    }
    else
    {
        q->data[q->size++] = (void *)data;
        i = q->size - 1;
    }
    int p = PARENT_IDX(i);
    while (i > 0 && q->compar(q->data[i], q->data[p]) > 0)
    {
        tmp = q->data[i];
        q->data[i] = q->data[p];
        q->data[p] = tmp;
        i = p;
        p = PARENT_IDX(i);
    }
}

void *nn_pqueue_dequeue(struct nn_graph *nn, struct nn_pqueue *q)
{
    void *data = NULL;
    if (q->size < 1)
    {
        //Queue empty
        return NULL;
    }
    data = q->data[0];
    q->data[0] = q->data[q->size - 1];
    q->size--;
    nn_pqueue_heapify(nn, q, 0);
    return data;
}

static inline void __attribute__((always_inline)) nn_pqueue_heapify(struct nn_graph *nn, struct nn_pqueue *q, unsigned index)
{
    void *tmp = NULL;
    unsigned left_index = LEFT_CHILD_IDX(index);
    unsigned right_index = RIGHT_CHILD_IDX(index);
    unsigned larger_index;

    if (left_index < q->size && q->compar(q->data[left_index], q->data[index]) > 0)
    {
        larger_index = left_index;
    }
    else
    {
        larger_index = index;
    }

    if (right_index < q->size && q->compar(q->data[right_index], q->data[larger_index]) > 0)
    {
        larger_index = right_index;
    }

    if (larger_index != index)
    {
        tmp = q->data[larger_index];
        q->data[larger_index] = q->data[index];
        q->data[index] = tmp;
        nn_pqueue_heapify(nn, q, larger_index);
    }
}
