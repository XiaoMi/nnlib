
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
#include <nn_graph.h>
#include <stdlib.h>
//#ifndef USE_OS_QURT

struct nn_pipe *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements)
{
	struct nn_pipe *pipe;
	uint64_t *buf;
	if ((buf = nn_malloc(sizeof(uint64_t)*pipe_elements))  == NULL) {
		logmsg(nn,0,"nn_pipe_alloc:buf Fatal ERROR!!!");
		return NULL;
	}
	if ((pipe = nn_malloc(sizeof(struct nn_pipe))) == NULL) {
		nn_free(buf);
		logmsg(nn,0,"nn_pipe_alloc:pipe Fatal ERROR!!!");
		return NULL;
	}
	pipe->data = buf;
	nn_mutex_init(&pipe->mutex);
	nn_sem_init(&pipe->howfull,0);
	nn_sem_init(&pipe->howempty,pipe_elements);
	pipe->elements = pipe_elements;
	pipe->send_idx = 0;
	pipe->recv_idx = 0;
	//logmsg(nn,0,"nn_pipe_alloc: elements=%d", pipe->elements);
	return pipe;
}

void nn_pipe_free(struct nn_pipe *pipe)
{
	nn_free(pipe->data);
	nn_free(pipe);
}

void nn_pipe_send_multi_slowpath(struct nn_pipe *pipe, uint64_t *data, int n_items_left)
{
	int i;
	int idx;
	do {
		int n_items = Q6_R_min_RR(n_items_left,(pipe->elements + 1)/2);
		n_items_left = n_items_left-n_items;
		nn_sem_sub(&pipe->howempty,n_items);
		nn_mutex_lock(&pipe->mutex);
		idx = pipe->send_idx;
		for (i = 0; i < n_items; i++) {
			pipe->data[idx] = data[i];
			idx = idx+1;
			if (idx >= pipe->elements) idx = 0;
		}
		pipe->send_idx = idx;
		nn_mutex_unlock(&pipe->mutex);
		nn_sem_add(&pipe->howfull,n_items);
		data += n_items;
	} while (n_items_left > 0);
}

#if 0
uint64_t nn_pipe_recv_portable(struct nn_pipe *pipe)
{
	int oldidx;
	uint64_t ret;
	nn_sem_wait(&pipe->howfull);
	nn_mutex_lock(&pipe->mutex);
	oldidx = pipe->recv_idx;
	ret = pipe->data[oldidx];
	pipe->recv_idx = ((oldidx == (pipe->elements-1)) ? 0 : oldidx+1);
	nn_mutex_unlock(&pipe->mutex);
	nn_sem_post(&pipe->howempty);
	//printf("Recv data from the pipe idx %d: val %llu",oldidx, ret);
	return ret;
}
#endif

uint64_t nn_pipe_recv_slowpath(struct nn_pipe *pipe)
{
	uint32_t oldidx;
	uint32_t newidx;
	uint64_t ret;
	/* Ensure not empty */
	nn_sem_wait(&pipe->howfull);
	do {
		oldidx = pipe->recv_idx;
		ret = pipe->data[oldidx];
		newidx = oldidx + 1;
		if (newidx >= pipe->elements) newidx = 0;
	} while (!__sync_bool_compare_and_swap(&pipe->recv_idx,oldidx,newidx));
	nn_sem_post(&pipe->howempty);
	return ret;
}

//#endif
