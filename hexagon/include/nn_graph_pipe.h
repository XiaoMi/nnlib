
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
#ifndef NN_GRAPH_PIPE_H
#define NN_GRAPH_PIPE_H 1

struct nn_pipe {
	union {
		struct {
			nn_sem_t howfull;
			volatile int recv_idx;
		};
		uint64_t recvidx_howfull;
	};

	nn_sem_t howempty;
	nn_mutex_t mutex;

	volatile int send_idx;
	int pad;

	uint64_t *data;
	int elements;
};

typedef struct nn_pipe nn_pipe_t;

struct nn_pipe *nn_pipe_alloc(struct nn_graph *nn, uint32_t pipe_elements);
void nn_pipe_free(struct nn_pipe *pipe);

void nn_pipe_send_multi_slowpath(struct nn_pipe *pipe, uint64_t *data, int n_items);
void nn_pipe_send_multi_fastpath(struct nn_pipe *pipe, uint64_t *data, int n_items);

//static inline void nn_pipe_send_multi(struct nn_pipe *pipe, uint64_t *data, int n_items) { return nn_pipe_send_multi_slowpath(pipe,data,n_items); }
static inline void nn_pipe_send_multi(struct nn_pipe *pipe, uint64_t *data, int n_items) { return nn_pipe_send_multi_fastpath(pipe,data,n_items); }

static inline void nn_pipe_send(struct nn_pipe *pipe, uint64_t val) { return nn_pipe_send_multi(pipe,&val,1); }

uint64_t nn_pipe_recv_slowpath(struct nn_pipe *pipe);
uint64_t nn_pipe_recv_fastpath(struct nn_pipe *pipe);

static inline uint64_t nn_pipe_recv(struct nn_pipe *pipe) { return nn_pipe_recv_fastpath(pipe); }
//static inline uint64_t nn_pipe_recv(struct nn_pipe *pipe) { return nn_pipe_recv_slowpath(pipe); }

#endif
