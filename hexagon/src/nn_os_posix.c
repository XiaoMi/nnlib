/*
 * Copyright (c) 2016-2017, The Linux Foundation. All rights reserved.
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

#include <pthread.h>
#include <semaphore.h>

#ifdef H2_H

h2_vecaccess_state_t vecstate;
// h2_mutex_t init_mutex;
int vec_initted;

int nn_os_vector_acquire()
{
	h2_vecaccess_ret_t ret = h2_vecaccess_acquire(&vecstate);
	return ret.idx;
}

void nn_os_vector_release(int idx)
{
	h2_vecaccess_release(&vecstate,idx);
}

#endif


struct nn_pipe {
	uint64_t *data;
	pthread_mutex_t mutex;
	nn_sem_t howempty;
	nn_sem_t howfull;
	int elements;
	int send_idx;
	int recv_idx;
};

void nn_pipe_send(nn_pipe_t *pipe, uint64_t data)
{
	int oldidx;
	nn_sem_wait(&pipe->howempty);
	pthread_mutex_lock(&pipe->mutex);
	oldidx = pipe->send_idx;
	pipe->sendidx = ((oldidx == (pipe->elements-1)) ? 0 : oldidx+1);
	pipe->data[oldidx] = data;
	pthread_mutex_unlock(&pipe->mutex);
	nn_sem_post(&pipe->howfull);
}

uint64_t nn_pipe_recv(nn_pipe_t *pipe)
{
	int oldidx;
	uint64_t ret;
	nn_sem_wait(&pipe->howfull);
	pthread_mutex_lock(&pipe->mutex);
	oldidx = pipe->recv_idx;
	ret = pipe->data[oldidx];
	pipe->recv_idx = ((oldidx == (pipe->elements-1)) ? 0 : oldidx+1);
	pthread_mutex_unlock(&pipe->mutex);
	nn_sem_post(&pipe->howempty);
	return ret;
}

nn_pipe_t *nn_pipe_alloc(uint32_t pipe_elements)
{
	nn_pipe_t *pipe;
	uint64_t buf;
	if ((buf = malloc(sizeof(uint64_t)*pipe_elements);
	if ((pipe = malloc(sizeof(nn_pipe_t))) == NULL) {
		free(buf);
		return NULL;
	}
	pipe->data = buf;
	pthread_mutex_init(&pipe->mutex,NULL);
	sem_init(&pipe->howfull,0,0);
	sem_init(&pipe->howempty,0,pipe_elements);
	pipe->elements = pipe_elements;
	pipe->send_idx = 0;
	pipe->recv_idx = 0;
}

int nn_os_workers_spawn(struct nn_graph *nn)
{
	pthread_t vec,scal1,scal2;
	pthread_attr_t attrs;
	pthread_attr_init(&attrs);
	pthread_attr_setstacksize(&attrs,8192);
	struct tinfo info;
	if (!vec_initted) {
		vec_initted = 1;
		h2_vecaccess_init(&vecstate,H2_VECACCESS_HVX_128);
	}
	nn->vec_work = nn_pipe_alloc(16);
	nn->nonvec_work = nn_pipe_alloc(16);
	nn_sem_init(&info.sem,0);
	info.nn = nn;

	/* Create vector */
	info.pipe = nn->vec_work;
	pthread_create(&vec,&attrs,nn_os_worker,&info);
	nn_sem_wait(&info.sem);
	/* create scalar */
	info.pipe = nn->nonvec_work;
	pthread_create(&scal1,&attrs,nn_os_worker,&info);
	nn_sem_wait(&info.sem);
	pthread_create(&scal2,&attrs,nn_os_worker,&info);
	nn_sem_wait(&info.sem);
	return 0;
}

