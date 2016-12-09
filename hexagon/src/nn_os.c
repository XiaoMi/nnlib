
/*
 * Copyright (c) 2016, The Linux Foundation. All rights reserved.
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

union workitem {
	struct {
		void (*f)(struct nn_graph *, void *);
		void *arg;
	};
	uint64_t raw;
};

struct tinfo {
	struct nn_graph *nn;
	nn_pipe_t *pipe;
	nn_sem_t sem;
};

static void *nn_os_worker(void *vinfo)
{
	struct tinfo *info = vinfo;
	struct nn_graph *nn = info->nn;
	nn_pipe_t *pipe = info->pipe;
	union workitem work;
	nn_sem_post(&info->sem);
	while (1) {
		work.raw = nn_pipe_recv(pipe);
		//logmsg(nn,0,"nn_pipe_recv work.raw=%x", work.raw);
		if (work.f == NULL) break;
		work.f(nn,work.arg);
	}
	return NULL;
}

void nn_os_workers_kill(struct nn_graph *nn)
{
	nn_os_work_for_vector(nn,NULL,NULL);
	nn_os_work_for_scalar(nn,NULL,NULL);
	nn_os_work_for_scalar(nn,NULL,NULL);
}

void nn_os_work_for_vector(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg)
{
	union workitem msg;
	msg.f = f;
	msg.arg = arg;
	//logmsg(nn,0,"nn_pipe_send msg.raw=%x", msg.raw);
	nn_pipe_send(nn->vec_work, msg.raw);
}


void nn_os_work_for_scalar(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg)
{
	union workitem msg;
	msg.f = f;
	msg.arg = arg;
	nn_pipe_send(nn->nonvec_work, msg.raw);
}

#if defined(USE_OS_H2)
h2_vecaccess_state_t vecstate;
// h2_mutex_t init_mutex;
int vec_initted = 0;

int nn_os_vector_acquire()
{
	h2_vecaccess_ret_t ret = h2_vecaccess_acquire(&vecstate);
	return ret.idx;
}

void nn_os_vector_release(int idx)
{
	h2_vecaccess_release(&vecstate,idx);
}

void nn_os_vector_init()
{
	if (!vec_initted) {
		vec_initted = 1;
		h2_vecaccess_init(&vecstate,H2_VECACCESS_HVX_128);
	}
	
}

#elif defined(USE_OS_QURT)
#include "dspCV_hvx.h"
#include <qurt.h>

#define PIPESIZE_ELEMENTS 4
#define PIPESIZE_BYTES ((PIPESIZE_ELEMENTS)*8)
#define STACK_SIZE 8192

static void __attribute__((unused)) qurt_worker(void *p)
{
	nn_os_worker(p);
	qurt_thread_exit(0);
}

int nn_os_vector_acquire()
{
		if (dspCV_hvx_lock(DSPCV_HVX_MODE_128B, 0) < 0) {
		return 0;
	}
	return 0;
}

void nn_os_vector_release(int idx)
{
	dspCV_hvx_unlock();
}

int nn_os_workers_spawn(struct nn_graph *nn)
{
	qurt_thread_t vec,scal1,scal2;
	struct tinfo info;
	qurt_thread_attr_t attrs[3];
	qurt_pipe_attr_t pattr;
	int i;

	nn_sem_init(&info.sem,0);
	info.nn = nn;

	qurt_pipe_attr_init(&pattr);
	qurt_pipe_attr_set_buffer(&pattr,malloc(PIPESIZE_BYTES));
	qurt_pipe_attr_set_elements(&pattr,PIPESIZE_ELEMENTS);
	qurt_pipe_create(&nn->vec_work,&pattr);

	qurt_pipe_attr_init(&pattr);
	qurt_pipe_attr_set_buffer(&pattr,malloc(PIPESIZE_BYTES));
	qurt_pipe_attr_set_elements(&pattr,PIPESIZE_ELEMENTS);
	qurt_pipe_create(&nn->nonvec_work,&pattr);

	for (i = 0; i < 3; i++) {
		qurt_thread_attr_t *a = &attrs[i];
		qurt_thread_attr_init(a);
		qurt_thread_attr_set_name(a,(char *)"nn_worker");
		qurt_thread_attr_set_stack_addr(a,malloc(STACK_SIZE));
		qurt_thread_attr_set_stack_size(a,STACK_SIZE);
		qurt_thread_attr_set_priority(a, QURT_THREAD_ATTR_PRIORITY_DEFAULT/2+i);
	}
	logmsg(nn,0,"thread priority: %d\n",QURT_THREAD_ATTR_PRIORITY_DEFAULT/2);
	info.pipe = nn->vec_work;
	if ((i=qurt_thread_create(&vec,&attrs[0],qurt_worker,&info)) != QURT_EOK) {
		return errlog(nn,"thread create fail: %d",i);
	}
	nn_sem_wait(&info.sem);
	info.pipe = nn->nonvec_work;
	qurt_thread_create(&scal1,&attrs[1],qurt_worker,&info);
	nn_sem_wait(&info.sem);
	qurt_thread_create(&scal2,&attrs[2],qurt_worker,&info);
	nn_sem_wait(&info.sem);
	return 0;
}

void nn_os_hvx_power_on(struct nn_graph *nn)
{
	if (dspCV_hvx_power_on() != 0) {
		errlog(nn,"couldn't power on hvx\n");
	}
}

void nn_os_hvx_power_off(struct nn_graph *nn)
{
	dspCV_hvx_power_off();
}

/* depending on config, get pcycles or PMU events */
unsigned long long int nn_os_get_perfcount(struct nn_graph *nn) {
	uint32_t lo;
	uint32_t hi;
	uint64_t ret;
	if (nn->perf_event < NN_GRAPH_PERFEVENT_HWPMU) {
		if (nn->perf_event == 0) return qurt_get_core_pcycles();
	}
	lo = qurt_pmu_get(QURT_PMUCNT0);
	hi = qurt_pmu_get(QURT_PMUCNT1);
	ret = hi;
	ret <<= 32;
	ret |= lo;
	return ret;
}
#endif

#if !defined(USE_OS_QURT)

int nn_os_workers_spawn(struct nn_graph *nn)
{
	pthread_t vec,scal1,scal2;
	pthread_attr_t attrs;
	pthread_attr_init(&attrs);
	pthread_attr_setstacksize(&attrs,8192);
	struct tinfo info;

	nn->vec_work = nn_pipe_alloc(nn, 128);
	nn->nonvec_work = nn_pipe_alloc(nn, 128);
	//logmsg(nn,0,"nn_pipe_alloc vec: elements=%d", nn->vec_work->elements);
	//logmsg(nn,0,"nn_pipe_alloc sca: elements=%d", nn->nonvec_work->elements);
	nn_sem_init(&info.sem,0);
	info.nn = nn;

	nn_os_vector_init();

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
#endif
