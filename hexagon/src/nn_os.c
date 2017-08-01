
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
/*
 */
#include <nn_graph.h>
#include <stdlib.h>

#ifndef NUM_VECTOR_THREADS
#define NUM_VECTOR_THREADS 2
#endif

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
	struct tinfo *info = (struct tinfo *)vinfo;
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
	int wait_for_context = 1;
	if (dspCV_hvx_lock(DSPCV_HVX_MODE_128B, wait_for_context) < 0) {
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
	qurt_thread_t *worker_ids;
	struct tinfo info;
	qurt_thread_attr_t attrs[NUM_VECTOR_THREADS+2];
	qurt_pipe_attr_t pattr;
	int i;

	if (nn->os_opaque != NULL) {
		return errlog(nn,"OS workers already spawned?");
	}
	if ((worker_ids = malloc(sizeof(*worker_ids)*(2+NUM_VECTOR_THREADS))) == NULL) {
		return errlog(nn,"OS malloc fail");
	}
	nn->os_opaque = worker_ids;

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

	for (i = 1; i < NUM_VECTOR_THREADS+2; i++) {
		qurt_thread_attr_t *a = &attrs[i];
		qurt_thread_attr_init(a);
		qurt_thread_attr_set_name(a,(char *)"nn_worker");
		qurt_thread_attr_set_stack_addr(a,malloc(STACK_SIZE));
		qurt_thread_attr_set_stack_size(a,STACK_SIZE);
		qurt_thread_attr_set_priority(a, QURT_THREAD_ATTR_PRIORITY_DEFAULT/2+i);
		if (i < NUM_VECTOR_THREADS) info.pipe = nn->vec_work;
		else info.pipe = nn->nonvec_work;
		qurt_thread_create(&worker_ids[i],a,qurt_worker,&info);
		nn_sem_wait(&info.sem);
	}
	return 0;
}

void nn_os_workers_kill(struct nn_graph *nn)
{
	int i;
	int status;
	qurt_thread_t *worker_ids = nn->os_opaque;
	if (worker_ids == NULL) {
		errlog(nn,"OS workers already killed?");
		return;
	}
	for (i = 0; i < (NUM_VECTOR_THREADS); i++) {
		nn_os_work_for_vector(nn,NULL,NULL);
	}
	nn_os_work_for_scalar(nn,NULL,NULL);
	nn_os_work_for_scalar(nn,NULL,NULL);
	for (i = 1; i < (NUM_VECTOR_THREADS+2); i++) {
		qurt_thread_join(worker_ids[i],&status);
	}
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
	if (nn->perf_event == NN_GRAPH_PERFEVENT_UTIME) return nn_os_get_usecs(nn);
	lo = qurt_pmu_get(QURT_PMUCNT0);
	hi = qurt_pmu_get(QURT_PMUCNT1);
	ret = hi;
	ret <<= 32;
	ret |= (unsigned long long int) lo;
	return ret;
}



#endif

static int nn_os_vecinfo[NUM_VECTOR_THREADS];
nn_sem_t worker_acquired_sem;
nn_sem_t worker_go_sem;


static void __attribute__((unused)) worker_acquire(struct nn_graph *nn, void *vptr)
{
	int *ptr = (int *)vptr;
	*ptr = nn_os_vector_acquire();
	nn_sem_post(&worker_acquired_sem);
	nn_sem_wait(&worker_go_sem);
}

static void __attribute__((unused)) worker_release(struct nn_graph *nn, void *vptr)
{
	int *ptr = (int *)vptr;
	nn_os_vector_release(*ptr);
	nn_sem_post(&worker_acquired_sem);
	nn_sem_wait(&worker_go_sem);
}

void nn_os_vector_workers_acquire(struct nn_graph *nn)
{
	int i;
	nn_sem_init(&worker_acquired_sem,0);
	nn_sem_init(&worker_go_sem,0);
	for (i = 1; i < (NUM_VECTOR_THREADS); i++) {
		nn_os_work_for_vector(nn,worker_acquire,&nn_os_vecinfo[i]);
		nn_sem_wait(&worker_acquired_sem);
	}
	for (i = 1; i < (NUM_VECTOR_THREADS); i++) {
		nn_sem_post(&worker_go_sem);
	}
	nn_os_vecinfo[0] = nn_os_vector_acquire();
}

void nn_os_vector_workers_release(struct nn_graph *nn)
{
	int i;
	nn_sem_init(&worker_acquired_sem,0);
	nn_sem_init(&worker_go_sem,0);
	for (i = 1; i < (NUM_VECTOR_THREADS); i++) {
		nn_os_work_for_vector(nn,worker_release,&nn_os_vecinfo[i]);
		nn_sem_wait(&worker_acquired_sem);
	}
	for (i = 1; i < (NUM_VECTOR_THREADS); i++) {
		nn_sem_post(&worker_go_sem);
	}
	nn_os_vector_release(nn_os_vecinfo[0]);
}

#if !defined(USE_OS_QURT)

int nn_os_workers_spawn(struct nn_graph *nn)
{
	pthread_t vec,scal1,scal2;
	pthread_attr_t attrs;
	pthread_attr_init(&attrs);
	pthread_attr_setstacksize(&attrs,8192);
	pthread_t *worker_ids;
	//pthread_attr_setdetachstate(&attrs,1);
	struct tinfo info;
	int i;
	if (nn->os_opaque != NULL) {
		return errlog(nn,"OS workers already spawned?");
	}
	if ((worker_ids = malloc(sizeof(*worker_ids)*(2+NUM_VECTOR_THREADS-1))) == NULL) {
		return errlog(nn,"OS malloc fail");
	}
	nn->os_opaque = worker_ids;

	nn->vec_work = nn_pipe_alloc(nn, 128);
	nn->nonvec_work = nn_pipe_alloc(nn, 128);
	//logmsg(nn,0,"nn_pipe_alloc vec: elements=%d", nn->vec_work->elements);
	//logmsg(nn,0,"nn_pipe_alloc sca: elements=%d", nn->nonvec_work->elements);
	nn_sem_init(&info.sem,0);
	info.nn = nn;

	nn_os_vector_init();

	/* Create vector */
	info.pipe = nn->vec_work;
	for (i = 0; i < (NUM_VECTOR_THREADS-1); i++) {
		pthread_create(&vec,&attrs,nn_os_worker,&info);
		worker_ids[2+i] = vec;
		nn_sem_wait(&info.sem);
	}
	/* create scalar */
	info.pipe = nn->nonvec_work;
	pthread_create(&scal1,&attrs,nn_os_worker,&info);
	worker_ids[0] = scal1;
	nn_sem_wait(&info.sem);
	pthread_create(&scal2,&attrs,nn_os_worker,&info);
	worker_ids[1] = scal2;
	nn_sem_wait(&info.sem);

	return 0;
}

void nn_os_workers_kill(struct nn_graph *nn)
{
	int i;
	pthread_t *worker_ids = nn->os_opaque;
	if (worker_ids == NULL) {
		errlog(nn,"OS workers already killed?");
		return;
	}
	for (i = 0; i < (NUM_VECTOR_THREADS-1); i++) {
		nn_os_work_for_vector(nn,NULL,NULL);
		pthread_join(worker_ids[2+i],NULL);
	}
	nn_os_work_for_scalar(nn,NULL,NULL);
	nn_os_work_for_scalar(nn,NULL,NULL);
	pthread_join(worker_ids[0],NULL);
	pthread_join(worker_ids[1],NULL);
}


#endif
