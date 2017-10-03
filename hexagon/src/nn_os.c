
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

//#define NUM_VECTOR_THREADS 1

#ifndef NUM_VECTOR_THREADS
#ifdef HEXAGON_V66
#define NUM_VECTOR_THREADS 4
#else
#define NUM_VECTOR_THREADS 2
#endif
#endif

#define TOTAL_THREADS (NUM_VECTOR_THREADS+2)
#define STACK_SIZE 8192

/*
 * OK, well, we need to cut the # of threads and make things non-global.
 * Create a new, global structure.
 * It will include worker thread information
 * It will include a pointer to all the nn graph contexts, this will facilitate debug
 */

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

struct nn_thread_info {
	nn_thread_t tid;
	void *stack;
	int vecinfo;
	nn_sem_t go;
	nn_sem_t ack;
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
	//logmsg(nn,0,"worker exiting");
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

struct nn_os_veccall_data {
	int (*f)(struct nn_graph *, void *);
	void *arg;
	int ret;
	nn_sem_t donesem;
};

static void nn_os_veccall_stub(struct nn_graph *nn, void *vdata)
{
	struct nn_os_veccall_data *data = vdata;
	data->ret = data->f(nn,data->arg);
	nn_sem_post(&data->donesem);
}

int nn_os_vector_call(struct nn_graph *nn, int (*f)(struct nn_graph *, void *), void *arg)
{
	struct nn_os_veccall_data calldata = {
		.f = f,
		.arg = arg,
		.ret = 0,
	};
	nn_sem_init(&calldata.donesem,0);
	nn_os_work_for_vector(nn,nn_os_veccall_stub,&calldata);
	nn_sem_wait(&calldata.donesem);
	return calldata.ret;
}


static void __attribute__((unused)) worker_acquire(struct nn_graph *nn, void *vptr)
{
	struct nn_thread_info *info = vptr;
	info->vecinfo = nn_os_vector_acquire();
	nn_sem_post(&info->ack);
	nn_sem_wait(&info->go);
}

static void __attribute__((unused)) worker_release(struct nn_graph *nn, void *vptr)
{
	struct nn_thread_info *info = vptr;
	nn_os_vector_release(info->vecinfo);
	nn_sem_post(&info->ack);
	nn_sem_wait(&info->go);
}

/* EJP: FIXME: for non-linux targets, we need a global mutex here to prevent deadlock */
void nn_os_vector_workers_acquire(struct nn_graph *nn)
{
	int i;
	logmsg(nn,4,"acquire");
	struct nn_thread_info *info = nn->os_opaque;
	/* Tell all the vector threads to release vectors */
	for (i = 0; i < (NUM_VECTOR_THREADS); i++) {
		nn_os_work_for_vector(nn,worker_acquire,&info[i]);
	}
	/* wait for all the threads to release vectors */
	for (i = 0; i < (NUM_VECTOR_THREADS); i++) {
		nn_sem_wait(&info[i].ack);
	}
	/* tell all the threads to keep going */
	for (i = 0; i < (NUM_VECTOR_THREADS); i++) {
		nn_sem_post(&info[i].go);
	}
	logmsg(nn,4,"acquire done");
	// nn_os_vecinfo[0] = nn_os_vector_acquire();
}

void nn_os_vector_workers_release(struct nn_graph *nn)
{
	int i;
	logmsg(nn,4,"release");
	struct nn_thread_info *info = nn->os_opaque;
	for (i = 0; i < (NUM_VECTOR_THREADS); i++) {
		nn_os_work_for_vector(nn,worker_release,&info[i]);
		nn_sem_wait(&info[i].ack);
	}
	for (i = 0; i < (NUM_VECTOR_THREADS); i++) {
		nn_sem_post(&info[i].go);
	}
	logmsg(nn,4,"release done");
	// nn_os_vector_release(nn_os_vecinfo[0]);
}

int nn_os_workers_spawn(struct nn_graph *nn)
{
	struct nn_thread_info *worker_info;
	int i;
	struct tinfo info;
	int alloc_err = 0;
	nn_thread_attr_t attrs;
	nn_thread_attr_init(&attrs);

	logmsg(nn,4,"workers spawn");
	if (nn->os_opaque != NULL) {
		return errlog(nn,"OS workers already spawned?");
	}
	if ((worker_info = nn_malloc(sizeof(*worker_info)*(2+NUM_VECTOR_THREADS))) == NULL) {
		return errlog(nn,"OS malloc fail");
	}
	nn->os_opaque = worker_info;

	if ((nn->vec_work = nn_pipe_alloc(nn, 128)) == NULL) return errlog(nn,"os pipe alloc fail");
	if ((nn->nonvec_work = nn_pipe_alloc(nn, 128)) == NULL) return errlog(nn,"os pipe alloc fail");

	for (i = 0; i < TOTAL_THREADS; i++) {
		if ((worker_info[i].stack = nn_malloc(STACK_SIZE)) == NULL) {
			alloc_err = 1;
		}
		nn_sem_init(&worker_info[i].go,0);
		nn_sem_init(&worker_info[i].ack,0);
	}

	if (alloc_err) {
		for (i = 0; i < TOTAL_THREADS; i++) {
			if (worker_info[i].stack) nn_free(worker_info[i].stack);
		}
		nn_free(worker_info);
		nn->os_opaque = NULL;
		return errlog(nn,"os stack / thread alloc fail");
	}

	nn_sem_init(&info.sem,0);
	info.nn = nn;
	nn_os_vector_init();

	for (i = 0; i < TOTAL_THREADS; i++) {
		nn_thread_attr_setstack(&attrs,worker_info[i].stack,STACK_SIZE);
		if (i < NUM_VECTOR_THREADS) info.pipe = nn->vec_work;
		else info.pipe = nn->nonvec_work;
		if (nn_thread_create(nn,&worker_info[i].tid,&attrs,nn_os_worker,&info) != 0) {
			return errlog(nn,"can't create worker thread");
		}
		nn_sem_wait(&info.sem);
	}
	logmsg(nn,4,"workers spawn done");
	return 0;
}

void nn_os_workers_kill(struct nn_graph *nn)
{
	struct nn_thread_info *worker_info = nn->os_opaque;
	int i;
	logmsg(nn,4,"workers kill");
	if (worker_info == NULL) {
		errlog(nn,"OS workers already killed?");
		return;
	}
	for (i = 0; i < TOTAL_THREADS; i++) {
		if (i < NUM_VECTOR_THREADS) nn_os_work_for_vector(nn,NULL,NULL);
		else nn_os_work_for_scalar(nn,NULL,NULL);
	}
	for (i = 0; i < TOTAL_THREADS; i++) {
		nn_thread_join(worker_info[i].tid,NULL);
		nn_free(worker_info[i].stack);
	}
	nn_free(worker_info);
	nn_pipe_free(nn->vec_work);
	nn_pipe_free(nn->nonvec_work);
	nn->vec_work = NULL;
	nn->nonvec_work = NULL;
	nn->os_opaque = NULL;
	logmsg(nn,4,"workers kill done");
}

