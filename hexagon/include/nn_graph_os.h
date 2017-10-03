
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
#ifndef NN_GRAPH_OS_H
#define NN_GRAPH_OS_H 1
/*
 */

#if defined(USE_OS_LINUX)
#include "nn_graph_os_linux.h"
#elif defined(USE_OS_H2)
#include "nn_graph_os_h2.h"
#elif defined(USE_OS_QURT)
#include "nn_graph_os_qurt.h"
#else
#error please define a USE_OS or be smarter about how we figure out our OS
#endif

int nn_os_workers_spawn(struct nn_graph *nn);
void nn_os_workers_kill(struct nn_graph *nn);
void nn_os_work_for_vector(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg);
void nn_os_work_for_scalar(struct nn_graph *nn, void (*f)(struct nn_graph *, void *),void *arg);
void nn_os_vector_workers_acquire(struct nn_graph *nn);
void nn_os_vector_workers_release(struct nn_graph *nn);

int nn_os_vector_call(struct nn_graph *nn, int (*f)(struct nn_graph *, void *),void *arg);

int nn_os_vtcm_acquire(struct nn_graph *nn);
int nn_os_vtcm_release(struct nn_graph *nn);

struct nn_os_bufstack_t {
	nn_mutex_t mutex;
	void **top;
};

static inline void nn_os_bufstack_init(struct nn_os_bufstack_t *bufstack)
{
	bufstack->top = NULL;
	nn_mutex_init(&bufstack->mutex);
}

static inline void *nn_os_bufstack_pop(struct nn_os_bufstack_t *bufstack)
{
	void *top;
	nn_mutex_lock(&bufstack->mutex);
	top = bufstack->top;
	if (top != NULL) bufstack->top = (void **)*bufstack->top;
	nn_mutex_unlock(&bufstack->mutex);
	return top;
}

static inline void nn_os_bufstack_push(struct nn_os_bufstack_t *bufstack, void *vblock)
{
	void **block = (void**)vblock;
	nn_mutex_lock(&bufstack->mutex);
	*block = bufstack->top;
	bufstack->top = block;
	nn_mutex_unlock(&bufstack->mutex);
}









#ifdef DEBUG_MEMORY_CANARIES

#define CANARY_MEMALIGN 128
#define CANARY_WORDS (CANARY_MEMALIGN / (sizeof(unsigned int)))

#define CANARY_MAGIC 0xCAFEBEEFU

static inline size_t nn_canary_resize(size_t align, size_t bytes)
{
	return ((bytes+(CANARY_MEMALIGN-1))&(~(CANARY_MEMALIGN-1)))+(3*(CANARY_MEMALIGN));
}
static inline void *nn_canary_wrap_alloc(size_t align, size_t bytes, void *vin)
{
	char *bin = vin;
	unsigned int *pre_ptr;
	unsigned int *post_ptr;
	int i;
	align = CANARY_MEMALIGN;
	bytes = (bytes + (CANARY_MEMALIGN-1)) & (~(CANARY_MEMALIGN-1));
	pre_ptr = (unsigned int *)(bin + align);
	post_ptr = (unsigned int *)(bin + align + align + bytes);
	for (i = 0; i < CANARY_WORDS; i++) {
		pre_ptr[i] = post_ptr[i] = CANARY_MAGIC;
	}
	pre_ptr[-1] = bytes;
	return bin+2*align;
}

#include <stdio.h>
#ifdef CHECK_CANARIES
static inline void nn_canary_wrap_free(void *vin)
{
	char *bin = vin;
	unsigned int *wptr = vin;
	int i;
	wptr -= CANARY_WORDS;
	unsigned int size = wptr[-1];
	for (i = 0; i < CANARY_WORDS; i++) {
		if (wptr[i] != CANARY_MAGIC) printf("dead pre canary @ %p size=0x%x (%d,%x @ %p)\n", vin, size,i,wptr[i],wptr+i);
	}
	wptr = (unsigned int *)(bin + size);
	for (i = 0; i < CANARY_WORDS; i++) {
		if (wptr[i] != CANARY_MAGIC) printf("dead post canary @ %p size=0x%x (%d,%x @ %p)\n", vin, size,i,wptr[i],wptr+i);
	}
	bin -= 2*CANARY_MEMALIGN;
	free(bin);
}
#else
static inline void nn_canary_wrap_free(void *vin)
{
	char *bin = vin;
	bin -= 2*CANARY_MEMALIGN;
	free(bin);
}
#endif

static inline void *nn_calloc(size_t n, size_t size) {
	size_t allocsize = nn_canary_resize(CANARY_MEMALIGN,n*size);
	void *ptr = memalign(CANARY_MEMALIGN,allocsize);
	if (ptr == NULL) return NULL;
	memset(ptr,0,allocsize);
	return nn_canary_wrap_alloc(CANARY_MEMALIGN,n*size,ptr);
}
static inline void *nn_malloc(size_t size) { return nn_calloc(1,size); }

static inline void *nn_memalign(size_t a, size_t size) { return (CANARY_MEMALIGN < a) ? NULL : nn_calloc(1,size); }
static inline void nn_free(void *ptr) { return nn_canary_wrap_free(ptr); }
static inline void *nn_realloc(void *ptr, size_t size) {
	if (ptr == NULL) return nn_malloc(size);
	if (size == 0) {
		nn_free(ptr);
		return NULL;
	}
	unsigned int *words = ptr;
	words -= CANARY_WORDS;
	size_t oldsize = words[-1];
	size_t copysize = size;
	if (copysize > oldsize) copysize = oldsize;
	char *newdata;
	if ((newdata = nn_calloc(1,size)) == NULL) return NULL;
	memcpy(newdata,ptr,copysize);
	nn_free(ptr);
	return newdata;
}


#else

static inline void *nn_malloc(size_t size) { return malloc(size); }
static inline void *nn_calloc(size_t n, size_t size) { return calloc(n,size); }
#if defined(__hexagon__)
static inline void *nn_memalign(size_t a, size_t size) { return memalign(a,size); }
#else
static inline void *nn_memalign(size_t a, size_t size) { void *ptr; return posix_memalign(&ptr, a, size)==0 ? ptr: NULL;}
#endif
static inline void *nn_realloc(void *ptr, size_t size) { return realloc(ptr,size); }
static inline void nn_free(void *ptr) { return free(ptr); }

#endif

#ifdef DEBUG_MEM
#include <stdio.h>
static inline void *nn_malloc_debug(size_t size, const char *filename, int line) { void *ret = nn_malloc(size); printf("MEMDBG:%s:%d:ALLOC(%d):%p\n",filename,line,size,ret); return ret; }
static inline void *nn_calloc_debug(size_t n, size_t size, const char *filename, int line) { void *ret = nn_calloc(n,size); printf("MEMDBG:%s:%d:ALLOC(%d):%p\n",filename,line,n*size,ret); return ret; }
static inline void *nn_memalign_debug(size_t a, size_t size, const char *filename, int line) { void *ret = nn_memalign(a,size); printf("MEMDBG:%s:%d:ALLOC(%d):%p\n",filename,line,size,ret); return ret; }
static inline void *nn_realloc_debug(void *ptr, size_t size, const char *filename, int line) 
{
	void *ret = nn_realloc(ptr,size);
	printf("MEMDBG:%s:%d:FREE:%p\n",filename,line,ptr);
	printf("MEMDBG:%s:%d:ALLOC(%d):%p\n",filename,line,size,ret);
	return ret;
}
static inline void nn_free_debug(void *ptr, const char *filename, int line) { nn_free(ptr); printf("MEMDBG:%s:%d:FREE:%p\n",filename,line,ptr); }

#define nn_malloc(sz) nn_malloc_debug(sz,__FILE__,__LINE__)
#define nn_calloc(n,sz) nn_calloc_debug(n,sz,__FILE__,__LINE__)
#define nn_memalign(a,sz) nn_memalign_debug(a,sz,__FILE__,__LINE__)
#define nn_realloc(p,sz) nn_realloc_debug(p,sz,__FILE__,__LINE__)
#define nn_free(p) nn_free_debug(p,__FILE__,__LINE__)
#endif

#ifndef DONT_REDEF_ALLOC
#define malloc(size) OOPS MALLOC
#define calloc(n,size) OOPS CALLOC
#define memalign(n,size) OOPS MEMALIGN
#define realloc(p,s) OOPS REALLOC
#define free(p) OOPS FREE
#endif

#endif // NN_GRAPH_OS_H
