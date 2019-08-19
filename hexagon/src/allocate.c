
/*
 * Copyright (c) 2016-2019, The Linux Foundation. All rights reserved.
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
 * This contains memory allocation routines
 */


/*
 * In the future...
 * 
 * We can look at the entire graph and find the lifetime of allocations.
 * Many allocations will have short lifetimes.
 * Many lifetimes will end at the same point
 * Come up with a good algorithm for pre-allocating everything, but
 * not requiring a total memory size that can hold all allocations.
 * 
 * Current heuristic idea: Go through and allocate in one direction
 * When we about to free lots of stuff in the middle, start allocating from 
 * the other direction
 * 
 * Current non-heuristic idea: Just go through with a first-fit allocator
 * 
 */

/*
 * For starting up and initial testing, just allocate everything ahead of time.
 */

#include <nn_graph.h>
#include <stdlib.h>

#define ALIGN_AMT 128

#ifndef CANARY_VECTORS
//#define CANARY_VECTORS 0
#define CANARY_VECTORS 1
#endif

#define STARTUP_OFFSET ((1+CANARY_VECTORS)*ALIGN_AMT)
#define MAX_ALLOC_SIZE (3*512*1024*1024)

static inline size_t round_up(size_t size)
{
	return (size + ALIGN_AMT - 1) & ~(size_t)(ALIGN_AMT-1);
}


struct freelist_node {
	struct freelist_node *next;
	size_t base;
	unsigned long size;
};

static void *rechunk(
	struct nn_graph *nn,
	struct freelist_node **ptr, 
	unsigned long size)
{
	struct freelist_node *me = *ptr;
	size_t offset = me->base;

	if (me->size == size) {
		*ptr = me->next;
		nn_free(me);
	} else {
		me->base += size;
		me->size -= size;
		if (me->base > nn->watermark_offset) {
			nn->watermark_offset = me->base;
			logmsg(nn,3,"[[Pre-Allocation]]: Watermark now 0x%x",me->base);
		}
	}
	return (void *)(offset+CANARY_VECTORS*ALIGN_AMT);
}

static void *prealloc(
	struct nn_graph *nn,
	struct freelist_node **ptr, 
	unsigned long size)
{
	struct freelist_node *tmp;
	size = round_up(size);
	size += CANARY_VECTORS*ALIGN_AMT*2;
	for ( ; (*ptr) != NULL; ptr = &((*ptr)->next)) {
		tmp = *ptr;
		if (tmp->size < size) continue;
		return rechunk(nn,ptr,size);
	}
	return NULL;
}

static void try_coalesce(struct freelist_node *left)
{
	struct freelist_node *right = left->next;
	if (right == NULL) return;
	if ((left->base + left->size) == right->base) {
		left->size += right->size;
		left->next = right->next;
		nn_free(right);
	}
}

static int prefree(struct nn_graph *nn, struct freelist_node **ptr, void *baseptr, unsigned long size)
{
	size_t base = (size_t)baseptr-(CANARY_VECTORS*ALIGN_AMT);
	struct freelist_node *tmp;
	struct freelist_node *newnode;
	size = round_up(size);
	size += CANARY_VECTORS*ALIGN_AMT*2;
	for ( ; (*ptr) != NULL; ptr = &((*ptr)->next)) {
		tmp = *ptr;
		/* Is it lower and not contiguous? */
		if ((tmp->base + tmp->size) < base) continue;
		/* Is it contiguous on the left? */
		if ((tmp->base + tmp->size) == base) {
			tmp->size += size;
			try_coalesce(tmp);
			return 0;
		}
		/* Is it contiguous on the right? */
		if ((base + size) == tmp->base) {
			tmp->base = base;
			tmp->size += size;
			return 0;
		}
		/* Well, need to make a new node for this */
		break;
	}
	if ((newnode = nn_malloc(sizeof(*newnode))) == NULL) {
		return errlog(nn,"prefree malloc fail");
	}
	newnode->base = base;
	newnode->size = size;
	newnode->next = *ptr;
	*ptr = newnode;
	return 0;
}

static void alloc_init(struct nn_graph *nn)
{
	nn->watermark_offset = STARTUP_OFFSET;
	prefree(nn,&nn->root,(void *)((long)STARTUP_OFFSET),MAX_ALLOC_SIZE);
}

static int count_freelist_nodes(struct freelist_node *p)
{
	int count = 0;
	for ( ; p != NULL; p = p->next) count++;
	return count;
}

static void pprint_freelist(struct nn_graph *nn, struct freelist_node *p)
{
	for (; p != NULL; p = p->next) logmsg(nn,3,"freelist node @ %p: base=%lx size=%x",p,p->base,p->size);
}

static int check_allocations(struct nn_graph *nn)
{
	pprint_freelist(nn,nn->root);
	if (count_freelist_nodes(nn->root) != 1) return errlog(nn,"too chunky");
	if (nn->root->size != (MAX_ALLOC_SIZE + CANARY_VECTORS*ALIGN_AMT*2)) {
		return errlog(nn,"wrong size: %x vs %x", nn->root->size,MAX_ALLOC_SIZE);
	}
	logmsg(nn,2,"Watermark says: we used %d bytes",nn->watermark_offset);
	return 0;
}

static int allocate_storage(struct nn_graph *nn)
{
	size_t bulk_i;
	if (check_allocations(nn) != 0) return errlog(nn,"check");
	if (nn->bulk) return errlog(nn,"bulk already allocated!?");
	if ((nn->bulk = nn_malloc(nn->watermark_offset)) == NULL) {
		return errlog(nn,"bulk malloc fail, size requested==%d",nn->watermark_offset);
	}
	logmsg(nn,2,"Allocated %d bytes @ %p.  Hope that's enough!",
		nn->watermark_offset,
		nn->bulk);
	bulk_i = (size_t)nn->bulk;
	nn->root->base = round_up(bulk_i);
	nn->root->size = nn->watermark_offset-(bulk_i - nn->root->base);
	return 0;
}

#if 1
static void remove_freenodes(struct nn_graph *nn)
{
	struct nn_node **p = &nn->head;
	struct nn_node *tmp;
	while (*p != NULL) {
		tmp = *p;
		if (tmp->node_type == OP_PreFree) {
			*p = tmp->next;
			tmp->outputs = NULL;
			tmp->ops->dtor(tmp,nn);
			tmp = NULL;
			continue;
		}
		p = &((*p)->next);
	}
}

static int append_freenode(
	struct nn_graph *nn, 
	struct nn_node *dst, 
	struct nn_node *src, 
	int out_idx)
{
	struct nn_node *newnode;
	if ((newnode = optab[OP_PreFree]->ctor(
		nn,
		0,
		OP_PreFree,
		NN_PAD_NA,
		0,
		0,
		NULL,
		NULL)) == NULL) {
		return errlog(nn,"freenode ctor fail");
	}
	/* Mark as 0 inputs, but point input to mark src tensor */
	newnode->outputs = &src->outputs[out_idx]->self;
	/* Add into graph */
	newnode->next = dst->next;
	dst->next = newnode;
	return 0;
}

static int add_freenodes(struct nn_graph *nn)
{
	struct nn_node *tmp;
	struct nn_node *dst;
	int i;
	struct tensor *t;
	/* Add free nodes */
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		for (i = 0; i < tmp->n_outputs; i++) {
			t = tmp->outputs[i];
			if ((t->max_size > 0) && (t->data == NULL)) {
				dst = t->last_consumer;
				if (append_freenode(nn,dst,tmp,i) != 0) {
					return errlog(nn,"can't append");
				}
			}
		}
	}
	return 0;
}
#endif

static int allocate_and_free(struct nn_graph *nn)
{
	struct nn_node *tmp;
	struct tensor *t;
	int i;
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
#if 1
	// OLD: look for prefree ops
		if (tmp->node_type == OP_PreFree) {
			if (prefree(nn,
				&nn->root,
				tmp->outputs[0]->data,
				tmp->outputs[0]->max_size) != 0) {
				return errlog(nn,"free failed?!");
			}
			logmsg(nn,3,"prefree %d @ %p",
				tmp->outputs[0]->max_size,
				tmp->outputs[0]->data);
			continue;
		}
#endif
		/* Not a free node, allocate outputs */
		for (i = 0; i < tmp->n_outputs; i++) {
			t = tmp->outputs[i];
			if ((t->max_size > 0) && (t->data == NULL)) {
				/* Pre-Allocate data */
				if ((t->data = prealloc(nn,&nn->root,t->max_size))
					== NULL) {
					return errlog(nn,"alloc failed.  %p is output %d from node %p (id %x), requesting %d bytes", t, i, tmp, tmp->node_id, t->max_size);
				}
				logmsg(nn,3,"alloc %d bytes @ %p",
					t->max_size,
					t->data);
			}
		}
#if 0
	// NEW: Check for being last consumer for a tensor
	// FIXME: how do we signify that the data pointer is bulk and not malloc'd like for const nodes?
	// Until we fix this, we can't use this newer, somewhat more efficient code.
	// But the bulk of the performance benefit is from the last_consumer pointer.
		for (i = 0; i < tmp->n_inputs; i++) {
			const struct tensor *ct;
			ct = tmp->inputs[i];
			if (ct->last_consumer != tmp) continue;
			if (ct->max_size == 0) continue;
			if (prefree(nn,&nn->root,ct->data,ct->max_size) != 0) {
				return errlog(nn,"free failed?!");
			}
			logmsg(nn,3,"prefree %d @ %p",ct->max_size,ct->data);
		}
#endif
	}
	return 0;
}

static void reset_allocated_pointers(struct nn_graph *nn)
{
	struct nn_node *tmp;
	/* Go back and NULL out all the input pointers */
	/* Since we point output0 to output tensors, this resets data pointers */
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		if (tmp->node_type == OP_PreFree) {
			tmp->outputs[0]->data = NULL;
		}
	}
}

int allocate_graph_storage(struct nn_graph *nn)
{
	/* initialize mm system */
	set_last_consumers(nn);
	alloc_init(nn);
	/* Add the free nodes where they belong */
	if (add_freenodes(nn) != 0) return errlog(nn,"add freenodes");
	/* Instead of free nodes, mark last consumer in every tensor */
	/* Go through and figure out storage requirements */
	if (allocate_and_free(nn) != 0) return errlog(nn,"alloc/free");
	/* 
	 * We didn't really want those values since storage 
	 * hasn't been allocated yet... 
	 */
	reset_allocated_pointers(nn);
	/* Check results and actually allocate bulk storage */
	if (check_allocations(nn) != 0) return errlog(nn,"bad alloc check");
	if (allocate_storage(nn) != 0) return errlog(nn,"storage alloc");
	/* Now reassign pointers */
	if (allocate_and_free(nn) != 0) return errlog(nn,"real alloc/free");
	/* Now we shouldn't need the freenodes any more */
	remove_freenodes(nn);
	return 0;
}

static void freelist_teardown(struct nn_graph *nn)
{
	struct freelist_node *tmp = nn->root;
	struct freelist_node *next;
	while (tmp != NULL) {
		next = tmp->next;
		nn_free(tmp);
		tmp = next;
	}
	nn->root = NULL;
}

static inline int is_bulk_data(struct nn_graph *nn, void *p)
{
	size_t longp = (size_t)p;
	size_t bulk_i = (size_t)nn->bulk;
	unsigned long last_bulk_i = (unsigned long)(nn->watermark_offset);
	int is_bulk = ((longp >= bulk_i) && (longp < last_bulk_i));
	logmsg(nn,9,"p=%p longp=%lx bulk_i=%lx last_bulk_i=%lx is_bulk=%d",
		p,longp,bulk_i,last_bulk_i,is_bulk);
	return is_bulk;
}

void allocator_teardown(struct nn_graph *nn)
{
	freelist_teardown(nn);
	if (nn->bulk) nn_free(nn->bulk);
}

void canary_mark(struct nn_graph *nn, struct tensor *t)
{
	if (!is_bulk_data(nn,t->data)) return;
	if (CANARY_VECTORS == 0) return;
	uint32_t *start = t->data;
	uint32_t *end = (uint32_t *)((long)t->data + round_up(t->max_size));
	int i;
	for (i = -32; i < 0; i++) {
		start[i] = 0xCAFEBABE;
	}
	for (i = 0; i < 32; i++) {
		end[i] = 0xDEADBEEF;
	}
}

int canary_check(struct nn_graph *nn, const struct tensor *t)
{
	if (((unsigned long)t) & 0x3) return errlog(nn,"tensor %p is corrupted",t);
	if (!is_bulk_data(nn,t->data)) return 0;
	if (CANARY_VECTORS == 0) return 0;
	uint32_t *start = t->data;
	uint32_t *end = (uint32_t *)((long)t->data + round_up(t->max_size));
	int i;
	for (i = -32; i < 0; i++) {
		if (start[i] != 0xCAFEBABE) return errlog(nn,"tensor %p dead canary",t);
	}
	for (i = 0; i < 32; i++) {
		if (end[i] != 0xDEADBEEF) return errlog(nn,"tensor %p dead canary",t);
	}
	return 0;
}

