
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
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains a simple hashtable to convert a node ID to a node
 * 
 * Empty nodes have node_id == 0 OR have node == NULL (if deleted)
 * 
 * we delete by setting node == NULL
 * 
 * If we add a node and the hash is over 1/2 full, we grow the table.
 * 
 * Hashing function: upper bits of large prine * id
 * Chaining function: keep adding (id|1) and taking low bits of result
 */

#include <nn_graph.h>

#define LARGE_PRIME 2654435761UL
#define MIN_BITS 7

struct table_data {
	uint32_t node_id;
	struct nn_node *node;
};

struct lookup_info {
	uint32_t size;
	uint32_t entries;
	uint32_t shift;
	struct table_data *data;
	nn_mutex_t lock;
};

static inline uint32_t find_node_hash(struct lookup_info *table, uint32_t node_id)
{
	return (LARGE_PRIME*node_id) >> table->shift;
}

static inline uint32_t find_node_chain(struct lookup_info *table, uint32_t node_id)
{
	return node_id | 1;
}

static inline uint32_t find_node_next(struct lookup_info *table, uint32_t idx, uint32_t chain)
{
	return ((idx+chain) & (table->size - 1));
}

static struct lookup_info *alloc_table(struct nn_graph *nn)
{
	struct lookup_info *table;
	if ((table = nn_malloc(sizeof(*table))) == NULL) {
		logmsg(nn,0,"can't alloc lookup table");
		return NULL;
	}
	nn_mutex_init(&table->lock);
	table->entries = 0;
	table->size = (1<<(MIN_BITS));
	table->shift = 32-(MIN_BITS);
	if ((table->data = nn_calloc(1,sizeof(*table->data)*table->size)) == NULL) {
		nn_free(table);
		logmsg(nn,0,"can't alloc lookup table data");
		return NULL;
	}
	nn_mutex_lock(&nn->log_mutex);
	if (nn->find_node_opaque != NULL) {
		nn_mutex_unlock(&nn->log_mutex);
		nn_free(table);
		return nn->find_node_opaque;
	} else {
		nn->find_node_opaque = table;
	}
	nn_mutex_unlock(&nn->log_mutex);
	return table;
}


static void do_add_node(struct lookup_info *table, uint32_t node_id, struct nn_node *node)
{
	uint32_t idx = find_node_hash(table,node_id);
	uint32_t chain = find_node_chain(table,node_id);
	do {
		if (unlikely(
			(table->data[idx].node_id != 0) &&
				(table->data[idx].node != NULL))) {
			idx = find_node_next(table,idx,chain);
			continue;
		}
		table->data[idx].node_id = node_id;
		table->data[idx].node = node;
		return;
	} while (1);
}

static int grow_hash_locked(struct nn_graph *nn, struct lookup_info *table)
{
	uint32_t oldsize = table->size;
	uint32_t newsize = oldsize * 2;
	struct table_data *newdata;
	struct table_data *olddata = table->data;
	uint32_t i;
	logmsg(nn,2,"growing hash entries=%d oldsize=%d newsize=%d",table->entries,oldsize,newsize);
	if ((newdata = nn_calloc(1,newsize * sizeof(*newdata))) == NULL) {
		return errlog(nn,"can't alloc new data");
	}
	table->size = newsize;
	table->data = newdata;
	table->shift --;
	for (i = 0; i < oldsize; i++) {
		if (olddata[i].node_id == 0) continue;
		if (olddata[i].node == NULL) continue;
		do_add_node(table,olddata[i].node_id,olddata[i].node);
	}
	nn_free(olddata);
	return 0;
}

static void add_node_to_hash(struct nn_graph *nn, struct nn_node *node)
{
	struct lookup_info *table = nn->find_node_opaque;
	if (node == NULL) return;
	if (table == NULL) table = alloc_table(nn);
	if (table == NULL) return;
	nn_mutex_lock(&table->lock);
	if ((table->entries*2) >= table->size) {
		if (grow_hash_locked(nn,table) != 0) return;
	}
	do_add_node(table,node->node_id,node);
	table->entries += 1;
	nn_mutex_unlock(&table->lock);
}

void del_node_from_hash(struct nn_graph *nn, uint32_t node_id)
{
	struct lookup_info *table = nn->find_node_opaque;
	if (table == NULL) return;
	uint32_t idx = find_node_hash(table,node_id);
	uint32_t chain = find_node_chain(table,node_id);
	do {
		if (likely(table->data[idx].node_id == node_id)) {
			nn_mutex_lock(&table->lock);
			if (table->data[idx].node != NULL) table->entries -= 1;
			table->data[idx].node = NULL;
			nn_mutex_unlock(&table->lock);
			return;
		}
		if (table->data[idx].node_id == 0) return;
		idx = find_node_next(table,idx,chain);
	} while (1);
}

/*
 * Look through all the linked list of nodes to find it
 * We could insert the node when we create it.  That would probably be better.
 * But we sometimes create some nodes with the op that we are about to delete, 
 * rather than deleting the old op first and creating the new op second.
 * 
 * Instead of worrying too much about that, instead we just go the slow way
 * the first time.  
 * 
 * This will probably change when we have better infrastructure.  Certainly in
 * C++ We could use a std::map or std::unordered_something.
 */

static struct nn_node *find_slow_and_add_to_hash(struct nn_graph *nn, uint32_t node_id)
{
	struct nn_node *node;
	for (node = nn->head; node != NULL; node = node->next) {
		if (node->node_id == node_id) break;
	}
	add_node_to_hash(nn,node);
	return node;
}

struct nn_node *find_node(struct nn_graph *nn, uint32_t node_id)
{
	struct lookup_info *table = nn->find_node_opaque;
	if (table == NULL) return find_slow_and_add_to_hash(nn,node_id);
	uint32_t idx = find_node_hash(table,node_id);
	uint32_t chain = find_node_chain(table,node_id);
	do {
		if (likely(table->data[idx].node_id == node_id) && (table->data[idx].node != NULL)) return table->data[idx].node;
		if (table->data[idx].node_id == 0) return find_slow_and_add_to_hash(nn,node_id);
		/* If node ID matches our index, it means that the node was NULL, which means deleted, so search slowly */
		if (table->data[idx].node_id == node_id) return find_slow_and_add_to_hash(nn,node_id);
		idx = find_node_next(table,idx,chain);
	} while (1);
}

void find_node_teardown(struct nn_graph *nn)
{
	struct lookup_info *table = nn->find_node_opaque;
	if (table == NULL) return;
	if (table->data) nn_free(table->data);
	nn_free(table);
	nn->find_node_opaque = NULL;
}

