
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
 *
 */


//
// detailed rules on maintaining the hash:
//
//  There are 3 states for the hash entries:
//      node_id = 0,   node = NULL			 (empty slot)
//      node_id !=0,   node != NULL			(valid slot)
//      node_id !=0,   node = NULL			(deleted)
//
// Lookup process:
//   probe the hash according to the start & chain functions of 'node_id' until we find
//   an entry with node_id matching, or an empty entry.
//   - if we find a valid slot, that is the result;
//   - if we find a deleted entry matching node id, or an empty entry, the result is "nothing found"
//   - for other cases, keep following the chain.
//
// In order to get predictable behaviour with deletes, and with inserts of values that may exist already:
//
//  Rule 1: We can't ever change any other slot state to 'empty', it could break the lookup of another entry.
//    Deleted slots can be changed to valid by 'insert' operations. Also we clean them out when the hash is enlarged.
//  Rule 2: It is OK to have multiple entries with the same node_id, but at most one of them can be 'valid'
//    (the rest must be 'deleted'); in such a situation the 'valid' one (if there is one)
//    MUST be the first one to appear in the path for that node_id. Only the first of these will be seen
//    by a lookup for 'node_id'; but if the first is a 'deleted entry' and is subsequently replaced by an
//    insert of another node_id, the next one could then be seen.
//
//  Delete process:
//   - find the entry using the lookup process given above
//   - if not found, do nothing; if found, change the slot to deleted by clearing the 'node' pointer.
//  We have an option to delete the entry only if the pointer matches a supplied pointer.
//
//  Insert process (overwriting previous entry if any):
//   Need to first find any previous entry, to avoid violating Rule 2. Process is equivalent to:
//      (1) try to find the entry using 'lookup' process. If found, delete the entry
//      (2) go back to the start of the probe process, and look for the first entry in the chain which is deleted
//         or which is empty; place the new entry there. Note that this may come earlier in the chain than the value
//         deleted in (1).
//   We can, instead, perform step (1), and remember the location of the first 'deleted' entry seen, this is where the
//   new entry is placed.
//
// We have another insert which either inserts the new value if it doesn't exist, or returns the existing entry without
// inserting the new one. This is is above except step (1) is
//      (1) try to find the entry using 'lookup' process. If found, return the existing entry.
//
//

#include <nn_graph.h>

#define LARGE_PRIME 2654435761UL
#define MIN_BITS 7

struct table_data {
	uint32_t node_id;
	struct nn_node *node;
};

struct lookup_info {
	uint32_t size;
	uint32_t entries;		// also includes deleted entries (node_id != 0, node = NULL)
	uint32_t shift;
	struct table_data *data;
	nn_mutex_t lock;
};

static inline uint32_t find_node_hash(struct lookup_info *table, uint32_t node_id)
{
	return (uint32_t)(LARGE_PRIME*node_id) >> table->shift;
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
	unsigned padded_n = nn->node_count;
	padded_n += (padded_n>>4)+5;

	int tshift = (padded_n >= (1<<(MIN_BITS-1)) )? (__builtin_clz( padded_n )-1): (32-MIN_BITS);
	int tsize = 1<<(32-tshift);

	nn_mutex_init(&table->lock);
	table->entries = 0;
	table->size = tsize; //(1<<(MIN_BITS));
	table->shift = tshift; //32-(MIN_BITS);
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

//
// find the valid entry for node_id in the hash; return a pointer to it.
// If it does not exist, return NULL. This is used for lookup and for delete.
//
static inline
struct table_data *
find_existing_entry(struct lookup_info *table, uint32_t node_id )
{
	uint32_t idx = find_node_hash(table,node_id);
	uint32_t chain = find_node_chain(table,node_id);
	struct table_data *td = table->data;
	for(;;) {
		if (likely(td[idx].node_id == node_id)) {
			return (td[idx].node == NULL)?NULL : &td[idx];
		}
		if (td[idx].node_id == 0)
			break;
		idx = find_node_next(table,idx,chain);
	}
	return NULL;
}

//
// This is used for inserts.
// find a valid entry for node_id in the hash, if it exists, and also
// find the index of the first available slot for inserting (which always exists).
// Return value is a pointer to the existing valid entry for node_id, if one exists,
// or NULL if the lookup fails. The value 'inspoint' is the table index of where node_id
// can be inserted. if the return value is not NULL, 'inspoint' may indicate the same entry,
// or it may indicate an earlier (deleted) entry.
//
static inline
struct table_data *
find_entry_for_insert(struct lookup_info *table, uint32_t node_id, int * inspoint )
{
	uint32_t idx = find_node_hash(table,node_id);
	uint32_t chain = find_node_chain(table,node_id);
	int inspt = -1;
	struct table_data *td = table->data;

	struct table_data *tdi;
	for(;;) {
		tdi = &td[idx];
		if( tdi->node == NULL){		// empty or deleted slot
			if( inspt < 0 ) inspt = idx;	// available
			if ( tdi->node_id == node_id || tdi->node_id == 0){	//empty, or deleted but current slot; done
				tdi = NULL;
				break;
			}
		}else{				// full slot - only interesting if matches node_id
			if( tdi->node_id == node_id ){
				if( inspt < 0 ) inspt = idx;
				break;
			}
		}
		idx = find_node_next(table,idx,chain);
	}
	*inspoint = inspt;
	return tdi;
}



// an 'add_node' which assumes no deleted or matching entries exist - only used in grow_hash_locked
static int do_add_node(struct lookup_info *table, uint32_t node_id, struct nn_node *node)
{
	uint32_t idx = find_node_hash(table,node_id);
	uint32_t chain = find_node_chain(table,node_id);
	while( unlikely( table->data[idx].node != NULL)){
		idx = find_node_next(table,idx,chain);
	}
	uint32_t old = table->data[idx].node_id;
	table->data[idx].node_id = node_id;
	table->data[idx].node = node;
	return old == 0;
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
	int new_entries = 0;		// new # can be smaller, we drop 'deleted' entries.
	for (i = 0; i < oldsize; i++) {
		if (olddata[i].node_id == 0) continue;
		if (olddata[i].node == NULL) continue;
		do_add_node(table,olddata[i].node_id,olddata[i].node);
		new_entries++;
	}
	table->entries = new_entries;
	nn_free(olddata);
	return 0;
}
//
// add a node to hash, over-writing any previous that might exist
//
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
	uint32_t node_id = node->node_id;
	int inspt;
	struct table_data * existing = find_entry_for_insert( table, node_id, &inspt);
	if( existing != NULL) existing->node = NULL;	// remove existing, if any;
	struct table_data * insert_at = &table->data[inspt];
	if( insert_at->node_id == 0){
		table->entries += 1;		// only count if it was an empty slot
	}
	insert_at->node_id = node_id;
	insert_at->node = node;

	nn_mutex_unlock(&table->lock);
}

//
// delete an entry under 'node_id', if one exists.
// if 'node' is not null, the entry will only be deleted if it's node pointer matches node.
//
void del_node_from_hash(struct nn_graph *nn, uint32_t node_id, struct nn_node * node )
{
	struct lookup_info *table = nn->find_node_opaque;
	if (table == NULL) return;

	nn_mutex_lock(&table->lock);
	struct table_data * ep = find_existing_entry(table, node_id );
	if( ep != NULL){			// found one
		if( node == NULL || ep->node == node)	// really delete it?
			ep->node = NULL;		// deleted.
	}
	nn_mutex_unlock(&table->lock);
}

// API for creating a new 'clean' hash at start of prepare.c
//  - initialize_hash: create empty hash table large enough for nn->node_count nodes
// - insert_node_to_hash will add a node to the hash, and return the same pointer;
//  if the id is already in the hash, it will return the pointer to the existing node
// (to detect collisions).
//
// Ensure the hash table exists and is large enough for nn_node_count.
// clear it.
int initialize_hash( struct nn_graph *nn)
{
	struct lookup_info *table = (struct lookup_info *) nn->find_node_opaque;
	if( table != NULL && table->size < nn->node_count * 2 ){
		find_node_teardown( nn );
		table = (struct lookup_info *) nn->find_node_opaque;
	}
	if (table == NULL){
		if( alloc_table(nn) == NULL ) return -1;
	}else{
		memset( table->data, 0,table->size *sizeof(*table->data) );
	}
	return 0;
}
// This does an insert node but if the node is already there, it returns
// the existing  pointer. Returns same pointer on normal insert.
// Returns NULL on error.
//
struct nn_node *
insert_node_to_hash( struct nn_graph *nn, struct nn_node *node)
{
	struct lookup_info *table = (struct lookup_info *) nn->find_node_opaque;
	if (table == NULL) table = alloc_table(nn);
	if (table == NULL) return NULL;
	nn_mutex_lock(&table->lock);
	if ((table->entries*2) >= table->size) {
		if (grow_hash_locked(nn,table) != 0) return NULL;
	}

	struct nn_node *result  = node;
	uint32_t node_id = node->node_id;

	int inspt;
	struct table_data * existing = find_entry_for_insert( table, node_id, &inspt);
	if( existing != NULL){
		result = existing->node;	// no insert - just return this.
	}else{
		struct table_data * insert_at = &table->data[inspt];
		if( insert_at->node_id == 0){
			table->entries += 1;		// only count if it was an empty slot
		}
		insert_at->node_id = node_id;
		insert_at->node = node;
	}
	nn_mutex_unlock(&table->lock);
	return result;
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

	nn_mutex_lock(&table->lock);
	struct table_data * ep = find_existing_entry(table, node_id );
	struct nn_node *res = (ep==NULL)?NULL : ep->node;
	nn_mutex_unlock(&table->lock);
	if( res == NULL)
		return find_slow_and_add_to_hash(nn,node_id);
	return res;
}

void find_node_teardown(struct nn_graph *nn)
{
	struct lookup_info *table = nn->find_node_opaque;
	if (table == NULL) return;
	if (table->data) nn_free(table->data);
	nn_free(table);
	nn->find_node_opaque = NULL;
}


#if !defined(NN_LOG_MAXLEV) || NN_LOG_MAXLEV >= 1
//
// used by graph-checker to traverse the hash table.
// For each ( node_id, ptr ) in the table representing an actual entry, call
//     callback(  nn, node_id, ptr, info )
//
void traverse_hash_for_check(struct nn_graph *nn,
		void (*callback)(struct nn_graph *, uint32_t nid, struct nn_node *, void *),
		void *info )
{
	struct lookup_info *table = nn->find_node_opaque;
	if (table == NULL) return;
	int n = table->size;
	struct table_data const * td = table->data;
	for( int i =0; i < n; i++, td++){
		if( td->node_id != 0 && td->node != NULL)
			(*callback)( nn, td->node_id, td->node, info);
	}
}

#endif


