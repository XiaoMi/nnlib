
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
#ifndef NN_GRAPH_OS_CHECKPOINT_H
#define NN_GRAPH_OS_CHECKPOINT_H 1

#include <stdint.h>
#include "nn_graph_builtin.h"


struct nn_graph;
struct nn_node;

typedef struct nn_checkpoint {
	union {
		struct {
			volatile uint32_t count;
			uint32_t required;
		};
		uint64_t required_count;
	};
	void (*func)(struct nn_graph *nn, struct nn_node *self, void *opaque);
	void *opaque;
} nn_checkpoint_t;

static inline int nn_checkpoint_init(nn_checkpoint_t *checkpoint, uint32_t required, void (*func)(struct nn_graph *, struct nn_node *, void *), void *opaque)
{
	checkpoint->count = 0;
	checkpoint->required = required;
	checkpoint->func = func;
	checkpoint->opaque = opaque;
	return 0;
}

/* Portable version */

#if 0
	/* With lock version */
static inline void nn_checkpoint_arrival(struct nn_graph *nn, nn_checkpoint_t *checkpoint)
{
	int last_arrival = 0;
	nn_mutex_lock(&checkpoint->mutex);
	checkpoint->count++;
	if (checkpoint->count == checkpoint->required) {
		checkpoint->count = 0;
		last_arrival = 1;
	};
	nn_mutex_unlock(&checkpoint->mutex);
	if (last_arrival) checkpoint->func(nn,checkpoint->opaque);
}
#else
static inline void nn_checkpoint_arrival(struct nn_graph *nn, struct nn_node *node, nn_checkpoint_t *checkpoint)
{
	int last_arrival;
	uint32_t old_count;
	uint32_t new_count;
	uint32_t required = checkpoint->required;
	do {
		last_arrival = 0;
		old_count = checkpoint->count;
		new_count = old_count + 1;
		if (new_count == required) {
			new_count = 0;
			last_arrival = 1;
		}
	} while (!__sync_bool_compare_and_swap(&checkpoint->count,old_count,new_count));
	if (last_arrival) checkpoint->func(nn,node,checkpoint->opaque);
}
#endif



#endif
