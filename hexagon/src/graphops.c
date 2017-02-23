
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
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains operations on the graph
 */

/* Returns the last node in the graph to reference the input. */
/* If no node references the input, producer is returned.  */
#include <nn_graph.h>

static inline void log_causality(
	struct nn_graph *nn, 
	struct nn_node *tmp, 
	struct nn_node *producer)
{
	logmsg(nn,0,
		"CAUSALITY VIOLATION: "
		"node %p (id=0x%x) referenced output of node %p (id=0x%x) "
		"before instantiated in the graph",
		tmp,
		tmp->node_id,
		producer,
		producer->node_id);
}

struct nn_node* find_last_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int out_idx)
{
	struct nn_node *tmp;
	struct nn_node *last_node = producer;
	struct input *in;
	int i;
	int seen_producer = 0;
	uint32_t prod_id = producer->node_id;
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		for (i = 0; i < tmp->n_inputs; i++) {
			in = &tmp->input_refs[i];
			if (in->src_id != prod_id) continue;
			if (in->output_idx != out_idx) continue;
			if (!seen_producer) {
				log_causality(nn,tmp,producer);
			} else {
				last_node = tmp;
			}
		}
		if (tmp == producer) seen_producer = 1;
	}
	return last_node;
}

/* Returns the last node in the graph to reference the input. */
/* If no node references the input, producer is returned.  */

struct nn_node* find_first_consumer(
	struct nn_graph *nn, 
	struct nn_node *producer, 
	int out_idx)
{
	struct nn_node *tmp;
	struct input *in;
	int i;
	int seen_producer = 0;
	uint32_t prod_id = producer->node_id;
	for (tmp = nn->head; tmp != NULL; tmp = tmp->next) {
		for (i = 0; i < tmp->n_inputs; i++) {
			in = &tmp->input_refs[i];
			if (in->src_id != prod_id) continue;
			if (in->output_idx != out_idx) continue;
			if (!seen_producer) {
				log_causality(nn,tmp,producer);
			} else {
				return tmp;
			}
		}
		if (tmp == producer) seen_producer = 1;
	}
	return producer;
}

