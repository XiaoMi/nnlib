
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
#include <nn_graph.h>

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains the code to execute the graph.
 */
#define ITERS 1
//#define ITERS (50*120)

/*
 * Since QuRT (especially) isn't very POSIX, we have a tricky time setting up a mutex.
 * We can't use PTHREAD_MUTEX_INITIALIZER
 * We can't use pthread_once to safely have an initialization hook.
 * So... we're going to assume that zero-initialized mutex means unlocked. 
 */

int do_execute(struct nn_graph *nn,
	const struct tensor *inputs,
	uint32_t n_inputs,
	struct tensor *outputs,
	uint32_t n_outputs)
{
	struct nn_node *node;
	int err = 0;
	uint64_t perf_start;
	uint64_t perf_stop;
	uint64_t pcycle_start;
	uint64_t pcycle_stop;
	int i;
	static nn_mutex_t exec_mutex = NN_MUTEX_INIT;
	nn->inputs = inputs;
	nn->n_inputs = n_inputs;
	nn->outputs = outputs;
	nn->n_outputs = n_outputs;
	nn_mutex_lock(&exec_mutex);
	nn_os_hvx_power_on(nn);
	nn_os_vector_workers_acquire(nn);
	pcycle_start = nn_os_get_cycles(nn);
	for (i = 0; i < ITERS; i++) {
	for (node = nn->head; node != NULL; node = node->next) {
		perf_start = nn_os_get_perfcount(nn);
		if ((err = node->ops->execute(node,nn)) != 0) break;
		perf_stop = nn_os_get_perfcount(nn);
		node->perfcounter += (perf_stop - perf_start);
		node->executions += 1;
	}
	}
	pcycle_stop = nn_os_get_cycles(nn);
	nn->execution_total_cycles = pcycle_stop - pcycle_start;
	nn_os_vector_workers_release(nn);
	nn_os_hvx_power_off(nn);
	nn_mutex_unlock(&exec_mutex);
	return err;
}

