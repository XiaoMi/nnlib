
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
#include <nn_graph.h>

/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains the code to execute the graph.
 */
#define ITERS 1
//#define ITERS (50*120)

nn_mutex_t graph_mutex = NN_MUTEX_INIT; // Should we move this to the graph table?
/*
 * Since QuRT (especially) isn't very POSIX, we have a tricky time setting up a mutex.
 * We can't use PTHREAD_MUTEX_INITIALIZER
 * We can't use pthread_once to safely have an initialization hook.
 * So... we're going to assume that zero-initialized mutex means unlocked.
 */

void execute_set_canaries(struct nn_graph *nn, struct nn_node *node)
{
	int i;
	for (i = 0; i < node->n_outputs; i++) {
		canary_mark(nn,node->outputs[i]);
	}
}

void execute_check_src_canaries(struct nn_graph *nn, struct nn_node *node)
{
	int i;
	for (i = 0; i < node->n_inputs; i++) {
		if (canary_check(nn,node->inputs[i]) != 0) {
			logmsg(nn,0,"src canary fail @ node=%p id=%x input=%d (%p @ %p)",node,node->node_id,i,node->inputs[i],&node->inputs[i]);
		}
	}
}

void execute_check_dst_canaries(struct nn_graph *nn, struct nn_node *node)
{
	int i;
	for (i = 0; i < node->n_outputs; i++) {
		if (canary_check(nn,node->outputs[i]) != 0) {
			logmsg(nn,0,"dst canary fail @ node=%p id=%x output=%d (%p)",node,node->node_id,i,node->outputs[i],&node->outputs[i]);
		}
	}
}

int do_execute(struct nn_graph *nn)
{
	struct nn_node *node;
	int err = 0;
	uint64_t perf_start;
	uint64_t perf_stop;
	uint64_t pcycle_node;
	uint64_t pcycle_start;
	uint64_t pcycle_stop;
	uint64_t pcycle_overhead;
	int i;

	struct nn_node *start_node = nn->head;
	struct nn_node *next_node = NULL;
	int saved_priority;
	if (nn_os_update_main_thread_priority(nn, &saved_priority)) return errlog(nn, "priority update failed");
	if (nn->nonconst_head_ptr && *nn->nonconst_head_ptr) start_node = *nn->nonconst_head_ptr;
	nn_mutex_lock(&graph_mutex);
	nn_os_hvx_power_on(nn); // THIS MUST BE CALLED WITHIN MUTEX LOCKED SECTION
	if (nn_os_vtcm_acquire(nn) != 0) {
		nn_os_hvx_power_off(nn); // THIS MUST BE CALLED WITHIN MUTEX LOCKED SECTION
		nn_mutex_unlock(&graph_mutex);
		if (nn_os_restore_main_thread_priority(nn, saved_priority)) errlog(nn, "priority restore failed");
		return errlog(nn,"vtcm acquire error");
	}
	nn_os_vector_workers_acquire(nn);
	pcycle_start = nn_os_get_cycles(nn);
	pcycle_overhead = nn_os_get_cycles(nn) - pcycle_start;
	for (i = 0; i < ITERS; i++) {

	// reset batch sequencing;
	nn_batchseqstate_before_outer_exec(&nn->batchseq);
    nn_loopstack_pre_execute( nn, &nn->loopstack);
	do{
	//print_tensors(inputs, n_inputs);
	for (node = start_node; node != NULL; node = next_node) {
		logmsg(nn,4,"do_execute(): node=%p id=%x, next at %p",node,node->node_id, node->next);
		//execute_check_src_canaries(nn,node);
		//execute_set_canaries(nn,node);
		perf_start = nn_os_get_perfcount(nn);
		pcycle_node = nn_os_get_cycles(nn);
		nn_scratch_reset(nn);
		/* for (int j = 0; j < node->n_inputs; j++) {
			print_tensor(node->inputs[j],"in");
		}*/
		if ((err = node->ops->execute(node,nn)) != 0) {
			errlog(nn,"execute() failed on node id=%x err=%d",node->node_id,err);
			goto quit;
		}
		pcycle_stop = nn_os_get_cycles(nn);
		perf_stop = nn_os_get_perfcount(nn);

		// Print the output tensors, if printing enabled
#if defined(V66)
		if (nn->debug_level && nn->enable_tensor_print) {
			for (int j = 0; j < node->n_outputs; j++) {
				print_tensor_to_file(nn, node->node_id, j, node->outputs[j]);
			}
		}
#endif

		/*
		if ((node->n_outputs > 0) && (node->outputs != NULL)) {
			for (int j = 0; j < node->n_outputs; j++) {
				print_tensor(node->outputs[j],"out");
			}
		}*/
		// show size & checksums of all outputs?
#if !defined(NN_LOG_MAXLEV) || (NN_LOG_MAXLEV >=2)
		if( unlikely( nn_option_get(nn,debug_show_output_tensors)))
			nn_report_node_outputs( nn, 0, node);
#endif
		//execute_check_dst_canaries(nn,node);
		node->perfcounter += (perf_stop - perf_start);
		node->executions += 1;
		node->iter_cycles = pcycle_stop - pcycle_node - pcycle_overhead;
		//print_node_checksum(nn, node);
		next_node = node->next;
		if(next_node == NULL){
			struct nn_loop_end_action endact = nn_loopstack_post_execute( nn, &nn->loopstack);
			if( endact.errcode !=0){
				errlog(nn,"loop update error");
				goto quit;
			}
			next_node = endact.rerun_node;	// NULL if all done
		}
	} // for node list
	}while( nn_batchseqstate_loop_update( &nn->batchseq )); // batch seq loop
	} // for ITERS
  quit:
	nn_os_vector_workers_release(nn);
	nn_os_vtcm_release(nn);
	nn_os_hvx_power_off(nn); // THIS MUST BE CALLED WITHIN MUTEX LOCKED SECTION
	nn_mutex_unlock(&graph_mutex);
	if (nn_os_restore_main_thread_priority(nn, saved_priority)) errlog(nn, "priority restore failed");
	return err;
}

