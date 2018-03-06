
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
 */
#include <stdio.h>
#include <nn_graph.h>
#ifndef NO_VERBOSE
#include "nn_string_map.h"
#endif

#define PRINTF_APPEND(BUF,N,LEN,FMT,...) \
 do { LEN = snprintf(BUF,N,FMT,__VA_ARGS__); N -= LEN; BUF += LEN; if (N == 0) return; } while (0)


void do_snpprint(struct nn_graph *nn, char *buf, uint32_t n)
{
#ifndef NO_VERBOSE
	int len;
	struct nn_node *node;
	unsigned int node_count = 0;
	int i;
	buf[n-1] = '\0';
	n -= 1;
	PRINTF_APPEND(buf,n,len,"nn @ %p: id=0x%lx debug_level=%d\n",nn,nn->id,nn->debug_level);
	for (node = nn->head; node != NULL; node = node->next) {
		const char *padname = padding_type_to_string_alt( node->padding,"**UNKNOWN**");
		const char *opname = op_type_to_string_alt( node->node_type, "**UNKNOWN**");
		PRINTF_APPEND(buf,n,len,"node @ %p: id=0x%x type=0x%x(%s) n_inputs=%d n_outputs=%d padding=%x(%s)\n",
			node,
			(unsigned int)node->node_id, 
			(unsigned int)node->node_type, 
			opname,
			(unsigned int)node->n_inputs, 
			(unsigned int)node->n_outputs,
			node->padding,
			padname);
		if (nn->debug_level > 0) for (i = 0; i < node->n_inputs; i++) {
			PRINTF_APPEND(buf,n,len,"... input %d @ %p <src_id %x out_idx %d>\n",
				i,
				node->inputs[i],
				(unsigned int)node->input_refs[i].src_id,
				(unsigned int)node->input_refs[i].output_idx);
		}
		if (nn->debug_level > 0) for (i = 0; i < node->n_outputs; i++) {
			PRINTF_APPEND(buf,n,len,"... output %d @ %p\n", i,node->outputs[i]);
		}
		node_count += 1;
	}
	PRINTF_APPEND(buf,n,len,"%d nodes total.\n",node_count);
#endif
}

#if STANDALONE_DEBUG
void pprint(struct nn_graph *nn)
{
	const unsigned int bufsize = 1024*1024;
	char *buf = malloc(bufsize);
	do_pprint(nn,buf,bufsize);
	printf("%s",buf);
	free(buf);
}
#endif
