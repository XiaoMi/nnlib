/*
 * Copyright (c) 2019, The Linux Foundation. All rights reserved.
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

#include <stdio.h>
#include "hexagon_nn.h"


#define CONST_CHUNK 4096
// Workaround for Qurt issues sharing large static objects
int hexagon_nn_append_empty_const_node_large_array(
	int nng_id,
	uint32_t node_id,
	uint32_t batches,
	uint32_t height,
	uint32_t width,
	uint32_t depth,
	const uint8_t *data,
	uint32_t data_len) {
	uint32_t offset = 0;
	// Create an empty node
	int err = hexagon_nn_append_empty_const_node(nng_id, node_id, batches, height, width, depth, data_len);
	if (err) {
		printf("ERROR appending const node (%d)\n",err);
		return err;
	}
	// Copy the source-data into the node, one chunk at a time (4kb?)
	for (offset=0; data_len; offset+=CONST_CHUNK) {
		int len = (data_len>CONST_CHUNK) ? CONST_CHUNK : data_len;
//		printf("DEBUG: Appending another %d bytes at %d\n",len,offset);
		err = hexagon_nn_populate_const_node(nng_id, node_id, data+offset, len, offset);
		if (err) {
			printf("ERROR populating const node (%d)\n",err);
			return err;
		}
		data_len -= len;
	}
	return 0;
}
