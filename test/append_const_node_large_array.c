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
