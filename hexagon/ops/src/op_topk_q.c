
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
#include <nn_graph.h>
#include <string.h>

#define NUM_VALUES_IN_BYTE 256
#define MAX_BYTE_VALUE 255

// finds topk with a histogram and a map
// to find top k, we need the values of sorted in decreasing order, and use index value(in increasing order) to break ties
// since there are only 256 values that can be represented in bytes (NUM_VALUES_IN_BYTE),
// we can store the the first k values we see of each byte value in a map
// and store how many values of a byte value we have seen in a histogram

static int topk_8_execute(struct nn_node *self, struct nn_graph *nn){
    logmsg(nn,2,"topkq execute. self=%p ",self);

    const struct tensor *input_val_tensor = self->inputs[0];
    const struct tensor *k_tensor = self->inputs[1];
    const struct tensor *input_min_tensor = self->inputs[2];
    const struct tensor *input_max_tensor = self->inputs[3];

    struct tensor *out_val_tensor = self->outputs[0];
    struct tensor *out_idx_tensor = self->outputs[1];
    struct tensor *output_min_tensor = self->outputs[2];
    struct tensor *output_max_tensor = self->outputs[3];

    const uint8_t * in_val = input_val_tensor->data;
    uint8_t * out_val = out_val_tensor->data;
    int32_t * out_idxs = out_idx_tensor->data;
    int32_t given_k = tensor_get_int32(k_tensor,0);

    struct shape outshape = input_val_tensor->shape;
    int32_t row_size = outshape.depth;
    int32_t batch_size = outshape.batches*outshape.height*outshape.width;
    int32_t cur_depth_idx;

    //if depth is smaller than k, use depth instead of k. 
    int32_t k = (row_size < given_k)?row_size:given_k;
    outshape.depth = k;
    if(tensor_out_prepare_normal_fromshape(out_val_tensor, &outshape, NN_TYPE_UINT8)!=0){
        return errlog(nn,"out too small");
    }
    if(tensor_out_prepare_normal_fromshape(out_idx_tensor, &outshape, NN_TYPE_INT32)!=0){
        return errlog(nn,"idx too small");
    }
    if(tensor_set_single_float(output_min_tensor, tensor_get_float(input_min_tensor,0))){
        return errlog(nn,"min too small");
    }
    if(tensor_set_single_float(output_max_tensor, tensor_get_float(input_max_tensor,0))){
        return errlog(nn,"max too small");
    }

    int32_t byte_count_histogram[NUM_VALUES_IN_BYTE] = {0};
    int32_t* byte_count_histogram_last_ptr = byte_count_histogram + MAX_BYTE_VALUE;
    int32_t map_bucket_size = k*sizeof(int32_t);
    int32_t map_size = map_bucket_size*NUM_VALUES_IN_BYTE;
    if (nn_scratch_grow(nn,map_size)){
        return errlog(nn,"scratch too small");
    }
    int32_t* byte_index_map = nn->scratch;
    for (int b=0; b<batch_size; b++, in_val+=row_size){
        //start from the end of the row
        const uint8_t * in_val_row = in_val+row_size-1;

        //zero out the memory before traversing the row again
        memset(byte_index_map,0,map_size);
        memset(byte_count_histogram,0,sizeof(byte_count_histogram));

        //read row front to back, to make larger indices appear first
        for (int r = row_size -1; r >=0; r--, in_val_row--){
            cur_depth_idx = r;
            uint8_t histogram_byte_key = *in_val_row;
            int32_t* byte_count_ptr = byte_count_histogram+histogram_byte_key;
            int32_t byte_count = *byte_count_ptr;

            if (byte_count<k){
                int32_t* idx_in_map_ptr = byte_index_map+(k*histogram_byte_key)+byte_count;// the 
                *idx_in_map_ptr=cur_depth_idx;
                *byte_count_ptr=byte_count+1;
            }else if(histogram_byte_key == MAX_BYTE_VALUE){// shortcutting if k 255s found
                break;
            }
            
        }

        if(*byte_count_histogram_last_ptr==k) {//if shortcutted above
            memset(out_val,MAX_BYTE_VALUE,k);
            memcpy(out_idxs,byte_index_map+k*MAX_BYTE_VALUE,map_bucket_size);
            out_val+=k;
            out_idxs+=k;
            continue;
        }

        int32_t remaining=k;
        int32_t* byte_count_ptr = byte_count_histogram_last_ptr;
        int32_t* idx_in_map_ptr = byte_index_map+k*MAX_BYTE_VALUE;
        //traverse the histogram and map backwards, and write to the output while there's space to write
        //start from 255, end at 0, while not all k values written to output
        for (int byte = MAX_BYTE_VALUE; remaining > 0 && byte >= 0  ; byte--){
            int32_t byte_count =*byte_count_ptr;
            int32_t copy_len = (remaining > byte_count)? byte_count:remaining;
            if(byte_count > 0){
                remaining-=copy_len;

                //since its the same byte value, just write it over and over again
                memset(out_val,byte,copy_len);

                //the indices are already in increasing order in the map, so just copy them
                memcpy(out_idxs,idx_in_map_ptr,copy_len*sizeof(int32_t)); 

                out_val+=copy_len;
                out_idxs+=copy_len;
            }
            //traverse pointer backwards
            byte_count_ptr--;
            idx_in_map_ptr-=k;
        }
    }
    return 0;
}


struct nn_node_ops nn_ops_for_TopK_8 = {
	.execute = topk_8_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(4),
};

