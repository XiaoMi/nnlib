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

/*
 * This contains implementation for hash table lookup
 */

#include <nn_graph.h>
#include <string.h>
#include <math.h>
#include <quantize.h>

#define OP_HASHTABLE_LOOKUPS_IDX 0
#define OP_HASHTABLE_KEYS_IDX 1
#define OP_HASHTABLE_VALUES_IDX 2
#define OP_HASHTABLE_VALUES_MIN_IDX 3
#define OP_HASHTABLE_VALUES_MAX_IDX 4
#define OP_HASHTABLE_NUM_OPS 5

#define OP_HASHTABLE_OUTPUT_IDX 0
#define OP_HASHTABLE_OUTPUT_MIN_IDX 1
#define OP_HASHTABLE_OUTPUT_MAX_IDX 2
#define OP_HASHTABLE_HITS_IDX 3
#define OP_HASHTABLE_HITS_MIN_IDX 4
#define OP_HASHTABLE_HITS_MAX_IDX 5

#ifdef HEXAGON_V66
#define NUM_MAX_THREADS 4
#else
#define NUM_MAX_THREADS 2
#endif

// data structure for passing info to each thread worker
struct lookups_data {
	int32_t *keys;
	int32_t *lookups;
	uint8_t *values;
	float values_min;
	float values_max;
	int32_t num_keys;
	int32_t num_lookups;
	int32_t values_size_per_batch;
	int32_t lu_per_job;
        int32_t totaljobs;
	int32_t cur_work_unit;
	uint8_t *out;
	uint8_t *hits;
	nn_sem_t donesem;
};

// binary search that returns index of found key in input data array
// returns -1 if key not found
static int32_t bisearch(int32_t *data, int32_t start, int32_t end, int32_t key){
    if (start>end){
	return -1;
    }
    int32_t mid = (start+end)/2;
    int32_t cur = *(data+mid);
    if (cur==key){
	return mid;
    }else if (cur<key){
	return bisearch(data, mid+1, end, key);
    }else{
	return bisearch(data, start, mid-1, key);
    }
}

// worker function that does hashtable lookup for one thread 
static void hashtableLookup_qu8_worker_thread(struct nn_graph *nn, void *thread_data) {
    struct lookups_data *td = thread_data;
    int32_t *lu_data = td->lookups;
    uint8_t *out_data = td->out;
    uint8_t *hits_data = td->hits;
    int32_t njobs = td->totaljobs; 
    int32_t nlus_per_job = td->lu_per_job;
    int32_t bytes_per_batch = (td->values_size_per_batch)*sizeof(uint8_t);
    int32_t start_lu, end_lu, diff, found;
    int job_id;
    uint8_t quantized_zero = quantize_uint8(0.0f, td->values_min, td->values_max);

    // accepts jobs until all jobs are done
    while( job_id = __sync_fetch_and_add( &td->cur_work_unit, 1),  job_id < njobs ){
	start_lu = job_id*nlus_per_job;
	end_lu = min_i32((job_id+1)*nlus_per_job, td->num_lookups);
	diff = end_lu-start_lu;
	int32_t found_ind[diff];

	// finds each lookup in the current job in keys tensor, and defines hits tensor
	for (int32_t i=start_lu; i<end_lu; i++) {
            found = bisearch(td->keys, 0, td->num_keys-1, lu_data[i]);
            found_ind[i-start_lu] = found;
	    if (found>=0 && found<td->num_keys){   // key found
	        hits_data[i] = 1;   // quantized 1
	    }else{   // key not found
		hits_data[i] = 0;   // quantized 0
	    }
	}

	// computes output tensor
	for (int32_t i=0; i<diff; i++) {
	    if (i+1<diff && hits_data[start_lu+i+1]==1){   // next key exists and found
		//prefetch next value to be copied 
		l2pref( (td->values)+found_ind[i+1]*(td->values_size_per_batch), 1, bytes_per_batch, 1);
	    }
	    if (hits_data[start_lu+i]==1){   // key found
                vmemcpy_asm(out_data+(start_lu+i)*(td->values_size_per_batch), (td->values)+found_ind[i]*(td->values_size_per_batch), bytes_per_batch);
            }else{  // key not found, stores 0 in output
                vmemset_asm(out_data+(start_lu+i)*(td->values_size_per_batch), quantized_zero, bytes_per_batch);     // quantized 0
	    }
	}
    }
		
    nn_sem_post(&td->donesem);
}


static int hashtableLookup_qu8_execute(struct nn_node *self, struct nn_graph *nn)
{
    const struct tensor *lookups_tensor = self->inputs[OP_HASHTABLE_LOOKUPS_IDX];
    const struct tensor *keys_tensor = self->inputs[OP_HASHTABLE_KEYS_IDX];
    const struct tensor *values_tensor = self->inputs[OP_HASHTABLE_VALUES_IDX];
    const struct tensor *values_min_tensor = self->inputs[OP_HASHTABLE_VALUES_MIN_IDX];
    const struct tensor *values_max_tensor = self->inputs[OP_HASHTABLE_VALUES_MAX_IDX];

    struct tensor *output_tensor = self->outputs[OP_HASHTABLE_OUTPUT_IDX];
    struct tensor *output_min_tensor = self->outputs[OP_HASHTABLE_OUTPUT_MIN_IDX];
    struct tensor *output_max_tensor = self->outputs[OP_HASHTABLE_OUTPUT_MAX_IDX];
    struct tensor *hits_tensor = self->outputs[OP_HASHTABLE_HITS_IDX];
    struct tensor *hits_min_tensor = self->outputs[OP_HASHTABLE_HITS_MIN_IDX];
    struct tensor *hits_max_tensor = self->outputs[OP_HASHTABLE_HITS_MAX_IDX];

    int32_t num_lookups = lookups_tensor->shape.depth; 
    int32_t num_keys = keys_tensor->shape.depth; 
    int32_t values_height = values_tensor->shape.height;
    int32_t values_width = values_tensor->shape.width;
    int32_t values_depth = values_tensor->shape.depth;
    tensor_set_shape(output_tensor, num_lookups, values_height, values_width, values_depth);
    tensor_set_shape(hits_tensor, 1, 1, 1, num_lookups);

    int32_t* lookups_data = lookups_tensor->data;
    int32_t* keys_data = keys_tensor->data;
    uint8_t* values_data = values_tensor->data;  
    uint8_t* dst = output_tensor->data; 
    uint8_t* hits = hits_tensor->data;   

    int32_t values_per_batch = values_height*values_width*values_depth;
    output_tensor->data_size=num_lookups*values_per_batch;
    hits_tensor->data_size=num_lookups;

    const int32_t lookups_per_job = 10;
    int32_t n_jobs = (int32_t)ceil((float)num_lookups/(float)lookups_per_job);
    int32_t n_threads = min_i32(n_jobs, NUM_MAX_THREADS);

    // initializes thread data
    struct lookups_data lu_tdata;
    lu_tdata.keys = keys_data;
    lu_tdata.lookups = lookups_data;
    lu_tdata.values = values_data;
    lu_tdata.values_min = tensor_get_float(values_min_tensor,0);
    lu_tdata.values_max = tensor_get_float(values_max_tensor,0);
    lu_tdata.out = dst;
    lu_tdata.hits = hits;
    lu_tdata.lu_per_job = lookups_per_job;
    lu_tdata.cur_work_unit = 0;
    lu_tdata.num_keys = num_keys;
    lu_tdata.num_lookups = num_lookups;
    lu_tdata.values_size_per_batch = values_per_batch;
    lu_tdata.totaljobs = n_jobs;
    nn_sem_init(&lu_tdata.donesem, 0);

    // sends workers to run algorithms
    for (int32_t tid=0; tid<n_threads; tid++){
	nn_os_work_for_vector(nn, hashtableLookup_qu8_worker_thread, &lu_tdata);
    }

    nn_sem_wait_n_times(&lu_tdata.donesem, n_threads);
    
    // output tensor min and max should be same as input values tensor
    tensor_copy(output_min_tensor,values_min_tensor);
    tensor_copy(output_max_tensor,values_max_tensor);
    // hits tensor is a 1-d tensor
    // it has fixed min of 0 and max of 255, i.e. scale 1.0 and offset 0
    float* hits_min = hits_min_tensor->data;
    *hits_min = 0.0;
    tensor_set_shape(hits_min_tensor, 1, 1, 1, 1);
    float* hits_max = hits_max_tensor->data;
    *hits_max = 255.0;
    tensor_set_shape(hits_max_tensor, 1, 1, 1, 1);

    return 0;
}


struct nn_node_ops nn_ops_for_QuantizedHashtableLookup_8 = { 
    .execute = hashtableLookup_qu8_execute,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(OP_HASHTABLE_NUM_OPS),
    .n_outputs = NN_IOCOUNT(6),
};

