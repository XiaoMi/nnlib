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

struct topk_f_element{
    float value;
    int32_t tensor_index;
};

#define PARENT_IDX(IDX) ((IDX-1)/2)
#define LEFT_CHILD_IDX(IDX) (2*IDX+1)
#define RIGHT_CHILD_IDX(IDX) (2*IDX+2)

struct topk_f_heap
{
    struct topk_f_element* list; 
    int32_t len;
    int32_t k; 
    int32_t capacity_size;
};

static inline void  topk_f_swap_elements(struct topk_f_heap * heap,int i, int j){
    struct topk_f_element* a = heap->list+i;
    struct topk_f_element* b = heap->list+j;
    struct topk_f_element t[1];
    memcpy(t, a, sizeof(struct topk_f_element));
    memcpy(a, b, sizeof(struct topk_f_element));
    memcpy(b, t, sizeof(struct topk_f_element));
}

int static inline topk_f_less_than(struct topk_f_element *a, struct topk_f_element *b){
	if (b->value > a->value){
		return 1;
	}
	if (b->value < a->value){
		return -1;
	}
	if (b->tensor_index > a->tensor_index){
		return 1;
	}
    if (b->tensor_index  < a->tensor_index){
        return -1;
    }
    return 0;
}

void __attribute__((unused))printheap(struct nn_graph* nn, struct topk_f_heap * heap){
    logmsg(nn,2,"printing heap of len %ld",heap->len);
    for (int i = 0; i< heap->len;i++){
       logmsg(nn,2,"[%d]=(value=%f,id=%ld)",i,heap->list[i].value,heap->list[i].tensor_index);
       int inum=i+2;
       if((inum&(inum-1))==0){
        logmsg(nn,2,"");
       }
    }
    logmsg(nn,2,"------------------");
}

static inline void topk_f_clear_heap(struct topk_f_heap * heap){
	memset(heap->list,0,heap->capacity_size);
    heap->len = 0;
}

static inline void topk_f_heapify(struct topk_f_heap * heap, int i){
    int l = LEFT_CHILD_IDX(i);
    int r = RIGHT_CHILD_IDX(i);
    int smallest;
    if (l < heap->len && topk_f_less_than(heap->list+l,heap->list+i)==1){
        //left child value is smaller current value in min-heap
        smallest = l;
    } else{
        smallest = i;
    }
    if (r < heap->len && topk_f_less_than(heap->list+r,heap->list+smallest)==1){
        //right child value is smaller current value in min-heap
        smallest = r;
    }
    if (smallest != i){
        topk_f_swap_elements(heap, smallest, i);
        topk_f_heapify(heap, smallest);
    }
}

static inline void topk_f_remove_min_from_heap(struct topk_f_heap * heap){
    heap->len--;
    topk_f_swap_elements(heap,heap->len,0);
    topk_f_heapify(heap,0);
}

static inline void topk_f_push_value_to_heap(struct topk_f_heap * heap,float value, int32_t index){
    int i = heap->len;
    heap->list[i].value = value;
    heap->list[i].tensor_index = index;
    heap->len++;
    int p = PARENT_IDX(i);
    while (i != 0 && topk_f_less_than(heap->list+i,heap->list+p)==1) {
        //current value is smaller than parent value in min-heap
        topk_f_swap_elements(heap,p,i);
        i = p;
        p = PARENT_IDX(i);
   }
}

static inline void topk_f_add_value(struct topk_f_heap * heap,float value, int32_t index){
    if(heap->len<=heap->k){
        topk_f_push_value_to_heap(heap,value,index);
    }
    else if(heap->list[0].value < value){
        topk_f_push_value_to_heap(heap,value,index);
        topk_f_remove_min_from_heap(heap);
    }
}

static inline void topk_sort_heap(struct topk_f_heap * heap){
    while(heap->len!=0){
        topk_f_remove_min_from_heap(heap);
    }
}

static inline void topk_get_sorted_result(struct topk_f_heap* heap, float* data,int32_t* idx){
    struct topk_f_element* copy_ptr = heap->list;
    int32_t copy_num = heap->k;
    topk_sort_heap(heap);
    for (int i =0;i<copy_num;i++){
        *data = copy_ptr->value;
        *idx = copy_ptr->tensor_index;
        data++;
        idx++;
        copy_ptr++;
    }
}

static int topk_f_execute(struct nn_node *self, struct nn_graph *nn){
    logmsg(nn,2,"topkf execute. self=%p ",self);
	const struct tensor *input_val_tensor = self->inputs[0];
    const struct tensor *k_tensor = self->inputs[1];
    int32_t given_k = tensor_get_int32(k_tensor,0);
    struct tensor *out_val_tensor = self->outputs[0];
    struct tensor *out_idx_tensor = self->outputs[1];
    float * in_val = input_val_tensor->data;
    struct shape outshape = input_val_tensor->shape;
    int32_t row_size=outshape.depth;
    int32_t batch_size=outshape.batches*outshape.height*outshape.width;
    int32_t cur_idx;
    int32_t k = (row_size < given_k)?row_size:given_k;
    outshape.depth = k;
    if( tensor_out_prepare_normal_fromshape( out_val_tensor, &outshape, NN_TYPE_FLOAT)!=0 ){
        return errlog(nn,"out too small");
    }
    if( tensor_out_prepare_normal_fromshape( out_idx_tensor, &outshape, NN_TYPE_INT32)!=0 ){
        return errlog(nn,"out too small");
    }
    struct topk_f_heap heap;
    heap.k = k;
    heap.capacity_size = (k+2)*sizeof(struct topk_f_element);
    if (nn_scratch_grow(nn,heap.capacity_size)){
        return errlog(nn,"scratch too small");
    }
    heap.list = nn->scratch;
    float * out_val = out_val_tensor->data;
    int32_t * out_idxs = out_idx_tensor->data;
    for (int b =0; b<batch_size; b++){
        topk_f_clear_heap(&heap);
        for (int r =0; r < row_size; r++){
            cur_idx = r;
            //cur_idx = b*row_size +r;
            topk_f_add_value(&heap,*in_val,cur_idx);
            in_val++;
        }
        topk_get_sorted_result(&heap,out_val,out_idxs);
        out_val+=k;
        out_idxs+=k;
    }
	return 0;
}


struct nn_node_ops nn_ops_for_TopK_f = {
	.execute = topk_f_execute,
	.check = NULL,
	.ctor = node_alloc_common,
	.dtor = node_free_common,
	.n_inputs = NN_IOCOUNT(2),
	.n_outputs = NN_IOCOUNT(2),
};

