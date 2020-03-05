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

#include "udo_impl_dsp_hexnn_internal_v2.h"
#include "nn_graph.h"

#ifdef HEXAGON_V66
#define NUM_MAX_THREADS 4
#else
#define NUM_MAX_THREADS 2
#endif

#define API_VERSION_MAJOR 1
#define API_VERSION_MINOR 3
#define API_VERSION_TEENY 0

#define N_HEXNN_TENSORS_PER_UDO_TENSOR_NONQUANT 1
#define N_HEXNN_TENSORS_PER_UDO_TENSOR_TF 3


SnpeUdo_DspGlobalInfrastructure_t* dspInfra = NULL;


int udo_set_output_tensor_size (void* hexnn_reserve, uint32_t out_idx, uint32_t size){ // returns err code
        graph_node* gn = (graph_node*)hexnn_reserve;
        if (out_idx >= (gn->node->udo_info).udo_num_output_tensors){
                return errlog(gn->graph, "UDO global infrastructure udoSetOutputTensorSize failed. index out of range");
        }
        uint32_t hexnn_out_idx = 0;
        SnpeUdo_QuantizationType_t* q_types = (SnpeUdo_QuantizationType_t*)(gn->node->udo_info).udo_output_q_types;
        for (uint32_t i=0; i<out_idx; i++) {
                if (q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        hexnn_out_idx += N_HEXNN_TENSORS_PER_UDO_TENSOR_NONQUANT;
                } else if (q_types[i] == SNPE_UDO_QUANTIZATION_TF) { 
                        hexnn_out_idx += N_HEXNN_TENSORS_PER_UDO_TENSOR_TF;
                } else {
                        return errlog(gn->graph, "UDO global infrastructure udoSetOutputTensorSize failed. invalid quantization type found");
                }
        }
        struct tensor* out_tensor = gn->node->outputs[hexnn_out_idx];
        if (size > out_tensor->max_size)  return errlog(gn->graph, "UDO global infrastructure udoSetOutputTensorSize failed. size exceeds max size");
        out_tensor->data_size = size;
        return 0;
}

int udo_get_input_d32_paddings (void* hexnn_reserve, uint32_t in_idx,
                                uint32_t* height_pad_before, uint32_t* height_pad_after,
                                uint32_t* width_pad_before, uint32_t* width_pad_after,
                                uint32_t* depth_pad_before, uint32_t* depth_pad_after) {  // returns err code
        graph_node* gn = (graph_node*)hexnn_reserve;
        if (in_idx >= (gn->node->udo_info).udo_num_input_tensors){
                return errlog(gn->graph, "UDO global infrastructure udoGetInputD32Paddings failed. index out of range");
        }
        uint32_t hexnn_in_idx = 0;
        SnpeUdo_QuantizationType_t* q_types = (SnpeUdo_QuantizationType_t*)(gn->node->udo_info).udo_input_q_types;
        for (uint32_t i=0; i<in_idx; i++) {
                if (q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        hexnn_in_idx += N_HEXNN_TENSORS_PER_UDO_TENSOR_NONQUANT;
                } else if (q_types[i] == SNPE_UDO_QUANTIZATION_TF) {
                        hexnn_in_idx += N_HEXNN_TENSORS_PER_UDO_TENSOR_TF;
                } else {
                        return errlog(gn->graph, "UDO global infrastructure udoGetInputD32Paddings failed. invalid quantization type found");
                }
        }
        struct input in_ref = (gn->node->input_refs)[hexnn_in_idx];
        struct nn_node *in_ref_node = gn->graph->head;
        while(in_ref_node) {
                if(in_ref_node->node_id == in_ref.src_id) {
                        break;
                }
                in_ref_node = in_ref_node->next;
        }
        struct tensor *in_tensor = in_ref_node->outputs[in_ref.output_idx];
        *height_pad_before = (uint32_t)(((in_tensor->format).height_pad)[0]);
        *height_pad_after = (uint32_t)(((in_tensor->format).height_pad)[1]);
        *width_pad_before = (uint32_t)(((in_tensor->format).width_pad)[0]);
        *width_pad_after = (uint32_t)(((in_tensor->format).width_pad)[1]);
        *depth_pad_before = (uint32_t)(((in_tensor->format).depth_pad)[0]);
        *depth_pad_after = (uint32_t)(((in_tensor->format).depth_pad)[1]);
        return 0;
}

int udo_set_output_d32_shape_size_paddings (void* hexnn_reserve, uint32_t out_idx,
                            uint32_t batch,
                            uint32_t height, uint32_t height_pad_before, uint32_t height_pad_after,
                            uint32_t width, uint32_t width_pad_before, uint32_t width_pad_after,
                            uint32_t depth, uint32_t depth_pad_before, uint32_t depth_pad_after,
                            SnpeUdo_DataType_t data_type){  // returns err code
        graph_node* gn = (graph_node*)hexnn_reserve;
        if (out_idx >= (gn->node->udo_info).udo_num_output_tensors){
                return errlog(gn->graph, "UDO global infrastructure udoSetOutputD32ShapeSizePaddings failed. index out of range");
        }
        uint32_t hexnn_out_idx = 0;
        SnpeUdo_QuantizationType_t* q_types = (SnpeUdo_QuantizationType_t*)(gn->node->udo_info).udo_output_q_types;
        for (uint32_t i=0; i<out_idx; i++) {
                if (q_types[i] == SNPE_UDO_QUANTIZATION_NONE) {
                        hexnn_out_idx += N_HEXNN_TENSORS_PER_UDO_TENSOR_NONQUANT;
                } else if (q_types[i] == SNPE_UDO_QUANTIZATION_TF) {
                        hexnn_out_idx += N_HEXNN_TENSORS_PER_UDO_TENSOR_TF;
                } else {
                        return errlog(gn->graph, "UDO global infrastructure udoSetOutputD32ShapeSizePaddings failed. invalid quantization type found");
                }
        }
        struct tensor* out_tensor = gn->node->outputs[hexnn_out_idx];
        uint32_t dt;
        switch(data_type) {
                case SNPE_UDO_DATATYPE_FIXED_8:   dt = NN_TYPE_QINT8; break;
                case SNPE_UDO_DATATYPE_FLOAT_16:  dt = NN_TYPE_QINT16; break;
                case SNPE_UDO_DATATYPE_FIXED_16:  dt = NN_TYPE_QINT16; break;
                case SNPE_UDO_DATATYPE_FLOAT_32:  dt = NN_TYPE_FLOAT; break;
                case SNPE_UDO_DATATYPE_FIXED_32:  dt = NN_TYPE_INT32; break;
                default:                          return errlog(gn->graph, "UDO global infrastructure udoSetOutputD32ShapeSizePaddings failed. invalid data type for output tensor");
        }

        uint32_t rmdr = 0;
        if ((rmdr = (depth + depth_pad_before + depth_pad_after) % 32) != 0) {
                return errlog(gn->graph, "UDO global infrastructure udoSetOutputD32ShapeSizePaddings failed. total depth must be multiple of 32");
        }
        if ((rmdr = (width + width_pad_before + width_pad_after) % 4) != 0) {
                return errlog(gn->graph, "UDO global infrastructure udoSetOutputD32ShapeSizePaddings failed. total width must be multiple of 4");
        }

        (((SnpeUdo_TensorParam_t*)((gn->node->udo_info).udo_output_tensors))[out_idx]).dataType = data_type;
        int res;
        if((res = tensor_out_prepare_padded_d32(out_tensor, batch, height, height_pad_before, height_pad_after, 
                                      width, width_pad_before, width_pad_after,
                                      depth, depth_pad_before, depth_pad_after,
                                      dt))!=0) {
                return errlog(gn->graph, "UDO global infrastructure udoSetOutputD32ShapeSizePaddings failed.");
        }
        return 0;
}

void* udo_memalign (size_t n, size_t size){
        return nn_memalign(n, size);
}

void* udo_malloc (size_t size){
        return nn_malloc(size);
}

void* udo_calloc (size_t n, size_t size){
        return nn_calloc(n, size);
}

void udo_free (void* ptr){
        if(ptr)  nn_free(ptr);
}

uint32_t udo_get_vtcm_size (void* hexnn_reserve) {   // available after init
        return (((graph_node*)hexnn_reserve)->graph)->vtcm_size;
}

void* udo_get_vtcm_ptr (void* hexnn_reserve) {  // available only during execute
        return (((graph_node*)hexnn_reserve)->graph)->vtcm_ptr;

}

void threading_function (struct nn_graph* nn, void* t) {
        thread_structure* stru = (thread_structure*)t;
        graph_node gn;
        gn.node = NULL;
        gn.graph = nn;
        (*stru->function) (&gn, stru->data->info);
        nn_sem_post(&(stru->data->sem));
}

void udo_run_worker_threads (void* hexnn_reserve, uint32_t n_threads, workerThread_t w, void* user_data){
        thread_structure t;
        thread_structure_data d;
        t.function = w;
        d.info = user_data;
        nn_sem_init(&(d.sem), 0);
        t.data = &d;
        int n_threads_actual = n_threads<NUM_MAX_THREADS? (int)n_threads:NUM_MAX_THREADS;
        for (int tid=0; tid<n_threads_actual; tid++) {
                nn_os_work_for_vector(((graph_node*)hexnn_reserve)->graph, threading_function, &t);
        }
        nn_sem_wait_n_times(&(t.data->sem), n_threads_actual);
}

int initialize_udo_infra(){

        dspInfra = (SnpeUdo_DspGlobalInfrastructure_t*)nn_malloc(sizeof(SnpeUdo_DspGlobalInfrastructure_t));
        if (dspInfra == NULL)  return -1;
        SnpeUdo_Version_t dsp_ver = {API_VERSION_MAJOR, API_VERSION_MINOR, API_VERSION_TEENY};
        dspInfra->dspInfraVersion = dsp_ver;
        dspInfra->infraType = UDO_INFRA_HEXNN_V2;
        SnpeUdo_HexNNv2GlobalInfra_t* funcs = &(dspInfra->hexNNv2Infra);

        funcs->udoSetOutputTensorSize = udo_set_output_tensor_size;
        funcs->udoGetInputD32Paddings = udo_get_input_d32_paddings;
        funcs->udoSetOutputD32ShapeSizePaddings = udo_set_output_d32_shape_size_paddings;
        funcs->udoMemalign = udo_memalign;
        funcs->udoMalloc = udo_malloc;
        funcs->udoCalloc = udo_calloc;
        funcs->udoFree = udo_free;
        funcs->udoGetVtcmSize = udo_get_vtcm_size;
        funcs->udoGetVtcmPtr = udo_get_vtcm_ptr;
        funcs->udoRunWorkerThreads = udo_run_worker_threads;
        return 0;
}; 

