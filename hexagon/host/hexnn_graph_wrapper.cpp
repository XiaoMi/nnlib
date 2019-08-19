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
#include <tuple>
#include <map>
#include <mutex>
#include <malloc.h>

#include "rpcmem.h"
#include "hexnn_graph_wrapper.hpp"


#define SYSTEM_HEAP_ID      25
#define ROUNDUP_8BYTES(X)   ((X+7)&(~7))

typedef struct _memory_info
{
    unsigned char *poi;
    unsigned char *ion_poi;
    unsigned long size;
    unsigned long num_op;
}memory_info;

static std::mutex graph_mutex;
static std::map<std::tuple<remote_handle64, hexagon_nn_nn_id>, memory_info> cached_graph_param;
static int hexnn_initialized = 0;

inline unsigned long calculate_op_param_size(batch_ops_params params)
{
    unsigned long size = 0;

    size = sizeof(flat_batch_ops_params) - sizeof(unsigned char);
    switch(params.op) {
        case HEXNN_BATCH_OP_APPEND_NODE:
            size += ROUNDUP_8BYTES(params.inputsLen * sizeof(hexagon_nn_input));
            size += ROUNDUP_8BYTES(params.outputsLen * sizeof(hexagon_nn_output));
            break;

        case HEXNN_BATCH_OP_APPEND_CONST_NODE:
            size += ROUNDUP_8BYTES(params.dataLen * sizeof(unsigned char));
            break;

        case HEXNN_BATCH_OP_APPEND_EMPTY_CONST_NODE:
            size += ROUNDUP_8BYTES(8);      // created 8 byte dummy data for alignment
            break;

        case HEXNN_BATCH_OP_POPULATE_CONST_NODE:
            size += ROUNDUP_8BYTES(params.dataLen * sizeof(unsigned char));
            break;

        default:
            // error, should not reach here
            size = 0;
            break;
    }
    return size;
}

inline void setup_params(unsigned char *mem, const batch_ops_params params)
{
    flat_batch_ops_params *fbo_poi = (flat_batch_ops_params*)mem;
    unsigned char *tmp;

    fbo_poi->op = params.op;
    switch(params.op) 
    {
        case HEXNN_BATCH_OP_APPEND_NODE:
            fbo_poi->U.an_params.node_id   = params.node_id;
            fbo_poi->U.an_params.operation = params.operation;
            fbo_poi->U.an_params.padding   = params.padding;

            fbo_poi->U.an_params.inputsLen = params.inputsLen * sizeof(hexagon_nn_input);
            fbo_poi->U.an_params.outputsLen = params.outputsLen * sizeof(hexagon_nn_output);

            // copy inputs
            tmp = fbo_poi->c;
            memcpy(tmp, (unsigned char*)params.inputs, fbo_poi->U.an_params.inputsLen);

            //copy outputs 
            tmp += ROUNDUP_8BYTES(fbo_poi->U.an_params.inputsLen);
            memcpy(tmp, (unsigned char*)params.outputs, fbo_poi->U.an_params.outputsLen);           

            break;

        case HEXNN_BATCH_OP_APPEND_CONST_NODE:
            fbo_poi->U.acn_params.node_id = params.node_id;
            fbo_poi->U.acn_params.batches = params.batches;
            fbo_poi->U.acn_params.height  = params.height;
            fbo_poi->U.acn_params.width   = params.width;
            fbo_poi->U.acn_params.depth   = params.depth;

            fbo_poi->U.acn_params.dataLen = params.dataLen * sizeof(unsigned char);

            //copy data
            tmp = fbo_poi->c;
            memcpy(tmp, (unsigned char*)params.data, fbo_poi->U.acn_params.dataLen);

            break;

        case HEXNN_BATCH_OP_APPEND_EMPTY_CONST_NODE:
            fbo_poi->U.aecn_params.node_id = params.node_id;       
            fbo_poi->U.aecn_params.batches = params.batches;
            fbo_poi->U.aecn_params.height  = params.height;
            fbo_poi->U.aecn_params.width   = params.width;
            fbo_poi->U.aecn_params.depth   = params.depth;
            fbo_poi->U.aecn_params.size    = params.size;

            break;

        case HEXNN_BATCH_OP_POPULATE_CONST_NODE:
            fbo_poi->U.pcn_params.node_id       = params.node_id;     
            fbo_poi->U.pcn_params.target_offset = params.target_offset;

            fbo_poi->U.pcn_params.dataLen       = params.dataLen * sizeof(unsigned char);

            //copy data
            tmp = fbo_poi->c;
            memcpy(tmp, (unsigned char*)params.data, fbo_poi->U.pcn_params.dataLen);
 
            break;

        default:
            // error, should not reach here
            break;
    }

}

int batch_append_ops(remote_handle64 h, hexagon_nn_nn_id id, batch_ops_params params)
{
    int sts = 0;

    std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);

    graph_mutex.lock();
    if (cached_graph_param.find(tup) == cached_graph_param.end()){
        // something is wrong, we should already setup mode during prepare
        // host does not call add_nn_id()
        sts = -1;
    }
    else{
        // insert the OP to vector list
        auto info  = &cached_graph_param[tup];
        
        unsigned long size;
        size = calculate_op_param_size(params);
        if(size == 0)
        {
            // error, the size should always none zero
            sts = -1;
        }
        else
        {
            if( info->poi == NULL ) 
            {   
                info->poi = (unsigned char*)malloc(size);
                memset(info->poi, 0, info->size );
                info->size = 0;
                info->num_op = 0;
            }
            else
            {
                info->poi = (unsigned char*)realloc(info->poi, (info->size + size));
                memset((info->poi+info->size), 0, size);
            }
        }

        // insert ops params
        unsigned char *poi = (info->poi + (info->size));
        setup_params(poi , params);

        info->size += size;
        info->num_op += 1;
    }
    graph_mutex.unlock();
    
    return sts;
}

int add_nn_id(remote_handle64 h, hexagon_nn_nn_id id)
{
    int sts = 0;

    graph_mutex.lock();
    if(hexnn_initialized == 0)
    {
        // initlized the map first
        cached_graph_param.clear();

        hexnn_initialized = 1; // initialized
    }

    std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
    if (cached_graph_param.find(tup) == cached_graph_param.end()) {
        //ok, this is new node, i insert it
        memory_info info={0};
        cached_graph_param[tup] = info;
    }
    else
    {
        // something is wrong, we should never add same node twice
        sts = -1;
    }
    graph_mutex.unlock();

    return sts;
}

int remove_nn_id(remote_handle64 h, hexagon_nn_nn_id id)
{
    int sts = 0;

    graph_mutex.lock();
    std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
    auto it = cached_graph_param.find(tup);
    if (it == cached_graph_param.end()) {
        //something is worng, we should never remove node twice.
        // but since it is not there,do nothing for now
    }
    else{
        memory_info *info = &cached_graph_param[tup];
        if(info->poi != NULL) {
            free_batch_op_memory(h, id);
        }

        // remove the node
        cached_graph_param.erase(it);
    }
    graph_mutex.unlock();

    return sts;
}

int copy_batch_ops_to_ion_memory(remote_handle64 h, hexagon_nn_nn_id id, unsigned char **poi, unsigned int *size)
{
    int sts = 0;

    // copy data to ion
    std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
    auto it = cached_graph_param.find(tup);
    if (it == cached_graph_param.end()) {
        // something is wrong, i should always find it
        sts = -1;
    }
    else
    {
        auto info = &cached_graph_param[tup];

        *poi = info->ion_poi = (unsigned char*)rpcmem_alloc(SYSTEM_HEAP_ID, RPCMEM_DEFAULT_FLAGS, info->size);

        if (*poi == NULL)
        {
            // can not allocate
            sts = -1;
            *size = 0;
        }
        else {
            memcpy(info->ion_poi, info->poi, info->size);
            *size = info->size;
        }
    }
    return sts;
}

int free_batch_op_memory(remote_handle64 h, hexagon_nn_nn_id id)
{
    int sts = 0;

    std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
    auto it = cached_graph_param.find(tup);
    if (it == cached_graph_param.end()) {
        // something is wrong, i should always find it
        sts = -1;
    }
    else
    {
        std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
        auto info = &cached_graph_param[tup];

        if(info->poi != NULL){
            free(info->poi);
            info->poi = NULL;
        }

        if(info->ion_poi != NULL){
            rpcmem_free(info->ion_poi);
            info->ion_poi = NULL;
        }
        info->size = 0;
        info->num_op = 0;
    }

    return sts;
}

#ifdef GRAPH_WRAPPER_SAVE_TO_DISK
int write_graph_to_disk(remote_handle64 h, hexagon_nn_nn_id id)
{
    int sts = 0;

    std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
    auto it = cached_graph_param.find(tup);
    if (it == cached_graph_param.end()) {
        // something is wrong, i should always find it
        sts = -1;
    }
    else
    {
        std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
        auto info = &cached_graph_param[tup];


        std::fstream ofile("/data/hexnn_graph_content.data", std::fstream::out | std::fstream::binary);

        //write header
        ofile.write((char*)info, sizeof(memory_info));
        //write graph content
        ofile.write((char*)info->poi, info->size);

        ofile.close();
    }

    return sts;

}
#endif

#ifdef GRAPH_WRAPPER_READ_FROM_DISK
int read_graph_from_disk(remote_handle64 h, hexagon_nn_nn_id id)
{
    int sts = 0;

    std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
    auto it = cached_graph_param.find(tup);
    if (it == cached_graph_param.end()) {
        // something is wrong, i should always find it
        sts = -1;
    }
    else
    {
        std::tuple<remote_handle64, hexagon_nn_nn_id> tup(h, id);
        auto info = &cached_graph_param[tup];

        std::fstream ifile("/data/hexnn_graph_content.data", std::fstream::in | std::fstream::binary);

        //read header
        ifile.read((char*)info, sizeof(memory_info));
        //read graph content
        info->poi = (unsigned char*)malloc(info->size);
        ifile.read((char*)info->poi, info->size);

        ifile.close();
    }

    return sts;
}
#endif
