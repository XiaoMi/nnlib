//==============================================================================
//
// Copyright (c) 2019 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Header to be used by DSP Hexnn

#ifndef SNPE_UDO_HEXNN_IMPL_HEXNN_H
#define SNPE_UDO_HEXNN_IMPL_HEXNN_H
#include "SnpeUdo/UdoImplDspHexNNv2.h"
#include "nn_graph.h"

typedef struct impl_library_t {
        uint32_t index;
        void* lib_handle;
        char* package_name;
        char* op_types;
        int num_ops;
        SnpeUdo_CreateOpFactoryFunction_t create_op_factory;
        SnpeUdo_CreateOperationFunction_t create_operation;
        SnpeUdo_ExecuteOpFunction_t execute_op;
        SnpeUdo_ReleaseOpFunction_t release_op;
        SnpeUdo_ReleaseOpFactoryFunction_t release_op_factory;
        SnpeUdo_TerminateImplLibraryFunction_t terminate_lib;
        SnpeUdo_ValidateOperationFunction_t validate_op;
        SnpeUdo_QueryOperationFunction_t query_op;
        struct impl_library_t* next;
} impl_library;

typedef struct graph_node_t {
        struct nn_node *node;
        struct nn_graph *graph;
} graph_node;

typedef struct thread_structure_data_t {
        void* info;
        nn_sem_t sem;
} thread_structure_data;

typedef struct thread_structure_t {
        thread_structure_data* data;
        workerThread_t function;
} thread_structure;

int initialize_udo_infra();

int check_udo_library_existence(uint32_t lib_id);

impl_library* find_udo_library(const char* package_name, const char* op_type);

#endif // SNPE_UDO_HEXNN_IMPL_HEXNN_H

