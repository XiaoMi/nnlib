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


/*
 * structs and enums used in UDO implementation libraries
 */

#include "SnpeUdo/UdoImpl.h"

// structs of op factories

typedef struct opFactoryType1 {
        SnpeUdo_String_t opType;
        // possibly more fields, including ptrs which memory can be allocated during create op factory function calls
} opFactoryType1_t;

typedef struct opFactoryType2 {
        SnpeUdo_String_t opType;
        uint32_t numStaticParams;
        SnpeUdo_Param_t* staticParams;
        // possibly more fields, including ptrs which memory can be allocated during create op factory function calls
} opFactoryType2_t;

// type 3 serves as an example of extra storage in op factories
typedef struct opFactoryType3 {
        SnpeUdo_String_t opType;
        uint32_t numStaticParams;
        SnpeUdo_Param_t* staticParams;
        // possibly more fields, including ptrs which memory can be allocated during create op factory function calls
        uint8_t* someStorage;
} opFactoryType3_t;


// struct for operation instances
typedef struct operationIns {
        SnpeUdo_OpFactory_t opFactory;
        uint32_t numInputParams;
        SnpeUdo_TensorParam_t* inputParams;
        uint32_t numOutputParams;
        SnpeUdo_TensorParam_t* outputParams;
        SnpeUdo_HexNNv2OpInfra_t opInfra;
        // possibly some indicator of vtcm capacity
        uint32_t inputsFitVtcm;  // eg. 0 - inputs cannot fit into vtcm, 1 - inputs can fit into vtcm
        // possibly more fields, including ptrs which memory can be allocated during execute function calls
} operationIns_t;

