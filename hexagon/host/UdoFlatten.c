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

#include "SnpeUdo/UdoFlatten.h"
#include "rpcmem.h"
#include <string.h>

#define RPCMEM_CONST 25

static int calculateParamSizes (SnpeUdo_Param_t* param, paramSizes_t* sizes, uint32_t* cumulativeSize) {
        if(param->paramName == NULL) {
                sizes->nameStructSize = sizeof(udoString_t);
        } else {
                sizes->nameStructSize = sizeof(udoString_t) + DSP_ALIGN((strlen(param->paramName)+1), DSP_STRUCT_ALIGNMENT);
        }
        sizes->descriptorSize = sizeof(dspStaticParamDescriptor_t) + sizes->nameStructSize - sizeof(udoString_t);
        sizes->dimsSize = 0;
        sizes->dataStructSize = 0;
        sizes->dataSize = 0;
        sizes->stringDataStructSize = 0;
        if(param->paramType == SNPE_UDO_PARAMTYPE_TENSOR) {
                if ((param->tensorParam).layout == SNPE_UDO_LAYOUT_NULL) {
                        (*cumulativeSize) += sizes->descriptorSize;
                        return 0;
                } else if ((param->tensorParam).tensorRank == 0 || (param->tensorParam).maxDimensions==NULL || (param->tensorParam).currDimensions==NULL || (param->tensorParam).tensorData==NULL || ((param->tensorParam).quantizeParams).quantizeType == SNPE_UDO_QUANTIZATION_QMN) {
                        return -1;
                }
                uint32_t rank = (param->tensorParam).tensorRank;
                for (int i=0; i<rank; i++) {
                        if (((param->tensorParam).maxDimensions)[i] != ((param->tensorParam).currDimensions)[i]) {
                                return -1;
                        }
                }
                uint32_t* dimensions = (param->tensorParam).maxDimensions;
                sizes->dimsSize = 2*sizeof(uint32_t) + rank * 2 * sizeof(uint32_t);
                uint32_t elementSize = 0;
                switch((param->tensorParam).dataType) {
                        case SNPE_UDO_DATATYPE_FIXED_4:
                        case SNPE_UDO_DATATYPE_FIXED_8:
                        case SNPE_UDO_DATATYPE_UINT_8:
                        case SNPE_UDO_DATATYPE_INT_8:      elementSize = sizeof(int8_t); break;
                        case SNPE_UDO_DATATYPE_FLOAT_16:
                        case SNPE_UDO_DATATYPE_FIXED_16:
                        case SNPE_UDO_DATATYPE_UINT_16:
                        case SNPE_UDO_DATATYPE_INT_16:     elementSize = sizeof(int16_t); break;
                        case SNPE_UDO_DATATYPE_FLOAT_32:
                        case SNPE_UDO_DATATYPE_FIXED_32:
                        case SNPE_UDO_DATATYPE_UINT_32:
                        case SNPE_UDO_DATATYPE_INT_32:
                        default:                          elementSize = sizeof(int32_t);
                }
                uint32_t dataSize = elementSize;
                for(int i=0; i<rank; i++) {
                        dataSize *= dimensions[i];
                }
                if(dataSize == 0)  return -1;
                sizes->dataSize = dataSize;
                sizes->dataStructSize = sizeof(tensorData_t) + DSP_ALIGN(dataSize, DSP_STRUCT_ALIGNMENT);
                sizes->descriptorSize += sizes->dimsSize + sizes->dataStructSize;
        } else if(param->paramType == SNPE_UDO_PARAMTYPE_STRING) {
                if(param->stringParam == NULL)  return -1;
                sizes->stringDataStructSize = sizeof(udoString_t) + DSP_ALIGN((strlen(param->stringParam)+1), DSP_STRUCT_ALIGNMENT); 
                sizes->descriptorSize += sizes->stringDataStructSize;
        } else if(param->paramType != SNPE_UDO_PARAMTYPE_SCALAR) {
                return -1;
        }
        (*cumulativeSize) += sizes->descriptorSize;
        return 0;
}

int SnpeUdo_flattenStaticParams (SnpeUdo_Param_t** paramList, uint32_t numParams, uint32_t* flattenedSize, void** flattened) {
        uint32_t metaSize = sizeof(dspStaticParamsMeta_t);
        paramSizes_t sizes[numParams];
        for(int i=0; i<numParams; i++) {
                if(calculateParamSizes (paramList[i], &sizes[i], &metaSize) != 0){
                        return -1;
                }
        }   
        *flattened = rpcmem_alloc(RPCMEM_CONST, RPCMEM_DEFAULT_FLAGS, metaSize);
        if (*flattened == NULL)  return -1;
        *flattenedSize = metaSize;
        dspStaticParams_t* fParamsStart = (dspStaticParams_t*)(*flattened);
        (fParamsStart->meta).size = metaSize;
        (fParamsStart->meta).numParams = numParams;
        dspStaticParamDescriptor_t* curDesc = NULL;
        SnpeUdo_Param_t* curParam;
        for(int i=0; i<numParams; i++) {
                curParam = paramList[i];
                if(i==0){
                        curDesc = &(fParamsStart->paramDesc);
                } else {
                        curDesc = (dspStaticParamDescriptor_t*) (((uint8_t*)curDesc)+curDesc->size);
                }
                curDesc->size = sizes[i].descriptorSize;
                curDesc->paramType = curParam->paramType; 
                if(curDesc->paramType == SNPE_UDO_PARAMTYPE_SCALAR) {
                        (curDesc->scalarInfo).dataType = (curParam->scalarParam).dataType; 
                        (curDesc->scalarInfo).dataValue.floatValue = (curParam->scalarParam).dataValue.floatValue;
                } else if(curDesc->paramType == SNPE_UDO_PARAMTYPE_TENSOR) {
                        (curDesc->tensorInfo).layout = (curParam->tensorParam).layout;
                        (curDesc->tensorInfo).quantizeInfo = (curParam->tensorParam).quantizeParams;
                        (curDesc->tensorInfo).dataType = (curParam->tensorParam).dataType; 
                }
                (curDesc->name).sizeStruct = sizes[i].nameStructSize;
                (curDesc->name).lengthString = curParam->paramName? strlen(curParam->paramName):0;
                char* nameStart = ((char*)curDesc) + sizeof(dspStaticParamDescriptor_t);
                if(curParam->paramName) {
                        strncpy(nameStart, curParam->paramName, (curDesc->name).lengthString);
                        nameStart[(curDesc->name).lengthString] = '\0';
                }
                if(curDesc->paramType == SNPE_UDO_PARAMTYPE_TENSOR && (curDesc->tensorInfo).layout!= SNPE_UDO_LAYOUT_NULL) {
                        dims_t* dimsStart = (dims_t*)(nameStart + sizes[i].nameStructSize - sizeof(udoString_t));
                        dimsStart->size = sizes[i].dimsSize;
                        dimsStart->rank = (curParam->tensorParam).tensorRank; 
                        uint32_t rank = dimsStart->rank;
                        uint32_t* ds = &(dimsStart->ds);
                        for(int j=0; j<rank; j++) {
                                ds[j] = (curParam->tensorParam).maxDimensions[j];
                                ds[j+rank] = (curParam->tensorParam).currDimensions[j];
                        }
                        tensorData_t* dataStart = (tensorData_t*)(ds+2*rank);
                        dataStart->structSize = sizes[i].dataStructSize;
                        dataStart->dataSize = sizes[i].dataSize;
                        memcpy(ds+2*rank+2, (curParam->tensorParam).tensorData, sizes[i].dataSize);
               } else if (curDesc->paramType == SNPE_UDO_PARAMTYPE_STRING) {
                        udoString_t* stringDataStart = (udoString_t*)(nameStart + sizes[i].nameStructSize - sizeof(udoString_t));
                        stringDataStart->sizeStruct = sizes[i].stringDataStructSize;
                        stringDataStart->lengthString = strlen(curParam->stringParam);
                        char* stringDataCharStart = ((char*)stringDataStart)+sizeof(udoString_t);
                        strncpy(stringDataCharStart, curParam->stringParam, stringDataStart->lengthString);
                        stringDataCharStart[stringDataStart->lengthString] = '\0';
               }
        }
        return 0;
}

void SnpeUdo_freeFlattenedStaticParams (void** flattened) {
        if (*flattened)  rpcmem_free(*flattened);
        *flattened = NULL;
}

