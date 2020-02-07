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

#include <string.h>
#include <math.h>
#include "SnpeUdo/UdoImplDspHexNNv2.h"
#include "udoExampleImplLib.h"

#define NUM_OPS 2
#define min_i32(a,b) (a<b?a:b)

// implementation library info
SnpeUdo_HexNNv2GlobalInfra_t* infra = NULL;
SnpeUdo_LibVersion_t ver = {{1, 0, 0}, {1, 3, 0}};  // lib version, api version
SnpeUdo_HexNNInfraType_t iType= UDO_INFRA_HEXNN_V2;
char udoPackageName[] = "udoExampleLib";
char udoOpTypes[] = "opUdoPlusOne opUdoPlusStat";
SnpeUdo_ImpInfo_t libInfo = {SNPE_UDO_CORETYPE_DSP, udoPackageName, udoOpTypes, NUM_OPS};

// ops info

// op 1
char udoOpType1[] = "opUdoPlusOne";
uint32_t numStatic1 = 0;
uint32_t numIn1 = 1;
uint32_t numOut1 = 1;
SnpeUdo_QuantizationType_t inQType1 [] = {SNPE_UDO_QUANTIZATION_NONE};
SnpeUdo_QuantizationType_t outQType1 [] = {SNPE_UDO_QUANTIZATION_NONE};
SnpeUdo_HexNNTensorLayout_t* inLayout1 = NULL;   // plain tensor, no d32
SnpeUdo_HexNNTensorLayout_t* outLayout1 = NULL;

// op 2
char udoOpType2[] = "opUdoPlusStat";
// num static parameters can be 1 or 2
uint32_t numStatic2_1 = 1;
uint32_t numStatic2_2 = 2;   // second static parameter is optional, op uses d32 format if 2nd parameter is set to value 1
uint32_t numIn2 = 1;
uint32_t numOut2 = 1;
SnpeUdo_QuantizationType_t inQType2 [] = {SNPE_UDO_QUANTIZATION_TF};
SnpeUdo_QuantizationType_t outQType2 [] = {SNPE_UDO_QUANTIZATION_TF};
// non d32 version, default
SnpeUdo_HexNNTensorLayout_t inLayout2_1 [] = {SNPE_UDO_DSP_TENSOR_LAYOUT_PLAIN};   // plain tensor, no d32
SnpeUdo_HexNNTensorLayout_t outLayout2_1 [] = {SNPE_UDO_DSP_TENSOR_LAYOUT_PLAIN};
// d32 version
SnpeUdo_HexNNTensorLayout_t inLayout2_2 [] = {SNPE_UDO_DSP_TENSOR_LAYOUT_D32};
SnpeUdo_HexNNTensorLayout_t outLayout2_2 [] = {SNPE_UDO_DSP_TENSOR_LAYOUT_D32};


// enum of op types
typedef enum opTypeEnum {
        OP_UDO_PLUS_ONE,
        OP_UDO_PLUS_STAT
} opTypeEnum_t;

// enum of op factories, represents factory types defined in udoExampleImplLib.h
typedef enum opFactoryTypes {
        OP_FACTORY_TYPE_1,
        OP_FACTORY_TYPE_2,
        OP_FACTORY_TYPE_3
} opFactoryTypes_t;

// structs for managing and keeping track of op factories

typedef struct opFactoryNode {
        opFactoryTypes_t type;
        SnpeUdo_OpFactory_t opFactory;
        struct opFactoryNode* next;
} opFactoryNode_t;

typedef struct opFactoryHeadTail {
        opFactoryNode_t* head;
        opFactoryNode_t* tail;
} opFactoryHeadTail_t;

// mutex lock each op factory list
typedef struct opFactoriesPerGraph {
        unsigned long gId;
        opFactoryHeadTail_t opFactoryList[NUM_OPS];
        struct opFactoriesPerGraph* next;
} opFactoriesPerGraph_t;

opFactoryHeadTail_t opFactoryNullList = {NULL, NULL};
opFactoriesPerGraph_t* opFactoriesAll = NULL;


int libInitialized = 0;


SnpeUdo_ErrorType_t SnpeUdo_getVersion (SnpeUdo_LibVersion_t** version){
        *version = &ver;
        return SNPE_UDO_NO_ERROR;
}


SnpeUdo_ErrorType_t SnpeUdo_getImpInfo (SnpeUdo_ImpInfo_t** implementationInfo){
        *implementationInfo = &libInfo;
        return SNPE_UDO_NO_ERROR;
}

SnpeUdo_ErrorType_t SnpeUdo_initImplLibrary (void* globalInfrastructure){
        if (libInitialized==0) {  // check if global variables are not mem alloc'ed more than one time without being freed in case this function is called multiple times
                SnpeUdo_DspGlobalInfrastructure_t* hInfra = (SnpeUdo_DspGlobalInfrastructure_t*)globalInfrastructure;
                SnpeUdo_Version_t hVer = hInfra->dspInfraVersion;
                SnpeUdo_Version_t apiVer = ver.apiVersion;
                if(hVer.major==apiVer.major && hVer.minor==apiVer.minor && hVer.teeny==apiVer.teeny && hInfra->infraType==iType){
                        infra = &(hInfra->hexNNv2Infra);
                        // possibly allocate memory and initialize some udo global variables, maybe based on global infrastructure
                        // mutex lock/unlock global variables here for thread safety
                        libInitialized = 1;
                        return SNPE_UDO_NO_ERROR;
                }
                return SNPE_UDO_INVALID_ARGUMENT;    
        }
        return SNPE_UDO_NO_ERROR;
}


// op specific execution struct for storing inputs, outputs and multithreading info
typedef struct plusSomeInfo {
        uint8_t* in;
        uint32_t inLen;
        uint8_t* out;
        uint8_t increment;
        uint32_t nJobs;
        uint32_t curJobUnit;
        uint32_t elePerJob;
} plusSomeInfo_t;

/*
 * function to be passed into multi threading infrastructure function
 * each thread calls this same function
 */
void workerThreadUdoPlusSome (void* perOpInfrastructure, void* userData){
        plusSomeInfo_t* data = (plusSomeInfo_t*)userData;
        uint8_t* thisIn = data->in;
        uint8_t* thisOut = data->out;
        uint8_t thisIncrement = data->increment;
        uint32_t nJobs = data->nJobs; 
        uint32_t elePerJob = data->elePerJob; 
        int jobId, startEle, endEle;

        // all jobs share same job pool
        while( jobId = __sync_fetch_and_add(&data->curJobUnit, 1),  jobId < nJobs ){
//                printf("JOB   %d\n", jobId);
                startEle = jobId*elePerJob;
                endEle = min_i32((jobId+1)*elePerJob, data->inLen); 
                for(int j=startEle;j<endEle;j++){
                        thisOut[j] = thisIn[j]+thisIncrement;
                }
        }
}

/*
 * function called in execute function
 * single input, single output
 * copies each uint8 number from input and pluses inc, and then saves to the output
 */
SnpeUdo_ErrorType_t opUdoPlusSome (SnpeUdo_Operation_t operation, uint8_t inc, uint32_t elePerJob, uint32_t nThreads, uint32_t useD32) {
        operationIns_t* thisOpIns = (operationIns_t*)operation;
        SnpeUdo_TensorParam_t* in = thisOpIns->inputParams;
        SnpeUdo_TensorParam_t* out = thisOpIns->outputParams;

        if (in->layout == SNPE_UDO_LAYOUT_NULL || out->layout == SNPE_UDO_LAYOUT_NULL)  return SNPE_UDO_UNSUPPORTED_FEATURE;
        if (in->dataType != SNPE_UDO_DATATYPE_FIXED_8 || out->dataType != SNPE_UDO_DATATYPE_FIXED_8)  return SNPE_UDO_UNSUPPORTED_FEATURE;
        uint32_t inLen = 1;
        uint32_t inSize = (uint32_t)sizeof(uint8_t);

        uint32_t heightPadBefore = 0, heightPadAfter = 0, widthPadBefore = 0, widthPadAfter = 0, depthPadBefore = 0, depthPadAfter = 0;
        if (useD32) {
                if ((infra->udoGetInputD32Paddings)(thisOpIns->opInfra, 0, &heightPadBefore, &heightPadAfter, &widthPadBefore, &widthPadAfter, &depthPadBefore, &depthPadAfter)!=0) {
                        return SNPE_UDO_UNSUPPORTED_FEATURE;
                }
        }
        uint32_t paddings[] = {0, heightPadBefore+heightPadBefore, widthPadBefore+widthPadAfter, depthPadBefore+depthPadAfter};

        for(int k=0; k<in->tensorRank; k++){
                inLen *= (in->currDimensions[k] + paddings[k]);
                if (in->currDimensions[k] > out->maxDimensions[k]) {
                        return SNPE_UDO_UNSUPPORTED_FEATURE;   
                }
                out->currDimensions[k] = in->currDimensions[k];
        }
        inSize *= inLen;
        uint32_t nJobs = (uint32_t)ceil((float)inLen/(float)elePerJob);
        uint32_t maxNThreads = min_i32(nJobs,nThreads);
        
        // uses vtcm when possible for accelerated performance
        // data on vtcm gets overwritten during each execute
        uint8_t* src;
        void* vtcmPtr = (infra->udoGetVtcmPtr)(thisOpIns->opInfra);
        if (vtcmPtr && useD32 == 0 && thisOpIns->inputsFitVtcm) {  // uses vtcm for fast input data access
                memcpy(vtcmPtr, in->tensorData, inSize);  // possibly there are faster methods for memcpy, eg. HVX
                src = (uint8_t*)vtcmPtr;
        } else {  // vtcm not available or does not have enough capacity
                src = (uint8_t*)(in->tensorData);
        }
        uint8_t* dst = (uint8_t*)(out->tensorData);

        if (useD32) {
                if ((infra->udoSetOutputD32ShapeSizePaddings)(thisOpIns->opInfra, 0, out->currDimensions[0], out->currDimensions[1], heightPadBefore, heightPadAfter, out->currDimensions[2],  widthPadBefore, widthPadAfter, out->currDimensions[3], depthPadBefore, depthPadAfter, SNPE_UDO_DATATYPE_FIXED_8)!=0) {
                        return SNPE_UDO_UNSUPPORTED_FEATURE;
                }
        } else {
                out->dataType = SNPE_UDO_DATATYPE_FIXED_8;
                if((*(infra->udoSetOutputTensorSize))(thisOpIns->opInfra, 0, inSize)!=0){  // required to set output tensor sizes
                        return SNPE_UDO_UNSUPPORTED_FEATURE;
                }
        }

        plusSomeInfo_t i = {src, inLen, dst, inc, nJobs, 0, elePerJob};
        (*(infra->udoRunWorkerThreads))(thisOpIns->opInfra, maxNThreads, workerThreadUdoPlusSome, &i);
        return SNPE_UDO_NO_ERROR;
}

/*
 * op 1 execute
 * op 1 - input and output are not quantized
 */
SnpeUdo_ErrorType_t opUdoPlusOne (SnpeUdo_Operation_t operation) {
        return opUdoPlusSome(operation, 1, 2, 4, 0);
}

/*
 * op 2 execute
 * op 2 - input and output are quantized
 */
SnpeUdo_ErrorType_t opUdoPlusStat (SnpeUdo_Operation_t operation) {
        operationIns_t* thisOpIns = (operationIns_t*)operation;
        opFactoryType2_t* thisOf = (opFactoryType2_t*)(thisOpIns->opFactory);
        SnpeUdo_Param_t* thisStatic = thisOf->staticParams;
        uint8_t incr = ((thisStatic[0].scalarParam).dataValue).uint8Value;
        uint32_t useD32 = 0;
        if (thisOf->numStaticParams == numStatic2_2 && thisStatic[1].scalarParam.dataValue.uint8Value != 0)  useD32 = 1;   // d32 format

        return opUdoPlusSome(operation, incr, 3, 4, useD32);
}


SnpeUdo_ErrorType_t SnpeUdo_validateOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams, const SnpeUdo_Param_t* staticParams) {
        if(strcmp(operationType, udoOpType1) == 0) {
                if (numOfStaticParams != numStatic1) {
                        return SNPE_UDO_UNSUPPORTED_FEATURE;
                }
        } else if (strcmp(operationType, udoOpType2) == 0) {
                if ((numOfStaticParams != numStatic2_1 && numOfStaticParams != numStatic2_2) || staticParams == NULL) {
                        return SNPE_UDO_UNSUPPORTED_FEATURE;
                }
                if (staticParams[0].paramType != SNPE_UDO_PARAMTYPE_SCALAR || (staticParams[0].scalarParam).dataType != SNPE_UDO_DATATYPE_UINT_8) {
                        return SNPE_UDO_UNSUPPORTED_FEATURE;
                }
                if (numOfStaticParams == numStatic2_2) {
                        if (staticParams[1].paramType != SNPE_UDO_PARAMTYPE_SCALAR || (staticParams[1].scalarParam).dataType != SNPE_UDO_DATATYPE_UINT_8) {
                                return SNPE_UDO_UNSUPPORTED_FEATURE;
                        }
                }
        } else {
                return SNPE_UDO_WRONG_OPERATION;
        }
        return SNPE_UDO_NO_ERROR;
}


SnpeUdo_ErrorType_t SnpeUdo_queryOperation (SnpeUdo_String_t operationType, uint32_t numOfStaticParams, const SnpeUdo_Param_t* staticParams, uint32_t* numOfInputs, SnpeUdo_QuantizationType_t** inputsQuantTypes, SnpeUdo_HexNNTensorLayout_t** inputsLayouts, uint32_t* numOfOutputs, SnpeUdo_QuantizationType_t** outputsQuantTypes, SnpeUdo_HexNNTensorLayout_t** outputsLayouts) {
        if(strcmp(operationType, udoOpType1) == 0) {
                *numOfInputs = numIn1;
                *inputsQuantTypes = inQType1;
                *inputsLayouts = inLayout1;
                *numOfOutputs = numOut1;
                *outputsQuantTypes = outQType1;
                *outputsLayouts = outLayout1;
        } else if (strcmp(operationType, udoOpType2) == 0) {
                // possibly uses static parameters to dynamically determine inputs and outputs info
                *numOfInputs = numIn2;
                *inputsQuantTypes = inQType2;
                *numOfOutputs = numOut2;
                *outputsQuantTypes = outQType2;
                if ((numOfStaticParams == numStatic2_1) || (numOfStaticParams == numStatic2_2 && staticParams[1].scalarParam.dataValue.uint8Value == 0)) {
                        *inputsLayouts = inLayout2_1;
                        *outputsLayouts = outLayout2_1;
                } else {
                        *inputsLayouts = inLayout2_2;
                        *outputsLayouts = outLayout2_2;
                }
        } else {
                return SNPE_UDO_WRONG_OPERATION;   
        }
        return SNPE_UDO_NO_ERROR;
}


/*
 * Function for managing and keeping track of op factories created by this udo library.
 * Op factories are grouped by graph id's and further grouped by operation types.
 * The purpose of keeping track of op factories is enabling checking existing op factories and avoiding duplicated factories being created.
 * Checking for duplicated op factories is demonstrated for opUdoPlusOne as an example
 */
int appendToOpFactoryList (SnpeUdo_OpFactory_t* of, unsigned long graphId, opTypeEnum_t opType, opFactoryTypes_t ofType) {
        opFactoryNode_t* newOfNode = (*(infra->udoMalloc))(sizeof(opFactoryNode_t));
        newOfNode->type = ofType;
        newOfNode->opFactory = *of;
        newOfNode->next = NULL;
        opFactoryHeadTail_t* opFactoryList;
        opFactoryNode_t* cur = NULL;

        if (opFactoriesAll == NULL) {
                opFactoriesAll = (*(infra->udoMalloc))(sizeof(opFactoriesPerGraph_t));
                opFactoriesAll->gId = graphId;
                opFactoriesAll->opFactoryList[0] = opFactoryNullList;
                opFactoriesAll->opFactoryList[1] = opFactoryNullList;
                opFactoriesAll->next = NULL;
                opFactoryList = opFactoriesAll->opFactoryList;
        } else {
                opFactoriesPerGraph_t* curG = opFactoriesAll;
                while (curG && curG->next && curG->gId!=graphId) {
                        curG = curG->next;
                }
                if (curG->gId==graphId) {
                        opFactoryList = curG->opFactoryList;
                } else {
                        opFactoriesPerGraph_t* newG = (*(infra->udoMalloc))(sizeof(opFactoriesPerGraph_t));
                        newG->gId = graphId;
                        newG->opFactoryList[0] = opFactoryNullList;
                        newG->opFactoryList[1] = opFactoryNullList;
                        newG->next = NULL;
                        curG->next = newG;
                        opFactoryList = newG->opFactoryList;
                }
        }

        if (opFactoryList[opType].head == NULL) {
                opFactoryList[opType].head = newOfNode;
                opFactoryList[opType].tail = newOfNode;
        } else {
                // check for existing op factories with same op type and static params and reuse them
                cur = opFactoryList[opType].head;
                if (ofType == OP_FACTORY_TYPE_1) {
                        while (cur) {
                                if (strcmp(((opFactoryType1_t*)(*of))->opType, ((opFactoryType1_t*)(cur->opFactory))->opType) == 0) {
                                        (*(infra->udoFree))(newOfNode);
                                        (*(infra->udoFree))(((opFactoryType1_t*)(*of))->opType);
                                        (*(infra->udoFree))((opFactoryType1_t*)(*of));
                                        *of = cur->opFactory;
                                        return 0;
                                }
                                cur = cur->next;
                        }
                }
                (opFactoryList[opType].tail)->next = newOfNode;
                opFactoryList[opType].tail = newOfNode;
        }
        return 0;
}

/*
 * UDO library has the ownership op factories and static parameters. 
 * UDO library needs to allocate memory and copy by value.
 * Keeping track of op factories and sharing op factories within a graph are optional.
 */
SnpeUdo_ErrorType_t SnpeUdo_createOpFactory (SnpeUdo_CoreType_t udoCoreType, void* perFactoryInfrastructure, SnpeUdo_String_t operationType, uint32_t numOfStaticParams, SnpeUdo_Param_t* staticParams, SnpeUdo_OpFactory_t* opFactory) {
        if(infra == NULL) {
                return SNPE_UDO_UNSUPPORTED_FEATURE; 
        }
        if(operationType == NULL) {
                return SNPE_UDO_INVALID_ARGUMENT;
        }
        SnpeUdo_HexNNv2OpFactoryInfra_t* facInfra = (SnpeUdo_HexNNv2OpFactoryInfra_t*)perFactoryInfrastructure;
        if(strcmp(operationType, udoOpType1) == 0) {   // no static parameters
                opFactoryType1_t* thisFactory = (*(infra->udoMalloc))(sizeof(opFactoryType1_t));
                thisFactory->opType = (*(infra->udoMalloc))(strlen(operationType)+1);
                strncpy(thisFactory->opType, operationType, strlen(operationType));
                (thisFactory->opType)[strlen(operationType)] = '\0';
                appendToOpFactoryList ((SnpeUdo_OpFactory_t*)(&thisFactory), facInfra->graphId, OP_UDO_PLUS_ONE, OP_FACTORY_TYPE_1) ;
                *opFactory = (SnpeUdo_OpFactory_t)thisFactory;
        } else if(strcmp(operationType, udoOpType2) == 0) {   // with static parameters
                opFactoryType2_t* thisFactory = (*(infra->udoMalloc))(sizeof(opFactoryType2_t));
                thisFactory->opType = (*(infra->udoMalloc))(strlen(operationType)+1);
                strncpy(thisFactory->opType, operationType, strlen(operationType));
                (thisFactory->opType)[strlen(operationType)] = '\0';
                thisFactory->numStaticParams = numOfStaticParams;
                SnpeUdo_Param_t* thisStatic;
                if(numOfStaticParams == 0) {
                        thisStatic = NULL;
                } else {
                        thisStatic = (*(infra->udoMalloc))(numOfStaticParams*sizeof(SnpeUdo_Param_t));
                }
                uint32_t rank, dataSize;
                for(int i=0; i<numOfStaticParams; i++) {
                        // thisStatic[i]   <-  staticParams[i]
                        thisStatic[i].paramType = staticParams[i].paramType;
                        if(staticParams[i].paramName == NULL) {
                                thisStatic[i].paramName = NULL;
                        } else {
                                thisStatic[i].paramName = (*(infra->udoMalloc))(strlen(staticParams[i].paramName)+1);
                                strncpy(thisStatic[i].paramName, staticParams[i].paramName, strlen(staticParams[i].paramName));
                                (thisStatic[i].paramName)[strlen(staticParams[i].paramName)] = '\0';
                        }
                        if(thisStatic[i].paramType == SNPE_UDO_PARAMTYPE_SCALAR) {
                                ((thisStatic[i]).scalarParam).dataType = (staticParams[i].scalarParam).dataType;
                                ((thisStatic[i].scalarParam).dataValue).floatValue = ((staticParams[i].scalarParam).dataValue).floatValue;
                        } else if(thisStatic[i].paramType == SNPE_UDO_PARAMTYPE_TENSOR) {
                                ((thisStatic[i]).tensorParam).dataType = (staticParams[i].tensorParam).dataType;
                                (thisStatic[i].tensorParam).layout = (staticParams[i].tensorParam).layout;
                                (thisStatic[i].tensorParam).quantizeParams = (staticParams[i].tensorParam).quantizeParams;
                                rank = (staticParams[i].tensorParam).tensorRank;
                                (thisStatic[i].tensorParam).tensorRank = rank;
                                if ((thisStatic[i].tensorParam).layout == SNPE_UDO_LAYOUT_NULL) {
                                        (thisStatic[i].tensorParam).maxDimensions = NULL;
                                        (thisStatic[i].tensorParam).currDimensions = NULL;
                                        (thisStatic[i].tensorParam).tensorData = NULL;
                                } else {
                                        (thisStatic[i].tensorParam).maxDimensions = (*(infra->udoMalloc))(rank*sizeof(uint32_t));
                                        (thisStatic[i].tensorParam).currDimensions = (*(infra->udoMalloc))(rank*sizeof(uint32_t));
                                        memcpy((thisStatic[i].tensorParam).maxDimensions, (staticParams[i].tensorParam).maxDimensions, rank*sizeof(uint32_t));
                                        memcpy((thisStatic[i].tensorParam).currDimensions, (staticParams[i].tensorParam).currDimensions, rank*sizeof(uint32_t));
                                	switch(((thisStatic[i]).tensorParam).dataType) {
                                        	case SNPE_UDO_DATATYPE_FIXED_4:
                                                case SNPE_UDO_DATATYPE_FIXED_8:
                                                case SNPE_UDO_DATATYPE_UINT_8:
                                                case SNPE_UDO_DATATYPE_INT_8:      dataSize = 1; break;
                                                case SNPE_UDO_DATATYPE_FLOAT_16:
                                                case SNPE_UDO_DATATYPE_FIXED_16:
                                                case SNPE_UDO_DATATYPE_UINT_16:
                                                case SNPE_UDO_DATATYPE_INT_16:     dataSize = 2; break;
                                                case SNPE_UDO_DATATYPE_FLOAT_32:
                                                case SNPE_UDO_DATATYPE_FIXED_32:
                                                case SNPE_UDO_DATATYPE_UINT_32:
                                                case SNPE_UDO_DATATYPE_INT_32:
                                                default:                          dataSize = 4;
                                        }
                                        for(int j=0;j<rank;j++) {
                                                dataSize *= (thisStatic[i].tensorParam).maxDimensions[j];
                                        }
                                        (thisStatic[i].tensorParam).tensorData = (*(infra->udoMalloc))(dataSize);   // or udo_memalign for hvx purpose
                                        memcpy((thisStatic[i].tensorParam).tensorData, (staticParams[i].tensorParam).tensorData, dataSize);
                                }
                        } else if(thisStatic[i].paramType == SNPE_UDO_PARAMTYPE_STRING) {
                                thisStatic[i].stringParam = (*(infra->udoMalloc))(strlen(staticParams[i].stringParam)+1);
                                strncpy(thisStatic[i].stringParam, staticParams[i].stringParam, strlen(staticParams[i].stringParam));
                                (thisStatic[i].stringParam)[strlen(staticParams[i].stringParam)] = '\0';

                        }
                }
                thisFactory->staticParams = thisStatic;
                appendToOpFactoryList ((SnpeUdo_OpFactory_t*)(&thisFactory), facInfra->graphId, OP_UDO_PLUS_STAT, OP_FACTORY_TYPE_2);
                *opFactory = (SnpeUdo_OpFactory_t)thisFactory;
        } else {
                return SNPE_UDO_WRONG_OPERATION;
        }
        return SNPE_UDO_NO_ERROR;
}

/*
 * UDO library has the ownership of operation instances. 
 * Runtime has the ownership of inputs and outputs tensors, so inputs and outputs are copied as pointers into operation instances
 */
SnpeUdo_ErrorType_t SnpeUdo_createOperation (SnpeUdo_OpFactory_t OpFactory, void* perOpInfrastructure, uint32_t numOfInputs, SnpeUdo_TensorParam_t* inputs, uint32_t numOfOutputs, SnpeUdo_TensorParam_t* outputs, SnpeUdo_Operation_t* operation) {
        if(OpFactory == NULL)  return SNPE_UDO_INVALID_ARGUMENT;
        if((numOfInputs == 0 || inputs == NULL) && (numOfOutputs == 0 || outputs == NULL))  return SNPE_UDO_INVALID_ARGUMENT;
        if(infra == NULL)  return SNPE_UDO_UNSUPPORTED_FEATURE; 
        operationIns_t* newOperationIns = (*(infra->udoMalloc))(sizeof(operationIns_t));
        // newOperationIns  <-  parameters
        newOperationIns->opInfra = (SnpeUdo_HexNNv2OpInfra_t)perOpInfrastructure;
        newOperationIns->opFactory = OpFactory;
        newOperationIns->numInputParams = numOfInputs;
        if(numOfInputs == 0 || inputs == NULL) {
                newOperationIns->numInputParams = 0;
                newOperationIns->inputParams = NULL;
        } else {
                newOperationIns->inputParams = inputs;
        }
        newOperationIns->numOutputParams = numOfOutputs;
        if(numOfOutputs == 0 || outputs == NULL) {
                newOperationIns->numOutputParams = 0;
                newOperationIns->outputParams = NULL;
        } else {
                newOperationIns->outputParams = outputs;
        }

        // reads vtcm available size and possibly does some pre-calculations and checks
        uint32_t vtcmSize = (infra->udoGetVtcmSize)(newOperationIns->opInfra);
        uint32_t curInSize = sizeof(uint8_t);
        uint32_t maxInSize = 0;  // maximum input size among all inputs
        for(int i=0; i<numOfInputs; i++) {
                if (inputs[i].dataType == SNPE_UDO_DATATYPE_FIXED_8) {
                        curInSize = sizeof(uint8_t);
                } else if (inputs[i].dataType == SNPE_UDO_DATATYPE_FIXED_16) {
                        curInSize = sizeof(uint16_t);
                } else if (inputs[i].dataType == SNPE_UDO_DATATYPE_FIXED_32) {
                        curInSize = sizeof(uint32_t);
                }
                for (int d=0; d<inputs[i].tensorRank; d++) {
                        curInSize *= (inputs[i].maxDimensions)[d];
                }
                maxInSize = curInSize>maxInSize? curInSize : maxInSize;
        }
        newOperationIns->inputsFitVtcm = vtcmSize>=maxInSize? 1 : 0;  // indicates whether vtcm is large enough to hold any input for current operation instance

        *operation = (SnpeUdo_Operation_t)newOperationIns;
        return SNPE_UDO_NO_ERROR;
}


SnpeUdo_ErrorType_t SnpeUdo_executeOp (SnpeUdo_Operation_t operation, bool blocking, const uint32_t ID, SnpeUdo_ExternalNotify_t notifyFunc) {
        if(operation == NULL)  return SNPE_UDO_INVALID_ARGUMENT;
        operationIns_t* thisOpIns = (operationIns_t*)operation;
        char* thisOpType = ((opFactoryType1_t*)(thisOpIns->opFactory))->opType;
        if(thisOpType == NULL)   return SNPE_UDO_INVALID_ARGUMENT;
        if(infra == NULL)  return SNPE_UDO_UNSUPPORTED_FEATURE;
        if(strcmp(thisOpType, udoOpType1) == 0) {
                return opUdoPlusOne(operation);
        } else if(strcmp(thisOpType, udoOpType2) == 0) {
                return opUdoPlusStat(operation);
        } else {
                return SNPE_UDO_WRONG_OPERATION;
        }
}


SnpeUdo_ErrorType_t SnpeUdo_releaseOp (SnpeUdo_Operation_t operation) {
        if(operation == NULL)  return SNPE_UDO_NO_ERROR;
        if(infra == NULL)  return SNPE_UDO_UNSUPPORTED_FEATURE;
        operationIns_t* thisOpIns = (operationIns_t*)operation;
        (*(infra->udoFree))(thisOpIns);
        return SNPE_UDO_NO_ERROR;
}


int free_params(SnpeUdo_Param_t* params, int numParams){
        for (int i=0; i<numParams; i++) {
                if (params[i].paramName)  (infra->udoFree)(params[i].paramName);
                if (params[i].paramType == SNPE_UDO_PARAMTYPE_TENSOR) {
                        if ((params[i].tensorParam).maxDimensions)  (infra->udoFree)((params[i].tensorParam).maxDimensions);
                        if ((params[i].tensorParam).currDimensions)  (infra->udoFree)((params[i].tensorParam).currDimensions);
                        if ((params[i].tensorParam).tensorData)  (infra->udoFree)((params[i].tensorParam).tensorData);
                } else if (params[i].paramType == SNPE_UDO_PARAMTYPE_STRING) {
                        if (params[i].stringParam)  (infra->udoFree)(params[i].stringParam);
                }
        }
        if (params)  (infra->udoFree)(params);
        return 0;
}

SnpeUdo_ErrorType_t SnpeUdo_releaseOpFactory (SnpeUdo_OpFactory_t opFactory) {
        if(infra == NULL) {
                return SNPE_UDO_UNSUPPORTED_FEATURE;
        }
        if(opFactory == NULL) {
                return SNPE_UDO_NO_ERROR;
        }
        opFactoryTypes_t type = OP_FACTORY_TYPE_1;
        if(strcmp(((opFactoryType1_t*)opFactory)->opType, udoOpType2) == 0) {
                type = OP_FACTORY_TYPE_2;
        }
        if(type == OP_FACTORY_TYPE_1) {
                opFactoryType1_t* factory = (opFactoryType1_t*)opFactory;
                (*(infra->udoFree))(factory->opType);
                (*(infra->udoFree))(factory);
        } else if(type == OP_FACTORY_TYPE_2) {
                opFactoryType2_t* factory = (opFactoryType2_t*)opFactory;
                (*(infra->udoFree))(factory->opType);
                free_params(factory->staticParams, factory->numStaticParams);
                (*(infra->udoFree))(factory);
        } else if (type == OP_FACTORY_TYPE_3) {
                opFactoryType3_t* factory = (opFactoryType3_t*)opFactory;
                (*(infra->udoFree))(factory->opType);
                free_params(factory->staticParams, factory->numStaticParams);
                if (factory->someStorage)  (*(infra->udoFree))(factory->someStorage);
                (*(infra->udoFree))(factory);
        }
        return SNPE_UDO_NO_ERROR;
}


SnpeUdo_ErrorType_t SnpeUdo_terminateImplLibrary() {
        // free global variables allocated
        if(infra == NULL)  return SNPE_UDO_UNSUPPORTED_FEATURE;  
        libInitialized = 0;
        if(opFactoriesAll==NULL)  return SNPE_UDO_NO_ERROR;
        opFactoriesPerGraph_t* curGraphOf = opFactoriesAll;
        opFactoriesPerGraph_t* nextGraphOf = NULL;
        opFactoryNode_t *curOfNode, *nextOfNode;
        while (curGraphOf) {
                nextGraphOf = curGraphOf->next;
                for (int opN = 0; opN<NUM_OPS; opN++) {
                        curOfNode = (curGraphOf->opFactoryList[opN]).head;
                        nextOfNode = NULL;
                        while (curOfNode) {
                                nextOfNode = curOfNode->next;
                                (*(infra->udoFree))(curOfNode);
                                curOfNode = nextOfNode;
                        }
                }
                (*(infra->udoFree))(curGraphOf);
                curGraphOf = nextGraphOf;
        }
        opFactoriesAll = NULL;
        return SNPE_UDO_NO_ERROR;
}

