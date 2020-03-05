//==============================================================================
//
// Copyright (c) 2019 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SNPE_UDO_BASE_H
#define SNPE_UDO_BASE_H

#include <stdint.h>

// Provide values to use for API version.
#define API_VERSION_MAJOR 1
#define API_VERSION_MINOR 3
#define API_VERSION_TEENY 0

// Defines a bitmask of enum values.
typedef uint32_t SnpeUdo_Bitmask_t;

// A string of characters, rather than an array of bytes.
// Assumed to be UTF-8.
typedef char* SnpeUdo_String_t;

// The maximum allowable length of a SnpeUdo_String_t in bytes,
// including null terminator. SNPE will truncate strings longer
// than this.
#define SNPE_UDO_MAX_STRING_SIZE 1024

/*!
  * An enum which holds the various error types.
  * The error types are divided to classes :
  * 0 - 99    : generic errors
  * 100 - 200 : errors related to configuration
  *
  */
typedef enum
{
   SNPE_UDO_NO_ERROR                    = 0,
   SNPE_UDO_WRONG_CORE                  = 1,
   SNPE_UDO_INVALID_ARGUMENT            = 2,
   SNPE_UDO_UNSUPPORTED_FEATURE         = 3,
   SNPE_UDO_MEM_ALLOC_ERROR             = 4,
   /* Configuration Specific errors */
   SNPE_UDO_WRONG_OPERATION             = 100,
   SNPE_UDO_WRONG_CORE_TYPE             = 101,
   SNPE_UDO_WRONG_NUM_OF_PARAMS         = 102,
   SNPE_UDO_WRONG_NUM_OF_DIMENSIONS     = 103,
   SNPE_UDO_WRONG_NUM_OF_INPUTS         = 104,
   SNPE_UDO_WRONG_NUM_OF_OUTPUTS        = 105,
   SNPE_UDO_UNKNOWN_ERROR               = 0xFFFFFFFF
} SnpeUdo_ErrorType_t;

/*!
  * An enum which holds the various data types.
  * Designed to be used as single values or combined into a bitfield parameter
  * (0x1, 0x2, 0x4, etc)
  * \n FIXED_XX types are targeted for data in tensors.
  * \n UINT / INT types are targeted for scalar params
  */
typedef enum
{
   SNPE_UDO_DATATYPE_FLOAT_16       = 0x01,
   SNPE_UDO_DATATYPE_FLOAT_32       = 0x02,
   SNPE_UDO_DATATYPE_FIXED_4        = 0x04,
   SNPE_UDO_DATATYPE_FIXED_8        = 0x08,
   SNPE_UDO_DATATYPE_FIXED_16       = 0x10,
   SNPE_UDO_DATATYPE_FIXED_32       = 0x20,
   SNPE_UDO_DATATYPE_UINT_8         = 0x100,
   SNPE_UDO_DATATYPE_UINT_16        = 0x200,
   SNPE_UDO_DATATYPE_UINT_32        = 0x400,
   SNPE_UDO_DATATYPE_INT_8          = 0x1000,
   SNPE_UDO_DATATYPE_INT_16         = 0x2000,
   SNPE_UDO_DATATYPE_INT_32         = 0x4000,
   SNPE_UDO_DATATYPE_LAST           = 0xFFFFFFFF
} SnpeUdo_DataType_t;

/*!
  * An enum which holds the various layouts.
  * Designed to be used as single values or combined into a bitfield parameter
  * (0x1, 0x2, 0x4, etc)
  */
typedef enum
{
   SNPE_UDO_LAYOUT_NHWC             = 0x01,
   SNPE_UDO_LAYOUT_NCHW             = 0x02,
   SNPE_UDO_LAYOUT_NDHWC            = 0x04,
   SNPE_UDO_LAYOUT_GPU_OPTIMAL1     = 0x08,
   SNPE_UDO_LAYOUT_GPU_OPTIMAL2     = 0x10,
   SNPE_UDO_LAYOUT_DSP_OPTIMAL1     = 0x11,
   SNPE_UDO_LAYOUT_DSP_OPTIMAL2     = 0x12,
   // Indicates no data will be allocated for this tensor.
   // Used to specify optional inputs/outputs positionally.
   SNPE_UDO_LAYOUT_NULL             = 0x13,
   SNPE_UDO_LAYOUT_LAST             = 0xFFFFFFFF
} SnpeUdo_TensorLayout_t;

/*!
  * An enum which holds the UDO library Core type .
  * Designed to be used as single values or combined into a bitfield parameter
  * (0x1, 0x2, 0x4, etc)
  */
typedef enum
{
   /// Library target IP Core is undefined
   SNPE_UDO_CORETYPE_UNDEFINED   = 0x00,
   /// Library target IP Core is CPU
   SNPE_UDO_CORETYPE_CPU         = 0x01,
   /// Library target IP Core is GPU
   SNPE_UDO_CORETYPE_GPU         = 0x02,
   /// Library target IP Core is DSP
   SNPE_UDO_CORETYPE_DSP         = 0x04,
   SNPE_UDO_CORETYPE_LAST         = 0xFFFFFFFF
} SnpeUdo_CoreType_t;

/*!
  * An enum to specify the parameter type : Scalar or Tensor
  */
typedef enum
{
   SNPE_UDO_PARAMTYPE_SCALAR,
   SNPE_UDO_PARAMTYPE_STRING,
   SNPE_UDO_PARAMTYPE_TENSOR,
   SNPE_UDO_PARAMTYPE_LAST   = 0xFFFFFFFF
} SnpeUdo_ParamType_t;

/*!
  * An enum to specify quantization type
  */
typedef enum
{
   SNPE_UDO_QUANTIZATION_NONE,
   SNPE_UDO_QUANTIZATION_TF,
   SNPE_UDO_QUANTIZATION_QMN,
   SNPE_UDO_QUANTIZATION_LAST   = 0xFFFFFFFF
} SnpeUdo_QuantizationType_t;

/**
 * @brief A struct which is used to provide a version number using 3 values : major, minor, teeny
 *
 */
typedef struct
{
   uint32_t major;
   uint32_t minor;
   uint32_t teeny;
} SnpeUdo_Version_t;

/**
 * @brief A struct returned from version query, contains the Library version and API version
 *
 */
typedef struct
{
   SnpeUdo_Version_t libVersion;
   SnpeUdo_Version_t apiVersion;
} SnpeUdo_LibVersion_t;

/**
 * @brief A union to hold the value of a generic type. allows defining a parameter struct
 * in a generic way, with a "value" location that holds the data regradless of the type.
 *
 */
typedef union
{
   float    floatValue;
   uint32_t uint32Value;
   int32_t  int32Value;
   uint16_t uint16Value;
   int16_t  int16Value;
   uint8_t  uint8Value;
   int8_t   int8Value;
} SnpeUdo_Value_t;

/**
 * @brief A struct which defines a scalar parameter : name, data type, and union of values
 *
 */
typedef struct
{
   /// The parameter data type : float, int, etc.
   SnpeUdo_DataType_t  dataType;
   /// a union of specified type which holds the data
   SnpeUdo_Value_t dataValue;
} SnpeUdo_ScalarParam_t;

/**
 * @brief A struct which defines the quantization parameters in case of Tensorflow style quantization
 *
 */
typedef struct
{
   float minValue;
   float maxValue;
} SnpeUdo_TFQuantize_t;

/**
 * @brief A struct which defines the quantization type, and union of supported quantization structs
 *
 */
typedef struct
{
   SnpeUdo_QuantizationType_t quantizeType;
   union
   {
      SnpeUdo_TFQuantize_t TFParams;
   };
} SnpeUdo_QuantizeParams_t;

/**
 * @brief A struct which defines a tensor parameter : name, data type, layout, quantization, more.
 *        Also holds a pointer to the tensor data.
 *
 */
typedef struct
{
   /// The maximum allowable dimensions of the tensor. The memory held in
   /// _tensorData_ is guaranteed to be large enough for this.
   uint32_t*               maxDimensions;
   /// The current dimensions of the tensor. An operation may modify the current
   /// dimensions of its output, to indicate cases where the output has been
   /// "resized".
   /// Note that for static parameters, the current and max dimensions must
   /// match.
   uint32_t*               currDimensions;
   SnpeUdo_QuantizeParams_t quantizeParams;
   uint32_t                tensorRank;
   /// The parameter data type : float, int, etc.
   SnpeUdo_DataType_t       dataType;
   /// The tensor layout type : NCHW, NHWC, etc.
   SnpeUdo_TensorLayout_t   layout;
   /// Cast to the right structure GPU tensor Data or DSP tensor data
   void*                   tensorData;
} SnpeUdo_TensorParam_t;

/**
 * @brief A struct which defines a UDO parameter- a union of scalar and tensor parameters
 *
 */
typedef struct
{
   /// Type is scalar or tensor
   SnpeUdo_ParamType_t paramType;
   /// The param name, for example : "offset", "activation_type"
   SnpeUdo_String_t    paramName;
   union
   {
      SnpeUdo_ScalarParam_t scalarParam;
      SnpeUdo_TensorParam_t tensorParam;
      SnpeUdo_String_t      stringParam;
   };

} SnpeUdo_Param_t;

/**
 * @brief A struct which defines Operation information which is specific for IP core (CPU, GPU, DSP ...)
 *
 */
typedef struct
{
   /// The IP Core
   ///
   SnpeUdo_CoreType_t     udoCoreType;
   /// Bitmask, defines supported internal calculation types (like FLOAT_32, etc)
   /// Based on SnpeUdo_DataType
   SnpeUdo_Bitmask_t      operationCalculationTypes;
} SnpeUdo_OpCoreInfo_t;

/**
 * @brief A struct which defines the Operation information - both the Shared and the Core Specific
 *
 */
typedef struct
{
   /// The operation type, for example : "MY_COOL_OP"
   SnpeUdo_String_t  operationType;
   /// A bitmask describing which IP Cores (CPU, GPU, DSP ...) support this operation
   /// Translated based on SnpeUdo_CoreType
   SnpeUdo_Bitmask_t supportedByCores;
   uint32_t numOfStaticParams;
   SnpeUdo_Param_t* staticParams;
   uint32_t numOfInputs;
   /// Names of the inputs to this operation. Length is numOfInputs.
   SnpeUdo_String_t* inputNames;
   uint32_t numOfOutputs;
   // Names of the outputs of this operation. Length is numOfOutputs.
   SnpeUdo_String_t* outputNames;
   uint32_t numOfCoreInfo;
   /// Array which defines per-core information like input/output types, etc.
   SnpeUdo_OpCoreInfo_t* opPerCoreInfo;
} SnpeUdo_OperationInfo_t;

/**
 * @brief A struct which provides the implementation library info : type, name
 *
 */
typedef struct
{
   /// Defines the IP Core that this implementation library is targeting
   SnpeUdo_CoreType_t     udoCoreType;
   /// library name. will be looked at in the standard library path
   SnpeUdo_String_t       libraryName;
} SnpeUdo_LibraryInfo_t;

/**
 * @brief A struct returned by the registration library and contains information on the UDO package :
 * name, operations, libraries, etc.
 *
 */
typedef struct
{
   /// A string containing the package name
   SnpeUdo_String_t         packageName;
   /// A bitmask describing supported IP cores (CPU, GPU, DSP ...)
   /// Translated based on SnpeUdo_CoreType
   SnpeUdo_Bitmask_t        supportedCoreTypes;
   /// The number of implementation libraries in the package
   uint32_t                numOfImplementationLib;
   /// Array of implementation libraries names/types
   SnpeUdo_LibraryInfo_t*   implementationLib;
   /// A string containing all operation types separated by space
   SnpeUdo_String_t         operationsString;
   /// Number of supported operations
   uint32_t                numOfOperations;
   /// Array of Operation info structs. Each entry describes one
   /// Operation (name, params, inputs, outputs)
   SnpeUdo_OperationInfo_t* operationsInfo;
} SnpeUdo_RegInfo_t;

/**
* @brief A struct returned by the implementation library and contains information on the specific library :
* name, IP Core, operations, etc.
*
*/
typedef struct
{
   /// Defines the IP Core that this implementation library is targeting
   SnpeUdo_CoreType_t     udoCoreType;
   /// A string containing the package name
   SnpeUdo_String_t       packageName;
   /// A string containing all operation types separated by space
   SnpeUdo_String_t       operationsString;
   /// Number of supported operations
   uint32_t              numOfOperations;
} SnpeUdo_ImpInfo_t;

/**
 * @brief This struct defines an operation. It is used for validation
 * or creation of an operation.
 * In case of using it for creation, the static params which are tensors
 * contain pointers to the real data (weights, for example), and input/output
 * tensors also include pointers to the buffers used.
 */
typedef struct
{
   /// The IP Core that the operation is defined for - CPU, GPU, DSP...
   SnpeUdo_CoreType_t      udoCoreType;
   /// The operation type, for example : "MY_COOL_OP"
   SnpeUdo_String_t        operationType;
   /// The number of static parameters provided in the staticParams array.
   /// this number has to match the number provided by the UDO Registration library information
   uint32_t               numOfStaticParams;
   /// Array of static parameters
   SnpeUdo_Param_t*        staticParams;
   /// The number of input parameters provided in inputs array.
   /// this number has to match the number provided by the UDO Registration library information
   uint32_t               numOfInputs;
   /// Array of input tensors, providing layout, data type, sizes, etc
   /// When used to create an operation, also contains the initial location of the data
   SnpeUdo_TensorParam_t*  inputs;
   /// The number of output parameters provided in inputs array.
   /// this number has to match the number provided by the UDO Registration library information
   uint32_t               numOfOutputs;
   /// Array of output tensors, providing layout, data type, sizes, etc
   /// When used to create an operation, also contains the initial location of the data
   SnpeUdo_TensorParam_t*  outputs;
} SnpeUdo_OpDefinition_t;

#endif //SNPE_UDO_BASE_H

