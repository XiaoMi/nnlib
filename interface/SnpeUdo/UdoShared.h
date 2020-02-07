//==============================================================================
//
// Copyright (c) 2019 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SNPE_UDO_SHARED_H
#define SNPE_UDO_SHARED_H

#include "SnpeUdo/UdoBase.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief A function to return the various versions
 *        The function returns a struct containing the library version, API version and more
 *
 * @param[in, out] version A pointer to Version struct of type snpeLibraryVersion
 *
 * @return Error code
 *
 */
SnpeUdo_ErrorType_t
SnpeUdo_getVersion (SnpeUdo_LibVersion_t** version);

typedef SnpeUdo_ErrorType_t
(*SnpeUdo_GetVersionFunction_t) (SnpeUdo_LibVersion_t** version);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SNPE_UDO_SHARED_H

