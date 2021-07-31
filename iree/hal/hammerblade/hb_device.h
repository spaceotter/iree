// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_HAMMERBLADE_HAMMERBLADE_DEVICE_H_
#define IREE_HAL_HAMMERBLADE_HAMMERBLADE_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/hammerblade/api.h"
#include "iree/hal/hammerblade/hb_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device that owns and manages its own CUcontext.
iree_status_t iree_hal_hammerblade_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hammerblade_device_params_t* params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_HAMMERBLADE_HAMMERBLADE_DEVICE_H_
