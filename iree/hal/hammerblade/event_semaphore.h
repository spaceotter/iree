// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_HAMMERBLADE_SEMAPHORE_H_
#define IREE_HAL_HAMMERBLADE_SEMAPHORE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/hammerblade/context_wrapper.h"
#include "iree/hal/hammerblade/status_util.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Create a hammerblade allocator.
iree_status_t iree_hal_hammerblade_semaphore_create(
    iree_hal_hammerblade_context_wrapper_t* context, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_HAMMERBLADE_SEMAPHORE_H_
