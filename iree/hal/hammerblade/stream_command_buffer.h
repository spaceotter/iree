// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_HAMMERBLADE_STREAM_COMMAND_BUFFER_H_
#define IREE_HAL_HAMMERBLADE_STREAM_COMMAND_BUFFER_H_

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/hammerblade/context_wrapper.h"
#include "iree/hal/hammerblade/hb_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a hammerblade stream command buffer that immediately
// issues commands against the given |stream|.
// Access to |stream| must be synchronized by the user.
// Used for replaying commands in special situations and
// never returned to a user from the device_create_command_buffer
iree_status_t iree_hal_hammerblade_stream_command_buffer_create(
    iree_hal_hammerblade_context_wrapper_t *context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, CUstream stream,
    iree_hal_command_buffer_t **out_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_HAMMERBLADE_STREAM_COMMAND_BUFFER_H_
