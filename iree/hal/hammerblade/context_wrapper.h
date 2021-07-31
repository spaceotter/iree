// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_HAMMERBLADE_CONTEXT_WRAPPER_H_
#define IREE_HAL_HAMMERBLADE_CONTEXT_WRAPPER_H_

#include "iree/hal/api.h"
#include "iree/hal/hammerblade/hb_headers.h"

// Structure to wrap all objects constant within a context. This makes it
// simpler to pass it to the different objects and saves memory.
typedef struct iree_hal_hammerblade_context_wrapper_t {
  hb_mc_device_t hb_device;
  iree_allocator_t host_allocator;
} iree_hal_hammerblade_context_wrapper_t;

#endif  // IREE_HAL_HAMMERBLADE_CONTEXT_WRAPPER_H_
