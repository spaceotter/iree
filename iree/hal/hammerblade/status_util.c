// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/hammerblade/status_util.h"

#include "bsg_manycore_errno.h"

#include <stddef.h>

iree_status_t iree_hal_hammerblade_result_to_status(int result,
                                                    const char* file,
                                                    uint32_t line) {
  if (IREE_LIKELY(result == HB_MC_SUCCESS)) {
    return iree_ok_status();
  }

  const char* error_name = NULL;
  if (result < HB_MC_SUCCESS && result >= HB_MC_UNALIGNED) {
    error_name = hb_mc_strerror(result);
  } else {
    error_name = "Unknown";
  }

  return iree_make_status(IREE_STATUS_INTERNAL,
                          "HAMMERBLADE driver error '%s' (%d)", error_name, result);
}
