// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LIBRT_SRC_LIBM_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LIBRT_SRC_LIBM_H_

#include "librt.h"

// https://en.cppreference.com/w/c/numeric/math/fma
LIBRT_EXPORT float fmaf(float x, float y, float z);
LIBRT_EXPORT short __gnu_f2h_ieee(float param);
LIBRT_EXPORT float __gnu_h2f_ieee(short param);

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_LLVM_LIBRT_SRC_LIBM_H_
