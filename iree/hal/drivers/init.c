// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/init.h"

#include "iree/base/tracing.h"

#if defined(IREE_HAL_HAVE_CUDA_DRIVER_MODULE)
#include "iree/hal/cuda/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_CUDA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_HAMMERBLADE_DRIVER_MODULE)
#include "iree/hal/hammerblade/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_CUDA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_DRIVER_MODULE)
#include "iree/hal/dylib/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_DYLIB_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE)
#include "iree/hal/dylib/registration/driver_module_sync.h"
#endif  // IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMVX_DRIVER_MODULE)
#include "iree/hal/vmvx/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_VMVX_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VULKAN_DRIVER_MODULE)
#include "iree/hal/vulkan/registration/driver_module.h"
#endif  // IREE_HAL_HAVE_VULKAN_DRIVER_MODULE

#if defined(IREE_BUILD_EXPERIMENTAL_ROCM)
#include "experimental/rocm/registration/driver_module.h"
#endif  // IREE_BUILD_EXPERIMENTAL_ROCM

IREE_API_EXPORT iree_status_t
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t* registry) {
  IREE_TRACE_ZONE_BEGIN(z0);

#if defined(IREE_HAL_HAVE_CUDA_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_CUDA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_HAMMERBLADE_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hammerblade_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_CUDA_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_DYLIB_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_dylib_sync_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_DYLIB_SYNC_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VMVX_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vmvx_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_VMVX_DRIVER_MODULE

#if defined(IREE_HAL_HAVE_VULKAN_DRIVER_MODULE)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_vulkan_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_VULKAN_DRIVER_MODULE

#if defined(IREE_BUILD_EXPERIMENTAL_ROCM)
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_rocm_driver_module_register(registry));
#endif  // IREE_HAL_HAVE_ROCM_DRIVER_MODULE

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
