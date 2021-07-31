// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/hammerblade/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/hammerblade/api.h"

#define IREE_HAL_HAMMERBLADE_DRIVER_ID 0x48414D42u  // HAMB

static iree_status_t iree_hal_hammerblade_driver_factory_enumerate(
    void* self, const iree_hal_driver_info_t** out_driver_infos,
    iree_host_size_t* out_driver_info_count) {
  // NOTE: we could query supported hammerblade versions or featuresets here.
  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_id = IREE_HAL_HAMMERBLADE_DRIVER_ID,
      .driver_name = iree_string_view_literal("hammerblade"),
      .full_name = iree_string_view_literal("HAMMERBLADE (dynamic)"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_hammerblade_driver_factory_try_create(
    void* self, iree_hal_driver_id_t driver_id, iree_allocator_t allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = NULL;
  if (driver_id != IREE_HAL_HAMMERBLADE_DRIVER_ID) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver with ID %016" PRIu64
                            " is provided by this factory",
                            driver_id);
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hammerblade_device_params_t default_params;
  iree_hal_hammerblade_device_params_initialize(&default_params);
  // TODO(jinchen62): set up default_params.use_deferred_submission by flag
  // When we expose more than one driver (different hammerblade versions, etc) we
  // can name them here:
  iree_string_view_t identifier = iree_make_cstring_view("hammerblade");

  iree_hal_hammerblade_driver_options_t driver_options;
  iree_hal_hammerblade_driver_options_initialize(&driver_options);
  iree_status_t status = iree_hal_hammerblade_driver_create(
      identifier, &default_params, &driver_options, allocator, out_driver);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_hammerblade_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_hammerblade_driver_factory_enumerate,
      .try_create = iree_hal_hammerblade_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
