// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/hammerblade/api.h"
#include "iree/hal/hammerblade/hb_device.h"
#include "iree/hal/hammerblade/status_util.h"

#define IREE_HAL_TASK_DEVICE_ID_DEFAULT 0

typedef struct iree_hal_hammerblade_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple HAMMERBLADE versions can be exposed in the
  // same process.
  iree_string_view_t identifier;
  iree_hal_hammerblade_device_params_t default_params;
  int default_device_index;
} iree_hal_hammerblade_driver_t;

// Pick a fixed lenght size for device names.
#define IREE_MAX_HAMMERBLADE_DEVICE_NAME_LENGTH 100

extern const iree_hal_driver_vtable_t iree_hal_hammerblade_driver_vtable;

static iree_hal_hammerblade_driver_t* iree_hal_hammerblade_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hammerblade_driver_vtable);
  return (iree_hal_hammerblade_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hammerblade_driver_options_initialize(
    iree_hal_hammerblade_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

static iree_status_t iree_hal_hammerblade_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_hammerblade_device_params_t* default_params,
    const iree_hal_hammerblade_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_hammerblade_driver_t* driver = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));

  iree_hal_resource_initialize(&iree_hal_hammerblade_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + iree_sizeof_struct(*driver));
  memcpy(&driver->default_params, default_params,
         sizeof(driver->default_params));
  driver->default_device_index = options->default_device_index;

  return iree_ok_status();
}

static void iree_hal_hammerblade_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_hammerblade_driver_t* driver = iree_hal_hammerblade_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_hammerblade_driver_create(
    iree_string_view_t identifier,
    const iree_hal_hammerblade_device_params_t* default_params,
    const iree_hal_hammerblade_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(default_params);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_hammerblade_driver_create_internal(
      identifier, default_params, options, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hammerblade_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count) {
  static const iree_hal_device_info_t device_infos[1] = {
      {
          .device_id = IREE_HAL_TASK_DEVICE_ID_DEFAULT,
          .name = iree_string_view_literal("default"),
      },
  };
  *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  return iree_allocator_clone(
      host_allocator, iree_make_const_byte_span(device_infos, sizeof(device_infos)),
      (void**)out_device_infos);
}

static iree_status_t iree_hal_hammerblade_driver_create_device(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_hammerblade_driver_t* driver = iree_hal_hammerblade_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_view_t device_name = iree_make_cstring_view("hammerblade");

  // Attempt to create the device.
  iree_status_t status = iree_hal_hammerblade_device_create(
      base_driver, device_name, &driver->default_params,
      host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_driver_vtable_t iree_hal_hammerblade_driver_vtable = {
    .destroy = iree_hal_hammerblade_driver_destroy,
    .query_available_devices = iree_hal_hammerblade_driver_query_available_devices,
    .create_device = iree_hal_hammerblade_driver_create_device,
};
