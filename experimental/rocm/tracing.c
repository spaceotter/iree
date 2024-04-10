// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/tracing.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include "experimental/rocm/status_util.h"

// Total number of events per tracing context. This translates to the maximum
// number of outstanding timestamp queries before collection is required.
// To prevent spilling pages we leave some room for the context structure.
#define IREE_HAL_ROCM_TRACING_DEFAULT_QUERY_CAPACITY (16 * 1024 - 256)

typedef void (*SmuDumpCallback)(uint64_t, const char*, const char*, double);
typedef bool (*SmuDumpInitFunc)(SmuDumpCallback callback);
typedef void (*SmuDumpEndFunc)(void);
typedef void (*SmuDumpOnceFunc)(void);
typedef void (*RegDumpOnceFunc)(void);
typedef void (*SviDumpOnceFunc)(void);
typedef uint32_t (*RegGetTraceRate)(void);
typedef uint32_t (*SmuGetTraceRate)(void);

#define SMUTRACE_DLL "libsmutrace.so"

struct iree_hal_rocm_tracing_context_t {
  iree_hal_rocm_context_wrapper_t* rocm_context;
  hipStream_t stream;
  iree_arena_block_pool_t* block_pool;
  iree_allocator_t host_allocator;

  // A unique GPU zone ID allocated from Tracy.
  // There is a global limit of 255 GPU zones (ID 255 is special).
  uint8_t id;

  // Base event used for computing relative times for all recorded events.
  // This is required as ROCM (without CUPTI) only allows for relative timing
  // between events and we need a stable base event.
  hipEvent_t base_event;

  // Indices into |event_pool| defining a ringbuffer.
  uint32_t query_head;
  uint32_t query_tail;
  uint32_t query_capacity;

  // Event pool reused to capture tracing timestamps.
  hipEvent_t event_pool[IREE_HAL_ROCM_TRACING_DEFAULT_QUERY_CAPACITY];

  void* smutrace_handle;
  SmuDumpInitFunc f_smuDumpInit;
  SmuDumpEndFunc f_smuDumpEnd;
  SmuDumpOnceFunc f_smuDumpOnce;
  RegDumpOnceFunc f_regDumpOnce;
  SviDumpOnceFunc f_sviDumpOnce;
  RegGetTraceRate f_regGetTraceRate;
  SmuGetTraceRate f_smuGetTraceRate;
  bool smutrace_enabled;
  pthread_t var_dump_thread;
  int64_t m_reg_period;
};

static uint64_t clocktime_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}

static void* dump_smu_variables_thread(void* ctx) {
  iree_hal_rocm_tracing_context_t* context =
      (iree_hal_rocm_tracing_context_t*)ctx;

  int64_t startTime = clocktime_ns() / 1000;

  if (context != NULL) {
    while (context->smutrace_enabled == true) {
      context->f_regDumpOnce();

      int64_t endTime = clocktime_ns() / 1000;
      int64_t sleepTime = startTime + context->m_reg_period - endTime;
      sleepTime = (sleepTime > 0) ? sleepTime : 0;
      usleep(sleepTime);
      startTime = clocktime_ns() / 1000;
    }
  }

  pthread_exit(NULL);
}

static void iree_smutrace_callback(uint64_t did, const char* type,
                                   const char* name, double value) {
  IREE_TRACE_SET_PLOT_TYPE(name, IREE_TRACING_PLOT_TYPE_NUMBER, true, true,
                           0xFF0000FF);
  IREE_TRACE_PLOT_VALUE_F64(name, value);
}

static iree_status_t iree_hal_rocm_tracing_context_initial_calibration(
    iree_hal_rocm_context_wrapper_t* rocm_context, hipStream_t stream,
    hipEvent_t base_event, int64_t* out_cpu_timestamp,
    int64_t* out_gpu_timestamp, float* out_timestamp_period) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_cpu_timestamp = 0;
  *out_gpu_timestamp = 0;
  *out_timestamp_period = 1.0f;

  // Record event to the stream; in the absence of a synchronize this may not
  // flush immediately.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, ROCM_RESULT_TO_STATUS(rocm_context->syms,
                                hipEventRecord(base_event, stream)));

  // Force flush the event and wait for it to complete.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, ROCM_RESULT_TO_STATUS(rocm_context->syms,
                                hipEventSynchronize(base_event)));

  // Track when we know the event has completed and has a reasonable timestamp.
  // This may drift from the actual time differential between host/device but is
  // (maybe?) the best we can do without CUPTI.
  *out_cpu_timestamp = iree_tracing_time();

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_rocm_tracing_context_allocate(
    iree_hal_rocm_context_wrapper_t* rocm_context,
    iree_string_view_t queue_name, hipStream_t stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_rocm_tracing_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(rocm_context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;

  iree_hal_rocm_tracing_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);
  if (iree_status_is_ok(status)) {
    context->rocm_context = rocm_context;
    context->stream = stream;
    context->block_pool = block_pool;
    context->host_allocator = host_allocator;
    context->query_capacity = IREE_ARRAYSIZE(context->event_pool);
  }

  // Pre-allocate all events in the event pool.
  if (iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_BEGIN_NAMED(
        z_event_pool, "iree_hal_rocm_tracing_context_allocate_event_pool");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z_event_pool,
                                     (int64_t)context->query_capacity);
    for (iree_host_size_t i = 0; i < context->query_capacity; ++i) {
      status = ROCM_RESULT_TO_STATUS(rocm_context->syms,
                                     hipEventCreate(&context->event_pool[i]));
      if (!iree_status_is_ok(status)) break;
    }
    IREE_TRACE_ZONE_END(z_event_pool);
  }

  // Create the initial GPU event and insert it into the stream.
  // All events we record are relative to this event.
  int64_t cpu_timestamp = 0;
  int64_t gpu_timestamp = 0;
  float timestamp_period = 0.0f;
  if (iree_status_is_ok(status)) {
    status = ROCM_RESULT_TO_STATUS(rocm_context->syms,
                                   hipEventCreate(&context->base_event));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_tracing_context_initial_calibration(
        rocm_context, stream, context->base_event, &cpu_timestamp,
        &gpu_timestamp, &timestamp_period);
  }

  // Allocate the GPU context and pass initial calibration data.
  if (iree_status_is_ok(status)) {
    context->id = iree_tracing_gpu_context_allocate(
        IREE_TRACING_GPU_CONTEXT_TYPE_VULKAN, queue_name.data, queue_name.size,
        /*is_calibrated=*/false, cpu_timestamp, gpu_timestamp,
        timestamp_period);
  }

  context->smutrace_enabled = false;
  context->smutrace_handle = dlopen(SMUTRACE_DLL, RTLD_LAZY);
  if (context->smutrace_handle) {
    context->f_smuDumpInit =
        (SmuDumpInitFunc)dlsym(context->smutrace_handle, "smuDumpInit");
    context->f_smuDumpEnd =
        (SmuDumpEndFunc)dlsym(context->smutrace_handle, "smuDumpEnd");
    context->f_smuDumpOnce =
        (SmuDumpOnceFunc)dlsym(context->smutrace_handle, "smuDumpOnce");
    context->f_regDumpOnce =
        (RegDumpOnceFunc)dlsym(context->smutrace_handle, "regDumpOnce");
    context->f_sviDumpOnce =
        (SviDumpOnceFunc)dlsym(context->smutrace_handle, "sviDumpOnce");
    context->f_smuGetTraceRate = (SmuGetTraceRate)dlsym(
        context->smutrace_handle, "getSmuVariablesCaptureRate");
    context->f_regGetTraceRate = (RegGetTraceRate)dlsym(
        context->smutrace_handle, "getRegisterExpressionCaptureRate");
    context->smutrace_enabled =
        (context->f_smuDumpInit && context->f_smuDumpEnd &&
         context->f_smuDumpOnce && context->f_smuGetTraceRate &&
         context->f_regGetTraceRate && context->f_regDumpOnce &&
         context->f_sviDumpOnce &&
         context->f_smuDumpInit(iree_smutrace_callback));
  } else {
    fprintf(stderr, "Warning: " SMUTRACE_DLL " could not be loaded!\n");
  }

  if (context->smutrace_enabled) {
    context->m_reg_period = context->f_regGetTraceRate();
    int ret = pthread_create(&context->var_dump_thread, NULL,
                             dump_smu_variables_thread, context);
    if (ret != 0) {
      fprintf(stderr, "Warning: failed to create variables dump thread\n");
    }
  }

  if (iree_status_is_ok(status)) {
    *out_context = context;
  } else {
    iree_hal_rocm_tracing_context_free(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_rocm_tracing_context_free(
    iree_hal_rocm_tracing_context_t* context) {
  if (!context) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Always perform a collection on shutdown.
  iree_hal_rocm_tracing_context_collect(context);

  // Release all events; since collection completed they should all be unused.
  IREE_TRACE_ZONE_BEGIN_NAMED(z_event_pool,
                              "iree_hal_rocm_tracing_context_free_event_pool");
  for (iree_host_size_t i = 0; i < context->query_capacity; ++i) {
    if (context->event_pool[i]) {
      ROCM_IGNORE_ERROR(context->rocm_context->syms,
                        hipEventDestroy(context->event_pool[i]));
    }
  }
  IREE_TRACE_ZONE_END(z_event_pool);
  if (context->base_event) {
    ROCM_IGNORE_ERROR(context->rocm_context->syms,
                      hipEventDestroy(context->base_event));
  }

  iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  if (context->smutrace_enabled) {
    context->smutrace_enabled = false;
    pthread_join(context->var_dump_thread, NULL);
    context->f_smuDumpEnd();
    dlclose(context->smutrace_handle);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_rocm_tracing_context_collect(
    iree_hal_rocm_tracing_context_t* context) {
  if (!context) return;
  if (context->query_tail == context->query_head) {
    // No outstanding queries.
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_rocm_dynamic_symbols_t* syms = context->rocm_context->syms;

  while (context->query_tail != context->query_head) {
    // Compute the contiguous range of queries ready to be read.
    // If the ringbuffer wraps around we'll handle that in the next loop.
    uint32_t try_query_count =
        context->query_head < context->query_tail
            ? context->query_capacity - context->query_tail
            : context->query_head - context->query_tail;
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)try_query_count);

    // Scan and feed the times to tracy, stopping when we hit the first
    // unavailable query.
    uint32_t query_base = context->query_tail;
    uint32_t read_query_count = 0;
    for (uint32_t i = 0; i < try_query_count; ++i) {
      // Ensure the event has completed; will return hipErrorNotReady if
      // recorded but not retired or any other deferred error.
      uint16_t query_id = (uint16_t)(query_base + i);
      hipEvent_t query_event = context->event_pool[query_id];
      hipError_t result = syms->hipEventQuery(query_event);
      if (result != hipSuccess) break;

      // Calculate context-relative time and notify tracy.
      float relative_millis = 0.0f;
      ROCM_IGNORE_ERROR(
          syms, hipEventElapsedTime(&relative_millis, context->base_event,
                                    query_event));
      int64_t gpu_timestamp = (int64_t)((double)relative_millis * 1000000.0);
      iree_tracing_gpu_zone_notify(context->id, query_id, gpu_timestamp);

      read_query_count = i + 1;
    }
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)read_query_count);

    context->query_tail += read_query_count;
    if (context->query_tail >= context->query_capacity) {
      context->query_tail = 0;
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

static uint16_t iree_hal_rocm_tracing_context_insert_query(
    iree_hal_rocm_tracing_context_t* context, hipStream_t stream) {
  // Allocate an event from the pool for use by the query.
  uint32_t query_id = context->query_head;
  context->query_head = (context->query_head + 1) % context->query_capacity;

  // TODO: check to see if the read and write heads of the ringbuffer have
  // overlapped. If they have we could try to collect but it's not guaranteed
  // that collection will complete (e.g. we may be reserving events for use in
  // graphs that haven't yet been launched).
  //
  // For now we just allow the overlap and tracing results will be inconsistent.
  IREE_ASSERT_NE(context->query_head, context->query_tail);

  hipEvent_t event = context->event_pool[query_id];
  ROCM_IGNORE_ERROR(context->rocm_context->syms, hipEventRecord(event, stream));

  return query_id;
}

// TODO: optimize this implementation to reduce the number of events required:
// today we insert 2 events per zone (one for begin and one for end) but in
// many cases we could reduce this by inserting events only between zones and
// using the differences between them.

void iree_hal_rocm_tracing_zone_begin_impl(
    iree_hal_rocm_tracing_context_t* context, hipStream_t stream,
    const iree_tracing_location_t* src_loc) {
  if (!context) return;
  uint16_t query_id =
      iree_hal_rocm_tracing_context_insert_query(context, stream);
  iree_tracing_gpu_zone_begin(context->id, query_id, src_loc);
}

void iree_hal_rocm_tracing_zone_begin_external_impl(
    iree_hal_rocm_tracing_context_t* context, hipStream_t stream,
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  if (!context) return;
  uint16_t query_id =
      iree_hal_rocm_tracing_context_insert_query(context, stream);
  iree_tracing_gpu_zone_begin_external(context->id, query_id, file_name,
                                       file_name_length, line, function_name,
                                       function_name_length, name, name_length);
}

void iree_hal_rocm_tracing_zone_end_impl(
    iree_hal_rocm_tracing_context_t* context, hipStream_t stream) {
  if (!context) return;
  uint16_t query_id =
      iree_hal_rocm_tracing_context_insert_query(context, stream);
  iree_tracing_gpu_zone_end(context->id, query_id);
}

#else

iree_status_t iree_hal_rocm_tracing_context_allocate(
    iree_hal_rocm_context_wrapper_t* rocm_context,
    iree_string_view_t queue_name, hipStream_t stream,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_rocm_tracing_context_t** out_context) {
  *out_context = NULL;
  return iree_ok_status();
}

void iree_hal_rocm_tracing_context_free(
    iree_hal_rocm_tracing_context_t* context) {}

void iree_hal_rocm_tracing_context_collect(
    iree_hal_rocm_tracing_context_t* context) {}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE
