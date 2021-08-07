# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Doesn't use bazel_to_cmake because of various special logic throughout.

if(NOT "${IREE_TARGET_BACKEND_HAMMERBLADE}")
  return()
endif()

iree_add_all_subdirs()

file(RELATIVE_PATH manycore_lib "${CMAKE_CURRENT_SOURCE_DIR}" "${BLADERUNNER_DIR}/bsg_manycore/software/spmd/fib/bsg_manycore_lib.a")

# TODO(#5893): iree_c_embed_data requires a relative path and we need it to be
# here to keep the path consistent with bazel build file. Fix it by making cmake
# target more flexible.
iree_c_embed_data(
  NAME
    hb_manycore_lib
  SRCS
    "${manycore_lib}"
  C_FILE_OUTPUT
    "bsg_manycore_lib.c"
  H_FILE_OUTPUT
    "bsg_manycore_lib.h"
  FLATTEN
  PUBLIC
)

iree_cc_library(
  NAME
    HammerBlade
  HDRS
    "HBTarget.h"
  SRCS
    "HBTarget.cpp"
  DEPS
    ::hb_manycore_lib
    LLVMBitReader
    LLVMCore
    LLVMipo
    LLVMLinker
#    LLVMNVPTXCodeGen
    LLVMSupport
    LLVMTarget
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRNVVMIR
    MLIRNVVMToLLVMIRTranslation
    MLIRPass
    MLIRSupport
    MLIRTargetLLVMIRExport
    iree::base::internal::flatcc
    iree::compiler::Codegen::LLVMHB
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Utils
    iree::schemas::hammerblade_executable_def_c_fbs
  PUBLIC
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
