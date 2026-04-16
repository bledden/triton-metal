cmake_minimum_required(VERSION 3.20)
project(triton_metal_backend CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Paths — hardcoded defaults, overridable via -D at configure time
# ---------------------------------------------------------------------------
set(LLVM_SYSPATH "$ENV{HOME}/.triton/llvm/llvm-979132a0-macos-arm64"
    CACHE PATH "Path to LLVM/MLIR install")
set(TRITON_SRCPATH "$ENV{HOME}/Documents/triton"
    CACHE PATH "Path to Triton source tree")
set(TRITON_BUILDPATH "$ENV{HOME}/Documents/triton/build/cmake.macosx-15.0-arm64-cpython-3.14"
    CACHE PATH "Path to Triton build tree (for generated headers)")

# ---------------------------------------------------------------------------
# Find LLVM & MLIR (for include paths and cmake modules only)
# ---------------------------------------------------------------------------
set(LLVM_DIR "${LLVM_SYSPATH}/lib/cmake/llvm" CACHE PATH "")
set(MLIR_DIR "${LLVM_SYSPATH}/lib/cmake/mlir" CACHE PATH "")

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")
message(STATUS "Found MLIR at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
# LLVM/MLIR headers (needed for compilation, NOT for linking)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})

# Triton source headers (Dialect definitions, etc.)
include_directories(SYSTEM "${TRITON_SRCPATH}/include")

# Triton build-generated headers (TableGen outputs)
include_directories(SYSTEM "${TRITON_BUILDPATH}/include")

# Our own headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# Compiler flags — match Triton conventions
# ---------------------------------------------------------------------------
add_definitions(${LLVM_DEFINITIONS})

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -fPIC")
endif()

# Default to -fno-rtti (LLVM/MLIR convention); pybind11 source overrides below
if(NOT LLVM_ENABLE_RTTI)
  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
  endif()
endif()

# ---------------------------------------------------------------------------
# Triton dialect object files — Triton builds as OBJECT libraries so we
# collect the pre-built .o files for the Triton IR dialect definitions.
# These are needed because the Triton ops (GetProgramIdOp, LoadOp, etc.)
# have non-inline methods defined in these compilation units.
# ---------------------------------------------------------------------------
set(TRITON_IR_OBJS_DIR
    "${TRITON_BUILDPATH}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir")

file(GLOB TRITON_IR_OBJS "${TRITON_IR_OBJS_DIR}/*.o")

# TritonGPU dialect object files — needed for type-safe matching on
# TTG ops (LocalAllocOp, LocalLoadOp, etc.). TTG symbols are not
# exported from libtriton.so on macOS, so we link the .o files directly.
set(TRITON_GPU_OBJS_DIR
    "${TRITON_BUILDPATH}/lib/Dialect/TritonGPU/IR/CMakeFiles/TritonGPUIR.dir")

file(GLOB TRITON_GPU_OBJS "${TRITON_GPU_OBJS_DIR}/*.o")

if(NOT TRITON_GPU_OBJS)
  message(FATAL_ERROR "TritonGPU dialect objects not found at ${TRITON_GPU_OBJS_DIR}")
endif()
list(LENGTH TRITON_GPU_OBJS _ttg_obj_count)
message(STATUS "Found ${_ttg_obj_count} TritonGPU objects")

# TritonNvidiaGPU dialect objects — TTG Ops.cpp references TMEM ops
# (TMEMStoreOp, TensorMemoryEncodingAttr, etc.) in canonicalize patterns,
# so the TypeID symbols for those types must be present in the final
# module. We don't register the dialect — just provide the symbols.
set(TRITON_NVIDIA_GPU_OBJS_DIR
    "${TRITON_BUILDPATH}/lib/Dialect/TritonNvidiaGPU/IR/CMakeFiles/TritonNvidiaGPUIR.dir")

file(GLOB TRITON_NVIDIA_GPU_OBJS "${TRITON_NVIDIA_GPU_OBJS_DIR}/*.o")

if(NOT TRITON_NVIDIA_GPU_OBJS)
  message(FATAL_ERROR "TritonNvidiaGPU dialect objects not found at ${TRITON_NVIDIA_GPU_OBJS_DIR}")
endif()
list(LENGTH TRITON_NVIDIA_GPU_OBJS _tng_obj_count)
message(STATUS "Found ${_tng_obj_count} TritonNvidiaGPU objects")

# Triton support object files — LinearLayout, LayoutUtils, supportMMA,
# inferDstEncoding, etc. These helpers are referenced by TTG Dialect.cpp
# / Ops.cpp in verify/canonicalize paths. On macOS they live in libtriton.so
# as local symbols (not exported), so we must pull in the specific .o files.
set(TRITON_SUPPORT_OBJS
    "${TRITON_BUILDPATH}/lib/Tools/CMakeFiles/TritonTools.dir/LinearLayout.cpp.o"
    "${TRITON_BUILDPATH}/lib/Tools/CMakeFiles/TritonTools.dir/LayoutUtils.cpp.o"
    "${TRITON_BUILDPATH}/lib/Analysis/CMakeFiles/TritonAnalysis.dir/Utility.cpp.o"
    "${TRITON_BUILDPATH}/lib/Dialect/TritonGPU/Transforms/CMakeFiles/TritonGPUTransforms.dir/Utility.cpp.o"
    "${TRITON_BUILDPATH}/third_party/f2reduce/CMakeFiles/f2reduce.dir/f2reduce.cpp.o"
)
foreach(_obj ${TRITON_SUPPORT_OBJS})
  if(NOT EXISTS "${_obj}")
    message(FATAL_ERROR "Triton support object not found: ${_obj}")
  endif()
endforeach()
list(LENGTH TRITON_SUPPORT_OBJS _ts_obj_count)
message(STATUS "Using ${_ts_obj_count} Triton support objects")

# ---------------------------------------------------------------------------
# libtriton.so path — Triton's shared library that contains all MLIR symbols.
# By linking against this instead of standalone MLIR libraries, we avoid
# duplicate MLIR dialect registrations when both are loaded in the same
# Python process.
# ---------------------------------------------------------------------------
set(LIBTRITON_PATH "${TRITON_SRCPATH}/python/triton/_C/libtriton.so")
if(NOT EXISTS "${LIBTRITON_PATH}")
  message(FATAL_ERROR "libtriton.so not found at ${LIBTRITON_PATH}")
endif()
message(STATUS "Using libtriton.so at ${LIBTRITON_PATH}")

# ---------------------------------------------------------------------------
# Standalone shared library — aggregates all conversion objects.
# This links against individual MLIR libraries and is used for standalone
# testing only (NOT loaded alongside libtriton.so in the same process).
# ---------------------------------------------------------------------------
add_subdirectory(lib/Conversion)

add_library(triton_metal_backend SHARED
    $<TARGET_OBJECTS:TritonMetalToLLVM>
    ${TRITON_IR_OBJS}
    python_bindings_bridge.cpp
)

target_link_libraries(triton_metal_backend
    PRIVATE
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRTransformUtils
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRSCFDialect
    MLIRMathDialect
    MLIRArithUtils
    MLIRFuncDialect
    MLIRTensorDialect
    LLVMSupport
    LLVMCore
)

install(TARGETS triton_metal_backend LIBRARY DESTINATION lib)

# ---------------------------------------------------------------------------
# pybind11 Python extension — exposes C++ MLIR passes to Python
#
# KEY CHANGE: This module links against Triton's libtriton.so for MLIR
# symbols instead of standalone MLIR libraries. This prevents the duplicate
# MLIR dialect registration crash when both our .so and libtriton.so are
# loaded in the same Python process.
#
# All conversion source files are compiled directly into this module (no
# intermediate library), avoiding the two-library symbol resolution problem.
# ---------------------------------------------------------------------------
find_package(pybind11 REQUIRED PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python3.14/site-packages/pybind11/share/cmake/pybind11")

pybind11_add_module(_triton_metal_cpp
    python_bindings.cpp
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
)

# Include our headers for the conversion sources compiled into this module
target_include_directories(_triton_metal_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against libtriton.so for ALL MLIR/LLVM symbols — this is the same
# copy of MLIR that Triton uses, so no duplicate registrations.
target_link_libraries(_triton_metal_cpp PRIVATE ${LIBTRITON_PATH})

# Also link the Triton IR dialect object files — these contain non-inline
# method definitions for Triton ops (GetProgramIdOp, LoadOp, etc.)
target_sources(_triton_metal_cpp PRIVATE ${TRITON_IR_OBJS})

# TritonGPU dialect object files — enables type-safe matching on TTG ops.
target_sources(_triton_metal_cpp PRIVATE ${TRITON_GPU_OBJS})

# TritonNvidiaGPU objects — provide TypeID symbols referenced by TTG ops.
target_sources(_triton_metal_cpp PRIVATE ${TRITON_NVIDIA_GPU_OBJS})

# Triton helper .o files — LinearLayout methods, supportMMA, etc.
target_sources(_triton_metal_cpp PRIVATE ${TRITON_SUPPORT_OBJS})

# Per-file RTTI settings:
# - python_bindings.cpp needs RTTI for pybind11 (typeid)
# - Everything else needs -fno-rtti to match LLVM/MLIR conventions
set_source_files_properties(
    python_bindings.cpp
    PROPERTIES COMPILE_FLAGS "-frtti"
)
set_source_files_properties(
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti"
)

# ---------------------------------------------------------------------------
# triton-ext compatible plugin shared library
# Loads via TRITON_PASS_PLUGIN_PATH without pybind11 dependency.
# ---------------------------------------------------------------------------
add_library(triton_metal_plugin SHARED
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
    ${TRITON_IR_OBJS}
    ${TRITON_GPU_OBJS}
    ${TRITON_NVIDIA_GPU_OBJS}
    ${TRITON_SUPPORT_OBJS}
)

target_link_libraries(triton_metal_plugin PRIVATE ${LIBTRITON_PATH})
target_include_directories(triton_metal_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# TritonGPU/LinearLayout methods live in libtriton.so as "local" symbols
# (internal linkage on macOS). Allow dynamic lookup so they resolve at
# runtime when loaded alongside libtriton.so, matching the pybind11
# module's bundle semantics.
if(APPLE)
  target_link_options(triton_metal_plugin PRIVATE
      "LINKER:-undefined,dynamic_lookup")
endif()
set_source_files_properties(
    python_bindings_bridge.cpp
    lib/Conversion/TritonMetalToLLVM.cpp
    lib/Conversion/ElementwiseOpToLLVM.cpp
    PROPERTIES COMPILE_FLAGS "-fno-rtti"
)
