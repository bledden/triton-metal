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
