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
# Find LLVM & MLIR
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
# LLVM/MLIR headers
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

# Match LLVM's RTTI setting — LLVM/MLIR are built without RTTI
if(NOT LLVM_ENABLE_RTTI)
  if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR-")
  endif()
endif()

# ---------------------------------------------------------------------------
# Triton dialect object files — Triton builds as OBJECT libraries so we
# collect the pre-built .o files and link them into our shared library.
# ---------------------------------------------------------------------------
set(TRITON_IR_OBJS_DIR
    "${TRITON_BUILDPATH}/lib/Dialect/Triton/IR/CMakeFiles/TritonIR.dir")

file(GLOB TRITON_IR_OBJS "${TRITON_IR_OBJS_DIR}/*.o")

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(lib/Conversion)

# ---------------------------------------------------------------------------
# Shared library target — aggregates all conversion objects
# ---------------------------------------------------------------------------
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

# Install target (optional — for packaging later)
install(TARGETS triton_metal_backend LIBRARY DESTINATION lib)

# ---------------------------------------------------------------------------
# pybind11 Python extension — exposes C++ MLIR passes to Python
# ---------------------------------------------------------------------------
find_package(pybind11 REQUIRED PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python3.14/site-packages/pybind11/share/cmake/pybind11")

pybind11_add_module(_triton_metal_cpp python_bindings.cpp)
target_link_libraries(_triton_metal_cpp PRIVATE triton_metal_backend)
# pybind11 requires RTTI (typeid).  The global CMAKE_CXX_FLAGS contain
# -fno-rtti to match LLVM/MLIR, so we must override it for this target.
# python_bindings.cpp does NOT include any MLIR headers — it calls
# through a C-linkage bridge (python_bindings_bridge.cpp) that lives in
# the dylib and is compiled with -fno-rtti.
target_compile_options(_triton_metal_cpp PRIVATE -frtti)
