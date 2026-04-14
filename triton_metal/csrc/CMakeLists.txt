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

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(lib/Conversion)

# ---------------------------------------------------------------------------
# Shared library target — aggregates all conversion objects
# ---------------------------------------------------------------------------
add_library(triton_metal_backend SHARED
    $<TARGET_OBJECTS:TritonMetalToLLVM>
)

target_link_libraries(triton_metal_backend
    PRIVATE
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
    LLVMSupport
    LLVMCore
)

# Install target (optional — for packaging later)
install(TARGETS triton_metal_backend LIBRARY DESTINATION lib)
