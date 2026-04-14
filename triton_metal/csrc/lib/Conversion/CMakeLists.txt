# Build conversion sources as an OBJECT library so the top-level shared lib
# can aggregate them without creating an intermediate static archive.
add_library(TritonMetalToLLVM OBJECT
    TritonMetalToLLVM.cpp
    ElementwiseOpToLLVM.cpp
)

target_include_directories(TritonMetalToLLVM
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
