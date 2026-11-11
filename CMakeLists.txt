cmake_minimum_required(VERSION 3.16)

project(RingAttentionMPI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(ATTN_ENABLE_MPI "Build the distributed benchmark" ON)
option(ATTN_ENABLE_OPENMP "Enable query-tile threading" ON)
option(ATTN_ENABLE_CUDA "Build CUDA backends (requires a CUDA toolkit)" OFF)
option(ATTN_SANITIZERS "Enable address and undefined-behaviour sanitizers" OFF)
option(ATTN_NATIVE "Tune for this CPU; compile on the target node type" OFF)

# Alignment in bytes of every tensor and scratch buffer (16 = plain malloc).
set(ATTN_ALIGNMENT "4096" CACHE STRING "Alignment of tensor storage in bytes: 16, 64 or 4096")
if(NOT ATTN_ALIGNMENT MATCHES "^(16|64|4096)$")
    message(FATAL_ERROR "ATTN_ALIGNMENT must be 16, 64 or 4096")
endif()
add_compile_definitions(ATTN_ALIGNMENT=${ATTN_ALIGNMENT})

add_library(attn_core src/core.cpp)
target_include_directories(attn_core PUBLIC include)
target_compile_options(attn_core PRIVATE -Wall -Wextra -Wpedantic)
if(ATTN_NATIVE)
    target_compile_options(attn_core PRIVATE -march=native)
endif()
# Extra kernel flags for controlled experiments, e.g. -mprefer-vector-width=512.
set(ATTN_EXTRA_FLAGS "" CACHE STRING "Additional compile options for the attention core")
if(ATTN_EXTRA_FLAGS)
    separate_arguments(ATTN_EXTRA_FLAGS_LIST NATIVE_COMMAND "${ATTN_EXTRA_FLAGS}")
    target_compile_options(attn_core PRIVATE ${ATTN_EXTRA_FLAGS_LIST})
endif()
if(ATTN_SANITIZERS)
    target_compile_options(attn_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(attn_core PUBLIC -fsanitize=address,undefined)
endif()

if(ATTN_ENABLE_OPENMP)
    # Apple Clang needs a separate OpenMP runtime. Respect other toolchains.
    if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        find_path(ATTN_OMP_INCLUDE omp.h PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        find_library(ATTN_OMP_LIBRARY omp PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
        if(ATTN_OMP_INCLUDE AND ATTN_OMP_LIBRARY)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_INCLUDE_DIR "${ATTN_OMP_INCLUDE}")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${ATTN_OMP_LIBRARY}")
        endif()
    endif()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(attn_core PUBLIC OpenMP::OpenMP_CXX)
endif()
if(ATTN_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(attn_mpi src/distributed.cpp)
    target_link_libraries(attn_mpi PUBLIC attn_core MPI::MPI_CXX)
    target_compile_options(attn_mpi PRIVATE -Wall -Wextra -Wpedantic)
    add_executable(attn_bench apps/attn_bench.cpp)
    target_link_libraries(attn_bench PRIVATE attn_mpi)
    add_executable(attn_analyze apps/attn_analyze.cpp)
    target_link_libraries(attn_analyze PRIVATE attn_core MPI::MPI_CXX)
    target_compile_options(attn_analyze PRIVATE -Wall -Wextra -Wpedantic)
    add_executable(attn_netbench apps/attn_netbench.cpp)
    target_link_libraries(attn_netbench PRIVATE MPI::MPI_CXX)
    target_compile_options(attn_netbench PRIVATE -Wall -Wextra -Wpedantic)
    add_executable(attn_ringbench apps/attn_ringbench.cpp)
    target_link_libraries(attn_ringbench PRIVATE MPI::MPI_CXX)
    target_compile_options(attn_ringbench PRIVATE -Wall -Wextra -Wpedantic)
    execute_process(COMMAND git describe --always --dirty
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE ATTN_GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(NOT ATTN_GIT_VERSION)
        set(ATTN_GIT_VERSION "unknown")
    endif()
    target_compile_definitions(attn_bench PRIVATE ATTN_GIT_COMMIT="${ATTN_GIT_VERSION}")
endif()
if(ATTN_ENABLE_CUDA)
    # Hand-written CUDA kernels (no cuBLAS/cuDNN). Needs a CUDA toolkit; for the
    # CUDA-aware path the MPI library itself must be CUDA-aware.
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(attn_gpu src/gpu_kernels.cu)
    set_target_properties(attn_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(attn_gpu PUBLIC include)
    target_link_libraries(attn_gpu PUBLIC CUDA::cudart)
    if(ATTN_ENABLE_MPI)
        add_executable(attn_gpu_bench apps/attn_gpu_bench.cpp)
        target_link_libraries(attn_gpu_bench PRIVATE attn_core attn_gpu MPI::MPI_CXX)
        target_compile_options(attn_gpu_bench PRIVATE -Wall -Wextra)
    endif()
endif()

add_executable(ring_attention_mpi
    src/main.cpp
    src/attention.cpp
)

target_include_directories(ring_attention_mpi PRIVATE include)

enable_testing()

add_test(
    NAME attention_correctness
    COMMAND ring_attention_mpi
)

add_executable(attn_core_tests tests/core_tests.cpp)
target_link_libraries(attn_core_tests PRIVATE attn_core)
add_test(NAME core_validation COMMAND attn_core_tests)

if(ATTN_ENABLE_MPI)
    add_executable(attn_mpi_tests tests/mpi_tests.cpp)
    target_link_libraries(attn_mpi_tests PRIVATE attn_mpi)
    # Larger P and oversubscription are explicit local validation invocations;
    # keep default CTest within two slots on developer and cluster allocations.
    foreach(ranks 1 2)
        add_test(NAME mpi_validation_${ranks}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:attn_mpi_tests> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(mpi_validation_${ranks} PROPERTIES TIMEOUT 90 PROCESSORS ${ranks})
    endforeach()
endif()
