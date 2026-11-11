# Vanda GPU toolchain (Lecture 12, p. 40): NVIDIA HPC SDK 26.1 with HPC-X
# (CUDA-aware Open MPI). The MPI wrapper is pointed at GCC for the host code;
# nvcc compiles the kernels. CMake comes from the EasyBuild tree.
module load nvhpc-hpcx-cuda12/26.1
export ATTN_CLUSTER=vanda
export OMPI_CXX=g++ OMPI_CC=gcc
for candidate in /app1/ebapps/arches/flat-avx2/software/CMake/3.31.8-GCCcore-14.3.0/bin \
                 /app1/ebapps/arches/flat-avx2/software/CMake/3.29.3-GCCcore-13.3.0/bin; do
    if [[ -x "$candidate/cmake" ]]; then export PATH="$candidate:$PATH"; break; fi
done
