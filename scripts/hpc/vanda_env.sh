# Verified on Vanda (stdct-login-01), 10 November 2026: GCC 14.3.0, Open MPI 5.0.8
# (UCX 1.19.0, PRRTE 3.0.11) and CMake 3.31.8 from the site EasyBuild tree.
# Source this file inside a job; it only loads modules and names the cluster.
module load OpenMPI/5.0.8-GCC-14.3.0 CMake/3.31.8-GCCcore-14.3.0
export ATTN_CLUSTER=vanda
