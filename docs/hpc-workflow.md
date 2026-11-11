# Running the experiments on Vanda

Everything below was verified on NUS Vanda on 10-11 November 2026 (Atlas was
probed too, but its `parallel` queue allows at most 96 cores per job, so the
256-core requirement can only be met on Vanda).

## Cluster facts

| Item | Value |
| --- | --- |
| CPU nodes | 72 cores (2 x Xeon Platinum 8452Y), 512 GiB; node names `cn-NNN` |
| GPU nodes | 2 x NVIDIA A40 per node; node names `gn-a40-NNN` |
| Charged route | no `-q`, `-P <project>`: `batch_cpu` (<= 720 cores, <= 10 nodes), `batch_gpu` (<= 2 GPUs, <= 2 nodes) |
| Free route | `-q auto_free`: <= 160 cores, <= 2 GPUs, 36 cores per GPU job |
| Accounting | credits are debited when a job **starts**: requested walltime x allocated cores (`batch_cpu`) or x GPUs (`batch_gpu`, which always allocates 36 cores per GPU and debits no CPU hours); `hpc project` shows the balance after `amgr login` |
| CPU toolchain | `module load OpenMPI/5.0.8-GCC-14.3.0 CMake/3.31.8-GCCcore-14.3.0` (`scripts/hpc/vanda_env.sh`) |
| GPU toolchain | `module load nvhpc-hpcx-cuda12/26.1` (`scripts/hpc/vanda_gpu_env.sh`) |
| Transport | shared memory inside a node, TCP over Ethernet between nodes (the Open MPI module selects `pml ob1`, `btl sm,tcp`, `osc rdma`) |

## Pitfalls met, and what the scripts do about them

- **`qsub -o <path>`** made every start attempt fail ("job held, too many failed
  attempts"). Jobs are submitted from their run directory and PBS writes
  `<name>.o<id>` there.
- **Bad nodes.** A job that lands on a faulty node is retried 21 times and then
  held. The helper is called with an explicit host list of nodes that are free
  and have run a full-node job before:
  `-l select=1:ncpus=72:mpiprocs=72:ompthreads=1:host=cn-042+1:...:host=cn-069`.
- **Daemon start.** This Open MPI starts its per-node daemon over ssh, where no
  module is loaded, and the daemon belongs to a separate PRRTE installation:
  `mpirun --prefix $(dirname $(dirname $(which prted)))`, plus
  `-x LD_LIBRARY_PATH -x PATH` for the processes. `common.sh` tests this on every
  allocated node before measuring.
- **Wide ranks.** `--map-by slot:PE=72 --bind-to core` is refused (one rank across
  two sockets); such ranks are placed per node and the OpenMP runtime pins them.
- **Shared GPU nodes.** A GPU job owns 36 of 72 cores, so MPI must not bind to
  absolute core numbers: `--bind-to none`.
- **Clock ramp.** A core stays at 2.0 GHz for the first seconds of load (about
  4 s in the measurement of the report) and then steps to its turbo clock. Every
  formal run therefore warms up for at least 5 s (`--warmup-seconds 5`).
- **Alignment.** With 512-bit vectors a `std::vector<float>` from `malloc`
  (16-byte alignment) makes nearly every load cross a cache line: the kernel is
  1.5 times slower, and its speed depends on where the allocator places each
  array, which changes with the rank count. All tensors and scratch buffers are
  therefore page-aligned (`ATTN_ALIGNMENT`, suite `align`).
- **Requested walltime is what is paid.** A job is debited for its requested
  walltime when it starts, however long it runs, also when it fails after a few
  seconds or is cancelled. Measure a suite once, then request about 1.3 times
  its duration; keep work that fits on one node in one-node jobs and run only
  the rest in a four-node job (`ATTN_MIN_CORES=73`, `ATTN_RANKS=72:144:288`,
  `ATTN_SERIAL_BASELINE=1`); never cancel a job that has just started.
- **Project name.** Pass `-P` and the project name as two words. In zsh an
  unquoted variable holding `-P name` is one word; the job is then accepted
  with a project name that starts with a space and never starts (`Budgets:
  ... project ... is not found`). `qalter -P name <job>` repairs a queued job.
- **Exit status of a killed job.** `common.sh` traps SIGTERM so that a cancelled
  job or one that reached its walltime records a non-zero `exit-status.txt`;
  the figure scripts only read runs that ended with status 0.
- **Shared build directory.** Jobs that start together build under a lock
  directory, one at a time. A job that cannot get the lock stops; a lock is
  taken over only if the job that holds it has finished. A build directory is
  reused only if it carries the marker of a completed build.
- **Nodes are taken quickly.** A node that becomes free is usually given to
  another user's job within minutes. A job pinned to hosts waits until all of
  them are free at the same moment.
- **Never replace a select statement with `qalter`.** The queue adds the memory
  of every chunk (`:mem=500GB` for a full node) when a job is *submitted*. A
  select statement replaced later with `qalter -l select=...` has no memory,
  the job's vnodes get none (`exec_vnode` without `mem=`), and the control group
  then kills every rank that grows beyond about 1 GB per node (`signal 9`).
  One job was lost this way. To pin hosts, give the host list to `qsub`; to
  check before paying, submit with `-h`, read `qstat -f` and release with
  `qrls`. `common.sh` stops a job whose vnodes carry no memory.
- **The MPI of the GPU toolchain does not know PBS.** The HPC-X Open MPI 4.1.9
  of the NVIDIA HPC SDK has no `tm` components (`ompi_info`: `ras: slurm`
  only), so `mpirun` sees only the node it runs on; a job on two GPU nodes fails
  with "more processes than the ppr". `detect_launcher` falls back to
  `--hostfile $PBS_NODEFILE`, which starts the other node's rank over ssh. On
  Vanda that rank is still confined to the job's GPU (the node shows it one
  device); `gpu_rank.sh` checks this on every node, selects the allocated GPU by
  its NUMA node if more are visible, and stops if it cannot tell.
- **Settlement.** The balance is reduced by requested walltime x cores when a
  job starts; some hours later the difference to the actual run time is
  returned (observed on 11 November: 4,789 to 8,937 core-hours).

## Workflow

```bash
H=USER@vanda.nus.edu.sg
TAG=$(scripts/hpc/cluster.sh push $H)          # immutable copy under ~/ring-attention/src/<TAG>
ATTN_JOB_ENV="ATTN_SUITE=strong,ATTN_RANKS=1:2:4:8:18:36:72" \
scripts/hpc/cluster.sh submit $H $TAG experiment.pbs e2-strong -P PROJECT \
    -l select=1:ncpus=72:mpiprocs=72:ompthreads=1:host=cn-042 -l walltime=01:10:00
scripts/hpc/cluster.sh status $H
scripts/hpc/cluster.sh fetch $H                 # -> results/hpc/vanda/<run>/
```

Each run directory holds the job log, `environment.txt` (nodes, modules,
compiler, CPU, limits), the build cache, the CSV files and `exit-status.txt`.
The run directories that the report uses are named in
`results/selected_runs.json`; every other directory is listed there with the
reason it is not used. `cluster.sh` stages exactly the files it hashes (the tag)
and checks the staged copy against the tag.

Job scripts (`scripts/hpc/jobs/`): `experiment.pbs` (suites `serial`, `strong`,
`weak`, `hybrid`, `comm`, `causal`, `memory`, `gqa`, `hardware`, `model`,
`align`, `extras`, `tune`, `realdata`, `fatcheck`; several per job with
`ATTN_SUITE=a:b`), `gpu_experiment.pbs` (`ladder`, `ring`, `ringcheck`),
`unit_tests.pbs` (all tests of the staged source), `real_data.pbs`,
`extract_qkv.pbs`, the per-rank wrapper `gpu_rank.sh`, and the diagnostics
`build_validate.pbs`, `gpu_validate.pbs`, `launch_probe.pbs`,
`transport_probe.pbs`. `tests/hpc_scripts_test.sh` tests the launcher detection,
the build lock and the GPU selection locally with mocked scheduler commands.
