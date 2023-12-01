"""Defines a launcher to launch a Slurm training job."""

import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from mlfab.nn.parallel import (
    init_parallelism,
    init_process_group_from_backend,
    is_master,
    set_init_method,
    set_master_addr,
    set_master_port,
    set_rank,
    set_world_size,
)
from mlfab.task.base import RawConfigType
from mlfab.task.launchers.staged import StagedLauncher
from mlfab.task.mixins.train import Config, TrainMixin
from mlfab.utils.logging import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_MASTER_PORT = 29500


def set_slurm_rank_and_world_size() -> tuple[int, int]:
    node_id = int(os.environ["SLURM_NODEID"])
    local_id = int(os.environ["SLURM_LOCALID"])
    tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    num_nodes = int(os.environ["SLURM_NNODES"])
    rank = node_id * tasks_per_node + local_id
    world_size = num_nodes * tasks_per_node
    set_rank(rank)
    set_world_size(world_size)
    return rank, world_size


def set_slurm_master_addr_and_port() -> str:
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    assert node_list is not None, "`SLURM_JOB_NODELIST` environment variable not set"
    hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
    host = hostnames.split()[0].decode("utf-8")
    port = int(os.environ.get("MASTER_PORT", str(DEFAULT_MASTER_PORT)))
    set_master_addr(host)
    set_master_port(port)
    return host


def requeue_job() -> None:
    if is_master():
        if "SLURM_JOB_ID" in os.environ:
            cmd = ["scontrol", "requeue", os.environ["SLURM_JOB_ID"]]
            logger.info("Running %s", " ".join(cmd))
            subprocess.check_call(cmd)
        else:
            logger.info("SLURM_JOB_ID environment variable not found; not requeueing")


class SlurmLauncher(StagedLauncher):
    """Defines a launcher to launch a Slurm training job."""

    def __init__(
        self,
        partition: str,
        gpus_per_node: int,
        cpus_per_gpu: int,
        num_nodes: int = 1,
        gpu_type: str | None = None,
        exclusive: bool = False,
        time_limit: str = "3-00:00:00",
        num_jobs: int = 1,
        comment: str | None = None,
        master_port: int = DEFAULT_MASTER_PORT,
        model_parallelism: int = 1,
        pipeline_parallelism: int = 1,
        backend: str | None = None,
        model_parallel_backend: str | None = None,
        pipeline_parallel_backend: str | None = None,
        data_parallel_backend: str | None = None,
        account: str | None = None,
    ) -> None:
        super().__init__()

        self.partition = partition
        self.gpus_per_node = gpus_per_node
        self.cpus_per_gpu = cpus_per_gpu
        self.num_nodes = num_nodes
        self.gpu_type = gpu_type
        self.exclusive = exclusive
        self.time_limit = time_limit
        self.num_jobs = num_jobs
        self.comment = comment
        self.master_port = master_port
        self.model_parallelism = model_parallelism
        self.pipeline_parallelism = pipeline_parallelism
        self.backend = backend
        self.model_parallel_backend = model_parallel_backend
        self.pipeline_parallel_backend = pipeline_parallel_backend
        self.data_parallel_backend = data_parallel_backend
        self.account = account

    @property
    def gres(self) -> str:
        return f"gpu:{self.gpus_per_node}" if self.gpu_type is None else f"gpu:{self.gpu_type}:{self.gpus_per_node}"

    @property
    def gpu_bind(self) -> str:
        return f"map_gpu:{','.join(str(i) for i in range(self.gpus_per_node))}"

    @property
    def extra_sbatch_lines(self) -> list[str]:
        sbatch_lines: list[str] = []
        if "EMAIL" in os.environ:
            sbatch_lines += [f"--mail-user={os.environ['EMAIL']}", "--mail-type=ALL"]
        if self.account is not None:
            sbatch_lines += [f"--account={self.account}"]
        if self.exclusive:
            sbatch_lines += ["--exclusive"]
        return sbatch_lines

    @property
    def extra_export_lines(self) -> str:
        export_lines: dict[str, str] = {}
        if self.model_parallelism > 1:
            export_lines["MODEL_PARALLELISM"] = str(self.model_parallelism)
        if self.pipeline_parallelism > 1:
            export_lines["PIPELINE_PARALLELISM"] = str(self.pipeline_parallelism)
        if self.backend is not None:
            export_lines["BACKEND"] = self.backend
        if self.model_parallel_backend is not None:
            export_lines["MODEL_PARALLEL_BACKEND"] = self.model_parallel_backend
        if self.pipeline_parallel_backend is not None:
            export_lines["PIPELINE_PARALLEL_BACKEND"] = self.pipeline_parallel_backend
        if self.data_parallel_backend is not None:
            export_lines["DATA_PARALLEL_BACKEND"] = self.data_parallel_backend
        return "\n".join(f"export {k}={v}" for k, v in sorted(export_lines.items()))

    def pythonpath(self, stage_dir: str | Path | None) -> str:
        pythonpath_paths = ([] if stage_dir is None else [str(stage_dir)]) + os.environ.get("PYTHONPATH", "").split(":")
        return ":".join(p for p in pythonpath_paths if p)

    def sbatch_file_contents(self, task: "TrainMixin[Config]") -> str:
        output_path, error_path = task.exp_dir / "slurm.out", task.exp_dir / "slurm.err"
        stage_dir = task.stage_environment()
        comments = ([] if self.comment is None else [self.comment]) + [f"Log directory: {task.exp_dir}"]
        config_path = self.get_config_path(task)

        # Gets the extra sbatch lines.
        extra_sbatch_lines = self.extra_sbatch_lines
        if stage_dir is not None:
            extra_sbatch_lines += [f"--chdir={stage_dir}"]
            comments += [f"Code location: {stage_dir}"]
        extra_sbatch_lines_str = "\n".join(f"#SBATCH {line}" for line in extra_sbatch_lines)

        return f"""
#!/bin/bash
#SBATCH --job-name={task.task_name}
#SBATCH --partition={self.partition}
#SBATCH --requeue
#SBATCH --signal=USR1@60
#SBATCH --time={self.time_limit}
#SBATCH --comment='{'; '.join(comments)}'
#SBATCH --nodes={self.num_nodes}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --cpus-per-task={self.cpus_per_gpu}
#SBATCH --gres={self.gres}
#SBATCH --gpu-bind={self.gpu_bind}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --open-mode=append
{extra_sbatch_lines_str}

# Sets the environment variables.
export SLURM_EXPORT_ENV=ALL
export PYTHONPATH={self.pythonpath(stage_dir)}
export MASTER_PORT={self.master_port}
{self.extra_export_lines}

# Set some debugging flags.
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=1
export SHOW_FULL_IMPORT_ERROR=1
export IGNORE_REGISTRY_CACHE=1

# Make a new line in the stdout file.
echo ""
echo "***"
echo "Job ID: ${{SLURM_JOBID}} - $(date)"
echo "***"
echo ""

# Also make a new line in the stderr file.
echo "" >&2
echo "***" >&2
echo "Job ID: ${{SLURM_JOBID}} - $(date)" >&2
echo "***" >&2
echo "" >&2

# Runs the training command.
srun \\
    --nodes={self.num_nodes} \\
    --ntasks-per-node={self.gpus_per_node} \\
    --cpus-per-task={self.cpus_per_gpu} \\
    --gres={self.gres} \\
    --gpu-bind={self.gpu_bind} \\
    python -m {self.__module__} {task.task_key} {config_path}
""".strip()

    def launch(self, task: "type[TrainMixin[Config]]", *cfgs: RawConfigType) -> None:
        task_obj = task.get_task(*cfgs)

        # Writes the sbatch file.
        sbatch_path = task_obj.exp_dir / "sbatch.sh"
        with open(sbatch_path, "w", encoding="utf-8") as f:
            f.write(self.sbatch_file_contents(task_obj))

        # Calls `sbatch` on the given file.
        all_run_ids: list[str] = []
        for _ in range(self.num_jobs):
            command = ["sbatch", str(sbatch_path)]
            if all_run_ids:
                command += ["--dependency", all_run_ids[-1]]
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert proc is not None and proc.stdout is not None
            proc.wait()
            log_line = proc.stdout.read().decode("utf-8").strip()
            run_ids = re.findall(r"Submitted batch job (\d+)", log_line)
            assert len(run_ids) == 1, f"Unexpected log line: {log_line}"
            all_run_ids += [run_ids[0]]

        run_ids_str = "".join(f"\n - {run_id}" for run_id in all_run_ids)
        logger.info("Launched %d job(s):%s", len(all_run_ids), run_ids_str)

        task_obj.add_lock_file("scheduled", exists_ok=False)

    @classmethod
    def run(cls) -> None:
        if len(sys.argv) != 3:
            raise RuntimeError(f"Usage: python -m {cls.__module__} <task_key> <config_path>")
        task_key, config_path = sys.argv[1:]
        task = cls.from_components(task_key, Path(config_path))

        # Adding the "running" lock file before rmoving the "scheduled" lock
        # file in order to prevent accidentally launching another job while the
        # current job is being set up.
        task.add_lock_file("running", exists_ok=True)
        task.remove_lock_file("scheduled", missing_ok=True)

        # Sets environment variables from Slurm environment variables.
        set_slurm_master_addr_and_port()
        rank, world_size = set_slurm_rank_and_world_size()

        # Sets the initialization method and configures per-rank logging.
        set_init_method("env://")
        configure_logging(rank=rank, world_size=world_size)
        init_process_group_from_backend()

        # Gets parallelism environment variables.
        model_parallelism = int(os.environ.get("MODEL_PARALLELISM", "1"))
        pipeline_parallelism = int(os.environ.get("PIPELINE_PARALLELISM", "1"))
        backend = os.environ.get("BACKEND", None)
        model_parallel_backend = os.environ.get("MODEL_PARALLEL_BACKEND", None)
        pipeline_parallel_backend = os.environ.get("PIPELINE_PARALLEL_BACKEND", None)
        data_parallel_backend = os.environ.get("DATA_PARALLEL_BACKEND", None)

        # Sets model parallelism.
        init_parallelism(
            model_parallelism=model_parallelism,
            pipeline_parallelism=pipeline_parallelism,
            mp_backend=model_parallel_backend or backend,
            pp_backend=pipeline_parallel_backend or backend,
            dp_backend=data_parallel_backend or backend,
        )

        task.add_signal_handler(signal.SIGUSR1, requeue_job)

        # Runs the base training loop.
        task.run_training_loop()


if __name__ == "__main__":
    SlurmLauncher.run()
