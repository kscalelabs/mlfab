"""Defines a launcher to launch a Slurm training job."""

import argparse
import datetime
import functools
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from mlfab.nn.parallel import MultiProcessConfig, init_dist, is_master
from mlfab.task.base import RawConfigType
from mlfab.task.launchers.staged import StagedLauncher
from mlfab.task.mixins.artifacts import ArtifactsMixin, Config as ArtifactsConfig
from mlfab.task.mixins.runnable import Config as RunnableConfig, RunnableMixin
from mlfab.utils.experiments import get_random_port
from mlfab.utils.logging import LOG_INFO_ALL, configure_logging
from mlfab.utils.text import outlined, show_info

logger = logging.getLogger(__name__)

DEFAULT_MASTER_PORT = 29500


def set_slurm_rank_and_world_size() -> tuple[int, int, int, int]:
    node_id = int(os.environ["SLURM_NODEID"])
    local_id = int(os.environ["SLURM_LOCALID"])
    tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    num_nodes = int(os.environ["SLURM_NNODES"])

    rank = node_id * tasks_per_node + local_id
    local_rank = local_id

    world_size = num_nodes * tasks_per_node
    local_world_size = tasks_per_node

    return rank, local_rank, world_size, local_world_size


def get_slurm_master_addr_and_port() -> tuple[str, int]:
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    assert node_list is not None, "`SLURM_JOB_NODELIST` environment variable not set"
    hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
    master_addr = hostnames.split()[0].decode("utf-8")
    master_port = int(os.environ.get("MASTER_PORT", str(DEFAULT_MASTER_PORT)))
    return master_addr, master_port


def write_message(message: str) -> None:
    sys.stderr.write(message)
    sys.stderr.flush()


def requeue_job() -> None:
    if is_master():
        if "SLURM_JOB_ID" in os.environ:
            cmd = ["scontrol", "requeue", os.environ["SLURM_JOB_ID"]]
            write_message(f"Requeuing job {os.environ['SLURM_JOB_ID']}\n")
            subprocess.check_call(cmd)
        else:
            write_message("SLURM_JOB_ID environment variable not found; not requeueing\n")


@dataclass
class PartitionInfo:
    name: str
    gpus_per_node: int
    cpus_per_node: int
    time_limit: str


@functools.lru_cache()
def parse_sinfo_output() -> list[PartitionInfo]:
    sinfo_output = subprocess.check_output(["sinfo", "--format", "%c %G %P %l"])
    partition_infos: list[PartitionInfo] = []
    lines = sinfo_output.decode("utf-8").splitlines()[1:]
    for line in lines:
        cpus_per_node, gres, name, time_limit = line.split()

        # Parses GPUs per node from gres.
        gpus_per_node_re = re.search(r"gpu:(?:[^:]*:)?(\d+)", gres)
        if gpus_per_node_re is None:
            continue
        gpus_per_node = int(gpus_per_node_re.group(1))

        # Cleans up partition name.
        name = name.replace("*", "")

        partition_infos += [PartitionInfo(name, int(gpus_per_node), int(cpus_per_node), time_limit)]

    return partition_infos


@dataclass(kw_only=True)
class SlurmArgs:
    partition: str | None
    gpus_per_node: int | None
    cpus_per_gpu: int | None
    num_nodes: int
    gpu_type: str | None
    exclusive: bool
    time_limit: str | None
    num_jobs: int
    comment: str | None
    account: str | None
    nodelist: list[str] | None
    master_port: int | None
    nccl_debug: str
    nccl_debug_subsys: str
    requeue: bool


class SlurmLauncher(StagedLauncher):
    """Defines a launcher to launch a Slurm training job.

    If no parameters are supplied, this launcher will attempt to use `sinfo` to
    find a partition with at least one GPU, and will use the first such
    partition found, with the default number of GPUs per node and CPUs per GPU.
    """

    def __init__(
        self,
        partition: str | None = None,
        gpus_per_node: int | None = None,
        cpus_per_gpu: int | None = None,
        num_nodes: int = 1,
        gpu_type: str | None = None,
        exclusive: bool = False,
        time_limit: str | None = None,
        num_jobs: int = 1,
        comment: str | None = None,
        account: str | None = None,
        nodelist: list[str] | None = None,
        master_port: int | None = None,
        tensor_parallelism: int | str = 1,
        nccl_debug: str = "WARN",
        nccl_debug_subsys: str = "ALL",
        requeue: bool = False,
    ) -> None:
        super().__init__()

        if partition is None:
            if len(sinfo_output := parse_sinfo_output()) == 0:
                raise RuntimeError("`sinfo` did not return any partitions with available GPUs!")
            partition = sinfo_output[0].name

        if gpus_per_node is None or cpus_per_gpu is None:
            try:
                first_partition = next(p for p in parse_sinfo_output() if p.name == partition)
            except StopIteration:
                raise RuntimeError(f"Partition {partition} not found in `sinfo` output")
            if gpus_per_node is None:
                gpus_per_node = first_partition.gpus_per_node
            if cpus_per_gpu is None:
                cpus_per_gpu = first_partition.cpus_per_node // first_partition.gpus_per_node

        self.partition: str = partition
        self.gpus_per_node: int = gpus_per_node
        self.cpus_per_gpu: int = cpus_per_gpu

        self.num_nodes = num_nodes
        self.gpu_type = gpu_type
        self.exclusive = exclusive
        self.time_limit = time_limit
        self.num_jobs = num_jobs
        self.comment = comment
        self.master_port = get_random_port(DEFAULT_MASTER_PORT) if master_port is None else master_port
        self.tensor_parallelism = tensor_parallelism
        self.account = account
        self.nodelist = nodelist
        self.nccl_debug = nccl_debug
        self.nccl_debug_subsys = nccl_debug_subsys
        self.requeue = requeue

    @classmethod
    def parse_args_from_cli(cls, args: list[str] | None = None) -> tuple[SlurmArgs, list[str]]:
        parser = argparse.ArgumentParser(description="Launches a Slurm job.")
        parser.add_argument("--partition", type=str, default=None, help="The partition to use")
        parser.add_argument("--gpus-per-node", type=int, default=None, help="The number of GPUs per node")
        parser.add_argument("--cpus-per-gpu", type=int, default=None, help="The number of CPUs per GPU")
        parser.add_argument("--num-nodes", type=int, default=1, help="The number of nodes to use")
        parser.add_argument("--gpu-type", type=str, default=None, help="Type of GPU to use")
        parser.add_argument("--exclusive", action="store_true", help="If set, use exclusive nodes")
        parser.add_argument("--time-limit", type=str, default=None, help="Time limit for each job")
        parser.add_argument("--num-jobs", type=int, default=1, help="The number of jobs to launch")
        parser.add_argument("--comment", type=str, default=None, help="Comment to add to each job")
        parser.add_argument("--account", type=str, default=None, help="The account to use")
        parser.add_argument("--nodelist", type=str, nargs="+", default=None, help="The list of nodes to use")
        parser.add_argument("--master-port", type=int, default=None, help="Specific master port to use")
        parser.add_argument("--nccl-debug", type=str, default="WARN", help="If set, turn off NCCL debug logs")
        parser.add_argument("--nccl-debug-subsys", type=str, default="INIT,P2P", help="Subsystem debugging options")
        parser.add_argument("--requeue", action="store_true", help="If set, requeue the job on USR1 signal")
        args, remaining_args = parser.parse_known_intermixed_args(args=args)

        return (
            SlurmArgs(
                partition=args.partition,
                gpus_per_node=args.gpus_per_node,
                cpus_per_gpu=args.cpus_per_gpu,
                num_nodes=args.num_nodes,
                gpu_type=args.gpu_type,
                exclusive=args.exclusive,
                time_limit=args.time_limit,
                num_jobs=args.num_jobs,
                comment=args.comment,
                account=args.account,
                nodelist=args.nodelist,
                master_port=args.master_port,
                nccl_debug=args.nccl_debug,
                nccl_debug_subsys=args.nccl_debug_subsys,
                requeue=args.requeue,
            ),
            remaining_args,
        )

    @property
    def extra_sbatch_lines(self) -> list[str]:
        sbatch_lines: list[str] = []
        if "EMAIL" in os.environ:
            sbatch_lines += [f"--mail-user={os.environ['EMAIL']}", "--mail-type=ALL"]
        if self.account is not None:
            sbatch_lines += [f"--account={self.account}"]
        if self.exclusive:
            sbatch_lines += ["--exclusive"]
        if self.time_limit is not None:
            sbatch_lines += [f"--time={self.time_limit}"]
        if self.nodelist is not None and len(self.nodelist) > 0:
            sbatch_lines += [f"--nodelist={','.join(self.nodelist)}"]
        if self.requeue:
            sbatch_lines += ["--requeue"]
        return sbatch_lines

    @property
    def extra_export_lines(self) -> str:
        export_lines: dict[str, str] = {}
        if self.tensor_parallelism != 1:
            export_lines["TENSOR_PARALLELISM"] = str(self.tensor_parallelism)
        if self.requeue:
            export_lines["REQUEUE_SLURM_JOB"] = "1"
        return "".join(f"\nexport {k}={v}" for k, v in sorted(export_lines.items()))

    def pythonpath(self, stage_dir: str | Path | None) -> str:
        pythonpath_paths = ([] if stage_dir is None else [str(stage_dir)]) + os.environ.get("PYTHONPATH", "").split(":")
        return ":".join(p for p in pythonpath_paths if p)

    def sbatch_file_contents(self, task: "ArtifactsMixin[ArtifactsConfig]") -> str:
        output_path = task.exp_dir / "slurm" / "job-%4t.slurm"
        nccl_path = task.exp_dir / "nccl.txt"
        stage_dir = task.stage_environment()
        comments = ([] if self.comment is None else [self.comment]) + [f"Log directory: {task.exp_dir}"]
        config_path = self.get_config_path(task, use_cli=False)

        # Gets the extra sbatch lines.
        extra_sbatch_lines = self.extra_sbatch_lines
        if stage_dir is not None:
            extra_sbatch_lines += [f"--chdir={stage_dir}"]
            comments += [f"Code location: {stage_dir}"]
        extra_sbatch_lines_str = "".join(f"\n#SBATCH {line}" for line in extra_sbatch_lines)

        # Adds some extra information to the job information line.
        job_info: dict[str, str] = {
            "launch_time": datetime.datetime.now().isoformat(),
            "task_key": task.task_key,
            "exp_dir": str(task.exp_dir),
        }
        if self.comment is not None:
            job_info["comment"] = self.comment
        if self.nodelist is not None:
            job_info["nodelist"] = ", ".join(self.nodelist)
        if self.account is not None:
            job_info["account"] = self.account
        if self.time_limit is not None:
            job_info["time_limit"] = self.time_limit

        log_data = {
            "job_id": "${SLURM_JOB_ID}",
            "job_start_time": "${launch_timestamp}",
            "node_list": "${SLURM_NODELIST}",
            "job": job_info,
        }

        return f"""
#!/bin/bash
#SBATCH --job-name={task.task_name}
#SBATCH --partition={self.partition}
#SBATCH --signal=USR1@60
#SBATCH --comment='{'; '.join(comments)}'
#SBATCH --nodes={self.num_nodes}
#SBATCH --ntasks-per-node={self.gpus_per_node}
#SBATCH --cpus-per-gpu={self.cpus_per_gpu}
#SBATCH --gpus-per-node={self.gpus_per_node}
#SBATCH --output={output_path}
#SBATCH --error={output_path}
#SBATCH --open-mode=append{extra_sbatch_lines_str}

# Sets the environment variables.
export SLURM_EXPORT_ENV=ALL
export PYTHONPATH={self.pythonpath(stage_dir)}
export MASTER_PORT={self.master_port}{self.extra_export_lines}

# Torch debugging flags.
export TORCH_DISABLE_ADDR2LINE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# NCCL debugging flags.
export NCCL_DEBUG_FILE={nccl_path}
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG={self.nccl_debug}
export NCCL_DEBUG_SUBSYS={self.nccl_debug_subsys}

# Disable Tensorboard in Slurm.
export TENSORBOARD_PORT=-1

# Create a dictionary in JSON format
launch_timestamp=$(date)
slurm_info_file={task.exp_dir}/slurm_info.json
log_data=$(cat <<EOF
{json.dumps(log_data, indent=2)}
EOF
)

# Check if the file exists and if not, create it with an empty array
if [ ! -f $slurm_info_file ]; then
    echo "[]" > $slurm_info_file
fi

# Append the log data to the JSON file
jq ". + [$log_data]" $slurm_info_file > tmp.$$.json && mv tmp.$$.json $slurm_info_file

# Runs the training command.
srun \\
    --ntasks-per-node={self.gpus_per_node} \\
    --cpus-per-gpu={self.cpus_per_gpu} \\
    --gpus-per-node={self.gpus_per_node} \\
    --output={output_path} \\
    --error={output_path} \\
    python -m {self.__module__} {task.task_key} {config_path}
""".strip()

    def launch(
        self,
        task: "type[RunnableMixin[RunnableConfig]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        if not issubclass(task, ArtifactsMixin):
            raise RuntimeError(f"Task {task} must be an `ArtifactsMixin`")

        # Creates the task using the meta device to avoid instantiating weights.
        with torch.device("meta"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            task_obj = task.get_task(*cfgs, use_cli=use_cli)

        # Writes the sbatch file.
        sbatch_path = task_obj.exp_dir / "sbatch.sh"
        with open(sbatch_path, "w", encoding="utf-8") as f:
            f.write(self.sbatch_file_contents(task_obj))

        # Calls `sbatch` on the given file.
        all_run_ids: list[str] = []
        for _ in range(self.num_jobs):
            command = ["sbatch"]
            if all_run_ids:
                command += ["--dependency", all_run_ids[-1]]
            command += [str(sbatch_path)]
            logger.info("Command: %s", command)
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
        show_info(f"Launched {len(all_run_ids)} job(s) to {task_obj.exp_dir}:{run_ids_str}")

        task_obj.add_lock_file("scheduled", exists_ok=False)

    @classmethod
    def run(cls) -> None:
        if len(sys.argv) != 3:
            raise RuntimeError(f"Usage: python -m {cls.__module__} <task_key> <config_path>")

        # Gets Slurm information.
        master_addr, master_port = get_slurm_master_addr_and_port()
        node_rank = int(os.environ["SLURM_NODEID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_world_size = int(os.environ["SLURM_NNODES"])
        local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        rank = node_rank * local_world_size + local_rank
        world_size = node_world_size * local_world_size

        # Sets the initialization method and configures per-rank logging.
        configure_logging(rank=rank, world_size=world_size)

        # Logs Nvidia information.
        if shutil.which("nvidia-smi") is not None:
            with subprocess.Popen(
                ["nvidia-smi", "topo", "--matrix"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as proc:
                stdout, stderr = proc.communicate()
                if stderr:
                    logger.error("Error running `nvidia-smi`: %s", stderr.decode("utf-8"))
                else:
                    logger.log(LOG_INFO_ALL, "Nvidia GPUs:\n%s", stdout.decode("utf-8"))

        # Gets parallelism environment variables.
        tensor_parallelism = os.environ.get("TENSOR_PARALLELISM", "1")
        requeue = os.environ.get("REQUEUE_SLURM_JOB", "0") == "1"

        # Sets tensor parallelism.
        cfg = MultiProcessConfig(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            local_world_size=local_world_size,
            master_addr=master_addr,
            master_port=master_port,
            tensor_parallelism=tensor_parallelism,
        )
        init_dist(cfg)

        task_key, config_path = sys.argv[1:]
        task = cls.from_components(task_key, Path(config_path), use_cli=False)

        if not isinstance(task, RunnableMixin):
            raise RuntimeError(f"Task {task} must be a `RunnableMixin`")

        # Adding the "running" lock file before rmoving the "scheduled" lock
        # file in order to prevent accidentally launching another job while the
        # current job is being set up.
        task.add_lock_file("running", exists_ok=True)
        task.remove_lock_file("scheduled", missing_ok=True)
        if requeue:
            task.add_signal_handler(requeue_job, signal.SIGUSR1)

        # Runs the base training loop.
        task.run()


if __name__ == "__main__":
    # Prints a header with some information about the job.
    print(
        outlined(
            [
                f"Job ID: {os.environ.get('SLURM_JOBID', 'MISSING')}",
                f"Node ID: {os.environ.get('SLURM_NODEID', 'MISSING')}",
                f"Local ID: {os.environ.get('SLURM_LOCALID', 'MISSING')}",
                f"Host: {os.environ.get('SLURMD_NODENAME', 'MISSING')}",
                f"All nodes: {os.environ.get('SLURM_NODELIST', 'MISSING')}",
                f"Launch time: {datetime.datetime.now()}",
            ],
        ),
        flush=True,
    )

    SlurmLauncher.run()
