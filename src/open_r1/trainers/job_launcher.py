import atexit
import os
import re
import subprocess
import time


# We need a special environment setup to launch vLLM from within Slurm training jobs.
# - Reference code: https://github.com/huggingface/brrr/blob/c55ba3505686d690de24c7ace6487a5c1426c0fd/brrr/lighteval/one_job_runner.py#L105
# - Slack thread: https://huggingface.slack.com/archives/C043JTYE1MJ/p1726566494958269
user_home_directory = os.path.expanduser("~")
SLURM_PREFIX = [
    "env",
    "-i",
    "bash",
    "-c",
    f"for f in /etc/profile.d/*.sh; do source $f; done; export HOME={user_home_directory}; sbatch --qos=high --output=/fsx/h4/logs/%x-%j.out --err=/fsx/h4/logs/%x-%j.err ",
]


class SGLangSlurmJobLauncher:
    def __init__(
        self,
        model_id_or_path,
        model_revision="main",
        num_gpus=1,
        sglang_port=30010,
        slurm_script="slurm/launch_sglang.slurm",
        check_interval=5,
    ):
        """
        Initialize the job launcher.

        :param slurm_script: Path to the SLURM script.
        :param check_interval: Time interval (seconds) to check job status.
        """
        self.slurm_script = slurm_script
        self.job_id = None
        self.node_name = None
        self.check_interval = check_interval
        self.model_id_or_path = model_id_or_path
        self.model_revision = model_revision
        self.num_gpus = num_gpus
        self.sglang_port = sglang_port

        # Register cleanup function to cancel job on exit
        atexit.register(self.cleanup)

    def submit_job(self):
        """Submits the SLURM job and extracts the job ID."""

        cmd = SLURM_PREFIX.copy()
        cmd_args = [
            f"--gres=gpu:{self.num_gpus}",
            self.slurm_script,
            self.model_id_or_path,
            self.model_revision,
            str(self.sglang_port),
        ]
        cmd[-1] += " " + " ".join(cmd_args)
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                self.job_id = match.group(1)
                print(f"Job submitted with ID: {self.job_id}")
            else:
                raise RuntimeError("Failed to retrieve job ID from sbatch output.")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e.stderr}")
            raise


    def get_job_status(self):
        """Checks the job status using squeue."""
        if not self.job_id:
            raise ValueError("Job ID is not set. Submit the job first.")

        result = subprocess.run(
            ["squeue", "--job", self.job_id, "--noheader"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if not result.stdout.strip():
            return None  # Job is no longer in queue
        status = result.stdout.split()[4]  # Typically, state is the 5th column
        return status

    def wait_for_job_to_start(self):
        """Waits for the job to start running and fetches its node."""
        print("Waiting for job to start...")
        while True:
            status = self.get_job_status()
            if status is None:
                raise RuntimeError("Job disappeared from queue, it may have failed.")
            if status == "R":  # Running
                print("Job is running. Fetching node information...")
                self.node_name = self.get_node_name()
                return
            time.sleep(self.check_interval)

    def get_node_name(self):
        """Gets the node where the job is running."""
        result = subprocess.run(
            ["squeue", "--job", self.job_id, "--noheader", "--format=%N"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.stdout.strip():
            return result.stdout.strip()
        else:
            raise RuntimeError("Failed to retrieve node name.")

    def get_node_ip(self):
        """Retrieves the IP address of the node running the job."""
        if not self.node_name:
            raise ValueError("Node name is not set. Wait for the job to start first.")

        result = subprocess.run(
            ["scontrol", "show", "node", self.node_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        match = re.search(r"NodeAddr=(\S+)", result.stdout)
        if match:
            return match.group(1)
        else:
            raise RuntimeError("Failed to retrieve node IP address.")

    def launch(self):
        """Launches the job, waits for it to start, and retrieves the node IP."""
        self.submit_job()
        self.wait_for_job_to_start()
        ip_address = self.get_node_ip()
        print(f"Job is running on {self.node_name} with IP: {ip_address}")
        self.ip_address = ip_address
        return ip_address

    def cleanup(self):
        """Cancels the SLURM job if it is still running."""
        if self.job_id is not None:
            print(f"Cleaning up: Cancelling job {self.job_id}...")
            subprocess.run(["scancel", self.job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Job cancelled.")

    def __del__(self):
        """Ensure job cleanup when the instance is destroyed."""
        self.cleanup()


if __name__ == "__main__":
    from open_r1.trainers.remote_model import RemoteModel

    launcher = SGLangSlurmJobLauncher("HuggingFaceTB/SmolLM2-135M-Instruct")
    ip_address = launcher.launch()
    launcher.ip_address
    time.sleep(15)
    remote_model = RemoteModel(f"{ip_address}", 30010)
    remote_model.wait_for_server()

    result = remote_model.generate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    assert 0
    
    print(result)
