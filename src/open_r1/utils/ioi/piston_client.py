import asyncio
import os
import random
import re
import subprocess
from collections import Counter
from functools import lru_cache

import aiohttp


class PistonError(Exception):
    pass


@lru_cache(maxsize=1)
def get_piston_client_from_env():
    piston_endpoints = os.getenv("PISTON_ENDPOINTS")
    if piston_endpoints is None:
        raise ValueError(
            "For IOI problems Piston endpoints running our IOI package are required. Please add a list of valid Piston endpoints to a PISTON_ENDPOINTS varialbe in a `.env` file."
        )
    piston_endpoints = piston_endpoints.split(",") if piston_endpoints != "slurm" else get_slurm_piston_endpoints()
    random.shuffle(piston_endpoints)
    max_requests_per_endpoint = os.getenv("PISTON_MAX_REQUESTS_PER_ENDPOINT", "1")
    return PistonClient(piston_endpoints, max_requests_per_endpoint=int(max_requests_per_endpoint))


class PistonClient:
    """
    A client that will automatically load balance across multiple Piston (https://github.com/engineer-man/piston) workers.
    This assumes piston is running our custom cms_ioi package: https://github.com/guipenedo/piston/releases/
    We recommend starting the instances with the following script as otherwise some IOI problems will hit default limits:
    ```
    export PISTON_COMPILE_TIMEOUT=60000
    export PISTON_RUN_TIMEOUT=60000
    export PISTON_OUTPUT_MAX_SIZE=1000000000
    export PISTON_MAX_FILE_SIZE=1000000000
    export PISTON_DISABLE_NETWORKING=true
    export PISTON_REPO_URL=https://github.com/guipenedo/piston/releases/download/pkgs/index
    mkdir /piston

    sed -i '/app.use(body_parser.urlencoded/c\    app.use(body_parser.urlencoded({ extended: true, limit: \"512mb\" }));' src/index.js
    sed -i '/app.use(body_parser.json/c\    app.use(body_parser.json({ limit: \"512mb\" }));' src/index.js

    # Start server in background
    node src```

    Piston docs for API usage: https://piston.readthedocs.io/en/latest/api-v2/
    """

    def __init__(
        self,
        base_endpoint: str | list[str] = "http://ip-10-53-80-65:3223/api/v2",
        session=None,
        max_requests_per_endpoint=1,
    ):
        self.max_requests_per_endpoint = max_requests_per_endpoint
        self.base_endpoints = [base_endpoint] if isinstance(base_endpoint, str) else base_endpoint
        self.endpoint_ids = {endpoint: i for i, endpoint in enumerate(self.base_endpoints)}

        self._session = session
        self.endpoint_tokens = asyncio.Queue(maxsize=max_requests_per_endpoint * len(self.base_endpoints))

        for _ in range(max_requests_per_endpoint):
            for base_endpoint in self.base_endpoints:
                self.endpoint_tokens.put_nowait(base_endpoint)
        self._endpoint_failures = Counter()
        self._unhealthy_endpoints = set()
        self._endpoint_failures_lock = asyncio.Lock()

    @property
    def session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(sock_read=10),
                connector=aiohttp.TCPConnector(
                    limit=self.max_requests_per_endpoint * len(self.base_endpoints),
                    ttl_dns_cache=300,
                    keepalive_timeout=5 * 60,
                ),
            )
        return self._session

    async def _wait_for_endpoint(self):
        endpoint = await self.endpoint_tokens.get()
        return endpoint

    async def _release_endpoint(self, endpoint):
        await self.endpoint_tokens.put(endpoint)

    async def _send_request(self, endpoint, route, data=None, method="post"):
        async with self.session.request(
            method, f"{endpoint.rstrip('/')}/{route}", json=data, headers={"Content-Type": "application/json"}
        ) as response:
            return await response.json(content_type=None)

    async def _send_to_all(self, route, data=None, method="post"):
        return await asyncio.gather(
            *[self._send_request(endpoint, route, data, method) for endpoint in self.base_endpoints]
        )

    async def _send_to_one(self, endpoint, route, data=None, method="post"):
        return await self._send_request(endpoint, route, data, method)

    async def install_package(self, language, version):
        return await self._send_to_all("packages", {"language": language, "version": version}, method="post")

    async def uninstall_package(self, language, version):
        return await self._send_to_all("packages", {"language": language, "version": version}, method="delete")

    async def get_supported_runtimes(self):
        return await self._send_to_all("runtimes", method="get")

    async def execute(self, data) -> tuple[str, str]:
        """
        Requests to the IOI package return the score as a float in the stdout, as well as optional feedback/errors in stderr.
        Returns a tuple of (score, feedback).
        """
        response = await self._send_execute(data)

        if "message" in response:
            raise PistonError(response["message"])

        if "compile" in response and response["compile"]["code"] != 0:
            return "0", "Compilation error exit code " + str(response["compile"]["code"]) + "\n" + response["compile"][
                "stderr"
            ]

        if "run" not in response:
            raise PistonError(response)

        if response["run"]["code"] == 1 and "MemoryError" in response["run"]["stderr"]:
            return "0", "Memory limit exceeded"

        # successful result
        if response["run"]["stdout"]:
            return response["run"]["stdout"], response["run"]["stderr"]

        if response["run"]["signal"] == "SIGKILL":
            return "0", "Time limit exceeded"

        # other issues
        if response["run"]["code"] != 0:
            raise PistonError(
                f"language={response['language']}, version={response['version']}, exit code={response['run']['code']}, stderr={response['run']['stderr']}, signal={response['run']['signal']}"
            )
        return "0", "Unknown error"

    async def _check_failed_endpoint(self, endpoint):
        async with self._endpoint_failures_lock:
            if endpoint in self._unhealthy_endpoints:
                return
            try:
                await asyncio.sleep(5)
                await self.get_supported_runtimes()
            except Exception as e:
                print(f"Error checking endpoint {endpoint}, dropping it ({e})")
                self._unhealthy_endpoints.add(endpoint)

    async def _send_execute(self, data):
        data = data | {
            "language": "cms_ioi",
            "version": "*",
        }

        max_retries = 5
        base_delay = 1.0

        status = None
        endpoint = None

        for attempt in range(max_retries + 1):
            try:
                endpoint = await self._wait_for_endpoint()
                if attempt > 0:
                    await asyncio.sleep(1)
                async with self.session.post(
                    f"{endpoint.rstrip('/')}/execute", json=data, headers={"Content-Type": "application/json"}
                ) as response:
                    status = response.status
                    res_json = await response.json(content_type=None)

                    if status != 200:
                        raise PistonError(f"Server error. status={status}")
                    if res_json is None:
                        raise PistonError(f"Empty response. status={status}")
                    # piston overloaded
                    if "run" in res_json and "Resource temporarily unavailable" in res_json["run"].get("stderr", ""):
                        raise PistonError(f"Piston overloaded: {res_json['run']['stderr']}")
                    return res_json

            except (PistonError, asyncio.TimeoutError, aiohttp.ClientConnectionError, RuntimeError) as e:
                # Only retry if we haven't reached max retries yet
                if attempt < max_retries:
                    # Calculate backoff with jitter
                    delay = min(base_delay * (2**attempt), 10)  # Exponential backoff, capped at 10 seconds
                    jitter = delay * 0.2 * (2 * asyncio.get_event_loop().time() % 1 - 0.5)  # Add ±10% jitter
                    retry_delay = delay + jitter
                    print(f"Retrying in {retry_delay} seconds [{self.endpoint_ids[endpoint]}] {endpoint}")

                    # special case: worker died
                    if isinstance(e, aiohttp.ClientConnectionError) and "Connect call failed" in str(e):
                        await self._check_failed_endpoint(endpoint)
                    else:
                        # hopefully we won't get this one again
                        await self._release_endpoint(endpoint)
                    endpoint = None

                    await asyncio.sleep(retry_delay)
                else:
                    print(f"Giving up on retries. {e}")
                    raise e
            except Exception as e:
                print(f"Propagating exception {type(e)}: {e}")
                raise e
            finally:
                # Ensure endpoint is always released, even if an exception occurs
                if endpoint is not None:
                    try:
                        await self._release_endpoint(endpoint)
                    except Exception as e:
                        print(f"Error releasing endpoint {endpoint}: {e}")
                    endpoint = None


def get_slurm_piston_endpoints():
    """Get list of active piston worker endpoints from squeue output"""
    # Run squeue command to get job name, hostname and status, filtering for RUNNING state
    result = subprocess.run(
        ["squeue", '--format="%j %N %T"', "--noheader", "--states=RUNNING"], capture_output=True, text=True
    )

    # Split output into lines and skip header
    lines = result.stdout.strip().split("\n")

    endpoints = []
    for line in lines:
        # Parse job name from squeue output
        fields = line.split()
        job_name = fields[0].strip('"')  # Remove quotes
        hostname = fields[1]

        # Extract port if job name matches pattern
        match = re.match(r"piston-worker-(\d+)", job_name)
        if match:
            port = match.group(1)
            endpoints.append(f"http://{hostname}:{port}/api/v2")

    return endpoints
