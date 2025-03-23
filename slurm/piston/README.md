# Piston workers (slurm)

We have built a [piston](https://github.com/engineer-man/piston) package to run IOI problems.

To launch a fleet of piston workers on a slurm cluster, you can adapt the paths in `launch_piston_workers.sh` and `launch_single_piston.sh` and run:
```bash
slurm/piston/launch_piston_workers.sh (number of workers to launch)
```

This command will launch a slurm job for each worker, which will be called `piston-worker-<port>`, where `<port>` is the port where the worker will be listening.

## First time setup
You will need to install the [IOI package](https://github.com/guipenedo/piston/tree/master/packages/cms_ioi/1.0.0) in the workers.
1. Launch a single worker:
```bash
slurm/piston/launch_piston_workers.sh 1
```

2. Assuming it's running on `ip-10-53-86-146:1234`, send the package install request:
```bash
curl -X POST http://ip-10-53-86-146:1234/api/v2/packages -H "Content-Type: application/json" -d '{"language": "cms_ioi", "version": "1.0.0"}'
```

3. You can now launch more workers and due to the shared mounted packages directory, they should already have the package installed.

To have the main script find the workers automatically, you can export the following environment variable:
```bash
export PISTON_ENDPOINTS=slurm
```
Alternatively your can add `PISTON_ENDPOINTS=slurm` to your .env file.

You can also change `PISTON_MAX_REQUESTS_PER_ENDPOINT`, which tries to limit how many simultaneous requests each worker will handle (1 by default). Keep in mind that this is a local limit and in distributed setups, as there is no global limit, workers might sometimes be overwhelmed when some processes hit the same worker.

If you would like to adapt the code to run without piston, please see the [ioi repo](https://github.com/huggingface/ioi).

# Piston workers (local docker)
This will launch a single worker in a docker container. Consider launching multiple workers for better scalability. Replace 2000 with the port you want to use.
Make sure to change `/path/to/local/packages` to the path you want to persist for package installs.

```bash
docker run -d \
  --name piston_worker \
  -v /path/to/local/packages:/piston/packages \
  -e PORT=2000 \
  -e PISTON_COMPILE_TIMEOUT=60000 \
  -e PISTON_RUN_TIMEOUT=60000 \
  -e PISTON_OUTPUT_MAX_SIZE=1000000000 \
  -e PISTON_MAX_FILE_SIZE=1000000000 \
  -e PISTON_DISABLE_NETWORKING=true \
  -e PISTON_REPO_URL=https://github.com/guipenedo/piston/releases/download/pkgs/index \
  -p 2000:2000 \
  --entrypoint /bin/bash \
  ghcr.io/engineer-man/piston@sha256:63b5654156a89c5a2ad281aface21416615d62ec056d88efe8fcd307ce73575a \
  -c "sed -i '/app.use(body_parser.urlencoded/c\    app.use(body_parser.urlencoded({ extended: true, limit: \"512mb\" }));' src/index.js && \
      sed -i '/app.use(body_parser.json/c\    app.use(body_parser.json({ limit: \"512mb\" }));' src/index.js && \
      node src"
```

Install the package:
```bash
curl -X POST http://localhost:2000/api/v2/packages -H "Content-Type: application/json" -d '{"language": "cms_ioi", "version": "1.0.0"}'
```

Remember to set `PISTON_ENDPOINTS`:
```bash
export PISTON_ENDPOINTS=http://localhost:2000/api/v2,http://localhost:2001/api/v2,http://localhost:2002/api/v2
```
