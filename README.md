# docker-eval
Source code for an experiment evaluating to what extent docker can make experimental results repeatable, regardless of the platform and hardware.

## Getting Started

1. Install Docker [here](https://docs.docker.com/get-docker/).
2. Build the docker image with `docker build --tag imagebench:1.0 .`.
3. Run the docker container with
   `docker run -it --name ib imagebench:1.0 python3 benchmark.py`.
   This will run the benchmarks in the container and let you interact with
   the prompt after it finishes.
4. Stop the docker container `docker rm --force ib`.
