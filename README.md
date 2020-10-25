# docker-eval
Source code for an experiment evaluating to what extent docker can make experimental results repeatable, regardless of the platform and hardware.

## Getting Started

1. Install Docker [here](https://docs.docker.com/get-docker/).
2. Go to the `opencv-contrib` folder and run `docker build --tag opencvcontrib:1.0 .`. This will build a Docker image containing the non-free features (SIFT/SURF).
3. Go to the root folder and build the docker image with `docker build --tag imagebench:1.0 .`.
4. Create and run the docker container with
   `docker run -dit --name ib imagebench:1.0`. This will start the
   container in the background (hence the `-dit` flags).
   Note that every time you make changes to the `Dockerfile` and rebuild
   the image, you should create a new container (and probably delete the
   old one---see step 7).
5. Run the benchmarks using `docker exec -it ib python3 benchmark.py`. This
   will run all benchmarks and show results as it progresses. It may take a
   while.
6. If you want to `ssh` into the container, run `docker exec -it ib /bin/bash`.
   This allows you to manually run commands in the container. Make sure you
   type `exit` to leave `ssh`.
7. If you want to stop the container, run `docker stop ib`. Now that your
   container has been created, you can start it back up again with `docker
   start ib`. Note that the stop command may take a while.
8. If you want to permanently remove the container before creating a fresh
   one, run `docker rm --force ib`.
