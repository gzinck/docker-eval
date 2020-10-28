# Benchmarking OpenCV with Python

1. Install Docker [here](https://docs.docker.com/get-docker/).
2. Build the opencv-py container with:
```sh
docker build --tag opencv-py:1.0 .
```
3. Create and run the docker container with:
```sh
docker run -dit \
	--mount type=bind,source="$(pwd)",target=/benchmarks \
	--name opencv-py \
	opencv-py:1.0
```
   This will start the container in the background (hence the `-dit` flags).
   Note that every time you make changes to the `Dockerfile` and rebuild
   the image, you should create a new container (and probably delete the
   old one---see step 7).
4. Run the benchmarks with:
```sh
docker exec -it opencv-py python3 benchmark.py
```
5. To `ssh` into the container, run:
```sh
docker exec -it opencv-py /bin/bash
```
6. To stop and start the container, run:
```sh
docker stop opencv-py # stops the container
docker start opencv-py # starts the container
```
7. To permanently remove a container, run:
```sh
docker rm --force opencv-py
```
