# docker-eval

Source code for an experiment evaluating to what extent docker can make
experimental results repeatable, regardless of the platform and hardware.

# Running the Benchmarks

1. Install Docker [here](https://docs.docker.com/get-docker/).
2. Build the opencv-py container with:
```sh
docker build --tag opencv-py:1.0 .
```
3. Add the nvidia runtime repository to apt-get:
For Linux, use this code:
```sh
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
```
For other distributions, visit https://nvidia.github.io/nvidia-container-runtime/

For Windows/Mac, it might just work with a device pass-through? (something like adding "--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599" to the run command (varies per GPU). On Ubuntu, this might still work with "--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm" but is definitely deprecated.)
```
```
4. Make sure you have a dedicated NVidia graphics driver installed
Follow for example this guide: https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/
Verify that you can run from the host system:
```sh
nvidia-smi
```
5. Install the nvidia container runtime and restart docker
```sh
apt-get install nvidia-container-runtime
sudo systemctl restart docker
```
6. If you are on mac/linux, create and run the docker container with:
```sh
docker run -dit \
	--mount type=bind,source="$(pwd)/benchmarks",target=/benchmarks \
	--mount type=bind,source="$(pwd)/output",target=/output \
	--name opencv-py \
	opencv-py:1.0
```
   If you are on Windows with PowerShell, use:
```sh
docker run -dit \
	--mount type=bind,source=${PWD}/benchmarks,target=/benchmarks \
	--mount type=bind,source=${PWD}/output,target=/output \
	--name opencv-py \
	opencv-py:1.0
```
   This will start the container in the background (hence the `-dit` flags).
   It also mounts two folders:
   - The `/benchmarks` bind mount makes the benchmarks available in the
     container.
   - The `/output` bind mount makes a folder available for any benchmark
     output which can be checked to ensure consistent results of image
     processing tasks. The results are available on the host machine and the
     docker container.
7. To add GPU support, replace the first line when starting the docker container:
(Add "--rm --gpus all" )
```sh
docker run -dit --rm --gpus all \
```

For copy/paste Linux:
```sh
docker run -dit --rm --gpus all \
	--mount type=bind,source="$(pwd)/benchmarks",target=/benchmarks \
	--mount type=bind,source="$(pwd)/output",target=/output \
	--name opencv-py \
	opencv-py:1.0
```
Windows:
```sh
docker run -dit --rm --gpus all \
	--mount type=bind,source=${PWD}/benchmarks,target=/benchmarks \
	--mount type=bind,source=${PWD}/output,target=/output \
	--name opencv-py \
	opencv-py:1.0
```
8. Run the benchmarks with:
```sh
docker exec -it opencv-py python3 benchmark.py
```
9. To `ssh` into the container, run:
```sh
docker exec -it opencv-py /bin/bash
```
10. To stop and start the container, run:
```sh
docker stop opencv-py # stops the container
docker start opencv-py # starts the container
```
11. To permanently remove a container, run:
```sh
docker rm --force opencv-py
```
