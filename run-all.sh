#!/bin/bash

# run-all.sh runs the entire benchmark suite twenty times.
# First, it runs it natively once. Then, it starts up the docker
# container and runs it in docker once. After stopping the docker
# container, this whole process repeats 9 times.

# Get the source command we will need to use, based on OS
# (on Linux, rather than doing "source SOME_FILE", you do
# ". SOME_FILE").
SOURCE='source'
ENV_ACTIVATE='venv/bin/activate'

# Make sure output directory exists
if ! [ -d 'output' ]; then
	mkdir output
fi

# Make the experiment's output subdirectory
OUT="output/$(date +'%m-%d-%Y_%H-%M-%S')"
mkdir $OUT

#===========================================================
# Ensure docker and native is set up and ready to go
#===========================================================

# Step 1: build the image if not already built
#if [ $(docker images | grep 'opencv-py' | wc -l) -eq 0 ]; then
#	echo 'Building docker image opencv-py:1.0'
#	docker build --tag opencv-py:1.0 .
#fi

# Step 2: check if the venv is set up.
if ! $SOURCE $ENV_ACTIVATE; then
	echo 'CRITICAL ERROR: Failed to activate venv for native python setup'
	exit 1
fi

# Step 3: check if the packages are available inside of venv
if ! python3 -c 'import cv2 ; import numpy ; import psutil'; then
	echo 'CRITICAL ERROR: Failed to import cv2, numpy, or psutil in the native venv'
	exit 1
fi

# Shut down the venv for now
deactivate

#===========================================================
# Run 10 trials
#===========================================================

for n in {0..9}; do
	echo '=================================='
	echo "Performing trial $n"
	echo '=================================='
	
	# Native trial
	echo "Performing trial $n---native"
	$SOURCE $ENV_ACTIVATE
	cd benchmarks && python3 benchmark.py && cd ..
	deactivate

	# Get the most recent item, move it
	mv output/results.csv "$OUT/native-$n.csv"

	# Docker trial
	echo "Performing trial $n---docker"
	docker run -it --rm \
        --mount type=bind,source="$(pwd)/benchmarks",target=/benchmarks \
        --mount type=bind,source="$(pwd)/output",target=/output \
        opencv-py:1.0 python3 benchmark.py
	
	# Get the most recent item, move it
	mv output/results.csv "$OUT/docker-$n.csv"
done

echo "Experiment complete! See the results in the folder $OUT"
