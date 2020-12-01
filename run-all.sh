#!/bin/bash

# run-all.sh runs the entire benchmark suite twenty times.
# First, it runs it natively once. Then, it starts up the docker
# container and runs it in docker once. After stopping the docker
# container, this whole process repeats 9 times.

# Get the source command we will need to use, based on OS
# (on Linux, rather than doing "source SOME_FILE", you do
# ". SOME_FILE").

ENV_ENABLED=0 # To enable, change to 1
PYTHON='python3' # Change to 'python' if this doesn't work
SOURCE='source'
ENV_ACTIVATE='venv/bin/activate'

# Exit on control-c
trap "exit" INT

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
if [ $ENV_ENABLED -eq 1 ]; then
	if ! $SOURCE $ENV_ACTIVATE; then
		echo 'CRITICAL ERROR: Failed to activate venv for native python setup'
		exit 1
	fi
fi

# Step 3: check if the packages are available inside of venv
if ! $PYTHON -c 'import cv2 ; import numpy ; import psutil'; then
	echo 'CRITICAL ERROR: Failed to import cv2, numpy, or psutil in the native setup'
	exit 1
fi

# Shut down the venv for now
if [ $ENV_ENABLED -eq 1 ]; then
	deactivate
fi

#===========================================================
# Run 10 trials
#===========================================================

for n in {0..9}; do
	echo '=================================='
	echo "Performing trial $n"
	echo '=================================='
	
	# Native trial
	echo "Performing trial $n---native"
	if [ $ENV_ENABLED -eq 1 ]; then
		$SOURCE $ENV_ACTIVATE
	fi
	cd benchmarks && $PYTHON benchmark.py && cd ..
	if [ $ENV_ENABLED -eq 1 ]; then
		deactivate
	fi

	# Get the most recent item, move it
	mv "$(pwd)/output/results.csv" "$OUT/native-$n.csv"

	# not sure why tf we have to do this but in Docker (Windows), it won't create a new results.csv file if it doesn't exist
	touch "$(pwd)/output/results.csv"

	# Docker trial
	echo "Performing trial $n---docker"
	docker run -i --rm \
        --mount type=bind,source="$(pwd)/benchmarks",target=/benchmarks \
        --mount type=bind,source="$(pwd)/output",target=/output \
        opencv-py:1.0 python3 benchmark.py
	
	# Get the most recent item, move it
	mv "$(pwd)/output/results.csv" "$OUT/docker-$n.csv"
done

echo "Experiment complete! See the results in the folder $OUT"
