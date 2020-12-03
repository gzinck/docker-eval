import cv2
import statistics
import time
import numpy as np
import gc
import csv
import os
import socket
import random
import hashlib
import sys
from datetime import datetime
from thread_logging import pre_benchmark_logging
from thread_logging import post_benchmark_logging
from thread_logging import initialize_logging
from thread_logging import return_logging
from thread_logging import memory_logging

np.set_printoptions(threshold=sys.maxsize)

now = datetime.now()
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
log_name = str(socket.gethostname()) + "-" + str(date_time)

if not os.path.exists("../output"):
	os.makedirs("../output")
csv_file = open("../output/results.csv", "w")
csv_file_timed = open("../output/" + log_name + ".csv", "w")
results_text = open("../output/results_text.txt", "w")

csv_writer = csv.writer(csv_file)
csv_writer_timed = csv.writer(csv_file_timed)

#Kernel widths to test
widths = [3, 11]

csv_contents = [['Benchmark', 
				'Source Video', 
				'Image Path', 
				'Kernel Width (# pixels)', 
				'Average Time Elapsed (milliseconds)', 
				'Variance of Time Elapsed (milliseconds)', 
				'Time Elapsed 1 (milliseconds)', 
				'Time Elapsed 2 (milliseconds)', 
				'Time Elapsed 3 (milliseconds)', 
				'Time Elapsed 4 (milliseconds)', 
				'Time Elapsed 5 (milliseconds)', 
				'Time Elapsed 6 (milliseconds)', 
				'Time Elapsed 7 (milliseconds)', 
				'Time Elapsed 8 (milliseconds)', 
				'Time Elapsed 9 (milliseconds)', 
				'Time Elapsed 10 (milliseconds)', 
				'Average of average CPU usage (% of CPU)', 
				'Variance of average CPU usage (% of CPU)', 
				'CPU usage 1 (% of CPU)', 
				'CPU usage 2 (% of CPU)', 
				'CPU usage 3 (% of CPU)', 
				'CPU usage 4 (% of CPU)', 
				'CPU usage 5 (% of CPU)', 
				'CPU usage 6 (% of CPU)', 
				'CPU usage 7 (% of CPU)', 
				'CPU usage 8 (% of CPU)', 
				'CPU usage 9 (% of CPU)', 
				'CPU usage 10 (% of CPU)', 
				'Average memory usage (% of memory)', 
				'Variance of memory usage (% of memory)', 
				'memory usage 1 (% of memory)', 
				'memory usage 2 (% of memory)', 
				'memory usage 3 (% of memory)', 
				'memory usage 4 (% of memory)', 
				'memory usage 5 (% of memory)', 
				'memory usage 6 (% of memory)', 
				'memory usage 7 (% of memory)', 
				'memory usage 8 (% of memory)', 
				'memory usage 9 (% of memory)', 
				'memory usage 10 (% of memory)']]

global benchCounter

def measureMemoryUsage():
	return memory_logging()

def loadImageBench(imgPath):
	return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def gaussianBlur(img):
	#cv2.imwrite("gaussianBlur.png", img) 
	blurred = cv2.GaussianBlur(img, (region_width, region_width), 0)
	#cv2.imwrite("gaussianBlur2.png", blurred) 
	#cv2.imwrite("gaussianBlur3.png", img) 

def gradientSobel(img):
	# Example converted it to CV_64F, which caused a massive slow down. Keeping it integer
	cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=1)
	cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=1)
	#cv2.imwrite("gradientSobel.png", img) 

def meanThresh(img):
	cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, region_width, 0)
	#cv2.imwrite("meanThresh.png", img) 

def computeHistogram(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	results_text.write(str(hist))

def goodFeatures(img):
	# Documentation was ambiguous if this was a weighted or unweighted variant. This ambiguity was resolved
	# by running it on a chessboard image and seeing where the corners were found. it found them inside the square
	# and not on the corner, therefor it was the unweighted variant. I also inspected the C++ source code
	# and couldn't found any indication that Gaussian blur was applied
	kp = cv2.goodFeaturesToTrack(img,0,qualityLevel=0.016,minDistance=10,blockSize=21)
	#results_text.write("goodFeatures")
	results_text.write(str(kp))


def computeCanny(img):
	# OpenCV's canny edge creates a binary image. This isn't very useful by itself. To process the edges you need
	# to extract the contours from the output binary image. I've used the values specified in an opencv example
	# https://docs.opencv.org/3.4.3/df/d0d/tutorial_find_contours.html
	edges = cv2.Canny(img, 15, 110)
	#cv2.imwrite("edges.png", img) 
	results_text.write(str(edges))
	contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

if hasattr(cv2, 'xfeatures2d'):
	# This has been configured to be the same as the Lowe's paper. 3 layers per octave.
	# It's not clear how many octaves are used and if the first layer is at twice the input as recommend by Lowe but
	# frequently not done due to speed hit
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
	def detectSift(img):
		kp,des = sift.detectAndCompute(img, None)
		#results_text.write("kp")
		results_text.write(str(kp))
		results_text.write(str(des))

	surf = cv2.xfeatures2d.SURF_create(hessianThreshold=420, nOctaves=4, nOctaveLayers=4, extended=False, upright=False)
	def detectSurf(img):
		kp,des = surf.detectAndCompute(img, None)
		#results_text.write("kp")
		results_text.write(str(kp))
		results_text.write(str(des))


def contour(img):
	# Convert to black and white first
	img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
	#cv2.imwrite("contour.png", img_binary) 

	# Find Contours
	contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	#results_text.write("contours")
	results_text.write(str(contours))
	results_text.write(str(hierarchy))
	#results_text.write(hashlib.sha256(contours).digest())

def houghLine(img):
	lines = cv2.HoughLines(img, rho=5, theta=np.pi/180, threshold=15000)
	results_text.write(str(lines))


def resize(img):
	img_width_original = img.shape[0]
	img_height_original = img.shape[1]
	#reduce the size of the image to a quarter of the original -- linear by default
	smaller_image = cv2.resize(img,(img_width_original//2,img_height_original//2)) 

	#Bring it back up to the size it was before -- linear by default
	larger_image = cv2.resize(img,(img_width_original,img_height_original)) 
	#cv2.imwrite("resize.png", smaller_image) 

def rotate(img):
	rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	#cv2.imwrite("rotate.png", rotated) 
  
def mirror(img):
	mirrored = cv2.flip(img, 0)
	mirrored = cv2.flip(img, 1)
	mirrored = cv2.flip(img, -1)
	#cv2.imwrite("mirror.png", mirrored) 

def benchmark(f, img, num_trials=1):
	gc.collect()
	times=[]
	initialize_logging()
	for trials in range(num_trials):
		t0 = time.time()
		new_img = img
		f(img)
		t1 = time.time()
		times.append((t1-t0)*1000)
	cpuMean = return_logging()
	memoryUsageAfterBenchmark = measureMemoryUsage()
	runtime = times[0]
	return [runtime] + [cpuMean] + [memoryUsageAfterBenchmark]

def addToTempStorage(storage, data):
	global benchCounter
	storage[benchCounter].append(data)
	benchCounter += 1
	return storage

for width in widths:
	print("Kernel Width: " + str(width))
	region_width = width

	dataDir = "./data"
	subfolders = [ entry.name for entry in os.scandir(dataDir) if entry.is_dir() ]
	for folder in subfolders:
		directory = os.path.join(dataDir, folder)
		source_video = str(folder)
		tempDataStorage = [[], [], [], [], [], [], [], [], [], [], [], [], [], []] # one for each benchmark
		for filename in os.listdir(directory):
			if filename.endswith(".jpg") or filename.endswith(".png"):

				random.seed(69420)
				cv2.setRNGSeed(69420)
				image_path = os.path.join(directory, filename)
				print("Image path: " + str(image_path))

				global benchCounter
				benchCounter = 0

				tempDataStorage = addToTempStorage(tempDataStorage, [measureMemoryUsage.__name__, source_video, image_path, region_width] + benchmark(loadImageBench, image_path))

				img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

				#Run all the tests again with the new parameters
				tempDataStorage = addToTempStorage(tempDataStorage, [resize.__name__, source_video, image_path, region_width] + benchmark(resize, img, 10))
				tempDataStorage = addToTempStorage(tempDataStorage, [rotate.__name__, source_video, image_path, region_width] + benchmark(rotate, img, 10))
				tempDataStorage = addToTempStorage(tempDataStorage, [mirror.__name__, source_video, image_path, region_width] + benchmark(mirror, img, 10))
				tempDataStorage = addToTempStorage(tempDataStorage, [contour.__name__, source_video, image_path, region_width] + benchmark(contour, img))
				tempDataStorage = addToTempStorage(tempDataStorage, [gaussianBlur.__name__, source_video, image_path, region_width] + benchmark(gaussianBlur, img, 10))
				tempDataStorage = addToTempStorage(tempDataStorage, [meanThresh.__name__, source_video, image_path, region_width] + benchmark(meanThresh, img))
				tempDataStorage = addToTempStorage(tempDataStorage, [gradientSobel.__name__, source_video, image_path, region_width] + benchmark(gradientSobel, img))
				tempDataStorage = addToTempStorage(tempDataStorage, [computeHistogram.__name__, source_video, image_path, region_width] + benchmark(computeHistogram, img, 10))
				tempDataStorage = addToTempStorage(tempDataStorage, [computeCanny.__name__, source_video, image_path, region_width] + benchmark(computeCanny, img))
				if hasattr(cv2, 'xfeatures2d'):
					tempDataStorage = addToTempStorage(tempDataStorage, [detectSift.__name__, source_video, image_path, region_width] + benchmark(detectSift, img))
					tempDataStorage = addToTempStorage(tempDataStorage, [detectSurf.__name__, source_video, image_path, region_width] + benchmark(detectSurf, img))
				else:
					print("Skipping SIFT and SURF. Not installed")
				tempDataStorage = addToTempStorage(tempDataStorage, [goodFeatures.__name__, source_video, image_path, region_width] + benchmark(goodFeatures, img))
				tempDataStorage = addToTempStorage(tempDataStorage, [houghLine.__name__, source_video, image_path, region_width] + benchmark(houghLine, img))
				
		
			else:
				continue

		for tempData in tempDataStorage:
			times=[]
			cpus=[]
			memorys=[]
			for trial_data in tempData:
				times.append(trial_data[4])
				cpus.append(trial_data[5])
				memorys.append(trial_data[6])

			if (len(times) <= 2):
				time_variance = 'N/A'
				cpu_variance = 'N/A'
				memory_variance = 'N/A'
			else:
				time_variance = statistics.variance(times)
				cpu_variance = statistics.variance(cpus)
				memory_variance = statistics.variance(memorys)

			csv_contents.append([tempData[0][0], tempData[0][1], tempData[0][2] + " and 9 others", tempData[0][3]] + [statistics.mean(times)] + [time_variance] + times + [statistics.mean(cpus)] + [cpu_variance] + cpus + [statistics.mean(memorys)] + [memory_variance] + memorys)


post_benchmark_logging(date_time)

csv_writer.writerows(csv_contents)
csv_writer_timed.writerows(csv_contents)
results_text.close()
print("Done!")
