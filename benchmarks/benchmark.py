import cv2
import statistics
import time
import numpy as np
import gc
import csv
from thread_logging import pre_benchmark_logging
from thread_logging import post_benchmark_logging

csv_file = open("./results.csv", "w")

csv_writer = csv.writer(csv_file)

csv_contents = [['Benchmark', 'Image Path', 'Binary Path', 'Kernel Width', 'Average Time Elapsed (seconds)', 'Variance of Time Elapsed', 'Time Elapsed 1', 'Time Elapsed 2', 'Time Elapsed 3', 'Time Elapsed 4', 'Time Elapsed 5', 'Time Elapsed 6', 'Time Elapsed 7', 'Time Elapsed 8', 'Time Elapsed 9', 'Time Elapsed 10']]


def gaussianBlur():
	cv2.GaussianBlur(img, (region_width, region_width), 0)

def gradientSobel():
	# Example converted it to CV_64F, which caused a massive slow down. Keeping it integer
	cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=1)
	cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=1)

def meanThresh():
	cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, region_width, 0)

def computeHistogram():
	hist =cv2.calcHist([img],[0],None,[256],[0,256])
	# for i in range(len(hist)):
	#	  print("[{:3d}] {}".format(i,hist[i]))
	# print()
	# print("total = {}".format(sum(hist)))

def goodFeatures():
	# Documentation was ambiguous if this was a weighted or unweighted variant. This ambiguity was resolved
	# by running it on a chessboard image and seeing where the corners were found. it found them inside the square
	# and not on the corner, therefor it was the unweighted variant. I also inspected the C++ source code
	# and couldn't found any indication that Gaussian blur was applied
	kp=cv2.goodFeaturesToTrack(img,0,qualityLevel=0.016,minDistance=10,blockSize=21)
	# print("Shi-Tomasi count {}".format(len(kp)))

def computeCanny():
	# OpenCV's canny edge creates a binary image. This isn't very useful by itself. To process the edges you need
	# to extract the contours from the output binary image. I've used the values specified in an opencv example
	# https://docs.opencv.org/3.4.3/df/d0d/tutorial_find_contours.html
	edges = cv2.Canny(img, 15, 110)
	# print("total canny {}".format((np.asarray(edges) > 100).sum()))
	# Not approximating chains here since in the general purpose algorithms I've done a chain approximation would
	# require additional work.
	contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# total = sum( len(c) for c in contours )
	# print("total canny contour {}".format(total))

# SIFT and SURF are not included in the standard distribution of Python OpenCV due to legal concerns
# Doing a custom build of OpenCV is beyond the scope scope of this benchmark since its only supposed to
# include what's easily available
if hasattr(cv2, 'xfeatures2d'):
	# This has been configured to be the same as the Lowe's paper. 3 layers per octave.
	# It's not clear how many octaves are used and if the first layer is at twice the input as recommend by Lowe but
	# frequently not done due to speed hit
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
	def detectSift():
		kp,des = sift.detectAndCompute(img, None)
		# print("SIFT found {:d}".format(len(kp)))

	# original paper had 4 scales per octave
	# threshold tuned to detect 10,000 features
	
	surf = cv2.xfeatures2d.SURF_create(hessianThreshold=420, nOctaves=4, nOctaveLayers=4, extended=False, upright=False)
	def detectSurf():
		kp,des = surf.detectAndCompute(img, None)
		# print("SURF found {:d}".format(len(kp)))


def contour():
	# Code in Validation Boof attempted to replicate the same behavior in both libraries. Configuration
	# values are taken from there.
	# The algorithm in OpenCV and the two contour algorithms in BoofCV operate internally very different
	# How OpenCV defines external and BoofCV define external is different
	contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	# total = sum( len(c) for c in contours )
	# print("total pixels {}".format(total))

def houghLine():
	# OpenCV contains 3 Hough Line algorithms. CV_HOUGH_STANDARD is the closest to the variants included
	# in boofcv. The other OpenCV variants should have very different behavior based on their description

	lines = cv2.HoughLines(img, rho=5, theta=np.pi/180, threshold=15000)
	# print("total lines {}".format(len(lines)))

#common operations to expand ML dataset:
def resize():
  
	img_width_original = img.shape[0]
	img_height_original = img.shape[1]
	#reduce the size of the image to a quarter of the original -- linear by default
	smaller_image = cv2.resize(img,(img_width_original//2,img_height_original//2)) 

	#Bring it back up to the size it was before -- linear by default
	larger_image = cv2.resize(img,(img_width_original,img_height_original)) 

def rotate():
	rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
  
def mirror():
	mirrored = cv2.flip(img, 0)
	mirrored = cv2.flip(img, 1)
	mirrored = cv2.flip(img, -1)

def benchmark( f , num_trials=10):
	
	gc.collect()
	times=[]
	for trials in range(num_trials):
		t0 = time.time()
		f()
		t1 = time.time()
		times.append((t1-t0)*1000)
	if (len(times) <= 2):
		time_variance = 'N/A'
	else:
		time_variance = statistics.variance(times)
	return [statistics.mean(times)] + [time_variance] + times

image_paths = ["./data/chessboard_large.jpg"]
binary_paths = ["./data/binary.png"]
radii = [3]

pre_benchmark_logging()
for i in range(len(image_paths)):
	
	image_path = image_paths[i]
	binary_path = binary_paths[i]
	print("Image path: " + str(image_path))
	print("Binary path: " + str(binary_path))
	# NOTE: OpenCV by RGB into Gray using a weighted average. BoofCV uses just the average
	img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

	# Binary image. 0 and 255. This matches what OpenCV expects
	img_binary = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

	for radius in radii:
		region_radius = radius
		print("Radius: " + str(radius))
		region_width = region_radius*2+1
		#run all the tests again with the new parameters
		csv_contents.append([resize.__name__, image_path, binary_path, region_width] + benchmark(resize))
		csv_contents.append([rotate.__name__, image_path, binary_path, region_width] + benchmark(rotate))
		csv_contents.append([mirror.__name__, image_path, binary_path, region_width] + benchmark(mirror))
		csv_contents.append([contour.__name__, image_path, binary_path, region_width] + benchmark(contour))
		img_binary = None
		csv_contents.append([gaussianBlur.__name__, image_path, binary_path, region_width] + benchmark(gaussianBlur))
		csv_contents.append([meanThresh.__name__, image_path, binary_path, region_width] + benchmark(meanThresh))
		csv_contents.append([gradientSobel.__name__, image_path, binary_path, region_width] + benchmark(gradientSobel))
		csv_contents.append([computeHistogram.__name__, image_path, binary_path, region_width] + benchmark(computeHistogram, 1))
		csv_contents.append([computeCanny.__name__, image_path, binary_path, region_width] + benchmark(computeCanny))
		if hasattr(cv2, 'xfeatures2d'):
			csv_contents.append([detectSift.__name__, image_path, binary_path, region_width] + benchmark(detectSift, 10))
			sift = None
			csv_contents.append([detectSurf.__name__, image_path, binary_path, region_width] + benchmark(detectSurf, 10))
			surf = None
		else:
			print("Skipping SIFT and SURF. Not installed")
		csv_contents.append([goodFeatures.__name__, image_path, binary_path, region_width] + benchmark(goodFeatures))
		csv_contents.append([houghLine.__name__, image_path, binary_path, region_width] + benchmark(houghLine))
		
post_benchmark_logging()

csv_writer.writerows(csv_contents)
print("Done!")