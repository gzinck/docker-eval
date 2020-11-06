#Linux
# sudo apt-get install gcc python3-dev
# sudo pip3 install psutil

#Windows
# pip3 install psutil

#!/usr/bin/env python
import os, sys
import threading
import psutil
import sched, time
from datetime import datetime

# Setup

threads = []

user_path = ""
if (user_path == ""):
	status_path = sys.path[0]
else:
	status_path = user_path

status_filename = os.path.join(status_path, "../output/PC_status_log.txt")

if os.path.exists(status_filename):
    append_write = 'a' # append if already exists
else:
    append_write = 'w' # make a new file if not

# Thread class and functions

def log_status(): 
	#if not end_flag:
	#	status_scheduler.enter(60, 1, log_status, (scheduler,))
	load = psutil.getloadavg()
	status_log_file = open(status_filename, append_write)
	status_log_file.write(
		str(psutil.virtual_memory().percent) + "," + # virtual memory in use
		str(psutil.swap_memory().percent) + "," + # swap memory in use
		str(load[0] / psutil.cpu_count() * 100) + "\n" # average CPU load over one minute 
	)
	
	status_log_file.close()
	
def log_status_start():
	status_log_file = open(status_filename, append_write)
	# Log general info
	status_log_file.write(get_environment_string())
		
	# Signify start of logged CSV data
	start_time = datetime.now()
	status_log_file.write("PC_stats_log_start: " + str(start_time) + "\n")
	status_log_file.close()
	load = psutil.getloadavg() # do once initially to start the first minute measurement
	
def log_status_end():
	status_log_file = open(status_filename, append_write)
	
	# Just for sanity check, log environment again
	status_log_file.write(get_environment_string())
	end_time = datetime.now()
	status_log_file.write("PC_stats_log_end: " + str(end_time) + "\n")
	status_log_file.close()

def get_environment_string():
	return_string = ("Environment details:" +
		"\nNumber of logical cores: " + str(psutil.cpu_count()) + 
		"\nNumber of physical cores: " + str(psutil.cpu_count(logical=False)) + 
		"\nCore frequency: " + str(psutil.cpu_freq()) + # This is likely to change due to core throtteling?
		"\nTotal virtual memory: " + str(psutil.virtual_memory().total) +
		"\nTotal swap memory: " + str(psutil.swap_memory().total) +
		"\n")
	return return_string

def pre_benchmark_logging():
	logging_thread = status_log_thread(1, "Logging_Thread_1")
	logging_thread.start()
	threads.append(logging_thread)
	

def post_benchmark_logging():
	log_status_end() # Not inside the thread since it can add up to 5 second delay until it finishes.
	for th in threads:
		if (th.name == "Logging_Thread_1"):
			th.end_flag = True
			break

def initialize_logging():
	psutil.cpu_percent(interval=None) # First call returns bogus 0, next calls reset interval
	
def return_logging():
	return psutil.cpu_percent(interval=None) # Percentage of CPU usage (over all CPUs)
	# If this times the number of cores is only slightly above 100%, benchmark uses a single CPU core
	
def memory_logging():
	return psutil.virtual_memory().percent # Percentage of memory in use
	# Does not include swap memory in this version 
	# We might want to use .available instead (and turn it into a percentage manually) 

# Thread logger class

class status_log_thread(threading.Thread):
	def __init__(self, thread_ID, name):
		threading.Thread.__init__(self)
		self.thread_ID = thread_ID
		self.name = name
		self.end_flag = False
		self.start_time = time.time()
		self.minute_counter = 0
	def run(self):
		print("Starting thread: " + self.name)
		log_status_start()
		
		# Sleep for 5 seconds and log every 60 seconds (with self-correcting timings)
		while not self.end_flag:
			if (self.minute_counter >= 11):
				log_status()
				self.minute_counter = 0
			else:
				self.minute_counter += 1
			time.sleep(5.0 - ((time.time() - self.start_time) % 5.0))
			
		print("Ending thread: " + self.name)



# Start logging

#pre_benchmark_logging()

		
# Do all the things!

#time.sleep(180) # Benchmark!
	

# Stop logging

#post_benchmark_logging()
