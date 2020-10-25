FROM opencvcontrib:1.0

RUN git clone https://github.com/lessthanoptimal/SpeedTestCV.git
WORKDIR /SpeedTestCV/opencv

# See https://github.com/lessthanoptimal/SpeedTestCV/tree/master/opencv
RUN python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt
