FROM python:3.7

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy

RUN mkdir ~/opencv
ENV OPENCV_VERSION="4.1.1"
RUN cd ~/opencv && git clone https://github.com/opencv/opencv_contrib.git && cd opencv_contrib && git checkout ${OPENCV_VERSION}
RUN cd ~/opencv && git clone https://github.com/opencv/opencv.git && cd opencv && git checkout ${OPENCV_VERSION}

RUN cd ~/opencv/opencv && mkdir release && cd release && \
          cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_ENABLE_NONFREE=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D INSTALL_C_EXAMPLES=ON \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D BUILD_EXAMPLES=ON \
          -D WITH_OPENGL=ON \
          -D WITH_V4L=ON \
          -D WITH_XINE=ON \
          -D WITH_TBB=ON ..

RUN cd ~/opencv/opencv/release && make -j $(nproc) && make install
RUN pip3 install vse
# RUN apt-get update && apt-get install -y vim # for development

# Make sure to attach a volume for this
RUN mkdir /benchmarks
WORKDIR /benchmarks
COPY requirements.txt /benchmarks/requirements.txt

# See https://github.com/lessthanoptimal/SpeedTestCV/tree/master/opencv
RUN python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt
