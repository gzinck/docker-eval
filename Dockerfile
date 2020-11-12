FROM python:3.7

RUN apt-get update \
    && apt-get install -y \
        build-essential=12.6 \
        cmake=3.13.4-1 \
        git=1:2.20.1-2+deb10u3 \
        wget=1.20.1-1.1 \
        unzip=6.0-23+deb10u1 \
        yasm=1.3.0-2+b1 \
        pkg-config=0.29-6 \
        libswscale-dev=7:4.1.6-1~deb10u1 \
        libtbb2=2018~U6-4 \
        libtbb-dev=2018~U6-4 \
        libjpeg-dev=1:1.5.2-2 \
        libpng-dev=1.6.36-6 \
        libtiff-dev=4.1.0+git191117-2~deb10u1 \
        libavformat-dev=7:4.1.6-1~deb10u1 \
        libpq-dev=11.9-0+deb10u1 \
		vim=2:8.1.0875-5 \
	&& pip3 install psutil==5.7.3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy==1.19.3

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

# Make sure to attach a volume for this
RUN mkdir /benchmarks
WORKDIR /benchmarks
COPY ./benchmarks/requirements.txt /benchmarks/requirements.txt

# See https://github.com/lessthanoptimal/SpeedTestCV/tree/master/opencv
RUN python3 -m venv venv && \
		. venv/bin/activate && \
		pip3 install -r requirements.txt
