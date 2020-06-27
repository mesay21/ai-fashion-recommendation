# Dockerfile for building streamline app

# pull miniconda image
FROM continuumio/miniconda3

# copy local files into container
#COPY streamlit_app.py /tmp/
#COPY requirements.txt /tmp/
#COPY pair /tmp/pair
#COPY output/indexes /tmp/output/indexes
#COPY data /tmp/data
# .streamlit for something to do with making enableCORS=False
#COPY .streamlit /tmp/.streamlit 

# install python 3.6 (needed to work with tensor flow) and faiss
RUN conda install python=3.6
#RUN conda install faiss-cpu=1.5.1 -c pytorch -y
#RUN conda install keras
#ENV PORT 8080

# change directory
#WORKDIR /tmp
WORKDIR /app

COPY requirements.txt .

# install dependencies
RUN apt-get update && apt-get install -y vim
RUN pip install -r requirements.txt
RUN pip install opencv-python
#RUN pip install pytorch torchvision
RUN conda install pytorch torchvision

#FROM nvidia/cuda:10.0-base-ubuntu16.04
#RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#RUN conda install pytorch torchvision cuda100 -c pytorch
#RUN apt-get install nvidia-container-runtime
#FROM nvidia/cuda:10.0-base-ubuntu16.04
#RUN apt-get install nvidia-container-runtime



#FROM amd64/ubuntu:latest

#RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
 #   && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
#ENV LANG en_US.utf8

#RUN apt-get update && apt-get install -y gnupg2

#RUN apt-get update
#RUN apt-get install ca-certificates

#RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

#RUN apt-get update && apt-get install -y nvidia-container-toolkit
#RUN systemctl restart docker

#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#zypper ar https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo

#RUN zypper install -y nvidia-docker2  # accept the overwrite of /etc/docker/daemon.json
#RUN systemctl restart docker#
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

#RUN apt-get update && apt-get install -y --no-install-recommends \
#        apt-utils \
        #python3.6 \
        #python-dev \
	#add-apt-repository ppa:fkrull/deadsnakes \
 #	add-apt-repository ppa:jonathonf/python-3.6
#	python3.6
#	python-pip \
 #       python-setuptools \
 #       && \
 #   rm -rf /var/lib/apt/lists/* && \
 #   apt-get update

#RUN pip install --upgrade pip==9.0.3 && \
#    pip --no-cache-dir install --upgrade torch==1.1.0 && \
#    pip --no-cache-dir install --upgrade torchvision==0.3.0

COPY . .

# run commands
#CMD ["streamlit", "run", "bolor_streamlit_app.py"]
WORKDIR "/root"

CMD ["/bin/bash"]

EXPOSE 8501
