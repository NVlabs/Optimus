# FROM nvcr.io/nvidia/pytorch:21.08-py3
FROM nvcr.io/nvidia/cudagl:11.3.0-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility
# env variables for tzdata install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin"
ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/usr/lib/nvidia"
ENV PYTHONPATH ${PYTHONPATH}:/workspace
ENV MUJOCO_GL 'egl'
ENV PATH "/usr/local/cuda-new/bin:$PATH"
ENV MESHSETS_LOCAL_ROOT_DIR /meshsets/
ENV ACRONYM_PATH /acronym/grasps
COPY pip.conf pip.conf
ENV PIP_CONFIG_FILE pip.conf
ENV PYTHONPATH ${PYTHONPATH}:/workspace/tamp_imitation
ENV PYTHONPATH ${PYTHONPATH}:/home/robosuite
ENV COPPELIASIM_ROOT /home/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/
ENV LD_LIBRARY_PATH $COPPELIASIM_ROOT:$LD_LIBRARY_PATH # fix for CoppeliaSim
ENV QT_QPA_PLATFORM_PLUGIN_PATH $COPPELIASIM_ROOT
ENV NVIDIA_DRIVER_CAPABILITIES compute,graphics,utility

# NOTE: each RUN command creates a new image layer, so minimize the number of run commands if possible
# installing other nice functionalities and system packages required by e.g. robosuite
RUN apt-get update -y && \
    apt-get install -y \
        htop screen tmux \
        sshfs libosmesa6-dev wget curl git \
        libeigen3-dev \
        liborocos-kdl-dev \
        libkdl-parser-dev \
        liburdfdom-dev \
        libnlopt-dev \
        libnlopt-cxx-dev \
        swig \
        python3 \
        python3-pip \
        python3-dev \
        vim \
        git-lfs \
        cmake \
        software-properties-common \
        libxcursor-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxi-dev \
        mesa-common-dev \
        zip \
        unzip \
        make \
        g++ \
        python2.7 \
        wget \
        vulkan-utils \
        mesa-vulkan-drivers \
        apt nano rsync \
        libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev \
        software-properties-common  net-tools  unzip  virtualenv \
        xpra xserver-xorg-dev libglfw3-dev patchelf python3-pip -y \
        && add-apt-repository -y ppa:openscad/releases && apt-get update && apt-get install -y openscad

# Copy over private key, and set permissions
# Warning! Anyone who gets their hands on this image will be able
# to retrieve this private key file from the corresponding image layer
ADD id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# install mujoco
RUN mkdir /root/.mujoco/ \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    && tar -xvf mujoco210-linux-x86_64.tar.gz \
    && mv mujoco210 /root/.mujoco/ \
    && rm mujoco210-linux-x86_64.tar.gz

# robomimic dependencies and installing other packages
RUN python3 -m pip install h5py psutil tqdm termcolor tensorboard tensorboardX imageio imageio-ffmpeg egl_probe>=1.0.1 ipdb wandb \
    && python3 -m pip install ipython patchelf robosuite jupyterlab notebook black flake8 isort pytest protobuf==3.20.1 pytorch_lightning \
    && python3 -m pip install seaborn mujoco pygame signatory==1.2.6.1.9.0 pyopengl==3.1.6 vit-pytorch stable_baselines3 \
    && pip uninstall torch torchvision torchaudio -y \
    && python3 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run --no-check-certificate
RUN sh cuda_11.3.0_465.19.01_linux.run --toolkit --silent --toolkitpath=/usr/local/cuda-new
ENV TORCH_CUDA_ARCH_LIST "7.0 7.5 8.0"
RUN git clone https://github.com/NVIDIA/apex /home/apex && cd /home/apex \
    && python3 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

#install PyRep
RUN wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
--no-check-certificate -O /home/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
&& cd /home/ && tar -xvf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
&& git clone https://github.com/stepjam/PyRep.git /home/PyRep \
&& cd /home/PyRep && python3 -m pip install -r requirements.txt && python3 -m pip install .

RUN git clone https://github.com/stepjam/RLBench.git /home/RLBench \
&& cd /home/RLBench/ && python3 -m pip install -r requirements.txt && python3 -m pip install . \
&& python3 -m pip install gym \
&& python3 -m pip uninstall opencv-python -y && python3 -m pip install opencv-python-headless


#dummy arg for forcing rebuild of the dockerfile with updated code
ARG dummy

RUN python3 -m pip install trimesh==3.12.6 pyopengl==3.1.6  # breaks scene_synthesizer otherwise

RUN python3 -m pip install numpy==1.23.5 #numpy 1.24.0 breaks with np.float calls
