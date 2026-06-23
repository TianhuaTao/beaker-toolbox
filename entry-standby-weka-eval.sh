
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        unzip \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gnupg \
        jq \
        gcc \
        net-tools \
        openssh-server \
        git-lfs \
        bc \
        sudo \
        htop \
        tmux \
        vmtouch \
        bwm-ng \
        git


source beaker-toolbox/entry-standby-weka.sh

