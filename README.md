# Multi-Human Swarm Control

![screenrecord](https://gitlab.com/genki_miyauchi/multi-human-swarm-control/-/wikis/uploads/52f4b6fd30ddee915c32070af8f8149d/screenrecord.gif)

[**Citation**](#citation) | [**Docker image**](#docker-image) | [**Installation**](#installation) | [**Usage**](#usage) | [**License**](#license)

This repository contains the code for the paper:
- [Sharing the Control of Robot Swarms Among Multiple Human Operators: A User Study](https://eprints.whiterose.ac.uk/202313/1/IROS23_1687_FI.pdf)

Watch the presentation video of the paper.

<p align="center">

[![MHSC IROS2023 Video](https://img.youtube.com/vi/DRYU4v8kkuo/0.jpg)](https://youtu.be/DRYU4v8kkuo)
</p>

# Citation

If you use this repository in your research, **cite** it using:

```
@inproceedings{miyauchi2023Sharing,
  author = {Miyauchi, Genki and Lopes, Yuri K. and Groß, Roderich},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title = {Sharing the Control of Robot Swarms Among Multiple Human Operators: A User Study}, 
  year = {2023},
  pages = {8847-8853}
```

# Docker image

If you have Docker installed on your computer, you can try it out using the Docker image found [here](https://hub.docker.com/r/genki15/multi-human-swarm-control).

```bash
docker pull genki15/multi-human-swarm-control
mkdir results
```

Then start a container with:

```bash
docker run --name argos -it --network="host" -v $(pwd)/results:/home/docker/multi-human-swarm-control/results -w /home/docker/multi-human-swarm-control genki15/multi-human-swarm-control python3 src/web_app/app.py -m $condition -o $order
```

- Use the ```-m``` flag to specify the **mode**. Replace ```$condition``` with either ```indirect``` (default) or ```direct```.
- Use the ```-o``` flag to specify the **order** of the trials. Replace ```$order``` with either ```1``` (default) or ```2```.

# Installation

The following steps have been tested in Ubuntu 22, but it should be applicable to other Ubuntu distributions as well.

You need to have [ARGoS](https://www.argos-sim.info/) installed on your computer before proceeding.

Install the following apt packages:

```bash
sudo apt update
sudo apt install python3 python3-venv pip git libyaml-cpp-dev nlohmann-json3-dev
```

### Install plugins

Install the [plugins](https://gitlab.com/genki_miyauchi/multi-human-swarm-control-plugins) used in this project:

```bash
git clone https://gitlab.com/genki_miyauchi/multi-human-swarm-control-plugins.git
cd multi-human-swarm-control-plugins
mkdir build 
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ../src
make
sudo make install
```

After executing the above commands, you should see ```e-puck_leader``` and ```rectangle_task``` appear when using ```argos3 -q entities```

Install the [ARGoS Webviz](https://gitlab.com/genki_miyauchi/argos3-webviz) plugin. This is a modified version of the [origianl](https://github.com/NESTLab/argos3-webviz) Webviz plugin modified for this project.

```bash
git clone https://gitlab.com/genki_miyauchi/argos3-webviz.git
cd argos3-webviz
mkdir build 
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ../src
make
sudo make install
```
After executing the above commands, you should see ```webviz``` appear when using ```argos3 -q visualizations```

### Install protobuf

Protobuf is used to log the experiment in binary format. Install protobuf using the following commands.

```bash
wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protobuf-cpp-3.21.12.tar.gz
tar xzf protobuf-cpp-3.21.12.tar.gz
cd protobuf-3.21.12/
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
```
Once the protobuf compiler is successfully installed, you should be able to run ```protoc``` in the terminal.

### Python dependencies

Install the Python dependencies in requirements.txt using your choice of virtual environment. Here, we assume using venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage

### Build the project

```bash
cd multi-human-swarm-control
mkdir results
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ../src
make
```

### Run the project

Use the following command to run the experiment:

```bash
python3 src/web_app/app.py
```

This hosts the main page at to access the rest of the experiments.
Open a web browser and access ```localhost:5000```.
The app has been tested on Chromium. There are known issues in Firefox.

# License
The code in this repository is released under the terms of the MIT license.
