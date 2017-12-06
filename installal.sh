#!/bin/bash

#sudo apt-get update
#sudo apt-get upgrade -y
#sudo apt-get install git
#git clone https://github.com/whatisAI/STL10
sudo apt-get install python3
update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
sudo apt-get install python3-pip
pip3 install --update pip
pip3 install --upgrade pip
pip3 install numpy
pip3 install matplotlib
pip3 install tensorflow
pip3 install keras




python download_data.py
