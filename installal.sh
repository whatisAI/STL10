#!/bin/bash

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install git
sudo apt-get install python3
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.5 1
sudo apt-get install python3-pip
sudo pip3 install pip
sudo pip3 install --upgrade pip 
sudo pip3 install numpy --upgrade 
sudo pip3 install matplotlib  --upgrade  
sudo pip3 install sklearn  --upgrade 
sudo pip3 install pandas  --upgrade   
sudo pip3 install tensorflow  --upgrade   
#pip3 install --upgrade tensorflow-gpu
sudo pip3 install keras --upgrade 
#pip3 install networkx==1.11

sudo apt-get install libhdf5
sudo pip3 install h5py

