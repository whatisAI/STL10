#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:09:06 2017

@author: clcaste
"""
import os
def downloaddata(urlfile,namedisk):
    import urllib.request
    print('Start downloading data')
    (filename, headers)=urllib.request.urlretrieve('http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz', 'stl10_binary.tar.gz')
    print('Finished downloading data')
    


def decompress(namecompressed):    
    import tarfile
    print('extracting data')
    tar = tarfile.open(namecompressed)
    tar.extractall()
    tar.close()
    print('Finished extracting data')
    
#downloaddata('http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz', 'stl10_binary.tar.gz')
#decompress('stl10_binary.tar.gz')    
os.remove('stl10_binary/unlabeled_X.bin')  #remove large unnecessary file

