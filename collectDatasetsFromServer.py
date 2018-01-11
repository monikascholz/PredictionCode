# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:33:57 2017
goes through the list of datasets and gets them from the server.
@author: monika
"""

import os
import subprocess
import numpy as np

# read in the dataset table and create date-time pairs
#dataLog = "AML18_moving/AML18_datasets.txt"
#outfolder = "AML18_moving"
dataLog = "AML32_immobilized/AML32_immobilized_datasets.txt"
outfolder = "AML32_immobilized/"
for lindex, line in enumerate(np.loadtxt(dataLog, dtype=str, ndmin = 2)):
    print "Creating dataset ", line[0]
    date, time = line[0].split('_')[0][-8:], line[0].split('_')[1]
    subprocess.call(['./assembleData.sh', date, time, outfolder])
