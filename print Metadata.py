# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 17:11:28 2018
write out a table of all datasets with metainformation.
@author: monika
"""
import dataHandler as dh
import numpy as np
################################################
#
# grab all the data we will need
#
################################################
strains = {'AML32':'pan-neuronal GCaMP6s and tagRFP',
              'AML18':'pan-neuronal GFP and tagRFP',
              'AML175':'pan-neuronal GFP and tagRFP in lite-1 background',
              'AML70':'pan-neuronal GCaMP6s and tagRFP in lite-1 background',
              'Special': 'pan-neuronal GCaMP6s and tagRFP'
    }
    
conditions = {'moving': 'freely moving on imaging plate',
              'chip': 'freely moving in microfluidic device',
              'transition': 'transiently paralyzed with tetramisole',
              'immobilized': 'physically constrained with microbeads',
              }
              


data = {}
for typ in ['AML32', 'AML18', 'AML175', 'AML70', 'Special']:
    for condition in ['moving', 'chip', 'immobilized', 'transition']:
        folder = '{}_{}/'.format(typ, condition)
        dataLog = '{0}_{1}/{0}_{1}_datasets.txt'.format(typ, condition)
        outLoc = "Analysis/{}_{}_results.hdf5".format(typ, condition)
        outLocData = "Analysis/{}_{}.hdf5".format(typ, condition)
        #print strains[typ], conditions[condition]
        try:
            # load multiple datasets
            dataSets = dh.loadDictFromHDF(outLocData)
            keyList = np.sort(dataSets.keys())
            results = dh.loadDictFromHDF(outLoc) 
            # store in dictionary by typ and condition
            key = '{}_{}'.format(typ, condition)
            
            for key in keyList:
                print typ, key, conditions[condition], dataSets[key]['Neurons']['Activity'].shape[0], int(dataSets[key]['Neurons']['Activity'].shape[1]/6.), 's'
            
        except IOError:
            #print typ, condition , 'not found.'
            pass
print 'Done reading data.'

