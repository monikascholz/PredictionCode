# PredictionCode

## requirements
scikit_image==0.13.0  
scipy==0.19.1  
numpy==1.11.0  
h5py==2.7.0  
matplotlib==1.5.1  
scikit_learn==0.19.1  

## key pieces

* dataHandler: Reads in wholebrain data.
* dimReduction: contains actual analysis functions
* dataQualityCheck: run diverse set of analysis methods, such as PCA, linear Regression or SVM. Here, output is plotted, and the program is more verbose.
* runAnalysis: basically the same as dataQualityCheck, but runs quietly and saves results in a hdf5 file.

## how to use it
dataFolder is the Brainscanner folder with your dataset
1. in MATLAB, on tigress: run rerunDataCollectionMS(dataFolder)
This creates a new version of the heatmap with some extra information, including a regularized derivative.
2. (Optional) If you are running the code in a different location and want to copy the datasets, copy heatdataMS.mat, centerline.mat and
pointStatsNew.mat in to a folder.
3. create a metadatafile. This should be called strain_condition.dat
For example, AML32_moving.dat, and contain the name of your dataFolder. Each line has one dataset.
4. Change the paths in the code (dataQualityCheck) to point to your data and to your desired output locations.

