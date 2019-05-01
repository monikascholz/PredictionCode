# PredictionCode

This reporsitory contains code and visualizations needed to reproduce the results presented in https://doi.org/10.1101/445643.
The associated data can be found here:https://osf.io/mw5zs/

## requirements
python 2.7
scikit_image==0.13.0  
scipy==0.19.1  
numpy==1.11.0  
h5py==2.7.0  
matplotlib==1.5.1  
scikit_learn==0.19.1  

## Getting started
To reproduce the figures and panels from 'Predicting natural behavior from whole-brain neural dynamics', download the data from the OSF repository. Each dataset is indentified by a strain name and condition eg. AML32_moving and contains two hdf5 files:
(1) data [strain]_[condition].hdf5  - contains the raw calcium data, behavior data and other meta data 
(2) results [strain]_[condition]_results.hdf5 - contains models, model prediction, PCA, ... 

hdf5 is a container format that allows storing variable datatypes in an organized structure with its associated metadata. A good way to see the nested data structure is by using an hdf5 viewer. Internally, our code converts the data to a nested dictionary. The basic structure of the input data is this:
### Input/raw data

* [strain]_[condition].hdf5  
  * __BrainScanner_Date_Time1__  
    * Neurons  
      * Activity   
      * Time  
      * RawFluorescence  
      * ...  
    * Behavior  
      * AngleVelocity  
      * Eigenworms  
      * ...  
    * Centerlines  
    * goodVolumes  
  * __BrainScanner_Date_Time2__ 
  * __BrainScanner_...__ 
       
### Output data/results
The results data is similarly structured by dataset. Each dataset contains analysis output for the 
* [strain]_[condition].hdf5  
  * __BrainScanner_Date_Time1__
    * PCA
    * LASSO
    * ...
### figure code
The code to make figures is located in the subdirectory 'figures'. You need to modify the path that points to the data according to where you downloaded it to. After that, you should see the figures.


## Analyzing new data or running models from scratch
### key pieces
* dataHandler: Reads in wholebrain data.
* dimReduction: contains actual analysis functions
* dataQualityCheck: run diverse set of analysis methods, such as PCA, linear Regression or SVM. Here, output is plotted, and the program is more verbose. This is mostly for checking basic features of the data before running the full analysis.
* runAnalysis: basically the same as dataQualityCheck, but runs quietly and saves results in a hdf5 file for input data and oputput data as described above.
### prepare your data
You need a sucessfully analyzed BrainScanner dataset from the wholebrain analysis pipeline. dataFolder is the Brainscanner folder containing your dataset.
1. in MATLAB, on tigress: run rerunDataCollectionMS(dataFolder)
This creates a new version of the heatmap with some extra information, including a regularized derivative and the time synchornization between behavior cameras and calcium data.
2. (Optional) If you are running the code in a different location and want to copy the datasets, copy heatdataMS.mat, centerline.mat and
pointStatsNew.mat in to a folder.
3. create a metadatafile. This should be called strain_condition.dat
For example, AML32_moving.dat, and contain the name of your dataFolder. Each line has one dataset.
4. Change the paths in the code (dataQualityCheck) to point to your data and to your desired output locations.

