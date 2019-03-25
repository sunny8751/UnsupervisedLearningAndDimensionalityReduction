# Unsupervised Learning and Dimensionality Reduction

Both Weka GUI and Java was used to run the clustering algorithms, dimensionality reduction, and neural network analysis. The python script "icaAnalysis.py" was used to generate the kurtosis of the ICA features.

The two databases used were: [White Wine Quality](http://archive.ics.uci.edu/ml/datasets/Wine) and [German Credit](https://archive.ics.uci.edu/ml/datasets/statlog+%28german+credit+data%29).

## Setup Instructions
1. Download the White Wine Quality and German Credit datasets
2. Use Weka's ArffViewer to convert csv files to arff format and name "data.arff".

## Run instructions
The "src" folder contains the Java code to run the clustering algorithms and NN. In the "Main.java" file, the functions for these are commented out.

1. KMeans and EM
In "Main.java", specify the data file to use (raw data or data after dimensionality reduction) and the range of k values to test. Plug the output into an Excel sheet to graph to give the error over k.
To get the cluster visualizations, use the WEKA GUI.

2. Dimensionality Reduction
Using Weka's GUI, use the Preprocess tab to create the following new transformed datasets, specifying the number of features to reduce to:
PCA: PrincipalComponents
ICA: IndependentComponents
RP: RandomProjection
Using Weka's GUI, use the Feature Selection tab to get information to create the following new transformed datasets:
PCA: PrincipalComponents. The ranked eigenvalues can be found here.
IG: InfoGainAttributeEval

3. Neural Network on dimensionality reduced dataset OR Neural Network on clusters as features
In "Main.java", specify the neural network's hyperparameters and the set of datasets to test over. These datasets should be created from part 2. This will output the learning curve for the wine dataset for the specified datasets. The splits for the learning curve were created via StratifiedRemoveFolds. 