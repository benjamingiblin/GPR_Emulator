# README for GPR emulator tool
# 26/02/2019, B. M. Giblin, PhD Student, Edinburgh

----- DEPENDABLES (must be installed) -----
sklean
matplotlib
numpy

----- BASIC SYNTAX-----
python GPR_Emulator_Tool.py <parameter filename>


----- INSIDE THE PARAMETER FILE -----
When changing the parameter file, always put a space between '=' and the argument. Don't put whitespace in the argument itself.

1. Specify a training set
   - NumNodes: Number of training nodes.
   - TrainFile: Give the address and general filename of training set predictions, e.g. PredictionXXXX.dat for predictions in separate files, formatted [x_array, Prediction] as columns. Alternatively, you can give all predictions as one pickled file with extension '.npy', such that Data[1,j,k] gives j'th prediction at k'th x-array element. Corresponding x-arrays are at element Data[0,j,k].
   - TrainIDs: an array that replaces the 'XXXX' in TrainFile when reading in. Not used if TrainFile has '.npy' extension.
   - TrainNodesFile: address of the nodes corresponding to the training predicitons.
   - Cols: which columns in TrainNodesFile to use. Default is all.
   - Train_Error: If True, reads in error on training set specified by TrainErrFile. Default is False, no error on training set.
   - Cov_Error: If True, TrainErrFile is interpretted as a covariance matrix and SQRT of diagonal is taken as error. THE OFF-DIAGONAL ELEMENTS ARE NOT USED.
   - alpha: If set to a number, it overrules TrainErrFile, and is interpretted as a constant noise across all training predictions. Safest to leave this alone.
   - Scale Nodes: True or False; Scale the training set nodes to [0,1]? Worth doing if nodes have large dynamic range. Otherwise doesn't make a difference.

2. Specify a trial Set
   - Run_Trial: If True, it will make predictions for the coordinates set in TrialNodesFile
   - TrialFile: Same rules as TrainFile but for the Trial predictions. If set, trial predictions are read in a compared to emulated predictions to assess accuracy. If not set, then no trial predictions are read in and Trial_Pred in code is set to an array of ones. In this case, pay no attention to the 'GPAcc...' plot and .npy file that are saved; these will contain the emulated predictions themselves.
   - TrialIDs: same as TrainIDs, they replace the 'XXXX' part of TrialFile when reading in. This is not used if testing for just one trial prediction.
   - TrialNodesFile: the file containing coordinates at which to test the emulator. Same format as TrainNodesFile.
   - savedirectory: sets where the output of the emulation is saved (including the Cross Validation if Cross_Val set to True). Default value of this is the the directory specified in TrialFile. If TrialFile is empty, savedirectory needs to be set.  

3. PCA Variables
   - Perform PCA: If True, it will decompose the training set into n_components number of basis functions, and train on predicting the weights of these basis functions instead of the statistic itself. This is recommended if your statistic is defined in tens/hundreds of bins as it will save time.
   - n_components: Number of principal components used to reconstruct statistic. Cannot exceed number of bins in which statistic is defined.
   - BFsFile: Ignore this for now. If specified it is supposed to read in basis functions from this file instead of calculating them from the Training set. But currently it does nothing.
   - BFsMean: Same as above - the mean that would be used in the PCA instead of that of the training set.

4. Extra Analysis Choices
   Include_x: Ignore this also. Option to fold the x_array into the emulation, increasing the dimensionality of the emulation by 1 and the size of the training set by len(x-array). Was useful when emulating with george. With scikit-learn, it's redundant.
   MCMC_HPs: Ignore. Was used to train the emulator with george by doing an MCMC for the hyperparameters. Now redundant.
   Cross_Val: True or False. Useful for testing the emulator accuracy. It leaves one node of the training set out, trains on the rest and makes a prediction for the missing node. Then cycles through all nodes. Saves output predictions showing accuracy with prefix 'CV'.

5. GPR Variables
   - n_restarts_optimizer: If set to integer, the emulator restarts its optimisation for the hyperparameters this many times (i.e. it does this many attempts at finding the best values). Default value is 20, a good safe bet if you've never ran the emulator before. Set to None if you want to run it without training, using the hyperparemeters set in HP.
   - HPs: When you train the emulator, it will output the optimal values of the HPs that it found to the terminal. You can save these in this array and set n_restarts_optimizer to None, such that next time it doesn't need any training. NB: if running an emulation with an error on the training set, it will need to re-train in each bin of the emulated statistic, so HPs cannot be pre-set and n_restarts_optimizer should not be set to None.
   - NDim: if HPs is not specified, the emulator figures out the dimensionality of the emulation with this integer. Does nothing if HPs is set.


----- INSIDE GPR_Emulator_Tool.py -----
Most things are self explanatory.

THINGS TO NOTE:
       - Line 47: creates a string used in all of the output saved data of the emulator. This can be changed to whatever you like.
       - Lines 51-53: These take the logarithm of the Training & Trial predictions & errors. NOT NECESSARY IN GENERAL. This is just a smart thing to do when emulating many statistics since log will have narrower dynamic range, but this will cause issues if your statistic is negative (obviously). The taking of the log is the reason that 'np.exp(...)' in all the places in which plots are made and data saved, since we are interested in this quantity rather than the emulated (logarithmic) statistic. If you are not taking the log of the input statistic, remove these lines and all cases where 'np.exp()' appears.


