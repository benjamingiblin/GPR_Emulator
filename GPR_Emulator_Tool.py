# 23/04/2018, B. M. Giblin, PhD Student, Edinburgh
# General code to do Gaussian process regression (GPR) from a set of predictions to a test set.

import pylab as plt
import numpy as np
import sys
import os
import matplotlib
from matplotlib import rc

rc('text',usetex=True)
rc('font',size=18)
rc('legend',**{'fontsize':14})
rc('font',**{'family':'serif','serif':['Computer Modern']})

# For doing Gaussian processes, PCAs and plotting output
from GPR_Classes import Get_Input, PCA_Class, GPR_Emu, Diagnostic_Plots

# ----- Load Input ----- #
paramfile = sys.argv[1]
GI = Get_Input(paramfile)
NumNodes = GI.NumNodes()
Train_x, Train_Pred, Train_ErrPred, Train_Nodes = GI.Load_Training_Set()	# Training set
if GI.Run_Trial():
	Trial_x, Trial_Pred, Trial_Nodes = GI.Load_Trial_Set()					# Trial set
else:
	Trial_Nodes = None


# Analysis choices
Run_Trial = GI.Run_Trial()													# Make predictions for a trial set? (T/F)
Perform_PCA = GI.Perform_PCA()												# Emulate for PCA weights? (T/F)
Include_x = GI.Include_x()													# Include x-coord of statistic as dimension in emulator (T/F)
MCMC_HPs = GI.MCMC_HPs()													# Perform an MCMC for the emulator hyperparameters (T/F)
Cross_Val = GI.Cross_Val()													# Perform "leave-one-out" cross-validation with training set (T/F)
Scale_Nodes = GI.Scale_Nodes()												# Scale training/trial nodes to [0,1] (T/F)
alpha = GI.alpha()															# Noise term, if not None then it supercedes errors in set in TrainErrorFile
HPs = GI.HPs()																# pre-set hyperparameters
n_restarts_optimizer = GI.n_restarts_optimizer()							# If None, emulator is not trained & goes with HPs. If an integer, trains this many times.


# Savedirectory for output - if not specified,
savedir = GI.savedirectory()
if not os.path.exists(savedir):
	os.makedirs(savedir)
savename = 'TrainN%s_PCA%s_ScaleNodes%s' %(NumNodes,Perform_PCA,Scale_Nodes)


###### CONVERSION OF THE INPUT STATISTIC TO EMULATED ONE: IN THIS CASE, TAKING THE LOG ######
#Train_ErrPred = ((1./Train_Pred)*Train_ErrPred)  
#Train_Pred = np.log(Train_Pred)
#Trial_Pred = np.log(Trial_Pred)


# PCA?
if Perform_PCA:
	n_components = GI.n_components()
	BFsFile = GI.BFsFile()
	PCAC = PCA_Class(n_components)
	if BFsFile == '': 
		Train_BFs, Train_Weights, Train_Recons = PCAC.PCA_BySKL(Train_Pred)
	#else # do manual PCA

	# Get the errors
	Train_Pred_Mean = np.zeros( len(Train_x) )
	for i in range(len(Train_x)):
		Train_Pred_Mean[i] = np.mean( Train_Pred[:,i] )
	Train_Weights_Upper, Train_Recons_Upper = PCAC.PCA_ByHand(Train_BFs, Train_Pred+Train_ErrPred, Train_Pred_Mean)
	Train_Weights_Lower, Train_Recons_Lower = PCAC.PCA_ByHand(Train_BFs, Train_Pred-Train_ErrPred, Train_Pred_Mean)
	Train_ErrWeights = abs(Train_Weights_Upper - Train_Weights_Lower)

	inTrain_Pred = np.copy(Train_Weights)
	inTrain_ErrPred = np.copy(Train_ErrWeights)

else:
	n_components = None
	Train_BFs = None
	Train_Pred_Mean = None
	inTrain_Pred = np.copy(Train_Pred)
	inTrain_ErrPred = np.copy(Train_ErrPred)



# GPR for Trail set
if Run_Trial:

	if GI.Train_Error():
		# If providing an error, SKL requires you run each x-bin separately:
		GP_AVOUT = np.zeros([ Trial_Nodes.shape[0], inTrain_Pred.shape[1] ])
		GP_STDOUT = np.zeros([ Trial_Nodes.shape[0], inTrain_Pred.shape[1] ])	
		GP_HPs = np.zeros([ inTrain_Pred.shape[1], Trial_Nodes.shape[1]+1 ])
		for i in range(GP_AVOUT.shape[1]):
			print( "Now on x bin %s of %s" %(i,GP_AVOUT.shape[1]) )
			GPR_Class = GPR_Emu(Train_Nodes, inTrain_Pred[:,i], inTrain_ErrPred[:,i], Trial_Nodes)
			if len(HPs.shape) == 1: 
				# We do not have individual HPs per bin
				GP_AVOUT[:,i], GP_STDOUT[:,i], GP_HPs[i,:] = GPR_Class.GPRsk(HPs, inTrain_ErrPred[:,i], n_restarts_optimizer)
			else:
				# We DO have individual HPs per bin!
				GP_AVOUT[:,i], GP_STDOUT[:,i], GP_HPs[i,:] = GPR_Class.GPRsk(HPs[i], inTrain_ErrPred[:,i], n_restarts_optimizer)

	else:
		GPR_Class = GPR_Emu(Train_Nodes, inTrain_Pred, inTrain_ErrPred, Trial_Nodes)
		GP_AVOUT, GP_STDOUT, GP_HPs = GPR_Class.GPRsk(HPs, alpha, n_restarts_optimizer) 				# with Scikit-Learn	
		GP_STDOUT = np.repeat(np.reshape(GP_STDOUT, (-1,1)), GP_AVOUT.shape[1], axis=1)			# SKL only returns 1 error bar per trial here
																								# stack them to same dimension as predicitons.
	
	if Perform_PCA:
		GP_Pred = PCAC.Convert_PCAWeights_2_Predictions(GP_AVOUT, Train_BFs, Train_Pred_Mean)
		Upper = PCAC.Convert_PCAWeights_2_Predictions(GP_AVOUT+GP_STDOUT, Train_BFs, Train_Pred_Mean)
		Lower = PCAC.Convert_PCAWeights_2_Predictions(GP_AVOUT-GP_STDOUT, Train_BFs, Train_Pred_Mean)
		GP_ErrPred = abs(Upper - Lower) / 2.

	else:
		GP_Pred = np.copy(GP_AVOUT)
		GP_ErrPred = np.copy(GP_STDOUT)

	# Save pickled GP predictions & accuracies plus a datafile containing corresponding x-array	
	np.save(savedir + 'GPPred_' + savename, GP_Pred)
	np.save(savedir + 'GPAcc_'  + savename, GP_Pred/Trial_Pred)
	np.savetxt(savedir + 'xArray_' + savename + '.dat', np.c_[Train_x])
	np.savetxt(savedir + 'GPHPs_' + savename + '.dat', GP_HPs)

	GPsavename = savedir + 'GPPred_' + savename + '.png'
	# Plot the result
	Diagnostic_Plots(Train_x, GP_Pred, GP_Pred*GP_ErrPred, Trial_Pred, np.zeros_like(Trial_Pred)).Plot_GPvsTrial(GPsavename)


if Cross_Val:
	# Do cross-validation
	GPR_Class = GPR_Emu(Train_Nodes, inTrain_Pred, inTrain_ErrPred, Trial_Nodes)
	CV_Pred, CV_HPs = GPR_Class.Cross_Validation(HPs, Perform_PCA, n_components, Train_BFs, Train_Pred_Mean, Include_x, Train_x, GI.Train_Error(), alpha, n_restarts_optimizer)

	# Save pickled GP predictions & accuracies plus a datafile containing corresponding x-array	
	np.save(savedir + 'CVPred_' + savename, CV_Pred)
	np.save(savedir + 'CVAcc_'  + savename, CV_Pred/Train_Pred)
	np.savetxt(savedir + 'xArray_' + savename + '.dat', np.c_[Train_x])
	np.savetxt(savedir + 'CVHPs_' + savename + '.dat', np.reshape(CV_HPs, (CV_HPs.shape[0]*CV_HPs.shape[1], CV_HPs.shape[2])) )

	# Stuff for plotting
	# First show the accuracy of the cross-validation
	CVplotname = savedir + 'CVAcc_' + savename + '.png'
	DP = Diagnostic_Plots(Train_x, CV_Pred, np.zeros_like(CV_Pred), Train_Pred, np.zeros_like(Train_Pred))
	DP.Plot_CV(Include_x, CVplotname)

	# Secondly, visualise which of the training nodes had mean accuracies worse than a certain threshold
	threshold = 5.0 		# Highlight nodes less accurate than this % on average
	labels = None			# array of labels for nodes used in emulation. e.g. ['Omega_m', 'S_8',...]
	limits = None			# Limits of training nodes used in plotting. e.g. [ [0.1,0.9], [0.7,0.9],...] 
	CVplotname2 = savedir + 'CVNodes_' + savename + '.png'

	DP.Plot_CV_Inacc(Train_Nodes,threshold,labels,limits,CVplotname2)






















