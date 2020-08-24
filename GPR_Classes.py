import numpy as np
import pylab as plt
import os
from matplotlib import rc
from matplotlib import rcParams

# Some lines just to make nice plot fonts
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)
plt.rcParams["mathtext.fontset"] = "cm"


# For doing Gaussian processes with SK-learn
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

# For doing Gaussian processes with george
#import george
#from george import kernels
#from george.kernels import ExpSquaredKernel

# For doing PCA
from sklearn.decomposition import PCA

# Class for reading in training set (predictions, nodes, errors) and trail set (prediction, nodes)
class Get_Input:

	
	# --------------------------------------------------------------------- READ IN PARAMS ------------------------------------------------------------------------------
	def __init__(self, paramfile):
		self.paramfile = paramfile
		self.paraminput = open(self.paramfile).read()

	# --------------------------------------------------------------------- Training set ------------------------------------------------------------------------------


	def NumNodes(self):
		return int(self.paraminput.split('NumNodes = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def TrainIDs(self):
		return eval(self.paraminput.split('TrainIDs = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def TrainFile(self):
		return self.paraminput.split('TrainFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def TrainPredCols(self): 
		try:
			return eval(self.paraminput.split('TrainPredCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
		except SyntaxError:
			return [0,1]

	def TrainNodesFile(self):
		return self.paraminput.split('TrainNodesFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def TrainNodeCols(self): 
		try:
			return eval(self.paraminput.split('TrainNodeCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
		except SyntaxError:
			return None

	def Train_Error(self):
		try:
			a = eval(self.paraminput.split('Train_Error = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
			if isinstance(a, bool):
				return a
			else:
				return False
		except SyntaxError:
			return False


	def Cov_Error(self):
		try:
			a = eval(self.paraminput.split('Cov_Error = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
			if isinstance(a, bool):
				return a
			else:
				return False
		except SyntaxError:
			return False


	def TrainErrFile(self):
		return self.paraminput.split('TrainErrFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def TrainPredErrCol(self): 
		try:
			return eval(self.paraminput.split('TrainPredErrCol = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
		except SyntaxError:
			return 1

	def alpha(self):
		try:
			return float(self.paraminput.split('alpha = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
		except (ValueError, TypeError) as error:
			return None

	def Scale_Nodes(self):
		return eval(self.paraminput.split('Scale_Nodes = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])


	# --------------------------------------------------------------------- Trial set ------------------------------------------------------------------------------
	def Run_Trial(self):
		try:
			a = eval(self.paraminput.split('Run_Trial = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
			if isinstance(a, bool):
				return a
			else:
				return False
		except SyntaxError:
			return False


	def TrialIDs(self):
		#print( self.paraminput.split('TrialIDs = ')[-1].split(' ')[0].split('\t')[0] )
		try:
			i = eval(self.paraminput.split('TrialIDs = ')[-1].split(' ')[0].split('\t')[0])	
		except SyntaxError:
			i = None
		return i

	def TrialFile(self):
		return self.paraminput.split('TrialFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def TrialPredCols(self): 
		try:
			return eval(self.paraminput.split('TrialPredCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
		except SyntaxError:
			return [0,1]

	def TrialNodesFile(self):
		return self.paraminput.split('TrialNodesFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
	
	def TrialNodeCols(self): 
		try:
			return eval(self.paraminput.split('TrialNodeCols = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
		except SyntaxError:
			return None

	def savedirectory(self):
		def Make_TF_the_savedir():
			print( "No save directory specified/not correctly specified (should start with / character).")
			print( "Setting savedirectory up inside trial directory.")
			TF = self.paraminput.split('TrialFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
			directory = os.path.dirname(os.path.abspath(TF))
			directory += '/GPPredictions/'
			return directory

		directory = self.paraminput.split('savedirectory = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		if directory == "":
			return Make_TF_the_savedir()
		elif directory[0] != '/':
			return Make_TF_the_savedir()	
		else:
			return directory


	# --------------------------------------------------------------------- PCA ------------------------------------------------------------------------------
	def Perform_PCA(self):
		try:
			a = eval(self.paraminput.split('Perform_PCA = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
			if isinstance(a, bool):
				return a
			else:
				return False
		except SyntaxError:
			return False


	def n_components(self):
		try:
			a = int(self.paraminput.split('n_components = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
		except SyntaxError:
			a = 9
		return a

	def BFsFile(self):
		return self.paraminput.split('BFsFile = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	def BFsDataMean(self):
		return self.paraminput.split('BFsDataMean = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]

	# --------------------------------------------------------------------- Extra choices ----------------------------------------------------------------------	
	def Include_x(self):
		try:
			a = eval(self.paraminput.split('Include_x = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])	
			if isinstance(a, bool):
				return a
			else:
				return False
		except SyntaxError:
			return False

	
	def MCMC_HPs(self):
		try:
			a = eval(self.paraminput.split('MCMC_HPs = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
			if isinstance(a, bool):
				return a
			else:
				return False
		except SyntaxError:
			return False


	def Cross_Val(self):
		try:
			a = eval(self.paraminput.split('Cross_Val = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
			if isinstance(a, bool):
				return a
			else:
				return False
		except SyntaxError:
			return False



	# --------------------------------------------------------------------- GPR ----------------------------------------------------------------------
	def NDim(self):
		return int(self.paraminput.split('NDim = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])

	def n_restarts_optimizer(self):
		try:
			a = eval(self.paraminput.split('n_restarts_optimizer = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
			if isinstance(a, int):
				return a
			elif a == None:
				return a
			else:
				return 20
		except SyntaxError:
			return 20
	

	def HPs(self):
		HP_File = self.paraminput.split('HP_File = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0]
		#print("HP_File is", HP_File )
		if HP_File != "" and HP_File != '#':
			# Then read in the values saved to file.
			h = np.loadtxt( HP_File )

		else:
			# Use the HP array if it's saved. If not, just create the default HP array of unity values.
			try: 
				h = eval(self.paraminput.split('HPs = ')[-1].split(' ')[0].split('\n')[0].split('\t')[0])
				h = np.array(h)
			except SyntaxError:
				print( "Could not interpret input HP array (if there is whitespace, remove this). Setting starting HPs to unity.")
				NDim = self.NDim() + 1 	# Add 1 for amplitude of kernel.
				h = np.ones(NDim)

		return h







	# --------------------------------------------------------------------- LOAD TRAINING SET ------------------------------------------------------------------------------
	def Fix_Single_Elem(self, a):
		# These lines avoid an error if a is only 1 element long and you try to take the length of it:
		try: 
			tmp = len(a) 
		except TypeError: 
			a = np.array([a]) 
		return a


	def Load_Training_Set(self):
		TF = self.TrainFile()		# Training Filename	(single string)
		IDs = self.TrainIDs()		# IDs of training sets (array)
		PC = self.TrainPredCols()	# Which columns of TrainFile to read (array)

		C = self.TrainNodeCols()	# Which columns of TrainNodesFile to read (array/None)
		NF = self.TrainNodesFile()	# File containing nodes for training, one per row.
		SN = self.Scale_Nodes()		# Scale training/trial nodes to [0,1] (True/False)

		Train_Nodes = np.loadtxt(NF, usecols=C, unpack=True).transpose()												# Coords of the nodes of the training set.
		if SN:		
			# Normalise Train_Nodes to be in range[0,1]
			for i in range(0,Train_Nodes.shape[1]):
				Train_Nodes[:,i] = abs(Train_Nodes[:,i]-Train_Nodes[:,i].min()) / abs(Train_Nodes[:,i].max()-Train_Nodes[:,i].min())

		if TF[-4:] == ".npy":		# The training file is a pickle. 
			#tmp = np.load('%s%s%s' %(TF.split('XXXX')[0],IDs[0],TF.split('XXXX')[1]))[0]	
			tmp = np.load(TF)[0] 			# Assumes x-array is the zero'th dimension.
			if len(tmp.shape) == 2:
				Train_x = tmp[0,:]
			elif len(tmp.shape) == 3:
				Train_x = tmp[0,0,:]
			else:
				print( "Training set contained in input file:\n %s\n Has dimensionality > 3. Too many to handle. Change this!" %TF)
				import sys				
				sys.exit()
			#Train_Pred = np.load('%s%s%s' %(TF.split('XXXX')[0],IDs[0],TF.split('XXXX')[1]))[1]
			Train_Pred = np.load(TF)[1]

		else:					# Read-in the multiple data files containing the training set predictions for each node.
			Train_x = np.loadtxt('%s%s%s' %(TF.split('XXXX')[0],IDs[0],TF.split('XXXX')[1]), usecols=(PC[0],), unpack=True) 	# x-coordinate of training set predictions
																															# currently assumes all training set predictions
																															# defined at same x-coords.
			Train_x = self.Fix_Single_Elem(Train_x)
			Train_Pred = np.empty([len(IDs), len(Train_x)])
			for i in range(len(IDs)):
				Train_Pred[i,:] = np.loadtxt('%s%s%s' %(TF.split('XXXX')[0],IDs[i],TF.split('XXXX')[1]), usecols=(PC[1],), unpack=True) 

		
		# If there's errors on Training set, load them. Else, set Error to 1e-6 * Train_Pred.
		TE = self.Train_Error()			# Error on training set? (True/False)
		CE = self.Cov_Error()			# Covariance matrix input? (True/False)
		TEF = self.TrainErrFile()		# Training error filename	(string)
		PEC = self.TrainPredErrCol()	# Which column of TrainErrFile to read (int)


		# Training set errors
		if TE:						# Read in an error on training set	
			Train_ErrPred = np.zeros_like(Train_Pred)		
			for i in range(len(IDs)):
				if 'XXXX' in TEF:	# Different error for each node. Read them all in.
					name = '%s%s%s' %(TEF.split('XXXX')[0],IDs[i],TEF.split('XXXX')[1])
					if CE: 			# Read in a covariance matrix; take sqrt of diag
						Train_ErrPred[i,:] = np.sqrt( np.diag( np.load(name) ) )
					else:
						Train_ErrPred[i,:] = np.loadtxt(name, usecols=(PEC,), unpack=True)
				else:
					if CE: 			# Read in a covariance matrix; take sqrt of diag
						Train_ErrPred[i,:] = np.sqrt( np.diag( np.load(TEF) ) )
					else:
						Train_ErrPred[i,:] = np.loadtxt(TEF, usecols=(PEC,), unpack=True)	
					
		else:						# Don't read in error on training set
			Train_ErrPred = 1.e-5 *  Train_Pred		


		return Train_x, Train_Pred, Train_ErrPred, Train_Nodes





	# --------------------------------------------------------------------- LOAD TRIAL SET COORDS ------------------------------------------------------------------------------
	def Interpolate_Trial_Onto_Train(self, Train_x, Trial_x, Trials):
		new_Trials = np.zeros([ Trials.shape[0], len(Train_x) ])
		for i in range(Trials.shape[0]):
			new_Trials[i,:] = np.interp( Train_x, Trial_x, Trials[i,:] )
		return	new_Trials




	def Load_Trial_Set(self):

		TF = self.TrialFile()		# Trial Filename	(single string)
		IDs = self.TrialIDs()		# IDs of training sets (array)
		PC = self.TrialPredCols()	# Which columns of TrialFile to read (array)

		C = self.TrialNodeCols()	# Which columns of TrialNodesFile to read (array/None)
		NF = self.TrialNodesFile()	# File containing nodes for training, one per row.
		SN = self.Scale_Nodes()		# Scale training/trial nodes to [0,1] (True/False)
		
		Trial_Nodes = np.loadtxt(NF, usecols=C, unpack=True)		# Coords of the nodes of the training set.

		if TF[-4:] == ".npy":		# The trial file is a pickle. 	
			tmp = np.load(TF)[0] 			# Assumes x-array is the zero'th dimension.
			if len(tmp.shape) == 2:
				Trial_x = tmp[0,:]
			elif len(tmp.shape) == 3:
				Trial_x = tmp[0,0,:]
			else:
				print( "Trial set contained in input file:\n %s\n Has dimensionality > 3. Too many to handle. Change this!" %TF )
				import sys				
				sys.exit()
			Trial_Pred = np.load(TF)[1]
			
			# If multiple Trial predictions, need to transpose the Trial_Nodes
			Trial_Nodes = Trial_Nodes.transpose()

		else:
			if 'XXXX' in TF:
				# If multiple Trial predictions, need to transpose the Trial_Nodes
				Trial_Nodes = Trial_Nodes.transpose()
				Trial_x = np.loadtxt('%s%s%s' %(TF.split('XXXX')[0],IDs[0],TF.split('XXXX')[1]), usecols=(PC[0],), unpack=True) 	# x-coordinate of training set predictions
																																	# currently assumes all trial set predictions
																																	# defined at same x-coords.
				# Avoids an error if Trial_x is only 1 element long.
				Trial_x = self.Fix_Single_Elem(Trial_x)
				Trial_Pred = np.empty([len(IDs), len(Trial_x)])
				for i in range(len(IDs)):
					Trial_Pred[i,:] = np.loadtxt('%s%s%s' %(TF.split('XXXX')[0],IDs[i],TF.split('XXXX')[1]), usecols=(PC[1],), unpack=True)

			elif TF == "":
				# Do not read in trial predictions as none are specified.
				Trial_Nodes = Trial_Nodes.transpose()
				Trial_x = None
				
			else:
				# There is one single Trial Pred prediction specified.
				Trial_x, Trial_Pred = np.loadtxt(TF, usecols=PC, unpack=True)	
				Trial_x = self.Fix_Single_Elem(Trial_x)	
				Trial_Pred = self.Fix_Single_Elem(Trial_Pred)	

			if len(Trial_Nodes.shape)==1:
				# If only one trial node specified, reshape the Trial_Nodes and Trial_Pred arrays
				# to be [1,len(Trial_Nodes)] and [1,len(Trial_Pred)]
				Trial_Nodes = np.reshape( Trial_Nodes, (1,len(Trial_Nodes)) )
				Trial_Pred = np.reshape( Trial_Pred, (1,len(Trial_Pred)) )

		if SN:
			# Scale Trial_Nodes to be in range [0,1] using Training set
			TrNF = self.TrainNodesFile()												# File containing nodes for training, one per row.
			C = self.TrainNodeCols()													# Columns of Training Files to read (array/None)
			Train_Nodes = np.loadtxt(TrNF, usecols=C, unpack=True).transpose()			# Coords of the nodes of the training set.
			# Normalise Train_Nodes to be in range[0,1]
			for i in range(0,Train_Nodes.shape[1]):
				Trial_Nodes[:,i] = abs(Trial_Nodes[:,i]-Train_Nodes[:,i].min()) / abs(Train_Nodes[:,i].max()-Train_Nodes[:,i].min())


		# Check if Trial_x and Train_x are different. If they are, interpolate the trial predictions onto the training.
		TF_Train = self.TrainFile()		# Training Filename	(single string)
		IDs_Train = self.TrainIDs()		# IDs of training sets (array)

		# Check if training files are a pickle...
		if TF_Train[-4:] == ".npy":		# The training file is a pickle. 	
			tmp = np.load(TF_Train)[0] 			# Assumes x-array is the zero'th dimension.
			if len(tmp.shape) == 2:
				Train_x = tmp[0,:]
			elif len(tmp.shape) == 3:
				Train_x = tmp[0,0,:]
			else:
				print( "Training set contained in input file:\n %s\n Has dimensionality > 3. Too many to handle. Change this!" %TF )
				import sys				
				sys.exit()
			Train_x = self.Fix_Single_Elem(Train_x) # Avoids an error if Train_x is 1 element long.

		# ...if not, read in Train_x as first column of first Train prediction file
		else:					
			Train_x = np.loadtxt('%s%s%s' %(TF_Train.split('XXXX')[0],IDs_Train[0],TF_Train.split('XXXX')[1]), usecols=(PC[0],), unpack=True) 	# x-coordinate of training set predictions
																																		# currently assumes all training set predictions
																																		# defined at same x-coords.
			Train_x = self.Fix_Single_Elem(Train_x) # Avoids an error if Train_x is 1 element long.

		if TF != "":
			if np.array_equal(Train_x, Trial_x) == False:
				print( "Interpolating the trial predictions onto the x-array of the training predictions." )
				Trial_Pred = self.Interpolate_Trial_Onto_Train(Train_x, Trial_x, Trial_Pred)
				Trial_x = Train_x		
		else:
			# No TrialFile specified
			Trial_x = Train_x
			Trial_Pred = np.ones([ Trial_Nodes.shape[0], len(Train_x) ])

		return Trial_x, Trial_Pred, Trial_Nodes







# Class for performing pincipal component analysis up to a specified number of basis functions.
# Options are to find the basis functions for some input data, 
# or to use pre-specified basis functions to some data to find corresponding weights
class PCA_Class:

	def __init__(self, n_components):
		self.n_components = n_components

	def Load_BFs(self, BFsFile, BFsDataMean):		# write this
		return #xvalues, BFs
	
	def PCA_BySKL(self, Data):	# use scikit-learn
		pca = PCA(n_components=self.n_components)
		Weights = pca.fit_transform(Data)			# Weights of the PCA
		Recons = pca.inverse_transform(Weights)	# The reconstructions
		BFs = pca.components_                       # Derived basis functions
		return BFs, Weights, Recons

	# Accept some basis functions, BFs, Data to perform PCA on,
    # + the mean (in each bin) of the data for which the BFs were identified,
    # and manually do the PC Reconstruction
	def PCA_ByHand(self, BFs, Data, data_Mean):
		Data_MinusMean = np.empty_like( Data )
		for i in range(len(Data[0,:])):
			Data_MinusMean[:,i] = Data[:,i] - data_Mean[i]
		Weights = np.dot(Data_MinusMean,np.transpose(BFs))
        
		Recons = np.zeros([ Data.shape[0], BFs.shape[1] ]) 
		for j in range(len(Weights[:,0])):		    # Scroll through the the Data
			for i in range(len(Weights[0,:])): 	    # scroll through the basis functions
				Recons[j,:] += Weights[j,i]*BFs[i,:]
			Recons[j,:] += data_Mean
		return Weights, Recons


	# Read in Weights, BFS, and data_Mean to recover original statistic
	def Convert_PCAWeights_2_Predictions(self, Weights, BFs, Mean):
		if len(Weights.shape) == 1:
			Weights = Weights.reshape(1, len(Weights))
		Predictions = np.zeros([ Weights.shape[0], BFs.shape[1] ]) 
		for j in range(Weights.shape[0]):		# Scroll through number of predictions one needs to make.
			Predictions[j,:] += Mean
			for i in range(BFs.shape[0]):		# Scroll through BFs adding the correct linear combination to the Mean
				Predictions[j,:] += Weights[j,i] * BFs[i,:]
		return Predictions










		






# Class for doing emulation via Gaussian process regression
class GPR_Emu:

	def __init__(self, nodes_train, y_train, yerr_train, nodes_trial):
		self.nodes_train = nodes_train
		self.y_train = y_train
		self.yerr_train = yerr_train
		self.nodes_trial = nodes_trial


	def GPR(self, p, iterations):

		kernel = np.exp(p[0])*ExpSquaredKernel(np.exp(p[1:]), ndim=len(p)-1)
		gp = george.GP(kernel)

		# Gauge dimensions of y_train, so as to make appropriate output storer
		# L is the length of the statistic you're predicting
		L = self.y_train.shape[1]

		# Gauge size of trial set, so as to make appropriate output storer
		# D is the number of nodes.
		if len(self.nodes_trial.shape) == 1:
			D = 1
		else:
			D = self.nodes_trial.shape[0]


		GP_OUT = np.zeros([ iterations, D, L ])	# GP_OUT[i,j,k] = i'th iteration, j'th cosmology, k'th theta bin

		for k in range(L):
			#print( "Generating %s predictions for %s trial cosmologies in bin %s" %(iterations, len(self.nodes_trial), k) )
			if L == 1:
				gp.compute(self.nodes_train, self.yerr_train)
			else:
				gp.compute(self.nodes_train, self.yerr_train[:,k])
			GP_OUT[:,:,k] = gp.sample_conditional(self.y_train[:,k], self.nodes_trial, iterations)

		# Now average the iterations and take a standev
		GP_AVOUT = np.zeros([D, L])
		GP_STDOUT = np.zeros([D, L])
		for j in range(0, D):
			for k in range(0, L):
				GP_AVOUT[j,k] = np.mean(GP_OUT[:,j,k])
				GP_STDOUT[j,k] = np.std(GP_OUT[:,j,k])

		return gp, GP_OUT, GP_AVOUT, GP_STDOUT


	def GPRsk(self, p, alpha, n_restarts_optimizer):
		
		kernel = p[0] * RBF(p[1:])
		if n_restarts_optimizer is not None:
			print("Optimising the emulator with %s restarts..." % n_restarts_optimizer)
		#else:
			#print("Not training the emulator, using pre-set hyperparameters.")

		if alpha is not None:
			gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, alpha=alpha)		
		else:
			gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
		gp.fit(self.nodes_train, self.y_train)
		#print("Right, we're running emulator with kernel hyperparameters:")
		#print(gp.kernel_)
		GP_AVOUT, GP_STDOUT = gp.predict(self.nodes_trial, return_std=True)
		
		# -------------------------------------------------------------------------
		# THIS BIT OF CODE EXTRACTED MANY SAMPLES FROM THE GP SO THE USER CAN CALC
		# THE FULL COVARIANCE OF THE PREDICTIONS. IT IS SLOW AND CAUSES DIMENSIONALITY
		# PROBLEMS WHEN RAN WITH Train_Error SET TO True .

		#Get_Samples = False
		#n_samples = 10
		#if Get_Samples:		
			# Get many samples of the prediction and return these to estimate the full 
			# covariance of the samples.
		#	GP_SAMPLES = gp.sample_y(self.nodes_trial, n_samples=n_samples, random_state=0)
		#else:
		#	GP_SAMPLES = np.zeros([ GP_AVOUT.shape[0], GP_AVOUT.shape[1], n_samples ])

		#return GP_AVOUT, GP_STDOUT, GP_SAMPLES, self.Process_gpkernel(gp.kernel_)
		# -------------------------------------------------------------------------

		return GP_AVOUT, GP_STDOUT, self.Process_gpkernel(gp.kernel_)

	def Process_gpkernel(self, gpkernel):
		hp_amp = eval( str(gpkernel).split()[0] )
		hp_rest = eval( str(gpkernel).split('length_scale=')[-1].split(')')[0] ) 
		return np.append( hp_amp, hp_rest ) 

	def Cross_Validation(self, HPs, Perform_PCA, n_components, Train_BFs, Mean_TS, Include_x, x_coord, Train_Error, alpha, n_restarts_optimizer):
		import time
		# Cycle through training set, omitting one, training on the rest, and testing accuracy with the omitted.
		t1 = time.time()
		NumNodes = self.y_train.shape[0]

		if Perform_PCA:
			PCACall =  PCA_Class(n_components)
			if Include_x:
				Predictions = np.empty([NumNodes*len(x_coord),1])
				rel_length = n_components							# the length of the statistic predicted
			else:
				Predictions = np.empty([NumNodes, len(x_coord)])
		else:
			Predictions = np.empty_like( self.y_train )
			rel_length = len(x_coord)								# the length of the statistic predicted

		GP_HPs_AllNodes = np.zeros([ NumNodes, Predictions.shape[1], self.nodes_train.shape[1]+1 ])
		for i in range(NumNodes):
	
			print( "Performing cross-val. for node %s of %s..." %(i, NumNodes) )
			if Include_x: 
				new_nodes_trial = self.nodes_train[i*rel_length:(i+1)*rel_length,:]
				new_y_trial = self.y_train[i*rel_length:(i+1)*rel_length,:]
				new_nodes_train = np.delete(self.nodes_train, slice(i*rel_length, (i+1)*rel_length), axis=0)
				new_y_train = np.delete(self.y_train, slice(i*rel_length, (i+1)*rel_length), axis=0)
				new_yerr_train = np.delete(self.yerr_train, slice(i*rel_length, (i+1)*rel_length), axis=0)
			else:
				new_nodes_trial  = self.nodes_train[i,:].reshape((1,len(self.nodes_train[i,:])))
				#print( self.y_train.shape )
				new_y_trial = self.y_train[i,:]
				new_nodes_train = np.delete(self.nodes_train, i, axis=0)
				new_y_train = np.delete(self.y_train, i, axis=0)
				new_yerr_train = np.delete(self.yerr_train, i, axis=0)
	
			if Train_Error:
				# If providing an error, SKL requires you run each x-bin separately:
				GP_AVOUT = np.zeros([ new_nodes_trial.shape[0], new_y_train.shape[1] ])
				GP_STDOUT = np.zeros([ new_nodes_trial.shape[0], new_y_train.shape[1] ])
				for j in range(GP_AVOUT.shape[1]):
					new_GPR_Class = GPR_Emu(new_nodes_train, new_y_train[:,j], new_yerr_train[:,j], new_nodes_trial)
					if len(HPs.shape) == 1: 
						# We do not have individual HPs per bin
						GP_AVOUT[:,j], GP_STDOUT[:,j], GP_HPs_AllNodes[i,j,:] = new_GPR_Class.GPRsk(HPs, new_yerr_train[:,j], n_restarts_optimizer)
					else:
						# We do have individual HPs per bin!
						GP_AVOUT[:,j], GP_STDOUT[:,j], GP_HPs_AllNodes[i,j,:] = new_GPR_Class.GPRsk(HPs[j], new_yerr_train[:,j], n_restarts_optimizer)

			else:
				new_GPR_Class = GPR_Emu(new_nodes_train, new_y_train, alpha, new_nodes_trial)				
				GP_AVOUT, GP_STDOUT, GP_HPs_AllNodes[i,:,:] = new_GPR_Class.GPRsk(HPs, alpha, n_restarts_optimizer) 				# with Scikit-Learn	
				GP_STDOUT = np.repeat(np.reshape(GP_STDOUT, (-1,1)), GP_AVOUT.shape[1], axis=1)				# SKL only returns 1 error bar per trial here
		

			if Perform_PCA:		# If Perform_PCA is True, need to convert weights returned from emulator to predictions
				if Include_x:	
					Predictions[i*len(x_coord):(i+1)*len(x_coord),:] = PCACall.Convert_PCAWeights_2_Predictions(GP_AVOUT.transpose(), Train_BFs, Mean_TS).transpose()
				else:
					Predictions[i,:] = PCACall.Convert_PCAWeights_2_Predictions(GP_AVOUT, Train_BFs, Mean_TS)
			else:
				if Include_x:
					Predictions[i*len(x_coord):(i+1)*len(x_coord),:] = GP_AVOUT
				else:
					Predictions[i,:] = GP_AVOUT

		t2 = time.time()
		print( "Whole cross-val. took %.1f s for %i nodes..." %((t2-t1), NumNodes) )

		return Predictions, GP_HPs_AllNodes





	##### THE FOLLOWING FUNCTIONS ARE ALL USED IN THE MCMC FOR HYPERPARAMS ###################################

	# log prior
	def lnprior(self, p, lnprior_ranges):
		for i in range(len(p)):
			if (p[i] < lnprior_ranges[i,0]) or (p[i] > lnprior_ranges[i,1]):
				return -np.inf
		return 0.

	# log likelihood
	def lnlike(self, p, y_trial):

		L = self.y_train.shape[1]

		# New likelihood
		kernel = np.exp(p[0])*ExpSquaredKernel(np.exp(p[1:]), ndim=len(p)-1)
		gp = george.GP(kernel)

		SumLnL = 0.
		for k in range(L):
			if L == 1:
				gp.compute(self.nodes_train, self.yerr_train)
			else:
				gp.compute(self.nodes_train, self.yerr_train[:,k])
			SumLnL += gp.lnlikelihood(self.y_train[:,k])
		return SumLnL


	# log posterior
	def lnprob(self, p, y_trial, lnprior_ranges):
		lp = self.lnprior(p, lnprior_ranges)
		return lp + self.lnlike(p, y_trial) if np.isfinite(lp) else -np.inf


	def Run_MCMC(self, y_trial, nwalkers, burn_steps, real_steps, lnprior_ranges, p):

		import emcee
		import time
		ndim = len(p)
		p0 = [p + 1e-2*np.random.randn(ndim) for i in range(nwalkers)]		# starting position
		sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, args=[y_trial, lnprior_ranges])

		t0 = time.time()
		print( "Running burn in of %s steps per walker...." %burn_steps )
		p0,lnp, _ = sampler.run_mcmc(p0, burn_steps)
		sampler.reset()

		t1 = time.time()
		print( "First burn-in took %.1f minutes. Running 2nd burn-in..." %((t1-t0)/60.) )
		# set new start point to be a tiny gauss ball around position of whatever walker reached max posterior during burn-in
		p = p0[np.argmax(lnp)]
		p0 = [p + 1.e-2* p * np.random.randn(ndim) for i in xrange(nwalkers)]
		p0, _, _ = sampler.run_mcmc(p0, burn_steps)
		sampler.reset()

		t2 = time.time()
		print( "Second burn-in took %.1f minutes. Running the MCMC proper with %s steps per walker" %(((t2-t1)/60.), real_steps) )
		sampler.run_mcmc(p0, real_steps)
		# sampler has an attribute called chain that is 3D: nwalkers * real_steps * ndim in dimensionality
		# Following line turns it into 2D: (nwalkers*real_steps) * ndim. 
		# As if there was only one walker.
		samples = sampler.chain[:, :, :].reshape((-1, ndim))
		t3 = time.time()
		print( "Finished. Main MCMC took %.1f minutes. The whole MCMC took %.1f minutes." %( ((t3-t2)/60.), ((t3-t0)/60.) ) )

		return samples



	##### THE ABOVE	 FUNCTIONS ARE ALL USED IN THE MCMC FOR HYPERPARAMS ###################################

class Diagnostic_Plots:

	def __init__(self, x, y_GP, yerr_GP, y_compare, yerr_compare):
		self.x = x
		self.y_GP = y_GP
		self.yerr_GP = yerr_GP
		self.y_compare = y_compare
		self.yerr_compare = yerr_compare

	def errRatio(self, num, errnum, denom, errdenom):
		Ratio = num / denom -1.
		errRatio = Ratio * np.sqrt( (errnum/num)**2. + (errdenom/denom)**2. )
		return Ratio, errRatio

	# Following are diagnostic plots to access accuracy of emulator
	def Plot_GPvsTrial(self, savename):
		FD, errFD = self.errRatio( self.y_GP, self.yerr_GP, self.y_compare, self.yerr_compare ) 
		plt.figure()
		for i in range(FD.shape[0]):
			plt.errorbar( self.x, FD[i,:], yerr=errFD[i,:] )
		#plt.plot([self.x.min(),self.x.max()], [0.,0.], 'k--')
		plt.xscale('log')
		plt.xlim([self.x.min(), self.x.max()])
		plt.ylim([FD.min(), FD.max()])
		plt.xlabel(r'$x$')
		plt.ylabel(r'(GP - Data) / Data')
		#plt.savefig(savename)
		#plt.show()
		return


	def Plot_CV(self, include_x, savename):

		NumNodes = self.y_GP.shape[0]
		plt.figure()
		for i in range(NumNodes):
			if include_x:
				R = (self.y_GP[i*len(self.x):(i+1)*len(self.x),:].transpose() / self.y_compare[i,:])[0,:]
			else:
				R = self.y_GP[i,:] / self.y_compare[i,:]
			plt.plot( self.x, R - 1. )
		plt.title(r'Cross-Validation: %s Nodes in training set' %NumNodes)
		plt.plot([self.x.min(),self.x.max()], [0.,0.], 'k--')
		plt.xscale('log')
		plt.xlim([self.x.min(), self.x.max()])
		plt.ylim([-0.5, 0.5])
		plt.xlabel(r'$x$')
		plt.ylabel(r'(GP - Data) / Data')
		plt.savefig(savename)
		plt.show()
		return


	def Plot_CV_Inacc(self,coords,threshold,labels,limits,savename):
		import matplotlib.gridspec as gridspec
		if labels == None or len(labels) != coords.shape[1]:
			new_labels = []
			for i in range(coords.shape[0]):
				new_labels.append('X%s'%i)
			labels = new_labels

		statistic = 100*(self.y_GP / self.y_compare -1)
		if len(statistic.shape) > 1:
			statistic_avg = np.zeros(statistic.shape[0])
			for i in range(len(statistic_avg)):
				statistic_avg[i] = np.mean( statistic[i,:] )
			statistic = statistic_avg

		count_inacc = len(np.where(statistic > threshold)[0])
		cmap = plt.get_cmap('jet')
		colors = [cmap(i) for i in np.linspace(0, 1, count_inacc)]

		fig = plt.figure(figsize = (12,10)) #figsize = (20,14)
		gs1 = gridspec.GridSpec(coords.shape[1]-1,coords.shape[1]-1)
		p=0	# subplot number
		for i in range(coords.shape[1]-1):
			l=i+1 # which y-axis statistic is plotted on each row.	
			for j in range(coords.shape[1]-1):
				ax1 = plt.subplot(gs1[p])
				if j>i:
					ax1.axis('off')
				else:
					ax1.scatter(coords[:,j], coords[:,l], color='dimgrey',s=20)
					color_idx=0
					for s in range(coords.shape[0]):
						if statistic[s] > threshold: 
							ax1.scatter(coords[s,j], coords[s,l], color=colors[color_idx], s=30 )
							color_idx+=1

					# Decide what the axis limits should be. If limits=None, it doesn't set any.
					# If limits is [a,b], limits are set to be a*min and b*max in each dimension.
					# Else, limits is interpreted as [ [x1,x2],[y1,y2],[z1,z2]...] for each dimension. 
					if limits != None:
						if len(np.array(limits).shape) == 1:
							ax1.set_xlim([ limits[0]*coords[:,j].min(),limits[1]*coords[:,j].max() ]) 
							ax1.set_ylim([ limits[0]*coords[:,l].min(),limits[1]*coords[:,l].max() ]) 
						else:
							ax1.set_xlim([ limits[j][0],limits[j][1] ]) 
							ax1.set_ylim([ limits[l][0],limits[l][1] ])
                                                        
                                        # If the axes tick labels are too busy, uncomment the following!:
					# Set every other x/y-tick to be invisible
					#if len(ax1.get_xticklabels()) > 5: # Too many things will plot on x-axis, 
													# so only plot every third 
					#	for thing in ax1.get_xticklabels():
					#		if thing not in ax1.get_xticklabels()[::3]:
					#			thing.set_visible(False)
					#else: 
					#	for thing in ax1.get_xticklabels()[::2]:
					#		thing.set_visible(False)
					#for thing in ax1.get_yticklabels()[::2]:
					#	thing.set_visible(False)

				
					# Get rid of x/y ticks completely for subplots not at the edge
					if j==0:
						ax1.set_ylabel(labels[l])
					else:
						ax1.set_yticks([])
					if i==coords.shape[1]-2:
						ax1.set_xlabel(labels[j])
					else:				
						ax1.set_xticks([])
				p+=1
		fig.suptitle(r'Training nodes for which mean accuracy worse than %s per cent (coloured points)' %threshold)
		plt.savefig(savename)
		plt.show()
		return



	def Plot_MCMC_Results(self,samples, labels, savename):
		import matplotlib.gridspec as gridspec
		steps2plot = np.linspace(0., samples.shape[0]-1, samples.shape[0]) 
		if labels == None:
			labels = [r'$\ln(a)$']
			for i in range(samples.shape[1]-1):				
				labels.append('$\ln(\rm{HP}_%s)$'%i)
		l = len(labels)

		fig = plt.figure(figsize = (8,16))
		gs1 = gridspec.GridSpec(len(labels), 1)
		for i in range(l):
			# i = i + 1 # grid spec indexes from 0
			ax1 = plt.subplot(gs1[i])
			ax1.plot(steps2plot, samples[:,i], color='magenta', linewidth=1)
			if i != (l-1):
				ax1.set_xticklabels([])
			ax1.set_ylabel(labels[i])
			ax1.set_xlabel(r'Number of steps')
		#gs1.update(wspace=0., hspace=0.) # set the spacing between axes.
		#fig.subplots_adjust(hspace=0, wspace=0)
		plt.savefig('%s_Steps.png'%savename)
		#plt.show()

		import corner
		fig = corner.corner(samples, labels=labels)
		if savename != None and savename != '':
			plt.savefig('%s_Contours.png' %savename)

		# Find the peak HPs
		peak_HPs = []
		for i in range(samples.shape[1]):
			hist, edges = np.histogram( samples[:,i], bins=int(samples.shape[0]/500))
			dw = 0.5*(edges[1]-edges[0])
			peak_HPs.append( float('%.3f'%(edges[np.argmax(hist)]+dw))  )

		plt.show()
		return peak_HPs






	
