from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def GaussianProcessInterpolation(measurement_data,max_insize, max_outsize, insize_window, outsize_window):
	#df90 = measurement_data.groupby(["conv1_out","conv2_out"]).quantile(0.75).reset_index()
	X_total = measurement_data[['conv1_out', 'conv2_out']].as_matrix()
	Y_total = measurement_data['measurements'].as_matrix()

	input_list = np.arange(0, max_insize+1, insize_window)
	output_list = np.arange(0, max_outsize+1, outsize_window)

	intermediate_indata = measurement_data[measurement_data['conv1_out'].isin(input_list)]
	final_outdata = intermediate_indata[intermediate_indata['conv2_out'].isin(output_list)]
	#summary_stats = final_outdata.groupby(["conv1_out","conv2_out"]).quantile(0.75).reset_index()

	X = final_outdata[['conv1_out','conv2_out']].as_matrix()
	Y = final_outdata[['measurements']].as_matrix()

	kernel = DotProduct() + WhiteKernel()
	gpr = GaussianProcessRegressor(kernel = kernel , random_state =0).fit(X,Y)
	#print(gpr.score(X,Y))
	y_pred, sigma = gpr.predict(X_total, return_std = True)
	return (np.sum(np.abs(y_pred - Y_total)**2)), np.mean(sigma)
	#####
	#plt.figure()
	#plt.plot(X_total, Y_total)
	#plt.errorbar(X, y_pred, 'b-', label = u'Predictions')
###########################################################
## get_IEs(file_name,iter_num): returns a bootstrap of the dataset everytime called
## Return type : pd data frame
###########################################################
def get_IEs(file_name, iter_num):
	ie_sigmas = np.ones((iter_num,2))*-1
	data = pd.read_csv(file_name)
	for i in range(iter_num):
		boot = data.groupby(["conv1_out", "conv2_out"]).apply(lambda x : x.sample(1, replace = True)).reset_index(drop = True)
		ie_sigmas[i] = GaussianProcessInterpolation(boot, 20, 50, 4,5)
		#print(sigma)
		#print(error)
		#ie_sigmas[i,0] = error
		#ie_sigmas[i,1] = sigma
	print(ie_confidences)
if __name__ == "__main__":
	a = get_IEs('lenet5_conv2.csv', 200)
	
	
	
