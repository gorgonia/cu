#include <cudnn.h>

/* THIS IS TEMPORARY, UNTIL I FIGURE OUT A BETTER WAY TO SET UNION VALUES IN GO */

cudnnAlgorithm_t makeConvFwdAlgo(cudnnConvolutionFwdAlgo_t algo) {
	cudnnAlgorithm_t retVal;
	retVal.algo.convFwdAlgo = algo;
	return retVal;
}

cudnnAlgorithm_t makeConvBwdFilterAlgo(cudnnConvolutionBwdFilterAlgo_t algo){
	cudnnAlgorithm_t retVal;
	retVal.algo.convBwdFilterAlgo = algo;
	return retVal;
}


cudnnAlgorithm_t makeConvBwdDataAlgo(cudnnConvolutionBwdDataAlgo_t algo){
	cudnnAlgorithm_t retVal;
	retVal.algo.convBwdDataAlgo = algo;
	return retVal;
}

cudnnAlgorithm_t makeRNNAlgo(cudnnRNNAlgo_t algo) {
       	cudnnAlgorithm_t retVal;
	retVal.algo.RNNAlgo = algo;
	return retVal;
}

cudnnAlgorithm_t makeCTCLossAlgo(cudnnCTCLossAlgo_t algo) {
	cudnnAlgorithm_t retVal;
	retVal.algo.CTCLossAlgo = algo;
	return retVal;
}
