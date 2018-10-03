import numpy as np 
from hmmlearn import hmm
np.random.seed(42)


# model = hmm.GaussianHMM(n_components=3, covariance_type = "full")
# model.startprob_ = np.array([0.6,0.3,0.1])
# model.transmat_ = np.array([[0.7,0.2,0.1],[0.3,0.5,0.2],[0.3,0.3,0.4]])
# model.means_ = np.array([[0.0,0.0],[3.0,-3.0],[5.0,10.0]])
# model.covars_ = np.tile(np.identity(2),(3,1,1))
# X, Z = model.sample(100)

# print X;
# print Z;


lr = hmm.GaussianHMM(n_components=3, covariance_type="diag", 
	init_params="cm",params="cmt")

lr.startprob_ = np.array([1.0,0.0,0.0])
lr.transmat_ = np.array([[0.5,0.5,0.0],[0.0,0.5,0.5],[0.0,0.0,1.0]])
lr.means_ = np.array([[0.0,0.0],[3.0,-3.0],[5.0,10.0]])
lr.covars_ = np.tile(np.identity(2),(3,1,1))

X,Z=lr.sample(100)

print X
print Z