import numpy as np 
from hmmlearn import hmm
np.random.seed(42)
import warnings
warnings.filterwarnings('ignore')
import pickle
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.utils import check_random_state

L = 15

ghmm0 = hmm.GaussianHMM(n_components=4, covariance_type="diag")

ghmm0.startprob_ = np.array([0.6,0.3,0.1,0.0])
ghmm0.transmat_ = np.array([[0.7,0.2,0.0,0.1],[0.3,0.5,0.2,0.0],[0.0,0.3,0.5,0.2],[0.2,0.0,0.2,0.6]])
ghmm0.means_ = np.array([[0.0,0.0],[0.0,11.0],[9.0,10.0],[11.0,-1.0]])
ghmm0.covars_ = 0.5 * np.tile(np.ones([1,2]),(4,1))

print "startprob_: \n", ghmm0.startprob_
print "transmat_: \n", ghmm0.transmat_
print "means_: \n", ghmm0.means_
print "covars_: \n", ghmm0.covars_


X0_a,Z0_a = ghmm0.sample(L)
X0_b,Z0_b = ghmm0.sample(L)
X0_c,Z0_c = ghmm0.sample(L)

print
print
print "============= first pair ============="
print "-------------    x0:     -------------\n", X0_a
print "-------------    z0:     -------------\n", Z0_a
print 
print "============= second pair ============="
print "-------------    x0:     -------------\n", X0_b
print "-------------    z0:     -------------\n", Z0_b
print 
print "============= third pair ============="
print "-------------    x0:     -------------\n", X0_c
print "-------------    z0:     -------------\n", Z0_c
print
print

POSTERIOR_0 = ghmm0.predict_proba(X0_c)

formatted_output = np.asarray([[0 if x < 0.1 else x for x in z] for z in POSTERIOR_0])

print POSTERIOR_0
print
print formatted_output
print 
print
print

# =========================== Task 2.2.1 ==============================

X0, Z0 = X0_c, Z0_c

ghmm1 = hmm.GaussianHMM(n_components=4, covariance_type="diag")
ghmm1.fit(X0)


print " ------------------- for ghmm0 ---------------------- " 
print "startprob_: \n", ghmm0.startprob_
print "transmat_: \n", ghmm0.transmat_
print "means_: \n", ghmm0.means_
print "covars_: \n", ghmm0.covars_
print
print " ------------------- for ghmm1 ---------------------- " 
print "startprob_: \n", ghmm1.startprob_
print "transmat_: \n", ghmm1.transmat_
print "means_: \n", ghmm1.means_
print "covars_: \n", ghmm1.covars_

# =========================== Task 2.2.2 ==============================

x_con = np.concatenate([X0_a,X0_b,X0_c])

lengths_con = [len(X0_a), len(X0_b), len(X0_c)]
ghmm1_con = hmm.GaussianHMM(n_components=4, covariance_type="diag")
ghmm1_con.fit(x_con,lengths_con)

print " ------------------- for ghmm1_con ---------------------- " 
print "startprob_: \n", ghmm1_con.startprob_
print "transmat_: \n", ghmm1_con.transmat_
print "means_: \n", ghmm1_con.means_
print "covars_: \n", ghmm1_con.covars_

# =========================== Task 2.2.3 ==============================
# predict the states with the two Gaussian hMMs

Z1 = ghmm0.predict(X0)
Z2 = ghmm1.predict(X0)

print
print "z0: ",Z0
print "z1: ",Z1
print "z2: ",Z2
print
print


# =========================== Task 2.3 ==============================

quotes = pickle.load(open('my_quotes.obj','r'))
diff_c = np.diff(quotes.Close)
binom1 = np.column_stack([diff_c[:100], quotes.Volume[1:101]/3e7])

ghmm2 = hmm.GaussianHMM(n_components=3, covariance_type="diag")
ghmm2.fit(binom1)

states = ghmm2.predict(binom1)

print 
print "---------------- ghmm2 params ------------------"
print "startprob_: \n", ghmm2.startprob_
print "transmat_: \n", ghmm2.transmat_
print "means_: \n", ghmm2.means_
print "covars_: \n", ghmm2.covars_

# # Visualization code
# close_p = quotes.Close[1:101]
# dates = np.arange(len(close_p))

# fig, axs = plt.subplots(ghmm2.n_components+1, sharex=True, sharey=True)
# colours = cm.rainbow(np.linspace(0, 1, ghmm2.n_components))
# axs[0].plot(dates, close_p)
# axs[0].set_title("Closing prices from day 1 to day 100.")
# axs[0].grid(True)
# for i  in range(1,ghmm2.n_components+1):
#     mask = states == i-1
#     axs[i].plot(dates[mask], close_p[mask], ".-", c=colours[i-1])
#     axs[i].set_title("#{0} hidden state".format(i-1))

#     axs[i].grid(True)

# plt.show()
# # end of visualization code


# --------------------- Predict future ---------------------------

print 
print
print
print "---------------------- Predict future ---------------------------"
print 
L=15 # We would like to predict the following 15 days' trend
Niter = 10 # A hyper parameter of generating samples

warnings.filterwarnings('ignore')
binom0 = np.column_stack([np.diff(quotes.Close), np.array(quotes.Volume)[1:]/3e7])
binom2 = np.copy(binom1)

startprob_cdf = np.cumsum(ghmm2.startprob_)
transmat_cdf = np.cumsum(ghmm2.transmat_, axis=1)
random_state = ghmm2.random_state

rs = check_random_state(None)

for l in range(L):
    binom2 = np.append(binom2,[[0,0]],axis=0) # Add a pair of empty (d,v)
    true_binom = np.copy(binom0[:len(binom1)+l])
    state_seq = ghmm2.predict(true_binom)
    previous_state = state_seq[-1]
    
    maxLL = -1e10
    for n in range(Niter):
        currstate = (transmat_cdf[previous_state]> rs.rand() ).argmax() # Go through transmat to get a new state
       
        new_sample = ghmm2._generate_sample_from_state(currstate) # generate from the new state
        tmp_binom = np.copy(true_binom)
        tmp_binom = np.append(tmp_binom,[new_sample],axis=0) # Append the new_sample for score
        tmp_maxLL = ghmm2.score(tmp_binom) # 
        if tmp_maxLL > maxLL :

                maxLL = tmp_maxLL
                binom2[-1][0] = new_sample[0]
                binom2[-1][1] = new_sample[1]

# The curve after day 100 is the predicted trend.

date2 = dates = np.arange(len(binom2))
# print len(date2)
plt.figure()
plt.plot(date2, quotes.Close[0]+np.cumsum(binom2[:,0]))
plt.plot(date2, quotes.Close[:len(binom1)+L])#[100:100+25])
plt.grid(True)
plt.legend(('predicted', 'ground truth'))
plt.title("Closing Prices")

# The curve after day 100 is the predicted trend.

plt.figure()
plt.plot(date2, binom2[:,1]*3e7)
plt.plot(date2, quotes.Volume[0:len(binom1)+L])#[100:100+25])
plt.grid(True)
plt.legend(('predicted', 'ground truth'))
plt.title("Volume")
plt.show()




















