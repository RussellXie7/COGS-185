import re
import numpy as np

from matplotlib import pyplot
from numpy.linalg import norm, svd
import sklearn.datasets
from sklearn import preprocessing

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

from sklearn.decomposition import PCA

# Source: http://kastnerkyle.github.io/posts/robust-matrix-decomposition/
# I did not modify the function from the source.
def inexact_augmented_lagrange_multiplier(X, lmbda=.01, tol=1e-3,
                                          maxiter=100, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break
    if verbose:
        print("Finished at iteration %d" % (itr))  
    return A, E

# I use the code snippet from https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm/7369986
# It read PGM image buffer string and return a numpy array
def read_pgm2(buffer, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    #with open(filename, 'rb') as f:
    #    buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    #print 'width',width, 'height', height
    
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=15#len(header)
                            ).reshape((int(height), int(width)))



dataset0 = sklearn.datasets.load_files('CroppedYale', shuffle=True)
print dir(dataset0)

# Here is some sanity checks of the parsed dataset
idx0 = len(dataset0.filenames)-1
print dataset0.filenames[idx0]
print dataset0.target[idx0]
print dataset0.target_names[dataset0.target[idx0]]
print len(dataset0.filenames)
image0 = read_pgm2(dataset0.data[idx0])
#pyplot.imshow(image0, pyplot.cm.gray)


# This cell includes the splittign of training and test set.
size0 = 1280
separate = 1024
X = np.stack(map(read_pgm2, dataset0.data))
X = X[:size0]

shapeX = X.shape
X = X.reshape((shapeX[0],shapeX[1]*shapeX[2]))
X = preprocessing.scale(X.astype(float),axis=1) # Mean removal and variance scaling

y = dataset0.target
y = y[:size0]
#imageX = X[0].reshape((shapeX[1],shapeX[2]))
#pyplot.imshow(imageX, pyplot.cm.gray)

print X.shape, y.shape

X_train = X[:separate]
y_train = y[:separate]
X_test = X[separate:]
y_test = y[separate:]

print y_test[:10]

print
print
print "..."
print "......"
print "---------------------- Training SVM Scheme ---------------------"
print

clf = LinearSVC(random_state=0)
print clf.fit(X_train,y_train)

# print (clf.coef_)
# print (clf.intercept_)

print "----- Testing.... -----"
print "The test score result is: "
print clf.score(X_test,y_test);
print


print "---------------------- Training PCA-SVM Scheme ---------------------"
print
print "Reduce dimension..."

pca = PCA(n_components=300)
pc_samples = pca.fit_transform(X)

print "Principle components samples with dimension reduced: "
print pc_samples.shape
pc_X_train = pc_samples[:separate]
pc_y_train = y_train
pc_X_test = pc_samples[separate:]
pc_y_test = y_test

print "Training using linear svm..."
clf = LinearSVC(random_state=0)
print clf.fit(pc_X_train, pc_y_train)
print
print "Testing...."
print clf.score(pc_X_test, pc_y_test)
print 
print


print "---------------------- Training RPCA-PCA-SVM Scheme ---------------------"

print "Obtaining the low rank matrix..."
# The cell is to demonstrate how to use the IALM function
A, E = inexact_augmented_lagrange_multiplier(X)
    
print A.shape
print E.shape

print "Reduce dimension..."

pca = PCA(n_components=300)
pc_samples = pca.fit_transform(A)

print "Principle components samples with dimension reduced: "
print pc_samples.shape
pc_X_train = pc_samples[:separate]
pc_y_train = y_train
pc_X_test = pc_samples[separate:]
pc_y_test = y_test

print "Training using linear svm..."
clf = LinearSVC(random_state=42)
print clf.fit(pc_X_train, pc_y_train)
print
print "Testing...."
print clf.score(pc_X_test, pc_y_test)
print 
print



# # How do I get the reduced training data set...??
















