#Anomaly detection system
#In this exercise we will try to train a system to detect anomalous behavior of a machine on the basis of its vibration & voltage surge.

#Steps:
#1. We will first train a Gaussian distribution based AI model
#2. Then we will estimate F1 score at different probability value to get most optimal probability value
import numpy             as np
import matplotlib.pyplot as plt
import scipy.optimize    as opt
import scipy.io          as io
from scipy.stats         import norm

################################# All functions Begin ###############################
def calcMeanVar(X):
    #Get number of columns = number of features
    col = np.shape(X)[1]

    #Create empty array to store Mean and Std. deviation for each feature
    feat = np.mat(np.zeros(shape=(2,col)))

    #Calculate mean and variance for each column and store
    feat[0,:] = np.mean(X, axis=0)  #Mu
    feat[1,:] = np.var(X, axis=0)   #Sigma2
    print(feat)
    return feat

#Calculate probability as per parameters, mv is features of
# xt stores all the parameters in a row fashion for each data
# mv stores all the mu and sigma2 in [2 X n] matrix
def calcProb(xt, mv):
    #Number of features & examples
    fcount  = np.shape(mv)[1]
    rows    = np.shape(xt)[0]
        
    #compute probability multiplication for each feature
    prob = 1
    result_v = np.zeros(shape=(rows,1))
    #print(np.shape(xt))

    #For each example
    for r in range(rows):
        xtemp = xt[r,:]

        #For each feature
        prob = 1
        for f in range(fcount):
            #Get parameters of fth feature => (0,0) is mean && (1,0) is variance
            param = mv[:,f]
            mu    = param[0,0]
            sig2  = param[1,0]
            sig   = sig2 ** (0.5)

            #Calculate probability for each and every feature multiplication
            prob = prob * norm.pdf(xtemp[0,f], mu, sig)
            #prob = prob * ( (1 / (( 2 * np.pi )**0.5)) * (sig2**0.5)) * np.exp((-1) * ((xtemp[0,f] - mu)**2) / (2 * sig2))
        
        result_v[r,0] = prob
    return result_v

# Calculate F-Score and plot a graph Vs different epsilon
# mv stores all the mu and sigma2 in [2 X n] matrix
def calcFscore(X, Yval, mv, start_eps, divider, times):
    plt.figure()
    epsilon = start_eps
    bestF1 = 0
    besteps = 0
    for i in range(times):
        #Calculate probability for the input set
        Prob = calcProb(X, mv)

        #Convert P to probability
        Prob = 1*( Prob < epsilon )

        #true positives => Both output are 1
        tp = len(np.nonzero(((Prob == 1) & (Yval == 1)) == 1 )[0])

        #false positive => prob = 1 but yval = 0
        fp = len(np.nonzero(((Prob == 1) & (Yval == 0)) == 1 )[0])

        #false negative => prob = 0 but yval = 1
        fn = len(np.nonzero(((Prob == 0) & (Yval == 1)) == 1 )[0])

        #precision
        prec = 0
        if tp+fp > 0:
            prec = tp/(tp + fp)

        #recall
        rec = 0    
        if tp+fn > 0:
            rec = tp/(tp + fn)

        #F1 Score
        F1 = np.float(0)
        if prec + rec > 0:
            F1 = (2*prec*rec)/(prec+rec)  

        #Set as BestF1
        if F1 > bestF1:
            bestF1 = F1
            besteps = epsilon
            
        #Decrease epsilon
        print("At epsilon", epsilon, "- F1 Score:", F1)
        print("TP:",tp," - FP:",fp," - FN:",fn)
        plt.scatter(epsilon,F1,marker='x',color='blue')
        epsilon = epsilon / divider

    #Plot and print Best F1
    print("Best epsilon is:",besteps)    
    plt.xlabel('Epsilon divider')
    plt.ylabel('F1 Score')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)    
    plt.show()    
        

#Determize Z for plotting contour plot with respect to MV
def calcZ(u, v, mv):
    m_x = np.shape(u)[0]
    m_y = np.shape(v)[0]

    #Matrification
    u       = np.mat(u)
    v       = np.mat(v)

    #Number of features
    fcount  = np.shape(mv)[1]

    #Calculate Z
    xtemp   = np.mat(np.zeros(shape=(1,fcount)))
    z       = np.zeros(shape=(m_x,m_y))          #To store resultant Z for each combo of X1 and X2

    #Predict output for the whole grid array ( each point )
    #U and V are row vectors
    for i in range(0, np.shape(u)[1]):
        for j in range(0, np.shape(v)[1]):
            #Calculate probability for all points in grid
            xtemp[0,0] = u[0,i]
            xtemp[0,1] = v[0,j]
            prob = calcProb(xtemp, mv)

            #Set probability
            z[i,j] = prob

    #VERY IMPORTANT => Transpose contour output( Took nearly 2 hrs to figure this out )
    return z.T

################################# All functions End #################################

#Load MATLAB file directly into variables X and y
data = io.loadmat('ex8data1.mat')
y = data['yval']
X = data['X']

#Matrification of input data
X = np.mat(X)
y = np.mat(y)

#Plot input Data
if 1 == 2:
    #Plot input data
    plt.scatter(X[:,0],X[:,1],marker='x',color='blue',label='Inputs')
    plt.xlabel('Machine vibration')
    plt.ylabel('Voltage surge')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.show()

#Calculate Mean and variance for each parameter
print("Computing Mean & Variance parameters ...")
MV = calcMeanVar(X)

#Calculate Fscore
if 1 == 1:
    starteps = 0.00261792634794614
    divider = 1.000000001
    times = 100
    calcFscore(X, y, MV, starteps, divider, times)

#Plot computed decision boundary as per parameters MV
if ( 1 == 2 ):
    bestEPS = 0.00261792634794614 #As estimated from F1 score
    print("Plotting contour for predicted boundaries")
    #Prepare Contour plot grid
    u = np.arange(0, 25, 0.1)
    v = np.arange(0, 30, 0.1)
    z = calcZ(u, v, MV)

    #Plot positive negative examples along with computed Contour
    plt.figure()
    plt.scatter(X[:,0],X[:,1],marker='x',color='blue',label='Inputs')
    plt.xlabel('Machine vibration')
    plt.ylabel('Voltage surge')
    cont = plt.contour(u, v, z, levels=[bestEPS],legend="Boundaries", colors=['r'])
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.title('Anomaly ML model')
    plt.show()
