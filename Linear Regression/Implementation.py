#In this part, you will implement linear regression with multiple variables to
#predict the prices of houses. Suppose you are selling your house and you
#want to know what a good market price would be. One way to do this is to
#collect information on recent houses sold and make a model of housing prices.

#The ex1data2.txt contains a training set of housing prices
import numpy             as np
import matplotlib.pyplot as plt
import scipy.optimize    as opt

################################# All functions Begin ###############################

#Function computeCost( Return statement has to be right aligned to def statement)
def computeCost(theta, X, y):
    # Training set size and initialize J
    m = y.size
    J = 0
    #print("Compute cost")
    colsX = np.shape(X)[1]
    theta = np.reshape(theta,(colsX,1))

    # Convert to matrices
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)

    # Calculate Cost with Theta ( Axis = 0, sum for values in a column )
    J = np.sum(np.power((X*theta - y),2),axis=0)/(2*m)
    print('Cost iteration:', J)
    return J

#Function computeGradient derivative for Theta0 and Theta1 ... => Unregularized
def computeGradient(theta, X, y):
    # Training set size
    m = y.size
    #print("Gradient Run")
    colsX = np.shape(X)[1]
    theta = np.reshape(theta,(colsX,1))

    # Convert to matrices
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)

    # For matrix, '*' means matrix multiplication, and the multiply() function is used for element-wise multiplication.
    grad = np.sum(np.multiply(X*theta - y,X),axis=0) / m

    # Convert matrix grad to Array
    grad = np.asarray(grad).reshape(-1)
    #print(grad)
    return grad

#Function calcMinGrad => Run FMINCG to compute minimizing theta
def calcMinGrad(init_theta, X, y):
    #theta_min = opt.fmin_cg(computeCost,x0=init_theta,fprime=computeGradient,args=(X,y),maxiter=1500)
    theta_min = opt.minimize(computeCost,x0=init_theta,jac=computeGradient,args=(X,y),method='BFGS')
    return theta_min

#Price predictor
def predictPrice(xval,theta,mu,sigma):
    #Normalize input xval to be predicted
    x_norm  = np.divide((xval - mu),sigma)    
    colsX   = np.shape(xval)[1]
    theta   = np.reshape(theta,(colsX+1,1))

    #Matrification
    x_norm  = np.mat(x_norm)
    theta   = np.mat(theta)

    #Insert first column
    x_norm = np.insert(x_norm,0,values=1,axis=1)
    return (x_norm*theta)

#Feature normalize => Normalize features on mean and sigma
def normalizeFeatures(xval):  
    #Matrification
    xval    = np.mat(xval)

    #MU and Sigma calculation
    mu      = xval.mean(0)
    sigma   = xval.std(0)
    mu      = np.mat(mu)
    sigma   = np.mat(sigma)

    #Normalization
    x_norm  = np.divide((xval - mu),sigma)
    return x_norm, mu, sigma  
################################# All functions End #################################

#Load text file data in "data"
data = np.loadtxt('ex1data2.txt',delimiter=',',usecols=(0,1,2))

#Get size of loaded Data => [47,N]
print("Load size")
print(np.shape(data))

#Load data in X and Y as Numpy Array
#Notation np.mat is used to convert array "(47,)" to matrix "(47,N)"
X = np.mat(data[:,0:2])
y = np.mat(data[:,2]).T
m = X.size

#Normalize X
xnorm, mu, sigma = normalizeFeatures(X)
print("Normalization ... Mu, Sigma")
print(mu , sigma)

#Insert ones as first column into X
X = np.insert(xnorm,0,values=1,axis=1)
print("Size of X after adding ones")
print(np.shape(X))
colsX = np.shape(X)[1]

#Initialize Theta and Compute initial cost with Theta = 1,1
if ( 1 == 2 ):
    theta = np.ones((1,colsX))
    print("Cost at Theta = [1,1]")
    print(computeCost(theta, X, y))

#Run GradientDescent to calculate theta
init_theta = np.zeros((1,colsX)).reshape(-1) #np.zeros((1,colsX)).reshape(-1)
theta_min = calcMinGrad(init_theta, X, y)
print("Minimum found Theta")
print(theta_min.x)

#Predict price for 1650 sq-ft, 3BR house
pred = [[1650, 3]]
print('Prediction price for 1650 sq-ft, 3BR house')
print(predictPrice(pred,theta_min.x,mu,sigma))
