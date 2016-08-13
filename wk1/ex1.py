#Implementation of Coursera ML - Ex1 by Hasan Rafiq 31-Jul-2016
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
    #print('Cost:', J)
    return J

#Function computeGradient derivative for Theta0 and Theta1 => Unregularized
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
def myfit(xval,theta):
    return theta[0] + theta[1]*xval    
################################# All functions End #################################

#Load text file data in "data"
data = np.loadtxt('ex1data1.txt',delimiter=',',usecols=(0,1))

#Get size of loaded Data => [97,2]
print("Load size")
print(np.shape(data))

#Load data in X and Y as Numpy Array
#Notation np.mat is used to convert array "(97,)" to matrix "(97,1)"
X = np.mat(data[:,0]).T
y = np.mat(data[:,1]).T
m = X.size

#Insert ones as first column into X
X = np.insert(X,0,values=1,axis=1)
print("Size of X after adding ones")
print(np.shape(X))
colsX = np.shape(X)[1]

#Scatter Plot X second column data Marker "X" and Color "Red = r"
if ( 1 == 2 ):
    plt.scatter(X[:,1],y,marker='x',color='r')
    plt.xlabel('Population of City: 10,000s')
    plt.ylabel('Profit: $10,000s')
    plt.show()

#Initialize Theta and Compute initial cost with Theta = 1,1
if ( 1 == 2 ):
    theta = np.ones((1,colsX))
    print("Cost at Theta = [1,1]")
    computeCost(theta, X, y)

#Run GradientDescent to calculate theta
init_theta = np.zeros((1,colsX)).reshape(-1) #[ 0.08991601,  1.00598962] #np.zeros((1,2)).reshape(-1)
theta_min = calcMinGrad(init_theta, X, y)
print("Minimum found Theta")
print(theta_min.x)

#Plot output
if ( 1 == 2 ):
    plt.figure(figsize=(10,6))
    plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
    plt.plot(X[:,1],myfit(X[:,1],theta_min.x),'b-',label = 'Hypothesis: h(x) = %0.2f + %0.2fx'%(theta_min.x[0],theta_min.x[1]))
    plt.grid(True) #Always plot.grid true!
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.legend()
    plt.show()
