#Implementation of Coursera ML - Ex2 Logistic Regression(Regularized) by Hasan Rafiq 13-Aug-2016
import numpy             as np
import matplotlib.pyplot as plt
import scipy.optimize    as opt

################################# All functions Begin ###############################
#Compute Sigmoid
def sigmoid(z):
    # Convert to matrices
    z = np.mat(z)

    # Sigmoid function
    s = np.divide(1,( 1 + np.exp(-1 * z)))
    return s

#Function computeCost( Return statement has to be right aligned to def statement) => Regularized
def computeCost(theta, X, y, lam):
    # Training set size and initialize J
    m = y.size
    J = 0
    
    #Reshape theta from any shape to Row vector
    colsX = np.shape(X)[1]
    theta = np.reshape(theta,(colsX,1))

    # Convert to matrices
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)

    # Calculate Cost with Theta ( Axis = 0, sum for values in a column )
    J = np.sum(-1 * (np.multiply(np.log(sigmoid(X*theta)),y) + np.multiply(np.log(1 - sigmoid(X*theta)),1-y)),axis=0 )/ m \
        +  \
        ( np.sum(np.power(theta,2),axis=0) - np.power(theta[0,0],2) ) * (lam / (2 * m)) #Subtract contribution of first element
    return J

#Function computeGradient derivative for Theta => Regularized
def computeGradient(theta, X, y, lam):
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
    grad = np.sum(np.multiply(sigmoid(X*theta) - y,X),axis=0) / m
    reg  = theta.T * lam / m

    # Apply regularization => First element of reg has to be reverted back
    temp     = grad[0,0]
    grad     = grad + reg
    grad[0,0] = temp
    
    # Convert matrix grad to Array
    grad = np.asarray(grad).reshape(-1)
    return grad

#Function calcMinGrad => Run FMINCG to compute minimizing theta
def calcMinGrad(init_theta, X, y, lam):
    theta_min = opt.minimize(computeCost,x0=init_theta,jac=computeGradient,args=(X,y,lam),method='CG')
    return theta_min

#Convert 2 dimension features to multiple features by power factorization; Also add ones column
def mapFeature(xval):
    degree = 6
    m = np.shape(xval)[0]
    
    #Convert to matrices; extract first and second column
    xval    = np.mat(xval)
    x1      = np.mat(xval[:,0])
    x2      = np.mat(xval[:,1])

    #Initialize empty matrix( row , 27 )
    xout    = np.mat(np.zeros(shape=(m,27)))
    col     = 0

    #start conversion=> X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    for i in range(1,degree+1):
        for j in range(0,i+1):
            #Element wise multiplication => Dot product
            xout[:,col] = np.multiply( np.power(x1,(i-j)) , np.power(x2,j) )
            col = col + 1

    #Insert ones as first column into X
    xout = np.insert(xout,0,values=1,axis=1)

    #Return featurized matrix
    return xout

#Determize Z for plotting contour plot with respect to MIN Theta
def calcZ(u, v, theta):
    m_z = np.shape(u)[0]

    #Matrification
    u       = np.mat(u)
    v       = np.mat(v)
    theta   = np.mat(theta).T

    #Calculate Z
    z       = np.zeros(shape=(m_z,m_z))         #To store resultant Z for each combo of X1 and X2
    ztempin = np.mat(np.zeros(shape=(1,2)))     #Temporary matrix to pass value to MapFeature

    #Predict output for dummy values
    ztempin[0,0] = -0.4
    ztempin[0,1] = 0.87
    trial = mapFeature(ztempin)*theta
    print("Trial prediction: ",trial)

    #U and V are row vectors
    for i in range(0, np.shape(u)[1]):
        for j in range(0, np.shape(v)[1]):
            #Calculate Z
            ztempin[0,0] = u[0,i]
            ztempin[0,1] = v[0,j]
            z[i,j] = mapFeature(ztempin)*theta

    #VERY IMPORTANT => Transpose contour output( Took nearly 2 hrs to figure this out )
    return z.T
    
################################# All functions End #################################

#Load text file data in "data" as per column size in "usecols"
data = np.loadtxt('ex2data2.txt',delimiter=',',usecols=(0,1,2))

#Get size of loaded Data => [118,2]
print("Load size")
print(np.shape(data))

#Load data in X and Y as Numpy Array
#Notation np.mat is used to convert array "(118,2)" to matrix "(118,2)"
X = np.mat(data[:,0:2])
y = np.mat(data[:,2]).T
m = np.shape(X)[0]

#Segrate positive and negative examples
#NONZERO operator in Python works similar to FIND in Octave
posindex = np.nonzero(y[:,0]==1)
negindex = np.nonzero(y[:,0]==0)
pos = X[posindex[0],:]
neg = X[negindex[0],:]

#Scatter Plot Positive and Negative examples
if ( 1 == 2 ):
    #Plot positive negative examples
    plt.scatter(pos[:,0],pos[:,1],marker='+',color='black',label='y = 1')
    plt.scatter(neg[:,0],neg[:,1],marker='o',color='y',label='y = 0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.show()

#Featurize input X with degree 6 and Add ones column too
X = mapFeature(X)
print("Size of X after featurizing: ",np.shape(X))
colsX = np.shape(X)[1]

#Run OPTIMIZE to calculate best theta
init_theta = np.zeros((1,colsX)).reshape(-1)
init_lamb  = 1
theta_min = calcMinGrad(init_theta, X, y, init_lamb)
print("Minimum found Theta")
print(theta_min)

#Plot computed decision boundary as per THETA
if ( 1 == 1 ):
    print("Plotting contour")
    #Prepare Contour plot grid
    u = np.arange(-1, 1.5, 0.05)
    v = np.arange(-1, 1.5, 0.05)
    z = calcZ(u, v, theta_min.x)
    
    #Plot positive negative examples along with computed Contour
    plt.figure()
    plt.scatter(pos[:,0],pos[:,1],marker='+',color='black',label='y = 1')
    plt.scatter(neg[:,0],neg[:,1],marker='o',color='y',label='y = 0')
    cont = plt.contour(u, v, z, levels=[0],legend="Decision boundary")
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)    
    plt.show()
