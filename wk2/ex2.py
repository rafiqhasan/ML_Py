#Implementation of Coursera ML - Ex2 Logistic Regression by Hasan Rafiq 07-Aug-2016
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
    J = np.sum(-1 * (np.multiply(np.log(sigmoid(X*theta)),y) + np.multiply(np.log(1 - sigmoid(X*theta)),1-y)),axis=0 )/ m
    return J

#Function computeGradient derivative for Theta => Unregularized
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
    grad = np.sum(np.multiply(sigmoid(X*theta) - y,X),axis=0) / m

    # Convert matrix grad to Array
    grad = np.asarray(grad).reshape(-1)
    return grad

#Function calcMinGrad => Run FMINCG to compute minimizing theta
def calcMinGrad(init_theta, X, y):
    theta_min = opt.minimize(computeCost,x0=init_theta,jac=computeGradient,args=(X,y),method='CG')
    return theta_min

#Boundary determiner for plotting decision boundary [ solve for X2 s.t; X0 = 0 , X1 = marks1 in X*Theta = 0 ]
def predictX2ForX1(xval,theta):
    x2 = (-1/(theta)[2])*((xval)[0]*(theta)[0] + (xval)[1]*(theta)[1])
    return x2

#Predict Admitted or Not admitted
def predict(xval,theta):
    #Reshape Theta
    theta = np.mat(theta)
    colsX = np.shape(xval)[0]
    theta = np.reshape(theta,(colsX,1))
    
    #Convert to matrices
    xval    = np.mat(xval)
    theta   = np.mat(theta)

    #Compute X*Theta
    g = xval*theta
    if ( g >= 0 ):
        return 1
    else:
        return 0
    
################################# All functions End #################################

#Load text file data in "data" as per column size in "usecols"
data = np.loadtxt('ex2data1.txt',delimiter=',',usecols=(0,1,2))

#Get size of loaded Data => [100,2]
print("Load size")
print(np.shape(data))

#Load data in X and Y as Numpy Array
#Notation np.mat is used to convert array "(100,2)" to matrix "(100,2)"
X = np.mat(data[:,0:2])
y = np.mat(data[:,2]).T
m = X.size

#Segrate positive and negative examples
#NONZERO operator in Python works similar to FIND in Octave
posindex = np.nonzero(y[:,0]==1)
negindex = np.nonzero(y[:,0]==0)
pos = X[posindex[0],:]
neg = X[negindex[0],:]

#Scatter Plot Positive and Negative examples
if ( 1 == 2 ):
    #Plot positive negative examples, w.r.t scores in both the exams
    plt.scatter(pos[:,0],pos[:,1],marker='+',color='black',label='Admitted')
    plt.scatter(neg[:,0],neg[:,1],marker='o',color='y',label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.show()

#Insert ones as first column into X
X = np.insert(X,0,values=1,axis=1)
print("Size of X after adding ones")
print(np.shape(X))
colsX = np.shape(X)[1]

#Initialize Theta and Compute initial cost with Theta = 0,0,0
if ( 1 == 2 ):
    theta = np.zeros((1,colsX))
    print("Cost at Initial theta")
    print(computeCost(theta, X, y))

#Run OPTIMIZE to calculate best theta
init_theta = np.zeros((1,colsX)).reshape(-1)
theta_min = calcMinGrad(init_theta, X, y)
print("Minimum found Theta")
print(theta_min)

#Plot computed decision boundary as per THETA
if ( 1 == 2 ):
    #Predict marks required in Exam2 for Exam1 = min(Exam1) and max(Exam1) to plot decision boundary( straight line )
    x2    = np.array([0,0])
    minx1 = np.amin(X[:,1])
    maxx1 = np.amax(X[:,1])
    x2[0] = predictX2ForX1(np.array([1,minx1,0]),theta_min.x)
    x2[1] = predictX2ForX1(np.array([1,maxx1,0]),theta_min.x)
    
    #Plot positive negative examples, w.r.t scores in both the exams 
    plt.scatter(pos[:,0],pos[:,1],marker='+',color='black',label='Admitted')
    plt.scatter(neg[:,0],neg[:,1],marker='o',color='y',label='Not Admitted')
    plt.plot([minx1,maxx1],x2,label='Decision')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.show()

#Predict for Score 45,85
prob = predict([1,45,85],theta_min.x)
print("Probability for admission on score 45,85 is: ",prob)
