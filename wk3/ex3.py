#Implementation of Coursera ML - Ex3 Logistic Regression character recog by Hasan Rafiq 15-Aug-2016
import numpy             as np
import matplotlib.pyplot as plt
import scipy.optimize    as opt
import scipy.io          as io   

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
    theta_min = opt.minimize(computeCost,x0=init_theta,jac=computeGradient,args=(X,y,lam),method='BFGS',options={'maxiter': 100})
    return theta_min

#Implementation to train model for each number
def trainModel(X,y):
    #initialize theta
    theta = np.mat(np.zeros(shape=(np.shape(X)[1],10)))

    #Training model
    for i in range(10):
        if i == 0:
            #0 is mapped with 10
            numindex = np.nonzero(y[:,0] == 10)
        else:
            #Get indices from y for each matching number; Initialize others
            numindex = np.nonzero(y[:,0] == i)
            
        print("... Training model for number", i)        
        numy     = np.mat(np.zeros(shape=(np.shape(X)[0],1)))

        #Create Y = 1, for particular indexes => Only these are positive classified
        numy[numindex[0],:] = 1

        #Run OPTIMIZE to calculate best theta for each number
        cols = np.shape(X)[1]
        init_theta  = np.zeros((1,cols)).reshape(-1)
        init_lambda = 0.1
        theta_min = calcMinGrad(init_theta, X, numy, init_lambda)

        #Store theta
        theta[:,i] = np.reshape(theta_min.x, (np.shape(theta)[0],1))
        print(theta_min.message)
        
    return theta


#Show images of 100 random characters from data set
def showRandomChar(m, X, img):
    print("Picturizing",img,"random number(s) ...")
    imgax   = int(img**0.5) #Images per each row of the square shaped axis
    
    #Random shuffle training index numbers
    randm = np.random.permutation(m)

    #Pick up top 100 random input data sets
    Xrand = X[randm[0:img],:]

    #Side of each image( square shape ) => Sqrt of pixel per image
    side = np.shape(Xrand)[1]
    side = int(side**0.5)

    #Prepare blank displays of ones => 10 by 10 images
    xsize = imgax*(side + 1) - 1    #padding of 1px between two images
    ysize = xsize                   #Square box hence lengths x = y
    canvas = np.mat(np.ones(shape=(xsize,ysize)))

    #Start replacing ones with pixel data for all 100 images
    xs  = int(0)
    ys  = int(0)
    for i in range(img):
        #Reshape Xrand for each image into 20x20 canvas from (i,400)       
        canvas[xs:(xs + side),ys:(ys + side)] = np.reshape(Xrand[i,:],(side,side)).T
        
        #Calculate values of X start and Y start for next images
        if ( (i - (imgax-1)) % imgax == 0 ):
            xs = 0
            ys = ys + side + 1
        else:    
            xs = xs + side + 1

    #Show images' matrix 
    fig, (ax1) = plt.subplots()
    ax1.imshow(canvas, extent=[0, xsize, 0, ysize], cmap=plt.gray())
    plt.show()
    return Xrand
    
################################# All functions End #################################

#Load MATLAB file directly into variables X and y
data = io.loadmat('ex3data1.mat')
y = data['y']
X = data['X']

#Matrification of input data
X = np.mat(X)
y = np.mat(y)
Xorig = X

#Get size of loaded Data => [100,2]
print("X Loaded size")
print(np.shape(X))
m = np.shape(X)[0]

#Picturize 100 random characters from data set
if 1 == 2:
    showRandomChar(m, X, 100)

#Insert ones as first column into X
X = np.insert(X,0,values=1,axis=1)
print("Size of X after adding ones")
print(np.shape(X))

#Train model for each number 0 to 9
theta_min = trainModel(X,y)

#Classify for random 10 numbers from set
for i in range(10):
    #Pick and show random digit
    xrandom = showRandomChar(m, Xorig, 1)
    
    #Insert ones as first column into xrandom
    xrandom = np.insert(xrandom,0,values=1,axis=1)
    output  = xrandom*theta_min
    digit   = np.argmax(output)
    print("Shown digit was:", digit)
