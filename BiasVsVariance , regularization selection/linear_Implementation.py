#This part below is the extension to the house price( single variable ) prediction exercise.
#Here you will apply ML analysis techniques like validation curve/ learning curve to diagnose Bias / Variance problems and
#also estimate the right model, regularization parameters for your model

#You will have the following objective
#1. Plot learning curve to measure Bias Vs Variance
#1. Model to be used( 1, 2, 3 ... ) 
#2. Regularization parameter estimation
import numpy             as np
import matplotlib.pyplot as plt
import scipy.optimize    as opt

################################# All functions Begin ###############################
#Function computeError( Return statement has to be right aligned to def statement)
def computeError(theta, X, y):
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
    J = np.sum(np.power((X*theta - y),2),axis=0)/(2*m)
    return J

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
    J = np.sum(np.power((X*theta - y),2),axis=0)/(2*m) \
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
    grad = np.sum(np.multiply(X*theta - y,X),axis=0) / m
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
    theta_min = opt.minimize(computeCost,x0=init_theta,jac=computeGradient,args=(X,y,lam),method='BFGS')
    return theta_min

#Convert 1 dimension feature to multiple features by power factorization; Also add ones column
def mapFeature(xval,degree):
    #Calculate number of columns which will be created
    noc     = degree
    m       = np.shape(xval)[0]
    
    #Convert to matrices; extract first and second column
    xval    = np.mat(xval)
    x1      = np.mat(xval[:,0])

    #Initialize empty matrix( row , 27 )
    xout    = np.mat(np.zeros(shape=(m,noc)))
    col     = 0

    #start conversion=> X1, X1.^2, X1.^3 ....
    for i in range(degree):
        #Element wise multiplication => Dot product
        xout[:,col] = np.power(x1,i+1)
        col = col + 1

    #Insert ones as first column into X
    xout = np.insert(xout,0,values=1,axis=1)

    #Return featurized matrix
    return xout

#Price predictor
def myfit(xval,theta):
    return np.mat(xval)*np.mat(theta).T

# Function to plot Train vs CV error on different lambdas( Validation curve )
def regSelectionPlotter(xt,yt,xc,yc,lv_lamb,lc_model,lv_trial,lv_multiplier):
    print("Running regularization estimation")
    #Number of training set examples
    mt      = np.shape(xt)[0]

    #Empty matrix to store error_train and error_cv values
    error_train  = np.zeros(shape=(lv_trial,2))
    error_cv     = np.zeros(shape=(lv_trial,2))

    #Featurize training set
    Xtraintemp  = mapFeature(xt, lc_model)
    Xcvtemp     = mapFeature(xc, lc_model)

    #For different power sizes compute CV and Train set error
    for i in range(lv_trial):
        lv_lamb = lv_lamb * lv_multiplier
        print("Computing errors for regularization:", lv_lamb)
        colsX       = np.shape(Xtraintemp)[1]
        init_theta  = np.zeros((1,colsX)).reshape(-1)

        #Calculate best theta
        theta_min   = calcMinGrad(init_theta, Xtraintemp, yt, lv_lamb)
        print(theta_min.message)

        #Compute error on Train set using the above min Theta
        error_train[i,0] = lv_lamb
        error_train[i,1] = computeError(theta_min.x, Xtraintemp, yt)

        #Compute error on CV set using the above min Theta
        costcv      = computeError(theta_min.x, Xcvtemp, yc)
        error_cv[i,0] = lv_lamb
        error_cv[i,1] = costcv

    #Show plots for each power
    plt.plot(error_train[:,0],error_train[:,1],color='blue',label='Train set error')
    plt.plot(error_cv[:,0],error_cv[:,1],color='green',label='CV set error')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.xlabel('Regularization parameter')
    plt.show()

# Function to plot Train vs CV error on different model( Validation curve )
def modelSelectionPlotter(xt,yt,xc,yc,lc_lamb,model_size):
    print("Running model estimation")
    #Number of training set examples
    mt      = np.shape(xt)[0]
    model_size = model_size - 1 

    #Empty matrix to store error_train and error_cv values
    error_train  = np.zeros(shape=(model_size,2))
    error_cv     = np.zeros(shape=(model_size,2))

    #For different power sizes compute CV and Train set error
    for i in range(model_size):
        power       = i + 1
        print("Computing errors for model with power:", power)
        Xtraintemp  = mapFeature(xt, power)
        colsX       = np.shape(Xtraintemp)[1]
        init_theta  = np.zeros((1,colsX)).reshape(-1)

        #Calculate best theta
        theta_min   = calcMinGrad(init_theta, Xtraintemp, yt, lc_lamb)
        print(theta_min.message)

        #Compute error on Train set using the above min Theta
        error_train[i,0] = power
        error_train[i,1] = computeError(theta_min.x, Xtraintemp, yt)    

        #Compute error on CV set using the above min Theta
        Xcvtemp     = mapFeature(xc, power)
        costcv      = computeError(theta_min.x, Xcvtemp, yc)
        error_cv[i,0] = power
        error_cv[i,1] = costcv

    #Show plots for each power
    plt.plot(error_train[:,0],error_train[:,1],color='blue',label='Train set error')
    plt.plot(error_cv[:,0],error_cv[:,1],color='green',label='CV set error')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.xlabel('Degree of polynomial')
    plt.show()

#Learning curve plotter
def learningCurve(xt,yt,xc,yc,lc_lamb):
    print("Running learning curve plotter")
    #Number of training set examples
    mt      = np.shape(xt)[0]

    #Empty matrix to store error_train and error_cv values
    error_train  = np.zeros(shape=(mt,2))
    error_cv     = np.zeros(shape=(mt,2))

    for i in range(mt):
        xt_temp     = xt[0:i+1,:]
        yt_temp     = yt[0:i+1,:]
        xcv_temp    = xc
        
        #Insert ones as first column into X
        xt_temp = np.insert(xt_temp,0,values=1,axis=1)

        #Initialize variables
        colsX       = np.shape(xt_temp)[1]
        init_theta  = np.zeros((1,colsX)).reshape(-1)

        #Calculate best theta and store in error_train
        theta_min        = calcMinGrad(init_theta, xt_temp, yt_temp, lc_lamb)

        #Compute error on Train set using the above min Theta
        error_train[i,0] = i
        error_train[i,1] = computeError(theta_min.x, xt_temp, yt_temp)          

        #Calculate error on full XCV set and store in error_cv
        xcv_temp        = np.insert(xcv_temp,0,values=1,axis=1)
        costcv          = computeError(theta_min.x, xcv_temp, yc)
        error_cv[i,0]   = i
        error_cv[i,1]   = costcv

    plt.plot(error_train[:,1],color='blue',label='Train set error')
    plt.plot(error_cv[:,1],color='green',label='CV set error')
    plt.legend(loc='upper right', numpoints=1, ncol=1, fontsize=8)
    plt.xlabel('Training example size')
    plt.show()
    
################################# All functions End #################################

#Load text file data in "data" as per column size in "usecols"
data = np.loadtxt('ex1data1.txt',delimiter=',',usecols=(0,1))

#Get size of loaded Data => [97,2]
print("Load size")
print(np.shape(data))

#Random shuffle training input data
data = np.mat(data)
np.random.shuffle(data)

#Divide data into Train(59) , CV(19) and Test(19) sets
dataTrain   = np.mat(data[0:59,:])
dataCV      = np.mat(data[59:78,:])
dataTest    = np.mat(data[78:,:])

#Notation np.mat is used to convert array "(118,2)" to matrix "(118,2)"
X       = data[:,0]
y       = data[:,1]
Xtrain  = dataTrain[:,0]
ytrain  = dataTrain[:,1]
Xcv     = dataCV[:,0]
ycv     = dataCV[:,1]
Xtest   = dataTest[:,0]
ytest   = dataTest[:,1]

#Scatter Plot X second column data Marker "X" and Color "Red = r"
if ( 1 == 2 ):
    plt.scatter(X[:,0],y,marker='x',color='r')
    plt.xlabel('Population of City: 10,000s')
    plt.ylabel('Profit: $10,000s')
    plt.show()

############## Learning curve plot: Bias Vs Variance ##############
# Analyze whether there is a bias or variance
# Important notes:
# 1. Learning curve should always be run for Lambda = 0(error is different than cost) and no extra featurization
# 2. Theta should be estimated on Training set for different train sizes
# 3. Estimated theta should be used to compute error(J) on Train and CV set
# 4. Training set error will be computed on training subset train(0:m,:)
# 5. However, CV set error will be computed on full CV set
# 6. Always run it multiple times for better analysis

if 1 == 2:
    learningCurve(Xtrain,ytrain,Xcv,ycv,0)
    #Result => High bias plot

###################### End learning curve #########################

################ Model selection: Validation curve ################
# Analyze using plots, which model(power) to chose as it is a high bias situation
# Important notes:
# 1. Since the current model has high bias we need to featurize X
# 2. Use validation curve logic
# 3. Estimated theta should be used to compute error(J) on Train and CV set
# 4. We will run it for 15 power models
# 5. Always run it multiple times for better analysis

if 1 == 2:
    modelSelectionPlotter(Xtrain,ytrain,Xcv,ycv,0,6)
    #Result => Chose model with power = 4

###################### End model selection #########################

###### Regularization parameter selection: Validation curve ########
# Analyze using plots, which value of regularization param to chose
# Important notes:
# 1. Since the current model has high bias we need to featurize X
# 2. Use validation curve logic
# 3. Estimated theta should be used to compute error(J) on Train and CV set
# 4. We will run it on multiples of 0.0001 = 0.0001 * 5(multiplier)

if 1 == 2:
    #Initialize starting lambda & model to use
    start_lamb  = 0.00001
    optim_model = 4         #Estimated from model selection
    trials      = 10
    multiplier  = 3

    #Run selection plotter
    regSelectionPlotter(Xtrain,ytrain,Xcv,ycv,start_lamb,optim_model,trials,multiplier)
    #Result => Regularization parameter should be zero or minimum

###################### End regularization selection #########################     

#Featurize input X with model degree 4 and Add ones column too
X = mapFeature(X,4)
colsX = np.shape(X)[1]

#Run OPTIMIZE to calculate best theta
print("Training model as per estimated parameters")
init_theta = np.zeros((1,colsX)).reshape(-1)
init_lamb  = 0.00001 #Estimated from regularization selection
theta_min  = calcMinGrad(init_theta, X, y, init_lamb)
print("Minimum Theta found")

#Plot output
if ( 1 == 1 ):
    plt.figure(figsize=(10,6))
    plt.plot(X[:,1],y[:,0],'rx',markersize=10,label='Training Data')
    plt.plot(X[:,1],myfit(X,theta_min.x),'gx',label = 'Predictor boundary')
    plt.grid(True) #Always plot.grid true!
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
    plt.legend()
    plt.show()
