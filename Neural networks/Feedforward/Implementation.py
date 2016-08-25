#In this part of the exercise, you will implement a neural network to rec-ognize handwritten digits using the same training set as before. The neural
#network will be able to represent complex models that form non-linear hy-potheses.

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

#Classification of digit => Feed-forward propagation
def predict(x, y, t1, t2):
    #Add ones to X and set as layer1_a => [1,401]
    layer1_a = np.insert(x,0,values=1,axis=1)

    #Compute Theta1*layer1_a => layer2_z [25,401]X[401,1] => [25,1]
    layer2_z = t1*layer1_a.T

    #Compute sigmoid(layer2_z) => layer2_a and add ones row => [26,1]
    layer2_a = sigmoid(layer2_z)
    layer2_a = np.insert(layer2_a,0,values=1,axis=0)

    #Compute Theta2'*layer2_a => layer3_z [10,26]x[26,1] => [10,1]
    print(np.shape(t2))
    layer3_z = t2*layer2_a

    #Compute sigmoid(layer2_z) => layer3_a => [10,1]
    layer3_a = sigmoid(layer3_z)

    #Find max in layer3_a
    digit    = np.argmax(layer3_a)
    if digit == 9:
        #Since 0 is mapped at 9th index
        digit = 0
    else:
        #Since 1 is mapped at 0th index and 9 is at 8th index
        digit = digit + 1
        
    print("Shown digit was:", digit)

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

#Load MATLAB data file directly into variables X and y
data    = io.loadmat('ex3data1.mat')
y       = data['y']
X       = data['X']

#Load MATLAB weights file directly into variables Theta1 and Theta2
dataw   = io.loadmat('ex3weights.mat')
Theta1  = dataw['Theta1']
Theta2  = dataw['Theta2']

#Matrification of input data
X       = np.mat(X)
y       = np.mat(y)
Theta1  = np.mat(Theta1)
Theta2  = np.mat(Theta2)
Xorig   = X

#Get size of loaded Data => [5000,400]
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

#Classify for random 10 numbers from set, using precalculated Theta1,2
for i in range(10):
    #Pick and show random digit
    xrandom = showRandomChar(m, Xorig, 1)
    
    #Predict characters as per existing Theta
    predict(xrandom,y,Theta1,Theta2)
