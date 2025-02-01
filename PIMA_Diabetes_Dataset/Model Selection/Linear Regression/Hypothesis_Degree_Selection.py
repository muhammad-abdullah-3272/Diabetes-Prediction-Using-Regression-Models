#==============================================================================
#-----------------------------------------------------------------------------
# LAMBDA SELECTION FOR DIABETES USING LINEAR REGRESSION MODEL
#-----------------------------------------------------------------------------
#==============================================================================


#------------------------------------------------------------------------------
## Importing Necessary Libraries
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(10)

plt.rcParams['figure.figsize'] = (10.0, 5.0)


#==============================================================================
# Reading the Dataset 
#==============================================================================
data = pd.read_csv('diabetes.csv', encoding='unicode_escape')
print("Data Size: ", data.shape)
data.head()

#==============================================================================
# Feature Scaling
#==============================================================================
def Scale(data):
    '''
    This fucntion performs data scaling from -1 to 1 using min-max critera.
    The model takes the following arguments:
    
    Data (numpy array): Input dataset
    
    returns:
    Scaled Data (numpy array)
    '''
    
    dataScale = 2*((data - data.min()) / (data.max() - data.min())) - 1    # Feature Scaling from -1 to 1
    dataScale['Outcome'] = data['Outcome']                                 # Not applying Scaling on Y
    return dataScale

data = Scale(data)                                # Calling the Feature Scaling Function

#==============================================================================
# Dataset Splitting
#==============================================================================
# Splitting the Dataset into Train set(60%), Cross Validation set(20%) and Test set(20%)
train, val, test = np.split(data.sample(frac=1), [int(0.7 * len(data)), int(0.85 * len(data))])
print("Training Set: ", train.shape)
print("Validation Set: ",val.shape)
print("Test Set: ",test.shape)


#==============================================================================
# Features and Labels Extraction
#==============================================================================
X_data = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
          "DiabetesPedigreeFunction", "Age"]                      # Extracting Features
Y_data = ["Outcome"]                                              # Extracting Labels   

X_train = train[X_data]                                           # Assigning Features to X_train               
Y_train = train[Y_data]                                           # Assigning Features to Y_train

X_val = val[X_data]                                               # Assigning Features to X_val
Y_val = val[Y_data]                                               # Assigning Features to Y_val

X_test = test[X_data]                                             # Assigning Features to X_test
Y_test = test[Y_data]                                             # Assigning Features to Y_test


X_train = X_train.values                                          # Extracting values from X_train
Y_train = Y_train.values                                          # Extracting values from Y_train

X_val = X_val.values                                              # Extracting values from X_val
Y_val = Y_val.values                                              # Extracting values from Y_val

X_test = X_test.values                                            # Extracting values from X_test
Y_test = Y_test.values                                            # Extracting values from Y_test


X_train = X_train.T                                               # Taking Transpose of the Training Features
Y_train = Y_train.reshape(1, X_train.shape[1])                    # Reshaping Training Labels 
print("Shape of X_train : ", X_train.shape)
print("Shape of Y_train : ", Y_train.shape)

X_val = X_val.T                                                   # Taking Transpose of the Cross Validation Features
Y_val = Y_val.reshape(1, X_val.shape[1])                          # Reshaping Cross Validation Labels 
print("Shape of X_val : ", X_val.shape)
print("Shape of Y_val : ", Y_val.shape)

X_test = X_test.T                                                 # Taking Transpose of the Testing Features
Y_test = Y_test.reshape(1, X_test.shape[1])                       # Reshaping Testing Labels 
print("Shape of X_test : ", X_test.shape)
print("Shape of Y_test : ", Y_test.shape)


#==============================================================================
# ReLU Activation
#==============================================================================
def ReLU(x):
    '''
    This fucntion creates a Sigmoid Funtion on given features and features.
    The model takes the following arguments:
    
    X (numpy array): Input data
    
    returns:
    Sigmoid Function
    '''
    return np.maximum(0, x)


#==============================================================================
# Logistic Regression and Linear Regression Model
#==============================================================================
def Model(X, Y, X_val, Y_val, learning_rate, iterations, lamda, verbose = True):
    '''
    This fucntion creates a model and trains it on the supplied dataset using the
    provided learning rate and regularization for a given number of iterations.
    The model takes the following arguments:
    
    X (numpy array): Input dataset
    Y (numpy array): Actual labels
    learning_rate (float): 0.001
    iterations (int): 30000
    lamda (float): [0, 0.01, 0.1, 1, 10]
    
    returns:
    W (numpy array): The weights of the model
    B (numpy array): The Bias value of the model
    cost_list (list): list of all the cost values for the given iterations
    '''
    
    m = X.shape[1]                                           # Number of Training Examples
    n = X.shape[0]                                           # Number of Training Features 
    
    m_val = X_val.shape[1]                                   # Number of Cross Validation Examples
    
    W = np.zeros((n,1))                                      # Intiallizing wieghts to zero
    B = 0                                                    # Initiallizing bias to zero
    
    train_cost_list = []                                     # Initialling Cost Function List as empty
    val_cost_list = []                                       # Initialling Validation Cost Function List as empty
    
    for i in range(iterations):
        
        Z = np.dot(W.T, X) + B                               # Function for Training Set 
        h = ReLU(Z)                                          # Implementing Sigmoid Funtion on the Hypothesis Model
        
        
        Z_val = np.dot(W.T, X_val) + B                       # Function for Validation Set
        h_val = ReLU(Z_val)                                  # Implementing Sigmoid Funtion on the Hypothesis Model
        
        
        # Mean Squared Error
        cost = 1/(2*m) * (np.sum(h - Y)**2) + ((lamda / (2 * m)) * np.sum(W**2))
        
        # Train Mean Squared Error
        cost_train = 1/(2*m) * (np.sum(h - Y)**2) 
           
        # Valodation Mean Squared Error
        cost_val = 1/(2*m_val) * (np.sum(h_val - Y_val)**2)
        
        # Gradient Descent
        dW = (1 / m) * np.dot(h - Y, X.T)
        dB = (1 / m) * np.sum(h - Y)
        
        # Updating Weights and Bias Parameters
        W = W - learning_rate * (dW.T + ((lamda / m) * W))    # Applying Regularization
        B = B - learning_rate * dB
        
        # Keeping track of our cost function values
        train_cost_list.append(cost_train)                    # Stacking all train costs in a list
        val_cost_list.append(cost_val)                        # Stacking all validation costs in a list

        
        if(i%(iterations/10) == 0):
            print("Train cost after ", i , "iteration is : ", cost_train)
            print("Val cost after ", i , "iteration is : ", cost_val)
            
        if verbose:
            if(i%(iterations/10) == 0):
                print("cost after ", i , "iteration is : ", cost_train)
                h[h>0.5] = 1
                h[h<=0.5] = 0
                print("Accuracy:", ((h == Y).sum())/Y.shape[1])
                print("Weight afters", i , "iteration is : ", W)
        
    return W, B, train_cost_list, val_cost_list



#==============================================================================
# Polynomial Degree Selection
#==============================================================================
iterations = 10000                                              # Number of times the Model will run to optimize parameters
learning_rate = 0.1                                             # Alpha is the step frequency for Gradient Descent
lamdas = [0, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.05, 0.1, 0.5, 1]      # Regularization Parameters
polynomials = [1, 2, 4, 8, 10, 12]                              # Degree of Polynomials

minCostArray = []                                               
minValArray = []                                               

for poly in polynomials:
    train_X = X_train.copy()                                    # Make a copy of Training Features and Assign it to the variable
    val_X = X_val.copy()                                        # Make a copy of Validation Features and Assign it to the variable 
        
    for i in range(1, poly):
        train_X = np.append(train_X, X_train**poly, axis = 0)   # Designing multiple Degree Polynomial Features for Training Set   
        val_X = np.append(val_X, X_val**poly, axis = 0)         # Designing 2nd Degree Polynomial Features for Validation Set
    
    for reg_param in lamdas:
        print(f"Polynomial {poly} and Lambda {reg_param}")
        W, B, cost_list, val_cost_list = Model(train_X, Y_train, val_X, Y_val, learning_rate = learning_rate, iterations = iterations, lamda = reg_param, verbose = False)
    
        print('Min Train Cost Error = ' + str(np.min(cost_list)))            # Determining Minimum Training Cost Error
        print('Min Validation Error = ' + str(np.min(val_cost_list)))        # Determining Minimum Validation Error
        
        
        if reg_param == 0:
            minCost = np.min(cost_list)                          # Determine min cost error from cost list
            minCostArray.append(minCost)                         # Append all min cost errors
            minValCost = np.min(val_cost_list)                   # Determine min val cost error from val cost list
            minValArray.append(minValCost)                       # Append all min cost errors
            break
            
plt.figure()
plt.plot(polynomials, minCostArray)
plt.plot(polynomials, minValArray)
plt.title("Training Cost vs Validation Cost")
plt.xlabel('Degree of Polynomials (d)')
plt.ylabel('Cost Error')
plt.legend(["Training Cost", "Validation Error"], loc ="upper right")
plt.show()