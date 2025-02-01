#==============================================================================
#-----------------------------------------------------------------------------
# CLASSIFICATION OF DIABETES USING LINEAR REGRESSION MODEL
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
def Model(X, Y, X_val, Y_val, learning_rate, iterations, lamda):
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
            print("val cost after ", i , "iteration is : ", cost_val)
            
        
    return W, B, train_cost_list, val_cost_list


iterations = 5000                                             # Number of times the Model will run to optimize parameters
learning_rate = 0.01                                            # Alpha is the step frequency for Gradient Descent
lamda = 0.01                                                      # Regularization Parameter
       

## Training the Selected Model on Training Set and retrieving optimized Weights and Bias, also training and validation cost functions 
W, B, train_cost_list, val_cost_list = Model(X_train, Y_train, X_val, Y_val, learning_rate = learning_rate, iterations = iterations, lamda = lamda)


print('Min Train Cost Error = ' + str(np.min(train_cost_list)))      # Determining Minimum Training Cost Error
print('Min Validation Error = ' + str(np.min(val_cost_list)))        # Determining Minimum Validation Error


plt.figure()
plt.title("Training Cost vs Validation Cost")
plt.plot(train_cost_list, label = "Training Cost")
plt.plot(val_cost_list, label = "Validation Cost")
plt.xlabel('Iterations')
plt.ylabel('Cost Error')
plt.legend()
plt.show()



def Accuracy(X, Y, W, B):
    '''
    This fucntion tests the trained model that is selected on the basis of minimum cost function. 
    The model will have have its minimized wieght parameters and optimum regularization parameter. 
    The model takes the following arguments:
    
    X (numpy array): Input dataset
    Y (numpy array): Actual labels
    W (numpy array): Minimized Weights
    B (int): Bias
    
    returns:
    Model's Accuracy (float)
    '''
    m = X.shape[1]                                                 # Number of Dataset Examples

    Z = np.dot(W.T, X) + B                                         # Calculation Z
    h = ReLU(Z)                                                    # Linear Regression Model
    
    loss = 1/(2*m) * (np.sum(h - Y)**2)                            # Mean Error for Corresponding Dataset
    
    h[h>=0.5] = 1                                                  # Thresholding where if h >= 0.5, then h = 1
    h[h<0.5] = 0                                                   # Thresholding where if h < 0, then h = 0
    h = np.array(h, dtype = 'int64')
        
    precision = np.std(Y) / 250                                    
    comparisons = np.absolute(h - Y) < precision    # Calculating Accuracy of Model on Corresponding Dataset
    acc = np.mean(comparisons) * 100
    return acc, loss
    
    
## Testing the Training Dataset on Trained Model and Determining Accuracy 
accuracy, loss = Accuracy(X_train, Y_train, W, B)
print("Train Accuracy: ", round(accuracy, 2), "%. Loss: ", loss)

## Testing the Validation Dataset on Trained Model and Determining Accuracy
accuracy, loss = Accuracy(X_val, Y_val, W, B)
print("Validation Accuracy: ", round(accuracy, 2), "%. Loss: ", loss)

## Testing the Test Dataset on Trained Model and Determining Accuracy
accuracy, loss = Accuracy(X_test, Y_test, W, B)
print("Test Accuracy: ", round(accuracy, 2), "%. Loss: ", loss)



