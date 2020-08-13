import numpy as np
import matplotlib.pyplot as plt

class bens_lm:


    def __init__(self, x_matrix, y, alpha=0.5, iterations=100):
        ''' params matrix x_matrix: input matrix where each row is a single sample and each column is a single attribute
            params 1d matrix y: 1d matrix
            params float alpha: learning rate
            params int iterations: iterations
        '''

        num_of_features = x_matrix.shape[1]
        num_of_samples = x_matrix.shape[0]
        
        #Set matrix of thetas
        thetas = np.ones((num_of_features + 1,1))
        
        #Adding column of 1's for theta_zero
        x_matrix = np.hstack((np.ones((num_of_samples,1)), x_matrix))
        
        #Storing errors for plotting
        errors = []

        #Iterations begin
        for i in range(iterations):

            #Our predictions, as a 1d array
            predicted = np.matmul(x_matrix, thetas)
            
            #Updating each of our thetas (vectorized)
            absolute_difference = predicted - y
            absolute_difference = absolute_difference.T

            loss_matrix = alpha * np.matmul(absolute_difference, x_matrix) / num_of_samples
            thetas -= loss_matrix.T
            
            #We need to add our sum squared error into our errors list - for plotting later
            error = np.sum(absolute_difference**2)
            errors.append(error)

        self.thetas = thetas
        self.errors = errors
        self.y = y
        self.final_error = error

    def get_thetas(self):
        ''' 
        Returns the coefficients as a (n + 1) x 1 numpy array 
        (Where n is number of features)
        '''
        return self.thetas


    def plot_errors(self):
        ''' Returns a plot of the sum squared errors versus the number of iterations '''
        plt.plot([i + 1 for i in range(len(self.errors))], self.errors)
        return plt.show()


    def predict(self, input_x):
        ''' 
        Returns predictions as a m x 1 numpy array
        (Where m is number of training samples)
        '''
        #As we have done before, must add column of 1's before matrix multiplication
        input_x = np.hstack((np.ones((input_x.shape[0], 1)), input_x))
        return np.matmul(input_x, self.thetas)


    def r_squared(self):
        '''
        Returns the r-squared value of our line of best fit
        '''
        variation_in_data = self.y - self.y.mean()
        variation_in_data = np.sum(variation_in_data**2)
        #Note that our error for our line has already been calculated from our last iteration
        return 1 - (self.final_error / variation_in_data)



## Testing
xs = np.random.rand(100,3)
x_matrixs = np.hstack((np.ones((100,1)), xs))
y = np.matmul(x_matrixs, np.array([[9],[11],[-6],[1]]))
x = bens_lm(xs, y, alpha=0.5, iterations=300)
print(x.thetas())
x.plot_errors()
print(x.predict(np.array([[2,3,7],[6,1,9]])))
print(x.r_squared())