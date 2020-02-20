import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import autograd.numpy as np_auto  # Thinly-wrapped numpy
from autograd import grad
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.mse=[]
        self.coef_ = None
        self.all_coef=None

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        # importing the data x,Y

        x = X.values
        Y = y.values

        # Adding intercept term

        if self.fit_intercept:
            theta = np.ones(x.shape[1] + 1)
            x_pad1 = np.ones((x.shape[0], x.shape[1] + 1))
            x_pad1[:, 1:] = x
        else:
            theta = np.ones(x.shape[1])
            x_pad1 = x

        num_batch = np.floor(x.shape[0] / batch_size)
        num_batch = int(num_batch)
        last_batch = x.shape[0] % batch_size

        if not (last_batch == 0):
            resi_sample_size = batch_size - last_batch
            resi_sample_ind = np.random.choice(X.shape[0], resi_sample_size, replace=False)
            resi_sample = x[resi_sample_ind]
            resi_sample_Y = Y[resi_sample_ind]
            x_new = np.vstack((x, resi_sample))
            Y1 = np.hstack((Y, resi_sample_Y))
            num_batch += 1

        else:
            x_new = x
            Y1 = Y


        self.all_coef=np.ones(theta.shape[0])

        for i in range(n_iter):

            if lr_type == 'constant':
                lr = lr
            elif lr_type == 'inverse':
                lr *= 1 / i + 1

            # batch gradient descent
            for j in range(num_batch):
                x_pad = x_new[j * batch_size:(j + 1) * batch_size]
                Y_new = Y1[j * batch_size:(j + 1) * batch_size]

                if self.fit_intercept:
                    x_temp = np.ones((x_pad.shape[0], x_pad.shape[1] + 1))
                    x_temp[:, 1:] = x_pad
                    x_pad = x_temp

                theta_temp = theta

                for t in range(theta_temp.shape[0]):
                    y_pred = np.dot(x_pad, theta_temp)
                    mse = (np.sum((Y_new - y_pred) * (-1 * x_pad[:, t]))) / x_new.shape[0]
                    theta[t] = theta_temp[t] - lr * mse

            Y_pred1 = np.dot(x_pad1, theta)
            self.mse.append((np.sum((Y - Y_pred1) ** 2)) / Y.shape[0])
            self.all_coef = np.vstack((self.all_coef,theta))

        self.coef_ = theta
        return self.coef_,self.mse,self.all_coef

    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        x = X.values
        Y = y.values

        # Adding intercept term

        if self.fit_intercept:
            theta = np.ones(x.shape[1] + 1)
            x_pad1 = np.ones((x.shape[0], x.shape[1] + 1))
            x_pad1[:, 1:] = x
        else:
            theta = np.ones(x.shape[1])
            x_pad1 = x

        num_batch = np.floor(x.shape[0] / batch_size)
        num_batch = int(num_batch)
        last_batch = x.shape[0] % batch_size
        if not (last_batch == 0):
            resi_sample_size = batch_size - last_batch
            resi_sample_ind = np.random.choice(X.shape[0], resi_sample_size, replace=False)
            resi_sample = x[resi_sample_ind]
            resi_sample_Y = Y[resi_sample_ind]
            x_new = np.vstack((x, resi_sample))
            Y1 = np.hstack((Y, resi_sample_Y))
            num_batch += 1

        else:
            x_new = x
            Y1 = Y

        self.all_coef = np.ones(theta.shape[0])

        for i in range(n_iter):

            if lr_type == 'constant':
                lr = lr
            elif lr_type == 'inverse':
                lr *= 1 / i + 1

            # batch gradient descent
            for j in range(num_batch):
                x_pad = x_new[j * batch_size:(j + 1) * batch_size]
                Y_new = Y1[j * batch_size:(j + 1) * batch_size]

                if self.fit_intercept:
                    x_temp = np.ones((x_pad.shape[0], x_pad.shape[1] + 1))
                    x_temp[:, 1:] = x_pad
                    x_pad = x_temp

                theta_temp = theta

                a = np.dot(x_pad.T, x_pad)
                a = np.dot(a, theta)
                b = np.dot(x_pad.T, Y_new)
                theta = theta_temp - lr * a + lr * b

            Y_pred1 = np.dot(x_pad1, theta)
            self.mse.append((np.sum((Y - Y_pred1) ** 2)) / Y.shape[0])
            self.all_coef = np.vstack((self.all_coef, theta))

        self.coef_ = theta

        return self.coef_, self.mse, self.all_coef



    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''



        x = X.values
        Y = y.values

        # Adding intercept term

        if self.fit_intercept:
            theta = np.ones(x.shape[1] + 1)
            x_pad1 = np.ones((x.shape[0], x.shape[1] + 1))
            x_pad1[:, 1:] = x
        else:
            theta = np.ones(x.shape[1])
            x_pad1 = x

        num_batch = np.floor(x.shape[0] / batch_size)
        num_batch = int(num_batch)
        last_batch = x.shape[0] % batch_size
        if not (last_batch == 0):
            resi_sample_size = batch_size - last_batch
            resi_sample_ind = np.random.choice(X.shape[0], resi_sample_size, replace=False)
            resi_sample = x[resi_sample_ind]
            resi_sample_Y = Y[resi_sample_ind]
            x_new = np.vstack((x, resi_sample))
            Y1 = np.hstack((Y, resi_sample_Y))
            num_batch += 1

        else:
            x_new = x
            Y1 = Y

        self.all_coef = np.ones(theta.shape[0])

        for i in range(n_iter):

            if lr_type == 'constant':
                lr = lr
            elif lr_type == 'inverse':
                lr *= 1 / i + 1

            # batch gradient descent
            for j in range(num_batch):
                x_pad = x_new[j * batch_size:(j + 1) * batch_size]
                Y_new = Y1[j * batch_size:(j + 1) * batch_size]

                if self.fit_intercept:
                    x_temp = np.ones((x_pad.shape[0], x_pad.shape[1] + 1))
                    x_temp[:, 1:] = x_pad
                    x_pad = x_temp

                theta_temp = theta

                # mse = (np.sum((Y_new - y_pred) * (-1 * x_pad[:, t]))) / x_new.shape[0]
                grad_mse = grad(objective_fn)
                mse = grad_mse(theta_temp, x_pad, Y_new)

                theta = theta_temp - lr * mse

            Y_pred1 = np.dot(x_pad1, theta)
            self.mse.append((np.sum((Y - Y_pred1) ** 2)) / Y.shape[0])
            self.all_coef = np.vstack((self.all_coef, theta))

        self.coef_ = theta

        return self.coef_, self.mse, self.all_coef



    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''



        self.X=X
        self.y=y
        x=X.values
        if self.fit_intercept == True:
            x1=np.ones([x.shape[0],x.shape[1]+1])
            x1[:,1:]=x
        else:
            x1=x
        Y=y.values
        x2=x1.T
        x3=np.dot(x2,x1)
        x4=np.dot(x2,Y)
        x5=np.linalg.pinv(x3)
        self.coef_ = np.dot(x5,x4)
        return self.coef_







        #pass

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        x1 = X.values

        if self.fit_intercept:
            x=np.ones((x1.shape[0],x1.shape[1]+1))
            x[:,1:]=x1
        else:
            x=x1

        Y=np.dot(x,self.coef_)
        y=pd.Series(Y)
        return y


    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS

        """

        t_0_max=np.max(t_0)
        t_1_max=np.max(t_1)
        #w0 = np.linspace(-(all_coef[0, 0] + 0.25), (all_coef[0, 0] + 0.25), 500)
       # w1 = np.linspace(-(all_coef[0, 1] + 0.25), (all_coef[0, 1] + 0.25), 500)
        w0 = np.linspace(-(t_0_max +0.25), (t_0_max + 0.25), 500)
        w1 = np.linspace(-(t_1_max + 0.25), (t_1_max + 0.25), 500)
        x = X.values
        Y = y.values
        x1 = np.ones((x.shape[0], x.shape[1] + 1))
        x1[:, 1:] = x
        W0, W1 = np.meshgrid(w0, w1)
        mse = np.ones(W0.shape)
        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                theta = np.array([W0[i, j], W1[i, j]])
                y_pred = np.dot(x1, theta)
                mse[i, j] = (np.sum((Y - y_pred) ** 2)) * 1 / (Y.shape[0])

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.plot_surface(W0, W1, mse, cmap='viridis', edgecolor='none')
        # ax.colorbar
        # ax.scatter(all_coef[1:,0],all_coef[1:,1],mse,c='r')
        #ax.scatter(self.all_coef[1:, 0], self.all_coef[1:, 1], mse1, c='r')

        ax.scatter(t_0[1:], t_1[1:], self.mse, c='r')
        plt.xlabel('theta_0')
        plt.ylabel('theta_1')

        plt.show()

        #pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """



        fig=plt.figure()
        Y=y.values
        x1 = X.values

        if self.fit_intercept:
            x = np.ones((x1.shape[0], x1.shape[1] + 1))
            x[:, 1:] = x1
        else:
            x = x1
        theta=np.array([t_0,t_1])
        y_pred=np.dot(x,theta)
        plt.scatter(x1,Y)
        plt.plot(x1,y_pred)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()




        #pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        t_0_max = np.max(t_0)
        t_1_max = np.max(t_1)
        w0 = np.linspace(-(t_0_max + 0.25), (t_0_max + 0.25), 500)
        w1 = np.linspace(-(t_1_max + 0.25), (t_1_max + 0.25), 500)
        #w0 = np.linspace(-(all_coef[0, 0] + 0.25), (all_coef[0, 0] + 0.25), 500)
        #w1 = np.linspace(-(all_coef[0, 1] + 0.25), (all_coef[0, 1] + 0.25), 500)
        x = X.values
        Y = y.values
        x1 = np.ones((x.shape[0], x.shape[1] + 1))
        x1[:, 1:] = x
        W0, W1 = np.meshgrid(w0, w1)
        mse = np.ones(W0.shape)
        for i in range(W0.shape[0]):
            for j in range(W0.shape[1]):
                theta = np.array([W0[i, j], W1[i, j]])
                y_pred = np.dot(x1, theta)
                mse[i, j] = (np.sum((Y - y_pred) ** 2)) * 1 / (Y.shape[0])

        cp = plt.contour(W0, W1, mse)
        plt.clabel(cp, inline=1, fontsize=10)
        plt.xlabel('theta_0')
        plt.ylabel('theta_1')
        plt.scatter(t_0[1:], t_1[1:], self.mse)
        plt.show()



def objective_fn(theta, x, y):
    # y_pred = np.dot(x, theta)
    cost = (np.sum((y - np.dot(x, theta)) ** 2)) / x.shape[0]

    return cost