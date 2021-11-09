from gpflow import optimizers
from gpflow.kernels import ChangePoints
from gpflow.kernels import Matern32
from gpflow.models import GPR
from gpflow.optimizers import Scipy
import numpy as np

class ChangePoint:
    def __init__(self, lbw, alpha, burnin_time, 
                 is_verbose = True, is_save=False, output_file=''):
        """
        Parameter:
            lbw: (int), lookback window for ChangePoint Detection
            alpha: (int), """
        self.lbw = lbw
        self.alpha = alpha
        self.burnin_time = burnin_time
        self.output_file = output_file
        self.is_verbose = is_verbose
        self.is_save = is_save
        if is_save:
            with open(output_file, 'w') as f:
                f.write('Start\tEnd\tLocation\tLocationNumber\n')
                f.close()
        
    
    def cal_group(self, X, Y, X_mapping):
        """
        Calculate CPD in moving method

        Parameter:
            X: (np.array), array for independent variable
            Y: (np.array), array for dependent variable
            X_Mapping: (pd.Series/dict), mapping for X -> mapped info, e.g. date
        """
        N = len(X)
        cpd_locations = []
        # Iterate on series to get 
        for w in range(self.lbw, N):
            x = X[w - self.lbw : w].astype('float')
            y = Y[w - self.lbw : w]
            v, gamma, location = self.cal_single(x, y)
            if self.is_verbose:
                print(f"Window Start: {X_mapping[w - self.lbw]}, End: {X_mapping[w]}, Significance: {v}")
            # Set Burn in time and significance level for CPD
            if v < self.alpha:
                continue
            if len(cpd_locations) == 0 or location - float(cpd_locations[-1][3]) > self.burnin_time:
                location = int(location)
                # todo 改成constraint
                if location >= w or location < w - self.lbw:
                    continue
                if location >= N:
                    cpd_locations.append((X_mapping[w - self.lbw], X_mapping[w], '', str(location)))
                else:
                    cpd_locations.append((X_mapping[w - self.lbw], X_mapping[w], X_mapping.iloc[location], str(location)))
                if self.is_save:
                    with open(self.output_file, 'a') as f:
                        f.write('\t'.join(cpd_locations[-1]) + '\n')
                        f.close()
        return cpd_locations

    @staticmethod
    def cal_single(x, y):
        """
        Parameter: 
            x: (np.array), araay for indices
            y: (np.array), array for dependent variables
        Return:
            v: (float), significance for changepoint detection result
            gamma: (float), normalized location for CPD detected, on range (0, 1)
            location: (float), location for CPD detected
            """
        cp_init = ChangePoint.init_point(x)
        LBW = len(x)
        y = (y - y.mean()) / y.std()
        # reshape x and y for GP regression
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # GP regression
        # single kernel
        k_single = Matern32(1.0)
        m_single = GPR(data=(x, y), kernel=k_single, mean_function=None)
        opt_single = Scipy()
        opt_logs_single = opt_single.minimize(closure=m_single.training_loss, 
                                              variables=m_single.trainable_variables, 
                                              options = dict(maxiter=1000))
        # double kernel
        k1 = Matern32(1.0)
        k2 = Matern32(1.0)
        k_double = ChangePoints(kernels=[k1, k2], locations=[cp_init], steepness=1.0)
        m_double = GPR(data=(x, y), kernel=k_double, mean_function=None)
        opt_double = Scipy()
        try:
            opt_logs_double = opt_double.minimize(closure=m_double.training_loss, 
                                                  variables=m_double.trainable_variables, 
                                                  options = dict(maxiter=1000))
        except:
            return
        
        # Calculate significance and location
        loss_single = m_single.log_marginal_likelihood().numpy()
        loss_double = m_double.log_marginal_likelihood().numpy()
        location = k_double.locations.numpy()[0]
        v = ChangePoint.sigmoid_kernels(loss_double, loss_single)
        gamma = (location - x[0]) / LBW

        return v, gamma, location
    
    @staticmethod
    def init_point(x):
        cp_init = (x[-1] + x[0]) / 2
        return 
    
    @staticmethod
    def sigmoid_kernels(loss_modify, loss_origin):
        """
        loss_modify > loss_origin theoretically
        """
        return 1 - 1/(1+np.exp(loss_modify-loss_origin))
        