import numpy as np
from numpy.random import randn, random, uniform, multivariate_normal, seed
import pdb
import random

class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here

        """

        self.data_robustness_threshold = 0.65

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        # [doc 1]: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
        # [doc 2]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.searchsorted.html
        # [doc 3]: https://filterpy.readthedocs.io/en/latest/_modules/filterpy/monte_carlo/resampling.html#multinomial_resample
        # --------------------------------------------------------
        # Get the number of particles that
        particle_num = np.size(X_bar, 0);  

        # Extract the weight of each particle as a 1D vector
        weights = X_bar[:,-1]
    
        # [See doc 1 & 3] calculate the cumulative weights given each individual weight of the particles
        cumulative_sum = np.cumsum(weights)

        # [See doc 1 & 3] To avoid round-off errors, so everything sums to 1
        cumulative_sum[-1] = 1.  
        
        # [See doc 1 & 2 & 3] Use Python default binary search method to return the indices of weighted resampled points
        resampled_idx = np.searchsorted(cumulative_sum, random(particle_num))

        # Using the re-sampled indices to extract the n x 4 nd array containing 
        X_bar_resampled = X_bar[resampled_idx,:]

        # [Optional] Re-weight the re-sampled points
        X_bar_resampled[:,-1] = X_bar_resampled[:,-1]/np.sum(X_bar_resampled[:,-1])


        # --------------------------------------------------------

        return X_bar_resampled

    def low_variance_sampler_legacy(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        """
        TODO : Add your code here
        """
        # [doc 1]: Sebastian Thurn, probablistic robotics - p 110
        # --------------------------------------------------------
        # Get the number of particles that
        particle_num = np.size(X_bar, 0); 

        # intialize X_bar_resampled as an empty list
        X_bar_resampled = []

        # Extract the weight of each particle as a 1D vector
        weights = X_bar[:,-1]

        # optional, in case I didn't normalize the weights, here we do it again
        weights = self._prob_rebalance(weights)

        # [doc 1] constant r
        r = random()/particle_num

        # [doc 1] step counter i
        i = 0

        # [doc 1] weight storage variable c
        c = weights[i]

        # [doc 1] double loop
        for m in range(particle_num):
            U = r + (m - 1)/particle_num

            while U>c: 
                i = i + 1
                c = c + weights[i]
            # end while
             
            X_bar_resampled.append(X_bar[i])

        # Convert from list to array
        X_bar_resampled = np.asarray(X_bar_resampled)

        # [Optional] Re-weight the re-sampled points
        X_bar_resampled[:,-1] = X_bar_resampled[:,-1]/np.sum(X_bar_resampled[:,-1])

        # --------------------------------------------------------
        
        return X_bar_resampled



    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, weights] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, weights] values for resampled set of particles
        """
        X_bar_resampled = []
        M = len(X_bar)
        weights = X_bar[:,3]

        r = random.uniform(0, 1.0/M)

        weights /= weights.sum()
        
        c = weights[0]
        i = 0
        for m_idx in range(M):
            u_idx = r + (m_idx)*(1.0/M)
            while u_idx > c:
                i = i + 1
                c = c + weights[i]
            X_bar_resampled.append(X_bar[i])
        X_bar_resampled = np.asarray(X_bar_resampled)

        return X_bar_resampled    



if __name__ == "__main__":
    # pass

    # Test X_bar
    Test_X_bar = np.asarray([[1,2,3,0.4],[3,-1,4,0.2],[5,5,0,0.1],[3,2,3,0.2], [5, -1,0.4, 0.1]])

    # Make an instantiation of the test
    Test = Resampling()

    # Try either of the methods to see if they are correct
    # results = Test.multinomial_sampler(Test_X_bar)  #  1) 
    results = Test.low_variance_sampler(Test_X_bar)   #  2)

    # Print results
    # print(results)

