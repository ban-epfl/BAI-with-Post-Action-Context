from scipy.optimize import bisect
from scipy.optimize import root_scalar
from utils import *






class NSTS:
    def __init__(self, n, k, delta, contexts=None, context_estimate = False, acc_optimization = 0.01, optimization_method = 'binary search'):
        self.context_estimate = context_estimate
        self.contexts = contexts
        self.n = n 
        self.k = k 
        self.T = 0 
        self.N_times_seen = np.zeros((n, k))
        self.sum_of_rewards = np.zeros((n, k))
        self.delta = delta
        self.acc_optimization = acc_optimization
        self.optimization_method = optimization_method
        
    


    def best_empirical_arm_calculator(self): #Code for estimator, handles the cases that contexts are not completely seen yet but in the code we have a initilization phase which pulls each arm until the corresponding contexts are seen
        
        post_actions_mu_hat = np.divide(self.sum_of_rewards, self.N_times_seen
                                        , out=np.zeros_like(self.sum_of_rewards, dtype=float), where=self.N_times_seen != 0)
        #safe division
        actions_mu_hat = np.sum(post_actions_mu_hat * self.contexts, axis=1)
        actions_mu_hat_ignore_context = np.sum(self.sum_of_rewards,axis=1)/np.sum(self.N_times_seen, axis = 1)
        
        mask = np.all(self.N_times_seen != 0, axis=1)
        
        result = np.where(mask, actions_mu_hat, actions_mu_hat_ignore_context)
        #result refers to estimated means
        
        delta_hat = np.max(result) - result
        best_arm = np.argmax(result)  
        return best_arm, result, delta_hat


    def lambda_hat(self):
        #compute delta hat from estimated means
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()

        #compute number of actions taken from number of post actions seen
        N_actions = np.sum(self.N_times_seen, axis=1)

        #compute lambda hat based on the equation
        

        best_arm_times_pulled = self.N_times_seen[best_arm]
        
        best_arm_constant = np.sum(self.contexts[best_arm]**2 / best_arm_times_pulled)

        result = 1 / (np.sum(self.contexts**2 / self.N_times_seen,axis=1) + best_arm_constant)

        result = result * (delta_hat**2) / 2

        #since we have 0 in the i*-th element and that should not be computed in arg min
            
            
        
        return SecondMin(result)


    def Stopping_Rule(self): #returns True if stopping criteria is not met and False otherwise

        lambda_hat_t = self.lambda_hat() 
        
        c_t = c_hat_t(self.n, self.k , self.T, self.delta)

        return c_t >= lambda_hat_t
    
    
    def Alternate_Stopping_Rule(self): 

        lambda_hat_t = self.lambda_hat() 
        
        c_t = c_hat_t(self.n, 1 , self.T, self.delta)

        return c_t >= lambda_hat_t






    def f(self, Delta, x):
        # because i!=i* we should deduct 1 in the end

        result = np.sum(1 / (x * Delta ** 2 - 1) ** 2) - 1
        return result 
   

    def df(self, Delta, x): #derivative of f used for newton method for optimization
        return np.sum((-2 * Delta ** 2 * (x * Delta ** 2 - 1) ) / (x * Delta ** 2 - 1) ** 2)




    def binary_search(self, Delta, start, end): #manual binary search function
        epsilon = self.acc_optimization
        alpha = self.f(Delta, (start + end)/2)
        while alpha > 1 + epsilon or alpha < 1 - epsilon:
            mid = (start + end)/2
            alpha = self.f(Delta, mid)
            if alpha > 1:
                start = mid
            else:
                end = mid
        return (start + end)/2



    def optimization_oracle(self, Delta_hat): #different optimization methods handled, the best method is scipy's binary search

        Delta_min = SecondMin(Delta_hat)
        n = Delta_hat.shape[0]

        if self.optimization_method == 'manual binary search':
            w_opt = self.binary_search(Delta_hat, start = 2/Delta_min**2, end = (1 + (n - 1)**0.5)/Delta_min**2) #get the number of steps as input? set it theoretically?
        elif self.optimization_method == 'binary search':
            w_opt = bisect(lambda x: self.f(Delta_hat, x) - 1, 1.5/Delta_min**2, (1 + n**0.5)/Delta_min**2)
            
        elif self.optimization_method == 'newton':
            
            result = root_scalar(f = lambda x: self.f(Delta_hat, x) - 1, fprime = lambda x: self.df(Delta_hat, x), bracket=[1/Delta_min**2, (1 + (n - 1)**0.5)/Delta_min**2], method='newton', x0 = (1/Delta_min**2+ (1 + (n - 1)**0.5)/Delta_min**2) / 2)
            w_opt = result.root
            
        elif self.optimization_method == 'root scalar':
            result = root_scalar(lambda x: self.f(Delta_hat, x) - 1, bracket=[1/Delta_min**2, (1 + (n - 1)**0.5)/Delta_min**2], method='bisect')
            w_opt = result.root
        
        return w_opt



    def Optimal_W(self): #computing the optimal w based on estimators and optimization problem described in paper

        best_arm, _, Delta_hat = self.best_empirical_arm_calculator()

        w_opt = self.optimization_oracle(Delta_hat)

        result = 1 / (w_opt * Delta_hat ** 2 - 1)

        result[best_arm] = 1
        


        #normalize and output
        return result/np.sum(result)



    def D_Tracking(self):
        #If U_t is non-empty output the least pulled arm
        N_actions = np.sum(self.N_times_seen, axis=1)
        if np.min(N_actions) <= max(0, self.T ** 0.5 - self.n/2):
            return np.argmin(N_actions)
        
        result = self.T * self.Optimal_W() - N_actions
        return np.argmax(result)
    


    
    def Initialization_Phase(self, means, samples): #1 handles proper initialization and 2 initiates the means with empirical mean. The empirical mean for each arm will be used until we have seen all of the contexts and then our estimator will be used.
        #1
        while np.any(self.N_times_seen == 0):
            result = np.where(self.N_times_seen == 0)
            i = result[0][0]
            j = hidden_action_sampler(self.contexts[i])
            self.N_times_seen[i, j] += 1
            self.sum_of_rewards[i, j] += samples[self.T] + means[i, j] #sample according to x ~ N(u, 1) === x = y + u , u ~ N(0, 1)
            self.T += 1
            
        #2
#         for i in range(self.n):
#             j = hidden_action_sampler(self.contexts[i])
#             self.N_times_seen[i, j] += 1
#             self.sum_of_rewards[i, j] += samples[self.T] + means[i, j] #sample according to x ~ N(u, 1) === x = y + u , u ~ N(0, 1)
#             self.T += 1



        if self.context_estimate == True:
            self.contexts = self.N_times_seen / self.N_times_seen.sum(axis=1, keepdims=True)

        return self.T
        

    def update(self, main_action, post_action, reward): #updates the means, number of times every arm with its contexts has been seen and T
        self.T += 1
        self.N_times_seen[main_action, post_action] += 1
        self.sum_of_rewards[main_action, post_action] += reward

        if self.context_estimate == True:
            self.contexts = self.N_times_seen / self.N_times_seen.sum(axis=1, keepdims=True)




class TS:
    def __init__(self, n, delta, contexts=None, acc_optimization = 0.01, optimization_method = 'binary search'):
        self.n = n 
        self.T = 0 
        self.N_times_seen = np.zeros(n)
        self.sum_of_rewards = np.zeros(n)
        self.delta = delta
        self.contexts = contexts
        self.acc_optimization = acc_optimization
        self.optimization_method = optimization_method        
    


    def best_empirical_arm_calculator(self): #simple empirical means for arms, disregarding the contexts information for estimators
        actions_mu_hat = self.sum_of_rewards/self.N_times_seen
        delta_hat = np.max(actions_mu_hat) - actions_mu_hat
        best_arm = np.argmax(actions_mu_hat)  
        return best_arm, actions_mu_hat, delta_hat



    def lambda_hat(self):
        #compute delta hat from estimated means
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()
        

        best_arm_times_pulled = self.N_times_seen[best_arm]

        result = self.N_times_seen * best_arm_times_pulled / (self.N_times_seen + best_arm_times_pulled)

        result = result * (delta_hat**2) / 2
        
        

        #since we have 0 in the i*-th element and that should not be computed in arg min
        return SecondMin(result)


    def Stopping_Rule(self):

        lambda_hat_t = self.lambda_hat() / 26 #since the means are generated between 0,10 we have to have a division by (1+mu**2/4)=26 for the theoretical threshold

        c_t = c_hat_t(self.n, 1 , self.T, self.delta)

        return c_t >= lambda_hat_t
    
    
    def Alternate_Stopping_rule(self):
        lambda_hat_t = self.lambda_hat()

        c_t = c_hat_t(self.n, 1 , self.T, self.delta)

        return c_t >= lambda_hat_t
    
    
    def f(self, Delta, x):
        # because i!=i* we should deduct 1 in the end

        result = np.sum(1 / (x * Delta ** 2 - 1) ** 2) - 1
        return result 
   

    def df(self, Delta, x):
        return np.sum((-2 * Delta ** 2 * (x * Delta ** 2 - 1) ) / (x * Delta ** 2 - 1) ** 2)




    def binary_search(self, Delta, start, end):
        epsilon = self.acc_optimization
        alpha = self.f(Delta, (start + end)/2)
        while alpha > 1 + epsilon or alpha < 1 - epsilon:
            mid = (start + end)/2
            alpha = self.f(Delta, mid)
            if alpha > 1:
                start = mid
            else:
                end = mid
        return (start + end)/2



    def optimization_oracle(self, Delta_hat):

        Delta_min = SecondMin(Delta_hat)
        n = Delta_hat.shape[0]

        if self.optimization_method == 'manual binary search':
            w_opt = self.binary_search(Delta_hat, start = 2/Delta_min**2, end = (1 + (n - 1)**0.5)/Delta_min**2) #get the number of steps as input? set it theoretically?
        elif self.optimization_method == 'binary search':
            w_opt = bisect(lambda x: self.f(Delta_hat, x) - 1, 1.5/Delta_min**2, (1 + n**0.5)/Delta_min**2)
            
        elif self.optimization_method == 'newton':
            
            result = root_scalar(f = lambda x: self.f(Delta_hat, x) - 1, fprime = lambda x: self.df(Delta_hat, x), bracket=[1/Delta_min**2, (1 + (n - 1)**0.5)/Delta_min**2], method='newton', x0 = (1/Delta_min**2+ (1 + (n - 1)**0.5)/Delta_min**2) / 2)
            w_opt = result.root
            
        elif self.optimization_method == 'root scalar':
            result = root_scalar(lambda x: self.f(Delta_hat, x) - 1, bracket=[1/Delta_min**2, (1 + (n - 1)**0.5)/Delta_min**2], method='bisect')
            w_opt = result.root
        
        return w_opt


    
    def Optimal_W(self):
        best_arm, _, Delta_hat = self.best_empirical_arm_calculator()
        
        
        w_opt = self.optimization_oracle(Delta_hat)

        result = 1 / (w_opt * Delta_hat ** 2 - 1)

        result[best_arm] = 1

        

        return result / np.sum(result)




    def D_Tracking(self):
        #If U_t is non-empty output the least pulled arm
        if np.min(self.N_times_seen) <= max(0, self.T ** 0.5 - self.n/2):
            return np.argmin(self.N_times_seen)
        
        result = self.T * self.Optimal_W() - self.N_times_seen
        return np.argmax(result)
    
    def Initialization_Phase(self, means, samples):
        for i in range(self.n):
            j = hidden_action_sampler(self.contexts[i])
            self.N_times_seen[i] += 1
            self.sum_of_rewards[i] += samples[self.T] + means[i, j] #sample according to x ~ N(u, 1) === x = y + u , u ~ N(0, 1)
            self.T += 1

        return self.T
        

    def update(self, main_action, reward):
        self.T += 1
        self.N_times_seen[main_action] += 1
        self.sum_of_rewards[main_action] += reward

        

