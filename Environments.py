import numpy as np
from Algorithms1 import *
from Algorithms2 import *
from scipy.optimize import bisect
from utils import *




class Environment1:#non-seperator environment
    def __init__(self, means, algorithm, n, k, delta, contexts=None, context_estimate = False, acc_optimization = 0.01, Theoretical_stopping_rule = True):

        self.contexts = contexts 
        self.algorithm = algorithm 
        self.means = means
        self.n = n
        self.k = k
        self.T = 0
        self.delta = delta
        self.samples = np.random.normal(loc=0, scale=1, size=20000000)
        self.context_estimate = context_estimate
        self.acc_optimization = acc_optimization
        self.best_arm = np.argmax(np.sum(means*contexts, axis=1))
        self.action_mus = np.sum(means*contexts, axis=1)
        self.optimal_W = self.compute_optimal_w()
        
        self.delta_hat = self.action_mus[self.best_arm] - self.action_mus

        result = self.optimal_W[self.best_arm] * self.optimal_W / (self.optimal_W[self.best_arm] + self.optimal_W)
        result = result * self.delta_hat / 2
        
        self.T_star = SecondMin(result)
        
        self.mu_hats = []
        self.w_s = []
        self.N_times_seens = []
        
        
        
        self.log_period = 500 #for NSTS
        
        if algorithm=='TS':
            self.log_period = 15000
        
        self.Theoretical_stopping_rule = Theoretical_stopping_rule
        
        
        
        
        
        
    def compute_optimal_w(self):
        
        
        Delta_hat = self.action_mus[self.best_arm] - self.action_mus

        w_opt = self.optimization_oracle(Delta_hat)

        result = 1 / (w_opt * Delta_hat ** 2 - 1)

        result[self.best_arm] = 1
        return result/np.sum(result)
    
    
    def f(self, Delta, x):
        result = np.sum(1 / (x * Delta ** 2 - 1) ** 2) - 1
        return result 
        
        
    def optimization_oracle(self, Delta_hat):

        Delta_min = SecondMin(Delta_hat)
        n = Delta_hat.shape[0]

        w_opt = bisect(lambda x: self.f(Delta_hat, x) - 1, 2/Delta_min**2, (1 + (n - 1)**0.5)/Delta_min**2)

        
        return w_opt
    

    def loop(self): #implements the interaction of environment and algorithms, for every algorithm, its stopping rule and Tracking rule are used for stopping criteria checking and choosing the arm to play and the .update method is used to update the rewards and number of times seen along with T for the Algorithm object. 


        if self.algorithm == 'NSTS':#Non-Seperator Track and Stop
            #Initialization phase
            alg = NSTS(self.n, self.k, self.delta, self.contexts, self.context_estimate, self.acc_optimization)
            self.T = alg.Initialization_Phase(self.means, self.samples)
            
            print("initialization finished with ",self.T," rounds")
            
            if self.Theoretical_stopping_rule:
            
                while alg.Stopping_Rule():
                    # Select an action using the algorithm
                    action = alg.D_Tracking()
                    post_action = hidden_action_sampler(self.contexts[action])
                    reward = self.samples[self.T] + self.means[action, post_action]
                    alg.update(action, post_action, reward)

                    self.T += 1

                    if self.T%self.log_period == 0:
                        self.w_s.append(alg.Optimal_W().tolist())
                        _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                        self.mu_hats.append(actions_mu_hat.tolist())

                        self.N_times_seens.append(np.sum(alg.N_times_seen, axis=1).tolist())


                best_arm, _, _ = alg.best_empirical_arm_calculator()
        
        
            else:
                while alg.Alternate_Stopping_rule():  #using the alternative stopping rule
                    # Select an action using the algorithm
                    action = alg.D_Tracking()
                    post_action = hidden_action_sampler(self.contexts[action])
                    reward = self.samples[self.T] + self.means[action, post_action]
                    alg.update(action, post_action, reward)

                    self.T += 1

                    if self.T%self.log_period == 0:
                        self.w_s.append(alg.Optimal_W().tolist())
                        _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                        self.mu_hats.append(actions_mu_hat.tolist())

                        self.N_times_seens.append(np.sum(alg.N_times_seen, axis=1).tolist())


                best_arm, _, _ = alg.best_empirical_arm_calculator()            
#             explored_vector = np.sum(alg.N_times_seen, axis=1)
        
            
            
        if self.algorithm == 'TS': #Track and Stop disregarding contexts information
            #Initialization phase
            alg = TS(self.n, self.delta, self.contexts)
            self.T = alg.Initialization_Phase(self.means, self.samples)
            
            print("initialization finished with ",self.T," rounds")
            
            if self.Theoretical_stopping_rule:
            
                while alg.Stopping_Rule():
                    # Select an action using the algorithm
                    action = alg.D_Tracking()
                    post_action = hidden_action_sampler(self.contexts[action])
                    reward = self.samples[self.T] + self.means[action, post_action]
                    alg.update(action, reward)

                    self.T += 1


                    if self.T%self.log_period == 0:
                        self.w_s.append(alg.Optimal_W().tolist())
                        _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                        self.mu_hats.append(actions_mu_hat.tolist())

                        self.N_times_seens.append(alg.N_times_seen.tolist())



                best_arm, _, _ = alg.best_empirical_arm_calculator()
            
            
            else:
                while alg.Alternate_Stopping_rule():
                    # Select an action using the algorithm
                    action = alg.D_Tracking()
                    post_action = hidden_action_sampler(self.contexts[action])
                    reward = self.samples[self.T] + self.means[action, post_action]
                    alg.update(action, reward)

                    self.T += 1


                    if self.T%self.log_period == 0:
                        self.w_s.append(alg.Optimal_W().tolist())
                        _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                        self.mu_hats.append(actions_mu_hat.tolist())

                        self.N_times_seens.append(alg.N_times_seen.tolist())


                best_arm, _, _ = alg.best_empirical_arm_calculator()
                
#             explored_vector = alg.N_times_seen
        
        
#         explored_vector /= np.sum(explored_vector)
        
#         explored_vector_distance = np.linalg.norm(self.optimal_W - explored_vector)

        return best_arm, self.mu_hats, self.N_times_seens, self.w_s, self.T









class Environment2: #seperator environment
    def __init__(self, means, algorithm, n, k, delta, contexts=None, mode = {'use_optimized_p': False, 'average_w': False, 'average_points_played': False}, stopping_rule = 'd_stopping_rule'):

        self.contexts = contexts
        self.algorithm = algorithm  
        self.means = means
        self.n = n
        self.k = k
        self.T = 0
        self.delta = delta
        self.samples = np.random.normal(loc=0, scale=1, size=1000000) 
        self.mode = mode
        
        self.mus = np.dot(self.contexts, self.means)
        self.best_arm = np.argmax(self.mus)
        self.delta_hat = self.mus[self.best_arm] - self.mus
        
        self.optimal_W, self.T_star = self.optimal_weight()
        
        #self.T_star *= np.sum(self.optimal_W)
        self.T_star = 0.5/self.T_star
        #self.optimal_W /= np.sum(self.optimal_W)
        
        
        self.mu_hats = []
        self.w_s = []
        self.N_times_seens = []
        
        self.log_period = 20 #every 20 iteration save the data of the arms played up to now and optimal w
        
        self.stopping_rule = stopping_rule #for c-tracking 
        
        
        
       
    def optimal_weight(self):
        
        delta = self.delta_hat
        i_star = self.best_arm


        lambda_var = cp.Variable(self.n, nonneg=True)  # Weights for convex combination
        w = self.contexts.T @ lambda_var  # w is a convex combination of rows of A
        t = cp.Variable() 

        constraints = [cp.sum(lambda_var) == 1]  
        
        constraints += [w[j] >= 0 for j in range(self.k)]
        
        
        for i in range(self.n):
            if i == i_star: #constraints are for non-optimal arms
                continue
                

            middle = cp.sum([(self.contexts[i, j] - self.contexts[i_star, j])**2 * cp.inv_pos(w[j]) for j in range(self.k)])
            constraints.append(
                t >= middle / (delta[i] ** 2)
            )

        objective = cp.Minimize(t)
        

        problem = cp.Problem(objective, constraints)
        
        
        
        try:
            problem.solve()
        except SolverError as e:
            self.optimization_failed_flag = True


        if problem.status not in ["optimal", "optimal_inaccurate"]:
            self.optimization_failed_flag = True
            return 0,0
            #raise ValueError("Optimization problem did not converge.")

        #print(t.value)
        return w.value, t.value 

    def loop(self):
        if self.algorithm == 'STS': #Seperator Track and Stop
            alg = STS(self.n, self.k, self.delta, self.contexts, self.mode)
            self.T = alg.Initialization_Phase(self.means, self.samples)
            
            print("initialization finished with ",self.T," rounds")
            
            while alg.Stopping_Rule():
                # Select an action using the algorithm
                action = alg.G_Tracking()
                post_action = hidden_action_sampler(self.contexts[action])
                reward = self.samples[self.T] + self.means[post_action]
                alg.update(action, post_action, reward)
    
                self.T += 1
                
                w = alg.Optimal_W()[1]
                w = w.tolist()
                
                if w and self.T%self.log_period==0:
                    self.w_s.append(w)
                    _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                    self.mu_hats.append(actions_mu_hat.tolist())

                    self.N_times_seens.append(alg.N_times_seen.tolist())
                
            best_arm, _, _ = alg.best_empirical_arm_calculator()
            print("number of failed optimization rounds is ", alg.optimization_failed_number_of_rounds)
            
            
        
        if self.algorithm == 'STS_C_Tracking':
            #Initialization phase
            alg = STS(self.n, self.k, self.delta, self.contexts, self.mode)
            self.T = alg.Initialization_Phase(self.means, self.samples)
            
            print("initialization finished with ",self.T," rounds")
            
            if self.stopping_rule == 'd_stopping_rule':
            
                while alg.Stopping_Rule():
                    # Select an action using the algorithm
                    action = alg.C_Tracking()
                    post_action = hidden_action_sampler(self.contexts[action])
                    reward = self.samples[self.T] + self.means[post_action]
                    alg.update(action, post_action, reward)

                    self.T += 1

                    if self.T%self.log_period==0:
                        w = alg.projected_c_optimal_w( alg.C_optimal_W() )
                        if w.size != 0:
                            w = np.dot(self.contexts.T, w) 
                        w = w.tolist()
                        if w:
                            self.w_s.append(w)
                            _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                            self.mu_hats.append(actions_mu_hat.tolist())

                            self.N_times_seens.append(alg.N_times_seen.tolist())
                            
            else:
            
                while alg.C_Stopping_Rule():
                    # Select an action using the algorithm
                    action = alg.C_Tracking()
                    post_action = hidden_action_sampler(self.contexts[action])
                    reward = self.samples[self.T] + self.means[post_action]
                    alg.update(action, post_action, reward)

                    self.T += 1

                    if self.T%self.log_period==0:
                        w = alg.projected_c_optimal_w( alg.C_optimal_W() )
                        if w.size != 0:
                            w = np.dot(self.contexts.T, w) 
                        w = w.tolist()
                        if w:
                            self.w_s.append(w)
                            _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                            self.mu_hats.append(actions_mu_hat.tolist())

                            self.N_times_seens.append(alg.N_times_seen.tolist())
                            
                           
                
            best_arm, _, _ = alg.best_empirical_arm_calculator()
            print("number of failed optimization rounds is ", alg.optimization_failed_number_of_rounds)
            
        if self.algorithm == 'TSS':
            #Initialization phase
            alg = TSS(self.n, self.delta, self.contexts)
            self.T = alg.Initialization_Phase(self.means, self.samples)
            
            print("initialization finished with ",self.T," rounds")
            
            while alg.Stopping_Rule():
                # Select an action using the algorithm
                action = alg.D_Tracking()
                post_action = hidden_action_sampler(self.contexts[action])
                reward = self.samples[self.T] + self.means[post_action]
                alg.update(action, reward)
    
                self.T += 1
        
        
                if self.T%self.log_period==0:
                    self.w_s.append(alg.Optimal_W().tolist())
                    _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                    self.mu_hats.append(actions_mu_hat.tolist())

                    self.N_times_seens.append(alg.N_times_seen.tolist())
                
                
            best_arm, _, _ = alg.best_empirical_arm_calculator()
            
        if self.algorithm == 'LTS':
            #Initialization phase
            alg = LTS(self.n, self.k, self.delta, self.contexts)
            
            while alg.Stopping_Rule():
                # Select an action using the algorithm
                action = alg.Tracking()
                post_action = hidden_action_sampler(self.contexts[action])
                reward = self.samples[self.T] + self.means[post_action]
                alg.update(action, reward)
    
                self.T += 1
                
            
                if self.T%self.log_period==0:
                    
                    #print(np.argmax(alg.contexts @ alg.mu_hat))
        
                    self.w_s.append(alg.design.tolist())

                    self.mu_hats.append((alg.contexts @ alg.mu_hat).tolist())

                    self.N_times_seens.append(alg.N_times_seen.tolist())
                
                
            best_arm = alg.best_arm_hat

        return best_arm, self.mu_hats, self.N_times_seens, self.w_s, self.T