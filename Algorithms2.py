from utils import *
from scipy.optimize import linprog
from scipy.optimize import bisect
from scipy.optimize import root_scalar
from cvxpy.error import SolverError





class STS:
    def __init__(self, n, k, delta, contexts=None, mode = {'use_optimized_p': False, 'average_w': False, 'average_points_played': False}):
        self.contexts = contexts
        self.n = n 
        self.k = k 
        self.T = 0 
        self.N_times_seen = np.zeros(n)
        self.K_times_seen = np.zeros(k)
        self.sum_of_rewards = np.zeros(k)
        self.delta = delta
        self.exploration_vector = find_projected_on_simplex_equivalent_in_X_space(contexts, 1/k * np.ones(k))
        
        self.optimization_failed_flag = False
        self.optimization_failed_number_of_rounds = 0
        
        self.mode = mode
        
        self.exploration_vector_Z = (contexts.T @ self.exploration_vector) / np.sum(contexts.T @ self.exploration_vector)
        
        
        self.sum_points_played = np.zeros(k)
        self.sum_ws = np.zeros(k)#only used for G-tracking
        
        self.sum_ws_c = np.zeros(n) #only used for c-tracking
        
    


    def best_empirical_arm_calculator(self):
        mean_hat = self.sum_of_rewards / self.K_times_seen
        
        actions_mu_hat = np.dot(self.contexts, mean_hat)
        delta_hat = np.max(actions_mu_hat) - actions_mu_hat
        
        best_arm = np.argmax(actions_mu_hat)  
        return best_arm, actions_mu_hat, delta_hat
    #changed


    def lambda_hat(self):
        #compute delta hat from estimated means
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()

        result = 1 / (np.sum((self.contexts - self.contexts[best_arm])**2 / self.K_times_seen, axis=1)) 
        result = result * (delta_hat**2) / 2    

        #since we have 0 in the i*-th element and that should not be computed in arg min
        return SecondMin(result)


    def Stopping_Rule(self):

        lambda_hat_t = self.lambda_hat() 

        c_t = c_hat_sep(self.k, self.K_times_seen, self.delta)

        return c_t >= lambda_hat_t
    
    
    def Alternate_Stopping_Rule(self):

        lambda_hat_t = self.lambda_hat() 
        
        c_t = c_hat_t(self.n, 1 , self.T, self.delta)

        return c_t >= lambda_hat_t
    
    
    def C_Stopping_Rule(self): 
        
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()
        
        V = - self.contexts + self.contexts[best_arm]
        
        expected_K = self.contexts.T @ self.N_times_seen
        
        confidence = np.sqrt(2*(self.k * Cg(np.log((self.n - 1) / self.delta)/self.k) + np.sum (2 * np.log(4 + np.log(self.K_times_seen)))) * np.sum(V ** 2 / expected_K, axis = 1))
        
        
        return SecondMin(delta_hat - confidence) < 0


    
    def optimization_line_coefficient(self, w_t, v_k): 

        w_t = np.asarray(w_t)
        v_k = np.asarray(v_k)
        original_array = np.asarray(self.contexts)


        def target_vector(alpha):
            return (1 + alpha) * w_t - alpha * v_k


        c = np.zeros(self.n + 1)  # n lambdas + 1 alpha
        c[-1] = -1  # Coefficient for -alpha

        A_eq = np.zeros((self.k + 1, self.n + 1))
        A_eq[:self.k, :self.n] = original_array.T
        A_eq[:self.k, -1] = -(w_t - v_k) 
        A_eq[self.k, :self.n] = 1  
        A_eq[self.k, -1] = 0  # Alpha does not contribute to sum of lambdas

        b_eq = np.zeros(self.k + 1)
        b_eq[:self.k] = w_t
        b_eq[self.k] = 1 

        A_ub = np.zeros((self.n, self.n + 1))
        A_ub[:, :self.n] = -np.eye(self.n)  # -lambda_i <= 0
        b_ub = np.zeros(self.n)

        bounds = [(0, None)] * self.n + [(None, None)]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            return result.x[-1]  #alpha
        else:
            self.optimization_failed_flag = True
            
    
    def optimization_for_Z_optimal_vector(self):
        
        i_star, _, delta = self.best_empirical_arm_calculator()

        lambda_var = cp.Variable(self.n, nonneg=True)  # Weights for convex combination
        w = self.contexts.T @ lambda_var  # w is a convex combination of rows of A
        t = cp.Variable()

        constraints = [cp.sum(lambda_var) == 1]  # Convex combination constraint
        
        constraints += [w[j] >= 0 for j in range(self.k)]
        
        
        for i in range(self.n):
            if i == i_star:
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

        # Return the optimal w and t
        w_opt = w.value
        w_opt /= np.sum(w_opt)
        t_opt = t.value
        
        return w_opt, t_opt
    
   




    def optimization_for_p(self, w):
        
        N_t = self.K_times_seen

        lambda_ = cp.Variable(self.n)
        p = self.contexts.T @ lambda_
        

        # ||N_t + p - w||^2 = ||p - (w - N_t)||^2
        y = w - N_t  
        objective = cp.Minimize(cp.norm((p + N_t)/(np.sum(N_t)+1) - w/np.sum(w), 2)**2)

        constraints = [
            lambda_ >= 0, 
            cp.sum(lambda_) == 1
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver failed: {problem.status}")

            
        lambda_opt = lambda_.value  
        p_opt = self.contexts.T @ lambda_opt 
        
        p_opt /= np.sum(p_opt)

        return p_opt, lambda_opt
        



    def Optimal_W(self):
        
        if not self.mode['use_optimized_p']:
            
            if self.mode['average_points_played']:
                #handling the first round
                v_k = self.sum_points_played / np.sum(self.sum_points_played) if np.sum(self.sum_points_played)!=0 else self.sum_points_played
                
            else:
                v_k = self.K_times_seen / np.sum(self.K_times_seen)

            w_t, _ = self.optimization_for_Z_optimal_vector()
            


            if self.optimization_failed_flag == True:
                self.optimization_failed_number_of_rounds += 1
                print("failed1")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector, np.array([])

            
            if self.mode['average_w']:
                w = self.sum_ws + w_t
                w /= np.sum(w)
            else:
                w = w_t

                
            alpha = self.optimization_line_coefficient(w, v_k)

            if self.optimization_failed_flag == True:
                self.optimization_failed_number_of_rounds += 1
                print("failed2")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector, np.array([])

            OPT_Z = (1 + alpha) * w_t - alpha * v_k

            OPT_X = convert_back_to_X_space(self.contexts, OPT_Z)


            if not isinstance(OPT_X, np.ndarray):
                self.optimization_failed_number_of_rounds += 1
                print("failed3")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector ,  np.array([])

            self.sum_ws += w_t
            self.sum_points_played += OPT_Z
         
        
        
            return OPT_X, w_t
        
        else:
            
            w_t, _ = self.optimization_for_Z_optimal_vector()
            
            if self.optimization_failed_flag == True:
                self.optimization_failed_number_of_rounds += 1
                print("failed1")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector,  np.array([])
            
            w = self.sum_ws + w_t
            
            OPT_Z, _ = self.optimization_for_p(w) 
            
            
            
            
            OPT_X = convert_back_to_X_space(self.contexts, OPT_Z)


            if not isinstance(OPT_X, np.ndarray):
                self.optimization_failed_number_of_rounds += 1
                print("failed3")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector, np.array([])

            
            self.sum_points_played += OPT_Z
            self.sum_ws += w_t
            
            return OPT_X, w_t
            

    
    def C_optimal_W(self):
        
        
        i_star, _, delta = self.best_empirical_arm_calculator()

        lambda_var = cp.Variable(self.n, nonneg=True)  # Weights for convex combination
        w = self.contexts.T @ lambda_var  # w is a convex combination of rows of A
        t = cp.Variable() 


        # Constraints
        constraints = [cp.sum(lambda_var) == 1]  # Convex combination constraint
        
        constraints += [w[j] >= 0 for j in range(self.k)]
        
        
        for i in range(self.n):
            if i == i_star:
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
            return 0
            #raise ValueError("Optimization problem did not converge.")

        # Return the optimal lambda_var
        lambda_var_opt = lambda_var.value / np.sum(lambda_var.value)
        
        return lambda_var_opt
    
    
    
    def projected_c_optimal_w(self, w):
        
        eps = 0.5/np.sqrt(self.n **2 + self.T)

        v = cp.Variable(self.n)
        t = cp.Variable()

        objective = cp.Minimize(t)

        # Constraints
        constraints = [
            v >= eps,   
            v <= 1,     
            cp.sum(v) == 1,    
            v - w <= t,     
            v - w >= -t 
        ]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve()
            if problem.status != cp.OPTIMAL:
                self.optimization_failed_flag = True
                #print("1")
                return np.array([])
        except Exception as e:
            self.optimization_failed_flag = True
            #print("2")
            #print(f"An error occurred during optimization: {str(e)}")
            return np.array([])

        # Return the projected vector
        return v.value
        
            
    
    
    def C_Tracking(self):
        w = self.C_optimal_W()
        if self.optimization_failed_flag:
            print("optimization failed 1")
            self.optimization_failed_number_of_rounds += 1
            return 0
        
        w_eps = self.projected_c_optimal_w(w)
        if self.optimization_failed_flag:
            self.optimization_failed_number_of_rounds += 1
            print("optimization failed 2")
            return 0
            
        self.sum_ws_c += w/np.sum(w) #make sure w sums up to 1
        
        sum_w_scaled = (self.sum_ws_c)/ np.sum(self.sum_ws_c) * np.sum(self.N_times_seen) #some optimization rounds may have failed so scale up to match it
        result = sum_w_scaled - self.N_times_seen
        return np.argmax(result)
        
        



    def G_Tracking(self):
        w, _ = self.Optimal_W()
        result = self.T * w - self.N_times_seen
        return np.argmax(result)
    


    
    def Initialization_Phase(self, means, samples):
        while np.any(self.K_times_seen == 0):
            result = np.where(self.K_times_seen == 0)
            j_prime = result[0][0]
            i = np.argmax(self.contexts[:, j_prime])
            j = hidden_action_sampler(self.contexts[i])
            self.N_times_seen[i] += 1
            self.K_times_seen[j] += 1
            self.sum_of_rewards[j] += samples[self.T] + means[j] #sample according to x ~ N(u, 1) === x = y + u , u ~ N(0, 1)
            self.T += 1
            self.sum_points_played += self.contexts[i]

        return self.T
        

    def update(self, main_action, post_action, reward):
        self.T += 1
        self.N_times_seen[main_action] += 1
        self.sum_of_rewards[post_action] += reward
        self.K_times_seen[post_action] += 1
        
        self.optimization_failed_flag = False
        
        
        
        
      
    
    


class TSS: #code for Track and Stop. This is very similar to the code of class TS in environment but they differ in the estimator calculators
    def __init__(self, n, delta, contexts=None, acc_optimization = 0.01, optimization_method = 'binary search'):
        self.n = n 
        self.T = 0 
        self.N_times_seen = np.zeros(n)
        self.sum_of_rewards = np.zeros(n)
        self.delta = delta
        self.contexts = contexts
        self.acc_optimization = acc_optimization
        self.optimization_method = optimization_method        
    


    def best_empirical_arm_calculator(self):
        actions_mu_hat = self.sum_of_rewards/self.N_times_seen
        delta_hat = np.max(actions_mu_hat) - actions_mu_hat
        best_arm = np.argmax(actions_mu_hat)  
        return best_arm, actions_mu_hat, delta_hat



    def lambda_hat(self):
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()
        

        best_arm_times_pulled = self.N_times_seen[best_arm]

        result = self.N_times_seen * best_arm_times_pulled / (self.N_times_seen + best_arm_times_pulled)

        result = result * (delta_hat**2) / 2
        

        #since we have 0 in the i*-th element and that should not be computed in arg min
        return SecondMin(result)


    def Stopping_Rule(self):

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
            w_opt = self.binary_search(Delta_hat, start = 1/Delta_min**2, end = (1 + (n - 1)**0.5)/Delta_min**2) #get the number of steps as input? set it theoretically?
        elif self.optimization_method == 'binary search':
            w_opt = bisect(lambda x: self.f(Delta_hat, x) - 1, 1/Delta_min**2, (1 + (n - 1)**0.5)/Delta_min**2)
            
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
    


    
    def Initialization_Phase(self, means, samples):#pull until we have seen every context
        for i in range(self.n):
            j = hidden_action_sampler(self.contexts[i])
            self.N_times_seen[i] += 1
            self.sum_of_rewards[i] += samples[self.T] + means[j] #sample according to x ~ N(u, 1) === x = y + u , u ~ N(0, 1)
            self.T += 1

        return self.T
        

    def update(self, main_action, reward):
        self.T += 1
        self.N_times_seen[main_action] += 1
        self.sum_of_rewards[main_action] += reward

        
        
        
        
        
        
class LTS: #code of https://github.com/jedray/LT-S adapted to our environment
    def __init__(self, n, k, delta, contexts=None):
        self.n = n 
        self.k = k
        self.T = 1 
        self.N_times_seen = np.zeros(n)
        self.sum_of_rewards = np.zeros(n)
        self.delta = delta
        self.contexts = contexts
        
        self.u = 0.1
        self.A = np.zeros((k, k))
        self.i0 = 0
        self.A0 = np.zeros(k).astype(int)
        self.c0 = 0 # forced exploration constant
        self.c1 = 0
        
        self.design = np.ones(self.n)
        self.design /= self.design.sum()
        
        self.sum_reward_times_a_t = np.zeros(k)
        
        self.sigma = 1
        self.c2 = (1+self.u) * (self.sigma**2)
        
        self.cumulative_design = np.zeros(self.n)
        
        
        self.g = self.k
        self.laziness_factor = 2
        self.averaging = True
        self.mu_hat = np.zeros(self.k)
        self.best_arm_hat = 0
        self.range_arms = [i for i in range(self.n-1)]
        self.Y_A = None
        self.Y_bar = None
        self.count_numerical_errors = 0
        self.psi_inv = 0
        self.count_fe = 0
        
        self.__build_A0()

    
    
    def __build_A0(self):
        r = 0
        k = 0
        while r < self.k:
            arm = np.random.randint(self.n)
            if np.linalg.matrix_rank(self.A +
                                     np.outer(self.contexts[arm], self.contexts[arm])) > r:
                self.A += np.outer(self.contexts[arm], self.contexts[arm])
                self.A0[r] = arm
                r += 1
        self.c0 = np.min(np.linalg.eigvals(self.A)) / np.sqrt(self.k)
        self.c1 = np.min(np.linalg.eigvals(self.A))



    def __Z(self): 
        
        # We compute rho in the following manner to avoid numerical errors
        rewards = self.contexts @ self.mu_hat
        gaps = rewards[self.best_arm_hat] - rewards
        gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        if np.any(gaps == 0):
            return 0
        
        #TODO change
        Y = self.contexts[self.best_arm_hat, :] - self.contexts
        Y = np.delete(Y, self.best_arm_hat, 0)
        Y = (1/gaps)*Y
        _I = np.ones((self.k, 1))
        design = self.N_times_seen;
        A_inv = np.linalg.pinv(self.contexts.T @ np.diag(design) @ self.contexts)
        U, D, V = np.linalg.svd(A_inv)
        Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T
        newY = (Y @ Ainvhalf)**2
        rho = newY @ _I
        Z = (1/(2*np.max(rho)))
        return Z


    def Stopping_Rule(self):
        lambda_hat_t = self.lambda_hat()

        c_t = c_hat_t(self.n, 1 , self.T, self.delta)
        
        #c_t = np.log((1 + np.log(self.T))/self.delta)

        return c_t >= lambda_hat_t or (np.min(np.linalg.eigvals(self.A)) < np.max(np.sum(self.contexts**2, axis=1)) )
    
    
    def lambda_hat(self):
        
        mus = self.contexts @ self.mu_hat
        
        delta_hat = mus[self.best_arm_hat] - mus

        best_arm_times_pulled = self.N_times_seen[self.best_arm_hat]

        result = self.N_times_seen * best_arm_times_pulled / (self.N_times_seen + best_arm_times_pulled)

        result = result * (delta_hat**2) / 2

        #since we have 0 in the i*-th element and that should not be computed in arg min
        return SecondMin(result)



    def Tracking(self):
        f = self.c0 * np.sqrt(self.T);
        if np.min(np.linalg.eigvals(self.A)) < f:
            arm = self.A0[self.i0]
            self.i0 = np.mod(self.i0 + 1, self.k)
            self.count_fe += 1;
        else:
            if self.averaging:
                # Averaging
                self.cumulative_design += self.design
                self.support = (self.cumulative_design > 0)
                self.map_support = np.array(range(self.n))[self.support]
                index = np.argmin((self.N_times_seen -
                                   self.cumulative_design)[self.map_support])
            else:
                # Agressive update
                self.support = (self.design > 0)
                self.map_support = np.array(range(self.n))[self.support]
                index = np.argmin(
                    (self.N_times_seen - self.T * self.design)[self.map_support])
            arm = self.map_support[index]
        return arm
    
        

    def update(self, main_action, reward):

        self.N_times_seen[main_action] += 1
        self.sum_of_rewards[main_action] += reward
        
        self.A += np.outer(self.contexts[main_action], self.contexts[main_action])
        
        self.sum_reward_times_a_t += reward * self.contexts[main_action]
        self.mu_hat = np.linalg.pinv(self.A) @ self.sum_reward_times_a_t
        self.best_arm_hat = np.argmax(self.contexts @ self.mu_hat)
        self.__Lazy_update()
        self.T += 1        
        
     
    
    def __Lazy_update(self):
        if self.T == int(self.g):
            self.g = np.max([np.ceil(self.laziness_factor *self.T), self.T+1])
            self.design, rho = self.__optimal_allocation(self.design)
            self.psi_inv = 2 *rho;
        
        
   

    def __optimal_allocation(self, init):

        #design = init
        design = np.ones(self.n)
        design /= design.sum()
        # maximum number of iterations
        max_iter = 1000
        # construct Y
        rewards = self.contexts @ self.mu_hat
        gaps = np.max(rewards) - rewards
        gaps = np.delete(gaps, self.best_arm_hat, 0)
        gaps = gaps.reshape(len(gaps), 1)
        Y = self.contexts[self.best_arm_hat, :] - self.contexts
        Y = np.delete(Y, self.best_arm_hat, 0)
        Y = (1 / gaps) * Y
        _I = np.ones((self.k, 1))
        for count in range(1, max_iter):
            A_inv = np.linalg.pinv(self.contexts.T @ np.diag(design) @ self.contexts)
            U, D, V = np.linalg.svd(A_inv)
            Ainvhalf = U @ np.diag(np.sqrt(D)) @ V.T

            newY = (Y @ Ainvhalf)**2
            rho = newY @ _I  # np.ones((newY.shape[1], 1))

            idx = np.argmax(rho)
            y = Y[idx, :, None]
            g = ((self.contexts @ A_inv @ y) * (self.contexts @ A_inv @ y)).flatten()
            g_idx = np.argmax(g)

            gamma = 2 / (count + 2)
            design_update = -gamma * design
            design_update[g_idx] += gamma

            relative = np.linalg.norm(design_update) / (np.linalg.norm(design))

            design += design_update

            if relative < 0.01:
                break

        idx_fix = np.where(design < 1e-5)[0]
        design[idx_fix] = 0
        design /= np.sum(design)
        return design, np.max(rho)
