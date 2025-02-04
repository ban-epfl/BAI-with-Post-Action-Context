import numpy as np
import cvxpy as cp
from scipy.optimize import nnls
import matplotlib.pyplot as plt




def SecondMin(arr):
    n = len(arr)
    smallest = np.min(arr)
    second_smallest = 100000
    for i in range(n):
        if second_smallest > arr[i] and arr[i] != smallest:
            second_smallest = arr[i]
    return second_smallest


def Cg(x):
    return x + np.log(x)


def c_hat_t(n, k , t, delta):
    return 4 * k * np.log(4 + np.log(t / (2 * k))) + 2 * k * Cg(np.log((n - 1) / delta)/(2 * k))

def c_hat_sep(k, N_z, delta):
    return 2 * np.sum(np.log(4 + np.log(N_z))) + k * Cg(np.log(1/delta) / k)

def beta(t, delta):
    return np.log((np.log(t) + 1) / delta)/t


def hidden_action_sampler(distribution, n_samples = None):
    if n_samples:
        return np.random.choice(np.arange(len(distribution)), size = n_samples, p = distribution)
        
    return np.random.choice(np.arange(len(distribution)), p = distribution)


def context_initiator(n, k, mode = "random"):

    if mode == "random":
        samples = np.random.uniform(low=0.0, high=1.0, size=(n, k))
        row_sums = samples.sum(axis=1, keepdims=True)
        normalized_samples = samples / row_sums
        
    if mode == "random with min 1/2k":
        samples = np.random.uniform(low=(k-1)/(2*k-1), high=1.0, size=(n, k))
        row_sums = samples.sum(axis=1, keepdims=True)
        normalized_samples = samples / row_sums
        
    if mode == "random with min 1/4k":
        samples = np.random.uniform(low=1/4, high=1.0, size=(n, k))
        row_sums = samples.sum(axis=1, keepdims=True)
        normalized_samples = samples / row_sums

    return normalized_samples




def convert_back_to_X_space(A, z):#solve for Ax=z
    
    n = A.shape[0]

    #optimization variable
    x = cp.Variable(n)
    constraints = [x >= 0, cp.sum(x) == 1]

#objective function: minimize ||Ax - b||² (least squares)
    objective = cp.Minimize(cp.norm(A.T @ x - z, 2))
    problem = cp.Problem(objective, constraints)
    problem.solve()
#    assert problem.status == cp.OPTIMAL, f"Solution was not found"
    
#     print(x.value, problem.value)
    
    if problem.status != cp.OPTIMAL:
        return False
    
    return x.value




def find_projected_on_simplex_equivalent_in_X_space(A, y): #minimize |Ax-y| subject to x on simplex


    n = A.shape[0]           
    x = cp.Variable(n)          

    objective = cp.Minimize(cp.norm(A.T @ x - y, 2)**2)
    constraints = [x >= 0, cp.sum(x) == 1]

    problem = cp.Problem(objective, constraints)
    problem.solve()
    
#     print("Optimal x:", x.value)
#     print("Optimal objective value ||Ax - y||^2:", problem.value)
    
    return x.value

    


    
    
def plot_mus(list1, list2, list3):
    plt.figure(figsize=(20, 15))

    
    l1 = mean_of_columns(list1)
    l2 = mean_of_columns(list2)
    l3 = mean_of_columns(list3)
    
    
    plt.plot(np.arange(len(l1)), l1, color='blue', label='NSTS with known contexts')
    plt.plot(np.arange(len(l2)), l2, color='orange', label='NSTS with unknown contexts')
    plt.plot(np.arange(len(l3)), l3, color='green', label='TS')

    plt.xlabel("rounds")
    plt.ylabel("mean of distance of estimated w of each algorithm with actual w ")
    plt.title("Plot of Two Lists of Lists")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()
    
    
    
def mean_of_columns(list_of_lists):
    max_length = max(len(sublist) for sublist in list_of_lists)
    
    means = []
    for i in range(max_length):
        column = [sublist[i] for sublist in list_of_lists if i < len(sublist)]
        
        if column:
            means.append(sum(column) / len(column))
        else:
            means.append(None)  

    return means

    
