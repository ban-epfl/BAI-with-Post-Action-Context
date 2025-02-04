import argparse
import numpy as np
from utils import *
from Environments import *
from Instance_json import *
from output_json import *



def main():
    parser = argparse.ArgumentParser(description="Parse variables for main1.py")
    parser.add_argument("--Algorithm", type=str, required=False, default='NSTS', help="Description for var1 (e.g., a string variable)")
    parser.add_argument("--instance_index", type=int, required=False, default=0)
    parser.add_argument("--theoretical_stopping_rule", type=int, required=False, default=0)
    parser.add_argument("--store", type=int, required=False, default=1)

    # Parse the arguments 
    args = parser.parse_args() 
    
    
    
    
    ns, ks, deltas, means, contexts, w_stars, T_stars= read_instances_from_json('instances1.json')
    
    index = args.instance_index
    
    n = ns[index]
    k = ks[index]
    delta = deltas[index]
    means = means[index]
    contexts = contexts[index]
    
    theoretical_stopping_rule = False
    if args.theoretical_stopping_rule:
        theoretical_stopping_rule = True

    
    env = Environment1(means, args.Algorithm, n, k, delta, contexts, False, theoretical_stopping_rule)
    best_arm, mu_hats, N_times_seens, w_s, T = env.loop()

    path = '1output_'
        
    path +=args.Algorithm
    
    if theoretical_stopping_rule:
        path += '_theoreticalStoppingRule'
    
    if args.store:
        add_output_to_json(index, mu_hats, N_times_seens, w_s, T, best_arm, file_path = 'results/' + path)
        
    print("The number of time steps is :", T)
    print("The actual best arm is :", np.argmax(np.sum(means * contexts, axis=1)))
    print("The best arm identified is :", best_arm)
    print("########################################################")
    
    

if __name__ == "__main__":
    main()