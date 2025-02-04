import argparse
import numpy as np
from utils import *
from Environments import *
from Instance_json import *
from output_json import *



def main():
    parser = argparse.ArgumentParser(description="Parse variables for main1.py")
    parser.add_argument("--Algorithm", type=str, required=False, default='STS', help="Description for var1 (e.g., a string variable)")
    parser.add_argument("--instance_index", type=int, required=False, default=0)
    parser.add_argument("--store", type=int, required=False, default=1)
    parser.add_argument("--use_optimized_p", type=int, required=False, default=0) 
    parser.add_argument("--average_points_played", type=int, required=False, default=0) 
    parser.add_argument("--average_w", type=int, required=False, default=0) 
    parser.add_argument("--c_stopping_rule", type=int, required=False, default=0) 

    # Parse the arguments 
    args = parser.parse_args() 
    
    
    
    
    ns, ks, deltas, means, contexts, w_stars, T_stars= read_instances_from_json('instances2.json')
    
    index = args.instance_index
    
    n = ns[index]
    k = ks[index]
    delta = deltas[index]
    means = means[index]
    contexts = contexts[index]
    
    use_optimized_p = False
    average_w = False
    average_points_played = False
    stopping_rule = 'd_stopping_rule'
    
    if args.average_points_played:
        average_points_played = True
        
    if args.use_optimized_p:
        use_optimized_p = True
        
    if args.average_w:
        average_w = True
        
    if args.c_stopping_rule:
        stopping_rule = 'c_stopping_rule'
        
    
    
    
    
    mode = {'use_optimized_p': use_optimized_p, 'average_w': average_w, 'average_points_played': average_points_played}

    
    env = Environment2(means, args.Algorithm, n, k, delta, contexts, mode = mode, stopping_rule = stopping_rule)
    best_arm, mu_hats, N_times_seens, w_s, T = env.loop()

    path = '2output_'
    path += args.Algorithm
    
    if args.Algorithm == 'STS':
        if use_optimized_p:
            path += '_optimizedPTrue'
        else:
            if average_w:
                path += '_averagedWTrue'
            if average_points_played:
                path += '_averagePointsPlayedTrue'
    
    if args.Algorithm == 'STS_C_Tracking':
        if stopping_rule == 'c_stopping_rule':
            path += '_CStoppingRule'
    
    
    if args.store:
        add_output_to_json(index, mu_hats, N_times_seens, w_s, T, best_arm, file_path = 'results/'+path)
    
    
    print("The number of time steps is :", T)
    print("The actual best arm is :", np.argmax(np.dot(contexts, means)))
    print("The best arm identified is :", best_arm)
    print("########################################################")
    

if __name__ == "__main__":
    main()