import json
import numpy as np

def add_output_to_json(instance_number, mu_hats, N_times_seens, w_s, T, best_arm, file_path = '1output'):
    #mu_hats and N_times_seens and w_s are tracked after the initialization rounds
    
#     mu_hats_list = mu_hats.tolist()
#     N_times_seens_list = N_times_seens.tolist()
#     w_s_list = w_s.tolist()


    instance = {
        "mu_hats": mu_hats,
        "N_times_seens": N_times_seens,
        "w_s": w_s,
        "T": T,
        "best_arm": int(best_arm)
    }
    
    #print(instance)
    
    file_path += str(instance_number)
    file_path += '.json'
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    
    data.append(instance)
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def read_outputs_from_json(instance_number, file_path = '1output'):
    file_path += str(instance_number)
    file_path += '.json'
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Prepare lists for each parameter
            mu_hat_list = []
            N_times_seens_list = []
            T_list = []
            w_s_list = []
            best_arm_list = []
           
            
            # Populate the lists with data from the JSON file
            for instance in data:
                T_list.append(instance["T"])
                best_arm_list.append(instance["best_arm"])
                w_s_list.append(instance["w_s"])
                mu_hat_list.append(instance["mu_hats"])
                N_times_seens_list.append(instance["N_times_seens"])
            
            return mu_hat_list, N_times_seens_list, T_list, w_s_list, best_arm_list
            
    except FileNotFoundError:
        return [], [], [], []

    