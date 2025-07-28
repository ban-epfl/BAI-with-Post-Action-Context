import json
import numpy as np

def add_instance_to_json(n, k, delta, contexts, means, w_star, T_star, file_path = 'instance1.json'):
    contexts_list = contexts.tolist()
    means_list = means.tolist()
    w_star_list = w_star.tolist()

    instance = {
        "n": n,
        "k": k,
        "delta": delta,
        "means": means_list,
        "contexts": contexts_list,
        "w_star": w_star_list,
        "T_star": T_star
    }
    
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []
    
    data.append(instance)
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def read_instances_from_json(file_path = 'instance1.json'):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Prepare lists for each parameter
            n_list = []
            k_list = []
            delta_list = []
            means_list = []
            contexts_list = []
            w_star_list = []
            T_star_list = []
            
            # Populate the lists with data from the JSON file
            for instance in data:
                n_list.append(instance["n"])
                k_list.append(instance["k"])
                delta_list.append(instance["delta"])
                means_list.append(np.array(instance["means"]))
                contexts_list.append(np.array(instance["contexts"]))
                w_star_list.append(np.array(instance["w_star"]))
                T_star_list.append(instance["T_star"])
            
            return n_list, k_list, delta_list, means_list, contexts_list, w_star_list, T_star_list
            
    except FileNotFoundError:
        return [], [], [], [], [], [], []

    