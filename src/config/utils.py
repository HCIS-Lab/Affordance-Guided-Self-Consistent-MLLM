import os
import yaml

def read_yaml(file_path, task_type='general', env_idx=1):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    assert task_type in config.keys(), f"{task_type} desn't exist"
    config = config[task_type][env_idx]
    return config

def get_task_type_list(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return list(config.keys())    

def get_task_env_num(file_path, task_type):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return len(config.get(task_type, []))    
