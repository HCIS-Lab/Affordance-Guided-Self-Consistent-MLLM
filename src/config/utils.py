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


def read_txt(file_path):
    with open(file_path, 'r') as f:
        txt = f.read()
    return txt


def concat_config(config_file_folder):
    config_files = os.listdir(config_file_folder)
    config_files = [f for f in config_files if f.endswith('.yaml') and 'all' not in f]
    config_files = [os.path.join(config_file_folder, f) for f in config_files]
    all_config = ""
    for config_file in config_files:
        with open(config_file, 'r') as f:
            config = read_txt(config_file)
        all_config += config
    return all_config


def write_txt(config, file_path):
    with open(file_path, 'w') as f:
        f.write(config)