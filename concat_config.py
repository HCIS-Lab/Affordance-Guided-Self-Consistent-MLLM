import yaml
import os

def read_txt(file_path, task_type=None, env_idx=None):
    # read as txt
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
    
if __name__ == '__main__':
    config_file_folder = 'src/config/final_task'
    all_config = concat_config(config_file_folder)
    write_txt(all_config, os.path.join(config_file_folder, 'all.yaml'))
    print('All config files are concatenated into all.yaml')