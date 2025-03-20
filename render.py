import os
import textwrap
from environment import IsaacSim
from src.config import *
def render_all_config(config_file, show_text=False, force=False):
    task_types = get_task_type_list(config_file)
    file_root, file_name = config_file.split('/')[:-1], config_file.split('/')[-1]
    folder_name = file_name.replace('.yaml', show_text * '_text')
    file_root = os.path.join('/'.join(file_root), 'rendered', folder_name)
    if force:
        os.system(f'rm -rf {file_root}')
    os.makedirs(file_root, exist_ok=True)
    for task_type in task_types:
        for env_idx in range(1, get_task_env_num(config_file, task_type)+1):
            print(f"{'='*20}\n{task_type.upper()} {env_idx}\n{'='*20}")
            config = read_yaml(config_file, task_type=task_type, env_idx=env_idx)
            img_filename = os.path.join(file_root, f"{task_type}_{env_idx}.jpg")
            print(img_filename)
            txt_filename = os.path.join(file_root, f"{task_type}_{env_idx}.txt")
            if force or not os.path.exists(img_filename):
                Environment = IsaacSim(env_cfg_dict=config)
                answer_sequence = '\n'.join(config.get('answer', [])) 
                instruction = textwrap.fill(config.get('instruction', ''))
                text = f"{instruction}\n{answer_sequence}" if show_text else ""  
                Environment.render_config(img_filename, text=text)
if __name__ == '__main__':
    config_name = 'general_easy' # 'general_easy',  'spatial_relationship', 'amount_ambiguity', 'distance', 'obstacles'
    config_file = f'src/config/{config_name}.yaml'
    render_all_config(config_file, show_text=False, force=True)