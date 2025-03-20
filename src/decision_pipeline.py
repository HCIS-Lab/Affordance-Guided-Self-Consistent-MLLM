import inspect
import os
import cv2
import random
from copy import deepcopy
from src.affordance.agents import *
from src.semantic import *
from src.utils import *
from src.cot_prompt import cot2 as cot # replace to other version
from src.cot_prompt import cot_baseline2 as cot_baseline # replace to other version

def get_action_list(tool_list, object_list):
    action_list = ["scoop", "stir", "drop_food", "pull_bowl_closer", "open_dumbwaiter", "close_dumbwaiter", "start_dumbwaiter", "put_bowl_into_dumbwaiter", "DONE"]
    action_list.extend([f"grasp_{tool}" for tool in tool_list])
    action_list.extend([f"put_{tool}_back" for tool in tool_list])
    action_list.extend([f"move_to_{object.split(' (')[0]}" for object in object_list])
    return action_list

class Decision_pipeline():
    def __init__(self, instruction, init_object_list, tool_list, log_folder) -> None:
        self.instruction = instruction
        self.init_object_list = init_object_list
        self.action_list = get_action_list(tool_list, init_object_list)
        self.log_folder = log_folder
        self.obs_id = 0
        self.affordance_agent = Affordance_agent(self.init_object_list, self.action_list)
        self.record = {}
        self._record_buffer = {}
        self.affordance_info_list = {}
        self.action_sequence = []
    
    def set_affordance_agent(self, affordance_type, **kwargs):
        affordance_agent_list = {
            "our": Affordance_agent_ours
        } 
        self.affordance_agent = affordance_agent_list.get(affordance_type, Affordance_agent)(self.init_object_list, self.action_list, **kwargs)
    
    def set_obs_id(self):
        self.obs_id += 1
        
    def random_action(self):
        return random.choice(self.action_list)
    
    def update_record(self):
        self.record = deepcopy(self._record_buffer)
        
    def clear_record(self):
        self.record = {}
        self._record_buffer = {}
    
    def update_action_sequence(self, action):
        self.action_sequence.append(action)
        
    def chain_of_thought(
        self, 
        observation_rgb_path,
        action_candidate=[],
    ) -> dict:
        if not action_candidate: 
            action_candidate = self.action_list
        action_candidate_scores = cot(
            instruction=self.instruction, 
            container_list=self.init_object_list, 
            action_list=self.action_list, 
            action_seq=self.action_sequence, 
            obs_url=encode_image(observation_rgb_path), 
            action_candidate=action_candidate, 
            log_folder=self.log_folder, 
            obs_id=self.obs_id
        )
        return action_candidate_scores
    
    
    def chain_of_thought_baseline(
        self, 
        observation_rgb_path,
        action_candidate=[],
    ) -> dict:
        if not action_candidate: 
            action_candidate = self.action_list
        action_candidate_scores = cot_baseline(
            instruction=self.instruction, 
            container_list=self.init_object_list, 
            action_list=self.action_list, 
            action_seq=self.action_sequence, 
            obs_url=encode_image(observation_rgb_path), 
            action_candidate=action_candidate, 
            log_folder=self.log_folder, 
            obs_id=self.obs_id
        )
        return action_candidate_scores
     
    def get_affordance_score(
        self, 
        observation_rgb_path, 
        observation_d_path,
        action_candidate=[],
        with_info = False,
        **kwargs
    ):
        if len(action_candidate) == 0:
            action_candidate = self.action_list
        affordance = self.affordance_agent.get_affordance(
            observation_rgb_path, 
            observation_d_path, 
            action_seq=self.action_sequence, 
            action_candidate=action_candidate, 
            **kwargs
        )
        if with_info:
            info = self.affordance_agent.get_affordance_info()
            if info:
                key = len(self.action_sequence) + 1
                if key not in self.affordance_info_list:
                    self.affordance_info_list[key] = []
                self.affordance_info_list[key].append(info)
        affordance = sort_scores_dict(affordance)
        open(os.path.join(self.log_folder, f"affordance_{self.obs_id}.txt"), 'w').write(f"{affordance}")
        print(f"affordance {max(affordance, key=affordance.get)}")
        print(affordance)
        print("=" * 20)
        return affordance
    
    def get_semantic_score(
        self, 
        observation_rgb_path, 
        use_vlm=False,
        use_affordance_info=False,
        segmentation_prompt=False,
        update_record=True,
    ):
        obs_image = None
        additional_info = {}
        if use_vlm:
            base64_image = encode_image(observation_rgb_path)
            obs_image = base64_image
            cv2.imwrite(os.path.join(self.log_folder, f'observation_{self.obs_id}.png'), cv2.imread(observation_rgb_path))
        if use_affordance_info:
            affordance_info_string = []
            for key, infos in self.affordance_info_list.items():
                info_string = ''.join(infos)
                affordance_info_string.append(f'\t\tIn iteration {key}, {info_string}')
            additional_info['Previous Affordance Feedback'] = '\n' + '\n'.join(affordance_info_string)
        semantic, record = get_selection_score_openai(
            instruction=self.instruction, 
            object_list=self.init_object_list, 
            action_list=self.action_list, 
            action_seq=self.action_sequence, 
            use_vlm=use_vlm,
            current_obs_url=obs_image,
            additional_info=additional_info,
            log_folder=self.log_folder, 
            obs_id=self.obs_id,
            segmentation_prompt=segmentation_prompt,
            example_with_image=False,
            example_in_system=True,
            record=deepcopy(self.record),
        )
        self._record_buffer = record
        if update_record:
            self.update_record()

        print(f"semantic {max(semantic, key=semantic.get)}")
        print(semantic)
        print("=" * 20)
        return semantic
    
    def get_combined_score(
        self, 
        observation_rgb_path, 
        observation_d_path, 
        use_vlm=False, 
        action_candidate=[]
    ):
        affordance = self.get_affordance_score(observation_rgb_path, observation_d_path, self.action_sequence, action_candidate)
        semantic = self.get_semantic_score(self.instruction, observation_rgb_path, self.action_sequence, use_vlm)
        score = {action: affordance[action] * semantic[action] for action in self.action_list}
        score = sort_scores_dict(score)
        open(os.path.join(self.log_folder, f"combined_{self.obs_id}.txt"), 'w').write(f"{score}")
        print(f"combined {max(score, key=score.get)}")
        print(score)
        print("=" * 20)
        
        return score
    
if __name__ == "__main__":
    observation_rgb_path = 'affordance/data/spoon/30/0_rgb/000.png'
    observation_d_path = observation_rgb_path.replace('_rgb', '_depth')
    instruction = "Stir the beans in the bowl, then scoop it to the round plate. 1."
    decision_pipeline = Decision_pipeline()
    combined_score = decision_pipeline.get_score(instruction, observation_rgb_path, observation_d_path)
    print(f"combined_score {max(combined_score, key=combined_score.get)}")
    print(combined_score)