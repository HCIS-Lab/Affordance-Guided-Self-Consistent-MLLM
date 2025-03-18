from abc import abstractmethod
from typing import List
class Affordance_agent:
    def __init__(self, init_object_list, action_list):
        self.init_object_list = init_object_list
        self.action_list = action_list
        
    @abstractmethod
    def get_affordance(self, rgb_img_path, gray_scale_img_path, action_seq: List[str], action_candidate=[]):
        if not action_candidate:
            return {action: 0 for action in self.action_list}
        return (int(action in self.action_list) for action in action_candidate)
    
    @abstractmethod
    def get_affordance_info(self):
        return None