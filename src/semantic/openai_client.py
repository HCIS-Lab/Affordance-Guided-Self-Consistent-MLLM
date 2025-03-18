import os
import socket
from openai import OpenAI
import pickle
import torch
import numpy as np
from collections import Counter

from .utils import *
from src.utils import *

def call_openai_api(messages, model='gpt-4o'):
    """
    return: raw content
    - .choices[0].message.content: text content
    - .choices[0].logprobs.content[0].top_logprobs: top 20 log probabilities
    """
    openai_client = OpenAI()
    content = openai_client.chat.completions.create(
        # model='gpt-3.5-turbo',
        model=model,
        messages=messages,
        logprobs=True,
        top_logprobs=20,
        temperature=0,
    )
    return content

def get_semantic(
    instruction: str, 
    object_list=None, 
    action_list=["scoop", "fork", "cut", "move", "stir", "DONE"], 
    action_seq=None, 
    use_vlm=False, 
    obs_url=None,
    log_folder=None,
    obs_id=None,
    record={}
) -> dict:
    """no more used"""
    
    action_description = {preprocess_action(action): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    system_prompt = get_system_prompt(use_vlm)
    user_prompt = get_user_prompt(instruction, action_seq, action_dict, object_list)
    model = 'gpt-4o'
    if use_vlm:
        assert obs_url is not None, "Observation url could not be None"
        description_system_prompt = "You are a robot arm in food manipulation scneario. You should focus on your gripper. You need to describe the food manipulation table top scenario from the image."
        description_user_prompt = "Describe the food manipulation table top scenario from the image. Including what the robot are holding, spoon, knife, fork, or None"
        scenario_prompt = get_messages(description_system_prompt, description_user_prompt, user_image_url=obs_url)
        scenario_description = call_openai_api(scenario_prompt, model).choices[0].message.content
        messages = get_messages(system_prompt, user_prompt, user_image_url=obs_url)
    else:
        scenario_description = ''
        messages = get_messages(system_prompt, user_prompt)
    
    response_content = call_openai_api(messages, model).choices[0].message.content
    
    explanation_prompt = get_messages(system_prompt, user_prompt + f"{response_content} \nPlease explain why you choose the action.", user_image_url=obs_url)
    # explanation = call_openai_api(explanation_prompt, model).choices[0].message.content
    
    print(response_content)
    if use_vlm:
        print(scenario_description)
    print(explanation)
    
    _semantic = {}
    for action in action_description:
        same = [x if x in response_content.split() else None for x in action.split()]
        while None in same:
            same.remove(None)
        _semantic[action] = len(same) / len(response_content.split())
    scores = torch.softmax(torch.Tensor(list(_semantic.values())), dim=-1).cpu().tolist()
    
    _semantic = {action: score for action, score in zip(_semantic.keys(), scores)}

    semantic = {}
    for key, val in _semantic.items():
        semantic[action_description[key]] = val
    
    if log_folder is not None:
        assert obs_id is not None, "Please provide observation id"
        with open(os.path.join(log_folder, f'prompt_{obs_id}.txt'), 'w') as f:
            f.write('\n'.join(['\n[SYS]\n' + system_prompt, '\n[USER]\n' + user_prompt, '\n[DES]\n' + scenario_description, '\n[RES]\n' + response_content, '\n[EXP]\n' + explanation]))
    return sorted(semantic, key=lambda x: x.value(), reverse=True)

'''
This function.
'''
def get_selection_score(
    instruction: str, 
    object_list=None, 
    action_list=["scoop", "move", "stir", "DONE"],
    action_seq=None, 
    use_vlm=False, 
    current_obs_url=None,
    additional_info: Dict={},
    log_folder=None,
    obs_id=None,
    example_with_image=False,
    example_in_system=True,
    segmentation_prompt=False,
    record={}
) -> dict:
    
    print("instruction", instruction)
    print("object_list", object_list)
    print("action_list", action_list)
    print("action_seq", action_seq)
    print("use_vlm", use_vlm)
    print("log_folder", log_folder)
    print("obs_id", obs_id)
    
    assert not (example_with_image and example_in_system), "Cannot have both example_with_image and example_in_system"
    action_description = {preprocess_action(action): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    additional_info_key = list(additional_info.keys())
    additional_info_key += ['Current Observation'] if use_vlm else []
    system_prompt = get_system_prompt(selection=True, with_example=example_in_system, additional_info=additional_info_key)
    current_user_prompt = get_user_prompt(instruction, action_seq, action_dict, object_list, additional_info=additional_info, segmentation=segmentation_prompt)
    
    if not example_in_system:
        example_prompt, example_img_url = get_example_prompt(use_vlm, selection=True)
        user_prompt = example_prompt + current_user_prompt
        obs_url = example_img_url if example_with_image else []
    else:
        user_prompt = current_user_prompt
        obs_url = []
    if current_obs_url is not None:
        obs_url.append(current_obs_url)
        
    model = 'gpt-4o'
    # description_system_prompt = "You are a robot arm in food manipulation scneario. You should focus on your gripper. You need to describe the food manipulation table top scenario from the image."
    # description_user_prompt = "Describe the food manipulation table top scenario from the image. Including what the robot are holding, spoon, knife, fork, or None"
    # scenario_prompt = get_messages(description_system_prompt, description_user_prompt, user_image_url=obs_url[-1])
    # scenario_description = call_openai_api(scenario_prompt, model).choices[0].message.content
    scenario_description = ''
    messages = get_messages(system_prompt, user_prompt, user_image_url=obs_url)
    
    response = call_openai_api(messages, model)
    response_content = response.choices[0].message.content
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
    
    # explanation_system_prompt = "You are a robot arm in food manipulation scneario. You should focus on your gripper. You need to explain why you choose the action."
    # explanation_prompt = get_messages(explanation_system_prompt, current_user_prompt + f"{response_content} \nPlease explain why you choose the last action.", user_image_url=current_obs_url)
    # explanation = call_openai_api(explanation_prompt, model).choices[0].message.content
    print(response_content)
    previous_best_action = max(record[len(action_seq)+1], key=record[len(action_seq)+1].get, default=None) if len(action_seq)+1 in record.keys() else None
    
    try:
        answer = response_content.split("Iteration")[1].split(".")[0][-1]
    except:
        answer = previous_best_action

    
    # print(action_dict)
    if previous_best_action is None or previous_best_action == answer:
        # Write result to record
        lines = response_content.strip().split("\n")
        iteration = None
        processing = False  # Flag to indicate when to start parsing

        for line in lines:
            line = line.strip()
            
            if line.startswith("Iteration"):
                processing = True  # Start processing once "Iteration" is found
                iteration = int(line.split()[1].strip(":"))
            
            elif processing and line.startswith("Output:") and iteration is not None:
                content = line.split("Output:")[1].strip().split(". ")[0]
                char_counts = Counter(content)
                if iteration in record:
                    for char, count in char_counts.items():
                        record[iteration][char] = record[iteration].get(char, 0) + count
                else:
                    record[iteration] = dict(char_counts)
    else:
        print(f"answer: {answer}, previous answer: {previous_best_action}")
        current_answer = response_content.split("Output: ")[1].split("\n")[0].strip()
        reverse_dict = {v:k for k, v in action_dict.items()}
        possible_actions = " or ".join([current_answer, f"{previous_best_action}. {reverse_dict[previous_best_action]}"])
        choose_sys_prompt = get_system_prompt_choose_one(selection=True, with_example=example_in_system, additional_info=additional_info_key)
        choose_user_prompt = get_user_prompt_choose_one(instruction, action_seq, action_dict, object_list, additional_info=additional_info, possible_actions=possible_actions)
        choose_messages = get_messages(choose_sys_prompt, choose_user_prompt, user_image_url=obs_url)
        choose_response_content = call_openai_api(choose_messages, model).choices[0].message.content
        answer = choose_response_content.split("The action I should execute in next iteration is ")[1].split(".")[0][-1]
        if log_folder is not None:
            assert obs_id is not None, "Please provide observation id"
            with open(os.path.join(log_folder, f'correction_{obs_id}.txt'), 'w') as f:
                f.write('\n'.join([
                    '\n[SYS]\n' + choose_sys_prompt, 
                    '\n[USER]\n' + choose_user_prompt, 
                    '\n[RES]\n' + choose_response_content, 
                    # '\n[EXP]\n' + explanation, 
                    '\n[ANSWER]\n' + f"{answer}"
                ]))
        
    print(record)
    # print(messages[0]["content"][0]['text'])
    print('=' * 80)
    # print(top_logprobs)
    answer = {answer: 0}
    semantic = {action_description[key]: np.exp(answer.get(value, float('-inf'))) for key, value in action_dict.items()}
    if use_vlm:
        print(scenario_description)
    # print(explanation)
    # _semantic = {action: np.exp(top_logprobs.get(choice, float('-inf'))) for action, choice in action_dict.items()}
    # semantic = {}
    # for key, val in _semantic.items():
    #     semantic[action_description[key]] = val
    # semantic = {action_description[key]: np.exp(top_logprobs.get(value, float('-inf'))) for key, value in action_dict.items()}
    semantic = sort_scores_dict(semantic)
    # print("here")
    # print(semantic)
    
    if log_folder is not None:
        assert obs_id is not None, "Please provide observation id"
        with open(os.path.join(log_folder, f'prompt_{obs_id}.txt'), 'w') as f:
            f.write('\n'.join([
                '\n[SYS]\n' + system_prompt, 
                '\n[USER]\n' + user_prompt, 
                '\n[DES]\n' + scenario_description, 
                '\n[RES]\n' + response_content, 
                # '\n[EXP]\n' + explanation, 
                '\n[SCORE]\n' + f"{semantic}"
            ]))
    return semantic, record

def get_calibration_data(
    instruction: str, 
    action: str,
    action_list: List[str],
    object_list=None, 
    action_seq=None,
    use_vlm=False
) -> str:
    """
    Given action and instruction to get question and answer pair in text for confidence calibration
    """
    object_list = [preprocess_object(container) for container in object_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    
    action_description = {preprocess_action(action): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    
    system_prompt = get_system_prompt(selection=True, with_example=True, additional_info=['Current Observation'] if use_vlm else [])
    user_prompt = get_user_prompt(instruction, action_seq, action_dict, object_list)
    answer = action_dict[preprocess_action(action)]
    return system_prompt, user_prompt, answer
    
    
if __name__ == '__main__':
    instruction = "Stir the beans in the bowl, then scoop it to the round plate."
    object_list = ["red_bowl (empty)", "white_round_plate (empty)", "green_bowl (with beans)"]
    action_list = ['grasp_spoon', 'take_tool (fork)', 'put_spoon_back', 'put_tool (fork)', 'move_to_green_bowl', 'move_to_red_bowl', 'move_to_white_round_plate', 'scoop', 'fork', 'cut', 'move', 'stir', 'put_food', 'DONE']
    action_seq = ['grasp_spoon', 'move_to_green_bowl', 'stir', 'scoop', 'move_to_white_round_plate', 'put_food', 'put_spoon_back']
    print("SELECTION\n\n")
    system_prompt, user_prompt, answer = get_calibration_data(instruction, 'take_tool (fork)', action_list, object_list, action_seq, True)
    print(system_prompt)
    print(user_prompt)
    print(answer)