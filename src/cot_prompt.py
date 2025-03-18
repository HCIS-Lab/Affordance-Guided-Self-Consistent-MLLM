import os
import datetime
import sys

import numpy as np
sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env')
from src.semantic.openai_client import call_openai_api
from src.semantic.utils import *
from src.utils import encode_image, sort_scores_dict


def cot1(
    instruction: str, 
    container_list=None, 
    action_list=["scoop", "fork", "cut", "move", "stir", "DONE"], 
    action_seq=None, 
    use_vlm=False, 
    obs_url=None,
    log_folder=None,
    obs_id=None,
) -> dict:
 
    
    container_list = [container.replace('_', ' ') for container in container_list]
    action_seq = [action.replace('(', '').replace(')', '').replace('_', ' ') for action in action_seq]
    
    
    ## Get the next action description
    system_prompt, user_prompt = next_action_prompt(action_seq, instruction)   
    messages = get_messages(system_prompt, user_prompt)
    next_action_description = call_openai_api(messages).choices[0].message.content
    
    print(next_action_description)
    
    ## Logging
    os.makedirs(log_folder, exist_ok=True)
    with open(os.path.join(log_folder, f"{obs_id}_next_action_description.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + next_action_description
        ]))
        
    ## Get the important information from the image
    system_prompt, user_prompt = extract_important_information_prompt(instruction, next_action_description)
    messages = get_messages(system_prompt, user_prompt, user_prompt=obs_url)
    important_information = call_openai_api(messages).choices[0].message.content
    
    print(important_information)
    
    ## Logging
    with open(os.path.join(log_folder, f"{obs_id}_important_information.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + important_information
        ]))
    
    return

def cot2(
    instruction: str, 
    container_list, 
    action_list,
    action_seq=None, 
    obs_url=None,
    action_candidate=[],
    log_folder=None,
    obs_id=None,
) -> dict:
        
    '''
    This is the second version
    1. summarize the important information to make the next decision given the instruction and the current action sequence (no image)
    2. extract the important information from the image to make the next decision given the instruction and the current action sequence (with image)
    3. make decision!
    '''
    action_description = {preprocess_action(action): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    action_candidate = [preprocess_action(action) for action in action_candidate]
    print(action_dict)

    ## Get the next action description
    system_prompt, user_prompt = extract_from_choice_prompt_selection(instruction, action_seq, action_candidate, action_dict, container_list)   
    messages = get_messages(system_prompt, user_prompt)
    key_consideration = call_openai_api(messages).choices[0].message.content
    
    print(key_consideration)
    
    ## Logging
    os.makedirs(log_folder, exist_ok=True)
    with open(os.path.join(log_folder, f"{obs_id}_cot_next_action_description.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + key_consideration
        ]))
    
    ## Get the important information from the image
    system_prompt, user_prompt = extract_important_information_prompt_selection(instruction, key_consideration, container_list)
    messages = get_messages(system_prompt, user_prompt, user_image_url=obs_url)
    important_information = call_openai_api(messages).choices[0].message.content
    
    print(important_information)
    
    ## Logging
    with open(os.path.join(log_folder, f"{obs_id}_cot_important_information.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + important_information
        ]))
    
    system_prompt, user_prompt = choose_from_information(instruction, important_information, action_candidate, action_dict, container_list)
    messages = get_messages(system_prompt, user_prompt)
    response = call_openai_api(messages)
    final_choice = response.choices[0].message.content
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
    action_candidate_probs = {action_description[action]: np.exp(top_logprobs.get(action_dict[action], float('-inf'))) for action in action_candidate}
    action_candidate_probs = sort_scores_dict(action_candidate_probs)
    with open(os.path.join(log_folder, f"{obs_id}_cot_final_choice.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + final_choice,
            '\n[TOP_LOGPROBS]\n' + str(top_logprobs),
            '\n[PROBS]\n' + str(action_candidate_probs)
        ]))
    
    return action_candidate_probs

def cot_baseline1(
    instruction: str, 
    container_list, 
    action_list,
    action_seq=None, 
    obs_url=None,
    action_candidate=[],
    log_folder=None,
    obs_id=None,
) -> dict:
    '''
    Chain of Thought baseline 1
    1. summarize the important information to make the next decision given the instruction and the current action sequence (no image)
    2. extract the important information from the image to make the next decision given the instruction and the current action sequence (with image)
    3. make decision (with example and image input in user prompt)
    '''
    decode_image(obs_url, f"{log_folder}/observation_{obs_id}.png")
    action_description = {preprocess_action(action): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    action_candidate = [preprocess_action(action) for action in action_candidate]
    
    ## Get the next action description
    system_prompt, user_prompt = extract_from_choice_prompt_selection(instruction, action_seq, action_candidate, action_dict, container_list)   
    messages = get_messages(system_prompt, user_prompt)
    key_consideration = call_openai_api(messages).choices[0].message.content
    
    
    ## Logging
    os.makedirs(log_folder, exist_ok=True)
    with open(os.path.join(log_folder, f"{obs_id}_cot_next_action_description.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + key_consideration
        ]))
    
    ## Get the important information from the image
    system_prompt, user_prompt = extract_important_information_prompt_selection(instruction, key_consideration, container_list)
    messages = get_messages(system_prompt, user_prompt, user_image_url=obs_url)
    important_information = call_openai_api(messages).choices[0].message.content
    
    
    ## Logging
    with open(os.path.join(log_folder, f"{obs_id}_cot_important_information.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + important_information
        ]))
    
    # difference between cot2 and cot_baseline1
    system_prompt, user_prompt = choose_from_information(instruction, important_information, action_candidate, action_dict, container_list)
    example_prompt, example_image_url = get_example_prompt(with_image=True, selection=True)
    user_prompt = f"Here are some decision making examples {example_prompt}" + user_prompt
    messages = get_messages(system_prompt, user_prompt, user_image_url=example_image_url)
    
    response = call_openai_api(messages)
    final_choice = response.choices[0].message.content
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
    action_candidate_probs = {action_description[action]: np.exp(top_logprobs.get(action_dict[action], float('-inf'))) for action in action_candidate}
    action_candidate_probs = sort_scores_dict(action_candidate_probs)
    with open(os.path.join(log_folder, f"{obs_id}_cot_final_choice.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + final_choice,
            '\n[TOP_LOGPROBS]\n' + str(top_logprobs),
            '\n[PROBS]\n' + str(action_candidate_probs)
        ]))
    
    return action_candidate_probs


def cot_baseline2(
    instruction: str, 
    container_list, 
    action_list,
    action_seq=None, 
    obs_url=None,
    action_candidate=[],
    log_folder=None,
    obs_id=None,
) -> dict:
    '''
    Chain of Thought baseline 2
    1. make LLM think what goal should be achieved next given the instruction and the current action sequence (with image)
    2. make decision based on the next goal description(use original system & user prompt, no image)
    '''
    
    decode_image(obs_url, f"{log_folder}/observation_{obs_id}.png")
    action_description = {preprocess_action(action): action for action in action_list}
    action_dict = format_action_choices(list(action_description.keys()))
    action_candidate = [preprocess_action(action) for action in action_candidate]
    

    ## Get the important information from the image
    system_prompt, user_prompt = next_goal_description_prompt(instruction, action_seq, container_list)
    messages = get_messages(system_prompt, user_prompt, user_image_url=obs_url)
    next_goal_description = call_openai_api(messages).choices[0].message.content
    
    
    ## Logging
    with open(os.path.join(log_folder, f"{obs_id}_cot_next_goal_description.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + next_goal_description
        ]))
    
    additional_info = {
        'Goal Description': next_goal_description
    }
    system_prompt = get_system_prompt(selection=True, with_example=True, additional_info=list(additional_info.keys()))
    user_prompt = get_user_prompt(instruction, action_seq, action_dict, container_list, additional_info=additional_info)
    messages = get_messages(system_prompt, user_prompt)
    
    response = call_openai_api(messages)
    response_content = response.choices[0].message.content
    final_choice = response_content.split('\n')[0].split('. ')
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs = {top_logprob.token: top_logprob.logprob for top_logprob in top_logprobs}
    character_action_map_error = False
    if len(final_choice) == 2:
        action = ' '.join(final_choice[1].split())
        character = final_choice[0]
        if action in action_dict.keys() and character != action_dict[action]:
            wrong_score = top_logprobs.pop(action_dict[action], float('-inf'))
            top_logprobs[action_dict[action]] = top_logprobs.pop(character)
            top_logprobs[character] = wrong_score
            character_action_map_error = True
    action_candidate_probs = {action_description[action]: np.exp(top_logprobs.get(action_dict[action], float('-inf'))) for action in action_candidate}
    action_candidate_probs = sort_scores_dict(action_candidate_probs)
    with open(os.path.join(log_folder, f"{obs_id}_cot_final_choice{'_mapping_error' if character_action_map_error else ''}.txt"), 'w') as f:
        f.write('\n'.join([
            '\n[SYS]\n' + system_prompt, 
            '\n[USER]\n' + user_prompt, 
            '\n[ANS]\n' + response_content,
            '\n[TOP_LOGPROBS]\n' + str(top_logprobs),
            '\n[PROBS]\n' + str(action_candidate_probs)
        ]))
    
    return action_candidate_probs

if __name__ == '__main__':
    root = '/home/hcis-s17/multimodal_manipulation/scoop_env/'
    instruction = "I need one scoop of green beans, and I'm in a hurry. Can you make it quick?"
    container_list = ['white_bowl (with green beans)', 'red_bowl (with green beans)', 'skyblue_bowl (with green beans)', 'blue_bowl (with black beans)', 'yellow_bowl (empty)']
    action_list = ['scoop', 'fork', 'cut', 'stir', 'put_food', 'pull_bowl_closer', 'DONE', 'grasp_spoon', 'take_tool (fork)', 'take_tool (knife)', 'put_spoon_back', 'put_tool (fork)', 'put_tool (knife)', 'move_to_white_bowl', 'move_to_red_bowl', 'move_to_skyblue_bowl', 'move_to_blue_bowl', 'move_to_yellow_bowl']
    action_seq = ['grasp_spoon']
    use_vlm = True
    log_folder = f"{root}experiment_log/correction2/" + '{date:%Y-%m-%d_%H:%M:%S}'.format( date=datetime.datetime.now())
    obs_id = 1
    obs_url = encode_image(f'{root}experiment_log/archive/hard_9/20241120000205/observation_1.png')
    action_candidate = ['move_to_white_bowl', 'move_to_red_bowl', 'move_to_skyblue_bowl']
    probs = cot2(instruction, container_list, action_list, action_seq, use_vlm, obs_url=obs_url, log_folder=log_folder, obs_id=obs_id, action_candidate=action_candidate)
    print(probs)