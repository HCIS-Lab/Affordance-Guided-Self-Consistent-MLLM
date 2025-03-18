import os
import socket
import json
from typing import List, Dict

import sys
sys.path.append('/home/hcis-s25/Desktop/yuhong/from_s21/scoop_env/')
# sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env/')

from src.utils import encode_image, decode_image

AFFORDANCE_DESCRIPTION = {
    'Current Observation': "An image of the robot's current environment.",
    'Goal Description': "A description of the subgoal that the robot must accomplish in next iteration.",
    "Additional important information": "A detailed description of the environment and task. Please consider this information when making your decision.",
    "Previous Affordance Feedback": "A record of action names, their failure reasons, and some suggestion from previous iterations. Please consider this information when making your decision.",
}

def preprocess_action(action):
    """
    Remove the parentheses and underscores in the action string
    """
    return action.replace('(', '').replace(')', '').replace('_', ' ')

def preprocess_object(object):
    return object.replace('_', ' ')

def format_action_choices(action_list: List[str]):
    """
    Create a dictionary that maps action to a character
    """
    return {action: chr(ord('A') + i)  for i, action in enumerate(action_list)}

def segmentation_process(rgb_img_path):
    """
    send image to localhost to process image with segmentation model
    """
    host = '127.0.0.1'
    port = 12345
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        data = {'img_path': rgb_img_path}
        s.sendall(json.dumps(data).encode('utf-8'))
        response = s.recv(4096)
        response = response.decode('utf-8')
    if os.path.exists(response):
        return response
    return None
    
def generate_prompt(action_seq, indent=False):
    ret = ""
    for i, action in enumerate(action_seq):
        executed_action = " ".join([f'{j+1}. {action_seq[j]}' for j in range(i)])
        if indent:
            ret += f"""
    Iteration {i+1}:
        Output: {action}"""
        else:
            ret += f"""Iteration {i+1}:
    Output: {action}
"""
    # Input: {executed_action}
    return ret

def get_action_description_prompt():
    '''
    Description of each action
    '''
    action_descrption = {
        'grasp_spoon': "Grasp the spoon from the tool holder. The robot arm must have no tools in the gripper when choosing this action.",
        'put_spoon_back': "Put the spoon back to the tool holder.",
        'move_to_container': "Move to a container for actions like pulling or scooping.",
        'scoop': "Scoop food, with the speed adapted to the food's state. You should select the bowl to scoop with move_to_container.",
        'stir': "Stir the food.",
        'drop_food': "When the robot arm is positioned above a container, drop the food from the spoon into the container.",
        'pull_bowl_closer': "When the gripper is empty, pull the bowl toward the center of the table. You should select the bowl to pull with move_to_container.",
        'open_dumbwaiter': "Open the dumbwaiter door.",
        'close_dumbwaiter': "Close the dumbwaiter door.",
        'put_bowl_into_dumbwaiter': "Place the nearest bowl into the dumbwaiter.",
        'start_dumbwaiter': "Start the dumbwaiter.",
        'DONE': "Indicate that the task is complete."

    }
    
    action_description_prompt = '\n'.join([f'{i + 1}. {action_name}: {action_descrption[action_name]}' for i, action_name in enumerate(action_descrption)])
    return action_description_prompt

def get_key_considerations():
    '''
    Manuanlly define the key considerations for each action
    '''

def get_example_prompt(with_image=False, selection=False):
    system_prompt = ""
    system_image_url = []
    example_path = 'src/semantic/example/text'
    # example_path = '/home/hcis-s25/Desktop/yuhong/from_s21/scoop_env/src/semantic/example/text'
    # example_path = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/semantic/example/text'
    example_id = 1
    for txt in os.listdir(example_path):
        if not txt.endswith('.txt'):
            continue
        txt_file = os.path.join(example_path, txt)
        img_file = txt_file.replace('txt', 'jpg').replace('text', 'image')
        content = ''.join(open(txt_file).readlines()).split('\n\n')
        example_instruction = content[0]
        example_action_seq = content[1].split('\n')
        example_container_list = content[2].split('\n')
        example_action_list = content[3].split('\n')
        if len(content) > 4:
            example_environment = content[4].replace('\n', '\n        ')
        
        example_action_seq = [preprocess_action(action) for action in example_action_seq]
        example_action_list = [preprocess_action(action) for action in example_action_list]
        example_container_list = [preprocess_object(container) for container in example_container_list]
        example_action_dict = format_action_choices(example_action_list)
        # print(example_action_dict)
        if selection:
            # example_action_seq = [example_action_dict[action] for action in example_action_seq] only character
            # example_action_seq = [f"{example_action_dict[action]}. {action}" for action in example_action_seq]
            # example_action_list = [f'{selection}. {action}' for action, selection in example_action_dict.items()]
            example_action_seq = [f"{action}" for action in example_action_seq]
            example_action_list = [f'{action}' for action, selection in example_action_dict.items()]
        system_prompt += f"""Example {example_id}:
    Action list: {example_action_list}
    Initial object list: {example_container_list}
    Instruction: {example_instruction}
    {generate_prompt(example_action_seq, indent=True)}
    
    Explanation:
        {example_environment}

"""
        if with_image and os.path.exists(img_file):
            system_image_url.append(encode_image(img_file))
        example_id += 1
    return system_prompt, system_image_url

def get_system_prompt(selection=False, with_example=True, additional_info=[]):
    '''
    Generate the system prompt for the user to make a decision
    Args:
        selection: whether to use character selection for the action list
        with_example: whether to include examples in the prompt
        additional_info: additional information to include in the prompt, format: List[name of information], including Current Observation, Goal Description
    Returns:
        system_prompt: the text content of the system prompt
        system_image_url: the url of the image used in the examples
    '''
    base_prompt = """# Scenario
You are a robotic arm specialized in food manipulation tasks. Your mission is to complete the assigned task step-by-step by selecting the most appropriate actions from the provided list. Your decisions should balance precision, safety, efficiency, and task progression.
Take the previous actionns into consideration and choose the best actions for the remaining sequence from the action list.
You should describe the reasoning behind your decision and consider the high-level goal of the task before making a choice.

# Additional Knowledges
1. Scooping guidelines
A single scoop should be done by selecting [move_to_container(with food), scoop, move_to_container(destination), drop_food] when the spoon is on the gripper.
2. Collision Avoidance
If an action risks a collision or task failure, pull the bowl to a safer location before proceeding.
3. Scooping limitations
Avoid scooping from bowls with insufficient food (e.g., only a few beans).
If a bowl is too far, pull it closer before attempting to scoop.
"""
    additional_info_description = AFFORDANCE_DESCRIPTION
    additional_info_prompt = '\n'.join([f"{name}: {additional_info_description[name]}" for name in additional_info])
    additional_info_prompt += '\n' if additional_info else ''
    system_prompt = f"""{base_prompt}
# Action Description
{get_action_description_prompt()}

# Scenario Format
You will be presented with a single scenario containing the following details:
Action List: A list of all actions that the robot can perform, formatted as character. action.
Initial Object List: A detailed inventory of objects present in the environment, formatted as container_name (food inside).
Instruction: The high-level task or goal that the robot must accomplish.
Iterative Previous Actions: A chronological record of the actions the robot has executed in prior iterations.
{additional_info_prompt}
# Input Format
You will be provided with several examples, each illustrating a unique scenario in the format described above.
Following these, another scenario will be presented, requiring you to deduce and choose the next optimal action.

# Output Requirements
Select and output some actions from the provided Action List in your task as the actions to execute in order.
The response should exclude all formatting characters such as backticks, quotes, or additional symbols.
You should provide a sequence of action as answer, starting from current iteration until selecting DONE.
 
Format the first line of your response strictly as: Description: [your description].
Format the rest of the line of your response strictly as: "
Iteration [number]: 
    Output: [character]. [action]". Please use the format in the examples as a reference.

"""
    if with_example:
        system_prompt += "\n# Examples\n"
        example_system_prompt, _ = get_example_prompt(with_image=False, selection=selection)
        system_prompt += example_system_prompt
    return system_prompt

def get_system_prompt_choose_one(selection=False, with_example=True, additional_info: List=[]):
    '''
    Generate the system prompt for the user to make a decision
    Args:
        selection: whether to use character selection for the action list
        with_example: whether to include examples in the prompt
        additional_info: additional information to include in the prompt, format: List[name of information], including Current Observation, Goal Description
    Returns:
        system_prompt: the text content of the system prompt
        system_image_url: the url of the image used in the examples
    '''
    base_prompt = """# Scenario
You are a robotic arm specialized in food manipulation tasks. Your mission is to complete the assigned task step-by-step by selecting the most appropriate actions from the provided list. Your decisions should balance precision, safety, efficiency, and task progression.
You will be provided with a sequence which the robot already executed and a few actions that you can choose for the next iteration at the end of the prompt. You should assume all the action is executed successfully.
You should describe the reasoning behind your decision and consider the high-level goal of the task before making a choice.

# Additional Knowledges
1. Scooping guidelines
A single scoop should be done by selecting [move_to_container(with food), scoop, move_to_container(destination), drop_food] when the spoon is on the gripper.
2. Collision Avoidance
If an action risks a collision or task failure, pull the bowl to a safer location before proceeding.
3. Scooping limitations
Avoid scooping from bowls with insufficient food (e.g., only a few beans).
If a bowl is too far, pull it closer before attempting to scoop.
"""
    additional_info_description = AFFORDANCE_DESCRIPTION
    additional_info_prompt = '\n'.join([f"{name}: {additional_info_description[name]}" for name in additional_info])
    additional_info_prompt += '\n' if additional_info else ''
    system_prompt = f"""{base_prompt}
# Action Description
{get_action_description_prompt()}

# Scenario Format
You will be presented with a single scenario containing the following details:
Action List: A list of all actions that the robot can perform, formatted as character. action.
Initial Object List: A detailed inventory of objects present in the environment, formatted as container_name (food inside).
Instruction: The high-level task or goal that the robot must accomplish.
Iterative Previous Actions: A chronological record of the actions the robot has executed in prior iterations.
{additional_info_prompt}
# Input Format
You will be provided with several examples, each illustrating a unique scenario in the format described above.
Following these, another scenario will be presented, requiring you to deduce and choose the next optimal action.

# Output Requirements
You should only select and output actions that is mentioned in the question at the end of the prompt.
The response should exclude all formatting characters such as backticks, quotes, or additional symbols.
 
Format the first line of your response strictly as: Description: [your description].
Format the second line of your response strictly as: The action I should execute in next iteration is [character]. [action name]

"""
    if with_example:
        system_prompt += "\n# Examples\n"
        example_system_prompt, _ = get_example_prompt(with_image=False, selection=selection)
        system_prompt += example_system_prompt
    return system_prompt

def get_user_prompt(instruction, action_seq, action_dict, container_list, additional_info: Dict={}, segmentation=False) -> str:
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    
    action_choices = [f"{v}. {k}" for k, v in action_dict.items()]
    action_seq_choices = [f"{action_dict[action]}. {action}" for action in action_seq]
    
    additional_info_prompt = '\n'.join([f"{name}: {info}" for name, info in additional_info.items()]) if additional_info else ''
    additional_info_prompt += '\n' if additional_info else ''
    segmentation_prompt = "Please focus on the segmentation result of the robot to make the decision.\n" if segmentation else ""
    
    user_prompt = f"""Your task:
    Action list: {action_choices}
    Initial object list: {container_list}
    Instruction: {instruction}
    {additional_info_prompt}{segmentation_prompt}{generate_prompt(action_seq_choices, indent=True)}
    Iteration {len(action_seq_choices)+1}:
        Output: """
    return user_prompt

def get_user_prompt_choose_one(instruction, action_seq, action_dict, container_list, additional_info={}, possible_actions=str) -> str:
    print(additional_info)
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    # possible_actions = [preprocess_action(action) for action in possible_actions]
    action_choices = [f"{v}. {k}" for k, v in action_dict.items()]
    action_seq_choices = [f"{action_dict[action]}. {action}" for action in action_seq]
    
    additional_info_prompt = '\n'.join([f"{name}: {info}" for name, info in additional_info.items()]) if additional_info else ''
    additional_info_prompt += '\n' if additional_info else ''
    
    # possible_actions = ' or '.join([f"{action_dict[action]. action}" for action in possible_actions])
    user_prompt = f"""Your task:
    Action list: {action_choices}
    Initial object list: {container_list}
    Instruction: {instruction}
    {additional_info_prompt}{generate_prompt(action_seq_choices, indent=True)}
    Which action should I execute next? {possible_actions}
    """
    return user_prompt

def next_action_prompt(instruction, action_seq):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties or information should be take into consideration before deciding the next action for a robot about a food manipulation task. Consider all previous actions and their outcomes when deciding. Please do not decide the action and specify what you should achieve in the next step and what you should take into consideration to achieve the goal in high level. Provide your answer in 50 words or fewer.
{get_action_description_prompt()}"""
    action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
    user_prompt = f"""Instruction: {instruction}
Previous actions:
{action_sequence}"""
    return system_prompt, user_prompt


def extract_from_choice_prompt(instruction, action_seq, choices, container_list):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties (extrinsic, amount, distribution) or information that should be take into consideration in a food manipulation task. You will be given some candidate actions, please think through all the choices as thorough as possible and list what I should know in order to make the choice. Please do not choose the action directly.
{get_action_description_prompt()}"""
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    
    action_sequence = ", ".join([f"{i+1}. {action}" for i, action in enumerate(action_seq)])
    user_prompt = f"""Instruction: {instruction}
Previous actions: {action_sequence}
Object list: {', '.join(container_list)}
Possible next actions: {', '.join(choices)}"""
    return system_prompt, user_prompt

def extract_important_information_prompt(instruction, important_considerations, object_list=None):
    object_list = [preprocess_object(container) for container in object_list]
    system_prompt = f"""You are a great observer that can describe the environment in detail. Given an image of the food manipulation scenario, the overall goal, the object list, and the key considerations of determining the next move, extract the important information from the image and the instruction. Please provide your answer in 50 words or fewer."""
    user_prompt = f"""Overall goal: {instruction}
Key considerations: {important_considerations}
Object list: {', '.join(object_list)}"""
    return system_prompt, user_prompt

def extract_from_choice_prompt_selection(instruction, action_seq, action_candidate, action_dict, container_list):
    system_prompt = f"""You are a smart assistant tasked with identifying what food properties (extrinsic, amount, distribution) or information that should be take into consideration in a food manipulation task. You will be given some candidate actions, please think through all the choices as thorough as possible and list what I should know in order to make the choice. Please do not choose the action directly.
{get_action_description_prompt()}"""
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    
    action_sequence = ", ".join([f"{action_dict[action]}. {action}" for action in action_seq])
    action_choices = [f"{action_dict[action]}. {action}" for action in action_candidate]
    user_prompt = f"""Instruction: {instruction}
Previous actions: {action_sequence}
Object list: {', '.join(container_list)}
Possible next actions: {', '.join(action_choices)}"""
    return system_prompt, user_prompt

def extract_important_information_prompt_selection(instruction, important_considerations, object_list):
    object_list = [preprocess_object(container) for container in object_list]
    system_prompt = f"""You are a great observer that can describe the environment in detail. Given an image of the food manipulation scenario, the overall goal, the object list, and the key considerations of determining the next move, extract the important information from the image and the instruction. Please provide your description with important information in 50 words or fewer without choosing the action directly."""
    user_prompt = f"""Overall goal: {instruction}
Key considerations: {important_considerations}
Object list: {', '.join(object_list)}"""
    return system_prompt, user_prompt

def next_goal_description_prompt(instruction, action_seq, container_list):
    container_list = [preprocess_object(container) for container in container_list]
    action_seq = [preprocess_action(action) for action in action_seq]
    action_sequence = ", ".join([action for action in action_seq])
    system_prompt = f"""You are a smart assistant tasked with summarizing current stage of a food manipulation task and reasonig about the next subgoal of the task in one step. Given instruction, previous action sequence, list of containers in the scenario, and current observation. Please specify what you should achieve in the next step and explain the reason. If you think the goal specified by the instruction is achieved, tell me you think the task is done and why. Do not choose the action directly and provide your answer in 50 words or fewer."""
    user_prompt =  f"""Instruction: {instruction}
Previous actions: {action_sequence}
Object list: {', '.join(container_list)}
Current Observation: An image of the robot's current environment.
"""
    return system_prompt, user_prompt

def choose_from_information(instruction, important_information, action_candidate, action_dict, object_list):
    object_list = [preprocess_object(container) for container in object_list]
    action_choices = [f"{action_dict[action]}. {action}" for action in action_candidate]
    system_prompt = f"""You are a decision maker that can choose the next action based on the information extracted from the image and the instruction. Given the information extracted from the image and the instruction, choose the next action from the action list. Please only answer a single character from the action list."""
    user_prompt = f"""Overall goal: {instruction}
Important information: {important_information}
Object list: {', '.join(object_list)}
Action choices: {', '.join(action_choices)}"""
    return system_prompt, user_prompt
    

def get_messages(system_prompt, user_prompt, system_image_url=None, user_image_url=None):
    """system_image_url is not supported yet"""
    if system_image_url is not None:
        if isinstance(system_image_url, list):
            system_content = [{"type": "text", "text": system_prompt}]
            for url in system_image_url:
                system_content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        elif isinstance(system_image_url, str):
            system_content = [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": system_image_url, "detail": "high"}}]
        else:
            raise ValueError("system_image_url should be a list or a string")
    else:
        system_content = [{"type": "text", "text": system_prompt}]
    if user_image_url is not None:
        if isinstance(user_image_url, list):
            user_content = [{"type": "text", "text": user_prompt}]
            for url in user_image_url:
                user_content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        elif isinstance(user_image_url, str):
            user_content = [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": user_image_url, "detail": "high"}}]
        else:
            raise ValueError("user_image_url should be a list or a string")
    else:
        user_content = [{"type": "text", "text": user_prompt}]
    messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
    ]
    return messages

if __name__ == '__main__':
    # system_prompt, system_image_url = get_system_prompt(with_obs=True, selection=True)
    # print(system_prompt)
    # for i, url in enumerate(system_image_url):
    #     img = decode_image(url, f'test_{i + 1}.png')
    # response = segmentation_process('/home/hcis-s17/multimodal_manipulation/scoop_env/src/semantic/output.png')
    # print(response)
    # print(get_system_prompt(with_obs=True, selection=True, with_example=True)[0])
    pass