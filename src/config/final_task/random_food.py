import yaml
import os
import random
foodname_dict = {
    'sweet potato balls': 'darkorange',
    'taro balls': 'plum',
    'purple sweet potato balls': 'indigo',
    'red tanyuan': 'lightcoral',
    'white tanyuan': 'white',
    'bubble': 'black',
    'mung beans': 'coral',
    'red beans': 'brown',
    'kidney beans': 'coral',
}
yaml_files = [file for file in os.listdir() if file.endswith('.yaml')]
for file in yaml_files:
    with open(file, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in data.items():
        for i, config in v.items():
            instruction = config['instruction']
            origin_foodname = None
            fix_food = random.choice(list(foodname_dict.keys()))
            for idx, container in enumerate(config['containers']):
                origin_food = container['food']
                food = origin_food.copy()
                if len(origin_food['foodname']) > 0:
                    in_instruction = False
                    food_num = len(origin_food['foodname'])
                    for foodname in origin_food['foodname']:
                        foodname = foodname + ' beans'
                        if foodname in instruction:
                            in_instruction = True
                            break
                    random_food = [random.choice(list(foodname_dict.keys())) for _ in range(food_num)]
                    if in_instruction:
                        random_food[0] = fix_food
                        origin_foodname = foodname
                        revise_instruction = True
                    if not in_instruction:
                        while fix_food in random_food:
                            random_food = [random.choice(list(foodname_dict.keys())) for _ in range(food_num)]
                    food['foodname'] = random_food
                    food['colorcode'] = [foodname_dict[foodname] for foodname in random_food]
                container['food'] = food
                config['containers'][idx] = container
            if origin_foodname is not None:
                instruction = instruction.replace(origin_foodname, fix_food)
                config['instruction'] = instruction
            v[i] = config
        data[k] = v
    with open(file.replace('.yaml', '_temp.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=None, sort_keys=False)
            
