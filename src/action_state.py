import math
import torch

def scoop(init_pos: torch.Tensor):
    goal_pos_set = [
        init_pos + torch.tensor([[0.0000, 0.0000, 0.7836]]),
        init_pos + torch.tensor([[-0.0300,  0.0000,  0.7228]]),
        init_pos + torch.tensor([[-0.0299,  0.0109,  0.6728]]),
        init_pos + torch.tensor([[-0.0271,  0.0050,  0.6808]]),
        init_pos + torch.tensor([[-0.0264,  0.0054,  0.6716]]),
        init_pos + torch.tensor([[0.0169, 0.0043, 0.6700]]),
        init_pos + torch.tensor([[-0.0288,  0.0043,  0.6638]]),
        init_pos + torch.tensor([[-0.0295,  0.0043,  0.6638]]),
        init_pos + torch.tensor([[-0.0300,  0.0038,  0.6650]]),
        init_pos + torch.tensor([[-0.0352,  0.0035,  0.6668]]),
        init_pos + torch.tensor([[-0.0505,  0.0031,  0.6698]]),
        init_pos + torch.tensor([[-0.0807,  0.0022,  0.6728]]),
        init_pos + torch.tensor([[-0.1024,  0.0019,  0.6758]]),
        init_pos + torch.tensor([[-0.1016,  0.0021,  0.6708]]),
        init_pos + torch.tensor([[-0.1370,  0.0019,  0.6808]]),
        init_pos + torch.tensor([[-0.2130,  0.0017,  0.7000]])
    ]
    goal_rot_set = [
        torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]]),
        torch.tensor([[ 0.9945,  0.0410, -0.0808,  0.0522]]),
        torch.tensor([[ 0.9953,  0.0393, -0.0701,  0.0542]]),
        torch.tensor([[ 0.9976,  0.0345, -0.0243,  0.0549]]),
        torch.tensor([[ 0.9977,  0.0345, -0.0176,  0.0563]]),
        torch.tensor([[0.9975, 0.0310, 0, 0.0575]]),
        torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]]),
        torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]]),
        torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]]),
        torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]]),
        torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]]),
        torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]]),
        torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]]),
        torch.tensor([[0.9337, 0.0093, 0.4071, 0.0639]]),
        torch.tensor([[0.8848, 0.0023, 0.5116, 0.0641]]),
        torch.tensor([[0.8844, 0.0025, 0.5116, 0.0640]])
    ]
    
    return goal_pos_set, goal_rot_set

def scoop_put(init_pos: torch.Tensor):
    x_shift = 0.03
    z_shift = 0.15
    goal_pos_set = [ 
        init_pos + torch.tensor([[-x_shift-0.1024,  0.0019,  0.6758+z_shift]]), 
        init_pos + torch.tensor([[-x_shift-0.0807,  0.0022,  0.6728+z_shift]]), 
        init_pos + torch.tensor([[-x_shift-0.0505,  0.0031,  0.6698+z_shift]]), 
        init_pos + torch.tensor([[-x_shift-0.0352,  0.0035,  0.6668+z_shift]]), 
        init_pos + torch.tensor([[-x_shift-0.0300,  0.0038,  0.6650+z_shift]]), 
        init_pos + torch.tensor([[-x_shift-0.0295,  0.0043,  0.6638+z_shift]]), 
        init_pos + torch.tensor([[-x_shift-0.0288,  0.0043,  0.6638+z_shift]]), 
        init_pos + torch.tensor([[-x_shift+0.0169, 0.0043, 0.6700+z_shift]]), 
        init_pos + torch.tensor([[-x_shift+0.0000, 0.0000, 0.7836+z_shift]])
    ]
    goal_rot_set = [ 
        torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]]), 
        torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]]), 
        torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]]), 
        torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]]), 
        torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]]), 
        torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]]), 
        torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]]), 
        torch.tensor([[0.9975, 0.0310, 0.0000, 0.0575]]),
        torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]])
    ]
    return goal_pos_set, goal_rot_set

def stir(init_pos: torch.Tensor):
    stage_num = 10
    stir_center = init_pos + torch.Tensor([[0.04, 0, 0]])
    radius = 0.04
    x = -0.04
    middle = False
    step = radius * 4 / stage_num
    goal_pos_set = [init_pos, init_pos + torch.Tensor([0, 0, -0.17])]
    print(goal_pos_set)
    for _ in range(stage_num):
        x += step if not middle else -step
        y = math.sqrt(radius ** 2 - x ** 2)
        y *= 1 if not middle else -1
        if x >= radius and not middle:
            middle = True
        goal_pos_set.append(stir_center + torch.Tensor([x, y, 0]))
    goal_pos_set.append(goal_pos_set[-1] + torch.Tensor([[0., 0., 0.1]]))