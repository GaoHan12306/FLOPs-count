# Author:Han
# @Time : 2019/5/20 17:23

import torch
from torchvision import models
from ptflops import get_model_complexity_info

device = torch.device("cpu")
net = models.AlexNet()
flops, params = get_model_complexity_info(net, (3, 224, 224), True, True)
print('Flops:' + flops)
print('Params:' + params)

model = str(net)  # 将模型强制转换成字符串以便写入文件
file = open("Alexnet.txt", 'w')
file.write(model)
