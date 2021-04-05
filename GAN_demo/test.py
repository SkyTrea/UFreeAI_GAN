import torch
import cv2
import torch.nn.functional as F
from train import LeNet  ##重要，虽然显示灰色(即在次代码中没用到)，但若没有引入这个模型代码，加载模型时会找不到模型
from torch.autograd import Variable
# from torchvision import datasets, transforms
import numpy as np

# import torchvision
# from skimage import io,transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model/model.pth')  # 加载模型
model = model.to(device)
model.eval()  # 把模型转为test模式
img = cv2.imread("./6.jpg")  # 读取要预测的图片
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
img = np.array(img).astype(np.float32)
img = np.expand_dims(img, 0)
img = np.expand_dims(img, 0)  # 扩展后，为[1，1，28，28]
img = torch.from_numpy(img)
img = img.to(device)
output = model(Variable(img))
prob = F.softmax(output, dim=1)
prob = Variable(prob)
prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
print(prob)  # prob是10个分类的概率
pred = np.argmax(prob)  # 选出概率最大的一个
print(pred.item())
