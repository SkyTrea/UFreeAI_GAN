from torch.functional import Tensor
from torchvision import datasets, transforms
from torchvision.transforms.transforms import ToTensor

train_data = datasets.MNIST('../../data/mnist', train=True,
                             transform = transforms.ToTensor(),
                             download=True)
test_data = datasets.MNIST('../../data/mnist', train=False,
                             transform=transforms.ToTensor(),
                             download=True)