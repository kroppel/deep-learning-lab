"""
# ResNet
In this lesson we will be implementing all the different pieces of a ResNet, 
in order to finally build a fully parametrized network.
"""
from torchvision.models import resnet18
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import resnet50

"""
## 3x3 Convolution
The first block we will need is a 3x3 Convolutional layer with `padding = 1`
### Exercise 1
Implement the 3x3 Convolutional layer, using the already existing pytorch Conv2d
"""

def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, padding=1) -> nn.Conv2d:
    """3x3 convolution with padding = 1"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups, dilation=dilation, bias=False, padding=1)


"""
## 1x1 Convolution
Another block is a 1x1 convolutional layer, used to change dimensionality in the channels.

### Exercise 2
Implement the 1x1 Convolutional layer, using the already existing pytorch Conv2d
"""

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


"""
## Basic Block
As stated before, we are going to build ResNet through use of blocks.
In particular the BasicBlock (or ResidualBlock) is a combination of 3x3 convolution, batch normalization and, finally, a skip connection summing the input (residual) to the output of the block.

For a visual aid of what a BasicBlock should look like, refer to image:

![ResidualBlock](ResidualBlockSmall.jpg)

### Exercise 3:
Implement the `__init__` and `forward` functions of the BasicBlock.
You can see the BasicBlock as a very simple convolutional network, like we have seen in the previous lessons! 

"""

class BasicBlock(nn.Module):
    expansion: int = 1
    
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """
        param: inplanes (int): the input channels
        param: planes (int): the channels in the final layer
        param: stride (int): the stride used during convolution
        param: downsample (nn.Module): an optional layer that will downsample the input, 
            if necessary, to match the output dimension for the skip connection

        output dimensions: B, planes
        """
        super(BasicBlock, self).__init__()

        # TODO: looking at the structure, implement the layers.
        #  Remember that after each convolutional layer there is 
        #  the batch normalization 2d
        #  HINT: you may want to implement the method called conv3x3()
        self.downsample = downsample
        self.stride = stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ReLU = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)


    def forward(self, x: Tensor) -> Tensor:
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        # TODO: implement the forward step
        out = self.bn2(self.conv2(self.ReLU(self.bn1(self.conv1(x)))))

        # TODO: implement the residual
        print(identity.shape)
        print(out.shape)
        print("####")
        out = self.ReLU(identity + out)

        return out

"""
## BottleNeck
Another type of block used for ResNets (for example ResNet50) is the BottleNeck.
This block changes the dimensions of the original input. Do not worry now about the parameters for reduction of dimensionality.
Your only goal is to implement the parametrized structure of the Bottleneck block, following this image:

![BottleNeck](bottleneckSmall.png)

### Exercise 4:
Implement the Bottleneck block, in particular the `__init__` and `forward` functions.
You can see it as a very simple Convolutional network!
"""

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        """
        param: inplanes (int): the input channels
        param: planes (int): the channels in the middle layer. 
            The output channels will be planes*self.expansion
        param: stride (int): the stride used during convolution
        param: downsample (nn.Module): an optional layer that will downsample the input, 
            if necessary, to match the output dimension for the skip connection

        output dimensions: B, planes*self.expansion
        """
        super(Bottleneck, self).__init__()

        # TODO: looking at the structure, implement the layers.
        #  Remember that after each convolutional layer there is 
        #  the batch normalization 2d
        #  HINT: you may want to implement the method called conv3x3() and conv1x1()
        self.downsample = downsample
        self.stride = stride
        self.conv1 = conv1x1(inplanes, planes)
        # self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ReLU = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        # self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, 4*planes)
        self.bn3 = nn.BatchNorm2d(4*planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        # TODO: implement the forward step
        out = self.ReLU(self.bn1(self.conv1(x)))
        out = self.ReLU(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # TODO: implement the residual
        print(identity.shape)
        print(out.shape)
        print("####")
        out = self.ReLU(identity + out)

        return out

"""
## Resnet
Let's build a ResNet!
Most of the structure is already taken care of, your only job is to understand the structure and implementthe forward pass of the model, in particular ```_forward_impl```.
For a visual aid of the structure and flow of the network, refer to image 

![resnet](Resnet18_arch.png)

### Exercise 5:
Implement the `_forward_impl`, representing the forward pass of the resnet. 
Follow the image for a visual aid of the structure and flow.
"""

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # TODO: Implement the forward step.
        #  take a look to the Resnet18_arch.png 
        #  For resnet50 is the same, as the changes are 
        #  only in the structure of layers

        out_maxpool = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out_layers = self.layer4.forward(
                        self.layer3.forward(
                            self.layer2.forward(
                                self.layer1.forward(out_maxpool))))
        out_avgpool = self.avgpool(out_layers)
        out_fc = self.fc(out_avgpool[:,:,0,0])

        return(out_fc)
        
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

"""
### Testing our network
If everything works correctly and the model is built like the original implementation, 
we should be able to load the weights from the online repository.
"""

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
}

# ResNet18
block_type = BasicBlock
layers = [2, 2, 2, 2]
model = ResNet(block_type, layers)

# Load pre-trained
state_dict = load_state_dict_from_url(model_urls["resnet18"])
model.load_state_dict(state_dict)

xin_fake = torch.rand((1, 3, 224, 224)).type(torch.FloatTensor)
yout_fake = model(xin_fake)

baseline = resnet18(True)
yout_baseline = baseline(xin_fake)

print(f'ResNet18 outputs are the same: {not(False in (yout_baseline == yout_fake))}')

# ResNet50
block_type = Bottleneck
layers = [3, 4, 6, 3]
model = ResNet(block_type, layers)

# Load pre-trained
state_dict = load_state_dict_from_url(model_urls["resnet50"])
model.load_state_dict(state_dict)


xin_fake = torch.rand((1, 3, 224, 224)).type(torch.FloatTensor)
yout_fake = model(xin_fake)

baseline = resnet50(True)
yout_baseline = baseline(xin_fake)

print(f'ResNet50 outputs are the same: {not(False in (yout_baseline == yout_fake))}')
