from Resnet50_v1 import resnet50_v1
from Resnet50_v2 import resnet50_v2
from Resnet50_v3 import resnet50_v3
from Resnet50_v4 import resnet50_v4
from Resnet50_v5 import resnet50_v5
from Resnet50_v6 import resnet50_v6
from Resnet50_v7 import resnet50_v7
from Resnet50_v8 import resnet50_v8

def model_choose(model_name="ResNet_v1", pretrained: bool = False):
    """
    Resnet_v1: The original version of Resnet50
    """
    print(f"Current Model {model_name}")
    if model_name == "ResNet_v1":      
        model = resnet50_v1(pretrained)
    if model_name == "ResNet_v2":
        model = resnet50_v2(pretrained)
    if model_name == "ResNet_v3":
        model = resnet50_v3(pretrained)
    if model_name == "ResNet_v4":
        model = resnet50_v4(pretrained)
    if model_name == "ResNet_v5":
        model = resnet50_v5(pretrained)
    if model_name == "ResNet_v6":
        model = resnet50_v6(pretrained)
    if model_name == "ResNet_v7":
        model = resnet50_v7(pretrained)
    if model_name == "ResNet_v8":
        model = resnet50_v8(pretrained)
    return model