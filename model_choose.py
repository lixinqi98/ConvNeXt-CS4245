from Resnet50_v1 import resnet50_v1
from Resnet50_v2 import resnet50_v2

def model_choose(model_name="ResNet_v1", pretrained: bool = False):
    """
    Resnet_v1: The original version of Resnet50
    """
    print(model_name)
    if model_name == "ResNet_v1":
        print(f"Current Model {model_name}")
        model = resnet50_v1(pretrained)
    if model_name == "ResNet_v2":
        print(f"Current Model {model_name}")
        model = resnet50_v2(pretrained)

    return model