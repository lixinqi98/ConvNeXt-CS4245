from Resnet50_v1 import resnet50_v1
from Resnet50_v2 import resnet50_v2
from Resnet50_v3 import resnet50_v3
from Resnet50_v4 import resnet50_v4
from Resnet50_v5 import resnet50_v5
from Resnet50_v6 import resnet50_v6
from Resnet50_v7 import resnet50_v7
from Resnet50_v8 import resnet50_v8
from Resnet50_v9 import resnet50_v9
from Resnet50_v10 import resnet50_v10
from Resnet50_v11 import resnet50_v11
from Resnet50_v12 import resnet50_v12
from Resnet50_v13 import resnet50_v13
from Resnet50_v14 import resnet50_v14

def model_choose(model_name="ResNet_v1", pretrained: bool = False):
    """
    Resnet_v1: The original version of Resnet50
    """
    print(f"Current Model {model_name}")
    if model_name == "ResNet_v1":      
        model = resnet50_v1(pretrained)
    elif model_name == "ResNet_v2":
        model = resnet50_v2(pretrained)
    elif model_name == "ResNet_v3":
        model = resnet50_v3(pretrained)
    elif model_name == "ResNet_v4":
        model = resnet50_v4(pretrained)
    elif model_name == "ResNet_v5":
        model = resnet50_v5(pretrained)
    elif model_name == "ResNet_v6":
        model = resnet50_v6(pretrained)
    elif model_name == "ResNet_v7":
        model = resnet50_v7(pretrained)
    elif model_name == "ResNet_v8":
        model = resnet50_v8(pretrained)
    elif model_name == "ResNet_v9":
        model = resnet50_v9(pretrained)
    elif model_name == "ResNet_v10":
        model = resnet50_v10(pretrained)
    elif model_name == "ResNet_v11":
        model = resnet50_v11(pretrained)
    elif model_name == "ResNet_v12":
        model = resnet50_v12(pretrained)
    elif model_name == "ResNet_v13":
        model = resnet50_v13(pretrained)
    elif model_name == "ResNet_v14":
        model = resnet50_v14(pretrained)
    return model