from Resnet50_v1 import resnet50

def model_choose(model_name='Resnet_v1', pretrained: bool = False):
    """
    Resnet_v1: The original version of Resnet50
    """
    # print(model_name)
    # if model_name == 'Resnet_v1':
    #     print(model_name)
    model = resnet50(pretrained)
    return model