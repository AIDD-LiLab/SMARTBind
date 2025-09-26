
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
