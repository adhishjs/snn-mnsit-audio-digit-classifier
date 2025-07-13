import torch



#for training
def load_checkpoint(model, optimizer, filename="parameters.pt"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']



#for testing
def load_model_from_checkpoint(model, checkpoint_path="parameters.pt"):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")