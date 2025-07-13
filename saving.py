import torch


def save_checkpoint(model, optimizer, epoch, filename="parameters.pt"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, filename)
