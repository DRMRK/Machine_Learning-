import torch
import torch.nn as nn
import torch.nn.functional as F


def train(data_loader,model,optimizer, device):
    """ This trains the model for one epoch
    :param data_loader: this is torch dataloader
    :param model: model network
    :param optimizer: torch optimizer
    :param device: device can be cuda or cpu
    """
    # set model to train mode
    model.train()
    total_loss = 0
    total_correct = 0
    for batch in data_loader:
        features, labels = batch
        preds = model(features)

        loss = F.cross_entropy(preds, labels) # Calculate Loss

        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights using Adam
        total_loss += loss.item()
        total_correct+=get_num_correct(preds,labels)
    return total_loss, total_correct

def evaluate(data_loader, model, device):
    # empty lists to store predictions and targets
    #put the model in eval mode
    model.eval()
    total_correct = 0
    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            features, labels = data 
            # make predictions
            predictions = model(features)
            total_correct+=get_num_correct(predictions,labels)
    return (total_correct)
               

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

   
