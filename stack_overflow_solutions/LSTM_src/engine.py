import torch 
import torch.nn as nn

def train(data_loader, model,optimizer,device):
    """
    :param data_loader: torch dataloader
    :param model: model (lstm)
    :param optimizer: torch optimizer
    :param device: 'cuda' or 'cpu'

    """
    # set model to training mode
    model.train()
    # go through batches of data in dataloader
    for data in data_loader:
        # get BodyMarkdown and OpenStatus
        BodyMarkdown = data["BodyMarkdown"]
        OpenStatus = data["OpenStatus"]

        # move data to device
        BodyMarkdown=BodyMarkdown.to(device,dtype=torch.long)
        OpenStatus =OpenStatus.to(device,dtype=torch.float)

        #clear the gradients
        optimizer.zero_grad()

        # make predictions from model
        predictions = model(BodyMarkdown)

        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(
            predictions,
            OpenStatus.view(-1,1)
        )
        # compute gradiet w.r.t to all trainable
        #parameters
        loss.backward()
        # single optimization step
        optimizer.step()

def evaluate(data_loader,model,device):
    # initialize empty array to store predictions
    # and OpenStatus
    final_predictions = []
    final_targets = []
    # put model in eval mode
    model.eval()
        # disable gradient calculations
    with torch.no_grad():
        for data in data_loader:
            BodyMarkdown = data["BodyMarkdown"]
            OpenStatus = data["OpenStatus"]

            BodyMarkdown=BodyMarkdown.to(device,dtype=torch.long)
            OpenStatus =OpenStatus.to(device,dtype=torch.long)

            predictions = model(BodyMarkdown)

            # move predictions and targets to list
            predictions = predictions.cpu().numpy().tolist()
            OpenStatus = data["OpenStatus"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(OpenStatus)

    return final_predictions, final_targets        