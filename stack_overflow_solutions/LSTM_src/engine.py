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
        # fetch question and OpenStatus from the dict
        question = data["question"]
        OpenStatus = data["OpenStatus"]
        # move the data to device that we want to use
        question = question.to(device, dtype=torch.long)
        OpenStatus = OpenStatus.to(device, dtype=torch.float)
        # clear the gradients
        optimizer.zero_grad()
        # make predictions from the model
        predictions = model(question)
        # calculate the loss
        #loss = nn.BCELoss()(
        #    predictions,
        #    OpenStatus.view(-1, 1)
        #    )
        # calculate the loss
        weight =torch.tensor([33.33,1.1]).cuda()
        criterion = nn.BCELoss(reduction='none')
        loss = criterion(predictions,OpenStatus.view(-1, 1))
        loss= loss*weight
        loss =loss.mean()

        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()
        # single optimization step
        optimizer.step()


def evaluate(data_loader,model,device):
    
    # initialize empty array to store predictions
    # and OpenStatus
    final_predictions = []
    final_OpenStatus = []
    # put model in eval mode
    model.eval()
        # disable gradient calculations
    with torch.no_grad():
        for data in data_loader:
            question = data["question"]
            OpenStatus = data["OpenStatus"]

            question=question.to(device,dtype=torch.long)
            OpenStatus =OpenStatus.to(device,dtype=torch.float)

            predictions = model(question)

            # move predictions and OpenStatus to list
            predictions = predictions.cpu().numpy().tolist()
            OpenStatus = data["OpenStatus"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_OpenStatus.extend(OpenStatus)

    return final_predictions, final_OpenStatus        