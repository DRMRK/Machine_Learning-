import torch 

class QUORADataset:
    
    def __init__(self,BodyMarkdown,OpenStatus):
        """
        param BodyMarkdown: this is a numpy array
        param OpenStatus: a vector, numpy array
        """
        self.BodyMarkdown = BodyMarkdown
        self.OpenStatus = OpenStatus
    def __len__(self):
        #  return length of the dataset
        return len(self.BodyMarkdown)

    def __getitem__(self,item):
        # for any item (int) return BodyMarkdown
        # and targets as torch tensor
        # item is the index of rows
        BodyMarkdown = self.BodyMarkdown[item,:]
        OpenStatus = self.OpenStatus[item]

        return {
            "BodyMarkdown":torch.tensor(BodyMarkdown, dtype=torch.long),
            "OpenStatus":torch.tensor(OpenStatus,dtype=torch.long)
        }   

