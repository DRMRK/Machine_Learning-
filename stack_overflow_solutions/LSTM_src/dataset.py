import torch 

class QUORADataset:
    
    def __init__(self,question,OpenStatus):
        """
        param question: this is a numpy array
        param OpenStatus: a vector, numpy array
        """
        self.question = question
        self.OpenStatus = OpenStatus
    def __len__(self):
        #  return length of the dataset
        return len(self.question)

    def __getitem__(self,item):
        # for any item (int) return question
        # and targets as torch tensor
        # item is the index of rows
        question = self.question[item,:]
        OpenStatus = self.OpenStatus[item]

        return {
            "question":torch.tensor(question, dtype=torch.long),
            "OpenStatus":torch.tensor(OpenStatus,dtype=torch.float)
        }   
