import torch

class FertilzerModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(3, 3)
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(50, 3)
        

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.activation(out)
        out = self.linear(out)
        
        return out