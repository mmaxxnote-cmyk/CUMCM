import torch
import torch.nn as nn



class LSTMModel(nn.Module):


    def __init__(
            self,
            input_size,
            hidden_size=64,
            num_layers=1
    ):

        super().__init__()


        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )


        self.fc = nn.Linear(
            hidden_size,
            1
        )



    def forward(self, x):


        out, (h,c)=self.lstm(x)


        # 取最后时间步

        out = out[:,-1,:]


        result=self.fc(out)


        return result