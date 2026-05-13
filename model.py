import torch
from torch import nn, Tensor

class LowpassRNN(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, conditioned: bool = True):
        super(LowpassRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Le modèle reste basé sur une seule GRUCell, mais on déroule la séquence
        # buffer par buffer dans forward() pour conserver la compatibilité avec
        # l'entraînement et l'évaluation existants.
        input_size = 2 if conditioned else 1
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor, hidden: Tensor):
        # Accepte soit un seul pas de temps en (batch, features),
        # soit un buffer complet en (batch, seq_len, features).
        squeeze_time = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_time = True

        if hidden is None:
            hidden = x.new_zeros(x.size(0), self.hidden_size)

        outputs = []
        for t in range(x.size(1)):
            hidden = self.gru_cell(x[:, t, :], hidden)
            outputs.append(self.fc(hidden).unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        if squeeze_time:
            output = output.squeeze(1)

        return output, hidden