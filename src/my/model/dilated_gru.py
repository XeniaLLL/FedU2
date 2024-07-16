import torch
import torch.nn
import torch.nn.functional


class GRUEncoder(torch.nn.Module):
    """3-layers bidirectional dilated RNN with GRU units."""

    def __init__(self, hidden_sizes: tuple[int, int, int] = (100, 50, 50), dilations: tuple[int, int, int] = (1, 4, 16)):
        super().__init__()
        self.forward1 = torch.nn.GRUCell(1, hidden_sizes[0])
        self.forward2 = torch.nn.GRUCell(hidden_sizes[0], hidden_sizes[1])
        self.forward3 = torch.nn.GRUCell(hidden_sizes[1], hidden_sizes[2])
        self.backward1 = torch.nn.GRUCell(1, hidden_sizes[0])
        self.backward2 = torch.nn.GRUCell(hidden_sizes[0], hidden_sizes[1])
        self.backward3 = torch.nn.GRUCell(hidden_sizes[1], hidden_sizes[2])
        self.dilations = dilations

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: shape(batch_size, sequence_length, 1)
        :return: shape(batch_size, sum(hidden_sizes) * 2)
        """
        hiddens: list[torch.Tensor] = []  # tensors with shape(batch_size, hidden_size)
        x = data
        for cell, dilation in zip((self.forward1, self.forward2, self.forward3), self.dilations):
            x = dilated_gru(cell, x, dilation)
            hn = x[:, -1, :]
            hiddens.append(hn)
        x = torch.flip(data, dims=(1,))
        for cell, dilation in zip((self.backward1, self.backward2, self.backward3), self.dilations):
            x = dilated_gru(cell, x, dilation)
            hn = x[:, -1, :]
            hiddens.append(hn)
        return torch.concat(hiddens, dim=1)


def dilated_gru(cell: torch.nn.GRUCell, data: torch.Tensor, dilation: int) -> torch.Tensor:
    """
    :param data: shape(batch_size, sequence_length, input_size)
    :return: output (hidden state) of each step, shape(batch_size, sequence_length, hidden_size)
    """
    hiddens: list[torch.Tensor] = []  # list of tensors with shape(batch_size, hidden_size)
    seq_len = data.shape[1]
    for i in range(seq_len):
        h = cell.forward(data[:, i, :], hiddens[i - dilation] if i - dilation >= 0 else None)
        hiddens.append(h)
    output = torch.concat([h.unsqueeze(dim=1) for h in hiddens], dim=1)  # shape(batch_size, seq_len, hidden_size)
    return output
