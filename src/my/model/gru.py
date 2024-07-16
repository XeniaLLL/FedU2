import torch
import torch.nn
import torch.nn.functional


class GRUEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 3, input_size: int = 1):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: shape(batch_size, sequence_length, input_size)
        :return: shape(batch_size, hidden_size)
        """
        output, _ = self.gru(data)  # output: shape(batch_size, sequence_length, 2 * hidden_size)
        output = output[:, -1, :].squeeze(dim=1)  # shape(batch_size, 2 * hidden_size)
        return self.fc(output)


class GRUDecoder(torch.nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 1, input_size: int = 1) -> None:
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.ffn = torch.nn.Conv1d(hidden_size, input_size, kernel_size=1)
        self.first_input = torch.nn.Parameter(torch.zeros(input_size))

    def forward(self, vectors: torch.Tensor, targets: torch.Tensor, mode: str = 'teacher_forcing') -> torch.Tensor:
        if mode == 'teacher_forcing':
            return self.teacher_forcing(vectors, targets)
        else:
            assert mode == 'free_running'
            return self.free_running(vectors, targets.shape[1])

    def teacher_forcing(self, vectors: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param vectors: shape(batch_size, hidden_size)
        :param targets: ground truth, shape(batch_size, seq_len, input_size)
        :return: shape(batch_size, seq_len, input_size)
        """
        batch_size = vectors.shape[0]
        decoder_inputs = self.first_input.repeat(batch_size, 1, 1)  # shape(batch_size, 1, input_size)
        decoder_inputs = torch.concat([decoder_inputs, targets[:, :-1, :]], dim=1)
        h = vectors.repeat(self.gru.num_layers, 1, 1)  # shape(num_layers, batch_size, hidden_size)
        output, _ = self.gru.forward(decoder_inputs, h)  # output: shape(batch_size, seq_len, hidden_size)
        reconstructed = self.ffn(output.permute(0, 2, 1)).permute(0, 2, 1)
        return reconstructed

    def free_running(self, vectors: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """
        :param vectors: shape(batch_size, hidden_size)
        :return: shape(batch_size, seq_len, input_size)
        """
        batch_size = vectors.shape[0]
        data = self.first_input.repeat(batch_size, 1, 1)  # shape(batch_size, seq_len=1, input_size)
        h = vectors.repeat(self.gru.num_layers, 1, 1)  # shape(num_layers, batch_size, hidden_size)
        reconstructed_list: list[torch.Tensor] = []
        for _ in range(sequence_length):
            output, h = self.gru(data, h)  # output: shape(batch_size, seq_len=1, hidden_size)
            data = self.ffn(output.permute(0, 2, 1)).permute(0, 2, 1)  # shape(batch_size, seq_len=1, input_size)
            reconstructed_list.append(data)
        return torch.concat(reconstructed_list, dim=1)
