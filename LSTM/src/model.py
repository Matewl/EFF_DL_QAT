import torch
import torch.nn as nn
from .quantization import get_quantizer, QuantizedLinear, QuantizedEmbedding, WeightParametrization
import torch.nn.utils.parametrize as parametrize

class FakeQuantizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, 
                 quantizer_type='lsq', bit_width=8, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True)
        
        act_q_type = 'lsq' if quantizer_type == 'adaround' else quantizer_type
        self.act_quant = get_quantizer(act_q_type, bit_width, **kwargs)
        

        if quantizer_type != 'none':
            #Собираем имена в список
            weight_names = [name for name, _ in self.lstm.named_parameters() if 'weight' in name]
            
            for name in weight_names:
                param = getattr(self.lstm, name)
                q = get_quantizer(quantizer_type, bit_width, weight_shape=param.shape, **kwargs)
                parametrize.register_parametrization(
                    self.lstm, name, WeightParametrization(q)
                )

    def forward(self, x, lengths=None):
        # 1. Квантуем вход
        x = self.act_quant(x)
        
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed_x)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, (h_n, c_n) = self.lstm(x)

        return out, (h_n, c_n)

class CustomQAT_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, 
                 quantizer_type='lsq', bit_width=8, **kwargs):
        super().__init__()
        
        self.embedding = QuantizedEmbedding(
            vocab_size, embedding_dim, 
            quantizer_type=quantizer_type, bit_width=bit_width, **kwargs
        )
        
        self.lstm = FakeQuantizedLSTM(
            embedding_dim, hidden_dim, 
            quantizer_type=quantizer_type, bit_width=bit_width, 
            bidirectional=True, **kwargs
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            QuantizedLinear(hidden_dim * 2, 64, quantizer_type=quantizer_type, bit_width=bit_width, **kwargs),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, x, lengths):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x, lengths)
        
        last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        return self.fc(last_hidden)