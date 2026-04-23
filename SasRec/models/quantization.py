import torch
from torch import nn
from torch.nn import functional as F
from models.original import SASRec
from quantizations import LSQQuantStrategy, APoTQuantStrategy, QDropQuantStrategy, AdaRoundQuantStrategy

class QuantMultiheadAttention(nn.Module):
    """
    MultiheadAttention implementation using nn.Linear for projections,
    compatible with quantization wrappers.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        if query is key and key is value:
             qkv = self.in_proj(query)
             q, k, v = qkv.chunk(3, dim=-1)
        else:
             qkv = self.in_proj(query)
             q, k, v = qkv.chunk(3, dim=-1)

        q = q.contiguous().view(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.contiguous().view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.contiguous().view(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        attn_output_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
             if attn_mask.dim() == 2:
                 attn_output_weights += attn_mask.unsqueeze(0).unsqueeze(0)
             else:
                 attn_output_weights += attn_mask

        if key_padding_mask is not None:
             attn_output_weights = attn_output_weights.masked_fill(
                 key_padding_mask.unsqueeze(1).unsqueeze(2),
                 float('-inf')
             )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_output_weights


class QuantSASRec(SASRec):
    """
    SASRec with QAT/PTQ support.
    Replaces standard MultiheadAttention with QuantMultiheadAttention.
    """
    def __init__(self, user_num, item_num, args):
        super().__init__(user_num, item_num, args)
        
        # Replace standard MHA with QuantMHA
        for i in range(len(self.attention_layers)):
            old_mha = self.attention_layers[i]
            new_mha = QuantMultiheadAttention(
                embed_dim=args.hidden_units,
                num_heads=args.num_heads,
                dropout=args.dropout_rate
            )

            new_mha.in_proj.weight.data = old_mha.in_proj_weight.data
            new_mha.in_proj.bias.data = old_mha.in_proj_bias.data
            new_mha.out_proj.weight.data = old_mha.out_proj.weight.data
            new_mha.out_proj.bias.data = old_mha.out_proj.bias.data
            
            self.attention_layers[i] = new_mha

        self.quant_strategy = None
        self.quant_enabled = True

    def prepare_quant(self, strategy_name: str, config: dict):
        """Attach quantization strategy (for QAT or PTQ)."""
        if strategy_name == "lsq":
            self.quant_strategy = LSQQuantStrategy(config)
        elif strategy_name == "apot":
            self.quant_strategy = APoTQuantStrategy(config)
        elif strategy_name == "qdrop":
            self.quant_strategy = QDropQuantStrategy(config)
        elif strategy_name == "adaround":
            self.quant_strategy = AdaRoundQuantStrategy(config)
        else:
            raise ValueError

        self.quant_strategy.attach(self)

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        return super().forward(user_ids, log_seqs, pos_seqs, neg_seqs)

    def calibrate(self, dataloader):
        """For PTQ strategies like AdaRound."""
        if hasattr(self.quant_strategy, "calibrate"):
            self.quant_strategy.calibrate(dataloader)
