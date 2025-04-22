import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from .modules.layers import MORTMEncoder
from .modules.config import MORTMArgs


class MeanPoolingWithMask(nn.Module):
    """
    Applies mean pooling to the input tensor, considering an optional attention mask.

    This module calculates the mean of the token embeddings along the sequence dimension,
    ignoring positions marked as padding in the attention mask.
    """
    def __init__(self):
        """Initializes the MeanPoolingWithMask module."""
        super().__init__()

    def forward(self, last_hidden_state: Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the mean pooling operation.

        Args:
            last_hidden_state (torch.Tensor): The tensor containing the hidden states from the encoder.
                                               Expected shape: [batch_size, sequence_length, hidden_size] (B, S, D).
            attention_mask (torch.Tensor, optional): The mask indicating which tokens are valid (1) and which are padding (0).
                                                     Expected shape: [batch_size, sequence_length] (B, S).
                                                     If None, all tokens are considered valid. Defaults to None.

        Returns:
            torch.Tensor: The pooled output tensor.
                          Shape: [batch_size, hidden_size] (B, D).
        """
        if attention_mask is None:
            # If no mask is provided, perform simple mean pooling over the sequence dimension.
            pooled_output = torch.mean(last_hidden_state, dim=1)
        else:
            # --- Padding-aware Mean Pooling ---
            # 1. Expand attention mask to match the hidden state dimensions: [B, S] -> [B, S, D]
            #    We need to ensure the mask is float for multiplication.
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            # 2. Sum the embeddings for non-padding tokens.
            #    Multiply hidden states by the expanded mask (padding tokens become zero vectors).
            #    Then sum across the sequence dimension (dim=1).
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

            # 3. Calculate the number of valid (non-padding) tokens for each sequence.
            #    Sum the expanded mask across the sequence dimension.
            #    Clamp the sum to a minimum value of 1e-9 to avoid division by zero if a sequence has only padding (edge case).
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

            # 4. Compute the mean by dividing the sum of embeddings by the number of valid tokens.
            pooled_output = sum_embeddings / sum_mask

        return pooled_output


class BERTM(nn.Module):

    def __init__(self, args: MORTMArgs, progress):
        super(BERTM, self).__init__()
        self.encoder = MORTMEncoder(args=args,
                                     layer_norm_eps=1e-5,
                                     progress=progress)
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)

        self.pooling = MeanPoolingWithMask()

        self.norm = nn.LayerNorm(args.d_model, eps=1e-5)
        self.linear = nn.Linear(args.d_model, args.d_model)

        self.Wout = nn.Linear(args.d_model, 2)

    def forward(self, src: Tensor, input_padding_mask=None):
        """
        src: 入力テンソル (バッチサイズ, シーケンス長, 特徴量次元)
        input_padding_mask: パディングマスク (バッチサイズ, シーケンス長)
        """
        src = self.embedding(src)
        # Encoderのforwardメソッドを呼び出す
        out = self.encoder(src=src, mask=None,
                           src_key_padding_mask=input_padding_mask,
                           is_causal=False)
        out = self.pooling(out, input_padding_mask)
        out = self.norm(out)
        out = self.linear(out)
        out = F.gelu(out)
        out = self.Wout(out)

        return out