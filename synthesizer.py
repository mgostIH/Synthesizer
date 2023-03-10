# Follows the paper Synthesizer: Rethinking Self-Attention for Transformer Models
# https://arxiv.org/abs/2005.00743

# Instead of self-attention, each token gets mapped to a vector of size M (values length)
# Each token has its own neural network that maps it to a vector of size M, so we don't have keys
# No dot product is taken between the vectors, but instead the vectors are concatenated into a N x M matrix
# We compute the values in the standard way.
# In self attention, M = N, but here we stay general


import torch
import torch.nn as nn

class SynthesizerLayer(nn.Module):
    # N: sequence length
    # M: number of values
    # D: embedding dimension
    # F: dimension of values
    # H: number of heads
    # K: factoring coefficient, now unused
    def __init__(self, N, M, D, F, H, _K) -> None:
        super().__init__()
        self.N = N
        self.D = D
        self.F = F
        self.H = H
        self.K = _K


        self.W_Q_1 = nn.Parameter(torch.empty((H, N, D//H, D//H)))
        self.W_Q_2 = nn.Parameter(torch.empty((H, N, M, D//H)))
        self.activation = nn.GELU()

        self.W_V = nn.Parameter(torch.empty((H, F//H, F//H)))
        self.O = nn.Linear(F, F)
        
        self.W_Q_1.data.normal_(0, 0.02)
        self.W_Q_2.data.normal_(0, 0.02)
        self.W_V.data.normal_(0, 0.02)
        self.O.weight.data.normal_(0, 0.02)

    def forward(self, X, V, mask = None):
        # X : (B, N, D)
        B, N, D = X.shape
        _, M, F = V.shape

        H = self.H
        _K = self.K
        D_H = D // H
        F_H = F // H
        # Split into H heads
        X = X.view(B, N, H, D_H)
        V = V.view(B, M, H, F_H)

        # X : (B, N, H, D_H)
        # W_Q_1 : (H, N, D_H, D_H)
        # Output: (B, H, N, D_H)
        Q_1 = torch.einsum('bnhd,hndd->bhnd', X, self.W_Q_1)
        # Activation
        Q_1 = self.activation(Q_1)

        # Now we obtain an NxM matrix for each head, where M is the amount of values
        # We softmax this on the rows and then compute the values
        # Q_1 : (B, H, N, D_H)
        # W_Q_2 : (H, N, M, D_H)
        # Output: (B, H, N, N)
        Q_2 = torch.einsum('bhnd,hnmd->bhnm', Q_1, self.W_Q_2)

        # Masking
        if mask is not None:
            Q_2 = Q_2.masked_fill(mask == 0, -float('inf'))

        # Softmax on rows
        Q_2 = torch.softmax(Q_2, dim=3)

        # Q_2 : (B, H, N, M)
        # V : (B, M, H, F_H)
        # W_V : (H, F_H, F_H)
        # Output: (B, N, H, F_H)
        Y = torch.einsum('bhnm,bmhf,hff->bnhf', Q_2, V, self.W_V)

        # Recombine heads
        Y = Y.reshape(B, N, F)
        return self.O(Y)


# Uses the equivalent of self-attention, so V = X, N = M, D = F, K is unused
# We use pre-normalization for X
class Synthesizer(nn.Module):
    def __init__(self, N, D, heads, tokens, layers, mask = None, _K = None) -> None:
        super().__init__()
        self.N = N
        self.D = D
        self.H = heads
        self.K = _K
        self.layers = layers
        self.mask = mask
        
        self.embedding = nn.Embedding(tokens, D)

        self.layers = nn.ModuleList([SynthesizerLayer(N, N, D, D, heads, _K) for _ in range(layers)])
        self.layer_norm = nn.LayerNorm(D)

    def forward(self, X):
        # X : (B, N)
        X = self.embedding(X)
        # Replace a few dimensions with uniformly drawn random numbers
        X[:, :, 0] = torch.rand(X.shape[0], X.shape[1])
        X[:, :, 1] = torch.rand(X.shape[0], X.shape[1])
        X[:, :, -1] = torch.rand(X.shape[0], X.shape[1])
        X[:, :, -2] = torch.rand(X.shape[0], X.shape[1])

        for layer in self.layers:
            X_norm = self.layer_norm(X)
            X = X + layer(X_norm, X_norm, self.mask)
        return X