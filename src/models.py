import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="valid")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        residual = X  # スキップ接続のための残差

        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))

        if residual.shape == X.shape:
            X = X + residual  # スキップ接続

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return self.dropout(X)

# テスト用の簡単な例
if __name__ == "__main__":
    model = BasicConvClassifier(num_classes=10, seq_len=300, in_channels=64)
    x = torch.randn(32, 64, 300)  # バッチサイズ32、チャンネル数64、シーケンス長300
    out = model(x)
    print(out.shape)  # (32, 10) となるはず
