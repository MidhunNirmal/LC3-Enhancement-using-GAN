import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 32, stride=2, padding=15),  # [B x 16 x 8192]
            nn.PReLU(),
            nn.Conv1d(16, 32, 32, stride=2, padding=15),  # [B x 32 x 4096]
            nn.PReLU(),
            nn.Conv1d(32, 64, 32, stride=2, padding=15),  # [B x 64 x 2048]
            nn.PReLU(),
            nn.Conv1d(64, 128, 32, stride=2, padding=15),  # [B x 128 x 1024]
            nn.PReLU(),
            nn.Conv1d(128, 256, 32, stride=2, padding=15),  # [B x 256 x 512]
            nn.PReLU(),
            nn.Conv1d(256, 512, 32, stride=2, padding=15),  # [B x 512 x 256]
            nn.PReLU(),
            nn.Conv1d(512, 1024, 32, stride=2, padding=15),  # [B x 1024 x 128]
            nn.PReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 32, stride=2, padding=15),  # [B x 512 x 256]
            nn.PReLU(),
            nn.ConvTranspose1d(512, 256, 32, stride=2, padding=15),  # [B x 256 x 512]
            nn.PReLU(),
            nn.ConvTranspose1d(256, 128, 32, stride=2, padding=15),  # [B x 128 x 1024]
            nn.PReLU(),
            nn.ConvTranspose1d(128, 64, 32, stride=2, padding=15),  # [B x 64 x 2048]
            nn.PReLU(),
            nn.ConvTranspose1d(64, 32, 32, stride=2, padding=15),  # [B x 32 x 4096]
            nn.PReLU(),
            nn.ConvTranspose1d(32, 16, 32, stride=2, padding=15),  # [B x 16 x 8192]
            nn.PReLU(),
            nn.ConvTranspose1d(16, 8, 32, stride=2, padding=15),  # [B x 1 x 16384]
            nn.PReLU(),
            nn.ConvTranspose1d(8, 1, 32, stride=2, padding=15),  # [B x 1 x 16384]
            nn.Tanh(),
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Test the Generator
generator = Generator()
input_tensor = torch.randn(5, 1, 8192, dtype=torch.float)  # Ensure input tensor is float type
output_tensor = generator(input_tensor)
print(output_tensor.shape)