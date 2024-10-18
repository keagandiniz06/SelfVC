import torch.nn as nn

class ConformerEncoder256(nn.Module):
    def __init__(self, original_encoder):
        super(ConformerEncoder256, self).__init__()
        self.encoder = original_encoder
        # Add a 1D convolution with stride to downsample to 256
        self.downsample = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, stride=2) # Example

    def forward(self, audio_signal, length):
        # Pass through original encoder
        embeddings, lengths = self.encoder(audio_signal=audio_signal, length=length)
        # Reduce the embedding size
        embeddings = self.downsample(embeddings)
        return embeddings, lengths