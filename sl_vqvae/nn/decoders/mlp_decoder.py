from torch import nn, Tensor
from sl_vqvae.nn.quantizers.quantizer import QuantizerOutput


class MLPBlock(nn.Module):
    def __init__(self, c_in: int, c_hidden: int, c_out: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(c_in, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MLPDecoder(nn.Module):

    def __init__(
        self,
        c_in: int = 512,
        c_hidden: int = 256,
        n_hand_landmarks: int = 21,
        n_pose_landmarks: int = 23,
        n_coordinates: int = 2,
    ):
        super().__init__()
        self.n_coordinates = n_coordinates
        pose_decoder = MLPBlock(c_in, c_hidden, n_pose_landmarks * n_coordinates)
        hand_decoder = MLPBlock(c_in, c_hidden, n_hand_landmarks * n_coordinates)
        self.decoders = nn.ModuleDict({
            'upper_pose': pose_decoder,
            'left_hand': hand_decoder,
            'right_hand': hand_decoder,
        })

    def forward(self, embeddings: dict[str, QuantizerOutput]) -> dict[str, Tensor]:
        out = dict()
        for body_part, quantizer_out in embeddings.items():
            x = quantizer_out.quantized_vectors
            N, T, _ = x.shape
            x = self.decoders[body_part](x)
            out[body_part] = x.view(N, T, -1, self.n_coordinates)
        return out

