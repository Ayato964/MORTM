
from .modules.layers import *
from einops import rearrange

class Vision(nn.Module):
    def __init__(self, args: MORTM_LIVE_Args):
        super().__init__()
        encoder_output_dim = args.instrument_num * 8 * (args.pianoroll_time_step // 4) * 16
        self.encoder = VisionEncoder(args, encoder_output_dim)
        self.decoder = VisionDecoder(args, encoder_output_dim=encoder_output_dim, encoder_output_shape=(args.instrument_num * 8, 16, args.pianoroll_time_step // 4))

    def forward(self, pianoroll: Tensor):
        pianoroll = rearrange(pianoroll, 'b w h c -> b c h w')
        encoded, mu, log_var = self.encoder(pianoroll)
        decoded = self.decoder(encoded)

        return decoded, mu, log_var


class MORTMLive(nn.Module):
    def __init__(self, args: MORTM_LIVE_Args, progress=None):
        super(MORTMLive, self).__init__()
        self.args = args
        self.vision = Vision(args)
        self.mortm = MORTMDecoder(args=args, progress=progress)

