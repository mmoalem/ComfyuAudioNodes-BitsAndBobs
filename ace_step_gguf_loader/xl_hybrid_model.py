import torch
import comfy.model_base
import comfy.supported_models
from comfy.ldm.ace.ace_step15 import (
    AceStepConditionGenerationModel,
    AceStepConditionEncoder,
    AceStepAudioTokenizer,
    AudioTokenDetokenizer
)

class AceStepConditionGenerationModelXL(AceStepConditionGenerationModel):
    """
    A hybridized wrapper for AceStepConditionGenerationModel that supports
    the XL architectures where the text/timbre encoders run at a smaller
    hidden dimension (2048) than the DiT decoder (2560).
    """
    def __init__(self, **kwargs):
        # XL models have dimension 2560 for the DiT
        decoder_hidden_size = kwargs.get("hidden_size", 2560)
        # The encoder conditioning components are always 2048-dim regardless of XL scale
        encoder_hidden_size = 2048
        # Encoder head parameters are derived from encoder_hidden_size, NOT from kwargs
        # (kwargs carries the DiT's num_heads which is read from GGUF and can be 32 for XL,
        # while the lyric encoder is always 2048-dim = 16 heads × 128 head_dim)
        enc_head_dim  = 128  # fixed for all ACE-Step variants
        enc_num_heads    = encoder_hidden_size // enc_head_dim        # = 16
        enc_num_kv_heads = encoder_hidden_size // (enc_head_dim * 2)  # = 8

        # 1. Initialize the parent normally (this builds the 2560-dim DiT correctly)
        super().__init__(**kwargs)

        # 2. Re-build the encoder, tokenizer, detokenizer, and null-cond using 2048 dims
        self.encoder = AceStepConditionEncoder(
            text_hidden_dim=kwargs.get("text_hidden_dim", 1024),
            timbre_hidden_dim=kwargs.get("timbre_hidden_dim", 64),
            hidden_size=encoder_hidden_size,
            num_lyric_layers=kwargs.get("num_lyric_layers", 8),
            num_timbre_layers=kwargs.get("num_timbre_layers", 4),
            num_heads=enc_num_heads,
            num_kv_heads=enc_num_kv_heads,
            head_dim=enc_head_dim,
            intermediate_size=encoder_hidden_size * 3,  # 6144
            rms_norm_eps=kwargs.get("rms_norm_eps", 1e-06),
            dtype=kwargs.get("dtype"),
            device=kwargs.get("device"),
            operations=kwargs.get("operations")
        )

        self.tokenizer = AceStepAudioTokenizer(
            audio_acoustic_hidden_dim=kwargs.get("audio_acoustic_hidden_dim", 64),
            hidden_size=encoder_hidden_size,
            pool_window_size=kwargs.get("pool_window_size", 5),
            fsq_dim=kwargs.get("fsq_dim", 2048),
            fsq_levels=kwargs.get("fsq_levels", [8, 8, 8, 5, 5, 5]),
            fsq_input_num_quantizers=kwargs.get("fsq_input_num_quantizers", 1),
            num_layers=kwargs.get("num_tokenizer_layers", 2),
            head_dim=enc_head_dim,
            rms_norm_eps=kwargs.get("rms_norm_eps", 1e-06),
            dtype=kwargs.get("dtype"),
            device=kwargs.get("device"),
            operations=kwargs.get("operations")
        )

        self.detokenizer = AudioTokenDetokenizer(
            hidden_size=encoder_hidden_size,
            pool_window_size=kwargs.get("pool_window_size", 5),
            audio_acoustic_hidden_dim=kwargs.get("audio_acoustic_hidden_dim", 64),
            num_layers=2,
            head_dim=enc_head_dim,
            dtype=kwargs.get("dtype"),
            device=kwargs.get("device"),
            operations=kwargs.get("operations")
        )

        self.null_condition_emb = torch.nn.Parameter(
            torch.empty(1, 1, encoder_hidden_size, dtype=kwargs.get("dtype"), device=kwargs.get("device"))
        )

class ACEStep15_XL_BaseModel(comfy.model_base.ACEStep15):
    def __init__(self, model_config, model_type=comfy.model_base.ModelType.FLOW, device=None):
        # We explicitly skip comfy.model_base.ACEStep15.__init__ and call BaseModel
        # to inject our XL model class instead of the standard one.
        comfy.model_base.BaseModel.__init__(
            self, model_config, model_type, device=device, 
            unet_model=AceStepConditionGenerationModelXL
        )

class ACEStep15_XL_Config(comfy.supported_models.ACEStep15):
    def get_model(self, state_dict, prefix="", device=None):
        out = ACEStep15_XL_BaseModel(self, device=device)
        return out
