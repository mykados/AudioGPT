# %pip install torch torchaudio
# %pip install encodec
# %pip install tqdm
# %pip install soundfile

import os
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Torchinductor does not support code generation for complex operators")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

AUDIO_FOLDER = "path_to_audio_pt_folder"
CHECKPOINT_PATH = "checkpoint_v2.pt"
OUTPUT_PATH = "generated_v2.wav"

MAX_FILES = 2500
MAX_AUDIO_SECONDS = 600

D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 32
MAX_LEN = 4096
DROPOUT = 0.1

FINE_D_MODEL = 512
FINE_N_HEADS = 8
FINE_N_LAYERS = 8

BAR_D_MODEL = 256
BAR_N_HEADS = 4
BAR_N_LAYERS = 4

EPOCHS = 40
BATCH_SIZE = 40
LEARNING_RATE = 3e-4
CHUNK_SIZE = 2048
RESUME = True
GRAD_CKPT = True
ACCUM_STEPS = 1

SCHEDULED_SAMPLING_START = 0.0
SCHEDULED_SAMPLING_END = 0.3
SCHEDULED_SAMPLING_WARMUP = 0.5

GENERATE_SECONDS = 40.0
TEMPERATURE = 0.75
TOP_K = 35
CONTEXT_WINDOW = 4096
PROMPT_AUDIO = "path_to_prompt_audio.pt"
PROMPT_SECONDS = 15.0

BPM = 120
BEAT_SUBDIVISIONS = 4

# Training stages: "coarse" -> train only coarse, "fine" -> train only fine,
TRAIN_STAGE = "fine"
MODE = "generate"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def clear_mem():
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(3.0)
encodec_model.eval()
encodec_model.cpu()

with torch.no_grad():
    _dummy = torch.zeros(1, 1, 24000)
    _encoded = encodec_model.encode(_dummy)
    NUM_Q = _encoded[0][0].shape[1]
print(f"Active codebooks (NUM_Q): {NUM_Q}")

CODEBOOK_SIZE = 1024
VOCAB_SIZE = CODEBOOK_SIZE
ENCODEC_HOP = 320
ENCODEC_FPS = 24000 / ENCODEC_HOP
COARSE_TOKENS_PER_SEC = ENCODEC_FPS

BEAT_STRIDE_FRAMES = max(1, round(ENCODEC_FPS / (BPM / 60) / BEAT_SUBDIVISIONS))
BEAT_STRIDE_TOKENS = BEAT_STRIDE_FRAMES
BEAT_TOKEN_ID = VOCAB_SIZE
COARSE_VOCAB_SIZE = VOCAB_SIZE + 1

BAR_STRIDE_TOKENS = BEAT_STRIDE_TOKENS * BEAT_SUBDIVISIONS


def load_audio(path, sr=24000, max_seconds=None):
    try:
        waveform = torch.load(path, weights_only=True).float()
        if waveform is None or waveform.numel() == 0:
            return None
        if max_seconds is not None:
            waveform = waveform[:, : int(sr * max_seconds)]
        return waveform
    except Exception as e:
        print(f"[SKIP] {path} -> {e}")
        return None


import torchaudio.transforms as T
def denoise_audio(audio, sr=24000):
    n_fft = 1024
    hop = 256
    spec = torch.stft(audio.squeeze(0), n_fft=n_fft, hop_length=hop,
                      return_complex=True)
    mag = spec.abs()

    threshold = mag.mean() * 0.05
    mask = (mag > threshold).float()
    denoised_spec = spec * mask

    denoised = torch.istft(denoised_spec, n_fft=n_fft, hop_length=hop,
                           length=audio.shape[-1])
    return denoised.unsqueeze(0)

def audio_to_tokens(waveform, sr=24000):
    enc = encodec_model
    waveform = waveform.cpu()
    waveform = convert_audio(waveform, sr, enc.sample_rate, enc.channels)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        encoded = enc.encode(waveform)

    codes = encoded[0][0].squeeze(0).cpu()
    coarse = codes[0].long()

    fine_list = []
    if NUM_Q > 1:
        for k in range(1, NUM_Q):
            fine_list.append(codes[k].long())

    coarse_with_beats = []
    for i in range(0, len(coarse), BEAT_STRIDE_TOKENS):
        coarse_with_beats.append(torch.tensor([BEAT_TOKEN_ID], dtype=torch.long))
        coarse_with_beats.append(coarse[i: i + BEAT_STRIDE_TOKENS])
    coarse_with_beats = torch.cat(coarse_with_beats)

    return coarse_with_beats, fine_list


def tokens_to_audio(coarse_tokens, fine_tokens=None):
    coarse_tokens = coarse_tokens.to(device)
    coarse_tokens = coarse_tokens[coarse_tokens != BEAT_TOKEN_ID]
    T = coarse_tokens.shape[0]

    codes = torch.zeros(NUM_Q, T, dtype=torch.long, device=device)
    codes[0] = coarse_tokens.clamp(0, CODEBOOK_SIZE - 1)

    if fine_tokens is not None and NUM_Q > 1:
        fine_tokens = fine_tokens.to(device)
        usable = min(T, fine_tokens.shape[0])
        for k in range(1, NUM_Q):
            if k - 1 < fine_tokens.shape[1]:
                codes[k, :usable] = fine_tokens[:usable, k - 1].clamp(0, CODEBOOK_SIZE - 1)

    enc = encodec_model.to(device)
    with torch.no_grad():
        decoded = enc.decode([(codes.unsqueeze(0), None)])
    enc.cpu()
    clear_mem()
    return decoded[0].cpu()


def inject_beat_tokens(tokens):
    out = []
    for i in range(0, len(tokens), BEAT_STRIDE_TOKENS):
        out.append(torch.tensor([BEAT_TOKEN_ID], dtype=torch.long))
        out.append(tokens[i: i + BEAT_STRIDE_TOKENS])
    return torch.cat(out)


def strip_beat_tokens(tokens):
    return tokens[tokens != BEAT_TOKEN_ID]


def get_beat_mask(tokens):
    return tokens != BEAT_TOKEN_ID


def precompute_freqs_cis(head_dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    cos = freqs_cis.real.repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def rotate(x):
        return (x.float() * cos + rotate_half(x.float()) * sin).type_as(x)

    return rotate(xq), rotate(xk)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        freqs = precompute_freqs_cis(self.head_dim, max_len)
        self.register_buffer("freqs_cis", freqs)

    def forward(self, x, attn_mask=None, is_causal=True):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T])

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.drop.p if self.training else 0.0, is_causal=is_causal,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class KVCachedAttention(nn.Module):
    def __init__(self, attn_module, max_len=4096):
        super().__init__()
        self.attn = attn_module
        self.max_len = max_len
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.seq_len = 0

    def forward_cached(self, x):
        B, T, C = x.shape
        H, D = self.attn.n_heads, self.attn.head_dim

        q = self.attn.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.attn.k_proj(x).reshape(B, T, H, D).transpose(1, 2)
        v = self.attn.v_proj(x).reshape(B, T, H, D).transpose(1, 2)

        max_pos = self.attn.freqs_cis.shape[0] - T
        pos = min(self.seq_len, max_pos)
        q, k = apply_rotary_emb(q, k, self.attn.freqs_cis[pos:pos + T])

        if self.k_cache is not None and self.k_cache.shape[2] >= self.max_len:
            self.k_cache = self.k_cache[:, :, -(self.max_len - T):]
            self.v_cache = self.v_cache[:, :, -(self.max_len - T):]

        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)

        self.seq_len += T

        if self.k_cache.shape[2] > self.max_len:
            self.k_cache = self.k_cache[:, :, -self.max_len:]
            self.v_cache = self.v_cache[:, :, -self.max_len:]

        out = F.scaled_dot_product_attention(q, self.k_cache, self.v_cache, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.attn.out_proj(out)


def attach_kv_cache(model, max_len=MAX_LEN):
    for block in model.layers:
        block._cached_attn = KVCachedAttention(block.attn, max_len=max_len)


def reset_kv_cache(model):
    for block in model.layers:
        if hasattr(block, '_cached_attn'):
            block._cached_attn.reset_cache()


def forward_one_token_cached(model, x_single, bar_ctx):
    B = x_single.shape[0]
    h = model.token_emb(x_single)
    h = model.drop(h)

    for i, block in enumerate(model.layers):
        normed = block.ln1(h)
        h = h + block._cached_attn.forward_cached(normed)

        if block.has_cross and bar_ctx is not None:
            h = h + block._cross_attn(block.ln_cross(h), bar_ctx)

        h = h + block.ff(block.ln2(h))

    h = model.ln_f(h)
    return model.head(h)


@torch.no_grad()
def generate_coarse_fast(model, prompt_tokens, total_audio_tokens, temperature=0.8, top_k=100, context=4096):
    REPETITION_PENALTY = 1.05
    REPETITION_WINDOW = 300
    BAR_CTX_REFRESH_TOKENS = 512

    model.eval()
    attach_kv_cache(model, max_len=context)
    reset_kv_cache(model)

    tokens = prompt_tokens.to(device).unsqueeze(0)
    print(f"Prefilling KV cache with {tokens.shape[1]} prompt tokens...")

    def compute_bar_ctx(token_ids_1d):
        window = token_ids_1d[-context:].unsqueeze(0)  # [1, T]
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            tok_emb = model.token_emb(window)
            return model.bar_model(tok_emb, BAR_STRIDE_TOKENS)

    bar_ctx = compute_bar_ctx(tokens[0])

    reset_kv_cache(model)
    with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        for t in range(tokens.shape[1]):
            forward_one_token_cached(model, tokens[:, t:t + 1], bar_ctx)

    generated = list(prompt_tokens.cpu().numpy())
    audio_token_count = int((prompt_tokens != BEAT_TOKEN_ID).sum().item())
    audio_since_last_beat = audio_token_count % BEAT_STRIDE_TOKENS
    tokens_since_bar_refresh = 0

    print(f"Generating {total_audio_tokens} audio tokens...")
    pbar = tqdm(total=total_audio_tokens, initial=audio_token_count)

    while audio_token_count < total_audio_tokens:
        if tokens_since_bar_refresh >= BAR_CTX_REFRESH_TOKENS:
            recent = torch.tensor(generated[-context:], dtype=torch.long, device=device)
            bar_ctx = compute_bar_ctx(recent)
            tokens_since_bar_refresh = 0

        if audio_since_last_beat == 0:
            beat_tok = torch.tensor([[BEAT_TOKEN_ID]], device=device)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                forward_one_token_cached(model, beat_tok, bar_ctx)
            generated.append(BEAT_TOKEN_ID)
            audio_since_last_beat = 1
            tokens_since_bar_refresh += 1
            continue

        last_tok = torch.tensor([[generated[-1]]], device=device)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = forward_one_token_cached(model, last_tok, bar_ctx)
        logits = logits[0, 0] / temperature

        if len(generated) > REPETITION_WINDOW:
            recent = torch.tensor(generated[-REPETITION_WINDOW:], device=device)
            audio_recent = recent[recent != BEAT_TOKEN_ID]
            unique_toks = audio_recent.unique()
            penalty_mask = torch.ones(logits.shape[0], device=device)
            penalty_mask[unique_toks] = REPETITION_PENALTY
            logits = torch.where(logits > 0, logits / penalty_mask, logits * penalty_mask)

        topv, topi = torch.topk(logits, top_k)
        probs = torch.softmax(topv, dim=-1)
        next_tok = topi[torch.multinomial(probs, 1)].item()

        generated.append(next_tok)
        audio_token_count += 1
        audio_since_last_beat += 1
        tokens_since_bar_refresh += 1
        if audio_since_last_beat >= BEAT_STRIDE_TOKENS:
            audio_since_last_beat = 0

        pbar.update(1)

    pbar.close()
    reset_kv_cache(model)
    return torch.tensor(generated, dtype=torch.long)


@torch.no_grad()
def generate_fine_fast(fine_model, coarse_audio_tokens, chunk_size=4096, overlap=1024):
    fine_model.eval()

    coarse = coarse_audio_tokens.unsqueeze(0).to(device)  # [1, T]
    T = coarse.shape[1]

    all_fine = []

    with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        i = 0
        while i < T:
            start = i
            end = min(i + chunk_size, T)
            chunk = coarse[:, start:end]
            logits = fine_model(chunk, start_pos=start)
            sampled = torch.argmax(logits[0], dim=-1)

            if start == 0:
                all_fine.append(sampled)
            else:
                all_fine.append(sampled[overlap:])

            i += (chunk_size - overlap)

    return torch.cat(all_fine, dim=0)


def generate_continuous_fast(
        coarse_model,
        fine_model,
        total_seconds=30.0,
        temperature=0.8,
        top_k=100,
        context=4096,
        prompt_tokens=None,
):
    total_audio_toks = int(total_seconds * COARSE_TOKENS_PER_SEC)

    if prompt_tokens is not None:
        prompt_with_beats = prompt_tokens
    else:
        prompt_with_beats = torch.tensor([BEAT_TOKEN_ID, torch.randint(0, CODEBOOK_SIZE, (1,)).item()], dtype=torch.long)

    coarse_tokens = generate_coarse_fast(
        coarse_model,
        prompt_with_beats,
        total_audio_toks,
        temperature=temperature,
        top_k=top_k,
        context=context,
    )

    print("Generating fine codebooks...")

    coarse_audio = strip_beat_tokens(coarse_tokens)

    if prompt_tokens is not None:
        prompt_audio_len = len(strip_beat_tokens(prompt_tokens))
        # Split prompt vs generated
        coarse_prompt = coarse_audio[:prompt_audio_len]
        coarse_gen = coarse_audio[prompt_audio_len:]
        # Generate fine tokens separately
        fine_prompt = generate_fine_fast(fine_model, coarse_prompt)
        fine_gen = generate_fine_fast(fine_model, coarse_gen)
        # Merge
        fine_tokens = torch.cat([fine_prompt, fine_gen], dim=0)

    else:
        fine_tokens = generate_fine_fast(fine_model, coarse_audio)

    usable = min(coarse_audio.shape[0], fine_tokens.shape[0])

    return (coarse_audio[:usable], fine_tokens[:usable])


class SwiGLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, max_len, cross_attn_dim=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model)
        self.drop = nn.Dropout(dropout)

        self.has_cross = cross_attn_dim > 0
        if self.has_cross:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross_q = nn.Linear(d_model, d_model, bias=False)
            self.cross_k = nn.Linear(cross_attn_dim, d_model, bias=False)
            self.cross_v = nn.Linear(cross_attn_dim, d_model, bias=False)
            self.cross_out = nn.Linear(d_model, d_model, bias=False)
            self.cross_drop = nn.Dropout(dropout)
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

    def _cross_attn(self, x, ctx):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.cross_q(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.cross_k(ctx).reshape(B, -1, H, D).transpose(1, 2)
        v = self.cross_v(ctx).reshape(B, -1, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.cross_drop.p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.cross_out(out)

    def forward(self, x, attn_mask=None, bar_ctx=None, is_causal=True):
        x = x + self.drop(self.attn(self.ln1(x), attn_mask, is_causal))
        if self.has_cross and bar_ctx is not None:
            x = x + self.drop(self._cross_attn(self.ln_cross(x), bar_ctx))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

    def forward_with_ckpt(self, x, attn_mask=None, bar_ctx=None, is_causal=True):
        import torch.utils.checkpoint as ckpt
        def attn_fn(x):
            return self.drop(self.attn(self.ln1(x), attn_mask, is_causal))

        x = x + ckpt.checkpoint(attn_fn, x, use_reentrant=False)
        if self.has_cross and bar_ctx is not None:
            def cross_fn(x, ctx):
                return self.drop(self._cross_attn(self.ln_cross(x), ctx))

            x = x + ckpt.checkpoint(cross_fn, x, bar_ctx, use_reentrant=False)

        def ff_fn(x):
            return self.drop(self.ff(self.ln2(x)))

        x = x + ckpt.checkpoint(ff_fn, x, use_reentrant=False)
        return x


class BarContextModel(nn.Module):
    def __init__(self, token_embed_dim, d_model=BAR_D_MODEL,
                 n_heads=BAR_N_HEADS, n_layers=BAR_N_LAYERS,
                 max_bars=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.proj_in = nn.Linear(token_embed_dim, d_model, bias=False)
        self.pos_emb = nn.Embedding(max_bars, d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, token_embs, bar_stride):
        B, T, E = token_embs.shape
        num_bars = max(1, T // bar_stride)
        usable = num_bars * bar_stride
        bar_embs = token_embs[:, :usable, :].reshape(B, num_bars, bar_stride, E).mean(dim=2)
        x = self.drop(self.proj_in(bar_embs))
        pos = torch.arange(num_bars, device=bar_embs.device).unsqueeze(0)
        x = x + self.pos_emb(pos)
        x = self.encoder(x)
        return self.ln_out(x)


class CoarseModel(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS,
                 max_len=MAX_LEN, dropout=DROPOUT,
                 bar_d_model=BAR_D_MODEL):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model, n_heads, dropout, max_len,
                cross_attn_dim=(bar_d_model if i % 2 == 1 else 0)
            )
            for i in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.bar_model = BarContextModel(
            token_embed_dim=d_model,
            d_model=bar_d_model,
            n_heads=BAR_N_HEADS,
            n_layers=BAR_N_LAYERS,
            max_bars=max_len // max(1, BEAT_STRIDE_TOKENS * BEAT_SUBDIVISIONS) + 4,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        std_base = 0.02
        std_resid = std_base / math.sqrt(2 * self.n_layers)
        nn.init.normal_(self.token_emb.weight, std=std_base)
        for block in self.layers:
            nn.init.normal_(block.attn.q_proj.weight, std=std_base)
            nn.init.normal_(block.attn.k_proj.weight, std=std_base)
            nn.init.normal_(block.attn.v_proj.weight, std=std_base)
            nn.init.normal_(block.attn.out_proj.weight, std=std_resid)
            nn.init.normal_(block.ff.w1.weight, std=std_base)
            nn.init.normal_(block.ff.w2.weight, std=std_base)
            nn.init.normal_(block.ff.w3.weight, std=std_resid)
            if block.has_cross:
                nn.init.normal_(block.cross_q.weight, std=std_base)
                nn.init.normal_(block.cross_k.weight, std=std_base)
                nn.init.normal_(block.cross_v.weight, std=std_base)
                nn.init.normal_(block.cross_out.weight, std=std_resid)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_len
        tok_emb = self.token_emb(x)
        h = self.drop(tok_emb)
        bar_ctx = self.bar_model(tok_emb, BAR_STRIDE_TOKENS)
        use_ckpt = GRAD_CKPT and self.training
        for i, block in enumerate(self.layers):
            ctx = bar_ctx if block.has_cross else None
            if use_ckpt:
                h = block.forward_with_ckpt(h, attn_mask=None, bar_ctx=ctx)
            else:
                h = block(h, attn_mask=None, bar_ctx=ctx)
        h = self.ln_f(h)
        return self.head(h)


class FineModel(nn.Module):
    def __init__(self, coarse_vocab_size, d_model=FINE_D_MODEL,
                 n_heads=FINE_N_HEADS, n_layers=FINE_N_LAYERS,
                 max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_fine_codebooks = NUM_Q - 1

        self.coarse_emb = nn.Embedding(coarse_vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout, max_len, cross_attn_dim=d_model)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # One head per codebook
        self.heads = nn.ModuleList([
            nn.Linear(d_model, CODEBOOK_SIZE, bias=False)
            for _ in range(self.num_fine_codebooks)
        ])

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.coarse_emb.weight, std=std)
        nn.init.normal_(self.pos_emb.weight, std=std)
        for head in self.heads:
            nn.init.normal_(head.weight, std=std)
        for block in self.layers:
            for w in [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj,
                      block.ff.w1, block.ff.w2]:
                nn.init.normal_(w.weight, std=std)
            nn.init.normal_(block.attn.out_proj.weight, std=std)
            nn.init.normal_(block.ff.w3.weight, std=std)

    def forward(self, coarse_audio_tokens, start_pos=0):
        B, T = coarse_audio_tokens.shape
        T = min(T, MAX_LEN)
        coarse_audio_tokens = coarse_audio_tokens[:, :T]

        coarse_emb = self.coarse_emb(coarse_audio_tokens)
        positions = torch.arange(start_pos, start_pos + T, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        h = self.drop(coarse_emb + pos_emb)

        for block in self.layers:
            h = block(h, attn_mask=None, bar_ctx=coarse_emb, is_causal=False)

        h = self.ln_f(h)

        # Predict each codebook
        outputs = []
        for head in self.heads:
            outputs.append(head(h))
        return torch.stack(outputs, dim=2)


def coarse_loss(logits, targets, beat_mask):
    B, T, V = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction='none'
    ).reshape(B, T)
    ce = ce * beat_mask.float()
    denom = beat_mask.sum() + 1e-8
    return ce.sum() / denom


def fine_loss(fine_logits, fine_targets, pad_mask=None):
    B, T, K, V = fine_logits.shape
    logits_flat = fine_logits.reshape(B * T * K, V)
    targets_flat = fine_targets.reshape(B * T * K)

    weights = torch.tensor([2.0 / (2 ** k) for k in range(K)], device=device)
    weights = weights / weights.sum()

    ce = F.cross_entropy(logits_flat, targets_flat, reduction='none').reshape(B, T, K)

    if pad_mask is not None:
        ce = ce * pad_mask.unsqueeze(-1).float()
        denom = pad_mask.sum() * K + 1e-8
        return ce.sum() / denom

    per_cb = ce.mean(dim=1)
    weighted = (per_cb * weights.unsqueeze(0)).sum(dim=1).mean()
    return weighted


def create_optimizer(model, lr=LEARNING_RATE):
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "ln", "norm", "pos_emb"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW([
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=lr, betas=(0.9, 0.95))


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, d):
        self.step_count = d["step_count"]


def get_scheduled_sampling_prob(step, total_steps):
    warmup_steps = int(total_steps * SCHEDULED_SAMPLING_WARMUP)
    if step < warmup_steps:
        return SCHEDULED_SAMPLING_START
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min(SCHEDULED_SAMPLING_END,
               SCHEDULED_SAMPLING_START + (SCHEDULED_SAMPLING_END - SCHEDULED_SAMPLING_START) * progress)


scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
_accum_step = 0


def train_step_coarse(model, optimizer, tokens, step, total_steps, scheduler=None):
    global _accum_step
    model.train()
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    beat_mask = get_beat_mask(y)

    with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        logits = model(x)
        loss = coarse_loss(logits, y, beat_mask)
        loss = loss / ACCUM_STEPS

    scaler.scale(loss).backward()

    _accum_step += 1
    if _accum_step % ACCUM_STEPS == 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    return loss.item() * ACCUM_STEPS


def train_step_fine(coarse_model, fine_model, optimizer,
                    coarse_tokens, fine_targets, scheduler=None):
    global _accum_step
    coarse_model.eval()
    fine_model.train()

    with torch.no_grad():
        # Strip beats for fine model conditioning
        B = coarse_tokens.shape[0]
        coarse_audio = []
        max_len = 0
        for b in range(B):
            ca = strip_beat_tokens(coarse_tokens[b])
            coarse_audio.append(ca)
            max_len = max(max_len, ca.shape[0])

        # Pad to same length
        coarse_padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b in range(B):
            ca = coarse_audio[b]
            coarse_padded[b, :ca.shape[0]] = ca

    with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        fine_logits = fine_model(coarse_padded)
        # Truncate fine_logits to match fine_targets length
        T_target = fine_targets.shape[1]
        fine_logits = fine_logits[:, :T_target]
        loss = fine_loss(fine_logits, fine_targets)
        loss = loss / ACCUM_STEPS

    scaler.scale(loss).backward()

    _accum_step += 1
    if _accum_step % ACCUM_STEPS == 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(fine_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    return loss.item() * ACCUM_STEPS


@torch.no_grad()
def generate_coarse(model, tokens, max_new, context, temperature, top_k):
    REPETITION_PENALTY = 1.2
    REPETITION_WINDOW = 600

    for _ in range(max_new):
        if tokens.shape[1] <= context:
            inp = tokens
        else:
            start = tokens.shape[1] - context
            recent = tokens[0, start:]
            beat_pos = (recent == BEAT_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(beat_pos) > 0:
                start = start + beat_pos[0].item()
            inp = tokens[:, start:]

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(inp)[:, -1] / temperature

        if tokens.shape[1] > REPETITION_WINDOW:
            recent = tokens[0, -REPETITION_WINDOW:]
            audio_recent = recent[recent != BEAT_TOKEN_ID]
            for tid in audio_recent.unique():
                if logits[0, tid] > 0:
                    logits[0, tid] /= REPETITION_PENALTY
                else:
                    logits[0, tid] *= REPETITION_PENALTY

        # Check if next should be beat
        stripped = strip_beat_tokens(tokens[0])
        if stripped.shape[0] % BEAT_STRIDE_TOKENS == 0 and tokens.shape[1] > 0:
            tokens = torch.cat([
                tokens,
                torch.tensor([[BEAT_TOKEN_ID]], device=device)
            ], dim=-1)
            continue

        topv, topi = torch.topk(logits, top_k)
        probs = torch.softmax(topv, dim=-1)
        next_tok = topi.gather(-1, torch.multinomial(probs, 1))
        tokens = torch.cat([tokens, next_tok], dim=-1)

    return tokens


@torch.no_grad()
def generate_fine(coarse_model, fine_model, coarse_tokens):
    fine_model.eval()
    coarse_audio = strip_beat_tokens(coarse_tokens[0]).unsqueeze(0)

    chunk_size = 2048
    all_fine = []
    for i in range(0, coarse_audio.shape[1], chunk_size):
        chunk = coarse_audio[:, i:i + chunk_size]
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = fine_model(chunk)
        probs = torch.softmax(logits / TEMPERATURE, dim=-1)
        fine_chunk = torch.multinomial(
            probs.reshape(-1, CODEBOOK_SIZE), 1
        ).reshape(logits.shape[0], logits.shape[1], NUM_Q - 1)
        all_fine.append(fine_chunk[0].cpu())
        clear_mem()

    return torch.cat(all_fine, dim=0)


def generate_continuous(coarse_model, fine_model, total_seconds=10.0,
                        context=8192, temperature=0.9, top_k=100,
                        prompt_tokens=None):
    total_coarse = int(total_seconds * COARSE_TOKENS_PER_SEC)

    if prompt_tokens is not None:
        prompt_with_beats = inject_beat_tokens(prompt_tokens)
        tokens = prompt_with_beats.unsqueeze(0).to(device)
    else:
        first_tok = torch.randint(0, CODEBOOK_SIZE, (1, 1), device=device)
        tokens = torch.cat([
            torch.tensor([[BEAT_TOKEN_ID]], device=device),
            first_tok
        ], dim=-1)

    print(f"Generating {total_coarse} coarse tokens (~{total_seconds}s)...")

    with tqdm(total=total_coarse, initial=max(0, strip_beat_tokens(tokens[0]).shape[0])) as pbar:
        while strip_beat_tokens(tokens[0]).shape[0] < total_coarse:
            batch = min(128, total_coarse)
            prev = strip_beat_tokens(tokens[0]).shape[0]
            tokens = generate_coarse(coarse_model, tokens, max_new=batch,
                                     context=context, temperature=temperature,
                                     top_k=top_k)
            pbar.update(strip_beat_tokens(tokens[0]).shape[0] - prev)

    print("Generating fine codebooks...")
    fine_tokens = generate_fine(coarse_model, fine_model, tokens)

    coarse_audio = strip_beat_tokens(tokens[0])
    usable = min(coarse_audio.shape[0], fine_tokens.shape[0])
    return coarse_audio[:usable], fine_tokens[:usable]


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)
    print(f"  -> Saved: epoch {epoch}, loss {loss:.4f}  [{path}]")


def load_checkpoint(model, optimizer, scheduler=None, path="checkpoint.pt"):
    if not os.path.exists(path):
        print("No checkpoint found.")
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    print(f"Resumed from epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
    return ckpt["epoch"]


def save_model(model, path="model.pt"):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model": raw.state_dict()}, path)
    print(f"Model saved to {path}")


def find_audio_files(folder, max_files=2000):
    files = []
    for root, dirs, filenames in os.walk(folder):
        dirs.sort()
        for f in sorted(filenames):
            if f.lower().endswith(".pt"):
                files.append(os.path.join(root, f))
    print(f"Found {len(files)} files, using up to {max_files}")
    return files[:max_files]


def encode_file(path):
    tok_path = os.path.splitext(path)[0] + ".tok2"

    if os.path.exists(tok_path):
        try:
            return torch.load(tok_path, weights_only=True)
        except Exception:
            pass

    waveform = load_audio(path)
    if waveform is None:
        return None
    try:
        coarse, fine_list = audio_to_tokens(waveform)
        torch.save({"coarse": coarse, "fine": fine_list}, tok_path)
        return {"coarse": coarse, "fine": fine_list}
    except Exception as e:
        print(f"  [SKIP ENCODE] {path} -> {e}")
        return None


class CoarseDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


class FineDataset(Dataset):
    def __init__(self, data_list, chunk_size):
        self.samples = []
        for data in data_list:
            coarse = data["coarse"]
            fine_list = data["fine"]

            coarse_audio = strip_beat_tokens(coarse)
            T_audio = coarse_audio.shape[0]

            fine_aligned = torch.zeros(T_audio, len(fine_list), dtype=torch.long)
            for k, fine_tok in enumerate(fine_list):
                usable = min(T_audio, fine_tok.shape[0])
                fine_aligned[:usable, k] = fine_tok[:usable]

            for i in range(0, T_audio - chunk_size + 1, chunk_size):
                audio_start = i
                audio_end = i + chunk_size

                coarse_audio_chunk = coarse_audio[audio_start:audio_end]
                coarse_with_beats = inject_beat_tokens(coarse_audio_chunk)

                fine_chunk = fine_aligned[audio_start:audio_end]

                self.samples.append((coarse_with_beats, fine_chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def coarse_collate(batch):
    return torch.stack(batch).long()


def fine_collate(batch):
    coarse_list, fine_list = zip(*batch)

    # Find max coarse length
    max_coarse = max(c.shape[0] for c in coarse_list)
    max_fine = max(f.shape[0] for f in fine_list)

    B = len(batch)
    coarse_padded = torch.full((B, max_coarse), BEAT_TOKEN_ID, dtype=torch.long)
    fine_padded = torch.zeros(B, max_fine, fine_list[0].shape[1], dtype=torch.long)

    for b, (c, f) in enumerate(zip(coarse_list, fine_list)):
        coarse_padded[b, :c.shape[0]] = c
        fine_padded[b, :f.shape[0]] = f

    return coarse_padded, fine_padded


def train_coarse_only(coarse_model, dataloader, optimizer, scheduler,
                      epochs, start_epoch, checkpoint_path):
    total_steps = len(dataloader) * epochs
    global _accum_step
    _accum_step = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0.0
        num_batches = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Coarse]", leave=True)

        for step, batch in enumerate(loop):
            batch = batch.to(device)
            global_step = epoch * len(dataloader) + step
            loss = train_step_coarse(coarse_model, optimizer, batch,
                                     global_step, total_steps, scheduler)
            total_loss += loss
            num_batches += 1
            loop.set_postfix(
                loss=f"{loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                ss=f"{get_scheduled_sampling_prob(global_step, total_steps):.2f}",
            )

        avg_loss = total_loss / max(num_batches, 1)
        print(f"\n  Epoch {epoch + 1} coarse loss: {avg_loss:.4f}\n")

        ckpt = checkpoint_path.replace("checkpoint", f"coarse_epoch{epoch + 1}")
        save_checkpoint(coarse_model, optimizer, scheduler, epoch + 1, avg_loss, ckpt)

    save_model(coarse_model, "coarse_model.pt")


def train_fine_only(coarse_model, fine_model, dataloader, optimizer, scheduler,
                    epochs, start_epoch, checkpoint_path):
    total_steps = len(dataloader) * epochs
    global _accum_step
    _accum_step = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0.0
        num_batches = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Fine]", leave=True)

        for step, (coarse_batch, fine_batch) in enumerate(loop):
            coarse_batch = coarse_batch.to(device)
            fine_batch = fine_batch.to(device)
            loss = train_step_fine(coarse_model, fine_model, optimizer,
                                   coarse_batch, fine_batch, scheduler)
            total_loss += loss
            num_batches += 1
            loop.set_postfix(
                loss=f"{loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_loss = total_loss / max(num_batches, 1)
        print(f"\n  Epoch {epoch + 1} fine loss: {avg_loss:.4f}\n")

        ckpt = checkpoint_path.replace("checkpoint", f"fine_epoch{epoch + 1}")
        save_checkpoint(fine_model, optimizer, scheduler, epoch + 1, avg_loss, ckpt)

    save_model(fine_model, "fine_model.pt")


def save_audio(audio, path="out.wav", sr=24000):
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    torchaudio.save(path, audio, sr)
    duration = audio.shape[-1] / sr
    print(f"Saved {path} ({duration:.1f}s)")


def test_roundtrip(path, seconds=10):
    print(f"\n--- Roundtrip test: {path} ---")
    wf = load_audio(path, max_seconds=seconds)
    coarse, fine_list = audio_to_tokens(wf)
    coarse_strip = strip_beat_tokens(coarse)

    audio_c = tokens_to_audio(coarse_strip, None)
    save_audio(audio_c, "test_coarse_only.wav")

    if fine_list:
        fine = torch.stack(fine_list, dim=1)  # [T, K]
        audio_f = tokens_to_audio(coarse_strip, fine)
        save_audio(audio_f, "test_full_rvq.wav")
    print("--- End roundtrip test ---\n")


def main():
    coarse_model = CoarseModel(
        vocab_size=COARSE_VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bar_d_model=BAR_D_MODEL,
    ).to(device)

    fine_model = FineModel(
        coarse_vocab_size=COARSE_VOCAB_SIZE,
        d_model=FINE_D_MODEL,
        n_heads=FINE_N_HEADS,
        n_layers=FINE_N_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    clear_mem()

    coarse_params = sum(p.numel() for p in coarse_model.parameters()) / 1e6
    fine_params = sum(p.numel() for p in fine_model.parameters()) / 1e6
    print(f"Coarse model: {coarse_params:.1f}M params")
    print(f"Fine model:   {fine_params:.1f}M params")
    print(f"Total:        {coarse_params + fine_params:.1f}M params")

    if TRAIN_STAGE == "coarse":
        optimizer = create_optimizer(coarse_model)
    else:
        optimizer = create_optimizer(fine_model)

    clear_mem()

    # train
    if MODE in ("train", "both"):
        print("=" * 50)
        print(f"  TRAINING: {TRAIN_STAGE.upper()}")
        print("=" * 50)

        start_epoch = 0
        if RESUME:
            stage_path = CHECKPOINT_PATH.replace("checkpoint", f"{TRAIN_STAGE}_epoch")
            import glob
            ckpts = glob.glob(stage_path + "*.pt")
            if ckpts:
                latest = max(ckpts, key=os.path.getmtime)
                if TRAIN_STAGE == "coarse":
                    start_epoch = load_checkpoint(coarse_model, optimizer, path=latest)
                else:
                    start_epoch = load_checkpoint(fine_model, optimizer, path=latest)

        print("\nLoading files...")
        files = find_audio_files(AUDIO_FOLDER, MAX_FILES)

        print("Encoding/loading cached tokens...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cached = sum(1 for f in files if os.path.exists(os.path.splitext(f)[0] + ".tok2"))
        print(f"  {cached}/{len(files)} cached")

        token_list = []
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = {executor.submit(encode_file, f): f for f in files}
            for future in tqdm(as_completed(futures), total=len(files),
                               desc="Loading" if cached == len(files) else "Encoding/Loading"):
                result = future.result()
                if result is not None:
                    token_list.append(result)

        if not token_list:
            print("ERROR: No audio encoded.")
            exit(1)

        total_audio = sum(strip_beat_tokens(t["coarse"]).shape[0] for t in token_list)
        print(f"\nTotal audio tokens: {total_audio:,} (~{total_audio / COARSE_TOKENS_PER_SEC:.0f}s)")

        if TRAIN_STAGE == "coarse":
            all_coarse = []
            for data in token_list:
                coarse = data["coarse"]
                for i in range(0, len(coarse) - CHUNK_SIZE + 1, CHUNK_SIZE):
                    all_coarse.append(coarse[i:i + CHUNK_SIZE])

            print(f"Coarse chunks: {len(all_coarse)}")
            if not all_coarse:
                print("ERROR: No chunks")
                exit(1)

            loader = DataLoader(
                CoarseDataset(all_coarse),
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=coarse_collate,
                num_workers=8,
                pin_memory=(device == "cuda"),
                persistent_workers=True,
            )

            total_steps = len(loader) * EPOCHS
            scheduler = CosineWarmupScheduler(
                optimizer,
                warmup_steps=min(2000, total_steps // 20),
                total_steps=total_steps,
            )

            if RESUME and start_epoch > 0:
                for _ in range(len(loader) * start_epoch):
                    scheduler.step()

            print(f"\nTraining coarse for {EPOCHS} epochs...")
            train_coarse_only(coarse_model, loader, optimizer, scheduler,
                              EPOCHS, start_epoch, CHECKPOINT_PATH)

            del loader, all_coarse
            clear_mem()

        elif TRAIN_STAGE == "fine":
            fine_dataset = FineDataset(token_list, CHUNK_SIZE)
            print(f"Fine samples: {len(fine_dataset)}")

            if len(fine_dataset) == 0:
                print("ERROR: No fine samples")
                exit(1)

            loader = DataLoader(
                fine_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                collate_fn=fine_collate,
                num_workers=8,
                pin_memory=(device == "cuda"),
                persistent_workers=True,
            )

            total_steps = len(loader) * EPOCHS
            scheduler = CosineWarmupScheduler(
                optimizer,
                warmup_steps=min(2000, total_steps // 20),
                total_steps=total_steps,
            )

            if RESUME and start_epoch > 0:
                for _ in range(len(loader) * start_epoch):
                    scheduler.step()

            print(f"\nTraining fine for {EPOCHS} epochs...")
            coarse_ckpt = "coarse_model.pt"
            if os.path.exists(coarse_ckpt):
                ckpt = torch.load(coarse_ckpt, map_location=device, weights_only=False)
                coarse_model.load_state_dict(ckpt["model"])
                print("Loaded coarse model")
            coarse_model.eval()

            train_fine_only(coarse_model, fine_model, loader, optimizer, scheduler,
                            EPOCHS, start_epoch, CHECKPOINT_PATH)

    # generate single audio clip
    if MODE in ("generate", "both"):
        print("=" * 50)
        print("  GENERATING")
        print("=" * 50)

        if MODE == "generate":
            ckpt = torch.load("coarse_epoch15_v2.pt", map_location=device, weights_only=False)
            coarse_model.load_state_dict(ckpt["model"])
            ckpt_fine = torch.load("fine_epoch15_v2.pt", map_location=device, weights_only=False)
            fine_model.load_state_dict(ckpt_fine["model"])
            print("Loaded both models")

        prompt_tokens = None
        if PROMPT_AUDIO is not None:
            print(f"Loading prompt: {PROMPT_AUDIO}")
            waveform = load_audio(PROMPT_AUDIO, max_seconds=PROMPT_SECONDS)
            if waveform is not None:
                prompt_tokens, _ = audio_to_tokens(waveform)
                print(f"Prompt: {prompt_tokens.shape[0]} tokens "
                      f"(~{prompt_tokens.shape[0] / COARSE_TOKENS_PER_SEC:.1f}s)")

        coarse_audio, fine_tokens = generate_continuous_fast(
            coarse_model, fine_model,
            total_seconds=GENERATE_SECONDS,
            context=CONTEXT_WINDOW,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            prompt_tokens=prompt_tokens,
        )

        print("Decoding to audio...")
        print("coarse_audio:", coarse_audio.shape)
        print("fine_tokens:", fine_tokens.shape)
        audio = tokens_to_audio(coarse_audio, fine_tokens)
        save_audio(audio, OUTPUT_PATH)

    print("\nDone!")


def run_rand():
    import random
    import re

    NUM_GENERATIONS = 12

    START_TEMP = 0.72
    TEMP_STEP = 0.04

    START_TOPK = 30
    TOPK_STEP = 5

    GENERATE_SECONDS = 40.0
    PROMPT_SECONDS = 15.0

    OUTPUT_DIR = "generated_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    coarse_model = CoarseModel(
        vocab_size=COARSE_VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        bar_d_model=BAR_D_MODEL,
    ).to(device)

    fine_model = FineModel(
        coarse_vocab_size=COARSE_VOCAB_SIZE,
        d_model=FINE_D_MODEL,
        n_heads=FINE_N_HEADS,
        n_layers=FINE_N_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    clear_mem()

    coarse_params = sum(p.numel() for p in coarse_model.parameters()) / 1e6
    fine_params = sum(p.numel() for p in fine_model.parameters()) / 1e6
    print(f"Coarse model: {coarse_params:.1f}M params")
    print(f"Fine model:   {fine_params:.1f}M params")
    print(f"Total:        {coarse_params + fine_params:.1f}M params")

    if TRAIN_STAGE == "coarse":
        optimizer = create_optimizer(coarse_model)
    else:
        optimizer = create_optimizer(fine_model)

    clear_mem()

    all_prompt_files = []

    for root, dirs, files in os.walk(AUDIO_FOLDER):
        for f in files:
            if f.lower().endswith(".pt"):
                all_prompt_files.append(os.path.join(root, f))

    if len(all_prompt_files) == 0:
        raise RuntimeError("No .pt files found!")

    print(f"Found {len(all_prompt_files)} prompt files")

    # model loading
    ckpt = torch.load("coarse_epoch15_v2.pt", map_location=device, weights_only=False)
    coarse_model.load_state_dict(ckpt["model"])

    ckpt_fine = torch.load("fine_epoch15_v2.pt", map_location=device, weights_only=False)
    fine_model.load_state_dict(ckpt_fine["model"])

    coarse_model.eval()
    fine_model.eval()

    print("Loaded both models")

    # generation
    for gen_idx in range(NUM_GENERATIONS):

        clear_mem()

        # Random prompt selection
        prompt_path = random.choice(all_prompt_files)

        # Increment sampling params
        temperature = START_TEMP + (gen_idx * TEMP_STEP)
        top_k = START_TOPK + (gen_idx * TOPK_STEP)

        print("\n" + "=" * 60)
        print(f"GENERATION {gen_idx + 1}/{NUM_GENERATIONS}")
        print(f"Prompt File : {prompt_path}")
        print(f"Temperature : {temperature:.2f}")
        print(f"Top-K       : {top_k}")
        print("=" * 60)

        waveform = load_audio(
            prompt_path,
            max_seconds=PROMPT_SECONDS
        )

        if waveform is None:
            print("Skipping invalid waveform")
            continue

        try:
            prompt_tokens, _ = audio_to_tokens(waveform)

        except Exception as e:
            print(f"Tokenization failed: {e}")
            continue

        print(
            f"Prompt tokens: {prompt_tokens.shape[0]} "
            f"(~{prompt_tokens.shape[0] / COARSE_TOKENS_PER_SEC:.1f}s)"
        )

        try:
            coarse_audio, fine_tokens = generate_continuous_fast(
                coarse_model,
                fine_model,
                total_seconds=GENERATE_SECONDS,
                context=CONTEXT_WINDOW,
                temperature=temperature,
                top_k=top_k,
                prompt_tokens=prompt_tokens,
            )

            print("Decoding audio...")

            audio = tokens_to_audio(coarse_audio, fine_tokens)

            base_name = os.path.splitext(os.path.basename(prompt_path))[0]
            base_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", base_name)
            out_name = f"generated_v2_{base_name}_temp{temperature:.2f}_topk{top_k}.wav"

            out_path = os.path.join(OUTPUT_DIR, out_name)
            save_audio(audio, out_path)
            print(f"Saved -> {out_path}")

        except Exception as e:
            print(f"Generation failed: {e}")

        clear_mem()

    print("\nALL GENERATIONS COMPLETE")


if __name__ == "__main__":
    # change to main() if need to train
    run_rand()