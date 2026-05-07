###### %pip install torch torchaudio
# %pip install encodec
# %pip install tqdm
# %pip install ffmpeg-python
# %pip install soundfile

import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm
import math

AUDIO_FOLDER = "path_to_pt_folder"
CHECKPOINT_PATH = "checkpoint21.pt"
OUTPUT_PATH = "generated.wav"

MAX_FILES = 2000
MAX_AUDIO_SECONDS = 600

# model
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 32
MAX_LEN = 8192
DROPOUT = 0.1

EPOCHS = 25
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
CHUNK_SIZE = 8192
RESUME = True

GENERATE_SECONDS = 30.0
TEMPERATURE = 0.8
TOP_K = 100
CONTEXT_WINDOW = 8192
PROMPT_AUDIO = "path_to_prompt_audio_pt"
PROMPT_SECONDS = 10.0

MODE = "generate"  # "train", "generate", or "both"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# torch.backends.cuda.matmul.allow_tf32 = True

encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(3.0)
encodec_model = encodec_model.to(device)
encodec_model.eval()

with torch.no_grad():
    _dummy = torch.zeros(1, 1, 24000, device=device)
    _encoded = encodec_model.encode(_dummy)
    NUM_Q = _encoded[0][0].shape[1]
print(f"Active codebooks (NUM_Q): {NUM_Q}")

CODEBOOK_SIZE = 1024
VOCAB_SIZE = NUM_Q * CODEBOOK_SIZE
ENCODEC_HOP = 320
TOKENS_PER_SECOND = (24000 / ENCODEC_HOP) * NUM_Q

import subprocess
import tempfile


def load_audio(path, sr=24000, max_seconds=None):
    try:
        waveform = torch.load(path, weights_only=True).float()
        if waveform is None or waveform.numel() == 0:
            return None
        if max_seconds is not None:
            waveform = waveform[:, :int(sr * max_seconds)]
        return waveform
    except Exception as e:
        print(f"[SKIP] {path} -> {e}")
        return None


def encode_file(path):
    waveform = load_audio(path)
    if waveform is None:
        return None
    try:
        tokens = audio_to_tokens(waveform)
        return tokens
    except Exception as e:
        print(f"  [SKIP ENCODE] {path} -> {e}")
        return None


def audio_to_tokens(waveform, sr=24000):
    waveform = waveform.to(device)

    waveform = convert_audio(
        waveform,
        sr,
        encodec_model.sample_rate,
        encodec_model.channels,
    )

    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        encoded = encodec_model.encode(waveform)

    codes = encoded[0][0]
    codes = codes.squeeze(0)

    offsets = (
            torch.arange(NUM_Q, device=codes.device).unsqueeze(1) * CODEBOOK_SIZE
    )
    codes = codes + offsets

    T = codes.shape[1]
    delayed = torch.zeros(NUM_Q, T + NUM_Q, dtype=codes.dtype)
    for k in range(NUM_Q):
        delayed[k, k:k + T] = codes[k]

    tokens = delayed.permute(1, 0).reshape(-1)
    return tokens.cpu().long()


def tokens_to_audio(tokens):
    tokens = tokens.to(device)

    total_frames = tokens.shape[0] // NUM_Q
    codes = tokens.reshape(total_frames, NUM_Q).transpose(0, 1)

    T = total_frames - NUM_Q
    undelayed = torch.zeros(NUM_Q, T, dtype=codes.dtype, device=codes.device)
    for k in range(NUM_Q):
        undelayed[k] = codes[k, k:k + T]

    offsets = (
            torch.arange(NUM_Q, device=undelayed.device).unsqueeze(1) * CODEBOOK_SIZE
    )
    undelayed = undelayed - offsets
    undelayed = undelayed.clamp(0, CODEBOOK_SIZE - 1)

    codes_out = undelayed.unsqueeze(0)

    with torch.no_grad():
        decoded = encodec_model.decode([(codes_out, None)])

    return decoded[0].cpu()


class AudioGPT(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model=768,
            n_heads=12,
            n_layers=12,
            max_len=4096,
            dropout=0.1,
    ):
        super().__init__()
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.drop(self.token_emb(x) + self.pos_emb(positions))

        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )

        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)


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
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, d):
        self.step_count = d["step_count"]


scaler = torch.amp.GradScaler(enabled=(device == "cuda"))


def train_step(model, optimizer, tokens, scheduler=None):
    model.train()
    x = tokens[:, :-1]
    y = tokens[:, 1:]

    with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
        logits = model(x)
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()

    if scheduler is not None:
        scheduler.step()

    return loss.item()


@torch.no_grad()
def generate(model, start_tokens, max_new=512, context=4096, temperature=0.9, top_k=120):
    model.eval()
    tokens = start_tokens

    REPETITION_PENALTY = 1.2
    REPETITION_WINDOW = 600

    for _ in range(max_new):
        if tokens.shape[1] <= context:
            inp = tokens
        else:
            start = tokens.shape[1] - context
            start = math.ceil(start / NUM_Q) * NUM_Q
            inp = tokens[:, start:]

        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(inp)[:, -1] / temperature

        # repetition penalty
        if tokens.shape[1] > REPETITION_WINDOW:
            recent = tokens[0, -REPETITION_WINDOW:]
            for token_id in recent.unique():
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= REPETITION_PENALTY
                else:
                    logits[0, token_id] *= REPETITION_PENALTY

        # codebook masking
        codebook_idx = tokens.shape[1] % NUM_Q
        mask = torch.full_like(logits, float("-inf"))
        mask[:, codebook_idx * CODEBOOK_SIZE:(codebook_idx + 1) * CODEBOOK_SIZE] = 0
        logits = logits + mask

        # top-k sampling
        topv, topi = torch.topk(logits, top_k)
        probs = torch.softmax(topv, dim=-1)
        next_token = topi.gather(-1, torch.multinomial(probs, 1))
        tokens = torch.cat([tokens, next_token], dim=-1)

    return tokens


def generate_continuous(model, total_seconds=10, context=2048, temperature=0.9,
                        top_k=40, prompt_tokens=None):
    total_tokens = int(total_seconds * TOKENS_PER_SECOND)

    if prompt_tokens is not None:
        tokens = prompt_tokens.unsqueeze(0).to(device)
    else:
        tokens = torch.randint(0, CODEBOOK_SIZE, (1, 1), device=device)

    print(f"Generating {total_tokens} tokens (~{total_seconds}s of audio)...")

    with tqdm(total=total_tokens, initial=tokens.shape[1]) as pbar:
        while tokens.shape[1] < total_tokens:
            batch_size = min(256, total_tokens - tokens.shape[1])
            tokens = generate(
                model, tokens,
                max_new=batch_size,
                context=context,
                temperature=temperature,
                top_k=top_k,
            )
            pbar.update(batch_size)

    tokens = tokens[0, :total_tokens]
    usable = (tokens.shape[0] // NUM_Q) * NUM_Q
    return tokens[:usable]


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path="checkpoint.pt"):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )
    print(f"  -> Checkpoint saved: epoch {epoch}, loss {loss:.4f}")


def load_checkpoint(model, optimizer, scheduler=None, path="checkpoint.pt"):
    if not os.path.exists(path):
        print("No checkpoint found, starting fresh.")
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
    torch.save({"model": model.state_dict()}, path)
    print(f"Model saved to {path}")


def find_audio_files(folder, max_files=2000):
    files = []
    for root, dirs, filenames in os.walk(folder):
        dirs.sort()
        for f in sorted(filenames):
            if f.lower().endswith(".pt"):
                files.append(os.path.join(root, f))
    print(f"Found {len(files)} preprocessed files, using up to {max_files}")
    return files[:max_files]


def chunk_tokens(tokens, chunk_size=2048):
    chunks = []
    for i in range(0, len(tokens) - chunk_size + 1, chunk_size):
        chunks.append(tokens[i: i + chunk_size])
    return chunks


def build_chunks(token_list, chunk_size=2048):
    all_chunks = []
    for tokens in token_list:
        all_chunks.extend(chunk_tokens(tokens, chunk_size))
    return all_chunks


class AudioDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def collate_fn(batch):
    return torch.stack(batch).long()


def train_loop(model, dataloader, optimizer, scheduler, epochs=10,
               start_epoch=0, checkpoint_path="checkpoint.pt"):
    for epoch in range(start_epoch, start_epoch + epochs):
        total_loss = 0
        num_batches = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)

        for batch in loop:
            batch = batch.to(device)
            loss = train_step(model, optimizer, batch, scheduler)
            total_loss += loss
            num_batches += 1
            loop.set_postfix(
                loss=f"{loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_loss = total_loss / max(num_batches, 1)
        print(f"\n  Epoch {epoch + 1} avg loss: {avg_loss:.4f}\n")
        checkpoint_path = checkpoint_path.replace("checkpoint", f"checkpoint{epoch + 1}")
        save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_loss, checkpoint_path)

    save_model(model)


def save_audio(audio, path="out.wav", sr=24000):
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    torchaudio.save(path, audio, sr)
    duration = audio.shape[-1] / sr
    print(f"Saved {path} ({duration:.1f}s)")


if __name__ == "__main__":

    # build model
    model = AudioGPT(
        VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {param_count:.1f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    if MODE in ("train", "both"):
        print("=" * 50)
        print("  TRAINING")
        print("=" * 50)

        start_epoch = 0
        if RESUME:
            start_epoch = load_checkpoint(model, optimizer, path=CHECKPOINT_PATH)

        print("\nLoading preprocessed files...")
        files = find_audio_files(AUDIO_FOLDER, MAX_FILES)

        print("Encoding audio to tokens...")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        token_list = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(encode_file, f): f for f in files}
            for future in tqdm(as_completed(futures), total=len(files), desc="Encoding"):
                result = future.result()
                if result is not None:
                    token_list.append(result)

        if not token_list:
            print("ERROR: No audio was successfully encoded.")
            exit(1)

        total_tokens = sum(t.shape[0] for t in token_list)
        print(f"\nTotal tokens: {total_tokens:,} (~{total_tokens / TOKENS_PER_SECOND:.0f}s of audio)")

        print(f"Chunking into sequences of {CHUNK_SIZE}...")
        chunks = build_chunks(token_list, chunk_size=CHUNK_SIZE)
        print(f"Training chunks: {len(chunks)}")

        if not chunks:
            print("ERROR: No chunks. Audio too short for chunk size.")
            print(f"  Longest sequence: {max(t.shape[0] for t in token_list)}")
            print(f"  Chunk size: {CHUNK_SIZE}")
            exit(1)

        loader = DataLoader(
            AudioDataset(chunks),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=(device == "cuda"),
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

        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                print("torch.compile enabled")
            except Exception:
                print("torch.compile not available, continuing without it")

        print(f"\nTraining for {EPOCHS} epochs...")
        train_loop(model, loader, optimizer, scheduler, EPOCHS, start_epoch, CHECKPOINT_PATH)

    if MODE in ("generate", "both"):
        print("=" * 50)
        print("  GENERATING")
        print("=" * 50)

        if MODE == "generate":
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            # strip _orig_mod. prefix if saved from a torch.compile'd model
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print(f"Loaded model from {CHECKPOINT_PATH}")

        prompt_tokens = None
        if PROMPT_AUDIO is not None:
            print(f"Loading prompt audio: {PROMPT_AUDIO}")
            waveform = load_audio(PROMPT_AUDIO, max_seconds=PROMPT_SECONDS)
            if waveform is not None:
                prompt_tokens = audio_to_tokens(waveform)
                print(f"Prompt: {prompt_tokens.shape[0]} tokens (~{prompt_tokens.shape[0] / TOKENS_PER_SECOND:.1f}s)")

        tokens = generate_continuous(
            model,
            total_seconds=GENERATE_SECONDS,
            context=CONTEXT_WINDOW,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            prompt_tokens=prompt_tokens,
        )

        print("Decoding tokens to audio...")
        audio = tokens_to_audio(tokens)
        save_audio(audio, OUTPUT_PATH)

    print("\nDone!")