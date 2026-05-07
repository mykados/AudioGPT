import os
import torch
import torchaudio
import torchaudio.functional as F
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

INPUT_FOLDER  = "path_to_mp3"
OUTPUT_FOLDER = "path_to_output_pt"
TARGET_SR     = 24000
MAX_SECONDS   = None   # None = full file
NUM_WORKERS   = cpu_count()  # max your CPU

SUPPORTED = (".mp3", ".wav", ".flac", ".ogg", ".m4a")


def find_audio_files(folder):
    files = []
    for root, dirs, filenames in os.walk(folder):
        dirs.sort()
        for f in sorted(filenames):
            if f.lower().endswith(SUPPORTED):
                files.append(os.path.join(root, f))
    return files


def process_file(path, input_folder, output_folder, target_sr, max_seconds):
    try:
        # mirror folder structure
        rel_path = os.path.relpath(path, input_folder)
        out_path = os.path.join(
            output_folder,
            os.path.splitext(rel_path)[0] + ".pt"
        )

        # skip if already done
        if os.path.exists(out_path):
            return (path, "skipped")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        waveform, orig_sr = torchaudio.load(path)

        if waveform is None or waveform.numel() == 0:
            return (path, "empty")

        # trim
        if max_seconds is not None:
            waveform = waveform[:, :int(orig_sr * max_seconds)]

        # resample
        if orig_sr != target_sr:
            waveform = F.resample(waveform, orig_sr, target_sr)

        # normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        torch.save(waveform.half(), out_path)

        return (path, "ok")

    except Exception as e:
        return (path, f"error: {e}")


if __name__ == "__main__":
    files = find_audio_files(INPUT_FOLDER)
    print(f"Found {len(files)} audio files")
    print(f"Using {NUM_WORKERS} workers")
    print(f"Output: {OUTPUT_FOLDER}")

    worker = partial(
        process_file,
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        target_sr=TARGET_SR,
        max_seconds=MAX_SECONDS,
    )

    ok = skipped = errors = 0

    with Pool(NUM_WORKERS) as pool:
        for path, status in tqdm(
            pool.imap_unordered(worker, files),
            total=len(files),
            desc="Processing"
        ):
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"[X] {path} -> {status}")


    print(f"\nDone: {ok} converted, {skipped} skipped, {errors} errors")