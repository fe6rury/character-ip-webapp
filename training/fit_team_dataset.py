import csv
import hashlib
import json
import os
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CODES = ('B', 'BY', 'G', 'GP', 'H', 'HB', 'HM', 'J', 'R', 'T')
POSITIVES = {'positive', 'known_en'}
IMAGE_GROUPS = {'og', 'or', 'ep', 'hp', 'hn'}


def live_images(root):
    result = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        code = folder.name.rsplit('(', 1)[-1].rstrip(')')
        is_en = folder.name.casefold().startswith('easy_negative')
        if not is_en and code not in CODES:
            continue
        for group in sorted(folder.iterdir()):
            if not group.is_dir() or (not is_en and group.name.casefold() not in IMAGE_GROUPS):
                continue
            for image in sorted(group.iterdir()):
                if image.is_file() and image.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                    key = unicodedata.normalize('NFC', image.relative_to(root).as_posix())
                    if key in result:
                        raise ValueError(f'Duplicate normalized path: {key}')
                    result[key] = image
    return result


def read_inputs(root, manifest):
    paths = live_images(Path(root))
    with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    declared = {r['path'] for r in rows}
    if len(declared) != len(rows):
        raise ValueError('Duplicate path in the training manifest.')
    aliases = {}
    original_by_digest = {}
    for row in rows:
        if row['source_group'] == 'og' and row['exact_duplicate_group'] and row['path'] in paths:
            original_by_digest.setdefault(row['exact_duplicate_group'], []).append(row['path'])
    for row in rows:
        key = row['path']
        digest = row['exact_duplicate_group']
        candidates = original_by_digest.get(digest, [])
        if key in paths or row['role'] != 'known_en' or len(candidates) != 1:
            continue
        original = paths[candidates[0]]
        if hashlib.sha256(original.read_bytes()).hexdigest() == digest:
            paths[key] = original
            aliases[key] = candidates[0]
    if declared != paths.keys():
        raise ValueError(f'Drive differs from audited snapshot: missing={len(declared - paths.keys())}, '
                         f'new={len(paths.keys() - declared)}')
    return rows, [paths[row['path']] for row in rows], aliases


def targets(rows):
    y = np.zeros((len(rows), len(CODES)), dtype=np.float32)
    mask = np.ones_like(y)
    duplicates = Counter(row['exact_duplicate_group'] for row in rows if row['exact_duplicate_group'])
    for i, row in enumerate(rows):
        role = row['role']
        if role in POSITIVES:
            y[i, CODES.index(row['character'])] = 1
        elif role == 'hard_negative':
            mask[i] = 0
            mask[i, CODES.index(row['negative_for'])] = 1
        elif role != 'unknown_en':
            raise ValueError(f'Unexpected role: {role}')
        group = row['exact_duplicate_group']
        if group:
            mask[i] /= duplicates[group]
    return y, mask


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit_head(embeddings, rows, epochs=30):
    y, mask = targets(rows)
    positives = (y * mask).sum(axis=0)
    negatives = ((1 - y) * mask).sum(axis=0)
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError('All ten classes need positive and negative evidence.')
    weights = mask * np.where(y == 1, 0.5 / positives, 0.5 / negatives) * len(rows)
    rng = np.random.default_rng(42)
    features = embeddings.astype(np.float32, copy=False) * np.sqrt(embeddings.shape[1])
    kernel = np.zeros((features.shape[1], len(CODES)), dtype=np.float32)
    bias = np.zeros(len(CODES), dtype=np.float32)
    for epoch in range(epochs):
        order = rng.permutation(len(rows))
        for indices in np.array_split(order, max(1, len(rows) // 256)):
            x = features[indices]
            error = (sigmoid(x @ kernel + bias) - y[indices]) * weights[indices]
            rate = 0.025 / (1 + epoch / 8)
            kernel -= rate * (x.T @ error / len(indices) + 0.001 * kernel)
            bias -= rate * error.mean(axis=0)
    return kernel * np.sqrt(embeddings.shape[1]), bias


def preprocess(source, mean, std):
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert('RGBA')
    image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('RGB')
    w, h = image.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), 'white')
    canvas.paste(image, ((side-w)//2, (side-h)//2))
    pixels = np.asarray(canvas.resize((224, 224), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return ((pixels - mean) / std).transpose(2, 0, 1)


def benchmark(features, rows):
    split = np.array([r['benchmark_split'] for r in rows])
    fit = np.flatnonzero(split == 'fit')
    kernel, bias = fit_head(features[fit], [rows[i] for i in fit])
    scores = sigmoid(features @ kernel + bias)
    correct = np.array([bool(r['character']) and CODES[scores[i].argmax()] == r['character']
                        for i,r in enumerate(rows)])
    known = np.array([bool(r['character']) for r in rows])
    peaks = scores.max(axis=1)
    def rates(group, threshold):
        positives = group & known
        negatives = group & ~known
        return dict(known_count=int(positives.sum()), unknown_count=int(negatives.sum()),
                    known_top1=int((positives & correct).sum()),
                    correctly_accepted=int((positives & correct & (peaks >= threshold)).sum()),
                    unknown_rejected=int((negatives & (peaks < threshold)).sum()))
    def metric(threshold):
        r = rates(split == 'calibration', threshold)
        return (r['correctly_accepted']/r['known_count'] + r['unknown_rejected']/r['unknown_count']) / 2
    threshold = max(np.linspace(0.02,0.995,196), key=lambda x:(metric(x),x))
    test = rates(split == 'test', threshold)
    test['by_character'] = {code: dict(n=int(sum(r['character']==code and r['benchmark_split']=='test' for r in rows)),
                  top1=int(sum(r['character']==code and r['benchmark_split']=='test' and correct[i]
                               for i,r in enumerate(rows)))) for code in CODES}
    return float(threshold), dict(training_indices=len(fit), calibration=rates(split=='calibration',threshold),
                                  grouped_test=test,
                                  note='Separate benchmark head. Grouped OG/EP and unknown EN, excludes HP and known EN in benchmark fitting. Final head fits all data after calibration.')


def train(team_root, manifest, output, model_source='facebook/dinov2-base', batch_size=16):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, paths, aliases = read_inputs(team_root, manifest)
    signature = hashlib.sha256(Path(manifest).read_bytes() + str(model_source).encode()).hexdigest()[:16]
    cache = output / signature
    cache.mkdir(exist_ok=True)
    workers = ThreadPoolExecutor(max_workers=8)
    processor = AutoImageProcessor.from_pretrained(model_source)
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f'DINOv2 embedding device={device}; files={len(paths)}', flush=True)
    model = AutoModel.from_pretrained(model_source).to(device).eval()
    if device == 'cuda':
        model.half()
    blocks = []
    for offset in range(0, len(paths), 128):
        shard = paths[offset:offset+128]
        name = cache / f'{offset:05d}.npy'
        if name.exists():
            block = np.load(name, allow_pickle=False)
            if block.shape != (len(shard), 768) or not np.isfinite(block).all():
                raise ValueError(f'Invalid saved embedding block: {name}')
        else:
            vectors = []
            for start in range(0, len(shard), batch_size):
                batch = list(workers.map(lambda path: preprocess(path,mean,std),
                                         shard[start:start+batch_size]))
                tensor = torch.from_numpy(np.stack(batch)).to(device=device,dtype=model.dtype)
                with torch.inference_mode():
                    feature = model(pixel_values=tensor).last_hidden_state[:, 0].float()
                    vectors.append(torch.nn.functional.normalize(feature, dim=1).cpu().numpy())
            block = np.concatenate(vectors)
            pending = name.with_suffix('.pending')
            with pending.open('wb') as handle:
                np.save(handle, block, allow_pickle=False)
            os.replace(pending, name)
        blocks.append(block)
        print(f'Features {min(offset+128,len(paths))}/{len(paths)}', flush=True)
    workers.shutdown()
    features = np.concatenate(blocks)
    threshold, evaluation = benchmark(features, rows)
    kernel, bias = fit_head(features, rows)
    head = output / 'character_head.npz'
    np.savez_compressed(head, classes=np.array(CODES), weights=kernel, bias=bias, review_threshold=np.array(threshold))
    audit = {'training_image_rows': len(rows), 'source_groups': dict(Counter(r['source_group'] for r in rows)),
             'roles': dict(Counter(r['role'] for r in rows)), 'device': device,
             'backbone': 'frozen DINOv2', 'trained_component': 'ten-class masked binary classification head',
             'grouped_benchmark': evaluation, 'review_threshold': threshold,
             'drive_exact_duplicate_aliases': aliases}
    (output / 'full_training_report.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    return head
