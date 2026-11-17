from pathlib import Path
from collections import Counter
from zipfile import ZipFile, ZIP_DEFLATED
import csv
import json
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from prepare_data import prepare_data

IMG_SIZE = 224
MODEL_ID = "facebook/dinov2-base"
S = {}

def preprocess(im: Image.Image) -> np.ndarray:
    """알파→흰색 합성, 비율 유지 정사각 패딩, 224 리사이즈, 정규화.

    코랩에서 갤러리 임베딩을 뽑을 때와 **똑같은** 전처리여야 한다.
    하나라도 다르면 유사도가 통째로 틀어진다.
    """
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255,) * 4), im)
    im = im.convert("RGB")

    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))

    arr = np.asarray(canvas.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), np.float32) / 255.0
    return ((arr - S["mean"]) / S["std"]).transpose(2, 0, 1)

def embed(im: Image.Image) -> np.ndarray:
    """이미지 1장 → L2 정규화된 768차원 벡터 (DINOv2 CLS 토큰)."""
    x = torch.from_numpy(preprocess(im)[None]).to(S["device"])
    with torch.no_grad():
        v = S["model"](pixel_values=x).last_hidden_state[:, 0]
    v = torch.nn.functional.normalize(v.float(), dim=1)
    return v.cpu().numpy()[0]

def overlap(box, truth):
    a, b, c, d = box
    x, y, z, w = truth
    intersection = max(0, min(c, z) - max(a, x)) * max(0, min(d, w) - max(b, y))
    union = (c-a)*(d-b) + (z-x)*(w-y) - intersection
    return intersection / (union + 1e-9)


def select_deployment(prepared, weight_dir, device):
    examples = []
    for path in sorted((prepared / 'yolo/images/val').glob('*.jpg')):
        label = prepared / 'yolo/labels/val' / (path.stem + '.txt')
        _, x, y, w, h = map(float, label.read_text().split())
        with Image.open(path) as photo:
            width, height = photo.size
        examples.append((path, ((x-w/2)*width, (y-h/2)*height,
                                (x+w/2)*width, (y+h/2)*height)))

    thresholds = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
    trials = []
    for name in ('best.pt', 'epoch5.pt', 'epoch6.pt', 'epoch8.pt', 'epoch9.pt', 'epoch11.pt', 'epoch13.pt'):
        model_path = weight_dir / name
        if not model_path.is_file():
            continue
        predictor = YOLO(str(model_path))
        outputs = predictor.predict([str(path) for path, _ in examples],
                                    conf=min(thresholds), imgsz=640, max_det=5,
                                    device=device, stream=True, verbose=False)
        predictions = []
        for (_, truth), output in zip(examples, outputs):
            predictions.append([(score, overlap(box, truth)) for box, score in
                                zip(output.boxes.xyxy.tolist(), output.boxes.conf.tolist())])
        for confidence in thresholds:
            kept = [[iou for score, iou in image if score >= confidence] for image in predictions]
            tp = sum(any(iou >= 0.50 for iou in image) for image in kept)
            fp = sum(map(len, kept)) - tp
            fn = len(examples) - tp
            f1 = 2*tp / max(1, 2*tp+fp+fn)
            trials.append((f1, tp/max(1, tp+fp), confidence, model_path, tp, fp, fn))

    f1, precision, confidence, path, tp, fp, fn = max(trials, key=lambda row: row[:3])
    report = dict(checkpoint=path.name, confidence=confidence, true_positives=tp,
                  false_positives=fp, false_negatives=fn, f1=f1, precision=precision,
                  validation_images=len(examples))
    print('Validation selection:', report, flush=True)
    return path, confidence, report


def execute(team, product_archive, work, epochs=25):
    started = time.time()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    prepared = work / 'prepared'
    inventory = prepare_data(team, product_archive, prepared)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, flush=True)
    if device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    detector_output = work / 'detector'
    last = detector_output / 'weights' / 'last.pt'
    best = detector_output / 'weights' / 'best.pt'
    if last.exists():
        detector = YOLO(str(last))
        try:
            detector.train(resume=True)
        except AssertionError as exc:
            if not best.exists() or 'nothing to resume' not in str(exc).lower():
                raise
    else:
        detector = YOLO('yolo11n.pt')
        detector.train(data=str(prepared / 'yolo/dataset.yaml'), imgsz=640, epochs=epochs,
                       batch=4 if device=='cpu' else 8, device=device, workers=0,
                       seed=42, project=str(work), name='detector', exist_ok=True,
                       patience=12, save_period=1, plots=False, verbose=False)
    best, confidence, selection = select_deployment(prepared, best.parent, device)
    detector = YOLO(str(best))
    test = detector.val(data=str(prepared / 'yolo/dataset.yaml'), split='test',
                        imgsz=640, device=device, batch=4, workers=0, plots=False)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    S.update(mean=np.array(processor.image_mean, np.float32), std=np.array(processor.image_std, np.float32),
             model=AutoModel.from_pretrained(MODEL_ID).to(device).eval(), device=device)
    rows = list(csv.DictReader((prepared / 'og_meta.csv').open(encoding='utf-8')))
    matrix = []
    for offset in range(0,len(rows),8):
        batch = []
        for row in rows[offset:offset+8]:
            with Image.open(prepared / 'gallery' / row['file']) as item:
                batch.append(preprocess(item))
        with torch.inference_mode():
            tensor = torch.from_numpy(np.stack(batch)).to(device)
            v = S['model'](pixel_values=tensor).last_hidden_state[:,0]
            matrix.append(torch.nn.functional.normalize(v.float(),dim=1).cpu().numpy())
    refs = np.concatenate(matrix).astype(np.float32)
    np.save(work / 'og_emb.npy',refs,allow_pickle=False)
    def top_code(sims):
        return rows[int(sims.argmax())]['character']
    by_product = {}
    with ZipFile(product_archive) as z:
        metadata = list(csv.DictReader(z.read('product_manifest.csv').decode('utf-8-sig').splitlines()))
    for row in metadata:
        if row['split'] != 'test':
            continue
        with Image.open(prepared / 'yolo' / row['prepared_image']) as im:
            im = im.convert('RGB')
        full = refs @ embed(im)
        sims = full.copy()
        prediction = detector.predict(im, imgsz=640, conf=confidence, max_det=5, device=device, verbose=False)[0]
        for x1,y1,x2,y2 in prediction.boxes.xyxy.tolist():
            margin=0.10*max(x2-x1,y2-y1)
            crop=im.crop((max(0,int(x1-margin)),max(0,int(y1-margin)),min(im.width,int(x2+margin+1)),min(im.height,int(y2+margin+1))))
            if min(crop.size)>=2:
                sims=np.maximum(sims,refs @ embed(crop))
        counts=by_product.setdefault(row['product'],Counter())
        truth='c_'+row['character_id'].lower()
        counts.update(total=1,full=int(top_code(full)==truth),combined=int(top_code(sims)==truth),detected=int(len(prediction.boxes)>0))
    report = dict(device=device, elapsed_seconds=round(time.time()-started), inventory=inventory,
                  detector_test={k:float(v) for k,v in test.results_dict.items()},
                  product_top1=by_product,
                  test_scope='상품 예시 15장(부기·갓평이). 10종 실촬영 일반화 검증에 해당하지 않음.')
    report['deployment_selection'] = selection
    report['deployment_confidence'] = confidence
    report_path=work/'evaluation.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    exported=work/'ai_oncamp_model_export.zip'
    with ZipFile(exported,'w',ZIP_DEFLATED) as z:
        z.write(best,'data/character_print.pt')
        z.write(work/'og_emb.npy','data/og_emb.npy')
        z.write(prepared/'og_meta.csv','data/og_meta.csv')
        z.write(report_path,'evaluation.json')
        for row in rows:
            z.write(prepared/'gallery'/row['file'],'data/og/'+row['file'])
    print(report_path.read_text(encoding='utf-8'),flush=True)
    return exported
