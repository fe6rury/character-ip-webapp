from pathlib import Path
from collections import Counter
from zipfile import ZipFile
from PIL import Image
import csv
import json
import math
import shutil
import unicodedata

CHARACTERS = {"B":"부기","BY":"벼리","G":"고양고양이","GP":"갓평이","H":"해치","HB":"해봄이","HM":"하모","J":"조아용","R":"랑랑이","T":"뚜니"}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

def image_files(path):
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)

def prepare_data(drive_root, product_archive, output):
    drive_root, product_archive, output = map(Path, (drive_root, product_archive, output))
    folders = {}
    for path in drive_root.iterdir():
        code = path.name.rsplit('(', 1)[-1].rstrip(')')
        if path.is_dir() and code in CHARACTERS:
            if code in folders:
                raise ValueError(f'중복 캐릭터 폴더: {code}')
            folders[code] = path
    if set(folders) != set(CHARACTERS):
        raise ValueError(f'드라이브 폴더 누락: {sorted(set(CHARACTERS) - set(folders))}')

    gallery = output / 'gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    inventory = {}
    gallery_rows = []
    for code, name in CHARACTERS.items():
        folder = folders[code]
        groups = {group: image_files(folder / group) for group in ('og', 'ep', 'hp', 'hn')}
        inventory[code] = {group: len(paths) for group, paths in groups.items()}
        if not groups['og']:
            raise ValueError(f'등록 원본 없음: {name}')
        for i, source in enumerate(groups['og'], 1):
            target = gallery / f'c_{code.lower()}__{i:03d}{source.suffix.lower()}'
            with Image.open(source) as original:
                original.load()
            shutil.copyfile(source, target)
            gallery_rows.append(dict(file=target.name, character=f'c_{code.lower()}', char_name=name,
                                     source=unicodedata.normalize('NFC',str(source.relative_to(drive_root)))))

    yolo = output / 'yolo'
    split_codes = {}
    counts = Counter()
    with ZipFile(product_archive) as package:
        rows = list(csv.DictReader(package.read('product_manifest.csv').decode('utf-8-sig').splitlines()))
        if len(rows) != 71 or len({row['image'] for row in rows}) != 71:
            raise ValueError('상품 사진 목록이 누락되거나 중복되었습니다.')
        for row in rows:
            split = row['split']
            if split == 'review':
                continue
            if split not in {'train','val','test'}:
                raise ValueError(f'지원하지 않는 분할: {split}')
            code = row['character_id']
            if code not in CHARACTERS or split_codes.setdefault(code, split) != split:
                raise ValueError(f'캐릭터 분할 오류: {code}')
            source = Path(row['prepared_image'])
            stem = Path(row['image']).stem
            if source.as_posix() != f'images/{split}/{stem}.jpg':
                raise ValueError(f'잘못된 이미지 경로: {source}')
            label = Path('labels') / split / (stem + '.txt')
            fields = package.read(label.as_posix()).decode().split()
            if len(fields) != 5 or fields[0] != '0':
                raise ValueError(f'라벨 형식 오류: {label}')
            x,y,w,h = map(float,fields[1:])
            if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0 or not (
                0 <= x-w/2 < x+w/2 <= 1 and 0 <= y-h/2 < y+h/2 <= 1
            ):
                raise ValueError(f'박스 좌표 오류: {label}')
            for member in (source,label):
                path = yolo / member
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(package.read(member.as_posix()))
            with Image.open(yolo / source) as photo:
                photo.load()
            counts[split] += 1
    if sum(counts.values()) != 68 or set(counts) != {'train','val','test'} or set(split_codes) != set(CHARACTERS):
        raise ValueError(f'상품 학습 세트 검증 실패: {dict(counts)}')

    with (output / 'og_meta.csv').open('w',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['file','character','char_name','source'])
        writer.writeheader()
        writer.writerows(gallery_rows)
    (yolo / 'dataset.yaml').write_text(
        'path: ' + json.dumps(str(yolo.resolve())) + '\n'
        'train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: character_print\n',
        encoding='utf-8',
    )
    summary = {'gallery':len(gallery_rows),'drive_inventory':inventory,
               'yolo_products':dict(counts),'product_review':3}
    (output / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary
