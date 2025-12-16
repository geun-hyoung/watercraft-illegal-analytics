# =============================================================
# 과업 2-1: 수상 레저 속 번호판 탐지 모델 학습
# =============================================================
# 
# 기능:
# - data/raw/plate_detection/ 폴더의 이미지와 라벨을 사용하여 YOLO 모델 학습
# - 이미지와 라벨을 train/val로 8:2 분할
# - YOLOv11s 사전 학습 모델을 기반으로 번호판 탐지 모델 학습
# - 학습 완료 후 model/plate_detection_baseline.pt로 저장
# - 학습 결과 그래프 및 평가 지표 출력
#
# 입력: data/raw/plate_detection/images/, data/raw/plate_detection/labels/
# 출력: model/plate_detection_baseline.pt (학습된 모델)
# =============================================================

from pathlib import Path
import os, shutil, random, yaml, csv, math
from ultralytics import YOLO

# ---------- 0) 기본 설정 ----------
# OpenMP 라이브러리 중복 로드 경고 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==== Robust pandas shim (optional, 없으면 자동 대체) ====
try:
    import pandas as _pd  # noqa
except Exception:
    import sys, types, importlib.machinery as _machinery

    class _Series(list):
        def to_list(self): return list(self)
        def tolist(self):  return list(self)

    class _Index(list):
        @property
        def values(self): return list(self)

    class _RangeIndex(_Index):
        def __init__(self, start=0, stop=None, step=1):
            if stop is None: stop = start; start = 0
            super().__init__(range(start, stop, step))

    class _DF:
        def __init__(self, data=None, columns=None, *a, **k):
            self._data = {}
            if isinstance(data, dict):
                for k2, v2 in data.items():
                    self._data[str(k2)] = list(v2 if isinstance(v2, (list, tuple)) else [v2])
                self._columns = list(self._data.keys())
            else:
                self._columns = list(columns) if columns is not None else []
            if 'Suffix' not in self._data:
                self._data['Suffix'] = []
                if 'Suffix' not in self._columns:
                    self._columns.append('Suffix')
            self._index = _RangeIndex(len(next(iter(self._data.values()), [])))

        def __getitem__(self, key):
            if isinstance(key, str):
                return _Series(self._data.get(key, []))
            return self

        def __setitem__(self, key, val):
            self._data[str(key)] = list(val if isinstance(val, (list, tuple)) else [val])
            if key not in self._columns: self._columns.append(str(key))

        def __getattr__(self, name):
            if name in self._data:
                return _Series(self._data[name])
            lname = name.lower()
            for k in self._data.keys():
                if k.lower() == lname:
                    return _Series(self._data[k])
            return _Series([])

        def to_csv(self, *a, **k): pass
        def to_excel(self, *a, **k): pass
        def to_dict(self, orient="dict"):
            if orient == "list":
                return {k: list(v) for k, v in self._data.items()}
            return {"data": {k: list(v) for k, v in self._data.items()}}

        @property
        def columns(self): return _Index(self._columns)
        @property
        def index(self):   return self._index

    def _read_csv(*a, **k):  # safe empty DF
        return _DF({})

    class _ExcelWriter:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass

    _pandas = types.ModuleType("pandas")
    _pandas.__spec__ = _machinery.ModuleSpec("pandas", loader=None)
    _pandas.__file__ = "<shim:pandas>"
    _pandas.DataFrame = _DF
    _pandas.Series = _Series
    _pandas.Index = _Index
    _pandas.RangeIndex = _RangeIndex
    _pandas.read_csv = _read_csv
    _pandas.ExcelWriter = _ExcelWriter
    def _concat(objs, *a, **k): return _DF({})
    def _to_numeric(*a, **k):   return _Series([])
    _pandas.concat = _concat
    _pandas.to_numeric = _to_numeric
    sys.modules["pandas"] = _pandas

def _sanitize_path(s: str) -> str:
    return s.replace('\u202a','').replace('\u202b','').replace('\u202c','').replace('\u202d','').replace('\u202e','')

# === 사용자 경로 수정 ===
# 프로젝트 루트 기준으로 경로 설정
BASE_DIR = Path(__file__).parent.parent
IMG_DIR   = BASE_DIR / "data" / "raw" / "plate_detection" / "images"
LABEL_DIR = BASE_DIR / "data" / "raw" / "plate_detection" / "labels"
assert IMG_DIR.exists(),  f"이미지 폴더 없음: {IMG_DIR}"
assert LABEL_DIR.exists(), f"라벨 폴더 없음: {LABEL_DIR}"

# ---------- 하이퍼파라미터 설정 ----------
EPOCHS      = 100                # 학습 에포크 수
IMGSZ       = 1280               # 학습/검증 이미지 해상도 (1280x1280)
BATCH       = -1                 # 배치 크기 (-1: 자동 설정)
DEVICE      = 0                  # GPU 장치 번호 (0: 첫 번째 GPU, CPU: "cpu")
RUN_NAME    = "y11s_plate_det_baseline"  # 실행 이름
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # 지원 이미지 확장자

# ---------- 1) 이미지-라벨 매핑 ----------
# 이미지 파일과 해당하는 라벨 파일(.txt)을 매칭
# 파일명이 동일한 이미지와 라벨을 쌍으로 구성
all_items = []
for p in sorted(IMG_DIR.rglob("*")):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        # 이미지 파일명(확장자 제외)과 동일한 이름의 라벨 파일 찾기
        lab = LABEL_DIR / f"{p.stem}.txt"
        all_items.append({"img": p, "label": lab if lab.exists() else None})
print(f"[INFO] 총 이미지: {len(all_items)}, 라벨 있음: {sum(1 for x in all_items if x['label'])}, 라벨 없음: {sum(1 for x in all_items if not x['label'])}")

# ---------- 2) 8:2 분할 (라벨 유/무 기준으로 각각 8:2) ----------
# 라벨이 있는 이미지와 없는 이미지를 각각 train:val = 8:2로 분할
SEED = 42                        # 랜덤 시드 (재현 가능성)
TRAIN_RATIO = 0.8                # 학습 데이터 비율 (80%)

# 라벨이 있는 이미지(pos_items)와 없는 이미지(neg_items)로 분류
pos_items, neg_items = [], []
for it in all_items:
    (pos_items if it["label"] else neg_items).append(it)

rng = random.Random(SEED)
rng.shuffle(pos_items)
rng.shuffle(neg_items)

n_pos = len(pos_items)
n_neg = len(neg_items)
n_pos_train = int(n_pos * TRAIN_RATIO)
n_neg_train = int(n_neg * TRAIN_RATIO)

splits = {
    "train": pos_items[:n_pos_train] + neg_items[:n_neg_train],
    "val":   pos_items[n_pos_train:] + neg_items[n_neg_train:]}
rng.shuffle(splits["train"])
rng.shuffle(splits["val"])

print(f"[SPLIT] total={len(all_items)} (pos={n_pos}, neg={n_neg})")
print(f"        train={len(splits['train'])} (pos={n_pos_train}, neg={n_neg_train})")
print(f"        val  ={len(splits['val'])}   (pos={n_pos-n_pos_train}, neg={n_neg-n_neg_train})")

# ---------- 3) YOLO 디렉토리 구성 ----------
DATASET_ROOT = Path.cwd() / "boats_plate_dataset"
for sub in ["images/train","images/val","labels/train","labels/val"]:
    (DATASET_ROOT / sub).mkdir(parents=True, exist_ok=True)

# ---------- 4) 복사 (라벨 QA/네거티브 샘플링 없이 그대로) ----------
def copy_pair(item, split):
    img_src = item["img"]; lbl_src = item["label"]
    img_dst = DATASET_ROOT / "images" / split / (img_src.stem + img_src.suffix.lower())
    lbl_dst = DATASET_ROOT / "labels" / split / (img_src.stem + ".txt")
    shutil.copy2(img_src, img_dst)
    if lbl_src and lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)  # 라벨 그대로
    else:
        lbl_dst.write_text("", encoding="utf-8")  # 라벨 없으면 빈 파일

for split, items in splits.items():
    for it in items:
        copy_pair(it, split)
print(f"[INFO] 데이터 구성 완료: {DATASET_ROOT}")

# ---------- 5) data.yaml ----------
DATA_YAML = DATASET_ROOT / "data.yaml"
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.dump({
        "path": str(DATASET_ROOT.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc": 1,
        "names": {0: "plate"}
    }, f, allow_unicode=True)
print(f"[INFO] data.yaml 생성: {DATA_YAML}")

# ---------- 6) 모델 학습 ----------
# YOLOv11s 사전 학습 모델을 기반으로 번호판 탐지 모델 학습
# 번호판 특성상 좌우반전, 모자이크 등은 사용하지 않음
model = YOLO("yolo11s.pt")  # 사전 학습된 YOLOv11s 모델 로드
train_results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=0,
    name="y11s_plate_det_aug",
    project="runs/detect_aug",
    save=True,
    val=True,
    amp=False,

    cos_lr=False,
    warmup_epochs=0,

    mosaic=0.0,        # 번호판은 모자이크 과하면 망가질 수 있어서 보통 끔
    mixup=0.0,
    copy_paste=0.0,
    translate=0.10,
    degrees=10.0,
    scale=0.50,
    shear=2.0,
    perspective=0.001,

    fliplr=0.0,        # 좌우반전은 번호판 의미 바뀔 수 있어서 보통 끔
    flipud=0.0,
    hsv_h=0.015, hsv_s=0.70, hsv_v=0.40,
    rect=True,
    close_mosaic=0,
    cache=False,
    plots=True)

RUN_DIR = Path(train_results.save_dir)
WEIGHTS_DIR = RUN_DIR / "weights"
BEST = WEIGHTS_DIR / "best.pt"
LAST = WEIGHTS_DIR / "last.pt"
use_weight = BEST if BEST.exists() else LAST
assert use_weight.exists(), f"체크포인트 없음: {WEIGHTS_DIR}"
print(f"[TRAIN] 완료 → RUN_DIR={RUN_DIR.name}, CKPT={( 'best.pt' if use_weight==BEST else 'last.pt')}")

RUN_DIR = Path(train_results.save_dir)
WEIGHTS_DIR = RUN_DIR / "weights"
BEST = WEIGHTS_DIR / "best.pt"
LAST = WEIGHTS_DIR / "last.pt"
use_weight = BEST if BEST.exists() else LAST
assert use_weight.exists(), f"체크포인트 없음: {WEIGHTS_DIR}"
print(f"[TRAIN] 완료 → RUN_DIR={RUN_DIR.name}, CKPT={( 'best.pt' if use_weight==BEST else 'last.pt')}")

# === 추가: 다른 경로/이름으로 고정 저장 ===
BASE_DIR = Path(__file__).parent.parent
FINAL_DIR = BASE_DIR / "model"     # ← model 폴더
FINAL_DIR.mkdir(parents=True, exist_ok=True)
FINAL_WEIGHT = FINAL_DIR / "plate_detection_baseline.pt" # ← 새로운 파일명
shutil.copy2(use_weight, FINAL_WEIGHT)
print(f"[SAVE] 추론용 고정 경로로 복사: {FINAL_WEIGHT}")

# ---------- 8) 평가(Val, 학습과 동일 해상도) ----------
val_model = YOLO(str(use_weight))
val_metrics = val_model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=IMGSZ,      # train과 동일
    device=DEVICE,
    workers=0,
    project=str(RUN_DIR.parent),
    name=f"{RUN_DIR.name}_val",
    plots=False)
print("[EVAL] 검증(val) 완료")

# ---------- 9) 결과 읽기 및 발표용 Fig/표 ----------
results_csv = RUN_DIR / "results.csv"

def _read_results_csv(csv_path: Path):
    if not csv_path.exists(): return [], []
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows: return [], []
    return rows[0], rows[1:]

def _get_epoch_index(hdr):
    for i, c in enumerate(hdr):
        if c.strip().lower() == "epoch":
            return i
    return None

header, rows = _read_results_csv(results_csv)

def _dedup_latest_by_epoch(header, rows):
    ep_col = _get_epoch_index(header)
    if ep_col is None:
        return rows
    latest = {}
    for r in rows:
        try:
            ep = int(float(r[ep_col]))
        except Exception:
            continue
        latest[ep] = r
    return [latest[k] for k in sorted(latest.keys())]

def _col(h, name_a, name_b=None):
    if name_a in h: return h.index(name_a)
    if name_b and name_b in h: return h.index(name_b)
    return None

if header and rows:
    rows_ep = _dedup_latest_by_epoch(header, rows)

    try:
        import matplotlib.pyplot as plt

        def to_float_safe(x):
            try: return float(x)
            except Exception: return math.nan

        epochs = []
        if _get_epoch_index(header) is not None:
            for r in rows_ep:
                try: epochs.append(int(float(r[_get_epoch_index(header)])))
                except Exception: epochs.append(math.nan)
        else:
            epochs = list(range(1, len(rows_ep)+1))

        m50_col = _col(header, "metrics/mAP50", "metrics/mAP50(B)")
        m95_col = _col(header, "metrics/mAP50-95", "metrics/mAP50-95(B)")

        if m50_col is not None or m95_col is not None:
            plt.figure(figsize=(10,5))
            if m50_col is not None:
                ys50 = [to_float_safe(r[m50_col]) if m50_col < len(r) else math.nan for r in rows_ep]
                plt.plot(epochs, ys50, label="Val mAP@0.5")
            if m95_col is not None:
                ys95 = [to_float_safe(r[m95_col]) if m95_col < len(r) else math.nan for r in rows_ep]
                plt.plot(epochs, ys95, label="Val mAP@0.5:0.95")
            plt.xlabel("Epoch"); plt.ylabel("mAP")
            plt.title("Val mAP@0.5 & mAP@0.5:0.95 vs. Epoch")
            plt.legend(loc="best"); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()
        else:
            print("[INFO] mAP 컬럼을 찾지 못했습니다. (train(..., val=True) 필요)")
    except Exception as e:
        print("[WARN] 그래프 표시 실패:", e)

    map50_col = _col(header, "metrics/mAP50", "metrics/mAP50(B)")
    prec_col  = _col(header, "metrics/precision", "metrics/precision(B)")
    rec_col   = _col(header, "metrics/recall", "metrics/recall(B)")

    best_row = None; best_val = -1
    if map50_col is not None:
        for r in rows_ep:
            try:
                v = float(r[map50_col])
                if v > best_val:
                    best_val = v; best_row = r
            except: pass

    if best_row is not None:
        ep_col = _get_epoch_index(header)
        ep = int(float(best_row[ep_col])) if ep_col is not None else None

        def _val(col):
            try: return float(best_row[col]) if col is not None else float('nan')
            except: return float('nan')

        p  = _val(prec_col)
        rc = _val(rec_col)
        m5 = _val(map50_col)

        print("\n=== Best Epoch (by Val mAP@0.5) ===")
        print(f"epoch: {ep} | precision: {p:.4f} | recall: {rc:.4f} | mAP@0.5: {m5:.4f}")
    else:
        print("\n[INFO] mAP@0.5 열을 찾지 못했습니다.")
else:
    print("\n[INFO] results.csv를 찾지 못했습니다. 학습이 정상 종료/저장되었는지 확인하세요.")

# ---------- 10) 최종 validation 요약 ----------
def _normalize_val_results(vm):
    d = {}
    if hasattr(vm, "results_dict") and isinstance(vm.results_dict, dict) and vm.results_dict:
        d = dict(vm.results_dict)
    keys_lower = {k.lower(): k for k in d.keys()} if d else {}

    def pick(name_list, fallback=None):
        for k in name_list:
            if k in d: return d[k]
            if d and k.lower() in keys_lower: return d[keys_lower[k.lower()]]
        return fallback

    out = {
        "precision": pick(["metrics/precision(B)", "metrics/precision"], getattr(getattr(vm, "box", None), "p", None)),
        "recall":    pick(["metrics/recall(B)",    "metrics/recall"],    getattr(getattr(vm, "box", None), "r", None)),
        "mAP50":     pick(["metrics/mAP50(B)",     "metrics/mAP50"],     getattr(getattr(vm, "box", None), "map50", None)),
        "mAP50-95":  pick(["metrics/mAP50-95(B)",  "metrics/mAP50-95"],  getattr(getattr(vm, "box", None), "map", None)),}
    for k, v in list(out.items()):
        try: out[k] = float(v)
        except Exception: pass
    return out

val_summary = _normalize_val_results(val_metrics)
print("\n=== Final Validation (val split, summary) ===")
for k in ["precision","recall","mAP50","mAP50-95"]:
    v = val_summary.get(k, None)
    print(f"{k:12s}: {v}" if v is not None else f"{k:12s}: N/A")