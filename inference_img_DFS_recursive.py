#!/usr/bin/env python3
# python inference_img_DFS_recursive.py \
#  --src ./temp2 --dst ./temp2_result --exp 4 --half

# !ffmpeg -framerate 240 -i temp2_result/%06d.png \
#       -c:v libx264 -profile:v high -crf 18 -pix_fmt yuv420p \
#       output.mp4

import os, re, cv2, gc, torch, argparse, warnings
from torch.nn import functional as F
from contextlib import nullcontext
warnings.filterwarnings("ignore")

# ------------------------- 인자 ------------------------- #
parser = argparse.ArgumentParser(description="Memory-efficient VFI (RIFE)")
parser.add_argument('--src', required=True, help='입력 폴더')
parser.add_argument('--dst', required=True, help='출력 폴더')
parser.add_argument('--exp', default=4, type=int, help='2^exp-1 개 중간 프레임')
parser.add_argument('--ratio', type=float, default=0, help='단일 비율 보간')
parser.add_argument('--half',  action='store_true', help='FP16 추론(메모리 절감)')
parser.add_argument('--modelDir', default='train_log', help='RIFE 모델 폴더')
parser.add_argument('--ext',    default='png', choices=['png', 'exr'])
args = parser.parse_args()

# ------------------------- 장치 ------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark = True, True
amp_ctx = torch.cuda.amp.autocast if (args.half and torch.cuda.is_available()) else nullcontext

# ------------------------- RIFE 로드 ------------------------- #
try:
    from model.RIFE_HDv2 import Model; model = Model(); model.load_model(args.modelDir, -1)
except Exception:
    try:
        from train_log.RIFE_HDv3 import Model; model = Model(); model.load_model(args.modelDir, -1)
    except Exception:
        from model.RIFE_HD import Model; model = Model(); model.load_model(args.modelDir, -1)
if not hasattr(model, 'version'): model.version = 0
model.eval(); model.device()
print(f"✔ RIFE version {model.version} loaded")

# ------------------------- 유틸 ------------------------- #
ALLOW = {'.png', '.jpg', '.jpeg', '.exr'}
def natsort_key(s):  # 자연어식 정렬
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def load_tensor(path):
    if path.endswith('.exr'):
        img  = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        t    = torch.tensor(img.transpose(2, 0, 1)).to(device).unsqueeze(0)
        is_e = True
    else:
        img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        t    = torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.
        t    = t.unsqueeze(0)
        is_e = False
    h, w  = img.shape[:2]
    ph, pw = ((h - 1)//64+1)*64, ((w - 1)//64+1)*64
    return F.pad(t, (0, pw-w, 0, ph-h)), (h, w), is_e

def save_tensor(tensor, shape, idx, is_exr):
    os.makedirs(args.dst, exist_ok=True)
    h, w = shape
    arr  = tensor[0][:, :h, :w].cpu().numpy().transpose(1, 2, 0)
    name = f"{idx:06d}.{args.ext}"
    if is_exr:
        cv2.imwrite(os.path.join(args.dst, name), arr,
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        cv2.imwrite(os.path.join(args.dst, name), (arr*255).astype('uint8'))

# ------------------------- 입력 목록 ------------------------- #
files = sorted([f for f in os.listdir(args.src)
                if os.path.splitext(f)[1].lower() in ALLOW], key=natsort_key)
if len(files) < 2:
    raise RuntimeError("2장 이상의 이미지가 필요합니다.")

# ------------------------- DFS 보간 ------------------------- #
def dfs(a, b, depth, shape, is_exr, counter):
    """a->b 사이 depth 단계 DFS, counter 는 [list] 로 전달해 전역 갱신"""
    if depth == 0: return
    with amp_ctx():
        mid = model.inference(a, b)
    dfs(a, mid, depth-1, shape, is_exr, counter)   # 왼쪽
    save_tensor(mid, shape, counter[0], is_exr)     # 중간 저장
    counter[0] += 1
    del a; dfs(mid, b, depth-1, shape, is_exr, counter)   # 오른쪽
    del mid; torch.cuda.empty_cache(); gc.collect()

global_idx = [0]   # 리스트를 통해 mutable 참조
for i in range(len(files)-1):
    path0, path1 = os.path.join(args.src, files[i]), os.path.join(args.src, files[i+1])
    t0, shape, is_e0 = load_tensor(path0)
    t1, _,     is_e1 = load_tensor(path1)
    is_exr = is_e0 or is_e1
    if i == 0:                       # 첫 원본 프레임 저장
        save_tensor(t0, shape, global_idx[0], is_exr); global_idx[0] += 1
    if args.ratio:                   # 단일 비율 보간
        with amp_ctx():
            mid = model.inference(t0, t1, args.ratio)
        save_tensor(mid, shape, global_idx[0], is_exr); global_idx[0] += 1
        del mid
    else:                            # 2^exp 방식
        dfs(t0, t1, args.exp, shape, is_exr, global_idx)
    save_tensor(t1, shape, global_idx[0], is_exr); global_idx[0] += 1
    # 메모리 해제
    del t0, t1; torch.cuda.empty_cache(); gc.collect()

print(f"✅  {global_idx[0]} frames saved in '{args.dst}' (OOM-safe)")
