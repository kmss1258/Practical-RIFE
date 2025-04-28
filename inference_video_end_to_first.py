import os
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from train_log.RIFE_HDv3 import Model

def pad_image(img, scale, fp16):
    _, _, h, w = img.shape
    tmp = max(128, int(128/scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)  # left, right, top, bottom
    if fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

def make_inference(model, I0, I1, n, scale):
    # model.version >=3.9 기준
    res = []
    for i in range(n):
        res.append(model.inference(I0, I1, (i+1)/(n+1), scale))
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',    type=str, required=True, help='입력 비디오 경로')
    parser.add_argument('--multi',    type=int, default=2, help='추가할 보간 프레임 수')
    parser.add_argument('--modelDir', type=str, default='train_log', help='학습된 모델 디렉토리')
    parser.add_argument('--fp16',     action='store_true', help='FP16 모드')
    parser.add_argument('--scale',    type=float, default=1.0, help='스케일 (일반 1.0)')
    parser.add_argument('--output',   type=str, default=None, help='출력 비디오 파일명 (mp4)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available() and args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # 모델 로드
    model = Model()
    model.load_model(args.modelDir, -1)
    model.eval()
    model.device()

    # 비디오 프레임 읽기
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        # BGR -> RGB
        frames.append(frame[:, :, ::-1].copy())
    cap.release()

    first = frames[0]
    last  = frames[-1]

    # 텐서 변환 및 패딩
    to_tensor = lambda img: torch.from_numpy(img.transpose(2,0,1))[None].float()/255.
    I_last  = pad_image(to_tensor(last).to(device), args.scale, args.fp16)
    I_first = pad_image(to_tensor(first).to(device), args.scale, args.fp16)

    # 마지막→첫 보간 프레임 생성
    mids = make_inference(model, I_last, I_first, args.multi, args.scale)

    # 출력 비디오 준비
    basename = os.path.splitext(os.path.basename(args.video))[0]
    out_name = args.output or f"{basename}_loop_{args.multi}x.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_name, fourcc, fps, (w, h))

    # 원본 프레임 그대로 쓰기
    for fr in frames:
        writer.write(fr[:, :, ::-1])  # RGB→BGR

    # 보간 프레임 추가
    for mid in mids:
        img = (mid[0] * 255).byte().cpu().numpy().transpose(1,2,0)
        # 패딩 제거
        img = img[:h, :w]
        writer.write(img[:, :, ::-1])

    writer.release()
    print(f"완료: {out_name}")
