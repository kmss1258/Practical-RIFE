#!/usr/bin/env python3
# folder2webp.py

#python folder2webp.py \
#  --src temp_result \
#  --out result--long.webp \
#  --fps 15 \
#  --size 300 \
#  --radius 110


import os, re, cv2, numpy as np, imageio.v3 as iio, argparse

def images2webp(images, wfp, **kwargs):          # _MS_250311
    if not wfp.lower().endswith('.webp'):
        raise ValueError(f"output_path must be a .webp file (got '{wfp}')")

    resize_square               = kwargs.get('resize_square',               300)
    circle_transparent_radius   = kwargs.get('circle_transparent_crop_radius', 110)
    fps                         = kwargs.get('fps',                           30)
    duration                    = int(1000 / fps)             # ms per frame

    # 원형 알파 마스크
    mask = np.zeros((resize_square, resize_square), np.uint8)
    cv2.circle(mask,
               (resize_square // 2, resize_square // 2),
               circle_transparent_radius, 255, -1)

    frames = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)                 # BGRA
        img = cv2.resize(img, (circle_transparent_radius * 2,
                               circle_transparent_radius * 2))
        canvas = np.zeros((resize_square, resize_square, 4), np.uint8)

        pad = (resize_square - circle_transparent_radius * 2) // 2
        canvas[pad : resize_square - pad, pad : resize_square - pad, :3] = img[..., :3]
        canvas[:, :, 3] = mask
        frames.append(canvas)

    # “long/b-link” 여부에 따라 loop 값 결정
    stem = os.path.basename(wfp).split("--")[1] if "--" in wfp else ""
    loop = 1 if ("long" in stem or "blink" in stem) else 0
    iio.imwrite(wfp, frames, format="webp", duration=duration, loop=loop)
    print(f"✅  {len(frames)} frames → '{wfp}'")


# ------------------------------------------------------------
# 2) 폴더에서 이미지 읽어와 images2webp 호출
# ------------------------------------------------------------
ALLOW = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
def natural_key(name: str):
    """사람이 보기 좋은 자연어식 정렬 키"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def folder2webp(src_dir: str, out_path: str, **kwargs):
    files = [f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in ALLOW]
    if not files:
        raise RuntimeError("No image files found in the source directory.")
    files.sort(key=natural_key)

    images = [cv2.imread(os.path.join(src_dir, f), cv2.IMREAD_UNCHANGED) for f in files]
    images2webp(images, out_path, **kwargs)


# ------------------------------------------------------------
# 3) CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert image sequence to animated WebP")
    p.add_argument('--src',  required=True, help='보간 이미지 폴더')
    p.add_argument('--out',  required=True, help='출력 WebP 파일 (*.webp)')
    p.add_argument('--fps',  type=int, default=30,  help='프레임 속도')
    p.add_argument('--size', type=int, default=300, help='정사각형 출력 크기(px)')
    p.add_argument('--radius', type=int, default=110, help='원형 알파 반경(px)')
    args = p.parse_args()

    folder2webp(
        args.src, args.out,
        fps=args.fps,
        resize_square=args.size,
        circle_transparent_crop_radius=args.radius
    )
