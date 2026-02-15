"""
顔モーフィング動画生成ツール

使い方:
  python main.py --image1 image.jpg --image2 image.jpg

オプション:
  --image1    (必須) 入力画像A（変形元）
  --image2    (必須) 入力画像B（変形先）
  --output    出力動画パス          (default: morphing_output.mp4)
  --model     FacemarkLBF モデル    (default: lbfmodel.yaml)
  --frames    総フレーム数          (default: 120)
  --fps       フレームレート        (default: 30)
  --debug_dir デバッグ画像出力先    (default: debug_morphing)
  --preview   プレビューウィンドウ表示
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めませんでした: {path}")
    return img


def _resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    if src_w == w and src_h == h:
        return img

    src_ratio = float(src_w) / float(src_h)
    dst_ratio = float(w) / float(h)

    if src_ratio > dst_ratio:
        crop_w = int(round(dst_ratio * float(src_h)))
        crop_w = max(1, min(crop_w, src_w))
        x0 = (src_w - crop_w) // 2
        cropped = img[:, x0 : x0 + crop_w]
    else:
        crop_h = int(round(float(src_w) / dst_ratio))
        crop_h = max(1, min(crop_h, src_h))
        y0 = (src_h - crop_h) // 2
        cropped = img[y0 : y0 + crop_h, :]

    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)


def _detect_face_bbox(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    h_img, w_img = gray.shape[:2]

    cascade_names = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_profileface.xml",
    ]
    best: tuple[int, int, int, int] | None = None

    for name in cascade_names:
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
        if cascade.empty():
            continue

        for g in (gray_eq, gray):
            faces = cascade.detectMultiScale(
                g,
                scaleFactor=1.05,
                minNeighbors=3,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(30, 30),
            )
            if faces is not None and len(faces) > 0:
                cand = max((tuple(int(v) for v in f) for f in faces), key=lambda r: r[2] * r[3])
                if best is None or cand[2] * cand[3] > best[2] * best[3]:
                    best = cand

            g_flip = cv2.flip(g, 1)
            faces_flip = cascade.detectMultiScale(
                g_flip,
                scaleFactor=1.05,
                minNeighbors=3,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(30, 30),
            )
            if faces_flip is not None and len(faces_flip) > 0:
                x_f, y_f, w_f, h_f = max((tuple(int(v) for v in f) for f in faces_flip), key=lambda r: r[2] * r[3])
                x = w_img - (x_f + w_f)
                cand = (int(x), int(y_f), int(w_f), int(h_f))
                if best is None or cand[2] * cand[3] > best[2] * best[3]:
                    best = cand

    if best is None:
        raise RuntimeError("顔が検出できませんでした。")

    x, y, w, h = best
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return (x, y, w, h)


def _extract_landmarks68(img_bgr: np.ndarray, face_bbox: tuple[int, int, int, int], model_path: Path) -> np.ndarray:
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(model_path))

    rect = np.array([face_bbox], dtype=np.int32)
    ok, landmarks = facemark.fit(img_bgr, rect)
    if not ok or landmarks is None or len(landmarks) == 0:
        raise RuntimeError("Facemark(68点)の抽出に失敗しました。")
    pts = landmarks[0].reshape(-1, 2).astype(np.float32)
    if pts.shape[0] != 68:
        raise RuntimeError("68点ランドマークが取得できませんでした。")
    return pts


def _add_boundary_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    b = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
            [w // 2, 0],
            [w - 1, h // 2],
            [w // 2, h - 1],
            [0, h // 2],
        ],
        dtype=np.float32,
    )
    return np.vstack([points.astype(np.float32), b])


def _delaunay_triangles(points: np.ndarray, w: int, h: int) -> np.ndarray:
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for x, y in points.astype(np.float32):
        subdiv.insert((float(x), float(y)))

    tris = subdiv.getTriangleList()
    if tris is None or len(tris) == 0:
        raise RuntimeError("三角形分割に失敗しました。")

    pts_int = np.round(points).astype(np.int32)
    key_to_idx: dict[tuple[int, int], int] = {(int(x), int(y)): i for i, (x, y) in enumerate(pts_int)}

    out: list[tuple[int, int, int]] = []
    for t in tris:
        x1, y1, x2, y2, x3, y3 = (float(v) for v in t)
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        p3 = (int(round(x3)), int(round(y3)))
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h and 0 <= p3[0] < w and 0 <= p3[1] < h):
            continue
        if p1 not in key_to_idx or p2 not in key_to_idx or p3 not in key_to_idx:
            continue
        i1, i2, i3 = key_to_idx[p1], key_to_idx[p2], key_to_idx[p3]
        if i1 != i2 and i2 != i3 and i1 != i3:
            out.append((i1, i2, i3))
    if not out:
        raise RuntimeError("三角形分割のインデックス化に失敗しました。")
    return np.array(out, dtype=np.int32)


def _warp_triangle(src: np.ndarray, dst: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> None:
    r_src = cv2.boundingRect(t_src)
    r_dst = cv2.boundingRect(t_dst)

    t_src_rect = np.array([(t_src[i, 0] - r_src[0], t_src[i, 1] - r_src[1]) for i in range(3)], dtype=np.float32)
    t_dst_rect = np.array([(t_dst[i, 0] - r_dst[0], t_dst[i, 1] - r_dst[1]) for i in range(3)], dtype=np.float32)

    h_dst, w_dst = dst.shape[:2]
    x_dst, y_dst, w_rect, h_rect = r_dst
    if x_dst < 0 or y_dst < 0 or w_rect <= 0 or h_rect <= 0:
        return
    w_rect = min(w_rect, w_dst - x_dst)
    h_rect = min(h_rect, h_dst - y_dst)
    if w_rect <= 0 or h_rect <= 0:
        return

    mask = np.zeros((h_rect, w_rect, 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), 16, 0)

    h_src, w_src = src.shape[:2]
    x_src, y_src, w_src_rect, h_src_rect = r_src
    if x_src < 0 or y_src < 0 or w_src_rect <= 0 or h_src_rect <= 0:
        return
    w_src_rect = min(w_src_rect, w_src - x_src)
    h_src_rect = min(h_src_rect, h_src - y_src)
    if w_src_rect <= 0 or h_src_rect <= 0:
        return

    src_rect = src[y_src : y_src + h_src_rect, x_src : x_src + w_src_rect]
    if src_rect.size == 0:
        return

    m = cv2.getAffineTransform(t_src_rect, t_dst_rect)
    warped = cv2.warpAffine(
        src_rect,
        m,
        (w_rect, h_rect),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    ).astype(np.float32)

    dst_slice = dst[y_dst : y_dst + h_rect, x_dst : x_dst + w_rect]
    dst_slice *= (1.0 - mask)
    dst_slice += warped * mask
    dst[y_dst : y_dst + h_rect, x_dst : x_dst + w_rect] = dst_slice


def _morph_frame(img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, tri: np.ndarray, alpha: float) -> np.ndarray:
    h, w = img1.shape[:2]
    a = np.float32(alpha)
    pts_mid = (1.0 - a) * pts1 + a * pts2

    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    w1 = np.zeros((h, w, 3), dtype=np.float32)
    w2 = np.zeros((h, w, 3), dtype=np.float32)

    for i1, i2, i3 in tri:
        t1 = np.array([pts1[i1], pts1[i2], pts1[i3]], dtype=np.float32)
        t2 = np.array([pts2[i1], pts2[i2], pts2[i3]], dtype=np.float32)
        tm = np.array([pts_mid[i1], pts_mid[i2], pts_mid[i3]], dtype=np.float32)
        _warp_triangle(img1_f, w1, t1, tm)
        _warp_triangle(img2_f, w2, t2, tm)

    out = (1.0 - a) * w1 + a * w2
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_points(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    out = img.copy()
    for x, y in pts.astype(np.float32):
        cv2.circle(out, (int(round(float(x))), int(round(float(y)))), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return out


def _draw_matches(img1: np.ndarray, img2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1 : w1 + w2] = img2
    n = min(p1.shape[0], p2.shape[0])
    for i in range(n):
        pt1 = (int(round(float(p1[i, 0]))), int(round(float(p1[i, 1]))))
        pt2 = (int(round(float(p2[i, 0]) + float(w1))), int(round(float(p2[i, 1]))))
        cv2.circle(canvas, pt1, 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt2, 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, pt1, pt2, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    return canvas


def _draw_triangulation(img: np.ndarray, pts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    out = img.copy()
    pts_i = np.round(pts).astype(np.int32)
    for i1, i2, i3 in tri:
        p1 = tuple(int(v) for v in pts_i[i1])
        p2 = tuple(int(v) for v in pts_i[i2])
        p3 = tuple(int(v) for v in pts_i[i3])
        cv2.line(out, p1, p2, (255, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(out, p2, p3, (255, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.line(out, p3, p1, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    return out


def _save_debug(debug_dir: Path, img1: np.ndarray, img2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, tri: np.ndarray) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / "points_image1.png"), _draw_points(img1, pts1))
    cv2.imwrite(str(debug_dir / "points_image2.png"), _draw_points(img2, pts2))
    cv2.imwrite(str(debug_dir / "matches.png"), _draw_matches(img1, img2, pts1, pts2))
    mid = np.clip(0.5 * img1.astype(np.float32) + 0.5 * img2.astype(np.float32), 0, 255).astype(np.uint8)
    cv2.imwrite(str(debug_dir / "triangulation.png"), _draw_triangulation(mid, 0.5 * (pts1 + pts2), tri))
    np.savez_compressed(str(debug_dir / "points_and_tri.npz"), pts1=pts1, pts2=pts2, tri=tri)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image1", type=Path, required=True)
    ap.add_argument("--image2", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("morphing_output.mp4"))
    ap.add_argument("--model", type=Path, default=Path("lbfmodel.yaml"))
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--debug_dir", type=Path, default=Path("debug_morphing"))
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Facemarkモデルが見つかりません: {args.model}")

    img1 = _load_bgr(args.image1)
    img2 = _load_bgr(args.image2)
    h, w = img1.shape[:2]
    img2 = _resize_to(img2, w, h)

    face1 = _detect_face_bbox(img1)
    img2_aligned = img2
    face2 = _detect_face_bbox(img2_aligned)

    pts1 = _extract_landmarks68(img1, face1, args.model)
    pts2 = _extract_landmarks68(img2_aligned, face2, args.model)
    pts1 = _add_boundary_points(pts1, w, h)
    pts2 = _add_boundary_points(pts2, w, h)

    tri = _delaunay_triangles(0.5 * (pts1 + pts2), w, h)
    _save_debug(args.debug_dir, img1, img2_aligned, pts1, pts2, tri)
    cv2.imwrite(str(args.debug_dir / "image2_aligned.png"), img2_aligned)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, float(args.fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter を開けませんでした。")

    if args.preview:
        cv2.namedWindow("morph_preview", cv2.WINDOW_NORMAL)

    try:
        total = max(2, int(args.frames))
        for i in range(total):
            alpha = float(i) / float(total - 1)
            frame = _morph_frame(img1, img2_aligned, pts1, pts2, tri, alpha)
            writer.write(frame)
            if args.preview:
                cv2.imshow("morph_preview", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        writer.release()
        if args.preview:
            cv2.destroyWindow("morph_preview")


if __name__ == "__main__":
    main()
