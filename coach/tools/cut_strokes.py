#!/usr/bin/env python3
"""Нарезка теннисных видео: оставляет только удары игрока у камеры и выбрасывает остальное.

Как это работает:
1. По звуку находим кандидатов: удар ракеткой по мячу — короткий громкий щелчок.
   Это быстро даже для часовых записей.
2. Отбираем удары игрока у камеры. Щелчки дают и удары партнёра, и отскоки мяча.
   - ai (по умолчанию, если задан ANTHROPIC_API_KEY): на каждый щелчок Claude получает
     полоску из трёх кадров (до, во время, после) и отвечает, бьёт ли по мячу ближний игрок.
     Это точнее всего; стоит примерно 1–2 доллара за час исходного видео.
   - pose: проверка позой MediaPipe (скорость кисти в момент щелчка). Бесплатно, но надёжна,
     только если игрок крупный в кадре (снят близко и сбоку).
   - none: оставить все щелчки.
3. Вокруг каждого удара вырезаем фрагмент одинаковой длины (по умолчанию 2.2 с, удар на
   отметке 1.3 с) и склеиваем фрагменты в ролики по 30 ударов в 720p. Длина фрагмента и момент
   удара записаны в имени файла (..._s2.2_h1.3.mp4), поэтому приложение знает, где удары.
   С --rallies дополнительно сохраняется вся игра без пауз между розыгрышами.

Примеры:
    python3 cut_strokes.py ~/Videos/tennis -o ~/Videos/tennis_strokes
    python3 cut_strokes.py "https://youtu.be/XXXX" -o strokes
    python3 cut_strokes.py match.mov --filter pose --rallies

Нужно: Python 3.9+, ffmpeg, numpy. Для проверки через Claude — anthropic и ключ API
в ANTHROPIC_API_KEY; для проверки позой — mediapipe; для YouTube — yt-dlp:
    pip install numpy anthropic mediapipe yt-dlp
"""

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".3gp"}
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task")
CLAUDE_MODEL = "claude-opus-5-5"
SR = 16000          # частота звука для поиска щелчков
POSE_FPS = 15       # кадров в секунду для проверки позой
POSE_WIDTH = 640    # кадр для модели уменьшаем: так в разы быстрее
AI_BATCH = 12       # кандидатов в одном запросе к Claude
AI_TOKENS_PER_CANDIDATE = 600  # полоска из трёх кадров высотой 256 px с подписью, для оценки цены
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 15, 16, 23, 24


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def run(cmd):
    return subprocess.run(cmd, check=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    kinds = {s.get("codec_type") for s in info.get("streams", [])}
    return float(info["format"]["duration"]), "audio" in kinds, "video" in kinds


# ---------- 1. Кандидаты по звуку ----------

def audio_hits(path, threshold):
    """Времена резких щелчков и их сила (во сколько раз громче фона вокруг)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    if len(x) < SR:
        return []
    # Щелчок широкополосный, а гул ветра и голоса — низкие: разность соседних отсчётов
    # подчёркивает высокие частоты. Энергию меряем окнами по 5 мс.
    hp = np.diff(x, prepend=x[0])
    win = SR // 200
    n = len(hp) // win
    e = np.sqrt((hp[:n * win] ** 2).reshape(n, win).mean(axis=1))
    fps = SR / win
    # Фон: медианы по 0.25 с, затем медиана из 9 соседних блоков (±1 с).
    blk = int(fps / 4)
    nb = n // blk
    bm = np.median(e[:nb * blk].reshape(nb, blk), axis=1)
    bg_blocks = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(bm, 4, mode="edge"), 9), axis=1)
    bg = np.repeat(bg_blocks, blk)
    bg = np.concatenate([bg, np.full(n - len(bg), bg[-1] if len(bg) else 1e-6)]) + 1e-5
    r = e / bg
    hits = []
    for i in np.where((r[1:-1] > threshold) & (r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:]))[0] + 1:
        t = i / fps
        if hits and t - hits[-1][0] < 0.5:
            if r[i] > hits[-1][1]:
                hits[-1] = (t, float(r[i]))
            continue
        hits.append((t, float(r[i])))
    return hits


# ---------- 2а. Проверка через Claude ----------

AI_SYSTEM = """Ты помогаешь нарезать любительское теннисное видео. Камера стоит на стороне одного игрока;
это «ближний игрок» — тот, кто ближе всего к камере (обычно виден со спины или сбоку, крупнее остальных).
Тебе присылают кандидатов. Каждый кандидат — полоска из трёх кадров слева направо: за 0.2 с до звука удара,
в момент звука и через 0.2 с после. Для каждого кандидата реши, бьёт ли ближний игрок по мячу ракеткой в этот момент
(замах, контакт или завершение удара). Удары игрока на другой стороне сетки, отскоки мяча, ходьба, сбор мячей
и подбрасывание мяча рукой без удара — это не удар ближнего игрока. Если ближнего игрока не видно, это тоже не удар."""

AI_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "near_player_hits": {"type": "boolean"},
            "stroke": {"type": "string", "enum": ["forehand", "backhand", "serve", "volley", "smash", "other", "none"]},
        },
        "required": ["index", "near_player_hits", "stroke"],
        "additionalProperties": False,
    }}},
    "required": ["candidates"],
    "additionalProperties": False,
}


class AiChecker:
    def __init__(self):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()

    @staticmethod
    def strip(path, t):
        """Три кадра (t−0.2, t, t+0.2) в одну картинку высотой 256 px."""
        jpg = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t - 0.2):.3f}", "-i", str(path),
                              "-vf", "fps=5,scale=-2:256,tile=3x1", "-frames:v", "1", "-q:v", "5",
                              "-f", "image2pipe", "-vcodec", "mjpeg", "-"], capture_output=True, check=True).stdout
        return base64.standard_b64encode(jpg).decode()

    def ask(self, batch):
        content = [{"type": "text", "text": f"Кандидатов: {len(batch)}."}]
        for k, img in batch:
            content.append({"type": "text", "text": f"Кандидат {k}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}})
        response = self.client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": AI_SCHEMA}},
            system=AI_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            log(f"  Claude не ответил на пачку кандидатов ({response.stop_reason}), пропускаю её")
            return {}
        text = next(b.text for b in response.content if b.type == "text")
        return {c["index"]: c for c in json.loads(text)["candidates"]}

    def ask_safe(self, batch):
        a = self.anthropic
        try:
            return self.ask(batch)
        except a.AuthenticationError:
            log("  Ключ ANTHROPIC_API_KEY не подходит.")
            os._exit(1)
        except a.RateLimitError:
            log("  Слишком много запросов к Claude, пачка пропущена. Запустите позже или уменьшите объём.")
        except a.BadRequestError as e:
            log(f"  Claude отклонил запрос: {e.message}")
        except a.APIStatusError as e:
            log(f"  Ошибка Claude ({e.status_code}), пачка пропущена")
        except a.APIConnectionError:
            log("  Нет связи с Claude, пачка пропущена")
        return {}

    def check(self, path, times):
        """{номер кандидата: ответ Claude} для всех щелчков видео."""
        imgs = [(k, self.strip(path, t)) for k, t in enumerate(times)]
        batches = [imgs[i:i + AI_BATCH] for i in range(0, len(imgs), AI_BATCH)]
        out = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for res in pool.map(self.ask_safe, batches):
                out.update(res)
        return out


# ---------- 2б. Проверка позой ----------

def model_path():
    p = Path.home() / ".cache" / "tennis-coach" / "pose_landmarker_full.task"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        log("Скачиваю модель позы (≈10 МБ)…")
        urllib.request.urlretrieve(MODEL_URL, p)
    return p


class PoseChecker:
    """Распознаёт позу только в окнах вокруг кандидатов и меряет скорость кистей."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        self.vision = vision
        self.options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path()), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=2)
        # Пробный запуск: если модель не стартует (нет нужных системных библиотек), узнаём сразу.
        vision.PoseLandmarker.create_from_options(self.options).close()

    def track(self, path, windows):
        """Позы игрока у камеры в кадрах, попавших в окна: {номер кадра: 33 точки (x, y, vis)}."""
        dur, _, _ = probe(path)
        st = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height:stream_side_data=rotation", "-of", "json", str(path)],
                                       capture_output=True, text=True, check=True).stdout)["streams"][0]
        w, h = st["width"], st["height"]
        rot = 0
        for sd in st.get("side_data_list", []) or []:
            rot = int(sd.get("rotation", 0) or 0)
        if abs(rot) in (90, 270):
            w, h = h, w
        H = int(round(h * POSE_WIDTH / w / 2) * 2)
        want = np.zeros(int(dur * POSE_FPS) + 2, dtype=bool)
        for a, b in windows:
            want[max(0, int(a * POSE_FPS)):int(b * POSE_FPS) + 1] = True
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
                                 f"fps={POSE_FPS},scale={POSE_WIDTH}:{H}", "-f", "rawvideo",
                                 "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        frame_bytes = POSE_WIDTH * H * 3
        poses, prev, i, ts = {}, None, 0, 0
        with self.vision.PoseLandmarker.create_from_options(self.options) as lm:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                if i < len(want) and want[i]:
                    img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                        data=np.frombuffer(buf, np.uint8).reshape(H, POSE_WIDTH, 3))
                    ts += 1000 // POSE_FPS
                    res = lm.detect_for_video(img, ts)
                    # Следим за тем же человеком, что в прошлом кадре; в начале окна — за самым крупным.
                    if i == 0 or not want[i - 1]:
                        prev = None
                    best, score = None, math.inf
                    for p in res.pose_landmarks:
                        xs = [q.x for q in p]
                        ys = [q.y for q in p]
                        cx, cy = (p[L_HIP].x + p[R_HIP].x) / 2, (p[L_HIP].y + p[R_HIP].y) / 2
                        s = math.hypot(cx - prev[0], cy - prev[1]) if prev else -(max(xs) - min(xs)) * (max(ys) - min(ys))
                        if s < score:
                            best, score = p, s
                    if best is not None:
                        prev = ((best[L_HIP].x + best[R_HIP].x) / 2, (best[L_HIP].y + best[R_HIP].y) / 2)
                        poses[i] = [(q.x * POSE_WIDTH, q.y * H, q.visibility or 0) for q in best]
                i += 1
        proc.wait()
        return poses

    @staticmethod
    def swing(poses, t):
        """Наибольшая скорость кистей относительно плеч (корпусов в секунду) в пределах ±0.35 с."""
        def rel(p, j):
            return p[j][0] - (p[L_SH][0] + p[R_SH][0]) / 2, p[j][1] - (p[L_SH][1] + p[R_SH][1]) / 2

        best = 0.0
        for i in range(int((t - 0.35) * POSE_FPS), int((t + 0.35) * POSE_FPS) + 1):
            p0, p1 = poses.get(i - 1), poses.get(i)
            if not p0 or not p1:
                continue
            torso = abs((p1[L_HIP][1] + p1[R_HIP][1]) / 2 - (p1[L_SH][1] + p1[R_SH][1]) / 2)
            if torso < 8:
                continue
            for j in (L_WR, R_WR):
                (x0, y0), (x1, y1) = rel(p0, j), rel(p1, j)
                v = math.hypot(x1 - x0, y1 - y0) * POSE_FPS / torso
                if v < 40:  # быстрее рука не двигается — это сбой модели
                    best = max(best, v)
        return best


# ---------- 3. Нарезка ----------

def encode(path, start, dur, out, args, has_audio):
    a_opts = (["-af", f"aresample=async=1,apad,atrim=0:{dur:.3f}", "-c:a", "aac", "-b:a", "64k", "-ar", "44100", "-ac", "1"]
              if has_audio else ["-an"])
    run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{dur:.3f}",
         "-vf", f"scale=-2:'min({args.height},ih)',fps=30", "-frames:v", str(int(round(dur * 30))),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
         *a_opts, str(out)])


def concat(parts, out, tmp):
    lst = tmp / (out.stem + ".txt")
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
         "-movflags", "+faststart", str(out)])


def cut_strokes(path, hits, out_dir, stem, args, has_audio):
    """Фрагменты одинаковой длины вокруг ударов, склеенные в ролики по --per-file ударов."""
    seg = args.pre + args.post
    tmp = Path(tempfile.mkdtemp(prefix="strokes_"))
    names = []
    try:
        parts = []
        for k, t in enumerate(hits):
            f = tmp / f"{k:05d}.mp4"
            encode(path, t - args.pre, seg, f, args, has_audio)
            parts.append(f)
        groups = [parts[i:i + args.per_file] for i in range(0, len(parts), args.per_file)]
        for g, group in enumerate(groups, 1):
            name = f"{stem}{f'_part{g}' if len(groups) > 1 else ''}_strokes_s{seg:.1f}_h{args.pre:.1f}.mp4"
            concat(group, out_dir / name, tmp)
            names.append(name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return names


def cut_rallies(path, clicks, dur, out_dir, stem, args, has_audio):
    """Вся игра без пауз: розыгрыш — щелчки с промежутками меньше --rally-gap секунд."""
    spans = []
    for t in clicks:
        a, b = max(0, t - 2.0), min(dur, t + 1.5)
        if spans and a - spans[-1][1] < args.rally_gap:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    tmp = Path(tempfile.mkdtemp(prefix="rallies_"))
    try:
        parts = []
        for k, (a, b) in enumerate(spans):
            f = tmp / f"{k:05d}.mp4"
            encode(path, a, b - a, f, args, has_audio)
            parts.append(f)
        name = f"{stem}_rallies.mp4"
        concat(parts, out_dir / name, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return name, sum(b - a for a, b in spans)


# ---------- Входные файлы ----------

def download(url, dest):
    """Скачивает видео с YouTube через yt-dlp (не выше 1080p) и возвращает пути к файлам."""
    if not shutil.which("yt-dlp"):
        sys.exit("Для ссылок на YouTube нужен yt-dlp: pip install yt-dlp")
    dest.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(["yt-dlp", "-f", "bv*[height<=1080]+ba/b[height<=1080]", "--merge-output-format", "mp4",
                          "-o", str(dest / "%(title).80s [%(id)s].%(ext)s"), "--print", "after_move:filepath",
                          "--no-simulate", url], capture_output=True, text=True)
    if out.returncode:
        log(out.stderr.strip())
        return []
    return [Path(line) for line in out.stdout.strip().splitlines() if line]


def inputs(items, out_dir):
    for item in items:
        if item.startswith(("http://", "https://")):
            log(f"Скачиваю {item}")
            for p in download(item, out_dir / "_downloads"):  # ссылка на плейлист даёт несколько файлов
                yield p, item
            continue
        p = Path(item).expanduser()
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in VIDEO_EXT and "_strokes_" not in f.name and not f.name.endswith("_rallies.mp4"):
                    yield f, str(f)
        elif p.exists():
            yield p, str(p)
        else:
            log(f"Не найдено: {item}")


def process(path, source, out_dir, args, checker):
    dur, has_audio, has_video = probe(path)
    if not has_video:
        log("  нет видеодорожки, пропускаю")
        return None
    if not has_audio:
        log("  в видео нет звука: удары по щелчку не найти, пропускаю")
        return None
    clicks = audio_hits(path, args.threshold)
    cands = [(t, s) for t, s in clicks if t >= args.pre and t + args.post <= dur]
    log(f"  {dur / 60:.1f} мин, щелчков: {len(cands)}")
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(path).stem)[:60]
    manifest = {"source": source, "source_minutes": round(dur / 60, 1), "filter": args.filter,
                "clip_seconds": args.pre + args.post, "hit_offset": args.pre, "files": [], "hits": []}
    if args.rallies and clicks:
        name, secs = cut_rallies(path, [t for t, _ in clicks], dur, out_dir, stem, args, has_audio)
        manifest["rallies"] = {"file": name, "minutes": round(secs / 60, 1)}
        log(f"  без пауз: {secs / 60:.1f} мин → {name}")

    hits = []
    if args.filter == "ai" and cands:
        log(f"  спрашиваю Claude про {len(cands)} щелчков (≈${len(cands) * AI_TOKENS_PER_CANDIDATE * 4 / 1e6:.2f})…")
        answers = checker.check(path, [t for t, _ in cands])
        for k, (t, s) in enumerate(cands):
            a = answers.get(k)
            if a and a["near_player_hits"]:
                hits.append({"source_time": round(t, 2), "click": round(s, 1), "stroke": a["stroke"]})
    elif args.filter == "pose" and cands:
        poses = checker.track(path, [(t - 0.5, t + 0.5) for t, _ in cands])
        for t, s in cands:
            v = checker.swing(poses, t)
            if v >= args.min_swing:
                hits.append({"source_time": round(t, 2), "click": round(s, 1), "swing": round(v, 1)})
    else:
        hits = [{"source_time": round(t, 2), "click": round(s, 1)} for t, s in cands]
    if args.filter != "none":
        log(f"  ударов игрока у камеры: {len(hits)}")

    if hits:
        files = cut_strokes(path, [h["source_time"] for h in hits], out_dir, stem, args, has_audio)
        seg = args.pre + args.post
        for k, h in enumerate(hits):
            h["file"] = files[k // args.per_file]
            h["time_in_file"] = round((k % args.per_file) * seg + args.pre, 2)
        manifest["files"], manifest["hits"] = files, hits
    if not hits and "rallies" not in manifest:
        return None
    (out_dir / f"{stem}_strokes.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Оставляет в теннисных видео только удары игрока у камеры.")
    ap.add_argument("inputs", nargs="+", help="видеофайлы, папки или ссылки на YouTube")
    ap.add_argument("-o", "--out", default="strokes", help="папка для результата (по умолчанию ./strokes)")
    ap.add_argument("--filter", choices=["ai", "pose", "none"],
                    help="как отбирать удары игрока у камеры: ai (Claude), pose (MediaPipe) или none (все щелчки); "
                         "по умолчанию ai, если задан ANTHROPIC_API_KEY, иначе pose")
    ap.add_argument("--rallies", action="store_true", help="ещё сохранить всю игру без пауз между розыгрышами")
    ap.add_argument("--pre", type=float, default=1.3, help="секунд до удара (замах), по умолчанию 1.3")
    ap.add_argument("--post", type=float, default=0.9, help="секунд после удара (завершение), по умолчанию 0.9")
    ap.add_argument("--per-file", type=int, default=30, help="ударов в одном ролике, по умолчанию 30")
    ap.add_argument("--height", type=int, default=720, help="высота кадра на выходе, по умолчанию 720")
    ap.add_argument("--crf", type=int, default=26, help="качество x264: меньше — лучше и тяжелее, по умолчанию 26")
    ap.add_argument("--threshold", type=float, default=6.0, help="насколько щелчок громче фона, по умолчанию 6")
    ap.add_argument("--rally-gap", type=float, default=5.0, help="пауза, после которой начинается новый розыгрыш (с)")
    ap.add_argument("--min-swing", type=float, default=6.0, help="для --filter pose: минимальная скорость кисти (корпусов/с)")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"Не найден {tool}. Установите ffmpeg: https://ffmpeg.org/download.html")
    if args.filter is None:
        args.filter = "ai" if os.environ.get("ANTHROPIC_API_KEY") else "pose"
    checker = None
    if args.filter == "ai":
        try:
            checker = AiChecker()
        except ImportError:
            sys.exit("Для --filter ai нужен пакет anthropic: pip install anthropic")
    elif args.filter == "pose":
        try:
            checker = PoseChecker()
            log("Отбор по позе: работает, если игрок крупный в кадре. Точнее — с ANTHROPIC_API_KEY (--filter ai).")
        except ImportError:
            log("mediapipe не установлен: останутся все щелчки (pip install mediapipe)")
            args.filter = "none"
        except Exception as e:  # noqa: BLE001 — поломка модели не должна мешать нарезке по звуку
            log(f"Модель позы не запустилась ({e}): останутся все щелчки")
            args.filter = "none"
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    total_in = total_hits = 0.0
    for path, source in inputs(args.inputs, out_dir):
        log(f"\n{path.name}")
        try:
            m = process(path, source, out_dir, args, checker)
        except subprocess.CalledProcessError as e:
            log(f"  ошибка ffmpeg: {e}")
            continue
        if m:
            total_in += m["source_minutes"]
            total_hits += len(m["hits"])
            if m["files"]:
                log("  удары: " + ", ".join(m["files"]))
    seg = args.pre + args.post
    log(f"\nИтого: {total_in:.0f} мин видео → {int(total_hits)} ударов, {total_hits * seg / 60:.1f} мин. Папка: {out_dir}")


if __name__ == "__main__":
    main()
