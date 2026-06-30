import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIG_DIR = Path("figures")
WIDE = (1200, 620)
WHITE = (255, 255, 255)
INK = (28, 34, 45)
BLUE = (35, 94, 168)
TEAL = (28, 132, 121)
ORANGE = (219, 120, 49)
RED = (186, 55, 58)
GRAY = (230, 234, 240)
DARK_GRAY = (100, 111, 125)


def font(size, bold=False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save(img, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    img.save(path)
    print(path)


def rounded(draw, box, fill, outline=None, width=2, radius=18):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill=INK, width=4):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    angle = np.arctan2(y2 - y1, x2 - x1)
    size = 14
    pts = [
        (x2, y2),
        (x2 - size * np.cos(angle - 0.45), y2 - size * np.sin(angle - 0.45)),
        (x2 - size * np.cos(angle + 0.45), y2 - size * np.sin(angle + 0.45)),
    ]
    draw.polygon(pts, fill=fill)


def architecture():
    img = Image.new("RGB", WIDE, WHITE)
    d = ImageDraw.Draw(img)
    title = font(34, True)
    body = font(22)
    small = font(18)
    d.text((48, 32), "LeWM-SDN Predictive Latent Defense Pipeline", fill=INK, font=title)

    boxes = [
        ((55, 135, 255, 265), "SDN graph\nG_t=(V,E,X,F)", BLUE),
        ((335, 135, 555, 265), "Temporal GNN\nmessage passing", TEAL),
        ((635, 105, 855, 205), "z_prog\nphase compass", ORANGE),
        ((635, 235, 855, 335), "z_cont\nSIGReg", BLUE),
        ((930, 155, 1135, 285), "Predict z_{t+1}\n+ anomaly score", RED),
    ]
    for box, label, color in boxes:
        rounded(d, box, fill=(248, 250, 252), outline=color, width=4)
        lines = label.split("\n")
        y = box[1] + 28
        for line in lines:
            bbox = d.textbbox((0, 0), line, font=body)
            d.text(((box[0] + box[2] - bbox[2]) / 2, y), line, fill=INK, font=body)
            y += 30

    arrow(d, (255, 200), (335, 200), BLUE)
    arrow(d, (555, 190), (635, 155), INK)
    arrow(d, (555, 210), (635, 285), INK)
    arrow(d, (855, 155), (930, 205), INK)
    arrow(d, (855, 285), (930, 235), INK)

    rounded(d, (155, 410, 500, 525), "white", outline=DARK_GRAY, width=3)
    d.text((185, 430), "Mitigation action u_t", fill=INK, font=body)
    d.text((185, 464), "no-op/drop/rate-limit/reroute", fill=DARK_GRAY, font=small)
    arrow(d, (500, 468), (930, 265), DARK_GRAY, width=3)

    d.text((620, 430), "A_t = surprise + phase drift", fill=INK, font=body)
    d.text((620, 464), "DDoS is detected as a violation of normal latent dynamics.", fill=DARK_GRAY, font=small)
    save(img, "architecture_pipeline.png")


def dataset_timeline():
    data = np.load("data/processed/synthetic_sdn_ddos.npz")
    labels = data["label"]
    img = Image.new("RGB", (1200, 360), WHITE)
    d = ImageDraw.Draw(img)
    d.text((48, 30), "Synthetic SDN/DDoS Timeline", fill=INK, font=font(34, True))
    d.text((48, 78), "1200 timesteps: normal windows followed by periodic DDoS bursts", fill=DARK_GRAY, font=font(20))
    x0, y0, width, height = 70, 160, 1060, 70
    d.rectangle((x0, y0, x0 + width, y0 + height), fill=GRAY)
    for i, label in enumerate(labels):
        x = x0 + int(i * width / len(labels))
        x_next = x0 + int((i + 1) * width / len(labels))
        color = RED if label else TEAL
        d.rectangle((x, y0, max(x + 1, x_next), y0 + height), fill=color)
    d.rectangle((x0, y0, x0 + width, y0 + height), outline=INK, width=2)
    d.text((70, 250), "0", fill=INK, font=font(18))
    d.text((1080, 250), "1199", fill=INK, font=font(18))
    d.rectangle((70, 300, 100, 322), fill=TEAL)
    d.text((112, 296), "normal", fill=INK, font=font(20))
    d.rectangle((230, 300, 260, 322), fill=RED)
    d.text((272, 296), "DDoS", fill=INK, font=font(20))
    save(img, "dataset_timeline.png")


def baseline_bars():
    payload = json.loads(Path("outputs/eval/baseline_comparison.json").read_text())
    results = payload["results"]
    supervised_path = Path("outputs/eval/torch_supervised_baselines.json")
    if supervised_path.exists():
        supervised_payload = json.loads(supervised_path.read_text())
        results = results[:3] + supervised_payload["results"] + results[3:]
    torch_path = Path("outputs/eval/torch_sdn_jepa_eval.json")
    if torch_path.exists():
        torch_payload = json.loads(torch_path.read_text())
        results = results + [
            {
                "method": "pytorch_temporal_gnn_jepa",
                "metrics": torch_payload["metrics"],
            }
        ]
    short_names = {
        "one_class_feature_distance": "One-class\nfeature\ndistance",
        "supervised_logistic_regression": "Logistic\nregression",
        "supervised_numpy_mlp": "NumPy\nMLP",
        "supervised_gradient_boosted_stumps": "Boosted\nstumps",
        "supervised_torch_mlp": "Torch\nMLP",
        "supervised_torch_cnn": "Torch\nCNN",
        "supervised_torch_lstm": "Torch\nLSTM",
        "raw_feature_ridge_surprise": "Raw ridge\nsurprise",
        "lewm_sdn_latent_surprise_phase": "LeWM-SDN\nlatent +\nphase",
        "pytorch_temporal_gnn_jepa": "Temporal\nGNN\nJEPA",
    }
    names = [short_names.get(r["method"], r["method"].replace("_", "\n")) for r in results]
    f1 = [r["metrics"]["f1"] for r in results]
    img = Image.new("RGB", WIDE, WHITE)
    d = ImageDraw.Draw(img)
    d.text((48, 24), "Baseline Comparison on Offline SDN/DDoS Dataset", fill=INK, font=font(30, True))
    left, top, chart_w, chart_h = 85, 100, 1030, 330
    d.line((left, top + chart_h, left + chart_w, top + chart_h), fill=INK, width=2)
    d.line((left, top, left, top + chart_h), fill=INK, width=2)
    for tick in range(0, 6):
        val = tick / 5
        y = top + chart_h - int(val * chart_h)
        d.line((left - 8, y, left + chart_w, y), fill=(235, 238, 242), width=1)
        d.text((35, y - 10), f"{val:.1f}", fill=DARK_GRAY, font=font(16))
    slot = chart_w // len(f1)
    bar_w = max(42, slot - 20)
    colors = [GRAY, DARK_GRAY, BLUE, ORANGE, TEAL, (118, 84, 162), (145, 92, 57), RED, (70, 130, 180)]
    for i, val in enumerate(f1):
        x = left + 12 + i * slot
        y = top + chart_h - int(val * chart_h)
        d.rectangle((x, y, x + bar_w, top + chart_h), fill=colors[i % len(colors)], outline=INK)
        d.text((x - 1, y - 25), f"{val:.3f}", fill=INK, font=font(15, True))
        d.multiline_text((x - 5, top + chart_h + 14), names[i], fill=INK, font=font(13), spacing=1)
    d.text((470, 570), "F1 score", fill=INK, font=font(22, True))
    save(img, "baseline_f1_comparison.png")


def public_baseline_bars():
    payload = json.loads(Path("outputs/eval/nf_unsw_nb15_baseline_comparison.json").read_text())
    results = payload["results"]
    supervised_path = Path("outputs/eval/nf_unsw_nb15_torch_supervised_baselines.json")
    if supervised_path.exists():
        supervised_payload = json.loads(supervised_path.read_text())
        insert_at = 4
        results = results[:insert_at] + supervised_payload["results"] + results[insert_at:]
    torch_path = Path("outputs/eval/nf_unsw_torch_sdn_jepa_eval.json")
    if torch_path.exists():
        torch_payload = json.loads(torch_path.read_text())
        results = results + [
            {
                "method": "temporal_gnn_jepa_public",
                "metrics": torch_payload["metrics"],
            }
        ]

    short_names = {
        "one_class_feature_distance": "One-class\nfeature\ndistance",
        "supervised_logistic_regression": "Logistic\nregression",
        "supervised_numpy_mlp": "NumPy\nMLP",
        "supervised_gradient_boosted_stumps": "Boosted\nstumps",
        "supervised_torch_mlp": "Torch\nMLP",
        "supervised_torch_cnn": "Torch\nCNN",
        "supervised_torch_lstm": "Torch\nLSTM",
        "raw_feature_ridge_surprise": "Raw ridge\nsurprise",
        "lewm_sdn_latent_surprise_phase": "LeWM-SDN\nlatent +\nphase",
        "temporal_gnn_jepa_public": "Temporal\nGNN\nJEPA",
    }
    names = [short_names.get(r["method"], r["method"].replace("_", "\n")) for r in results]
    f1 = [r["metrics"]["f1"] for r in results]
    img = Image.new("RGB", WIDE, WHITE)
    d = ImageDraw.Draw(img)
    d.text((48, 24), "Public NF-UNSW-NB15 Test F1 Comparison", fill=INK, font=font(30, True))
    left, top, chart_w, chart_h = 85, 100, 1030, 330
    d.line((left, top + chart_h, left + chart_w, top + chart_h), fill=INK, width=2)
    d.line((left, top, left, top + chart_h), fill=INK, width=2)
    for tick in range(0, 6):
        val = tick / 5
        y = top + chart_h - int(val * chart_h)
        d.line((left - 8, y, left + chart_w, y), fill=(235, 238, 242), width=1)
        d.text((35, y - 10), f"{val:.1f}", fill=DARK_GRAY, font=font(16))
    slot = chart_w // len(f1)
    bar_w = max(38, slot - 18)
    colors = [GRAY, DARK_GRAY, BLUE, (145, 92, 57), ORANGE, TEAL, (118, 84, 162), RED, (70, 130, 180), (30, 64, 96)]
    for i, val in enumerate(f1):
        x = left + 10 + i * slot
        y = top + chart_h - int(val * chart_h)
        d.rectangle((x, y, x + bar_w, top + chart_h), fill=colors[i % len(colors)], outline=INK)
        d.text((x - 1, y - 25), f"{val:.3f}", fill=INK, font=font(14, True))
        d.multiline_text((x - 5, top + chart_h + 14), names[i], fill=INK, font=font(12), spacing=1)
    d.text((470, 570), "F1 score", fill=INK, font=font(22, True))
    save(img, "public_baseline_f1_comparison.png")


def score_separation():
    payload = json.loads(Path("outputs/eval/baseline_comparison.json").read_text())
    lewm = [r for r in payload["results"] if r["method"] == "lewm_sdn_latent_surprise_phase"][0]
    normal = lewm["normal_score_mean"]
    attack = lewm["attack_score_mean"]
    img = Image.new("RGB", (1000, 560), WHITE)
    d = ImageDraw.Draw(img)
    d.text((48, 30), "LeWM-SDN Reference Anomaly Score Separation", fill=INK, font=font(30, True))
    d.text((48, 76), "Attack windows have much higher predictive latent surprise.", fill=DARK_GRAY, font=font(19))
    left, top, chart_w, chart_h = 140, 130, 700, 300
    max_val = attack * 1.12
    d.line((left, top + chart_h, left + chart_w, top + chart_h), fill=INK, width=2)
    d.line((left, top, left, top + chart_h), fill=INK, width=2)
    for label, val, color, x in [
        ("Normal", normal, TEAL, left + 140),
        ("DDoS", attack, RED, left + 430),
    ]:
        h = int(val / max_val * chart_h)
        d.rectangle((x, top + chart_h - h, x + 150, top + chart_h), fill=color, outline=INK)
        d.text((x + 12, top + chart_h + 20), label, fill=INK, font=font(22, True))
        d.text((x - 18, top + chart_h - h - 34), f"{val:.2f}", fill=INK, font=font(20, True))
    d.text((50, 250), "Mean score", fill=INK, font=font(18))
    save(img, "score_separation.png")


def main():
    architecture()
    dataset_timeline()
    baseline_bars()
    public_baseline_bars()
    score_separation()


if __name__ == "__main__":
    main()
