# Visualizations

`qismet_ml_explained.mp4` is a narrated explainer (about 7 minutes, 1920×1080, 60 fps) of the machine learning in Qismet. It covers:

1. The Fashion-MNIST task and how an image becomes 784 numbers
2. The 784 → 16 → 10 network, weights, and ReLU
3. Training: the network's real learning curve and learned neuron templates
4. A real prediction, traced through the trained network
5. Pruning: magnitude pruning, and pruning connections vs. whole neurons
6. Qismet's grouped pruning results (50% and 75% of neurons removed)
7. Retraining from a fresh start vs. rewinding to the original weights
8. The Lottery Ticket Hypothesis (Frankle & Carbin, 2019), tested on this network

The video does not cover QAOA.

Every image, weight map, curve and number comes from this repository:

| File | What it does |
|---|---|
| `lottery_ticket_experiment.py` | One-shot and iterative magnitude pruning on Qismet's exact data, model and training recipe, 5 seeds each. It compares rewinding to the original weights against fresh random weights. Writes `lottery_ticket_*.csv` / `.json`. |
| `prepare_assets.py` | Pulls the trained seed-11 network, its original initialization, example images and aggregate results into `video_assets.npz` / `.json`. |
| `make_video.py` | Draws every frame with skia, narrates with the macOS `say` command, and encodes with ffmpeg (H.264, CRF 15, AAC audio). |

## Lottery ticket result (iterative pruning, mean of 5 seeds)

| Weights removed | Rewound to original start | Fresh random start |
|---:|---:|---:|
| 0% | 82.2% | 82.3% |
| 49% | 82.0% | 81.4% |
| 79% | 82.1% | 80.5% |
| 89% | 81.3% | 78.8% |
| 93% | 80.3% | 76.7% |

## Rebuild

```bash
pip install skia-python imageio-ffmpeg numpy pandas pillow torch
python lottery_ticket_experiment.py   # ~10 min on CPU
python prepare_assets.py
python make_video.py                  # --scale 2 for 4K, --voice Daniel, --no-captions
python make_video.py --still 13:20    # check one frame: scene 13 at 20 s
```

Narration needs macOS. For a more natural voice, download a Premium or Enhanced voice in System Settings → Accessibility → Spoken Content, then pass its name with `--voice`.
