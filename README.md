# ML Systems Labs

One repository, one folder per lab (`lab01/` ... `lab11/`, `project/`).

## Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lab 1

```
python lab01/src/print_versions.py > lab01/results/versions.txt
python lab01/src/run_lab01.py
```

## Lab 2

```
python -m pytest lab02/tests
cd lab02/src && python run_lab02.py
```

## Lab 3

Architectures and their costs (MLP, CNN, patch attention on Fashion-MNIST). Needs the dataset in `data/FashionMNIST/raw` (torchvision's `FashionMNIST(root="data", download=True)` fetches it).

```
python -m pytest lab03/tests
cd lab03/src
python cost.py     # parameters, FLOPs (hand vs thop), latency
python train.py    # 5 epochs per model, about 10 minutes on 2 CPU cores
python plots.py    # accuracy vs parameters / latency
```

Write-up: `lab03/report.md`.

## Lab 4

Leakage-free data pipeline for UCI Adult (validation, cleaning, derived features, label audit, hash, data card). One command:

```
python lab04/src/run_pipeline.py              # add --source openml to use fetch_openml
python -m pytest lab04/tests
```

Settings: `lab04/config.yaml`. Write-up: `lab04/report.md`; data card: `lab04/data_card.md`. Rerunning the pipeline rewrites the data card (`--skip-card` keeps it).


## Lab 5 (autograd engine)

```
python -m pytest lab05/tests
cd lab05/src
python verify.py            # item 4: engine vs PyTorch vs numerical gradients
python moons.py             # item 5: make_moons training, engine vs PyTorch (~5 min)
python inspect_graph.py     # item 6: graphs, shapes, strides, contiguity
python compile_bench.py     # item 6: eager vs torch.compile vs jit.trace; contiguity timings (~3 min)
```
Report: `lab05/report.md`.
