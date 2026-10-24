# ML Systems Labs

One repository, one folder per lab (`lab01/` ... `lab11/`, `project/`).

## Setup

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

## Lab 1

```
python lab01/src/print_versions.py > lab01/results/versions.txt
python lab01/src/run_lab01.py
```
