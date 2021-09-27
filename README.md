# Dynamic KD

Code for EMNLP 2021 Main Conference Paper: Dynamic Knowledge Distillation for Pre-traiend Language Models

## Environment Setup

We recommend using Anaconda for setting up the environment of experiments:
```bash
git clone https://github.com/lancopku/DynamicKD.git
cd DynamicKD
conda create -n dkd python=3.7
conda activate dkd
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

## Train Teacher Model

Using the provided `scripts/train_teacher.sh` script to train corresponding teacher model like BERT-base and BERT-large on the target datasets. Note that the teacher and student performance on some small datasets may different from the reported numbers in the paper due to the randomness.


## Dynamic Teacher Adoption
See `scripts/teacher.sh` and `dynamic_teacher.py` for details.

## Dynamic Data Selection
See `scripts/data.sh` and `dynamic_data.py` for details.

## Dynamic Supervision Adjustment
See `scripts/objective.sh` and `dynamic_objective.py` for details.
