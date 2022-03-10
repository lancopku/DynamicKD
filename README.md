# Dynamic KD

Code for EMNLP 2021 Main Conference Paper: 

Dynamic Knowledge Distillation for Pre-traiend Language Models

[pdf](https://aclanthology.org/2021.emnlp-main.31/)

## Code Example

We provide a plug-in module for ease of use of our Dynamic KD idea:

```python3
import torch
import torch.nn as nn
from torch.nn import functional as F

def dynamic_kd_loss(student_logits, teacher_logits, temperature=1.0):

  with torch.no_grad():
    student_probs = F.softmax(student_logits, dim=-1)
    student_entropy = - torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1) # student entropy, (bsz, )
    # normalized entropy score by student uncertainty:
    # i.e.,  entropy / entropy_upper_bound
    # higher uncertainty indicates the student is more confusing about this instance
    instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))

  input = F.log_softmax(student_logits / temperature, dim=-1)
  target = F.softmax(teacher_logits / temperature, dim=-1)
  batch_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
  weighted_kld = torch.mean(batch_loss * instance_weight)

return weighted_kld

```

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

## Citation

If you find this repo useful, please kindly cite our paper:
```
@inproceedings{Li2021DynamicKD,
  title={Dynamic Knowledge Distillation for Pre-trained Language Models},
  author={Lei Li and Yankai Lin and Shuhuai Ren and Peng Li and Jie Zhou and Xu Sun},
  booktitle={EMNLP},
  year={2021}
}
```
