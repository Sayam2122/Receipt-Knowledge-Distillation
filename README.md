# Receipt Field Extraction by Knowledge Distillation

**IE 643 Course Project · IIT Bombay · Team Neural Ninjas**
Arnav Ashish Deshmukh (24B3033) · Sayam (24B2240)

We distil a frozen **Qwen2.5-VL-3B-Instruct** teacher into a compact **17,987-parameter** MLP that picks the **vendor, date and total** from OCR candidates on receipts (SROIE).
At inference only the small student runs:

```
receipt image -> RapidOCR -> candidate generation -> student MLP -> vendor / date / total (+ boxes, confidence)
```

## Main finding

Comparing the distilled student with the *same* network trained on annotations alone shows **when the teacher helps**:
distillation is most useful when annotations are scarce and gives no advantage once enough clean annotations exist.

Mean value-exact accuracy (vendor, date, total) on the 347 SROIE test receipts, 3 seeds:

| Annotated receipts | CE (annotations only) | KD (annotations + teacher) | Gain |
|---|---|---|---|
| 25 | 87.9 | 91.9 | **+4.0** |
| 50 | 89.0 | 91.3 | **+2.3** |
| 100 | 92.3 | 91.5 | -0.8 |
| 300 | 94.5 | 93.5 | -1.0 |
| 0 (teacher labels only) | - | 91.3 | - |

With all 532 annotations (CE, no teacher): vendor 92.2 / date 96.9 / total 97.8 (mean 95.6). Value-exact and localisation (IoU >= 0.5) are always reported separately.

## Method in brief

* **Candidates.** Vendor candidates are the first OCR lines (and two-line spans); dates and amounts are found by regular expressions.
* **Student.** 246 features per candidate (192 hashed character n-grams of the candidate and its row context + 54 scalars: geometry, row rank, keyword flags, amount rank, ...), a shared trunk 246 -> 64 -> 32 and one head per field, softmax over the receipt's candidates.
* **Teacher labels.** Qwen reads the image freely and its answer is mapped onto the candidates (string similarity for vendor, exact value for date and total). No annotation is used.
* **Loss.** `L = (1 - lambda) * CE + lambda * T^2 * KL(q || p)`, lambda = 0.5, T = 2. Receipts without an annotation use the KL term only.

## Setup

```bash
pip install -r requirements.txt     # Windows PowerShell first: $env:PYTHONUTF8 = "1"
python -m pytest tests              # 11 tests, synthetic data only
```

The SROIE 2019 dataset ([ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13)) is **not** included. Place it as `receipt_dataset/{train,test}/{img,box,entities}`.

## Quick demo (no dataset needed)

The repo includes one trained checkpoint, `runs/ce_all/seed0.pt` (student trained on all 532 annotations).

```bash
python predict.py --ckpt runs/ce_all/seed0.pt --image receipt.jpg --save-vis out.png --show-ocr
python app.py --ckpt runs/ce_all/seed0.pt        # web page: upload, fields, boxes, confidence
```

The first run downloads the RapidOCR models and takes about 20 s; later runs take 5-7 s per receipt on CPU, almost all of it OCR.

## Reproducing the experiments

```bash
# 1. candidates and features from SROIE's annotated boxes
python prepare_data.py --dataset receipt_dataset --out kd_data

# 2. teacher (GPU, e.g. Colab T4): free-form image reading, then map answers onto candidates
python teacher_qwen.py --mode json --split train --data kd_data --dataset receipt_dataset --limit 300
python teacher_qwen.py --mode json --split test  --data kd_data --dataset receipt_dataset --limit 100
python soft_from_json.py --data kd_data --split train --out kd_data/teacher_soft_train_v2.json

# 3. students: CE vs KD with the same seeds, initialisation and data order
python train.py --data kd_data --mode ce --subset all --seeds 0 1 2 --out runs/ce_all
python train.py --data kd_data --mode ce --subset teacher --teacher kd_data/teacher_soft_train_v2.json --annotated-n 50 --seeds 0 1 2 --out runs/ce_50
python train.py --data kd_data --mode kd --subset teacher --teacher kd_data/teacher_soft_train_v2.json --annotated-n 50 --seeds 0 1 2 --out runs/kd_50
#    label-budget sweep: --annotated-n 25 / 50 / 100 / 300;  --annotated-n 0 = teacher labels only

# 4. evaluation
python evaluate.py --data kd_data --dataset receipt_dataset --split test --ckpt runs/kd_50/seed0.pt \
       --ce-ckpt runs/ce_50/seed0.pt --teacher-json kd_data/teacher_json_test.json --only-teacher-covered --out report_eval
python eval_ocr.py --ckpt runs/ce_all/seed0.pt --dataset receipt_dataset --limit 100   # SROIE boxes vs real RapidOCR boxes
python rule_check.py --data kd_data                                                    # vendor-rule check on training data only
python eval_unseen.py --data kd_data --ckpts runs/ce_all/seed0.pt runs/ce_all/seed1.pt runs/ce_all/seed2.pt
```

## Repository layout

| File | Purpose |
|---|---|
| `sroie_data.py` | parse SROIE, candidate generation, ground-truth matching, vendor cleaning |
| `features.py`, `student.py` | 246-d candidate features; the MLP student and the CE + KL loss |
| `teacher_qwen.py`, `soft_from_json.py` | run Qwen2.5-VL and turn its answers into candidate distributions |
| `train.py` | CE / KD / confidence-weighted KD training, label-budget option `--annotated-n` |
| `metrics.py`, `evaluate.py` | value-exact and localisation metrics, side-by-side teacher / student / ground-truth report |
| `ocr.py`, `extract.py`, `prettify.py` | RapidOCR wrapper, image-to-fields extractor, display-only space restoration |
| `predict.py`, `app.py` | command-line and Gradio interfaces |
| `eval_ocr.py`, `eval_unseen.py`, `rule_check.py` | real-OCR evaluation and leakage controls |
| `line_items.py` | heuristic line items (not evaluated) |
| `tests/` | unit tests on synthetic data |

## Limitations (please read)

* **Real OCR costs accuracy.** The student is trained on SROIE's clean boxes. With RapidOCR boxes the same student falls from 96.0% to 82.0% mean value-exact (100 test receipts, one seed). Training on OCR boxes is the obvious next step.
* **Vendor rule and test leakage.** `clean_vendor` (strips a trailing company registration number) was first noticed in test errors. On training data only it fixes 126 of 471 vendor lines and breaks none; on the 146 test receipts the error analysis never reached, the mean is 95.7% (all test: 95.6%). Without the rule the mean is 88.0%.
* **Zero-annotation run.** It uses no training annotations, but ground-truth validation labels still select the checkpoint.
* **Small samples.** The teacher covers 300 training and 100 test receipts; the teacher-vs-student table uses single-seed students.
* **Line items** come from a heuristic and have no labels, so they are not evaluated. Claims here cover vendor, date and total only.

## References

* Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu and C. V. Jawahar. ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction, ICDAR, 2019. https://doi.org/10.1109/ICDAR.2019.00244
* Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou. LayoutLM: Pre-training of Text and Layout for Document Image Understanding, KDD, 2020. https://arxiv.org/abs/1912.13318
* Shuai Bai et al. Qwen2.5-VL Technical Report, arXiv, 2025. https://arxiv.org/abs/2502.13923
* Geoffrey Hinton, Oriol Vinyals and Jeff Dean. Distilling the Knowledge in a Neural Network, NeurIPS Deep Learning Workshop, 2015. https://arxiv.org/abs/1503.02531
