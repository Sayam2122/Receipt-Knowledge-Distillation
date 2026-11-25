# Receipt extraction via knowledge distillation (v2)

Teacher: frozen **Qwen2.5-VL-3B-Instruct**. Student: **~18k-parameter** layout-aware candidate scorer (<<1/10 of teacher).
Student runs alone at inference: `image -> OCR (RapidOCR) -> candidates -> student -> JSON`.

## Run order
```bash
pip install -r requirements.txt
# 1. candidates (put SROIE in receipt_dataset/{train,eval,test}/{img,box,entities})
python prepare_data.py --dataset receipt_dataset --out kd_data                 # GT boxes
python prepare_data.py --dataset receipt_dataset --out kd_data_ocr --box-source ocr   # real-OCR boxes (recommended for the final model)

# 2. teacher (GPU: Colab/Kaggle T4). Copy kd_data/ + receipt_dataset/ there.
python teacher_qwen.py --mode soft --split train --data kd_data --dataset receipt_dataset --limit 300   # soft labels (true logits)
python teacher_qwen.py --mode json --split test  --data kd_data --dataset receipt_dataset               # teacher answers + boxes for the figure

# 3. ablation: same seeds => identical init + order. 'teacher' subset = matched data.
python train.py --data kd_data --mode ce      --subset teacher --teacher kd_data/teacher_soft_train.json --seeds 0 1 2 --out runs/ce
python train.py --data kd_data --mode kd      --subset teacher --teacher kd_data/teacher_soft_train.json --seeds 0 1 2 --out runs/kd
python train.py --data kd_data --mode kd_conf --subset teacher --teacher kd_data/teacher_soft_train.json --seeds 0 1 2 --out runs/kdconf
python train.py --data kd_data --mode ce      --subset all --seeds 0 1 2 --out runs/ce_all   # upper reference (all annotations)

# 4. report: value-exact and localisation reported separately; teacher vs student vs GT table + images
python evaluate.py --data kd_data --dataset receipt_dataset --split test --ckpt runs/kd/seed0.pt \
       --ce-ckpt runs/ce/seed0.pt --teacher-json kd_data/teacher_json_test.json --out report_eval
# 5. demo
python predict.py --ckpt runs/kd/seed0.pt --image any_receipt.jpg --save-vis out.jpg
python app.py --ckpt runs/kd/seed0.pt
```
Tests: `python -m pytest tests` (uses synthetic data: `tests/make_synthetic.py`, `tests/make_fake_teacher.py` are TEST-ONLY).

## Status (be honest in the report)
* Verified here: unit tests, data->train->evaluate->predict->UI on **synthetic** receipts incl. real OCR on an unseen image.
  100% on synthetic data is meaningless (too easy) - it only proves the plumbing.
* **Not run**: SROIE (not available to the author), and `teacher_qwen.py` against the real model (the digit-probability
  combiner is unit-tested; the HF model calls are not). Expect to debug small API details on the first Colab run.
* Line items: heuristic only, unevaluated (SROIE has no line-item labels).
