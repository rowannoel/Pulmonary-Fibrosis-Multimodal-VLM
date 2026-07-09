# Pulmonary-Fibrosis-Multimodal-VLM

PyTorch implementation of a multimodal pulmonary fibrosis detection framework using chest X-ray images, radiology reports, and pretrained medical vision-language models.

This repository accompanies the research paper:

**Multimodal Pulmonary Fibrosis Detection Using Chest X-rays and Radiology Reports**

---

## Overview

This project investigates multimodal pulmonary fibrosis classification by combining chest X-ray images with their corresponding radiology reports.

Three pretrained medical vision-language models are evaluated:

- BioViL-T
- PubMedCLIP
- KAD

Image and text embeddings are projected into a shared latent space and concatenated before being classified using a lightweight multilayer perceptron (MLP).

The framework was evaluated using both the PadChest and MIMIC-CXR datasets.

---

## Repository Structure

```
.
├── train_padchest.py
├── train_mimic.py
├── training.py
├── models.py
├── datasets.py
├── metrics.py
├── utils.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Requirements

Install the required packages with

```bash
pip install -r requirements.txt
```

---

## Datasets

This repository does **not** include the datasets or generated embeddings.

The experiments use:

- PadChest
- MIMIC-CXR

Please obtain these datasets from their official sources.

---

## Embeddings

This framework expects precomputed image, text, and prompt embeddings generated from the selected vision-language model.

Embedding files should correspond exactly to the associated CSV file (same ordering and number of samples).

---

## Running Experiments

Run the PadChest experiments with

```bash
python train_padchest.py
```

Run the MIMIC-CXR experiments with

```bash
python train_mimic.py
```

---

## Results

Performance is reported using

- Accuracy
- F1 Score
- Area Under the Precision-Recall Curve (AUPRC)

Experiments are repeated using five random seeds with patient-level train/validation/test splits.

---

## Citation

If you use this repository in your research, please cite the accompanying paper.

```bibtex
% Citation will be added after publication.
```

---

## License

This project is released under the MIT License.
