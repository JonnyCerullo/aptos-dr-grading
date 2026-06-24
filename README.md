# Diabetic Retinopathy Severity Grading — AI for Medicine

Final project for the **AI for Medicine** course (M.Sc. Artificial Intelligence, University of Bologna, Prof. Stefano Diciotti, A.Y. 2025–2026).

## Project

Deep learning classifier for the 5-class severity grading of diabetic retinopathy from fundus photographs, trained on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset (3,662 labelled images).

Pipeline: EfficientNet-B0 (ImageNet-pretrained) + Ben Graham intensity harmonisation + stratified 70/15/15 split + 5-fold cross-validation with multi-seed stability analysis + Grad-CAM interpretability.

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JonnyCerullo/aptos-dr-grading/blob/main/aptos-dr-grading.ipynb)

## Setup

1. Create a free Kaggle account, generate an API token in the new KGAT_ format, and store it as a Colab secret named `KAGGLE_API_TOKEN` with notebook access enabled.
2. Click the badge above to open the notebook in Colab.
3. Runtime → Change runtime type → T4 GPU.
4. Run cells top to bottom. Dataset is fetched automatically via `kagglehub`; preprocessing is cached after the first run.

## License

- **Code** (notebook, scripts, README): MIT License — see [LICENSE](LICENSE).
- **Dataset**: APTOS 2019 is released by the Asia Pacific Tele-Ophthalmology Society
  on Kaggle under the [competition data-use terms](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules);
  non-commercial academic/research use only, redistribution of the dataset is not permitted.
  The dataset is **not redistributed** through this repository — it is fetched at runtime
  via the Kaggle API and remains subject to its original terms.
- **Pretrained weights**: ImageNet-pretrained EfficientNet-B0 is loaded via the
  [timm](https://github.com/huggingface/pytorch-image-models) library, released under
  Apache License 2.0.

## Author

Jonathan Ted Benson Cerullo Uyi — jonathan.cerullouyi@studio.unibo.it
