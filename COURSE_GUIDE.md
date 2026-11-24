# Choosing a coursework entry point

Each assignment has its own environment and inputs. Choose a row below, then use the linked assignment README for its commands.

| Task | Working directory | Inputs and outputs |
|---|---|---|
| [HW1 · Object detection](hw1_R1443001/README.md) | Install in `hw1_R1443001/`. Prepare data under its `src/dataset/`; use the selected directory under `src/model/` for the detector. | HW1 competition images and annotations → YOLO data and trained weights. Hyper-YOLO validation can export COCO JSON; the YOLOv12 prediction script produces submission CSV. |
| [HW2 · Long-tailed detection](hw2_R1443001/README.md) | Install in `hw2_R1443001/`, prepare data in its own `src/dataset/`, then select a detector under its `src/model/`. | HW2 competition images and labels → prepared/augmented data, detector weights, and model-specific predictions for submission. Keep the assignment's dataset configuration with the detector. |
| [HW3 · Image generation](hw3_R1443001/code/README.md) | Install in `hw3_R1443001/code/`; the training and generation scripts are in its `src/`. | Training images → generator checkpoints → generated images. The assignment guide compares generated and reference image folders for FID. |
| [Final project · Citrus classification](taica_final_project/README.md) | Install in `taica_final_project/` and use `src/` for the numbered experiments. | Citrus images and split lists → baseline and teacher results → distilled student checkpoints. The [project report](taica_final_project/TAICA_Final%20Project%20Report.pdf) records the submitted work. |

## Which final-project checkpoint goes where?

[Teacher fine-tuning](taica_final_project/src/001_finetune_teacher.py) writes into `checkpoints_finetune/`. [Distillation](taica_final_project/src/002_distillation.py) expects `best_teacher_model_convnextv2_large.pth` from that directory and writes the ConvNeXt V2 Atto student's outputs to `checkpoints_kd/`. Check that the selected teacher matches that input filename before continuing.

[ONNX export](taica_final_project/src/005_to_onnx.py) follows a separate MobileViT comparison run. It reads `checkpoints_comparison_mobilevit/best_finetune_model.pth` and writes `onnx_ckpt/model.onnx`. Use the MobileViT checkpoint for that script.
