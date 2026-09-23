# Animal Classifier Specification

## Goal

Predict which animal appears in a photograph and return the predicted animal,
its confidence score, and scores for every discovered animal class.

## Data contract

- Raw images are organized as `data/animals/<animal_name>/`; the folder name is
  the label.
- Class names and the number of classes are always read from directories. No
  animal name or class count is hard-coded.
- Every image must be readable by Pillow. There must be at least two classes and
  at least 10 images in each class.
- Data is split independently per class into 80% training and 20% test data,
  using fixed random seed 42. The command line may override the ratio and seed.

## Model and preprocessing

- Use torchvision MobileNetV2 with ImageNet weights for the normal training
  path.
- Freeze `model.features`, replace `model.classifier[1]` with a new
  `torch.nn.Linear` sized for the discovered classes, and keep
  `model.features` in evaluation mode while training.
- `--from_scratch` disables ImageNet weights and leaves every layer trainable.
- Training accepts `--curve_out` (default `training_curve.png`) and saves a
  loss-per-epoch chart at that path after training.
- The same transform is used in training and prediction: resize to 224x224 and
  normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
  `(0.229, 0.224, 0.225)`.

## Artifacts and interfaces

The saved model is a directory containing:

- `model.pt`: the model `state_dict`.
- `classes.json`: the ordered list of class names.

The prediction response is `{ "animal", "confidence", "all_scores" }`.
Azure scoring accepts `{ "image": "<base64 image>" }` and returns that
response, or `{ "error": "<message>" }` on errors.

## Quality gate

Evaluation prints overall accuracy, per-animal accuracy, and a confusion matrix
(real animals as rows and predicted animals as columns), writes `metrics.json`,
and exits with code 1 when test accuracy is below `min_accuracy` (default 0.70).

## Acceptance criteria

- **AC1:** pytest data checks pass.
- **AC2:** `prep.py` finds five animals and creates train/test folders.
- **AC3:** `train.py` finishes and saves `model.pt` and `classes.json`.
- **AC4:** `evaluate.py` prints accuracy and a confusion matrix and writes `metrics.json`.
- **AC5:** Test accuracy is at least 0.70 and the quality gate passes.
- **AC6:** `predict.py` prints an animal and confidence for a demo image.
- **AC7:** `score.py` returns an animal and confidence for `sample-request.json`.
- **AC8:** No script hard-codes animal names or the number of animals.
- **AC9:** `README.md` explains every step.