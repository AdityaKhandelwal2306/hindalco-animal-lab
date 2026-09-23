# Animal Classifier

This beginner-friendly project trains a torchvision MobileNetV2 classifier on
folders of animal images. The complete design is in
[`specs/animal_classifier_spec.md`](specs/animal_classifier_spec.md).

## Install

On Linux, from the project root:

```bash
python -m pip install -r requirements.txt
```

The first requirements line selects CPU PyTorch wheels. Azure's supplied
environment files are unchanged and provide the cloud runtime dependencies.

## Check, split, train, and evaluate

Put images in `data/animals/<lowercase-animal-name>/`. Then run:

```bash
python -m pytest tests -v
python src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
python src/train.py --train_data split/train --model_dir models
python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
python src/predict.py --model_dir models --image demo_images/cat_demo.jpg
```

`prep.py` validates every image and makes an 80/20 per-animal split using seed
42. Training saves `models/model.pt` and `models/classes.json`. Evaluation
prints the confusion matrix and exits with status 1 below 70% accuracy. You can
change the optional `--epochs`, `--lr`, `--batch_size`, `--test_ratio`,
`--min_images`, `--seed`, and `--min_accuracy` arguments. Add `--from_scratch`
to train every layer without ImageNet weights. Training also saves the loss
chart `training_curve.png`; use `--curve_out path/to/chart.png` to choose a
different output path.

## Test the Azure scoring contract locally

```bash
python src/make_request.py --image demo_images/cat_demo.jpg
AZUREML_MODEL_DIR=models python -c \
  'import sys, json; sys.path.insert(0, "src"); import score; score.init(); print(score.run(open("sample-request.json").read()))'
```

The request is resized to at most 512 pixels and encoded as JPEG to stay below
Azure endpoint request limits. Azure ML uses `src/score.py` and its
`AZUREML_MODEL_DIR` model mount automatically.

## Streamlit app

Install the UI dependencies and set the endpoint credentials:

```bash
python -m pip install -r app/requirements.txt
export ENDPOINT_URL='https://your-endpoint'
export ENDPOINT_KEY='your-key'
streamlit run app/app.py
```

The app uploads a photo, sends the same compact JSON body, shows the prediction,
confidence, score chart, and a warning below 60% confidence. The supplied
`azureml/` folder and `azure-pipelines.yml` describe the optional Azure ML
pipeline/deployment and are not modified by this project.