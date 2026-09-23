You are an ML engineer. Use SPEC-DRIVEN DEVELOPMENT: write the spec first, 

then code exactly to the spec, then prove it works. 

I am on Linux OS. 

  

CONTEXT 

- Images are in data/animals/<animal_name>/ (5 animals, 20 .jpg each). 

  The folder name is the label. 

- Extra test photos are in demo_images/. 

- The folder azureml/ and the file azure-pipelines.yml are GIVEN. 

  Do NOT change them. Read them: your scripts must accept exactly the 

  arguments they use. 

  

PHASE 1 - SPEC 

Create specs/animal_classifier_spec.md describing: 

- Goal: predict which animal is in a photo, with a confidence score. 

- Model: transfer learning with torchvision MobileNetV2 (ImageNet weights). 

  Freeze model.features, replace model.classifier[1] with a new Linear 

  layer. Keep model.features in eval mode while training. 

- Images: resize to 224x224 + ImageNet normalisation. The SAME transform 

  must be used for training and prediction. 

- Never hard-code the number or names of animals. Read them from the 

  folder names, so a 6th animal works with no code change. 

- Split: 80% train / 20% test for each animal, fixed seed 42. 

- Quality gate: exit with code 1 if test accuracy < min_accuracy (0.70). 

- Saved model = a folder containing model.pt (state_dict) and 

  classes.json (list of animal names). 

- Acceptance criteria AC1-AC9 (below). 

  

PHASE 2 - CODE (exact files and command-line arguments) 

- src/model_utils.py: get_transform(); build_model(num_classes, 

  pretrained=True, freeze=True); save_model(model, classes, model_dir); 

  load_model(model_dir) - search sub-folders for model.pt and classes.json; 

  predict_image(model, classes, pil_image) returning 

  {"animal", "confidence", "all_scores"}; log_metric(name, value, step=None) 

  that logs with mlflow ONLY when env var MLFLOW_TRACKING_URI starts with 

  "azureml", otherwise does nothing. 

- src/prep.py --raw_data --train_out --test_out [--test_ratio 0.2] 

  [--min_images 10] [--seed 42]: check every image opens, need >=2 animals 

  and >=10 images each, copy into <out>/<animal>/ folders, print a table. 

  If raw_data holds a single wrapper folder, look inside it. 

- src/train.py --train_data --model_dir [--epochs 10] [--lr 0.001] 

  [--batch_size 16] [--from_scratch]: torchvision ImageFolder, print loss 

  and train accuracy per epoch, save the model. --from_scratch = no 

  pretrained weights and every layer trainable (an experiment). 

- src/evaluate.py --model_dir --test_data --metrics_out 

  [--min_accuracy 0.70]: print overall and per-animal accuracy and a 

  confusion matrix (rows = real, columns = predicted), write metrics.json, 

  apply the quality gate. 

- src/predict.py --model_dir --image: print 

  "Prediction: <animal>  (confidence NN%)". 

- src/score.py: Azure ML scoring script. init() loads the model from env 

  var AZUREML_MODEL_DIR. run(raw_data) accepts JSON 

  {"image": "<base64 image>"} and returns {"animal", "confidence", 

  "all_scores"}. Print one log line per prediction. On error return 

  {"error": "<message>"}. 

- src/make_request.py --image [--out sample-request.json]: shrink the photo 

  to max 512 px (JPEG), then write the JSON body that score.py expects. 

  (Azure endpoints reject requests larger than about 1.5 MB.) 

- tests/test_data.py: pytest checks on data/animals using ONLY Pillow and 

  pytest (no torch): folder exists, >=2 animals, lowercase folder names 

  without spaces, no loose images at the top level, each animal has >=10 

  readable images. 

- app/app.py: Streamlit page "Animal Predictor". Read ENDPOINT_URL and 

  ENDPOINT_KEY from environment variables. Upload a photo, show it, shrink 

  it to max 512 px JPEG, POST {"image": <base64>} with header 

  "Authorization: Bearer <key>", show the animal, the confidence, a bar 

  chart of all_scores, and a warning if confidence < 0.60. 

  app/requirements.txt: streamlit, requests, pillow, pandas. 

- requirements.txt, first line: 

  --extra-index-url https://download.pytorch.org/whl/cpu 

  then: torch, torchvision, pillow, pytest 

- .gitignore: models/, split/, metrics/, mlruns/, __pycache__/, .env, 

  sample-request.json 

- README.md: how to run every step. 

Keep the code simple and commented for beginners. No other cloud services. 

  

PHASE 3 - VALIDATE 

Run: 

  python -m pip install -r requirements.txt 

  python -m pytest tests -v 

  python src/prep.py --raw_data data/animals --train_out split/train --test_out split/test 

  python src/train.py --train_data split/train --model_dir models 

  python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics 

  python src/predict.py --model_dir models --image demo_images/cat_demo.jpg 

Test score.py: run make_request.py on a demo image, then set 

AZUREML_MODEL_DIR=models, import score, call init() and run(). 

Create specs/validation_report.md with AC1-AC9 as PASS/FAIL plus evidence 

(the key output lines). If anything fails, fix it and run again. 

  

ACCEPTANCE CRITERIA 

AC1 pytest data checks pass 

AC2 prep.py finds 5 animals and creates train/test folders 

AC3 train.py finishes and saves model.pt + classes.json 

AC4 evaluate.py prints accuracy + confusion matrix and writes metrics.json 

AC5 test accuracy >= 0.70 (quality gate passes) 

AC6 predict.py prints an animal + confidence for a demo image 

AC7 score.py returns animal + confidence for sample-request.json 

AC8 no script hard-codes the number or names of animals 

AC9 README explains every step 

 