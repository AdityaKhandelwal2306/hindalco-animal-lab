# Validation Report

Validation was run on Linux from `/root/training_animal_classification` after
installing `requirements.txt`. The supplied `azureml/` directory and
`azure-pipelines.yml` were not changed.

| Criterion | Result | Evidence |
|---|---|---|
| AC1 | PASS | `python -m pytest tests -v`: **8 passed in 0.63s** (and a final rerun: `8 passed in 0.12s`). |
| AC2 | PASS | `prep.py` table reported `cat`, `chicken`, `cow`, `dog`, `horse`, each with `16` train and `4` test images; both `split/train` and `split/test` contain all five folders. |
| AC3 | PASS | Training completed 10/10 epochs and printed `Saved model to /root/training_animal_classification/models`; files are `models/model.pt` (9,144,523 bytes) and `models/classes.json` (53 bytes). |
| AC4 | PASS | Evaluation printed `Overall accuracy: 0.9500`, per-animal accuracy, and the labeled confusion matrix; `metrics/metrics.json` exists with the same results. |
| AC5 | PASS | Evaluation printed `QUALITY GATE PASSED: 0.9500 >= 0.7000`. |
| AC6 | PASS | `python src/predict.py ...cat_demo.jpg` printed `Prediction: cat  (confidence 85%)`. |
| AC7 | PASS | `make_request.py` wrote `sample-request.json` (23,661 bytes); `score.init()` and `score.run()` printed `Prediction: cat confidence=0.8548` and returned `animal`, `confidence`, and `all_scores`. |
| AC8 | PASS | Search of `src`, `tests`, and `app` found no animal-name literals or fixed five-class logic. Classes are read from folder names/ImageFolder and saved in `classes.json`. |
| AC9 | PASS | `README.md` documents installation, tests, prep, training options, evaluation/quality gate, prediction, scoring, and Streamlit deployment configuration. |

Additional checks:

- `python -m compileall -q src tests app`: PASS.
- Root dependency installation completed successfully, including CPU
  `torchvision`.
- The scoring shell wrapper reported a non-zero status because its nested
  terminal session did not report completion, but its captured output showed
  the expected result and assertion inputs. The direct prediction and score
  response were both correct.