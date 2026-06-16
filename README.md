# Weather Vision Classifier

A computer vision data science project that classifies weather images into 11 categories:
dew, fogsmog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, and snow.

The app uses Flask for deployment and TensorFlow/Keras for model training and inference.

## What Makes It Data Science

- Image dataset organized by class labels
- Deep learning model for multi-class classification
- Transfer learning with MobileNetV2
- Data augmentation
- Class weighting for imbalanced classes
- Validation accuracy and top-3 accuracy tracking
- Classification report and confusion matrix artifacts
- Grad-CAM explainability in the web app

## Project Structure

```text
weather_project/
  app.py
  train_model.py
  class_names.json
  weather_model.h5
  templates/
    index.html
  dataset/
    dew/
    fogsmog/
    ...
```

After retraining, the project also creates:

```text
artifacts/
  classification_report.json
  confusion_matrix.png
  training_log.csv
  training_plot.png
weather_model.keras
```

## Recommended Environment

Use Python 3.11 or 3.12 for best TensorFlow compatibility. If TensorFlow does not install or training stops early on Python 3.13, create a fresh Python 3.11 environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The Current App

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

## Retrain With The Advanced Model

For a quick smoke test:

```bash
python train_model.py --quick
```

For real training:

```bash
python train_model.py --epochs 25 --fine-tune-epochs 10
```

The retrained model is saved as:

```text
weather_model.keras
```

The Flask app automatically uses `weather_model.keras` when it exists. Until then, it falls back to the older `weather_model.h5`.

## Live Deployment

For Render or Railway:

```text
Build command: pip install -r requirements.txt
Start command: gunicorn app:app
```

Keep the model file in the project or store it externally if the hosting platform has file size limits.

## Accuracy Upgrade Notes

The original custom CNN can underperform because it learns image features from scratch. The upgraded training script uses MobileNetV2 pretrained on ImageNet, which usually improves accuracy for small or medium image datasets.

If accuracy is still low:

- Add more images to weak classes like `rainbow` and `lightning`.
- Check mislabeled or duplicate images.
- Train longer after confirming validation accuracy is still improving.
- Use a separate test set for a fair final score.
- Try EfficientNetB0 if MobileNetV2 is not accurate enough.
