import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ROOT_DIR / "weather_model.keras"
SMOKE_MODEL_PATH = OUTPUT_DIR / "weather_model_smoke.keras"
CLASS_NAMES_PATH = ROOT_DIR / "class_names.json"
IMG_SIZE = (224, 224)
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train an advanced weather image classifier.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs for the classifier head.")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="Fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5, help="Fine-tuning learning rate.")
    parser.add_argument("--quick", action="store_true", help="Run a tiny smoke training job.")
    return parser.parse_args()


def set_reproducibility():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_datasets(batch_size, quick=False):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    if quick:
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(1)

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        class_names,
    )


def save_class_names(class_names):
    class_indices = {name: index for index, name in enumerate(class_names)}
    with CLASS_NAMES_PATH.open("w") as f:
        json.dump(class_indices, f, indent=2)


def class_weights_from_dataset(dataset, class_names):
    labels = []
    for _, batch_labels in dataset.unbatch():
        labels.append(int(np.argmax(batch_labels.numpy())))

    if not labels:
        return {index: 1.0 for index in range(len(class_names))}

    present_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=np.array(labels),
    )
    class_weights = {index: 1.0 for index in range(len(class_names))}
    class_weights.update({
        int(index): float(weight)
        for index, weight in zip(present_classes, weights)
    })
    return class_weights


def build_model(num_classes, learning_rate):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="weather_class")(x)

    model = tf.keras.Model(inputs, outputs, name="weather_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return model, base_model


def callbacks(model_path):
    OUTPUT_DIR.mkdir(exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(OUTPUT_DIR / "training_log.csv"),
    ]


def fine_tune(model, base_model, train_ds, val_ds, class_weights, epochs, learning_rate, model_path):
    if epochs <= 0:
        return None

    base_model.trainable = True
    for layer in base_model.layers[:-35]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks(model_path),
    )


def merge_history(first, second):
    history = dict(first.history)
    if second:
        for key, values in second.history.items():
            history.setdefault(key, []).extend(values)
    return history


def plot_training(history, update_project_plot=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("top_3_accuracy", "val_top_3_accuracy", "Top-3 Accuracy"),
        ("loss", "val_loss", "Loss"),
    ]
    for axis, (train_key, val_key, title) in zip(axes, metrics):
        axis.plot(history.get(train_key, []), label="Train")
        axis.plot(history.get(val_key, []), label="Validation")
        axis.set_title(title)
        axis.legend()
    plt.tight_layout()
    if update_project_plot:
        plt.savefig(ROOT_DIR / "training_plot.png", dpi=150)
    plt.savefig(OUTPUT_DIR / "training_plot.png", dpi=150)
    plt.close(fig)


def evaluate_model(model, val_ds, class_names):
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    with (OUTPUT_DIR / "classification_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Weather Classifier Confusion Matrix")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return report


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.fine_tune_epochs = 0
        args.batch_size = min(args.batch_size, 8)

    set_reproducibility()
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_model_path = SMOKE_MODEL_PATH if args.quick else MODEL_PATH

    train_ds, val_ds, class_names = load_datasets(args.batch_size, args.quick)
    save_class_names(class_names)
    class_weights = class_weights_from_dataset(train_ds, class_names)

    model, base_model = build_model(len(class_names), args.learning_rate)
    first_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks(output_model_path),
    )
    second_history = fine_tune(
        model,
        base_model,
        train_ds,
        val_ds,
        class_weights,
        args.fine_tune_epochs,
        args.fine_tune_learning_rate,
        output_model_path,
    )

    model.save(output_model_path)
    history = merge_history(first_history, second_history)
    plot_training(history, update_project_plot=not args.quick)
    report = evaluate_model(model, val_ds, class_names)

    print("\nTraining complete.")
    print(f"Model saved to: {output_model_path}")
    print(f"Validation accuracy: {report['accuracy']:.4f}")
    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
