import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import MaxPooling2D, Dense, Flatten, Conv2D
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


train_path = "dataset/train"
test_path = "dataset/test"


os.makedirs("model", exist_ok=True)


labels = ["healthy", "parkinson"]
label_map = {label: i for i, label in enumerate(labels)}

def load_images_from_path(data_path):
    X, Y = [], []
    for category in labels:  
        folder_path = os.path.join(data_path, category)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist!")
            continue

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Couldn't read image {img_path}")
                continue

            img = cv2.resize(img, (64, 64))
            img = img.astype("float32") 
            X.append(img)
            Y.append(label_map[category]) 

    return np.array(X), np.array(Y)


X_train, Y_train = load_images_from_path(train_path)
X_test, Y_test = load_images_from_path(test_path)

np.save("model/images_X_train.npy", X_train)
np.save("model/images_Y_train.npy", Y_train)
np.save("model/images_X_test.npy", X_test)
np.save("model/images_Y_test.npy", Y_test)


Y_train = to_categorical(Y_train, num_classes=len(labels))
Y_test = to_categorical(Y_test, num_classes=len(labels))

print("Dataset Loaded Successfully!")
print(f"Train Data: {X_train.shape}, Labels: {Y_train.shape}")
print(f"Test Data: {X_test.shape}, Labels: {Y_test.shape}")


class MetricsCallback(Callback):
    def on_train_begin(self, logs=None):
        self.precision = []
        self.recall = []
        self.f1 = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred_probs = self.model.predict(X_test)  
        y_pred = np.argmax(y_pred_probs, axis=1)  
        y_true = np.argmax(Y_test, axis=1)

        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)

        print(f"Epoch {epoch+1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


if os.path.exists("model/images_model.json"):
    with open("model/images_model.json", "r") as json_file:
        classifier = model_from_json(json_file.read())

    classifier.save("model/images_model.h5")
    print("Loaded existing model")

else:
    classifier = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(len(labels), activation="softmax"),
    ])
    classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    metrics_callback = MetricsCallback()
    history = classifier.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=32,
        epochs=20,
        shuffle=True,
        verbose=2,
        callbacks=[metrics_callback]  
    )


    classifier.save("model/images_model.h5")

    with open("model/images_model.json", "w") as json_file:
        json_file.write(classifier.to_json())

    
    history_dict = history.history
    history_dict["precision"] = metrics_callback.precision
    history_dict["recall"] = metrics_callback.recall
    history_dict["f1"] = metrics_callback.f1

    with open("model/DeepCNN_history.pckl", "wb") as f:
        pickle.dump(history_dict, f)

print("Model training complete!")
os.makedirs("model", exist_ok=True)


epochs = range(1, len(history_dict["accuracy"]) + 1)


def save_plot(metric_name, values, val_values=None):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, values, label=f'Training {metric_name}', marker='o')
    
    if val_values:
        plt.plot(epochs, val_values, label=f'Validation {metric_name}', marker='s')

    plt.xlabel("Epochs")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model/{metric_name}.png")  # Save to model folder
    plt.close()



def save_comparison_plot(metric_names, metric_values, title, filename):
    plt.figure(figsize=(8, 6))

    for name, values in zip(metric_names, metric_values):
        plt.plot(epochs, values, label=name, marker='o')

    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"model/{filename}.png") 
    plt.close()


save_plot("accuracy", history_dict["accuracy"], history_dict["val_accuracy"])
save_plot("loss", history_dict["loss"], history_dict["val_loss"])
save_plot("precision", history_dict["precision"])
save_plot("recall", history_dict["recall"])
save_plot("f1", history_dict["f1"])


save_comparison_plot(
    metric_names=["Precision", "Recall", "F1-score"],
    metric_values=[history_dict["precision"], history_dict["recall"], history_dict["f1"]],
    title="Precision vs Recall vs F1-score Over Epochs",
    filename="precision_recall_f1"
)

print("Graphs saved to 'model' folder successfully!")
