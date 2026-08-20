import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix


train_path = "dataset/train"
test_path = "dataset/test"

os.makedirs("model2", exist_ok=True)


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

            img = cv2.resize(img, (224, 224))  
            img = img.astype("float32") / 255.0  
            X.append(img)
            Y.append(label_map[category])  

    return np.array(X), np.array(Y)


X_train, Y_train = load_images_from_path(train_path)
X_test, Y_test = load_images_from_path(test_path)


Y_train_categorical = to_categorical(Y_train, num_classes=len(labels))
Y_test_categorical = to_categorical(Y_test, num_classes=len(labels))

print("Dataset Loaded Successfully!")
print(f"Train Data: {X_train.shape}, Labels: {Y_train_categorical.shape}")
print(f"Test Data: {X_test.shape}, Labels: {Y_test_categorical.shape}")


if os.path.exists("model2/vgg16_model.json"):
    with open("model2/vgg16_model.json", "r") as json_file:
        classifier = model_from_json(json_file.read())
    classifier.load_weights("model2/vgg16_model.h5")
    print("Loaded existing model")
else:
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    classifier = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(labels), activation='softmax')
    ])
    
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
    
    history = classifier.fit(X_train, Y_train_categorical, validation_data=(X_test, Y_test_categorical), 
                             batch_size=32, epochs=10, shuffle=True, verbose=2)
    
    classifier.save("model2/vgg16_model.h5")
    with open("model2/vgg16_model.json", "w") as json_file:
        json_file.write(classifier.to_json())
    with open("model2/vgg16_history.pckl", "wb") as f:
        pickle.dump(history.history, f)
    
    print("Model training complete!")


y_pred = np.argmax(classifier.predict(X_test), axis=1)
y_true = Y_test
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)


with open("model2/vgg16_classification_report.pckl", "wb") as f:
    pickle.dump(report, f)


metrics = ['precision', 'recall', 'f1-score']
for metric in metrics:
    values = [report[label][metric] for label in labels]
    plt.figure()
    plt.bar(labels, values, color=['blue', 'red'])
    plt.xlabel("Classes")
    plt.ylabel(metric.capitalize())
    plt.title(f"VGG16 {metric.capitalize()} per Class")
    plt.savefig(f"model2/vgg16_{metric}.png")
    plt.close()

print("Precision, Recall saved successfully!")
