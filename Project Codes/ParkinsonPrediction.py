from tkinter import messagebox, simpledialog, filedialog
from tkinter import *
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.models import model_from_json
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

main = Tk()
main.title("Parkinson Disease Detection Using CNNs")
main.geometry("1300x700")
main.config(bg='lightgray')

global filename, image_panel, text_panel
image_classifier = None
model_choice = None

def show_info():
    clear_ui()
    def toggle_section(section):
        info_text.config(state=NORMAL)
        info_text.delete(1.0, END)
        info_text.insert(END, section_titles[section] + "\n", "bold")
        info_text.insert(END, section_texts[section] + "\n")
        info_text.config(state=DISABLED)
    
    section_titles = {
        "What is Parkinson’s Disease?": "What is Parkinson’s Disease?",
        "Causes of Parkinson’s Disease": "Causes of Parkinson’s Disease",
        "Symptoms of Parkinson’s Disease": "Symptoms of Parkinson’s Disease",
        "Diagnosis of Parkinson’s Disease": "Diagnosis of Parkinson’s Disease",
        "Treatment and Management": "Treatment and Management",
        "Early Detection and AI": "Early Detection and AI",
    }
    
    section_texts = {
        "What is Parkinson’s Disease?": """Parkinson’s disease (PD) is a progressive neurological disorder that affects movement and motor control. It occurs when nerve cells in the brain’s substantia nigra deteriorate and stop producing dopamine, a chemical essential for smooth and coordinated muscle movements. The gradual loss of dopamine leads to symptoms such as tremors, stiffness, and difficulty with balance and coordination. Other non-motor symptoms include depression, sleep disturbances, and cognitive impairment.""",
        
        "Causes of Parkinson’s Disease": """The exact cause of Parkinson’s disease is unknown, but several factors may contribute to its development:
- Genetic Factors: Some cases are linked to inherited gene mutations, although they are rare.
- Environmental Triggers: Exposure to toxins, pesticides, or heavy metals may increase the risk.
- Aging: The risk of Parkinson’s increases with age, typically affecting people over 60.
- Neuroinflammation: Chronic inflammation in the brain may contribute to nerve cell damage.""",
        
        "Symptoms of Parkinson’s Disease": """Symptoms develop gradually and worsen over time. The most common include:
- Tremors: Uncontrollable shaking, usually starting in the hands or fingers.
- Bradykinesia (Slowness of Movement): Reduced ability to move quickly, making daily tasks difficult.
- Muscle Rigidity: Stiffness in the limbs and joints, causing pain and limited movement.
- Postural Instability: Impaired balance, leading to a higher risk of falls.
- Speech Changes: Speaking softly, slurring words, or hesitating before speaking.
- Cognitive and Mood Symptoms: Memory problems, anxiety, and depression.""",
        
        "Diagnosis of Parkinson’s Disease": """There is no definitive test for Parkinson’s disease. Doctors diagnose it based on:
- Medical History: Examining symptoms and family history.
- Physical Examination: Evaluating motor functions and reflexes.
- Imaging Tests: MRI, CT scans, and DaTscans help rule out other conditions.
- Response to Medication: Improvement with dopamine-related drugs may confirm diagnosis.""",
        
        "Treatment and Management": """While there is no cure, treatments can help manage symptoms:
- Medications: Dopamine-replacement drugs like Levodopa help control movement issues.
- Physical Therapy: Exercises improve balance, flexibility, and coordination.
- Deep Brain Stimulation (DBS): A surgical procedure that implants electrodes in the brain to reduce severe motor symptoms.
- Lifestyle Changes: A healthy diet, regular exercise, and stress management can slow symptom progression.
- Occupational Therapy: Helps patients adapt to daily tasks.
- Speech Therapy: Assists with communication difficulties.""",
        
        "Early Detection and AI": """AI and deep learning models are revolutionizing Parkinson’s detection. By analyzing medical images and movement patterns, AI-powered models like VGG16, ResNet50, and DeepCNN can assist doctors in early diagnosis, improving treatment outcomes. 
- Machine Learning Models: Algorithms analyze MRI scans and voice patterns to detect early signs.
- Wearable Technology: Smart devices monitor motor symptoms in real time.
- Predictive Analytics: AI can help predict disease progression, aiding in personalized treatment plans.""",
    }
    
    font1 = ('times', 14, 'bold')
    y_position = 150
    
    for section in section_texts:
        btn = Button(main, text=section, command=lambda s=section: toggle_section(s), font=font1, bg='darkblue', fg='white', width=30)
        btn.place(x=50, y=y_position)
        y_position += 50
    
    global info_text
    info_text = Text(main, font=('times', 14), wrap=WORD, height=15, width=70, bg='white', relief=SOLID, padx=10, pady=10)
    info_text.place(x=450, y=150)
    info_text.config(state=DISABLED)  

    info_text.tag_configure("bold", font=('times', 16, 'bold'))




def show_test_ui():
    clear_ui()
    
    font1 = ('times', 13, 'bold')
    Button(main, text="Choose Model", command=chooseModel, font=font1, bg='darkblue', fg='white', width=15).place(x=50, y=150)
    Button(main, text="Load Model", command=loadDLModel, font=font1, bg='darkblue', fg='white', width=15).place(x=50, y=200)
    Button(main, text="Detect Parkinson", command=imageDetection, font=font1, bg='darkblue', fg='white', width=15).place(x=50, y=250)
    Button(main, text="Performance Graph", command=graph, font=font1, bg='darkblue', fg='white', width=15).place(x=50, y=300)
    Button(main, text="Exit", command=main.quit, font=font1, bg='red', fg='white', width=15).place(x=50, y=350)
    
    global text_panel
    text_panel = Text(main, height=10, width=60, font=font1, bg='white', fg='black')
    text_panel.place(x=300, y=150)

def chooseModel():
    global model_choice
    model_choice = simpledialog.askstring("Choose Model", "Enter the model (VGG16/ResNet50/DeepCNN):")
    if model_choice not in ['VGG16', 'ResNet50', 'DeepCNN']:
        messagebox.showerror("Error", "Invalid choice! Choose VGG16, ResNet50, or DeepCNN.")
        model_choice = None
    else:
        text_panel.insert(END, f"Selected Model: {model_choice}\n")

def loadDLModel():
    global image_classifier, model_choice
    if not model_choice:
        messagebox.showerror("Error", "Please choose a model first!")
        return
    
    model_folder = {'VGG16': 'model2', 'ResNet50': 'model1', 'DeepCNN': 'model'}.get(model_choice, 'model')
    if model_choice == "ResNet50":
        model_file = f'{model_folder}/resnet_model.json'
        weights_file = f'{model_folder}/resnet_model.h5'
        accuracy = 77.8
    elif model_choice == 'DeepCNN':
        model_file = f'{model_folder}/images_model.json'
        weights_file = f'{model_folder}/images_model.h5'
        accuracy = 98.6
    else:  # VGG16
        model_file = f'{model_folder}/{model_choice.lower()}_model.json'
        weights_file = f'{model_folder}/{model_choice.lower()}_model.h5'
        accuracy = 99.31
    
    try:
        with open(model_file, "r") as json_file:
            image_classifier = model_from_json(json_file.read())
        image_classifier.load_weights(weights_file)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return
    
    text_panel.insert(END, f"{model_choice} Model Loaded Successfully!\n")
   
    print(f"{model_choice} Model Loaded Successfully! \n Accuracy: {accuracy}%")

def imageDetection():
    global image_classifier, model_choice, image_panel, prediction_label
    
    if not model_choice:
        messagebox.showerror("Error", "Please choose a model first!")
        return
    if not image_classifier:
        messagebox.showerror("Error", "Please load the model before detecting Parkinson!")
        return
    
    labels = ['Healthy', 'Parkinson']
    filename = filedialog.askopenfilename(initialdir="testImages")
    if not filename:
        return
    
    image = cv2.imread(filename)
    img = cv2.resize(image, (224, 224) if model_choice != 'DeepCNN' else (64, 64))
    img = np.asarray(img).reshape(1, img.shape[0], img.shape[1], 3) / 255
    
    preds = image_classifier.predict(img)
    predict = np.argmax(preds)
    result = labels[predict]
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (300, 225))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    
    if image_panel:
        image_panel.destroy()
    image_panel = Label(main, image=image, bg='lightgray')
    image_panel.image = image
    image_panel.place(x=700, y=150)
    
    if 'prediction_label' in globals() and prediction_label:
        prediction_label.destroy()
    
    prediction_label = Label(main, text=f"Prediction: {result}", font=("times", 14, "bold"),
                             fg="green" if result == "Healthy" else "red", bg="lightgray")
    prediction_label.place(x=800, y=380)

def graph():
    if not model_choice:
        messagebox.showerror("Error", "Please choose a model first!")
        return
    if not image_classifier:
        messagebox.showerror("Error", "Please load the model before viewing the performance graph!")
        return
    
    model_folder = {'VGG16': 'model2', 'ResNet50': 'model1','DeepCNN':'model'}.get(model_choice, 'model')
    history_file = f'{model_folder}/{model_choice.lower()}_history.pckl'
    if model_choice == "ResNet50":
        history_file = f'{model_folder}/ResNet50_history.pckl'
    
    try:
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load performance data: {str(e)}")
        return
    
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.plot(history['accuracy'], 'ro-', color='red')
    plt.plot(history['loss'], 'ro-', color='green')
    plt.legend(['Training Accuracy', 'Training Loss'], loc='upper left')
    plt.title(f'{model_choice} Performance')
    plt.show()


def clear_ui():
    for widget in main.winfo_children():
        if widget not in [compareButton, title, infoButton, testButton]:
            widget.destroy()

import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from tkinter import messagebox


def compare_models():
    model_folders = {'VGG16': 'model2', 'ResNet50': 'model1', 'DeepCNN': 'model'}
    history_files = {
        'VGG16': f"{model_folders['VGG16']}/vgg16_history.pckl",
        'ResNet50': f"{model_folders['ResNet50']}/ResNet50_history.pckl",
        'DeepCNN': f"{model_folders['DeepCNN']}/deepcnn_history.pckl"
    }

    histories = {}

    
    for model, file_path in history_files.items():
        try:
            with open(file_path, 'rb') as f:
                histories[model] = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load performance data for {model}: {str(e)}")
            return

    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    
    ax1 = axes[0, 0]
    ax1.grid(True)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
    for model, history in histories.items():
        ax1.plot(history['precision'], label=f"{model} Precision", linewidth=2)
    ax1.legend(fontsize=10)

    
    ax2 = axes[0, 1]
    ax2.grid(True)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Recall', fontsize=12)
    ax2.set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
    for model, history in histories.items():
        ax2.plot(history['recall'], label=f"{model} Recall", linewidth=2)
    ax2.legend(fontsize=10)

    
    ax3 = axes[1, 0]
    ax3.grid(True)
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    for model, history in histories.items():
        ax3.plot(history['accuracy'], label=f"{model} Accuracy", linewidth=2)
    ax3.legend(fontsize=10)

    
    ax4 = axes[1, 1]
    ax4.grid(True)
    ax4.set_xlabel('Epochs', fontsize=12)
    ax4.set_ylabel('Loss', fontsize=12)
    ax4.set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
    for model, history in histories.items():
        ax4.plot(history['loss'], label=f"{model} Loss", linewidth=2)
    ax4.legend(fontsize=10)

   
    plt.tight_layout()
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='PARKINSON DISEASE DETECTION', anchor=CENTER, justify=CENTER, bg='darkblue', fg='white', font=font, height=2, width=120)
title.place(relx=0.5, y=25, anchor=CENTER)

title.place(x=0, y=25)

font1 = ('times', 13, 'bold')
infoButton = Button(main, text="Know About Parkinson", command=show_info, font=font1, bg='darkgreen', fg='white', width=20)
infoButton.place(x=50, y=100)

testButton = Button(main, text="Test for Parkinson", command=show_test_ui, font=font1, bg='darkgreen', fg='white', width=20)
testButton.place(x=250, y=100)

compareButton = Button(main, text="Compare Models", command=compare_models, font=font1, bg='purple', fg='white', width=20)
compareButton.place(x=450, y=100)



image_panel = None
main.mainloop()
