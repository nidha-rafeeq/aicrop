# AI-Based Crop Disease Prediction and Management System

**Jawaharlal College of Engineering and Technology (Autonomous), Lakkidi, Palakkad – 679301**

**Group members**
- Amritha KB (JCE24CS012)
- Fayiz K (JCE24CS041)
- Mohammed Adnan M P (JCE24CS046)
- Nidha Rafeeq (JCE24CS049)
- Sreeraj K (JCE24CS061)

---

## 1. About the project

Crop diseases cause heavy yield loss, and manual inspection is slow and error-prone. This project is a simple web app where a farmer **uploads a leaf photo** and the system:

1. **Predicts the disease** using a deep learning model (CNN with transfer learning).
2. **Gives treatment measures** (organic, chemical) and prevention tips in simple language.

## 2. Features

- Upload a leaf photo (from phone camera or gallery)
- Disease name and confidence percentage
- Treatment and prevention advice
- Low-confidence warning ("Please upload a clearer photo")

## 3. Tech stack

| Layer | Choice |
|---|---|
| Model | TensorFlow/Keras, MobileNetV2 (transfer learning) |
| Dataset | PlantVillage (Kaggle) |
| Backend | Flask |
| Frontend | HTML + CSS + Bootstrap |
| Treatment data | `treatments.json` file |

## 4. How it works

```
Leaf photo -> resize to 224x224 -> CNN model -> disease + confidence
                                                     |
                                                     v
                                   treatments.json lookup (treatment + prevention)
                                                     |
                                                     v
                                               Result page
```

## 5. Project structure

```
crop-disease-system/
├── README.md
├── requirements.txt
├── train_model.ipynb          # model training (run on Google Colab)
├── model/
│   ├── crop_disease_model.keras  # trained model
│   └── class_names.json       # list of disease names
├── treatments.json            # disease -> treatment / prevention
├── app.py                     # Flask server: upload, predict, show result
├── templates/
│   ├── index.html             # upload page
│   └── result.html            # result page
└── static/
    ├── style.css
    └── uploads/               # uploaded images
```

## 6. Build steps

**Step 1 – Train the model (Colab)**
1. Download PlantVillage from Kaggle. Use 5–10 classes (e.g. Tomato, Potato, Pepper) to keep training quick.
2. Resize images to 224x224, split into train / validation / test (80/10/10).
3. Build MobileNetV2 (ImageNet weights, frozen) + a new output layer. Train about 10 epochs.
4. Check accuracy on the test set and save the confusion matrix for the report.
5. Save `crop_disease_model.keras` and `class_names.json`.

**Step 2 – Create `treatments.json`**
```json
{
  "Tomato___Early_blight": {
    "organic": "Spray neem oil; remove infected lower leaves.",
    "chemical": "Mancozeb or chlorothalonil as per label.",
    "prevention": "Rotate crops, avoid overhead watering, keep plants spaced."
  }
}
```
Take advice from reliable sources (agricultural university leaflets) and note them in the report.

**Step 3 – Write `app.py`**
- Load the model once at start.
- Route `/` shows the upload form.
- Route `/predict` receives the image, preprocesses it, predicts, looks up the treatment, and renders `result.html`.

**Step 4 – Build the two HTML pages**
Upload page (file input with `capture` for mobile camera) and result page (disease, confidence, treatment, prevention).

**Step 5 – Test**
Try real leaf photos and wrong or blurry images. Take screenshots for the report.

## 7. Run locally

```bash
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

`requirements.txt`:
```
flask
tensorflow
pillow
numpy
```

## 8. Limitations

- Trained on PlantVillage (lab-style images), so accuracy may drop on real field photos.
- Covers only the selected crops and diseases.
- Treatment advice is guidance only; follow the product label or consult an agricultural officer.

## 9. Future enhancements

- More crops and real field images
- Weather-based risk alerts
- Malayalam / Hindi language support
- Offline mobile app (TensorFlow Lite)

## 10. Tools

Python · TensorFlow/Keras · Flask · Bootstrap · Google Colab · VS Code
