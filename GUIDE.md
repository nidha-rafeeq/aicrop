# Build Guide (from zero)

You will do 3 big parts: **(A) set up your computer, (B) train the AI model on Google Colab, (C) run the website.**
Total time: about 1 day.

---

## PART A – Set up your computer (one time)

1. **Install Python 3.10 or 3.11** from python.org. On the first installer screen, tick **"Add Python to PATH"**.
   (TensorFlow may not work on very new Python versions. If `pip install tensorflow` fails, install Python 3.11.)
2. **Install VS Code** from code.visualstudio.com.
3. Copy the `crop-disease-system` folder to your Desktop and open it in VS Code (File > Open Folder).
4. Open the VS Code terminal (Terminal > New Terminal) and run:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (Mac/Linux)
   pip install -r requirements.txt
   ```
   Wait until it finishes. This installs Flask and TensorFlow.

## PART B – Train the AI model (Google Colab)

The model learns from thousands of leaf photos. Your laptop is slow for this, so use free Google Colab.

**Step 1 – Get the dataset**
1. Create a free account on kaggle.com.
2. Search for **"PlantVillage"** by *emmarex* (dataset name: `plantdisease`) and click Download. It has 15 folders for Pepper, Potato and Tomato diseases.
3. Unzip it on your computer. Upload the unzipped folder **PlantVillage** to your Google Drive (My Drive).
   Check that this path exists: `My Drive/PlantVillage/Tomato_healthy/` (the disease folders sit directly inside `PlantVillage`).
   If there is a second `PlantVillage` folder inside, use that inner path in `DATA_DIR` in the next step.

**Step 2 – Train**
1. Go to colab.research.google.com > New notebook.
2. Menu: Runtime > Change runtime type > **T4 GPU** > Save.
3. Copy everything from `train_model.py` into one cell and press Run. Allow Drive access when asked.
4. Wait about 15–30 minutes. At the end it prints validation accuracy (should be about 90%+) and downloads two files:
   - `crop_disease_model.keras`
   - `class_names.json`
5. Put **both files** inside the `model/` folder of your project.
6. Screenshot the accuracy output for your report.

## PART C – Run the website

1. In the VS Code terminal (venv active):
   ```
   python app.py
   ```
2. Open **http://127.0.0.1:5000** in your browser.
3. Upload a leaf photo (pick images from the dataset to start, then try real leaf photos).
4. You should see the disease name, confidence, treatment and prevention.

---

## What each file does

| File | Job |
|---|---|
| `train_model.py` | Trains the AI (runs in Colab) |
| `model/` | Holds the trained model and the class names |
| `treatments.json` | Treatment advice for each disease (you can edit and add more) |
| `app.py` | The server: receives the photo, predicts, finds the treatment, shows the result |
| `templates/index.html` | Upload page |
| `templates/result.html` | Result page |
| `static/style.css` | Page colours |

## Common problems

| Problem | Fix |
|---|---|
| `No module named tensorflow` | The venv is not active, or run `pip install tensorflow` again |
| `FileNotFoundError: model/crop_disease_model.keras` | Copy the downloaded model files into `model/` |
| Colab says folder not found | Fix `DATA_DIR` in `train_model.py` to match your Drive path |
| Wrong or generic advice shown | The disease name is missing in `treatments.json`; add it |
| Low confidence warning on every photo | Use a close, clear photo of one leaf on a plain background |

## Checklist for the college demo

- [ ] Website runs without errors
- [ ] Test 3 healthy and 3 diseased leaves live
- [ ] Accuracy screenshot and confusion matrix in the report
- [ ] Be ready to explain: dataset, MobileNetV2 (transfer learning), how prediction becomes advice, limitations
