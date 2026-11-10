# 🌿 Plant Disease Advisor

**Action-Group Conformal Prediction for Safe Plant Disease Advisory under Lab-to-Field Shift**
MCA Major Project (MCA_MP_22), Presidency University, Bengaluru
Sarthak Vijay · Himanshu Kumar Singh · Arun Das · Sahil Kumar — guide: Ms Lekhya Sree

A farmer uploads a photo of a leaf. The system does **not** just say "this is disease X". It asks:
*"Can I be sure which **treatment** is needed?"*

* sure about one treatment group → it gives that group's advice
* unsure between treatments → it says **"ask an expert (KVK)"**
* the photo looks unlike anything it was trained on → **"photo not recognised"**

---

## 1. How the files are connected

```
                        ┌─────────────────────────────────┐
  data/datasets/  ───►  │ notebooks/train_model.ipynb     │  trains the network and calibrates it
  (downloaded)          └───────────────┬─────────────────┘
                                        │ SAVES
                                        ▼
                        saved_model/plant_model.onnx    (the network)
                        saved_model/model_info.json     (classes, temperature, thresholds)
                                        │ LOADED BY
                                        ▼
                        src/predictor.py  ◄──── app/streamlit_app.py   (web page, the frontend)
                                          ◄──── app/api.py             (REST API, the backend)
                                        │
                        src/database.py  ──►  database/predictions.db  (history + feedback)
```

The notebook and the app **share** the code in `src/`, so a photo is prepared and judged in exactly the same way during
testing and in the app.

## 2. Folder structure

```
plant_disease_advisor/
├── README.md                  this file
├── config.py                  ALL paths and settings in one place
├── requirements.txt           packages for the app
├── requirements-train.txt     packages for training (app + PyTorch)
│
├── notebooks/
│   └── train_model.ipynb      STEP 1: train, test and SAVE the model
│
├── src/                       shared code (used by the notebook AND the app)
│   ├── disease_info.py        38 classes → 6 treatment groups, PlantDoc name mapping
│   ├── download_data.py       downloads the 3 datasets with git
│   ├── dataset_split.py       lists photos, splits PlantVillage by LEAF, finds duplicate PlantDoc photos
│   ├── image_utils.py         photo → numbers (resize, crop, normalise)
│   ├── leaf_dataset.py        PyTorch dataset + training augmentation (training only)
│   ├── conformal.py           temperature, group scores, thresholds, prediction sets, decision
│   ├── metrics.py             coverage, advice rate, wrong-advice rate (IAR), ECE
│   ├── advice.py              reads data/advice.csv
│   ├── predictor.py           loads saved_model/ and predicts ONE photo
│   └── database.py            SQLite: predictions + feedback
│
├── app/
│   ├── streamlit_app.py       STEP 2: the web page (frontend)
│   └── api.py                 optional REST API (backend, e.g. for a mobile app)
│
├── data/
│   ├── advice.csv             what to tell the farmer for each group
│   └── datasets/              PlantVillage, PlantDoc, rice (downloaded by the notebook)
│
├── saved_model/               written by the notebook, read by the app
├── outputs/                   results.json, results.png, photo lists (for the report / paper)
├── database/                  predictions.db (created by the app)
└── tests/                     pytest tests
```

## 3. How to run it

### Step 1 — Train the model (Google Colab, free GPU, ~1–1.5 h)

1. Zip the project folder as `plant_disease_advisor.zip`.
2. Open https://colab.research.google.com → *File → Upload notebook* → `notebooks/train_model.ipynb`.
3. *Runtime → Change runtime type → T4 GPU*.
4. Upload `plant_disease_advisor.zip` in the *Files* panel on the left.
5. *Runtime → Run all*. Step 0 unzips the project, Step 3 downloads the datasets (~4 GB).
6. At the end the notebook downloads `saved_model.zip`. Unzip it into the project's `saved_model/` folder on your computer.

> First time? Set `QUICK_TEST = True` in Step 2 → the whole notebook runs in a few minutes, so you can check that
> everything works before the long run.
> No GPU at all? Set `FINETUNE_EPOCHS = 0`: only the last layer is trained (works on a CPU in about an hour).

### Step 2 — Run the app (your own computer, no GPU needed)

Open a terminal **inside the `plant_disease_advisor` folder**:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux / macOS:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The browser opens http://localhost:8501. Upload a leaf photo.

Optional REST API (same model, same database):

```bash
uvicorn app.api:app --reload
```
Then open http://localhost:8000/docs to try `/predict` in the browser.

### Tests

```bash
pytest -q
```

## 4. What the notebook does (20 steps)

| Step | What happens | Code |
|---|---|---|
| 0–2 | open the project, import libraries, settings | notebook |
| 3 | download PlantVillage (lab), PlantDoc (field), rice (unknown crop) | `src/download_data.py` |
| 4 | show the 38 classes and their treatment groups | `src/disease_info.py` |
| 5 | split PlantVillage **by leaf** into 70 % train / 10 % val / 10 % cal / 10 % test | `src/dataset_split.py` |
| 6 | PlantDoc + rice lists; remove duplicate photos with conflicting labels | `src/dataset_split.py` |
| 7 | data loaders (augmentation for training photos) | `src/leaf_dataset.py` |
| 8 | build the model **layer by layer**: backbone → dropout → classifier | notebook |
| 9–11 | train: stage 1 = only the last layer, stage 2 = whole network | notebook |
| 12–13 | logits for all photos, temperature scaling, accuracy, ECE | `src/conformal.py`, `src/metrics.py` |
| 14 | action-group conformal prediction vs Top-1 and APS (lab and field) | `src/conformal.py` |
| 15 | field re-calibration experiment (10 / 25 / 50 field photos per group) | notebook |
| 16 | final thresholds for the app + "unknown photo" energy gate | `src/conformal.py` |
| 17 | figures → `outputs/results.png` | notebook |
| 18 | **save** `saved_model/plant_model.onnx` + `model_info.json` | notebook |
| 19 | load the saved model with `src/predictor.py` exactly like the app | `src/predictor.py` |
| 20 | zip `saved_model/` for download | notebook |

## 5. The method in four formulas

| | Formula | Where |
|---|---|---|
| Temperature scaling | p = softmax(z / T), T chosen to minimise the negative log-likelihood on val photos | `conformal.fit_temperature` |
| Group score (ours) | s(x, g) = 1 − Σ<sub>y∈g</sub> p(y \| x) | `conformal.group_scores` |
| Threshold per group | q̂<sub>g</sub> = ⌈(n<sub>g</sub>+1)(1−α)⌉-th smallest score of calibration photos of group g, α = 0.10 | `conformal.calibrate_group_thresholds` |
| Decision | C(x) = {g : s(x,g) ≤ q̂<sub>g</sub>}; \|C\| = 1 → advise, \|C\| ≥ 2 → refer, else unknown | `conformal.decide` |
| Unknown photo | E(x) = −T · log Σ exp(z/T) > τ (95th percentile on field calibration photos) | `conformal.energy_score` |
| Safety metric (ours) | IAR = wrong advice / all advice given | `metrics.advisory_metrics` |

## 6. Shipped demo model

`saved_model/` already contains a model, so the app runs straight away. It was trained **by this notebook** on a
CPU with all 54,305 PlantVillage photos, but with `BACKBONE = "mobilenetv3_small_100"` and `FINETUNE_EPOCHS = 0`
(frozen backbone, 4 epochs). The executed notebook is in `outputs/train_model_example_run.ipynb`.

| Result (field = PlantDoc photos the model never saw) | Value |
|---|---|
| Lab accuracy (leaf-grouped test split) | 93.6 % |
| Field accuracy | 21.2 % (mean confidence 63.5 % → over-confident) |
| Coverage, lab thresholds → field photos | 0.39 (the 90 % promise breaks) |
| Coverage after re-calibration with 25 field photos per group | 0.93 |
| Final app thresholds, field test photos: coverage / advice rate | 0.89 / 8.3 % |
| Wrong advice among advised photos (IAR): ours vs top-1 | 21.2 % vs 44.9 % |
| Wrong advice among **all** field photos: ours vs top-1 | 1.8 % vs 44.9 % |
| Rice photos (unknown crop): referred / advised | 117 / 3 of 120 |

Honest limits of this demo model: it gives advice for only ~8 % of field photos (it refers the rest), and the energy
gate does not catch rice photos (the prediction set still refers them). The **fine-tuned EfficientNet-B0** from a Colab
run (`FINETUNE_EPOCHS = 8`) is expected to do better — replace `saved_model/` with that one.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `No saved model found in saved_model/` | run the notebook (or unzip `saved_model.zip` into `saved_model/`) |
| `OSError: [WinError 4551] ... torch_python.dll` | Windows Smart App Control blocks PyTorch. The **app does not need PyTorch** (it runs ONNX) — only training does: train on Colab. |
| Notebook hangs at the data loader on Windows | `NUM_WORKERS` is set to 0 automatically on Windows; check that you did not change it |
| `ModuleNotFoundError: No module named 'src'` | start Jupyter / Streamlit **from the project folder**, not from inside `app/` or `notebooks/` |
| Colab: "Upload plant_disease_advisor.zip first" | upload the zip in the Files panel before running Step 0 |

## 8. Before publication

* Fill the `source` column of `data/advice.csv` (marked "TO CITE") with ICAR / TNAU package-of-practices references.
* Ask a plant-pathology faculty member to check the disease → group table in `src/disease_info.py`.
* Train the fine-tuned EfficientNet-B0 on Colab and use its `outputs/results.json` in the paper.
