"""
predictor.py — loads the SAVED model and answers "what is wrong with this leaf, and what should I do?"

This is the connection between training and the app:

    notebook  --saves-->  saved_model/plant_model.onnx + saved_model/model_info.json
                                      |
    app       --uses-->   PlantDiseasePredictor  (this file)

Steps for one photo:
  1. prepare_image()        photo -> numbers                    (src/image_utils.py)
  2. ONNX model             numbers -> 38 logits
  3. energy_score()         is the photo familiar at all?       (src/conformal.py)
  4. softmax + group scores probabilities of the 6 groups
  5. prediction_sets()      which groups pass their threshold?
  6. decide()               advise / refer / unknown
  7. advice text            from data/advice.csv                (src/advice.py)

Needs only numpy, Pillow and onnxruntime (no PyTorch).
"""
import json

import numpy as np
import onnxruntime

from config import MODEL_FILE, MODEL_INFO_FILE
from src.advice import load_advice
from src.conformal import softmax, group_probabilities, prediction_sets, energy_score, decide
from src.disease_info import DISEASES, GROUP_CODES, GROUP_NAMES, NUM_GROUPS, pretty_class_name
from src.image_utils import prepare_image


class PlantDiseasePredictor:

    def __init__(self, model_file=MODEL_FILE, model_info_file=MODEL_INFO_FILE):
        if not model_file.exists() or not model_info_file.exists():
            raise FileNotFoundError(
                f"No saved model found in {model_file.parent}. "
                "Run notebooks/train_model.ipynb first: its last steps save the model there.")

        # everything the notebook learned besides the network weights
        with open(model_info_file, encoding="utf-8") as file:
            self.info = json.load(file)
        self.class_names = self.info["class_names"]
        self.class_groups = np.array(self.info["class_groups"])
        self.temperature = self.info["temperature"]
        self.thresholds = np.array(self.info["group_thresholds"])
        self.energy_threshold = self.info["energy_threshold"]

        # the network itself (ONNX format, runs on CPU)
        self.session = onnxruntime.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.advice = load_advice()

    def predict(self, image):
        """image: a PIL image. Returns a dictionary with everything the app shows."""
        # 1-2. photo -> logits
        pixels = prepare_image(image)[np.newaxis]               # add a "batch" dimension: (1, 3, 224, 224)
        logits = self.session.run(None, {self.input_name: pixels})[0]   # shape (1, 38)

        # 3. familiar photo?
        energy = float(energy_score(logits, self.temperature)[0])
        is_unfamiliar = energy > self.energy_threshold

        # 4. probabilities of classes and groups
        probs = softmax(logits, self.temperature)                # shape (1, 38)
        group_probs = group_probabilities(probs, self.class_groups, NUM_GROUPS)[0]

        # 5. prediction set = groups whose score passes their threshold
        in_set = prediction_sets(probs, self.class_groups, NUM_GROUPS, self.thresholds)[0]
        groups_in_set = [g for g in range(NUM_GROUPS) if in_set[g]]

        # 6. decision
        decision, advised_group = decide(groups_in_set, is_unfamiliar)

        # 7. advice text
        if decision == "advise":
            advice = self.advice[GROUP_CODES[advised_group]]
        elif decision == "refer":
            advice = self.advice["REFER"]
        else:
            advice = self.advice["UNKNOWN"]

        # the 3 most likely diseases (for information only — the DECISION uses the groups)
        top_classes = []
        for k in np.argsort(-probs[0])[:3]:
            class_name = self.class_names[k]
            top_classes.append({
                "class_name": class_name,
                "pretty_name": pretty_class_name(class_name),
                "probability": float(probs[0][k]),
                "cause": DISEASES[class_name][0],
                "group": GROUP_NAMES[GROUP_CODES[self.class_groups[k]]],
            })

        return {
            "decision": decision,                                         # "advise" / "refer" / "unknown"
            "advised_group": None if advised_group is None else GROUP_NAMES[GROUP_CODES[advised_group]],
            "prediction_set": [GROUP_NAMES[GROUP_CODES[g]] for g in groups_in_set],
            "group_probabilities": {GROUP_NAMES[GROUP_CODES[g]]: float(group_probs[g]) for g in range(NUM_GROUPS)},
            "top_classes": top_classes,
            "energy": energy,
            "is_unfamiliar": bool(is_unfamiliar),
            "advice": advice,
        }
