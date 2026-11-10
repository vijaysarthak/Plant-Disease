"""
disease_info.py — the 38 PlantVillage classes and the ACTION GROUP of each one.

An action group = "what the farmer has to do", not just "what the disease is called".
    G0 Healthy           -> nothing to do
    G1 Fungal            -> fungicide + remove infected leaves
    G2 Oomycete          -> late blight: needs a special (anti-oomycete) fungicide, acts very fast
    G3 Bacterial         -> copper spray + hygiene (normal fungicides do not work)
    G4 Viral / systemic  -> no cure: remove plants, control the insects that spread it
    G5 Pest (mite)       -> acaricide (mite killer)

Two choices that look strange at first:
  * Late blight is not a true fungus (it is an oomycete), so it gets its own group G2.
  * Citrus greening (Huanglongbing) is caused by a bacterium, but like a virus it cannot be
    cured by spraying -> G4.

Used by: the notebook (training) and src/predictor.py (the app).
"""

# Group code -> group name. The ORDER matters: G0 is group number 0, G1 is number 1, ...
GROUP_NAMES = {
    "G0": "Healthy",
    "G1": "Fungal",
    "G2": "Oomycete (late blight)",
    "G3": "Bacterial",
    "G4": "Viral / systemic",
    "G5": "Pest (mite)",
}
GROUP_CODES = list(GROUP_NAMES)          # ["G0", "G1", "G2", "G3", "G4", "G5"]
NUM_GROUPS = len(GROUP_CODES)            # 6

# PlantVillage class name -> (what causes it, group code)
DISEASES = {
    "Apple___Apple_scab": ("Venturia inaequalis (fungus)", "G1"),
    "Apple___Black_rot": ("Botryosphaeria obtusa (fungus)", "G1"),
    "Apple___Cedar_apple_rust": ("Gymnosporangium juniperi-virginianae (fungus)", "G1"),
    "Cherry_(including_sour)___Powdery_mildew": ("Podosphaera clandestina (fungus)", "G1"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Cercospora zeae-maydis (fungus)", "G1"),
    "Corn_(maize)___Common_rust_": ("Puccinia sorghi (fungus)", "G1"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Exserohilum turcicum (fungus)", "G1"),
    "Grape___Black_rot": ("Guignardia bidwellii (fungus)", "G1"),
    "Grape___Esca_(Black_Measles)": ("Phaeomoniella / Phaeoacremonium (fungi)", "G1"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Pseudocercospora vitis (fungus)", "G1"),
    "Potato___Early_blight": ("Alternaria solani (fungus)", "G1"),
    "Squash___Powdery_mildew": ("Podosphaera xanthii (fungus)", "G1"),
    "Strawberry___Leaf_scorch": ("Diplocarpon earlianum (fungus)", "G1"),
    "Tomato___Early_blight": ("Alternaria solani (fungus)", "G1"),
    "Tomato___Leaf_Mold": ("Passalora fulva (fungus)", "G1"),
    "Tomato___Septoria_leaf_spot": ("Septoria lycopersici (fungus)", "G1"),
    "Tomato___Target_Spot": ("Corynespora cassiicola (fungus)", "G1"),
    "Potato___Late_blight": ("Phytophthora infestans (oomycete)", "G2"),
    "Tomato___Late_blight": ("Phytophthora infestans (oomycete)", "G2"),
    "Peach___Bacterial_spot": ("Xanthomonas arboricola pv. pruni (bacterium)", "G3"),
    "Pepper,_bell___Bacterial_spot": ("Xanthomonas spp. (bacterium)", "G3"),
    "Tomato___Bacterial_spot": ("Xanthomonas spp. (bacterium)", "G3"),
    "Orange___Haunglongbing_(Citrus_greening)": ("Candidatus Liberibacter (bacterium, spread by psyllids)", "G4"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Begomovirus (virus, spread by whiteflies)", "G4"),
    "Tomato___Tomato_mosaic_virus": ("Tobamovirus (virus)", "G4"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tetranychus urticae (mite)", "G5"),
}

# The 12 healthy classes all go to group G0
HEALTHY_CLASSES = [
    "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___healthy", "Grape___healthy", "Peach___healthy", "Pepper,_bell___healthy",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Strawberry___healthy",
    "Tomato___healthy",
]
for name in HEALTHY_CLASSES:
    DISEASES[name] = ("none", "G0")

# Class number 0..37 = position in this alphabetical list.
# The model's output number k always means CLASS_NAMES[k].
CLASS_NAMES = sorted(DISEASES)
NUM_CLASSES = len(CLASS_NAMES)           # 38

# CLASS_GROUPS[k] = group number (0..5) of class number k
CLASS_GROUPS = []
for name in CLASS_NAMES:
    group_code = DISEASES[name][1]
    CLASS_GROUPS.append(GROUP_CODES.index(group_code))

# PlantDoc uses different folder names for the same diseases.
# PlantDoc folder name -> the matching PlantVillage class name
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def pretty_class_name(class_name):
    """'Tomato___Late_blight'  ->  'Tomato - Late blight'  (for showing in the app)."""
    plant, disease = class_name.split("___")
    plant = plant.replace("_", " ").replace(",", "")
    disease = disease.replace("_", " ").strip()
    return f"{plant} - {disease}"


# Small self-check: runs every time this file is imported, and stops with an error if the table is broken.
assert NUM_CLASSES == 38, "there must be 38 PlantVillage classes"
for plantvillage_name in PLANTDOC_TO_PLANTVILLAGE.values():
    assert plantvillage_name in DISEASES, plantvillage_name
