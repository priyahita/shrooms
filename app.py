
import pandas as pd
import pickle
import json
import streamlit as lit

with open('shrooms_model.pkl', 'rb') as f:
    rf = pickle.load(f)

with open('shrooms_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('shrooms_features.json', 'r') as f:
    features = json.load(f)


odor_map = {
    "pungent": 6, "almond": 0, "anise": 3, "none": 5, "foul": 2,
    "creosote": 1, "spicy": 8, "fishy": 7, "musty": 4
}

gill_size_map = {"broad": 1, "narrow": 0}

spore_print_color_map = {
    "black": 2, "brown": 3, "buff": 6, "chocolate": 1, "green": 7,
    "orange": 5, "purple": 4, "white": 8, "yellow": 0
}

ring_type_map = {
    "pendant": 4, "evanescent": 0, "flaring": 2, "none": 1, "large": 3
}

gill_color_map = {
    "black": 4, "brown": 5, "gray": 2, "pink": 7, "white": 10,
    "chocolate": 3, "purple": 9, "red": 1, "orange": 0, "yellow": 8,
    "green": 11, "buff": 6
}

population_map = {
    "scattered": 3, "numerous": 2, "abundant": 0,
    "several": 4, "solitary": 5, "clustered": 1
}

bruises_map = {"bruises": 1, "no": 0}



lit.title("🍄 Mushroom Edible Detector")
lit.subheader("Select features to predict whether it’s edible or poisonous.")

odor_sel = lit.selectbox("Odor", list(odor_map.keys()))
gill_size_sel = lit.selectbox("Gill Size", list(gill_size_map.keys()))
spore_sel = lit.selectbox("Spore Print Color", list(spore_print_color_map.keys()))
ring_sel = lit.selectbox("Ring Type", list(ring_type_map.keys()))
gill_col_sel = lit.selectbox("Gill Color", list(gill_color_map.keys()))
population_sel = lit.selectbox("Population", list(population_map.keys()))
bruises_sel = lit.selectbox("Bruises", list(bruises_map.keys()))


input_data = pd.DataFrame([{
    "odor": odor_map[odor_sel],
    "gill-size": gill_size_map[gill_size_sel],
    "spore-print-color": spore_print_color_map[spore_sel],  
    "ring-type": ring_type_map[ring_sel],
    "gill-color": gill_color_map[gill_col_sel],
    "population": population_map[population_sel],
    "bruises": bruises_map[bruises_sel]
}])


# make sure same column order
input_data = input_data[features]

# Scale + Predict
input_scaled = scaler.transform(input_data)
prediction = rf.predict(input_scaled)[0]

lit.subheader("Prediction Result:")
lit.write("☠️ Poisonous" if prediction == 0 else "🍽️ Edible")
