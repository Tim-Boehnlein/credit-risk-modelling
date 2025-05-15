import streamlit as st
import pandas as pd
import numpy as np
import pickle
import statsmodels.api as sm

# Titel
st.title("PD-Modell: Ausfallwahrscheinlichkeit vorhersagen")

# Modell laden
with open('pd_model.pkl', 'rb') as file:
    modell_pickle = pickle.load(file)

params = np.ravel(modell_pickle['params'])  # Flatten params if necessary
columns = modell_pickle['columns']

# Benutzer-Eingaben über Sidebar
st.sidebar.header("Kundendaten eingeben")

umsatz = st.sidebar.slider("Umsatz (€)", 100_000, 1_000_000, 500_000, step=10_000)
sicherheiten = st.sidebar.selectbox("Sicherheiten vorhanden?", ["Ja", "Nein"])
branche_bau = st.sidebar.selectbox("Branche: Bau?", ["Ja", "Nein"])
kreditlinie = st.sidebar.slider("Kreditlinie (€)", 50_000, 500_000, 200_000, step=10_000)

# In numerische Werte umwandeln
sicherheiten_val = 1 if sicherheiten == "Ja" else 0
branche_bau_val = 1 if branche_bau == "Ja" else 0

# Daten vorbereiten
eingabe_df = pd.DataFrame([{
    'Umsatz': umsatz,
    'Sicherheiten': sicherheiten_val,
    'Branche_Bau': branche_bau_val,
    'Kreditlinie': kreditlinie
}])

# Debugging: Check shape of eingabe_df before adding the constant
# st.write("Shape of eingabe_df before adding constant:", eingabe_df.shape)

# Manually add constant column (for the intercept term)
eingabe_df['const'] = 1

# Reorder the columns to match the model's expected order (const first)
eingabe_df = eingabe_df[columns]

# Debugging: Check the shape after manually adding the constant
# st.write("Shape of eingabe_df after manually adding constant:", eingabe_df.shape)

# Now that we know the shape of the input matches the model's expected input, perform the prediction
try:
    lineare_komb = np.dot(eingabe_df, params)  # Berechnung des linearen Terms
    pd_score = 1 / (1 + np.exp(-lineare_komb))  # Sigmoid-Funktion für die Wahrscheinlichkeit
    wahrscheinlichkeit = float(pd_score[0])  # Die Wahrscheinlichkeit des Ausfalls

    # Ergebnis anzeigen
    st.subheader("Ergebnis")
    st.metric(label="Ausfallwahrscheinlichkeit (PD)", value=f"{wahrscheinlichkeit:.2%}")
except ValueError as e:
    st.error(f"Fehler bei der Berechnung: {e}")
