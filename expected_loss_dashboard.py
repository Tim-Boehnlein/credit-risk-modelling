import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Titel
st.title("Credit Risk Dashboard: PD, LGD, EAD & Expected Loss")

# --- 1. PD-Modell laden ---
with open('pd_model.pkl', 'rb') as file:
    modell_pickle = pickle.load(file)

params = np.ravel(modell_pickle['params'])
columns = modell_pickle['columns']

# --- 2. Benutzereingaben ---
st.sidebar.header("Kundendaten eingeben")

umsatz = st.sidebar.slider("Umsatz (€)", 100_000, 1_000_000, 500_000, step=10_000)
sicherheiten = st.sidebar.selectbox("Sicherheiten vorhanden?", ["Ja", "Nein"])
branche_bau = st.sidebar.selectbox("Branche: Bau?", ["Ja", "Nein"])
kreditlinie = st.sidebar.slider("Kreditlinie (€)", 50_000, 500_000, 200_000, step=10_000)

# LGD und EAD ergänzen
lgd = st.sidebar.slider("LGD (Verlustquote bei Ausfall in %)", 0, 100, 45) / 100  # z.B. 45%
ead = st.sidebar.slider("EAD (Exposure at Default in €)", 10_000, 1_000_000, 300_000, step=10_000)

# --- 3. Datentransformation ---
sicherheiten_val = 1 if sicherheiten == "Ja" else 0
branche_bau_val = 1 if branche_bau == "Ja" else 0

eingabe_df = pd.DataFrame([{
    'Umsatz': umsatz,
    'Sicherheiten': sicherheiten_val,
    'Branche_Bau': branche_bau_val,
    'Kreditlinie': kreditlinie
}])
eingabe_df['const'] = 1
eingabe_df = eingabe_df[columns]

# --- 4. PD berechnen ---
try:
    lineare_komb = np.dot(eingabe_df, params)
    pd_wahrscheinlichkeit = 1 / (1 + np.exp(-lineare_komb))
    pd_wert = float(pd_wahrscheinlichkeit[0])

    # --- 5. Expected Loss berechnen ---
    expected_loss = pd_wert * lgd * ead

    # --- 6. Ergebnisse anzeigen ---
    st.subheader("Ergebnisse")
    st.metric("Ausfallwahrscheinlichkeit (PD)", f"{pd_wert:.2%}")
    st.metric("Verlustquote bei Ausfall (LGD)", f"{lgd:.0%}")
    st.metric("Exposure at Default (EAD)", f"{ead:,.0f} €")
    st.metric("Expected Loss (EL)", f"{expected_loss:,.0f} €")

except Exception as e:
    st.error(f"Fehler bei der Berechnung: {e}")
