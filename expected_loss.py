import numpy as np
import pandas as pd
import pickle

# Hilfsfunktion zur Vorhersage mit geladenem Modell
def vorhersage_mit_modell(model_dict, df):
    X = df[model_dict['columns'][1:]]  # Erste Spalte ist "const"
    X = sm.add_constant(X)
    return np.dot(X, model_dict['params'])

# 1. Beispielhafte Neudaten erzeugen (können auch aus einer Datenbank stammen)
np.random.seed(42)     # Reproduzierbarkeit
n = 1000
df_pd = pd.DataFrame({
    'Umsatz': np.random.normal(500000, 150000, n),
    'Sicherheiten': np.random.binomial(1, 0.6, n),
    'Branche_Bau': np.random.binomial(1, 0.2, n),
    'Kreditlinie': np.random.normal(200000, 50000, n)
})

df_lgd = pd.DataFrame({
    'Kredithöhe': df_pd['Kreditlinie'],
    'Sicherheitenwert': np.random.normal(100000, 30000, n),
    'Branche_Risiko': df_pd['Branche_Bau']
})

df_ead = pd.DataFrame({
    'Kreditlinie': df_pd['Kreditlinie'],
    'Nutzungshistorie': np.random.normal(0.6, 0.1, n),
    'Branche_Kapitalintensiv': df_pd['Branche_Bau']
})

# 2. Modelle laden
import statsmodels.api as sm

with open('pd_model.pkl', 'rb') as f:
    pd_model = pickle.load(f)

with open('lgd_model.pkl', 'rb') as f:
    lgd_model = pickle.load(f)

with open('ead_model.pkl', 'rb') as f:
    ead_model = pickle.load(f)

# 3. Vorhersagen berechnen
linear_pd = vorhersage_mit_modell(pd_model, df_pd)
pd_wahrscheinlichkeit = 1 / (1 + np.exp(-linear_pd))  # Logistische Umwandlung

lgd = vorhersage_mit_modell(lgd_model, df_lgd).clip(0, 1)
ead = vorhersage_mit_modell(ead_model, df_ead).clip(0)

# 4. EL berechnen
el = pd_wahrscheinlichkeit * lgd * ead

# 5. Ergebnis zusammenfassen
ergebnisse = pd.DataFrame({
    'PD': pd_wahrscheinlichkeit,
    'LGD': lgd,
    'EAD': ead,
    'Expected_Loss': el
})

print(ergebnisse.head())

# Optional: Durchschnittlicher EL
print("\nDurchschnittlicher Expected Loss:", ergebnisse['Expected_Loss'].mean())
