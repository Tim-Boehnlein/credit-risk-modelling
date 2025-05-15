# Importieren der benötigten Bibliotheken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle

# 1. Datengenerierung – in echt beobachtete EAD-Werte
np.random.seed(42)
n = 1000
kreditlinie = np.random.normal(300000, 70000, n)           # Maximal nutzbare Kreditlinie
nutzungshistorie = np.random.normal(0.6, 0.1, n)           # Durchschnittliche Nutzung (60 %)
branche_kapitalintensiv = np.random.binomial(1, 0.4, n)    # Kapitalintensive Branche

# EAD = Anteil der Kreditlinie, der bei Ausfall offen ist (zwischen 0 und kreditlinie)
ead_quote = 0.5 + 0.2 * nutzungshistorie + 0.1 * branche_kapitalintensiv
ead_quote = np.clip(ead_quote + np.random.normal(0, 0.05, n), 0, 1)
ead = kreditlinie * ead_quote

# DataFrame erstellen
df = pd.DataFrame({
    'Kreditlinie': kreditlinie,
    'Nutzungshistorie': nutzungshistorie,
    'Branche_Kapitalintensiv': branche_kapitalintensiv,
    'EAD': ead
})

# 2. Daten fitten mittels linearer Regression
X = df[['Kreditlinie', 'Nutzungshistorie', 'Branche_Kapitalintensiv']]
X = sm.add_constant(X)  # Konstante hinzufügen, um den Achsenabschnitt zu berücksichtigen    
y = df['EAD']

modell = sm.OLS(y, X)
ergebnis = modell.fit()
print(ergebnis.summary())

# 3. Plot: Tatsächliche vs. vorhergesagte EAD
vorhersage = ergebnis.predict(X)

plt.figure(figsize=(6, 4))
plt.scatter(y, vorhersage, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Tatsächliche EAD')
plt.ylabel('Vorhergesagte EAD')
plt.title('EAD: Modellgüte')
plt.tight_layout()
plt.show()

# 4. Modell speichern
ead_model = {
    'params': ergebnis.params,
    'columns': X.columns
}

with open('ead_model.pkl', 'wb') as file:
    pickle.dump(ead_model, file)

print("EAD-Modell wurde gespeichert als ead_model.pkl")
