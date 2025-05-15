# Importieren der benötigten Bibliotheken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle

# 1. Datengenerierung für LGD – in der Realität historische Workout-Daten
np.random.seed(42)
n = 1000
kredithöhe = np.random.normal(200000, 40000, n)       # Ursprüngliche Kredithöhe
sicherheitenwert = np.random.normal(100000, 30000, n) # Wert der Sicherheiten
branche_risiko = np.random.binomial(1, 0.3, n)         # Risikoreiche Branche (z. B. Bau)

# Simulierter LGD-Wert: Anteil des Kredits, der bei Ausfall verloren geht (zwischen 0 und 1)
linear_lgd = 0.4 + 0.000002 * kredithöhe - 0.000003 * sicherheitenwert + 0.1 * branche_risiko
lgd = np.clip(linear_lgd + np.random.normal(0, 0.05, n), 0, 1)  # LGD zwischen 0 und 1 begrenzen

# DataFrame erstellen
df = pd.DataFrame({
    'Kredithöhe': kredithöhe,
    'Sicherheitenwert': sicherheitenwert,
    'Branche_Risiko': branche_risiko,
    'LGD': lgd
})

# 2. Daten fitten mittels linearer Regression
X = df[['Kredithöhe', 'Sicherheitenwert', 'Branche_Risiko']]
X = sm.add_constant(X) # Konstante hinzufügen, um den Achsenabschnitt zu berücksichtigen    
y = df['LGD']

modell = sm.OLS(y, X)
ergebnis = modell.fit()
print(ergebnis.summary())

# 3. Plot: Tatsächliche vs. vorhergesagte LGD
vorhersage = ergebnis.predict(X)

plt.figure(figsize=(6, 4))
plt.scatter(y, vorhersage, alpha=0.3)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Tatsächliche LGD')
plt.ylabel('Vorhergesagte LGD')
plt.title('LGD: Modellgüte')
plt.tight_layout()
plt.show()

# 4. Modell speichern
lgd_model = {
    'params': ergebnis.params,
    'columns': X.columns
}

with open('lgd_model.pkl', 'wb') as file:
    pickle.dump(lgd_model, file)

print("LGD-Modell wurde gespeichert als lgd_model.pkl")
