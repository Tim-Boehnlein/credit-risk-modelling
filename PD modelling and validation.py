# pd_model_example.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 1. Datengenerierung (synthetisch)
np.random.seed(42)
n = 1000
umsatz = np.random.normal(500000, 150000, n)
sicherheiten = np.random.binomial(1, 0.6, n)
branche_bau = np.random.binomial(1, 0.2, n)
kreditlinie = np.random.normal(200000, 50000, n)

# PD beeinflusst durch geringe Sicherheiten, hohe Kreditlinie, Branche Bau
linear_comb = -3 + 0.000002*umsatz - 1.5*sicherheiten + 1.2*branche_bau + 0.00001*kreditlinie
p_default = 1 / (1 + np.exp(-linear_comb))

default = np.random.binomial(1, p_default)

# DataFrame erstellen
df = pd.DataFrame({
    'Umsatz': umsatz,
    'Sicherheiten': sicherheiten,
    'Branche_Bau': branche_bau,
    'Kreditlinie': kreditlinie,
    'Ausfall': default
})

# 2. Logistische Regression
X = df[['Umsatz', 'Sicherheiten', 'Branche_Bau', 'Kreditlinie']]
X = sm.add_constant(X)
y = df['Ausfall']

modell = sm.Logit(y, X)
ergebnis = modell.fit()
print(ergebnis.summary())

# 3. ROC-Kurve und AUC
wahrscheinlichkeiten = ergebnis.predict(X)
fpr, tpr, schwellen = roc_curve(y, wahrscheinlichkeiten)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Zufall')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-Kurve für PD-Modell')
plt.legend()
plt.tight_layout()
plt.show()

# 4. Confusion Matrix bei Cutoff 0.5
cutoff = 0.5
vorhersagen = (wahrscheinlichkeiten >= cutoff).astype(int)
cm = confusion_matrix(y, vorhersagen)
ConfusionMatrixDisplay(cm).plot()
plt.title(f'Confusion Matrix bei Cutoff {cutoff}')
plt.show()