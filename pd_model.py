# Imortieren der benötigten Bibliotheken    

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 1. Datengenerierung per Zuffal, in echt historische Daten 
np.random.seed(42)                               # Reproduzierbarkeit 
n = 1000                                         # Anzahl der Beobachtungen 
umsatz = np.random.normal(500000, 150000, n)     # Umsatz in Euro mit Normalverteilung  
sicherheiten = np.random.binomial(1, 0.6, n)     # Sicherheiten 0 oder 1 mit Binomialverteilung
branche_bau = np.random.binomial(1, 0.2, n)      # Branche Bau 0 oder 1 mit Binomialverteilung
kreditlinie = np.random.normal(200000, 50000, n) # Kreditlinie in Euro mit Normalverteilung 

# PD beeinflusst durch geringe Sicherheiten, hohe Kreditlinie, Branche Bau
linear_comb = -3 + 0.000002*umsatz - 1.5*sicherheiten + 1.2*branche_bau + 0.00001*kreditlinie # Summe aus Einflussfaktoren mal Daten 
p_default = 1 / (1 + np.exp(-linear_comb)) # Ausfallwahrscheinlichkeit mit logistischem Modell, sinnvoll für binärische Daten (Ausfall, kein Ausfall)

default = np.random.binomial(1, p_default) # Ausfall 0 oder 1 mit Binomialverteilung    

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
X = sm.add_constant(X)  # Konstante hinzufügen   
y = df['Ausfall']       # Zielvariable    

modell = sm.Logit(y, X)     # Logistische Regression    
ergebnis = modell.fit()     # Modell anpassen   
print(ergebnis.summary())   # Zusammenfassung des Modells

# 3. ROC-Kurve (Receiver Operating Characteristic Curve) und AUC (Area Under the Curve)
# ROC-Kurve zeigt die Sensitivität (True Positive Rate TPR = TP/(TP + FN)) gegen die Falsch-Positiv-Rate FPR = FP/(FP + TN)
# Die ROC-Kurve ensteht durch Variation des Schwellenwerts für die Klassifikation
# AUC ist die Fläche unter der ROC-Kurve, ein Maß für die Trennschärfe des Modells
# Ein hoher AUC-Wert (nahe 1) zeigt eine gute Trennschärfe an, während ein Wert von 0.5 auf Zufall hinweist
# Der Wert 0.5 muss aber nicht immer erreicht werden, da die ROC-Kurve auch unterhalb der Diagonalen liegen kann
# In diesem Fall sollte man versuchen zu erkennen, woran das liegt und beispeilsweise die Trainingsdaten verbessern
wahrscheinlichkeiten = ergebnis.predict(X)                  # Vorhersage der Wahrscheinlichkeiten    
fpr, tpr, schwellen = roc_curve(y, wahrscheinlichkeiten)    # Berechnung der ROC-Kurve 
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
# Ein anderes Maß für die Sensitivität wäre der KS-Wert, der die Differenz zwischen der kumulierten Verteilung der positiven und negativen Klassen anzeigt

# 4. Confusion Matrix bei Cutoff 0.5
# Die Confusion Matrix zeigt die Anzahl der True Positives (TP), False Positives (FP), True Negatives (TN) und False Negatives (FN)
# TP: richtig als Ausfall klassifiziert
cutoff = 0.5
vorhersagen = (wahrscheinlichkeiten >= cutoff).astype(int)
cm = confusion_matrix(y, vorhersagen)
ConfusionMatrixDisplay(cm).plot()
plt.title(f'Confusion Matrix bei Cutoff {cutoff}')
plt.show()

# 5. Modell speichern mit Pickle
import pickle

# Nur das Modellobjekt speichern (koeffizientenbasiert, ohne statsmodels-Summary)
modell_pickle = {
    'params': ergebnis.params,  # Koeffizienten des Modells
    'columns': X.columns        # Spaltennamen
}

with open('pd_model.pkl', 'wb') as file:   # Öffnen der Datei im Binärmodus
    pickle.dump(modell_pickle, file)       # Speichern des Modells    

print("Modell wurde gespeichert als pd_model.pkl")
