# Credit Risk Modelling

Dieses Projekt beschäftigt sich mit der Modellierung und Validierung der **Probability of Default (PD)** im Rahmen des Kreditrisikomanagements.  
Es nutzt zufällig generierte Kreditdaten und wendet statistische Methoden wie **logistische Regression** und **Maximum-Likelihood-Schätzung (MLE)** an.

## Inhalt

- Datenimport und Vorverarbeitung
- Schätzung der PD mittels logistischer Regression
- Modellvalidierung mit ROC- und AUC-Analyse
- Grafische Auswertung der Ergebnisse (Confusion Matrix)
- Streamlit Dashboard zur PD-Berechnung 

## Verwendete Tools 

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- Steramlit

## Ausführen des Hauptprogramms

```bash
python 'PD modelling and validation.py'
```

## Ausführung des Dashboards

1. Virtuelle Umgebung aktivieren  
2. Modell vorbereiten (`pd_model.pkl`)  
3. Dashboard starten:

```bash
streamlit run pd_dashboard.py
```
