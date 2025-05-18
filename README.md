# Expected Loss Modell (EL) – Kreditrisikoanalyse

Dieses Projekt ist ein interaktives Dashboard zur Berechnung des **Expected Loss (EL)** auf Einzelkundenbasis. Der erwartete Verlust ergibt sich aus:

$$
\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}.
$$

Dabei werden folgende Komponenten berücksichtigt:

- **PD (Probability of Default):** geschätzt mittels logistischer Regression
- **LGD (Loss Given Default):** Eingabe oder Annahme durch den Benutzer
- **EAD (Exposure at Default):** basierend auf Kreditlinie und Annahmen zur Inanspruchnahme

## Funktionen im Dashboard

- Vorhersage der **PD** mittels logistischer Regression
- Eingabemöglichkeiten für LGD und EAD
- Berechnung und Anzeige des erwarteten Verlusts (EL)
- Intuitive Benutzeroberfläche mit **Streamlit**
- Modular aufgebaut und erweiterbar

## Eingabeparameter (über Sidebar)

- **Umsatz (€)** – numerischer Wert
- **Kreditlinie (€)** – numerischer Wert
- **Sicherheiten vorhanden?** – Ja/Nein
- **Branche: Bau?** – Ja/Nein
- **LGD (%)** – Verlustquote bei Ausfall (manuelle Eingabe)
- **Konvertierungsfaktor (%)** – Anteil der Kreditlinie, der im Ausfall in Anspruch genommen wird

## Installation

1. Projekt klonen:
   ```bash
   git clone https://github.com/dein-user/el-modell.git
   cd el-model
   ```

2. Virtuelle Umgebung: 
```bash 
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. Abhängigkeiten installieren:
```bash 
   pip install -r requirements.txt
   ```

4. Dashboard starten:
```bash 
   streamlit run el_dashboard.py
   ```
## Mögliche Erweiterung

- Die Module ```lgd_model.py``` und ```ead_model.py``` schätzen den $LGD$ und die $EAD$ mittels linearer Regression und zufällig generierten Daten. Man kann diese auch noch in das interaktive Umfeld einbinden. Z.B. in dem dort csv.Dateien hochgeladen werden, welche als Trainingsdaten dienen, und zum Schluss der $EL$ ausgegeben wird.  
