# Konstantprodukt-Markt und LVR – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-lvr-pool-demo.streamlit.app/)** -->

> **Modell, keine Marktdaten, keine Anlageberatung.** Alle Pfade und Zahlen sind synthetisch; die Demo erklärt einen Mechanismus und gibt keine Empfehlung für Käufe, Verkäufe oder Anlagen.

Sechstes und letztes Kernstück der Methodenreihe **Marktmechanismen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Ein Mechanismus macht aus Geboten, Budgets oder Überzeugungen Preise und eine Zuteilung; die Reihe prüft ihn an Effizienz, Anreizen, Budgetausgleich und Erlös und rechnet ihn, wo es geht, als Programm. Nach dem [Fisher-Markt](https://github.com/sebastian-hanisch/fisher-markt-demo), der [Positionsauktion](https://github.com/sebastian-hanisch/positionsauktion-demo), dem [Pacing-Gleichgewicht](https://github.com/sebastian-hanisch/pacing-gleichgewicht-demo), der [Doppelauktion](https://github.com/sebastian-hanisch/doppelauktion-demo) und dem [Prognosemarkt mit Marktmacher](https://github.com/sebastian-hanisch/prognosemarkt-demo) steht hier ein Marktmacher ohne Orderbuch: ein **Pool**, der nach der Regel $x \cdot y = k$ jederzeit handelt. Er ist ein Kostenfunktionsmarkt wie die LMSR aus dem Vorgänger (die Brücke wird am Beispiel gerechnet), und er zahlt für den Preis, den er nicht kennt: den **LVR**.

## Warum dieses Problem

Ein Pool braucht keine Gegenseite, aber er kennt den Preis **außerhalb** nicht. Bewegt sich der, holen Arbitrageure die Differenz ab; das, was der Pool dabei gegenüber dem **Rebalancieren** (denselben Bestand laufend zum Marktpreis nachziehen) verliert, heißt **Loss-versus-Rebalancing (LVR)** und ist die laufende Last der Liquiditätsgeber. Die Demo rechnet nach, **wie viel** das ist (exakt, auch für diskrete Schritte), wie es sich vom **Impermanent Loss** (Pool gegen Halten) unterscheidet und was **Gebühren** daran ändern.

## Modell

Reserven $(x, y)$, $xy = k$, Preis $P = y/x$. Wer $\Delta x$ kauft, zahlt $\Delta y = y\,\Delta x/(x - \Delta x)$; mit Gebühr $f$ gehen nur $1 - f$ des Eingangs in die Invariante ein, die Gebühr bleibt im Pool und lässt $k$ wachsen. Kauft man den Anteil $s$ der Reserve, ist der mittlere Preis das $1/(1-s)$-fache und der Preis danach das $1/(1-s)^2$-fache des Anfangspreises.

**Brücke zur Kostenfunktion.** Zwei Ergebnis-Anteile $A$, $B$ (zahlen 1 bei Eintritt), ein vollständiger Satz lässt sich für 1 prägen. Ein Pool mit Reserven $(a, b)$, $ab = r^2$, hat den Preis $p_A = b/(a+b)$; wer $q$ will, zahlt $m = C(q)$ mit $(r + m - q_A)(r + m - q_B) = r^2$. Das ist eine Kostenfunktion: Preise $= \nabla C$, Summe 1, pfadunabhängige Kasse. Gerechnet wird nur dieses Beispiel, nicht der allgemeine Satz von Frongillo, Papireddygari und Waggoner.

**LVR.** Der Referenzpreis folgt $dP = P(\mu\,dt + \sigma\,dW)$; Arbitrageure bringen den Pool nach jedem von $n$ Schritten im Jahr auf ihn. Poolwert bei Preis $P$: $V(P) = 2\sqrt{kP}$. **LVR** ist die Summe ihrer Gewinne, $\sum_i [x_i P_{i+1} + y_i - V(P_{i+1})] \ge 0$; die Rate ist $\sigma^2/8$ des Poolwerts. Erwartet: $E[\mathrm{LVR}] = V_0 \sum_{i<n} g^i (a - g)$ mit $g = e^{(\mu/2 - \sigma^2/8)\,dt}$ und $a = (e^{\mu\,dt}+1)/2$ (eigene Herleitung, siehe 📐-Expander der App); bei $\mu = 0$ ist das $V_0(1 - e^{-\sigma^2 T/8})$. Alle Märkte und Pfade sind synthetisch (Start: Pool 1000 / 1000, Preis 1, Wert $V_0 = 2000$; Horizont ein Jahr).

## Methodik

- **Exakte Erwartungswerte neben der Simulation:** für den LVR (jede Schrittzahl, jede Drift), den Poolwert und den Impermanent Loss gibt es geschlossene Formeln; die Simulation (vektorisiert über die Pfade, feste Seeds) steht mit Standardfehler daneben.
- **Arbitrage in geschlossener Form:** Die Arbitrage-Menge (mit und ohne Gebühr) ist die Lösung einer quadratischen Gleichung; die Tests rechnen sie zusätzlich durch Bisektion über die Tauschfunktionen.
- **Gleiche Pfade für alle Gebühren:** Der Zufallszug hängt nur vom Seed ab, nicht von der Gebühr.
- **Ausgleichsvolumen** ist eine Rechengröße (Arbitrage-Gewinn / $V_0$ geteilt durch die Gebühr), keine Simulation von Noise-Tradern.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind deterministisch (feste Seeds, nur `exp`, `log`, `sqrt` und Summen) und stehen in `tests/test_claims.py`.

**Tabelle 1: Vier Handbeispiele** (jeder Wert von Hand nachrechenbar; A: $\Delta y = 1000 \cdot 100/900$; B: Reserven $(100\sqrt{0{,}2/0{,}8};\ 100\sqrt{0{,}8/0{,}2}) = (50;\ 200)$, 100 Sätze geprägt, $100 + 100 - 50 = 150$ Anteile A; C: Poolwert $2\sqrt{10^6 \cdot 4} = 4000$ gegen Halten $4 \cdot 1000 + 1000 = 5000$; D: $0{,}05^2/8 = 3{,}125 \cdot 10^{-4}$).

| Beispiel | Aufgabe | Ergebnis |
|---|---|---|
| A: Pool 1000 / 1000, 100 Einheiten kaufen, ohne Gebühr | Zahlung 111,11, mittlerer Preis 1,1111, Preis danach 1,2346 | mit Gebühr 0,3 %: Zahlung 111,45 |
| B: Konstantprodukt als Prognosemarkt, Reserve 100, Preis auf 0,8 | 150,00 Anteile A für 100,00; Reserven (50,00; 200,00) | Verlust bei A 50,00; LMSR (b = 100): 138,63 Anteile für 91,63 |
| C: ein Schritt von P = 1 auf P = 4, Pool 1000 / 1000 | Reserven (500; 2000), Poolwert 4000, Halten 5000 | LVR 1000, Impermanent Loss -20,0 % |
| D: Tagesvolatilität 5 % | LVR-Rate σ²/8 | 3,125 Basispunkte je Tag |

**Tabelle 2: Preisauswirkung** (Pool 1000 / 1000; Kauf des Anteils $s$ der Reserve; Gebühr 0,3 %).

| Anteil s der Reserve | mittlerer Preis / Anfangspreis | Preis danach / Anfangspreis | Zahlung ohne Gebühr | Zahlung mit Gebühr 0,30 % |
|---|---|---|---|---|
| 0,1 % | 1,001001 | 1,002003 | 1,0010 | 1,0040 |
| 1,0 % | 1,010101 | 1,020304 | 10,1010 | 10,1314 |
| 10,0 % | 1,111111 | 1,234568 | 111,1111 | 111,4454 |
| 50,0 % | 2,000000 | 4,000000 | 1000,0000 | 1003,0090 |

Ohne Gebühr bleibt $k$ gleich $1\,000\,000$; wer die Hälfte der Reserve kauft, zahlt im Mittel das Doppelte und lässt den Preis auf das Vierfache steigen. **Mit Gebühr ist der Pfad nicht mehr gleichgültig:** Dasselbe Ziel (10 % der Reserve) kostet in Stücken mehr.

| Stücke | Zahlung ohne Gebühr | Zahlung mit Gebühr 0,30 % |
|---|---|---|
| 1 | 111,11111 | 111,44545 |
| 2 | 111,11111 | 111,45427 |
| 10 | 111,11111 | 111,46157 |
| 100 | 111,11111 | 111,46324 |

Und **hin und zurück** (kaufen, sofort alles zurückverkaufen) ist ohne Gebühr gratis:

| Anteil s der Reserve | Verlust beim Hin und zurück ohne Gebühr | mit Gebühr 0,30 % |
|---|---|---|
| 1 % | 0,0000 % | 0,5931 % |
| 10 % | 0,0000 % | 0,5394 % |
| 50 % | 0,0000 % | 0,3000 % |

**Tabelle 3: Konstantprodukt gegen LMSR als Prognosemarkt** (Anfangsreserve $r = 100$, Start $p_A = 0{,}5$; LMSR mit $b = r$, die am Start gleich reagiert: $1/(4r)$ je Anteil).

| Preis p | Konstantprodukt: Anteile / Kosten | LMSR (b = r): Anteile / Kosten | Verlust Konstantprodukt (in r) | Verlust LMSR (in r) |
|---|---|---|---|---|
| 0,60 | 40,825 / 22,474 | 40,547 / 22,314 | 0,183503 | 0,182322 |
| 0,80 | 150,000 / 100,000 | 138,629 / 91,629 | 0,500000 | 0,470004 |
| 0,90 | 266,667 / 200,000 | 219,722 / 160,944 | 0,666667 | 0,587787 |
| 0,99 | 984,937 / 894,987 | 459,512 / 391,202 | 0,899496 | 0,683097 |

Der Konstantprodukt-Markt ist **an den Rändern viel tiefer** (bis zum Preis 0,99 braucht man 984,937 Anteile statt 459,512), dafür ist seine **Verlustgrenze $r$**, die er nie erreicht, größer als die der LMSR, $r \ln 2 = 0{,}6931\,r$. Kasse, Preise und Summe 1 stimmen mit einer expliziten Kette „Sätze prägen, dann tauschen“ überein.

**Tabelle 4: LVR nach der Zahl der Schritte** ($\sigma = 0{,}8$, $\mu = 0$, ein Jahr; 5 000 Pfade je Zeile, Seed 3, ohne Gebühr).

| Schritte im Jahr | E[LVR] / V₀ (exakt) | E[LVR] / V₀ (Simulation) |
|---|---|---|
| 12 | 7,688 % | 7,669 % ± 0,056 % |
| 52 | 7,688 % | 7,693 % ± 0,035 % |
| 365 | 7,688 % | 7,688 % ± 0,027 % |
| 2000 | 7,688 % | 7,654 % ± 0,025 % |

Bei Drift 0 ist der erwartete LVR $V_0(1 - e^{-\sigma^2 T/8}) = 7{,}688\ \%$ des Anfangswerts, **unabhängig von der Zahl der Schritte**; die Simulation liegt höchstens 1,4 Standardfehler daneben. **Kein simulierter Pfad hat negativen LVR.**

**Tabelle 5: Impermanent Loss gegen LVR** (Drift $\mu = m\,\sigma^2$ mit $m = -\tfrac12, 0, \tfrac12, 1$, $\sigma = 0{,}8$, 100 Schritte, 5 000 Pfade, Seed 2, ohne Gebühr; „exakt“: Formel für diskrete Schritte beziehungsweise $E[V_T] - E[H_T]$).

| Drift μ | E[LVR] / V₀ (exakt) | E[LVR] / V₀ (Simulation) | Impermanent Loss / V₀ (exakt) | Impermanent Loss / V₀ (Simulation) |
|---|---|---|---|---|
| -0,32 | 7,12 % | 7,12 % ± 0,03 % | -7,64 % | -7,63 % ± 0,14 % |
| 0,00 | 7,69 % | 7,69 % ± 0,03 % | -7,69 % | -7,78 % ± 0,22 % |
| 0,32 | 8,35 % | 8,36 % ± 0,03 % | -10,53 % | -10,80 % ± 0,37 % |
| 0,64 | 9,11 % | 9,13 % ± 0,04 % | -17,70 % | -18,23 % ± 0,61 % |

**Nur ohne Drift sind beide im Betrag gleich** (7,69 %); mit Drift wächst der Impermanent Loss (−17,70 % bei $\mu = 0{,}64$), der LVR kaum (9,11 %): Der LVR ist die laufende Last der Liquiditätsgeber, der Impermanent Loss ein Vergleich mit dem Halten. Alle acht Simulationswerte der LVR- und Impermanent-Loss-Spalten liegen höchstens einen Standardfehler vom exakten Wert.

**Tabelle 6: Gebühren** ($\sigma = 0{,}8$, $\mu = 0$, 365 Schritte, 5 000 Pfade, Seed 3, dieselben Pfade für alle Gebühren; „Arbitrage-Gewinn“ ist, was der Pool gegenüber dem Rebalancieren verliert, einschließlich der Gebühren, die im Pool bleiben; Ausgleichsvolumen = (Arbitrage-Gewinn / $V_0$) / Gebühr).

| Gebühr | Arbitrage-Gewinn / V₀ | Anteil des gebührenfreien | Gebühren / V₀ | Schritte mit Handel | Ausgleichsvolumen / V₀ |
|---|---|---|---|---|---|
| 0,00 % | 7,69 % | 100,0 % | 0,00 % | 100,0 % | – |
| 0,05 % | 7,55 % | 98,2 % | 0,14 % | 99,0 % | 151,0 |
| 0,30 % | 6,92 % | 90,0 % | 0,80 % | 94,3 % | 23,1 |
| 1,00 % | 5,56 % | 72,3 % | 2,20 % | 81,4 % | 5,6 |

**Mit reinem Arbitrage-Fluss bleibt der Pool in jedem simulierten Pfad hinter dem Rebalancieren zurück**, auch mit Gebühr (der kleinste Pfad verliert zwischen 2,6 % und 3,6 % von $V_0$); die Gebühren mindern den Verlust nur, bei 1 % auf 72,3 % des gebührenfreien Werts. Ausgleichen würde ihn erst der Handel von Nicht-Informierten im Wert des Ausgleichsvolumens, also bei 0,3 % Gebühr etwa das 23-Fache des Anfangswerts im Jahr. Mit feineren Schritten sinkt der Gewinn stärker: bei 2 000 Schritten im Jahr und 1 % Gebühr auf etwa die Hälfte des gebührenfreien Werts (Preset „Hohe Gebühr, feine Schritte“).

## Ehrliche Grenzen

- **Geometrische Brownsche Bewegung, Arbitrage zu festen Zeitpunkten:** Sprünge, schwankende Volatilität, Blockzeiten, Reihenfolgekonkurrenz und Gas sind nicht Teil des Modells; die Schrittzahl ist nur ein Stellvertreter.
- **Zwei Güter, volle Preisspanne:** Konzentrierte Liquidität (Uniswap v3) und mehr als zwei Güter werden nicht gerechnet.
- **Kein Fluss von Nicht-Informierten:** Das Ausgleichsvolumen ist eine Rechengröße, keine Simulation.
- **Brücke nur am Beispiel:** Der allgemeine Satz (jeder CFMM mit konkavem Potential ist ein Kostenfunktionsmarkt) wird nicht nachgerechnet.
- **Simulationen sind Stichproben:** Die Simulationswerte in Tabelle 4, 5 und 6 haben Standardfehler (Tabelle 4 und 5 angegeben); die exakten Werte nicht.
- **Literatur:** Alle Angaben beruhen auf Titel, Abstract und Trefferlisten, nicht auf Volltexten; die Formel für diskrete Schritte und die Brücke über vollständige Sätze sind eigene Herleitungen, durch Aufzählung eines Binomialbaums, numerische Integration und Simulation geprüft.
- **Modell, keine Marktdaten, keine Anlageberatung:** Die Zahlen sagen nichts über echte Pools oder Erträge.

## Verifikation

- **Kern:** die vier Handbeispiele von Hand; Tauschfunktionen von Hand; die exakte Formel für zwei Schritte von Hand; Pfad-Identität (Rebalancieren minus Pool gleich Summe der Gewinne).
- **Orakel (anderer Rechenweg):** die Handelsmenge gegen **Bisektion über die Tauschfunktion** (mit und ohne Gebühr); die **geschlossene Arbitrage-Menge gegen Bisektion**, Grenzpreis inklusive Gebühr gleich dem Referenzpreis, kein Handel im Band; die Kostenfunktion der Brücke gegen eine **explizite Kette „Sätze prägen, dann tauschen“** mit Bisektion statt quadratischer Formel, Preise gegen finite Differenzen, beide Ausgänge einzeln gegen die Grenze $r$; der **erwartete LVR gegen die Aufzählung aller $2^n$ Pfade eines Binomialbaums** mit expliziter Arbitrage Schritt für Schritt, einen Schritt gegen **numerische Integration** (`scipy.integrate.quad`), und gegen die Simulation mit vier Standardfehlern; die vektorisierte Simulation gegen **dieselben Zufallszüge Schritt für Schritt** über die Tauschfunktionen (mit und ohne Gebühr); der Impermanent Loss über die Reserven.
- **Eigenschaftstests:** Produkt bleibt ohne Gebühr und wächst mit Gebühr; Hin und zurück gratis ohne Gebühr, nie gratis mit Gebühr; Preis steigt monoton; Skalierung und Vertauschen der Güter; Arbitrage-Gewinn nichtnegativ; LVR skaliert mit dem Wert und mit $\sigma^2$; die Rate hängt nicht von der Drift ab; Impermanent Loss gleich −LVR nur ohne Drift; mehr Gebühr, weniger Arbitrage-Gewinn auf denselben Pfaden; Brücke: konvex, Verlust unter $r$, tiefer als die LMSR an jedem Preis über 1/2; Literaturtests ($\sigma^2/8$, LVR gegen Impermanent Loss, Kostenfunktionsmarkt am Beispiel).
- **Randfälle:** kleinste und größte Handelsgrößen, Gebühren, Preise, Volatilität, Drift und Schrittzahlen; ein Schritt; keine Volatilität; Preis im Gebührenband; ungültige Eingaben; Zahlen, die auf null runden, ohne Vorzeichen; Laufzeit am Worst Case.
- **Verdikte** als reine Funktionen an handgesetzten Werten; **Presets, Layout, AppTest-Rauchtest** (jeder Button, jeder Regler an Minimum und Maximum), **Footer**, **Laufzeit-Importe**.
- **Behauptungen:** `tests/test_claims.py` rechnet jede Zahl dieser Datei nach.
- **Umfang:** siehe `pytest tests/`; `tools/readme_rows.py` ist kein Test, sondern erzeugt die Tabellen, die `test_claims.py` prüft.
- **Nicht gelaufen:** Mutationstests (seit 2026-10-28 nicht mehr Teil des Ablaufs); die Aussagenprüfung durch unabhängige Prüfer steht noch aus.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Ergebnis, Kernfrage, drei Experimente, Grenzen, Formeln |
| `kpm_algorithm.py` | Kern: Pool, Brücke zur Kostenfunktion, Arbitrage, exakte Erwartungswerte, Simulation |
| `kpm_constants.py` | Regler-Grenzen, feste Annahmen, Seeds, Presets |
| `kpm_evaluation.py` | Analyse, Tabellenzeilen, Verdikte (reine Funktionen) |
| `kpm_layout.py` | Fester Kopf (Stylesheet) |
| `kpm_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button, Auswahlregler |
| `kpm_scenario.py` | Referenzpreis-Schritt, Parameterprüfung |
| `kpm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/readme_rows.py` | erzeugt die README-Tabellen über dieselben Funktionen wie die App |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur für das Orakel) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Konzentrierte Liquidität**, **mehr als zwei Güter**, andere Handelsfunktionen als das Konstantprodukt.
- **Sprünge, stochastische Volatilität, Blockzeiten, Gas, Reihenfolgekonkurrenz.**
- **Noise-Trader** und **echte Marktdaten.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12; für die Streamlit Community Cloud ist die Python-Version im Dashboard einzustellen (3.12 empfohlen).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
