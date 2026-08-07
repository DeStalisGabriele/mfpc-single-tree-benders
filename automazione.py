"""
Automatizza i test della tesi: per ogni scenario (n, p, d) genera l'istanza
e lancia strong.py / weak.py / benders.py / benders_v2.py, salvando TUTTO
l'output su file di testo -- sia il log completo (grezzo, riga per riga di
ogni run) sia un riassunto compatto con solo i numeri che servono per le
tabelle 4.1/4.2 (e per il confronto originale-vs-ottimizzato).

Metti questo file nella STESSA cartella di common.py/generatore.py/strong.py/
weak.py/benders.py/benders_v2.py/maxflow_fast.py (il progetto PyCharm).

Uso:
    python automazione.py

Alla fine trovi due file nella cartella:
    - risultati_completi.txt  (tutto l'output grezzo di ogni run, per controllo)
    - risultati_riassunto.txt (tabella compatta: scenario, formulazione,
      stato, flusso ottimo, tempo, nodi -- pronta da incollare nella tesi)

MODIFICA la lista SCENARI qui sotto per far combaciare i casi con le tue
tabelle 4.1 (Scalability Test) e 4.2 (Density Test): quelli gia' presenti
sono la Tabella 4.1 (n crescente, p/d fissi); per la 4.2 aggiungi le tue
combinazioni di p/d a n fisso (sono commentate come esempio).

NOTA su benders_v2: viene lanciato senza flag aggiuntivi (configurazione
base, quella validata come miglioramento consistente su tutte le scale --
vedi BENCHMARKS.md). Se vuoi includere anche la variante che batte Weak ad
alta densita' di conflitto (--no-strong-master --cuts 0), aggiungi le sue
combinazioni di (p, d) alla lista SCENARI_V2_ALTA_DENSITA piu' in basso.
"""

import os
import re
import subprocess
import sys
import time

# ===========================================================================
# CONFIGURAZIONE: modifica qui gli scenari da testare
# ===========================================================================
SCENARI = [
    # (n, p, d, seed)  -- Tabella 4.1: Scalability Test
    (40, 0.6, 0.3, 12345),
    (60, 0.6, 0.3, 12345),
    (80, 0.6, 0.3, 12345),
    (100, 0.6, 0.3, 12345),

    # Tabella 4.2: Density Test (n fisso, p/d variabili) -- SOSTITUISCI questi
    # valori con quelli esatti della tua tesi (bassa/media/alta densita')
    # (50, 0.3, 0.1, 12345),
    # (50, 0.6, 0.3, 12345),
    # (50, 0.9, 0.5, 12345),
]

# Scenari extra SOLO per benders_v2 --no-strong-master --cuts 0 (il regime ad
# alta densita' di conflitto dove batte Weak -- vedi README.md). Lasciato
# vuoto di default: aggiungi qui le combinazioni (n, p, d, seed) che vuoi
# testare in questo regime specifico.
SCENARI_V2_ALTA_DENSITA = [
    # (60, 0.6, 0.7, 12345),
]

TIMELIMIT_STRONG = 3600   # secondi -- 1 ora, come nel paper Suvak et al. 2020
TIMELIMIT_WEAK = 3600
TIMELIMIT_BENDERS = 3600  # secondi (MAXTIME per Benders e Benders v2)

CARTELLA_ISTANZE = "run_istanze"   # sottocartella dove salvare ogni istanza generata
LOG_COMPLETO = "risultati_completi.txt"
LOG_RIASSUNTO = "risultati_riassunto.txt"

PY = sys.executable  # usa lo stesso interprete Python (con l'ambiente/venv corrente)


# ===========================================================================
# PARSING OUTPUT (regex sui writeln di strong.py/weak.py/benders.py/benders_v2.py)
# ===========================================================================
def estrai_metriche_compatto(output, formulazione):
    """Ritorna dict con stato/flusso/gap/tempo/nodi estratti dall'output
    testuale. Ritorna None per campi non trovati (es. run interrotto/crashato).
    Il gap e' FONDAMENTALE per confrontare run che vanno in TIME_LIMIT: a
    parita' di flusso trovato, gap piu' piccolo = piu' vicino a una soluzione
    dimostrata ottima."""
    metriche = {"stato": None, "flusso": None, "gap": None, "tempo": None, "nodi": None}

    if formulazione in ("strong", "weak"):
        m = re.search(r"Stato del solver:\s*(\S+)", output)
        if m:
            metriche["stato"] = m.group(1)
        m = re.search(r"Flusso Ottimo Trovato:\s*([\d.]+)", output)
        if m:
            metriche["flusso"] = float(m.group(1))
        m = re.search(r"MIP Gap:\s*([\d.]+)", output)
        if m:
            metriche["gap"] = float(m.group(1))
        m = re.search(r"Tempo di calcolo \(sec\):\s*([\d.]+)", output)
        if m:
            metriche["tempo"] = float(m.group(1))
        m = re.search(r"Nodi esplorati \(B&B Tree\):\s*([\d.]+)", output)
        if m:
            metriche["nodi"] = float(m.group(1))
    else:  # benders / benders_v2
        m = re.search(r"Status Gurobi:\s*(\S+)", output)
        if m:
            metriche["stato"] = m.group(1)
        m = re.search(r"Flusso Ottimo v\*:\s*([\d.]+)", output)
        if m:
            metriche["flusso"] = float(m.group(1))
        m = re.search(r"MIP Gap:\s*([\d.]+)", output)
        if m:
            metriche["gap"] = float(m.group(1))
        m = re.search(r"Tempo Totale Esecuzione:\s*([\d.]+)\s*s", output)
        if m:
            metriche["tempo"] = float(m.group(1))
        m = re.search(r"Nodi B&B esplorati:\s*([\d.]+)", output)
        if m:
            metriche["nodi"] = float(m.group(1))

    return metriche


def stato_leggibile(stato, formulazione):
    if stato is None:
        return "ERRORE"
    return {"2": "OTTIMO", "9": "TIME_LIMIT"}.get(stato, stato)


# ===========================================================================
# ESECUZIONE DI UNO SCRIPT (subprocess, con timeout di sicurezza extra)
# ===========================================================================
def esegui(script, cartella_istanza, extra_args, timeout_sicurezza):
    cmd = [PY, script, "--dir", cartella_istanza] + extra_args
    t0 = time.time()
    try:
        risultato = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sicurezza
        )
        output = risultato.stdout + "\n" + risultato.stderr
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or "") + "\n" + (e.stderr or "") + \
            "\n[AUTOMAZIONE] Timeout di sicurezza superato, processo interrotto."
    tempo_wall = time.time() - t0
    return output, tempo_wall


def scrivi_riga_riassunto(log_riassunto, etichetta_scenario, formulazione, output):
    metriche = estrai_metriche_compatto(output, formulazione)
    stato_str = stato_leggibile(metriche["stato"], formulazione)
    flusso_str = f"{metriche['flusso']:.1f}" if metriche["flusso"] is not None else "N/A"
    gap_str = f"{metriche['gap']*100:.2f}" if metriche["gap"] is not None else "N/A"
    tempo_str = f"{metriche['tempo']:.1f}" if metriche["tempo"] is not None else "N/A"
    nodi_str = f"{metriche['nodi']:.0f}" if metriche["nodi"] is not None else "N/A"

    riga = (
        f"{etichetta_scenario:<22}{formulazione:<16}{stato_str:<12}"
        f"{flusso_str:<10}{gap_str:<10}{tempo_str:<12}{nodi_str:<10}\n"
    )
    log_riassunto.write(riga)
    log_riassunto.flush()
    print("    " + riga.strip())


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    os.makedirs(CARTELLA_ISTANZE, exist_ok=True)

    log_completo = open(LOG_COMPLETO, "w", encoding="utf-8")
    log_riassunto = open(LOG_RIASSUNTO, "w", encoding="utf-8")

    totale_scenari = len(SCENARI) + len(SCENARI_V2_ALTA_DENSITA)
    log_riassunto.write(f"RISULTATI TEST MFPC -- {totale_scenari} scenari\n")
    log_riassunto.write("=" * 100 + "\n")
    log_riassunto.write(
        f"{'Scenario (n,p,d)':<22}{'Formulazione':<16}{'Stato':<12}"
        f"{'Flusso*':<10}{'Gap(%)':<10}{'Tempo(s)':<12}{'Nodi':<10}\n"
    )
    log_riassunto.write("-" * 100 + "\n")
    log_riassunto.flush()

    idx_globale = 0

    # --- Scenari principali: Strong / Weak / Benders originale / Benders v2 -
    for n, p, d, seed in SCENARI:
        idx_globale += 1
        etichetta_scenario = f"n{n}_p{p}_d{d}"
        cartella = os.path.join(CARTELLA_ISTANZE, etichetta_scenario)
        os.makedirs(cartella, exist_ok=True)

        print(f"\n[{idx_globale}/{totale_scenari}] Scenario {etichetta_scenario} -- generazione istanza...")
        log_completo.write(f"\n{'#'*90}\n# SCENARIO {etichetta_scenario} (seed={seed})\n{'#'*90}\n")

        out_gen, t_gen = esegui(
            "generatore.py", cartella,
            ["--n", str(n), "--p", str(p), "--d", str(d), "--seed", str(seed)],
            timeout_sicurezza=300,
        )
        log_completo.write(f"\n--- generatore.py ({t_gen:.1f}s) ---\n{out_gen}\n")
        log_completo.flush()

        for formulazione, script, extra, timelimit in [
            ("strong", "strong.py", ["--timelimit", str(TIMELIMIT_STRONG)], TIMELIMIT_STRONG),
            ("weak", "weak.py", ["--timelimit", str(TIMELIMIT_WEAK)], TIMELIMIT_WEAK),
            ("benders", "benders.py", ["--maxtime", str(TIMELIMIT_BENDERS)], TIMELIMIT_BENDERS),
            ("benders_v2", "benders_v2.py", ["--maxtime", str(TIMELIMIT_BENDERS)], TIMELIMIT_BENDERS),
        ]:
            print(f"    -> {formulazione} (timeout {timelimit}s)...")
            output, t_wall = esegui(script, cartella, extra, timeout_sicurezza=timelimit + 60)
            log_completo.write(f"\n--- {script} (wall clock {t_wall:.1f}s) ---\n{output}\n")
            log_completo.flush()
            scrivi_riga_riassunto(log_riassunto, etichetta_scenario, formulazione, output)

    # --- Scenari extra: solo benders_v2 in configurazione alta densita' ----
    for n, p, d, seed in SCENARI_V2_ALTA_DENSITA:
        idx_globale += 1
        etichetta_scenario = f"n{n}_p{p}_d{d}_altadensita"
        cartella = os.path.join(CARTELLA_ISTANZE, etichetta_scenario)
        os.makedirs(cartella, exist_ok=True)

        print(f"\n[{idx_globale}/{totale_scenari}] Scenario {etichetta_scenario} (alta densita') -- generazione istanza...")
        log_completo.write(f"\n{'#'*90}\n# SCENARIO {etichetta_scenario} (seed={seed})\n{'#'*90}\n")

        out_gen, t_gen = esegui(
            "generatore.py", cartella,
            ["--n", str(n), "--p", str(p), "--d", str(d), "--seed", str(seed)],
            timeout_sicurezza=300,
        )
        log_completo.write(f"\n--- generatore.py ({t_gen:.1f}s) ---\n{out_gen}\n")
        log_completo.flush()

        for formulazione, script, extra, timelimit in [
            ("weak", "weak.py", ["--timelimit", str(TIMELIMIT_WEAK)], TIMELIMIT_WEAK),
            ("benders_v2_cuts0", "benders_v2.py",
             ["--maxtime", str(TIMELIMIT_BENDERS), "--no-strong-master", "--cuts", "0"],
             TIMELIMIT_BENDERS),
        ]:
            print(f"    -> {formulazione} (timeout {timelimit}s)...")
            output, t_wall = esegui(script, cartella, extra, timeout_sicurezza=timelimit + 60)
            log_completo.write(f"\n--- {script} (wall clock {t_wall:.1f}s) ---\n{output}\n")
            log_completo.flush()
            scrivi_riga_riassunto(log_riassunto, etichetta_scenario, formulazione, output)

    log_riassunto.write("-" * 100 + "\n")
    log_riassunto.write(f"\nCompletato. Log completo in: {LOG_COMPLETO}\n")
    log_completo.close()
    log_riassunto.close()

    print(f"\nFatto. Vedi '{LOG_RIASSUNTO}' per il riassunto e '{LOG_COMPLETO}' per i log completi.")


if __name__ == "__main__":
    main()