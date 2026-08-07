# Log completo degli esperimenti

Tutti i tempi in questa pagina, salvo dove esplicitamente indicato "macchina secondaria", sono stati misurati sulla macchina di riferimento della tesi (Intel Core i7-13620H, Windows 11, licenza Gurobi accademica WLS, come da Cap. 4 §4.1). I numeri della macchina secondaria sono segnalati chiaramente e vanno usati solo come indicazione di direzione, non come valori assoluti: si è verificato più volte che lo stesso identico modello, sulla stessa identica istanza (stesso seed), dà tempi assoluti diversi tra macchine diverse — coerente con quanto già discusso nella tesi (Cap. 4, "Search-Path Variability" e "Different solver and hardware").

## 0. Baseline: `benders.py` originale (Tabella 4.1 della tesi, p=0.6, d=0.3)

| n | Strong | Weak | Benders (originale) | Nodi B&B Benders |
|---|---|---|---|---|
| 40 | 6.3 s | 4.6 s | 29.6 s | 14.445 |
| 60 | 78.0 s | 131.3 s | 1058.5 s | 206.159 |
| 80 | 1655.9 s | 754.5 s | time limit, gap 20.08% | 191.820 |
| 100 | time limit | time limit | time limit, incumbent 21 (vs 53 di Strong/Weak) | 59.964 |

## 1. `benders_v2.py`, configurazione di default (p=0.6, d=0.3, macchina reale)

Modifiche attive: subproblem veloce, master forte (pairwise), warm start, batch cbGetSolution. Tagli frazionari disattivati.

| n | Tempo | Nodi B&B | Speedup vs originale |
|---|---|---|---|
| 40 | 17.87 s | 9.076 | 1.66x |
| 60 | 652.0 s | 153.610 | 1.62x |

**Unica configurazione a migliorare in modo consistente su entrambe le scale.**

## 2. Ablation a n=40 (p=0.6, d=0.3, macchina reale) — quali modifiche contano

| Configurazione | Tempo | Nodi |
|---|---|---|
| v2 completo (master forte + tagli frazionari) | 20.06 s | 7.254 |
| v2 solo master forte (no tagli frazionari) | **17.87 s** | 9.076 |
| v2 solo tagli frazionari (master debole) | 18.17 s | 7.937 |
| v2 solo subproblem veloce + warm start (nessuna delle due) | 21.82 s | 9.214 |

Conclusione: master forte e tagli frazionari aiutano entrambi presi singolarmente, ma **insieme si ostacolano a vicenda** (20.06s, peggio di ciascuno preso da solo) — probabile ridondanza informativa più overhead dei tagli frazionari. Per questo i tagli frazionari sono disattivati di default in v2.

## 3. `--extra-preseed` (tagli di pre-seeding statici) — macchina reale, p=0.6, d=0.3

| n | K (extra-preseed) | Tempo | Nodi |
|---|---|---|---|
| 40 | 0 (baseline) | 17.87 s | 9.076 |
| 40 | 20 | **16.01 s** | 5.849 |
| 40 | 50 | 16.08 s | 6.766 |
| 60 | 0 (baseline) | 652.0 s | 153.610 |
| 60 | 20 | 807.6 s | 177.038 |
| 60 | 50 | 761.5 s | 185.687 |

A n=40 aiuta (miglior risultato assoluto trovato per questa scala: 16.01s), a n=60 peggiora nettamente. Non generalizza tra scale — lasciato disattivato di default (`--extra-preseed 0`), disponibile come opzione da validare caso per caso.

## 4. Parametri Gurobi — macchina reale, p=0.6, d=0.3

### n=40

| Flag | Tempo | Nodi |
|---|---|---|
| (baseline) | 17.87 s | 9.076 |
| `--cuts 0` | **12.90 s** | 7.871 |
| `--cuts 1` | 19.41 s | 9.602 |
| `--cuts 3 --mipfocus 3` | 66.36 s | 11.189 |
| `--mipfocus 3` (isolato) | 65.17 s | 11.189 |
| `--threads 16` (isolato) | 18.06 s | 6.278 |

`--cuts 0` è la singola leva più efficace trovata a questa scala (1.39x). `--mipfocus 3` è nettamente dannoso, isolato o combinato. `--threads 16` è neutro sul tempo ma dimezza quasi i nodi (effetto da capire meglio, non sfruttato).

### n=60

| Flag | Tempo | Nodi |
|---|---|---|
| (baseline) | 652.0 s | 153.610 |
| `--cuts 3 --mipfocus 3` | 1030.9 s (gap 3.2e-5, non provato ottimo) | 126.259 |
| `--cuts 0` | 1121.5 s | 261.969 |

Entrambi i vincitori di n=40 **peggiorano nettamente** a n=60 — pattern non generalizzabile.

### Combinazioni di `--cuts 0` con altre leve (n=40) — tutte fallite

| Combinazione | Tempo | Nodi |
|---|---|---|
| `--cuts 0` da solo | 12.90 s | 7.871 |
| `--cuts 0 --extra-preseed 20` | 26.05 s | 7.117 |
| `--cuts 0 --threads 16` | 32.83 s | 8.195 |
| `--cuts 0 --presolve 2` | 36.04 s | 7.759 |
| `--cuts 0 --extra-preseed 50` | 30.28 s | 9.982 |

4 combinazioni su 4 testate peggiorano rispetto a `--cuts 0` isolato, spesso di 2-3x, pur con nodi simili o inferiori (quindi il costo per nodo sale parecchio quando si combinano più leve). Nessuna combinazione testata batte la singola modifica isolata.

## 5. `benders_v3.py` (upper bound iniziale stretto + taglio "gratis") — RIGETTATA

Idea: (6) risolvere il rilassamento LP del master intero per un upper bound iniziale più stretto su `v`; (7) iniettare anche il min-cut grezzo (non perturbato) della rete attiva come taglio lazy aggiuntivo, a costo computazionale nullo (già calcolato per decidere se serve un taglio).

| n | Configurazione | Tempo | Nodi |
|---|---|---|---|
| 60 | v3 completo (UB stretto + taglio gratis) | 1162.8 s | 198.111 |
| 60 | v3 `--no-raw-cut` (solo UB stretto) | 732.5 s | 153.610 (identico a v2 baseline) |
| 60 | v3 `--no-tight-ub` (solo taglio gratis), timeout 900s | 900.2 s, gap 18.4%, incumbent 49 (non ottimo) | 161.178 |

L'UB stretto non ha cambiato di una virgola la ricerca (nodi identici a v2 baseline) — Gurobi arriva comunque allo stesso bound da solo via presolve — ma calcolarlo costa ~80s extra su un master con decine di migliaia di vincoli pairwise: puro costo, zero beneficio. Il taglio "gratis" rallenta la ricerca (meno nodi/secondo) invece di aiutarla, nonostante non richieda calcoli aggiuntivi — un altro caso di "Search-Path Variability": più vincoli validi non garantiscono un albero di ricerca migliore.

## 6. `benders_v4.py` (Benders multi-tree classico, senza `LazyConstraints`) — RIGETTATA

Idea: evitare `LazyConstraints=1` del tutto (e quindi la disattivazione delle dual reduction — vedi README) ririsolvendo il master da zero, con presolve pieno, ogni volta che la soluzione trovata non rispetta il subproblem reale.

| Istanza | Tempo | Iterazioni (ririsolvi completi) |
|---|---|---|
| n=60, macchina secondaria (bassa densità) | 7.35 s | 277 |
| n=80, macchina secondaria (bassa densità) | 0.20 s | 32 |
| n=40, macchina reale, p=0.6/d=0.3 | timeout a 300s | 38 (non convergente; v* riportato non valido) |

Confronto diretto sulla macchina secondaria: contro v2 baseline, v4 è 10x peggio quando servono molte iterazioni (n=60 macchina secondaria: 277 iter) e 2.3x meglio quando ne servono poche (n=80 macchina secondaria: 32 iter) — il costo di ririsolvere da zero il master supera il vantaggio del presolve pieno se le iterazioni necessarie sono troppe. Sulla macchina reale, a densità di conflitto vera (p=0.6/d=0.3), servono centinaia/migliaia di iterazioni (confrontabile col numero di chiamate MIPSOL del single-tree, 340-2000 a seconda della scala) — con ~8s a iterazione osservati prima del timeout, la convergenza completa avrebbe richiesto ore. Architettura scartata per questa classe di istanze.

## 7. Il regime dove Benders batte Weak — densità di conflitto alta (macchina reale)

Vedi anche README.md. Parametri: p=0.6 fisso, d variabile, comando `benders_v2.py --no-strong-master --cuts 0` (il master forte pairwise supera il limite della licenza size-limited a densità di conflitto molto alta; qui non è comunque il fattore vincente).

| n | d | Weak | Benders | Rapporto |
|---|---|---|---|---|
| 30 | 0.3 | 0.25 s | 0.61 s | Benders 2.4x più lento |
| 30 | 0.5 | 0.30 s | 0.35 s | Benders 1.17x più lento |
| 30 | 0.7 | 0.28 s | 0.28 s | pareggio |
| 30 | 0.85 | 0.35 s | 0.18 s | Benders 1.91x più veloce |
| 60 | 0.7 | 18.42 s | 12.64 s | **Benders 1.46x più veloce** |
| 60 | 0.85 | 11.77 s | 12.35 s | Benders 1.05x più lento |

Ottimo confermato identico tra Weak e Benders su ogni riga. Nota: gli stessi identici parametri (n=60, p=0.3, d=0.6, seed 12345) testati prima sulla macchina secondaria avevano dato Weak e Benders sostanzialmente pari (12.80s vs 12.81s) — un risultato diverso da quello poi osservato sulla macchina reale con d=0.6 esatto (Weak 2.93s, Benders 6.33s, Benders perdente). La differenza è dovuta all'hardware/ambiente diverso (vedi discussione "Search-Path Variability" nel README), non a un errore di configurazione — motivo per cui da questo punto in poi tutti i test decisivi sono stati eseguiti solo sulla macchina reale.

## Riepilogo: cosa tenere

- **Sempre**: `benders_v2.py` senza flag (1.6-1.7x sull'originale, verificato a più scale, nessuna eccezione).
- **Se la densità di conflitto è alta (d ≳ 0.6-0.7) e la densità di rete è quella reale (p=0.6)**: `benders_v2.py --no-strong-master --cuts 0` può battere Weak — verificare caso per caso, il punto di incrocio non è uno spigolo netto.
- **Da evitare**: combinare più leve contemporaneamente (nessuna combinazione testata ha battuto la sua migliore componente isolata); `benders_v3.py` e `benders_v4.py` come architetture generali; `--mipfocus 3`.
