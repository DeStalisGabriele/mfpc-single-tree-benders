"""
Single-Tree Benders Decomposition per il Maximum Flow Problem with Conflicts
(Suvak, Altinel & Aras, 2020, EJOR 287:410-437) -- VERSIONE OTTIMIZZATA (v2).

Rispetto a benders.py, questa versione introduce 5 modifiche mirate a
rendere il Benders competitivo (e idealmente piu' veloce) rispetto alle
formulazioni compatte Strong/Weak, indirizzando le cause di lentezza
individuate nel Capitolo 4 della tesi:

  (1) SUBPROBLEM PIU' VELOCE (maxflow_fast.FastMaxFlow)
      La topologia della rete non cambia mai durante la risoluzione: solo le
      capacita' cambiano ad ogni chiamata di callback. La vecchia versione
      ricostruiva da zero una matrice sparsa e un dizionario di adiacenza ad
      ogni chiamata; questa versione precalcola tutto UNA volta e riusa
      buffer numpy. Guadagno misurato: 2-4x per chiamata al subproblem (vedi
      benchmark_v2.py), crescente con la dimensione dell'istanza.

  (2) VINCOLI DI CONFLITTO FORTI (Eq. 3) NEL MASTER, non solo quelli deboli
      aggregati (Eq. 6). La tesi (Cap. 4, "Solver Evolution Since 2020")
      osserva che Gurobi genera automaticamente molti clique cut sfruttando
      la struttura PAIRWISE dei conflitti nelle formulazioni compatte -- e'
      esattamente questo che rende Strong/Weak competitive. Il master del
      Benders, usando solo il vincolo aggregato debole, non dava a Gurobi la
      stessa struttura da sfruttare. Aggiungendo anche i vincoli pairwise
      (che sono impliciti/dominanti rispetto a quelli deboli, quindi non
      cambiano l'insieme di soluzioni intere ma stringono il rilassamento
      LP) il master ottiene lo stesso vantaggio di presolve.

  (3) TAGLI VALIDI A NODI FRAZIONARI (MIPNODE + cbCut). Dimostrazione di
      validita': il taglio di ottimalita' forte v <= sum_{a in delta(x_bar)}
      u_a * x_a e' costruito a partire da una PARTIZIONE DI NODI (raggiungi-
      bili da s / non raggiungibili) nella rete con capacita' perturbate.
      Una volta fissata questa partizione, l'insieme di archi delta(x_bar)
      che la attraversano e' un taglio (s,t) FISSO nella topologia originale
      -- che NON dipende da x. Per qualunque altro x' (frazionario o intero),
      ogni flusso ammissibile deve ancora attraversare quel taglio, quindi
      max-flow(x') <= sum_{a in delta(x_bar)} u_a * x'_a resta valida. Questo
      significa che il taglio si puo' generare anche usando un punto
      frazionario x_bar (dal rilassamento LP a un nodo B&B) e iniettarlo
      come "user cut" globale con cbCut, stringendo il bound PRIMA che il
      solver trovi soluzioni intere -- non solo dopo, come nella versione
      base. Impatto atteso: meno nodi esplorati.

  (4) cbGetSolution / cbGetNodeRel BATCH: un'unica chiamata sull'intera lista
      di variabili invece di una chiamata per variabile (riduce il numero di
      attraversamenti della frontiera Python/C di Gurobi da O(archi) a O(1)
      per callback).

  (5) WARM START: soluzione iniziale euristica (greedy sul grafo dei
      conflitti, arco a capacita' piu' alta prima) usata per pre-caricare
      x.Start / v.Start, cosi' il solver parte gia' da un incumbent
      ragionevole invece di scoprirlo da zero.

Uso:
    python benders_v2.py --dir . --maxtime 1200
"""

import argparse
import random
import time

import gurobipy as gp
from gurobipy import GRB

from common import carica_istanza, TOL
from maxflow_fast import FastMaxFlow

# Ogni quante chiamate MIPNODE tentare un taglio frazionario (throttle: la
# callback MIPNODE viene invocata molto piu' spesso di MIPSOL, quindi
# generare un taglio ad ogni nodo esplorato sarebbe inutilmente costoso).
MIPNODE_CUT_EVERY = 15
MIPNODE_CUT_MAX_TOTAL = 400


def genera_x_diversificati(archi, conflitti_per_arco, k, seed_base=0):
    """Genera k assegnazioni x ammissibili DIVERSE tra loro, con lo stesso
    greedy weighted-independent-set del warm start ma con un tie-break
    casuale diverso ad ogni run (piccolo rumore sulla capacita' usata per
    ordinare gli archi). Servono per costruire piu' tagli di pre-seeding
    STATICI (aggiunti come vincoli normali, non lazy, PRIMA di avviare il
    Branch-and-Bound -- vedi modifica 8 in main())."""
    risultati = []
    archi_keys = list(archi.keys())
    for seed in range(seed_base, seed_base + k):
        rng = random.Random(seed)
        ordine = sorted(archi_keys, key=lambda a: -(archi[a] + rng.random() * 1e-6))
        scelti = set()
        bloccati = set()
        for a in ordine:
            if a in bloccati:
                continue
            conflitti = conflitti_per_arco.get(a, [])
            if any(kl in scelti for kl in conflitti):
                continue
            scelti.add(a)
            for kl in conflitti:
                bloccati.add(kl)
        risultati.append(scelti)
    return risultati


def greedy_feasible_x(archi, conflitti_per_arco):
    """Costruzione euristica di un x ammissibile: ordina gli archi per
    capacita' decrescente e li attiva uno alla volta se non confliggono con
    un arco gia' scelto (greedy weighted independent set sul grafo dei
    conflitti). Usata solo per il warm start (.Start), non incide sulla
    correttezza/ottimalita' finale."""
    ordine = sorted(archi.keys(), key=lambda a: -archi[a])
    scelti = set()
    bloccati = set()
    for a in ordine:
        if a in bloccati:
            continue
        conflitti = conflitti_per_arco.get(a, [])
        if any(kl in scelti for kl in conflitti):
            continue
        scelti.add(a)
        for kl in conflitti:
            bloccati.add(kl)
    return scelti


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Cartella con meta.txt/archi.csv/conflitti.csv")
    parser.add_argument("--maxtime", type=float, default=1200.0, help="Time limit in secondi")
    parser.add_argument("--fractional-cuts", action="store_true",
                         help="Riattiva i tagli a nodi frazionari (modifica 3). Disattivati di default: "
                              "su test a densita' reale, combinati col master forte, peggioravano il tempo.")
    parser.add_argument("--no-warmstart", action="store_true", help="Disattiva il warm start euristico (modifica 5)")
    parser.add_argument("--extra-preseed", type=int, default=0,
                         help="Modifica 8 (sperimentale): aggiunge N tagli di ottimalita' STATICI, generati da N "
                              "assegnazioni x euristiche diverse, come vincoli normali PRIMA di avviare il "
                              "Branch-and-Bound (non durante, quindi non altera la dinamica di ricerca come i "
                              "tagli lazy). Verifica empiricamente se aiuta sulla tua istanza: default 0 (disattivato).")
    parser.add_argument("--no-strong-master", action="store_true",
                         help="Usa il vincolo di conflitto DEBOLE aggregato (Eq. 6) invece dei vincoli "
                              "pairwise forti (Eq. 3) nel master (modifica 2). Su istanze molto dense in "
                              "conflitti il numero di vincoli pairwise puo' essere O(archi^2): con una "
                              "licenza Gurobi size-limited o hardware limitato puo' convenire disattivarli.")
    parser.add_argument("--threads", type=int, default=None, help="Gurobi Params.Threads (default: lascia decidere a Gurobi)")
    parser.add_argument("--cuts", type=int, default=None, help="Gurobi Params.Cuts (0=off, 1=moderato, 2=aggressivo, 3=molto aggressivo)")
    parser.add_argument("--mipfocus", type=int, default=None, help="Gurobi Params.MIPFocus (1=incumbent, 2=bound, 3=migliora il bound quando il gap e' fermo)")
    parser.add_argument("--heuristics", type=float, default=None, help="Gurobi Params.Heuristics, frazione di tempo su euristiche (default Gurobi: 0.05)")
    parser.add_argument("--presolve", type=int, default=None, help="Gurobi Params.Presolve (-1=auto, 0=off, 1=moderato, 2=aggressivo)")
    parser.add_argument("--branch-priority", action="store_true",
                         help="Modifica 9 (sperimentale): assegna priorita' di branching piu' alta agli archi con "
                              "piu' conflitti e a quelli nel taglio di pre-seeding iniziale, cosi' Gurobi li decide "
                              "prima nell'albero invece che in ordine arbitrario.")
    args = parser.parse_args()

    n, archi, conflitti_per_arco = carica_istanza(args.dir)
    s, t = 1, n
    archi_list = list(archi.keys())

    n_archi = len(archi_list)
    n_conflitti = sum(len(v) for v in conflitti_per_arco.values())
    print(f"[DIAG] Nodi: {n}  Archi attivi: {n_archi}  Coppie in conflitto lette: {n_conflitti}")

    U_totale_capacita = sum(archi.values())
    m_archi_totale = n_archi

    # --- Subproblem veloce (modifica 1) -------------------------------------
    solver_mf = FastMaxFlow(n, archi_list, s, t)
    cap_array = [archi[a] for a in archi_list]

    # --- Pre-seeding (Sezione 3.1.3 del paper) ------------------------------
    flusso_rilassato, visited_rilassato = solver_mf.solve(cap_array)
    UB_relax = flusso_rilassato
    print(f"[DIAG] UB rilassata (rete completa, senza conflitti): {UB_relax}")
    cut0 = solver_mf.cut_arcs(visited_rilassato)

    # --- Modello master ------------------------------------------------------
    model = gp.Model("SingleTree_Benders_MFPC_v2")

    x = model.addVars(archi_list, vtype=GRB.BINARY, name="x")
    v = model.addVar(lb=0.0, ub=UB_relax, vtype=GRB.CONTINUOUS, name="v")
    model.setObjective(v, GRB.MAXIMIZE)

    # --- Modifica (2): vincoli di conflitto FORTI (Eq. 3, pairwise) al posto
    # dei soli vincoli deboli aggregati (Eq. 6). I vincoli forti implicano i
    # deboli (nessuna soluzione intera viene persa o aggiunta), ma stringono
    # il rilassamento LP dando a Gurobi la stessa struttura pairwise che
    # sfrutta con i clique cut nelle formulazioni compatte (vedi discussione
    # nel Cap. 4 della tesi).
    if not args.no_strong_master:
        visti = set()
        n_vincoli_forti = 0
        for (i, j), conflitti in conflitti_per_arco.items():
            for (k, l) in conflitti:
                pair = frozenset([(i, j), (k, l)])
                if pair in visti:
                    continue
                visti.add(pair)
                if (i, j) in x and (k, l) in x:
                    model.addConstr(x[i, j] + x[k, l] <= 1, name=f"conflitto_{i}_{j}_{k}_{l}")
                    n_vincoli_forti += 1
        print(f"[DIAG] Vincoli di conflitto forti (pairwise) aggiunti al master: {n_vincoli_forti}")
    else:
        n_vincoli_deboli = 0
        for (i, j), conflitti in conflitti_per_arco.items():
            if conflitti and (i, j) in x:
                k_conf = len(conflitti)
                model.addConstr(
                    gp.quicksum(x[kl] for kl in conflitti if kl in x) + k_conf * x[i, j] <= k_conf,
                    name=f"conflitto_{i}_{j}",
                )
                n_vincoli_deboli += 1
        print(f"[DIAG] Vincoli di conflitto deboli (aggregati) aggiunti al master: {n_vincoli_deboli}")

    # Vincoli di pre-seeding dal min-cut della rilassata
    model.addConstr(v <= gp.quicksum(archi[a] * x[a] for a in cut0), name="preseed_v")
    model.addConstr(gp.quicksum(x[a] for a in cut0) >= 1, name="preseed_cut")

    # --- Modifica (8, sperimentale): tagli di pre-seeding statici extra ----
    # A differenza dei tagli lazy/frazionari (aggiunti DURANTE il B&B, dove
    # possono cambiare in modo imprevedibile il cammino di ricerca -- vedi i
    # test su v3), questi sono vincoli NORMALI aggiunti PRIMA di avviare
    # optimize(): danno a Gurobi piu' materiale per il presolve fin dal nodo
    # radice, un po' come le formulazioni compatte hanno fin da subito tutti
    # i vincoli di flow-conservation. Ogni taglio e' valido per OGNI x (stessa
    # dimostrazione della modifica 3), quindi non toglie ne' aggiunge
    # soluzioni ammissibili: puo' solo stringere il rilassamento LP.
    if args.extra_preseed > 0:
        diversificati = genera_x_diversificati(archi, conflitti_per_arco, args.extra_preseed)
        n_extra = 0
        for scelti in diversificati:
            x_val_div = [1.0 if a in scelti else 0.0 for a in archi_list]
            cap_mod = [archi[a] * (U_totale_capacita * xv + 1) for a, xv in zip(archi_list, x_val_div)]
            _, visited_div = solver_mf.solve(cap_mod)
            cut_idx = solver_mf.cut_arc_indices(visited_div)
            model.addConstr(v <= gp.quicksum(archi[archi_list[k]] * x[archi_list[k]] for k in cut_idx))
            n_extra += 1
        print(f"[DIAG] Tagli di pre-seeding extra (statici, prima del B&B): {n_extra}")

    # --- Modifica (5): warm start euristico ---------------------------------
    if not args.no_warmstart:
        scelti = greedy_feasible_x(archi, conflitti_per_arco)
        cap_greedy = [archi[a] if a in scelti else 0.0 for a in archi_list]
        flow_greedy, _ = solver_mf.solve(cap_greedy)
        for a in archi_list:
            x[a].Start = 1.0 if a in scelti else 0.0
        v.Start = flow_greedy
        print(f"[DIAG] Warm start euristico: {len(scelti)} archi attivi, flusso={flow_greedy}")

    # --- Modifica (9, sperimentale): priorita' di branching -----------------
    # Gurobi, in assenza di indicazioni, decide da solo su quale variabile
    # ramificare. Qui suggeriamo di dare priorita' agli archi "strutturalmente
    # importanti": quelli con piu' conflitti (decidere il loro valore
    # inchioda anche molte altre variabili via i vincoli pairwise) e quelli
    # gia' comparsi nel taglio di pre-seeding iniziale (sappiamo gia' che sono
    # rilevanti per il valore ottimo di v). BranchPriority e' solo un
    # suggerimento euristico, non cambia la correttezza.
    if args.branch_priority:
        cut0_set = set(cut0)
        for a in archi_list:
            grado_conflitto = len(conflitti_per_arco.get(a, []))
            priorita = grado_conflitto + (10 if a in cut0_set else 0)
            if priorita > 0:
                x[a].BranchPriority = priorita
        print(f"[DIAG] Priorita' di branching assegnata in base al grado di conflitto + pre-seeding")

    # --- Parametri: NESSUN presolve/cuts/euristiche disattivati di default,
    # salvo override espliciti da riga di comando per esperimenti -----------
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = args.maxtime
    if args.threads is not None:
        model.Params.Threads = args.threads
    if args.cuts is not None:
        model.Params.Cuts = args.cuts
    if args.mipfocus is not None:
        model.Params.MIPFocus = args.mipfocus
    if args.heuristics is not None:
        model.Params.Heuristics = args.heuristics
    if args.presolve is not None:
        model.Params.Presolve = args.presolve

    x_list = [x[a] for a in archi_list]

    stato = {
        "call_counter": 0,
        "num_tagli": 0,
        "num_tagli_frazionari": 0,
        "mipnode_counter": 0,
        "start_time": None,
        "last_heartbeat": 0.0,
    }

    def aggiungi_taglio_ottimalita(x_val, is_lazy):
        """Costruisce il taglio di ottimalita' forte (Prop. 4) a partire da
        un vettore x_val (0/1 se da MIPSOL, eventualmente frazionario se da
        MIPNODE) e lo inietta come lazy constraint o come user cut."""
        cap_mod = [archi[a] * (U_totale_capacita * xv + 1) for a, xv in zip(archi_list, x_val)]
        _, visited_opt = solver_mf.solve(cap_mod)
        cut_idx = solver_mf.cut_arc_indices(visited_opt)
        expr = gp.quicksum(archi[archi_list[k]] * x[archi_list[k]] for k in cut_idx)
        if is_lazy:
            model.cbLazy(v <= expr)
        else:
            model.cbCut(v <= expr)
        return cut_idx

    def callback(model, where):
        # --- Modifica (4): batch cbGetSolution -----------------------------
        if where == GRB.Callback.MIPSOL:
            stato["call_counter"] += 1
            now = time.time() - stato["start_time"]
            if now - stato["last_heartbeat"] > 5:
                nodi = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
                print(f"[HEARTBEAT] t={now:.1f}s  chiamate={stato['call_counter']}  "
                      f"tagli={stato['num_tagli']} (+{stato['num_tagli_frazionari']} frazionari)  nodi={nodi:.0f}")
                stato["last_heartbeat"] = now

            master_v = model.cbGetSolution(v)
            x_val = model.cbGetSolution(x_list)  # UNA chiamata invece di n_archi chiamate

            cap_attiva = [archi[a] if xv > 0.5 else 0.0 for a, xv in zip(archi_list, x_val)]
            max_flow, visited = solver_mf.solve(cap_attiva)

            if master_v > max_flow + TOL:
                aggiungi_taglio_ottimalita(x_val, is_lazy=True)
                stato["num_tagli"] += 1

                if max_flow < TOL:
                    cap_mod2 = [m_archi_totale * archi[a] * xv + 1 for a, xv in zip(archi_list, x_val)]
                    _, visited_conn = solver_mf.solve(cap_mod2)
                    cut_idx2 = solver_mf.cut_arc_indices(visited_conn)
                    expr2 = gp.quicksum(x[archi_list[k]] for k in cut_idx2)
                    model.cbLazy(expr2 >= 1)
                    stato["num_tagli"] += 1

        # --- Modifica (3): tagli validi a nodi frazionari -------------------
        elif (where == GRB.Callback.MIPNODE and args.fractional_cuts
              and stato["num_tagli_frazionari"] < MIPNODE_CUT_MAX_TOTAL):
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            stato["mipnode_counter"] += 1
            if stato["mipnode_counter"] % MIPNODE_CUT_EVERY != 0:
                return

            master_v_rel = model.cbGetNodeRel(v)
            x_val_rel = model.cbGetNodeRel(x_list)

            # Validita' globale dimostrata sopra: il taglio costruito da un
            # punto frazionario e' comunque un taglio (s,t) fisso nella
            # topologia originale, quindi vale per OGNI x, non solo per
            # quello che lo ha generato. Lo iniettiamo solo se e' violato
            # nel punto corrente, per evitare tagli inutili.
            cap_mod = [archi[a] * (U_totale_capacita * xv + 1) for a, xv in zip(archi_list, x_val_rel)]
            _, visited_opt = solver_mf.solve(cap_mod)
            cut_idx = solver_mf.cut_arc_indices(visited_opt)
            cap_cut_rel = sum(archi[archi_list[k]] * x_val_rel[k] for k in cut_idx)

            if master_v_rel > cap_cut_rel + TOL:
                expr = gp.quicksum(archi[archi_list[k]] * x[archi_list[k]] for k in cut_idx)
                model.cbCut(v <= expr)
                stato["num_tagli_frazionari"] += 1

    stato["start_time"] = time.time()
    print("--- Inizio Ottimizzazione Single-Tree Benders v2 (Gurobi/Python) ---")
    model.optimize(callback)

    tempo_totale = time.time() - stato["start_time"]

    print("\n--- RISULTATI FINALI (v2) ---")
    print(f"Status Gurobi: {model.Status}  (2 = OPTIMAL, 9 = TIME_LIMIT)")
    print(f"Flusso Ottimo v*: {model.ObjVal if model.SolCount > 0 else 'N/A'}")
    print(f"MIP Gap: {model.MIPGap if model.SolCount > 0 else 'N/A'}")
    print(f"Tempo Totale Esecuzione: {tempo_totale:.4f} s")
    print(f"Tagli Benders (lazy, su soluzioni intere): {stato['num_tagli']}")
    print(f"Tagli aggiuntivi a nodi frazionari (user cuts): {stato['num_tagli_frazionari']}")
    print(f"Chiamate totali alla callback MIPSOL: {stato['call_counter']}")
    print(f"Nodi B&B esplorati: {model.NodeCount:.0f}")


if __name__ == "__main__":
    main()
