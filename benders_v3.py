"""
** RIGETTATA / SPERIMENTALE -- NON USARE COME DEFAULT **
Su istanze a densita' di conflitto reale (p=0.6, d=0.3), entrambe le modifiche
qui sotto (UB iniziale stretto, taglio "gratis" dalla partizione grezza)
peggiorano le prestazioni rispetto a benders_v2.py, anche se erano nate da un
ragionamento corretto e sembravano funzionare su istanze di prova piu' piccole
o meno dense. Dettagli e numeri completi in BENCHMARKS.md (sezione 5). File
tenuto per documentare il tentativo, non per l'uso pratico -- usa benders_v2.py.

Single-Tree Benders Decomposition per il Maximum Flow Problem with Conflicts
(Suvak, Altinel & Aras, 2020, EJOR 287:410-437) -- VERSIONE v3.

Basata sui risultati empirici raccolti su benders_v2.py (istanza n=40,
v*=42, confrontata con Benders originale a 29.6s / 14.445 nodi):

    config                                      tempo    nodi
    v2 completo (master forte + tagli fraz.)    20.06s   7.254
    v2 SOLO master forte (no tagli fraz.)       17.87s   9.076   <- migliore
    v2 SOLO tagli fraz. (master debole)         18.17s   7.937
    v2 solo subproblem veloce + warm start      21.82s   9.214

Il dato interessante e' che master forte + tagli frazionari INSIEME
peggiorano (20.06s) rispetto a uno dei due presi da soli: probabilmente
stringono il rilassamento nello stesso modo, e i tagli frazionari
aggiungono solo overhead (una max-flow extra ogni 15 nodi) senza portare
molta informazione che il master forte non dia gia'. Di conseguenza:

  - i tagli a nodi frazionari (modifica 3 di v2) sono ora DISATTIVATI DI
    DEFAULT (riattivabili con --fractional-cuts);
  - il master forte pairwise (modifica 2 di v2) resta attivo di default.

Questa versione aggiunge due nuove leve, entrambe pensate per essere "quasi
gratuite" (costo aggiuntivo trascurabile o nullo):

  (6) UPPER BOUND INIZIALE PIU' STRETTO. La v1/v2 fissavano v.UB al max-flow
      della rete COMPLETAMENTE rilassata (nessun conflitto attivo per
      nessun arco) -- un limite superiore validissimo ma molto lasco, non
      tiene conto della struttura di conflitto per niente. Qui, prima di
      avviare il Branch-and-Bound, si risolve UNA VOLTA il rilassamento
      continuo (x in [0,1]) dell'INTERO master gia' costruito (con i
      vincoli di conflitto forti e i tagli di pre-seeding gia' dentro): il
      suo valore ottimo e' un upper bound valido su v* strettamente piu'
      stretto di UB_relax, ottenuto con un singolo risolutivo LP (millisecondi
      anche su istanze da centinaia di nodi).

  (7) TAGLIO "GRATIS" DALLA PARTIZIONE GREZZA. Nella callback MIPSOL, prima
      di scegliere il taglio di ottimalita' forte (Prop. 4, che richiede una
      seconda max-flow su capacita' perturbate per selezionare il min-cut
      "migliore" tra eventuali pareggi), abbiamo GIA' calcolato gratis, per
      decidere se il taglio serve, il min-cut della rete attiva NON
      perturbata (quello restituito dalla prima chiamata a solver_mf.solve
      su cap_attiva). Questa partizione e' anch'essa un taglio (s,t) fisso
      nella topologia originale, quindi da' anch'essa una disuguaglianza
      v <= sum_{a in taglio_grezzo} u_a * x_a valida per OGNI x (stessa
      dimostrazione della modifica 3), tesa esattamente al punto corrente
      (il suo valore a x_bar e' proprio max_flow(x_bar), che e' < master_v,
      quindi e' gia' violata). La iniettiamo SEMPRE insieme al taglio
      raffinato di Prop. 4 quando i due tagli sono diversi: due
      disuguaglianze per il prezzo di una sola max-flow aggiuntiva (quella
      di Prop. 4), perche' il taglio grezzo non richiede nessun calcolo in
      piu' rispetto a quello che la callback fa comunque.

Uso:
    python benders_v3.py --dir . --maxtime 1200
"""

import argparse
import time

import gurobipy as gp
from gurobipy import GRB

from common import carica_istanza, TOL
from maxflow_fast import FastMaxFlow

MIPNODE_CUT_EVERY = 15
MIPNODE_CUT_MAX_TOTAL = 400


def greedy_feasible_x(archi, conflitti_per_arco):
    """Greedy weighted independent set sul grafo dei conflitti (solo per il
    warm start .Start, non incide su correttezza/ottimalita' finale)."""
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
                         help="Riattiva i tagli a nodi frazionari (MIPNODE + cbCut). Disattivati di "
                              "default: sui test n=40 peggioravano il tempo se combinati col master forte.")
    parser.add_argument("--no-warmstart", action="store_true", help="Disattiva il warm start euristico")
    parser.add_argument("--no-strong-master", action="store_true",
                         help="Usa il vincolo di conflitto DEBOLE aggregato (Eq. 6) invece dei vincoli "
                              "pairwise forti (Eq. 3) nel master. Su istanze molto dense in conflitti il "
                              "numero di vincoli pairwise puo' essere O(archi^2): con una licenza Gurobi "
                              "size-limited puo' convenire disattivarli.")
    parser.add_argument("--no-tight-ub", action="store_true",
                         help="Disattiva il calcolo dell'upper bound iniziale via rilassamento LP del master (modifica 6)")
    parser.add_argument("--no-raw-cut", action="store_true",
                         help="Disattiva il taglio extra 'gratis' dalla partizione grezza (modifica 7)")
    args = parser.parse_args()

    n, archi, conflitti_per_arco = carica_istanza(args.dir)
    s, t = 1, n
    archi_list = list(archi.keys())

    n_archi = len(archi_list)
    n_conflitti = sum(len(v) for v in conflitti_per_arco.values())
    print(f"[DIAG] Nodi: {n}  Archi attivi: {n_archi}  Coppie in conflitto lette: {n_conflitti}")

    U_totale_capacita = sum(archi.values())
    m_archi_totale = n_archi

    solver_mf = FastMaxFlow(n, archi_list, s, t)
    cap_array = [archi[a] for a in archi_list]

    flusso_rilassato, visited_rilassato = solver_mf.solve(cap_array)
    UB_relax = flusso_rilassato
    print(f"[DIAG] UB rilassata (rete completa, senza conflitti): {UB_relax}")
    cut0 = solver_mf.cut_arcs(visited_rilassato)

    model = gp.Model("SingleTree_Benders_MFPC_v3")

    x = model.addVars(archi_list, vtype=GRB.BINARY, name="x")
    v = model.addVar(lb=0.0, ub=UB_relax, vtype=GRB.CONTINUOUS, name="v")
    model.setObjective(v, GRB.MAXIMIZE)

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

    model.addConstr(v <= gp.quicksum(archi[a] * x[a] for a in cut0), name="preseed_v")
    model.addConstr(gp.quicksum(x[a] for a in cut0) >= 1, name="preseed_cut")

    # --- Modifica (6): upper bound iniziale piu' stretto --------------------
    if not args.no_tight_ub:
        model.update()
        rilassato = model.relax()
        rilassato.Params.OutputFlag = 0
        rilassato.optimize()
        if rilassato.Status == GRB.OPTIMAL:
            ub_lp = rilassato.ObjVal
            if ub_lp < UB_relax:
                v.UB = ub_lp
                print(f"[DIAG] Upper bound iniziale stretto (LP relax. del master): {ub_lp:.3f} "
                      f"(invece di {UB_relax})")
            else:
                print(f"[DIAG] LP relax. del master ({ub_lp:.3f}) non piu' stretta di UB_relax, "
                      f"mantengo {UB_relax}")

    # --- Warm start euristico -------------------------------------------------
    if not args.no_warmstart:
        scelti = greedy_feasible_x(archi, conflitti_per_arco)
        cap_greedy = [archi[a] if a in scelti else 0.0 for a in archi_list]
        flow_greedy, _ = solver_mf.solve(cap_greedy)
        for a in archi_list:
            x[a].Start = 1.0 if a in scelti else 0.0
        v.Start = flow_greedy
        print(f"[DIAG] Warm start euristico: {len(scelti)} archi attivi, flusso={flow_greedy}")

    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = args.maxtime

    x_list = [x[a] for a in archi_list]

    stato = {
        "call_counter": 0,
        "num_tagli": 0,
        "num_tagli_raw": 0,
        "num_tagli_frazionari": 0,
        "mipnode_counter": 0,
        "start_time": None,
        "last_heartbeat": 0.0,
    }

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            stato["call_counter"] += 1
            now = time.time() - stato["start_time"]
            if now - stato["last_heartbeat"] > 5:
                nodi = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
                print(f"[HEARTBEAT] t={now:.1f}s  chiamate={stato['call_counter']}  "
                      f"tagli={stato['num_tagli']} (+{stato['num_tagli_raw']} raw, "
                      f"+{stato['num_tagli_frazionari']} frazionari)  nodi={nodi:.0f}")
                stato["last_heartbeat"] = now

            master_v = model.cbGetSolution(v)
            x_val = model.cbGetSolution(x_list)

            cap_attiva = [archi[a] if xv > 0.5 else 0.0 for a, xv in zip(archi_list, x_val)]
            max_flow, visited_raw = solver_mf.solve(cap_attiva)

            if master_v > max_flow + TOL:
                # --- Taglio di ottimalita' FORTE (Prop. 4) ---
                cap_mod = [archi[a] * (U_totale_capacita * xv + 1) for a, xv in zip(archi_list, x_val)]
                _, visited_opt = solver_mf.solve(cap_mod)
                cut_idx = solver_mf.cut_arc_indices(visited_opt)
                model.cbLazy(v <= gp.quicksum(archi[archi_list[k]] * x[archi_list[k]] for k in cut_idx))
                stato["num_tagli"] += 1

                # --- Modifica (7): taglio extra "gratis" dalla partizione
                # grezza (visited_raw), se diversa da quella raffinata -----
                if not args.no_raw_cut:
                    cut_idx_raw = solver_mf.cut_arc_indices(visited_raw)
                    if set(cut_idx_raw.tolist()) != set(cut_idx.tolist()):
                        model.cbLazy(v <= gp.quicksum(archi[archi_list[k]] * x[archi_list[k]] for k in cut_idx_raw))
                        stato["num_tagli_raw"] += 1

                # --- Taglio di connettivita' FORTE (Prop. 2-3) ---
                if max_flow < TOL:
                    cap_mod2 = [m_archi_totale * archi[a] * xv + 1 for a, xv in zip(archi_list, x_val)]
                    _, visited_conn = solver_mf.solve(cap_mod2)
                    cut_idx2 = solver_mf.cut_arc_indices(visited_conn)
                    model.cbLazy(gp.quicksum(x[archi_list[k]] for k in cut_idx2) >= 1)
                    stato["num_tagli"] += 1

        elif (where == GRB.Callback.MIPNODE and args.fractional_cuts
              and stato["num_tagli_frazionari"] < MIPNODE_CUT_MAX_TOTAL):
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            stato["mipnode_counter"] += 1
            if stato["mipnode_counter"] % MIPNODE_CUT_EVERY != 0:
                return

            master_v_rel = model.cbGetNodeRel(v)
            x_val_rel = model.cbGetNodeRel(x_list)

            cap_mod = [archi[a] * (U_totale_capacita * xv + 1) for a, xv in zip(archi_list, x_val_rel)]
            _, visited_opt = solver_mf.solve(cap_mod)
            cut_idx = solver_mf.cut_arc_indices(visited_opt)
            cap_cut_rel = sum(archi[archi_list[k]] * x_val_rel[k] for k in cut_idx)

            if master_v_rel > cap_cut_rel + TOL:
                model.cbCut(v <= gp.quicksum(archi[archi_list[k]] * x[archi_list[k]] for k in cut_idx))
                stato["num_tagli_frazionari"] += 1

    stato["start_time"] = time.time()
    print("--- Inizio Ottimizzazione Single-Tree Benders v3 (Gurobi/Python) ---")
    model.optimize(callback)

    tempo_totale = time.time() - stato["start_time"]

    print("\n--- RISULTATI FINALI (v3) ---")
    print(f"Status Gurobi: {model.Status}  (2 = OPTIMAL, 9 = TIME_LIMIT)")
    print(f"Flusso Ottimo v*: {model.ObjVal if model.SolCount > 0 else 'N/A'}")
    print(f"MIP Gap: {model.MIPGap if model.SolCount > 0 else 'N/A'}")
    print(f"Tempo Totale Esecuzione: {tempo_totale:.4f} s")
    print(f"Tagli Benders lazy (Prop.4 + connettivita'): {stato['num_tagli']}")
    print(f"Tagli extra 'gratis' (partizione grezza): {stato['num_tagli_raw']}")
    print(f"Tagli a nodi frazionari (user cuts): {stato['num_tagli_frazionari']}")
    print(f"Chiamate totali alla callback MIPSOL: {stato['call_counter']}")
    print(f"Nodi B&B esplorati: {model.NodeCount:.0f}")


if __name__ == "__main__":
    main()
