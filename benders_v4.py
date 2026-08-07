"""
** RIGETTATA / SPERIMENTALE -- NON USARE COME DEFAULT **
Su istanze a densita' di conflitto reale (p=0.6, d=0.3) il numero di
iterazioni (ririsolvi completi del master) necessario e' troppo alto (centinaia
o migliaia, confrontabile col numero di chiamate MIPSOL del single-tree): a
~8s per iterazione osservati prima di un timeout a 300s (solo 38 iterazioni
completate, nessuna convergenza), la risoluzione completa richiederebbe ore.
Funziona bene SOLO quando servono poche decine di iterazioni (verificato su una
macchina secondaria a bassa densita'). Dettagli in BENCHMARKS.md (sezione 6).
File tenuto per documentare il tentativo -- usa benders_v2.py.

Multi-Tree (classico) Benders Decomposition per il Maximum Flow Problem with
Conflicts -- VERSIONE v4, architettura diversa da v1/v2/v3.

MOTIVAZIONE: la documentazione
ufficiale di Gurobi conferma che impostare LazyConstraints=1 (obbligatorio per
usare cbLazy dentro una callback) disattiva AUTOMATICAMENTE le "dual
reductions" del presolve, per l'intera durata della risoluzione, a prescindere
da qualunque valore di Presolve/Cuts/etc. Il motivo e' di correttezza: le dual
reduction assumono di conoscere gia' tutta la regione ammissibile, cosa non
vera quando i vincoli lazy la definiscono progressivamente. Le formulazioni
compatte (Strong/Weak) non hanno questo problema, quindi ottengono SEMPRE il
presolve completo -- e' un vantaggio strutturale che nessun tuning di
parametri puo' recuperare fintanto che restiamo dentro il meccanismo a
callback (v1/v2/v3).

v4 elimina il problema alla radice: NON usa LazyConstraints/cbLazy affatto.
Torna invece al Benders "classico" (multi-tree, pre-1990): risolve il master
ALL'OTTIMO COMPLETO (presolve pieno, dual reduction incluse), controlla se la
soluzione trovata rispetta davvero il vincolo di max-flow (con il subproblem
reale), e se non lo rispetta aggiunge il taglio violato come vincolo NORMALE
(non lazy) e ririsolve da capo. Ripete finche' la soluzione del master non e'
gia' coerente con il subproblem reale.

Il compromesso e' opposto a quello di v1/v2/v3: qui ogni iterazione ha
accesso al presolve pieno di Gurobi (potenzialmente molto piu' efficiente per
nodo), ma se servono MOLTE iterazioni (molti ririsolvi da capo) il costo puo'
superare quello di un singolo albero con tagli lazy. Quale dei due vince
dipende dall'istanza: e' esattamente il tipo di domanda che si puo' rispondere
solo misurando, non ragionando a tavolino -- per questo e' un file a parte
invece di un'ennesima modifica in-place a benders_v2.py.

Uso:
    python benders_v4.py --dir . --maxtime 1200
"""

import argparse
import time

import gurobipy as gp
from gurobipy import GRB

from common import carica_istanza, TOL
from maxflow_fast import FastMaxFlow
from benders_v2 import greedy_feasible_x, genera_x_diversificati


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Cartella con meta.txt/archi.csv/conflitti.csv")
    parser.add_argument("--maxtime", type=float, default=1200.0, help="Time limit TOTALE (su tutte le iterazioni) in secondi")
    parser.add_argument("--no-warmstart", action="store_true", help="Disattiva il warm start euristico iniziale")
    parser.add_argument("--no-strong-master", action="store_true", help="Usa il vincolo di conflitto debole aggregato invece di quello forte pairwise")
    parser.add_argument("--extra-preseed", type=int, default=0, help="Tagli di pre-seeding statici extra prima della prima iterazione")
    parser.add_argument("--max-iter", type=int, default=100000, help="Numero massimo di iterazioni (ririsolvi completi)")
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

    model = gp.Model("MultiTree_Benders_MFPC_v4")
    model.Params.OutputFlag = 1

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
        print(f"[DIAG] Tagli di pre-seeding extra (statici): {n_extra}")

    if not args.no_warmstart:
        scelti = greedy_feasible_x(archi, conflitti_per_arco)
        cap_greedy = [archi[a] if a in scelti else 0.0 for a in archi_list]
        flow_greedy, _ = solver_mf.solve(cap_greedy)
        for a in archi_list:
            x[a].Start = 1.0 if a in scelti else 0.0
        v.Start = flow_greedy
        print(f"[DIAG] Warm start euristico: {len(scelti)} archi attivi, flusso={flow_greedy}")

    # NOTA: NESSUN model.Params.LazyConstraints -- e' l'intero punto di v4.
    # Ogni model.optimize() qui sotto ha quindi accesso al presolve completo
    # di Gurobi, dual reductions incluse.

    inizio = time.time()
    iterazione = 0
    tagli_totali = 0
    x_list = [x[a] for a in archi_list]

    while iterazione < args.max_iter:
        iterazione += 1
        tempo_rimanente = args.maxtime - (time.time() - inizio)
        if tempo_rimanente <= 0:
            print("[DIAG] Tempo totale esaurito.")
            break

        model.Params.TimeLimit = tempo_rimanente
        model.optimize()

        if model.Status == GRB.TIME_LIMIT and model.SolCount == 0:
            print("[DIAG] Time limit raggiunto senza soluzione in questa iterazione.")
            break
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print(f"[DIAG] Stato Gurobi inatteso: {model.Status}, interrompo.")
            break

        master_v = v.X
        x_val = [x[a].X for a in archi_list]

        cap_attiva = [archi[a] if xv > 0.5 else 0.0 for a, xv in zip(archi_list, x_val)]
        max_flow, visited_raw = solver_mf.solve(cap_attiva)

        print(f"[ITER {iterazione}] t={time.time()-inizio:.1f}s  master_v={master_v:.3f}  "
              f"max_flow_reale={max_flow:.3f}  status={model.Status}")

        if model.Status == GRB.OPTIMAL and master_v <= max_flow + TOL:
            print("[DIAG] Soluzione del master coerente col subproblem reale: ottimo globale raggiunto.")
            break
        if model.Status == GRB.TIME_LIMIT:
            print("[DIAG] Time limit raggiunto durante un'iterazione (soluzione non necessariamente ottima).")
            break

        # --- Taglio di ottimalita' forte (Prop. 4), stavolta come vincolo
        # NORMALE (non lazy): entra a far parte del modello per tutte le
        # iterazioni successive, con pieno accesso al presolve.
        cap_mod = [archi[a] * (U_totale_capacita * xv + 1) for a, xv in zip(archi_list, x_val)]
        _, visited_opt = solver_mf.solve(cap_mod)
        cut_idx = solver_mf.cut_arc_indices(visited_opt)
        model.addConstr(v <= gp.quicksum(archi[archi_list[k]] * x[archi_list[k]] for k in cut_idx))
        tagli_totali += 1

        if max_flow < TOL:
            cap_mod2 = [m_archi_totale * archi[a] * xv + 1 for a, xv in zip(archi_list, x_val)]
            _, visited_conn = solver_mf.solve(cap_mod2)
            cut_idx2 = solver_mf.cut_arc_indices(visited_conn)
            model.addConstr(gp.quicksum(x[archi_list[k]] for k in cut_idx2) >= 1)
            tagli_totali += 1

        # Warm start della prossima iterazione con la soluzione corrente
        for a, xv in zip(archi_list, x_val):
            x[a].Start = xv
        v.Start = master_v

    tempo_totale = time.time() - inizio

    print("\n--- RISULTATI FINALI (v4, multi-tree) ---")
    print(f"Status Gurobi (ultima iterazione): {model.Status}")
    print(f"Flusso Ottimo v*: {v.X if model.SolCount > 0 else 'N/A'}")
    print(f"Tempo Totale Esecuzione: {tempo_totale:.4f} s")
    print(f"Iterazioni (ririsolvi completi del master): {iterazione}")
    print(f"Tagli aggiunti in totale: {tagli_totali}")


if __name__ == "__main__":
    main()
