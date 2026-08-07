"""
Single-Tree Benders Decomposition per il Maximum Flow Problem with Conflicts
(Suvak, Altinel & Aras, 2020, EJOR 287:410-437) -- porting da FICO Xpress
Mosel a Gurobi/Python.

DIFFERENZA CHIAVE rispetto alla versione Mosel:
Su Xpress, per iniettare tagli lazy in modo sicuro con addcut() dentro una
callback XPRS_CB_OPTNODE, e' stato necessario disattivare presolve, cut
nativi, euristiche e dual reductions (altrimenti il solver puo' "chiudere"
il gap prematuramente sul modello incompleto). Questo elimina quasi tutta
l'accelerazione nativa del solver, ed e' la causa della lentezza osservata
su istanze dense (es. 80 nodi).

Gurobi risolve il problema alla radice con il meccanismo LazyConstraints:
impostando model.Params.LazyConstraints = 1 e aggiungendo i tagli con
model.cbLazy(...) dentro una callback MIPSOL (eseguita solo quando il
solver trova una NUOVA soluzione intera candidata), Gurobi garantisce che
quella soluzione venga rifiutata se il taglio e' violato, MA continua a
usare presolve, i propri cut nativi (clique, cover, MIR, ...) ed euristiche
per il resto della ricerca. Non serve disattivare nulla.

Uso:
    python benders.py --dir . --maxtime 1200
"""

import argparse
import time

import gurobipy as gp
from gurobipy import GRB

from common import carica_istanza, edmonds_karp, min_cut_archi, TOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Cartella con meta.txt/archi.csv/conflitti.csv")
    parser.add_argument("--maxtime", type=float, default=1200.0, help="Time limit in secondi")
    args = parser.parse_args()

    n, archi, conflitti_per_arco = carica_istanza(args.dir)
    s, t = 1, n

    n_archi = len(archi)
    n_conflitti = sum(len(v) for v in conflitti_per_arco.values())
    print(f"[DIAG] Nodi: {n}  Archi attivi: {n_archi}  Coppie in conflitto lette: {n_conflitti}")

    U_totale_capacita = sum(archi.values())
    m_archi_totale = n_archi

    # --- Pre-seeding (Sezione 3.1.3 del paper) -----------------------------
    flusso_rilassato, visited_rilassato = edmonds_karp(n, archi, s, t)
    UB_relax = flusso_rilassato
    print(f"[DIAG] UB rilassata (rete completa, senza conflitti): {UB_relax}")
    cut0 = min_cut_archi(archi, visited_rilassato)

    # --- Modello master ------------------------------------------------------
    model = gp.Model("SingleTree_Benders_MFPC_Gurobi")

    x = model.addVars(archi.keys(), vtype=GRB.BINARY, name="x")
    v = model.addVar(lb=0.0, ub=UB_relax, vtype=GRB.CONTINUOUS, name="v")
    model.setObjective(v, GRB.MAXIMIZE)

    # Vincoli di conflitto deboli (Eq. 6 del paper)
    for (i, j), conflitti in conflitti_per_arco.items():
        if conflitti and (i, j) in x:
            k_conf = len(conflitti)
            model.addConstr(
                gp.quicksum(x[kl] for kl in conflitti if kl in x) + k_conf * x[i, j] <= k_conf,
                name=f"conflitto_{i}_{j}",
            )

    # Vincoli di pre-seeding dal min-cut della rilassata
    model.addConstr(v <= gp.quicksum(archi[a] * x[a] for a in cut0), name="preseed_v")
    model.addConstr(gp.quicksum(x[a] for a in cut0) >= 1, name="preseed_cut")

    # --- Parametri: NESSUN presolve/cuts/euristiche disattivati ------------
    model.Params.LazyConstraints = 1
    model.Params.TimeLimit = args.maxtime
    # MIPGap invariato (default) -- qui e' affidabile perche' i tagli lazy
    # vengono aggiunti PRIMA che una soluzione sia accettata come incumbent,
    # non dopo: il bound del solver resta valido.

    stato = {
        "call_counter": 0,
        "num_tagli": 0,
        "start_time": None,
        "last_heartbeat": 0.0,
    }

    def callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        stato["call_counter"] += 1
        now = time.time() - stato["start_time"]
        if now - stato["last_heartbeat"] > 5:
            nodi = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
            print(f"[HEARTBEAT] t={now:.1f}s  chiamate={stato['call_counter']}  "
                  f"tagli={stato['num_tagli']}  nodi={nodi:.0f}")
            stato["last_heartbeat"] = now

        master_v = model.cbGetSolution(v)
        x_val = {a: model.cbGetSolution(x[a]) for a in archi}

        cap_attiva = {a: archi[a] for a in archi if x_val[a] > 0.5}
        max_flow, visited = edmonds_karp(n, cap_attiva, s, t)

        if master_v > max_flow + TOL:
            # --- Taglio di ottimalita' FORTE (Prop. 4) ---
            cap_mod = {a: archi[a] * (U_totale_capacita * x_val[a] + 1) for a in archi}
            _, visited_opt = edmonds_karp(n, cap_mod, s, t)
            cut_archi = min_cut_archi(archi, visited_opt)

            model.cbLazy(v <= gp.quicksum(archi[a] * x[a] for a in cut_archi))
            stato["num_tagli"] += 1

            # --- Taglio di connettivita' FORTE (Prop. 2-3), solo se la rete
            # attiva e' sconnessa (max_flow ~ 0) ---
            if max_flow < TOL:
                cap_mod2 = {a: m_archi_totale * archi[a] * x_val[a] + 1 for a in archi}
                _, visited_conn = edmonds_karp(n, cap_mod2, s, t)
                cut_archi2 = min_cut_archi(archi, visited_conn)

                model.cbLazy(gp.quicksum(x[a] for a in cut_archi2) >= 1)
                stato["num_tagli"] += 1

    stato["start_time"] = time.time()
    print("--- Inizio Ottimizzazione Single-Tree Benders (Gurobi/Python) ---")
    model.optimize(callback)

    tempo_totale = time.time() - stato["start_time"]

    print("\n--- RISULTATI FINALI ---")
    print(f"Status Gurobi: {model.Status}  (2 = OPTIMAL, 9 = TIME_LIMIT)")
    print(f"Flusso Ottimo v*: {model.ObjVal if model.SolCount > 0 else 'N/A'}")
    print(f"MIP Gap: {model.MIPGap if model.SolCount > 0 else 'N/A'}")
    print(f"Tempo Totale Esecuzione: {tempo_totale:.4f} s")
    print(f"Tagli Benders iniettati: {stato['num_tagli']}")
    print(f"Chiamate totali alla callback (soluzioni intere candidate): {stato['call_counter']}")
    print(f"Nodi B&B esplorati: {model.NodeCount:.0f}")


if __name__ == "__main__":
    main()
