"""
MFPC Weak Formulation -- stessa struttura flow-conservation di strong.py, ma
con il vincolo di conflitto AGGREGATO (Eq. 6 del paper Suvak et al. 2020):
per ogni arco (i,j), sum_{(k,l) in conflitto con (i,j)} x(k,l) + |conflitti| *
x(i,j) <= |conflitti|. E' la stessa disuguaglianza usata come master problem
nel Benders (vedi benders.py), qui risolta in un unico MILP compatto.

Uso:
    python weak.py --dir . --timelimit 3600
"""

import argparse
import time

import gurobipy as gp
from gurobipy import GRB

from common import carica_istanza, costruisci_adiacenze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Cartella con meta.txt/archi.csv/conflitti.csv")
    parser.add_argument("--timelimit", type=float, default=3600, help="Time limit in secondi")
    args = parser.parse_args()

    n, archi, conflitti_per_arco = carica_istanza(args.dir)
    s, t = 1, n
    out_adj, in_adj = costruisci_adiacenze(archi)

    print(f"[DIAG] Nodi: {n}  Archi attivi: {len(archi)}")

    model = gp.Model("MFPC_Weak_Formulation")

    f = model.addVars(archi.keys(), lb=0.0, vtype=GRB.CONTINUOUS, name="f")
    x = model.addVars(archi.keys(), vtype=GRB.BINARY, name="x")
    v = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="v")
    model.setObjective(v, GRB.MAXIMIZE)

    # 1. Sorgente
    model.addConstr(
        gp.quicksum(f[a] for a in out_adj[s]) - gp.quicksum(f[a] for a in in_adj[s]) == v,
        name="sorgente",
    )
    # 2. Pozzo
    model.addConstr(
        gp.quicksum(f[a] for a in out_adj[t]) - gp.quicksum(f[a] for a in in_adj[t]) == -v,
        name="pozzo",
    )
    # 3. Nodi intermedi
    for i in range(2, n):
        model.addConstr(
            gp.quicksum(f[a] for a in out_adj[i]) - gp.quicksum(f[a] for a in in_adj[i]) == 0,
            name=f"nodo_{i}",
        )

    # 4. Capacita'
    for a in archi:
        model.addConstr(f[a] <= archi[a] * x[a], name=f"cap_{a[0]}_{a[1]}")

    # 5. Conflitti WEAK (Eq. 6): un vincolo aggregato per ogni arco che ha
    # almeno un conflitto
    n_vincoli_conflitto = 0
    for (i, j), lista in conflitti_per_arco.items():
        if lista and (i, j) in x:
            k_conf = len(lista)
            model.addConstr(
                gp.quicksum(x[kl] for kl in lista if kl in x) + k_conf * x[i, j] <= k_conf,
                name=f"conflitto_{i}_{j}",
            )
            n_vincoli_conflitto += 1

    print(f"[DIAG] Vincoli di conflitto deboli aggiunti: {n_vincoli_conflitto}")

    model.Params.TimeLimit = args.timelimit

    t0 = time.time()
    print("Risoluzione in corso con Gurobi MILP (WEAK)...")
    model.optimize()
    tempo = time.time() - t0

    print("\n--- RISULTATI WEAK FORMULATION ---")
    print(f"Stato del solver: {model.Status}  (2=OPTIMAL, 9=TIME_LIMIT)")
    print(f"Flusso Ottimo Trovato: {model.ObjVal if model.SolCount > 0 else 'N/A'}")
    print(f"Best Bound: {model.ObjBound if model.SolCount > 0 else 'N/A'}")
    print(f"MIP Gap: {model.MIPGap if model.SolCount > 0 else 'N/A'}")
    print(f"Tempo di calcolo (sec): {tempo:.6f}")
    print(f"Nodi esplorati (B&B Tree): {model.NodeCount:.0f}")


if __name__ == "__main__":
    main()