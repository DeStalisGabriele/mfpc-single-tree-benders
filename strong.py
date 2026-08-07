"""
MFPC Strong Formulation -- porting fedele di MFPC_Strong_Formulation.mos
(flow conservation classico + capacita' gate da x + vincolo di conflitto
FORTE Eq. 3: una coppia x(i,j) + x(k,l) <= 1 per ogni coppia di archi in
conflitto).

Uso:
    python strong.py --dir . --timelimit 3600
"""

import argparse
import time

import gurobipy as gp
from gurobipy import GRB

from common import carica_istanza, costruisci_adiacenze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Cartella con meta.txt/archi.csv/conflitti.csv")
    parser.add_argument("--timelimit", type=float, default=3600, help="Time limit in secondi (come nel .mos originale)")
    args = parser.parse_args()

    n, archi, conflitti_per_arco = carica_istanza(args.dir)
    s, t = 1, n
    out_adj, in_adj = costruisci_adiacenze(archi)

    print(f"[DIAG] Nodi: {n}  Archi attivi: {len(archi)}")

    model = gp.Model("MFPC_Strong_Formulation")

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

    # 5. Conflitti STRONG (Eq. 3): un vincolo per ogni coppia NON ordinata
    # (evita di scrivere due volte lo stesso vincolo, dato che
    # conflitti_per_arco e' simmetrico)
    visti = set()
    n_vincoli_conflitto = 0
    for (i, j), lista in conflitti_per_arco.items():
        for (k, l) in lista:
            pair = frozenset([(i, j), (k, l)])
            if pair in visti:
                continue
            visti.add(pair)
            if (i, j) in x and (k, l) in x:
                model.addConstr(x[i, j] + x[k, l] <= 1, name=f"conflitto_{i}_{j}_{k}_{l}")
                n_vincoli_conflitto += 1

    print(f"[DIAG] Vincoli di conflitto forti aggiunti: {n_vincoli_conflitto}")

    model.Params.TimeLimit = args.timelimit

    t0 = time.time()
    print("Risoluzione in corso con Gurobi MILP (STRONG)...")
    model.optimize()
    tempo = time.time() - t0

    print("\n--- RISULTATI STRONG FORMULATION ---")
    print(f"Stato del solver: {model.Status}  (2=OPTIMAL, 9=TIME_LIMIT)")
    print(f"Flusso Ottimo Trovato: {model.ObjVal if model.SolCount > 0 else 'N/A'}")
    print(f"Best Bound: {model.ObjBound if model.SolCount > 0 else 'N/A'}")
    print(f"MIP Gap: {model.MIPGap if model.SolCount > 0 else 'N/A'}")
    print(f"Tempo di calcolo (sec): {tempo:.6f}")
    print(f"Nodi esplorati (B&B Tree): {model.NodeCount:.0f}")


if __name__ == "__main__":
    main()