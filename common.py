"""
Funzioni condivise da generatore.py / strong.py / weak.py / benders.py:
lettura istanza (formato CSV comune) e Edmonds-Karp (usato solo da benders.py
per il subproblem e la costruzione dei tagli).

Metti questo file nella STESSA cartella di generatore.py / strong.py / weak.py
/ benders.py (progetto PyCharm).
"""

import csv
import os
from collections import defaultdict, deque

TOL = 1e-5


def carica_istanza(cartella):
    """Legge meta.txt / archi.csv / conflitti.csv da `cartella`.

    Ritorna:
        n: numero di nodi
        archi: dict (i,j) -> capacita
        conflitti_per_arco: dict (i,j) -> lista di (k,l) in conflitto con (i,j)
                             (simmetrico: se (i,j) e' in conflitto con (k,l),
                             compare sia in conflitti_per_arco[(i,j)] che in
                             conflitti_per_arco[(k,l)])
    """
    with open(os.path.join(cartella, "meta.txt")) as f:
        n = int(f.read().strip())

    archi = {}
    with open(os.path.join(cartella, "archi.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            i, j = int(row["i"]), int(row["j"])
            archi[(i, j)] = float(row["capacita"])

    conflitti_per_arco = defaultdict(list)
    with open(os.path.join(cartella, "conflitti.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            i, j, k, l = int(row["i"]), int(row["j"]), int(row["k"]), int(row["l"])
            conflitti_per_arco[(i, j)].append((k, l))

    return n, archi, conflitti_per_arco


def costruisci_adiacenze(archi):
    """out_adj[i] = lista di archi (i,j) uscenti da i
    in_adj[i]  = lista di archi (j,i) entranti in i"""
    out_adj = defaultdict(list)
    in_adj = defaultdict(list)
    for (i, j) in archi:
        out_adj[i].append((i, j))
        in_adj[j].append((i, j))
    return out_adj, in_adj


def edmonds_karp(n, cap, s, t):
    """BFS augmenting path (Edmonds-Karp). `cap` e' un dict (i,j)->capacita
    (solo archi con capacita' > 0 vanno passati, quelli assenti valgono 0).
    Ritorna (flusso_max, insieme_nodi_raggiungibili_da_s_nel_residuo_finale)."""
    rescap = defaultdict(float)
    for (i, j), c in cap.items():
        rescap[(i, j)] += c
        rescap.setdefault((j, i), 0.0)

    adj = defaultdict(set)
    for (i, j) in list(rescap.keys()):
        adj[i].add(j)
        adj[j].add(i)

    flusso = 0.0
    while True:
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == t:
                break
            for w in adj[u]:
                if w not in parent and rescap[(u, w)] > TOL:
                    parent[w] = u
                    q.append(w)
        if t not in parent:
            break

        path_cap = float("inf")
        curr = t
        while curr != s:
            prev = parent[curr]
            path_cap = min(path_cap, rescap[(prev, curr)])
            curr = prev

        curr = t
        while curr != s:
            prev = parent[curr]
            rescap[(prev, curr)] -= path_cap
            rescap[(curr, prev)] += path_cap
            curr = prev

        flusso += path_cap

    visited = set(parent.keys())
    return flusso, visited


def min_cut_archi(archi, visited):
    """Archi (i,j) della topologia originale che attraversano il taglio
    (i raggiungibile da s nel residuo, j no)."""
    return [(i, j) for (i, j) in archi if i in visited and j not in visited]
