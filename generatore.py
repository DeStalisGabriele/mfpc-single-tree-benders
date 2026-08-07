"""
Generatore di istanze per il Maximum Flow Problem with Partial Conflicts,
porting fedele (stessa logica, stessi parametri) di Generatore_Istanze_MFPC.mos.

NOTA IMPORTANTE: usa il generatore di numeri casuali di Python (non quello di
Xpress Mosel), quindi anche con lo stesso SEED_CASUALE l'istanza NON sara'
identica bit-per-bit a quella generata da Mosel. E' pero' riproducibile
all'interno di Python (stesso seed = sempre la stessa istanza) e con le
stesse proprieta' statistiche (stessa topologia a catena + archi long-range
con probabilita' p, stessi conflitti tra archi non di catena con probabilita'
d). Va bene per confrontare Strong/Weak/Benders TRA LORO su un'unica istanza
Python, ma non per confrontare numeri con vecchi run fatti in Mosel.

Uso:
    python generatore.py --n 80 --p 0.6 --d 0.3 --seed 12345 --dir .
"""

import argparse
import csv
import os
import random


def genera_istanza(n, prob_archi, prob_conflitti, seed):
    random.seed(seed)

    archi = {}       # (i,j) -> capacita
    e_catena = {}     # (i,j) -> True se arco di catena (i, i+1)

    # 1. Archi di catena: garantiscono che la rete sia sempre connessa da 1 a n
    for i in range(1, n):
        archi[(i, i + 1)] = round(1 + random.random() * 4)
        e_catena[(i, i + 1)] = True

    # 2. Archi long-range casuali (i, j) con j > i+1, presenti con probabilita' p
    for i in range(1, n + 1):
        for j in range(i + 2, n + 1):
            if random.random() < prob_archi:
                archi[(i, j)] = round(5 + random.random() * 15)
                e_catena[(i, j)] = False

    # 3. Conflitti: solo tra coppie di archi NON di catena (stesso criterio del
    # generatore Mosel: j <> i+1 and l <> k+1), ogni coppia non ordinata una
    # sola volta, con probabilita' d. Simmetrico nella matrice risultante.
    archi_non_catena = [a for a in archi if not e_catena[a]]
    conflitti = []
    for idx, (i, j) in enumerate(archi_non_catena):
        for (k, l) in archi_non_catena[idx + 1:]:
            if random.random() < prob_conflitti:
                conflitti.append((i, j, k, l))
                conflitti.append((k, l, i, j))

    return archi, conflitti


def salva_istanza(cartella, n, archi, conflitti):
    os.makedirs(cartella, exist_ok=True)

    with open(os.path.join(cartella, "meta.txt"), "w") as f:
        f.write(f"{n}\n")

    with open(os.path.join(cartella, "archi.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "capacita"])
        for (i, j), cap in archi.items():
            w.writerow([i, j, cap])

    with open(os.path.join(cartella, "conflitti.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "k", "l"])
        for (i, j, k, l) in conflitti:
            w.writerow([i, j, k, l])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80, help="Numero di nodi")
    parser.add_argument("--p", type=float, default=0.6, help="Probabilita' archi long-range (densita' rete)")
    parser.add_argument("--d", type=float, default=0.3, help="Probabilita' conflitto tra due archi non di catena")
    parser.add_argument("--seed", type=int, default=12345, help="Seed per riproducibilita'")
    parser.add_argument("--dir", default=".", help="Cartella di output (meta.txt/archi.csv/conflitti.csv)")
    args = parser.parse_args()

    print(f"Generazione rete in corso con N={args.n}, p={args.p}, d={args.d}, seed={args.seed}")
    archi, conflitti = genera_istanza(args.n, args.p, args.d, args.seed)
    salva_istanza(args.dir, args.n, archi, conflitti)

    n_conflitti_coppie = len(conflitti) // 2
    print(f"Generazione completata: {len(archi)} archi, {n_conflitti_coppie} coppie in conflitto")
    print(f"File salvati in '{os.path.abspath(args.dir)}': meta.txt, archi.csv, conflitti.csv")


if __name__ == "__main__":
    main()
