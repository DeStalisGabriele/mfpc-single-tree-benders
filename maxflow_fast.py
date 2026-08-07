"""
FastMaxFlow -- max-flow/min-cut solver ottimizzato per essere chiamato migliaia
di volte all'interno della callback lazy-constraint del Single-Tree Benders.

IDEA CHIAVE (perche' e' piu' veloce della versione in common.py):
La topologia del grafo (l'insieme di archi (i,j)) e' SEMPRE la stessa per
tutta la durata della risoluzione: cio' che cambia ad ogni chiamata e' solo
il VETTORE delle capacita' (rete attiva, oppure capacita' perturbate per il
taglio di ottimalita'/connettivita'). La versione precedente ricostruiva da
zero, ad ogni chiamata: la matrice sparsa CSR (righe/colonne/dati), un dict
Python per le capacita' residue e un dict di adiacenza basato su set --
tutto lavoro ripetuto migliaia di volte per una struttura che non cambia mai.

Questa versione costruisce la struttura sparsa (CSR) e la matrice di
adiacenza UNA SOLA VOLTA in __init__, e ad ogni chiamata a solve():
  1) sovrascrive in-place il solo array `.data` della CSR gia' allocata
     (nessuna nuova allocazione di righe/colonne);
  2) chiama scipy.sparse.csgraph.maximum_flow (Dinic compilato in C);
  3) ricostruisce la capacita' residua come matrice densa n x n riutilizzata
     (operazioni numpy vettorizzate, senza dict/set Python);
  4) fa la BFS del min-cut leggendo righe della matrice densa (numpy), non
     un dizionario hash Python.
  5) l'estrazione degli archi di taglio e' vettorizzata con numpy invece di
     un list-comprehension con test di appartenenza a un set per ogni arco.

Adatto a reti con n fino a qualche centinaio di nodi (la matrice residua
densa n x n e' irrilevante in memoria: 200x200 = 320 KB in float64). Per reti
molto piu' grandi servirebbe un'adiacenza sparsa anche per il residuo, ma per
la scala di questa tesi (n <= 100) la versione densa e' la piu' veloce.

Le capacita' devono essere intere (requisito di scipy): valori frazionari
(es. provenienti da un punto frazionario del rilassamento LP, usati per i
tagli generati a nodi non interi) vengono arrotondati all'intero piu' vicino,
esattamente come gia' faceva common.py per i valori 0/1 delle soluzioni
intere.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow as _scipy_maximum_flow

TOL = 1e-9


class FastMaxFlow:
    def __init__(self, n, archi_list, s, t):
        """
        n: numero di nodi (1..n)
        archi_list: lista di tuple (i,j) in un ORDINE FISSO -- questo stesso
                    ordine deve essere usato per costruire l'array di
                    capacita' passato a solve().
        s, t: sorgente e pozzo (1-indicizzati, come nel resto del progetto)
        """
        self.n = n
        self.m = len(archi_list)
        self.archi_list = list(archi_list)
        self.s0 = s - 1
        self.t0 = t - 1

        arr = np.array(self.archi_list, dtype=np.int64)
        self.edge_from = arr[:, 0] - 1
        self.edge_to = arr[:, 1] - 1

        # CSR con sparsita' FISSA (dati placeholder, sovrascritti ad ogni solve())
        placeholder = np.ones(self.m, dtype=np.int64)
        self._csr = csr_matrix((placeholder, (self.edge_from, self.edge_to)), shape=(n, n))
        if self._csr.nnz != self.m:
            raise ValueError(
                "archi_list contiene coppie (i,j) duplicate: non supportato "
                "(il grafo del progetto e' semplice, non dovrebbe accadere)."
            )

        # Mappa: indice in archi_list -> posizione nell'array .data della CSR
        # (la conversione COO->CSR di scipy ordina le entrate per riga poi
        # colonna, quindi l'ordine di .data NON coincide con l'ordine di
        # archi_list; questa mappa si calcola una sola volta).
        indptr, indices = self._csr.indptr, self._csr.indices
        lookup = {}
        for r in range(n):
            for k in range(indptr[r], indptr[r + 1]):
                lookup[(r, int(indices[k]))] = k
        self.pos_map = np.array(
            [lookup[(int(self.edge_from[idx]), int(self.edge_to[idx]))] for idx in range(self.m)],
            dtype=np.int64,
        )

        # Buffer riutilizzato per la matrice di capacita' residua (denso).
        self._residual = np.zeros((n, n), dtype=np.float64)

    def solve(self, capacities):
        """
        capacities: array-like di lunghezza self.m, capacita' nello STESSO
        ordine di self.archi_list (float o int; arrotondate internamente).

        Ritorna (flusso_massimo, visited) dove `visited` e' un array booleano
        di lunghezza n (indice 0-based) -- nodi raggiungibili da s nel grafo
        residuo finale.
        """
        cap_int = np.rint(np.asarray(capacities, dtype=np.float64)).astype(np.int64)
        cap_int[cap_int < 0] = 0

        self._csr.data[self.pos_map] = cap_int
        result = _scipy_maximum_flow(self._csr, self.s0, self.t0)

        flow_coo = result.flow.tocoo()
        pos_mask = flow_coo.data > 0
        pr = flow_coo.row[pos_mask]
        pc = flow_coo.col[pos_mask]
        pv = flow_coo.data[pos_mask]

        res = self._residual
        res.fill(0.0)
        res[self.edge_from, self.edge_to] = cap_int  # capacita' residua diretta (prima del flusso)
        res[pr, pc] -= pv                              # sottrae il flusso usato in avanti
        res[pc, pr] += pv                              # aggiunge capacita' residua all'indietro

        visited = np.zeros(self.n, dtype=bool)
        visited[self.s0] = True
        frontier = [self.s0]
        while frontier:
            nxt = []
            for u in frontier:
                cand = np.nonzero(res[u] > TOL)[0]
                for w in cand:
                    if not visited[w]:
                        visited[w] = True
                        nxt.append(int(w))
            frontier = nxt

        return float(result.flow_value), visited

    def cut_arc_indices(self, visited):
        """Indici (in archi_list) degli archi che attraversano il taglio
        (i raggiungibile da s, j no). Vettorizzato con numpy."""
        mask = visited[self.edge_from] & ~visited[self.edge_to]
        return np.nonzero(mask)[0]

    def cut_arcs(self, visited):
        """Come cut_arc_indices, ma ritorna direttamente le tuple (i,j)."""
        return [self.archi_list[k] for k in self.cut_arc_indices(visited)]
