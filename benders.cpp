// benders.cpp
// Single-Tree Benders Decomposition per MFPC -- porting C++ per confronto
// diretto con l'ambiente del paper Suvak et al. 2020 (CPLEX 12.7 + C++ +
// LEMON). Stessa identica logica di benders.py (Python/Gurobi): master
// debole (Eq. 6), pre-seeding (Sezione 3.1.3), taglio di ottimalita' forte
// (Prop. 4) e taglio di connettivita' forte (Prop. 2-3), iniettati via
// LazyConstraints/addLazy in una callback su MIPSOL.
//
// Legge lo STESSO formato di istanza generato da generatore.py
// (meta.txt / archi.csv / conflitti.csv) -- nessuna riscrittura del
// generatore, cosi' il confronto usa esattamente la stessa istanza.
//
// Uso:
//   benders.exe <cartella_istanza> <timelimit_secondi>
//
// Esempio:
//   benders.exe run_istanze\n80_p0.6_d0.3 3600

#include "gurobi_c++.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <chrono>
#include <algorithm>

using namespace std;

static const double TOL = 1e-5;

struct Arco {
    int i, j;
    double cap;
};

long long chiave(int i, int j) { return (long long)i * 1000000LL + j; }

struct Istanza {
    int n = 0;
    vector<Arco> archi;
    unordered_map<long long, int> arco_idx;                       // (i,j) -> indice in archi
    unordered_map<long long, vector<pair<int,int>>> conflitti_per_arco; // (i,j) -> lista (k,l) in conflitto
};

// ===========================================================================
// LETTURA ISTANZA
// ===========================================================================
Istanza carica_istanza(const string& cartella) {
    Istanza ist;

    ifstream fmeta(cartella + "/meta.txt");
    if (!fmeta) { cerr << "Impossibile aprire meta.txt in " << cartella << endl; exit(1); }
    fmeta >> ist.n;
    fmeta.close();

    ifstream farchi(cartella + "/archi.csv");
    if (!farchi) { cerr << "Impossibile aprire archi.csv in " << cartella << endl; exit(1); }
    string line;
    getline(farchi, line); // header
    while (getline(farchi, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string si, sj, scap;
        getline(ss, si, ',');
        getline(ss, sj, ',');
        getline(ss, scap, ',');
        int i = stoi(si), j = stoi(sj);
        double cap = stod(scap);
        ist.arco_idx[chiave(i, j)] = (int)ist.archi.size();
        ist.archi.push_back({i, j, cap});
    }
    farchi.close();

    ifstream fconf(cartella + "/conflitti.csv");
    if (!fconf) { cerr << "Impossibile aprire conflitti.csv in " << cartella << endl; exit(1); }
    getline(fconf, line); // header
    while (getline(fconf, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string si, sj, sk, sl;
        getline(ss, si, ',');
        getline(ss, sj, ',');
        getline(ss, sk, ',');
        getline(ss, sl, ',');
        int i = stoi(si), j = stoi(sj), k = stoi(sk), l = stoi(sl);
        ist.conflitti_per_arco[chiave(i, j)].push_back({k, l});
    }
    fconf.close();

    return ist;
}

// ===========================================================================
// EDMONDS-KARP (BFS augmenting path) su matrice di capacita' (n+1)x(n+1),
// indicizzata 1..n. Ritorna il flusso massimo e riempie `visited` con i nodi
// raggiungibili da s nel grafo residuo finale (= partizione del min-cut).
// ===========================================================================
double edmonds_karp(int n, const vector<vector<double>>& cap, int s, int t, vector<bool>& visited) {
    vector<vector<double>> rescap = cap;
    double flusso = 0.0;
    vector<int> parent(n + 1);

    while (true) {
        fill(parent.begin(), parent.end(), -1);
        parent[s] = s;
        queue<int> q;
        q.push(s);
        bool trovato = false;
        while (!q.empty() && !trovato) {
            int u = q.front(); q.pop();
            for (int w = 1; w <= n && !trovato; w++) {
                if (parent[w] == -1 && rescap[u][w] > TOL) {
                    parent[w] = u;
                    q.push(w);
                    if (w == t) trovato = true;
                }
            }
        }
        if (parent[t] == -1) break;

        double path_cap = 1e18;
        for (int cur = t; cur != s; ) {
            int prev = parent[cur];
            path_cap = min(path_cap, rescap[prev][cur]);
            cur = prev;
        }
        for (int cur = t; cur != s; ) {
            int prev = parent[cur];
            rescap[prev][cur] -= path_cap;
            rescap[cur][prev] += path_cap;
            cur = prev;
        }
        flusso += path_cap;
    }

    visited.assign(n + 1, false);
    for (int i = 1; i <= n; i++) visited[i] = (parent[i] != -1);
    visited[s] = true;
    return flusso;
}

vector<int> min_cut_archi(const Istanza& ist, const vector<bool>& visited) {
    vector<int> cut;
    for (int idx = 0; idx < (int)ist.archi.size(); idx++) {
        const Arco& a = ist.archi[idx];
        if (visited[a.i] && !visited[a.j]) cut.push_back(idx);
    }
    return cut;
}

// ===========================================================================
// CALLBACK DI BENDERS SU MIPSOL -- stessa logica di benders.py
// ===========================================================================
class BendersCallback : public GRBCallback {
public:
    const Istanza& ist;
    int n, s, t;
    vector<GRBVar>& x;
    GRBVar& v;
    double U_totale;
    int m_totale;
    long num_tagli = 0;
    long call_counter = 0;
    chrono::steady_clock::time_point start_time;
    chrono::steady_clock::time_point last_heartbeat;

    BendersCallback(const Istanza& ist_, vector<GRBVar>& x_, GRBVar& v_, double U_, int m_)
        : ist(ist_), n(ist_.n), s(1), t(ist_.n), x(x_), v(v_), U_totale(U_), m_totale(m_) {
        start_time = chrono::steady_clock::now();
        last_heartbeat = start_time;
    }

protected:
    void callback() override {
        try {
            if (where != GRB_CB_MIPSOL) return;

            call_counter++;
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - start_time).count();
            if (chrono::duration<double>(now - last_heartbeat).count() > 5) {
                double nodi = getDoubleInfo(GRB_CB_MIPSOL_NODCNT);
                cout << "[HEARTBEAT] t=" << elapsed << "s  chiamate=" << call_counter
                     << "  tagli=" << num_tagli << "  nodi=" << nodi << endl;
                last_heartbeat = now;
            }

            double master_v = getSolution(v);
            vector<double> x_val(ist.archi.size());
            for (size_t idx = 0; idx < ist.archi.size(); idx++)
                x_val[idx] = getSolution(x[idx]);

            // --- Subproblem reale: rete attiva (x > 0.5) ---
            vector<vector<double>> cap_attiva(n + 1, vector<double>(n + 1, 0.0));
            for (size_t idx = 0; idx < ist.archi.size(); idx++) {
                if (x_val[idx] > 0.5) {
                    const Arco& a = ist.archi[idx];
                    cap_attiva[a.i][a.j] = a.cap;
                }
            }
            vector<bool> visited;
            double max_flow = edmonds_karp(n, cap_attiva, s, t, visited);

            if (master_v > max_flow + TOL) {
                // --- Taglio di ottimalita' FORTE (Prop. 4) ---
                vector<vector<double>> cap_mod(n + 1, vector<double>(n + 1, 0.0));
                for (size_t idx = 0; idx < ist.archi.size(); idx++) {
                    const Arco& a = ist.archi[idx];
                    cap_mod[a.i][a.j] = a.cap * (U_totale * x_val[idx] + 1);
                }
                vector<bool> visited_opt;
                edmonds_karp(n, cap_mod, s, t, visited_opt);
                vector<int> cut_opt = min_cut_archi(ist, visited_opt);

                GRBLinExpr expr = v;
                for (int idx : cut_opt) expr -= ist.archi[idx].cap * x[idx];
                addLazy(expr <= 0.0);
                num_tagli++;

                // --- Taglio di connettivita' FORTE (Prop. 2-3), solo se la
                // rete attiva e' sconnessa (max_flow ~ 0) ---
                if (max_flow < TOL) {
                    vector<vector<double>> cap_mod2(n + 1, vector<double>(n + 1, 0.0));
                    for (size_t idx = 0; idx < ist.archi.size(); idx++) {
                        const Arco& a = ist.archi[idx];
                        cap_mod2[a.i][a.j] = m_totale * a.cap * x_val[idx] + 1;
                    }
                    vector<bool> visited_conn;
                    edmonds_karp(n, cap_mod2, s, t, visited_conn);
                    vector<int> cut_conn = min_cut_archi(ist, visited_conn);

                    GRBLinExpr expr2 = 0;
                    for (int idx : cut_conn) expr2 += x[idx];
                    addLazy(expr2 >= 1.0);
                    num_tagli++;
                }
            }
        } catch (GRBException& e) {
            cout << "Errore nella callback: " << e.getMessage() << endl;
        }
    }
};

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char* argv[]) {
    string cartella = (argc > 1) ? argv[1] : ".";
    double maxtime = (argc > 2) ? stod(argv[2]) : 3600.0;

    Istanza ist = carica_istanza(cartella);
    int n = ist.n, s = 1, t = n;

    long n_conflitti = 0;
    for (auto& kv : ist.conflitti_per_arco) n_conflitti += (long)kv.second.size();
    cout << "[DIAG] Nodi: " << n << "  Archi attivi: " << ist.archi.size()
         << "  Coppie in conflitto lette: " << n_conflitti << endl;

    double U_totale = 0.0;
    for (auto& a : ist.archi) U_totale += a.cap;
    int m_totale = (int)ist.archi.size();

    // --- Pre-seeding (Sezione 3.1.3) ---
    vector<vector<double>> cap_piena(n + 1, vector<double>(n + 1, 0.0));
    for (auto& a : ist.archi) cap_piena[a.i][a.j] = a.cap;
    vector<bool> visited_rilassato;
    double UB_relax = edmonds_karp(n, cap_piena, s, t, visited_rilassato);
    cout << "[DIAG] UB rilassata (rete completa, senza conflitti): " << UB_relax << endl;
    vector<int> cut0 = min_cut_archi(ist, visited_rilassato);

    try {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);

        vector<GRBVar> x(ist.archi.size());
        for (size_t idx = 0; idx < ist.archi.size(); idx++) {
            string nome = "x_" + to_string(ist.archi[idx].i) + "_" + to_string(ist.archi[idx].j);
            x[idx] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, nome);
        }
        GRBVar v = model.addVar(0.0, UB_relax, 0.0, GRB_CONTINUOUS, "v");
        model.setObjective(GRBLinExpr(v), GRB_MAXIMIZE);

        // Vincoli di conflitto deboli (Eq. 6)
        for (auto& kv : ist.conflitti_per_arco) {
            long long key = kv.first;
            auto it = ist.arco_idx.find(key);
            if (it == ist.arco_idx.end()) continue;
            int idx_ij = it->second;
            const vector<pair<int,int>>& conflitti = kv.second;
            if (conflitti.empty()) continue;

            GRBLinExpr expr = 0;
            int k_conf = 0;
            for (auto& kl : conflitti) {
                auto it2 = ist.arco_idx.find(chiave(kl.first, kl.second));
                if (it2 != ist.arco_idx.end()) {
                    expr += x[it2->second];
                    k_conf++;
                }
            }
            expr += k_conf * x[idx_ij];
            model.addConstr(expr <= (double)k_conf);
        }

        // Pre-seeding
        GRBLinExpr expr_v = 0;
        for (int idx : cut0) expr_v += ist.archi[idx].cap * x[idx];
        model.addConstr(v <= expr_v, "preseed_v");

        GRBLinExpr expr_cut = 0;
        for (int idx : cut0) expr_cut += x[idx];
        model.addConstr(expr_cut >= 1.0, "preseed_cut");

        // Parametri (nessun presolve/cuts disattivato: e' il punto centrale)
        model.set(GRB_IntParam_LazyConstraints, 1);
        model.set(GRB_DoubleParam_TimeLimit, maxtime);

        BendersCallback cb(ist, x, v, U_totale, m_totale);
        model.setCallback(&cb);

        cout << "--- Inizio Ottimizzazione Single-Tree Benders (C++/Gurobi) ---" << endl;
        auto t_start = chrono::steady_clock::now();
        model.optimize();
        auto t_end = chrono::steady_clock::now();
        double tempo_totale = chrono::duration<double>(t_end - t_start).count();

        cout << "\n--- RISULTATI FINALI ---" << endl;
        int status = model.get(GRB_IntAttr_Status);
        cout << "Status Gurobi: " << status << "  (2 = OPTIMAL, 9 = TIME_LIMIT)" << endl;
        if (model.get(GRB_IntAttr_SolCount) > 0) {
            cout << "Flusso Ottimo v*: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
            cout << "MIP Gap: " << model.get(GRB_DoubleAttr_MIPGap) << endl;
        } else {
            cout << "Flusso Ottimo v*: N/A" << endl;
            cout << "MIP Gap: N/A" << endl;
        }
        cout << "Tempo Totale Esecuzione: " << tempo_totale << " s" << endl;
        cout << "Tagli Benders iniettati: " << cb.num_tagli << endl;
        cout << "Chiamate totali alla callback: " << cb.call_counter << endl;
        cout << "Nodi B&B esplorati: " << model.get(GRB_DoubleAttr_NodeCount) << endl;

    } catch (GRBException& e) {
        cout << "Errore Gurobi (" << e.getErrorCode() << "): " << e.getMessage() << endl;
        return 1;
    } catch (exception& e) {
        cout << "Errore: " << e.what() << endl;
        return 1;
    }

    return 0;
}
