"""Solvers of the two optimization problems (Methods, Section 3.4).

Flight schedule optimization (1_schedule_optimization.py)
    Each flight may leave up to 1,800 s earlier or later. Flights that pass
    the same farm at the same time compete for its capacity, so the shifts are
    chosen to minimise the total time during which two flights are inside the
    same farm's range together. Two solvers are available:

    adam    (default, used for the paper) gradient descent on the total
            pairwise overlap with PyTorch's Adam; the shifts are continuous and
            clipped to +/-1,800 s after each step.
    gurobi  (optional) the same objective written as a mixed-integer linear
            program and solved with Gurobi in consecutive time windows
            (rolling horizon); flights decided in a window keep their shift in
            the next ones. Needs gurobipy and a Gurobi licence.

Farm-and-flight choice optimization (3_farm_flight_selection.py)
    Given penetration rates for farms and flights, choose which farms get a
    SoPhAr system and which flights get a receiving antenna so as to maximise
    the capacity-weighted overlap time sum(p_cap_safe x overlap seconds) of
    the selected pairs. Two solvers are available:

    greedy  (default, used for the paper) multi-start alternating local
            search: start from the farms and flights with the largest
            potential, then alternately pick the best flights for the chosen
            farms and the best farms for the chosen flights until the objective
            stops improving; keep the best of five starts.
    gurobi  (optional) the integer program: binary farm, flight and pair
            variables, a pair counts only when its farm and its flight are both
            chosen, at most rho_F|F| farms and rho_I|I| flights. At 10%/10% its
            objective is within 2% of the greedy search.
"""
import time

import numpy as np
import pandas as pd

MAX_SHIFT = 1800        # s, the largest allowed schedule shift


# ----------------------------------------------------------------------------
# Flight schedule optimization
# ----------------------------------------------------------------------------
def isolate_conflicts(df, max_shift=MAX_SHIFT):
    """Pairs of flights that could overlap at a farm after shifting.

    `df` has one row per flight-farm crossing with integer Entry_Time and
    Exit_Time in seconds. Two crossings of the same farm can be made to
    overlap only if the gap between them is at most 2 x max_shift. Returns the
    candidate pairs and the flights that take part in none (their shift is 0).
    """
    started = time.time()
    df_sorted = df.sort_values(["Solar_Farm_ID", "Entry_Time"]).reset_index(drop=True)
    pairs = {k: [] for k in ("solar_farm_id", "flight_id_1", "flight_id_2", "entry_time_1",
                             "exit_time_1", "entry_time_2", "exit_time_2")}
    all_flights = set(df["Trip_ID"].unique())
    at_risk = set()
    for farm_id, group in df_sorted.groupby("Solar_Farm_ID"):
        f_ids = group["Trip_ID"].values
        entries = group["Entry_Time"].values
        exits = group["Exit_Time"].values
        n = len(f_ids)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                # sorted by entry: once the gap exceeds 2 x max_shift, so do
                # all later crossings
                if entries[j] - exits[i] > max_shift * 2:
                    break
                pairs["solar_farm_id"].append(farm_id)
                pairs["flight_id_1"].append(f_ids[i])
                pairs["flight_id_2"].append(f_ids[j])
                pairs["entry_time_1"].append(entries[i])
                pairs["exit_time_1"].append(exits[i])
                pairs["entry_time_2"].append(entries[j])
                pairs["exit_time_2"].append(exits[j])
                at_risk.add(f_ids[i])
                at_risk.add(f_ids[j])
    df_pairs = pd.DataFrame(pairs)
    safe_flights = list(all_flights - at_risk)
    print("conflict pairs: %s among %s flights; %s flights need no shift  (%.0f s)"
          % (format(len(df_pairs), ","), format(len(at_risk), ","),
             format(len(safe_flights), ","), time.time() - started), flush=True)
    return df_pairs, safe_flights


def set_seed(seed=2026):
    """Fix every random seed and make PyTorch deterministic."""
    import os
    import random

    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def shifts_adam(df_pairs, safe_flights, max_shift=MAX_SHIFT, epochs=1500, lr=5.0,
                device=None):
    """Minimise the total pairwise overlap with Adam.

    Returns a table of flight_id and time_shift_seconds (rounded to 0.01 s)
    and the loss history.
    """
    import torch

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("adam on %s: %d epochs, learning rate %g" % (device, epochs, lr), flush=True)

    at_risk = list(set(df_pairs["flight_id_1"]).union(set(df_pairs["flight_id_2"])))
    flight_to_idx = {f: i for i, f in enumerate(at_risk)}
    idx1 = torch.tensor(df_pairs["flight_id_1"].map(flight_to_idx).values, dtype=torch.long,
                        device=device)
    idx2 = torch.tensor(df_pairs["flight_id_2"].map(flight_to_idx).values, dtype=torch.long,
                        device=device)
    E1 = torch.tensor(df_pairs["entry_time_1"].values, dtype=torch.float32, device=device)
    L1 = torch.tensor(df_pairs["exit_time_1"].values, dtype=torch.float32, device=device)
    E2 = torch.tensor(df_pairs["entry_time_2"].values, dtype=torch.float32, device=device)
    L2 = torch.tensor(df_pairs["exit_time_2"].values, dtype=torch.float32, device=device)

    x = torch.zeros(len(at_risk), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    history = []
    started = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        overlap = torch.relu(torch.minimum(L1 + x[idx1], L2 + x[idx2])
                             - torch.maximum(E1 + x[idx1], E2 + x[idx2]))
        loss = overlap.sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(-max_shift, max_shift)
        history.append(loss.item())
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print("  epoch %5d/%d  total overlap %.2f s" % (epoch + 1, epochs, loss.item()),
                  flush=True)
    print("adam finished in %.0f s" % (time.time() - started), flush=True)

    final = x.detach().cpu().numpy()
    rows = [{"flight_id": f, "time_shift_seconds": round(final[i], 2)}
            for f, i in flight_to_idx.items()]
    rows += [{"flight_id": f, "time_shift_seconds": 0.0} for f in safe_flights]
    return pd.DataFrame(rows), history


def shifts_gurobi(df_pairs, safe_flights, max_shift=MAX_SHIFT, window_hours=2,
                  big_m=100000, threads=4, time_limit=None):
    """Minimise the total pairwise overlap with Gurobi, window by window.

    For a pair (1, 2) with shifts x1, x2 the overlap W >= U - V, W >= 0, with
    U = min(L1 + x1, L2 + x2) and V = max(E1 + x1, E2 + x2) linearised with two
    binaries and big-M. Pairs are solved in windows of `window_hours` by the
    entry time of their first crossing; shifts fixed in earlier windows are
    kept. Returns the shift table and the objective of every window.
    """
    import gurobipy as gp
    from gurobipy import GRB

    df_pairs = df_pairs.sort_values("entry_time_1", kind="stable")
    t0, t1 = df_pairs["entry_time_1"].min(), df_pairs["entry_time_1"].max()
    window = window_hours * 3600
    locked, objectives = {}, []

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", threads)
    if time_limit:
        env.setParam("TimeLimit", time_limit)
    env.start()

    start, k = t0, 1
    while start <= t1:
        end = start + window
        chunk = df_pairs[(df_pairs["entry_time_1"] >= start) & (df_pairs["entry_time_1"] < end)]
        if len(chunk) == 0:
            start = end
            continue
        flights = set(chunk["flight_id_1"]).union(set(chunk["flight_id_2"]))
        m = gp.Model("window_%d" % k, env=env)
        x = {}
        n_new = 0
        for f in flights:
            if f in locked:
                x[f] = m.addVar(lb=locked[f], ub=locked[f], vtype=GRB.CONTINUOUS)
            else:
                x[f] = m.addVar(lb=-max_shift, ub=max_shift, vtype=GRB.CONTINUOUS)
                n_new += 1
        W = []
        for row in chunk.itertuples(index=False):
            x1, x2 = x[row.flight_id_1], x[row.flight_id_2]
            E1, L1, E2, L2 = row.entry_time_1, row.exit_time_1, row.entry_time_2, row.exit_time_2
            U = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
            V = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
            w = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
            b = m.addVar(vtype=GRB.BINARY)
            c = m.addVar(vtype=GRB.BINARY)
            m.addConstr(U <= L1 + x1)
            m.addConstr(U <= L2 + x2)
            m.addConstr(U >= L1 + x1 - big_m * b)
            m.addConstr(U >= L2 + x2 - big_m * (1 - b))
            m.addConstr(V >= E1 + x1)
            m.addConstr(V >= E2 + x2)
            m.addConstr(V <= E1 + x1 + big_m * c)
            m.addConstr(V <= E2 + x2 + big_m * (1 - c))
            m.addConstr(w >= U - V)
            W.append(w)
        m.setObjective(gp.quicksum(W), GRB.MINIMIZE)
        m.optimize()
        if m.SolCount > 0:
            objectives.append(m.ObjVal)
            for f in flights:
                if f not in locked:
                    locked[f] = x[f].X
            print("  window %02d: %s pairs, %d new flights, overlap %.0f s (status %d)"
                  % (k, format(len(chunk), ","), n_new, m.ObjVal, m.Status), flush=True)
        else:
            print("  window %02d: no solution (status %d); its new flights keep shift 0"
                  % (k, m.Status), flush=True)
        m.dispose()
        start, k = end, k + 1
    env.dispose()

    rows = [{"flight_id": f, "time_shift_seconds": v} for f, v in locked.items()]
    solved = set(locked)
    at_risk = set(df_pairs["flight_id_1"]).union(set(df_pairs["flight_id_2"]))
    rows += [{"flight_id": f, "time_shift_seconds": 0.0}
             for f in list(at_risk - solved) + list(safe_flights)]
    return pd.DataFrame(rows), objectives


def total_overlap(df, entry_col, exit_col):
    """Total pairwise overlap (s) of crossings of the same farm."""
    total = 0.0
    df_sorted = df.sort_values(["Solar_Farm_ID", entry_col]).reset_index(drop=True)
    for _, group in df_sorted.groupby("Solar_Farm_ID"):
        entries, exits = group[entry_col].values, group[exit_col].values
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if entries[j] >= exits[i]:
                    break
                overlap = min(exits[i], exits[j]) - entries[j]
                if overlap > 0:
                    total += overlap
    return total


# ----------------------------------------------------------------------------
# Farm-and-flight choice optimization
# ----------------------------------------------------------------------------
def select_greedy(df, p_farm_rate, p_flight_rate, n_starts=5, max_iter=30):
    """Multi-start alternating local search.

    `df` has Solar_Farm_ID, Trip_ID, p_cap_safe and Duration_Overlap_Sec.
    Returns (selected farms, selected flights, objective).
    """
    df["w_ij"] = df["p_cap_safe"] * df["Duration_Overlap_Sec"]
    k_farm = max(1, int(df["Solar_Farm_ID"].nunique() * p_farm_rate))
    k_flight = max(1, int(df["Trip_ID"].nunique() * p_flight_rate))

    top_edges = df.nlargest(n_starts, "w_ij")
    best_power, best_farms, best_flights = -1, set(), set()
    for i in range(n_starts):
        seed = top_edges.iloc[i]
        farms, flights = {seed["Solar_Farm_ID"]}, {seed["Trip_ID"]}
        if k_farm > 1:
            farms.update(df.groupby("Solar_Farm_ID")["w_ij"].sum().nlargest(k_farm).index)
        if k_flight > 1:
            flights.update(df.groupby("Trip_ID")["w_ij"].sum().nlargest(k_flight).index)

        prev = 0
        for _ in range(max_iter):
            gains = df[df["Solar_Farm_ID"].isin(farms)].groupby("Trip_ID")["w_ij"].sum()
            flights = set(gains.nlargest(k_flight).index)
            gains = df[df["Trip_ID"].isin(flights)].groupby("Solar_Farm_ID")["w_ij"].sum()
            farms = set(gains.nlargest(k_farm).index)
            current = df[df["Solar_Farm_ID"].isin(farms) & df["Trip_ID"].isin(flights)]["w_ij"].sum()
            if current <= prev:
                break
            prev = current
        if current > best_power:
            best_power, best_farms, best_flights = current, farms, flights
    return best_farms, best_flights, best_power


def select_gurobi(df, p_farm_rate, p_flight_rate, time_limit=None, mip_gap=None,
                  threads=None, verbose=False):
    """The farm-and-flight choice integer program, solved with Gurobi.

    Returns (selected farms, selected flights, objective); (None, None, 0) when
    Gurobi finds no solution.
    """
    import gurobipy as gp
    from gurobipy import GRB

    df["w_ij"] = df["p_cap_safe"] * df["Duration_Overlap_Sec"]
    farms = df["Solar_Farm_ID"].unique()
    flights = df["Trip_ID"].unique()
    k_farm = int(len(farms) * p_farm_rate)
    k_flight = int(len(flights) * p_flight_rate)

    m = gp.Model("farm_flight_choice")
    m.Params.OutputFlag = 1 if verbose else 0
    if time_limit:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if threads:
        m.Params.Threads = threads

    x = m.addVars(farms.tolist(), vtype=GRB.BINARY, name="farm")
    y = m.addVars(flights.tolist(), vtype=GRB.BINARY, name="flight")
    rows = list(range(len(df)))
    z = m.addVars(rows, vtype=GRB.BINARY, name="transmit")
    w = df["w_ij"].to_numpy()
    farm_of = df["Solar_Farm_ID"].to_numpy()
    flight_of = df["Trip_ID"].to_numpy()

    m.setObjective(gp.quicksum(w[r] * z[r] for r in rows), GRB.MAXIMIZE)
    m.addConstr(x.sum() <= k_farm, name="farm_quota")
    m.addConstr(y.sum() <= k_flight, name="flight_quota")
    for r in rows:
        # a pair transmits only if its farm and its flight are both selected
        m.addConstr(z[r] <= x[farm_of[r]])
        m.addConstr(z[r] <= y[flight_of[r]])
    m.optimize()

    if m.SolCount == 0:
        return None, None, 0
    if m.Status != GRB.OPTIMAL:
        print("  gurobi status %d, gap %.3g%%: using the best solution found"
              % (m.Status, 100 * m.MIPGap), flush=True)
    sel_farms = {f for f in farms if x[f].X > 0.5}
    sel_flights = {f for f in flights if y[f].X > 0.5}
    return sel_farms, sel_flights, m.ObjVal
