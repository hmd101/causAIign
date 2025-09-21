# normative_cot_analysis.py

# use like this: python normative_cot_analysis.py normative_by_agent_matrix.csv cot_outputs/
# It writes two CSVs: cot_normative_shares.csv and cot_transitions_numeric_to_cot.csv.


import pandas as pd, os, sys
from math import comb

def mcnemar_exact(lost, gained):
    n = lost + gained
    if n == 0:
        return 1.0
    probs=[comb(n,i)*0.5**n for i in range(n+1)]
    pk=probs[min(lost,gained)]
    return float(min(1.0, sum(p for p in probs if p<=pk)))

def load_long(path):
    df = pd.read_csv(path)
    val_cols = [c for c in df.columns if "__" in c]
    long = df.melt(id_vars=["agent"], value_vars=val_cols, var_name="cond", value_name="normative")
    long = long.dropna(subset=["normative"])
    long[["experiment","prompt"]] = long["cond"].str.split("__", n=1, expand=True)
    long["experiment"] = long["experiment"].replace({
        "random_abstract":"Abstract",
        "rw17_indep_causes":"RW17",
        "rw17_overloaded_e":"RW17-Over",
        "abstract_overloaded_lorem_de":"Abstract-Over"
    })
    long["prompt"] = long["prompt"].replace({
        "CoT":"cot",
        "numeric":"numeric",
        "single_numeric_response":"numeric"
    })
    long["normative"] = long["normative"].astype(bool)
    return long

def shares(long):
    g = (long.groupby(["experiment","prompt"])
              .agg(N=("agent","nunique"),
                   N_norm=("normative", "sum"))
              .reset_index())
    g["pct_norm"] = g["N_norm"]/g["N"]*100
    return g.sort_values(["experiment","prompt"])

def transitions(long, exp):
    A = long[(long["experiment"]==exp) & (long["prompt"]=="numeric")].set_index("agent")["normative"]
    B = long[(long["experiment"]==exp) & (long["prompt"]=="cot")].set_index("agent")["normative"]
    inter = sorted(set(A.index)&set(B.index))
    a = A.loc[inter]; b = B.loc[inter]
    kept       = int(((a) & (b)).sum())
    lost       = int(((a) & (~b)).sum())
    gained     = int(((~a) & (b)).sum())
    stayed_non = int(((~a) & (~b)).sum())
    from_pct   = float(a.mean()*100)
    to_pct     = float(b.mean()*100)
    p          = mcnemar_exact(lost, gained)
    return {"experiment":exp,"N_overlap":len(inter),"from_pct":round(from_pct,1),
            "to_pct":round(to_pct,1),"kept":kept,"lost":lost,"gained":gained,
            "stayed_non":stayed_non,"mcnemar_p":p}

def main(in_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    long = load_long(in_path)
    shares(long).to_csv(os.path.join(out_dir, "cot_normative_shares.csv"), index=False)
    pd.DataFrame([transitions(long, e) for e in ["RW17","Abstract","RW17-Over","Abstract-Over"]]
                ).to_csv(os.path.join(out_dir, "cot_transitions_numeric_to_cot.csv"), index=False)

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv)>1 else "normative_by_agent_matrix.csv"
    out_dir = sys.argv[2] if len(sys.argv)>2 else "cot_outputs"
    main(in_path, out_dir)
