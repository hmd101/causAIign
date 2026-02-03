# This script quick and dirty read the masters_classified_strategy_metrics.csv outputted by the script aggregate mormative strategies
# to detect parameter shifts / signatures per non-normative and normative reasoning agents
import pandas as pd, numpy as np, os, sys
from math import comb

def binom_test_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    probs=[comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(n+1)]
    pk=probs[k]
    pval=sum(pr for pr in probs if pr<=pk)
    return min(1.0, pval)

def mcnemar_p(lost, gained):
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    return binom_test_two_sided(k, n, 0.5)

def normalize(df):
    df = df.copy()
    df["prompt"] = df["prompt_category"].replace(
        {"single_numeric_response":"numeric","CoT":"cot","numeric":"numeric"}
    )
    df["exp"] = df["experiment"].replace({
        "random_abstract":"Abstract",
        "rw17_indep_causes":"RW17",
        "abstract_overloaded_lorem_de":"Abstract-Over",
        "rw17_overloaded_e":"RW17-Over"
    })
    return df

def shares_table(df):
    g = df.groupby(["exp","prompt"]).agg(
        N_tested=("agent","nunique"),
        N_rows=("agent","size"),
        N_norm=("normative_reasoner","sum")
    ).reset_index()
    g["pct_norm"] = g["N_norm"]/g["N_tested"]*100.0
    return g.sort_values(["exp","prompt"])

def pairwise_transition(df, exp_from, prompt_from, exp_to, prompt_to):
    A = df[(df["exp"]==exp_from) & (df["prompt"]==prompt_from)].set_index("agent")[["normative_reasoner"]]
    B = df[(df["exp"]==exp_to) & (df["prompt"]==prompt_to)].set_index("agent")[["normative_reasoner"]]
    overlap = sorted(set(A.index) & set(B.index))
    if not overlap:
        return {"Pair": f"{exp_from}({prompt_from})->{exp_to}({prompt_to})", "N_overlap":0,
                "Kept":0,"Lost":0,"Gained":0,"Stayed_non":0,
                "From_pct":float("nan"),"To_pct":float("nan"),"mcnemar_p":1.0}
    A = A.loc[overlap]; B = B.loc[overlap]
    kept = int(((A["normative_reasoner"]) & (B["normative_reasoner"])).sum())
    lost = int(((A["normative_reasoner"]) & (~B["normative_reasoner"])).sum())
    gained = int(((~A["normative_reasoner"]) & (B["normative_reasoner"])).sum())
    stayed_non = int(((~A["normative_reasoner"]) & (~B["normative_reasoner"])).sum())
    from_pct = float(A["normative_reasoner"].mean()*100.0)
    to_pct   = float(B["normative_reasoner"].mean()*100.0)
    pval = mcnemar_p(lost, gained)
    return {"Pair": f"{exp_from}({prompt_from})->{exp_to}({prompt_to})", "N_overlap":len(overlap),
            "Kept":kept,"Lost":lost,"Gained":gained,"Stayed_non":stayed_non,
            "From_pct":from_pct,"To_pct":to_pct,"mcnemar_p":pval}

def paired_deltas(df, exp_from, prompt_from, exp_to, prompt_to, cols=("loocv_r2","b","m_bar")):
    A = df[(df["exp"]==exp_from) & (df["prompt"]==prompt_from)].set_index("agent")
    B = df[(df["exp"]==exp_to) & (df["prompt"]==prompt_to)].set_index("agent")
    overlap = A.index.intersection(B.index)
    out = {"Pair": f"{exp_from}({prompt_from})->{exp_to}({prompt_to})", "N_overlap":len(overlap)}
    for c in cols:
        d = (B.loc[overlap, c] - A.loc[overlap, c]).dropna()
        if d.empty:
            out[f"med_delta_{c}"] = float("nan")
            out[f"q1_delta_{c}"]  = float("nan")
            out[f"q3_delta_{c}"]  = float("nan")
        else:
            out[f"med_delta_{c}"] = float(d.median())
            out[f"q1_delta_{c}"]  = float(d.quantile(0.25))
            out[f"q3_delta_{c}"]  = float(d.quantile(0.75))
    return out

def main(in_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = normalize(pd.read_csv(in_path))

    shares = shares_table(df)
    shares.to_csv(os.path.join(out_dir, "normative_shares.csv"), index=False)

    pairs = [
        ("RW17","numeric","Abstract","numeric"),
        ("RW17","numeric","RW17-Over","numeric"),
        ("Abstract","numeric","Abstract-Over","numeric"),
        ("RW17","numeric","RW17","cot"),
        ("Abstract","numeric","Abstract","cot"),
        ("RW17-Over","numeric","RW17-Over","cot"),
        ("Abstract-Over","numeric","Abstract-Over","cot"),
    ]
    trans_rows = [pairwise_transition(df, *p) for p in pairs]
    pd.DataFrame(trans_rows).to_csv(os.path.join(out_dir, "pairwise_transitions.csv"), index=False)

    delta_rows = [paired_deltas(df, *p) for p in pairs]
    pd.DataFrame(delta_rows).to_csv(os.path.join(out_dir, "paired_deltas.csv"), index=False)

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "masters_classified_strategy_metrics.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "q4_outputs"
    main(in_path, out_dir)
