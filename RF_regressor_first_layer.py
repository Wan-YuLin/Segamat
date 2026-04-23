# -*- coding: utf-8 -*-


# --- IMPORT PACKAGES ---
import os
import gc
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# --- IMPORT MODULES ---
from evaluation import filter_predictable_models



# --- DEFINE FUNCTIONS ---
def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def load_and_clean_data(path, is_csv=True, sheet_name=None):
    df = pd.read_csv(path, index_col=0) if is_csv else pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.astype(str).str.replace(".", "-", regex=False)
    return df

def preprocess_abundance(abun_df, meta_ls_path, sheet, metab_abun_df, pseudo=1):
    # Get Mapping
    ls = pd.read_excel(meta_ls_path, sheet_name=sheet)
    ls = ls[ls["presence"] == True][["Metabolite name", "Features"]].dropna().drop_duplicates()
    
    # Map and Align
    mapped = ls.merge(abun_df, left_on="Features", right_index=True, how="inner")
    mapped = mapped.sort_values(by="Metabolite name").set_index(["Metabolite name", "Features"])
    
    metab = metab_abun_df[metab_abun_df.index.isin(mapped.index.get_level_values("Metabolite name"))].sort_index()
    
    return np.log2(mapped.T + pseudo), np.log2(metab.T + pseudo)


def run_pipeline(x_abun, y_metab, pipeline_name, config):
    out_path = os.path.join(config["paths"]["output_dir"], pipeline_name)
    os.makedirs(out_path, exist_ok=True)
    
    param_grid = config["rf_params"]
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for metabolite in y_metab.columns:
        print(f"Starting {pipeline_name} modeling for: {metabolite}")
        
        # Merge and Filter
        combined = x_abun[metabolite].merge(y_metab[metabolite], left_index=True, right_index=True)
        combined = combined[(combined.sum(axis=1) != 0) & (combined.iloc[:, -1] != 0)]
        
        if combined.empty: continue
        X, y = combined.iloc[:, :-1], combined.iloc[:, -1]
        results = []

        for seed in range(0, 100):
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

            for params in combinations:
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(x_train, y_train)
                
                y_tr_pred, y_ts_pred = model.predict(x_train), model.predict(x_test)
                
                results.append({
                    "random_state": seed,
                    **params,
                    "train_R2": r2_score(y_train, y_tr_pred),
                    "test_R2": r2_score(y_test, y_ts_pred)
                })
            del x_train, x_test, y_train, y_test
            gc.collect()
        
        pd.DataFrame(results).to_csv(f"{out_path}/{metabolite}.csv", index=False)






# --- EXECUTION ---
if __name__ == "__main__":
    
    config = load_config()
    p = config["layer1_paths"]

    # 16S approach
    x, y = preprocess_abundance(
        load_and_clean_data(p["six_pathway"]), 
        p["metabolite_list"], "16S_metabolite_list",
        load_and_clean_data(p["six_metabolite"]), 1
    )
    run_pipeline(x, y, "16S", config)

    # ref-shotgun approach 
    x, y = preprocess_abundance(
        load_and_clean_data(p["ref-shot_pathway"]), 
        p["metabolite_list"], "ref-shotgun_metabolite_list",
        load_and_clean_data(p["ref-shot_metabolite"]), 1
    )
    run_pipeline(x, y, "ref-shotgun", config)
    
    # de novo-contigs approach 
    x, y = preprocess_abundance(
        load_and_clean_data(p["contigs_pathway"]), 
        p["metabolite_list"], "contigs_metabolite_list",
        load_and_clean_data(p["contigs_metabolite"]), 1
    )
    run_pipeline(x, y, "denovo-contigs", config)
    
    # de novo-mags approach 
    x, y = preprocess_abundance(
        load_and_clean_data(p["mags_pathway"]), 
        p["metabolite_list"], "mags_metabolite_list",
        load_and_clean_data(p["mags_metabolite"]), 1
    )
    run_pipeline(x, y, "denovo-mags", config)