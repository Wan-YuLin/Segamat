# -*- coding: utf-8 -*-


# --- IMPORT PACKAGES ---
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




# --- DEFINE FUNCTIONS ---
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)



def preprocess_data(abun_path, metab_path, params_path, pseudo=1):
    # Load abundance
    abun = pd.read_csv(abun_path).set_index(["Name", "final_chosen"])
    metab = pd.read_csv(metab_path).set_index("Name")
    
    # Load Best Parameters got from First Layer
    params = pd.read_csv(params_path).set_index("metabolite")
    
    # Filter for intersection
    common_metabs = params.index.intersection(metab.columns)
    abun = abun[abun.index.get_level_values('Name').isin(common_metabs)]
    metab = metab[common_metabs]
    
    # Log2 transform
    abun_t = np.log2(abun + pseudo).T
    metab_t = np.log2(metab + pseudo).T
    
    return abun_t, metab_t, params



def second_RF_layer(x_data, y_data, params):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

    model = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        criterion=params['criterion'],
        max_depth=None if pd.isna(params['max_depth']) else int(params['max_depth']),
        min_samples_leaf=int(params['min_samples_leaf'])
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rho, p = spearmanr(y_test, y_pred)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
        "Spearman_Rho": rho,
        "p_value": p
    }




def run_pipeline(x_abun, y_metab, param_df):
    results = []
    for metabolite in y_metab.columns:
        combined = pd.concat([x_abun[metabolite], y_metab[metabolite]], axis=1, join='inner')
        combined = combined[(combined.iloc[:, :-1].sum(axis=1) != 0) & (combined.iloc[:, -1] != 0)]
        
        if combined.empty: continue
        
        X, y = combined.iloc[:, :-1], combined.iloc[:, -1]
        metrics = second_RF_layer(X, y, param_df.loc[metabolite])
        metrics["Metabolite"] = metabolite
        results.append(metrics)

    df = pd.DataFrame(results).set_index("Metabolite")
    
    # FDR Correction
    if not df.empty:
        _, p_adj, _, _ = multipletests(df["p_value"].fillna(1), method='fdr_bh')
        df["Adjusted_p"] = p_adj
        
    return df





# --- EXECUTION ---
if __name__ == "__main__":
    
    config = load_config()
    pipelines = ["16S", "ref-shotgun", "denovo-contigs", "denovo-MAGs"]
       
    for pipe in pipelines:
        paths = config["layer2_paths"][pipe] 
        
        x, y, p = preprocess_data(
            paths["pathway"], 
            paths["metabolite"], 
            paths["params"]
        )
        
        report = run_pipeline(x, y, p)
        report.to_csv(f"./results/validation_{pipe}_report.csv")
