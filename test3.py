### Libraries

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score

### Read the data, extract features and label columns (4 features, 1000 rows)

data = pd.read_csv("data.csv")

X = data.drop("quality_label", axis=1)
Y = data["quality_label"]

### Binary encoding: "high" = 1, everything else (low, medium) = 0 (high : others == 343 : 657)

Y_enc = (Y == "high").astype(int)

print("Class distribution:")
print(Y_enc.value_counts())

### Split: 60% train, 20% validation, 20% test (600 : 200 : 200)

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y_enc, test_size=0.4, random_state=67)
X_val, X_test, Y_val, Y_test     = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=67)

### Scale the features (standardize them)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

### C candidates (for non-linear kernels consider higher values of C as mentioned in description for assigment)

C_values_linear = [0.005, 0.01 , 0.05, 0.1, 0.5, 1, 5, 10 ,50, 100]
C_values_kernel = [0.005, 0.01 , 0.05, 0.1, 0.5, 1, 5, 10 ,50, 100, 500, 1000, 5000, 10000]

### Choose correct b for Fb score, since classes are imbalanced (65.7% vs 34.3%), use Fb for selection of the best C as mentioned in description
### Pick b==2, which weights recall higher than accuracy, because accuracy is misleading for imbalanced classes (trivial classifier is one of the best using this metric)

BETA = 2

### Evaluation function

def evaluate(model, X, Y, beta=BETA):
    Y_pred = model.predict(X)

    acc  = accuracy_score(Y, Y_pred)
    prec = precision_score(Y, Y_pred, average="binary", zero_division=0) 
    rec  = recall_score(Y, Y_pred, average="binary", zero_division=0)
    fb   = fbeta_score(Y, Y_pred, beta=beta, average="binary", zero_division=0)
    sv   = model.n_support_.sum()

    return acc, prec, rec, fb, sv

### Define 4 models with different kernels, coefficients are picked as 1, as mentioned in description

kernels = {
    "Model I (Linear)":   ({"kernel": "linear"},                          C_values_linear),
    "Model II (Poly)":    ({"kernel": "poly", "degree": 3, "coef0": 1},   C_values_kernel),
    "Model III (RBF)":    ({"kernel": "rbf",  "gamma": 1},                C_values_kernel),
    "Model IV (Sigmoid)": ({"kernel": "sigmoid", "coef0": 1},             C_values_kernel),
}

final_results = []

for model_name, (params, C_values) in kernels.items():

    print( f"\n{model_name}")
    
    best_model  = None
    best_val_fb = -np.inf
    best_C      = None

    val_rows = []

    for C in C_values:
        
        model = SVC(C=C, class_weight="balanced", **params)
        model.fit(X_train, Y_train)

        tr_acc,  tr_prec,  tr_rec,  tr_fb,  tr_sv  = evaluate(model, X_train, Y_train)
        val_acc, val_prec, val_rec, val_fb, val_sv  = evaluate(model, X_val, Y_val)

        val_rows.append({
            "C":               C,
            "Train Acc":       round(tr_acc,  4),
            "Train Prec":      round(tr_prec, 4),
            "Train Recall":    round(tr_rec,  4),
            f"Train F{BETA}":  round(tr_fb,   4),
            "Train SVs":       tr_sv,
            "Val Acc":         round(val_acc,  4),
            "Val Prec":        round(val_prec, 4),
            "Val Recall":      round(val_rec,  4),
            f"Val F{BETA}":    round(val_fb,   4),
        })

        ### Choosing the best C: pick the one with highest Fb on validation set as mentioned b4

        if val_fb > best_val_fb:
            best_val_fb = val_fb
            best_model  = model
            best_C      = C

    print(pd.DataFrame(val_rows).to_string(index=False))
    print(f" Best C = {best_C}  (Val F{BETA} = {best_val_fb:.4f})\n")

    ### Check the best models on the test set 
    te_acc, te_prec, te_rec, te_fb, te_sv = evaluate(best_model, X_test, Y_test)

    final_results.append({
        "Model":           model_name,
        "Best C":          best_C,
        "Test Accuracy":   round(te_acc,  4),
        "Test Precision":  round(te_prec, 4),
        "Test Recall":     round(te_rec,  4),
        f"Test F{BETA}":   round(te_fb,   4),
        "Support Vectors": te_sv,
    })

### Final comparison among the best models 

df_final = pd.DataFrame(final_results)
print(df_final.to_string(index=False))

### Choose best model out of the best models again by Fb score

best_row = df_final.loc[df_final[f"Test F{BETA}"].idxmax()]
print(f"\nBest model: {best_row['Model']} with Test F{BETA} = {best_row[f'Test F{BETA}']}")

### Manual evaluation for Model III (RBF) with C=0.5 on the Test Set (For testing other models on test set)
model_rbf_05 = SVC(C=0.1, kernel="sigmoid", coef0=1, class_weight="balanced")
model_rbf_05.fit(X_train, Y_train)

# Check the scores of this example model
te_acc_05, te_prec_05, te_rec_05, te_fb_05, te_sv_05 = evaluate(model_rbf_05, X_test, Y_test)

print(f"\n--- Specific Evaluation: \n")
print(f"Test Accuracy:  {te_acc_05:.4f}")
print(f"Test Precision: {te_prec_05:.4f}")
print(f"Test Recall:    {te_rec_05:.4f}")
print(f"Test F{BETA} Score: {te_fb_05:.4f}")
print(f"Support Vectors: {te_sv_05}")