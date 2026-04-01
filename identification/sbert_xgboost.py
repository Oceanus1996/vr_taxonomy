

import os
import time
import webbrowser

from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
import xgboost as xgb
import joblib
import json
from imblearn.over_sampling import SMOTE
import psutil
import re
import urllib.parse

input_file = f"github_clawer\\merged_all.csv"    


def biaozhu(num,input_file = f"github_clawer\\merged_all.csv"
            ,output_file = f"D:\\vr_detection_2.0\data_process\\labeled_sample.csv"):
 
    df = pd.read_csv(input_file)

 
    sample_df = df.sample(n=num, random_state=42) 


    # sample_df = df.head(100)


    sample_df.to_csv(output_file, index=False)






def read_and_sbert(file_path):
    """
    数据加载+
    """

    text_col = "text"

    df = pd.read_csv(file_path)
    if text_col not in df.columns:


    texts = df[text_col].astype(str).tolist()
    print(f"✅ Loaded {len(texts)} rows for embedding.")
==
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    print(f"✅ Embeddings shape: {embeddings.shape}")
    
    unit_embeddings = normalize(embeddings, norm='l2', axis=1)
    print(f"✅ Normalized (unit) vectors shape: {unit_embeddings.shape}")

    os.makedirs("data_process", exist_ok=True)  
        "data_process/text_unit_embeddings_with_index.npz",
        embeddings=embeddings,
        index=df.index.values
    )


    emb_df = pd.concat([
        df[[text_col]].reset_index(drop=True),
        pd.DataFrame(unit_embeddings, columns=[f"dim_{i + 1}" for i in range(unit_embeddings.shape[1])])
    ], axis=1)

    emb_df.to_csv(f"data_process\\text_embeddings.csv", index=False, encoding="utf-8-sig")
    print("💾 Saved to 'data_process\\text_embeddings.csv' and 'text_unit_embeddings.csv'")


def train_sbert_xgboost(csv_path, text_col="text", label_col="label",
                        model_out=f"data_process\\xgb_model.json", params_out=f"data_process\\xgb_params.json"):


 
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
  
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    embeddings = sbert.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
 

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(embeddings,labels)

    print(f"After SMOTE: {np.bincount(y_resampled)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )


    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
        "scale_pos_weight": len(y_train) / sum(y_train),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
    }

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("🎯 Training complete.")



    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    importance = clf.feature_importances_
    top_features = np.argsort(importance)[-10:][::-1]
    print("\n🔥 Top 10 Important Dimensions:")
    for i, idx in enumerate(top_features):
        print(f"  {i+1:02d}. dim_{idx} → importance={importance[idx]:.4f}")

    
    
    os.makedirs("data_process/model", exist_ok=True)
  
    clf.save_model("data_process/model/xgb_model.json")
  
    joblib.dump(sbert, "data_process/model/sbert_encoder.pkl")
   
    joblib.dump({"encoder": sbert, "model": clf}, "data_process/model/final_model.pkl")

    params_out = "data_process/model/xgb_params.json"
    with open(params_out, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)



    return clf, sbert, params

def run_model(model_path, source_npy, raw_data, output_csv="predicted_results.csv"):

    print(f"📦 Loading model from {model_path} ...")
    model = xgb.Booster()
    model.load_model(model_path)


    data = np.load(source_npy)
    X = data["embeddings"]
    idx = data["index"]
  
    dmatrix = xgb.DMatrix(X)
    probs = model.predict(dmatrix)

    preds = (probs >= 0.5).astype(int)


    df_result = pd.DataFrame({
        "index": idx,
        "predicted_label": preds,
        "reproducible_prob": probs
    })

    
    merged_csv(raw_data, df_result, output_csv)

    return df_result


def merged_csv(raw_data, pred_df, output):
  
    df_raw = pd.read_csv(raw_data)
   
    merged = df_raw.reset_index().merge(
        pred_df,
        left_on="index",
        right_on="index",
        how="left"
    )
    
    merged.to_csv(output, index=False)
        print(merged.head())
    return merged



def create_high_confidence(num):

    df = pd.read_csv("prediction_result.csv")
    high_conf = df[df["reproducible_prob"] > num]
    high_conf.to_csv("high_confidence_results.csv", index=False)
   

def create_confidence_level(num1,num2):

    df = pd.read_csv("prediction_result.csv")
    high_conf = df[(df["reproducible_prob"] > num1) & (df["reproducible_prob"] <= num2)]
    high_conf.to_csv(f"high_confidence_results{num1}_{num2}.csv", index=False)
 
def filter_app(file="high_confidence_results.csv"):

   
    df = pd.read_csv(file)
 
    df["repo_link"] = "https://github.com/" + df["repo"]

    repo_counts = df["repo"].value_counts().reset_index()
    print("repo_counts",repo_counts)
    repo_counts.columns = ["repo", "count"]


    unique_repos = (
        df.drop_duplicates(subset=["repo"])
          .merge(repo_counts, on="repo", how="left")
          .reset_index(drop=True)
    )


    unique_repos[["repo", "repo_link", "count"]].to_csv("unique_high_conf_repos.csv", index=False)
    print("💾 已保存 unique_high_conf_repos.csv（含计数）")


def open_in_patch(file, batch_size=5, wait_time=30):

   
    df = pd.read_csv(file)
    if "repo_link" not in df.columns:
    
    links = df["repo_link"].dropna().tolist()



    for i in range(0, len(links), batch_size):
        batch = links[i:i + batch_size]
     
        for url in batch:
  
            webbrowser.open_new_tab(url)
        if i + batch_size < len(links):
            
            time.sleep(wait_time)


def record_open_github(output_txt="opened_repos.txt"):

    urls = []
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline']).lower()
            if "github.com" in cmdline:
                found = re.findall(r"https?://github\.com/[^\s\"']+", cmdline)
                urls.extend(found)
        except Exception:
            pass

    urls = list(set(urls))
    if not urls:
      
        return []

    repos = []
    for url in urls:
        path = urllib.parse.urlparse(url).path.strip("/")
        m = re.match(r"([\w\-_]+/[\w\-_]+)", path)
        if m:
            repos.append(m.group(1).lower())

    repos = sorted(set(repos))
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(repos))



    return repos


# ------------------------------------------------------------
# 🧩 第二步：根据仓库名筛选原始 CSV
# ------------------------------------------------------------
def filter_by_repos(repo_txt="opened_repos_90.txt",
                    raw_csv=r"D:\vr_detection_2.0\pythonProject\prediction_result.csv",
                    col_name="repo",
                    new_csv=r"D:\vr_detection_2.0\pythonProject\filter_new90.csv"):
  

    with open(repo_txt, "r", encoding="utf-8") as f:
        text = f.read()

    repos = [r.strip().lower() for r in text.replace(",", "\n").splitlines() if r.strip()]
 
    df = pd.read_csv(raw_csv)
    if col_name not in df.columns:
        raise ValueError(f" {raw_csv}  '{col_name}'")

    # 3️⃣ 标准化并筛选
    df[col_name] = df[col_name].astype(str).str.lower().str.strip()
    filtered = df[df[col_name].isin(repos)]

    # 4️⃣ 保存结果
    filtered.to_csv(new_csv, index=False)

    return filtered


"""重新进行超参的训练"""


def encode_file_npz(input_csv, output_npz, text_col="text"):
  

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=[text_col])

    texts = df[text_col].astype(str).tolist()
    idx = df.index.to_numpy()  
    print(f"📄 Loaded {len(texts)} rows from {input_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings)

    # 保存 npz
    np.savez(output_npz, embeddings=embeddings, index=idx)
    pri

    return embeddings, idx


def encode_file(input_csv, output_npy, text_col="text"):
 

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=[text_col])
    texts = df[text_col].astype(str).tolist()
    print(f"📄 Loaded {len(texts)} rows from {input_csv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.array(embeddings)

    np.save(output_npy, embeddings)
    return embeddings

def apply_smote(X_train, y_train, k_neighbors=3, random_state=42):
   
    print("🔧 Applying SMOTE only on TRAIN set...")
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    print(f"   ➤ Before SMOTE: {np.bincount(y_train)}")
    print(f"   ➤ After  SMOTE: {np.bincount(y_resampled)}")
    print("✅ SMOTE completed.\n")
    return X_resampled, y_resampled


import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(input_file, input_npy, label_col="visibility"):
    df = pd.read_csv(input_file)

  
    df[label_col] = df[label_col].map({"visual": 1, "non_visual": 0})

    # X = embeddings
    X = embeddings
    y = df[label_col].values


    X_main, X_test, y_main, y_test = train_test_split(
        X, y, test_size=200/len(df), stratify=y, random_state=42
    )

  
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_main, y_main, test_size=200/len(X_main), stratify=y_main, random_state=42
    )

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

def train_and_valid(input_file, input_npy):

    # 1️⃣ 加载文件
    df = pd.read_csv(input_file)
    embeddings = np.load(input_npy)

   

    print(f"📌 Total loaded samples: {len(df)}")

    (X_train_raw, y_train_raw), (X_dev, y_dev), (X_test, y_test) = split_data(input_file,input_npy)

    print(f"📌 train = {len(X_train_raw)}, dev = {len(X_dev)}")

    # 4️⃣ SMOTE only on train
    sm = SMOTE(k_neighbors=3, random_state=42)
    X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)
    print(f"🔧 After SMOTE: {np.bincount(y_train)}")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5]
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
    }

    grid = list(ParameterGrid(param_grid))
    print(f"🔍 Total hyperparameter combinations: {len(grid)}")

    best_f1 = -1
    best_model = None
    best_params = None


    for params in grid:
        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_dev)
        f1 = f1_score(y_dev, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = params

       
        print("\n📊 Dev Classification Report:")
        print(classification_report(y_dev, best_model.predict(X_dev)))
==
        probs_dev = best_model.predict_proba(X_dev)[:, 1]

        best_thr, df_thr = tune_threshold(y_dev, probs_dev, metric="f1")

        print("\n📈 Threshold Scan on Dev Set:")
        print(df_thr)
        print(f"\n🏆 B
        dmatrix_test = xgb.DMatrix(X_test)
        probs_test = best_model.predict(dmatrix_test)
        preds_test = (probs_test >= best_thr).astype(int)

        print("\n📊 Final Test Report (Using Best Threshold):")
        print(classification_report(y_test, preds_test))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds_test))

        
        os.makedirs("record", exist_ok=True)
        best_model.save_model("record/xgb_best_model.json")
        with open("record/xgb_best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
        with open("record/best_threshold.txt", "w") as f:
            f.write(str(best_thr))

        return best_model, best_params, best_thr, (X_test, y_test)

def tune_threshold(y_true, probs, metric="f1"):
    """
    在验证集上扫描不同阈值，选择表现最好的阈值
    -----------------------------------------------------
    参数:
        y_true : ndarray   → 验证集真实标签 (0/1)
        probs  : ndarray   → 模型在验证集上的预测概率
        metric : str       → 使用哪个指标选阈值 ("f1", "recall", "precision")

    返回:
        best_thr : float            → 最优阈值
        df_thr   : DataFrame        → 所有阈值的指标结果表格
    """

    thresholds = np.linspace(0.1, 0.9, 17)  # 0.1, 0.15, ..., 0.9
    records = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )

        records.append([t, precision, recall, f1])

    df_thr = pd.DataFrame(records, columns=["threshold", "precision", "recall", "f1"])

    # 根据 metric 选择最佳阈值
    if metric == "f1":
        best_thr = df_thr.iloc[df_thr["f1"].idxmax()]["threshold"]
    elif metric == "recall":
        best_thr = df_thr.iloc[df_thr["recall"].idxmax()]["threshold"]
    elif metric == "precision":
        best_thr = df_thr.iloc[df_thr["precision"].idxmax()]["threshold"]
    else:
        raise ValueError("metric must be one of: f1, recall, precision")

    return best_thr, df_thr


def evaluate_on_test(model, X_test, y_test):
    """使用最终模型在 test 集进行评估"""

    print("\n🔍 Evaluating on Test Set...\n")

    y_pred = model.predict(X_test)

    print("📊 Test Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))





if __name__ ==  "__main__" :
    # print(torch.__version__)
    # print(torch.version.cuda)
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    #read_and_sbert(input_file)

    #train_sbert_xgboost(f"D:\\vr_detection_2.0\\data_process\\labeled_sample_300.csv","text","reproducible")
    #run_model(f"D:\\vr_detection_2.0\\pythonProject\\data_process\\model\\xgb_model.json",f"data_process\\text_unit_embeddings.npy")

    #生成置信度>0.9，筛选独立仓库，先找bug多的

    #filter_app()
    #open_in_patch("unique_high_conf_repos.csv")
    #filter_by_repos(f"D:\\vr_detection_2.0\\opened_repos_90.txt")

    #挑选50-90的80存在pythonProject/high_confidence_results0.5_0.9.csv（然后开始复现）
    create_confidence_level(0.5,0.9)

    #筛出仓库数
    filter_app("high_confidence_results0.5_0.9.csv")
