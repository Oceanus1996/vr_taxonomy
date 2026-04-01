import json
import os
import joblib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid

from sbert_xgboost import split_data, apply_smote, tune_threshold

def encode_file_codebert(input_csv, output_npy, text_col="text"):
    df = pd.read_csv(input_csv).dropna(subset=[text_col])
    texts = df[text_col].astype(str).tolist()

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    embeddings = []
    for text in tqdm(texts, desc="Encoding with CodeBERT"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(emb.squeeze())

    embeddings = np.array(embeddings)
    np.save(output_npy, embeddings)          # ⭐ 保存为 NPY
    print(f"💾 Saved CodeBERT embeddings → {output_npy}")

    return embeddings

def train_and_valid_codebert(input_file, input_npy):
    """
    CodeBERT + Linear 模型的训练与验证：
    - 用 CodeBERT 的 CLS 向量做特征
    - Logistic Regression 做分类
    - 在 dev 上选最佳超参和阈值
    - 在 test 上做最终评估
    """

    # 1️⃣ 加载 CSV + CodeBERT 向量
    df = pd.read_csv(input_file)
    embeddings = np.load(input_npy)

    assert len(df) == len(embeddings), "CSV 与 CodeBERT NPY 行数不一致！"

    print(f"📌 Total loaded samples (CodeBERT): {len(df)}")

    # 2️⃣ 拆分 train / dev / test（复用你已有的 split_data）
    (X_train_raw, y_train_raw), (X_dev, y_dev), (X_test, y_test) = split_data(input_file, input_npy)
    print(f"📌 train = {len(X_train_raw)}, dev = {len(X_dev)}, test = {len(X_test)}")

    # 3️⃣ 只在 train 上做 SMOTE
    X_train, y_train = apply_smote(X_train_raw, y_train_raw, k_neighbors=3, random_state=42)

    # 4️⃣ Logistic Regression 超参网格
    param_grid = {
        "C": [0.1, 1.0, 3.0, 5.0],  # 扩展更大范围
        "class_weight": ["balanced"],
        "max_iter": [100,200, 500, 1000],  # 防止收敛问题
        "penalty": ["l2"],  # L2 正则足够
        "solver": ["lbfgs"],  # 最稳定的 solver
    }

    grid = list(ParameterGrid(param_grid))
    print(f"🔍 Total hyperparameter combinations (CodeBERT): {len(grid)}")

    best_f1 = -1
    best_model = None
    best_params = None

    # 5️⃣ 遍历超参组合，在 dev 上选 F1 最优
    for params in grid:
        clf = LogisticRegression(**params)
        clf.fit(X_train, y_train)

        y_pred_dev = clf.predict(X_dev)
        f1 = f1_score(y_dev, y_pred_dev)

        if f1 > best_f1:
            best_f1 = f1
            best_model = clf
            best_params = params

    print("\n📊 Best Dev Classification Report (CodeBERT + Linear):")
    print(classification_report(y_dev, best_model.predict(X_dev), digits=4))
    print("Best params:", best_params)

    # 6️⃣ 在 dev 上用同一个 tune_threshold 做阈值扫描
    probs_dev = best_model.predict_proba(X_dev)[:, 1]
    best_thr, df_thr = tune_threshold(y_dev, probs_dev, metric="f1")

    print("\n📈 Threshold Scan on Dev Set (CodeBERT):")
    print(df_thr)
    print(f"\n🏆 Best Threshold (CodeBERT) = {best_thr}")

    # 7️⃣ 在 test 上用最佳阈值评估
    probs_test = best_model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_thr).astype(int)

    print("\n📊 Final Test Report (CodeBERT + Linear, Using Best Threshold):")
    print(classification_report(y_test, preds_test, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds_test))

    # 8️⃣ 保存模型 & 超参 & 阈值
    os.makedirs("record", exist_ok=True)
    # 简单用 joblib/pickle 保存

    joblib.dump(best_model, "record/codebert_logreg_model.pkl")
    with open("record/codebert_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    with open("record/codebert_best_threshold.txt", "w") as f:
        f.write(str(best_thr))

    return best_model, best_params, best_thr, (X_test, y_test)


if __name__ == "__main__":
    input_csv = "record/labele_1200_labeled.csv"   # 有 text + visibility 的那个
    codebert_npy = "record/codebert_embeddings.npy"

    # 1）先跑一次编码（只要跑一次，后面可复用）
    encode_file_codebert(input_csv, codebert_npy, text_col="text")

    # 2）然后跑训练 + 验证 + 阈值 + 测试
    best_model, best_params, best_thr, (X_test, y_test) = train_and_valid_codebert(
        input_csv, codebert_npy
    )