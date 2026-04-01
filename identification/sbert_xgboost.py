"""
剪切merged_all.csv，然后让chat做一下人工标注，标注可复现，不可复现,这里就是持续剪切，筛掉框架类，然后进行一个个测试
(调用api做下面一步的放在5_chatreportbug)
"""

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

input_file = f"github_clawer\\merged_all.csv"     # 原始文件名


def biaozhu(num,input_file = f"github_clawer\\merged_all.csv"
            ,output_file = f"D:\\vr_detection_2.0\data_process\\labeled_sample.csv"):
    # 1️⃣ 读入原始文件
     # 原始文件名
    # 2️⃣ 读取 CSV
    df = pd.read_csv(input_file)

    # 3️⃣ 如果太大，可以先随机抽样 100 条
    sample_df = df.sample(n=num, random_state=42)  # 随机抽样 100 条

    # 4️⃣ 如果你想直接取前100条而非随机，改为：
    # sample_df = df.head(100)

    # 5️⃣ 保存
    sample_df.to_csv(output_file, index=False)

    print(f"✅ 已剪切 {len(sample_df)} 条样本，保存为 {output_file}")


# ===== Step 1️⃣：数据加载 =====

def read_and_sbert(file_path):
    """
    数据加载+
    """
    # ===== Step 1️⃣：读取 CSV =====
    text_col = "text"

    df = pd.read_csv(file_path)
    if text_col not in df.columns:
        raise ValueError(f"❌ 未找到列名 {text_col}，请检查文件。")

    texts = df[text_col].astype(str).tolist()
    print(f"✅ Loaded {len(texts)} rows for embedding.")

    # ===== Step 2️⃣：生成 SBERT 向量 =====
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # ===== Step 3️⃣：逐条单位化（L2 Normalization）=====
    unit_embeddings = normalize(embeddings, norm='l2', axis=1)
    print(f"✅ Normalized (unit) vectors shape: {unit_embeddings.shape}")

    # ===== Step 4️⃣：保存 =====
    # 1. 保存为 .npy 以便后续分析
    os.makedirs("data_process", exist_ok=True)  # ✅ 确保目录存在
    np.savez(
        "data_process/text_unit_embeddings_with_index.npz",
        embeddings=embeddings,
        index=df.index.values
    )

    # 2. 保存为 CSV（带原文方便对照）
    emb_df = pd.concat([
        df[[text_col]].reset_index(drop=True),
        pd.DataFrame(unit_embeddings, columns=[f"dim_{i + 1}" for i in range(unit_embeddings.shape[1])])
    ], axis=1)

    emb_df.to_csv(f"data_process\\text_embeddings.csv", index=False, encoding="utf-8-sig")
    print("💾 Saved to 'data_process\\text_embeddings.csv' and 'text_unit_embeddings.csv'")

"""训练sbert_xgboost"""
def train_sbert_xgboost(csv_path, text_col="text", label_col="label",
                        model_out=f"data_process\\xgb_model.json", params_out=f"data_process\\xgb_params.json"):
    """
    SBERT + XGBoost 二分类模型训练方法
    ----------------------------------------------------
    输入:
        csv_path : 带 'text' 和 'label' 列的 CSV 文件
        text_col : 文本列名
        label_col: 标签列名 (0/1)
    输出:
        1. XGBoost 模型文件 (.json)
        2. 模型参数文件 (.json)
        3. 控制台打印分类报告 & 混淆矩阵
    """

    # 1️⃣ 加载数据
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    print(f"✅ Loaded {len(df)} samples from {csv_path}")

    # 2️⃣ SBERT 向量化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    embeddings = sbert.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"✅ Generated embeddings with shape {embeddings.shape}")

    # X: SBERT 向量 (N, 384)
    # y: 标签 (0/1)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(embeddings,labels)

    print(f"After SMOTE: {np.bincount(y_resampled)}")

    # 3️⃣ 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    # 4️⃣ 定义 XGBoost 参数
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

    # 5️⃣ 训练模型
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("🎯 Training complete.")


    # 6️⃣ 模型评估
    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # 7️⃣ 输出特征重要性（前10）
    importance = clf.feature_importances_
    top_features = np.argsort(importance)[-10:][::-1]
    print("\n🔥 Top 10 Important Dimensions:")
    for i, idx in enumerate(top_features):
        print(f"  {i+1:02d}. dim_{idx} → importance={importance[idx]:.4f}")

    # 8️⃣ 保存模型与参数
    # 确保目录存在
    os.makedirs("data_process/model", exist_ok=True)
    # 1️⃣ 保存 XGBoost 模型 (.json)
    clf.save_model("data_process/model/xgb_model.json")
    # 2️⃣ 保存 SBERT 编码器 (.pkl)
    joblib.dump(sbert, "data_process/model/sbert_encoder.pkl")
    # 3️⃣ 打包一个整体模型（可选，用于直接预测）
    joblib.dump({"encoder": sbert, "model": clf}, "data_process/model/final_model.pkl")
    # 4️⃣ 保存参数 (.json)
    params_out = "data_process/model/xgb_params.json"
    with open(params_out, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)

    print(f"\n💾 Saved model → {model_out}")
    print(f"💾 Saved parameters → {params_out}")
    print("💾 Saved SBERT encoder → sbert_encoder.pkl")

    return clf, sbert, params

def run_model(model_path, source_npy, raw_data, output_csv="predicted_results.csv"):
    """
    model_path : str  → 已训练好的 xgboost 模型路径 (.json)
    source_npy : str  → SBERT embedding 文件路径 (.npz)
    raw_data   : str  → 原始 CSV 文件，用来合并输出
    output_csv : str  → 输出 CSV
    """
    print(f"📦 Loading model from {model_path} ...")
    model = xgb.Booster()
    model.load_model(model_path)

    print(f"📊 Loading embeddings from {source_npy} ...")
    data = np.load(source_npy)
    X = data["embeddings"]
    idx = data["index"]
    print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # 用 XGBoost 预测
    dmatrix = xgb.DMatrix(X)
    probs = model.predict(dmatrix)

    # 二分类阈值
    preds = (probs >= 0.5).astype(int)

    # 预测 DataFrame
    df_result = pd.DataFrame({
        "index": idx,
        "predicted_label": preds,
        "reproducible_prob": probs
    })

    # 合并 + 输出文件
    merged_csv(raw_data, df_result, output_csv)

    return df_result


def merged_csv(raw_data, pred_df, output):
    # 读取原始文本数据
    df_raw = pd.read_csv(raw_data)
    # 合并：用 pred_df 的 index 字段
    merged = df_raw.reset_index().merge(
        pred_df,
        left_on="index",
        right_on="index",
        how="left"
    )
    # 保存
    merged.to_csv(output, index=False)
    print("✅ 合并完成，前 5 行：")
    print(merged.head())
    return merged



def create_high_confidence(num):
    """挑出超过0.9的"""
    df = pd.read_csv("prediction_result.csv")
    high_conf = df[df["reproducible_prob"] > num]
    high_conf.to_csv("high_confidence_results.csv", index=False)
    print(f"✅ 已筛出 {len(high_conf)} 条置信度 > 0.9 的结果，保存为 high_confidence_results.csv")


def create_confidence_level(num1,num2):
    """挑出范围区间在这中间的"""
    df = pd.read_csv("prediction_result.csv")
    high_conf = df[(df["reproducible_prob"] > num1) & (df["reproducible_prob"] <= num2)]
    high_conf.to_csv(f"high_confidence_results{num1}_{num2}.csv", index=False)
    print(f"✅ 已筛出 {len(high_conf)} 条置信度{num1}_{num2}.  的结果，保存为 high_confidence_results.csv")

def filter_app(file="high_confidence_results.csv"):

    # 读入数据
    df = pd.read_csv(file)
    # 🧹 清洗 repo：去掉 issue id、空格、大小写统一
    # 构造链接
    df["repo_link"] = "https://github.com/" + df["repo"]

    # ✅ 1️⃣ 统计每个仓库的样本数
    repo_counts = df["repo"].value_counts().reset_index()
    print("repo_counts",repo_counts)
    repo_counts.columns = ["repo", "count"]

    # ✅ 2️⃣ 去重（每个仓库取一条）+ 合并计数
    unique_repos = (
        df.drop_duplicates(subset=["repo"])
          .merge(repo_counts, on="repo", how="left")
          .reset_index(drop=True)
    )

    # 输出统计信息
    print(f"✅ 原始高置信样本：{len(df)} 条")
    print(f"✅ 独立仓库：{len(unique_repos)} 个")
    print("🔥 Top 10 仓库:")
    print(unique_repos.sort_values("count", ascending=False)[["repo", "count"]].head(10))

    # 💾 保存结果
    unique_repos[["repo", "repo_link", "count"]].to_csv("unique_high_conf_repos.csv", index=False)
    print("💾 已保存 unique_high_conf_repos.csv（含计数）")


def open_in_patch(file, batch_size=5, wait_time=30):
    """
    每次打开 batch_size 个网页，每次等待 wait_time 秒
    :param file: 包含 repo_link 的 CSV 文件
    :param batch_size: 每次打开的网页数量
    :param wait_time: 每批之间等待的秒数
    """
    # 读取文件
    df = pd.read_csv(file)
    if "repo_link" not in df.columns:
        raise ValueError("❌ 文件中找不到列 'repo_link'。")

    links = df["repo_link"].dropna().tolist()

    print(f"✅ 总共 {len(links)} 个仓库链接，将分批打开，每批 {batch_size} 个。")

    for i in range(0, len(links), batch_size):
        batch = links[i:i + batch_size]
        print(f"\n🚀 打开第 {i // batch_size + 1} 批（{len(batch)} 个）：")
        for url in batch:
            print(f"   🌐 {url}")
            webbrowser.open_new_tab(url)
        if i + batch_size < len(links):
            print(f"⏳ 等待 {wait_time} 秒后继续...")
            time.sleep(wait_time)

    print("\n✅ 全部打开完成！")


# ------------------------------------------------------------
# 🧩 第一步：记录当前打开的 GitHub 仓库链接
# ------------------------------------------------------------
def record_open_github(output_txt="opened_repos.txt"):
    """
    检测当前打开的 GitHub 页面（通过浏览器进程命令行）
    提取仓库 owner/repo，保存到一个文本文件。
    """
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
        print("⚠️ 没检测到任何 GitHub 页面（可能浏览器未暴露 URL）。")
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

    print(f"✅ 检测到 {len(repos)} 个仓库，已保存到 {output_txt}")
    for r in repos:
        print("  🌐", r)

    return repos


# ------------------------------------------------------------
# 🧩 第二步：根据仓库名筛选原始 CSV
# ------------------------------------------------------------
def filter_by_repos(repo_txt="opened_repos_90.txt",
                    raw_csv=r"D:\vr_detection_2.0\pythonProject\prediction_result.csv",
                    col_name="repo",
                    new_csv=r"D:\vr_detection_2.0\pythonProject\filter_new90.csv"):
    """
    从 raw_csv 中筛选出仓库名在 repo_txt 文件中的记录
    """

    # 1️⃣ 读取 repo 名称列表
    with open(repo_txt, "r", encoding="utf-8") as f:
        text = f.read()

    # 支持换行或逗号分隔的两种格式
    repos = [r.strip().lower() for r in text.replace(",", "\n").splitlines() if r.strip()]
    print(f"📦 从 {repo_txt} 读取 {len(repos)} 个仓库名")

    # 2️⃣ 读取原始 CSV
    df = pd.read_csv(raw_csv)
    if col_name not in df.columns:
        raise ValueError(f"❌ {raw_csv} 中未找到列 '{col_name}'")

    # 3️⃣ 标准化并筛选
    df[col_name] = df[col_name].astype(str).str.lower().str.strip()
    filtered = df[df[col_name].isin(repos)]

    # 4️⃣ 保存结果
    filtered.to_csv(new_csv, index=False)
    print(f"✅ 筛选出 {len(filtered)} 条记录，已保存至 {new_csv}")
    return filtered


"""重新进行超参的训练"""


def encode_file_npz(input_csv, output_npz, text_col="text"):
    """SBERT 编码：读取 CSV → 编码 text → 保存为 NPZ（含 embeddings + index）"""

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=[text_col])

    texts = df[text_col].astype(str).tolist()
    idx = df.index.to_numpy()  # 保存原始行号
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

    print(f"✅ Embedding shape: {embeddings.shape}")

    # 保存 npz
    np.savez(output_npz, embeddings=embeddings, index=idx)
    print(f"💾 Saved embeddings + index → {output_npz}")

    return embeddings, idx


def encode_file(input_csv, output_npy, text_col="text"):
    """SBERT 编码函数：读取 CSV → 对 text 编码 → 保存为 NPY 文件"""

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
    print(f"✅ Embedding shape: {embeddings.shape}")
    # 5️⃣ 保存
    np.save(output_npy, embeddings)
    print(f"💾 Saved embeddings → {output_npy}")
    return embeddings

def apply_smote(X_train, y_train, k_neighbors=3, random_state=42):
    """
    对训练集应用 SMOTE（只对训练集进行！）
    ----------------------------------------------------
    输入:
        X_train: ndarray (N, D)  训练集特征
        y_train: ndarray (N,)    训练集标签
        k_neighbors: int         SMOTE 的近邻数
        random_state: int        随机数种子

    输出:
        X_resampled: ndarray (N', D)  SMOTE 后的训练集特征
        y_resampled: ndarray (N',)    SMOTE 后的训练集标签
    """
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

    # 标签映射
    df[label_col] = df[label_col].map({"visual": 1, "non_visual": 0})

    # 加载 SBERT 编码
    embeddings = np.load(input_npy)
    assert len(df) == len(embeddings), "CSV 与 NPY 行数不一致！"

    # X = embeddings（不是文本！！！）
    X = embeddings
    y = df[label_col].values

    # 1️⃣ 先拆 test（200）
    X_main, X_test, y_main, y_test = train_test_split(
        X, y, test_size=200/len(df), stratify=y, random_state=42
    )

    # 2️⃣ 再从 main 拆出 dev（200）
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_main, y_main, test_size=200/len(X_main), stratify=y_main, random_state=42
    )

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)

def train_and_valid(input_file, input_npy):
    """
    训练 + 验证：从 1000 条中划分 800/200，并在 dev 上选最优超参
    -----------------------------------------------------------------
    输入:
        input_file : CSV 文件（text + label）
        input_npy  : SBERT embedding 文件（np.load 即可用）
    输出:
        best_model : dev 上最佳 XGBoost 模型
        best_params: 最佳超参
        (X_test, y_test): 测试集数据（后续做最终评估）
    """

    # 1️⃣ 加载文件
    df = pd.read_csv(input_file)
    embeddings = np.load(input_npy)

    assert len(df) == len(embeddings), "CSV 与 NPY 行数不一致！"

    print(f"📌 Total loaded samples: {len(df)}")
    # 2️⃣ 前 1000 用于 train/dev，后 200 用 test
    (X_train_raw, y_train_raw), (X_dev, y_dev), (X_test, y_test) = split_data(input_file,input_npy)

    print(f"📌 train = {len(X_train_raw)}, dev = {len(X_dev)}")

    # 4️⃣ SMOTE only on train
    sm = SMOTE(k_neighbors=3, random_state=42)
    X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)
    print(f"🔧 After SMOTE: {np.bincount(y_train)}")

    # 5️⃣ Grid Search 超参
    param_grid = {
        "n_estimators": [100, 200, 300],#多少个tree
        "max_depth": [3, 4, 5],#树深度
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

    # 6️⃣ 遍历每组参数
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

        # 6️⃣ Dev 分类报告
        print("\n📊 Dev Classification Report:")
        print(classification_report(y_dev, best_model.predict(X_dev)))

        # === 7️⃣ 在 dev 上调阈值 ===
        probs_dev = best_model.predict_proba(X_dev)[:, 1]

        best_thr, df_thr = tune_threshold(y_dev, probs_dev, metric="f1")

        print("\n📈 Threshold Scan on Dev Set:")
        print(df_thr)
        print(f"\n🏆 Best Threshold = {best_thr}")

        # === 8️⃣ 使用 dev 上的最佳阈值，在 test 上评估  🔧 最重要的修改 ===
        dmatrix_test = xgb.DMatrix(X_test)
        probs_test = best_model.predict(dmatrix_test)
        preds_test = (probs_test >= best_thr).astype(int)

        print("\n📊 Final Test Report (Using Best Threshold):")
        print(classification_report(y_test, preds_test))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds_test))

        # 9️⃣ 保存模型 & 超参 & 阈值 🔧
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