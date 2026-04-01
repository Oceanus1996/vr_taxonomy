import json
import joblib
import re
import time

import numpy as np
import pandas as pd
from github import Github, Auth
from openai import OpenAI

import sbert_xgboost
from codebert_linear import encode_file_codebert
GPT_TOKEN = os.environ["OPENAI_API_KEY"]

def merge_2(input1: str, input2: str, output: str,
            dedup_subset: list = ["repo", "id"],
            keep: str = "first"):
    df1 = pd.read_csv(input1)
    df2 = pd.read_csv(input2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(subset=dedup_subset, keep=keep)
    df.to_csv(output, index=False, encoding="utf-8")
    print(f"Merged {input1} + {input2} -> {output}, rows: {len(df)}")


def read_repo(input_path: str, output_path: str):
    # 读入数据
    df = pd.read_csv(input_path)

    if "repo" not in df.columns:
        raise ValueError("输入文件中没有 'repo' 列！")
    # 按 repo 分组统计数量
    stats = (
        df.groupby("repo")
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )
    stats.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✔ 完成：共 {len(stats)} 个 repo，统计已写入 {output_path}")
    return stats



prompt_template = """
You are a classifier for software bug reports.

Task:
Determine whether each issue described by the user is a *visual bug* or *non_visual bug* based on user-perceivable effects.

Definition of *visual bug*:
A visual bug is any issue that the user can directly perceive during interaction.  
This includes:
- Visually observable problems (appearance issues, rendering errors, UI misalignment, flickering, incorrect position, visual artifacts, spatial drift, etc.)
- User-perceivable motion or interaction abnormalities (noticeable lag, stutter, frame freeze, jitter, delayed visual response, etc.)

Definition of *non_visual bug*:
A non_visual bug is a problem not directly perceivable by the user.  
Examples include:
- Pure performance optimization (CPU usage, render time, profiling, memory usage)
- Internal logic errors, API issues, state inconsistencies
- Crashes without visible symptoms
- Build issues, dependency updates, refactoring, configuration problems
- Backend or data processing errors

Your task:
Output JSON per report:
{{"index": i, "visibility": "visual" | "non_visual", "reason": "short explanation"}}

Classify the following:
---
{text_block}
---

"""


def biaozhu_gpt(inputfile, outputfile, batch=10):
    df = pd.read_csv(inputfile)
    texts = df["text"].fillna("").tolist()
    results = []

    for i in range(0, len(texts), batch):
        block = texts[i:i+batch]

        # ✔ 正确构造 batch prompt
        text_block = ""
        for idx, t in enumerate(block):
            text_block += f"[{idx}] {t}\n"

        prompt = prompt_template.format(text_block=text_block)

        # 调用 GPT
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        ).choices[0].message.content
        clean = resp.strip().lstrip("```json").lstrip("```").rstrip("```")

        # 解析 JSON
        try:
            print(clean)
            parsed = json.loads(clean)
        except:
            parsed = [{"visibility": "non_visual", "reason": "parse_error"} for _ in block]

        results.extend(parsed)
        time.sleep(0.1)

    # 写回 CSV
    df["visibility"] = [r["visibility"] for r in results]
    df["reason"] = [r["reason"] for r in results]
    df.to_csv(outputfile, index=False)

TOKEN = os.environ.get("GITHUB_TOKEN", "")

def issue_commit_from_api(input_csv, output_csv, threshold=0.55):

    g = Github(auth=Auth.Token(TOKEN), timeout=20)

    # 匹配任何 "#123" 形式
    pattern = re.compile(r"#(\d+)")

    # ===== 读取 CSV =====
    df = pd.read_csv(input_csv)
    df_f = df[df["reproducible_prob"] > threshold].reset_index(drop=True)

    # repo → issue_ids
    repo_to_issues = {}
    for _, row in df_f.iterrows():
        repo_to_issues.setdefault(row["repo"], set()).add(int(row["id"]))

    print(f"📌 {len(repo_to_issues)} repos, {len(df_f)} issues to analyze")

    # repo → issue_id → commits
    commit_map = {}

    # ===== 主循环：只扫 commit message =====
    for repo_name, issue_id_set in repo_to_issues.items():
        print(f"\n🚀 Repo: {repo_name} (need issues: {len(issue_id_set)})")

        try:
            repo = g.get_repo(repo_name)
        except:
            commit_map[repo_name] = {}
            print(f"❌ 无法访问 repo: {repo_name}")
            continue

        repo_commit_dict = {iid: [] for iid in issue_id_set}

        try:
            commits = repo.get_commits()
        except:
            commit_map[repo_name] = repo_commit_dict
            continue

        for commit in commits:
            msg = commit.commit.message or ""
            matches = pattern.findall(msg)
            if not matches:
                continue

            for issue_num in matches:
                iid = int(issue_num)
                if iid in repo_commit_dict:
                    repo_commit_dict[iid].append(commit.sha)

        commit_map[repo_name] = repo_commit_dict

    # ===== 输出结果 =====
    results = []
    for _, row in df_f.iterrows():
        repo = row["repo"]
        issue_id = int(row["id"])
        commits = commit_map.get(repo, {}).get(issue_id, [])

        results.append({
            "repo": repo,
            "issue_id": issue_id,
            "type": row["type"],
            "author": row["author"],
            "text": row["text"],
            "reproducible_prob": row["reproducible_prob"],
            "predicted_label": row["predicted_label"],
            "has_commit": len(commits) > 0,
            "commit_shas": commits,
            **row.to_dict()
        })
        print(f"repo:{repo},issue:{issue_id},commit:{commits}")

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)

    print("\n🎉 完成linking！")
    print("📊 Summary:")
    print(out["has_commit"].value_counts())

    return out

def apply_codebert_classifier(input_csv,codebert_npy,model_path,output_csv):
    # 1️⃣ 读取原始 CSV
    df = pd.read_csv(input_csv)
    print(f"📄 Loaded {len(df)} rows from {input_csv}")

    # 2️⃣ 加载 CodeBERT embedding
    embeddings = np.load(codebert_npy, allow_pickle=True)
    if isinstance(embeddings, dict):  # 如果 npz 格式
        embeddings = embeddings["embeddings"]

    assert len(embeddings) == len(df), "❗CSV 与 CodeBERT embedding 行数不一致！"

    # 3️⃣ 加载训练好的 Logistic Regression 模型
    clf = joblib.load(model_path)
    print("✅ Loaded CodeBERT + Linear classifier.")

    # 4️⃣ 预测概率
    probs = clf.predict_proba(embeddings)[:, 1]  # 取视觉缺陷概率

    # 5️⃣ 加载你在训练阶段的阈值
    with open("record/codebert_best_threshold.txt", "r") as f:
        thr = float(f.read().strip())

    preds = (probs >= thr).astype(int)

    # 6️⃣ 写入新列
    df["code_reproducible_prob"] = probs
    df["code_predicted_label"] = preds

    # 7️⃣ 导出
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 Saved result → {output_csv}")
    print("🎉 Done!")

if __name__ == "__main__":
    "合并了所有的相关仓库的repo.非全文（标题）"
    #merge_2("record/2_all_repos_issues.csv","record/2_all_repos_issues_extra.csv","3_merge_all_repo")
    #read_repo("record/3_merge_all_repo.csv","record/3-1_all_repo.csv")
    """拿出1200条 2,人工判断  3.训练一个xboost"""
    #sbert_xgboost.biaozhu(1200,"record/3_merge_all_repo.csv","record/labele_1200.csv")
    #biaozhu_gpt("record/labele_1200.csv","record/labele_1200_labeled.csv",50)
    #sbert_xgboost.encode_file("record/labele_1200_labeled.csv","encode_sbert_1200.npy")
    #sbert_xgboost.train_and_valid(input_file="record/labele_1200_labeled.csv"
    #                                            ,input_npy="record/encode_sbert_1200.npy")

    #sbert_xgboost.encode_file("record/3_merge_all_repo.csv","record/3_merge_all_repo.npy")
    #sbert_xgboost.encode_file_npz("record/3_merge_all_repo.csv","record/3_merge_all_repo.npz")
    #sbert_xgboost.run_model("record/xgb_best_model.json","record/3_merge_all_repo.npz ",
    #                                      "record/3_merge_all_repo.csv","record/4_all_repo_labeled.csv")

    """筛选出4000+条（0.4/0.55）"""

    #issue_commit_from_api("record/4_all_repo_labeled.csv","record/5_all_repo_labeled_commit.csv")


    """linear+codebert 在17074中筛选，训模型已经在H.py中有了，共7000条，输出新文件"""
    # input_csv = "record/6_all_repo_filter_commit.csv"




    # codebert_npy = "record/codebert_embeddings_all.npy"
    # output_pkl = "record/codebert_logreg_model.pkl"
    #encode_file_codebert(input_csv, codebert_npy, text_col="text")
    #apply_codebert_classifier(input_csv,codebert_npy,output_pkl,"record/7_all_repo_codebert.csv")

    # content = pd.read_csv("record/7_all_repo_codebert.csv")
    # high_conf = content[
    #     (content["code_predicted_label"] == 1) &
    #     (content["predicted_label"] == 1) &
    #     (content["has_commit"] == True)
    #     ]
    # print("高置信度视觉缺陷数量:", len(high_conf))
    #
    # # 4️⃣ 保存为新的 CSV
    # output_path = "record/A_visual_high_confidence.csv"
    # high_conf.to_csv(output_path, index=False, encoding="utf-8-sig")
    #
    # print("已保存:", output_path)

    """形成新的repo集合"""