# TASK 5 — KNN 代码审计

## 现有 KNN 是否严格限制同行业？

**否。** 现有 KNN（`outputs/revision2/audit_05_knn.py`）**不**严格限制同行业。

- **文件**：`outputs/revision2/audit_05_knn.py`

- **函数**：主循环中 `NearestNeighbors(n_neighbors=k).fit(Xt)`，其中 `Xt = prep['Xt']`

- **行业限制条件**：无硬约束。`Xt` 是「9 个 winsorize+标准化数值特征」与「industry_l2 one-hot」拼接后的矩阵；one-hot 只作为距离的一个分量（不同行业会增加欧氏距离），但**不禁止**跨行业邻居。

- **actual_k 处理**：现有代码**不**记录 `actual_k`，也**不**做「同行业不足 k 家」的处理——它始终取全样本的 k 近邻（可跨行业）。


## 结论与处理

现有 KNN 是「全市场（可跨行业）可比企业」benchmark，不是「同行业可比企业」benchmark。
因此本任务新增**严格同行业 KNN**（`task05_industry_knn.py`）：候选民企必须 `industry_l2` 相同，
距离用主模型同一套 9 个数值特征（scaler 只由民营训练样本拟合），
同行业民企不足 k 家用全部（记录 `actual_k`），同行业无民企则 benchmark 缺失且不跨行业。


## 新增输出

- `industry_restricted_knn_summary.csv`（year × sample）

- `industry_restricted_knn_firm_level.csv`（firm-year，含 `actual_k5`/`actual_k10`）
