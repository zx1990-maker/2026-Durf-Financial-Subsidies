# TASK 6 — EBI 会计口径审计

## 1. 当前使用的字段

- **原始 Wind 字段名**：`归属母公司股东的净利润`（net profit attributable to parent）。

- **Python dataframe 列名**：`run_counterfactual.py` 中 `classify_column()` 将其映射为语义类型 `net_profit`；`records_to_dataframe()` 用其与 `interest_expense`、`total_assets` 计算 `EBI_A`。

- **EBI/A 构造公式**：`EBI_A = (net_profit[t+1] + interest_expense[t+1]) / total_assets[t+1]`。

- **数据来源**：`0803/A_{2014..2024}_origin.xlsx`（Wind 终端导出）。

- **使用该变量的文件/函数**：`run_counterfactual.py` → `classify_column()`（第 186 行）与 `records_to_dataframe()`（第 325 行）。

## 2. 原始数据字段清单（决定性）

对全部 11 个 `A_*_origin.xlsx` 做逐列扫描，共 14 个不同字段：

- `企业所有制性质`（11 个文件）
- `利息支出`（11 个文件）
- `固定资产-净值`（11 个文件）
- `归属母公司股东的净利润`（11 个文件）
- `所属Wind行业名称(2024)`（11 个文件）
- `流动负债合计`（11 个文件）
- `流动资产合计`（11 个文件）
- `营业收入`（11 个文件）
- `营业收入(同比增长率)`（11 个文件）
- `证券代码`（11 个文件）
- `证券简称`（11 个文件）
- `负债合计`（11 个文件）
- `资产总计`（11 个文件）
- `资本性支出`（11 个文件）

其中与利润/利息/税/少数股东相关的字段：

- `利息支出`
- `归属母公司股东的净利润`

## 3. 是否存在 consolidated net income

- **合并净利润 / 净利润**：**不存在**。
- **少数股东损益**：**不存在**。
- **归属母公司股东的净利润**：**存在**（唯一被使用的利润字段）。


## 4. 结论

原始数据**只包含单一利润序列**（归属母公司股东的净利润），**不存在** 合并净利润/净利润、少数股东损益、利润总额、营业利润、所得税费用。因此：

> Consolidated NI robustness cannot be implemented with available data.

故本任务**不生成** `table_ebi_definition_robustness.csv`（数据可得性限制，而非分析选择）。主指标 `EBI/A`（归母净利润口径）保持不变。


## 5. 口径含义（非 EBIT）

分子为「税后、归母」净利润加利息支出，故**不是** EBIT（缺税加回、缺少数股东损益加回）。它是「(税后) 归母权益 + 有息债务」口径的资产回报率变体。跨组有效税率或少数股东结构差异会机械地进入 gap，但本数据无法量化该影响。
