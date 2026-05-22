## 1. 因变量 (标签 $Y$ — 当前期 $t$)

* **资产回报率 ($EBI/A$):**
    
    $$\text{EBI/A}_t = \frac{\text{EBI}_t}{\text{Assets}_t}$$
    
    注：$$\text{EBI} = \text{EBIT} - \text{Taxes}$$ (息税前利润 - 税收)。

---

## 2. 自变量 (特征 $X$ — 滞后一期 $t-1$)

为了缓解潜在的内生性问题，所有自变量均采用滞后一期 ($t-1$) 的数据：

### 2.1 公司规模与行业特征
* **公司规模 ($\text{Total Assets}_{t-1}$):** 衡量方式为总资产的自然对数，即 $\ln(\text{Total Assets})$。
* **行业集中度 ($\text{HHI}_{t-1}$):** 使用赫芬达尔-赫希曼指数 (Herfindahl-Hirschman Index) 进行衡量。

### 2.2 营运效率
* **资产周转率 ($\text{Asset Turnover}_{t-1}$):** $\text{营业收入} \div \text{平均总资产}$。
* **营运资金比率 ($\text{Working Capital Ratio}_{t-1}$):** $\text{营运资金} \div \text{总资产}$。
* **固定资产比率:** $\text{固定资产净值} \div \text{总资产}$。

### 2.3 财务风险
* **财务杠杆 ($\text{Leverage}_{t-1}$):** $\text{总负债} \div \text{总资产}$ (资产负债率)。
* **流动比率 ($\text{Current Ratio}_{t-1}$):** $\text{流动资产} \div \text{流动负债}$。

### 2.4 增长与投资
* **资本支出比率 ($\text{Investment Capex Ratio}_{t-1}$):** $\text{资本支出} \div \text{总资产}$。
* **营业收入增长率:** $(\text{当期营业收入} - \text{上期营业收入}) \div \text{上期营业收入}$。
