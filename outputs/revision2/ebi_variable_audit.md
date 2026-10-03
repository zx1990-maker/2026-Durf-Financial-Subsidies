# TASK 7 — EBI/A Variable-Definition Audit

**Question.** What is the exact numerator of the outcome variable `EBI_A`? Is it
(a) 归母净利润 (net profit *attributable to parent*), or (b) consolidated 净利润
(net income before minority interest)? Is the measure a true EBIT?

---

## 1. What the code actually computes

In `run_counterfactual.py`, `classify_column()` maps the profit field to `'net_profit'`
via the match

```python
elif '归属母公司股东的净利润' in field_name or '归母净利润' in field_name:
    return 'net_profit', year
```

and `records_to_dataframe()` computes the outcome as

```python
ebi_a = (net_profit[t+1] + interest_expense[t+1]) / total_assets[t+1]
```

**Answer: option (a).** The numerator profit is **归母净利润** — net profit
**attributable to the parent shareholders**, i.e. *after* tax **and** *after* minority
interest. The measure is therefore

```
EBI/A_t+1 = (归母净利润_t+1  +  利息支出_t+1) / 资产总计_t+1
```

---

## 2. What the raw data contains (decisive field scan)

A scan of **all 11** `A_*_origin.xlsx` files (via the project's own `parse_xlsx_xml`)
for every profit-, tax-, interest- and equity-related header returned exactly:

| Category | Fields present |
|---|---|
| Profit | `归属母公司股东的净利润` **only** |
| Interest | `利息支出` **only** |
| Tax | *(none)* — no 所得税 / 所得税费用 |
| Equity / minority | *(none)* — no 净利润 (consolidated), no 少数股东损益, no 利润总额, no 营业利润 |

**Conclusion.** The raw data contains a **single** profit series: net profit attributable
to parent. Consolidated net income (净利润) is **not** in the data, so the alternative
"净利润 vs 归母净利润" robustness comparison requested in TASK 7 **cannot be run** — there
is no consolidated series to compare against. (No `ebi_definition_robustness.csv` is
produced for this reason; this is a data-availability limitation, not an analysis choice.)

---

## 3. What the measure is — and is not

The variable `EBI_A` is a valid, computable ratio, but it is **not** EBIT:

- **Not EBIT.** EBIT = pre-tax operating profit. The numerator here uses *net income
  after tax*, so it is missing the tax add-back (and, because 归母净利润 excludes
  minority interest, the minority-interest add-back too).
- **Partial interest add-back.** `利息支出` (interest expense) is added back, but interest
  is a *pre-tax* deduction, so adding it to an *after-tax* profit figure understates true
  "earnings before interest" by the tax shield on interest.
- **Best description.** The numerator is a "net-profit-after-tax-attributable-to-parent +
  interest expense" figure; the ratio is a *return on assets* variant close to
  "(after-tax) return to debt + parent-equity holders", with the equity leg measured net of
  tax and net of minority interest.

### Implications for the gap interpretation

1. **Tax asymmetry.** Because the numerator is after-tax, cross-group differences in the
   effective tax rate (e.g. preferential SOE tax treatment) would mechanically flow into
   the gap. If SOEs face a systematically *lower* tax rate, the after-tax measure
   *narrows* the measured SOE underperformance relative to a pre-tax (EBIT) measure — i.e.
   the reported gap may be a **lower bound** on the true pre-tax efficiency gap.
2. **Minority-interest asymmetry.** 归母净利润 excludes minority interest. If SOEs
   systematically hold more consolidated subsidiaries with minority shareholders than
   private peers, the parent-attributable numerator understates SOE total earnings and
   biases the gap upward (SOE looks worse). The magnitude is unquantifiable here because
   consolidated 净利润 is absent.
3. **Not "subsidy" or "policy burden".** Consistent with the working principles, this ratio
   is a *relative return benchmark*, not an amount of government subsidy. The
   definitional caveats above reinforce that the gap should be read as a
   return differential, not a fiscal transfer.

---

## 4. Recommendation

- **Keep the variable** (it is the only computable ROA-style measure given the data) but
  **rename/relabel** it in the paper to something precise, e.g. *"net profit attributable
  to parent plus interest expense, over total assets (after tax)"*, and add a one-sentence
  caveat that it is **not** EBIT and is computed **after tax**.
- **State the two directional caveats** (tax asymmetry → likely lower bound; minority
  interest → possible upward bias) so reviewers cannot flag the definition as silently
  non-standard.
- If a pre-tax or consolidated series becomes available from Wind at revision time, add the
  `ebi_definition_robustness.csv` comparison (净利润 + 利息, and EBIT if tax is available).

---

## 5. Verdict (TASK 7)

**PASS WITH REVISION** — the variable is computed exactly as the code intends, but the
paper must label it correctly (after-tax, attributable-to-parent, not EBIT) and disclose
the two directional caveats. No data-correctness defect; a definitional/labeling issue only.
