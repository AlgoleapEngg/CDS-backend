# 🚀 Data Analysis Report

## Customer-Centric Highlights
{{ narrative }}

---

## 1. Dataset Snapshot
Your dataset contains **{{ analysis.dataset_overview.num_rows }}** records across **{{ analysis.dataset_overview.num_columns }}** fields, using about **{{ analysis.dataset_overview.memory_usage_mb | round(2) }} MB** of memory.
> **Takeaway:** At this scale, you can iterate quickly—but very wide tables may slow down some analyses.

---

## 2. Data Quality Narrative
- **Duplicate rows:** {{ analysis.data_quality.duplicate_rows }}
- **Constant columns:** {{ analysis.data_quality.constant_columns | length }}

**Fields with high missing rates (>20%):**
{% set any_miss = false %}
{% for col, pct in analysis.data_quality.missing_value_percentage.items() %}
  {% if pct >= 20 %}
    {% set any_miss = true %}
    - **{{ col }}**: {{ pct }}% missing
  {% endif %}
{% endfor %}
{% if not any_miss %}
All fields have under 20% missingness—great!
{% endif %}

> **Takeaway:**
{% if any_miss %}
You’ll likely drop or impute the high-missing fields listed above before modeling.
{% else %}
No major missingness to worry about.
{% endif %}

---

## 3. Distributions Narrative
**Numeric features summary:**
{% for col, dist in analysis.value_distributions.items() if dist.min is defined %}
- **{{ col }}** ranges from {{ dist.min }} to {{ dist.max }} (mean = {{ dist.mean | round(2) }}, median = {{ dist.median }}), std = {{ dist.std | round(2) }} (skew = {{ dist.skew | round(2) }})
{% endfor %}

**Categorical features (top 3 categories):**
{% for col, dist in analysis.value_distributions.items() if dist.min is not defined %}
- **{{ col }}** has {{ dist.unique_count }} unique values; top:
  {% set count = 0 %}
  {% for k, v in dist.top_values.items() %}
    {% if count < 3 %}
      - {{ k }} ({{ v }})
      {% set count = count + 1 %}
    {% endif %}
  {% endfor %}
{% endfor %}

> **Takeaway:**
Log-transform any numeric with |skew| > 1:
{% set any_skew = false %}
{% for col, dist in analysis.value_distributions.items() if dist.min is defined and (dist.skew > 1 or dist.skew < -1) %}
  {% set any_skew = true %}
  - {{ col }} (skew = {{ dist.skew | round(2) }})
{% endfor %}
{% if not any_skew %}
None exceed |1|—your numeric distributions look fairly symmetric.
{% endif %}

---

## 4. Correlations & Outliers Narrative
**Outliers detected** (|z| > 3) in:
{% set any_out = false %}
{% for col, cnt in analysis.outliers.items() if cnt > 0 %}
  {% set any_out = true %}
  - {{ col }}: {{ cnt }} records
{% endfor %}
{% if not any_out %}
None—your numeric ranges are tight!
{% endif %}

**Top features correlated with price:**
{% for feat, r in analysis.correlations.price.items() if feat != "price" and (r >= 0.6 or r <= -0.6) %}
- {{ feat }} (r = {{ r }})
{% endfor %}

> **Takeaway:**
Trim or cap outliers listed above; prioritize strongly-correlated features in your first models.

---

## 5. Recommendations & Next Steps
1. **Clean & Impute** – handle missingness in {% if any_miss %}the fields above{% else %}none needed{% endif %}.
2. **Transform** – apply log on skewed numerics if any.
3. **Feature Selection** – start with:
   {% for feat, r in analysis.correlations.price.items() if feat != "price" and r >= 0.8 %}
   - **{{ feat }}** (r = {{ r }})
   {% endfor %}
4. **Define Target** – decide whether to predict **price**, **fuel-efficiency**, or another goal.

> **Roadmap:**
With clean data, transformed distributions, and high-impact features in hand, you’re set to explore your next ML or statistical workflow.