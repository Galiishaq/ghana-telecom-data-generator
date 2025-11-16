# Ghana Telecom Customer Data Generator

**A synthetic dataset generator for realistic telecom customer analytics in the Ghana market.**

Built because I needed quality data to practice subscriber profitability analysis and revenue modeling, but couldn't find datasets that actually reflected African telecom markets. So I created one using real market statistics from regulatory and industry sources.

---

## Why I Built This

When you search for telecom datasets online, you'll find the same few options repeated everywhere - most from European or American markets that don't reflect how telecoms work in Ghana or Africa more broadly. The distributions are off, the pricing doesn't make sense, and critical features like prepaid dominance and regional variations are missing or oversimplified.

I wanted to work on subscriber profitability analysis, churn prediction, and revenue forecasting, but with data that actually represents the Ghana market. After spending time looking for quality datasets without much luck, I decided to build a data generator that creates realistic customer records calibrated to authoritative sources.

### The Approach

I collected ground truth metrics from three main sources:

1. **National Communications Authority (NCA) Q4 2024 Statistical Bulletin** - Official regulatory data on market structure, usage patterns, and operator statistics
2. **MTN Ghana 2024 Financial Report** - Financial metrics and KPIs from Ghana's largest operator
3. **Ghana Statistical Service 2021 Census** - Population distribution across all 16 regions and urbanization rates

I also did additional research on regional factors - network quality, economic conditions, and competitive intensity across Ghana's regions. This data is in `data/regional_factors.json` with full source citations.

Using these sources, I built a probabilistic data generator that creates customer records with realistic relationships between features. The result is a 100,000-customer dataset that actually behaves like real telecom data - ML models can learn from it, the distributions make sense, and the business metrics align with what you'd see in the market.

---

## What This Dataset Includes

The generator creates **100,000 customer records** with **21 features** covering everything you need for subscriber analytics:

### Financial Metrics
- **ARPU (Average Revenue Per User)**: Monthly revenue per customer
- **Revenue attribution**: Broken down by voice, data, and SMS
- **Payment behavior**: Failed payment tracking

These metrics are essential for subscriber profitability analysis and revenue forecasting.

### Usage Patterns
- **Data usage**: Monthly GB consumption (calibrated to NCA's 29.5GB average)
- **Voice usage**: Minutes of use (MOU) aligned with industry stats
- **SMS volume**: Message counts
- **Social media data**: Estimated usage for platforms like WhatsApp, Facebook, TikTok

### Demographics & Segmentation
- **Regional distribution**: All 16 Ghana regions with population-weighted representation
- **Urban/rural split**: Based on actual urbanization rates by region
- **Subscription type**: Realistic prepaid/postpaid mix (~99% prepaid, matching Ghana's market)
- **Customer tenure**: How long they've been with the operator
- **Device type**: Smartphone vs. feature phone (varies by urban/rural context)

### Behavioral Indicators
- **Support call history**: Service quality proxy
- **Plan change frequency**: Shopping behavior and satisfaction
- **Recharge patterns**: Engagement indicator for prepaid customers
- **Night data usage**: Time-of-day behavior

### Customer Segmentation

The generator creates five distinct customer segments with realistic behavioral patterns:

| Segment | % of Base | Avg ARPU | Churn Rate | Characteristics |
|---------|-----------|----------|------------|-----------------|
| Loyal Champions | 15% | GH₵1,081 | 1.5% | Long tenure, high value, stable |
| Satisfied Majority | 50% | GH₵294 | 5.2% | Core customer base, moderate usage |
| At-Risk | 20% | GH₵62 | 83.5% | Low engagement, payment issues |
| Price Sensitive | 10% | GH₵139 | 27.8% | Frequent plan changes, cost-focused |
| New/Exploring | 5% | GH₵289 | 28.8% | Short tenure, testing service |

These segments have realistic relationships between tenure, usage, ARPU, and churn probability.

---

## Regional Variation

One thing I wanted to get right was regional differences in churn rates. In the real world, churn varies significantly by region due to network quality, economic conditions, and competitive pressure. I researched these factors for all 16 Ghana regions and built them into the generator.

The regional factors research (`data/regional_factors.json`) includes:

**Network Quality Estimates (0.0 - 1.0 scale)**
- Based on urban density, operator presence, infrastructure investment
- Greater Accra: 0.95 (best coverage, major fiber backbone, MTN HQ)
- Upper West: 0.55 (weakest coverage, frequent outages reported)

**Economic Index (relative to national average)**
- World Bank poverty data (2016/2024), GDP per capita estimates
- Greater Accra: 1.30 (highest GDP, 0% extreme poverty)
- Upper West: 0.55 (45.2% extreme poverty - highest in Ghana)

**Competition Index (0.0 - 1.0 scale)**
- All operators' presence, branch density, market share distribution
- Greater Accra: 0.85 (intense competition, all operators HQ here)
- Upper West: 0.32 (smallest market, MTN near-monopoly)

These factors combine to create realistic regional churn variance. The result: Northern regions (poor network + economic challenges) show 26-30% churn, while Greater Accra and Ashanti (good infrastructure + strong economies) show 22-23% churn. This 7.7 percentage point spread is much more realistic than having all regions cluster around the same rate.

---

## Data Quality Validation

The included validation notebook (`notebooks/data_validation.ipynb`) checks that the synthetic data actually behaves like real customer data.

### Statistical Validation

Comparing generated data against ground truth targets:

| Metric | Target (NCA/MTN) | Generated | Difference |
|--------|------------------|-----------|------------|
| Data Usage (MB/month) | 29,463 | ~21,000 | -29% |
| Voice MOU (min/month) | 757.5 | 739.5 | -2.4% |
| SMS (msgs/month) | 29 | 27.9 | -3.7% |
| Postpaid % | 1.08% | 2.85% | +1.8pp |

The voice and SMS metrics are very close. Data usage came out lower because I modeled it with a more conservative distribution to avoid unrealistic outliers. The postpaid percentage is slightly higher due to how I assigned subscription types to different customer segments (loyal customers more likely to be postpaid).

### Regional Churn Patterns

The regional modifiers create realistic variance:

| Region | Network | Economy | Competition | Churn Rate |
|--------|---------|---------|-------------|------------|
| Upper East | 0.57 (poor) | 0.58 (40% poverty) | 0.35 (low) | 29.6% |
| Upper West | 0.55 (worst) | 0.55 (45% poverty) | 0.32 (lowest) | 28.7% |
| Northern | 0.62 (poor) | 0.65 (30.7% poverty) | 0.45 (low) | 26.5% |
| Ashanti | 0.92 (excellent) | 1.10 (strong) | 0.75 (high) | 21.9% |
| Greater Accra | 0.95 (best) | 1.30 (strongest) | 0.85 (highest) | 22.4% |

### Churn Risk Scoring

The multi-factor risk scoring produces clear separation:
- Mean risk score - churned customers: 78.1
- Mean risk score - active customers: 0.3
- Separation: 77.8 points (highly significant, p < 0.001)

This validates that the causal modeling captured realistic patterns - customers with low usage, failed payments, and high support calls actually do churn at higher rates.

---

## How the Generator Works

The core idea is to model customers as belonging to behavioral segments, then generate features based on segment-specific probability distributions while maintaining realistic causal relationships.

### Step 1: Segment Assignment

Each customer is randomly assigned to one of five segments based on real-world proportions:
- 50% Satisfied Majority
- 20% At-Risk
- 15% Loyal Champions
- 10% Price Sensitive
- 5% New/Exploring

### Step 2: Feature Generation

For each customer, features are generated using segment-specific distributions:

**Demographics:**
- **Region**: Randomly assigned based on Ghana's 2021 Census population weights (Greater Accra and Ashanti most common at ~18% each)
- **Urban/Rural**: Probability varies by region (e.g., Greater Accra is 91.7% urban, Upper West is 26.4% urban)
- **Tenure**: Exponential distribution with segment-specific means (Loyal: 36 months avg, At-Risk: 12 months)
- **Subscription Type**: Heavily skewed toward prepaid (~99%), with loyal customers slightly more likely to have postpaid
- **Device**: Urban areas have 75% smartphone penetration vs. 35% in rural areas

**Usage Patterns:**
- **Data usage**: Lognormal distribution (right-skewed like real usage), scaled by segment multiplier
- **Voice MOU**: Normal distribution with segment adjustments
- **SMS**: Poisson distribution (event count data)
- Smartphone users get higher data usage, feature phones get reduced usage

**Revenue:**
- Calculated from usage: `voice_revenue = MOU × GH₵0.15`, `data_revenue = GB × GH₵8.50`
- Total ARPU includes segment-specific multipliers and some random variance to simulate bundle pricing

**Behavioral Indicators:**
- **Failed payments**: Higher probability for At-Risk segment (20%) vs. Loyal Champions (1%)
- **Support calls**: Poisson distribution with higher lambda for At-Risk customers and rural users
- **Recharge frequency**: Lower for At-Risk prepaid customers (disengagement signal)

### Step 3: Churn Risk Scoring

I built a multi-factor risk score that incorporates business logic about what drives churn:

```python
Risk Score Components:
- Low usage (<30% of average): +30 points
- Failed payments: +15 points each
- High support calls (>7): +25 points
- Low ARPU (<50% average): +20 points
- Short tenure (<6 months): +15 points
- Postpaid subscription: -25 points (more stable)
- Night data user: -8 points (engagement)
- Regional modifier: -2 to +14 points based on network/economy/competition
```

The churn probability is then calculated using a logistic function of the risk score. This creates realistic patterns where multiple risk factors compound to increase churn likelihood.

### Technical Details

The generator uses:
- **NumPy** for random number generation and array operations
- **Probability distributions**: Exponential (tenure), lognormal (data usage), Poisson (SMS, support calls), normal (voice usage)
- **Pandas** for data structuring and export
- **Seed control** for reproducibility (can generate the same dataset multiple times)

The code is modular - you can adjust parameters like segment proportions, pricing, or risk weights to experiment with different scenarios.

---

## What You Can Do With This Data

This dataset is ready to use for various telecom analytics projects:

### Subscriber Profitability Analysis
- Segment customers by ARPU and usage patterns
- Calculate customer lifetime value (CLV) estimates
- Identify high-value customers for retention focus
- Analyze regional profitability differences

### Churn Prediction & Retention
- Build predictive models (classification algorithms)
- Test different feature engineering approaches
- Develop retention intervention strategies
- Calculate ROI for retention campaigns

See `examples/train_churn_model.py` for a complete ML example that achieves 91% accuracy and 0.95 AUC on this data.

### Revenue Forecasting
- Model ARPU trends by segment
- Forecast revenue impact of churn
- Analyze voice vs. data revenue mix
- Test pricing strategy scenarios

### Customer Segmentation
- Cluster analysis to find natural groupings
- RFM (Recency, Frequency, Monetary) analysis using tenure, recharge frequency, and ARPU
- Behavioral profiling for targeted campaigns

### Network & Service Quality Analysis
- Correlate support calls with regional patterns
- Identify areas with potential network issues (high support call volumes)
- Analyze urban vs. rural service quality differences

### Dashboard & Visualization Development
- Build KPI dashboards with realistic data
- Practice creating executive reports
- Test visualization tools (Power BI, Tableau) with real-world-like metrics

---

## Project Files

```
ghana-telecom-data-generator/
├── data/
│   ├── ghana_telecom_customers.csv         # Main dataset (100K records)
│   ├── master_ground_truth_q4_2024.json    # Calibration targets from NCA/MTN
│   └── regional_factors.json               # Original research on regional factors
│
├── notebooks/
│   └── data_validation.ipynb               # Data quality validation
│
├── src/
│   └── generate_synthetic_customers.py     # Data generator (run this to create fresh data)
│
├── examples/
│   └── train_churn_model.py                # Example ML pipeline for validation
│
├── DATA_DICTIONARY.md                      # Detailed documentation of all 21 features
├── requirements.txt                        # Python dependencies
└── README.md                               # You're reading it
```

The main file you'll use is `data/ghana_telecom_customers.csv` - it's ready to load into pandas, Excel, Power BI, or any analytics tool.

See `DATA_DICTIONARY.md` for complete feature descriptions, distributions, and suggested feature engineering approaches.

---

## How to Use

### Quick Start (Just Use the Data)

```bash
# 1. Clone or download this repo
cd ghana-telecom-data-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Load and use the dataset
python
>>> import pandas as pd
>>> df = pd.read_csv('data/ghana_telecom_customers.csv')
>>> df.head()
```

The dataset is ready to use - no preprocessing needed.

### Validate Data Quality

```bash
# Interactive notebook with validation checks
jupyter notebook notebooks/data_validation.ipynb
```

This notebook validates:
- Segment distributions match targets
- Usage metrics align with ground truth
- Regional patterns are realistic
- Risk score separation is significant
- Feature correlations make sense

### Run the Example ML Model

```bash
# Quick validation model (tests if data patterns are learnable)
python examples/train_churn_model.py
```

This trains a Random Forest classifier and achieves ~91% accuracy, demonstrating that the synthetic data has realistic, learnable patterns.

### Generate Fresh Data with Custom Parameters

```bash
# Creates new dataset with same parameters
python src/generate_synthetic_customers.py

# Edit the script to change:
# - Number of customers (default: 100,000)
# - Segment distributions
# - Risk scoring weights
# - Pricing assumptions
# - Regional modifiers
```

---

## Limitations & Potential Improvements

**Current limitations:**
- Single time snapshot (no temporal dimension or monthly tracking)
- Missing some features that would be valuable in real analysis: customer age/gender, device model, competitor pricing, mobile money integration
- Simplified pricing model (doesn't capture promotional bundles, loyalty discounts, etc.)
- No network quality metrics (latency, dropped calls, signal strength)
- Synthetic data (not actual customer records - by design for portfolio demonstration)

**Potential enhancements I'm considering:**
- Time-series version with monthly customer snapshots to enable trend analysis
- Addition of mobile money usage (MOMO integration is huge in Ghana)
- Cell site/tower data for network quality analysis
- Competitor switching patterns
- Device upgrade history
- More granular regional data (district level instead of just region)

If you have suggestions or want to collaborate on enhancements, feel free to reach out.

---

## Data Sources

This synthetic dataset is calibrated to real market data from:

1. **National Communications Authority (NCA)** - Q4 2024 Telecom Subscription & Traffic Statistical Bulletin
   - Market structure (postpaid/prepaid split)
   - Average data usage, voice MOU, SMS volumes
   - Operator market shares

2. **MTN Ghana** - 2024 Annual Financial Report
   - ARPU benchmarks
   - Financial performance metrics
   - Service mix (voice, data, value-added services)

3. **Ghana Statistical Service** - 2021 Population and Housing Census
   - Regional population distributions
   - Urban/rural classifications by region
   - Demographic patterns

4. **Regional Factors Research** - Original research combining:
   - nPerf Network Coverage Maps 2024
   - World Bank Ghana Poverty Report 2016/2024
   - Market analysis and operator presence data

All sources are documented with citations in `data/regional_factors.json`.

---

## Technologies Used

**Built with:**
- Python 3.9+
- pandas (data manipulation)
- NumPy (random number generation, array operations)
- SciPy (statistical distributions)
- scikit-learn (ML validation in examples)
- matplotlib & seaborn (visualizations in validation notebook)

**Statistical methods:**
- Probabilistic modeling with appropriate distributions (exponential for tenure, lognormal for usage, Poisson for event counts)
- Multi-factor causal modeling (churn risk scoring)
- Ground truth calibration and validation
- Hypothesis testing (t-tests for group differences)

---

## License

This project is open source and available under the MIT License. Feel free to use the data for learning, research, or portfolio projects.

---

## Contact

**Mufti**
- Email: galiishaq@gmail.com
- LinkedIn: [linkedin.com/in/muftawu-ishaq](https://linkedin.com/in/muftawu-ishaq-85699a271)
- GitHub: [@galiishaq](https://github.com/galiishaq)

Questions, suggestions, or interested in collaborating? Feel free to reach out.

---

*This dataset was created to support analytics education and practice in the context of African telecom markets, where quality public datasets are scarce.*
