# Data Dictionary: MTN Synthetic Customer Dataset

## Overview
This document describes all 21 features in the synthetic customer dataset (`mtn_synthetic_customers_100k.csv`).

---

## Feature Categories

### 1. Identifier
### 2. Demographics
### 3. Usage Metrics
### 4. Revenue
### 5. Behavioral Indicators
### 6. Target Variable
### 7. Metadata (for validation only)

---

## Detailed Feature Descriptions

### 1. IDENTIFIER

#### `customer_id`
- **Type:** String
- **Format:** MTN_XXXXXXX (7-digit zero-padded)
- **Example:** MTN_0000001, MTN_0042567
- **Description:** Unique identifier for each customer
- **Usage:** Index for tracking, not used in predictive modeling
- **Range:** MTN_0000000 to MTN_0099999

---

### 2. DEMOGRAPHICS

#### `region`
- **Type:** Categorical (16 levels)
- **Values:** Greater Accra, Ashanti, Eastern, Central, Northern, Western, Volta, Upper East, Bono, Bono East, Upper West, Western North, Oti, North East, Savannah, Ahafo
- **Description:** Customer's geographic region in Ghana
- **Source:** Ghana Statistical Service 2021 Census
- **Distribution:** Weighted by actual regional population percentages
- **Example:** Greater Accra (17.7%), Ashanti (17.6%), Eastern (9.5%)
- **Business Use:** Regional marketing, network planning, churn analysis by location

#### `locality_type`
- **Type:** Binary categorical
- **Values:**
  - `urban` - Customer in urban area
  - `rural` - Customer in rural area
- **Description:** Urbanization classification
- **Generation:** Probabilistic based on region-specific urban percentages (e.g., Greater Accra 91.7% urban)
- **Distribution:** ~60% urban, ~40% rural (nationally)
- **Business Use:** Device targeting, network coverage analysis, pricing strategies

#### `subscription_type`
- **Type:** Binary categorical
- **Values:**
  - `prepaid` - Pay-as-you-go service (~98.9%)
  - `postpaid` - Contract-based service (~1.1%)
- **Description:** Customer's billing arrangement
- **Source:** NCA Q4 2024 data (MTN postpaid: 1.08%)
- **Generation:** Segment-dependent probability (loyal customers more likely postpaid)
- **Churn Impact:** Postpaid customers have -25 risk score points (lower churn)
- **Business Use:** Contract retention, payment processing, service tier management

#### `tenure_months`
- **Type:** Continuous (float)
- **Range:** 0.5 - 120.0 months
- **Unit:** Months as active customer
- **Distribution:** Exponential with segment-specific means:
  - Loyal Champions: 36 months average
  - Satisfied Majority: 18 months
  - At-Risk: 12 months
  - Price Sensitive: 15 months
  - New/Exploring: 6 months
- **Description:** How long customer has been with MTN
- **Churn Impact:**
  - < 6 months: +15 risk points
  - < 12 months: +8 risk points
  - > 36 months: -20 risk points (loyal)
- **Business Use:** Loyalty analysis, lifetime value calculation, winback campaigns

#### `device_type`
- **Type:** Binary categorical
- **Values:**
  - `smartphone` - Modern smartphone (~57%)
  - `feature_phone` - Basic phone (~43%)
- **Description:** Type of device customer uses
- **Generation:** Influenced by:
  - Urban areas: 75% smartphone penetration
  - Rural areas: 35% smartphone penetration
  - Loyal customers: Higher smartphone adoption
  - Tenure: Longer customers more likely upgraded
- **Churn Impact:** Indirect (affects data usage, which affects engagement)
- **Business Use:** Data bundle targeting, app-based services, device financing

---

### 3. USAGE METRICS

#### `monthly_data_usage_gb`
- **Type:** Continuous (float, 3 decimals)
- **Range:** 0.001 - 300+ GB
- **Unit:** Gigabytes per month
- **Target Mean:** 28.8 GB (29,463 MB from NCA data)
- **Distribution:** Lognormal (right-skewed, realistic for usage data)
- **Generation Factors:**
  - Segment usage multiplier (loyal: 1.3-2.0x, at-risk: 0.3-0.6x)
  - Feature phones: 15% of base usage
  - Urban customers: 1.15x multiplier
- **Churn Impact:**
  - < 30% average usage: +30 risk points
  - < 50% average: +15 points
  - > 150% average: -10 points (high engagement)
- **Business Use:** Network capacity planning, bundle design, engagement scoring

#### `monthly_voice_mou`
- **Type:** Continuous (float, 1 decimal)
- **Range:** 0.0 - 2000+ minutes
- **Unit:** Minutes of Use (MOU) per month
- **Target Mean:** 757.52 minutes
- **Distribution:** Normal distribution, segment-adjusted
- **Generation Factors:**
  - Segment usage multiplier
  - Postpaid: 1.3x multiplier (higher usage)
  - Standard deviation: 150 minutes
- **Churn Impact:** Included in composite usage metric
- **Business Use:** Voice revenue forecasting, bundle minutes allocation

#### `monthly_sms_count`
- **Type:** Integer
- **Range:** 0 - 200+ messages
- **Unit:** SMS messages per month
- **Target Mean:** 29 messages (NCA data)
- **Distribution:** Poisson distribution, segment-adjusted
- **Generation:** Lambda parameter varies by segment usage multiplier
- **Churn Impact:** Included in composite usage metric (declining importance)
- **Business Use:** SMS bundle design, OTT messaging impact analysis

#### `social_media_data_gb`
- **Type:** Continuous (float, 3 decimals)
- **Range:** 0.001 - 150+ GB
- **Unit:** Gigabytes per month
- **Description:** Estimated social media (Facebook, WhatsApp, Instagram, TikTok) data usage
- **Generation:**
  - Smartphones: 25-55% of total data
  - Feature phones: ~10% (mainly WhatsApp)
- **Business Use:** Social media bundle targeting, zero-rating partnerships, youth market analysis

---

### 4. REVENUE

#### `estimated_monthly_arpu_gh`
- **Type:** Continuous (float, 2 decimals)
- **Range:** 10 - 3000+ GH₵
- **Unit:** Ghana Cedis (GH₵) per month
- **Description:** Average Revenue Per User
- **Target Mean:** GH₵52.36 (from MTN 2024 report, industry average)
- **Actual Mean:** GH₵349.29 (higher due to segment modeling with high-value customers)
- **Calculation:** (Voice revenue + Data revenue + SMS revenue) × ARPU multiplier × bundle variance (0.85-1.15)
- **Segment Multipliers:**
  - Loyal Champions: 1.5-2.5x
  - Satisfied Majority: 0.8-1.2x
  - At-Risk: 0.4-0.7x
  - Price Sensitive: 0.5-0.8x
  - New/Exploring: 0.7-1.3x
- **Churn Impact:**
  - < 50% average ARPU: +20 risk points
  - < 70% average: +10 points
  - > 150% average: -15 points (high value)
- **Business Use:** Revenue forecasting, customer value scoring, retention ROI

#### `voice_revenue_gh`
- **Type:** Continuous (float, 2 decimals)
- **Range:** 0.00 - 300+ GH₵
- **Unit:** Ghana Cedis per month
- **Calculation:** monthly_voice_mou × GH₵0.15/minute
- **Pricing:** GH₵0.15 per minute (estimated from industry rates)
- **Business Use:** Voice revenue attribution, bundle pricing optimization

#### `data_revenue_gh`
- **Type:** Continuous (float, 2 decimals)
- **Range:** 0.00 - 2500+ GH₵
- **Unit:** Ghana Cedis per month
- **Calculation:** monthly_data_usage_gb × GH₵8.50/GB
- **Pricing:** GH₵8.50 per GB (estimated from bundle rates)
- **Note:** Actual revenue may vary due to bundles, promotions
- **Business Use:** Data revenue attribution, bundle effectiveness analysis

---

### 5. BEHAVIORAL INDICATORS

#### `failed_payments`
- **Type:** Integer
- **Range:** 0 - 3
- **Unit:** Count of failed payment attempts
- **Description:** Number of unsuccessful payment/recharge attempts in recent period
- **Generation:** Segment-based probability:
  - Loyal Champions: 1% base probability
  - At-Risk: 20% base probability
  - Prepaid: 2x multiplier
  - If triggered: 60% chance of 1 failure, 30% chance of 2, 10% chance of 3
- **Churn Impact:** **+15 risk points per failed payment** (strong predictor)
- **Business Use:** Payment plan offers, grace period policies, collection strategy

#### `support_calls_last_3months`
- **Type:** Integer
- **Range:** 0 - 15
- **Unit:** Number of calls to customer support
- **Distribution:** Poisson with segment-specific lambda:
  - Loyal Champions: λ = 1.5
  - At-Risk: λ = 6.0
  - Rural customers: λ × 1.4 (more network issues)
- **Churn Impact:**
  - > 7 calls: +25 risk points (major dissatisfaction)
  - 5-7 calls: +15 points
  - ≤ 1 call: -5 points (satisfaction)
- **Business Use:** Service quality metrics, proactive support, network issue identification

#### `plan_changes_last_year`
- **Type:** Integer
- **Range:** 0 - 12+
- **Unit:** Number of times customer changed service plan
- **Distribution:** Poisson with tenure and segment adjustments:
  - At-Risk/Price Sensitive: λ = tenure / 6 (shopping around)
  - Others: λ = tenure / 24 (stable)
- **Churn Impact:** > 3 changes: +12 risk points (seeking better deals)
- **Business Use:** Plan satisfaction analysis, competitive offer alerts

#### `recharge_frequency_monthly`
- **Type:** Integer
- **Range:** 1 - 20+
- **Unit:** Number of recharges/payments per month
- **Description:** How often customer adds credit (prepaid) or pays bill (postpaid)
- **Generation:**
  - Prepaid At-Risk: λ = 4 (low engagement)
  - Prepaid Others: λ = 8
  - Postpaid: 1 (monthly bill)
- **Churn Impact:** Prepaid with < 4 recharges: +18 risk points (disengagement)
- **Business Use:** Engagement scoring, auto-recharge targeting, bundle duration optimization

#### `night_data_user`
- **Type:** Binary (0/1)
- **Values:**
  - 1 = Uses night data bundles
  - 0 = Does not use night data
- **Description:** Whether customer uses data during night hours (bonus/discounted data)
- **Probability:**
  - Loyal/Satisfied: 45% chance
  - Others: 25% chance
- **Churn Impact:** -8 risk points (engagement indicator)
- **Business Use:** Night bundle targeting, network load balancing, engagement programs

---

### 6. TARGET VARIABLE

#### `churned`
- **Type:** Binary (0/1)
- **Values:**
  - 0 = Active customer (retained) - 77.2%
  - 1 = Churned customer (left) - 22.8%
- **Description:** Whether customer has churned (left MTN for competitor or stopped service)
- **Generation:** Calculated from multi-factor churn risk score using logistic function
- **Formula:** P(churn) = 1 / (1 + exp(-(risk_score - 50) / 15))
- **Distribution:** Varies dramatically by segment:
  - Loyal Champions: 1.3% churn
  - Satisfied Majority: 4.3% churn
  - At-Risk: 82.6% churn
  - Price Sensitive: 25.2% churn
  - New/Exploring: 27.4% churn
- **Business Use:** **PRIMARY ML TARGET** for churn prediction models

---

### 7. METADATA (Validation Only - Not for Prediction)

#### `customer_segment`
- **Type:** Categorical (5 levels)
- **Values:** loyal_champions, satisfied_majority, at_risk, price_sensitive, new_exploring
- **Description:** Ground truth customer segment used during generation
- **Purpose:** Validation and analysis only
- **Warning:** ⚠️ **DO NOT use as ML feature** - this is how data was generated, not a real observable
- **Distribution:**
  - satisfied_majority: 50.0%
  - at_risk: 20.1%
  - loyal_champions: 15.0%
  - price_sensitive: 10.0%
  - new_exploring: 4.9%
- **Business Use:** Validation that synthetic data has realistic segment structure

#### `churn_risk_score`
- **Type:** Continuous (float, 2 decimals)
- **Range:** -50 to 150+
- **Unit:** Risk score points (arbitrary scale)
- **Description:** Multi-factor churn risk score calculated during generation
- **Purpose:** Analysis and validation only
- **Warning:** ⚠️ **DO NOT use as ML feature** - this is the intermediate calculation used to generate churn label
- **Mean Scores:**
  - Churned customers: 77.59
  - Active customers: -1.46
  - Separation: 79.05 points
- **Business Use:** Validate that churn model is learning the right patterns (scores should predict churn)

---

## Feature Engineering Suggestions

When using this dataset for ML, consider creating:

1. **Composite Usage Score:**
   ```python
   usage_score = (data_gb / avg_data + voice_mou / avg_voice + sms / avg_sms) / 3
   ```

2. **Revenue Efficiency:**
   ```python
   arpu_per_gb = estimated_monthly_arpu_gh / (monthly_data_usage_gb + 0.1)
   ```

3. **Payment Health:**
   ```python
   payment_health = 1 if failed_payments == 0 else 0
   ```

4. **Support Burden Flag:**
   ```python
   high_support = 1 if support_calls_last_3months > 5 else 0
   ```

5. **Engagement Level:**
   ```python
   engagement = tenure_months * (usage_score + night_data_user)
   ```

6. **ARPU Percentile:**
   ```python
   arpu_percentile = df['estimated_monthly_arpu_gh'].rank(pct=True)
   ```

---

## Data Quality Notes

### Strengths
- ✅ Calibrated to real-world ground truth metrics
- ✅ Causal relationships between features and target
- ✅ Realistic distributions (exponential, lognormal, Poisson)
- ✅ Segment-based behavioral patterns
- ✅ Statistical significance in churn patterns

### Limitations
- ⚠️ Synthetic data - not actual customer records
- ⚠️ ARPU higher than industry average (due to inclusion of high-value segments)
- ⚠️ Simplified pricing model (doesn't capture all bundle complexity)
- ⚠️ No temporal dimension (single snapshot, not time-series)
- ⚠️ Missing features: Age, gender, competitor info, device model, MoMo usage

### Use Cases
- ✅ Churn prediction model development
- ✅ Retention strategy analysis
- ✅ Segmentation analysis
- ✅ Revenue optimization
- ✅ ML pipeline testing
- ❌ Production deployment (requires real data validation)

---

## Example Queries

### High-value at-risk customers
```python
critical = df[(df['estimated_monthly_arpu_gh'] > df['estimated_monthly_arpu_gh'].quantile(0.75)) &
              (df['churn_risk_score'] > 50)]
```

### Low-engagement prepaid customers
```python
low_engagement = df[(df['subscription_type'] == 'prepaid') &
                     (df['recharge_frequency_monthly'] < 4) &
                     (df['monthly_data_usage_gb'] < 5)]
```

### Rural customers with network issues
```python
network_issues = df[(df['locality_type'] == 'rural') &
                     (df['support_calls_last_3months'] > 5)]
```

### Segment analysis
```python
segment_summary = df.groupby('customer_segment').agg({
    'customer_id': 'count',
    'churned': 'mean',
    'estimated_monthly_arpu_gh': 'mean'
})
```

---

## Version History

- **v1.0 (2025-11-15):** Initial release with 100,000 customers, 21 features
- **Methodology:** Advanced causal modeling with multi-factor churn risk scoring

---

**Documentation complete. For questions or clarifications, refer to README.md or source code comments.**
