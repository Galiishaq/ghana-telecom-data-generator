"""Synthetic telecom customer generator calibrated to Ghana market data.

Generates realistic customer records with causal churn patterns based on:
- NCA Q4 2024 Statistical Bulletin
- MTN Ghana 2024 Financial Report
- Ghana Statistical Service 2021 Census
"""

from dataclasses import dataclass, asdict
from typing import Literal
import json
import pandas as pd
import numpy as np


# ============================================================================
# Configuration
# ============================================================================

PRICING = {
    'voice_per_min': 0.15,
    'data_per_gb': 8.50,
    'sms': 0.05,
}

RISK_WEIGHTS = {
    'low_usage': 30,
    'moderate_low_usage': 15,
    'high_engagement': -10,
    'high_support': 25,
    'moderate_support': 15,
    'low_support': -5,
    'low_arpu': 20,
    'moderate_arpu': 10,
    'high_arpu': -15,
    'short_tenure': 15,
    'moderate_tenure': 8,
    'long_tenure': -20,
    'postpaid': -25,
    'plan_changes': 12,
    'night_user': -8,
    'low_recharge': 18,
    'urban': 5,
}

# Regional churn modifiers based on network quality, competition, and economic factors
# Source: /home/mufti/analytics/dataset_generator/data/regional_factors.json
REGIONAL_CHURN_MODIFIER = {
    'Greater Accra': 0.0,      # Excellent network (0.95), high competition (0.85), strongest economy (1.30)
    'Ashanti': -2.0,           # Excellent network (0.92), high competition (0.75), strong economy (1.10)
    'Eastern': 1.0,            # Good network (0.82), moderate competition (0.60), moderate economy (0.95)
    'Central': 0.5,            # Good network (0.80), moderate competition (0.58), moderate economy (0.92)
    'Western': -1.0,           # Good network (0.78), moderate-high competition (0.62), resource-rich (1.05)
    'Western North': 3.0,      # Poor network (0.68), low competition (0.48), developing region (0.88)
    'Volta': 4.0,              # Fair network (0.75), moderate competition (0.55), weaker economy (0.85)
    'Oti': 5.0,                # Poor network (0.65), low competition (0.42), new region (0.78)
    'Bono': 2.0,               # Fair network (0.74), moderate competition (0.54), middle-income (0.90)
    'Bono East': 2.5,          # Fair network (0.70), moderate competition (0.50), developing (0.88)
    'Ahafo': 1.5,              # Fair network (0.72), moderate competition (0.52), mining economy (0.92)
    'Northern': 8.0,           # Poor network (0.62), low competition (0.45), high poverty 30.7% (0.65)
    'North East': 10.0,        # Poor network (0.58), low competition (0.38), extreme poverty 38% (0.60)
    'Savannah': 9.0,           # Poor network (0.60), low competition (0.40), high poverty 36% (0.62)
    'Upper East': 12.0,        # Poor network (0.57), low competition (0.35), extreme poverty 40% (0.58)
    'Upper West': 14.0,        # Worst network (0.55), lowest competition (0.32), highest poverty 45.2% (0.55)
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CustomerSegment:
    name: str
    proportion: float
    base_churn: float
    avg_tenure: float
    arpu_range: tuple[float, float]
    usage_range: tuple[float, float]
    support_lambda: float
    p_postpaid: float


@dataclass
class Demographics:
    region: str
    locality: Literal['urban', 'rural']
    subscription: Literal['prepaid', 'postpaid']
    tenure: float
    device: Literal['smartphone', 'feature_phone']


@dataclass
class Usage:
    data_gb: float
    voice_mou: float
    sms: int


@dataclass
class Revenue:
    voice: float
    data: float
    sms: float
    total_arpu: float


@dataclass
class Behavior:
    support_calls: int
    plan_changes: int
    recharge_freq: int
    night_user: bool
    social_data_gb: float


@dataclass
class Customer:
    customer_id: str
    demographics: Demographics
    usage: Usage
    revenue: Revenue
    behavior: Behavior
    segment: str
    risk_score: float
    churned: bool

    def to_flat_dict(self):
        """Flatten nested structure for CSV output."""
        return {
            'customer_id': self.customer_id,
            'region': self.demographics.region,
            'locality_type': self.demographics.locality,
            'subscription_type': self.demographics.subscription,
            'tenure_months': round(self.demographics.tenure, 1),
            'device_type': self.demographics.device,
            'monthly_data_usage_gb': round(self.usage.data_gb, 3),
            'monthly_voice_mou': round(self.usage.voice_mou, 1),
            'monthly_sms_count': self.usage.sms,
            'estimated_monthly_arpu_gh': round(self.revenue.total_arpu, 2),
            'voice_revenue_gh': round(self.revenue.voice, 2),
            'data_revenue_gh': round(self.revenue.data, 2),
            'support_calls_last_3months': self.behavior.support_calls,
            'plan_changes_last_year': self.behavior.plan_changes,
            'recharge_frequency_monthly': self.behavior.recharge_freq,
            'night_data_user': int(self.behavior.night_user),
            'social_media_data_gb': round(self.behavior.social_data_gb, 3),
            'customer_segment': self.segment,
            'churn_risk_score': round(self.risk_score, 2),
            'churned': int(self.churned),
        }


# ============================================================================
# Customer Segments
# ============================================================================

def define_segments():
    """Return the five customer segment profiles."""
    return [
        CustomerSegment('loyal_champions', 0.15, 0.02, 36, (1.5, 2.5), (1.3, 2.0), 1.5, 0.15),
        CustomerSegment('satisfied_majority', 0.50, 0.05, 18, (0.8, 1.2), (0.8, 1.2), 2.5, 0.01),
        CustomerSegment('at_risk', 0.20, 0.25, 12, (0.4, 0.7), (0.3, 0.6), 6.0, 0.005),
        CustomerSegment('price_sensitive', 0.10, 0.15, 15, (0.5, 0.8), (0.6, 0.9), 3.5, 0.002),
        CustomerSegment('new_exploring', 0.05, 0.18, 6, (0.7, 1.3), (0.5, 1.5), 4.0, 0.005),
    ]


# ============================================================================
# Data Loading
# ============================================================================

def load_ground_truth(path):
    """Load and return ground truth data from JSON."""
    with open(path) as f:
        return json.load(f)


def extract_constraints(ground_truth):
    """Extract key constraints from ground truth."""
    return {
        'avg_data_mb': ground_truth['usage_metrics']['avg_data_usage_per_sub']['value'],
        'avg_voice_mou': ground_truth['usage_metrics']['avg_voice_mou']['value'],
        'avg_sms': ground_truth['usage_metrics']['avg_sms_per_sub']['value'],
        'avg_arpu': ground_truth['financial_metrics']['estimated_mtn_total_arpu']['value'],
    }


def build_region_distribution(ground_truth):
    """Build region names, probabilities, and urban percentages."""
    regions_data = ground_truth['demographics']['population_by_region']

    names, probs, urban_pcts = [], [], {}
    for key, info in regions_data.items():
        name = key.replace('_', ' ').title()
        names.append(name)
        probs.append(info['percent'] / 100)
        urban_pcts[name] = info['urban_percent'] / 100

    # Normalize probabilities
    total = sum(probs)
    probs = [p / total for p in probs]

    return names, probs, urban_pcts


# ============================================================================
# Sampling Functions
# ============================================================================

def sample_demographics(segment, regions, rng):
    """Sample customer demographics based on segment profile."""
    names, probs, urban_pcts = regions

    region = rng.choice(names, p=probs)
    locality = 'urban' if rng.random() < urban_pcts[region] else 'rural'
    subscription = 'postpaid' if rng.random() < segment.p_postpaid else 'prepaid'
    tenure = np.clip(rng.exponential(segment.avg_tenure), 0.5, 120)

    # Device type depends on location, segment, and tenure
    p_smartphone = 0.75 if locality == 'urban' else 0.35
    if segment.name == 'loyal_champions':
        p_smartphone = min(p_smartphone * 1.2, 0.95)
    elif segment.name == 'at_risk':
        p_smartphone *= 0.7
    p_smartphone += min(tenure / 120 * 0.1, 0.15)
    p_smartphone = min(p_smartphone, 0.95)

    device = 'smartphone' if rng.random() < p_smartphone else 'feature_phone'

    return Demographics(region, locality, subscription, tenure, device)


def sample_usage(demographics, segment, constraints, rng):
    """Sample usage patterns calibrated to ground truth."""
    usage_mult = rng.uniform(*segment.usage_range)

    # Data usage (lognormal distribution)
    mu = np.log(constraints['avg_data_mb'] * usage_mult) - 0.72
    data_mb = rng.lognormal(mu, 1.2)
    if demographics.device == 'feature_phone':
        data_mb *= 0.15
    if demographics.locality == 'urban':
        data_mb *= 1.15
    data_gb = max(data_mb / 1024, 0.001)

    # Voice and SMS
    voice_mou = max(rng.normal(constraints['avg_voice_mou'] * usage_mult, 150), 0)
    if demographics.subscription == 'postpaid':
        voice_mou *= 1.3

    sms = rng.poisson(constraints['avg_sms'] * usage_mult)

    return Usage(data_gb, voice_mou, sms)


def calculate_revenue(usage, demographics, segment, rng):
    """Calculate revenue from usage with segment-based multiplier."""
    arpu_mult = rng.uniform(*segment.arpu_range)

    voice_rev = usage.voice_mou * PRICING['voice_per_min']
    data_rev = usage.data_gb * PRICING['data_per_gb']
    sms_rev = usage.sms * PRICING['sms']

    total_arpu = (voice_rev + data_rev + sms_rev) * arpu_mult * rng.uniform(0.85, 1.15)

    return Revenue(voice_rev, data_rev, sms_rev, total_arpu)


def sample_behavior(demographics, segment, usage, rng):
    """Sample behavioral indicators correlated with segment."""
    # Support calls (higher in rural areas)
    lambda_support = segment.support_lambda * (1.4 if demographics.locality == 'rural' else 1.0)
    support_calls = min(rng.poisson(lambda_support), 15)

    # Plan changes (at-risk customers shop around more)
    if segment.name in ('at_risk', 'price_sensitive'):
        lambda_changes = min(demographics.tenure / 6, 4)
    else:
        lambda_changes = min(demographics.tenure / 24, 2)
    plan_changes = rng.poisson(lambda_changes)

    # Recharge frequency
    if demographics.subscription == 'prepaid':
        lambda_recharge = 4 if segment.name == 'at_risk' else 8
        recharge_freq = rng.poisson(lambda_recharge)
    else:
        recharge_freq = 1

    # Night data user (engagement indicator)
    p_night = 0.45 if segment.name in ('loyal_champions', 'satisfied_majority') else 0.25
    night_user = rng.random() < p_night

    # Social media data
    social_pct = rng.uniform(0.25, 0.55) if demographics.device == 'smartphone' else 0.1
    social_data_gb = usage.data_gb * social_pct

    return Behavior(support_calls, plan_changes, recharge_freq, night_user, social_data_gb)


def calculate_churn_risk(customer, segment, constraints):
    """Calculate churn risk score from multiple factors."""
    score = segment.base_churn * 100

    # Usage engagement
    avg_usage = (
        customer.usage.data_gb / (constraints['avg_data_mb'] / 1024) +
        customer.usage.voice_mou / constraints['avg_voice_mou'] +
        customer.usage.sms / constraints['avg_sms']
    ) / 3

    if avg_usage < 0.3:
        score += RISK_WEIGHTS['low_usage']
    elif avg_usage < 0.5:
        score += RISK_WEIGHTS['moderate_low_usage']
    elif avg_usage > 1.5:
        score += RISK_WEIGHTS['high_engagement']

    # Support issues
    if customer.behavior.support_calls > 7:
        score += RISK_WEIGHTS['high_support']
    elif customer.behavior.support_calls > 4:
        score += RISK_WEIGHTS['moderate_support']
    elif customer.behavior.support_calls <= 1:
        score += RISK_WEIGHTS['low_support']

    # ARPU (customer value)
    arpu_ratio = customer.revenue.total_arpu / constraints['avg_arpu']
    if arpu_ratio < 0.5:
        score += RISK_WEIGHTS['low_arpu']
    elif arpu_ratio < 0.7:
        score += RISK_WEIGHTS['moderate_arpu']
    elif arpu_ratio > 1.5:
        score += RISK_WEIGHTS['high_arpu']

    # Tenure (loyalty)
    if customer.demographics.tenure < 6:
        score += RISK_WEIGHTS['short_tenure']
    elif customer.demographics.tenure < 12:
        score += RISK_WEIGHTS['moderate_tenure']
    elif customer.demographics.tenure > 36:
        score += RISK_WEIGHTS['long_tenure']

    # Other factors
    if customer.demographics.subscription == 'postpaid':
        score += RISK_WEIGHTS['postpaid']
    if customer.behavior.plan_changes > 3:
        score += RISK_WEIGHTS['plan_changes']
    if customer.behavior.night_user:
        score += RISK_WEIGHTS['night_user']
    if customer.demographics.subscription == 'prepaid' and customer.behavior.recharge_freq < 4:
        score += RISK_WEIGHTS['low_recharge']
    if customer.demographics.locality == 'urban':
        score += RISK_WEIGHTS['urban']

    # Regional factors: network quality, competition, economic conditions
    score += REGIONAL_CHURN_MODIFIER.get(customer.demographics.region, 0)

    return score


def to_churn_probability(risk_score):
    """Convert risk score to churn probability using logistic function."""
    p_churn = 1 / (1 + np.exp(-(risk_score - 50) / 15))
    return np.clip(p_churn, 0.001, 0.95)


# ============================================================================
# Main Generation
# ============================================================================

def generate_customer(customer_id, segment, regions, constraints, rng):
    """Generate a single customer with all attributes."""
    demographics = sample_demographics(segment, regions, rng)
    usage = sample_usage(demographics, segment, constraints, rng)
    revenue = calculate_revenue(usage, demographics, segment, rng)
    behavior = sample_behavior(demographics, segment, usage, rng)

    customer = Customer(
        customer_id=customer_id,
        demographics=demographics,
        usage=usage,
        revenue=revenue,
        behavior=behavior,
        segment=segment.name,
        risk_score=0,
        churned=False,
    )

    risk_score = calculate_churn_risk(customer, segment, constraints)
    p_churn = to_churn_probability(risk_score)
    churned = rng.random() < p_churn

    customer.risk_score = risk_score
    customer.churned = churned

    return customer


def generate_customers(n, segments, regions, constraints, seed=42):
    """Generate n customers across all segments."""
    rng = np.random.default_rng(seed)

    # Sample segment assignments
    segment_names = [s.name for s in segments]
    segment_probs = [s.proportion for s in segments]
    segment_map = {s.name: s for s in segments}

    assignments = rng.choice(segment_names, size=n, p=segment_probs)

    customers = []
    for i, segment_name in enumerate(assignments):
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1:,} customers...")

        customer_id = f"MTN_{i:07d}"
        segment = segment_map[segment_name]
        customer = generate_customer(customer_id, segment, regions, constraints, rng)
        customers.append(customer)

    return customers


# ============================================================================
# Output and Reporting
# ============================================================================

def save_and_report(customers, output_path):
    """Save customers to CSV and print summary statistics."""
    df = pd.DataFrame([c.to_flat_dict() for c in customers])
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved {len(df):,} customers to {output_path}\n")
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nChurn Rate: {df['churned'].mean()*100:.1f}%")
    print(f"Avg ARPU: GH₵{df['estimated_monthly_arpu_gh'].mean():.2f}")
    print(f"Avg Tenure: {df['tenure_months'].mean():.1f} months")

    print("\nSegment Distribution:")
    for segment, group in df.groupby('customer_segment'):
        churn_pct = group['churned'].mean() * 100
        print(f"  {segment:20s}: {len(group):6,} ({len(group)/len(df)*100:4.1f}%) - Churn: {churn_pct:5.1f}%")

    print(f"\nRisk Score - Churned: {df[df['churned']==1]['churn_risk_score'].mean():.1f}")
    print(f"Risk Score - Active:  {df[df['churned']==0]['churn_risk_score'].mean():.1f}")
    print("=" * 70)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Generate synthetic telecom customer dataset."""
    print("=" * 70)
    print("GHANA TELECOM SYNTHETIC CUSTOMER GENERATOR")
    print("=" * 70)

    ground_truth = load_ground_truth('data/master_ground_truth_q4_2024.json')
    constraints = extract_constraints(ground_truth)
    regions = build_region_distribution(ground_truth)
    segments = define_segments()

    print("\nGenerating 100,000 customers with causal churn patterns...\n")
    customers = generate_customers(100_000, segments, regions, constraints, seed=42)

    save_and_report(customers, 'data/ghana_telecom_customers.csv')


if __name__ == '__main__':
    main()
