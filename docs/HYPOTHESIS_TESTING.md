# Hypothesis Testing in ML Systems

## Overview
Hypothesis testing is a critical statistical method used in ML systems to make data-driven decisions about model performance, feature impact, and system changes.

---

## 1. Model Performance Comparison

### Use Case
When comparing different models (e.g., Model A vs. Model B), use hypothesis testing to statistically determine if one model is significantly better than the other in terms of ETA accuracy.

### Statistical Tests
- **Paired t-test**
- **Wilcoxon signed-rank test** (for prediction accuracy)

### Hypotheses

**Null Hypothesis (H0)**: There is no significant difference in the mean prediction error between Model A and Model B.

**Alternative Hypothesis (H1)**: There is a significant difference in the mean prediction error between Model A and Model B.

### Implementation Example

```python
import numpy as np
from scipy import stats

def compare_model_performance(model_a_errors, model_b_errors, alpha=0.05):
    """
    Compare two models using paired t-test
    
    Args:
        model_a_errors: Array of prediction errors from Model A
        model_b_errors: Array of prediction errors from Model B
        alpha: Significance level (default 0.05)
    
    Returns:
        dict: Test results including p-value and decision
    """
    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(model_a_errors, model_b_errors)
    
    # Calculate mean errors
    mean_error_a = np.mean(np.abs(model_a_errors))
    mean_error_b = np.mean(np.abs(model_b_errors))
    
    # Make decision
    is_significant = p_value < alpha
    
    results = {
        't_statistic': t_statistic,
        'p_value': p_value,
        'mean_error_model_a': mean_error_a,
        'mean_error_model_b': mean_error_b,
        'is_significant': is_significant,
        'conclusion': (
            f"Model B is significantly better (p={p_value:.4f})" 
            if is_significant and mean_error_b < mean_error_a
            else f"Model A is significantly better (p={p_value:.4f})"
            if is_significant and mean_error_a < mean_error_b
            else f"No significant difference (p={p_value:.4f})"
        )
    }
    
    return results

# Example usage
model_a_errors = np.array([5.2, 4.8, 6.1, 5.5, 4.9])  # MAE in minutes
model_b_errors = np.array([4.1, 3.9, 4.5, 4.2, 3.8])  # MAE in minutes

results = compare_model_performance(model_a_errors, model_b_errors)
print(results['conclusion'])
```

---

## 2. Impact of New Features

### Use Case
When introducing new features, use hypothesis testing to assess if the new features significantly improve ETA accuracy. Compare model performance with and without the new features.

### Hypotheses

**H0**: Adding the new features does **not** improve ETA prediction accuracy.

**H1**: Adding the new features **significantly improves** ETA prediction accuracy.

### Implementation Example

```python
def test_new_features_impact(baseline_errors, new_feature_errors, alpha=0.05):
    """
    Test if new features significantly improve model performance
    
    Args:
        baseline_errors: Errors from model without new features
        new_feature_errors: Errors from model with new features
        alpha: Significance level
    
    Returns:
        dict: Test results and recommendation
    """
    # One-tailed t-test (we expect improvement)
    t_statistic, p_value = stats.ttest_rel(
        baseline_errors, 
        new_feature_errors,
        alternative='greater'  # Testing if baseline > new_features
    )
    
    mean_baseline = np.mean(np.abs(baseline_errors))
    mean_new = np.mean(np.abs(new_feature_errors))
    improvement_pct = ((mean_baseline - mean_new) / mean_baseline) * 100
    
    is_significant = p_value < alpha
    
    results = {
        'baseline_mae': mean_baseline,
        'new_features_mae': mean_new,
        'improvement_percentage': improvement_pct,
        'p_value': p_value,
        'is_significant': is_significant,
        'recommendation': (
            f"✅ Deploy new features! Significant improvement of {improvement_pct:.2f}% (p={p_value:.4f})"
            if is_significant and improvement_pct > 0
            else f"❌ Do not deploy. No significant improvement (p={p_value:.4f})"
        )
    }
    
    return results

# Example usage
baseline_errors = np.array([5.5, 5.8, 5.2, 5.9, 5.4])
new_feature_errors = np.array([4.2, 4.5, 4.1, 4.6, 4.3])

results = test_new_features_impact(baseline_errors, new_feature_errors)
print(results['recommendation'])
```

---

## 3. A/B Testing for System Changes

### Use Case
When rolling out changes to the ETA prediction system (e.g., model updates, feature engineering changes), perform A/B tests to compare the new system version against the old version in a live environment.

### Metrics to Monitor
- ETA accuracy
- Customer satisfaction
- Rider efficiency

### Hypotheses

**H0**: There is no significant difference in ETA accuracy or customer satisfaction between the old and new system versions.

**H1**: The new system version **significantly improves** ETA accuracy or customer satisfaction.

### Implementation Example

```python
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

class ABTestAnalyzer:
    """
    A/B Testing framework for system changes
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def test_eta_accuracy(self, control_errors, treatment_errors):
        """
        Test if new system (treatment) has better ETA accuracy than old (control)
        
        Args:
            control_errors: Errors from old system (Group A)
            treatment_errors: Errors from new system (Group B)
        
        Returns:
            dict: Test results
        """
        # Mann-Whitney U test (non-parametric alternative to t-test)
        statistic, p_value = mannwhitneyu(
            control_errors, 
            treatment_errors,
            alternative='greater'
        )
        
        control_mae = np.mean(np.abs(control_errors))
        treatment_mae = np.mean(np.abs(treatment_errors))
        improvement = ((control_mae - treatment_mae) / control_mae) * 100
        
        is_significant = p_value < self.alpha
        
        return {
            'control_mae': control_mae,
            'treatment_mae': treatment_mae,
            'improvement_pct': improvement,
            'p_value': p_value,
            'is_significant': is_significant,
            'decision': (
                f"✅ Roll out new system! {improvement:.2f}% improvement (p={p_value:.4f})"
                if is_significant and improvement > 0
                else f"❌ Keep old system (p={p_value:.4f})"
            )
        }
    
    def test_customer_satisfaction(self, control_satisfaction, treatment_satisfaction):
        """
        Test if new system improves customer satisfaction
        
        Args:
            control_satisfaction: [satisfied_count, unsatisfied_count] for control
            treatment_satisfaction: [satisfied_count, unsatisfied_count] for treatment
        
        Returns:
            dict: Chi-square test results
        """
        # Create contingency table
        contingency_table = np.array([
            control_satisfaction,
            treatment_satisfaction
        ])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        control_rate = control_satisfaction[0] / sum(control_satisfaction)
        treatment_rate = treatment_satisfaction[0] / sum(treatment_satisfaction)
        improvement = ((treatment_rate - control_rate) / control_rate) * 100
        
        is_significant = p_value < self.alpha
        
        return {
            'control_satisfaction_rate': control_rate,
            'treatment_satisfaction_rate': treatment_rate,
            'improvement_pct': improvement,
            'p_value': p_value,
            'is_significant': is_significant,
            'decision': (
                f"✅ New system improves satisfaction by {improvement:.2f}% (p={p_value:.4f})"
                if is_significant and improvement > 0
                else f"❌ No significant improvement in satisfaction (p={p_value:.4f})"
            )
        }

# Example usage
ab_tester = ABTestAnalyzer(alpha=0.05)

# Test ETA accuracy
control_errors = np.random.normal(5.5, 1.2, 1000)  # Old system
treatment_errors = np.random.normal(4.8, 1.1, 1000)  # New system

accuracy_results = ab_tester.test_eta_accuracy(control_errors, treatment_errors)
print("ETA Accuracy Test:", accuracy_results['decision'])

# Test customer satisfaction
control_satisfaction = [850, 150]  # [satisfied, unsatisfied]
treatment_satisfaction = [920, 80]  # [satisfied, unsatisfied]

satisfaction_results = ab_tester.test_customer_satisfaction(
    control_satisfaction, 
    treatment_satisfaction
)
print("Satisfaction Test:", satisfaction_results['decision'])
```

---

## 4. Complete A/B Testing Workflow

### Step-by-Step Process

```python
class ETASystemABTest:
    """
    Complete A/B testing workflow for ETA prediction system
    """
    
    def __init__(self, test_duration_days=7, alpha=0.05, min_sample_size=1000):
        self.test_duration_days = test_duration_days
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.results = {}
    
    def run_ab_test(self, control_data, treatment_data):
        """
        Run complete A/B test comparing old vs new system
        
        Args:
            control_data: DataFrame with columns ['predicted_eta', 'actual_eta', 'satisfied']
            treatment_data: DataFrame with columns ['predicted_eta', 'actual_eta', 'satisfied']
        
        Returns:
            dict: Comprehensive test results
        """
        # Calculate errors
        control_errors = control_data['actual_eta'] - control_data['predicted_eta']
        treatment_errors = treatment_data['actual_eta'] - treatment_data['predicted_eta']
        
        # 1. Test ETA Accuracy
        accuracy_test = self._test_accuracy(control_errors, treatment_errors)
        
        # 2. Test Customer Satisfaction
        satisfaction_test = self._test_satisfaction(
            control_data['satisfied'],
            treatment_data['satisfied']
        )
        
        # 3. Test Rider Efficiency (if available)
        if 'delivery_time' in control_data.columns:
            efficiency_test = self._test_efficiency(
                control_data['delivery_time'],
                treatment_data['delivery_time']
            )
        else:
            efficiency_test = None
        
        # Compile results
        self.results = {
            'accuracy': accuracy_test,
            'satisfaction': satisfaction_test,
            'efficiency': efficiency_test,
            'overall_recommendation': self._make_recommendation()
        }
        
        return self.results
    
    def _test_accuracy(self, control_errors, treatment_errors):
        """Test ETA accuracy improvement"""
        mae_control = np.mean(np.abs(control_errors))
        mae_treatment = np.mean(np.abs(treatment_errors))
        
        t_stat, p_value = stats.ttest_ind(
            np.abs(control_errors),
            np.abs(treatment_errors)
        )
        
        improvement = ((mae_control - mae_treatment) / mae_control) * 100
        
        return {
            'mae_control': mae_control,
            'mae_treatment': mae_treatment,
            'improvement_pct': improvement,
            'p_value': p_value,
            'is_significant': p_value < self.alpha and improvement > 0
        }
    
    def _test_satisfaction(self, control_satisfaction, treatment_satisfaction):
        """Test customer satisfaction improvement"""
        control_rate = control_satisfaction.mean()
        treatment_rate = treatment_satisfaction.mean()
        
        # Proportion test
        from statsmodels.stats.proportion import proportions_ztest
        
        counts = np.array([treatment_satisfaction.sum(), control_satisfaction.sum()])
        nobs = np.array([len(treatment_satisfaction), len(control_satisfaction)])
        
        z_stat, p_value = proportions_ztest(counts, nobs)
        
        improvement = ((treatment_rate - control_rate) / control_rate) * 100
        
        return {
            'satisfaction_control': control_rate,
            'satisfaction_treatment': treatment_rate,
            'improvement_pct': improvement,
            'p_value': p_value / 2,  # One-tailed
            'is_significant': (p_value / 2) < self.alpha and improvement > 0
        }
    
    def _test_efficiency(self, control_time, treatment_time):
        """Test delivery efficiency improvement"""
        mean_control = control_time.mean()
        mean_treatment = treatment_time.mean()
        
        t_stat, p_value = stats.ttest_ind(control_time, treatment_time)
        
        improvement = ((mean_control - mean_treatment) / mean_control) * 100
        
        return {
            'avg_time_control': mean_control,
            'avg_time_treatment': mean_treatment,
            'improvement_pct': improvement,
            'p_value': p_value,
            'is_significant': p_value < self.alpha and improvement > 0
        }
    
    def _make_recommendation(self):
        """Make final recommendation based on all tests"""
        significant_improvements = 0
        total_tests = 0
        
        for test_name, test_result in self.results.items():
            if test_result and isinstance(test_result, dict):
                total_tests += 1
                if test_result.get('is_significant', False):
                    significant_improvements += 1
        
        if significant_improvements >= 2:
            return "✅ RECOMMEND: Deploy new system to all users"
        elif significant_improvements == 1:
            return "⚠️ CAUTION: Some improvements seen, consider extended testing"
        else:
            return "❌ DO NOT DEPLOY: No significant improvements detected"

# Example usage
np.random.seed(42)

# Generate sample data
control_df = pd.DataFrame({
    'predicted_eta': np.random.normal(30, 5, 2000),
    'actual_eta': np.random.normal(35, 6, 2000),
    'satisfied': np.random.binomial(1, 0.75, 2000),
    'delivery_time': np.random.normal(35, 6, 2000)
})

treatment_df = pd.DataFrame({
    'predicted_eta': np.random.normal(30, 5, 2000),
    'actual_eta': np.random.normal(32, 5, 2000),  # Better accuracy
    'satisfied': np.random.binomial(1, 0.82, 2000),  # Higher satisfaction
    'delivery_time': np.random.normal(32, 5, 2000)  # Faster delivery
})

# Run A/B test
ab_test = ETASystemABTest(test_duration_days=7, alpha=0.05)
results = ab_test.run_ab_test(control_df, treatment_df)

print("\n" + "="*60)
print("A/B TEST RESULTS")
print("="*60)
print(f"\nAccuracy Improvement: {results['accuracy']['improvement_pct']:.2f}%")
print(f"P-value: {results['accuracy']['p_value']:.4f}")
print(f"Significant: {results['accuracy']['is_significant']}")

print(f"\nSatisfaction Improvement: {results['satisfaction']['improvement_pct']:.2f}%")
print(f"P-value: {results['satisfaction']['p_value']:.4f}")
print(f"Significant: {results['satisfaction']['is_significant']}")

print(f"\n{results['overall_recommendation']}")
print("="*60)
```

---

## 5. Key Metrics & Thresholds

### Significance Level (α)
- **Standard**: α = 0.05 (5% significance level)
- **Conservative**: α = 0.01 (1% significance level for critical changes)

### Minimum Detectable Effect (MDE)
- **ETA Accuracy**: 5% improvement in MAE
- **Customer Satisfaction**: 3% improvement in satisfaction rate
- **Delivery Time**: 2 minutes reduction

### Sample Size Requirements
```python
from statsmodels.stats.power import TTestIndPower

def calculate_sample_size(effect_size=0.2, alpha=0.05, power=0.8):
    """
    Calculate required sample size for A/B test
    
    Args:
        effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
        alpha: Significance level
        power: Statistical power (1 - β)
    
    Returns:
        int: Required sample size per group
    """
    analysis = TTestIndPower()
    sample_size = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='larger'
    )
    
    return int(np.ceil(sample_size))

# Example
required_n = calculate_sample_size(effect_size=0.2, alpha=0.05, power=0.8)
print(f"Required sample size per group: {required_n}")
```

---

## 6. Best Practices

### ✅ Do's
1. **Define hypotheses before testing** - Don't p-hack!
2. **Use appropriate sample sizes** - Ensure statistical power
3. **Run tests for sufficient duration** - Account for day-of-week effects
4. **Monitor multiple metrics** - Accuracy, satisfaction, efficiency
5. **Document all tests** - Keep audit trail of decisions

### ❌ Don'ts
1. **Don't stop tests early** - Even if results look good
2. **Don't test multiple changes simultaneously** - Isolate variables
3. **Don't ignore practical significance** - Statistical ≠ Practical
4. **Don't forget about seasonality** - Account for external factors
5. **Don't deploy without validation** - Always validate on holdout set

---

## 7. Monitoring Dashboard

```python
import matplotlib.pyplot as plt
import seaborn as sns

def create_ab_test_dashboard(control_errors, treatment_errors):
    """
    Create visualization dashboard for A/B test results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Error Distribution
    axes[0, 0].hist(control_errors, bins=50, alpha=0.5, label='Control', color='blue')
    axes[0, 0].hist(treatment_errors, bins=50, alpha=0.5, label='Treatment', color='green')
    axes[0, 0].set_title('Error Distribution Comparison')
    axes[0, 0].set_xlabel('Prediction Error (minutes)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 2. Box Plot
    axes[0, 1].boxplot([np.abs(control_errors), np.abs(treatment_errors)],
                        labels=['Control', 'Treatment'])
    axes[0, 1].set_title('Absolute Error Comparison')
    axes[0, 1].set_ylabel('Absolute Error (minutes)')
    
    # 3. Cumulative Distribution
    axes[1, 0].hist(np.abs(control_errors), bins=50, cumulative=True, 
                    density=True, alpha=0.5, label='Control', color='blue')
    axes[1, 0].hist(np.abs(treatment_errors), bins=50, cumulative=True,
                    density=True, alpha=0.5, label='Treatment', color='green')
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel('Absolute Error (minutes)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].legend()
    
    # 4. Summary Statistics
    summary_text = f"""
    Control Group:
    - MAE: {np.mean(np.abs(control_errors)):.2f} min
    - Median: {np.median(np.abs(control_errors)):.2f} min
    - Std: {np.std(control_errors):.2f} min
    
    Treatment Group:
    - MAE: {np.mean(np.abs(treatment_errors)):.2f} min
    - Median: {np.median(np.abs(treatment_errors)):.2f} min
    - Std: {np.std(treatment_errors):.2f} min
    
    Improvement: {((np.mean(np.abs(control_errors)) - np.mean(np.abs(treatment_errors))) / np.mean(np.abs(control_errors)) * 100):.2f}%
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
```

---

## Summary

**Hypothesis testing is NEEDED (as noted in your class notes!)** for:

1. ✅ **Model Comparison** - Determine which model performs better
2. ✅ **Feature Impact** - Assess if new features improve predictions
3. ✅ **System Changes** - Validate improvements before full rollout
4. ✅ **Continuous Monitoring** - Detect model drift and performance degradation

**Key Takeaway**: Never deploy changes without statistical validation!

---

**Document Version**: 1.0  
**Last Updated**: November 26, 2024  
**Reference**: Class notes on ML System Design - Hypothesis Testing section
