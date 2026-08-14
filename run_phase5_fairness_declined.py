import sys
sys.path.insert(0, "src")
import pandas as pd
import fairness

print("Loading Phase 3's declined-pool scores...")
declined = pd.read_parquet("data/processed/declined_scored.parquet")
print("Declined pool:", declined.shape)

THRESHOLD = 0.320  # Phase 3's corrected, profit-maximizing threshold
approve_flag = declined["corrected_predicted_pd"] < THRESHOLD
print(f"\nDecisions at threshold {THRESHOLD}: Approved={approve_flag.sum():,}  Declined={(~approve_flag).sum():,}")
print(f"Approval rate: {approve_flag.mean():.1%}  (much more balanced than Phase 5's 99.99% on the accepted-test population)")

print("\n--- Disparate impact check: approval rate by state (declined pool) ---")
rates = fairness.approval_rate_by_group(approve_flag, declined["addr_state"])
result = fairness.four_fifths_test(rates)
print(f"Most favored state: {result['most_favored_group']} ({rates[result['most_favored_group']]:.1%} approval)")
print(f"Least favored state: {result['least_favored_group']} ({rates[result['least_favored_group']]:.1%} approval)")
print(f"Impact ratio: {result['impact_ratio']:.3f}")
print(f"Passes four-fifths rule: {result['passes_four_fifths']}")

print("\n--- Full state-level breakdown, sorted ---")
print(rates.sort_values().to_string())

rates.sort_values().to_csv("reports/figures/phase5_declined_pool_approval_rate_by_state.csv")
print("\nSaved to reports/figures/phase5_declined_pool_approval_rate_by_state.csv")
