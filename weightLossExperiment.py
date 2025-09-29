# Weight loss drug experiment simulated data
# synthetic data for 3,000 individuals randomly assigned to four groups: Drug A, Drug B,
# Both A & B, and Placebo. Each individual will have:
# Weekly weight measurements over 30 weeks
# 10 health-related covariates (e.g., blood pressure, cholesterol, glucose level, etc.)
# additional covariates
# Sex: Male or Female, Height: Normally distributed by sex, Age: Randomized within a realistic adult range
# Occupation: Randomly assigned from a list, Hours of Exercise per Week: Based on occupation and age
# Calorie Intake per Day: Influenced by sex, exercise, and occupation

import numpy as np
import pandas as pd

# Set seed
np.random.seed(42)

# Constants
n_individuals = 3000
n_weeks = 30
n_covariates = 10
groups = ['Drug_A', 'Drug_B', 'Both_AB', 'Placebo']
group_size = n_individuals // len(groups)
group_labels = np.repeat(groups, group_size)
ids = np.arange(1, n_individuals + 1)

# Covariates
covariate_names = [f'covariate_{i+1}' for i in range(n_covariates)]
covariates = np.random.normal(0, 1, size=(n_individuals, n_covariates))

# Sex
sex = np.random.choice(['Male', 'Female'], size=n_individuals)

# Height (cm)
height = np.where(sex == 'Male',
                  np.random.normal(175, 7, size=n_individuals),
                  np.random.normal(162, 6, size=n_individuals))

# Age
age = np.random.randint(18, 65, size=n_individuals)

# Occupation
occupations = ['Office Worker', 'Manual Laborer', 'Student', 'Healthcare', 'Unemployed']
occupation = np.random.choice(occupations, size=n_individuals)

# Exercise hours per week
exercise_hours = np.random.normal(3, 1.5, size=n_individuals)
exercise_hours += np.where(occupation == 'Manual Laborer', 2, 0)
exercise_hours = np.clip(exercise_hours, 0, None)

# Calorie intake
base_calories = np.where(sex == 'Male', 2500, 2000)
calorie_intake = base_calories + exercise_hours * 100 + np.random.normal(0, 200, size=n_individuals)

# Weight simulation
def simulate_weight(group, weeks, base_weight):
    trend = {
        'Drug_A': -0.2,
        'Drug_B': -0.1,
        'Both_AB': -0.3,
        'Placebo': 0.0
    }[group]
    noise = np.random.normal(0, 1, weeks)
    return base_weight + trend * np.arange(weeks) + noise

# Build dataset
records = []
for i in range(n_individuals):
    base_weight = np.random.normal(70, 10)
    weight_series = simulate_weight(group_labels[i], n_weeks, base_weight)
    for week in range(n_weeks):
        record = {
            'ID': ids[i],
            'Group': group_labels[i],
            'Week': week + 1,
            'Weight': weight_series[week],
            'Sex': sex[i],
            'Height_cm': height[i],
            'Age': age[i],
            'Occupation': occupation[i],
            'Exercise_hours': exercise_hours[i],
            'Calorie_intake': calorie_intake[i]
        }
        for j, cov_name in enumerate(covariate_names):
            record[cov_name] = covariates[i, j]
        records.append(record)

# Create DataFrame
df = pd.DataFrame(records)

# Preview
print(df.head())

# visualization of weekly weight trends across all treatment groups — Drug A, Drug B, Both A&B, and Placebo —
# with 95% confidence intervals shaded around the mean lines

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# Apply aesthetic style
style.use('seaborn-v0_8')

# Group by 'Group' and 'Week' and calculate mean and 95% CI
summary = df.groupby(['Group', 'Week'])['Weight'].agg(['mean', 'count', 'std']).reset_index()
summary['ci95'] = 1.96 * summary['std'] / np.sqrt(summary['count'])

# Plotting
plt.figure(figsize=(12, 8))
for group in summary['Group'].unique():
    data = summary[summary['Group'] == group]
    plt.plot(data['Week'], data['mean'], label=group)
    plt.fill_between(data['Week'], data['mean'] - data['ci95'], data['mean'] + data['ci95'], alpha=0.3)

plt.title('Weekly Weight Trends by Treatment Group with 95% Confidence Interval')
plt.xlabel('Week')
plt.ylabel('Weight (kg)')
plt.legend(title='Treatment Group')
plt.grid(True)
plt.tight_layout()

# Save plot
output_path = '/mnt/data/weekly_weight_trends.png'
plt.savefig(output_path)
plt.close()

# updated visualization showing weekly weight trends across treatment groups —
# now enhanced with ANOVA-based significance testing against the placebo group
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Assume df is already defined from previous simulation
# Columns: ID, Group, Week, Weight, Sex, Height_cm, Age, Occupation, Exercise_hours, Calorie_intake, covariate_1...covariate_10

# Track significant weeks
significant_weeks = []

# Perform ANOVA weekly
for week in range(1, 31):
    week_data = df[df['Week'] == week]
    groups_data = [week_data[week_data['Group'] == g]['Weight'] for g in ['Drug_A', 'Drug_B', 'Both_AB', 'Placebo']]
    
    # One-way ANOVA
    f_stat, p_val = f_oneway(*groups_data)
    
    # Post-hoc Tukey HSD
    tukey = pairwise_tukeyhsd(endog=week_data['Weight'], groups=week_data['Group'], alpha=0.05)
    for row in tukey.summary().data[1:]:
        if 'Placebo' in row[0] or 'Placebo' in row[1]:
            if row[-1]:  # reject null hypothesis
                significant_weeks.append(week)
                break

# Plot weight trends with confidence intervals
plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")
sns.lineplot(data=df, x='Week', y='Weight', hue='Group', ci=95)

# Annotate significant weeks
for week in significant_weeks:
    y_pos = df[df['Week'] == week]['Weight'].mean() + 1.5
    plt.text(week, y_pos, '★', color='red', ha='center', fontsize=12)

plt.title('Weekly Weight Trends by Treatment Group\n★ = Significant Difference from Placebo (p < 0.05)')
plt.xlabel('Week')
plt.ylabel('Weight (kg)')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

# Covariates Affecting Drug Efficacy
# dentify which variables (e.g., age, sex, height, exercise, calorie intake, covariates 1–10)
# influence how effective the drugs are in reducing weight.
# Method: Linear Mixed Effects Model
# This accounts for repeated measures (weekly weights) and tests interactions between treatment and covariates.

import statsmodels.formula.api as smf

# Encode categorical variables
df['Sex'] = df['Sex'].astype('category')
df['Occupation'] = df['Occupation'].astype('category')

# Fit mixed effects model
model = smf.mixedlm(
    "Weight ~ Week + Group * (Sex + Age + Height_cm + Exercise_hours + Calorie_intake + covariate_1 + covariate_2 + covariate_3)",
    data=df,
    groups=df["ID"]
)
result = model.fit()
print(result.summary())

# Significant interaction terms like Group:Age or Group:Exercise_hours suggest that those covariates modify drug efficacy.
# Main effects of covariates show their independent influence on weight
# Drug Effects on Vitals (Covariates 1–10)
# Goal: Determine if any drug alters health-related covariates over time.
# Assuming covariates are tracked weekly (or at baseline and endpoint), we can test for changes by group.
# Example: test effect on covariate_1 at Week 1 vs Week 30
baseline = df[df['Week'] == 1]
endpoint = df[df['Week'] == 30]

# Merge for delta analysis
merged = baseline[['ID', 'Group', 'covariate_1']].merge(
    endpoint[['ID', 'covariate_1']], on='ID', suffixes=('_start', '_end')
)
merged['delta_cov1'] = merged['covariate_1_end'] - merged['covariate_1_start']

# ANOVA on change
import scipy.stats as stats
groups = [merged[merged['Group'] == g]['delta_cov1'] for g in ['Drug_A', 'Drug_B', 'Both_AB', 'Placebo']]
f_stat, p_val = stats.f_oneway(*groups)
print(f"ANOVA result for covariate_1 change: F={f_stat:.2f}, p={p_val:.4f}")

# Assuming drug response differs by sex, we’ll fit a Linear Mixed Effects Model that includes:
# Fixed effects: Week, Drug Group, Sex, and their interaction
# Random effects: Individual ID (to account for repeated measures)
import statsmodels.formula.api as smf

# Ensure categorical encoding
df['Sex'] = df['Sex'].astype('category')
df['Group'] = df['Group'].astype('category')

# Fit model with interaction
model = smf.mixedlm(
    "Weight ~ Week * Group * Sex",
    data=df,
    groups=df["ID"]
)
result = model.fit()
print(result.summary())

# Predict and Plot Weight Trends
import matplotlib.pyplot as plt
import seaborn as sns

# Create prediction grid
pred_df = df[['Week', 'Group', 'Sex']].drop_duplicates()
pred_df['Predicted_Weight'] = result.predict(pred_df)

# Plot
plt.figure(figsize=(14, 8))
sns.lineplot(data=pred_df, x='Week', y='Predicted_Weight', hue='Group', style='Sex', markers=False, ci=None)

plt.title('Predicted Weight Loss Trends by Drug and Sex')
plt.xlabel('Week')
plt.ylabel('Predicted Weight (kg)')
plt.legend(title='Group × Sex')
plt.tight_layout()
plt.show()

# Residuals Visualization Script
import matplotlib.pyplot as plt
import seaborn as sns

# Add predicted values and residuals to the original dataframe
df['Predicted_Weight'] = result.predict(df)
df['Residual'] = df['Weight'] - df['Predicted_Weight']

# Plot residuals over time
plt.figure(figsize=(14, 6))
sns.lineplot(data=df, x='Week', y='Residual', hue='Group', style='Sex', ci='sd')

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residuals Over Time by Group and Sex')
plt.xlabel('Week')
plt.ylabel('Residual (Observed - Predicted)')
plt.legend(title='Group × Sex')
plt.tight_layout()
plt.show()

# Flat residuals around zero → good fit
# Patterns or trends → model may be missing structure
# Widening bands → increasing variance over time
# Group/Sex differences → model may need more interaction terms or nonlinear effects

# Quantify Percent Weight Loss by Treatment Group
# To support the company’s claim about drug-induced weight loss and provide physician guidelines
# Calculate percent weight loss
baseline = df[df['Week'] == 1][['ID', 'Weight']].rename(columns={'Weight': 'Weight_start'})
endpoint = df[df['Week'] == 30][['ID', 'Weight', 'Group']].rename(columns={'Weight': 'Weight_end'})

merged = pd.merge(baseline, endpoint, on='ID')
merged['Percent_Loss'] = 100 * (merged['Weight_start'] - merged['Weight_end']) / merged['Weight_start']

# Group averages
group_stats = merged.groupby('Group')['Percent_Loss'].agg(['mean', 'std', 'count'])
print(group_stats)

# Estimate Probability of Clinically Meaningful Weight Loss
# Let’s define ≥5% weight loss as clinically meaningful and compute the proportion of individuals achieving it
merged['Clinically_Meaningful'] = merged['Percent_Loss'] >= 5
probability = merged.groupby('Group')['Clinically_Meaningful'].mean()
print(probability)

# Vital Sign Impact Analysis
# To assess whether drugs affect health covariates (vitals), we compare changes from Week 1 to Week 30
# Example for covariate_1
cov_start = df[df['Week'] == 1][['ID', 'covariate_1']]
cov_end = df[df['Week'] == 30][['ID', 'covariate_1']].rename(columns={'covariate_1': 'covariate_1_end'})
cov_merged = pd.merge(cov_start, cov_end, on='ID')
cov_merged = pd.merge(cov_merged, endpoint[['ID', 'Group']], on='ID')
cov_merged['delta_cov1'] = cov_merged['covariate_1_end'] - cov_merged['covariate_1']

# ANOVA
from scipy.stats import f_oneway
groups = [cov_merged[cov_merged['Group'] == g]['delta_cov1'] for g in ['Drug_A', 'Drug_B', 'Both_AB', 'Placebo']]
f_stat, p_val = f_oneway(*groups)
print(f"Covariate_1 change ANOVA: F={f_stat:.2f}, p={p_val:.4f}")

# Drug admistration duration
# To recommend how many weeks each drug should be administered to achieve clinically meaningful weight loss
# (defined as ≥5% reduction in body weight), we’ll analyze the simulated data to estimate:
# The average time to reach 5% weight loss
# The probability of achieving that threshold at each week
# A recommended minimum duration for physicians

# Define Meaningful Weight Loss
# For each individual, calculate baseline weight
baseline_weight = df[df['Week'] == 1][['ID', 'Weight']].rename(columns={'Weight': 'Weight_start'})

# Merge with weekly data
df_merged = pd.merge(df, baseline_weight, on='ID')
df_merged['Percent_Loss'] = 100 * (df_merged['Weight_start'] - df_merged['Weight']) / df_merged['Weight_start']
df_merged['Meaningful'] = df_merged['Percent_Loss'] >= 5

# Estimate Probability of Achieving ≥5% Loss by Week
# Group by Week and Drug Group
prob_by_week = df_merged.groupby(['Week', 'Group'])['Meaningful'].mean().reset_index()

# Visualize and Recommend Duration
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
sns.lineplot(data=prob_by_week, x='Week', y='Meaningful', hue='Group')

# Add threshold line
plt.axhline(0.8, color='gray', linestyle='--', label='80% Threshold')
plt.title('Probability of Achieving ≥5% Weight Loss Over Time')
plt.ylabel('Probability')
plt.xlabel('Week')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

# Simulated data for Phase 2: Weeks 31–60
# All drugs withdrawn
#Weight regain behavior: Drug A: faster regain, Drug B: slower regain, Both AB: moderate regain, Placebo: stable

# Simulate Weight Regain
# Simulate regain trends
def simulate_regain(group, start_weight, weeks=30):
    regain_rate = {
        'Drug_A': 0.15,
        'Drug_B': 0.05,
        'Both_AB': 0.10,
        'Placebo': 0.00
    }[group]
    noise = np.random.normal(0, 1, weeks)
    return start_weight + regain_rate * np.arange(1, weeks + 1) + noise

# Get last weight at Week 30
week30 = df[df['Week'] == 30][['ID', 'Group', 'Weight']].rename(columns={'Weight': 'Weight_30'})

# Simulate Weeks 31–60
regain_records = []
for _, row in week30.iterrows():
    regain_series = simulate_regain(row['Group'], row['Weight_30'], weeks=30)
    for week in range(31, 61):
        regain_records.append({
            'ID': row['ID'],
            'Group': row['Group'],
            'Week': week,
            'Weight': regain_series[week - 31]
        })

df_regain = pd.DataFrame(regain_records)

# Combine with original
df_extended = pd.concat([df, df_regain], ignore_index=True)

# Plot Weight Trends with Confidence Intervals
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
sns.set(style="whitegrid")

# Plot with CI
sns.lineplot(data=df_extended, x='Week', y='Weight', hue='Group', ci=95)

# Mark drug withdrawal
plt.axvline(30, color='black', linestyle='--', label='Drug Withdrawal')
plt.text(30.5, df_extended['Weight'].max() - 2, 'Withdrawal Point', rotation=90, color='black')

plt.title('Weight Trends Over 60 Weeks\nDrug Withdrawal at Week 30')
plt.xlabel('Week')
plt.ylabel('Weight (kg)')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

# bootstrap analysis to evaluate the stability of parameter estimates and confidence intervals
# Mixed Effects Model Recap
from statsmodels.formula.api import mixedlm

model = mixedlm("Weight ~ Week * Group * Sex", data=df_extended, groups=df_extended["ID"])
result = model.fit()

# This estimates fixed effects (Week, Group, Sex, interactions) and random effects (individual variation).
# Bootstrap Procedure
# We’ll resample individuals (not rows) with replacement, refit the model, and track parameter variability.
# Tests robustness of estimates
# Reveals confidence interval width
# Helps assess sample adequacy
import numpy as np
from tqdm import tqdm

n_boot = 100  # Increase to 1000+ for production
params = []

unique_ids = df_extended['ID'].unique()

for _ in tqdm(range(n_boot)):
    sampled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=True)
    boot_df = df_extended[df_extended['ID'].isin(sampled_ids)].copy()
    
    try:
        boot_model = mixedlm("Weight ~ Week * Group * Sex", data=boot_df, groups=boot_df["ID"])
        boot_result = boot_model.fit()
        params.append(boot_result.params)
    except:
        continue  # skip failed fits

# Analyze Bootstrap Results
import pandas as pd

param_df = pd.DataFrame(params)
summary = param_df.describe(percentiles=[0.025, 0.5, 0.975]).T
summary.columns = ['Mean', 'Std', '2.5%', '50%', '97.5%']
print(summary)

# Mean and Std of each parameter
# Bootstrap confidence intervals
# Stability check: narrow intervals = reliable estimates

# Sample Size Adequacy Criteria

# Metric	                Interpretation
# Narrow CI (e.g. ±5%)	        Sample size likely sufficient
# Stable parameter signs	Direction of effect is robust
# Low Std across bootstraps	Low sampling variability
# Failed fits < 5%	        Model is stable across resamples

# visualize the bootstrap distributions
# Convert Bootstrap Results to DataFrame
import pandas as pd

param_df = pd.DataFrame(params)  # Each row = one bootstrap sample

# Plot Distributions for Key Parameters
# Let’s visualize the distributions for a few critical fixed effects:
import matplotlib.pyplot as plt
import seaborn as sns

# Choose parameters to plot
key_params = ['Week', 'Group[T.Drug_A]', 'Group[T.Drug_B]', 'Group[T.Both_AB]', 'Week:Group[T.Drug_A]']

plt.figure(figsize=(14, 8))
for i, param in enumerate(key_params, 1):
    plt.subplot(2, 3, i)
    sns.histplot(param_df[param], kde=True, bins=30, color='skyblue')
    plt.axvline(param_df[param].mean(), color='red', linestyle='--', label='Mean')
    plt.title(f'Distribution of {param}')
    plt.xlabel('Estimate')
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.suptitle('Bootstrap Distributions of Mixed Model Parameters', fontsize=16, y=1.02)
plt.show()

# Narrow, symmetric distributions → stable estimates, good sample size
# Wide or skewed distributions → potential instability or need for more data
# Mean ≈ median → low bias
# Red line (mean) helps visualize central tendency

# Power Analysis for Mixed Effects Model
# To confirm whether your sample size (~3,000 individuals) is sufficient for detecting meaningful weight loss effects
#in a mixed effects model, we’ll run a power analysis tailored to longitudinal data.

# Estimate the minimum sample size required to detect a clinically meaningful weight loss effect (≥5%) with:
# α = 0.05 (significance level)
# Power = 0.80 (probability of detecting true effect)
# Effect size based on observed data
# Repeated measures over 60 weeks

# Estimate Effect Size
# Use your model or bootstrap results to extract the standardized effect size for the treatment group (e.g., Drug A vs Placebo):

# Example: Cohen's f² for fixed effect
f_squared = (R²_full - R²_reduced) / (1 - R²_full)

# If you don’t have R² values, approximate using:
# Cohen's d from bootstrap
mean_diff = param_df['Group[T.Drug_A]'].mean()
std_dev = param_df['Group[T.Drug_A]'].std()
cohen_d = mean_diff / std_dev

# Typical benchmarks:
# Small effect: d = 0.2
# Medium effect: d = 0.5
# Large effect: d = 0.8

# Run Power Analysis
# Use statsmodels or pingouin for repeated measures power calculation:
from statsmodels.stats.power import FTestAnovaPower

# Assume medium effect size (f = 0.25), 4 groups, 60 measurements
analysis = FTestAnovaPower()
required_n = analysis.solve_power(effect_size=0.25, alpha=0.05, power=0.8, k_groups=4)
print(f"Required sample size per group: {int(required_n)}")
print(f"Total required sample size: {int(required_n * 4)}")

# Visualize Power Curve
import matplotlib.pyplot as plt
effect_sizes = np.linspace(0.1, 0.5, 50)
sample_sizes = [analysis.solve_power(effect_size=es, alpha=0.05, power=0.8, k_groups=4) for es in effect_sizes]

plt.figure(figsize=(10, 6))
plt.plot(effect_sizes, sample_sizes, color='blue')
plt.axhline(3000, color='red', linestyle='--', label='Current Sample Size')
plt.xlabel("Effect Size (f)")
plt.ylabel("Required Sample Size")
plt.title("Power Analysis: Sample Size vs Effect Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# If your estimated effect size is ≥0.25, your current sample size of 3,000 is more than sufficient.
# The power curve confirms that for moderate-to-large effects, you’re well above the minimum threshold.
# For smaller effects (f < 0.15), you may need more participants or stronger modeling assumptions.

# FDA requirements 
# Intent-to-Treat (ITT) and Per-Protocol (PP) Analyses
# ITT: Include all randomized participants, regardless of adherence or dropout.
# PP: Include only those who completed the study as planned.
# ITT: Use full dataset
itt_df = df_extended.copy()

# PP: Filter for participants with complete data across all weeks
pp_ids = df_extended.groupby('ID')['Week'].nunique()
pp_df = df_extended[df_extended['ID'].isin(pp_ids[pp_ids == 60].index)]
# Then rerun your efficacy models (e.g., mixed effects) on both datasets and compare results.

# Handling of Missing Data
# Use multiple imputation or maximum likelihood estimation to handle missing weight or covariate values.
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = df_extended.copy()
df_imputed[['Weight']] = imputer.fit_transform(df_imputed[['Weight']])


# Multiplicity Adjustment
# Adjust for multiple comparisons (e.g., testing multiple drugs, time points, covariates) to control false positives.
from statsmodels.stats.multitest import multipletests

# Example: p-values from weekly ANOVA
p_values = [f_oneway(*[df_extended[(df_extended['Week'] == w) & (df_extended['Group'] == g)]['Weight']
                       for g in ['Drug_A', 'Drug_B', 'Both_AB', 'Placebo']])[1] for w in range(1, 61)]

# Adjust
adjusted = multipletests(p_values, method='bonferroni')
significant_weeks = [w for w, sig in zip(range(1, 61), adjusted[0]) if sig]

# Adverse Event Simulation and Analysis
#Simulate and analyze adverse events (AEs) by group, severity, and timing.
np.random.seed(42)
ae_df = pd.DataFrame({
    'ID': df_extended['ID'].unique(),
    'Group': df_extended.groupby('ID')['Group'].first().values,
    'AE_occurred': np.random.binomial(1, p=[0.05 if g == 'Placebo' else 0.15 for g in df_extended.groupby('ID')['Group'].first().values])
})

# AE rates by group
ae_rates = ae_df.groupby('Group')['AE_occurred'].mean()
print(ae_rates)

# Sensitivity and Subgroup Analyses
# Test robustness of results across subgroups (e.g., sex, age, BMI).
# Use interaction terms in models.
model = smf.mixedlm("Weight ~ Week * Group * Sex + Age + Height_cm", data=df_extended, groups=df_extended["ID"])
result = model.fit()
print(result.summary())

# Long-Term Safety and Rebound Risk Modeling
# Model weight regain post-withdrawal (Weeks 31–60).
# Identify predictors of rebound (e.g., drug type, sex, calorie intake)
df_extended['Phase'] = np.where(df_extended['Week'] <= 30, 'Treatment', 'Withdrawal')
regain_model = smf.mixedlm("Weight ~ Week * Group * Phase", data=df_extended, groups=df_extended["ID"])
regain_result = regain_model.fit()
print(regain_result.summary())

# prototype for a Streamlit dashboard called “Weight Loss Treatment Advisor”. This interactive app lets physicians input patient profiles
# and visualize predicted weight loss patterns, treatment duration, and drug recommendations.

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# Set style
sns.set(style="whitegrid")

# -----------------------------
# Simulate training data
# -----------------------------
np.random.seed(42)
n = 3000
weeks = np.tile(np.arange(1, 61), n)
groups = np.repeat(np.random.choice(['Drug_A', 'Drug_B', 'Both_AB', 'Placebo'], n), 60)
sexes = np.repeat(np.random.choice(['Male', 'Female'], n), 60)
ages = np.repeat(np.random.randint(18, 65, n), 60)
heights = np.repeat(np.random.normal(170, 10, n), 60)
exercise = np.repeat(np.random.uniform(0, 10, n), 60)
calories = np.repeat(np.random.randint(1500, 3500, n), 60)
baseline_weights = np.repeat(np.random.normal(80, 15, n), 60)

# Simulate weight based on group and week
def simulate_weight(group, week, base):
    if week <= 30:
        loss = {'Drug_A': -0.2, 'Drug_B': -0.1, 'Both_AB': -0.3, 'Placebo': 0.0}[group] * week
    else:
        regain = {'Drug_A': 0.15, 'Drug_B': 0.05, 'Both_AB': 0.10, 'Placebo': 0.0}[group] * (week - 30)
        loss = {'Drug_A': -0.2, 'Drug_B': -0.1, 'Both_AB': -0.3, 'Placebo': 0.0}[group] * 30 + regain
    return base + loss + np.random.normal(0, 1)

weights = [simulate_weight(g, w, b) for g, w, b in zip(groups, weeks, baseline_weights)]

df = pd.DataFrame({
    'Week': weeks,
    'Group': groups,
    'Sex': sexes,
    'Age': ages,
    'Height_cm': heights,
    'Exercise_hours': exercise,
    'Calorie_intake': calories,
    'Weight': weights
})

# -----------------------------
# Train model
# -----------------------------
features = ['Week', 'Group', 'Sex', 'Age', 'Height_cm', 'Exercise_hours', 'Calorie_intake']
target = 'Weight'

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), ['Group', 'Sex'])
], remainder='passthrough')

model = make_pipeline(preprocessor, GradientBoostingRegressor())
model.fit(df[features], df[target])

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💊 Weight Loss Treatment Advisor")
st.markdown("Predict weight loss trajectory and get drug recommendations based on patient profile.")

# Sidebar inputs
st.sidebar.header("Patient Profile")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 65, 35)
height = st.sidebar.slider("Height (cm)", 140, 200, 170)
exercise = st.sidebar.slider("Exercise Hours/Week", 0, 10, 3)
calories = st.sidebar.slider("Daily Calorie Intake", 1200, 4000, 2200)
weight_start = st.sidebar.slider("Baseline Weight (kg)", 50, 150, 80)

# -----------------------------
# Predict weight trajectory
# -----------------------------
weeks = np.arange(1, 61)
groups = ['Drug_A', 'Drug_B', 'Both_AB', 'Placebo']
predictions = []

for group in groups:
    input_df = pd.DataFrame({
        'Week': weeks,
        'Group': group,
        'Sex': sex,
        'Age': age,
        'Height_cm': height,
        'Exercise_hours': exercise,
        'Calorie_intake': calories
    })
    input_df['Weight'] = model.predict(input_df)
    input_df['Group'] = group
    predictions.append(input_df)

pred_df = pd.concat(predictions)

# -----------------------------
# Plot weight trajectory
# -----------------------------
st.subheader("📈 Predicted Weight Trajectory")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=pred_df, x='Week', y='Weight', hue='Group', ax=ax)
ax.axvline(30, color='black', linestyle='--')
ax.text(30.5, weight_start + 5, "Drug Withdrawal", rotation=90, color='black')
ax.set_ylabel("Weight (kg)")
ax.set_xlabel("Week")
st.pyplot(fig)

# -----------------------------
# Recommendation logic
# -----------------------------
recommendations = {}
for group in groups:
    group_df = pred_df[pred_df['Group'] == group].copy()
    baseline = group_df.iloc[0]['Weight']
    group_df['Percent_Loss'] = 100 * (baseline - group_df['Weight']) / baseline
    meaningful = group_df[group_df['Percent_Loss'] >= 5]
    if not meaningful.empty:
        week = meaningful.iloc[0]['Week']
        recommendations[group] = week

if recommendations:
    best_drug = min(recommendations, key=recommendations.get)
    best_week = recommendations[best_drug]
    st.subheader("🧠 Recommended Treatment")
    st.markdown(f"""
    - **Recommended Drug:** `{best_drug}`
    - **Minimum Duration:** `{int(best_week)} weeks`
    - **Expected Weight Loss:** ≥5% by Week {int(best_week)}
    - **Post-Withdrawal Monitoring:** Required for Drug A and Both_AB
    """)
else:
    st.warning("No drug achieves ≥5% weight loss with current profile.")


# Save the code as app.py
# In your terminal, run:
# streamlit run app.py



