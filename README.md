# Python Project: Weight loss drug treatments : simulated clinical trial 

This study evaluated the efficacy and durability of three drug treatments (Drug A, Drug B, and Both A&B) compared to placebo in promoting weight loss over a 60-week period. The first 30 weeks involved active drug administration, followed by a 30-week withdrawal phase to assess weight regain. The analysis was designed to meet regulatory standards, including those outlined by the FDA.

The study employed rigorous statistical methods including mixed modeling, bootstrap validation, and power analysis to evaluate drug efficacy and durability. The sample size was sufficient for reliable inference, and the dashboard provides a practical tool for clinical decision-making.

<b> Methodology Overview </b> 

<b> 1. Data Simulation </b> 
   
- A synthetic dataset was generated for 3,000 individuals across four treatment groups.

- Weekly weight measurements were recorded over 60 weeks.

- Covariates included sex, age, height, occupation, exercise hours, calorie intake, and 10 additional health-related variables.

<b> 2. Mixed Effects Modeling </b> 

- A linear mixed effects model was used to estimate weight trajectories, accounting for repeated measures and individual variability.

- Fixed effects included Week, Group, Sex, and their interactions.

- Random effects were modeled at the individual level.

<b> 3. ANOVA and Post-Hoc Testing </b> 

- Weekly ANOVA tests compared each drug group to placebo.

- Tukey HSD post-hoc tests identified statistically significant differences (p < 0.05).

- Significance was annotated on weight trend plots.

<b> 4. Responder and Probability Analysis </b> 

- Clinically meaningful weight loss was defined as ≥5% reduction from baseline.

- The proportion of responders was calculated weekly for each group.

- Minimum duration to reach ≥80% probability of response was estimated per drug.

<b> 5. Withdrawal Phase Simulation </b>
   
- After Week 30, drugs were withdrawn.

- Weight regain was modeled with group-specific rebound rates:

- Drug A: rapid regain

- Drug B: slow regain

- Both A&B: moderate regain

- Placebo: stable

<b> 6. Bootstrap Analysis </b> 

- 100 bootstrap samples were drawn to assess parameter stability.

- Distributions of key model coefficients were visualized.

- Confidence intervals and standard deviations confirmed robustness of estimates.

<b> 7. Power Analysis </b> 

- A power analysis was conducted using F-test for ANOVA.

- With an assumed medium effect size (f = 0.25), the required sample size per group was ~200.

- The actual sample size (750 per group) exceeded this threshold, confirming adequacy.

<b> 8. Dashboard Development </b> 

A Streamlit dashboard was built to visualize weight trajectories and provide treatment recommendations.

- Physicians can input patient profiles to receive personalized predictions and guidance on drug choice and duration.

