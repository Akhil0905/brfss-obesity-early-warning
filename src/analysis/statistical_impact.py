"""
src/analysis/statistical_impact.py
----------------------------------
Performs statistical tests (ANOVA, T-tests, Correlation) to quantify the impact
of socioeconomic and demographic factors on obesity prevalence.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import json
from pathlib import Path
from src.utils.helpers import get_logger, load_config
from src.utils.paths import INTERIM_DATA_DIR, REPORTS_DIR

logger = get_logger(__name__)

def run_statistical_analysis():
    config = load_config()
    interim_path = INTERIM_DATA_DIR / config["data"]["interim_filename"]
    
    if not interim_path.exists():
        logger.error(f"Interim data not found at {interim_path}")
        return

    df = pd.read_csv(interim_path)
    logger.info(f"Loaded {len(df)} rows for statistical analysis.")

    results = {}

    # 1. ANOVA for Stratum Categories
    logger.info("Running ANOVA for stratum_category...")
    model = ols('Data_Value ~ C(stratum_category)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    results["anova_stratum"] = {
        "f_stat": float(anova_table['F'].iloc[0]),
        "p_value": float(anova_table['PR(>F)'].iloc[0]),
        "significant": bool(anova_table['PR(>F)'].iloc[0] < 0.05)
    }

    # 2. Tukey HSD for Stratum Categories
    logger.info("Running Tukey HSD for stratum_category...")
    tukey = pairwise_tukeyhsd(endog=df['Data_Value'], groups=df['stratum_category'], alpha=0.05)
    results["tukey_stratum"] = tukey.summary().as_csv()

    # 3. T-Test for Gender (if available)
    if 'Sex' in df['stratum_category'].unique():
        logger.info("Running T-test for Gender...")
        gender_df = df[df['stratum_category'] == 'Sex']
        males = gender_df[gender_df['stratum_value'] == 'Male']['Data_Value']
        females = gender_df[gender_df['stratum_value'] == 'Female']['Data_Value']
        
        t_stat, p_val = stats.ttest_ind(males, females, nan_policy='omit')
        results["ttest_gender"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
            "male_mean": float(males.mean()),
            "female_mean": float(females.mean())
        }

    # 4. Impact of Income (Ordinal Correlation)
    if 'Income' in df['stratum_category'].unique():
        logger.info("Analyzing Income impact...")
        income_df = df[df['stratum_category'] == 'Income'].copy()
        # Define mapping based on dataset values
        income_map = {
            'Less than $15,000': 1,
            '$15,000 - $24,999': 2,
            '$25,000 - $34,999': 3,
            '$35,000 - $49,999': 4,
            '$50,000 - $74,999': 5,
            '$75,000 or greater': 6,
            'Data not reported': np.nan
        }
        income_df['income_level'] = income_df['stratum_value'].map(income_map)
        income_df = income_df.dropna(subset=['income_level'])
        
        corr, p_val = stats.spearmanr(income_df['income_level'], income_df['Data_Value'])
        results["income_correlation"] = {
            "spearman_rho": float(corr),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05)
        }

    # 5. Impact of Education
    if 'Education' in df['stratum_category'].unique():
        logger.info("Analyzing Education impact...")
        edu_df = df[df['stratum_category'] == 'Education'].copy()
        edu_map = {
            'Less than high school': 1,
            'High school graduate': 2,
            'Some college or technical school': 3,
            'College graduate': 4
        }
        edu_df['edu_level'] = edu_df['stratum_value'].map(edu_map)
        edu_df = edu_df.dropna(subset=['edu_level'])
        
        corr, p_val = stats.spearmanr(edu_df['edu_level'], edu_df['Data_Value'])
        results["education_correlation"] = {
            "spearman_rho": float(corr),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05)
        }

    # Save Results
    out_path = REPORTS_DIR / "metrics" / "statistical_impact.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Statistical analysis results saved to {out_path}")

if __name__ == "__main__":
    run_statistical_analysis()
