import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from linearmodels.panel.results import PanelEffectsResults
from statsmodels.stats.diagnostic import het_breuschpagan


def fe_analysis(df: pd.DataFrame, exog: list, dummy: dict[str, list], target: str = 'actual_employment_rate') -> PanelEffectsResults:
    """
    固定効果モデルによる分析
    """
    classcode_dummy = dummy['classcode']
    year_dummy = dummy['year']
    prefecture_dummy = dummy['prefecture']
    formula = f"{target} ~ " + ' + '.join(exog) + \
            ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in classcode_dummy]) + \
            ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in year_dummy]) + \
            ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in prefecture_dummy]) + ' + EntityEffects'
    result = PanelOLS.from_formula(formula, df, check_rank=False, drop_absorbed=True).fit()
    with open('datasets/results_fe.txt', 'w') as f:
        f.write(str(result))
    print(result)
    return result


def re_analysis(df: pd.DataFrame, exog: list, dummy: dict[str, list], target: str = 'actual_employment_rate') -> PanelEffectsResults:
    """
    ランダム効果モデルによる分析
    """
    classcode_dummy = dummy['classcode']
    year_dummy = dummy['year']
    prefecture_dummy = dummy['prefecture']
    formula = f"{target} ~ 1 +" + ' + '.join(exog) + \
            ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in classcode_dummy]) + \
            ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in year_dummy]) + \
            ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in prefecture_dummy])
    result = RandomEffects.from_formula(formula, df, check_rank=False).fit()
    with open('datasets/results_re.txt', 'w') as f:
        f.write(str(result))
    print(result)
    return result


def create_model_summary_and_hausman_test(df_original: pd.DataFrame, mode: str = 'fe'):
    """
    linearmodelsによる固定効果モデル分析
    ランダム効果モデルとしての固定効果モデル分析
    ハウスマン検定
    """
    # 説明変数の定義
    exog = ['total_regular_worker'] + [col for col in df_original.columns if 'major_class_code_' in col or 'prefecture_' in col or 'year_' in col] \
        + ['aging_rate', 'density', 'city_population_rate', 'cpi_regional_diff']
    classcode_dummy = [col for col in df_original.columns if 'major_class_code_' in col]
    year_dummy = [col for col in df_original.columns if 'year_' in col]
    prefecture_dummy = [col for col in df_original.columns if 'prefecture_' in col]
    df = df_original[exog + ['actual_employment_rate', 'id', 'year']]
    df = df.set_index(['id', 'year'])

    if mode == 'fe':
        fe_analysis(df, exog, {'classcode': classcode_dummy, 'year': year_dummy, 'prefecture': prefecture_dummy})
    # ランダム効果モデル
    elif mode == 're':
        re_analysis(df, exog, {'classcode': classcode_dummy, 'year': year_dummy, 'prefecture': prefecture_dummy})
    elif mode == 'fe_re':
        result_fe = fe_analysis(df, exog, {'classcode': classcode_dummy, 'year': year_dummy, 'prefecture': prefecture_dummy})
        result_re = re_analysis(df, exog, {'classcode': classcode_dummy, 'year': year_dummy, 'prefecture': prefecture_dummy})
        residuals_fe = result_fe.resids
        residuals_re = result_re.resids
        bp_test = het_breuschpagan(residuals_fe - residuals_re, result_re.model.exog.dataframe)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        print("ハウスマン検定")
        print(dict(zip(labels, bp_test)))
    elif mode == 'fe_extract':
        # result_fe = fe_analysis(df, exog, {'classcode': classcode_dummy, 'year': year_dummy, 'prefecture': prefecture_dummy})
        # extract_effect_and_reg_analysis(df, result_fe, exog)
        df = pd.read_csv('datasets/dummy_extended_fe.csv')
        df = df.set_index(['id', 'year'])
        extract_effect_and_reg_analysis(df, exog)


def extract_effect_and_reg_analysis(df: pd.DataFrame, exog: list):
    """
    固定効果モデルの結果から、効果量と回帰分析を行う
    """
    # fe = results.estimated_effects
    # df['estimated_effects'] = fe.values
    formula = 'estimated_effects ~ ' + ' + '.join(exog) + ' + EntityEffects'
    result = PanelOLS.from_formula(formula, df, check_rank=False, drop_absorbed=True).fit()

    with open('datasets/results_fe_extract.txt', 'w') as f:
        f.write(str(result))

    print(result)


def perform_pooled_ols(df: pd.DataFrame, target_year: int):
    """
    PooledOLSを実行する関数
    """
    exog = ['total_regular_worker'] + [col for col in df.columns if 'major_class_code_' in col or 'prefecture_' in col] \
        + ['aging_rate', 'density', 'city_population_rate', 'cpi_regional_diff']
    df = df[df['year'] == target_year]
    df = df.set_index(['id', 'year'])
    df = df[exog + ['actual_employment_rate']]
    classcode_dummy = [col for col in df.columns if 'major_class_code_' in col]
    prefecture_dummy = [col for col in df.columns if 'prefecture_' in col]
    formula_pooled = 'actual_employment_rate ~ 1 + ' + ' + '.join(exog) + \
        ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in classcode_dummy]) + \
        ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in prefecture_dummy])
    result_pooled = PooledOLS.from_formula(formula_pooled, df, check_rank=False).fit()
    print(f"Pooled OLS: {target_year}")
    print(result_pooled)
    


def add_external_data(df: pd.DataFrame, path: str, new_column_name: str):
    external_data = pd.read_csv(path, index_col=0).T
    external_data.index = external_data.index.astype(int)
    external_data_dict = external_data.to_dict()
    # pprint(external_data_dict)
    df[new_column_name] = df.apply(lambda row: external_data_dict.get(row['prefecture'], {}).get(row['year']), axis=1)
    # print(df[["name","year", "prefecture", "aging_rate"]])
    return df


def create_extended_dummy():
    df = pd.read_csv('datasets/dummy.csv')
    df = add_external_data(df, 'datasets/density_etc/aging_rate.csv', 'aging_rate')
    df = add_external_data(df, 'datasets/density_etc/density.csv', 'density')
    df = add_external_data(df, 'datasets/density_etc/city_population_rate.csv', 'city_population_rate')
    df = add_external_data(df, 'datasets/density_etc/cpi_regional_diff.csv', 'cpi_regional_diff')
    df.to_csv('datasets/dummy_extended.csv', index=False)


def filter_by_year_and_remove_unused_dummy(df: pd.DataFrame, years: list) -> pd.DataFrame:
    unused_years = [year for year in df['year'].unique().tolist() if year not in years]
    print(f"unused years: {unused_years}")
    df = df[df['year'].isin(years)]
    unused_dummy_columns = [f"year_{year}" for year in unused_years]
    print(f"unused dummy columns: {unused_dummy_columns}")
    df = df.drop(columns=unused_dummy_columns)
    return df


def main():
    # create_extended_dummy()
    # target_year = [2014, 2015, 2016, 2017, 2018, 2019, 2020]

    # 固定効果モデルによる推定
    df = pd.read_csv('datasets/dummy_extended_fe.csv')
    create_model_summary_and_hausman_test(df, mode='fe_extract')

    # df = filter_by_year_and_remove_unused_dummy(df, target_year)
    # df = add_external_data(df, 'datasets/pref_gdp.csv', 'pref_gdp')
    # df.to_csv('datasets/dummy_extended_2.csv', index=False)
    
    # for year in target_year:
    #     perform_pooled_ols(df, year)



if __name__ == '__main__':
    main()
