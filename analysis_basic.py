import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df, threshold=10.0):
    # 数値型の列のみを対象にする
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numerical_cols]

    # VIFを計算
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]
    vif["features"] = df_numeric.columns

    # VIFが閾値以上の変数を取得
    print(f"全体の変数の数：{len(vif)}")
    collinear_features = vif[vif["VIF Factor"] > threshold]["features"]

    return collinear_features


def calculate_basic(df: pd.DataFrame, column_name: str) -> tuple:
    # 欠損値の数をカウントして変数に格納します
    missing_values_count = df[column_name].isnull().sum()
    # 指定された列に欠損値があるレコードを無視します
    df = df.dropna(subset=[column_name])
    mean = df[column_name].mean()
    median = df[column_name].median()
    max_value = df[column_name].max()
    min_value = df[column_name].min()
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)

    return (column_name, mean, median, max_value, min_value, q1, q3, missing_values_count)


def create_basic_analysis(target_df: pd.DataFrame, target_columns: list, save_path: str):
    results = []
    for target_column in target_columns:
        results.append(calculate_basic(target_df, target_column))
    df_results = pd.DataFrame(results, columns=['column_name', 'mean', 'median', 'max_value', 'min_value', 'q1', 'q3', 'missing_values_count'])
    df_results.to_csv(save_path, index=False)


def create_model_summary(df_original: pd.DataFrame):
    """
    linearmodelsによる固定効果モデル分析
    ランダム効果モデルとしての固定効果モデル分析
    """
    # 説明変数の定義
    exog = ['total_regular_worker'] + [col for col in df_original.columns if 'major_class_code_' in col or 'prefecture_' in col or 'year_' in col]
    classcode_dummy = [col for col in df_original.columns if 'major_class_code_' in col]
    year_dummy = [col for col in df_original.columns if 'year_' in col]
    df = df_original[exog + ['actual_employment_rate', 'id', 'year']]
    df = df.set_index(['id', 'year'])

    # VIFの計算
    # collinear_features = calculate_vif(df)
    # print('多重共線性チェック')
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(collinear_features)
    # print('多重共線性チェック終了')

    # 固定効果モデル
    # formula_fe = 'actual_employment_rate ~ ' + ' + '.join(exog) + ' + EntitEffects'
    formula_fe = 'actual_employment_rate ~ ' + ' + '.join(exog) + \
        ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in classcode_dummy]) + \
        ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in year_dummy]) + ' + EntityEffects'
    result_fe = PanelOLS.from_formula(formula_fe, df, check_rank=False, drop_absorbed=True).fit()
    print(result_fe)
    with open('results/basic/results_fe.txt', 'w') as f:
        f.write(str(result_fe))

    # ランダム効果モデル
    df = df_original[exog + ['actual_employment_rate', 'id', 'year']]
    df = df.set_index(['id', 'year'])
    formula_re = 'actual_employment_rate ~ 1 +' + ' + '.join(exog) + \
        ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in classcode_dummy]) + \
        ' + ' + ' + '.join([f'total_regular_worker*{dummy}' for dummy in year_dummy])
    result_re = RandomEffects.from_formula(formula_re, df, check_rank=False).fit()
    print(result_re)
    with open('results/basic/results_re.txt', 'w') as f:
        f.write(str(result_re))


def main():
    # basic analysis
    df = pd.read_csv('datasets/dummy/dummy_extended.csv')
    target_columns = [
        'num_office', 
        'num_regular_worker', 
        'num_parttime_worker', 
        'total_regular_worker', 
        'num_legal_worker', 
        'total_disability',
        'new_total_disability',
        'actual_employment_rate',
        'num_deficient',
        'aging_rate',
        'density',
        'city_population_rate',
        'cpi_regional_diff'
    ]
    create_basic_analysis(df, target_columns, 'results/basic/basic.csv')
    
    # 固定効果モデルによる推定
    df = pd.read_csv('datasets/dummy/dummy_extended.csv')
    create_model_summary(df)


if __name__ == '__main__':
    main()

