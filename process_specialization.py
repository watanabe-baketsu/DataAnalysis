import pandas as pd
from linearmodels.panel import PanelOLS
from linearmodels.panel.results import PanelEffectsResults
import statsmodels.api as sm

from pref import prefecture_dict

class Processor:
    def __init__(self, pref_numbe_data_path: str, data_path: str):
        self.pref_num_df = pd.read_csv(pref_numbe_data_path)
        self.pref_num_df['en_name'] = self.pref_num_df['jp_name'].map(prefecture_dict)
        self.data_df = pd.read_csv(data_path)

    @staticmethod
    def _calculate_specialization(df: pd.DataFrame) -> pd.DataFrame:
        total_all_prefectures = df.loc[df['pref'] == 'all_prefectures'].drop(['pref', '小計'], axis=1)
        df_copy = df.loc[df['pref'] != 'all_prefectures'].copy()
        
        for column in df_copy.columns:
            if column not in ['pref', '小計']:
                df_copy.loc[:, f'{column}_特化係数'] = (df_copy[column] / df_copy['小計']) / (total_all_prefectures[column].values[0] / total_all_prefectures.sum(axis=1).values[0])
        
        return df_copy

    def make_specializationdataset(self):
        df_2017 = pd.read_csv('datasets/category_pref_gdp/2017.csv')
        df_2018 = pd.read_csv('datasets/category_pref_gdp/2018.csv')
        df_2019 = pd.read_csv('datasets/category_pref_gdp/2019.csv')
        df_2020 = pd.read_csv('datasets/category_pref_gdp/2020.csv')

        df_2017 = self._calculate_specialization(df_2017)
        df_2018 = self._calculate_specialization(df_2018)
        df_2019 = self._calculate_specialization(df_2019)
        df_2020 = self._calculate_specialization(df_2020)

        df_2017.to_csv('datasets/category_pref_gdp/2017_specialization.csv', index=False)
        df_2018.to_csv('datasets/category_pref_gdp/2018_specialization.csv', index=False)
        df_2019.to_csv('datasets/category_pref_gdp/2019_specialization.csv', index=False)
        df_2020.to_csv('datasets/category_pref_gdp/2020_specialization.csv', index=False)

    @staticmethod
    def update_year(df: pd.DataFrame) -> pd.DataFrame:
        df['year'] = df['year'].map({2017: 1, 2018: 2, 2019: 3, 2020: 4})
        return df

    def add_specialization_data(self):
        # Get prefecture names from self.pref_num_df
        self.data_df = self.data_df.merge(self.pref_num_df[['Prefecture_id', 'en_name']], left_on='Prefecuture_id', right_on='Prefecture_id', how='left')

        # Load specialization data and merge it to self.data_df
        specialization_df = pd.read_csv('datasets/category_pref_gdp/2017_specialization.csv')
        specialization_df['year'] = 2017
        for year in range(2018, 2021):
            temp_df = pd.read_csv(f'datasets/category_pref_gdp/{year}_specialization.csv')
            temp_df['year'] = year
            specialization_df = pd.concat([specialization_df, temp_df], axis=0, ignore_index=True)
        specialization_df = self.update_year(specialization_df)
        # rename columns
        self.data_df.rename(columns={'en_name': 'pref'}, inplace=True)
        # Get the columns with '特化係数' in their names
        coef_columns = [col for col in specialization_df.columns if '特化係数' in col]
        coef_columns.extend(['pref', 'year'])

        # Merge the specialization data to self.data_df
        self.data_df = pd.merge(self.data_df, specialization_df[coef_columns], on=['pref', 'year'], how='left')

    @staticmethod
    def calculate_specialization_sum(df: pd.DataFrame) -> dict:
        coef_columns = [col for col in df.columns if '特化係数' in col]
        year_sum_dict = {}
        for year in df['year'].unique():
            year_df = df[df['year'] == year]
            year_sum = year_df[coef_columns].sum(axis=1).sum()
            year_sum_dict[year] = year_sum
        return year_sum_dict
    
    @staticmethod
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
        std = df[column_name].std()  # 標準偏差を計算

        return (column_name, mean, median, max_value, min_value, q1, q3, std, missing_values_count)


class PanelProcessor:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)

    def drop_rows_with_missing_values(self, prefecture_id: int):
        self.df = self.df[self.df['Prefecture_id'] != prefecture_id]
    
    def _fe_analysis(self, exog: list, target: str) -> PanelEffectsResults:
        """
        固定効果モデルによる分析
        """
        print(self.df)
        print(self.df.columns)
        df_panel = self.df[exog + ['Prefecture_id', 'year', target]]
        jp_exog = False
        for column in exog:
            if '特化係数' in column:
                jp_exog =True
        if jp_exog:
            # 特化係数が列名に含まれる列をアルファベットに置き換える
            coef_columns = [col for col in df_panel.columns if '特化係数' in col]
            alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
            for i, col in enumerate(coef_columns):
                df_panel.rename(columns={col: alphabet[i]}, inplace=True)
                exog = [alphabet[i] if x==col else x for x in exog]
        if '特化係数' in target:
            df_panel.rename(columns={target: 'JP_TARGET'}, inplace=True)
            target = 'JP_TARGET'
        df_panel = df_panel.set_index(['Prefecture_id', 'year'])
        formula = f"{target} ~ " + ' + '.join(exog) + ' + EntityEffects'
        result = PanelOLS.from_formula(formula, df_panel, check_rank=False, drop_absorbed=True).fit()
        return result
    
    def execute_fe_analysis(self, save_path: str):
        exog = [
            'MLR', 'ILR', 'LCLR', 'SCLR', 'MDER', 'IDER', 
            'LCDER', 'SCDER', 'CPR', 'EPR', 'CPDI', 'LPSP', 'EMCW', 
            'DP', 'MLdammy'
        ] + [col for col in self.df.columns if '特化係数' in col]
        result = self._fe_analysis(exog, 'TDER')
        print(result)
        with open(save_path, 'w') as f:
            f.write(str(result))
    
    def execute_fe_analysis_2(self):
        exog = ['EPR', 'CPDI', 'LPSP', 'DP', 'MLdammy']
        targets = ['情報通信業_特化係数', '保険衛生・社会事業_特化係数', 'LCLR', 'SCLR']
        for target in targets:
            result = self._fe_analysis(exog, target)
            print(result)
            with open(f'results/specialization/panel_fe_{target}.txt', 'w') as f:
                f.write(str(result))
        


def make_specializationdataset():
    processor = Processor('datasets/都道府県番号と都道府県の対応.csv', 'datasets/研究用データ.csv')
    processor.add_specialization_data()
    processor.data_df.to_csv('datasets/研究用データ(特化係数追加済み).csv', index=False)
    


def perform_ols_analysis(df, target_year, save_path, target_column='TDER', exog=['LCLR', 'SCLR', 'EMCW']):
    df_target_year = df[df['year'] == target_year]

    X_columns = exog
    X = df_target_year[X_columns]
    y = df_target_year[target_column]
    model = sm.OLS(y, sm.add_constant(X))
    results = model.fit(cove_type='HC1')

    with open(save_path, 'w') as f:
        f.write(results.summary().as_text())
    print(results.summary())


def basic_analysis(df, processor, save_path):
    target_columns = [
        'TDER', 'MLR', 'ILR', 'LCLR', 'SCLR', 'MDER', 'IDER', 
        'LCDER', 'SCDER', 'CPR', 'EPR', 'CPDI', 'LPSP', 'EMCW', 
        'DP', 'MLdammy'
    ] + [col for col in df.columns if '特化係数' in col]
    results = []
    for target_column in target_columns:
        results.append(processor.calculate_basic(df, target_column))
    df_results = pd.DataFrame(results, columns=[
        'column_name', 'mean', 'median', 'max_value', 'min_value', 
        'q1', 'q3', 'std', 'missing_values_count'
        ])
    print(df_results)
    df_results.to_csv(save_path, index=False)


def one_year_ols_analysis(df: pd.DataFrame, target_year: int):
    # 2020年のみの単年度回帰分析
    df = pd.read_csv('datasets/研究用データ(特化係数追加済み).csv')
    # target = 'TDER'の場合
    save_path = 'results/specialization/ols_analysis_2020_TDER.txt'
    target = 'TDER'
    exog = ['LCLR', 'SCLR', 'EMCW'] + [col for col in df.columns if '特化係数' in col]
    perform_ols_analysis(df, target_year, save_path, target, exog)
    # target = '情報通信業_特化係数'の場合
    save_path = 'results/specialization/ols_analysis_2020_情報通信業_特化係数.txt'
    target = '情報通信業_特化係数'
    exog = ['EPR', 'CPDI', 'LPSP', 'DP', 'MLdammy']
    perform_ols_analysis(df, target_year, save_path, target, exog)
    # target = '保険衛生・社会事業_特化係数'の場合
    save_path = 'results/specialization/ols_analysis_2020_保険衛生・社会事業_特化係数.txt'
    target = '保険衛生・社会事業_特化係数'
    exog = ['EPR', 'CPDI', 'LPSP', 'DP', 'MLdammy']
    perform_ols_analysis(df, target_year, save_path, target, exog)
    # target = 'LCLR'の場合
    save_path = 'results/specialization/ols_analysis_2020_LCLR.txt'
    target = 'LCLR'
    exog = ['EPR', 'CPDI', 'LPSP', 'DP', 'MLdammy']
    perform_ols_analysis(df, target_year, save_path, target, exog)
    # target = 'SCLR'の場合
    save_path = 'results/specialization/ols_analysis_2020_SCLR.txt'
    target = 'SCLR'
    exog = ['EPR', 'CPDI', 'LPSP', 'DP', 'MLdammy']
    perform_ols_analysis(df, target_year, save_path, target, exog)




def main():
    # make dataset
    # processor = Processor()
    # processor.make_specializationdataset()
    processor = Processor('datasets/都道府県番号と都道府県の対応.csv', 'datasets/研究用データ.csv')

    # 2020年のみの単年度回帰分析
    # df = pd.read_csv('datasets/研究用データ(特化係数追加済み).csv')
    # one_year_ols_analysis(df, 4)


    # 基本統計量の算出
    # basic_analysis(df, processor, 'results/specialization/basic_analysis.csv')

    # パネルデータ分析
    panel_processor = PanelProcessor('datasets/研究用データ(特化係数追加済み).csv')
    panel_processor.drop_rows_with_missing_values(5)
    # panel_processor.execute_fe_analysis('results/specialization/panel_fe.txt')
    panel_processor.execute_fe_analysis_2()
    


if __name__ == '__main__':
    main()