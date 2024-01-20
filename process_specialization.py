import pandas as pd

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

    
def main():
    # processor = Processor()
    # processor.make_specializationdataset()

    processor = Processor('datasets/都道府県番号と都道府県の対応.csv', 'datasets/研究用データ.csv')
    processor.add_specialization_data()
    processor.data_df.to_csv('datasets/研究用データ(特化係数追加済み).csv', index=False)


if __name__ == '__main__':
    main()
    main()