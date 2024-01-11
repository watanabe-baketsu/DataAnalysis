import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


class Plotter1:
    def __init__(self, data_path: str, aer_path: str, target_year: int):
        self.df = pd.read_csv(data_path)
        self.aer_df = pd.read_csv(aer_path)
        self.target_year = target_year

    def calculate_specialization(self):
        total_all_prefectures = self.df.loc[self.df['pref'] == 'all_prefectures'].drop(['pref', '小計'], axis=1)
        df = self.df.loc[self.df['pref'] != 'all_prefectures']
        
        for column in df.columns:
            if column not in ['pref', '小計']:
                df[f'{column}_特化係数'] = (df[column] / df['小計']) / (total_all_prefectures[column].values[0] / total_all_prefectures.sum(axis=1).values[0])
        
        return df


    def extract_actual_employment_rate(self, year: int) -> pd.DataFrame:
        df = self.aer_df[['pref', str(year)]]
        df = df.rename(columns={str(year): '都道府県実雇用率'})
        return df

    @staticmethod
    def merge_dfs_on_pref(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        merged_df = pd.merge(df1, df2, on='pref')
        print(merged_df.columns)
        return merged_df

    @staticmethod
    def plot_correlation(df: pd.DataFrame, save_path: str = None):
        num_columns = len([col for col in df.columns if col.endswith('_特化係数')])
        num_rows = num_columns // 2 if num_columns % 2 == 0 else (num_columns // 2) + 1
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, num_rows * 5))  # サイズを調整
        axes = axes.flatten()
        
        for ax, column in zip(axes, [col for col in df.columns if col.endswith('_特化係数')]):
            sns.scatterplot(data=df, x='都道府県実雇用率', y=column, ax=ax)
            ax.set_title(f'都道府県実雇用率と{column}の相関', fontsize=10)  # フォントサイズを調整
            ax.set_xlabel('都道府県実雇用率', fontsize=8)
            ax.set_ylabel(column, fontsize=8)
            
            correlation = df['都道府県実雇用率'].corr(df[column])
            ax.text(0.05, 0.95, f'相関係数: {correlation:.2f}', transform=ax.transAxes, fontsize=8, verticalalignment='top')
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.6)  # スペースを調整
        if save_path:
            plt.savefig(save_path, format='pdf')
        else:
            plt.show()


class Plotter2:
    def __init__(self, dummy_data_path: str):
        self.df = pd.read_csv(dummy_data_path)
    
    def plot_correlation_aer(self, year: int, save_path: str = None):
        plt.figure()
        df_year = self.df[self.df['year'] == year]
        plt.scatter(df_year['total_regular_worker'], df_year['actual_employment_rate'])
        plt.xlabel('常用労働者総数')
        plt.ylabel('実雇用率')
        plt.title(f'実雇用率と常用労働者総数の相関({year}年)')
        correlation = df_year['total_regular_worker'].corr(df_year['actual_employment_rate'])
        plt.text(0.05, 0.95, f'相関係数: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf')
        else:
            plt.show()

    def plot_correlation_effects(self, year: int, save_path: str = None):
        plt.figure()
        df_year = self.df[self.df['year'] == year]
        plt.scatter(df_year['total_regular_worker'], df_year['estimated_effects'])
        plt.xlabel('常用労働者総数')
        plt.ylabel('固定効果量')
        plt.title(f'固定効果量と常用労働者総数の相関({year}年)')
        correlation = df_year['total_regular_worker'].corr(df_year['estimated_effects'])
        plt.text(0.05, 0.95, f'相関係数: {correlation:.2f}', transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, format='pdf')
        else:
            plt.show()


def exec_plot_1(target_year: int):
    data_path = 'datasets/category_pref_gdp.csv'
    aer_path = 'datasets/density_etc/pref_actual_employment_rate.csv'
    plotter = Plotter1(data_path, aer_path, target_year)
    df = plotter.calculate_specialization()
    df.to_csv('datasets/category_pref_gdp_specialization.csv', index=False)

    aer_df = plotter.extract_actual_employment_rate(target_year)

    merged_df = plotter.merge_dfs_on_pref(df, aer_df)
    plotter.plot_correlation(merged_df, save_path='results/correlation.pdf')


def exec_plot_2():
    dummy_data_path = 'datasets/dummy_extended_fe.csv'
    plotter = Plotter2(dummy_data_path)
    for year in range(2014, 2023):
        plotter.plot_correlation_aer(year, save_path=f'results/correlation_aer_trw/{year}.pdf')
        plotter.plot_correlation_effects(year, save_path=f'results/correlation_effects_trw/{year}.pdf')

def main():
    # exec_plot_1(2019)
    exec_plot_2()


if __name__ == '__main__':
    main()