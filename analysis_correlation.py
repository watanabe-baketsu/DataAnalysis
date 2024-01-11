import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns


def calculate_specialization(df):
    total_all_prefectures = df.loc[df['pref'] == 'all_prefectures'].drop(['pref', '小計'], axis=1)
    df = df.loc[df['pref'] != 'all_prefectures']
    
    for column in df.columns:
        if column not in ['pref', '小計']:
            df[f'{column}_特化係数'] = (df[column] / df['小計']) / (total_all_prefectures[column].values[0] / total_all_prefectures.sum(axis=1).values[0])
    
    return df


def extract_actual_employment_rate(df, year: int) -> pd.DataFrame:
    df = df[['pref', str(year)]]
    df = df.rename(columns={str(year): '都道府県実雇用率'})
    return df


def merge_dfs_on_pref(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(df1, df2, on='pref')
    print(merged_df.columns)
    return merged_df


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


def main():
    data_path = 'datasets/category_pref_gdp.csv'
    df = pd.read_csv(data_path)
    df = calculate_specialization(df)
    df.to_csv('datasets/category_pref_gdp_specialization.csv', index=False)

    aer_path = 'datasets/density_etc/pref_actual_employment_rate.csv'
    aer_df = pd.read_csv(aer_path)
    aer_df = extract_actual_employment_rate(aer_df, 2020)

    merged_df = merge_dfs_on_pref(df, aer_df)
    plot_correlation(merged_df, save_path='results/correlation.pdf')


if __name__ == '__main__':
    main()