import pandas as pd


def calculate_disability_ratio(df) -> pd.Series:
    total_regular_worker = df['total_regular_worker'].sum()
    disability_by_industry = df.groupby('major_class_code')['total_disability'].sum()
    disability_ratio = disability_by_industry / total_regular_worker
    return disability_ratio



def main():
    target_path = 'datasets/edit/name_edit/2022_add_year.csv'
    df = pd.read_csv(target_path)
    df = df[['middle_class_code', 'total_regular_worker', 'total_disability']]

    class_code_df = pd.read_csv('datasets/dic/class_code_custom.csv')
    df['middle_class_code'] = df['middle_class_code'].astype(str).str.zfill(2)
    class_code_df['middle_class_code'] = class_code_df['middle_class_code'].astype(str).str.zfill(2)
    df = pd.merge(df, class_code_df[['middle_class_code', 'major_class_code']], on='middle_class_code', how='left')
    # print(df.head())

    disability_ratio = calculate_disability_ratio(df)
    print(disability_ratio)
    disability_ratio.to_csv('results/raw/disability_ratio.csv')

if __name__ == '__main__':
    main()