import pandas as pd


def count_unique_names_per_prefecture(path: str, save_path: str):
    df = pd.read_csv(path)
    unique_names_per_prefecture = df.groupby('prefecture')['name'].nunique()
    unique_names_per_prefecture.to_csv(save_path)


def add_prefecture_dummy(df: pd.DataFrame):
    most_common_prefecture = df['prefecture'].value_counts().idxmax()
    df_dummy = pd.get_dummies(df['prefecture'], prefix='prefecture')
    df_dummy = df_dummy.drop('prefecture_' + most_common_prefecture, axis=1)
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print("ダミーとして追加した都道府県の数: ", len(df_dummy.columns))
    not_dummy_prefectures = set(df['prefecture'].unique()) - set(df_dummy.columns.str.replace('prefecture_', ''))
    print("ダミーとして追加されなかった都道府県: ", not_dummy_prefectures)
    return df


def add_major_class_dummy(df: pd.DataFrame):
    most_common_class = df['major_class_code'].value_counts().idxmax()
    df_dummy = pd.get_dummies(df['major_class_code'], prefix='major_class_code')
    df_dummy = df_dummy.drop('major_class_code_' + most_common_class, axis=1)
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print(f"データに存在している産業分類数：{df['major_class_code'].nunique()}")
    print("ダミーとして追加したクラスの数: ", len(df_dummy.columns))
    not_dummy_classes = set(df['major_class_code'].unique()) - set(df_dummy.columns.str.replace('major_class_code_', ''))
    print("ダミーとして追加されなかったクラス: ", not_dummy_classes)
    return df


def add_year_dummy(df: pd.DataFrame):
    df_dummy = pd.get_dummies(df['year'], prefix='year', drop_first=True)
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print("ダミーとして追加した年の数: ", len(df_dummy.columns))
    return df


def add_external_data(df: pd.DataFrame, path: str, new_column_name: str):
    external_data = pd.read_csv(path, index_col=0).T
    external_data.index = external_data.index.astype(int)
    external_data_dict = external_data.to_dict()
    # pprint(external_data_dict)
    df[new_column_name] = df.apply(lambda row: external_data_dict.get(row['prefecture'], {}).get(row['year']), axis=1)
    # print(df[["name","year", "prefecture", "aging_rate"]])
    return df



def main():
    # save pref data count of unique name
    count_unique_names_per_prefecture('datasets/panel/panel_base.csv', 'datasets/unique_names_per_prefecture.csv')

    # create prefecture dummy
    df = pd.read_csv('datasets/panel/panel_final.csv')
    df = add_prefecture_dummy(df)
    df = add_major_class_dummy(df)
    df = add_year_dummy(df)
    df.to_csv('datasets/dummy/dummy.csv', index=False)

    # add external data
    df = pd.read_csv('datasets/dummy/dummy.csv')
    df = add_external_data(df, 'datasets/density_etc/aging_rate.csv', 'aging_rate')
    df = add_external_data(df, 'datasets/density_etc/density.csv', 'density')
    df = add_external_data(df, 'datasets/density_etc/city_population_rate.csv', 'city_population_rate')
    df = add_external_data(df, 'datasets/density_etc/cpi_regional_diff.csv', 'cpi_regional_diff')
    df.to_csv('datasets/dummy/dummy_extended.csv', index=False)


if __name__ == '__main__':
    main()

