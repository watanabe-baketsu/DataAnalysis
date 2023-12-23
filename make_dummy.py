import pandas as pd


def create_prefecture_dummy(path: str, save_path: str):
    df = pd.read_csv(path)
    most_common_prefecture = df['prefecture'].value_counts().idxmax()
    df_dummy = pd.get_dummies(df['prefecture'], prefix='prefecture')
    df_dummy = df_dummy.drop('prefecture_' + most_common_prefecture, axis=1)
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print("ダミーとして追加した都道府県の数: ", len(df_dummy.columns))
    not_dummy_prefectures = set(df['prefecture'].unique()) - set(df_dummy.columns.str.replace('prefecture_', ''))
    print("ダミーとして追加されなかった都道府県: ", not_dummy_prefectures)
    df.to_csv(save_path, index=False)


def count_unique_names_per_prefecture(path: str, save_path: str):
    df = pd.read_csv(path)
    unique_names_per_prefecture = df.groupby('prefecture')['name'].nunique()
    unique_names_per_prefecture.to_csv(save_path)


def create_major_class_dummy(path: str, save_path: str):
    df = pd.read_csv(path)
    most_common_class = df['major_class_code'].value_counts().idxmax()
    df_dummy = pd.get_dummies(df['major_class_code'], prefix='major_class_code')
    df_dummy = df_dummy.drop('major_class_code_' + most_common_class, axis=1)
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print(f"データに存在している企業数：{df['major_class_code'].nunique()}")
    print("ダミーとして追加したクラスの数: ", len(df_dummy.columns))
    not_dummy_classes = set(df['major_class_code'].unique()) - set(df_dummy.columns.str.replace('major_class_code_', ''))
    print("ダミーとして追加されなかったクラス: ", not_dummy_classes)
    df.to_csv(save_path, index=False)


def create_year_dummy(path: str, save_path: str):
    df = pd.read_csv(path)
    df_dummy = pd.get_dummies(df['year'], prefix='year', drop_first=True)
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print("ダミーとして追加した年の数: ", len(df_dummy.columns))
    df.to_csv(save_path, index=False)



def main():
    # create prefecture dummy
    create_prefecture_dummy('datasets/panel_final.csv', 'datasets/dummy.csv')

    # save pref data count of unique name
    count_unique_names_per_prefecture('datasets/panel_base.csv', 'datasets/unique_names_per_prefecture.csv')

    create_major_class_dummy('datasets/dummy.csv', 'datasets/dummy.csv')

    create_year_dummy('datasets/dummy.csv', 'datasets/dummy.csv')

    pass


if __name__ == '__main__':
    main()

