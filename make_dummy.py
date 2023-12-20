import pandas as pd


def create_prefecture_dummy(path: str, save_path: str):
    df = pd.read_csv(path)
    df_dummy = pd.get_dummies(df['prefecture'], prefix='prefecture')
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print("ダミーとして追加した都道府県の数: ", len(df_dummy.columns))
    df.to_csv(save_path, index=False)


def count_unique_names_per_prefecture(path: str, save_path: str):
    df = pd.read_csv(path)
    unique_names_per_prefecture = df.groupby('prefecture')['name'].nunique()
    unique_names_per_prefecture.to_csv(save_path)


def create_major_class_dummy(path: str, save_path: str):
    df = pd.read_csv(path)
    df_dummy = pd.get_dummies(df['major_class_code'], prefix='major_class_code')
    df = pd.concat([df, df_dummy], axis=1)
    df[df_dummy.columns] = df[df_dummy.columns].astype(int)
    print(f"データに存在している企業数：{df['major_class_code'].nunique()}")
    print("ダミーとして追加したクラスの数: ", len(df_dummy.columns))
    df.to_csv(save_path, index=False)


def create_year_dummy(path: str, save_path: str):
    df = pd.read_csv(path)
    df_dummy = pd.get_dummies(df['year'], prefix='year')
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

