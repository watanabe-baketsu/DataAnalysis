import pandas as pd


# 都道府県名の英語表記
prefecture_dict = {
    '北海道': 'hokkaido', '青森県': 'aomori', '岩手県': 'iwate', '宮城県': 'miyagi', '秋田県': 'akita',
    '山形県': 'yamagata', '福島県': 'fukushima', '茨城県': 'ibaraki', '栃木県': 'tochigi', '群馬県': 'gunma',
    '埼玉県': 'saitama', '千葉県': 'chiba', '東京都': 'tokyo', '神奈川県': 'kanagawa', '新潟県': 'niigata',
    '富山県': 'toyama', '石川県': 'ishikawa', '福井県': 'fukui', '山梨県': 'yamanashi', '長野県': 'nagano',
    '岐阜県': 'gifu', '静岡県': 'shizuoka', '愛知県': 'aichi', '三重県': 'mie', '滋賀県': 'shiga',
    '京都府': 'kyoto', '大阪府': 'osaka', '兵庫県': 'hyogo', '奈良県': 'nara', '和歌山県': 'wakayama',
    '鳥取県': 'tottori', '島根県': 'shimane', '岡山県': 'okayama', '広島県': 'hiroshima', '山口県': 'yamaguchi',
    '徳島県': 'tokushima', '香川県': 'kagawa', '愛媛県': 'ehime', '高知県': 'kochi', '福岡県': 'fukuoka',
    '佐賀県': 'saga', '長崎県': 'nagasaki', '熊本県': 'kumamoto', '大分県': 'oita', '宮崎県': 'miyazaki',
    '鹿児島県': 'kagoshima', '沖縄県': 'okinawa', '全県計': 'all_prefectures'
}

def convert_xlsx_to_csv(path, sheet_name, output_path, header_row=0, index_col=0, nrows=None, drop_columns=[]):
    # エクセルファイルの読み込み
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, index_col=index_col, nrows=nrows)

    # 都道府県名を英語表記に変換
    df.index = df.index.map(prefecture_dict)

    for column in drop_columns:
        df.drop(column, axis=1, inplace=True)

    # CSVファイルに変換
    df.to_csv(output_path)

def remove_last_column_from_csv(path):
    df = pd.read_csv(path)
    df.drop(df.columns[-1], axis=1, inplace=True)
    df.to_csv(path, index=False)


def convert_prefecture_name_in_csv(path):
    df = pd.read_csv(path, encoding='shift-jis')
    df.rename(columns={df.columns[0]: 'pref'}, inplace=True)
    df['pref'] = df['pref'].map(prefecture_dict)
    df.to_csv(path, index=False)



def main():
    dataset_path = 'datasets/density_etc_data.xlsx'
    convert_prefecture_name_in_csv('datasets/category_pref_gdp.csv')
    # sheet_name_1 = '都道府県実雇用率'
    # sheet_name_2 = 'density'
    # sheet_name_3 = 'city_population_rate'
    # sheet_name_4 = 'cpi_regional_diff'
    drop_columns = ['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 13']

    # convert_xlsx_to_csv(dataset_path, sheet_name_1, 'datasets/density_etc/pref_actual_employment_rate.csv')
    # convert_xlsx_to_csv(dataset_path, sheet_name_2, 'datasets/density_etc/density.csv')
    # convert_xlsx_to_csv(dataset_path, sheet_name_3, 'datasets/density_etc/city_population_rate.csv')
    # convert_xlsx_to_csv(dataset_path, sheet_name_4, 'datasets/density_etc/cpi_regional_diff.csv')

    # remove_last_column_from_csv('datasets/pref_gdp.csv')

    print('Done')


if __name__ == '__main__':
    main()