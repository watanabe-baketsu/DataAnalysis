import os
import pandas as pd
import numpy as np

from pref import prefecture_dict


def add_year_to_datasets():
    files = os.listdir('datasets/raw')
    for file in files:
        if file.endswith('.xlsx'):
            df = pd.read_excel(f'datasets/raw/{file}')
            year = file.rstrip('.xlsx')
            df['年'] = year
            df.to_excel(f"datasets/edit/{file.rstrip('.xlsx')}_add_year.xlsx", index=False)


def display_record_count(path: str):
    files = os.listdir(path)
    for file in files:
        if file.endswith('.xlsx'):
            df = pd.read_excel(f'{path}/{file}')
            print(f'{file}: {len(df)} records')
            """
            2016_add_year.xlsx: 89359 records
            2018_add_year.xlsx: 100586 records
            2015_add_year.xlsx: 87935 records
            2020_add_year.xlsx: 102698 records
            2021_add_year.xlsx: 106924 records
            2022_add_year.xlsx: 107691 records
            2019_add_year.xlsx: 101889 records
            2014_add_year.xlsx: 86648 records
            2017_add_year.xlsx: 91024 records
            """


def replace_column_name_and_save_csv_files(path: str, save_path: str):
    files = os.listdir(path)
    for file in files:
        if file.endswith('.xlsx'):
            df = pd.read_excel(f'{path}/{file}')
            df = df[['法人名称','郵便番号','住所','電話番号','産業中分類番号','産業中分類名','事業所数','常用労働者数','短時間労働者数','常用労働者総数','法定雇用労働者数','障害者計','障害者新規計','実雇用率','不足数','年']]
            df.columns = ['name','postcode','address','tel','middle_class_code','middle_class_name','num_office','num_regular_worker','num_parttime_worker','total_regular_worker','num_legal_worker','total_disability','new_total_disability','actual_employment_rate','num_deficient','year']
            df.to_csv(f'{save_path}/{file.rstrip(".xlsx")}.csv', index=False)


def replace_specials_in_name(path: str, save_path: str):
    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(f'{path}/{file}')
            df['original_name'] = df['name']
            df.insert(1, 'original_name', df.pop(item='original_name'))
            df['name'] = df['name'].str.replace(' ', '')
            df['name'] = df['name'].str.replace('　', '')
            df['name'] = df['name'].str.replace('（株）', '株式会社')
            df['name'] = df['name'].str.replace('（福）', '社会福祉法人')
            df['name'] = df['name'].str.replace('医療法人', '（医）')
            df['name'] = df['name'].str.replace('医療法人社団', '（医）')
            df['name'] = df['name'].str.replace('医療法人財団', '（医）')
            df['name'] = df['name'].str.replace('社会医療法人', '（医）')
            df['name'] = df['name'].str.replace('（医）', '医療法人')
            df['name'] = df['name'].str.replace('（有）', '有限会社')
            df['name'] = df['name'].str.replace('（社福）', '社会福祉法人')
            df['name'] = df['name'].str.replace('（学）', '学校法人')
            df['name'] = df['name'].str.replace('（合同）', '合同会社')
            df['name'] = df['name'].str.replace('（一社）', '一般社団法人')
            df['name'] = df['name'].str.replace('（公財）', '公益財団法人')
            df['name'] = df['name'].str.replace('（特非）', '特定非営利活動法人')
            df['name'] = df['name'].str.replace('（名）', '合名会社')
            df['name'] = df['name'].str.replace('（資）', '合資会社')
            df['name'] = df['name'].str.replace('（同）', '合同会社')
            df['name'] = df['name'].str.replace('（相）', '相互会社')
            df['name'] = df['name'].str.replace('（社）', '社団法人')
            df['name'] = df['name'].str.replace('（公社）', '公益社団法人')
            df['name'] = df['name'].str.replace('（財）', '財団法人')
            df['name'] = df['name'].str.replace('（一財）', '一般財団法人')
            df['name'] = df['name'].str.replace('（宗）', '宗教法人')
            df['name'] = df['name'].str.replace('（独）', '独立行政法人')
            df['name'] = df['name'].str.replace('（地独）', '地方独立行政法人')
            df['name'] = df['name'].str.replace('（弁）', '弁護士法人')
            df['name'] = df['name'].str.replace('（司）', '司法書士法人')
            df['name'] = df['name'].str.replace('（税）', '税理士法人')
            df['name'] = df['name'].str.replace('（行）', '行政書士法人')
            df['name'] = df['name'].str.replace('有限責任中間法人', '（中）')
            df['name'] = df['name'].str.replace('無限責任中間法人', '（中）')
            df['name'] = df['name'].str.replace('（中）', '中間法人')
            df['name'] = df['name'].str.replace('公立大学法人', '（大）')
            df['name'] = df['name'].str.replace('国立大学法人', '（大）')
            df['name'] = df['name'].str.replace('（大）', '大学法人')
            df['name'] = df['name'].str.replace('（営）', '営業所')
            df['name'] = df['name'].str.replace('（出）', '出張所')
            df.to_csv(f"{save_path}/{file}", index=False)


def combine_all_files_into_dataframe(path: str):
    files = os.listdir(path)
    df_list = []
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(f'{path}/{file}')
            df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def remove_incomplete_records_by_name(df: pd.DataFrame):
    df = df.groupby('name').filter(lambda x: len(x) == 9)
    df = df.groupby('name').filter(lambda x: sorted(x['year'].unique().tolist()) == list(range(2014, 2023)))
    df = df.sort_values(['name', 'year'], ascending=[True, True])
    return df


def remove_incomplete_records_by_tel(df: pd.DataFrame):
    df = df.groupby('tel').filter(lambda x: len(x) == 9)
    df = df.groupby('tel').filter(lambda x: sorted(x['year'].unique().tolist()) == list(range(2014, 2023)))
    df = df.sort_values(['tel', 'year'], ascending=[True, True])
    return df


def format_postcode(df: pd.DataFrame):
    df['postcode'] = df['postcode'].astype(str).str.zfill(7)
    return df


def read_post_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='shift_jis', header=None)
    df = df.iloc[:, [2, 6]]
    df.columns = ['postcode', 'prefecture']
    df = format_postcode(df)
    return df


def add_prefecture_to_panel(post_csv_path: str, panel_df: pd.DataFrame) -> pd.DataFrame:
    post_pref_df = read_post_csv(post_csv_path)

    panel_df['postcode'] = panel_df['postcode'].astype(str)
    post_pref_df['postcode'] = post_pref_df['postcode'].astype(str)
    panel_df = format_postcode(panel_df)
    post_pref_df = format_postcode(post_pref_df)

    panel_df = panel_df.merge(post_pref_df, on='postcode', how='left')
    panel_df = panel_df.drop_duplicates(subset=['name', 'year'])

    # 同じnameのレコードはprefectureの値がどれか一つでも存在する場合はその値に統一
    panel_df['prefecture'] = panel_df.groupby('name')['prefecture'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else np.nan)
    # 県情報がないものは削除
    panel_df = panel_df.dropna(subset=['prefecture'])

    return panel_df


def create_classcode_csv(path: str, save_path: str):
    df = pd.read_csv(path, usecols=[0, 1], encoding='shift-jis')
    df.columns = ['major_class_code', 'middle_class_code']
    df['middle_class_code'] = df['middle_class_code'].astype(str).str.zfill(2)
    df['major_class_name'] = df['major_class_code']
    df['major_class_name'] = df['major_class_name'].map({
        'A': '農業、林業',
        'B': '漁業',
        'C': '鉱業、採石業、砂利採取業',
        'D': '建設業',
        'E': '製造業',
        'F': '電気・ガス・熱供給・水道業',
        'G': '情報通信業',
        'H': '運輸業、郵便業',
        'I': '卸売業・小売業',
        'J': '金融業・保険業',
        'K': '不動産業、物品賃貸業',
        'L': '学術研究、専門・技術サービス業',
        'M': '宿泊業、飲食サービス業',
        'N': '生活関連サービス業、娯楽業',
        'O': '教育、学習支援業',
        'P': '医療、福祉',
        'Q': '複合サービス事業',
        'R': 'サービス業（他に分類されないもの）',
        'S': '公務（他に分類されるものを除く）',
        'T': '分類不能の産業'
    })
    df = df.drop_duplicates(subset=['major_class_code', 'major_class_name', 'middle_class_code'])
    df.to_csv(save_path, index=False)


def add_id_to_name(df: pd.DataFrame) -> pd.DataFrame:
    df['id'] = df.groupby('name').ngroup()
    df.insert(0, 'id', df.pop(item='id'))
    return df


def add_major_class_code_and_name(panel_data: pd.DataFrame, class_code_data_path: str) ->pd.DataFrame:
    panel_data['middle_class_code'] = panel_data['middle_class_code'].astype(str).str.zfill(2)
    df_class_code = pd.read_csv(class_code_data_path)
    df_class_code['middle_class_code'] = df_class_code['middle_class_code'].astype(str).str.zfill(2)

    df_merged = pd.merge(panel_data, df_class_code, on='middle_class_code', how='left')
    df_merged.insert(5, 'major_class_code', df_merged.pop(item='major_class_code'))
    df_merged.insert(6, 'major_class_name', df_merged.pop(item='major_class_name'))
    return df_merged


def convert_prefecture_name(df: pd.DataFrame) -> pd.DataFrame:
    df['prefecture'] = df['prefecture'].map(prefecture_dict)
    return df


def main():
    # add_year_to_datasets()
    # display_record_count('datasets/edit/add_year')
    # replace_column_name_and_save_csv_files('datasets/edit/add_year', 'datasets/edit/base_csv')
    # replace_specials_in_name('datasets/edit/base_csv', 'datasets/edit/name_edit')

    df = combine_all_files_into_dataframe('datasets/edit/name_edit')
    df_filtered_by_name = remove_incomplete_records_by_name(df)

    # df_filtered_by_tel = remove_incomplete_records_by_tel(df)
    # print(df_filtered_by_tel)

    df_filtered_by_name = format_postcode(df_filtered_by_name)
    # 同じnameのレコードはmiddle_class_codeの値がどれか一つでも存在する場合はその値に統一
    df_filtered_by_name['middle_class_code'] = df_filtered_by_name.groupby('name')['middle_class_code'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else np.nan)
    # middle_class_codeがないものは削除
    df_filtered_by_name = df_filtered_by_name.dropna(subset=['middle_class_code'])
    df_filtered_by_name['middle_class_code'] = df_filtered_by_name['middle_class_code'].astype(int).astype(str)

    # add prefecture, major_class_code columns
    df_panel_base = add_prefecture_to_panel('datasets/dic/post_to_prefecture.CSV', df_filtered_by_name)
    # create classcode dictionary
    # create_classcode_csv('datasets/dic/class_code.csv', save_path='datasets/dic/class_code_custom.csv')
    df_panel_base = add_major_class_code_and_name(df_panel_base, 'datasets/dic/class_code_custom.csv')
    df_panel_base = add_id_to_name(df_panel_base)
    print(df_panel_base.shape)
    df_panel_base = convert_prefecture_name(df_panel_base)
    df_panel_base.to_csv('datasets/panel_base.csv', index=False)
    print(f"データに存在している県の数：{df_panel_base['prefecture'].nunique()}")
    print(f"データに存在している企業数：{df_panel_base['name'].nunique()}")

    # create panel_final.csv
    df_panel_final = df_panel_base.drop(columns=['postcode', 'address', 'tel'])
    df_panel_final.to_csv('datasets/panel_final.csv', index=False)


if __name__=='__main__':
    main()