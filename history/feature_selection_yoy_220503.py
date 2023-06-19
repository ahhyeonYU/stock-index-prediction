from collect_data import collect_ecos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import MinMaxScaler

def get_data():
    ## 1. Get data ##
    ecos_df = collect_ecos()

    ## 2. Preprocessing ##
    # nan이 있는 행 삭제
    ecos_df.dropna(inplace=True)
    ecos_df.reset_index(drop=True, inplace=True)

    # 칼럼명 내 '(', ')' 기호 -> '[', ']'로 수정
    ecos_df_cols_new = list(ecos_df.columns)
    for idx, i in enumerate(ecos_df_cols_new):
        temp_name = re.sub('[\( ]', '_', i)
        temp_name = re.sub('[\)\,\-]', '', temp_name)
        ecos_df_cols_new[idx] = temp_name

    ecos_df.columns = ecos_df_cols_new
    # 칼럼 type numeric으로 수정
    ecos_df = ecos_df.apply(pd.to_numeric, errors='coerce')
    '''
    다음 월(next month)의 종가를 값으로 갖는 새로운 칼럼 만들기
    '''
    # 기존 칼럼 갯수
    column_index = len(ecos_df.columns)

    # 새로운 칼럼에 임시로 0.1값 부여(float으로 만들기 위해)
    ecos_df.insert(column_index, column='KOSPI_NEW', value=0.1)
    #ecos_df.insert(column_index+1, column='KOSDAQ_NEW', value=0.1)

    # 새로운 칼럼에 다음 월 종가 기입
    for i in range(len(ecos_df)-1):
        ecos_df.loc[i, 'KOSPI_NEW'] = ecos_df.loc[i+1, 'KOSPI_종가']
        #ecos_df.loc[i, 'KOSDAQ_NEW'] = ecos_df.loc[i+1, 'KOSDAQ_종가']

    # 마지막 행 삭제(새로운 칼럼에 값이 없기 때문)
    ecos_df.drop(labels=ecos_df.index[-1], axis='index', inplace=True)

    # 코스피/코스닥 칼럼명 변경
    ecos_df.rename(columns={'KOSPI_종가':'KOSPI', 'KOSDAQ_종가':'KOSDAQ'}, inplace=True) 
    # 전월 대비 증가했으면 1, 감소했으면 0부여
    for i in range(len(ecos_df)):
        if ecos_df.loc[i, 'KOSPI'] <= ecos_df.loc[i, 'KOSPI_NEW']:
            ecos_df.loc[i, 'KOSPI_BINARY'] = 1
        else:
            ecos_df.loc[i, 'KOSPI_BINARY'] = 0

        # if ecos_df.loc[i, 'KOSDAQ'] <= ecos_df.loc[i, 'KOSDAQ_NEW']:
        #     ecos_df.loc[i, 'KOSDAQ_BINARY'] = 1
        # else:
        #     ecos_df.loc[i, 'KOSDAQ_BINARY'] = 0

    # BINARY 칼럼 type을 int로
    ecos_df = ecos_df.astype({'KOSPI_BINARY':'int'})
    #ecos_df = ecos_df.astype({'KOSPI_BINARY':'int', 'KOSDAQ_BINARY':'int'})
    # 전년 동기 대비 값으로 변경하기
    ecos_df_cols = list(ecos_df.columns)
    not_yoy_cols = ['TIME', '한국은행_기준금리', 'KOSPI_NEW', 'KOSPI_BINARY']

    for c in ecos_df_cols:
        if c in not_yoy_cols:
            continue
        else:
            ecos_df[f'{c}_yoy'] = None
            for i in range(12, len(ecos_df)):
                yoy_value = (ecos_df.loc[i, c] - (ecos_df.loc[i-12, c]))/(ecos_df.loc[i-12, c])
                ecos_df.loc[i, f'{c}_yoy'] = yoy_value
    erase_cols = ecos_df_cols
    for i in not_yoy_cols:
        erase_cols.remove(i)
    ecos_df = ecos_df.loc[12: ]
    ecos_df = ecos_df[ecos_df.columns.difference(erase_cols)]
    ecos_df.reset_index(drop=True, inplace=True)

    # 칼럼 type numeric으로 수정
    ecos_df = ecos_df.apply(pd.to_numeric, errors='coerce')

    #ecos_df.loc[12:,:].isna().any().sum()

    '''
    22.04.28. 추가 작업
    - 사유: 모델링 중 스케일링이 안 되어 있음을 발견
    - 작업 내용: ecos_df를 상관분석 하기 전에 스케일링
    - 유의점: py에서는 return에 scaler도 포함시켜야 한다.
    '''

    ecos_df_cols = list(ecos_df.columns)
    # 각 값들 소수점 둘째자리에서 반올림
    ecos_df = ecos_df.apply(lambda x : round(x, 2))

    # 스케일링 진행
    scaler = MinMaxScaler()
    scaler.fit(ecos_df)
    '''
    scaler에서 inf값 때문에 오류가 난다면, 해당 지표는 삭제하는 게 좋다고 생각한다. 결측값 처리하기가 참 애매하다. 그냥 빼는 게 낫다
    -> 신용 대주 잔고 삭제.
    '''
    ecos_df = scaler.transform(ecos_df)

    # 데이터프레임화
    ecos_df = pd.DataFrame(ecos_df)
    ecos_df.columns = ecos_df_cols

    ## 3. Correlation analysis ##
    # 상관분석
    ecos_corr = ecos_df.corr()
    ## heatmap
    #plt.figure(figsize=(15, 15))
    #sns.heatmap(data = ecos_corr, annot=True, fmt='.2f', linewidths=.5, cmap='Reds')
    '''
    높은 상관관계가 있는 칼럼 추출
    '''

    threshholds = 0.2

    # 필수 포함 feature: based on 도메인지식
    nece_ftrs = ['한국은행_기준금리', 'KOSPI_yoy', '국고채_10년_yoy', 'KOSPI_주가이익비율_3_yoy', 'KOSPI_BINARY']

    kospi_corr = ecos_corr['KOSPI_NEW']
    #kosdaq_corr = ecos_corr['KOSDAQ_NEW']

    # 불필요한 칼럼 삭제
    kospi_corr.drop(labels=['TIME', 'KOSPI_NEW'], axis='index', inplace=True)
    #kosdaq_corr.drop(labels=['TIME', 'KOSPI_NEW', 'KOSDAQ_NEW'], axis='index', inplace=True)

    # threshholds 기준에 부합하는 칼럼 출력
    kospi_ftrs = list(kospi_corr[(kospi_corr>=threshholds) | (kospi_corr<=-threshholds)].keys())
    #kosdaq_ftrs = kosdaq_corr[(kosdaq_corr>=0.3) | (kosdaq_corr<=-0.3)].keys()

    # 필수 feature가 없을 시 추가
    for f in nece_ftrs:
        if f not in kospi_ftrs:
            kospi_ftrs.append(f)
    # 상관분석 결과 바탕으로 데이터셋 수정
    #kospi_ftrs.insert(0, 'TIME')
    #kospi_ftrs.extend(['KOSPI_NEW'])
    ecos_df = ecos_df[kospi_ftrs]
    
    return ecos_df, scaler