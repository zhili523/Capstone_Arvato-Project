import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def clean(df):
    """
    Feature engineering for demographic data

    INPUT: Demographic dataframe
    OUTPUT: Cleaned demographic dataframe
    """

    # Fix mixed-type issue
    print("Step 1 - Fix mixed-type issue")
    df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'][df['CAMEO_DEUG_2015'].notnull()].replace("X","-1").astype('int')
    df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'][df['CAMEO_INTL_2015'].notnull()].replace("XX","-1").astype('int')
    df.CAMEO_DEU_2015 = df.CAMEO_DEU_2015.replace('XX', np.NaN)
    df.OST_WEST_KZ = df.OST_WEST_KZ.replace('W', 1.0).replace('O', 2.0)
    print("Shape after Step 1: {}".format(df.shape))

    # Convert NaN Code
    # Load attribute dataframe
    print('Step 2 - Convert NaN')
    feature = pd.read_csv('./feature_summary.csv')
    feature.drop(['Unnamed: 0'],axis=1, inplace=True)
    feature_notnull = feature[feature['missing_or_unknown'].notna()]
    feature_notnull['missing_or_unknown'] = feature_notnull['missing_or_unknown'].apply(lambda x: x.split(','))

    #Re-encode NaN in df
    for i in feature_notnull.index:
        # Convert each value in missing_or_unknown to 'int' variable if there is
        for value in range(len(feature_notnull.loc[i,'missing_or_unknown'])):
            feature_notnull.loc[i,'missing_or_unknown'][value] = int(feature_notnull.loc[i,'missing_or_unknown'][value])

            # Replace the unknown or missing value to NaN in azdias in the reference of feature dataframe
        df.loc[:,(feature_notnull.loc[i, 'attribute'])].replace(feature_notnull.loc[i,'missing_or_unknown'], np.nan, inplace=True)
    print("Shape after Step 2: {}".format(df.shape))

    # Drop column with above 30% missing rate
    print('Step 3 - Drop column with >30% missing rate')
    df_null_percent = df.isnull().sum()/len(df)
    drop_missing = df_null_percent[df_null_percent>0.3].index
    df = df.drop(columns = drop_missing, axis =1)
    print("Shape after Step 3: {}".format(df.shape))

    # Drop highly correlated features
    print('Step 4 - Drop highly correlated features')
    corr_df = df.corr().abs()
    mask = np.triu(np.ones_like(corr_df,dtype=bool))
    tri_df = corr_df.mask(mask)
    drop_corr = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
    df = df.drop(columns = drop_corr, axis = 1)
    print("Shape after Step 4: {}".format(df.shape))

    # Hot one-encode categorical features
    print('Step 5 - Re-encode categorical features')
    cat_col = feature[feature['type']=='categorical']['attribute']
    cat_col = [x for x in cat_col if x in df.columns]
    multilevel = []
    for col in cat_col:
        if (df[col].nunique() > 2) & (df[col].nunique() < 30):
            multilevel.append(col)

    df.drop(['CAMEO_DEU_2015', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM'], axis=1, inplace=True)
    for feature in multilevel:
        df_notnull = df[feature][df[feature].notnull()]
        dummie_df = pd.get_dummies(df_notnull,prefix=feature)
        df.drop(feature, axis=1, inplace=True)
        df = pd.concat([df, dummie_df], axis = 1)
    print("Shape after Step 5: {}".format(df.shape))

    # Transform mix-type features
    print('Step 6 - Transform some mix-type features')
    mix_col = ['LP_LEBENSPHASE_GROB', 'PRAEGENDE_JUGENDJAHRE', 'WOHNLAGE', 'CAMEO_INTL_2015','PLZ8_BAUMAX']
    #Translate 'PRAEGENDE_JUGENDJAHRE' to decade and movement
    decade = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60, 8: 70, 9: 70, 10: 80, 11: 80, 12: 80, 13: 80, 14: 90, 15: 90,
           np.nan: np.nan, -1: np.nan, 0: np.nan}
    movement = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1, 12: 0, 13: 1, 14: 0, 15: 1,
           np.nan: np.nan, -1: np.nan, 0: np.nan}
    df['Decade'] = df['PRAEGENDE_JUGENDJAHRE'].map(decade)
    df['Movement'] = df['PRAEGENDE_JUGENDJAHRE'].map(movement)
    #Translate 'CAMEO_INTL_2015' to wealth and life stage
    wealth = {11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 31: 3, 32: 3, 33: 3, 34: 3, 35: 3,
         41: 4, 42: 4, 43: 4, 44: 4, 45: 4, 51: 5, 52: 5, 53: 5, 54: 5, 55: 5, -1: np.nan}
    life_stage = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 21: 1, 22: 2, 23: 3, 24: 4, 25: 5, 31: 1, 32: 2, 33: 3, 34: 4, 35: 5,
             41: 1, 42: 2, 43: 3, 44: 4, 45: 5, 51: 1, 52: 2, 53: 3, 54: 4, 55: 5, -1: np.nan}
    df['Wealth'] = df['CAMEO_INTL_2015'].map(wealth)
    df['Life_stage'] = df['CAMEO_INTL_2015'].map(life_stage)
    # Get dummies for other mix-type features
    mix_dummies = ['LP_LEBENSPHASE_GROB', 'WOHNLAGE', 'PLZ8_BAUMAX']
    for feature in mix_dummies:
        df_notnull = df[feature][df[feature].notnull()]
        dummie_df = pd.get_dummies(df_notnull,prefix=feature)
        df = pd.concat([df, dummie_df], axis = 1)
    df = df.drop(mix_col, axis=1)
    print("Shape after Step 6: {}".format(df.shape))

    # Impute the missing value
    print('Step 7 - Impute missing value')
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df.values), columns= df.columns)
    print("Shape after Step 7: {}".format(df.shape))

    # Scale the values
    print('Step 8 - Scale the values')
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df.values),columns=df.columns)
    print("Shape after Step 8: {}".format(df.shape))
    return df
