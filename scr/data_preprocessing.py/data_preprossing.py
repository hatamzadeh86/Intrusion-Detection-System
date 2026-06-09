from sklearn.preprocessing import OneHotEncoder , StandardScaler ,LabelEncoder
import pandas as pd


#_______________________________________________________________________________
uf = pd.read_csv("Test_KDDTest.csv")
uf = uf.drop_duplicates()
#_______________________________________________________________________________
uf['target_ATTACK'] = (uf['target'] != 'normal').astype(int)
uf = uf.drop(columns=['target'],errors='ignore')
cols_string = uf.select_dtypes(exclude='number').columns
for i in cols_string :
    le = LabelEncoder()
    uf[i] = le.fit_transform(uf[i])
#_______________________________________________________________________________
# print(uf.head(10))
# print(uf.isnull().sum())

# print(uf['target_ATTACK'].value_counts())
# print(uf.columns)

# print(uf.info())
#_______________________________________________________________________________
cols_X = uf.corr()['target_ATTACK'].abs().sort_values(ascending=False).copy()
drop_corr = cols_X[cols_X >= 0.01].index.tolist()

uf_clean = uf[drop_corr]
#_______________________________________________________________________________

# print(uf_clean.corr()['target_ATTACK'].abs().sort_values(ascending=False))

# for h in uf_clean.columns:
#     sns.boxplot(x=uf_clean[h] , data=uf_clean)
#     plt.show()


for cols_outlier in uf_clean.columns :
    Q1 = uf_clean[cols_outlier].quantile(0.25)
    Q3 = uf_clean[cols_outlier].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    uf_clean[cols_outlier] = uf_clean[cols_outlier].clip(lower , upper)


# for h in uf_clean.columns:
#     sns.boxplot(x=uf_clean[h] , data=uf_clean)
#     plt.show()
print(uf_clean['target_ATTACK'].value_counts())

# for u in uf_clean.columns :
#     vars_cols = uf_clean[u].var()
#     print(vars_cols)
#_______________________________________________________________________________