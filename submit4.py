# -----------------------------------
# 学習データ、テストデータの読み込み
# -----------------------------------
from statistics import *
import numpy as np
import pandas as pd
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
train_x = train.values
train_y = train.columns.values
# 学習データを特徴量と目的変数に分ける
train_x = train.drop(['y'], axis=1)
train_y = train['y']
test_x = test.copy()

# -----------------------------------
# 特徴量作成
# -----------------------------------
from sklearn.preprocessing import LabelEncoder

# それぞれのカテゴリ変数にlabel encodingを適用する
for c in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
    # 学習データに基づいてどう変換するかを定める
    le = LabelEncoder()
    # 学習データ、テストデータを変換する
    train_x[c] = le.fit_transform(pd.DataFrame(train_x[c]))
    test_x[c] = le.fit_transform(pd.DataFrame(test_x[c]))
for i in train_x:
    print(i)
    
# -----------------------------------
# 正規化
# -----------------------------------
normalization_categories = ['day', 'month']
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
for c in normalization_categories:
    train_x[c] = (train_x[c]-min(train_x[c]))/(max(train_x[c])-min(train_x[c]))
    test_x[c] = (train_x[c]-min(train_x[c]))/(max(train_x[c])-min(train_x[c]))

# -----------------------------------
# 標準化
# -----------------------------------
print(train_x[c])
standardization_categories = ['balance', 'age', 'duration', 'campaign', 'pdays', 'previous']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for c in standardization_categories:
    # print((train_x[c] - mean(train_x[c]))/pstdev(train_x[c]))
    train_x[c] = (train_x[c] - mean(train_x[c]))/pstdev(train_x[c])
    test_x[c] = (train_x[c] - mean(train_x[c]))/pstdev(test_x[c])
    
for i in train_x:
    print(i)
    print(' max:', max(train_x[i]),' min:', min(train_x[i]),' mean:', mean(train_x[i]) ,' median:', median(train_x[i]), ' mode:', mode(train_x[i]), ' pvariance:', pvariance(train_x[i]))
    print('\n')

# -----------------------------------
# 行を落とす
# -----------------------------------
# train_x = train_x.drop('pdays', axis=1)
# test_x = test_x.drop('pdays', axis=1)
# train_x = train_x.drop('balance', axis=1)
# test_x = test_x.drop('balance', axis=1)
# train_x = train_x.drop('default', axis=1)
# test_x = test_x.drop('default', axis=1)

# train_x = train_x.drop('id', axis=1)
# test_x = test_x.drop('id', axis=1)


# -----------------------------------
# 相関係数
# -----------------------------------
for i in train_x:
    print('\n', i)
    print(train_y.corr(train_x[i]))


# -----------------------------------
# グラフ化
# -----------------------------------
import matplotlib.pyplot as plt

#figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
fig = plt.figure()
print(train_x.shape[1])
#add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
for i, label in enumerate(train_x):
    if(i + 1 < train_x.shape[1]/2):
        ax = fig.add_subplot(2, int(train_x.shape[1]/2) + 1, i + 1)
    else:
        ax = fig.add_subplot(2, int(train_x.shape[1]/2) + 1, i + 1)
    ax.hist(train_x[label])
    ax.set_title(label)
ax = fig.add_subplot(2, int(train_x.shape[1]/2) + 1, 18)
ax.hist(train_y)
ax.set_title('train_y')
plt.show()
# -----------------------------------
# モデル作成
# -----------------------------------
from xgboost import XGBClassifier

# モデルの作成および学習データを与えての学習
model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

# テストデータの予測値を確率で出力する
pred = model.predict_proba(test_x)[:, 1]

# テストデータの予測値を二値に変換する
pred_label = np.where(pred > 0.5, 1, 0)

# 提出用ファイルの作成
# submission = pd.DataFrame({'y': test['PassengerId'], 'Survived': pred_label})
# submission.to_csv('submission_first.csv', index=False)
# スコア：0.7799（本書中の数値と異なる可能性があります）

# -----------------------------------
# バリデーション
# -----------------------------------
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

# 各foldのスコアを保存するリスト
scores_accuracy = []
scores_logloss = []

# クロスバリデーションを行う
# 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
kf = KFold(n_splits=4, shuffle=True, random_state=71)
for tr_idx, va_idx in kf.split(train_x):
    # 学習データを学習データとバリデーションデータに分ける
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # モデルの学習を行う
    model = XGBClassifier(n_estimators=20, random_state=71)
    model.fit(tr_x, tr_y)

    # バリデーションデータの予測値を確率で出力する
    va_pred = model.predict_proba(va_x)[:, 1]

    # バリデーションデータでのスコアを計算する
    logloss = log_loss(va_y, va_pred)
    accuracy = accuracy_score(va_y, va_pred > 0.5)

    # そのfoldのスコアを保存する
    scores_logloss.append(logloss)
    scores_accuracy.append(accuracy)

# 各foldのスコアの平均を出力する
logloss = np.mean(scores_logloss)
accuracy = np.mean(scores_accuracy)
print(f'logloss: {logloss:.4f}, accuracy: {accuracy:.4f}')
# logloss: 0.4270, accuracy: 0.8148（本書中の数値と異なる可能性があります）

# -----------------------------------
# モデルチューニング
# -----------------------------------
import itertools

# チューニング候補とするパラメータを準備する
param_space = {
    'max_depth': [3, 5, 7],
    'min_child_weight': [1.0, 2.0, 4.0]
}

# 探索するハイパーパラメータの組み合わせ
param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

# 各パラメータの組み合わせ、それに対するスコアを保存するリスト
params = []
scores = []

# 各パラメータの組み合わせごとに、クロスバリデーションで評価を行う
for max_depth, min_child_weight in param_combinations:

    score_folds = []
    # クロスバリデーションを行う
    # 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す
    kf = KFold(n_splits=4, shuffle=True, random_state=123456)
    for tr_idx, va_idx in kf.split(train_x):
        # 学習データを学習データとバリデーションデータに分ける
        tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

        # モデルの学習を行う
        model = XGBClassifier(n_estimators=20, random_state=71,
        max_depth=max_depth, min_child_weight=min_child_weight)
        model.fit(tr_x, tr_y)

        # バリデーションデータでのスコアを計算し、保存する
        va_pred = model.predict_proba(va_x)[:, 1]
        logloss = log_loss(va_y, va_pred)
        score_folds.append(logloss)

    # 各foldのスコアを平均する
    score_mean = np.mean(score_folds)
    print('score_mean', score_mean)

    # パラメータの組み合わせ、それに対するスコアを保存する
    params.append((max_depth, min_child_weight))
    scores.append(score_mean)

# 最もスコアが良いものをベストなパラメータとする
best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
print(f'max_depth: {best_param[0]}, min_child_weight: {best_param[1]}')
# max_depth=7, min_child_weight=2.0のスコアが最もよかった


# -----------------------------------
# ロジスティック回帰用の特徴量の作成
# -----------------------------------
# from sklearn.preprocessing import OneHotEncoder

# # 元データをコピーする
# train_x2 = train.drop(['y'], axis=1)
# train_x2 = train_x2.drop(index=10810)
# test_x2 = test.copy()
# #print('______________')
# #print(train_x2['job'].unique())
# #print(train_x2['job'].value_counts())
# #print('______________')
# #print(test_x2['job'].unique())
# #print(test_x2['job'].value_counts())
# #print('______________')
# i = test_x2['job'] == 'un'
# #print(i.sort_values())
# test_x2 = test_x2.drop(index=10810)
# #print(test_x2['job'].value_counts())
# #print('______________')
# #print(test_x2.isnull().value_counts())
# #print('______________')
# # # 変数PassengerIdを除外する
# # train_x2 = train_x2.drop(['PassengerId'], axis=1)
# # test_x2 = test_x2.drop(['PassengerId'], axis=1)

# # # 変数Name, Ticket, Cabinを除外する
# # train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)
# # test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# # one-hot encodingを行う
# cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
# ohe = OneHotEncoder(categories='auto', sparse=False)
# ohe.fit(train_x2[cat_cols].fillna('NA'))

# # one-hot encodingのダミー変数の列名を作成する
# ohe_columns = []
# for i, c in enumerate(cat_cols):
#     ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]

# # one-hot encodingによる変換を行う
# ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)
# ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)

# # one-hot encoding済みの変数を除外する
# # one-hot encoding済みの変数を除外する
# train_x2 = train_x2.drop(cat_cols, axis=1)
# test_x2 = test_x2.drop(cat_cols, axis=1)

# # one-hot encodingで変換された変数を結合する
# train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)
# test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)

# # 数値変数の欠損値を学習データの平均で埋める
# num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
# for col in num_cols:
#     train_x2[col].fillna(train_x2[col].mean(), inplace=True)
#     test_x2[col].fillna(train_x2[col].mean(), inplace=True)

# 変数Fareを対数変換する
# train_x2['Fare'] = np.log1p(train_x2['Fare'])
# test_x2['Fare'] = np.log1p(test_x2['Fare'])

# -----------------------------------
# アンサンブル
# -----------------------------------
from sklearn.linear_model import LogisticRegression

# xgboostモデル
model_xgb = XGBClassifier(n_estimators=20, random_state=71)
model_xgb.fit(train_x, train_y)
pred_xgb = model_xgb.predict_proba(test_x)[:, 1]

# ロジスティック回帰モデル
# xgboostモデルとは異なる特徴量を入れる必要があるので、別途train_x2, test_x2を作成した
# model_lr = LogisticRegression(solver='lbfgs', max_iter=300)
# train_x2 = train_x2.fillna(train_x2.median())
# test_x2 = test_x2.fillna(test_x2.median())
# model_lr.fit(train_x2, train_y)
# #print(test_x2[test_x2.isnull()])
# pred_lr = model_lr.predict_proba(test_x2)[:, 1]

# 予測値の加重平均をとる
# pred = pred_xgb * 0.8 + pred_lr * 0.2
# pred_label = np.where(pred > 0.5, 1, 0)
pred = pred_xgb 

import glob
files = glob.glob('*.csv')
#print(files)
num = 1
for file in files:
    if('submit' in file):
        if(str(num) in file):
            num += 1

sub = pd.read_csv('submit_sample.csv', encoding = 'UTF-8', names=['idx', 'ans'])
sub['ans'] = list(pred)
# sub.to_csv("submit4.csv", header=False, index=False)
#print('fin')