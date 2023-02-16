import os
import re
import datetime
from typing import Union, Tuple
from copy import deepcopy
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# from lightgbm import log
pd.options.display.precision = 4

class EDA:

    def __init__(self, data: pd.DataFrame) -> None:
        self.raw_data = deepcopy(data)
        self.data = data
        self.set_categorical_and_numerical_feature()

        # 以下、各メソッド実行中に値が代入されるインスタンス変数。
        self.target_vname = None
        self.categorical_NA = None
        self.numerical_NA =None
        self.categorical_na_dict = None # カテゴリ特徴量と置換値がセットになったdictオブジェクト
        self.numerical_na_dict = None # 数値特徴量と置換値がセットになったdictオブジェクト
        

    def set_categorical_and_numerical_feature(self):
        
        self.categorical_feature = self.data.select_dtypes(include="object")
        self.numerical_feature = self.data.loc[:, self.data.dtypes!="object"]

    def reset_data(self):
        self.data = deepcopy(self.raw_data)
        self.categorical_feature = self.data.select_dtypes(include="object")
        self.numerical_feature = self.data.loc[:, self.data.dtypes!="object"]

    def fillna(self, categorical_na: str="NaN", numerical_na: np.float64=-99, threshold: float=0.5) -> Tuple[dict, dict, dict, dict]:

        """
        メソッド概要：
        サンプル数全体に対する欠損値の割合が設定した閾値(threshold)以上なら、定数(categorical_na or numrical_na)で置換。
        閾値以下なら、最頻値(categorical feature)または中央値(numerical feature)で置換。

        欠損値の割合が少なければ、それは偶発的に発生したものと仮定し、代表値（最頻値や中央値）で置換。
        割合が大きい場合は、偶発的ではなくデータの収集方法・過程で欠損値が発生する特定の理由があるものと仮定し、任意の定数で置換。

        引数：
        categorical_na -> 閾値以上のカテゴリ欠損値を置換する置換値
        numerical_na -> 閾値以上の数値欠損値を置換する置換値
        threshold -> サンプル数全体に対する欠損値の割合の閾値。これを超えると定数で置換される。

        Return:
        特徴量名(key)と置換値(value)で構成されるカテゴリ特徴量と数値特徴量の2つのdictオブジェクト。
        
        *メソッド実行後に以下の2つにインスタンス変数としてもアクセス可能。
        (1)self.categorical_NA_dict (2)self.numerical_NA_dict 
        """
        for df, df_type in (zip([self.categorical_feature, self.numerical_feature], ["categorical", "numerical"])):
            
            print(f"----------------{df_type}_feature----------------")
            df = deepcopy(df)
            for i in range(2): #閾値以上、以下の2ループ
                if i==0:
                    # 閾値以上の場合
                    flag = (((df.isna().sum() / self.data.shape[0])>=threshold).values)
                    name = df.loc[:, flag].columns
                    
                    if df_type == "categorical":
                        value = categorical_na
                        self.categorical_NA = categorical_na
                        map_dict = {n:value for n in name}
                        const_categorical_NA_dict = map_dict
                    else:
                        value = numerical_na
                        self.numerical_NA = numerical_na
                        map_dict = {n:value for n in name}
                        const_numerical_NA_dict = map_dict

                    # 重要!: df.fillnaに関して次のURLを参照 (https://stackoverflow.com/questions/21998354/pandas-wont-fillna-inplace)
                    self.data.fillna(map_dict, inplace=True) 
                    print(f"{name} \n accounts for more than 90% of the total at each feature and inplace by \n{value}")
                    
                else:
                    # 閾値以下の場合
                    flag = (((df.isna().sum() / self.data.shape[0])<threshold).values)
                    name = df.loc[:, flag].columns
                    temp = self.data.loc[:, name].mode() if df_type == "categorical" else self.data.loc[:, name].median()
                    value_type = "mode" if df_type == "categorical" else "median"
                    value = temp.values[0] if df_type == "categorical" else temp.values
                    map_dict = dict(zip(name, value))
                    
                    if df_type == "categorical":
                        temp = self.data.loc[:, name].mode()
                        value_type = "mode" # 最頻値
                        value = temp.values[0]
                        map_dict = dict(zip(name, value))
                        statistic_categorical_NA_dict = map_dict
                    else:
                        temp = self.data.loc[:, name].median()
                        value_type = "median" # 中央値
                        value = temp.values
                        map_dict = dict(zip(name, value))
                        statistic_numerical_NA_dict = map_dict

                    self.data.fillna(map_dict, inplace=True)
                    print(f"{name} \n don't accounts for more than 90% of the total at each feature and inplace by {value_type}")
                    # print(type(value))

        self.categorical_na_dict = deepcopy(const_categorical_NA_dict)
        self.numerical_na_dict = deepcopy(const_numerical_NA_dict)

        for d_1, d_2 in ([(self.categorical_na_dict, statistic_categorical_NA_dict), (self.numerical_na_dict, statistic_numerical_NA_dict)]):
            for key, value in d_2.items():
                d_1[key] = value

        return self.categorical_na_dict, self.numerical_na_dict

        # return self.const_categorical_NA_dict, self.const_numerical_NA_dict, self.statistic_categorical_NA_dict, self.statistic_numerical_NA_dict                    

    def set_target_variable(self, name="str"):

        if name in self.data.columns:
            self.target_vname = name
        else:
            raise ValueError(f"{name} isn't included in input data!")

    def _check_target_variable(self):

        if self.target_vname == None:
            raise ValueError("please set target_variavle name by using set_traget_variable methods!")

    def _check_name_include_dataset(self, name: list=None):

        for n in name:
            if not (n in self.data.columns):
                raise ValueError(f"{n} isn't include in this dataset!")
            

    def count_plots(self, cols: int=4):
        total_num = len(self.categorical_feature.columns)
        rows = np.ceil(total_num/cols).astype(np.int8)
        k = 0
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5.5*cols, 5.5*rows), squeeze=False, sharey=True)
        for i in range(rows):
            for j in range(cols):
                if k < total_num:
                    sns.countplot(data=self.categorical_feature, x=self.categorical_feature.columns[k], ax=axes[i, j])
                    axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(),rotation = 30)
                    k += 1

    def violin_plots(self, cols: int=4):
        self._check_target_variable()
        total_num = len(self.categorical_feature.columns)
        rows = np.ceil(total_num/cols).astype(np.int8)
        k = 0
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5.5*cols, 5.5*rows), squeeze=False, sharey=True)
        for i in range(rows):
            for j in range(cols):
                if k < total_num:
                    sns.violinplot(data=self.data, x=self.categorical_feature.columns[k], y=self.target_vname, ax=axes[i, j])
                    axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(),rotation = 30)
                    k += 1

    def dist_plots(self, cols=4):
        total_num = len(self.numerical_feature.columns)
        rows = np.ceil(total_num/cols).astype(np.int8)
        k = 0
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(5.5*cols, 5.5*rows), squeeze=False, sharey=False)
        for i in range(rows):
            for j in range(cols):
                if k < total_num:
                    sns.histplot(data=self.numerical_feature, x=self.numerical_feature.columns[k], kde=False, ax=axes[i, j], color=cm.jet(k/total_num))
                    k += 1

    def reg_plots(self, cols=4):
        self._check_target_variable()
        total_num = len(self.numerical_feature.columns)
        rows = np.ceil(total_num/cols).astype(np.int8)
        k = 0
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7.5*cols, 5.5*rows), squeeze=False, sharey=False)
        for i in range(rows):
            for j in range(cols):
                if k < total_num:
                    sns.regplot(data=self.numerical_feature, x=self.numerical_feature.columns[k], 
                                    y=self.data[self.target_vname], ax=axes[i, j], color=cm.jet(k/total_num), scatter_kws={'alpha':0.2})
                    k += 1

    def corr_heatmap(self):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
        corr = self.numerical_feature.corr(method="pearson")
        sns.heatmap(corr, annot=True, cmap="bwr")

    def search_low_variance_feature(self, threshold: float=0.9) -> list:
        # カテゴリカル特徴量、離散値特徴量(dtypes=int)は1種類の特徴量の中で相対度数が閾値以上(threshold)の階級が存在するものを返す。
        # 今回、連続値データ(float)は除外
        low_variace = []
        for i, vname in enumerate(self.data.columns):
            if self.data[vname].dtypes != np.float64:
                value_count = self.data[vname].value_counts()
                sum = value_count.sum()
                relative_freq = pd.Series(value_count.values / sum, index=value_count.index, name="relative_freq")
                flag = (relative_freq.values > threshold).sum() > 0
                if flag:
                    display(pd.concat([value_count, relative_freq], axis=1).head(5))
                    low_variace.append(vname)
        print("-----------------------------------------------------")
        print(f"low_variance_feature = {low_variace}")
        return low_variace

    def qqplots(self, name: list=None, make_log: bool=False, cols=4) -> None:
        
        self._check_name_include_dataset(name=name)
        total_num = len(name)*2 if make_log else len(name)
        rows = np.ceil(total_num/cols).astype(np.int8)
        k = 0
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7.5*cols, 5.5*rows), squeeze=False, sharey=False)
        for i in range(rows):
            for j in range(cols):
                if k < total_num:
                    if make_log:
                        if k % 2 == 0:
                            stats.probplot(x=self.data[name[int(k/2)]], dist="norm", plot=axes[i, j], fit=True)
                            axes[i, j].set_title(f"probability plot for {name[int(k/2)]}")
                        else:
                            stats.probplot(x=np.log1p(self.data[name[int((k-1)/2)]].values), dist="norm", plot=axes[i, j], fit=True)
                            axes[i, j].set_title(f"probability plot for logarithmic of {name[int((k-1)/2)]}")
                    else:
                        stats.probplot(x=self.data[name[k]], dist="norm", plot=axes[i, j], fit=True)
                        axes[i, j].set_title(f"probability plot for {name[k]}")
                    k += 1

    def search_outlier(self, feature_series: pd.Series=None, alpha: float=0.95) -> list:
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5))
        mean = feature_series.mean()
        std = feature_series.std()
        lower, upper = stats.norm.interval(alpha=alpha, loc=mean, scale=std)
        X_norm = np.linspace(feature_series.min(), feature_series.max(), 200)
        ax.hist(feature_series, bins=30, color="blue", density=True)
        ax.plot(X_norm, stats.norm.pdf(X_norm, loc=mean, scale=std), color="red", label="Norm", ls="-")
        # plt.plot(X_norm, stats.t.pdf(X_norm, loc=mean, scale=std, df=len(feature_series)-1), color="green", label="t", ls="--")
        ax.vlines(x=upper, ymax=3.0, ymin=0, colors='gray', linestyles='dashed')
        ax.vlines(x=lower, ymax=3.0, ymin=0, colors='gray', linestyles='dashed')
        ax.set_ylim(0, 3.0)
        ax.legend(fontsize=15)
        ax.set_title(f"{feature_series.name} with {alpha*100}% normal interval")
        
        outlier_flag = (feature_series.to_numpy()<lower) + (feature_series.to_numpy()>upper)
        outlier = self.data.loc[outlier_flag, :]
        print(f"outlier_index -> {outlier.index}")

        return outlier.index

    def outlier_plot(self, name: list=None, outlier_id: list=None, cols=4) -> None:

        self._check_name_include_dataset(name=name)
        self._check_target_variable()
        total_num = len(name)
        rows = np.ceil(total_num/cols).astype(np.int8)
        k = 0
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7.5*cols, 5.5*rows), squeeze=False, sharey=True)
        outlier = self.data.iloc[outlier_id, :]
        for i in range(rows):
            for j in range(cols):
                if k < total_num:
                    sns.scatterplot(x=f'{name[k]}', y=self.target_vname, data=self.data, color="black", label="normal", ax=axes[i, j])
                    sns.scatterplot(x=f'{name[k]}', y=self.target_vname, data=outlier, color='red', label="outlier", ax=axes[i, j])
                    k += 1

    def target_and_feature_realtion_plots(self, name: list=None, cols=4) -> None:
        self._check_name_include_dataset(name=name)
        self._check_target_variable()
        total_num = len(name)
        cols = cols if total_num > cols else total_num
        rows = np.ceil(total_num/cols).astype(np.int8)
        k = 0
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(7.5*cols, 6.5*rows), squeeze=False, sharey=False)
        for i in range(rows):
            for j in range(cols):
                if k < total_num:
                    if name[k] in self.categorical_feature.columns:
                        sns.violinplot(x=f'{name[k]}', y=self.target_vname, data=self.data, ax=axes[i, j])
                        axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(),rotation = 45)
                    elif name[k] in self.numerical_feature.columns:
                        sns.regplot(data=self.data, x=f'{name[k]}', 
                                    y=self.target_vname, ax=axes[i, j], color="blue", scatter_kws={'alpha':0.2})
                    k += 1

    def label_encoding(self, original_labels: list, encoding_labels: list) -> list:

        """
        引数
        original_labels: エンコーディング対象のカテゴリ変数の種類。例->Ex, Gd etc. これに該当するカテゴリ変数を全てエンコードする。
        encoging_labels: エンコーディングのラベル
        """
        encoded_feature = []
        map_dict = dict(zip(original_labels, encoding_labels))
        display(map_dict)
        unique_v_Series = self.categorical_feature.apply(np.unique, axis=0) # -> pd.Series
        for name, uniqe_value in zip(unique_v_Series.index, unique_v_Series):
            
            flag = False
            count = 0
            for v in uniqe_value:
                if v in original_labels:
                    count += 1
            
            # unique値が全てoriginal_labelsに含まれるのか
            flag = True if count==len(uniqe_value) else False     
            if flag:
                # print(f"name->{name}, uq_value->{uniqe_value}")
                encoded_feature.append(name)
        for name in encoded_feature:
            self.data[name] = self.data[name].map(map_dict)
        
        self.set_categorical_and_numerical_feature()
        display(self.data.loc[:,encoded_feature])
        return encoded_feature
        

def fill_diff_between_dfs(train: EDA, test: EDA, na_value: str="NaN") -> Tuple[EDA, EDA]:
    """
    テストデータには含まれていて訓練データには含まれいないカテゴリ変数値を欠損値に置換する。
    """
    uq_train = train.categorical_feature.apply(np.unique, axis=0)
    uq_test = test.categorical_feature.apply(np.unique, axis=0)
    names = []
    diffs = []
    for i, (tr, te) in enumerate(zip(uq_train, uq_test)):
        tr = set(tr)
        te = set(te)
        diff = te.difference(tr)
        if len(diff)!=0:
            names.append(uq_train.index[i])
            diffs.append(diff)
            print(f"name: {uq_train.index[i]}")
            print(f"train_data -> {tr}")
            print(f"test_data -> {te}")
            print(f"{diff} aren't included in train_data, but included in test_data")

    for name, diff in zip(names, diffs):
        for value in diff:
            flag = (test.data[name]==value)
            print(f"value => {value}\nflag -> {flag}")
            test.data[name][flag] = train.categorical_na_dict[name]

    return train, test