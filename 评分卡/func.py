
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


def SplitData(df, col, numOfSplit, special_attribute=[]):
    '''
    #等频？
    df: 按照col排序后的数据集
    col:待分箱的变量
    numOfSplit :切分的组别数
    special_attribute :在切分数据集的时候，某些特殊值需要排除在外
    return ：在原数据集上增加一列，把原始细粒度的col重新划分为粗粒度的值，便于分享中的合并处理
    '''
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attibute)]
    N = df2.shape[0]
    n = int(N/numOfSplit)
    splitPointIndex = [i * n for i in range(1, numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint
    


# In[ ]:


def Chi2(df, total_col, bad_col):
    '''
    df : 包含全部样本总计与坏样本总计的数据框
    total_col : 全部样本的个数
    bad_col : 坏样本的个数
    return :卡方值
    '''
    df2 = df.copy()
    #求出df中，总体的坏样本率和好样本率
    badRate = sum(df2[bad_col])*1.0/sum(df2[total_col])
    #当全部样本只有好或者坏样本时，卡方值为0
    if badRate in [0,1]:
        return 0
    df2['good'] = df2.apply(lambda x : x[total_col] - x[bad_col], axis=1)
    goodRate = sum(df2['good']) *1.0 / sum(df2[total_col])
    #期望坏（好）样本个数 = 全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df[total_col].apply(lambda x :x*badRate)
    df2['goodExpected'] = df[total_col].apply(lambda x: x*goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0] - i[1]) **2/i[0] for i in badCombined]
    goodChi = [(i[0] - i[1]) **2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2


# In[1]:


def BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left') # 每箱的坏样本数，总样本数
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1) # 加上一列坏样本率
    dicts = dict(zip(regroup[col],regroup['bad_rate'])) # 每箱对应的坏样本率组成的字典
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)


# In[ ]:


def chiMerge(df, col, target, max_interval=5, special_attribute=[], minBinPcnt=0):
    '''
    df : 包含目标变量与分箱属性的数据框
    col: 需要分箱的属性
    target:目标变量，取值0或1
    max_interval:最大分箱数，如果原始属性的取值个数低于该参数，不执行这段函数
    special_attribute:不参与分箱的属性取值
    minBinPcnt:最小箱的占比，默认为0
    return：分箱结果
    '''
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval: #如果原始属性的取值个数低于max_intervals,不执行这段函数
        print("the number od original level () is less " .format(col))
        print (colLevels[:-1])
    else:
        if len(special_attribute) >= 1:
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))
        #步骤一：通过col对数据集进行分组，求出分组的总样本数与坏样本数
        if N_distinct > 100:
            split_x = SplitData(df2, col, 100)
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
            #AssignGroup函数：每一行的数值和切分点做对比，返回原值在切分后的映射，经过map以后，生成该特征的值对象的
        else:
            df2['temp'] = df2[col]
        #总体bad rate 将被用来计算expected bad count
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp',target,grantRateIndicator=1)
        
        #首先，每个单独的属性值将被分为单独的一组
        #对属性值进行排序，然后两两组别进行合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]
        #把每个分箱的值打包成[[], []]的形式
        
        #步骤二：建立循环，不断合最优的相邻两个组别，直到：
        #1.最终分裂出来的箱数<=预设的最大的分箱数
        #2.每箱的占比不低于预设值（可选）
        #3.每箱同时包含于好坏样本
        #4.如果有特殊属性，那么最终分裂出来的分箱数=预设的分箱数-特殊属性的个数
        split_intervals = max_interval - len(special_attribute)
        while (len(groupIntervals) > split_intervals):
            #终止条件：当前分箱数=预设分箱数
            #每次循环时，计算合并相邻组别后的卡方值。具有最小卡方值的合并方案是最优方案
            chisqList = []
            for k in range(len(groupIntervals) -1):
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad')
                chisqList.append(chisq)
            best_combined = chisqList.index(min(chisqList))
            
            groupIntervals[best_combined] = groupIntervals[best_combined] + groupIntervals[best_combined]
            groupIntervals.remove(groupIntervals[best_combined+1])
            
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]
        #检查是否有箱没有坏或者好样本，如果有，需要和相邻的箱进行合并
        groupedvalues = df2['temp'].apply(lambda x : AssignBin(x, cutOffPoints))
        df2['temp_Bin'] = groupedvalues
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
        #返回（每箱坏样本率字典和包含列名、坏样本数、总样本数、坏样本率的数据框）
        [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
        while minBadRate == 0 or maxBadRate ==1:
            #找出全部为好/坏样本的箱
            indexForBad01 = regroup[regroup['bad_rate'].isin([0,1])].temp_Bin.tolist()
            bin = indexForBad01[0]
            #如果是最后一箱，则需要和上一箱进行合并，也就意味着分裂点最后一个CutOffPoints需要移除
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            #如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints的第一个需要移除
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            #如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
            else:
                #和前一箱合并，并且计算卡方值
                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex -1]
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                chisq1 = Chi2(df2b, 'total', 'bad')
                #和后一箱合并，计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex +1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex,bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                chisq2 = Chi2(df2b, 'total', 'bad')
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex -1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])
            #完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupedvalues = df2['temp'].apply(lambda x:AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
            (minBadRate, maxBadRate) = [min(binBadRate.values()), max(binBadRate.values())]
        #需要检查分箱后的最小占比
        if minBinPcnt > 0:
            groupedvalues = df2['temp'].apply(lambda x : AssignBin(x, cutOffPoins))
            df2['temp_Bin'] = groupedvalues
            valueCounts = groupedvalues.value_counts().to_frame()
            valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x :x*1.0/sum(valueCounts['temp']))
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])
            while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
                #找出占比最小的箱
                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
                #如果最小占比的箱是最后一箱，则需要和上一箱进行合并
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                #如果占比最小的箱是第一箱，则需要和下一箱进行合并
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                #如果占比最小的箱在中间，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                else:
                    #和前一箱进行合并，计算卡方值
                    currentIndex = list(valueCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex -1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                    chisq1 = Chi2(df2b,'taotal','bad')
                    #和后一箱进行合并，计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex +1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin',target)
                    chisq2 = Chi2(df2b, 'total', 'bad')
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex -1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex +1])
                groupedValues = df2['temp'].apply(lambda x:AssignBin(x, cutOffPoints))
                df2['temp_Bin'] = groupedvalues()
                valueCounts = groupedvalues.value_counts().to_frame()
                valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x : x*1.0/sum(valueCounts['temp']))
                valueCounts = valueCounts.sort_index()
                minPcnt = min(valueCounts['pcnt'])

        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints
    
                    
            
            


# In[1]:


def UnsupervisedSplitBin(df, var, numOfSplit =5, method = 'equal freq'):
    '''
    df : 数据集
    var : 需要分箱的变量，仅限于数值型
    numOfSplit : 需要分箱个数，默认是5
    method ： 分箱方法，默认是等频，否则等距
    '''
    if method == 'equal freq':
        N = df.shape[0]
        n = N/numOfSplit
        splitPointIndex = [i * n for i in range(1, numOfSplit)]
        rawValues = sorted(list(df[col]))
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
        return splitPoint
    else:
        var_max, var_min = max(df[var]), min(df[var])
        interval_len = (var_max - var_min)*1.0 / numOfSplit
        splitPoint = [var_min + i*interval_len for i in range(1, numOfSplit)]
        return splitPoint
    
        


# In[4]:


def AssignGroup(x, bin):
    '''
    x:某个变量的某个取值
    bin:上述变量的分箱结果
    return x在分箱结果下的映射
    '''
    N = len(bin)
    if x<= min(bin):
        return min(bin)
    elif x>max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]


# In[7]:


def BadRateEncoding(df, col, target):
    '''
    df : dataframe,包含变量和目标值
    col:需要用bad rate编码的变量，通常是分类变量
    target : good/bad 指标
    return：分类变量的bad rate编码
    '''
    
    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]
    br_dict = regroup[[col, 'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEncoding = df[col].map(lambda x: br_dict[x])
    return {'encoding' : badRateEncoding, 'bad_rate':br_dict}


# In[8]:


def AssignBin(x, cutOffPoints, special_attribute=[]):
    '''
    x : 某个变量的某个取值
    cutOffPoints :上述变量的分箱结果，用切分点表示
    special_attribute:不参与分箱的特殊取值
    return:分箱后的对应的第几个箱，从0开始
    '''
    numBin = len(cutOffPoints) +1 +len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x) +1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints[0]:
        return 'Bin 0'
    elif x>cutOffPoints[-1]:
        return 'Bin {}'.format(numBin -1)
    else:
        for i in range(0, numBin -1):
            if cutOffPoints[i] <x<=cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)


# In[9]:


def CalcWOE(df, col, target):
    '''
    df: 包含需要计算WOE的变量和目标变量
    col : 需要计算WOE、IV的变量，必须是分箱后的变量
    target : 目标变量,0/1表示好/坏
    return :返回woe和iv
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total' : total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad' : bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x : x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x : x*1.0/G)
    regroup['WOE'] = regroup.apply(lambda x : np.log(x.good_pcnt *1.0 /x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x:(x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt/x.bad_pcnt), axis=1)
    IV = sum(IV)
    return {'WOE' : WOE_dict, "IV" : IV}


# In[1]:


#判断某变量的坏样本率是否单调
def BadRateMonotone(df, sortByVar, target, special_attribute =[]):
    '''
    df : 包含检验坏样本率的变量和目标变量
    sortByVar : 需要检验坏样本率的变量
    target：目标变量
    special_attribute:不参与检验的特殊值
    return :坏样本率单调与否
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <=2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'], regroup['bad'])
    badRate = [x[1]*1.0 /x[0] for x in combined]
    
    badRateNotMonotone = [badRate[i] < badRate[i+1] and badRate[i] < badRate[i-1] or
                         badRate[i] > badRate[i+1] and badRate[i] <badRate[i] < badRate[i-1] for i in range(1, len(badRate)-1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True
    


# In[2]:


def MergeBad0(df, col, target, direction='bad'):
    '''
    df:包含检验0%和100%坏样本率
    col:分箱后的变量或者类别型变量，检验其中一组或者多组没有坏样本或好样本，如果是则需要合并
    target:目标变量
    direction:合并方案，使得每个组里同时包含好坏样本
    '''
    regroup = BinBadRate(df, col, target)[1]
    if direction == 'bad':
        # 如果是合并0坏样本率的组，则跟最小的非0坏样本率的组进行合并
        regroup = regroup.sort_values(by='bad_rate')
    else:
        #如果是合并0好样本率的组，则跟最小的非0好样本率的组进行合并
        regroup = regroup.sort_values(by='bad_rate', ascending=False)
    regroup.index = range(regroup.shape[0])
    col_regroup = [[i] for i in regroup[col]]
    del_index = []
    for i in range(regroup.shape[0]-1):
        col_regroup[i+1] = col_regroup[i] + col_regroup[i+1]
        del_index.append[i]
        if direction == 'bad':
            if regroup['bad_rate'][i+1]>0:
                break
        else:
            if regroup['bad_rate'][i+1]<1:
                break
    col_regroup2 = [col_regroup[i] for i in range(len(col_regroup)) if i not in del_index]
    newGroup = {}
    for i in range(len(col_regroup2)):
        for g2 in col_regroup2[i]:
            newGroup[g2] = 'Bin' + str[i]
    return newGroup


# In[3]:


def Prob2Score(prob, basePoint, PDO):
    #将概率转化为分数且为正整数
    y = np.log(prob/(1-prob))
    return int(basePoint+PDO/np.log(2)*(-y))


# In[4]:


#计算KS值
def KS(df, score, target):
    '''
    df : 包含目标变量与预测值的数据集
    score : 得分或概率
    target ： 目标变量
    return ： ks值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total' : total, "bad" : bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score, ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum()/all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum()/all['good'].sum()
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(KS)
    


# In[5]:


def MergeByCondition(x, condition_list):
    #condition_list是条件列表，满足第几个condition，就输出几
    s = 0
    for condition in conditionn_list:
        if eval(str(x) + condition):
            return s
        else:
            s +=1
    return s


# In[ ]:




