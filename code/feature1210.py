import pandas as pd
import numpy as np
from multiprocessing import Pool
# 读取数据
login = pd.read_csv('./Risk_Detection_Qualification/t_login_v1.csv')
trade = pd.read_csv('./Risk_Detection_Qualification/t_trade.csv')
login_test = pd.read_csv('./Risk_Detection_Qualification/t_login_test_v1.csv')
trade_test = pd.read_csv('./Risk_Detection_Qualification/t_trade_test.csv')
# 区分购买记录和登录记录
login['buy'] = np.zeros((len(login), 1));login['train'] = np.ones((len(login), 1))
trade['buy'] = np.ones((len(trade), 1));trade['train'] = np.ones((len(trade), 1))
# 区分测试数据和训练数据
login_test['buy'] = np.zeros((len(login_test), 1));login_test['train'] = np.zeros((len(login_test), 1))
trade_test['buy'] = np.ones((len(trade_test), 1));trade_test['train'] = np.zeros((len(trade_test), 1))
df = pd.concat([login, trade, login_test, trade_test])
# 转换时间序列格式
df.time = pd.to_datetime(df.time)
# 第一排序按照时间排序，第二排序按照id排序
df = df.sort_values(by=['time', 'id'])
df = df.reset_index()
device = df[df.buy == 0].device.tolist()
ip = df[df.buy == 0].ip.tolist()
city = df[df.buy == 0].city.tolist()
device_bad = df['is_risk'].device.tolist()
ip_bad = df['is_risk'].ip.tolist()
city_bad = df['is_risk'].city.tolist()
device_dict = {}
ip_dict = {}
city_dict = {}
device_bad_dict = {}
ip_bad_dict = {}
city_bad_dict = {}
for dev in device:
    if dev in device_dict.keys():
        device_dict[dev] += 1
    else:
        device_dict[dev] = 1
for ip1 in ip:
    if ip1 in ip_dict.keys():
        ip_dict[ip1] += 1
    else:
        ip_dict[ip1] = 1
for ci in city:
    if ci in city_dict.keys():
        city_dict[ci] += 1
    else:
        city_dict[ci] = 1

for dev2 in device_bad:
    if dev2 in device_bad_dict.keys():
        device_bad_dict[dev2] += 1
    else:
        device_bad_dict[dev2] = 1
for ip2 in ip:
    if ip2 in ip_bad_dict.keys():
        ip_bad_dict[ip2] += 1
    else:
        ip_bad_dict[ip2] = 1
for ci2 in city_bad:
    if ci2 in city_bad_dict.keys():
        city_bad_dict[ci2] += 1
    else:
        city_bad_dict[ci2] = 1

for item1 in device_dict.keys():
    device_dict[item1] = device_dict[item1]/len(device)
for item2 in ip_dict.keys():
    ip_dict[item2] = ip_dict[item2]/len(ip)
for item3 in city_dict.keys():
    city_dict[item3] = city_dict[item3]/len(city)

for item11 in device_bad_dict.keys():
    device_bad_dict[item11] = device_bad_dict[item11] / len(device_bad)
for item22 in ip_bad_dict.keys():
    ip_bad_dict[item22] = ip_bad_dict[item22] / len(ip_bad)
for item33 in city_bad_dict.keys():
    city_bad_dict[item33] = city_bad_dict[item33] / len(city_bad)

df.timelong = df.timelong.map(lambda x:x if x>100 else x*1000 )
def generate_feature(id):
    #针对每个id进行操作
    feature = []
    print(id)
    #获取当前id购买记录
    df_buy = df[ (df.buy == 1) & (df.id == id)]
    for row in df_buy.itertuples():
        df_login_all = df[(df.id == id ) & (df.buy == 0) & (df.time <= row[13])]
        df_login_10 = df[(df.id == id ) & (df.buy == 0) & (df.time <= row[13])].tail(10)
        df_login_6 = df[(df.id == id ) & (df.buy == 0) & (df.time <= row[13])].tail(6)
        df_login_3 = df[(df.id == id) & (df.buy == 0) & (df.time <= row[13])].tail(3)
        df_login_1 = df[(df.id == id) & (df.buy == 0) & (df.time <= row[13])].tail(1)
        #本次购买时间获取：
        buy_time = row[13].hour
        cnt_total = len(df_login_3)
        if cnt_total == 0:
            login_time = None #用众数填充
            is_scan_1 = 1
            log_from_1 = None #众数填充
            log_type_1 = None #众数填充
            log_result_1 = 1
            timelong_1 = None #平均数填充
            buy_login_period_1 = None #平均数填充
            city_rate_3 = None ; city_rate_6 = None  ; city_rate_10 = None #平均数填充
            device_rate_3 = None ;device_rate_6 = None ;device_rate_10 = None #平均数填充
            ip_rate_3 = None  ;ip_rate_6 = None  ;ip_rate_10 = None  # 平均数填充
            result_rate_3 = None;result_rate_6 = None;result_rate_10 = None #平均数填充
            device_chg_rate_3 = None ; device_chg_rate_6 = None ; device_chg_rate_10 = None #平均数填充
            city_chg_rate_3 = None;city_chg_rate_6 = None;city_chg_rate_10 = None  # 平均数填充
            ip_chg_rate_3 = None;ip_chg_rate_6 = None;ip_chg_rate_10 = None  # 平均数填充
            timelong_diff_avg_3 = None; timelong_diff_avg_6 = None; timelong_diff_avg_10 = None
            device_rate = None;ip_rate = None;city_rate = None
            device_bad_rate = None;ip_bad_rate = None;city_bad_rate=None
            weekday = 0;df_login_5_min = 0;df_login_30_min = 0
        elif cnt_total ==1:
            login_time = df_login_1.time.dt.hour.tolist()[0]
            is_scan_1 = df_login_1.is_scan.tolist()[0]
            log_from_1 = df_login_1.log_from.tolist()[0]
            log_type_1 = df_login_1.type.tolist()[0]
            log_result_1 = df_login_1.result.tolist()[0]
            timelong_1 = df_login_1.timelong.tolist()[0]
            buy_login_period_1 = int((row[13] - df_login_1.time.tolist()[0]).total_seconds())
            city_rate_3 = 1 ; city_rate_6 = 1  ; city_rate_10 = 1 #平均数填充
            device_rate_3 = 1 ;device_rate_6 = 1 ;device_rate_10 = 1 #平均数填充
            ip_rate_3 = 1  ;ip_rate_6 = 1  ;ip_rate_10 = 1  # 平均数填充
            result_rate_3 = 1;result_rate_6 = 1;result_rate_10 = 1 #平均数填充
            device_chg_rate_3 = 0 ; device_chg_rate_6 = 0 ; device_chg_rate_10 = 0 #平均数填充
            city_chg_rate_3 = 0;city_chg_rate_6 = 0;city_chg_rate_10 = 0  # 平均数填充
            ip_chg_rate_3 = 0;ip_chg_rate_6 = 0;ip_chg_rate_10 = 0  # 平均数填充
            timelong_diff_avg_3 = 0; timelong_diff_avg_6 = 0; timelong_diff_avg_10 = 0
            if df_login_1.time.tolist()[0].weekday() > 4:
                weekday = 1
            else:
                weekday = 0
            df_login_5_min = 0
            df_login_30_min = 0
            device_rate = device_dict[df_login_1.device.tolist()[0]]
            ip_rate = ip_dict[df_login_1.ip.tolist()[0]]
            city_rate = city_dict[df_login_1.city.tolist()[0]]
            device_bad_rate = device_bad_dict[df_login_1.device.tolist()[0]]
            ip_bad_rate = ip_bad_dict[df_login_1.ip.tolist()[0]]
            city_bad_rate = city_bad_dict[df_login_1.city.tolist()[0]]
        else:
            login_time = df_login_1.time.dt.hour.tolist()[0]
            is_scan_1 = df_login_1.is_scan.tolist()[0]
            log_from_1 = df_login_1.log_from.tolist()[0]
            log_type_1 = df_login_1.type.tolist()[0]
            log_result_1 = df_login_1.result.tolist()[0]
            timelong_1 = df_login_1.timelong.tolist()[0]
            buy_login_period_1 = int((row[13] - df_login_1.time.tolist()[0]).total_seconds())
            c3 = 0 ; i3 = 0 ; d3 = 0 ; r3 = 0
            for city_3 in df_login_3.city.tolist():
                if city_3 == df_login_1.city.tolist()[0]:
                    c3 +=1
            city_rate_3 = round(c3 / len(df_login_3), 2)
            for device_3 in df_login_3.device.tolist():
                if device_3 == df_login_1.device.tolist()[0]:
                    d3 +=1
            device_rate_3 = round(d3 / len(df_login_3), 2)
            for ip_3 in df_login_3.ip.tolist():
                if ip_3 == df_login_1.ip.tolist()[0]:
                    i3 +=1
            ip_rate_3 = round(i3 / len(df_login_3), 2)
            for result_3 in df_login_3.result.tolist():
                if (df_login_1.result.tolist()[0] == 1):
                    if (result_3 ==df_login_1.result.tolist()[0]):
                        r3 +=1
                elif (df_login_1.result.tolist()[0] != 1):
                    if (result_3 !=1):
                        r3 +=1
            result_rate_3 = round(r3 / len(df_login_3), 2)
            #------------------------------------------------
            c6 = 0 ; i6 = 0 ; d6 = 0 ; r6 = 0
            for city_6 in df_login_6.city.tolist():
                if city_6 == df_login_1.city.tolist()[0]:
                    c6 +=1
            city_rate_6 = round(c6 / len(df_login_6), 2)
            for device_6 in df_login_6.device.tolist():
                if device_6 == df_login_1.device.tolist()[0]:
                    d6 +=1
            device_rate_6 = round(d6 / len(df_login_6), 2)
            for ip_6 in df_login_6.ip.tolist():
                if ip_6 == df_login_1.ip.tolist()[0]:
                    i6 +=1
            ip_rate_6 = round(i6 / len(df_login_6), 2)
            for result_6 in df_login_6.result.tolist():
                if (df_login_1.result.tolist()[0] == 1):
                    if (result_6 ==df_login_1.result.tolist()[0]):
                        r6 +=1
                elif (df_login_1.result.tolist()[0] != 1):
                    if (result_6 !=1):
                        r6 +=1
            result_rate_6 = round(r6 / len(df_login_6), 2)
            #------------------------------------------------
            c10 = 0 ; i10 = 0 ; d10 = 0 ; r10 = 0
            for city_10 in df_login_10.city.tolist():
                if city_10 == df_login_1.city.tolist()[0]:
                    c10 +=1
            city_rate_10 = round(c10 / len(df_login_10), 2)
            for device_10 in df_login_10.device.tolist():
                if device_10 == df_login_1.device.tolist()[0]:
                    d10 +=1
            device_rate_10 = round(d10 / len(df_login_10), 2)
            for ip_10 in df_login_10.ip.tolist():
                if ip_10 == df_login_1.ip.tolist()[0]:
                    i10 +=1
            ip_rate_10 = round(i10 / len(df_login_10), 2)
            for result_10 in df_login_10.result.tolist():
                if (df_login_1.result.tolist()[0] == 1):
                    if (result_10 ==df_login_1.result.tolist()[0]):
                        r10 +=1
                elif (df_login_1.result.tolist()[0] != 1):
                    if (result_10 !=1):
                        r10 +=1
            result_rate_10 = round(r3 / len(df_login_10), 2)
            #-----------------------------------------------------
            device_chg_rate_3 = 0 ; device_chg_rate_6 = 0 ; device_chg_rate_10 = 0
            for dc3 in range(len(df_login_3) - 1):
                if df_login_3.device.tolist()[dc3+1] == df_login_3.device.tolist()[dc3]:
                    device_chg_rate_3 +=1
            device_chg_rate_3 = device_chg_rate_3/(len(df_login_3)-1)
            for dc6 in range(len(df_login_6) - 1):
                if df_login_6.device.tolist()[dc6+1] == df_login_6.device.tolist()[dc6]:
                    device_chg_rate_6 +=1
            device_chg_rate_6 = device_chg_rate_6/(len(df_login_6)-1)
            for dc10 in range(len(df_login_10) - 1):
                if df_login_10.device.tolist()[dc10+1] == df_login_10.device.tolist()[dc10]:
                    device_chg_rate_10 +=1
            device_chg_rate_10 = device_chg_rate_10/(len(df_login_10)-1)
            #-----------------------------------------------------------
            city_chg_rate_3 = 0;city_chg_rate_6 = 0;city_chg_rate_10 = 0
            for city3 in range(len(df_login_3) - 1):
                if df_login_3.city.tolist()[city3+1] == df_login_3.city.tolist()[city3]:
                    city_chg_rate_3 +=1
            city_chg_rate_3 = city_chg_rate_3/(len(df_login_3)-1)
            for city6 in range(len(df_login_6) - 1):
                if df_login_6.city.tolist()[city6+1] == df_login_6.city.tolist()[city6]:
                    city_chg_rate_6 +=1
            city_chg_rate_6 = city_chg_rate_6/(len(df_login_6)-1)
            for city10 in range(len(df_login_10) - 1):
                if df_login_10.city.tolist()[city10+1] == df_login_10.city.tolist()[city10]:
                    city_chg_rate_10 +=1
            city_chg_rate_10 = city_chg_rate_10/(len(df_login_10)-1)
            #----------------------------------------------------------
            ip_chg_rate_3 = 0;ip_chg_rate_6 = 0;ip_chg_rate_10 = 0
            for ip3 in range(len(df_login_3) - 1):
                if df_login_3.ip.tolist()[ip3+1] == df_login_3.ip.tolist()[ip3]:
                    ip_chg_rate_3 +=1
            ip_chg_rate_3 = ip_chg_rate_3/(len(df_login_3)-1)
            for ip6 in range(len(df_login_6) - 1):
                if df_login_6.ip.tolist()[ip6+1] == df_login_6.ip.tolist()[ip6]:
                    ip_chg_rate_6 +=1
            ip_chg_rate_6 = ip_chg_rate_6/(len(df_login_6)-1)
            for ip10 in range(len(df_login_10) - 1):
                if df_login_10.ip.tolist()[ip10+1] == df_login_10.ip.tolist()[ip10]:
                    ip_chg_rate_10 +=1
            ip_chg_rate_10 = ip_chg_rate_10/(len(df_login_10)-1)
            #----------------------------------------------------------
            timelong_diff_avg_3 = 0; timelong_diff_avg_6 = 0; timelong_diff_avg_10 = 0
            for timelong_3 in range(len(df_login_3)-1):
                timelong_diff_avg_3 = timelong_diff_avg_3 + (
                df_login_3.timelong.tolist()[timelong_3+1] - df_login_3.timelong.tolist()[timelong_3])
            timelong_diff_avg_3 = round(timelong_diff_avg_3/(len(df_login_3)-1),2)
            for timelong_6 in range(len(df_login_6) - 1):
                timelong_diff_avg_6 = timelong_diff_avg_6 + (
                df_login_6.timelong.tolist()[timelong_6 + 1] - df_login_6.timelong.tolist()[timelong_6])
            timelong_diff_avg_6 = round(timelong_diff_avg_6 / (len(df_login_6) - 1), 2)
            for timelong_10 in range(len(df_login_10) - 1):
                timelong_diff_avg_10 = timelong_diff_avg_10 + (
                df_login_10.timelong.tolist()[timelong_10 + 1] - df_login_10.timelong.tolist()[timelong_10])
            timelong_diff_avg_10 = round(timelong_diff_avg_10 / (len(df_login_10) - 1), 2)
            if df_login_1.time.tolist()[0].weekday()> 4:
                weekday = 1
            else:
                weekday = 0
            df_login_5_min = len(df_login_all[(df_login_1.time.tolist()[0] -df_login_all.time).dt.total_seconds()<300])
            df_login_30_min = len(df_login_all[(df_login_1.time.tolist()[0] -df_login_all.time).dt.total_seconds()<3600])
            device_rate = device_dict[df_login_1.device.tolist()[0]]
            ip_rate = ip_dict[df_login_1.ip.tolist()[0]]
            city_rate = city_dict[df_login_1.city.tolist()[0]]
            device_bad_rate = device_bad_dict[df_login_1.device.tolist()[0]]
            ip_bad_rate = ip_bad_dict[df_login_1.ip.tolist()[0]]
            city_bad_rate = city_bad_dict[df_login_1.city.tolist()[0]]
        rowkey = row[12]
        is_risk = row[7]
        is_train = row[-2]
        feature.append([rowkey,is_risk,is_train,buy_time,login_time,is_scan_1,log_from_1,log_type_1,
                        log_result_1,timelong_1,buy_login_period_1,city_rate_3,city_rate_6,
                        city_rate_10,device_rate_3,device_rate_6,device_rate_10,ip_rate_3,ip_rate_6,
                        ip_rate_10,result_rate_3,result_rate_6,result_rate_10,device_chg_rate_3,device_chg_rate_6,
                        device_chg_rate_10,city_chg_rate_3,city_chg_rate_6,city_chg_rate_10,
                        ip_chg_rate_3,ip_chg_rate_6,ip_chg_rate_10,timelong_diff_avg_3,timelong_diff_avg_6,
                        timelong_diff_avg_10,weekday,df_login_5_min,df_login_30_min,device_rate,ip_rate,city_rate,cnt_total,
                        device_bad_rate,ip_bad_rate,city_bad_rate])
    return feature

if __name__ == '__main__':
    pool = Pool(4)
    id_list = df.id.unique().tolist()
    data_list = pool.map(generate_feature, id_list)
    re = []
    for li in data_list:
         for lil in li:
             re.append(lil)
    df_feature = pd.DataFrame(re)
    df_feature.columns = ['rowkey','is_risk','is_train','buy_time','login_time','is_scan_1','log_from_1','log_type_1',
                          'log_result_1','timelong_1','buy_login_period_1','city_rate_3','city_rate_6','city_rate_10',
                          'device_rate_3','device_rate_6','device_rate_10','ip_rate_3','ip_rate_6','ip_rate_10',
                          'result_rate_3','result_rate_6','result_rate_10','device_chg_rate_3','device_chg_rate_6',
                          'device_chg_rate_10','city_chg_rate_3','city_chg_rate_6','city_chg_rate_10','ip_chg_rate_3',
                          'ip_chg_rate_6','ip_chg_rate_10','timelong_diff_avg_3','timelong_diff_avg_6',
                          'timelong_diff_avg_10','weekday','df_login_5_min','df_login_30_min','device_rate','ip_rate',
                          'city_rate','cnt_total','device_bad_rate','ip_bad_rate','city_bad_rate']
    df_feature.timelong_1 = df_feature.timelong_1/(df_feature.timelong_1.max()-df_feature.timelong_1.min())
    df_feature.buy_login_period_1 = df_feature.buy_login_period_1 / (df_feature.buy_login_period_1.max() - df_feature.buy_login_period_1.min())
    df_feature.timelong_diff_avg_3 = df_feature.timelong_diff_avg_3 / (
    df_feature.timelong_diff_avg_3.max() - df_feature.timelong_diff_avg_3.min())
    df_feature.timelong_diff_avg_6 = df_feature.timelong_diff_avg_6 / (
        df_feature.timelong_diff_avg_6.max() - df_feature.timelong_diff_avg_6.min())
    df_feature.timelong_diff_avg_10 = df_feature.timelong_diff_avg_10 / (
        df_feature.timelong_diff_avg_10.max() - df_feature.timelong_diff_avg_10.min())
    bins = [-1, 7, 9, 12, 14, 18, 19, 24]
    group_name = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    buy_time_box = pd.cut(df_feature['buy_time'], bins, labels=group_name)
    login_time_box = pd.cut(df_feature['login_time'], bins, labels=group_name)
    df_feature['buy_time_box'] = buy_time_box
    df_feature['login_time_box'] = login_time_box
    buy_time_one_hot = pd.get_dummies(df_feature['buy_time_box'], prefix='buy_time_box')
    login_time_one_hot = pd.get_dummies(df_feature['login_time_box'], prefix='login_time_box')
    last_login_from_one_hot = pd.get_dummies(df_feature['log_from_1'], prefix='log_from')
    last_login_result_one_hot = pd.get_dummies(df_feature['log_result_1'], prefix='result')
    last_login_type_one_hote = pd.get_dummies(df_feature['log_type_1'], prefix='type')
    feature_all_one_hot = pd.concat([df_feature, buy_time_one_hot, login_time_one_hot, last_login_from_one_hot,
                                     last_login_result_one_hot,last_login_type_one_hote], axis=1, join_axes=[df_feature.index])

    feature_all_one_hot_no_record = feature_all_one_hot[feature_all_one_hot.login_time.isnull()]
    feature_all_one_hot_no_record.to_csv('feature_all_one_hot_no_record.csv')

    feature_all_one_hot_multi_record = feature_all_one_hot[feature_all_one_hot.cnt_total > 1]
    feature_all_one_hot_multi_record.to_csv('feature_all_one_hot_multi_record.csv')

    feature_all_one_hot_one_record = feature_all_one_hot[feature_all_one_hot.cnt_total == 1]
    feature_all_one_hot_one_record.to_csv('feature_all_one_hot_one_record.csv')





