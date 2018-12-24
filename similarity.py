import numpy
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import jieba
import jieba.analyse
import pymysql
from gensim.models import word2vec
import re
from datetime import datetime, timedelta
import jieba.posseg as psg
import locale
import math
# 显示无向图
import networkx
import matplotlib.pyplot as plt
import numpy

from jpype import *


# =======================计算内容相似度==========================
# 去除文本中的无意义字符
def cleanSen(str):
    str = str.strip(' ')
    str = str.replace('\\t', '')
    str = str.replace('\\n', '')
    str = str.replace('\\r', '')
    str = str.replace('\\u3000', '')
    return str


def extractKeywords(data):
    tfidf = jieba.analyse.extract_tags
    keywords = tfidf(data)
    return keywords


# 获取关键词
def getKeywords(news, savePath):
    with open(savePath, 'w', encoding='UTF-8-SIG') as outf:
        keywords = extractKeywords(news)
        for word in keywords:
            outf.write(word + ' ')
        outf.write('\n')


# 关键词向量化
def w2v(keyword, model):
    wordVec = numpy.zeros(50)
    for data in keyword:
        # print(data)
        data = data.split()
        first_word = data[0]
        if model.__contains__(first_word):
            wordVec = wordVec + model[first_word]
        for i in range(len(data) - 1):
            word = data[i + 1]
            # print(word)
            if model.__contains__(word):
                wordVec = wordVec + model[word]
    return wordVec


# 计算两个文本的相似度
def similarity(vec1, vec2):
    vec1Mod = numpy.sqrt(sum(vec1**2))
    vec2Mod = numpy.sqrt(sum(vec2**2))
    if vec1Mod != 0 and vec2Mod != 0:
        similarity = (numpy.sum(vec1 * vec2)) / (vec1Mod * vec2Mod)
    else:
        similarity = 0
    return similarity


# 两个新闻文本的相似度
def similarityOfText(news1, news2):
    # 去除文档多余字符
    news1 = cleanSen(str(news1))
    news2 = cleanSen(str(news2))
    # 获取关键词
    news1Keyword = extractKeywords(news1)
    news2Keyword = extractKeywords(news2)
    # news1Keyword = './simResult/news1Keyword.txt'
    # news2Keyword = './simResult/news2Keyword.txt'
    # 用 word2vec 进行训练
    sentences = word2vec.Text8Corpus(u'./segmentResult/tmp.txt')
    # sentences为训练语料 min_count小于该数的单词会被剔除，默认值为5 windows为神经网络隐藏层单元数，默认100
    model = word2vec.Word2Vec(sentences, min_count=3, size=50, window=5, workers=4)
    # 关键词向量化
    news1Vec = w2v(news1Keyword, model)
    news2Vec = w2v(news2Keyword, model)
    # 相似度计算
    simOfText = similarity(news1Vec, news2Vec)
    return simOfText


# =======================计算时间相似度==========================
UTIL_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}
UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def cn2dig(src):
    if src == "":
        return None
    m = re.match("\d+", src)
    if m:
        return int(m.group(0))
    rsl = 0
    unit = 1
    for item in src[::-1]:
        if item in UTIL_CN_UNIT.keys():
            unit = UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM.keys():
            num = UTIL_CN_NUM[item]
            rsl += num * unit
        else:
            return None
    if rsl < unit:
        rsl += unit
    return rsl


def year2dig(year):
    res = ''
    for item in year:
        if item in UTIL_CN_NUM.keys():
            res = res + str(UTIL_CN_NUM[item])
        else:
            res = res + item
    m = re.match("\d+", res)
    if m:
        if len(m.group(0)) == 2:
            return int(datetime.datetime.today().year / 100) * 100 + int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None


# 利用jieba分词 识别出分词结果中的数字和日期词汇 只提取到年月日
def time_extract(text):
    locale.setlocale(locale.LC_CTYPE, 'chinese')
    time_res = []
    word = ''
    keyDate = {'今天': 0, '明天': 1, '后天': 2}  # keyDate = {k:v}
    for k, v in psg.cut(text.strip()):
        if k in keyDate:
            if word != '':
                time_res.append(word)
            word = (datetime.today() + timedelta(days=keyDate.get(k, 0))).strftime('%Y年%m月%d日')
        elif word != '':
            if v in ['m', 't']:
                word = word + k
            else:
                time_res.append(word)
                word = ''
        elif v in ['m', 't']:
            word = k
    if word != '':
        time_res.append(word)
    result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
    final_res = [parse_datetime(w) for w in result]
    return [x for x in final_res if x is not None]


# 将日期中统一替换成“日”
def check_time_valid(word):
    if '年' in word or '月' in word or '日' in word:
        m = re.match("\d+$", word)
        if m:
            if len(word) <= 6:
                return None
        word1 = re.sub('[号|日]\d+$', '日', word)
        if word1 != word:
            return check_time_valid(word1)
        else:
            return word1


# 通过正则表达式对日期进行切分，分为具体维度再对具体维度进行识别
def parse_datetime(msg):
    if msg is None or len(msg) == 0:
        return None
    # try:
    #     dt = parse(msg, fuzzy=True)
    #     return dt.strftime('%Y-%m-%d')
    # except Exception as e:
    else:
        m = re.match(
            r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?",
            msg)
        if m.group(0) is not None:
            res = {
                "year": m.group(1),
                "month": m.group(2),
                "day": m.group(3),
            }
            params = {}

            for name in res:
                if res[name] is not None and len(res[name]) != 0:
                    if name == 'year':
                        tmp = year2dig(res[name][:-1])
                    else:
                        tmp = cn2dig(res[name][:-1])
                    if tmp is not None:
                        params[name] = int(tmp)
            target_date = datetime.today().replace(**params)
            return target_date.strftime('%Y-%m-%d')
        else:
            return None


# 将获取的时间转换成datatime形式
def shiftDatatime(time):
    time = check_time_valid(time)
    time = parse_datetime(time)
    return time


# 计算时间的相似度
def timeSimilarity(time1, time2, H):
    # 将str转化成datatime再进行运算
    time1 = datetime.strptime(time1, '%Y-%m-%d')
    time2 = datetime.strptime(time2, '%Y-%m-%d')
    diff = time1 - time2
    simOfTime = -(math.log((abs(diff.days) + 1) / H) / math.log(H))
    return simOfTime


# 两个新闻时间的相似度
def similarityOfTime(time1, time2, H):
    # 通过正则表达式抽取文本中的时间
    # print(news1, time_extract(news1), sep=':')
    # print(news2, time_extract(news2), sep=':')

    # 将获取的时间转换成datatime形式
    time1 = shiftDatatime(time1)
    # print(time1)
    time2 = shiftDatatime(time2)
    # print(time2)
    # 计算时间的相似度
    simOfTime = timeSimilarity(time1, time2, H)
    return simOfTime


# ==========================计算实体相似度======================
# 地名识别，标注为ns
def Place_Recognize(sentence_str):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    segment = HanLP.newSegment().enablePlaceRecognize(True)
    return HanLP.segment(sentence_str)


# 人名识别,标注为nr
def PersonName_Recognize(sentence_str):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    segment = HanLP.newSegment().enableNameRecognize(True)
    return HanLP.segment(sentence_str)


# 机构名识别,标注为nt
def Organization_Recognize(sentence_str):
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    segment = HanLP.newSegment().enableOrganizationRecognize(True)
    return HanLP.segment(sentence_str)


# 标注结果转化成列表
def total_result(function_result_input):
    x = str(function_result_input)
    y = x[1:len(x) - 1]
    y = y.split(',')
    return y


# Type_Recognition 可以选 ‘place’,‘person’,‘organization’三种实体,
# 返回单一实体类别的列表
def single_result(Type_Recognition, total_result):
    if Type_Recognition == 'place':
        Type = '/ns'
    elif Type_Recognition == 'person':
        Type = '/nr'
    elif Type_Recognition == 'organization':
        Type = '/nt'
    else:
        print('请输入正确的参数：（place，person或organization）')
    z = []
    for i in range(len(total_result)):
        if total_result[i][-3:] == Type:
            z.append(total_result[i])
    return z


# 把单一实体结果汇总成一个字典
def dict_result(sentence):
    a = total_result(Place_Recognize(sentence))
    b = single_result('place', a)
    c = total_result(PersonName_Recognize(sentence))
    d = single_result('person', c)
    e = total_result(Organization_Recognize(sentence))
    f = single_result('organization', e)
    # g = total_result(NLP_tokenizer(sentence))
    # h = time_result(g)
    total_list = [i.strip() for i in b] + [i.strip() for i in d] + [i.strip() for i in f]
    # print('一维数组形式：', total_list)
    return total_list


# 除去重复实体
def deleteRedundant(entityList):
    # total_list = {}.fromkeys(entityList).keys()
    total_list = list(set(entityList))
    return total_list


# 求两个新闻实体的相似度
def entitySimilarity(entityOfNews1, entityOfNews2):
    # 求两个数组的交集
    intersection = list(set(entityOfNews1).intersection(set(entityOfNews2)))
    # print('实体交集', intersection)
    # 求两个数组的并集
    union = list(set(entityOfNews1).union(set(entityOfNews2)))
    # print('实体并集', union)
    simOfEntity = len(intersection) / len(union)
    return simOfEntity


# 两个新闻的实体相似度
def similarityOfEntity(news1, news2):
    # startJVM(getDefaultJVMPath(), "-Djava.class.path=D:\hanlp\hanlp-portable-1.7.0.jar;D:\hanlp", "-Xms1g", "-Xmx1g")

    # 获取新闻的实体列表，以一维数组的形式返回
    entityOfNews1 = dict_result(news1)
    entityOfNews2 = dict_result(news2)
    # 关闭JVM虚拟机
    # shutdownJVM()
    # 除去重复实体
    entityOfNews1 = deleteRedundant(entityOfNews1)
    # print('数组去除重复实体：', entityOfNews1)
    entityOfNews2 = deleteRedundant(entityOfNews2)
    # print('数组去除重复实体：', entityOfNews2)
    # 求两个新闻实体的相似度
    simOfEntity = entitySimilarity(entityOfNews1, entityOfNews2)
    return simOfEntity


# ========================新闻相似度==========================
def similarityOfNews(news1, news2, time1, time2, H):
    # 两个新闻文本相似度
    simOfText = similarityOfText(news1, news2)
    print('文本相似度：', simOfText)
    # 两个新闻时间的相似度
    simOfTime = similarityOfTime(time1, time2, H)
    print('时间相似度：', simOfTime)
    # 两个新闻的实体相似度
    simOfEntity = similarityOfEntity(news1, news2)
    print('实体相似度：', simOfEntity)
    # 事件的相似度
    simeos = simOfText * simOfTime * simOfEntity
    print('事件相似度：', simeos)
    return simeos


# 显示无向图
def graphShow(G, adjacencyMatrix):
    matrix = numpy.array(adjacencyMatrix)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            if matrix[i][j] == 1:
                G.add_edge(i, j)
    networkx.draw(G)
    plt.show()
    return


# Tanimoto系数(广义jaccard相似度)
def Tanimoto(vec1, vec2):
    vec1Mod = numpy.sqrt(sum(vec1**2))
    vec2Mod = numpy.sqrt(sum(vec2**2))
    similarity = (vec1.dot(vec2)) / (vec1Mod + vec2Mod - vec1.dot(vec2))
    return similarity


# jaccard相似度
def jaccard(news1, news2):
    # 去除文档多余字符
    news1 = cleanSen(str(news1))
    news2 = cleanSen(str(news2))
    # 获取关键词
    news1Keyword = extractKeywords(news1)
    news2Keyword = extractKeywords(news2)
    # 用 word2vec 进行训练
    sentences = word2vec.Text8Corpus(u'./segmentResult/tmp.txt')
    # sentences为训练语料 min_count小于该数的单词会被剔除，默认值为5 windows为神经网络隐藏层单元数，默认100
    model = word2vec.Word2Vec(sentences, min_count=3, size=50, window=5, workers=4)
    # 关键词向量化
    news1Vec = w2v(news1Keyword, model)
    news2Vec = w2v(news2Keyword, model)
    # 广义Jaccard相似度计算
    simOfText = Tanimoto(news1Vec, news2Vec)
    return simOfText


# 抛光图的过程
def polishGraph(G, textOfRelatedNews, P):
    VNeighbor = []
    for v in G.nodes():
        # 求每个点的邻近点N(v)
        NV = G.neighbors(v)
        # 将每个点的邻近点存入数组
        # 迭代器遍历，只前进不后退
        for i in NV:
            VNeighbor.append(i)

        for u in VNeighbor:
            minJaccard = 0
            maxJaccard = 0
            UNeighbor = []
            # 求每个邻近点的邻近点
            NU = G.neighbors(u)
            # 将每个邻近点的邻近点存储到数组中
            # 迭代器遍历，只前进不后退
            for i in NU:
                UNeighbor.append(i)
            # 计算公式f(u, v)
            # 计算ANeighbor与BNeighbor的交集
            intersection = list(set(VNeighbor).intersection(set(UNeighbor)))
            if len(intersection) != 0:
                # 计算ANeighbor与BNeighbor的并集
                union = list(set(VNeighbor).union(set(UNeighbor)))
                for n in union:
                    # max = 0
                    # 求max{J(u, n), J(v, n)}
                    # J(A, B)为广义Jaccard相似度 EJ(A,B)=(A*B)/(||A||^2+||B||^2-A*B)
                    jaccardUN = jaccard(textOfRelatedNews[u], textOfRelatedNews[n])
                    jaccardVN = jaccard(textOfRelatedNews[v], textOfRelatedNews[n])
                    maxJaccard += max(jaccardUN, jaccardVN)
                    if n in intersection:
                        minJaccard += min(jaccardUN, jaccardVN)
                print('~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('min：', minJaccard)
                print('max：', maxJaccard)
                fuv = minJaccard/maxJaccard
                print('f(u,v):', fuv)
                # 若f(u,v)大于阈值则添加uv之间的边，否则删除uv之间的边
                if fuv >= -1:
                    P[u][v] = 1
    print('polish over')
    return


# ========================主函数==========================
def main():
    # connect mysql
    # conn = pymysql.connect(
    #     host='localhost',  # mysql服务器地址
    #     user='root',  # 用户名
    #     passwd='root',  # 密码
    #     db='renminnews',  # 数据库
    #     charset='utf8'
    # )
    # # 获取游标
    # cursor = conn.cursor()
    # # 从数据库获取两篇新闻
    # sql1 = "SELECT content FROM news WHERE title='意向苏赔款一万万美元'"
    # sql2 = "SELECT content FROM news WHERE title='苏联钢铁生产超过计划'"
    # try:
    #     cursor.execute(sql1)
    #     news1 = cursor.fetchall()
    #     cursor.execute(sql2)
    #     news2 = cursor.fetchall()
    # except:
    #     print("select is failed")
    # cursor.close()
    # conn.close()

    # 从testData文件中读取新闻
    newsList = []
    file = open("./test/testData.txt", encoding='UTF-8-SIG')
    for line in file:
        newsList.append(line)

    # 从testTime文件中读取对应新闻的时间
    timeList = []
    file = open("./test/testTime.txt", encoding='UTF-8-SIG')
    for line in file:
        timeList.append(line)

    # 给定的新闻new1
    news1 = "强烈地震后十几分钟，大地还在抖动，刚刚脱险的开滦唐山煤矿工会副主任李玉林和他的战友们，带着唐山市几十万人民的迫切心情，驾驶着一辆矿山救护车，朝北京方向开去，去报告党中央，报告毛主席。他们都有家，此刻，谁都没有想到自己的家，谁都没有顾及自己的家。究竟是什么精神力量指挥着他们这一英勇的行动？李玉林激情满怀地回答说，“抗灾全靠党指引，红心向着毛主席！”"
    time1 = "1976年8月10日"
    H = 34

    # 设 𝜃1为0,大于𝜃1则认为是与给定的news1相关的新闻
    textOfRelatedNews = []
    timeOfRelatedNews = []

    startJVM(getDefaultJVMPath(), "-Djava.class.path=D:\hanlp\hanlp-portable-1.7.0.jar;D:\hanlp", "-Xms1g", "-Xmx1g")
    for i in range(0, len(newsList)):
        news2 = newsList[i].replace("\n", "")
        print(news2)
        time2 = timeList[i].replace("\n", "")
        print(time2)
        simeos = similarityOfNews(news1, news2, time1, time2, H)
        # 设 𝜃1为0,大于𝜃1则认为是与给定的news1相关的新闻,相关新闻集合C
        if simeos > 0.0:
            textOfRelatedNews.append(news2)
            timeOfRelatedNews.append(time2)
    print('=====================================')
    # 集合C的新闻相互计算相似度，设 𝜃2为0.1，大于𝜃2构成图
    G = networkx.Graph()
    # 邻接矩阵存储无向图
    adjacencyMatrix = [[0] * len(textOfRelatedNews) for x in range(len(textOfRelatedNews))]
    # 存储抛光后的图
    P = [[0] * len(textOfRelatedNews) for x in range(len(textOfRelatedNews))]
    for i in range(0, len(textOfRelatedNews)):
        news1 = textOfRelatedNews[i]
        time1 = timeOfRelatedNews[i]
        for j in range(i + 1, len(textOfRelatedNews)):
            news2 = textOfRelatedNews[j]
            time2 = timeOfRelatedNews[j]
            simeos = similarityOfNews(news1, news2, time1, time2, H)
            print('(', i, j, ')=', simeos)
            if simeos > 0.1:
                adjacencyMatrix[i][j] = 1
                adjacencyMatrix[j][i] = 1
    shutdownJVM()
    # 显示无向图
    graphShow(G, adjacencyMatrix)
    # 加权微聚类算法进行无向图抛光
    polishGraph(G, textOfRelatedNews, P)
    # 显示抛光后的无向图
    graphShow(G, P)
    #


if __name__ == "__main__":
    main()
