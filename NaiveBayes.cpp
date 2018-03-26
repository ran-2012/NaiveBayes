
#include "NaiveBayes.h"
#include <exception>
#include <iostream>
#include <algorithm>

using namespace std;

NaiveBayes::NaiveBayes():parameterGenerated(false)
{
}

bool NaiveBayes::loadSingleData(vector<char> data, size_t id)
{
	try
	{
		//对于第一个数据，决定参数的个数
		if (rawData.empty())
		{
			parameterSize = data.size();
			valueSpace.resize(parameterSize);
			resId = id;
		}
		//参数个数不正确
		if (parameterSize != data.size())
		{
			throw exception("Wrong lenth of parameter");
		}
		rawData.push_back(data);
	}
	catch (exception e)
	{
		cerr << e.what() << endl;
		return false;
	}
	return true;
}

bool NaiveBayes::generateParameter()
{
	try
	{
		//分析所有可能的取值，为下一步处理做准备
		for (const auto& data : rawData)
		{
			for (auto i = 0; i != data.size(); ++i)
			{
				//当前的取值空间
				auto& curMap = valueSpace[i];
				if (curMap.find(data[i]) == curMap.end())
					//插入并按个数赋值
					curMap.insert(make_pair(data[i], curMap.size()));
				//生成结果值的分布
				if (i == resId)
				{
					if (resValueDistr.find(data[i]) == resValueDistr.end())
						resValueDistr.insert(make_pair(data[i], 1));
					else
						++resValueDistr[data[i]];
				}
			}
		}
		//生成结果ID到值的映射
		for (auto it = valueSpace[resId].begin(); it != valueSpace[resId].end(); ++it)
		{
			resIdMapValue.insert(make_pair(it->second, it->first));
		}
		//分配空间
		conditionCounter.resize(parameterSize);
		for (auto i = 0; i != conditionCounter.size(); ++i)
		{
			auto &v = conditionCounter[i];
			v.resize(valueSpace[i].size());
			for (auto j = 0; j != v.size(); ++j)
				v[j].resize(valueSpace[0].size());
		}
		//生成条件概率计数数组
		for (const auto& data : rawData)
		{
			//结果值ID
			auto curResValId = valueSpace[resId][data[resId]];
			for (auto i = 0; i != parameterSize; ++i)
			{
				if (i == resId)
					continue;
				//参数值ID
				auto curParaValId = valueSpace[i][data[i]];
				++conditionCounter[i][curParaValId][curResValId];
			}
		}
	}
	catch (exception e)
	{
		cerr << e.what() << std::endl;
		return false;
	}
	return false;
}

char NaiveBayes::predict(vector<char> data, float lb)
{
	//每种结果的概率值
	vector<float> probability;
	probability.resize(resValueDistr.size());
	for (auto i = 0; i != resValueDistr.size(); ++i)
	{
		char curResValue = resIdMapValue[i];
		float temp = 1.0;
		//简单的概率计算
		for (auto j = 0; j != parameterSize; ++j)
		{
			if (j == resId)
				continue;
			//d,p,c定义见conditionCounter注释
			auto d = j;
			auto p = valueSpace[j][data[j]];
			auto c = i;
			float P = conditionCounter[d][p][c] + lb;
			P /= resValueDistr[curResValue] 
				+ lb * valueSpace[j].size();
			temp *= P;
		}
		probability[i] = temp;
	}
	//取概率最大者
	size_t retId = distance(probability.begin(), 
		max_element(probability.begin(), probability.end()));
	char ret=resIdMapValue[retId];
	return ret;
}


NaiveBayes::~NaiveBayes()
{
}
