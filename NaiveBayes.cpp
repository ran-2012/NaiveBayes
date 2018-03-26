
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
		//���ڵ�һ�����ݣ����������ĸ���
		if (rawData.empty())
		{
			parameterSize = data.size();
			valueSpace.resize(parameterSize);
			resId = id;
		}
		//������������ȷ
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
		//�������п��ܵ�ȡֵ��Ϊ��һ��������׼��
		for (const auto& data : rawData)
		{
			for (auto i = 0; i != data.size(); ++i)
			{
				//��ǰ��ȡֵ�ռ�
				auto& curMap = valueSpace[i];
				if (curMap.find(data[i]) == curMap.end())
					//���벢��������ֵ
					curMap.insert(make_pair(data[i], curMap.size()));
				//���ɽ��ֵ�ķֲ�
				if (i == resId)
				{
					if (resValueDistr.find(data[i]) == resValueDistr.end())
						resValueDistr.insert(make_pair(data[i], 1));
					else
						++resValueDistr[data[i]];
				}
			}
		}
		//���ɽ��ID��ֵ��ӳ��
		for (auto it = valueSpace[resId].begin(); it != valueSpace[resId].end(); ++it)
		{
			resIdMapValue.insert(make_pair(it->second, it->first));
		}
		//����ռ�
		conditionCounter.resize(parameterSize);
		for (auto i = 0; i != conditionCounter.size(); ++i)
		{
			auto &v = conditionCounter[i];
			v.resize(valueSpace[i].size());
			for (auto j = 0; j != v.size(); ++j)
				v[j].resize(valueSpace[0].size());
		}
		//�����������ʼ�������
		for (const auto& data : rawData)
		{
			//���ֵID
			auto curResValId = valueSpace[resId][data[resId]];
			for (auto i = 0; i != parameterSize; ++i)
			{
				if (i == resId)
					continue;
				//����ֵID
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
	//ÿ�ֽ���ĸ���ֵ
	vector<float> probability;
	probability.resize(resValueDistr.size());
	for (auto i = 0; i != resValueDistr.size(); ++i)
	{
		char curResValue = resIdMapValue[i];
		float temp = 1.0;
		//�򵥵ĸ��ʼ���
		for (auto j = 0; j != parameterSize; ++j)
		{
			if (j == resId)
				continue;
			//d,p,c�����conditionCounterע��
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
	//ȡ���������
	size_t retId = distance(probability.begin(), 
		max_element(probability.begin(), probability.end()));
	char ret=resIdMapValue[retId];
	return ret;
}


NaiveBayes::~NaiveBayes()
{
}
