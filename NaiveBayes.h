#pragma once

#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <unordered_map>

using namespace std;

class NaiveBayes
{
	//ԭʼ����
	vector<vector<char>> rawData;
	//ÿһ��������ȡֵ�ռ�
	vector<unordered_map<char, size_t>> valueSpace;
	//���ֵ�ķֲ�
	unordered_map<char, size_t> resValueDistr;
	//���ID��ֵ��ӳ��
	unordered_map<size_t, char> resIdMapValue;
	//����������( conter[d][p][c]=P(X_d=p|y=c) )
	vector<vector<vector<size_t>>> conditionCounter;
	//�������ݲ�������
	size_t parameterSize;
	//����ڵ��������е�ID
	size_t resId;
	//�Ƿ��Ѿ����ɲ���
	bool parameterGenerated;
public:
	NaiveBayes();
	//����һ�����ݣ�Ĭ�ϵ�һ��ΪԤ��ֵ
	bool loadSingleData(vector<char>, size_t);
	//����Ԥ����
	bool generateParameter();
	//Ԥ����
	char predict(vector<char>, float lb = 1.0);
	~NaiveBayes();
};

