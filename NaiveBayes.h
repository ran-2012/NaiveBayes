#pragma once

#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <unordered_map>

using namespace std;

class NaiveBayes
{
	//原始数据
	vector<vector<char>> rawData;
	//每一个参数的取值空间
	vector<unordered_map<char, size_t>> valueSpace;
	//结果值的分布
	unordered_map<char, size_t> resValueDistr;
	//结果ID到值的映射
	unordered_map<size_t, char> resIdMapValue;
	//条件概率数( conter[d][p][c]=P(X_d=p|y=c) )
	vector<vector<vector<size_t>>> conditionCounter;
	//单条数据参数数量
	size_t parameterSize;
	//结果在单条数据中的ID
	size_t resId;
	//是否已经生成参数
	bool parameterGenerated;
public:
	NaiveBayes();
	//加载一条数据，默认第一个为预测值
	bool loadSingleData(vector<char>, size_t);
	//数据预处理
	bool generateParameter();
	//预测结果
	char predict(vector<char>, float lb = 1.0);
	~NaiveBayes();
};

