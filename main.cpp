
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "NaiveBayes.h"
#include "timer.h"

using namespace std;

int main()
{
	ifstream in("mushroom.txt");

	NaiveBayes n;

	vector<vector<char>> datas;

	Timer T;
	T.begin();
	const size_t resId = 4;
	//load data
	while (!in.eof())
	{
		string str;
		in >> str;
		if (str.size() == 0)
			break;
		vector<char> data;
		for (auto i = 0; i != 22; ++i)
		{
			data.push_back(str[2 * i]);
		}
		datas.push_back(data);
		n.loadSingleData(data, resId);
	}

	n.generateParameter();

	size_t right = 0;
	size_t total = datas.size();

	for (auto &d : datas)
	{
		if (n.predict(d, 0) == d[resId])
			++right;
	}

	T.end();

	cout << "total:" << total << endl;
	cout << "right:" << right << endl;
	cout << " rate:" << 100.0*right / total << '%' << endl;
	cout << " time:" << T.time() << 's' << endl;

	return 0;
}
