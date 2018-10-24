#include "vipl_detector.h"
#include <QApplication>
#include <string>
#include <fstream> 
#include <istream>
#include <iostream>
using namespace std;
string line_1, line_2, line_3;



int main(int argc, char *argv[])
{
	ifstream in("data.txt");


	if (in) // 有该文件  
	{
		if (!getline(in, line_1))
			return 0;
		if (!getline(in, line_2))
			return 0;
		if (!getline(in, line_3))
			return 0;
	}
	else // 没有该文件  
	{
		//cout << "no such txt_file" << endl;
	}
	in.close();
	QApplication a(argc, argv);
	VIPL_Detector w;
	w.move(0, 0);
	w.show();
	
	#ifdef _ConsoleDisappear
	#pragma comment( linker, "/subsystem:/"windows/" /entry:/"wmainCRTStartup/"")
	#else
	#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" )
	#endif
	

	return a.exec();

}
