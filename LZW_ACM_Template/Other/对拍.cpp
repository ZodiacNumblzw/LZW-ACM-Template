//.bat����      ��data.exe,biaoda.exe,mycode.exe��.bat�ļ�����һ���ļ��к�ִ��.bat�ļ�

:again
data > input.txt
biaoda < input.txt > biaoda_output.txt
mycode < input.txt > mycode_output.txt
fc biaoda_output.txt test_output.txt
if not errorlevel 1 goto again
pause

//.cpp����

#include<iostream>
#include<windows.h>
using namespace std;
int main()
{
    //int t=200;
    while(1)
    {
//      t--;
        system("data.exe > data.txt");
        system("biaoda.exe < data.txt > biaoda.txt");
         system("test.exe < data.txt > test.txt");
        if(system("fc test.txt biaoda.txt"))   break;
    }
    if(t==0) cout<<"no error"<<endl;
    else cout<<"error"<<endl;
    //system("pause");
    return 0;
}
//https://blog.csdn.net/code12hour/article/details/51252457