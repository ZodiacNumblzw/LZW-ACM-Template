//ĸ����  ����ĸ����ͨ�����ת��Ϊ����ʽ�˷� ������FFT�Ż�
#include<bits/stdc++.h>
using namespace std;
#define maxn 305
int dp[maxn];
const int a[30]={1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289};

int main()
{
    int n;

    for(int i=0;i<17;i++)
    {
        dp[a[i]]++;
        for(int j=1;j+a[i]<maxn&&j<maxn;j++)
        {
            dp[j+a[i]]+=dp[j];
        }
    }
    while(cin>>n)
    {
        if(!n)
            break;
        cout<<dp[n]<<endl;
    }
    return 0;
}

#include<bits/stdc++.h>
using namespace std;
int a[125];
int b[125];
int main()
{
    ios::sync_with_stdio(false);
    int n;
    while(cin>>n)
    {
        memset(a,0,sizeof(a));
        memset(b,0,sizeof(b));
        a[0]=1;
        for(int i=1;i<=n;i++)     //i���Ͻ�Ϊ�ж��ٸ�����ʽ���
        {
            for(int j=0;j<=n;j++)  //j���Ͻ�Ϊ�����Ͻ磬��Ŀǰ�Ķ���ʽ�ж�����
            {
                for(int k=0;k+j<=n;k+=i) //k+j<=n����˼��������ʽ��ˣ�ֻ�ÿ����Ͻ����µ���
                {                        //kÿ�μ�i�����i������ʽÿ��������Ĵ�����
                    b[j+k]+=a[j];
                   // cout<<j+k<<' '<<b[j+k]<<endl;
                }
            }
            for(int j=0;j<=n;j++)
                a[j]=b[j];
            memset(b,0,sizeof(b));
        }
        cout<<a[n]<<endl;
    }
    return 0;
}