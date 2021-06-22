//��λdpģ��    HDU2089
//��[n,m]�в�����4���͡�62�������ĸ���
#include<bits/stdc++.h>
using namespace std;
#define Accepted 0
int dist[10];
int dp[10][2];


/*
dp����һ��Ҫ��ʼ��Ϊ-1
dp����һ��Ҫ��ʼ��Ϊ-1
dp����һ��Ҫ��ʼ��Ϊ-1
��Ҫ������˵����
*/


//pre����ǰһ�������Ƿ�Ϊ6��flag����ǰλ�Ƿ������ƣ��Ƿ���dist[len]��β��
int dfs(int len,int pre,int flag)
{
    if(len<0)
        return 1;
    //�����ǰѯ�ʵ�ֵ�Ѿ������仯��ֱ�ӷ���
    if(!flag&&dp[len][pre]!=-1)
        return dp[len][pre];
    //�жϵ�ǰλ��β
    int ed=flag?dist[len]:9;
    int ans=0;
    for(int i=0;i<=ed;i++)
    {
        if(i!=4&&!(pre&&i==2))
        {
            //�����ǰλ��endû�����ƣ���ô�ݹ���ȥ������λ��û������
            ans+=dfs(len-1,i==6,flag&&i==ed);
        }
    }
    //���仯
    if(!flag)
    {
        dp[len][pre]=ans;
    }
    return ans;
}
//����ط�solve(0)Ҳ����ֵ��
//solve�����[0,n]��������ĿҪ������ĸ���
int solve(int n)
{
    int len=0;
    while(n)
    {
        dist[len++]=n%10;
        n/=10;
    }
    return dfs(len-1,0,1);
}
signed main()
{
    ios::sync_with_stdio(false);
    int n,m;
    while(cin>>n>>m)
    {
        //����һ��Ҫ��ʼ��Ϊ-1
        memset(dp,-1,sizeof(dp));
        if(!n&&!m)
            break;
        cout<<solve(m)-solve(n-1)<<endl;
    }
    return Accepted;
}


//��λdp��������[l,r]�� ������ĳЩ�Ӵ������ĸ���
//Ҳ���Բ��������ĳЩ�Ӵ������ĸ���
//���Ҫ�󣬿�Լ�ݳ�ԭ��+��λdp   ��HDU3555

//���߲ο�����HDU3652
//��[1,n]�а���13��%13==0�����ĸ���
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
#define Accepted 0
#define IO ios::sync_with_stdio(false);cin.tie(0);cout.tie(0);
const int maxn=1e5+10;
const double eps=1e-6;
const int INF=0x3f3f3f3f;
const double pi=acos(-1);
const int mod=1e9+7;
inline ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
inline ll quick(ll a,ll b,ll m){ll sum=1;while(b){if(b&1)sum=(sum*a)%m;a=(a*a)%m;b>>=1;}return sum;}
int dp[50][10][20];
int dist[50];
int dfs(int len,int pre,int md,int flag)
{
    if(len<0)
    {
        return pre==2&&md==0;
    }
    if(!flag&&dp[len][pre][md]!=-1)
    {
        return dp[len][pre][md];
    }
    int ed=flag?dist[len]:9;
    int ans=0;
    for(int i=0;i<=ed;i++)
    {
        if(pre==2||(pre==1&&i==3))
        {
            ans+=dfs(len-1,2,(md*10+i)%13,flag&&i==ed);
        }
        else if(i==1)
        {
            ans+=dfs(len-1,1,(md*10+i)%13,flag&&i==ed);
        }
        else
        {
            ans+=dfs(len-1,0,(md*10+i)%13,flag&&i==ed);
        }
        
    }
    if(!flag)
    {
        dp[len][pre][md]=ans;
    }
    return ans;
}
int solve(int n)
{
    int len=0;
    while(n)
    {
        dist[len++]=n%10;
        n/=10;
    }
    return dfs(len-1,0,0,1);
}
signed main()
{
    IO
    int n;
    while(cin>>n)
    {
        memset(dp,-1,sizeof(dp));
        cout<<solve(n)<<endl;
    }
    return 0;
}
