//数位dp模板    HDU2089
//求[n,m]中不含“4”和“62”的数的个数
#include<bits/stdc++.h>
using namespace std;
#define Accepted 0
int dist[10];
int dp[10][2];


/*
dp数组一定要初始化为-1
dp数组一定要初始化为-1
dp数组一定要初始化为-1
重要的事情说三遍
*/


//pre代表前一个数字是否为6，flag代表当前位是否有限制（是否以dist[len]结尾）
int dfs(int len,int pre,int flag)
{
    if(len<0)
        return 1;
    //如果当前询问的值已经被记忆化，直接返回
    if(!flag&&dp[len][pre]!=-1)
        return dp[len][pre];
    //判断当前位结尾
    int ed=flag?dist[len]:9;
    int ans=0;
    for(int i=0;i<=ed;i++)
    {
        if(i!=4&&!(pre&&i==2))
        {
            //如果当前位的end没有限制，那么递归下去的所有位都没有限制
            ans+=dfs(len-1,i==6,flag&&i==ed);
        }
    }
    //记忆化
    if(!flag)
    {
        dp[len][pre]=ans;
    }
    return ans;
}
//这个地方solve(0)也是有值的
//solve求的是[0,n]中满足题目要求的数的个数
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
        //这里一定要初始化为-1
        memset(dp,-1,sizeof(dp));
        if(!n&&!m)
            break;
        cout<<solve(m)-solve(n-1)<<endl;
    }
    return Accepted;
}


//数位dp可以做求[l,r]中 不包含某些子串的数的个数
//也可以不能求包含某些子串的数的个数
//如果要求，可约容斥原理+数位dp   如HDU3555

//或者参考如下HDU3652
//求[1,n]中包含13且%13==0的数的个数
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
