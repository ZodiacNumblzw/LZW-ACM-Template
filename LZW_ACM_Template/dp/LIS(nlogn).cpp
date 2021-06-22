//O(nlogn)的LIS
int low[maxn];
int n,ans;
int binary_search(int *a,int r,int x)
{
    int l=1,mid;
    while(l<=r)
    {
        mid=(l+r)>>1;
        if(a[mid]<=x)
            l=mid+1;
        else
            r=mid-1;
    }
    return l;
}
int a[maxn];
int LIS(int tol)
{
    for(int i=1;i<=tol;i++)
    {
        low[i]=INF;
    }
    low[1]=a[1];
    ans=1;
    for(int i=2;i<=tol;i++)
    {
        if(a[i]>=low[ans])
            low[++ans]=a[i];
        else
            low[binary_search(low,ans,a[i])]=a[i];
    }
    cout<<ans<<endl;
}

//或者如下用lower_bound
#include<cstdio>
#include<algorithm>
const int MAXN=200001;
 
int a[MAXN];
int d[MAXN];
 
int main()
{
    int n;
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
        scanf("%d",&a[i]);
    d[1]=a[1];
    int len=1;
    for(int i=2;i<=n;i++)
    {
        if(a[i]>d[len])
            d[++len]=a[i];
        else
        {
            int j=std::lower_bound(d+1,d+len+1,a[i])-d;
            d[j]=a[i];
        }
    }
    printf("%d\n",len);   
    return 0;
}


//或者如下更简洁
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cstring>
typedef long long ll;
using namespace std;
const int maxn=40009;
const int INF=0x3f3f3f3f;
int a[maxn];
int dp[maxn];
int main()
{
    int n;
    while(scanf("%d",&n) && n!=-1)
    {
        //输入
        for(int i=0;i<n;i++)
            scanf("%d",&a[i]);
        //nlogn 的最长升序子序列的解法
        memset(dp,INF,sizeof(dp));
        for(int i=0;i<n;i++)
        {
            *lower_bound(dp,dp+n,a[i])=a[i];
        }
        printf("%d\n",lower_bound(dp,dp+n,INF)-dp);
    }
    return 0;
}