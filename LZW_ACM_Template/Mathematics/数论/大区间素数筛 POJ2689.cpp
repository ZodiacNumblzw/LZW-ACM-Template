//区间素数筛选       POJ2689
#include<iostream>
#include<cstdlib>
#include<cmath>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<vector>
#include<string>
#include<list>
#include<stack>
#include<queue>
#include<deque>
#include<map>
#include<set>
#include<bitset>
#include<utility>
#include<iomanip>
#include<climits>
#include<complex>
#include<cassert>
#include<functional>
#include<numeric>
#define Accepted 0
typedef long long ll;
typedef long double ld;
const int mod=1e9+7;
const int maxn=1e5+10;
const int maxm=1e6+10;
const double pi=acos(-1);
const double eps=1e-6;
const int INF=0x3f3f3f3f;
using namespace std;
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
ll lowbit(ll x){return x&(-x);}
int vis[maxn],prime[maxn],tol;
void liner_shai(int n)
{
    memset(vis,0,sizeof(vis));
    for(int i=2;i<n;i++)
    {
        if(!vis[i])
            prime[tol++]=i;
        for(int j=0;j<tol&&i*prime[j]<n;j++)
        {
            vis[i*prime[j]]=1;
            if(i%prime[j]==0)
            {
                break;
            }
        }
    }
}
int a[maxm];
void solve(int l,int r)
{
    if(l==1)
        a[0]=1;
    vector<int> v;
    for(int i=0;i<tol;i++)
    {
        for(ll j=max(((l-1)/prime[i]+1)*(ll)prime[i],2*(ll)prime[i]);j<=r;j+=prime[i])
        {
            a[j-l]=1;
        }
    }
    for(int i=0;i<=r-l;i++)
    {
        //cout<<a[i]<<endl;
        if(!a[i])
        {
            v.push_back(i);
        }
        a[i]=0;
    }
    int len=v.size();
    if(len<=1)
    {
        cout<<"There are no adjacent primes."<<endl;
        return;
    }
    else
    {
        int mi=INT_MAX,mi1,mi2,mm=0,mm1,mm2;
        for(int i=1;i<len;i++)
        {
            if(v[i]-v[i-1]>mm)
            {
                mm1=v[i-1],mm2=v[i],mm=v[i]-v[i-1];
            }
            if(v[i]-v[i-1]<mi)
            {
                mi1=v[i-1],mi2=v[i],mi=v[i]-v[i-1];
            }
        }
        cout<<mi1+l<<','<<mi2+l<<" are closest, "<<mm1+l<<','<<mm2+l<<" are most distant."<<endl;
    }
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    int inf=sqrt(INT_MAX)+1;
    liner_shai(inf);
    int l,r;
    while(cin>>l>>r)
    {
        solve(l,r);
    }
    return Accepted;
}
