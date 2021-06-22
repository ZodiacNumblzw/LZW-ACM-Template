//LCS O(nlogn) 适用范围：序列中每一个元素都不相同
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
#include<ext/rope>
using namespace std;
using namespace __gnu_cxx;
using namespace __gnu_pbds;
#define ll long long
#define ld long double
#define ull unsigned long long
#define rep(i,a,b) for(int i=a;i<b;i++)
#define Rep(i,a,b) for(int i=a;i<=b;i++)
#define per(i,a,b) for(int i=b-1;i>=a;i--)
#define Per(i,a,b) for(int i=b;i>=a;i--)
#define pb push_back
#define eb emplace_back
#define MP make_pair
#define fi first
#define se second
#define SZ(x) (x).size()
#define LEN(x) (x).length()
#define ALL(X) (X).begin(), (X).end()
#define MS0(X) memset((X), 0, sizeof((X)))
#define MS1(X) memset((X), -1, sizeof((X)))
#define MS(X,a) memset((X),a,sizeof(X))
#define CASET int ___T; scanf("%d", &___T); for(int cs=1;cs<=___T;cs++)
#define Read(x) scanf("%d",&x)
#define ReadD(x) scanf("%lf",&x)
#define ReadLL(x) scanf("%lld",&x)
#define ReadLD(x) scanf("%llf",&x)
#define Write(x) printf("%d\n",x)
#define WriteD(x) printf("%f\n",x)
#define WriteLL(x) printf("%lld\n",x)
#define WriteLD(x) printf("%Lf\n",x)
#define IO ios::sync_with_stdio(false); cin.tie(0); cout.tie(0);
constexpr ld pi=acos(-1);
constexpr ll mod=1e9+7;
#define lowbit(x) (x&(-x))
constexpr ld eps=1e-6;
constexpr int maxn=1e5+10;
constexpr int INF=0x3f3f3f3f;
constexpr double e=2.718281828459045;
typedef long long LL;
typedef unsigned long long ULL;
typedef long double LD;
typedef pair<int,int> PII;
typedef vector<int> VI;
typedef vector<LL> VL;
typedef vector<PII> VPII;
typedef pair<LL,LL> PLL;
typedef vector<PLL> VPLL;
typedef vector<int> VI;
typedef pair<int,int> PII;
#define Accepted 0
inline ll quick(ll a,ll b,ll m){ll ans=1;while(b){if(b&1)ans=(a*ans)%m;a=(a*a)%m;b>>=1;}return ans;}
inline ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
struct node
{
    int a,b;
}a[maxn];
bool cmp(node a,node b)
{
    return a.a<b.a;
}
int turn[maxn];
int d[maxn];
signed main()
{
    int n;
    Read(n);
    Rep(i,1,n)
    {
        Read(a[i].a);
    }
    Rep(i,1,n)
    {
        Read(a[i].b);
    }
    Rep(i,1,n)
    {
        turn[a[i].a]=i;
    }
    Rep(i,1,n)
    {
        a[i].b=turn[a[i].b];
    }
    d[1]=a[1].b;
    int len=1;
    for(int i=2;i<=n;i++)
    {
        if(a[i].b>d[len])
        {
            d[++len]=a[i].b;
        }
        else
        {
            int id=upper_bound(d+1,d+len+1,a[i].b)-d;
            d[id]=a[i].b;
        }
    }
    Write(len);
    return Accepted;
}