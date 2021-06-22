//Lagrange插值公式：
//拉格朗日插值还可以求出多项式f的前缀和f(0)+f(1)+...+f(n)的值
//F(0)=0,F(1)=f(0),F(2)=f(0)+f(1),...,F(n+1)=f(0)+f...f(n)
//只需要把多项式的前缀和作为点的值插值就行
//洛谷上大佬写的可以直接插出系数的板子
#include<cstdio>
#include<cstring>
#include<algorithm>
#define O 1ll
using namespace std;
const int N=2005,mod=998244353;
int n,k,x[N],y[N],num[N],tmp[N],res[N],inv[N];
void Add(int &x,int y)
{
    x+=y;
    if(x>=mod) x-=mod;
}
void exGCD(int a,int b,int &x,int &y)
{
    if(!b) x=1,y=0;
    else exGCD(b,a%b,y,x),y-=a/b*x;
}
int Inv(int x)
{
    int xx,yy;
    exGCD(x,mod,xx,yy);
    Add(xx,mod);
    return xx;
}
void Lagrange()
{
    for(int i=1; i<=n; i++)
    {
        int den=1,lst=0;
        for(int j=1; j<=n; j++)
            if(i!=j) den=O*den*(x[i]-x[j]+mod)%mod;
        den=O*y[i]*Inv(den)%mod;
        for(int j=0; j<n; j++)
        {
            tmp[j]=O*(num[j]-lst+mod)*inv[i]%mod;
            Add(res[j],O*den*tmp[j]%mod),lst=tmp[j];
        }
    }
}
void Pre()
{
    num[0]=1;
    for(int i=1; i<=n; swap(num,tmp),i++)
    {
        tmp[0]=0;
        inv[i]=Inv(mod-x[i]);
        for(int j=1; j<=i; j++) tmp[j]=num[j-1];
        for(int j=0; j<=i; j++) Add(tmp[j],O*num[j]*(mod-x[i])%mod);
    }
}
int Calc(int x)
{
    int ret=0,var=1;
    for(int i=0; i<n; var=O*var*x%mod,i++)
        Add(ret,O*var*res[i]%mod);
    return ret;
}
int main()
{
    scanf("%d%d",&n,&k);
    for(int i=1; i<=n; i++) //输入你要插值的n个点（n-1次多项式）
        scanf("%d%d",&x[i],&y[i]);
    Pre();                   //预处理
    Lagrange();
    printf("%d",Calc(k));       //输出多项式在k处的值
    return 0;
}







//dls的Lagrange板子
//这个板子是输入多项式的系数可以得到前缀和和单点值的方法
#include <bits/stdc++.h>
#define endl '\n'
#define ll long long
#define ull unsigned long long
#define fi first
#define se second
#define mp make_pair
#define pii pair<int,int>
#define ull unsigned long long
#define all(x) x.begin(),x.end()
#pragma GCC optimize("unroll-loops")
#define inline inline __attribute__(  \
(always_inline, __gnu_inline__, __artificial__)) \
__attribute__((optimize("Ofast"))) __attribute__((target("sse"))) \
__attribute__((target("sse2"))) __attribute__((target("mmx")))
#define IO ios::sync_with_stdio(false);
#define rep(ii,a,b) for(int ii=a;ii<=b;++ii)
#define per(ii,a,b) for(int ii=b;ii>=a;--ii)
#define for_node(x,i) for(int i=head[x];i;i=e[i].next)
#define show(x) cout<<#x<<"="<<x<<endl
#define showa(a,b) cout<<#a<<'['<<b<<"]="baidu<a[b]<<endl
#define show2(x,y) cout<<#x<<"="<<x<<" "<<#y<<"="<<y<<endl
#define show3(x,y,z) cout<<#x<<"="<<x<<" "<<#y<<"="<<y<<" "<<#z<<"="<<z<<endl
#define show4(w,x,y,z) cout<<#w<<"="<<w<<" "<<#x<<"="<<x<<" "<<#y<<"="<<y<<" "<<#z<<"="<<z<<endl
using namespace std;
const int maxn=1e6+10,maxm=2e6+10;
const int INF=0x3f3f3f3f;
const ll mod=1e9+7;
const double PI=acos(-1.0);
//head
int casn,n,m,k;
int num[maxn];
ll a[maxn];
ll pow_mod(ll a,ll b,ll c=mod,ll ans=1){while(b){if(b&1) ans=(a*ans)%c;a=(a*a)%c,b>>=1;}return ans;}

namespace polysum {
    const int maxn=101000;
    const ll mod=1e9+7;
    ll a[maxn],f[maxn],g[maxn],p[maxn],p1[maxn],p2[maxn],b[maxn],h[maxn][2],C[maxn];
    ll calcn(int d,ll *a,ll n) {//d次多项式(a[0-d])求第n项            求f(n)
        if (n<=d) return a[n];
        p1[0]=p2[0]=1;
        rep(i,0,d) {
            ll t=(n-i+mod)%mod;
            p1[i+1]=p1[i]*t%mod;
        }
        rep(i,0,d) {
            ll t=(n-d+i+mod)%mod;
            p2[i+1]=p2[i]*t%mod;
        }
        ll ans=0;
        rep(i,0,d) {
            ll t=g[i]*g[d-i]%mod*p1[i]%mod*p2[d-i]%mod*a[i]%mod;
            if ((d-i)&1) ans=(ans-t+mod)%mod;
            else ans=(ans+t)%mod;
        }
        return ans;
    }
    void init(int maxm) {//初始化预处理阶乘和逆元(取模乘法)
        f[0]=f[1]=g[0]=g[1]=1;
        rep(i,2,maxm+4) f[i]=f[i-1]*i%mod;
        g[maxm+4]=pow_mod(f[maxm+4],mod-2);
        per(i,1,maxm+3) g[i]=g[i+1]*(i+1)%mod;
    }
    ll polysum(ll n,ll *a,ll m) { // a[0].. a[m] \sum_{i=0}^{n-1} a[i]
        // m次多项式求第n项前缀和
        a[m+1]=calcn(m,a,m+1);                          //不包括n
        rep(i,1,m+1) a[i]=(a[i-1]+a[i])%mod;
        return calcn(m+1,a,n-1);
    }
    ll qpolysum(ll R,ll n,ll *a,ll m) { // a[0].. a[m] \sum_{i=0}^{n-1} a[i]*R^i
        if (R==1) return polysum(n,a,m);
        a[m+1]=calcn(m,a,m+1);
        ll r=pow_mod(R,mod-2),p3=0,p4=0,c,ans;
        h[0][0]=0;
        h[0][1]=1;
        rep(i,1,m+1) {
            h[i][0]=(h[i-1][0]+a[i-1])*r%mod;
            h[i][1]=h[i-1][1]*r%mod;
        }
        rep(i,0,m+1) {
            ll t=g[i]*g[m+1-i]%mod;
            if (i&1) p3=((p3-h[i][0]*t)%mod+mod)%mod,p4=((p4-h[i][1]*t)%mod+mod)%mod;
            else p3=(p3+h[i][0]*t)%mod,p4=(p4+h[i][1]*t)%mod;
        }
        c=pow_mod(p4,mod-2)*(mod-p3)%mod;
        rep(i,0,m+1) h[i][0]=(h[i][0]+h[i][1]*c)%mod;
        rep(i,0,m+1) C[i]=h[i][0];
        ans=(calcn(m,C,n)*pow_mod(R,n)-c)%mod;
        if (ans<0) ans+=mod;
        return ans;
    }
}


int main() {
//#define test
#ifdef test
    auto _start = chrono::high_resolution_clock::now();
    freopen("in.txt","r",stdin);freopen("out.txt","w",stdout);
#endif
    IO;
    ll n,r,k;
    cin>>n>>r>>k;
    polysum::init(k+5);
    rep(i,0,2010) a[i]=pow_mod(i,k);
    ll ans=polysum::qpolysum(r,n+1,a,k+1);
    if(k==0) ans=(ans-1+mod)%mod;
    cout<<ans<<endl;
#ifdef test
    auto _end = chrono::high_resolution_clock::now();
  cerr << "elapsed time: " << chrono::duration<double, milli>(_end - _start).count() << " ms\n";
    fclose(stdin);fclose(stdout);system("out.txt");
#endif
    return 0;
}