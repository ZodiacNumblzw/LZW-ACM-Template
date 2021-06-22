//Min_25筛               Loj6053
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
using namespace std;
#define ll long long
#define MAX 222222
#define MOD 1000000007
ll n,Sqr,w[MAX];
ll prime[MAX],id1[MAX],id2[MAX],h[MAX],g[MAX],m;
bool vis[MAX];
int tot,sp[MAX];
void pre(int n)          //线性筛预处理 筛sqrt(n)内的素数
{
    vis[1]=true;
    for(int i=2; i<=n; ++i)
    {
        if(!vis[i])
        {
            prime[++tot]=i;
            sp[tot]=(sp[tot-1]+i)%MOD;           //素数前缀和
            //次数可添加要求的前缀和
        }
        for(int j=1; j<=tot&&i*prime[j]<=n; ++j)
        {
            vis[i*prime[j]]=true;
            if(i%prime[j]==0)
                break;
        }
    }
}
int S(ll x,int y)
{
    if(x<=1||prime[y]>x)return 0;
    int k=(x<=Sqr)?id1[x]:id2[n/x],ret=(g[k]-sp[y-1]-h[k]+y-1)%MOD;//此处修改
    if(y==1)ret+=2;
    for(int i=y; i<=tot&&1ll*prime[i]*prime[i]<=x; ++i)
    {
        ll t1=prime[i],t2=1ll*prime[i]*prime[i];    //此处修改
        for(int e=1; t2<=x; ++e,t1=t2,t2*=prime[i])
        {
            (ret+=((1ll*S(x/t1,i+1)*(prime[i]^e)%MOD+(prime[i]^(e+1))%MOD)))%=MOD;  //此处根据公式修改 根据f的表达式
        }
    }
    return ret;
}
int main()
{
    scanf("%lld",&n);
    Sqr=sqrt(n);
    pre(Sqr);
    for(ll i=1,j; i<=n; i=j+1)           //整除分块
    {
        j=n/(n/i);
        w[++m]=n/i;
        h[m]=(w[m]-1)%MOD;
        g[m]=(w[m]%MOD)*((w[m]+1)%MOD)%MOD;
        if(g[m]&1)g[m]=g[m]+MOD;
        g[m]/=2;
        g[m]--;
        if(w[m]<=Sqr)id1[w[m]]=m;
        else id2[j]=m;
    }
    for(int j=1; j<=tot; ++j)
    {
        for(int i=1; i<=m&&prime[j]*prime[j]<=w[i]; ++i)
        {
            int k=(w[i]/prime[j]<=Sqr)?id1[w[i]/prime[j]]:id2[n/(w[i]/prime[j])];
            (g[i]-=1ll*prime[j]*(g[k]-sp[j-1])%MOD)%=MOD;
            (h[i]-=h[k]-j+1)%=MOD;//此处修改
        }
    }
    int ans=S(n,1)+1;
    printf("%d\n",(ans+MOD)%MOD);
    return 0;
}







//Min_25筛 by:yyb          待修改
#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<algorithm>
using namespace std;
#define ll long long
#define MAX 222222
#define MOD 1000000007
ll n,Sqr,w[MAX];
ll pri[MAX],id1[MAX],id2[MAX],h[MAX],g[MAX],m;
bool zs[MAX];
int tot,sp[MAX];
void pre(int n)                 //预处理出质数
{
    zs[1]=true;
    for(int i=2; i<=n; ++i)
    {
        if(!zs[i])pri[++tot]=i,sp[tot]=(sp[tot-1]+i)%MOD;       //sp为质数的前缀和
        for(int j=1; j<=tot&&i*pri[j]<=n; ++j)
        {
            zs[i*pri[j]]=true;
            if(i%pri[j]==0)break;
        }
    }
}
int S(ll x,int y)                //S(n,j)=g(n,|P|)-\sum{i=1}^{j-1}(f(i))+\sum{k>j&&p_k^(e+1)<=n}^{}(f(p_k^e)*S(n/(p_k^e),k+1)+f(p_k^(e+1)))
{
    if(x<=1||pri[y]>x)return 0;
    int k=(x<=Sqr)?id1[x]:id2[n/x],ret=(g[k]-sp[y-1]-h[k]+y-1)%MOD;   //初始化g(x,|P|)-\sum{i=1}^{y-1}(f(i))
    if(y==1)ret+=2;
    for(int i=y; i<=tot&&1ll*pri[i]*pri[i]<=x; ++i)
    {
        ll t1=pri[i],t2=1ll*pri[i]*pri[i];
        for(int e=1; t2<=x; ++e,t1=t2,t2*=pri[i])
            (ret+=((1ll*S(x/t1,i+1)*(pri[i]^e)%MOD+(pri[i]^(e+1))%MOD)))%=MOD;       //(f(p_k^e)*S(n/(p_k^e),k+1)+f(p_k^(e+1)))
    }
    return ret;
}
int main()
{
    scanf("%lld",&n);
    Sqr=sqrt(n);
    pre(Sqr);
    for(ll i=1,j; i<=n; i=j+1)                     //以下前缀和均为除1以外的前缀和
    {
        j=n/(n/i);
        w[++m]=n/i;                                   //积性函数f(p)由几个多项式生成就需要几个预处理的函数
                                                      //比如f(p)=p^3-p^2+p-1就需要预处理i^3 i^2 i 1这四个前缀和
        h[m]=(w[m]-1)%MOD;                            //此处h需要维护素数个数的前缀和，初始化为所有数的前缀和
                                                      //（把所有数都当做质数初始化）
        g[m]=(w[m]%MOD)*((w[m]+1)%MOD)%MOD;           //此处g维护f(p)的前缀和还是将所有数当做质数
                                                      //（因此初始化为所有数的前缀和）
        if(g[m]&1)g[m]=g[m]+MOD;
        g[m]/=2;
        g[m]--;
        if(w[m]<=Sqr)id1[w[m]]=m;         //用来存小于sqrt(n)整除分块的序号 w[m]<sqrt(n)
        else id2[j]=m;                   //用来存w[n/m] 减少数组内存为sqrt(n)数量级
    }                                    //我们需要求出每个g(n/i,|P|)的值，因为在递归求S中只需要这些位置的值

                                        //接下来利用g(n/l,0)来递推g(n/l,|P|);
    for(int j=1; j<=tot; ++j)
        for(int i=1; i<=m&&pri[j]*pri[j]<=w[i]; ++i)
        {
            int k=(w[i]/pri[j]<=Sqr)?id1[w[i]/pri[j]]:id2[n/(w[i]/pri[j])];     //找到k在哪个块 即[n/(l*prime[i])]被存在哪个g/h里面
            (g[i]-=1ll*pri[j]*(g[k]-sp[j-1])%MOD)%=MOD;   //g(n,j)=g(n-1,j)-f(p_j)(g(n/p_j,j-1)-sum{i=1}^{j-1}(f(pi)))
            (h[i]-=h[k]-j+1)%=MOD;
        }
    int ans=S(n,1)+1;         //最后加上f(1)
    printf("%d\n",(ans+MOD)%MOD);
    return 0;
}


//Min_25筛的预处理
int n,val[N*2],id1[N],id2[N];
//主函数（预处理）
int sqrt_n=sqrt(n),tot=0;
for(int i=1,j;i<=n;i=j+1) {
    j=n/(n/i);
    int w=n/i;val[++tot]=w;
    if(w<=sqrt_n) id1[w]=tot;
    else id2[n/w]=tot;
}
//查询某个m的编号
inline int get_id(int m) {
    if(m<=sqrt_n) return id1[m];
    else return id2[n/m];
}




//洛谷上筛phi和mu的板子
//P4213 min_25 
#include <bits/stdc++.h>
#define ll long long
#define pb push_back
#define mk make_pair
#define pii pair<int,int>
#define fst first
#define scd second
using namespace std;
/*  -----  by:duyi  -----  */
const int N=5e4;
const int MAXN=N*2+5;
int n,sn,cnt,tot,p[MAXN],sum[MAXN],id1[MAXN],id2[MAXN],val[MAXN];
ll g1[MAXN],g2[MAXN];
bool v[MAXN];
void sieve() {
    v[1]=1;
    for(int i=2;i<=N;++i) {
        if(!v[i]) p[++cnt]=i;
        for(int j=1;j<=cnt && (ll)i*p[j]<=N;++j) {
            v[i*p[j]]=1;
            if(i%p[j]==0) break;
        }
    }
    for(int i=1;i<=cnt;++i) sum[i]=sum[i-1]+p[i];
}
inline int get_id(int x) {
    if(x<=sn) return id1[x];
    else return id2[n/x];
}
ll S_phi(int x,int y) {
    if(x<=1 || p[y]>x) return 0;
    ll res=g1[get_id(x)]-g2[get_id(x)]-(sum[y-1]-(y-1));
    for(int i=y;i<=cnt && (ll)p[i]*p[i]<=x;++i) {
        ll pre=1,cur=p[i];
        for(int j=1;cur*p[i]<=x;++j) {
            res+=pre*(p[i]-1LL)*S_phi(x/cur,i+1)+cur*(p[i]-1LL);
            pre=cur;cur=cur*p[i];
        }
    }
    return res;
}
ll S_mu(int x,int y) {
    if(x<=1 || p[y]>x) return 0;
    ll res=-g2[get_id(x)]+y-1;
    for(int i=y;i<=cnt && (ll)p[i]*p[i]<=x;++i) {
        res+=(-S_mu(x/p[i],i+1));
    }
    return res;
}
void solve() {
    cin>>n;
    sn=sqrt(n);tot=0;
    for(int i=1,j;i<=n;i=j+1) {
        j=n/(n/i);int w=n/i;
        val[++tot]=w;
        if(w<=sn) id1[w]=tot;
        else id2[n/w]=tot;
        g1[tot]=(ll)w*(w+1LL)/2LL-1LL;
        g2[tot]=w-1;
    }
    for(int i=1;i<=cnt;++i) {
        for(int j=1;j<=tot && (ll)p[i]*p[i]<=val[j];++j) {
            int t=get_id(val[j]/p[i]);
            g1[j]-=(ll)p[i]*(g1[t]-sum[i-1]);
            g2[j]-=(g2[t]-(i-1));
        }
    }
    cout<<S_phi(n,1)+1LL<<" "<<S_mu(n,1)+1LL<<endl;
}
int main() {
    ios::sync_with_stdio(0);/*syn加速*/
    sieve();
    int T;cin>>T;while(T--)solve();
    return 0;
}









//Min_25    自己改的板子
struct Min_25
{
    ll n,Sqr,w[maxn];
    ll prime[maxn],id1[maxn],id2[maxn],h[maxn],g[maxn],m;
    bool vis[maxn];
    int tot,sp[maxn],sp2[maxn];
    void pre(int n)         //预处理出质数
    {                       //举例f=p^2-p
    	tot=0;              //则需要筛出p和p^2的素数处的前缀和
        memset(vis,0,sizeof(vis));   
        vis[1]=true;
        for(int i=2; i<=n; i++)
        {
            if(!vis[i])
                prime[++tot]=i,sp[tot]=(sp[tot-1]+i)%mod,sp2[tot]=(sp2[tot-1]+1ll*i*i%mod)%mod;       //sp为质数的前缀和 ,sp2为素数平方处的前缀和
            for(int j=1; j<=tot&&i*prime[j]<=n; j++)
            {
                vis[i*prime[j]]=true;
                if(i%prime[j]==0)
                    break;
            }
        }
    }
    void init(ll n)
    {
    	m=0;
        for(ll l=1,r; l<=n; l=r+1)
        {
            r=n/(n/l);
            w[++m]=n/l;
            //积性函数f(p)由几个多项式生成就需要几个预处理的函数
            //比如f(p)=p^3-p^2+p-1就需要预处理i^3 i^2 i 1这四个前缀和
            //此处h需要维护素数个数的前缀和，初始化为所有数的前缀和
            h[m]=(w[m]%mod)*((w[m]+1)%mod)%mod*inverse(2,mod)%mod;
            //举例f=p^2-p     h处理2,3,..n的和,g处理2^2,3^2,...n^2的前缀和
            h[m]--;
            //（把所有数都当做质数初始化）           h(x)=f1(2)+f1(3)+...+f1(n)
            g[m]=((w[m]%mod)*((w[m]+1)%mod)%mod)%mod*((2*w[m]+1)%mod)%mod*inverse(6,mod)%mod;
            g[m]--;
            //此处g维护f(p)的前缀和  还是将所有数当做质数
            //（因此初始化为所有数的前缀和）               g(x)=f2(2)+...+f2(x)

            if(w[m]<=Sqr)id1[w[m]]=m;         //用来存小于sqrt(n)整除分块的序号 w[m]<sqrt(n)
            else id2[r]=m;                   //用来存w[n/m] 减少数组内存为sqrt(n)数量级
        }
        //我们需要求出每个g(n/i,|P|)的值，因为在递归求S中只需要这些位置的值

        //接下来利用g(n/l,0)来递推g(n/l,|P|);
        for(int j=1; j<=tot; j++)
        {
            for(int i=1; i<=m&&prime[j]*prime[j]<=w[i]; i++)
            {
                int k=(w[i]/prime[j]<=Sqr)?id1[w[i]/prime[j]]:id2[n/(w[i]/prime[j])];
                //找到k在哪个块 即[n/(l*prime[i])]被存在哪个g/h里面
                g[i]=(g[i]-1ll*(prime[j]*prime[j]%mod)*((g[k]-sp2[j-1])%mod)%mod)%mod;
                //g(n,j)=g(n,j-1)-f(p_j)(g(n/p_j,j-1)-sum{i=1}^{j-1}(f(pi)))
                //此处分别f=p^2,f=p;
                h[i]=(h[i]-1ll*prime[j]*(h[k]-sp[j-1])%mod)%mod;
            }
        }
    }
    ll S(ll x,int y)
    {//S(n,j)=g(n,|P|)-\sum{i=1}^{j-1}(f(i))+\sum{k>j&&p_k^(e+1)<=n}^{}(f(p_k^e)*S(n/(p_k^e),k+1)+f(p_k^(e+1)))
        if(x<=1||prime[y]>x)return 0;
        ll k=(x<=Sqr)?id1[x]:id2[n/x],ret=((g[k]-h[k])%mod-((sp2[y-1]-sp[y-1])%mod))%mod;
                                     //初始化g(x,|P|)-\sum{i=1}^{y-1}(f(i))
        for(int i=y; i<=tot&&1ll*prime[i]*prime[i]<=x; ++i)
        {
            ll t1=prime[i],t2=1ll*prime[i]*prime[i];
            //t1->p^e  t2->p^(e+1)
            for(int e=1; t2<=x; ++e,t1=t2,t2*=prime[i])
            {
                ret=(ret+((1ll*S(x/t1,i+1)*((t1%mod)*((t1-1)%mod)%mod)%mod+((t2%mod)*((t2-1)%mod)%mod)%mod)))%mod;
                //(f(p_k^e)*S(n/(p_k^e),k+1)+f(p_k^(e+1)))
            }
        }
        return ret;
    }
    ll query(ll nn)
    {
        n=nn;
        memset(id1,0,sizeof(id1));
        memset(id2,0,sizeof(id2));
        Sqr=sqrt(nn);
        pre(Sqr);
        init(nn);
        ll ans=S(nn,1)+1;         //最后加上f(1)
        if(ans<0)
            ans+=mod;
        return ans;
    }
}M;