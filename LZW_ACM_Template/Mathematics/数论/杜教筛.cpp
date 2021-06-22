//杜教筛求phi和mu的前缀和 O(n^(2/3)) 洛谷P4213 注意long long乘法次数过多会被卡常

#include<bits/stdc++.h>
using namespace std;
#define ll long long
const int maxn = 5e6 + 10;
struct djs
{
    bool vis[maxn];
    int prime[maxn],mu[maxn];
    ll phi[maxn];
    unordered_map<int, int> mm;
    unordered_map<int, ll> pp;
    void init(int n)
    {
        memset(vis,0,sizeof(vis));
        int cnt=0;
        mu[1]=phi[1] = 1;
        for (int i=2; i<= n; i++)
        {
            if (!vis[i])
                prime[++cnt] = i, mu[i] = -1, phi[i] = i - 1;
            for (int j = 1; j <= cnt && prime[j] * i <= n; j++)
            {
                vis[i * prime[j]] = 1;
                if (i % prime[j] == 0)
                {
                    phi[i * prime[j]] =phi[i] * prime[j];
                    break;
                }
                else
                {
                    mu[i * prime[j]]=-mu[i];
                    phi[i * prime[j]]=phi[i] * (prime[j] - 1);
                }
            }
        }
        for (int i = 1; i <= n; i++)
            mu[i]+=mu[i-1],phi[i]+=phi[i-1];
    }
    djs()
    {
        init(maxn-5);
        mm.clear();
        pp.clear();
    }
    int pre_mu(int n)
    {
        if (n <= maxn - 5) return mu[n];
        if (mm[n]) return mm[n];

        int l = 2, r, ans = 1;
        for(l=2; l<=n; l=r+1)
        {
            r=n/(n/l);
            ans-=(r-l+1)*pre_mu(n/l);
        }
        return mm[n] = ans;
    }

    long long pre_phi(int n)
    {
        if (n <= maxn - 5) return phi[n];
        if (pp[n]) return pp[n];
        int l = 2, r;
        long long ans = 1LL*n*(n+1)/2;
        for(l=2; l<=n; l=r+1)
        {
            r=n/(n/l);
            ans-=(r-l+1)*pre_phi(n/l);
        }
        return pp[n] = ans;
    }
}D;

signed main()
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        int n;
        scanf("%d", &n);
        printf("%lld %d\n", D.pre_phi(n),D.pre_mu(n));
    }
    return 0;
}









//杜教筛  HDU5608  
struct djs
{
    bool vis[maxn];
    int prime[maxn],mu[maxn];
    ll f[maxn];
    unordered_map<ll,ll> Hash;         //保存f的前缀和在n/1....n/n 大约2sqrt(n)个点的值
    void init(int n)
    {
        memset(vis,0,sizeof(vis));
        int cnt=0;
        mu[1]=1;
        for(int i=2;i<=n;i++)
        {
            if(!vis[i])
            {
                prime[cnt++]=i;
                mu[i]=-1;
            }
            for(int j=0;j<cnt&&i*prime[j]<=n;j++)
            {
                vis[i*prime[j]]=1;
                if(i%prime[j]==0)
                {
                    break;
                }
                else
                {
                    mu[i*prime[j]]=-mu[i];
                }
            }
        }
        for(ll i=1;i<=n;i++)
        {
            ll g=(i*i-3*i+2)%mod;
            for(ll j=i;j<=n;j+=i)
            {
                f[j]=(f[j]+mu[j/i]*g)%mod;
            }
        }
        for(int i=1;i<=n;i++)                    //注意这个地方求前缀和的时候去掉负数的情况
            f[i]=(f[i]+f[i-1]+mod)%mod;          //或者在最后Query ans的时候判断ans的正负
    }                                            //如果为负则加上mod
    djs()
    {
        init(maxn-5);
        Hash.clear();
    }
    ll get_f(ll n)          //g[1]S[n]=(sum{i=1}^{n} h[i])-(sum{d=2}^{n} g[d]*S[n/d])
    {                       //S[n]为f[i]的前缀和   且必须 h=f*g (*为狄利克雷卷积符号)
        if (n <= maxn - 5) return f[n];
        ll temp=Hash[n];
        if(temp)return temp;
        ll ans =n*(n-1)%mod*(n-2)%mod*inv%mod;        //预处理 h[1]+h[2]+...+h[n] h的前缀和
        for(int l=2,r; l<=n; l=r+1)                   //整除分块
        {
            r=n/(n/l);
            ans=(ans-(r-l+1)*get_f(n/l)%mod)%mod;     //ans-=(g[r]-g[l-1])*S[x/p],所以g[i]的前缀和必须预处理出来
        }
        if(ans<0)
            ans+=mod;
        return Hash[n]=ans;
    }
}D;








//杜教筛 HashTable版本
struct djs
{
    bool vis[maxn];
    int prime[maxn],mu[maxn];
    ll f[maxn];
    struct HashTable
    {
        struct Line{int u,v,next;}e[1000000];
        int h[HashMod],cnt;
        void Hash(int u,int v,int w){e[++cnt]=(Line){w,v,h[u]};h[u]=cnt;}
        void Clear(){memset(h,0,sizeof(h));cnt=0;}
        void Add(int x,int k)
        {
            int s=x%HashMod;
            Hash(s,k,x);
        }
        int Query(int x)
        {
            int s=x%HashMod;
            for(int i=h[s];i;i=e[i].next)
                if(e[i].u==x)return e[i].v;
            return -1;
        }
    }Hash;
    void init(int n)
    {
        memset(vis,0,sizeof(vis));
        int cnt=0;
        mu[1]=1;
        for(int i=2;i<=n;i++)
        {
            if(!vis[i])
            {
                prime[cnt++]=i;
                mu[i]=-1;
            }
            for(int j=0;j<cnt&&i*prime[j]<=n;j++)
            {
                vis[i*prime[j]]=1;
                if(i%prime[j]==0)
                {
                    break;
                }
                else
                {
                    mu[i*prime[j]]=-mu[i];
                }
            }
        }
        for(ll i=1;i<=n;i++)
        {
            ll g=(i*i-3*i+2)%mod;
            for(ll j=i;j<=n;j+=i)
            {
                f[j]=(f[j]+mu[j/i]*g)%mod;
            }
        }
        for(int i=1;i<=n;i++)
            f[i]=(f[i]+f[i-1]+mod)%mod;            //注意这个地方求前缀和的时候去掉负数的情况
    }                                             //或者在最后Query ans的时候判断ans的正负
    djs()                                           //如果为负则加上mod
    {
        init(maxn-5);
        Hash.Clear();
    }
    ll get_f(ll n)
    {
        if (n <= maxn - 5) return f[n];
        ll temp=Hash.Query(n);
        if(temp!=-1)return temp;
        ll ans =n*(n-1)%mod*(n-2)%mod*inv%mod;
        for(int l=2,r; l<=n; l=r+1)
        {
            r=n/(n/l);
            ans=(ans-(r-l+1)*get_f(n/l)%mod)%mod;
        }
        if(ans<0)
            ans+=mod;
        Hash.Add(n,ans);
        return ans;
    }
}D;










//用块的id处是值来存前缀和的方式（非Hash）
//此代码对于Query>maxn次数过多的情形不太适用（Query中memset调用过多）
//G只需要比sqrt(n)大一点即可
struct djs
{
    #define G 100010
    const int inv=inverse(3,mod);
    ll id1[G],id2[G],n;
    int mu[maxn];
    int vis[maxn];
    int prime[maxn];
    ll f[maxn];
    void init(int n)
    {
        int cnt=0;
        mu[1]=1;
        for(int i=2;i<=n;i++)
        {
            if(!vis[i])
            {
                prime[cnt++]=i;
                mu[i]=-1;
            }
            for(int j=0;j<cnt&&i*prime[j]<=n;j++)
            {
                vis[i*prime[j]]=1;
                if(i%prime[j]==0)
                {
                    break;
                }
                else
                {
                    mu[i*prime[j]]=-mu[i];
                }
            }
        }
        for(ll i=1;i<=n;i++)
        {
            ll g=(i*i-3*i+2)%mod;
            for(ll j=i;j<=n;j+=i)
            {
                f[j]=(f[j]+mu[j/i]*g)%mod;
            }
        }
        for(int i=1;i<=n;i++)
        {
            f[i]=(f[i]+f[i-1]+mod)%mod;              //注意在线性筛部分保持非负
        }
    }
    djs()
    {
        init(maxn-5);
        memset(id1,0,sizeof(id1));
        memset(id2,0,sizeof(id2));
    }
    ll id(ll x)                  //获取该块处的前缀和的值
    {
        if(x<G) return id1[x];
        return id2[n/x];
    }
    ll get_f(ll x)          //g[1]S[n]=(sum{i=1}^{n} h[i])-(sum{d=2}^{n} g[d]*S[n/d])
    {                       //S[n]为f[i]的前缀和   且必须 h=f*g (*为狄利克雷卷积符号)
        if(x<=maxn-5) return f[x];
        ll temp=id(x);
        if(temp!=id2[0]) return temp;
        ll ans=x*(x-1)%mod*(x-2)%mod*inv%mod;       //h[1]+h[2]+...+h[n] h的前缀和
        for(ll l=2,r; l<=x; l=r+1)                  //整除分块
        {
            r=x/(x/l);
            ans=(ans-(r-l+1)*get_f(x/l)%mod)%mod;       //ans-=(g[r]-g[l-1])*S[x/p],所以g[i]的前缀和必须预处理出来
        }
        if(ans<0)
            ans+=mod;
        if(x<G)
            return id1[x]=ans;
        else
            return id2[n/x]=ans;
    }
    ll query(ll x)
    {
        if(x<=maxn-5)
            return f[x];
        memset(id1,-1,sizeof(id1));
        memset(id2,-1,sizeof(id2));
        n=x;
        return get_f(x);
    }
    #undef G
}D;