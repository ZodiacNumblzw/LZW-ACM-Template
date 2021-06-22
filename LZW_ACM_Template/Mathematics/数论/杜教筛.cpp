//�Ž�ɸ��phi��mu��ǰ׺�� O(n^(2/3)) ���P4213 ע��long long�˷���������ᱻ����

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









//�Ž�ɸ  HDU5608  
struct djs
{
    bool vis[maxn];
    int prime[maxn],mu[maxn];
    ll f[maxn];
    unordered_map<ll,ll> Hash;         //����f��ǰ׺����n/1....n/n ��Լ2sqrt(n)�����ֵ
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
        for(int i=1;i<=n;i++)                    //ע������ط���ǰ׺�͵�ʱ��ȥ�����������
            f[i]=(f[i]+f[i-1]+mod)%mod;          //���������Query ans��ʱ���ж�ans������
    }                                            //���Ϊ�������mod
    djs()
    {
        init(maxn-5);
        Hash.clear();
    }
    ll get_f(ll n)          //g[1]S[n]=(sum{i=1}^{n} h[i])-(sum{d=2}^{n} g[d]*S[n/d])
    {                       //S[n]Ϊf[i]��ǰ׺��   �ұ��� h=f*g (*Ϊ�������׾������)
        if (n <= maxn - 5) return f[n];
        ll temp=Hash[n];
        if(temp)return temp;
        ll ans =n*(n-1)%mod*(n-2)%mod*inv%mod;        //Ԥ���� h[1]+h[2]+...+h[n] h��ǰ׺��
        for(int l=2,r; l<=n; l=r+1)                   //�����ֿ�
        {
            r=n/(n/l);
            ans=(ans-(r-l+1)*get_f(n/l)%mod)%mod;     //ans-=(g[r]-g[l-1])*S[x/p],����g[i]��ǰ׺�ͱ���Ԥ�������
        }
        if(ans<0)
            ans+=mod;
        return Hash[n]=ans;
    }
}D;








//�Ž�ɸ HashTable�汾
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
            f[i]=(f[i]+f[i-1]+mod)%mod;            //ע������ط���ǰ׺�͵�ʱ��ȥ�����������
    }                                             //���������Query ans��ʱ���ж�ans������
    djs()                                           //���Ϊ�������mod
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










//�ÿ��id����ֵ����ǰ׺�͵ķ�ʽ����Hash��
//�˴������Query>maxn������������β�̫���ã�Query��memset���ù��ࣩ
//Gֻ��Ҫ��sqrt(n)��һ�㼴��
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
            f[i]=(f[i]+f[i-1]+mod)%mod;              //ע��������ɸ���ֱ��ַǸ�
        }
    }
    djs()
    {
        init(maxn-5);
        memset(id1,0,sizeof(id1));
        memset(id2,0,sizeof(id2));
    }
    ll id(ll x)                  //��ȡ�ÿ鴦��ǰ׺�͵�ֵ
    {
        if(x<G) return id1[x];
        return id2[n/x];
    }
    ll get_f(ll x)          //g[1]S[n]=(sum{i=1}^{n} h[i])-(sum{d=2}^{n} g[d]*S[n/d])
    {                       //S[n]Ϊf[i]��ǰ׺��   �ұ��� h=f*g (*Ϊ�������׾������)
        if(x<=maxn-5) return f[x];
        ll temp=id(x);
        if(temp!=id2[0]) return temp;
        ll ans=x*(x-1)%mod*(x-2)%mod*inv%mod;       //h[1]+h[2]+...+h[n] h��ǰ׺��
        for(ll l=2,r; l<=x; l=r+1)                  //�����ֿ�
        {
            r=x/(x/l);
            ans=(ans-(r-l+1)*get_f(x/l)%mod)%mod;       //ans-=(g[r]-g[l-1])*S[x/p],����g[i]��ǰ׺�ͱ���Ԥ�������
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