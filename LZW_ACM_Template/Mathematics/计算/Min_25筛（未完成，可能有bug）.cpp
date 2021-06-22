//Min_25ɸ               Loj6053
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
void pre(int n)          //����ɸԤ���� ɸsqrt(n)�ڵ�����
{
    vis[1]=true;
    for(int i=2; i<=n; ++i)
    {
        if(!vis[i])
        {
            prime[++tot]=i;
            sp[tot]=(sp[tot-1]+i)%MOD;           //����ǰ׺��
            //���������Ҫ���ǰ׺��
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
    int k=(x<=Sqr)?id1[x]:id2[n/x],ret=(g[k]-sp[y-1]-h[k]+y-1)%MOD;//�˴��޸�
    if(y==1)ret+=2;
    for(int i=y; i<=tot&&1ll*prime[i]*prime[i]<=x; ++i)
    {
        ll t1=prime[i],t2=1ll*prime[i]*prime[i];    //�˴��޸�
        for(int e=1; t2<=x; ++e,t1=t2,t2*=prime[i])
        {
            (ret+=((1ll*S(x/t1,i+1)*(prime[i]^e)%MOD+(prime[i]^(e+1))%MOD)))%=MOD;  //�˴����ݹ�ʽ�޸� ����f�ı��ʽ
        }
    }
    return ret;
}
int main()
{
    scanf("%lld",&n);
    Sqr=sqrt(n);
    pre(Sqr);
    for(ll i=1,j; i<=n; i=j+1)           //�����ֿ�
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
            (h[i]-=h[k]-j+1)%=MOD;//�˴��޸�
        }
    }
    int ans=S(n,1)+1;
    printf("%d\n",(ans+MOD)%MOD);
    return 0;
}







//Min_25ɸ by:yyb          ���޸�
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
void pre(int n)                 //Ԥ���������
{
    zs[1]=true;
    for(int i=2; i<=n; ++i)
    {
        if(!zs[i])pri[++tot]=i,sp[tot]=(sp[tot-1]+i)%MOD;       //spΪ������ǰ׺��
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
    int k=(x<=Sqr)?id1[x]:id2[n/x],ret=(g[k]-sp[y-1]-h[k]+y-1)%MOD;   //��ʼ��g(x,|P|)-\sum{i=1}^{y-1}(f(i))
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
    for(ll i=1,j; i<=n; i=j+1)                     //����ǰ׺�;�Ϊ��1�����ǰ׺��
    {
        j=n/(n/i);
        w[++m]=n/i;                                   //���Ժ���f(p)�ɼ�������ʽ���ɾ���Ҫ����Ԥ����ĺ���
                                                      //����f(p)=p^3-p^2+p-1����ҪԤ����i^3 i^2 i 1���ĸ�ǰ׺��
        h[m]=(w[m]-1)%MOD;                            //�˴�h��Ҫά������������ǰ׺�ͣ���ʼ��Ϊ��������ǰ׺��
                                                      //����������������������ʼ����
        g[m]=(w[m]%MOD)*((w[m]+1)%MOD)%MOD;           //�˴�gά��f(p)��ǰ׺�ͻ��ǽ���������������
                                                      //����˳�ʼ��Ϊ��������ǰ׺�ͣ�
        if(g[m]&1)g[m]=g[m]+MOD;
        g[m]/=2;
        g[m]--;
        if(w[m]<=Sqr)id1[w[m]]=m;         //������С��sqrt(n)�����ֿ����� w[m]<sqrt(n)
        else id2[j]=m;                   //������w[n/m] ���������ڴ�Ϊsqrt(n)������
    }                                    //������Ҫ���ÿ��g(n/i,|P|)��ֵ����Ϊ�ڵݹ���S��ֻ��Ҫ��Щλ�õ�ֵ

                                        //����������g(n/l,0)������g(n/l,|P|);
    for(int j=1; j<=tot; ++j)
        for(int i=1; i<=m&&pri[j]*pri[j]<=w[i]; ++i)
        {
            int k=(w[i]/pri[j]<=Sqr)?id1[w[i]/pri[j]]:id2[n/(w[i]/pri[j])];     //�ҵ�k���ĸ��� ��[n/(l*prime[i])]�������ĸ�g/h����
            (g[i]-=1ll*pri[j]*(g[k]-sp[j-1])%MOD)%=MOD;   //g(n,j)=g(n-1,j)-f(p_j)(g(n/p_j,j-1)-sum{i=1}^{j-1}(f(pi)))
            (h[i]-=h[k]-j+1)%=MOD;
        }
    int ans=S(n,1)+1;         //������f(1)
    printf("%d\n",(ans+MOD)%MOD);
    return 0;
}


//Min_25ɸ��Ԥ����
int n,val[N*2],id1[N],id2[N];
//��������Ԥ����
int sqrt_n=sqrt(n),tot=0;
for(int i=1,j;i<=n;i=j+1) {
    j=n/(n/i);
    int w=n/i;val[++tot]=w;
    if(w<=sqrt_n) id1[w]=tot;
    else id2[n/w]=tot;
}
//��ѯĳ��m�ı��
inline int get_id(int m) {
    if(m<=sqrt_n) return id1[m];
    else return id2[n/m];
}




//�����ɸphi��mu�İ���
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
    ios::sync_with_stdio(0);/*syn����*/
    sieve();
    int T;cin>>T;while(T--)solve();
    return 0;
}









//Min_25    �Լ��ĵİ���
struct Min_25
{
    ll n,Sqr,w[maxn];
    ll prime[maxn],id1[maxn],id2[maxn],h[maxn],g[maxn],m;
    bool vis[maxn];
    int tot,sp[maxn],sp2[maxn];
    void pre(int n)         //Ԥ���������
    {                       //����f=p^2-p
    	tot=0;              //����Ҫɸ��p��p^2����������ǰ׺��
        memset(vis,0,sizeof(vis));   
        vis[1]=true;
        for(int i=2; i<=n; i++)
        {
            if(!vis[i])
                prime[++tot]=i,sp[tot]=(sp[tot-1]+i)%mod,sp2[tot]=(sp2[tot-1]+1ll*i*i%mod)%mod;       //spΪ������ǰ׺�� ,sp2Ϊ����ƽ������ǰ׺��
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
            //���Ժ���f(p)�ɼ�������ʽ���ɾ���Ҫ����Ԥ����ĺ���
            //����f(p)=p^3-p^2+p-1����ҪԤ����i^3 i^2 i 1���ĸ�ǰ׺��
            //�˴�h��Ҫά������������ǰ׺�ͣ���ʼ��Ϊ��������ǰ׺��
            h[m]=(w[m]%mod)*((w[m]+1)%mod)%mod*inverse(2,mod)%mod;
            //����f=p^2-p     h����2,3,..n�ĺ�,g����2^2,3^2,...n^2��ǰ׺��
            h[m]--;
            //����������������������ʼ����           h(x)=f1(2)+f1(3)+...+f1(n)
            g[m]=((w[m]%mod)*((w[m]+1)%mod)%mod)%mod*((2*w[m]+1)%mod)%mod*inverse(6,mod)%mod;
            g[m]--;
            //�˴�gά��f(p)��ǰ׺��  ���ǽ���������������
            //����˳�ʼ��Ϊ��������ǰ׺�ͣ�               g(x)=f2(2)+...+f2(x)

            if(w[m]<=Sqr)id1[w[m]]=m;         //������С��sqrt(n)�����ֿ����� w[m]<sqrt(n)
            else id2[r]=m;                   //������w[n/m] ���������ڴ�Ϊsqrt(n)������
        }
        //������Ҫ���ÿ��g(n/i,|P|)��ֵ����Ϊ�ڵݹ���S��ֻ��Ҫ��Щλ�õ�ֵ

        //����������g(n/l,0)������g(n/l,|P|);
        for(int j=1; j<=tot; j++)
        {
            for(int i=1; i<=m&&prime[j]*prime[j]<=w[i]; i++)
            {
                int k=(w[i]/prime[j]<=Sqr)?id1[w[i]/prime[j]]:id2[n/(w[i]/prime[j])];
                //�ҵ�k���ĸ��� ��[n/(l*prime[i])]�������ĸ�g/h����
                g[i]=(g[i]-1ll*(prime[j]*prime[j]%mod)*((g[k]-sp2[j-1])%mod)%mod)%mod;
                //g(n,j)=g(n,j-1)-f(p_j)(g(n/p_j,j-1)-sum{i=1}^{j-1}(f(pi)))
                //�˴��ֱ�f=p^2,f=p;
                h[i]=(h[i]-1ll*prime[j]*(h[k]-sp[j-1])%mod)%mod;
            }
        }
    }
    ll S(ll x,int y)
    {//S(n,j)=g(n,|P|)-\sum{i=1}^{j-1}(f(i))+\sum{k>j&&p_k^(e+1)<=n}^{}(f(p_k^e)*S(n/(p_k^e),k+1)+f(p_k^(e+1)))
        if(x<=1||prime[y]>x)return 0;
        ll k=(x<=Sqr)?id1[x]:id2[n/x],ret=((g[k]-h[k])%mod-((sp2[y-1]-sp[y-1])%mod))%mod;
                                     //��ʼ��g(x,|P|)-\sum{i=1}^{y-1}(f(i))
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
        ll ans=S(nn,1)+1;         //������f(1)
        if(ans<0)
            ans+=mod;
        return ans;
    }
}M;