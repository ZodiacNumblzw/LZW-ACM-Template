//高次剩余可以求解x^k=a(mod p)的x的值，同时也是a的k次根的模p的意义
//BSGS可以求解a^x=b(mod p)中x的值 ，同时也是log(a,b)在模p意义下的值

//高次同余方程 曹家华学长的模板
//HDU3930
//x^a=b(mod p) 且p为素数
#include <algorithm>
#include <iostream>
#include <sstream>
#include <utility>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <cstring>
#include <cstdio>
#include <cmath>
#define met(a,b) memset(a, b, sizeof(a));
#define IN freopen("in.txt", "r", stdin);
#define OT freopen("ot.txt", "w", stdout);
using namespace std;
typedef long long LL;
typedef pair<int, int> PII;
const int maxn = 1e6 + 100;
const LL INF = 0x7fffffff;
const int dir[5][2] = {0,0,-1,0,1,0,0,-1,0,1};
const LL MOD = 1e9+7;
const double eps = 1e-6;

LL a, b, p;

//工具
bool is[maxn]; LL prm[maxn], id;
LL getprm(LL n) {
    if(n == 1) return 0;
    LL k = 0; met(is, 1);
    is[0] = is[1] = 0;
    for(LL i = 2; i < n; ++i) {
        if(is[i]) prm[k++] = i;
        for(LL j = 0; j < k && (i*prm[j] < n); ++j) {
            is[i*prm[j]] = 0;
            if(i % prm[j] == 0) break;
        }
    }
    return k;
}

LL Euler(LL x) {    //素数的欧拉函数
    return x-1;
}

LL gcd(LL a, LL b){
    return b ? gcd(b, a%b) : a;
}
LL extgcd(LL a, LL b, LL& x, LL& y) {
    if (b == 0) { x=1; y=0; return a; }
    LL d = extgcd(b, a % b, x, y);
    LL t = x; x = y; y = t - a / b * y;
    return d;
}

//快速乘 -- a*b % mod
LL pow_mul(LL a, LL b, LL p) {
    LL r = 0; a %= p;
    while(b) {
        if(b&1) r = (r+a) % p;
        a = (a+a) % p;
        b >>= 1;
    }
    return r;
}

LL pow_mod(LL a, LL b, LL p) {
    LL r = 1; a %= p;
    while(b) {
        if(b&1) r = pow_mul(r, a, p);
        a = pow_mul(a, a, p);
        b >>= 1;
    }
    return r;
}

//求原根
LL fac[maxn], num[maxn], tot;
LL Factor(LL n){
    LL ans = 1, temp = n; tot = 0;
    for (LL i = 0; i < id && prm[i] * prm[i] <= temp; i++){
        if (n % prm[i] == 0){
            fac[tot] = prm[i], num[tot] = 0;
            while (n%prm[i] == 0) n /= prm[i], ++num[tot];
            ans *= (num[tot] + 1);
            ++tot;
        }
    }
    if (n != 1){
        fac[tot] = n, num[tot] = 1;
        ans *=(num[tot]+1);
        ++tot;
    }
    return ans;
}

LL root(LL p) {
    LL phi = Euler(p);
    Factor(phi);
    for(LL g = 2; ; g++) {
        bool f = 1;
        for(int i = 0; i < tot; ++i) {
            LL t = phi / fac[i];
            if(pow_mod(g, t, p) == 1) { f = 0; break; }
        }
        if(f) return g;
    }
}

//BSGS
LL BSGS(LL a, LL b, LL p) {
    a %= p; b %= p;
    map<LL, LL> h;
    LL m = ceil(sqrt(p)), x, y, d, t = 1, v = 1;
    for(LL i = 0; i < m; ++i) {
        if(h.count(t)) h[t] = min(h[t], i);
        else h[t] = i;
        t = pow_mul(t, a, p);
    }
    for(LL i = 0; i < m; ++i) {
        d = extgcd(v, p, x, y);
        x = (x* b/d % p + p) % (p);
        if(h.count(x)) return i*m + h[x];
        v = pow_mul(v, t, p);
    }
    return -1;
}

//求模线性方程
LL modeq(LL a, LL b, LL p, LL r[]) {
    LL e, i, d, x, y;
    d = extgcd(a, p, x, y);
    if (b % d) { return -1; }
    e = (x * (b / d) + p) % p;
    for (i = 0; i < d; i++) {
        r[i] = (e + i*(p/d) + p) % p;
    }// 总共 (a, m) 个解
    return d;
}

//开始解决问题
LL solve(LL a, LL b, LL p, LL r[], LL ans[]) {
    LL g = root(p);
    LL t1 = BSGS(g, b, p);
    LL phi = Euler(p);
    LL cnt = modeq(a, t1, phi, r);
    if(cnt == -1) return -1;
    for(int i = 0; i < cnt; ++i) {
        ans[i] = pow_mod(g, r[i], p);
    }
    return cnt;
}

LL ans[maxn], res[maxn];

int main() {
    #ifdef _LOCAL
    IN; //OT;
    #endif // _LOCAL
    id = getprm(maxn-1); LL kase = 0;
    while(scanf("%lld%lld%lld", &a, &p, &b) == 3) {
        LL cnt = solve(a, b, p, res, ans);
        printf("case%lld:\n", ++kase);
        if(cnt == -1) { printf("-1\n"); continue; }
        sort(ans, ans + cnt);
        for(int i = 0; i < cnt; ++i) printf("%lld\n", ans[i]);
    }

    return 0;
}
//https://blog.csdn.net/a27038/article/details/77367076









//x^a=b(mod p)  当p不为素数时
#include <algorithm>
#include <iostream>
#include <sstream>
#include <utility>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <cstring>
#include <cstdio>
#include <cmath>
#define met(a,b) memset(a, b, sizeof(a));
#define IN freopen("in.txt", "r", stdin);
#define OT freopen("ot.txt", "w", stdout);
using namespace std;
typedef long long LL;
typedef pair<int, int> PII;
const int maxn = 1e6 + 100;
const LL INF = 0x7fffffff;
const int dir[5][2] = {0,0,-1,0,1,0,0,-1,0,1};
const LL MOD = 1e9+7;
const double eps = 1e-6;

LL a, b, p;

bool is[maxn]; LL prm[maxn], id;
LL getprm(LL n) {
    if(n == 1) return 0;
    LL k = 0; met(is, 1);
    is[0] = is[1] = 0;
    for(LL i = 2; i < n; ++i) {
        if(is[i]) prm[k++] = i;
        for(LL j = 0; j < k && (i*prm[j] < n); ++j) {
            is[i*prm[j]] = 0;
            if(i % prm[j] == 0) break;
        }
    }
    return k;
}

/*
LL Euler(LL x) {    //素数的欧拉函数
    return x-1;
}*/

LL Euler(LL x) {
    LL ans = x, m = (LL)sqrt(x*1.0)+1;
    for(LL i = 2; i < m; ++i) if(x%i == 0) {
        ans = ans / i * (i-1);
        while(x%i == 0) x /= i;
    }
    if(x > 1) ans = ans / x * (x-1);
    return ans;
}

LL gcd(LL a, LL b){
    return b ? gcd(b, a%b) : a;
}
LL extgcd(LL a, LL b, LL& x, LL& y) {
    if (b == 0) { x=1; y=0; return a; }
    LL d = extgcd(b, a % b, x, y);
    LL t = x; x = y; y = t - a / b * y;
    return d;
}

//快速乘 -- a*b % mod
LL pow_mul(LL a, LL b, LL p) {
    LL r = 0; a %= p;
    while(b) {
        if(b&1) r = (r+a) % p;
        a = (a+a) % p;
        b >>= 1;
    }
    return r;
}

LL pow_mod(LL a, LL b, LL p) {
    LL r = 1; a %= p;
    while(b) {
        if(b&1) r = pow_mul(r, a, p);
        a = pow_mul(a, a, p);
        b >>= 1;
    }
    return r;
}


LL Factor(LL n, LL fac[], LL num[], LL& tot){
    LL ans = 1, temp = n; tot = 0;
    for (LL i = 0; i < id && prm[i] * prm[i] <= temp; i++){
        if (n % prm[i] == 0){
            fac[tot] = prm[i], num[tot] = 0;
            while (n%prm[i] == 0) n /= prm[i], ++num[tot];
            ans *= (num[tot] + 1);
            ++tot;
        }
    }
    if (n != 1){
        fac[tot] = n, num[tot] = 1;
        ans *=(num[tot]+1);    //n的素因数中最多只有一个大于根号n的;
        ++tot;
    }
    return ans;
}

LL fac[maxn], num[maxn];
LL root(LL p, LL phi) {
    //LL phi = Euler(p);
    LL tot;
    Factor(phi, fac, num, tot);
    for(LL g = 2; ; g++) {
        bool f = 1;
        for(int i = 0; i < tot; ++i) {
            LL t = phi / fac[i];
            if(pow_mod(g, t, p) == 1) { f = 0; break; }
        }
        if(f && pow_mod(g, phi, p) == 1) return g;
    }
    return -1;
}

LL modeq(LL a, LL b, LL p, LL r[]) {
    LL e, i, d, x, y;
    d = extgcd(a, p, x, y);
    if (b % d) { return -1; }
    e = (x * (b / d) + p) % p;
    for (i = 0; i < d; i++) {
        r[i] = (e + i*(p/d) + p) % p;
    }// 总共 (a, m) 个解
    return d;
}

LL CRT(LL a[], LL m[], LL k) {
    LL i, d, x, y, Mi, ans = 0, M = 1;
    for (i = 0; i < k; i++) M *= m[i];  // !  注意不能overflow
    for (i = 0; i < k; i++) {
        Mi = M / m[i];
        d = extgcd(m[i], Mi, x, y);     // y 为逆元 -- Mi*y === 1 (% m[i])
        ans = (ans + a[i]*y*Mi) % M;
    }
    if (ans >= 0) return ans;
    else return (ans + M);
}

LL exBSGS(LL a, LL b, LL p) {
    a = (a%p+p)%p; b = (b%p+p)%p;
    LL ret = 1;
    for(LL i = 0; i <= 50; ++i) {
        if(ret == b) return i;
        ret = (ret*a) % p;
    }//枚举比较小的i

    LL x,y,d, v = 1, cnt = 0;
    while((d = gcd(a, p)) != 1) {
        if(b % d) return -1;
        b /= d, p /= d;
        v = (v * (a/d)) % p;
        ++cnt;
    }//约分直到(a, p) == 1

    map<LL, LL> h;
    LL m = ceil(sqrt(p)), t = 1;
    for(LL i = 0; i < m; ++i) {
        if(h.count(t)) h[t] = min(h[t], i);
        else h[t] = i;
        t = (t*a) % p;
    }
    for(LL i = 0; i < m; ++i) {
        d = extgcd(v, p, x, y);
        x = (x* (b/d) % p + p) % p;
        if(h.count(x)) return i*m + h[x] + cnt;
        v = (v*t) % p;
    }
    return -1;
}


LL F[maxn], N[maxn], P[maxn], E[maxn], r[maxn], tot;
vector< vector<LL> > ans;
vector<LL> ot;

LL A[maxn], M[maxn];
void dfs(LL dep, LL N){
    if( dep == N ){ ot.push_back(CRT(A, M, N)); }
    else {
        for(LL i = 0; i < ans[dep].size(); ++i) {
            A[dep] = ans[dep][i];
            dfs(dep+1, N);
        }
    }
}

LL solve(LL a, LL b, LL p, LL r[]) {
    //分解因式
    Factor(p, F, N, tot);
    for(LL i = 0; i < tot; ++i) {
        P[i] = pow_mod(F[i], N[i], p*2);
        E[i] = P[i] - P[i]/F[i];
    }

    ans.clear();
    for(LL i = 0; i < tot; ++i) {
        vector<LL> res;
        if(F[i] == 2) {
            LL tb = (b%P[i]+P[i])%P[i];
            for(LL j = 0; j < P[i]; ++j) {
                if(pow_mod(j, a, P[i]) == tb) res.push_back(j);
            }
            if(res.size() == 0) return -1;
            sort(res.begin(), res.end());
            res.erase(unique(res.begin(), res.end()), res.end());
            ans.push_back(res);
            continue;
        }
        if(b % P[i] == 0) {
            LL x = 0, ret = 1;
            while(pow_mod(ret, a, P[i]) != 0) ret *= F[i];
            for(int j = 0; j < P[i]/ret; ++j) res.push_back(ret*j);
            sort(res.begin(), res.end());
            res.erase(unique(res.begin(), res.end()), res.end());
            ans.push_back(res);
            continue;
        }

        LL tp = P[i], tb = b%tp,te = E[i];//, d = gcd(tp, tb);
        LL g = root(tp, te);        if(g == -1) return -1;  //求原根
        LL t1 = exBSGS(g, tb, tp);     if(t1 == -1) return -1; //求离散对数
        LL cnt = modeq(a, t1, te, r); if(cnt == -1) return -1;//求log_gX
        for(LL j = 0; j < cnt; ++j) {
            res.push_back(pow_mod(g, r[j], tp));
        }
        sort(res.begin(), res.end());
        res.erase(unique(res.begin(), res.end()), res.end());
        ans.push_back(res);
    }

    //CRT合并
    ot.clear();
    for(LL i = 0; i < tot; ++i) M[i] = P[i];
    dfs(0, tot);
    sort(ot.begin(), ot.end());
    ot.erase(unique(ot.begin(), ot.end()), ot.end());

}

int main() {
    #ifdef _LOCAL
    IN; //OT;
    #endif // _LOCAL

    id = getprm(maxn-1);
    int t; cin >> t;
    while(t--) {
        scanf("%lld%lld%lld", &a, &p, &b);
        if(solve(a, b, p, r) == -1) printf("No Solution\n");
        else {
            for(LL i = 0; i < ot.size(); ++i) printf("%lld ", ot[i]);
            printf("\n");
        }
    }

    return 0;
}









//高次同余方程/高次剩余ZZQ模板
#include<bits/stdc++.h>
using namespace std;
#define pb push_back
#define MP make_pair
typedef pair<int,int> pii;
typedef long long ll;
typedef double ld;
typedef vector<int> vi;
#define fi first
#define se second
#define fe first
#define FO(x) {freopen(#x".in","r",stdin);freopen(#x".out","w",stdout);}
#define Edg int M=0,fst[SZ],vb[SZ],nxt[SZ];void ad_de(int a,int b){++M;nxt[M]=fst[a];fst[a]=M;vb[M]=b;}void adde(int a,int b){ad_de(a,b);ad_de(b,a);}
#define Edgc int M=0,fst[SZ],vb[SZ],nxt[SZ],vc[SZ];void ad_de(int a,int b,int c){++M;nxt[M]=fst[a];fst[a]=M;vb[M]=b;vc[M]=c;}void adde(int a,int b,int c){ad_de(a,b,c);ad_de(b,a,c);}
#define es(x,e) (int e=fst[x];e;e=nxt[e])
#define esb(x,e,b) (int e=fst[x],b=vb[e];e;e=nxt[e],b=vb[e])
#define VIZ {printf("digraph G{\n"); for(int i=1;i<=n;i++) for es(i,e) printf("%d->%d;\n",i,vb[e]); puts("}");}
#define VIZ2 {printf("graph G{\n"); for(int i=1;i<=n;i++) for es(i,e) if(vb[e]>=i)printf("%d--%d;\n",i,vb[e]); puts("}");}
#define SZ 666666
template<class T>
inline T dw();
template<>
inline ll dw<ll>()
{
    return 1;
}
template<>
inline int dw<int>()
{
    return 1;
}
typedef pair<ll,ll> pll;
ll pll_s;
inline pll mul(pll a,pll b,ll p)
{
    pll ans;
    ans.fi=a.fi*b.fi%p+a.se*b.se%p*pll_s%p;
    ans.se=a.fi*b.se%p+a.se*b.fi%p;
    ans.fi%=p;
    ans.se%=p;
    return ans;
}
inline ll mul(ll a,ll b,ll c)
{
    return a*b%c;
}
//a^b mod c
template<class T>
T qp(T a,ll b,ll c)
{
    T ans=dw<T>();
    while(b)
    {
        if(b&1) ans=mul(ans,a,c);
        a=mul(a,a,c);
        b>>=1;
    }
    return ans;
}
inline ll ll_rnd()
{
    ll ans=0;
    for(int i=1; i<=5; i++)
        ans=ans*32768+rand();
    if(ans<0) ans=-ans;
    return ans;
}
//(x,y) -> x+sqrt(pll_s)*y
template<>
inline pll dw<pll>()
{
    return pll(1,0);
}
//find (possibly) one root of x^2 mod p=a
//correctness need to be checked
ll sqr(ll a,ll p)
{
    if(!a) return 0;
    if(p==2) return 1;
    ll w,q;
    while(1)
    {
        w=ll_rnd()%p;
        q=w*w-a;
        q=(q%p+p)%p;
        if(qp(q,(p-1)/2,p)!=1)
            break;
    }
    pll_s=q;
    pll rst=qp(pll(w,1),(p+1)/2,p);
    ll ans=rst.fi;
    ans=(ans%p+p)%p;
    return ans;
}
//solve x^2 mod p=a
vector<ll> all_sqr(ll a,ll p)
{
    vector<ll> vec;
    a=(a%p+p)%p;
    if(!a)
    {
        vec.pb(0);
        return vec;
    }
    ll g=sqr(a,p);
    ll g2=(p-g)%p;
    if(g>g2) swap(g,g2);
    if(g*g%p==a) vec.pb(g);
    if(g2*g2%p==a&&g!=g2) vec.pb(g2);
    return vec;
}
ll s3_a;
//f0+f1*x+f2*x^2 (for x^3=s3_a)
struct s3
{
    ll s[3];
    s3()
    {
        s[0]=s[1]=s[2]=0;
    }
    s3(ll* p)
    {
        s[0]=p[0];
        s[1]=p[1];
        s[2]=p[2];
    }
    s3(ll a,ll b,ll c)
    {
        s[0]=a;
        s[1]=b;
        s[2]=c;
    }
};
template<>
s3 dw<s3>()
{
    return s3(1,0,0);
}
s3 rs3(ll p)
{
    return s3(ll_rnd()%p,ll_rnd()%p,ll_rnd()%p);
}
s3 mul(s3 a,s3 b,ll p)
{
    ll k[3]= {};
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3; j++)
        {
            if(i+j<3) k[i+j]+=a.s[i]*b.s[j]%p;
            else k[i+j-3]+=a.s[i]*b.s[j]%p*s3_a%p;
        }
    }
    for(int i=0; i<3; i++) k[i]%=p;
    return s3(k[0],k[1],k[2]);
}
//solve x^3 mod p=a
vector<ll> all_cr(ll a,ll p)
{
    vector<ll> vec;
    a=(a%p+p)%p;
    if(!a)
    {
        vec.pb(0);
        return vec;
    }
    if(p<=3)
    {
        for(int i=0; i<p; i++)
        {
            if(i*i*i%p==a) vec.pb(i);
        }
        return vec;
    }
    if(p%3==2)
    {
        vec.pb(qp(a,(p*2-1)/3,p));
        return vec;
    }
    if(qp(a,(p-1)/3,p)!=1) return vec;
    ll l=(sqr(p-3,p)-1)*qp(2LL,p-2,p)%p,x;
    s3_a=a;
    while(1)
    {
        s3 u=rs3(p);
        s3 v=qp(u,(p-1)/3,p);
        if(v.s[1]&&!v.s[0]&&!v.s[2])
        {
            x=qp(v.s[1],p-2,p);
            break;
        }
    }
    x=(x%p+p)%p;
    vec.pb(x);
    vec.pb(x*l%p);
    vec.pb(x*l%p*l%p);
    sort(vec.begin(),vec.end());
    return vec;
}
map<ll,ll> gg;
ll yss[2333];
int yyn=0;
//find x's primitive root
inline ll org_root(ll x)
{
    ll& pos=gg[x];
    if(pos) return pos;
    yyn=0;
    ll xp=x-1;
    for(ll i=2; i*i<=xp; i++)
    {
        if(xp%i) continue;
        yss[++yyn]=i;
        while(xp%i==0) xp/=i;
    }
    if(xp!=1) yss[++yyn]=xp;
    ll ans=1;
    while(1)
    {
        bool ok=1;
        for(int i=1; i<=yyn; i++)
        {
            ll y=yss[i];
            if(qp(ans,(x-1)/y,x)==1)
            {
                ok=0;
                break;
            }
        }
        if(ok) return pos=ans;
        ++ans;
    }
}
map<ll,int> bsgs_mp;
//find smallest x: a^x mod p=b
ll bsgs(ll a,ll b,ll p)
{
    if(b==0) return 1;
    map<ll,int>& ma=bsgs_mp;
    ma.clear();
    //only /2.5 for speed...
    ll hf=sqrt(p)/2.5+2,cur=b;
    for(int i=0; i<hf; i++)
        ma[cur]=i+1, cur=cur*a%p;
    ll qwq=1,qh=qp(a,hf,p);
    for(int i=0;; i++)
    {
        if(i)
        {
            if(ma.count(qwq))
                return i*hf-(ma[qwq]-1);
        }
        qwq=qwq*(ll)qh%p;
    }
    return 1e18;
}
//ax+by=1
void exgcd(ll a,ll b,ll& x,ll& y)
{
    if(b==0)
    {
        x=1;
        y=0;
        return;
    }
    exgcd(b,a%b,x,y);
    ll p=x-a/b*y;
    x=y;
    y=p;
}
template<class T>
T gcd(T a,T b)
{
    if(b) return gcd(b,a%b);
    return a;
}
//solve x^a mod p=b
vector<ll> kr(ll a,ll b,ll p)
{
    vector<ll> rst;
    if(!b)
    {
        rst.pb(0);
        return rst;
    }
    ll g=org_root(p);
    ll pb=bsgs(g,b,p);
    ll b1=a,b2=p-1,c=pb;
    ll gg=gcd(b1,b2);
    if(c%gg) return rst;
    b1/=gg, b2/=gg, c/=gg;
    ll x1,x2;
    exgcd(b1,b2,x1,x2);
    x1*=c;
    x1=(x1%b2+b2)%b2;
    ll cs=qp(g,x1,p),ec=qp(g,b2,p);
    for(ll cur=x1; cur<p-1; cur+=b2)
        rst.pb(cs), cs=cs*ec%p;
    sort(rst.begin(),rst.end());
    return rst;
}
