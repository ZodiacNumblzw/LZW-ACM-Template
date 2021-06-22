#pragma GCC optimize(3)
#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
#include<ext/rope>
using namespace std;
using namespace __gnu_cxx;
using namespace __gnu_pbds;
void test(){cerr<<"\n";}
template<typename T,typename... Args>void test(T x,Args... args){cerr<<x<<" ";test(args...);}
#define ll long long
#define ld long double
#define ull unsigned long long
#define rep(i,a,b) for(register int i=a;i<b;i++)
#define Rep(i,a,b) for(register int i=a;i<=b;i++)
#define per(i,a,b) for(register int i=b-1;i>=a;i--)
#define Per(i,a,b) for(register int i=b;i>=a;i--)
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
#define CASET int ___T; scanf("%d", &___T); for(register int cs=1;cs<=___T;cs++)
#define Read(x) scanf("%d",&x)
#define Read2(x,y) scanf("%d%d",&x,&y)
#define Read3(x,y,z) scanf("%d%d%d",&x,&y,&z)
#define Read4(x,y,z,t) scanf("%d%d%d%d",&x,&y,&z,&t)
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
inline void ADD(int &a,int b) {a+=b; if(a>=mod) a-=mod; if(a<0) a+=mod;}


signed main()
{

    return Accepted;
}

/*
 * ┌───┐   ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┬───┐ ┌───┬───┬───┐
 * │Esc│   │ F1│ F2│ F3│ F4│ │ F5│ F6│ F7│ F8│ │ F9│F10│F11│F12│ │P/S│S L│P/B│  ┌┐    ┌┐    ┌┐
 * └───┘   └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┴───┘ └───┴───┴───┘  └┘    └┘    └┘
 * ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───────┐ ┌───┬───┬───┐ ┌───┬───┬───┬───┐
 * │~ `│! 1│@ 2│# 3│$ 4│% 5│^ 6│& 7│* 8│( 9│) 0│_ -│+ =│ BacSp │ │Ins│Hom│PUp│ │Num│ / │ * │ - │
 * ├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─────┤ ├───┼───┼───┤ ├───┼───┼───┼───┤
 * │ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{ [│} ]│ | \ │ │Del│End│PDn│ │ 7 │ 8 │ 9 │   │
 * ├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤ └───┴───┴───┘ ├───┼───┼───┤ + │
 * │ Caps │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  │               │ 4 │ 5 │ 6 │   │
 * ├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────────┤     ┌───┐     ├───┼───┼───┼───┤
 * │ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│  Shift   │     │ ↑ │     │ 1 │ 2 │ 3 │   │
 * ├─────┬──┴─┬─┴──┬┴───┴───┴───┴───┴───┴──┬┴───┼───┴┬────┬────┤ ┌───┼───┼───┐ ├───┴───┼───┤ E││
 * │ Ctrl│ Win│ Alt│         Space         │ Alt│ Win│Menu│Ctrl│ │ ← │ ↓ │ → │ │   0   │ . │←─┘│
 * └─────┴────┴────┴───────────────────────┴────┴────┴────┴────┘ └───┴───┴───┘ └───────┴───┴───┘
 */










/*
	快速读入
*/

//fread读入挂
inline char gc()//用fread把读入加速到极致，具体我也不懂，背板子就好
{
    static char BB[1000000],*S=BB,*T=BB;
    return S==T&&(T=(S=BB)+fread(BB,1,1000000,stdin),S==T)?EOF:*S++;
}
inline int getin()//getchar换成fread的快读
{
    register int x=0;register char ch=gc();
    while(ch<48)ch=gc();
    while(ch>=48)x=x*10+(ch^48),ch=gc();
    return x;
}



//read读入挂
template <typename _tp> inline _tp read(_tp&x){
	char ch=getchar(),sgn=0;x=0;
	while(ch^'-'&&!isdigit(ch))ch=getchar();if(ch=='-')ch=getchar(),sgn=1;
	while(isdigit(ch))x=x*10+ch-'0',ch=getchar();if(sgn)x=-x;return x;
}

int main(){int a,b;b=read(a);}  //read()即可读入a 可再赋值给b

//输入输出挂
template <class T>
inline bool read(T &ret)
{
    char c;
    int sgn;
    T bit=0.1;
    if(c=getchar(),c==EOF) return 0;
    while(c!='-'&&c!='.'&&(c<'0'||c>'9')) c=getchar();
    sgn=(c=='-')?-1:1;
    ret=(c=='-')?0:(c-'0');
    while(c=getchar(),c>='0'&&c<='9') ret=ret*10+(c-'0');
    if(c==' '||c=='\n')
    {
        ret*=sgn;
        return 1;
    }
    while(c=getchar(),c>='0'&&c<='9') ret+=(c-'0')*bit,bit/=10;
    ret*=sgn;
    return 1;
}
inline void out(int x)
{
    if(x>9) out(x/10);
    putchar(x%10+'0');
}




//随机Hash          //不会被107897和126271卡unordered_map
struct custom_hash {
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return x + FIXED_RANDOM;
    }
};

struct custom_hash {
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        x ^= FIXED_RANDOM;
        return x ^ (x >> 16);
    }
};






/*
	LZW的数学板子
*/



//计算方面：
//gcd
inline ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
//exgcd
ll exgcd(ll a, ll b, ll &x, ll &y) {     //ax+by=gcd(a,b)
    if (b == 0) {x = 1, y = 0; return a;}
    ll r = exgcd(b, a % b, x, y), tmp;
    tmp = x; x = y; y = tmp - (a / b) * y;
    return r;
}

int exgcd(int a,int b) //ax+by=gcd(a,b) return x;  非递归写法
{
    int x0=1,y0=0,x1=0,y1=1,x=a,y=b,r=a%b,q=a;
    while(r)
    {
        x=x0-q*x1;
        y=y0-q*y1;
        x0=x1,y0=y1;
        x1=x,y1=y;
        a=b;
        b=r;
        r=a%b;
        q=a;
    }
    return x;
}




ll inv(ll a, ll b) {
    ll r = exgcd(a, b, x, y);
    while (x < 0) x += b;
    return x;
}




//Python板子


/*def gcd(a,b):
    if b==0:
        return a
    else:
        return gcd(b,a%b)
 
 
def exgcd(r0, r1): # calc ax+by = gcd(a, b) return x
    x0, y0 = 1, 0
    x1, y1 = 0, 1
    x, y = r0, r1
    r = r0 % r1
    q = r0 // r1
    while r:
        x, y = x0 - q * x1, y0 - q * y1
        x0, y0 = x1, y1
        x1, y1 = x, y
        r0 = r1
        r1 = r
        r = r0 % r1
        q = r0 // r1
    return x
 
def inv(a,b):
    x=exgcd(a,b)
    while x<0:
        x+=b
    return x
*/

/*
    Author: wsnpyo
    Update Date: 2014-11-16
    Algorithm: 快速幂/Fermat, Solovay_Stassen, Miller-Rabin素性检验/Exgcd非递归版/中国剩余定理
*/
/*
import random

def QuickPower(a, n, p): # 快速幂算法
    tmp = a
    ret = 1
    while(n > 0):
        if (n&1):
            ret = (ret * tmp) % p
        tmp = (tmp * tmp) % p
        n>>=1
    return ret

def Jacobi(n, m): # calc Jacobi(n/m)
    n = n%m
    if n == 0:
        return 0
    Jacobi2 = 1
    if not (n&1): # 若有n为偶数, 计算Jacobi2 = Jacobi(2/m)^(s) 其中n = 2^s*t t为奇数
        k = (-1)**(((m**2-1)//8)&1)
        while not (n&1):
            Jacobi2 *= k
            n >>= 1
    if n == 1:
        return Jacobi2
    return Jacobi2 * (-1)**(((m-1)//2*(n-1)//2)&1) * Jacobi(m%n, n)

def Exgcd(r0, r1): # calc ax+by = gcd(a, b) return x
    x0, y0 = 1, 0
    x1, y1 = 0, 1
    x, y = r0, r1
    r = r0 % r1
    q = r0 // r1
    while r:
        x, y = x0 - q * x1, y0 - q * y1
        x0, y0 = x1, y1
        x1, y1 = x, y
        r0 = r1
        r1 = r
        r = r0 % r1
        q = r0 // r1
    return x

def Fermat(x, T): # Fermat素性判定
        if x < 2:
                return False
        if x <= 3:
                return True
        if x%2 == 0 or x%3 == 0:
                return False
        for i in range(T):
                ran = random.randint(2, x-2) # 随机取[2, x-2]的一个整数
                if QuickPower(ran, x-1, x) != 1:
                        return False
        return True

def Solovay_Stassen(x, T): # Solovay_Stassen素性判定
    if x < 2:
        return False
    if x <= 3:
        return True
    if x%2 == 0 or x%3 == 0:
        return False
    for i in range(T): # 随机选择T个整数
        ran = random.randint(2, x-2)
        r = QuickPower(ran, (x-1)//2, x)
        if r != 1 and r != x-1:
            return False
        if r == x-1:
            r = -1
        if r != Jacobi(ran, x):
            return False
    return True

def MillerRabin(x, ran): # x-1 = 2^s*t
    tx = x-1
    s2 = tx&(~tx+1) # 取出最后一位以1开头的二进制 即2^s
    r = QuickPower(ran, tx//s2, x)
    if r == 1 or r == tx:
        return True
    while s2>1: # 从2^s -> 2^1 循环s次
        r = (r*r)%x
        if r == 1:
            return False
        if r == tx:
            return True
        s2 >>= 1
    return False

def MillerRabin_init(x, T): #Miller-Rabin素性判定
    if x < 2:
        return False
    if x <= 3:
        return True
    if x%2 == 0 or x%3 == 0:
        return False
    for i in range(T): # 随机选择T个整数
        ran = random.randint(2, x-2)
        if not MillerRabin(x, ran):
            return False
    return True

def CRT(b, m, n): # calc x = b[] % m[]
    M = 1
    for i in range(n):
        M *= m[i]
    ans = 0
    for i in range(n):
        ans += b[i] * M // m[i] * Exgcd(M//m[i], m[i])
    return ans%M
*/

//快速幂
ll quick(ll a,ll b,ll m)
{
    ll ans=1;
    while(b)
    {
        if(b&1)
        {
            ans=(a*ans)%m;
        }
        a=(a*a)%m;
        b>>=1;
    }
    return ans;
}


//O((logn)^2)十进制快速幂 （需要欧拉降幂，慎用）
ll quick_mod_10(ll a,string b,ll m)                   //当模数为素数时
{
    ll ans=1,res[10],temp=1;;                         //res[]用来存a^0,a^1,a^2,a^3……a^9
    a%=m;
    int len=b.length();
    res[0]=1;
    for(int i=1;i<10;i++)
    {
        res[i]=(res[i-1]*a)%m;                        //预处理一下res[]
    }
    for(int i=len-1;i>=0;i--)
    {
        ans=(ans*quick(res[(int)(b[i]-'0')],temp,m))%m;
        temp=(temp*10)%(mod-1);                       //欧拉降幂
    }
    return ans;
}


//十进制快速幂
ll quickmod_10(ll a,string b,ll m)
{
    ll ans=1;
    a%=m;
    int len=b.length();
    for(int i=len-1;i>=0;i--)
    {
        ans=(ans*quick(a,(int)(b[i]-'0'),m))%m;
        a=quick(a,10,m);
    }
    return ans;
}


//快速乘 O(1)
inline ll quick_c(ll a, ll b, ll p)
{
    ll c = a*b-(ll)((long double)a*b/p+0.5)*p;
    return c < 0 ? c+p:c;
}
//快速乘O(logn)
ll quick_c(ll a,ll b,ll m)
{
    ll ans=0;
    while(b)
    {
        if(b&1)
        {
            ans=(ans+a)%m;
        }
        b>>=1;
        a=(a+a)%m;
    }
    return ans;
}


//矩阵快速幂
struct Matrix                       //矩阵结构体
{
    int sizen;
    ll m[105][105];
};
namespace Matrix_quick
{
const ll mod=1e9+7;
Matrix multi(Matrix a,Matrix b)           //矩阵乘法
{
    int sizen=a.sizen;
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(int i=1; i<=sizen; i++)
        for(int j=1; j<=sizen; j++)
            for(int k=1; k<=sizen; k++)
                c.m[i][j]=(c.m[i][j]+a.m[i][k]*b.m[k][j]+mod)%mod;
    return c;
}
Matrix quickpow(Matrix a,ll b)                   //矩阵二进制快速幂
{
    int sizen=a.sizen;
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(int i=1; i<=sizen; i++)                            //初始化为单位矩阵
        c.m[i][i]=1;
    while(b)
    {
        if(b&1)
            c=multi(c,a);                                       //快速幂
        a=multi(a,a);
        b>>=1;
    }
    return c;
}
Matrix qucikpow10(Matrix a,string b)           //矩阵十进制快速幂
{
    int sizen=a.sizen,len=b.length();
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(int i=1; i<=sizen; i++)                            //初始化为单位矩阵
        c.m[i][i]=1;
    for(int i=len-1; i>=0; i--)
    {
        c=multi(c,quickpow(a,(int)(b[i]-'0')));
        a=quickpow(a,10);
    }
    return c;
}
}
//Java BigInteger矩阵快速幂
/*
 public static BigInteger[][] multi(BigInteger [][]a,BigInteger [][]b)
    {
        BigInteger [][]c= new BigInteger[3][3];
        for(int i=1;i<=2;i++)
            for(int j=1;j<=2;j++)
                c[i][j]=BigInteger.ZERO;
        for(int i=1;i<=2;i++)
            for(int j=1;j<=2;j++)
                for(int k=1;k<=2;k++) {
                    c[i][k] = c[i][k].add(a[i][j].multiply(b[j][k]));
                }
        return c;
    }
    public static BigInteger qp(int b)
    {
        BigInteger ans[][]=new BigInteger[3][3];
        BigInteger cnt[][]=new BigInteger[3][3];
        ans[1][1]=BigInteger.ONE;ans[1][2]=BigInteger.ZERO;
        ans[2][1]=BigInteger.ZERO;ans[2][2]=BigInteger.ONE;
        cnt[1][1]=BigInteger.ZERO;cnt[1][2]=BigInteger.ONE;
        cnt[2][1]=BigInteger.ONE;cnt[2][2]=BigInteger.ONE;
        while(b>0)
        {
            if(b%2==1)
                ans=multi(ans,cnt);
            cnt=multi(cnt,cnt);
            b>>=1;
        }
        return ans[2][2];
    }
*/



//逆元
ll quick(ll a,ll b,ll m)
{
    ll ans=1;
    while(b)
    {
        if(b&1)
        {
            ans=(a*ans)%m;
        }
        a=(a*a)%m;
        b>>=1;
    }
    return ans;
}
inline ll inverse(ll a,ll p){return quick(a,p-2,p);}       //模素数求逆元


//组合数取模
ll comp(ll a,ll b,ll m)                                    //组合数取模
{
    if(a<b) return 0;
    if(a==b) return 1;
    else b=min(b,a-b);
    ll ans=1,ca=1,cb=1;
    for(int i=0;i<b;i++)
    {
        ca=(ca*(a-i))%m;
        cb=(cb*(b-i))%m;
    }
    ans=(ca*inverse(cb,m))%m;
    return ans;
}


//Lucas定理
ll lucas(ll a,ll b,ll p)                   //Lucas定理
{
	if(a<b)
		return 0;
    ll ans=1;
    while(a&&b)
    {
        ans=(ans*(comp(a%p,b%p,p)))%p;
        a/=p; b/=p;
    }
    return ans;
}

//Kummer定理
//设m，n为正整数，p为素数，则C(m+n,m)含p的幂次等于m+n在p进制下的进位次数。
//设m，n为正整数，p为素数，则C(n,m)含p的幂次等于n-m在p进制下的借位次数。
//https://www.luogu.org/blog/i207M/kummer-ding-li-shuo-lun-xue-xi-bi-ji





//求解二次同余方程
int read()
{
    int sum=0,f=1;char ch=getchar();
    while(ch<'0'||ch>'9'){if(ch='-')f=-1;ch=getchar();}
    while(ch>='0'&&ch<='9'){sum=sum*10+ch-'0';ch=getchar();}
    return sum*f;
}

int n,p,w;
namespace Quadratic_Congruence_Equation
{
#define random(a,b) (rand()%(b-a+1)+a)
#define int long long
bool flag;             //flag记录是否有解
struct Complex
{
    int x,y;
    Complex (int xx=0,int yy=0)
    {
        x=xx;
        y=yy;
    }
};
Complex operator *(Complex a,Complex b)
{
    return Complex(((a.x*b.x%p+w*a.y%p*b.y%p)%p+p)%p,((a.x*b.y%p+a.y*b.x%p)%p+p)%p);
}
Complex ksm(Complex x,int b)
{
    Complex tmp(1,0);
    while(b)
    {
        if(b&1)tmp=tmp*x;
        b>>=1;
        x=x*x;
    }
    return tmp;
}
int ksm(int x,int b)
{
    int tmp=1;
    while(b)
    {
        if(b&1)tmp=tmp*x%p;
        b>>=1;
        x=x*x%p;
    }
    return tmp;
}
int work(int n)          //求解x^2=n(mod p) 返回一个ans,另一个解为p-ans
{
    if(p==2)return n;
    if(ksm(n,(p-1)/2)+1==p)flag=true;   //勒让德符号判断是否有解
    int a;
    while(233)
    {
        a=random(0,p-1);
        w=((a*a-n)%p+p)%p;
        if(ksm(w,(p-1)/2)+1==p)break;
    }
    Complex res(a,1);
    Complex ans(0,0);
    ans=ksm(res,(p+1)/2);
    return ans.x;
}
#undef random(a,b)
#undef int
}
signed main()
{
    srand((unsigned)time(NULL));   //初始化随机数  调用work求解方程
    //如果是ax^2+bx+c=0(mod p) 左右乘4可以得 (2ax-b)^2=b^2-4c(mod 4p)
    //如果gcd(4,p)==1可以消去4
    p=read();n=read();
    n%=p;
    int ans1=work(n);
    int ans2=p-ans1;
    if(flag){printf("No Solution\n");return 0;}
    if(ans1==ans2)printf("%lld\n",ans1);
    else printf("%lld %lld",min(ans1,ans2),max(ans1,ans2));
    return 0;
}





//CSL的二次同余方程的解的板子
#include <bits/stdc++.h>
 
typedef long long i64;
typedef unsigned long long u64;
 
const int MOD = 1000000007;
 
i64 q1, q2, w;
struct P {
    i64 x, y;
};
 
P pmul(const P& a, const P& b, i64 p) {
    P res;
    res.x = (a.x * b.x + a.y * b.y % p * w) % p;
    res.y = (a.x * b.y + a.y * b.x) % p;
    return res;
}
i64 bin(i64 x, i64 n, i64 MOD) {
    i64 ret = MOD != 1;
    for (x %= MOD; n; n >>= 1, x = x * x % MOD)
        if (n & 1) ret = ret * x % MOD;
    return ret;
}
P bin(P x, i64 n, i64 MOD) {
    P ret = {1, 0};
    for (; n; n >>= 1, x = pmul(x, x, MOD))
        if (n & 1) ret = pmul(ret, x, MOD);
    return ret;
}
i64 Legendre(i64 a, i64 p) { return bin(a, (p - 1) >> 1, p); }
 
std::mt19937_64 g(std::clock());
i64 equation_solve(i64 b, i64 p) {
    if(b == 0) return 0;
    if (p == 2) return 1;
    if ((Legendre(b, p) + 1) % p == 0)
        return -1;
    i64 a;
    while (true) {
        a = g() % p;
        w = ((a * a - b) % p + p) % p;
        if ((Legendre(w, p) + 1) % p == 0)
            break;
    }
    return bin({a, 1}, (p + 1) >> 1, p).x;
}
 
i64 pow64(i64 a, i64 b) {
     
    if(b <= 0) return 1;
    i64 result = 1;
    do {
         
        if(b & 1) result = result * a % MOD;
        a = a * a % MOD;
        b >>= 1;
         
    } while(b);
     
    return result;
     
}
i64 inv64(i64 x) { return pow64(x, MOD - 2); }
 
int t;
int b, c;
 
const i64 INV2 = inv64(2);
 
int main() {
     
    std::scanf("%d", &t);
    for(int test = 0; test < t; ++test) {
         
        std::scanf("%d%d", &b, &c);
         
        i64 d = equation_solve(((i64(b) * b - i64(4) * c) % MOD + MOD) % MOD, MOD);
        if(d < 0) std::puts("-1 -1");
        else {
             
            i64 x = (b + d) * INV2 % MOD, y = ((b - d) * INV2 % MOD + MOD) % MOD;
            if(x > y) std::swap(x, y);
            if((x + y) % MOD == b && (x * y) % MOD == c) std::printf("%lld %lld\n", x, y);
            else std::puts("-1 -1");
             
        }
         
    }
     
}






//求高次同余方程的所有解   x^a=b(mod p)
#include<bits/stdc++.h>
using namespace std;
#define int long long
const int N=201010;
unordered_map<int,int> hsh;
int prime[N],tot,T,phi,ans[N],num,a,b,p,g,x,y;
int read()
{
    int sum=0,f=1;
    char ch=getchar();
    while(ch<'0'||ch>'9')
    {
        if(ch='-')f=-1;
        ch=getchar();
    }
    while(ch>='0'&&ch<='9')
    {
        sum=sum*10+ch-'0';
        ch=getchar();
    }
    return sum*f;
}
void work(int x)
{
    int tmp=x;
    tot=0;
    for(int i=2; i*i<=x; i++)
    {
        if(tmp%i==0)
        {
            prime[++tot]=i;
            while(tmp%i==0)tmp/=i;
        }
    }
    if(tmp>1)prime[++tot]=tmp;
}
int ksm(int x,int b)
{
    int tmp=1;
    while(b)
    {
        if(b&1)tmp=tmp*x%p;
        b>>=1;
        x=x*x%p;
    }
    return tmp;
}
int BSGS(int a,int b)
{
    hsh.clear();
    int block=sqrt(p)+1;
    int tmp=b;
    for(int i=0; i<block; i++,tmp=tmp*a%p)hsh[tmp]=i;
    a=ksm(a,block);
    tmp=1;
    for(int i=0; i<=block; i++,tmp=tmp*a%p)
    {
        if(hsh.count(tmp)&&i*block-hsh[tmp]>=0)return i*block-hsh[tmp];
    }
}
int exgcd(int &x,int &y,int a,int b)
{
    if(b==0)
    {
        x=1;
        y=0;
        return a;
    }
    int gcd=exgcd(x,y,b,a%b);
    int z=x;
    x=y;
    y=z-(a/b)*y;
    return gcd;
}
void solve(int a,int b,int p)      //求x^a=b(mod p)的所有解
{
    b%=p;
    phi=p-1;
    work(phi);
    for(int i=1; i<=p; i++)
    {
        bool flag=false;
        for(int j=1; j<=tot; j++)
        {
            if(ksm(i,phi/prime[j])==1)
            {
                flag=true;
                break;
            }
        }
        if(flag==false)
        {
            g=i;
            break;
        }
    }
    int r=BSGS(g,b);
    int gcd=exgcd(x,y,a,phi);
    if(r%gcd!=0)
    {
        printf("No Solution\n");
        return;
    }
    x=x*r/gcd;
    int k=phi/gcd;
    x=(x%k+k)%k;
    num=0;
    while(x<phi)
    {
        ans[++num]=ksm(g,x),x+=k;
    }
    sort(ans+1,ans+1+num);
    for(int i=1; i<=num; i++)
        printf("%lld ",ans[i]);
    printf("\n");
}
signed main()
{
    T=read();
    while(T--)
    {
        p=read(),a=read(),b=read();
        solve(a,b,p);
    }
    return 0;
}


//   https://www.cnblogs.com/Xu-daxia/p/10246664.html





//BSGS         求y^x=z(mod p)的最小非负 x    其中p为素数
void BSGS(int y,int z,int p)     //令x=am-b   其中m=sqrt(p)时复杂度最优
{
    //y^(am)=z*y^b(modp)
    if(y%p==0)                 //特判gcd(y,p)!=1时的情况
    {
        cout<<"Orz, I cannot find x!"<<endl;
        return ;
    }
    y%=p,z%=p;
    if(z==1)
    {
        cout<<0<<endl;
        return;
    }
    map<ll,ll> mp;                 //用来存z*y^b(modp)的值 也可以自己写hash
    mp.clear();
    int m=sqrt(p)+1;
    ll temp=quick(y,m,p),b=z;
    for(int i=0; i<m; i++,b=b*y%p)           //存z*y^b(modp)
    {
        mp[b]=i;
    }
    ll sum=temp;
    for(int a=1; (a-1)*(m)<=p; a++)   //用来验证左式是否存在和右式相等的值
    {
        if(mp.count(sum))
        {
            ll ans=a*m-mp[sum];
            cout<<ans<<endl;
            return;
        }
        sum=(sum*temp)%p;
    }
    cout<<"Orz, I cannot find x!"<<endl;
}
//PS:洛谷上可以过 poj上T飞了





//手写Hash表   
const int HashMod=100007;
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






//BSGS  yyb巨佬 poj2417
namespace BSGS
{
const int HashMod=100007;
struct HashTable
{
    struct Line
    {
        int u,v,next;
    } e[1000000];
    int h[HashMod],cnt;
    void Add(int u,int v,int w)
    {
        e[++cnt]=(Line)
        {
            w,v,h[u]
        };
        h[u]=cnt;
    }
    void Clear()
    {
        memset(h,0,sizeof(h));
        cnt=0;
    }
    void Hash(int x,int k)
    {
        int s=x%HashMod;
        Add(s,k,x);
    }
    int Query(int x)
    {
        int s=x%HashMod;
        for(int i=h[s]; i; i=e[i].next)
            if(e[i].u==x)return e[i].v;
        return -1;
    }
} Hash;
int fpow(int a,int b,int MOD)
{
    int s=1;
    while(b){if(b&1)s=1ll*s*a%MOD;a=1ll*a*a%MOD;b>>=1;}
    return s;
}
//BSGS(yyb巨佬)
void solve(int y,int z,int p)
{
    if(y%p==0)
    {
        printf("no solution\n");
        return;
    }
    y%=p;
    z%=p;
    if(z==1)
    {
        puts("0");
        return;
    }
    int m=sqrt(p)+1;
    Hash.Clear();
    for(register int i=0,t=z; i<m; ++i,t=1ll*t*y%p)Hash.Hash(t,i);
    for(register int i=1,tt=fpow(y,m,p),t=tt; i<=m+1; ++i,t=1ll*t*tt%p)
    {
        int k=Hash.Query(t);
        if(k==-1)continue;
        printf("%d\n",i*m-k);
        return;
    }
    printf("no solution\n");
}
}






//扩展BSGS (yyb巨佬)
namespace exBSGS
{
const int HashMod=100007;
struct HashTable
{
    struct Line
    {
        int u,v,next;
    } e[1000000];
    int h[HashMod],cnt;
    void Add(int u,int v,int w)
    {
        e[++cnt]=(Line)
        {
            w,v,h[u]
        };
        h[u]=cnt;
    }
    void Clear()
    {
        memset(h,0,sizeof(h));
        cnt=0;
    }
    void Hash(int x,int k)
    {
        int s=x%HashMod;
        Add(s,k,x);
    }
    int Query(int x)
    {
        int s=x%HashMod;
        for(int i=h[s]; i; i=e[i].next)
            if(e[i].u==x)return e[i].v;
        return -1;
    }
} Hash;
int fpow(int a,int b,int MOD)
{
    int s=1;
    while(b){if(b&1)s=1ll*s*a%MOD;a=1ll*a*a%MOD;b>>=1;}
    return s;
}
void NoAnswer(){puts("No Solution");}
//exBSGS(yyb巨佬)
void solve(int y,int z,int p)         //y^x=z(modp) p不一定为素数
{
    if(z==1){puts("0");return;}
    int k=0,a=1;
    while(233)
    {
        int d=__gcd(y,p);if(d==1)break;
        if(z%d){NoAnswer();return;}
        z/=d;p/=d;++k;a=1ll*a*y/d%p;
        if(z==a){printf("%d\n",k);return;}
    }
    Hash.Clear();
    int m=sqrt(p)+1;
    for(int i=0,t=z;i<m;++i,t=1ll*t*y%p)Hash.Hash(t,i);
    for(int i=1,tt=fpow(y,m,p),t=1ll*a*tt%p;i<=m;++i,t=1ll*t*tt%p)
    {
        int B=Hash.Query(t);if(B==-1)continue;
        printf("%d\n",i*m-B+k);return;
    }
    NoAnswer();
}
}







/*
Pell方程(求形如x*x-d*y*y=1的通解。)
佩尔方程x*x-d*y*y=1，当d不为完全平方数时，有无数个解，并且知道一个解可以推其他解。 如果d为完全平方数时，可知佩尔方程无解。

假设(x0,y0)是最小正整数解。

则：

xn=xn-1 * x0 + d* yn-1 *y0

yn=xn-1 * y0 + yn-1 * x0

证明只需代入。 如果忘记公式可以自己用(x0*x0-d*y0*y0)*(x1*x1-d*y1*y1)=1 推。

这样只要暴力求出最小特解，就可以用快速幂求出任意第K个解。
*/














//数论筛法：

//线性筛素数
int vis[maxn],prime[maxn],tol;
void liner_shai()
{
    memset(vis,0,sizeof(vis));
    for(int i=2;i<maxn;i++)
    {
        if(!vis[i])
            prime[tol++]=i;
        for(int j=0;j<tol&&i*prime[j]<maxn;j++)
        {
            vis[i*prime[j]]=1;
            if(i%prime[j]==0)
            {
                break;
            }
        }
    }
}



//num的一维存素因子，一维存对应素因子的次数次数
int num[2][maxn],cnt;
void fenjie_num(ll x)                //唯一标准分解
{
    cnt=0;
    for(int i=0; i<tol&&prime[i]<=x/prime[i]; i++)
    {
        if(x%prime[i]==0)
        {
            int temp=0;
            while(x%prime[i]==0)
            {
                x/=prime[i];
                temp++;
            }
            num[0][cnt]=prime[i];
            num[1][cnt++]=temp;
        }
    }
    if(x!=1)
    {
        num[0][cnt]=x;
        num[1][cnt++]=1;
    }
}




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




//线性筛莫比乌斯函数
int mu[maxn];
int vis[maxn];
int prim[maxn];
void get_mu(int n)
{
    int cnt=0;
    mu[1]=1;
    for(int i=2;i<=n;i++)
    {
        if(!vis[i])
        {
            prim[cnt++]=i;
            mu[i]=-1;
        }
        for(int j=0;j<cnt&&i*prim[j]<=n;j++)
        {
            vis[i*prim[j]]=1;
            if(i%prim[j]==0)
            {
                break;
            }
            else
            {
                mu[i*prim[j]]=-mu[i];
            }
        }
    }
}




//单点求值 Eular函数
ll Euler(ll x)
{
    ll ans = x, m = (ll)sqrt(x*1.0)+1;
    for(int i = 2; i < m; ++i)
    {
        if(x%i == 0)
        {
            ans = ans / i * (i-1);
            while(x%i == 0) x /= i;
        }
    }
    if(x > 1) ans = ans / x * (x-1);
    return ans;
}




//线性筛Eular函数
int vis[maxn],prime[maxn],tol;
ll phi[maxn];
void get_phi(int n)
{
    memset(vis,0,sizeof(vis));
    phi[1]=1;
    for(int i=2; i<=n; i++)
    {
        if(!vis[i])
        {
            prime[tol++]=i;
            phi[i]=(i-1);
        }
        for(int j=0; j<tol&&i*prime[j]<=n; j++)
        {
            vis[i*prime[j]]=1;
            if(i%prime[j]==0)
            {
                phi[i*prime[j]]=prime[j]*phi[i];
                break;
            }
            else
            {
                phi[i*prime[j]]=phi[i]*(prime[j]-1);
            }
        }
    }
}





//线性筛因数个数
int d[maxn];
int num_d[maxn];
void get_d(int n)
{
    memset(prime,0,sizeof(prime));
    memset(vis,0,sizeof(vis));
    int cnt=0;
    d[1]=1;
    for(int i=2; i<=n; i++)
    {
        if(!vis[i])
        {
            prime[++cnt]=i;
            d[i]=2;
            num_d[i]=1;
        }
        for(int j=1; j<=cnt&&prime[j]*i<=n; j++)
        {
            vis[prime[j]*i]=1;
            if(i%prime[j]==0)
            {
                num_d[i*prime[j]]=num_d[i]+1;
                d[i*prime[j]]=d[i]/(num_d[i]+1)*(num_d[i]+2);
                break;
            }
            else
            {
                num_d[i*prime[j]]=1;
                d[i*prime[j]]=d[i]*2;
            }
        }
    }
}






//米勒 罗宾算法模版 ==============//
LL prime[6] = {2, 3, 5, 233, 331};           //这个地方不是随机的
LL qu(LL x, LL y, LL mod) {
    LL ret = 0;
    while(y) {
        if(y & 1)
            ret = (ret + x) % mod;
        x = x * 2 % mod;
        y >>= 1;
    }
    return ret;
}
LL qpow(LL a, LL n, LL mod) {
    LL ret = 1;
    while(n) {
        if(n & 1) ret = qu(ret, a, mod);
        a = qu(a, a, mod);
        n >>= 1;
    }
    return ret;
}
bool MR(LL p) {
    if(p < 2) return 0;
    if(p != 2 && p % 2 == 0) return 0;
    LL s = p - 1;
    while(! (s & 1)) s >>= 1;
    for(int i = 0; i < 5; ++i) {
        if(p == prime[i]) return 1;
        LL t = s, m = qpow(prime[i], s, p);
        while(t != p - 1 && m != 1 && m != p - 1) {
            m = qu(m, m, p);
            t <<= 1;
        }
        if(m != p - 1 && !(t & 1)) return 0;
    }
    return 1;
}

//这个板子有点快 2018宁夏B






//MR素数测试
// Miller_Rabin + Pollard_rho

const int S=20;
mt19937 rd(time(0));
ll mul2(ll a,ll b,ll p)
{
	ll res=0;
	while(b)
	{
		if(b&1) res=(res+a)%p;
		a=(a+a)%p;
		b>>=1;
	}
	return res;
}
ll pow2(ll a,ll b,ll p)
{
	ll res=1;
	while(b)
	{
		if(b&1) res=mul2(res,a,p);
		a=mul2(a,a,p);
		b>>=1;
	}
	return res;
}
int check(ll a,ll n,ll x,ll t)//一定是合数返回1,不一定返回0
{
	ll now,nex,i;
	now=nex=pow2(a,x,n);
	for(i=1;i<=t;i++)
	{
		now=mul2(now,now,n);
		if(now==1&&nex!=1&&nex!=n-1) return 1;
		nex=now;
	}
	if(now!=1) return 1;
	return 0;
}
int Miller_Rabin(ll n)
{
	if(n<2) return 0;
	if(n==2) return 1;
	if((n&1)==0) return 0;
	ll x,t,i;
	x=n-1;
	t=0;
	while((x&1)==0) x>>=1,t++;
	for(i=0;i<S;i++)
	{
		if(check(rd()%(n-1)+1,n,x,t)) return 0;
	}
	return 1;
}
ll Pollard_rho(ll x,ll c)
{
	ll i,k,g,t,y;
	i=1;
	k=2;
	y=t=rd()%x;
	while(1)
	{
		i++;
		t=(mul2(t,t,x)+c)%x;
		g=__gcd(y-t+x,x);
		if(g!=1&&g!=x) return g;
		if(y==t) return x;
		if(i==k)
		{
			y=t;
			k+=k;
		}
	}
}
vector<ll> fac;
void findfac(ll n)
{
	if(Miller_Rabin(n))
	{
		fac.pb(n);
		return;
	}
	ll t=n;
	while(t>=n) t=Pollard_rho(t,rd()%(n-1)+1);
	findfac(t);
	findfac(n/t);
}
void work(ll x)
{
	fac.clear();
	findfac(x);
}








//异或运算算法：

//线性基
ll p[maxn];
ll a[maxn];
void get_linear_ji(ll n)            //线性基O(62*62*n)
{
    for(ll i=0;i<n;i++)
    {
        for(ll j=62;j>=0;j--)
        {
            if((a[i]>>j))
            {
               if(p[j])
               {
                   a[i]^=p[j];
               }
               else
               {
                   p[j]=a[i];
                   break;
               }
            }
        }
    }
    /* int r=0;
    for (int j=0;j<=62;j++) if (P[j]) r++;             //返回秩 线性基的个数
    return r; */
}





//网友写的很好的线性基模板
struct Linear_Basis
{
    LL b[63],nb[63],tot;

    Linear_Basis()
    {
        tot=0;
        memset(b,0,sizeof(b));
        memset(nb,0,sizeof(nb));
    }
    void clear()
    {
        tot=0;
        memset(b,0,sizeof(b));
        memset(nb,0,sizeof(nb));
    }
    bool ins(LL x)                  //插入一个数
    {
        for(int i=62;i>=0;i--)
            if (x&(1LL<<i))
            {
                if (!b[i]) {b[i]=x;break;}
                x^=b[i];
            }
        return x>0;
    }

    LL Max(LL x)                  //求异或最大值
    {
        LL res=x;
        for(int i=62;i>=0;i--)
            res=max(res,res^b[i]);
        return res;
    }

    LL Min(LL x)                 //求异或最小值
    {
        LL res=x;
        for(int i=0;i<=62;i++)
            if (b[i]) res^=b[i];
        return res;
    }

    void rebuild()                //重构化为对角矩阵 为求Kth_MAX做铺垫
    {
        for(int i=62;i>=0;i--)
            for(int j=i-1;j>=0;j--)
                if (b[i]&(1LL<<j)) b[i]^=b[j];
        for(int i=0;i<=62;i++)
            if (b[i]) nb[tot++]=b[i];
    }

    LL Kth_Min(LL k)             //第K小   在调用之前必须rebuild
    {
    	rebuild();
        LL res=0;
        for(int i=62;i>=0;i--)
            if (k&(1LL<<i)) res^=nb[i];
        return res;
    }

} LB;

Linear_Basis merge(const Linear_Basis &n1,const Linear_Basis &n2)   //暴力合并两个线性基
{
    Linear_Basis ret=n1;
    for (int i=62; i>=0; i--)
        if (n2.b[i])
            ret.ins(n2.b[i]);
    return ret;
}






//多项式乘法： FFT NTT FWT

//普通FFT 自己写的递归形式FFT，可能不太好用（极其不好用）
//SPOJ TSUM因不明原因运行错误（段错误/爆内存）
const ld pi=acos(-1);

struct Complex
{
    ld r,i;
    Complex(ld rr=0,ld ii=0):r(rr),i(ii){}
    friend Complex operator+(Complex a,Complex b){return Complex(a.r+b.r,a.i+b.i);}
    friend Complex operator-(Complex a,Complex b){return Complex(a.r-b.r,a.i-b.i);}
    friend Complex operator*(Complex a,Complex b){return Complex(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r);}
};


void FFT(int n,Complex *c,int type)
{
    if(n==1)return;
    Complex a1[n>>1],a2[n>>1];
    for(int i=0;i<=n;i+=2)
    {
        a1[i>>1]=c[i];
        a2[i>>1]=c[i+1];
    }
    FFT(n>>1,a1,type);
    FFT(n>>1,a2,type);
    Complex Wn=Complex(cos(2.0*pi/n),type*sin(2.0*pi/n)),w=Complex(1,0);
    for(int i=0;i<(n>>1);i++,w=w*Wn)
    {
        c[i]=a1[i]+w*a2[i];
        c[i+(n>>1)]=a1[i]-w*a2[i];
    }
}








//由kuangbin的FFT板子改编的板子


//复数结构体
struct Complex
{
    double x,y;//实部和虚部 x+yi
    Complex(double _x = 0.0,double _y = 0.0){x = _x;y = _y;}
    Complex operator -(const Complex &b)const{return Complex(x-b.x,y-b.y);}
    Complex operator +(const Complex &b)const{return Complex(x+b.x,y+b.y);}
    Complex operator *(const Complex &b)const{return Complex(x*b.x-y*b.y,x*b.y+y*b.x);}
};
/*
* 进行FFT和IFFT前的反转变换。
* 位置i和 （i二进制反转后位置）互换
* len必须取2的幂
*/
void change(Complex y[],int len)
{
    int i,j,k;
    for(i = 1, j = len/2; i <len-1; i++)
    {
        if(i < j)swap(y[i],y[j]);
//交换互为小标反转的元素，i<j保证交换一次
//i做正常的+1，j左反转类型的+1,始终保持i和j是反转的
        k = len/2;
        while(j >= k)
        {
            j -= k;
            k /= 2;
        }
        if(j < k)j += k;
    }
}
/*
* 做FFT
* len必须为2^k形式，
* on==1时是DFT，on==-1时是IDFT
*/
void fft(Complex y[],int len,int on)
{
    change(y,len);
    for(int h = 2; h <= len; h <<= 1)
    {
        Complex wn(cos(-on*2*PI/h),sin(-on*2*PI/h));
        for(int j = 0; j < len; j+=h)
        {
            Complex w(1,0);
            for(int k = j; k < j+h/2; k++)
            {
                Complex u = y[k];
                Complex t = w*y[k+h/2];
                y[k] = u+t;
                y[k+h/2] = u-t;
                w = w*wn;
            }
        }
    }
    if(on == -1)
        for(int i = 0; i < len; i++)
            y[i].x /= len;
}
Complex x1[maxn],x2[maxn];
int Multiply(int *a,int len1,const int *b,int len2)   //这里的len1和len2为数组的长度
{                                                    //在传参的时候记得len+1
    int len = 1;
    while(len < len1*2 || len < len2*2)len<<=1;
    for(int i=0; i<len1; i++)
        x1[i]=Complex(a[i],0);
    for(int i=len1; i<len; i++)
        x1[i]=Complex(0,0);
    for(int i=0; i<len2; i++)
        x2[i]=Complex(b[i],0);
    for(int i=len2; i<len; i++)
        x2[i]=Complex(0,0);
    fft(x1,len,1);
    fft(x2,len,1);
    for(int i = 0; i < len; i++)
        x1[i] = x1[i]*x2[i];
    fft(x1,len,-1);
    for(int i = 0; i < len; i++)
        a[i] = (int)(x1[i].x+0.5);
    len=len1+len2-1;
    while(a[len] <= 0 && len > 0)len--;
    return len;         //返回的len为数组的最后一个数的下标，是数组的长度-1
}
int a1[maxn],a11[maxn],a2[maxn],a3[maxn];
//玄学开maxn










//还是FFT的板子，代码比较简洁，非递归写法，不会炸
namespace FFT
{
    struct Complex
    {
        double r,i;
        Complex(double real=0.0,double image=0.0)
        {
            r=real; i=image;
        }
        Complex operator + (const Complex o)
        {
            return Complex(r+o.r,i+o.i);
        }
        Complex operator - (const Complex o)
        {
            return Complex(r-o.r,i-o.i);
        }
        Complex operator * (const Complex o)
        {
            return Complex(r*o.r-i*o.i,r*o.i+i*o.r);
        }
    };

    void brc(Complex *y, int l)
    {
        register int i,j,k;
        for( i = 1, j = l / 2; i < l - 1; i++)
        {
            if (i < j) swap(y[i], y[j]);
            k = l / 2; while ( j >= k) j -= k,k /= 2;
            if (j < k) j += k;
        }
    }

    void FFT(Complex *y, int len, double on)
    {
        register int h, j, k;
        Complex u, t; brc(y, len);
        for(h = 2; h <= len; h <<= 1)
        {
            Complex wn(cos(on * 2 * PI / h), sin(on * 2 * PI / h));
            for(j = 0; j < len; j += h)
            {
                Complex w(1, 0);
                for(k = j; k < j + h / 2; k++)
                {
                    u = y[k]; t = w * y[k + h / 2];
                    y[k] = u + t; y[k + h / 2] = u - t;
                    w = w * wn;
                }
            }
        }
        if (on<0) for (int i = 0; i < len; i++) y[i].r/=len;
    }

}

FFT::Complex A[N],B[N],C[N],ans[N];







//FFT Tourist神仙的板子
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define MAXN 100005
#define INF 1000000000
#define MOD 998244353
#define F first
#define S second
using namespace std;
typedef long long ll;
typedef pair<int,int> P;
int n,k;
const double PI=acos(-1.0);
//tourist
namespace fft
{
    struct num
    {
        double x,y;
        num() {x=y=0;}
        num(double x,double y):x(x),y(y){}
    };
    inline num operator+(num a,num b) {return num(a.x+b.x,a.y+b.y);}
    inline num operator-(num a,num b) {return num(a.x-b.x,a.y-b.y);}
    inline num operator*(num a,num b) {return num(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
    inline num conj(num a) {return num(a.x,-a.y);}

    int base=1;
    vector<num> roots={{0,0},{1,0}};
    vector<int> rev={0,1};
    const double PI=acosl(-1.0);

    void ensure_base(int nbase)
    {
        if(nbase<=base) return;
        rev.resize(1<<nbase);
        for(int i=0;i<(1<<nbase);i++)
            rev[i]=(rev[i>>1]>>1)+((i&1)<<(nbase-1));
        roots.resize(1<<nbase);
        while(base<nbase)
        {
            double angle=2*PI/(1<<(base+1));
            for(int i=1<<(base-1);i<(1<<base);i++)
            {
                roots[i<<1]=roots[i];
                double angle_i=angle*(2*i+1-(1<<base));
                roots[(i<<1)+1]=num(cos(angle_i),sin(angle_i));
            }
            base++;
        }
    }

    void fft(vector<num> &a,int n=-1)
    {
        if(n==-1) n=a.size();
        assert((n&(n-1))==0);
        int zeros=__builtin_ctz(n);
        ensure_base(zeros);
        int shift=base-zeros;
        for(int i=0;i<n;i++)
            if(i<(rev[i]>>shift))
                swap(a[i],a[rev[i]>>shift]);
        for(int k=1;k<n;k<<=1)
        {
            for(int i=0;i<n;i+=2*k)
            {
                for(int j=0;j<k;j++)
                {
                    num z=a[i+j+k]*roots[j+k];
                    a[i+j+k]=a[i+j]-z;
                    a[i+j]=a[i+j]+z;
                }
            }
        }
    }

    vector<num> fa,fb;

    vector<int> multiply(vector<int> &a, vector<int> &b)
    {
        int need=a.size()+b.size()-1;
        int nbase=0;
        while((1<<nbase)<need) nbase++;
        ensure_base(nbase);
        int sz=1<<nbase;
        if(sz>(int)fa.size()) fa.resize(sz);
        for(int i=0;i<sz;i++)
        {
            int x=(i<(int)a.size()?a[i]:0);
            int y=(i<(int)b.size()?b[i]:0);
            fa[i]=num(x,y);
        }
        fft(fa,sz);
        num r(0,-0.25/sz);
        for(int i=0;i<=(sz>>1);i++)
        {
            int j=(sz-i)&(sz-1);
            num z=(fa[j]*fa[j]-conj(fa[i]*fa[i]))*r;
            if(i!=j) fa[j]=(fa[i]*fa[i]-conj(fa[j]*fa[j]))*r;
            fa[i]=z;
        }
        fft(fa,sz);
        vector<int> res(need);
        for(int i=0;i<need;i++) res[i]=fa[i].x+0.5;
        return res;
    }

    vector<int> multiply_mod(vector<int> &a,vector<int> &b,int m,int eq=0)
    {
        int need=a.size()+b.size()-1;
        int nbase=0;
        while((1<<nbase)<need) nbase++;
        ensure_base(nbase);
        int sz=1<<nbase;
        if(sz>(int)fa.size()) fa.resize(sz);
        for(int i=0;i<(int)a.size();i++)
        {
            int x=(a[i]%m+m)%m;
            fa[i]=num(x&((1<<15)-1),x>>15);
        }
        fill(fa.begin()+a.size(),fa.begin()+sz,num{0,0});
        fft(fa,sz);
        if(sz>(int)fb.size()) fb.resize(sz);
        if(eq) copy(fa.begin(),fa.begin()+sz,fb.begin());
        else
        {
            for(int i=0;i<(int)b.size();i++)
            {
                int x=(b[i]%m+m)%m;
                fb[i]=num(x&((1<<15)-1),x>>15);
            }
            fill(fb.begin()+b.size(),fb.begin()+sz,num{0,0});
            fft(fb,sz);
        }
        double ratio=0.25/sz;
        num r2(0,-1),r3(ratio,0),r4(0,-ratio),r5(0,1);
        for(int i=0;i<=(sz>>1);i++)
        {
            int j=(sz-i)&(sz-1);
            num a1=(fa[i]+conj(fa[j]));
            num a2=(fa[i]-conj(fa[j]))*r2;
            num b1=(fb[i]+conj(fb[j]))*r3;
            num b2=(fb[i]-conj(fb[j]))*r4;
            if(i!=j)
            {
                num c1=(fa[j]+conj(fa[i]));
                num c2=(fa[j]-conj(fa[i]))*r2;
                num d1=(fb[j]+conj(fb[i]))*r3;
                num d2=(fb[j]-conj(fb[i]))*r4;
                fa[i]=c1*d1+c2*d2*r5;
                fb[i]=c1*d2+c2*d1;
            }
            fa[j]=a1*b1+a2*b2*r5;
            fb[j]=a1*b2+a2*b1;
        }
        fft(fa,sz);fft(fb,sz);
        vector<int> res(need);
        for(int i=0;i<need;i++)
        {
            ll aa=fa[i].x+0.5;
            ll bb=fb[i].x+0.5;
            ll cc=fa[i].y+0.5;
            res[i]=(aa+((bb%m)<<15)+((cc%m)<<30))%m;
        }
        return res;
    }
    vector<int> square_mod(vector<int> &a,int m)
    {
        return multiply_mod(a,a,m,1);
    }
};
vector<int> v;
int main()
{
    scanf("%d%d",&n,&k);
    v.resize(10,0);
    for(int i=0;i<k;i++)
    {
        int x;
        scanf("%d",&x);
        v[x]=1;
    }
    vector<int> ans;
    ans.push_back(1);
    int p=n/2;
    while(p)
    {
        if(p&1) ans=fft::multiply_mod(ans,v,MOD);
        v=fft::square_mod(v,MOD);
        p>>=1;
    }
    int res=0;
    for(auto t:ans) res=(res+1LL*t*t)%MOD;
    printf("%d\n",res);
    return 0;
}








//暴力求解原根
int G(int s)
{
    int q[1010]={0};
    for(int i=2;i<=s-2;i++) if ((s-1)%i==0) q[++q[0]]=i;
    for (int i=2;;i++)
    {
        bool B=1;
        for (int j=1;j<=q[0]&&B;j++) if (quick(i,q[j],s)==1) B=0;
        if (B) return i;
    }
    return -1;
}









//Tourist
//NTT算法
namespace ntt
{
struct num
{
    double x, y;
    num()
    {
        x = y = 0;
    }
    num(double x, double y) : x(x), y(y) {}
};
inline num operator+(num a, num b)
{
    return num(a.x + b.x, a.y + b.y);
}
inline num operator-(num a, num b)
{
    return num(a.x - b.x, a.y - b.y);
}
inline num operator*(num a, num b)
{
    return num(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline num conj(num a)
{
    return num(a.x, -a.y);
}

int base = 1;
vector<num> roots = { { 0, 0 }, { 1, 0 } };
vector<int> rev = { 0, 1 };
const double PI = acosl(-1.0);

void ensure_base(int nbase)
{
    if (nbase <= base)
        return;
    rev.resize(1 << nbase);
    for (int i = 0; i < (1 << nbase); i++) rev[i] = (rev[i >> 1] >> 1) + ((i & 1) << (nbase - 1));
    roots.resize(1 << nbase);
    while (base < nbase)
    {
        double angle = 2 * PI / (1 << (base + 1));
        for (int i = 1 << (base - 1); i < (1 << base); i++)
        {
            roots[i << 1] = roots[i];
            double angle_i = angle * (2 * i + 1 - (1 << base));
            roots[(i << 1) + 1] = num(cos(angle_i), sin(angle_i));
        }
        base++;
    }
}

void fft(vector<num> &a, int n = -1)
{
    if (n == -1)
        n = a.size();
    assert((n & (n - 1)) == 0);
    int zeros = __builtin_ctz(n);
    ensure_base(zeros);
    int shift = base - zeros;
    for (int i = 0; i < n; i++)
        if (i < (rev[i] >> shift))
            swap(a[i], a[rev[i] >> shift]);
    for (int k = 1; k < n; k <<= 1)
    {
        for (int i = 0; i < n; i += 2 * k)
        {
            for (int j = 0; j < k; j++)
            {
                num z = a[i + j + k] * roots[j + k];
                a[i + j + k] = a[i + j] - z;
                a[i + j] = a[i + j] + z;
            }
        }
    }
}

vector<num> fa, fb;

vector<int> multiply(vector<int> &a, vector<int> &b)
{
    int need = a.size() + b.size() - 1;
    int nbase = 0;
    while ((1 << nbase) < need) nbase++;
    ensure_base(nbase);
    int sz = 1 << nbase;
    if (sz > (int)fa.size())
        fa.resize(sz);
    for (int i = 0; i < sz; i++)
    {
        int x = (i < (int)a.size() ? a[i] : 0);
        int y = (i < (int)b.size() ? b[i] : 0);
        fa[i] = num(x, y);
    }
    fft(fa, sz);
    num r(0, -0.25 / sz);
    for (int i = 0; i <= (sz >> 1); i++)
    {
        int j = (sz - i) & (sz - 1);
        num z = (fa[j] * fa[j] - conj(fa[i] * fa[i])) * r;
        if (i != j)
            fa[j] = (fa[i] * fa[i] - conj(fa[j] * fa[j])) * r;
        fa[i] = z;
    }
    fft(fa, sz);
    vector<int> res(need);
    for (int i = 0; i < need; i++) res[i] = fa[i].x + 0.5;
    return res;
}

vector<int> multiply(vector<int> &a, vector<int> &b, int m, int eq = 0)
{
    int need = a.size() + b.size() - 1;
    int nbase = 0;
    while ((1 << nbase) < need) nbase++;
    ensure_base(nbase);
    int sz = 1 << nbase;
    if (sz > (int)fa.size())
        fa.resize(sz);
    for (int i = 0; i < (int)a.size(); i++)
    {
        int x = (a[i] % m + m) % m;
        fa[i] = num(x & ((1 << 15) - 1), x >> 15);
    }
    fill(fa.begin() + a.size(), fa.begin() + sz, num{ 0, 0 });
    fft(fa, sz);
    if (sz > (int)fb.size())
        fb.resize(sz);
    if (eq)
        copy(fa.begin(), fa.begin() + sz, fb.begin());
    else
    {
        for (int i = 0; i < (int)b.size(); i++)
        {
            int x = (b[i] % m + m) % m;
            fb[i] = num(x & ((1 << 15) - 1), x >> 15);
        }
        fill(fb.begin() + b.size(), fb.begin() + sz, num{ 0, 0 });
        fft(fb, sz);
    }
    double ratio = 0.25 / sz;
    num r2(0, -1), r3(ratio, 0), r4(0, -ratio), r5(0, 1);
    for (int i = 0; i <= (sz >> 1); i++)
    {
        int j = (sz - i) & (sz - 1);
        num a1 = (fa[i] + conj(fa[j]));
        num a2 = (fa[i] - conj(fa[j])) * r2;
        num b1 = (fb[i] + conj(fb[j])) * r3;
        num b2 = (fb[i] - conj(fb[j])) * r4;
        if (i != j)
        {
            num c1 = (fa[j] + conj(fa[i]));
            num c2 = (fa[j] - conj(fa[i])) * r2;
            num d1 = (fb[j] + conj(fb[i])) * r3;
            num d2 = (fb[j] - conj(fb[i])) * r4;
            fa[i] = c1 * d1 + c2 * d2 * r5;
            fb[i] = c1 * d2 + c2 * d1;
        }
        fa[j] = a1 * b1 + a2 * b2 * r5;
        fb[j] = a1 * b2 + a2 * b1;
    }
    fft(fa, sz);
    fft(fb, sz);
    vector<int> res(need);
    for (int i = 0; i < need; i++)
    {
        ll aa = fa[i].x + 0.5;
        ll bb = fb[i].x + 0.5;
        ll cc = fa[i].y + 0.5;
        res[i] = (aa + ((bb % m) << 15) + ((cc % m) << 30)) % m;
    }
    return res;
}
vector<int> square(vector<int> &a, int m)
{
    return multiply(a, a, m, 1);
}
};









//yyb的FWT板子 orz
void FWT_or(int *a,int opt)
{
    for(int i=1; i<N; i<<=1)
        for(int p=i<<1,j=0; j<N; j+=p)
            for(int k=0; k<i; ++k)
                if(opt==1)a[i+j+k]=(a[j+k]+a[i+j+k])%MOD;
                else a[i+j+k]=(a[i+j+k]+MOD-a[j+k])%MOD;
}
void FWT_and(int *a,int opt)
{
    for(int i=1; i<N; i<<=1)
        for(int p=i<<1,j=0; j<N; j+=p)
            for(int k=0; k<i; ++k)
                if(opt==1)a[j+k]=(a[j+k]+a[i+j+k])%MOD;
                else a[j+k]=(a[j+k]+MOD-a[i+j+k])%MOD;
}
void FWT_xor(int *a,int opt)
{
    for(int i=1; i<N; i<<=1)
        for(int p=i<<1,j=0; j<N; j+=p)
            for(int k=0; k<i; ++k)
            {
                int X=a[j+k],Y=a[i+j+k];
                a[j+k]=(X+Y)%MOD;
                a[i+j+k]=(X+MOD-Y)%MOD;
                if(opt==-1)a[j+k]=1ll*a[j+k]*inv2%MOD,a[i+j+k]=1ll*a[i+j+k]*inv2%MOD;
            }
}








//FWT                sjf板子
namespace FWT
{
	ll inv2;//2对p的逆元 仅在xor FWT逆变换用到
	const ll p=1e9+7;          //若不需取模时请注意把逆元替换掉
	ll pow2(ll a,ll b)
	{
		ll res=1;
		while(b)
		{
			if(b&1) res=res*a%p;
			a=a*a%p;
			b>>=1;
		}
		return res;
	}
    //对a数组进行FWT变换 n必须为2^m形式
    //若f==1是xor_FWT,若f==2是and_FWT,若f==3是or_FWT
    //v=0时是正向FWT变换，v=1时是逆变换
	void fwt(ll *a,int n,int f,int v)
	{  
		for(int d=1;d<n;d<<=1)
		{
			for(int m=d<<1,i=0;i<n;i+=m)
			{
				for(int j=0;j<d;j++)
				{  
					ll x=a[i+j],y=a[i+j+d];
					if(!v)
					{
						if(f==1) a[i+j]=(x+y)%p,a[i+j+d]=(x-y+p)%p;//xor
						else if(f==2) a[i+j]=(x+y)%p;//and
						else if(f==3) a[i+j+d]=(x+y)%p;//or
					}
					else
					{
						if(f==1) a[i+j]=(x+y)*inv2%p,a[i+j+d]=(x-y+p)%p*inv2%p;//xor
						else if(f==2) a[i+j]=(x-y+p)%p;//and
						else if(f==3) a[i+j+d]=(y-x+p)%p;//or
					}
				}
			}
		}
	}
	
	//结果存在a 
	void XOR(ll *a,ll *b,int n)
	{
		int len;
		for(len=1;len<=n;len<<=1);
		fwt(a,len,1,0);
		fwt(b,len,1,0);
		for(int i=0;i<len;i++) a[i]=a[i]*b[i]%p;
		inv2=pow2(2,p-2);
		fwt(a,len,1,1);
	}
	void AND(ll *a,ll *b,int n)
	{
		int len;
		for(len=1;len<=n;len<<=1);
		fwt(a,len,2,0);
		fwt(b,len,2,0);
		for(int i=0;i<len;i++) a[i]=a[i]*b[i]%p;
		fwt(a,len,2,1);
	}
	void OR(ll *a,ll *b,int n)
	{
		int len;
		for(len=1;len<=n;len<<=1);
		fwt(a,len,3,0);
		fwt(b,len,3,0);
		for(int i=0;i<len;i++) a[i]=a[i]*b[i]%p;
		fwt(a,len,3,1);
	}
};









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












//求一维线性递推式的神仙算法：

//杜教BM

#include <bits/stdc++.h>

using namespace std;
#define rep(i,a,n) for (long long i=a;i<n;i++)
#define per(i,a,n) for (long long i=n-1;i>=a;i--)
#define pb push_back
#define mp make_pair
#define all(x) (x).begin(),(x).end()
#define fi first
#define se second
#define SZ(x) ((long long)(x).size())
typedef vector<long long> VI;
typedef long long ll;
typedef pair<long long,long long> PII;
const ll mod=1e9+7;
ll powmod(ll a,ll b) {ll res=1;a%=mod; assert(b>=0); for(;b;b>>=1){if(b&1)res=res*a%mod;a=a*a%mod;}return res;}
// head

long long _,n;
namespace linear_seq
{
    const long long N=10010;
    ll res[N],base[N],_c[N],_md[N];

    vector<long long> Md;
    void mul(ll *a,ll *b,long long k)
    {
        rep(i,0,k+k) _c[i]=0;
        rep(i,0,k) if (a[i]) rep(j,0,k)
            _c[i+j]=(_c[i+j]+a[i]*b[j])%mod;
        for (long long i=k+k-1;i>=k;i--) if (_c[i])
            rep(j,0,SZ(Md)) _c[i-k+Md[j]]=(_c[i-k+Md[j]]-_c[i]*_md[Md[j]])%mod;
        rep(i,0,k) a[i]=_c[i];
    }
    long long solve(ll n,VI a,VI b)
    { // a 系数 b 初值 b[n+1]=a[0]*b[n]+...
//        printf("%d\n",SZ(b));
        ll ans=0,pnt=0;
        long long k=SZ(a);
        assert(SZ(a)==SZ(b));
        rep(i,0,k) _md[k-1-i]=-a[i];_md[k]=1;
        Md.clear();
        rep(i,0,k) if (_md[i]!=0) Md.push_back(i);
        rep(i,0,k) res[i]=base[i]=0;
        res[0]=1;
        while ((1ll<<pnt)<=n) pnt++;
        for (long long p=pnt;p>=0;p--)
        {
            mul(res,res,k);
            if ((n>>p)&1)
            {
                for (long long i=k-1;i>=0;i--) res[i+1]=res[i];res[0]=0;
                rep(j,0,SZ(Md)) res[Md[j]]=(res[Md[j]]-res[k]*_md[Md[j]])%mod;
            }
        }
        rep(i,0,k) ans=(ans+res[i]*b[i])%mod;
        if (ans<0) ans+=mod;
        return ans;
    }
    VI BM(VI s)
    {
        VI C(1,1),B(1,1);
        long long L=0,m=1,b=1;
        rep(n,0,SZ(s))
        {
            ll d=0;
            rep(i,0,L+1) d=(d+(ll)C[i]*s[n-i])%mod;
            if (d==0) ++m;
            else if (2*L<=n)
            {
                VI T=C;
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                L=n+1-L; B=T; b=d; m=1;
            }
            else
            {
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                ++m;
            }
        }
        return C;
    }
    long long gao(VI a,ll n)
    {
        VI c=BM(a);
        c.erase(c.begin());
        rep(i,0,SZ(c)) c[i]=(mod-c[i])%mod;
        return solve(n,c,VI(a.begin(),a.begin()+SZ(c)));
    }
};

int main()
{
    while(~scanf("%I64d", &n))
    {
    	printf("%I64d\n",linear_seq::gao(VI{1,5,11,36,95,281,781,2245,6336,18061, 51205},n-1));
    }
}










//非求逆元的BM板子 (未验证)
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define rep(i,a,n) for(int i=a;i<n;i++)
namespace linear
{
    ll mo=1000000009;
    vector<ll> v;
    double a[105][105],del;
    int k;
    struct matrix
    {
        int n;
        ll a[50][50];
        matrix operator * (const matrix & b)const
        {
            matrix c;
            c.n=n;
            rep(i,0,n)rep(j,0,n)c.a[i][j]=0;
            rep(i,0,n)rep(j,0,n)rep(k,0,n)
            c.a[i][j]=(c.a[i][j]+a[i][k]*b.a[k][j]%mo)%mo;
            return c;
        }
    }A;
    bool solve(int n)
    {
        rep(i,1,n+1)
        {
            int t=i;
            rep(j,i+1,n+1)if(fabs(a[j][i])>fabs(a[t][i]))t=j;
            if(fabs(del=a[t][i])<1e-6)return false;
            rep(j,i,n+2)swap(a[i][j],a[t][j]);
            rep(j,i,n+2)a[i][j]/=del;
            rep(t,1,n+1)if(t!=i)
            {
                del=a[t][i];
                rep(j,i,n+2)a[t][j]-=a[i][j]*del;
            }
        }
        return true;
    }
    void build(vector<ll> V)
    {
        v=V;
        int n=(v.size()-1)/2;
        k=n;
        while(1)
        {
            rep(i,0,k+1)
            {
                rep(j,0,k)a[i+1][j+1]=v[n-1+i-j];
                a[i+1][k+1]=1;
                a[i+1][k+2]=v[n+i];
            }
            if(solve(n+1))break;
            n--;k--;
        }
        A.n=k+1;
        rep(i,0,A.n)rep(j,0,A.n)A.a[i][j]=0;
        rep(i,0,A.n)A.a[i][0]=(int)round(a[i+1][A.n+1]);
        rep(i,0,A.n-2)A.a[i][i+1]=1;
        A.a[A.n-1][A.n-1]=1;
    }
    void formula()
    {
        printf("f(n) =");
        rep(i,0,A.n-1)printf(" (%lld)*f(n-%d) +",A.a[i][0],i+1);
        printf(" (%lld)\n",A.a[A.n-1][0]);
    }
    ll cal(ll n)
    {
        if(n<v.size())return v[n];
        n=n-k+1;
        matrix B,T=A;
        B.n=A.n;
        rep(i,0,B.n)rep(j,0,B.n)B.a[i][j]=i==j?1:0;
        while(n)
        {
            if(n&1)B=B*T;
            n>>=1;
            T=T*T;
        }
        ll ans=0;
        rep(i,0,B.n-1)ans=(ans+v[B.n-2-i]*B.a[i][0]%mo)%mo;
        ans=(ans+B.a[B.n-1][0])%mo;
        while(ans<0)ans+=mo;
        return ans;
    }
}

int main()
{
//  vector<ll> V={1 ,4 ,9 ,16,25,36,49};
//  vector<ll> V={1 ,1 ,2 ,3 ,5 ,8 ,13};
//  vector<ll> V={2 ,2 ,3 ,4 ,6 ,9 ,14};
    vector<ll> V={1,1,1,3,5,9,17};//<-----
    linear::build(V);
    linear::formula();
    ll n;
    while(~scanf("%lld",&n))
    {
        printf("%lld\n",linear::cal(n-1));
    }
    return 0;
}









//CSL的BM板子 （非求逆元）最牛逼的板子，牛逼啊蔡老板
#include <bits/stdc++.h>
using namespace std;
#ifndef ONLINE_JUDGE
#define debug(fmt, ...) fprintf(stderr, "[%s] " fmt "\n", __func__, ##__VA_ARGS__)
#else
#define debug(...)
#endif
 
// given first m items init[0..m-1] and coefficents trans[0..m-1] or
// given first 2 *m items init[0..2m-1], it will compute trans[0..m-1]
// for you. trans[0..m] should be given as that
//      init[m] = sum_{i=0}^{m-1} init[i] * trans[i]
struct LinearRecurrence
{
    using int64 = long long;
    using vec = std::vector<int64>;
 
    static void extand(vec& a, size_t d, int64 value = 0)
    {
        if (d <= a.size()) return;
        a.resize(d, value);
    }
    static vec BerlekampMassey(const vec& s, int64 mod)
    {
        std::function<int64(int64)> inverse = [&](int64 a) {
            return a == 1 ? 1 : (int64)(mod - mod / a) * inverse(mod % a) % mod;
        };
        vec A = {1}, B = {1};
        int64 b = s[0];
        for (size_t i = 1, m = 1; i < s.size(); ++i, m++)
        {
            int64 d = 0;
            for (size_t j = 0; j < A.size(); ++j)
            {
                d += A[j] * s[i - j] % mod;
            }
            if (!(d %= mod)) continue;
            if (2 * (A.size() - 1) <= i)
            {
                auto temp = A;
                extand(A, B.size() + m);
                int64 coef = d * inverse(b) % mod;
                for (size_t j = 0; j < B.size(); ++j)
                {
                    A[j + m] -= coef * B[j] % mod;
                    if (A[j + m] < 0) A[j + m] += mod;
                }
                B = temp, b = d, m = 0;
            }
            else
            {
                extand(A, B.size() + m);
                int64 coef = d * inverse(b) % mod;
                for (size_t j = 0; j < B.size(); ++j)
                {
                    A[j + m] -= coef * B[j] % mod;
                    if (A[j + m] < 0) A[j + m] += mod;
                }
            }
        }
        return A;
    }
    static void exgcd(int64 a, int64 b, int64& g, int64& x, int64& y)
    {
        if (!b)
            x = 1, y = 0, g = a;
        else
        {
            exgcd(b, a % b, g, y, x);
            y -= x * (a / b);
        }
    }
    static int64 crt(const vec& c, const vec& m)
    {
        int n = c.size();
        int64 M = 1, ans = 0;
        for (int i = 0; i < n; ++i) M *= m[i];
        for (int i = 0; i < n; ++i)
        {
            int64 x, y, g, tm = M / m[i];
            exgcd(tm, m[i], g, x, y);
            ans = (ans + tm * x * c[i] % M) % M;
        }
        return (ans + M) % M;
    }
    static vec ReedsSloane(const vec& s, int64 mod)
    {
        auto inverse = [](int64 a, int64 m) {
            int64 d, x, y;
            exgcd(a, m, d, x, y);
            return d == 1 ? (x % m + m) % m : -1;
        };
        auto L = [](const vec& a, const vec& b) {
            int da = (a.size() > 1 || (a.size() == 1 && a[0])) ? a.size() - 1 : -1000;
            int db = (b.size() > 1 || (b.size() == 1 && b[0])) ? b.size() - 1 : -1000;
            return std::max(da, db + 1);
        };
        auto prime_power = [&](const vec& s, int64 mod, int64 p, int64 e) {
            // linear feedback shift register mod p^e, p is prime
            std::vector<vec> a(e), b(e), an(e), bn(e), ao(e), bo(e);
            vec t(e), u(e), r(e), to(e, 1), uo(e), pw(e + 1);
            ;
            pw[0] = 1;
            for (int i = pw[0] = 1; i <= e; ++i) pw[i] = pw[i - 1] * p;
            for (int64 i = 0; i < e; ++i)
            {
                a[i] = {pw[i]}, an[i] = {pw[i]};
                b[i] = {0}, bn[i] = {s[0] * pw[i] % mod};
                t[i] = s[0] * pw[i] % mod;
                if (t[i] == 0)
                {
                    t[i] = 1, u[i] = e;
                }
                else
                {
                    for (u[i] = 0; t[i] % p == 0; t[i] /= p, ++u[i])
                        ;
                }
            }
            for (size_t k = 1; k < s.size(); ++k)
            {
                for (int g = 0; g < e; ++g)
                {
                    if (L(an[g], bn[g]) > L(a[g], b[g]))
                    {
                        ao[g] = a[e - 1 - u[g]];
                        bo[g] = b[e - 1 - u[g]];
                        to[g] = t[e - 1 - u[g]];
                        uo[g] = u[e - 1 - u[g]];
                        r[g] = k - 1;
                    }
                }
                a = an, b = bn;
                for (int o = 0; o < e; ++o)
                {
                    int64 d = 0;
                    for (size_t i = 0; i < a[o].size() && i <= k; ++i)
                    {
                        d = (d + a[o][i] * s[k - i]) % mod;
                    }
                    if (d == 0)
                    {
                        t[o] = 1, u[o] = e;
                    }
                    else
                    {
                        for (u[o] = 0, t[o] = d; t[o] % p == 0; t[o] /= p, ++u[o])
                            ;
                        int g = e - 1 - u[o];
                        if (L(a[g], b[g]) == 0)
                        {
                            extand(bn[o], k + 1);
                            bn[o][k] = (bn[o][k] + d) % mod;
                        }
                        else
                        {
                            int64 coef = t[o] * inverse(to[g], mod) % mod * pw[u[o] - uo[g]] % mod;
                            int m = k - r[g];
                            extand(an[o], ao[g].size() + m);
                            extand(bn[o], bo[g].size() + m);
                            for (size_t i = 0; i < ao[g].size(); ++i)
                            {
                                an[o][i + m] -= coef * ao[g][i] % mod;
                                if (an[o][i + m] < 0) an[o][i + m] += mod;
                            }
                            while (an[o].size() && an[o].back() == 0) an[o].pop_back();
                            for (size_t i = 0; i < bo[g].size(); ++i)
                            {
                                bn[o][i + m] -= coef * bo[g][i] % mod;
                                if (bn[o][i + m] < 0) bn[o][i + m] -= mod;
                            }
                            while (bn[o].size() && bn[o].back() == 0) bn[o].pop_back();
                        }
                    }
                }
            }
            return std::make_pair(an[0], bn[0]);
        };
 
        std::vector<std::tuple<int64, int64, int>> fac;
        for (int64 i = 2; i * i <= mod; ++i)
        {
            if (mod % i == 0)
            {
                int64 cnt = 0, pw = 1;
                while (mod % i == 0) mod /= i, ++cnt, pw *= i;
                fac.emplace_back(pw, i, cnt);
            }
        }
        if (mod > 1) fac.emplace_back(mod, mod, 1);
        std::vector<vec> as;
        size_t n = 0;
        for (auto&& x : fac)
        {
            int64 mod, p, e;
            vec a, b;
            std::tie(mod, p, e) = x;
            auto ss = s;
            for (auto&& x : ss) x %= mod;
            std::tie(a, b) = prime_power(ss, mod, p, e);
            as.emplace_back(a);
            n = std::max(n, a.size());
        }
        vec a(n), c(as.size()), m(as.size());
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < as.size(); ++j)
            {
                m[j] = std::get<0>(fac[j]);
                c[j] = i < as[j].size() ? as[j][i] : 0;
            }
            a[i] = crt(c, m);
        }
        return a;
    }
 
    LinearRecurrence(const vec& s, const vec& c, int64 mod) : init(s), trans(c), mod(mod), m(s.size()) {}
    LinearRecurrence(const vec& s, int64 mod, bool is_prime = true) : mod(mod)
    {
        vec A;
        if (is_prime)
            A = BerlekampMassey(s, mod);
        else
            A = ReedsSloane(s, mod);
        if (A.empty()) A = {0};
        m = A.size() - 1;
        trans.resize(m);
        for (int i = 0; i < m; ++i)
        {
            trans[i] = (mod - A[i + 1]) % mod;
        }
        std::reverse(trans.begin(), trans.end());
        init = {s.begin(), s.begin() + m};
    }
    int64 calc(int64 n)
    {
        if (mod == 1) return 0;
        if (n < m) return init[n];
        vec v(m), u(m << 1);
        int msk = !!n;
        for (int64 m = n; m > 1; m >>= 1) msk <<= 1;
        v[0] = 1 % mod;
        for (int x = 0; msk; msk >>= 1, x <<= 1)
        {
            std::fill_n(u.begin(), m * 2, 0);
            x |= !!(n & msk);
            if (x < m)
                u[x] = 1 % mod;
            else
            { // can be optimized by fft/ntt
                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0, t = i + (x & 1); j < m; ++j, ++t)
                    {
                        u[t] = (u[t] + v[i] * v[j]) % mod;
                    }
                }
                for (int i = m * 2 - 1; i >= m; --i)
                {
                    for (int j = 0, t = i - m; j < m; ++j, ++t)
                    {
                        u[t] = (u[t] + trans[j] * u[i]) % mod;
                    }
                }
            }
            v = {u.begin(), u.begin() + m};
        }
        int64 ret = 0;
        for (int i = 0; i < m; ++i)
        {
            ret = (ret + v[i] * init[i]) % mod;
        }
        return ret;
    }
 
    vec init, trans;
    int64 mod;
    int m;
};
 
const int mod = 1e9;
 
typedef long long ll;
 
ll Pow(ll a, ll n, ll mod)
{
    ll t = 1;
    for (; n; n >>= 1, (a *= a) %= mod)
        if (n & 1) (t *= a) %= mod;
    return t;
}
 
int main()
{
    int n, m;
    cin >> n >> m;
    std::vector<long long> f = {0, 1};
    for (int i = 2; i < m * 2 + 5; i++)
        f.push_back((f[i - 1] + f[i - 2]) % mod);
 
    for (auto& t : f) t = Pow(t, m, mod);
    for (int i = 1; i < m * 2 + 5; i++)
        f[i] = (f[i - 1] + f[i]) % mod;
    LinearRecurrence solver(f, mod, false);
    printf("%lld\n", solver.calc(n));
}







//咖啡鸡 BM板子
long long _,n,k,ans;
namespace linear_seq{
    const long long N=10010;
    ll res[N],base[N],_c[N],_md[N];
    vector<long long> Md;
    void mul(ll *a,ll *b,long long k){
        rep(i,0,k+k) _c[i]=0;
        rep(i,0,k) if (a[i]) rep(j,0,k)
            _c[i+j]=(_c[i+j]+a[i]*b[j])%mod;
        for (long long i=k+k-1;i>=k;i--) if (_c[i])
            rep(j,0,SZ(Md)) _c[i-k+Md[j]]=(_c[i-k+Md[j]]-_c[i]*_md[Md[j]])%mod;
        rep(i,0,k) a[i]=_c[i];
    }
    long long solve(ll n,VI a,VI b){
// a 系数 b 初值 b[n+1]=a[0]*b[n]+...
//        printf("%d\n",SZ(b));
        ll ans=0,pnt=0;
        long long k=SZ(a);
        assert(SZ(a)==SZ(b));
        rep(i,0,k) _md[k-1-i]=-a[i];_md[k]=1;
        Md.clear();
        rep(i,0,k) if (_md[i]!=0) Md.push_back(i);
        rep(i,0,k) res[i]=base[i]=0;
        res[0]=1;
        while ((1ll<<pnt)<=n) pnt++;
        for (long long p=pnt;p>=0;p--){
            mul(res,res,k);
            if ((n>>p)&1) {
                for (long long i=k-1;i>=0;i--) res[i+1]=res[i];res[0]=0;
                rep(j,0,SZ(Md)) res[Md[j]]=(res[Md[j]]-res[k]*_md[Md[j]])%mod;
            }
        }
        rep(i,0,k) ans=(ans+res[i]*b[i])%mod;
        if (ans<0) ans+=mod;
        return ans;
    }
    VI BM(VI s) {
        VI C(1,1),B(1,1);
        long long L=0,m=1,b=1;
        rep(n,0,SZ(s)) {
            ll d=0;
            rep(i,0,L+1) d=(d+(ll)C[i]*s[n-i])%mod;
            if (d==0) ++m;
            else if (2*L<=n) {
                VI T=C;
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                L=n+1-L; B=T; b=d; m=1;
            } else {
                ll c=mod-d*powmod(b,mod-2)%mod;
                while (SZ(C)<SZ(B)+m) C.pb(0);
                rep(i,0,SZ(B)) C[i+m]=(C[i+m]+c*B[i])%mod;
                ++m;
            }
        }
        return C;
    }
    long long gao(VI a,ll n){
        VI c=BM(a);
        c.erase(c.begin());
        rep(i,0,SZ(c)) c[i]=(mod-c[i])%mod;
        return solve(n,c,VI(a.begin(),a.begin()+SZ(c)));
    }
};











//CRT板子 中国剩余定理
int exgcd(int a,int b,int &x,int &y)
{
    if(b==0)
    {
        x=1;
        y=0;
        return a;
    }
    int r=exgcd(b,a%b,x,y);
    int tmp=x;
    x=y;
    y=(tmp-(a/b)*y);
    return r;
}
int crt(int *a,int *m,int n)
{
    int M=1,ans=0;
    for(int i=1;i<=n;i++)
        M*=m[i];
    int x,y;
    for(int i=1;i<=n;i++)
    {
        int nowm=(M/m[i]);
        int remain=exgcd(nowm,m[i],x,y);
        ans=(ans+a[i]*nowm*x)%M;
    }
    return ans;
}




//exCRT扩展中国剩余定理 （这里的inv需要用exgxd求逆元，因为不一定互质）
ll M[maxn];
ll C[maxn];
void exCRT()
{
    int n;
    scanf("%d",&n);
    for (LL i = 1; i <= n; i++)
        scanf("%lld%lld", &M[i], &C[i]);
    bool flag = 1;
    for (ll i = 2; i <= n; i++)
    {
        ll M1 = M[i - 1], M2 = M[i], C2 = C[i], C1 = C[i - 1], T = gcd(M1, M2);
        if ((C2 - C1) % T != 0)
        {
            flag = 0;
            break;
        }
        M[i] = (M1 * M2) / T;
        C[i] = ( inv( M1 / T, M2 / T ) * (C2 - C1) / T ) % (M2 / T) * M1 + C1;
        C[i] = (C[i] % M[i] + M[i]) % M[i];
    }
    printf("%lld\n", flag ? C[n] : -1);
}






//母函数  这类母函数通常最后转化为多项式乘法 可以用FFT优化
#include<bits/stdc++.h>
using namespace std;
#define maxn 305
int dp[maxn];
const int a[30]={1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289};

int main()
{
    int n;

    for(int i=0;i<17;i++)
    {
        dp[a[i]]++;
        for(int j=1;j+a[i]<maxn&&j<maxn;j++)
        {
            dp[j+a[i]]+=dp[j];
        }
    }
    while(cin>>n)
    {
        if(!n)
            break;
        cout<<dp[n]<<endl;
    }
    return 0;
}

#include<bits/stdc++.h>
using namespace std;
int a[125];
int b[125];
int main()
{
    ios::sync_with_stdio(false);
    int n;
    while(cin>>n)
    {
        memset(a,0,sizeof(a));
        memset(b,0,sizeof(b));
        a[0]=1;
        for(int i=1;i<=n;i++)     //i的上界为有多少个多项式相乘
        {
            for(int j=0;j<=n;j++)  //j的上界为数组上界，即目前的多项式有多少项
            {
                for(int k=0;k+j<=n;k+=i) //k+j<=n的意思是两多项式相乘，只用考虑上界以下的项
                {                        //k每次加i代表第i个多项式每相邻两项的次数差
                    b[j+k]+=a[j];
                   // cout<<j+k<<' '<<b[j+k]<<endl;
                }
            }
            for(int j=0;j<=n;j++)
                a[j]=b[j];
            memset(b,0,sizeof(b));
        }
        cout<<a[n]<<endl;
    }
    return 0;
}





//指数型母函数
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <iostream>
 
using namespace std;
 
typedef long long ll;
 
const int maxn = 1e5 + 10;
#define mem(a) memset(a, 0, sizeof a)
 
double a[maxn],b[maxn]; // 注意为浮点型
 
int s1[maxn];
 
double f[11];
void init() {
    mem(a);
    mem(b);
    mem(s1);
    f[0] = 1;
    for (int i = 1; i <= 10; i++) {
        f[i] = f[i - 1] * i;
    }
}
 
int main() {
    int n,m;
    while (~scanf("%d%d", &n, &m)) {
       init();
       for (int i = 0; i < n; i++) {
            scanf("%d", &s1[i]);
       }
        for (int i = 0; i <= s1[0]; i++) a[i] = 1.0 / f[i];
        for (int i = 1; i < n; i++) {
            mem(b);
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= s1[i] && k + j <= m; k++) {
                    b[j + k] += a[j] * 1.0 / f[k]; //注意这里
                }
            }
            memcpy(a, b, sizeof b);
        }
       printf("%.0f\n", a[m] * f[m]);
    }
    return 0;
}






//Gauss消元        O(n^3)
double a[N][N];
int  Gauss(int n,int m){
    int col,i,mxr,j,row;
    for(row=col=1;row<=n&&col<=m;row++,col++){
        mxr = row;
        for(i=row+1;i<=n;i++)
            if(fabs(a[i][col])>fabs(a[mxr][col]))
                mxr = i;
        if(mxr != row) swap(a[row],a[mxr]);
        if(fabs(a[row][col]) < eps){
            row--;
            continue;
        }
        for(i=1;i<=n;i++)///消成上三角矩阵
            if(i!=row&&fabs(a[i][col])>eps)
                for(j=m;j>=col;j--)
                    a[i][j]-=a[row][j]/a[row][col]*a[i][col];
    }
    row--;
    for(int i = row;i>=1;i--){///回代成对角矩阵
        for(int j = i + 1;j <= row;j++){
                a[i][m] -= a[j][m] * a[i][j];
        }
        a[i][m] /= a[i][i];
    }
    return row;          //返回秩
}









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

















/*
	博弈论算法
*/



//SG函数模板        Hdu1536 凡是可以化简为Impartial Combinatorial Games的抽象模型（有向图移动棋子）
//都可以尝试利用SG函数 复杂度目前还不太清楚，板子是抄的
//如果堆数或每堆的棋子数太大，可以考虑打标猜规律找循环节   如Codeforces1194D
#include<bits/stdc++.h>
#include<ext/rope>
using namespace std;
using namespace __gnu_cxx;
#define ll long long
#define ld long double
#define ull unsigned long long
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-6;
const int maxn=1e5+10;
const int INF=0x3f3f3f3f;
const double e=2.718281828459045;
#define Accepted 0
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}

//注意 S数组要按从小到大排序 SG函数要初始化为-1 对于每个集合只需初始化1边
//不需要每求一个数的SG就初始化一边
int SG[10100],n,m,s[102],k;//k是集合s的大小 S[i]是定义的特殊取法规则的数组
int dfs(int x)//求SG[x]模板
{
    if(SG[x]!=-1) return SG[x];
    bool vis[110];
    memset(vis,0,sizeof(vis));

    for(int i=0; i<k; i++)
    {
        if(x>=s[i])
        {
            dfs(x-s[i]);
            vis[SG[x-s[i]]]=1;
        }
    }
    int e;
    for(int i=0;; i++)
        if(!vis[i])
        {
            e=i;
            break;
        }
    return SG[x]=e;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    while(cin>>k&&k)
    {
        memset(SG,-1,sizeof(SG));
        for(int i=0; i<k; i++)
            cin>>s[i];
        cin>>m;
        for(int i=0; i<m; i++)
        {
            int sum=0;
            cin>>n;
            for(int i=0,a; i<n; i++)
            {
                cin>>a;
                sum^=dfs(a);
            }
            // printf("SG[%d]=%d\n",num,SG[num]);
            if(sum==0) putchar('L');
            else putchar('W');
        }
        putchar('\n');
    }
    return Accepted;
}

//还是SG函数
//神仙网友的板子也挺不错的，贴在这里    
//https://www.cnblogs.com/dyllove98/p/3194312.html

#include<stdio.h>
#include<string.h>
#include<algorithm>
using namespace std;
//注意 S数组要按从小到大排序 SG函数要初始化为-1 对于每个集合只需初始化1边
//不需要每求一个数的SG就初始化一边
int SG[10100],n,m,s[102],k;//k是集合s的大小 S[i]是定义的特殊取法规则的数组
int dfs(int x)//求SG[x]模板
{
    if(SG[x]!=-1) return SG[x];
    bool vis[110];
    memset(vis,0,sizeof(vis));

    for(int i=0;i<k;i++)
    {
        if(x>=s[i])
        {
           dfs(x-s[i]);
           vis[SG[x-s[i]]]=1;
         }
    }
    int e;
    for(int i=0;;i++)
      if(!vis[i])
      {
        e=i;
        break;
      }
    return SG[x]=e;
}
int main()
{
    int cas,i;
    while(scanf("%d",&k)!=EOF)
    {
        if(!k) break;
        memset(SG,-1,sizeof(SG));
        for(i=0;i<k;i++) scanf("%d",&s[i]);
        sort(s,s+k);
        scanf("%d",&cas);
        while(cas--)
        {
            int t,sum=0;
            scanf("%d",&t);
            while(t--)
            {
                int num;
                scanf("%d",&num);
                sum^=dfs(num);
               // printf("SG[%d]=%d\n",num,SG[num]);
            }
            if(sum==0) printf("L");
            else printf("W");
        }
        printf("\n");
    }
    return 0;
}


//下面是对SG打表的做法
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
const int K=101;
const int H=10001;//H是我们要打表打到的最大值
int k,m,l,h,s[K],sg[H],mex[K];///k是集合元素的个数 s[]是集合  mex大小大约和集合大小差不多
///注意s的排序
void sprague_grundy()
{
    int i,j;
    sg[0]=0;
    for (i=1; i<H; i++)
    {
        memset(mex,0,sizeof(mex));
        j=1;
        while (j<=k && i>=s[j])
        {
            mex[sg[i-s[j]]]=1;
            j++;
        }
        j=0;
        while (mex[j]) j++;
        sg[i]=j;
    }
}

int main()
{
    int tmp,i,j;
    scanf("%d",&k);
    while (k!=0)
    {
        for (i=1; i<=k; i++)
            scanf("%d",&s[i]);
        sort(s+1,s+k+1);            //这个不能少
        sprague_grundy();
        scanf("%d",&m);
        for (i=0; i<m; i++)
        {
            scanf("%d",&l);
            tmp=0;
            for (j=0; j<l; j++)
            {
                scanf("%d",&h);
                tmp=tmp^sg[h];
            }
            if (tmp)
                putchar('W');
            else
                putchar('L');
        }
        putchar('\n');
        scanf("%d",&k);
    }
    return 0;
}














/*
    动态规划(dp):
*/






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








//区间dp板子 未用四边形优化   O(n^3)          P1880 石子合并
for(int len=1;len<=n-1;len++)             //枚举区间长度
{
    for(int i=1;i<n;i++)           //枚举区间左端点
    {
        int j=i+len;                      //区间右端点
        dp1[i][j]=INF;
        for(int k=i;k<j;k++)              //枚举断点
        {
            dp1[i][j]=min(dp1[i][j],dp1[i][k]+dp1[k+1][j]+a[j]-a[i-1]);           //合并[i,k],[k+1,j]所需的花费
            dp2[i][j]=max(dp2[i][j],dp2[i][k]+dp2[k+1][j]+a[j]-a[i-1]);
        }
    }
}












//树形dp   (HDU1561改编)
vector<int> E[maxn];
int val[maxn];
int cost[maxn];
int dp[maxn][maxn];
void dfs(int x,int w)
{
    for(int i=0;i<=w;i++)
    {
        if(i>=cost[x])
        {
            dp[x][i]=val[x];
        }
        else
        {
            dp[x][i]=0;
        }
    }
    //dp[i][j]的意义是以i为根节点的子树选取j个结点的最大价值
    //这个地方先枚举x的子节点i 对于前i个子树，枚举第i个子树的结点数量k
    //此时取决于对于前i-1个子节点取j-k个节点的状态
    for(int i:E[x])
    {
        dfs(i,w-cost[x]);
        for(int j=w;j>=cost[x];j--)
        {
            for(int k=0;k<=j-cost[x];k++)
            {
                dp[x][j]=max(dp[x][j],dp[x][j-k]+dp[i][k]);
            }
        }
    }
}






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









//单调队列优化dp   （洛谷P3957跳房子）
int a[maxn];
int s[maxn];
ll dp[maxn];
ll k;
int n,d,ans=-1;
int q[maxn];        //q用来记录下标，在q[l]~q[r]内的下标都是合法的
bool check(int g)
{
    //cout<<"g="<<g<<endl;
    for(int i=1; i<=n; i++)
    {
        dp[i]=-1e18;
    }
    int x1=g+d,x2=max(d-g,1);
    int now=0;
    int l=0,r=-1;     //[l,r]记录队列的区间 初始状态l>r队列为空
    int i;
    for(i=1; i<=n; i++)
    {
        int L=a[i]-x1,R=a[i]-x2; //L,R用来记录合法范围
        //cout<<L<<' '<<R<<endl;
        while(a[now]<=R&&now<i)    //Insert(now)这个点
        {
            while(r>=l&&dp[q[r]]<=dp[now])r--;   //维护最大值，就保证是一个单调递减的单调队列
            q[++r]=now;
            now++;
        }
        while(a[q[l]]<L&&l<=r)l++;            //pop_front 把队列前端不合法的数去掉
        if(l>r||dp[q[l]]==-1e18)
            continue;
        dp[i]=dp[q[l]]+s[i];
        //cout<<a[i]<<' '<<dp[i]<<endl;
        if(dp[i]>=k)
            return true;
    }
    return false;
}




//单调队列优化dp (HDU板子)
typedef pair<long long,long long> P;
const int maxn=5010,maxk=2010;
 
struct Clique
{
    P q[maxn];
    int top,tail;
 
    void Init() {top=1,tail=0;}
 
    void push(long long k,long long b)
    {
        while (tail>top && (__int128)(b-q[tail-1].second)*(q[tail].first-q[tail-1].first)>=(__int128)(q[tail].second-q[tail-1].second)*(k-q[tail-1].first)) tail--;
        ++tail;
        q[tail].first=k;
        q[tail].second=b;
    }
 
    void autopop(long long x)
    {
        while (top<tail && q[top].first*x+q[top].second<=q[top+1].first*x+q[top+1].second) top++;
    }
 
    P front() {return q[top];}
 
}Q[maxk];

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41136357




//斜率优化dp板子
struct Line{
    ll k,b;
    ll f(ll x){
        return k*x+b;
    }
};
struct Hull{
    vector<Line>ve;
    int cnt,idx;
    bool empty(){return cnt==0;}
    void init(){ve.clear();cnt=idx=0;}
    void add(const Line& p){ve.push_back(p);cnt++;}
    void pop(){ve.pop_back();cnt--;}
    bool checkld(const Line& a,const Line& b,const Line& c){
        return (long double)(a.b-b.b)/(long double)(b.k-a.k)>(long double)(a.b-c.b)/(long double)(c.k-a.k);
    }
    bool checkll(const Line& a,const Line& b,const Line& c){
        return (a.b-b.b)*(c.k-a.k)>(a.b-c.b)*(b.k-a.k);
    }
    void insert(const Line& p){
        if(cnt&&ve.back().k==p.k){
            if(p.b<=ve.back().b)return;
            else pop();
        }
        while(cnt>=2&&checkld(ve[cnt-2],ve[cnt-1],p))pop();
        add(p);
    }
    ll query(ll x){
        while(idx+1<cnt&&ve[idx+1].f(x)>ve[idx].f(x))idx++;
        return ve[idx].f(x);
    }
}hull[2005];

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41129697





//斜率优化dp  quailty
struct Line {
    mutable ll k,m,p;
    bool operator <(const Line& o)const { return k<o.k; }
    bool operator <(ll x)const { return p<x; }
};
struct LineContainer : multiset<Line,less<> > {
    const ll inf = LLONG_MAX;
    ll div(ll a,ll b){
        return a/b-((a^b)<0&&a%b);
    }
    bool isect(iterator x,iterator y){
        if (y==end()){x->p=inf; return false; }
        if (x->k==y->k)x->p=x->m>y->m?inf:-inf;
        else x->p=div(y->m-x->m,x->k-y->k);
        return x->p>=y->p;
    }
    void add(ll k,ll m) {
        auto z=insert({k,m,0}),y=z++,x=y;
        while (isect(y,z))z=erase(z);
        if (x!=begin() && isect(--x,y))isect(x,y=erase(y));
        while ((y=x)!=begin() && (--x)->p>=y->p)
            isect(x,erase(y));
    }
    ll query(ll x){
        assert(!empty());
        auto l=*lower_bound(x);
        return l.k*x+l.m;
    }
}h;

//https://ac.nowcoder.com/acm/contest/view-submission?submissionId=41130719








/*
	计算几何板子整理（kuangbin的板子）
*/



//模拟退火算法   POJ2420
int n;
struct Point
{
    double x, y;
    Point(int x=0,int y=0):x(x),y(y) {}
} P[maxn];
double dist(Point A, Point B)
{
    return sqrt((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y) );
}
int dx[4]= {0,0,1,-1};
int dy[4]= {1,-1,0,0};

struct Simulated_Annealing
{
#define eps 1e-7
#define T 100
#define delta 0.985
//适当调参，delta越小会导致结果越不精准，但是程序，运行速度很快
//delta越大结果越精准，但是运行速度会变慢
#define INF 1e99
    double solve(Point p[],int n)
    {
        //选取初始温度，状态，答案
        Point s = p[0];
        double t = T;
        double ans = INF;
        while(t > eps)
        {
            /******** may need change ************/
            int flag=1;
            while(flag)
            {
                flag=0;
                for(int i=0; i<4; i++)
                {
                    double sum=0;
                    Point pp=Point(s.x+dx[i]*t,s.y+dy[i]*t);
                    for(int j=1; j<=n; j++)
                    {
                        sum+=dist(P[j],pp);
                    }
                    if(sum<ans)
                    {
                        ans=sum;
                        s=pp;
                        flag=1;
                    }
                }
            }

            /**********************************/
            t *= delta;
        }
        return ans;
    }

#undef eps
#undef T
#undef delta
#undef INF
} S;

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    cin>>n;
    for(int i=1; i<=n; i++)
    {
        cin>>P[i].x>>P[i].y;
    }
    cout<<(int)(S.solve(P,n)+0.5)<<endl;
    return Accepted;
}











// 二维计算几何模板`
const double eps = 1e-8;
const double inf = 1e20;
const double pi = acos(-1.0);
const int maxp = 1010;
//`Compares a double to zero`
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
//square of a double
inline double sqr(double x){return x*x;}
/*
 * Point
 * Point()               - Empty constructor
 * Point(double _x,double _y)  - constructor
 * input()             - double input
 * output()            - %.2f output
 * operator ==         - compares x and y
 * operator <          - compares first by x, then by y
 * operator -          - return new Point after subtracting curresponging x and y
 * operator ^          - cross product of 2d points
 * operator *          - dot product
 * len()               - gives length from origin
 * len2()              - gives square of length from origin
 * distance(Point p)   - gives distance from p
 * operator + Point b  - returns new Point after adding curresponging x and y
 * operator * double k - returns new Point after multiplieing x and y by k
 * operator / double k - returns new Point after divideing x and y by k
 * rad(Point a,Point b)- returns the angle of Point a and Point b from this Point
 * trunc(double r)     - return Point that if truncated the distance from center to r
 * rotleft()           - returns 90 degree ccw rotated point
 * rotright()          - returns 90 degree cw rotated point
 * rotate(Point p,double angle) - returns Point after rotateing the Point centering at p by angle radian ccw
 */
struct Point{               //点/向量
	double x,y;
	Point(){}
	Point(double _x,double _y){
		x = _x;
		y = _y;
	}
	void input(){
		scanf("%lf%lf",&x,&y);
	}
	void output(){
		printf("%.2f %.2f\n",x,y);
	}
	bool operator == (Point b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0;
	}
	bool operator < (Point b)const{                //优先比较x的大小，x相同时比较y，用于点排序
		return sgn(x-b.x)== 0?sgn(y-b.y)<0:x<b.x;
	}
	Point operator -(const Point &b)const{
		return Point(x-b.x,y-b.y);
	}
	//叉积
	double operator ^(const Point &b)const{
		return x*b.y - y*b.x;
	}
	//点积
	double operator *(const Point &b)const{
		return x*b.x + y*b.y;
	}
	//返回长度 点到原点的距离
	double len(){
		return hypot(x,y);//库函数
	}
	//返回长度的平方
	double len2(){
		return x*x + y*y;
	}
	//返回两点的距离
	double distance(Point p){
		return hypot(x-p.x,y-p.y);
	}
	Point operator +(const Point &b)const{
		return Point(x+b.x,y+b.y);
	}
	Point operator *(const double &k)const{
		return Point(x*k,y*k);
	}
	Point operator /(const double &k)const{
		return Point(x/k,y/k);
	}
	//`计算pa  和  pb 的夹角`
	//`就是求这个点看a,b 所成的夹角`
	//`测试 LightOJ1203`
	double rad(Point a,Point b){
		Point p = *this;
		return fabs(atan2( fabs((a-p)^(b-p)),(a-p)*(b-p) ));
	}
	//`化为长度为r的向量`
	Point trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point(x*r,y*r);
	}
	//`逆时针旋转90度`
	Point rotleft(){
		return Point(-y,x);
	}
	//`顺时针旋转90度`
	Point rotright(){
		return Point(y,-x);
	}
	//`绕着p点逆时针旋转angle`
	Point rotate(Point p,double angle){
		Point v = (*this) - p;
		double c = cos(angle), s = sin(angle);
		return Point(p.x + v.x*c - v.y*s,p.y + v.x*s + v.y*c);
	}
};



/*
 * Stores two points
 * Line()                         - Empty constructor
 * Line(Point _s,Point _e)        - Line through _s and _e
 * operator ==                    - checks if two points are same
 * Line(Point p,double angle)     - one end p , another end at angle degree
 * Line(double a,double b,double c) - Line of equation ax + by + c = 0
 * input()                        - inputs s and e
 * adjust()                       - orders in such a way that s < e
 * length()                       - distance of se
 * angle()                        - return 0 <= angle < pi
 * relation(Point p)              - 3 if point is on line
 *                                  1 if point on the left of line
 *                                  2 if point on the right of line
 * pointonseg(double p)           - return true if point on segment
 * parallel(Line v)               - return true if they are parallel
 * segcrossseg(Line v)            - returns 0 if does not intersect
 *                                  returns 1 if non-standard intersection
 *                                  returns 2 if intersects
 * linecrossseg(Line v)           - line and seg
 * linecrossline(Line v)          - 0 if parallel
 *                                  1 if coincides
 *                                  2 if intersects
 * crosspoint(Line v)             - returns intersection point
 * dispointtoline(Point p)        - distance from point p to the line
 * dispointtoseg(Point p)         - distance from p to the segment
 * dissegtoseg(Line v)            - distance of two segment
 * lineprog(Point p)              - returns projected point p on se line
 * symmetrypoint(Point p)         - returns reflection point of p over se
 *
 */
//结构体 直线/线段
struct Line{
	Point s,e;
	Line(){}
	Line(Point _s,Point _e){
		s = _s;
		e = _e;
	}
	bool operator ==(Line v){
		return (s == v.s)&&(e == v.e);
	}
	//`根据一个点和倾斜角angle确定直线,0<=angle<pi`
	Line(Point p,double angle){
		s = p;
		if(sgn(angle-pi/2) == 0){
			e = (s + Point(0,1));
		}
		else{
			e = (s + Point(1,tan(angle)));
		}
	}
	//ax+by+c=0
	Line(double a,double b,double c){
		if(sgn(a) == 0){
			s = Point(0,-c/b);
			e = Point(1,-c/b);
		}
		else if(sgn(b) == 0){
			s = Point(-c/a,0);
			e = Point(-c/a,1);
		}
		else{
			s = Point(0,-c/b);
			e = Point(1,(-c-a)/b);
		}
	}
	void input(){
		s.input();
		e.input();
	}
	void adjust(){            //反向
		if(e < s)swap(s,e);
	}
	//求线段长度
	double length(){
		return s.distance(e);
	}
	//`返回直线倾斜角 0<=angle<pi`
	double angle(){
		double k = atan2(e.y-s.y,e.x-s.x);
		if(sgn(k) < 0)k += pi;
		if(sgn(k-pi) == 0)k -= pi;
		return k;
	}
	//`点和直线关系`
	//`1  在左侧`
	//`2  在右侧`
	//`3  在直线上`
	int relation(Point p){
		int c = sgn((p-s)^(e-s));
		if(c < 0)return 1;
		else if(c > 0)return 2;
		else return 3;
	}
	// 点在线段上的判断
	bool pointonseg(Point p){
		return sgn((p-s)^(e-s)) == 0 && sgn((p-s)*(p-e)) <= 0;
	}
	//`两向量平行(对应直线平行或重合)`
	bool parallel(Line v){
		return sgn((e-s)^(v.e-v.s)) == 0;
	}
	//`两线段相交判断`
	//`2 规范相交`（两线段的交点不是两线段端点且无重合的线段）
	//`1 非规范相交`
	//`0 不相交`
	int segcrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		int d3 = sgn((v.e-v.s)^(s-v.s));
		int d4 = sgn((v.e-v.s)^(e-v.s));
		if( (d1^d2)==-2 && (d3^d4)==-2 )return 2;
		return (d1==0 && sgn((v.s-s)*(v.s-e))<=0) ||
			(d2==0 && sgn((v.e-s)*(v.e-e))<=0) ||
			(d3==0 && sgn((s-v.s)*(s-v.e))<=0) ||
			(d4==0 && sgn((e-v.s)*(e-v.e))<=0);
	}
	//`直线和线段相交判断`
	//`-*this line   -v seg`
	//`2 规范相交`
	//`1 非规范相交`
	//`0 不相交`
	int linecrossseg(Line v){
		int d1 = sgn((e-s)^(v.s-s));
		int d2 = sgn((e-s)^(v.e-s));
		if((d1^d2)==-2) return 2;
		return (d1==0||d2==0);
	}
	//`两直线关系`
	//`0 平行`
	//`1 重合`
	//`2 相交`
	int linecrossline(Line v){
		if((*this).parallel(v))
			return v.relation(s)==3;
		return 2;
	}
	//`求两直线的交点`
	//`要保证两直线不平行或重合`
	Point crosspoint(Line v){
		double a1 = (v.e-v.s)^(s-v.s);
		double a2 = (v.e-v.s)^(e-v.s);
		return Point((s.x*a2-e.x*a1)/(a2-a1),(s.y*a2-e.y*a1)/(a2-a1));
	}
	//点到直线的距离
	double dispointtoline(Point p){
		return fabs((p-s)^(e-s))/length();
	}
	//点到线段的距离
	double dispointtoseg(Point p){
		if(sgn((p-s)*(e-s))<0 || sgn((p-e)*(s-e))<0)
			return min(p.distance(s),p.distance(e));
		return dispointtoline(p);
	}
	//`返回线段到线段的距离`
	//`前提是两线段不相交，相交距离就是0了`
	double dissegtoseg(Line v){
		return min(min(dispointtoseg(v.s),dispointtoseg(v.e)),min(v.dispointtoseg(s),v.dispointtoseg(e)));
	}
	//`返回点p在直线上的投影`
	Point lineprog(Point p){
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`返回点p关于直线的对称点`
	Point symmetrypoint(Point p){
		Point q = lineprog(p);
		return Point(2*q.x-p.x,2*q.y-p.y);
	}
};



//圆
struct circle{
	Point p;//圆心
	double r;//半径
	circle(){}
	circle(Point _p,double _r){
		p = _p;
		r = _r;
	}
	circle(double x,double y,double _r){
		p = Point(x,y);
		r = _r;
	}
	//`三角形的外接圆`
	//`需要Point的+ /  rotate()  以及Line的crosspoint()`
	//`利用两条边的中垂线得到圆心`
	//`测试：UVA12304`
	circle(Point a,Point b,Point c){
		Line u = Line((a+b)/2,((a+b)/2)+((b-a).rotleft()));
		Line v = Line((b+c)/2,((b+c)/2)+((c-b).rotleft()));
		p = u.crosspoint(v);
		r = p.distance(a);
	}
	//`三角形的内切圆`
	//`参数bool t没有作用，只是为了和上面外接圆函数区别`
	//`测试：UVA12304`
	circle(Point a,Point b,Point c,bool t){
		Line u,v;
		double m = atan2(b.y-a.y,b.x-a.x), n = atan2(c.y-a.y,c.x-a.x);
		u.s = a;
		u.e = u.s + Point(cos((n+m)/2),sin((n+m)/2));
		v.s = b;
		m = atan2(a.y-b.y,a.x-b.x) , n = atan2(c.y-b.y,c.x-b.x);
		v.e = v.s + Point(cos((n+m)/2),sin((n+m)/2));
		p = u.crosspoint(v);
		r = Line(a,b).dispointtoseg(p);
	}
	//输入
	void input(){
		p.input();
		scanf("%lf",&r);
	}
	//输出
	void output(){
		printf("%.2lf %.2lf %.2lf\n",p.x,p.y,r);
	}
	bool operator == (circle v){
		return (p==v.p) && sgn(r-v.r)==0;
	}
	bool operator < (circle v)const{
		return ((p<v.p)||((p==v.p)&&sgn(r-v.r)<0));
	}
	//面积
	double area(){
		return pi*r*r;
	}
	//周长
	double circumference(){
		return 2*pi*r;
	}
	//`点和圆的关系`
	//`0 圆外`
	//`1 圆上`
	//`2 圆内`
	int relation(Point b){
		double dst = b.distance(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r)==0)return 1;
		return 0;
	}
	//`线段和圆的关系`
	//`比较的是圆心到线段的距离和半径的关系`
	int relationseg(Line v){
		double dst = v.dispointtoseg(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`直线和圆的关系`
	//`比较的是圆心到直线的距离和半径的关系`
	int relationline(Line v){
		double dst = v.dispointtoline(p);
		if(sgn(dst-r) < 0)return 2;
		else if(sgn(dst-r) == 0)return 1;
		return 0;
	}
	//`两圆的关系`
	//`5 相离`
	//`4 外切`
	//`3 相交`
	//`2 内切`
	//`1 内含`
	//`需要Point的distance`
	//`测试：UVA12304`
	int relationcircle(circle v){
		double d = p.distance(v.p);
		if(sgn(d-r-v.r) > 0)return 5;
		if(sgn(d-r-v.r) == 0)return 4;
		double l = fabs(r-v.r);
		if(sgn(d-r-v.r)<0 && sgn(d-l)>0)return 3;
		if(sgn(d-l)==0)return 2;
		if(sgn(d-l)<0)return 1;
	}
	//`求两个圆的交点，返回0表示没有交点，返回1是一个交点，2是两个交点`
	//`需要relationcircle`
	//`测试：UVA12304`
	int pointcrosscircle(circle v,Point &p1,Point &p2){
		int rel = relationcircle(v);
		if(rel == 1 || rel == 5)return 0;
		double d = p.distance(v.p);
		double l = (d*d+r*r-v.r*v.r)/(2*d);
		double h = sqrt(r*r-l*l);
		Point tmp = p + (v.p-p).trunc(l);
		p1 = tmp + ((v.p-p).rotleft().trunc(h));
		p2 = tmp + ((v.p-p).rotright().trunc(h));
		if(rel == 2 || rel == 4)
			return 1;
		return 2;
	}
	//`求直线和圆的交点，返回交点个数`
	int pointcrossline(Line v,Point &p1,Point &p2){
		if(!(*this).relationline(v))return 0;
		Point a = v.lineprog(p);
		double d = v.dispointtoline(p);
		d = sqrt(r*r-d*d);
		if(sgn(d) == 0){
			p1 = a;
			p2 = a;
			return 1;
		}
		p1 = a + (v.e-v.s).trunc(d);
		p2 = a - (v.e-v.s).trunc(d);
		return 2;
	}
	//`得到过a,b两点，半径为r1的两个圆`
	int gercircle(Point a,Point b,double r1,circle &c1,circle &c2){
		circle x(a,r1),y(b,r1);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r;
		return t;
	}
	//`得到与直线u相切，过点q,半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(Line u,Point q,double r1,circle &c1,circle &c2){
		double dis = u.dispointtoline(q);
		if(sgn(dis-r1*2)>0)return 0;
		if(sgn(dis) == 0){
			c1.p = q + ((u.e-u.s).rotleft().trunc(r1));
			c2.p = q + ((u.e-u.s).rotright().trunc(r1));
			c1.r = c2.r = r1;
			return 2;
		}
		Line u1 = Line((u.s + (u.e-u.s).rotleft().trunc(r1)),(u.e + (u.e-u.s).rotleft().trunc(r1)));
		Line u2 = Line((u.s + (u.e-u.s).rotright().trunc(r1)),(u.e + (u.e-u.s).rotright().trunc(r1)));
		circle cc = circle(q,r1);
		Point p1,p2;
		if(!cc.pointcrossline(u1,p1,p2))cc.pointcrossline(u2,p1,p2);
		c1 = circle(p1,r1);
		if(p1 == p2){
			c2 = c1;
			return 1;
		}
		c2 = circle(p2,r1);
		return 2;
	}
	//`同时与直线u,v相切，半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(Line u,Line v,double r1,circle &c1,circle &c2,circle &c3,circle &c4){
		if(u.parallel(v))return 0;//两直线平行
		Line u1 = Line(u.s + (u.e-u.s).rotleft().trunc(r1),u.e + (u.e-u.s).rotleft().trunc(r1));
		Line u2 = Line(u.s + (u.e-u.s).rotright().trunc(r1),u.e + (u.e-u.s).rotright().trunc(r1));
		Line v1 = Line(v.s + (v.e-v.s).rotleft().trunc(r1),v.e + (v.e-v.s).rotleft().trunc(r1));
		Line v2 = Line(v.s + (v.e-v.s).rotright().trunc(r1),v.e + (v.e-v.s).rotright().trunc(r1));
		c1.r = c2.r = c3.r = c4.r = r1;
		c1.p = u1.crosspoint(v1);
		c2.p = u1.crosspoint(v2);
		c3.p = u2.crosspoint(v1);
		c4.p = u2.crosspoint(v2);
		return 4;
	}
	//`同时与不相交圆cx,cy相切，半径为r1的圆`
	//`测试：UVA12304`
	int getcircle(circle cx,circle cy,double r1,circle &c1,circle &c2){
		circle x(cx.p,r1+cx.r),y(cy.p,r1+cy.r);
		int t = x.pointcrosscircle(y,c1.p,c2.p);
		if(!t)return 0;
		c1.r = c2.r = r1;
		return t;
	}

	//`过一点作圆的切线(先判断点和圆的关系)`
	//`测试：UVA12304`
	int tangentline(Point q,Line &u,Line &v){
		int x = relation(q);
		if(x == 2)return 0;
		if(x == 1){
			u = Line(q,q + (q-p).rotleft());
			v = u;
			return 1;
		}
		double d = p.distance(q);
		double l = r*r/d;
		double h = sqrt(r*r-l*l);
		u = Line(q,p + ((q-p).trunc(l) + (q-p).rotleft().trunc(h)));
		v = Line(q,p + ((q-p).trunc(l) + (q-p).rotright().trunc(h)));
		return 2;
	}
	//`求两圆相交的面积`
	double areacircle(circle v){
		int rel = relationcircle(v);
		if(rel >= 4)return 0.0;
		if(rel <= 2)return min(area(),v.area());
		double d = p.distance(v.p);
		double hf = (r+v.r+d)/2.0;
		double ss = 2*sqrt(hf*(hf-r)*(hf-v.r)*(hf-d));
		double a1 = acos((r*r+d*d-v.r*v.r)/(2.0*r*d));
		a1 = a1*r*r;
		double a2 = acos((v.r*v.r+d*d-r*r)/(2.0*v.r*d));
		a2 = a2*v.r*v.r;
		return a1+a2-ss;
	}
	//`求圆和三角形pab的相交面积`
	//`测试：POJ3675 HDU3982 HDU2892`
	double areatriangle(Point a,Point b){
		if(sgn((p-a)^(p-b)) == 0)return 0.0;
		Point q[5];
		int len = 0;
		q[len++] = a;
		Line l(a,b);
		Point p1,p2;
		if(pointcrossline(l,q[1],q[2])==2){
			if(sgn((a-q[1])*(b-q[1]))<0)q[len++] = q[1];
			if(sgn((a-q[2])*(b-q[2]))<0)q[len++] = q[2];
		}
		q[len++] = b;
		if(len == 4 && sgn((q[0]-q[1])*(q[2]-q[1]))>0)swap(q[1],q[2]);
		double res = 0;
		for(int i = 0;i < len-1;i++){
			if(relation(q[i])==0||relation(q[i+1])==0){
				double arg = p.rad(q[i],q[i+1]);
				res += r*r*arg/2.0;
			}
			else{
				res += fabs((q[i]-p)^(q[i+1]-p))/2.0;
			}
		}
		return res;
	}
};



/*
 * n,p  Line l for each side
 * input(int _n)                        - inputs _n size polygon
 * add(Point q)                         - adds a point at end of the list
 * getline()                            - populates line array
 * cmp                                  - comparision in convex_hull order
 * norm()                               - sorting in convex_hull order
 * getconvex(polygon &convex)           - returns convex hull in convex
 * Graham(polygon &convex)              - returns convex hull in convex
 * isconvex()                           - checks if convex
 * relationpoint(Point q)               - returns 3 if q is a vertex
 *                                                2 if on a side
 *                                                1 if inside
 *                                                0 if outside
 * convexcut(Line u,polygon &po)        - left side of u in po
 * gercircumference()                   - returns side length
 * getarea()                            - returns area
 * getdir()                             - returns 0 for cw, 1 for ccw
 * getbarycentre()                      - returns barycenter
 *
 */




//多边形
struct polygon{
	int n;
	Point p[maxp];
	Line l[maxp];
	void input(int _n){
		n = _n;
		for(int i = 0;i < n;i++)
			p[i].input();
	}
	void add(Point q){
		p[n++] = q;
	}
	void getline(){
		for(int i = 0;i < n;i++){
			l[i] = Line(p[i],p[(i+1)%n]);
		}
	}
	struct cmp{
		Point p;
		cmp(const Point &p0){p = p0;}
		bool operator()(const Point &aa,const Point &bb){
			Point a = aa, b = bb;
			int d = sgn((a-p)^(b-p));
			if(d == 0){
				return sgn(a.distance(p)-b.distance(p)) < 0;
			}
			return d > 0;
		}
	};
	//`进行极角排序`
	//`首先需要找到最左下角的点`
	//`需要重载号好Point的 < 操作符(min函数要用) `
	void norm(){
		Point mi = p[0];
		for(int i = 1;i < n;i++)mi = min(mi,p[i]);
		sort(p,p+n,cmp(mi));
	}
	//`得到凸包`
	//`得到的凸包里面的点编号是0~n-1的`
	//`两种凸包的方法`
	//`注意如果有影响，要特判下所有点共点，或者共线的特殊情况`
	//`测试 LightOJ1203  LightOJ1239`
	void getconvex(polygon &convex){
		sort(p,p+n);
		convex.n = n;
		for(int i = 0;i < min(n,2);i++){
			convex.p[i] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//特判
		if(n <= 2)return;
		int &top = convex.n;
		top = 1;
		for(int i = 2;i < n;i++){
			while(top && sgn((convex.p[top]-p[i])^(convex.p[top-1]-p[i])) <= 0)
				top--;
			convex.p[++top] = p[i];
		}
		int temp = top;
		convex.p[++top] = p[n-2];
		for(int i = n-3;i >= 0;i--){
			while(top != temp && sgn((convex.p[top]-p[i])^(convex.p[top-1]-p[i])) <= 0)
				top--;
			convex.p[++top] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//特判
		convex.norm();//`原来得到的是顺时针的点，排序后逆时针`
	}
	//`得到凸包的另外一种方法`
	//`测试 LightOJ1203  LightOJ1239`
	void Graham(polygon &convex){
		norm();
		int &top = convex.n;
		top = 0;
		if(n == 1){
			top = 1;
			convex.p[0] = p[0];
			return;
		}
		if(n == 2){
			top = 2;
			convex.p[0] = p[0];
			convex.p[1] = p[1];
			if(convex.p[0] == convex.p[1])top--;
			return;
		}
		convex.p[0] = p[0];
		convex.p[1] = p[1];
		top = 2;
		for(int i = 2;i < n;i++){
			while( top > 1 && sgn((convex.p[top-1]-convex.p[top-2])^(p[i]-convex.p[top-2])) <= 0 )
				top--;
			convex.p[top++] = p[i];
		}
		if(convex.n == 2 && (convex.p[0] == convex.p[1]))convex.n--;//特判
	}
	//`判断是不是凸的`
	bool isconvex(){
		bool s[2];
		memset(s,false,sizeof(s));
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			int k = (j+1)%n;
			s[sgn((p[j]-p[i])^(p[k]-p[i]))+1] = true;
			if(s[0] && s[2])return false;
		}
		return true;
	}
	//`判断点和任意多边形的关系`
	//` 3 点上`
	//` 2 边上`
	//` 1 内部`
	//` 0 外部`
	int relationpoint(Point q){
		for(int i = 0;i < n;i++){
			if(p[i] == q)return 3;
		}
		getline();
		for(int i = 0;i < n;i++){
			if(l[i].pointonseg(q))return 2;
		}
		int cnt = 0;
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			int k = sgn((q-p[j])^(p[i]-p[j]));
			int u = sgn(p[i].y-q.y);
			int v = sgn(p[j].y-q.y);
			if(k > 0 && u < 0 && v >= 0)cnt++;
			if(k < 0 && v < 0 && u >= 0)cnt--;
		}
		return cnt != 0;
	}
	//`直线u切割凸多边形左侧`
	//`注意直线方向`
	//`测试：HDU3982`
	void convexcut(Line u,polygon &po){
		int &top = po.n;//注意引用
		top = 0;
		for(int i = 0;i < n;i++){
			int d1 = sgn((u.e-u.s)^(p[i]-u.s));
			int d2 = sgn((u.e-u.s)^(p[(i+1)%n]-u.s));
			if(d1 >= 0)po.p[top++] = p[i];
			if(d1*d2 < 0)po.p[top++] = u.crosspoint(Line(p[i],p[(i+1)%n]));
		}
	}
	//`得到周长`
	//`测试 LightOJ1239`
	double getcircumference(){
		double sum = 0;
		for(int i = 0;i < n;i++){
			sum += p[i].distance(p[(i+1)%n]);
		}
		return sum;
	}
	//`得到面积`
	double getarea(){
		double sum = 0;
		for(int i = 0;i < n;i++){
			sum += (p[i]^p[(i+1)%n]);
		}
		return fabs(sum)/2;
	}
	//`得到方向`
	//` 1 表示逆时针，0表示顺时针`
	bool getdir(){
		double sum = 0;
		for(int i = 0;i < n;i++)
			sum += (p[i]^p[(i+1)%n]);
		if(sgn(sum) > 0)return 1;
		return 0;
	}
	//`得到重心`
	Point getbarycentre(){
		Point ret(0,0);
		double area = 0;
		for(int i = 1;i < n-1;i++){
			double tmp = (p[i]-p[0])^(p[i+1]-p[0]);
			if(sgn(tmp) == 0)continue;
			area += tmp;
			ret.x += (p[0].x+p[i].x+p[i+1].x)/3*tmp;
			ret.y += (p[0].y+p[i].y+p[i+1].y)/3*tmp;
		}
		if(sgn(area)) ret = ret/area;
		return ret;
	}
	//`多边形和圆交的面积`
	//`测试：POJ3675 HDU3982 HDU2892`
	double areacircle(circle c){
		double ans = 0;
		for(int i = 0;i < n;i++){
			int j = (i+1)%n;
			if(sgn( (p[j]-c.p)^(p[i]-c.p) ) >= 0)
				ans += c.areatriangle(p[i],p[j]);
			else ans -= c.areatriangle(p[i],p[j]);
		}
		return fabs(ans);
	}
	//`多边形和圆关系`
	//` 2 圆完全在多边形内`
	//` 1 圆在多边形里面，碰到了多边形边界`
	//` 0 其它`
	int relationcircle(circle c){
		getline();
		int x = 2;
		if(relationpoint(c.p) != 1)return 0;//圆心不在内部
		for(int i = 0;i < n;i++){
			if(c.relationseg(l[i])==2)return 0;
			if(c.relationseg(l[i])==1)x = 1;
		}
		return x;
	}
};
//`AB X AC`
double cross(Point A,Point B,Point C){
	return (B-A)^(C-A);
}
//`AB*AC`
double dot(Point A,Point B,Point C){
	return (B-A)*(C-A);
}
//`最小矩形面积覆盖`
//` A 必须是凸包(而且是逆时针顺序)`
//` 测试 UVA 10173`
double minRectangleCover(polygon A){
	//`要特判A.n < 3的情况`
	if(A.n < 3)return 0.0;
	A.p[A.n] = A.p[0];
	double ans = -1;
	int r = 1, p = 1, q;
	for(int i = 0;i < A.n;i++){
		//`卡出离边A.p[i] - A.p[i+1]最远的点`
		while( sgn( cross(A.p[i],A.p[i+1],A.p[r+1]) - cross(A.p[i],A.p[i+1],A.p[r]) ) >= 0 )
			r = (r+1)%A.n;
		//`卡出A.p[i] - A.p[i+1]方向上正向n最远的点`
		while(sgn( dot(A.p[i],A.p[i+1],A.p[p+1]) - dot(A.p[i],A.p[i+1],A.p[p]) ) >= 0 )
			p = (p+1)%A.n;
		if(i == 0)q = p;
		//`卡出A.p[i] - A.p[i+1]方向上负向最远的点`
		while(sgn(dot(A.p[i],A.p[i+1],A.p[q+1]) - dot(A.p[i],A.p[i+1],A.p[q])) <= 0)
			q = (q+1)%A.n;
		double d = (A.p[i] - A.p[i+1]).len2();
		double tmp = cross(A.p[i],A.p[i+1],A.p[r]) *
			(dot(A.p[i],A.p[i+1],A.p[p]) - dot(A.p[i],A.p[i+1],A.p[q]))/d;
		if(ans < 0 || ans > tmp)ans = tmp;
	}
	return ans;
}

//`直线切凸多边形`
//`多边形是逆时针的，在q1q2的左侧`
//`测试:HDU3982`
vector<Point> convexCut(const vector<Point> &ps,Point q1,Point q2){
	vector<Point>qs;
	int n = ps.size();
	for(int i = 0;i < n;i++){
		Point p1 = ps[i], p2 = ps[(i+1)%n];
		int d1 = sgn((q2-q1)^(p1-q1)), d2 = sgn((q2-q1)^(p2-q1));
		if(d1 >= 0)
			qs.push_back(p1);
		if(d1 * d2 < 0)
			qs.push_back(Line(p1,p2).crosspoint(Line(q1,q2)));
	}
	return qs;
}




//`半平面交`
//`测试 POJ3335 POJ1474 POJ1279`
//***************************
struct halfplane:public Line{
	double angle;
	halfplane(){}
	//`表示向量s->e逆时针(左侧)的半平面`
	halfplane(Point _s,Point _e){
		s = _s;
		e = _e;
	}
	halfplane(Line v){
		s = v.s;
		e = v.e;
	}
	void calcangle(){
		angle = atan2(e.y-s.y,e.x-s.x);
	}
	bool operator <(const halfplane &b)const{
		return angle < b.angle;
	}
};
struct halfplanes{
	int n;
	halfplane hp[maxp];
	Point p[maxp];
	int que[maxp];
	int st,ed;
	halfplanes(){n=0;}
	void push(halfplane tmp){
		hp[n++] = tmp;
	}
	//去重
	void unique(){
		int m = 1;
		for(int i = 1;i < n;i++){
			if(sgn(hp[i].angle-hp[i-1].angle) != 0)
				hp[m++] = hp[i];
			else if(sgn( (hp[m-1].e-hp[m-1].s)^(hp[i].s-hp[m-1].s) ) > 0)
				hp[m-1] = hp[i];
		}
		n = m;
	}
	bool halfplaneinsert(){
		for(int i = 0;i < n;i++)hp[i].calcangle();
		sort(hp,hp+n);
		unique();
		que[st=0] = 0;
		que[ed=1] = 1;
		p[1] = hp[0].crosspoint(hp[1]);
		for(int i = 2;i < n;i++){
			while(st<ed && sgn((hp[i].e-hp[i].s)^(p[ed]-hp[i].s))<0)ed--;
			while(st<ed && sgn((hp[i].e-hp[i].s)^(p[st+1]-hp[i].s))<0)st++;
			que[++ed] = i;
			if(hp[i].parallel(hp[que[ed-1]]))return false;
			p[ed]=hp[i].crosspoint(hp[que[ed-1]]);
		}
		while(st<ed && sgn((hp[que[st]].e-hp[que[st]].s)^(p[ed]-hp[que[st]].s))<0)ed--;
		while(st<ed && sgn((hp[que[ed]].e-hp[que[ed]].s)^(p[st+1]-hp[que[ed]].s))<0)st++;
		if(st+1>=ed)return false;
		return true;
	}
	//`得到最后半平面交得到的凸多边形`
	//`需要先调用halfplaneinsert() 且返回true`
	void getconvex(polygon &con){
		p[st] = hp[que[st]].crosspoint(hp[que[ed]]);
		con.n = ed-st+1;
		for(int j = st,i = 0;j <= ed;i++,j++)
			con.p[i] = p[j];
	}
};
//***************************









//多圆/圆簇

const int maxn = 1010;
struct circles{
	circle c[maxn];
	double ans[maxn];//`ans[i]表示被覆盖了i次的面积`
	double pre[maxn];
	int n;
	circles(){}
	void add(circle cc){
		c[n++] = cc;
	}
	//`x包含在y中`
	bool inner(circle x,circle y){
		if(x.relationcircle(y) != 1)return 0;
		return sgn(x.r-y.r)<=0?1:0;
	}
	//圆的面积并去掉内含的圆
	void init_or(){
		bool mark[maxn] = {0};
		int i,j,k=0;
		for(i = 0;i < n;i++){
			for(j = 0;j < n;j++)
				if(i != j && !mark[j]){
					if( (c[i]==c[j])||inner(c[i],c[j]) )break;
				}
			if(j < n)mark[i] = 1;
		}
		for(i = 0;i < n;i++)
			if(!mark[i])
				c[k++] = c[i];
		n = k;
	}
	//`圆的面积交去掉内含的圆`
	void init_add(){
		int i,j,k;
		bool mark[maxn] = {0};
		for(i = 0;i < n;i++){
			for(j = 0;j < n;j++)
				if(i != j && !mark[j]){
					if( (c[i]==c[j])||inner(c[j],c[i]) )break;
				}
			if(j < n)mark[i] = 1;
		}
		for(i = 0;i < n;i++)
			if(!mark[i])
				c[k++] = c[i];
		n = k;
	}
	//`半径为r的圆，弧度为th对应的弓形的面积`
	double areaarc(double th,double r){
		return 0.5*r*r*(th-sin(th));
	}
	//`测试SPOJVCIRCLES SPOJCIRUT`
	//`SPOJVCIRCLES求n个圆并的面积，需要加上init\_or()去掉重复圆（否则WA）`
	//`SPOJCIRUT 是求被覆盖k次的面积，不能加init\_or()`
	//`对于求覆盖多少次面积的问题，不能解决相同圆，而且不能init\_or()`
	//`求多圆面积并，需要init\_or,其中一个目的就是去掉相同圆`
	void getarea(){
		memset(ans,0,sizeof(ans));
		vector<pair<double,int> >v;
		for(int i = 0;i < n;i++){
			v.clear();
			v.push_back(make_pair(-pi,1));
			v.push_back(make_pair(pi,-1));
			for(int j = 0;j < n;j++)
				if(i != j){
					Point q = (c[j].p - c[i].p);
					double ab = q.len(),ac = c[i].r, bc = c[j].r;
					if(sgn(ab+ac-bc)<=0){
						v.push_back(make_pair(-pi,1));
						v.push_back(make_pair(pi,-1));
						continue;
					}
					if(sgn(ab+bc-ac)<=0)continue;
					if(sgn(ab-ac-bc)>0)continue;
					double th = atan2(q.y,q.x), fai = acos((ac*ac+ab*ab-bc*bc)/(2.0*ac*ab));
					double a0 = th-fai;
					if(sgn(a0+pi)<0)a0+=2*pi;
					double a1 = th+fai;
					if(sgn(a1-pi)>0)a1-=2*pi;
					if(sgn(a0-a1)>0){
						v.push_back(make_pair(a0,1));
						v.push_back(make_pair(pi,-1));
						v.push_back(make_pair(-pi,1));
						v.push_back(make_pair(a1,-1));
					}
					else{
						v.push_back(make_pair(a0,1));
						v.push_back(make_pair(a1,-1));
					}
				}
			sort(v.begin(),v.end());
			int cur = 0;
			for(int j = 0;j < v.size();j++){
				if(cur && sgn(v[j].first-pre[cur])){
					ans[cur] += areaarc(v[j].first-pre[cur],c[i].r);
					ans[cur] += 0.5*(Point(c[i].p.x+c[i].r*cos(pre[cur]),c[i].p.y+c[i].r*sin(pre[cur]))^Point(c[i].p.x+c[i].r*cos(v[j].first),c[i].p.y+c[i].r*sin(v[j].first)));
				}
				cur += v[j].second;
				pre[cur] = v[j].first;
			}
		}
		for(int i = 1;i < n;i++)
			ans[i] -= ans[i+1];
	}
};










//三维几何
const double eps = 1e-8;
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
struct Point3{
	double x,y,z;
	Point3(double _x = 0,double _y = 0,double _z = 0){
		x = _x;
		y = _y;
		z = _z;
	}
	void input(){
		scanf("%lf%lf%lf",&x,&y,&z);
	}
	void output(){
		printf("%.2lf %.2lf %.2lf\n",x,y,z);
	}
	bool operator ==(const Point3 &b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0 && sgn(z-b.z) == 0;
	}
	bool operator <(const Point3 &b)const{
		return sgn(x-b.x)==0?(sgn(y-b.y)==0?sgn(z-b.z)<0:y<b.y):x<b.x;
	}
	double len(){
		return sqrt(x*x+y*y+z*z);
	}
	double len2(){
		return x*x+y*y+z*z;
	}
	double distance(const Point3 &b)const{
		return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z));
	}
	Point3 operator -(const Point3 &b)const{
		return Point3(x-b.x,y-b.y,z-b.z);
	}
	Point3 operator +(const Point3 &b)const{
		return Point3(x+b.x,y+b.y,z+b.z);
	}
	Point3 operator *(const double &k)const{
		return Point3(x*k,y*k,z*k);
	}
	Point3 operator /(const double &k)const{
		return Point3(x/k,y/k,z/k);
	}
	//点乘
	double operator *(const Point3 &b)const{
		return x*b.x+y*b.y+z*b.z;
	}
	//叉乘
	Point3 operator ^(const Point3 &b)const{
		return Point3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
	double rad(Point3 a,Point3 b){
		Point3 p = (*this);
		return acos( ( (a-p)*(b-p) )/ (a.distance(p)*b.distance(p)) );
	}
	//变换长度
	Point3 trunc(double r){
		double l = len();
		if(!sgn(l))return *this;
		r /= l;
		return Point3(x*r,y*r,z*r);
	}
};
struct Line3
{
	Point3 s,e;
	Line3(){}
	Line3(Point3 _s,Point3 _e)
	{
		s = _s;
		e = _e;
	}
	bool operator ==(const Line3 v)
	{
		return (s==v.s)&&(e==v.e);
	}
	void input()
	{
		s.input();
		e.input();
	}
	double length()
	{
		return s.distance(e);
	}
	//点到直线距离
	double dispointtoline(Point3 p)
	{
		return ((e-s)^(p-s)).len()/s.distance(e);
	}
	//点到线段距离
	double dispointtoseg(Point3 p)
	{
		if(sgn((p-s)*(e-s)) < 0 || sgn((p-e)*(s-e)) < 0)
			return min(p.distance(s),e.distance(p));
		return dispointtoline(p);
	}
	//`返回点p在直线上的投影`
	Point3 lineprog(Point3 p)
	{
		return s + ( ((e-s)*((e-s)*(p-s)))/((e-s).len2()) );
	}
	//`p绕此向量逆时针arg角度`
	Point3 rotate(Point3 p,double ang)
	{
		if(sgn(((s-p)^(e-p)).len()) == 0)return p;
		Point3 f1 = (e-s)^(p-s);
		Point3 f2 = (e-s)^(f1);
		double len = ((s-p)^(e-p)).len()/s.distance(e);
		f1 = f1.trunc(len); f2 = f2.trunc(len);
		Point3 h = p+f2;
		Point3 pp = h+f1;
		return h + ((p-h)*cos(ang)) + ((pp-h)*sin(ang));
	}
	//`点在直线上`
	bool pointonseg(Point3 p)
	{
		return sgn( ((s-p)^(e-p)).len() ) == 0 && sgn((s-p)*(e-p)) == 0;
	}
};
struct Plane
{
	Point3 a,b,c,o;//`平面上的三个点，以及法向量`
	Plane(){}
	Plane(Point3 _a,Point3 _b,Point3 _c)
	{
		a = _a;
		b = _b;
		c = _c;
		o = pvec();
	}
	Point3 pvec()
	{
		return (b-a)^(c-a);
	}
	//`ax+by+cz+d = 0`
	Plane(double _a,double _b,double _c,double _d)
	{
		o = Point3(_a,_b,_c);
		if(sgn(_a) != 0)
			a = Point3((-_d-_c-_b)/_a,1,1);
		else if(sgn(_b) != 0)
			a = Point3(1,(-_d-_c-_a)/_b,1);
		else if(sgn(_c) != 0)
			a = Point3(1,1,(-_d-_a-_b)/_c);
	}
	//`点在平面上的判断`
	bool pointonplane(Point3 p)
	{
		return sgn((p-a)*o) == 0;
	}
	//`两平面夹角`
	double angleplane(Plane f)
	{
		return acos(o*f.o)/(o.len()*f.o.len());
	}
	//`平面和直线的交点，返回值是交点个数`
	int crossline(Line3 u,Point3 &p)
	{
		double x = o*(u.e-a);
		double y = o*(u.s-a);
		double d = x-y;
		if(sgn(d) == 0)return 0;
		p = ((u.s*x)-(u.e*y))/d;
		return 1;
	}
	//`点到平面最近点(也就是投影)`
	Point3 pointtoplane(Point3 p)
	{
		Line3 u = Line3(p,p+o);
		crossline(u,p);
		return p;
	}
	//`平面和平面的交线`
	int crossplane(Plane f,Line3 &u)
	{
		Point3 oo = o^f.o;
		Point3 v = o^oo;
		double d = fabs(f.o*v);
		if(sgn(d) == 0)return 0;
		Point3 q = a + (v*(f.o*(f.a-a))/d);
		u = Line3(q,q+oo);
		return 1;
	}
};


//最近平面点对
const int MAXN = 100010;
const double eps = 1e-8;
const double INF = 1e20;
struct Point{
	double x,y;
	void input(){
		scanf("%lf%lf",&x,&y);
	}
};
double dist(Point a,Point b){
	return sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}
Point p[MAXN];
Point tmpt[MAXN];
bool cmpx(Point a,Point b){
	return a.x < b.x || (a.x == b.x && a.y < b.y);
}
bool cmpy(Point a,Point b){
	return a.y < b.y || (a.y == b.y && a.x < b.x);
}
double Closest_Pair(int left,int right){
	double d = INF;
	if(left == right)return d;
	if(left+1 == right)return dist(p[left],p[right]);
	int mid = (left+right)/2;
	double d1 = Closest_Pair(left,mid);
	double d2 = Closest_Pair(mid+1,right);
	d = min(d1,d2);
	int cnt = 0;
	for(int i = left;i <= right;i++){
		if(fabs(p[mid].x - p[i].x) <= d)
			tmpt[cnt++] = p[i];
	}
	sort(tmpt,tmpt+cnt,cmpy);
	for(int i = 0;i < cnt;i++){
		for(int j = i+1;j < cnt && tmpt[j].y - tmpt[i].y < d;j++)
			d = min(d,dist(tmpt[i],tmpt[j]));
	}
	return d;
}
int main(){
	int n;
	while(scanf("%d",&n) == 1 && n){
		for(int i = 0;i < n;i++)p[i].input();
		sort(p,p+n,cmpx);
		printf("%.2lf\n",Closest_Pair(0,n-1));
	}
    return 0;
}


//三维凸包  Hud4273
const double eps = 1e-8;
const int MAXN = 550;
int sgn(double x){
	if(fabs(x) < eps)return 0;
	if(x < 0)return -1;
	else return 1;
}
struct Point3{
	double x,y,z;
	Point3(double _x = 0, double _y = 0, double _z = 0){
		x = _x;
		y = _y;
		z = _z;
	}
	void input(){
		scanf("%lf%lf%lf",&x,&y,&z);
	}
	bool operator ==(const Point3 &b)const{
		return sgn(x-b.x) == 0 && sgn(y-b.y) == 0 && sgn(z-b.z) == 0;
	}
	double len(){
		return sqrt(x*x+y*y+z*z);
	}
	double len2(){
		return x*x+y*y+z*z;
	}
	double distance(const Point3 &b)const{
		return sqrt((x-b.x)*(x-b.x)+(y-b.y)*(y-b.y)+(z-b.z)*(z-b.z));
	}
	Point3 operator -(const Point3 &b)const{
		return Point3(x-b.x,y-b.y,z-b.z);
	}
	Point3 operator +(const Point3 &b)const{
		return Point3(x+b.x,y+b.y,z+b.z);
	}
	Point3 operator *(const double &k)const{
		return Point3(x*k,y*k,z*k);
	}
	Point3 operator /(const double &k)const{
		return Point3(x/k,y/k,z/k);
	}
	//点乘
	double operator *(const Point3 &b)const{
		return x*b.x + y*b.y + z*b.z;
	}
	//叉乘
	Point3 operator ^(const Point3 &b)const{
		return Point3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);
	}
};
struct CH3D{
	struct face{
		//表示凸包一个面上的三个点的编号
		int a,b,c;
		//表示该面是否属于最终的凸包上的面
		bool ok;
	};
	//初始顶点数
	int n;
	Point3 P[MAXN];
	//凸包表面的三角形数
	int num;
	//凸包表面的三角形
	face F[8*MAXN];
	int g[MAXN][MAXN];
	//叉乘
	Point3 cross(const Point3 &a,const Point3 &b,const Point3 &c){
		return (b-a)^(c-a);
	}
	//`三角形面积*2`
	double area(Point3 a,Point3 b,Point3 c){
		return ((b-a)^(c-a)).len();
	}
	//`四面体有向面积*6`
	double volume(Point3 a,Point3 b,Point3 c,Point3 d){
		return ((b-a)^(c-a))*(d-a);
	}
	//`正：点在面同向`
	double dblcmp(Point3 &p,face &f){
		Point3 p1 = P[f.b] - P[f.a];
		Point3 p2 = P[f.c] - P[f.a];
		Point3 p3 = p - P[f.a];
		return (p1^p2)*p3;
	}
	void deal(int p,int a,int b){
		int f = g[a][b];
		face add;
		if(F[f].ok){
			if(dblcmp(P[p],F[f]) > eps)
				dfs(p,f);
			else {
				add.a = b;
				add.b = a;
				add.c = p;
				add.ok = true;
				g[p][b] = g[a][p] = g[b][a] = num;
				F[num++] = add;
			}
		}
	}
	//递归搜索所有应该从凸包内删除的面
	void dfs(int p,int now){
		F[now].ok = false;
		deal(p,F[now].b,F[now].a);
		deal(p,F[now].c,F[now].b);
		deal(p,F[now].a,F[now].c);
	}
	bool same(int s,int t){
		Point3 &a = P[F[s].a];
		Point3 &b = P[F[s].b];
		Point3 &c = P[F[s].c];
		return fabs(volume(a,b,c,P[F[t].a])) < eps &&
			fabs(volume(a,b,c,P[F[t].b])) < eps &&
			fabs(volume(a,b,c,P[F[t].c])) < eps;
	}
	//构建三维凸包
	void create(){
		num = 0;
		face add;

		//***********************************
		//此段是为了保证前四个点不共面
		bool flag = true;
		for(int i = 1;i < n;i++){
			if(!(P[0] == P[i])){
				swap(P[1],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		flag = true;
		for(int i = 2;i < n;i++){
			if( ((P[1]-P[0])^(P[i]-P[0])).len() > eps ){
				swap(P[2],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		flag = true;
		for(int i = 3;i < n;i++){
			if(fabs( ((P[1]-P[0])^(P[2]-P[0]))*(P[i]-P[0]) ) > eps){
				swap(P[3],P[i]);
				flag = false;
				break;
			}
		}
		if(flag)return;
		//**********************************

		for(int i = 0;i < 4;i++){
			add.a = (i+1)%4;
			add.b = (i+2)%4;
			add.c = (i+3)%4;
			add.ok = true;
			if(dblcmp(P[i],add) > 0)swap(add.b,add.c);
			g[add.a][add.b] = g[add.b][add.c] = g[add.c][add.a] = num;
			F[num++] = add;
		}
		for(int i = 4;i < n;i++)
			for(int j = 0;j < num;j++)
				if(F[j].ok && dblcmp(P[i],F[j]) > eps){
					dfs(i,j);
					break;
				}
		int tmp = num;
		num = 0;
		for(int i = 0;i < tmp;i++)
			if(F[i].ok)
				F[num++] = F[i];
	}
	//表面积
	//`测试：HDU3528`
	double area(){
		double res = 0;
		if(n == 3){
			Point3 p = cross(P[0],P[1],P[2]);
			return p.len()/2;
		}
		for(int i = 0;i < num;i++)
			res += area(P[F[i].a],P[F[i].b],P[F[i].c]);
		return res/2.0;
	}
	double volume(){
		double res = 0;
		Point3 tmp = Point3(0,0,0);
		for(int i = 0;i < num;i++)
			res += volume(tmp,P[F[i].a],P[F[i].b],P[F[i].c]);
		return fabs(res/6);
	}
	//表面三角形个数
	int triangle(){
		return num;
	}
	//表面多边形个数
	//`测试：HDU3662`
	int polygon(){
		int res = 0;
		for(int i = 0;i < num;i++){
			bool flag = true;
			for(int j = 0;j < i;j++)
				if(same(i,j)){
					flag = 0;
					break;
				}
			res += flag;
		}
		return res;
	}
	//重心
	//`测试：HDU4273`
	Point3 barycenter(){
		Point3 ans = Point3(0,0,0);
		Point3 o = Point3(0,0,0);
		double all = 0;
		for(int i = 0;i < num;i++){
			double vol = volume(o,P[F[i].a],P[F[i].b],P[F[i].c]);
			ans = ans + (((o+P[F[i].a]+P[F[i].b]+P[F[i].c])/4.0)*vol);
			all += vol;
		}
		ans = ans/all;
		return ans;
	}
	//点到面的距离
	//`测试：HDU4273`
	double ptoface(Point3 p,int i){
		double tmp1 = fabs(volume(P[F[i].a],P[F[i].b],P[F[i].c],p));
		double tmp2 = ((P[F[i].b]-P[F[i].a])^(P[F[i].c]-P[F[i].a])).len();
		return tmp1/tmp2;
	}
};
CH3D hull;
int main()
{
    while(scanf("%d",&hull.n) == 1){
		for(int i = 0;i < hull.n;i++)hull.P[i].input();
		hull.create();
		Point3 p = hull.barycenter();
		double ans = 1e20;
		for(int i = 0;i < hull.num;i++)
			ans = min(ans,hull.ptoface(p,i));
		printf("%.3lf\n",ans);
	}
    return 0;
}




















/*
	字符串算法
*/
//kmp算法    O(m+n)判断字符串匹配/求匹配数
int Next[maxn];
void getNext(const string s)
{
    //k:最大前后缀长度
    int len=s.length();//模版字符串长度
    Next[0] = 0;//模版字符串的第一个字符的最大前后缀长度为0
    for (int i = 1,k = 0; i < len; i++)//for循环，从第二个字符开始，依次计算每一个字符对应的next值
    {
        while(k > 0 && s[i] != s[k])//递归的求出P[0]···P[q]的最大的相同的前后缀长度k
            k = Next[k-1];          //不理解没关系看下面的分析，这个while循环是整段代码的精髓所在，确实不好理解
        if (s[i] == s[k])//如果相等，那么最大相同前后缀长度加1
        {
            k++;
        }
        Next[i] = k;
    }
}

int kmp(const string s,const string p)
{
    int cnt=0;
    int len1=s.length(),len2=p.length();
    getNext(p);
    for (int i = 0,j = 0; i < len1; i++)
    {
        while(j > 0 && s[i] != p[j])   //失配，j下标向前挪找到可以匹配的位置
            j = Next[j-1];
        if (s[i] == p[j])//该位匹配成功，j下标挪到模式串的下一个位置
        {
            j++;
        }
        if (j == len2)            //匹配完毕
        {
            cnt++;
            j=Next[j-1];
            //break;
        }
    }
    return cnt;
}



//马拉车算法(Manacher)  找出一个字符串中的最长回文子串 O(n)
int n,hw[maxn<<1],ans;              //hw记录每个点的回文半径，hw[i]-1即为以i为中心的最长回文长度（去掉'#'），n为字符串长度
char a[maxn],s[maxn<<1];       //a为原字符串，s为扩展后的字符串
void change()            //将相邻两个字符之间插上'#'
{
    s[0]=s[1]='#';
    for(int i=0; i<n; i++)
    {
        s[i*2+2]=a[i];
        s[i*2+3]='#';
    }
    n=n*2+2;
    s[n]=0;
}
int manacher()       //马拉车算法
{
    ans=1;
    change();
    int maxright=0,mid;        //maxright记录当前可以拓展到的最右的回文串的最右边界(不可达边界，maxright-1才是可达边界)，mid记录这个最长回文串的中心点
    for(int i=1; i<n; i++)
    {
        if(i<maxright)
            hw[i]=min(hw[(mid<<1)-i],hw[mid]+mid-i);  //((mid<<1)-i)是i关于mid对称的那个点   hw[mid]+mid-i是i到maxright的最长距离
        else
            hw[i]=1;
        for(; s[i+hw[i]]==s[i-hw[i]]; ++hw[i]); //暴力扩展maxright的外部部分
        if(hw[i]+i>maxright)
        {
            maxright=hw[i]+i;
            mid=i;
        }
       // cout<<s[i]<<' '<<hw[i]<<endl;
        ans=max(ans,hw[i]);
    }
    return ans-1;                 //返回最长回文串的长度
}

signed main()
{
    scanf("%s",a);
    n=strlen(a);
    printf("%d\n",manacher());
    return 0;
}

//PS:manacher算法求得最长回文串的回文半径hw[i]以及该回文串的中心位置i，
//则s[i-(hw[i]-1)+1]~s[i+(hw[i]-1)-1]中去掉#的就是最长回文子串，
//原位置为a[(i-(hw[i]-1)+1)/2-1]~a[(i+(hw[i]-1)-1)/2-1].
//a[(i-hw[i])/2]~a[(i+hw[i])/2-2]









//Trie 字典树模板
struct Trie
{
    #define type int
    #define MAX maxn*20 //Trie结点的最大数量 总字符数量
    int sz;
    struct TrieNode
    {
        int pre;
        bool ed;
        int nxt[26];
        type v;                //may change
    }trie[MAX];
    Trie()
    {
        sz=1;                     //记录Trie中结点的数量
        memset(trie,0,sizeof(trie));
    }
    void insert(const string& s)           //在根节点处插入一个字符串
    {
        int p=1;                         //默认为根节点
        for(char c:s)
        {
            int ch=c-'a';
            if(!trie[p].nxt[ch]) trie[p].nxt[ch]=++sz;   //从内存池中选取一个空间分配给新节点
            p=trie[p].nxt[ch];
            trie[p].pre++;
        }
        trie[p].ed=1;
    }
    bool search(const string& s)        //查询Trie中是否存在一个字符串，返回1/0
    {
        int p=1;
        for(char c:s)
        {
            p=trie[p].nxt[c-'a'];
            if(!p) return 0;
        }
        return trie[p].ed;
    }
    string prefix(const string& s)     //求出最小unique前缀
    {
        string res;
        int p=1;
        for(char c:s)
        {
            p=trie[p].nxt[c-'a'];
            res+=c;
            if(trie[p].pre<=1) break;
        }
        return res;
    }
    #undef type
}tr;

//PS:foreach循环是C++11里的，poj不能用...

//再写一遍
struct Trie
{
    #define type int
    #define MAX maxn*20 //Trie结点的最大数量 总字符数量
    int sz;
    struct TrieNode
    {
        int pre;
        bool ed;
        int nxt[26];
        type v;            //may change
    }trie[MAX];
    Trie()
    {
        sz=1;               //记录Trie中结点的数量
        memset(trie,0,sizeof(trie));
    }
    //在根节点处插入一个字符串
    void insert(const string& s)
    {
        int p=1,len=s.length();        //p默认起始为根节点
        for(int i=0;i<len;i++)
        {
            int ch=s[i]-'a';
            if(!trie[p].nxt[ch]) trie[p].nxt[ch]=++sz;
            //从内存池中选取一个空间分配给新节点
            p=trie[p].nxt[ch];
            trie[p].pre++;
        }
        trie[p].ed=1;
    }
    //查询Trie中是否存在一个字符串，返回1/0
    bool search(const string& s)
    {
        int p=1,len=s.length();
        for(int i=0;i<len;i++)
        {
            p=trie[p].nxt[s[i]-'a'];
            if(!p) return 0;
        }
        return trie[p].ed;
    }
    //求出最小unique前缀
    string prefix(const string& s)
    {
        string res;
        int p=1,len=s.length();
        for(int i=0;i<len;i++)
        {
            p=trie[p].nxt[s[i]-'a'];
            res+=s[i];
            if(trie[p].pre<=1) break;
        }
        return res;
    }
    #undef type
}tr;














//AC自动机板子 by:sjf
struct AC_Automaton
{
    #define MAX 100010
    static const int K=26;//may need change
    int next[MAX][K],fail[MAX],cnt[MAX],last[MAX];
    int root,tot;
    inline int getid(char c)//may need change
    {
        return c-'a';
    }
    int newnode()
    {
        memset(next[tot],0,sizeof(next[tot]));
        fail[tot]=0;
        cnt[tot]=0;
        return tot++;
    }
    void init()
    {
        tot=0;
        root=newnode();
    }
    void insert(char *s)
    {
        int len,now,i;
        len=strlen(s);
        now=root;
        for(i=0;i<len;i++)
        {
            int t=getid(s[i]);
            if(!next[now][t]) next[now][t]=newnode();
            now=next[now][t];
        }
        cnt[now]++;
    }
    void setfail()
    {
        int i,now;
        queue<int>q;
        for(i=0;i<K;i++)
        {
            if(next[root][i]) q.push(next[root][i]);
        }
        while(!q.empty())
        {
            now=q.front();
            q.pop();
            //suffix link
            if(cnt[fail[now]]) last[now]=fail[now];
            else last[now]=last[fail[now]];
            /*
            may need add something here:
            cnt[now]+=cnt[fail[now]];
            */
            for(i=0;i<K;i++)
            {
                if(next[now][i])
                {
                    fail[next[now][i]]=next[fail[now]][i];
                    q.push(next[now][i]);
                }
                else next[now][i]=next[fail[now]][i];
            }
        }
    }
    int query(char *s)
    {
        int len,now,i,res;
        len=strlen(s);
        now=root;
        res=0;
        for(i=0;i<len;i++)
        {
            int t=getid(s[i]);
            now=next[now][t];
            int tmp=now;
            while(tmp&&cnt[tmp]!=-1)
            {
                res+=cnt[tmp];
                cnt[tmp]=-1;
                tmp=last[tmp];
            }
        }
        return res;
    }
    //build fail tree
    vector<int> mp[MAX];
    void build_tree()
    {
        for(int i=0;i<=tot;i++) mp[i].clear();
        for(int i=1;i<tot;i++) mp[fail[i]].pb(i);
    }
    #undef MAX
}ac;







// Created by calabash_boy on 18/6/5.
// HDU 6138
//给定若干字典串。
// query:strx stry 求最长的p,p为strx、stry子串，且p为某字典串的前缀
#include<bits/stdc++.h>
using namespace std;
const int maxn = 1e5+100;
struct Aho_Corasick_Automaton
{
//basic
    int nxt[maxn*10][26],fail[maxn*10];
    int root,tot;
//special
    int flag[maxn*10];
    int len[maxn*10];
    void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        flag[tot] = len[tot]=0;
        return tot;
    }
    void insert(char *s )
    {
        int now = root;
        while (*s)
        {
            int id = *s-'a';
            if(!nxt[now][id])nxt[now][id] = newnode();
            len[nxt[now][id]] = len[now]+1;
            now = nxt[now][id];
        }
    }
    void insert(string str)
    {
        int now = root;
        for (int i=0; i<str.size(); i++)
        {
            int id = str[i]-'a';
            if(!nxt[now][id])nxt[now][id] = newnode();
            len[nxt[now][id]] = len[now]+1;
            now = nxt[now][id];
        }
    }
    void build()
    {
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (int i=0; i<26; i++)
            {
                if(!nxt[head][i])continue;
                int temp = nxt[head][i];
                fail[temp] = fail[head];
                while (fail[temp]&&!nxt[fail[temp]][i])
                {
                    fail[temp] = fail[fail[temp]];
                }
                if(head&&nxt[fail[temp]][i])fail[temp] = nxt[fail[temp]][i];
                Q.push(temp);
            }
        }
    }
    void search(string str,int QID);
    int query(string str,int QID);
} acam;
void Aho_Corasick_Automaton::search(string str,int QID)
{
    int now = root;
    for (int i=0; i<str.size(); i++)
    {
        int id = str[i]-'a';
        now = nxt[now][id];
        int temp = now;
        while (temp!=root&&flag[temp]!=QID)
        {
            flag[temp] = QID;
            temp = fail[temp];
        }
    }
}
int Aho_Corasick_Automaton::query(string str, int QID)
{
    int ans =0;
    int now = root;
    for (int i=0; i<str.size(); i++)
    {
        int id = str[i]-'a';
        now = nxt[now][id];
        int temp = now;
        while (temp!=root)
        {
            if(flag[temp]==QID)
            {
                ans = max(ans,len[temp]);
                break;
            }
            temp = fail[temp];
        }
    }
    return ans;
}
string a[maxn];
int m,n,qid;
int main()
{
    int T;
    cin>>T;
    while (T--)
    {
        acam.clear();
        cin>>n;
        for (int i=1; i<=n; i++)
        {
            cin>>a[i];
            acam.insert(a[i]);
        }
        acam.build();
        cin>>m;
        for (int i=1; i<=m; i++)
        {
            int x,y;
            cin>>x>>y;
            qid++;
            acam.search(a[x],qid);
            int ans = acam.query(a[y],qid);
            cout<<ans<<endl;
        }
    }
    return 0;
}






//AC自动机 by:LZW  HDU2896
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
const double pi=acos(-1);
const double eps=1e-6;
const int INF=0x3f3f3f3f;
using namespace std;
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}
ll lowbit(ll x)
{
    return x&(-x);
}

struct Aho_Corasick_Automaton
{
#define MAX maxn
#define type int
/***********************/
//basic
    //nxt为构建字典树的数组,fail维护失配时的转移后的结点的下标
    int nxt[MAX][100],fail[MAX];
    int root,tot;        //根节点下标和字典树中结点的数量
//special 结点中维护的信息，和结点同步更新
    type v[MAX];
/***********************/
    int getid(char c)
    {
        return c-32;     //may change
    }
    void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    //从内存池中新建一个结点
    int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        //////
        v[tot]=0;
        return tot;
    }
    //向Trie中插入一个字符串
    void insert(string str,int num)
    {
        int now = root;
        int len=str.size();
        for (int i=0; i<len; i++)
        {
            int id = getid(str[i]);
            if(!nxt[now][id])nxt[now][id] = newnode();
            now = nxt[now][id];
            if(i==len-1)
                v[now]=num;
        }
    }
    //BFS建立fail指针
    void build()
    {
        //root的fail为自己
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (int i=0; i<100; i++)
            {
                if(!nxt[head][i])continue;

                int temp = nxt[head][i];
                fail[temp] = fail[head];
                //若temp的fail没有到达root,且temp当前的fail位置的下一个位置的对应结点为空
                //则temp的fail再向前移动，这里的转移可以结合kmp算法理解
                while (fail[temp]&&!nxt[fail[temp]][i])
                {
                    fail[temp] = fail[fail[temp]];
                }
                //如果temp的fail位置的下一个对应结点存在，则直接赋值
                if(head&&nxt[fail[temp]][i])fail[temp] = nxt[fail[temp]][i];
                Q.push(temp);
            }
        }
    }
    int query(string str,int num);
#undef type
#undef MAX
} ac;

int Aho_Corasick_Automaton::query(string str,int num)
{
    set<int> s;
    int ans = 0;
    int now = root,len=str.length();
    for (int i=0; i<len; i++)
    {
        int id = getid(str[i]);
        while(now&&!nxt[now][id])
        {
            now=fail[now];
        }
        int temp=nxt[now][id];
        while(temp)
        {
            if(v[temp])
                s.insert(v[temp]);
            temp=fail[temp];
        }
        now=nxt[now][id];
    }
    if(s.size()>0)
    {
        cout<<"web "<<num<<':';
        for(int c:s)
        {
            cout<<' '<<c;
        }
        cout<<endl;
        return 1;
    }
    else
        return 0;
}
string str;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int n;
    cin>>n;
    for (int i=1; i<=n; i++)
    {
        cin>>str;
        ac.insert(str,i);
    }
    /*****别忘记build*****/
    ac.build();
    int m;
    cin>>m;
    int ans=0;
    for(int i=1;i<=m;i++)
    {
        cin>>str;
        ans+=ac.query(str,i);
    }
    cout<<"total: "<<ans<<endl;
    return 0;
}





//AC自动机求长度为n的包含一堆串中任一个串的字符串的个数
//AC自动机+矩阵快速幂模板 HDU2243
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define Accepted 0
#define ull unsigned long long
#define ll unsigned long long
using namespace std;
struct Matrix                       //矩阵结构体
{
    int sizen;
    ull m[105][105];
    void clear()
    {
        sizen=0;
        memset(m,0,sizeof(m));
    }
} M1,M2;

ll quick(ll a,ll b,ll m)
{
    ll ans=1;
    while(b)
    {
        if(b&1)
        {
            ans=ans*a;
        }
        a=a*a;
        b>>=1;
    }
    return ans;
}
namespace Matrix_quick
{
inline Matrix multi(Matrix a,Matrix b)           //矩阵乘法
{
    int sizen=a.sizen;
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(register int i=0; i<=sizen; i++)
        for(register int j=0; j<=sizen; j++)
        {
            for(register int k=0; k<=sizen; k++)
                c.m[i][j]=(c.m[i][j]+a.m[i][k]*b.m[k][j]);
        }
    return c;
}
inline Matrix quickpow(Matrix a,ll b)                   //矩阵二进制快速幂
{
    int sizen=a.sizen;
    Matrix c;
    c.sizen=sizen;
    memset(c.m,0,sizeof(c.m));
    for(register int i=0; i<=sizen; i++)                            //初始化为单位矩阵
        c.m[i][i]=1;
    while(b)
    {
        if(b&1)
            c=multi(c,a);                                       //快速幂
        a=multi(a,a);
        b>>=1;
    }
    return c;
}
}
struct Aho_Corasick_Automaton
{
#define MAX 105
#define K 26                  //may need change
#define type int
    /***********************/
//basic
    //nxt为构建字典树的数组,fail维护失配时的转移后的结点的下标
    int nxt[MAX][K],fail[MAX];
    int root,tot;        //根节点下标和字典树中结点的数量
//special 结点中维护的信息，和结点同步更新
    type v[MAX];
    /***********************/
    inline int getid(char c)
    {
        return c-'a';     //may need change
    }
    inline void clear()
    {
        memset(nxt[0],0,sizeof nxt[0]);
        root = tot=0;
    }
    //从内存池中新建一个结点
    inline int newnode()
    {
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        //////
        fail[tot]=0;
        v[tot]=0;
        return tot;
    }
    //向Trie中插入一个字符串
    inline void insert(string str)
    {
        int now = root;
        int len=str.size();
        for (register int i=0; i<len; i++)
        {
            int id = getid(str[i]);
            if(!nxt[now][id])nxt[now][id] = newnode();
            now = nxt[now][id];
            if(i==len-1)
                v[now]=1;
        }
    }
    //BFS建立fail指针
    inline void build()
    {
        //root的fail为自己
        fail[root] = root;
        queue<int>Q;
        Q.push(root);
        while (!Q.empty())
        {
            int head = Q.front();
            Q.pop();
            for (register int i=0; i<K; i++)          ////
            {
                if(nxt[head][i])
                {
                    Q.push(nxt[head][i]);
                    //此处fail不能指向自己，需要特判
                    /*
                    4 10
                    ATAA
                    TAA
                    AA
                    A
                    */
                    if(nxt[fail[head]][i]!=nxt[head][i])
                        fail[nxt[head][i]]=nxt[fail[head]][i];
                }
                else
                {
                    nxt[head][i]=nxt[fail[head]][i];
                }
                v[nxt[head][i]]|=v[nxt[fail[head]][i]];
            }
        }
    }
    inline void init()
    {
        M1.sizen=M2.sizen=tot+1;
        for(register int i=0; i<=tot; i++)
        {
            for(register int j=0; j<K; j++)
            {
                if(!v[i]&&!v[nxt[i][j]])
                {
                    M1.m[i][nxt[i][j]]++;
                }
                M2.m[i][nxt[i][j]]++;
            }
        }
        for(int i=0; i<=tot+1; i++)
            M1.m[i][tot+1]=1,M2.m[i][tot+1]=1;
         /*for(int i=0;i<=tot+1;i++)
            cout<<v[i]<<endl;
        for(int i=0;i<=tot+1;i++)
        {
            for(int j=0;j<=tot+1;j++)
            {
                cout<<M.m[i][j]<<' ';
            }
            cout<<endl;
        }*/
    }
#undef type
#undef MAX
} ac;

string str;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    register ull m,n;
    while(cin>>m>>n)
    {
        ac.clear();
        M1.clear();
        M2.clear();
        for(register int i=1; i<=m; i++)
            cin>>str,ac.insert(str);
        ac.build();
        ac.init();
        M1=Matrix_quick::quickpow(M1,n+1);
        M2=Matrix_quick::quickpow(M2,n+1);
        int sz=M1.sizen;
        /*for(int i=0;i<=sz;i++)
        {
            for(int j=0;j<=sz;j++)
            {
                cout<<M.m[i][j]<<' ';
            }
            cout<<endl;
        }*/
        ull ans=(M2.m[0][sz]-M1.m[0][sz]);
        cout<<ans<<endl;
    }
    return 0;
}















//回文树板子           by:sjf
struct Palindrome_Tree
{
    int len[MAX],next[MAX][26],fail[MAX],last,s[MAX],tot,n;
    int cnt[MAX],deep[MAX];
    int newnode(int l)
    {
        mem(next[tot],0);
        fail[tot]=0; 
        deep[tot]=cnt[tot]=0;
        len[tot]=l;
        return tot++;
    }
    void init()
    {
        tot=n=last=0;
        newnode(0);
        newnode(-1);
        s[0]=-1;
        fail[0]=1;
    }
    int get_fail(int x)
    {
        while(s[n-len[x]-1]!=s[n]) x=fail[x];
        return x;
    }
    void add(int t)//attention the type of t is int
    {
        int id,now;
        s[++n]=t;
        now=get_fail(last);
        if(!next[now][t])
        {
            id=newnode(len[now]+2);
            fail[id]=next[get_fail(fail[now])][t];
            deep[id]=deep[fail[id]]+1;
            next[now][t]=id;
        }
        last=next[now][t];
        cnt[last]++;
    }
    void count()
    {
        for(int i=tot-1;~i;i--) cnt[fail[i]]+=cnt[i];
    }
}pam; //pam.init(); 













//回文树板子              by:calabash_boy
struct Palindromic_AutoMaton{
    //basic
    int s[maxn],now;
    int nxt[maxn][26],fail[maxn],l[maxn],last,tot;
    // extension
    int num[maxn];/*节点代表的所有回文串出现次数*/
    void clear(){
        //1节点：奇数长度root 0节点：偶数长度root
        s[0]=l[1]=-1;
        fail[0] = tot = now =1;
        last = l[0]=0;
        memset(nxt[0],0,sizeof nxt[0]);
        memset(nxt[1],0,sizeof nxt[1]);
    }
    Palindromic_AutoMaton(){clear();}
    int newnode(int ll){
        tot++;
        memset(nxt[tot],0,sizeof nxt[tot]);
        fail[tot]=num[tot]=0;
        l[tot]=ll;
        return tot;
    }
    int get_fail(int x){
        while (s[now-l[x]-2]!=s[now-1])x = fail[x];
        return x;
    }
    void add(int ch){
        s[now++] = ch;
        int cur = get_fail(last);
        if(!nxt[cur][ch]){
            int tt = newnode(l[cur]+2);
            fail[tt] = nxt[get_fail(fail[cur])][ch];
            nxt[cur][ch] = tt;
        }
        last = nxt[cur][ch];num[last]++;
    }
    void build(){
        //fail[i]<i，拓扑更新可以单调扫描。
        for (int i=tot;i>=2;i--){
            num[fail[i]]+=num[i];
        }
        num[0]=num[1]=0;
        ans2 -= tot - 1;
    }
    void init(char* ss){
        while (*ss){
            add(*ss-'a');ss++;
        }
    }
    void init(string str){
        for (int i=0;i<str.size();i++){
            add(str[i]-'a');
        }
    }
    long long query();
}pam;
long long Palindromic_AutoMaton::query(){
    long long ret =1;
    for (int i=2;i<=tot;i++){
        ret = max(ret,1LL*l[i]*num[i]);
    }
    return ret;
}










//后缀自动机板子                     //by Calabash_boy
/*注意需要按l将节点基数排序来拓扑更新parent树*/
struct Suffix_Automaton{
    //basic
    int nxt[maxn*2][26],fa[maxn*2],l[maxn*2];
    int last,cnt;
    int flag[maxn*2];
    Suffix_Automaton(){ clear(); }
    void clear(){
        last =cnt=1;
        fa[1]=l[1]=0;
        memset(nxt[1],0,sizeof nxt[1]);
    }
    void init(char *s){
        while (*s){
            add(*s-'a');s++;
        }
    }
    void build(){
        int temp = 1;
        for (int i=0;i<n;i++){
            temp = nxt[temp][s[i] - 'a'];
            int now = temp;
            while (now && (flag[now] & 2) == 0){
                flag[now] |= 2;
                now = fa[now];
            }
        }
        temp = 1;
        for (int i=0;i<n;i++){
            temp = nxt[temp][t[i] - 'a'];
            int now = temp;
            while (now && (flag[now] & 1) == 0){
                flag[now] |= 1;
                now = fa[now];
            }
        }
        for (int i=1;i<=cnt;i++){
            if (flag[i] == 3){
                ans2 += l[i] - l[fa[i]];
            }
            if (flag[i] & 1){
                ans += l[i] - l[fa[i]];
            }
        }
    }
    void add(int c){
        int p = last;
        int np = ++cnt;
        memset(nxt[cnt],0,sizeof nxt[cnt]);
        l[np] = l[p]+1;last = np;
        while (p&&!nxt[p][c])nxt[p][c] = np,p = fa[p];
        if (!p)fa[np]=1;
        else{
            int q = nxt[p][c];
            if (l[q]==l[p]+1)fa[np] =q;
            else{
                int nq = ++ cnt;
                l[nq] = l[p]+1;
                memcpy(nxt[nq],nxt[q],sizeof (nxt[q]));
                fa[nq] =fa[q];fa[np] = fa[q] =nq;
                while (nxt[p][c]==q)nxt[p][c] =nq,p = fa[p];
            }
        }
    }
 
}sam;








/*
	数据结构算法
*/
//单调队列    (deque慎用，不如手写)
void mono_queue(int n,int k)
{
    deque<pair<int,int> >q;
    for(int i=1; i<=n; i++)
    {
        int m=a[i];
        while(!q.empty()&&q.back().second<m)
            q.pop_back();
        q.push_back(pair<int,int>(i,m));
        if(q.front().first<=i-k)
            q.pop_front();
    }
}






//Treap部分
//PS:按size来split可以达到区间操作，按value可达到集合操作

//不重复随机数生成函数               (Treap)
inline int Rand(){
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}

//还是随机数生成
int Seed = 19260817 ;
inline int Rand() {
    Seed ^= Seed << 7 ;
    Seed ^= Seed >> 5 ;
    Seed ^= Seed << 13 ;
    return Seed ;
}


//无旋Treap              （普通平衡树）
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-9;
const int maxn=1e5+10;
#define Accepted 0
ll gcd(ll a,ll b){return b?gcd(b,a%b):a;}
#define ls(x) arr[x].child[0]
#define rs(x) arr[x].child[1]
inline int Rand(){                                 //随机数
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}
struct node                                     //Treap结点
{
    int child[2],value,key,size;
}arr[maxn];
int tot;                                       //结点数量
inline void Push_Up(int x)                       //更新size
{
    arr[x].size=arr[ls(x)].size+arr[rs(x)].size+1;
}
void Split(int root,int& x,int& y,int value)         //按值切分无旋Treap树
{
    if(!root)
    {
        x=y=0;
        return ;
    }
    if(arr[root].value<=value) x=root,Split(rs(root),rs(x),y,value);
    else y=root,Split(ls(root),x,ls(y),value);
    Push_Up(root);
}
void Merge(int& root,int x,int y)                   //合并两个无旋Treap子树
{
    if(!x||!y)
    {
        root=x+y;
        return ;
    }
    if(arr[x].key<arr[y].key)root=x,Merge(rs(root),rs(x),y);
    else root=y,Merge(ls(root),x,ls(y));
    Push_Up(root);
}
void Insert(int& root,int value)                  //插入结点
{
    int x=0,y=0,z=++tot;
    arr[z].key=Rand(),arr[z].size=1,arr[z].value=value;
    Split(root,x,y,value);
    Merge(x,x,z);
    Merge(root,x,y);
}
void Erase(int& root,int value)                //删除结点
{
    int x=0,y=0,z=0;
    Split(root,x,y,value);
    Split(x,x,z,value-1);
    Merge(z,ls(z),rs(z));
    Merge(x,x,z);
    Merge(root,x,y);
}
int Kth_number(int root,int k)               //树上第k小
{
    while(arr[ls(root)].size+1!=k)
    {
        if(arr[ls(root)].size>=k)root=ls(root);
        else k-=arr[ls(root)].size+1,root=rs(root);
    }
    return arr[root].value;
}
int Get_rank(int& root,int value)             //树上名次
{
    int x=0,y=0;
    Split(root,x,y,value-1);
    int res=arr[x].size+1;
    Merge(root,x,y);
    return res;
}
int Pre(int& root,int value)                //value的前驱
{
    int x=0,y=0;
    Split(root,x,y,value-1);
    int res=Kth_number(x,arr[x].size);
    Merge(root,x,y);
    return res;
}
int Suf(int& root,int value)               //value的后继
{
    int x=0,y=0;
    Split(root,x,y,value);
    int res=Kth_number(y,1);
    Merge(root,x,y);
    return res;
}
int root;                                //root=0,初始化空树
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    int n;
    cin>>n;
    while(n--)
    {
        int opt,x;
        cin>>opt>>x;
        if(opt==1)
            Insert(root,x);
        else if(opt==2)
            Erase(root,x);
        else if(opt==3)
            cout<<Get_rank(root,x)<<endl;
        else if(opt==4)
            cout<<Kth_number(root,x)<<endl;
        else if(opt==5)
            cout<<Pre(root,x)<<endl;
        else if(opt==6)
            cout<<Suf(root,x)<<endl;
    }
    return Accepted;
}







//文艺平衡树 无旋Treap实现
#include<bits/stdc++.h>
using namespace std;
#define ll long long
#define ld long double
const ld pi=acos(-1);
const ll mod=1e9+7;
#define lowbit(x) (x&(-x))
const ld eps=1e-9;
const int maxn=1e5+10;
#define Accepted 0
ll gcd(ll a,ll b)
{
    return b?gcd(b,a%b):a;
}
#define ls(x) arr[x].child[0]
#define rs(x) arr[x].child[1]
inline int Rand()                                  //随机数
{
    static int seed=703; //seed可以随便取
    return seed=int(seed*48271LL%2147483647);
}
struct node                                     //Treap结点
{
    int child[2],value,key,size,tag;
} arr[maxn];
int tot;                                       //结点数量
inline void Push_Up(int x)
{
    arr[x].size=arr[ls(x)].size+arr[rs(x)].size+1;
}

inline void Push_Down(int x)                  //下推翻转标记
{
    if(arr[x].tag)
    {
        arr[ls(x)].tag^=1;
        arr[rs(x)].tag^=1;
        swap(ls(x),rs(x));
        arr[x].tag^=1;
    }
}

void Split(int root,int &x,int &y,int Sz)           //按Size切分
{
    if(!root)
    {
        x=y=0;
        return ;
    }
    Push_Down(root);
    if(Sz<=arr[ls(root)].size)
        y=root,Split(ls(root),x,ls(y),Sz);
    else x=root,Split(rs(root),rs(root),y,Sz-arr[ls(root)].size-1);
    Push_Up(root);
}

void Merge(int&root,int x,int y)             //合并子树
{
    if(!x||!y)
    {
        root=x+y;
        return;
    }
    Push_Down(x),Push_Down(y);
    if(arr[x].key<arr[y].key)root=x,Merge(rs(root),rs(x),y);
    else root=y,Merge(ls(root),x,ls(y));
    Push_Up(root);
}

inline void Insert(int&root,int value)          //插入一个结点
{
    int x=0,y=0,z=++tot;
    arr[z].size=1,arr[z].value=value,arr[z].key=Rand(),arr[z].tag=0;
    Split(root,x,y,value);
    Merge(x,x,z);
    Merge(root,x,y);
}

void Rever(int&root,int L,int R)                 //翻转一个区间
{
    int x=0,y=0,z=0,t=0;
    Split(root,x,y,R);
    Split(x,z,t,L-1);
    arr[t].tag^=1;
    Merge(x,z,t);
    Merge(root,x,y);
}
void Print(int x)                             //中序遍历输出
{
    if(!x)return ;
    Push_Down(x);
    Print(ls(x));
    cout<<arr[x].value<<' ';
    Print(rs(x));
}
int root;
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(Accepted);
    cout.tie(Accepted);
    int n,m;
    cin>>n>>m;
    for(int i=1; i<=n; i++)
        Insert(root,i);
    while(m--)
    {
        int a,b;
        cin>>a>>b;
        Rever(root,a,b);
    }
    Print(root);
    return Accepted;
}



//指针版Treap 省内存
// It is made by XZZ
#include<cstdio>
#include<algorithm>
#define pr pair<point,point>
#define mp make_pair
using namespace std;
#define rep(a,b,c) for(rg int a=b;a<=c;a++)
#define drep(a,b,c) for(rg int a=b;a>=c;a--)
#define erep(a,b) for(rg int a=fir[b];a;a=nxt[a])
#define il inline
#define rg register
#define vd void
typedef long long ll;
il int gi(){
    rg int x=0,f=1;rg char ch=getchar();
    while(ch<'0'||ch>'9')f=ch=='-'?-1:f,ch=getchar();
    while(ch>='0'&&ch<='9')x=x*10+ch-'0',ch=getchar();
    return x*f;
}
int seed=19260817;
il int Rand(){return seed=seed*48271ll%2147483647;}
typedef struct node* point;
point null;
struct node{
    char data;
    int size,rand;
    point ls,rs;
    bool rev;
    node(char ch){data=ch,size=1,rand=Rand(),rev=0,ls=rs=null;}
    il vd down(){if(rev)rev=0,ls->rev^=1,rs->rev^=1,swap(ls,rs);}
    il vd reset(){if(ls!=null)ls->down();if(rs!=null)rs->down();size=ls->size+rs->size+1;}
};
point root=null;
il point build(int n){
    point stack[n+1];
    int top=0;
    char ch;
    rep(i,1,n){
    ch=getchar();while(ch=='\n')ch=getchar();
    point now=new node(ch),lst=null;
    while(top&&stack[top]->rand>now->rand)lst=stack[top],stack[top--]->reset();
    now->ls=lst;if(top)stack[top]->rs=now;stack[++top]=now;
    }
    while(top)stack[top--]->reset();
    return stack[1];
}
il point merge(point a,point b){
    if(a==null)return b;
    if(b==null)return a;
    if(a->rand<b->rand){a->down(),a->rs=merge(a->rs,b),a->reset();return a;}
    else {b->down(),b->ls=merge(a,b->ls),b->reset();return b;}
}
il pr split(point now,int num){
    if(now==null)return mp(null,null);
    now->down();
    point ls=now->ls,rs=now->rs;
    if(num==ls->size){now->ls=null,now->reset();return mp(ls,now);}
    if(num==ls->size+1){now->rs=null,now->reset();return mp(now,rs);}
    if(num<ls->size){pr T=split(ls,num);now->ls=T.second,now->reset();return mp(T.first,now);}
    pr T=split(rs,num-ls->size-1);now->rs=T.first,now->reset();return mp(now,T.second);
}
il vd del(point now){if(now!=null)del(now->ls),del(now->rs),delete now;}
int main(){
    int m=gi()-1,pos=0;
    char opt[10];
    null=new node('%');
    null->size=0;
    {
    scanf("%*s");
    root=build(gi());
    }
    while(m--){
    scanf("%s",opt);
    if(opt[0]=='M')pos=max(0,min(gi(),root->size));
    else if(opt[0]=='P'){if(pos)--pos;}
    else if(opt[0]=='N'){if(pos!=root->size)++pos;}
    else if(opt[0]=='G'){
        pr T=split(root,pos);
        char lst;point now=T.second;
        while(now!=null)lst=now->data,now->down(),now=now->ls;
        printf("%c\n",lst);
        root=merge(T.first,T.second);
    }
    else if(opt[0]=='I'){
        pr T=split(root,pos);
        root=merge(T.first,merge(build(gi()),T.second));
    }
    else{
        pr T=split(root,pos),TT=split(T.second,gi());
        if(opt[0]=='D')root=merge(T.first,TT.second),del(TT.first);
        else TT.first->rev^=1,root=merge(T.first,merge(TT.first,TT.second));
    }
    }
    del(root);
    delete null;
    return 0;
}







//Treap
struct node{ //节点数据的结构
    int key,prio,size; //size是指以这个节点为根的子树中节点的数量
    node* ch[2]; //ch[0]指左儿子，ch[1]指右儿子
};

typedef node* tree;

node base[MAXN],nil;
tree top,null,root;

void init(){                //初始化top和null
    top=base;
    root=null=&nil;
    null->ch[0]=null->ch[1]=null;
    null->key=null->prio=2147483647;
    null->size=0;
}

inline tree newnode(int k){ //注意这种分配内存的方法也就比赛的时候用用，仅仅是为了提高效率
    top->key=k;				//看为val,是BST的键值
    top->size=1;
    top->prio=random();
    top->ch[0]=top->ch[1]=null;
    return top++;
}


//Treap结点的旋转
void rotate(tree &x,bool d){ //d指旋转的方向，0为左旋，1为右旋
    tree y=x->ch[!d];        //x为要旋的子树的根节点
    x->ch[!d]=y->ch[d];
    y->ch[d]=x;
    x->size=x->ch[0]->size+1+x->ch[1]->size;
    y->size=y->ch[0]->size+1+y->ch[1]->size;
    x=y;
}

void insert(tree &t,int key){ //插入一个节点
    if (t==null) t=newnode(key);
    else{
        bool d=key>t->key;
        insert(t->ch[d],key);
        t->size++;
        if (t->prio<t->ch[d]->prio) rotate(t,!d);
    }
}

void erase(tree &t,int key){ //删除一个节点
    if (t->key!=key){
        erase(t->ch[key>t->key],key);
        t->size--;
	}
    else if (t->ch[0]==null) t=t->ch[1];
    else if (t->ch[1]==null) t=t->ch[0];
    else{
        bool d=t->ch[0]->prio<t->ch[1]->prio;
        rotate(t,d);
        erase(t->ch[d],key);
    }
}

tree select(int k){ //选择第k小节点
    tree t=root;
    for (int tmp;;){
        tmp=t->ch[0]->size+1;
        if (k==tmp) return t;
        if (k>tmp){
            k-=tmp;
            t=t->ch[1];
        }
        else t=t->ch[0];
    }
}









//Treap   by:sjf
struct Treap
{
	#define type ll
	struct node
	{
		int ch[2],fix,sz,w;
		type v;
		node(){}
		node(type x)
		{
			v=x;
			fix=rand();
			sz=w=1;
			ch[0]=ch[1]=0;
		} 
	}t[MAX];  
	int tot,root,tmp;
	void init()
	{
		srand(unsigned(new char));
		root=tot=0;
		t[0].sz=t[0].w=0;
		mem(t[0].ch,0);
	}
	inline void maintain(int k)  
	{  
		t[k].sz=t[t[k].ch[0]].sz+t[t[k].ch[1]].sz+t[k].w ;  
	}  
	inline void rotate(int &id,int k)
	{
		int y=t[id].ch[k^1];
		t[id].ch[k^1]=t[y].ch[k];
		t[y].ch[k]=id;
		maintain(id);
		maintain(y);
		id=y;
	}
	void insert(int &id,type v)
	{
		if(!id) t[id=++tot]=node(v);
		else
		{
			if(t[id].sz++,t[id].v==v)t[id].w++;
			else if(insert(t[id].ch[tmp=v>t[id].v],v),t[t[id].ch[tmp]].fix>t[id].fix) rotate(id,tmp^1);
	    }
	}
	void erase(int &id,type v)
	{
		if(!id)return;
		if(t[id].v==v)
		{
			if(t[id].w>1) t[id].w--,t[id].sz--;
			else
			{
				if(!(t[id].ch[0]&&t[id].ch[1])) id=t[id].ch[0]|t[id].ch[1];
				else
				{
					rotate(id,tmp=t[t[id].ch[0]].fix>t[t[id].ch[1]].fix);
					t[id].sz--;
					erase(t[id].ch[tmp],v);
				}
			}
		}
		else
		{
			t[id].sz--;
			erase(t[id].ch[v>t[id].v],v);
		}
	}
	type kth(int k)//k small
	{
		int id=root;
		if(id==0) return 0;
		while(id)
		{
			if(t[t[id].ch[0]].sz>=k) id=t[id].ch[0];
			else if(t[t[id].ch[0]].sz+t[id].w>=k) return t[id].v;
			else
			{
				k-=t[t[id].ch[0]].sz+t[id].w;
				id=t[id].ch[1];
			}
		}
	}
	int find(type key,int f)
	{
		int id=root,res=0;
		while(id)
		{
			if(t[id].v<=key)
			{
				res+=t[t[id].ch[0]].sz+t[id].w;
				if(f&&key==t[id].v) res-=t[id].w;
				id=t[id].ch[1];
			}
			else id=t[id].ch[0];
		}
		return res;
	}
	type find_pre(type key)
	{
		type res=-LLINF;
		int id=root;
		while(id)
		{
			if(t[id].v<key)
			{
				res=max(res,t[id].v);
				id=t[id].ch[1];
			}
			else id=t[id].ch[0];
		}
		return res;
	}
	type find_suc(type key)
	{
		type res=LLINF;
		int id=root;
		while(id)
		{
			if(t[id].v>key)
			{
				res=min(res,t[id].v);
				id=t[id].ch[0];
			}
			else id=t[id].ch[1];
		}
		return res;
	}
	void insert(type v){insert(root,v);}
	void erase(type v){erase(root,v);}
	int upper_bound_count(type key){return find(key,0);}//the count >=key
	int lower_bound_count(type key){return find(key,1);}//the count >key
	int rank(type key){return lower_bound_count(key)+1;}
	#undef type
}t; //t.init();




//并查集
int fa[maxn];
int Find(int x)
{
    return x==fa[x]?x:fa[x]=Find(fa[x]);
}
void unit(int u,int v)
{
    int xx=Find(u);
    int yy=Find(v);
    if(xx!=yy)
    {
        fa[xx]=yy;
    }
}





//RMQ算法            求解区间最值，预处理O(nlogn) 查询O(1)
int a[maxn];
int maxsum[maxn][20],minsum[maxn][20];     //dp[i][j] 指a[i]-a[i+2^j-1]这长度为2^j串的最小（最大）值
void RMQ(int num) //预处理 O(nlogn)       一共num个数 且num<2^20（倍增）
{
    for(int i=1;i<=num;i++)
        maxsum[i][0]=minsum[i][0]=a[i];
    for(int j = 1; j < 20; j++)
    {
        for(int i = 1; i <= num; i++)
        {
            if(i + (1 << j) - 1 <= num)
            {
                maxsum[i][j] = max(maxsum[i][j - 1], maxsum[i + (1 << (j - 1))][j - 1]);
                minsum[i][j] = min(minsum[i][j - 1], minsum[i + (1 << (j - 1))][j - 1]);
            }
        }
    }
}
int maxl,minl;
void Query(int l,int r)
{
    int k=(int)((log(r-l+1))/log(2.0));
    maxl=max(maxsum[l][k],maxsum[r-(1<<k)+1][k]);
    minl=min(minsum[l][k],minsum[r-(1<<k)+1][k]);
}






//莫队算法  O(nsqrt(n))
ll block;
ll nowAns = 0;
ll a[maxn];
inline ll calc(ll x)
{
    return x*(x-1)/2;
}
ll ans[maxn];
struct Query
{
    int l, r, id;
    void input()
    {
        scanf("%d%d",&l,&r);
    }
    bool operator < (const Query &rhs) const
    {
        if(l/block == rhs.l/block) return r > rhs.r;
        return l > rhs.l;
    }
} q[maxn];

void add(int p)
{
    nowAns +=mp[a[p]];          //此处注意修改
    mp[a[p]]++;
}

void del(int p)
{
    mp[a[p]]--;                 //此处注意修改
    nowAns -= mp[a[p]];
}

void Mos(int n,int m)
{
    for(int i=0;i<m;i++)              //初始化询问的编号
        q[i].id=i;
    block = (int)sqrt(n);
    sort(q, q + m);
    int l = 1, r = 1;              //当前窗口
    mp[a[1]]++;                     //别忘记初始化
    for(int i = 0; i < m; i++)
    {
        const Query &c = q[i];
        if(c.l==c.r){                           //特判
            ans[c.id].fz = 0,ans[c.id].fm =1;
            continue;
        }
        while(l > c.l) add(--l);      //四种转移 O(1)
        while(r < c.r) add(++r);
        while(l < c.l) del(l++);
        while(r > c.r) del(r--);
        ans[c.id].fz = nowAns;
        ans[c.id].fm = calc(r+1-l);
    }
    for(int i = 0; i < m; i++)                //按序号输出
    {
        ll g=gcd(ans[i].fz,ans[i].fm);
        printf("%lld/%lld\n",ans[i].fz/g,ans[i].fm/g);
    }
}





//树状数组 维护区间和 板子
int a[maxn];
int c[maxn];
int lowbit(int x)
{
    return x&(-x);
}
void update(int pos,int r,int val)               //单点更新 在pos的位置+val O(logn)
{
    for(int i=pos;i<=r;i+=lowbit(i))
    {
        c[i]+=val;
    }
}

int getsum(int x)          //查询从1到x的前缀和O(logn)
{
    int sum=0;
    for(int i=x;i>=1;i-=lowbit(i))
    {
        sum+=c[i];
    }
    return sum;
}

int getsum(int l,int r)       //查询从l到r的区间和
{
    return getsum(r)-getsum(l-1);
}






//树状数组 by:sjf
//一维 
struct Fenwick_Tree
{
	#define type int
	type bit[MAX];
	int n;
	void init(int _n){n=_n;mem(bit,0);}
	int lowbit(int x){return x&(-x);}
	void insert(int x,type v)
	{
		while(x<=n)
		{
			bit[x]+=v;
			x+=lowbit(x);
		}
	}
	type get(int x)
	{
		type res=0;
		while(x)
		{
			res+=bit[x];
			x-=lowbit(x);
		}
		return res;
	}
	type query(int l,int r)
	{
		return get(r)-get(l-1);
	}
	#undef type
}tr;

//二维
struct Fenwick_Tree
{
	#define type int
	type bit[MAX][MAX];
	int n,m;
	void init(int _n,int _m){n=_n;m=_m;mem(bit,0);}
	int lowbit(int x){return x&(-x);}
	void update(int x,int y,type v)
	{
		int i,j;
		for(i=x;i<=n;i+=lowbit(i))
		{
			for(j=y;j<=m;j+=lowbit(j))
			{
				bit[i][j]+=v;
			}
		} 
	}
	type get(int x,int y)
	{
		type i,j,res=0;
		for(i=x;i>0;i-=lowbit(i))
		{
			for(j=y;j>0;j-=lowbit(j))
			{
				res+=bit[i][j];
			}
		}
		return res;
	}
	type query(int x1,int x2,int y1,int y2)
	{
		x1--;
		y1--;
		return get(x2,y2)-get(x1,y2)-get(x2,y1)+get(x1,y1);
	}
	#undef type
}tr;

//区间
struct Fenwick_Tree
{
	#define type int
	type bit[MAX][2];
	int n;
	void init(int _n){n=_n;mem(bit,0);}
	int lowbit(int x){return x&(-x);}
	void insert(int x,type v)
	{
		for(int i=x;i<=n;i+=lowbit(i))
		{
			bit[i][0]+=v;
			bit[i][1]+=v*(x-1);
		}
	}
	void upd(int l,int r,type v)
	{
		insert(l,v);
		insert(r+1,-v);
	}
	type get(int x)
	{
		type res=0;
		for(int i=x;i;i-=lowbit(i))
		{
			res+=x*bit[i][0]-bit[i][1];
		}
		return res;
	}
	type ask(int l,int r)
	{
		return get(r)-get(l-1);
	}
	#undef type
}tr;









//线段树 (NO lazy)
//[l,r]区间为一个活动的区间且为结点rt所指代的区间
int a[maxn],n;
int Sum[maxn<<2];
void update(int index)
{
    Sum[index]=Sum[index<<1]+Sum[index<<1|1];               //此为区间和的模板，如有其他区间合并运算修改请在此修改
}
void build(int l,int r,int rt)
{
    if(l==r)
    {
        Sum[rt]=a[l];
        return;
    }
    int m=(l+r)>>1;
    build(l,m,rt<<1);
    build(m+1,r,rt<<1|1);
    update(rt);
}
int ans_ans;
void ask(int A,int l,int r,int rt)     //单点查询
{
    if(l==r)
    {
        ans_ans=Sum[rt];
        return;
    }
    int m=(l+r)>>1;
    if(A<=m)
        ask(A,l,m,rt<<1);
    else
        ask(A,m+1,r,rt<<1|1);
}

void add(int L,int C,int l,int r,int rt)          //单点修改
{
    if(l==r)
    {
        Sum[rt]+=C;
        return ;
    }
    int m=(l+r)>>1;
    if(L<=m)
        add(L,C,l,m,rt<<1);
    else
        add(L,C,m+1,r,rt<<1|1);
    update(rt);
}
//求和区间[L,R]永不改变
int Query(int L,int R,int l,int r,int rt)          //区间求和
{
    if(l>=L&&r<=R)
    {
        return Sum[rt];
    }
    int m=(l+r)>>1;
    int ans=0;
    if(L<=m)
        ans+=Query(L,R,l,m,rt<<1);
    if(R>m)
        ans+=Query(L,R,m+1,r,rt<<1|1);
    return ans;
} 







//线段树板子2 SegmentTree
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
bool Finish_read;
template<class T>inline void read(T &x)
{
    Finish_read=0;
    x=0;
    int f=1;
    char ch=getchar();
    while(!isdigit(ch))
    {
        if(ch=='-')f=-1;
        if(ch==EOF)return;
        ch=getchar();
    }
    while(isdigit(ch))x=x*10+ch-'0',ch=getchar();
    x*=f;
    Finish_read=1;
}
template<class T>inline void print(T x)
{
    if(x/10!=0)print(x/10);
    putchar(x%10+'0');
}
template<class T>inline void writeln(T x)
{
    if(x<0)putchar('-');
    x=abs(x);
    print(x);
    putchar('\n');
}
template<class T>inline void write(T x)
{
    if(x<0)putchar('-');
    x=abs(x);
    print(x);
}
/*================Header Template==============*/
const int maxn=2e5+5;
#define ls(o) o<<1
#define rs(o) (o<<1|1)
int n,m;
ll p;
int a[maxn];
struct node
{
    int l,r,sz;
    ll val;
    ll addtag,multag;
} t[maxn<<2];
/*==================Define Area================*/
namespace SegmentTree
{
void update(int o)
{
    t[o].val=(t[ls(o)].val+t[rs(o)].val)%p;
}

void pushdown(int o)
{
    if(t[o].multag!=1)
    {
        t[ls(o)].multag*=t[o].multag;
        t[ls(o)].multag%=p;
        t[rs(o)].multag*=t[o].multag;
        t[rs(o)].multag%=p;
        t[ls(o)].addtag*=t[o].multag;
        t[ls(o)].addtag%=p;
        t[rs(o)].addtag*=t[o].multag;
        t[rs(o)].addtag%=p;
        t[ls(o)].val*=t[o].multag;
        t[ls(o)].val%=p;
        t[rs(o)].val*=t[o].multag;
        t[rs(o)].val%=p;
        t[o].multag=1;
    }
    if(t[o].addtag)
    {
        t[ls(o)].addtag+=t[o].addtag;
        t[ls(o)].addtag%=p;
        t[rs(o)].addtag+=t[o].addtag;
        t[rs(o)].addtag%=p;
        t[ls(o)].val+=t[o].addtag*t[ls(o)].sz;
        t[ls(o)].val%=p;
        t[rs(o)].val+=t[o].addtag*t[rs(o)].sz;
        t[rs(o)].val%=p;
        t[o].addtag=0;
    }
    return ;
}

void Build(int o,int l,int r)
{
    t[o].l=l,t[o].r=r;
    t[o].sz=r-l+1;
    t[o].multag=1;
    if(t[o].l==t[o].r)
    {
        t[o].val=a[l];
        return ;
    }
    int mid=(l+r)>>1;
    Build(ls(o),l,mid);
    Build(rs(o),mid+1,r);
    update(o);
}

ll IntervalSum(int o,int l,int r)
{
    if(t[o].l>r||t[o].r<l) return 0;
    if(t[o].l>=l&&t[o].r<=r)
    {
        pushdown(o);
        return t[o].val;
    }
    pushdown(o);
    ll ans=0;
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) ans+=IntervalSum(ls(o),l,r),ans%=p;
    if(mid<=r) ans+=IntervalSum(rs(o),l,r),ans%=p;
    return ans;
}

void IntervalAdd(int o,int l,int r,int v)
{
    if(t[o].l>r||t[o].r<l) return ;
    if(t[o].l>=l&&t[o].r<=r)
    {
        t[o].val+=t[o].sz*v;
        t[o].addtag+=v;
        t[o].addtag%=p;
        t[o].val%=p;
        return ;
    }
    pushdown(o);
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) IntervalAdd(ls(o),l,r,v);
    if(mid<=r) IntervalAdd(rs(o),l,r,v);
    update(o);
    return ;
}

void IntervalMul(int o,int l,int r,int v)
{
    if(t[o].l>r||t[o].r<l) return ;
    if(t[o].l>=l&&t[o].r<=r)
    {
        pushdown(o);
        t[o].val*=v;
        t[o].val%=p;
        t[o].addtag*=v;
        t[o].addtag%=p;
        t[o].multag*=v;
        t[o].multag%=p;
        return ;
    }
    pushdown(o);
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) IntervalMul(ls(o),l,r,v);
    if(mid<=r) IntervalMul(rs(o),l,r,v);
    update(o);
    return ;
}
}
using namespace SegmentTree;

int main()
{
    read(n);
    read(m);
    read(p);
    for(int i=1; i<=n; i++)
    {
        read(a[i]);
    }
    Build(1,1,n);
    for(int i=1; i<=m; i++)
    {
        int opt;
        read(opt);
        if(opt==1)
        {
            int x,y,k;
            read(x);
            read(y);
            read(k);
            IntervalMul(1,x,y,k);
        }
        if(opt==2)
        {
            int x,y,k;
            read(x);
            read(y);
            read(k);
            IntervalAdd(1,x,y,k);
        }
        if(opt==3)
        {
            int x,y;
            read(x);
            read(y);
            ll ans=IntervalSum(1,x,y);
            printf("%lld\n",ans);
        }
    }
    return 0;
}

#include <iostream>
#include <cstdio>
#include <cstring>
#include <stack>
#include <queue>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
const double eps = 1e-6;
const double pi = acos(-1.0);
const int INF = 0x3f3f3f3f;
const int MOD = 1000000007;
#define ll long long
#define CL(a,b) memset(a,b,sizeof(a))
#define MAXN 100010
 
struct node
{
    int l,r;
    ll s,add;//add为每次加的数
}t[MAXN<<2];
int hh[MAXN];
int n,q;
ll ans;
 
void build(int l, int r, int i)
{
    t[i].l = l;
    t[i].r = r;
    t[i].add = 0;
    if(l == r) return ;
    int mid = (l+r)>>1;
    build(l, mid, i<<1);
    build(mid+1, r, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
void update(int l, int r, int add, int i)
{
    if(t[i].l>r || t[i].r<l) return ;
    if(t[i].l>=l && t[i].r<=r)
    {
        t[i].s += (t[i].r-t[i].l+1)*add;
        t[i].add += add;
        return ;
    }
    if(t[i].add)
    {
        t[i<<1].s += (t[i<<1].r-t[i<<1].l+1)*t[i].add;
        t[i<<1].add += t[i].add;
        t[i<<1|1].s += (t[i<<1|1].r-t[i<<1|1].l+1)*t[i].add;
        t[i<<1|1].add += t[i].add;
        t[i].add = 0;
    }
    update(l, r, add, i<<1);
    update(l, r, add, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
void query(int l, int r, int i)
{
    if(t[i].l>r || t[i].r<l) return ;
    if(t[i].l>=l && t[i].r<=r)
    {
        ans += t[i].s;
        return ;
    }
    if(t[i].add)
    {
        t[i<<1].s += (t[i<<1].r-t[i<<1].l+1)*t[i].add;
        t[i<<1].add += t[i].add;
        t[i<<1|1].s += (t[i<<1|1].r-t[i<<1|1].l+1)*t[i].add;
        t[i<<1|1].add += t[i].add;
        t[i].add = 0;
    }
    query(l, r, i<<1);
    query(l, r, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
int main()
{
    int a,b,c;
    ll k;
    char ch;
    while(scanf("%d%d",&n,&q)==2)
    {
        for(int i=1; i<=n; i++)
            scanf("%d",&hh[i]);
        build(1, n, 1);
        for(int i=1; i<=n; i++)
            update(i, i, hh[i], 1);
        while(q--)
        {
            getchar();
            scanf("%c",&ch);
            if(ch == 'C')
            {
                scanf("%d%d%d",&a,&b,&c);
                update(a, b, c, 1);
            }
            if(ch == 'Q')
            {
                ans = 0;
                scanf("%d%d",&a,&b);
                query(a, b, 1);
                printf("%lld\n",ans);
            }
        }
    }
    return 0;
}

//还是线段树板子 维护区间和，区间平方和，支持区间修改
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
struct node
{
    ll sum;//当前节点所表示的区间的和
    ll asign;//加法延迟标记
    ll msign;//乘法延迟标记
    ll sq;
};
ll a[10009];//以此数组建树
ll n,m;//数组的大小,取模,询问次数
node t[4*10009];
void build(int root,int l,int r)//build(1,1,n)进行建树
{
    t[root].msign=1;
    if(l==r)
    {t[root].sum=a[l];
    t[root].sq=a[l]*a[l];
    return ;}
    int mid=(l+r)>>1;
    build(root<<1,l,mid);
    build(root<<1|1,mid+1,r);
    t[root].sum=t[root<<1].sum+t[root<<1|1].sum;
    t[root].sq=t[root<<1].sq+t[root<<1|1].sq;
}
void push_down(int rt,int l,int r)
{
    if(t[rt].msign!=1)
    {
        t[rt<<1].msign*=t[rt].msign;
        t[rt<<1].asign*=t[rt].msign;
        t[rt<<1|1].msign*=t[rt].msign;
        t[rt<<1|1].asign*=t[rt].msign;
        t[rt<<1].sum*=t[rt].msign;
        t[rt<<1].sq*=t[rt].msign*t[rt].msign;
        t[rt<<1|1].sum*=t[rt].msign;
        t[rt<<1|1].sq*=t[rt].msign*t[rt].msign;
        t[rt].msign=1;
    }
    if(t[rt].asign)
    {
        int mid=(l+r)>>1;
        t[rt<<1].sq+=(2*t[rt].asign*t[rt<<1].sum+t[rt].asign*t[rt].asign*(mid-l+1));
        t[rt<<1].sum+=(t[rt].asign*(mid-l+1));
        t[rt<<1|1].sq+=(2*t[rt].asign*t[rt<<1|1].sum+t[rt].asign*t[rt].asign*(r-mid));
        t[rt<<1|1].sum+=(t[rt].asign*(r-mid));
        t[rt<<1].asign+=t[rt].asign;
        t[rt<<1|1].asign+=t[rt].asign;
        t[rt].asign=0;
    }
}
void range_add(int rt,int l,int r,int x,int y,ll val)//[x,y]区间加上val
{
    if(x<=l&&y>=r)
    {
        t[rt].sq=t[rt].sq+2*val*t[rt].sum+(r-l+1)*val*val;
        t[rt].sum=t[rt].sum+(r-l+1)*val;
        t[rt].asign=t[rt].asign+val;
        return ;
    }
    if(t[rt].asign!=0||t[rt].msign!=1)
        push_down(rt,l,r);
    int mid=(l+r)>>1;
    if(x<=mid)
        range_add(rt<<1,l,mid,x,y,val);
    if(y>mid)
        range_add(rt<<1|1,mid+1,r,x,y,val);
    t[rt].sum=(t[rt<<1].sum+t[rt<<1|1].sum);
    t[rt].sq=t[rt<<1].sq+t[rt<<1|1].sq;
}
void range_mul(int rt,int l,int r,int x,int y,ll val)//[x,y]区间乘上val
{
    if(x<=l&&y>=r)
    {
        t[rt].sq=val*val*t[rt].sq;
        t[rt].sum=val*t[rt].sum;
        t[rt].asign=(t[rt].asign*val);
        t[rt].msign=(t[rt].msign*val);
        return ;
    }
    if(t[rt].asign!=0||t[rt].msign!=1)
        push_down(rt,l,r);
    int mid=(l+r)>>1;
    if(x<=mid)
        range_mul(rt<<1,l,mid,x,y,val);
    if(y>mid)
        range_mul(rt<<1|1,mid+1,r,x,y,val);
    t[rt].sum=(t[rt<<1].sum+t[rt<<1|1].sum);
    t[rt].sq=t[rt<<1].sq+t[rt<<1|1].sq;
}
ll query_sum(int rt,int l,int r,int x,int y)//查询[x,y]的和
{
    if(x<=l&&y>=r)
        return t[rt].sum;
    if(t[rt].asign!=0||t[rt].msign!=1)
    push_down(rt,l,r);
    int mid=(l+r)>>1;
    ll sum=0;
    if(x<=mid)
        sum+=query_sum(rt<<1,l,mid,x,y);
    if(y>mid)
        sum+=query_sum(rt<<1|1,mid+1,r,x,y);
    return sum;
}
ll query_sq(int rt,int l,int r,int x,int y)
{
    if(x<=l&&y>=r)
        return t[rt].sq;
    if(t[rt].asign!=0||t[rt].msign!=1)
    push_down(rt,l,r);
    int mid=(l+r)>>1;
    ll sum=0;
    if(x<=mid)
        sum+=query_sq(rt<<1,l,mid,x,y);
    if(y>mid)
        sum+=query_sq(rt<<1|1,mid+1,r,x,y);
    return sum;
}
int main()
{
    cin.sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int opt,l,r;
    ll x;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
        {
            cin>>a[i];
        }
    build(1,1,n);
    for(int i=1;i<=m;i++)
    {
        cin>>opt;
        if(opt==1)
        {
            cin>>l>>r;
            cout<<query_sum(1,1,n,l,r)<<'\n';
        }
        if(opt==2)
        {
            cin>>l>>r;
            cout<<query_sq(1,1,n,l,r)<<'\n';
        }
        if(opt==3)
        {
            cin>>l>>r>>x;
            range_mul(1,1,n,l,r,x);
        }
        if(opt==4)
        {
            cin>>l>>r>>x;
            range_add(1,1,n,l,r,x);
        }
    }
    return 0;
}




//sjf线段树板子
struct Segment_Tree
{
	#define type int
	#define ls (id<<1)
	#define rs (id<<1|1)
	int n,ql,qr;
	type a[MAX],v[MAX<<2],tag[MAX<<2],qv;
	void pushup(int id)
	{
		
	}
	void pushdown(int id)
	{
		if(!tag[id]) return;
		
	}
	void build(int l,int r,int id)
	{
		tag[id]=0;
		if(l==r)
		{
			v[id]=a[l];
			return;
		}
		int mid=(l+r)>>1;
		build(l,mid,ls);
		build(mid+1,r,rs);
		pushup(id);
	}
	void update(int l,int r,int id)
	{
		if(l>=ql&&r<=qr)
		{
			
			return;
		}
		pushdown(id);
		int mid=(l+r)>>1;
		if(ql<=mid) update(l,mid,ls);
		if(qr>mid) update(mid+1,r,rs);
		pushup(id);
	}
	type res;
	void query(int l,int r,int id)
	{
		if(l>=ql&&r<=qr)
		{
			res+=v[id];
			return;
		}
		pushdown(id);
		int mid=(l+r)>>1;
		if(ql<=mid) query(l,mid,ls);
		if(qr>mid) query(mid+1,r,rs);
	}
	void build(int _n){n=_n;build(1,n,1);}
	void upd(int l,int r,type v)
	{
		ql=l;
		qr=r;
		qv=v;
		update(1,n,1);
	}
	type ask(int l,int r)
	{
		ql=l;
		qr=r;
		res=0;
		query(1,n,1);
		return res;
	}
	#undef type
	#undef ls
	#undef rs
}tr;


//离散化模板
vector<int> v;
int scatter(vector<int> &v)
{
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
    return v.size();
}
inline int getid(int x)
{
    return lower_bound(v.begin(), v.end(), x) - v.begin() + 1;
}


//离散化板子  **未验证
void scatter(int a[],int n)
{
    for(int i=0;i<n;i++)
    {
        b[i]=a[i];
    }
    sort(b,b+n);
    int sz=unique(b,b+n)-b;
    for(int i=0;i<n;i++)
    {
        c[i]=lower_bound(b,b+sz,a[i])-b;
    }
}




//主席树 by:sjf
struct president_tree
{
	#define type int
	int root[MAX],ls[40*MAX],rs[40*MAX],tot,ql,qr;
	type sum[40*MAX],qv;
	void init()
	{
		mem(root,0);
		tot=1;
		ls[0]=rs[0]=sum[0]=0;
	}
	int newnode(int x)
	{
		ls[tot]=ls[x];
		rs[tot]=rs[x];
		sum[tot]=sum[x];
		return tot++;
	}
	void insert(int l,int r,int &id,int pre) //set(ql,ql,v)
	{
		id=newnode(pre);
		sum[id]+=qv;
		if(l==r) return;
		int mid=(l+r)>>1;
		if(ql<=mid) insert(l,mid,ls[id],ls[pre]);
		else insert(mid+1,r,rs[id],rs[pre]);
	}
	int kindcnt(int l,int r,int id) //set(ql,qr)
	{
		if(ql<=l&&r<=qr) return sum[id]; 
		int mid=(l+r)>>1;
		int res=0;
		if(ql<=mid) res+=kindcnt(l,mid,ls[id]);
		if(qr>=mid+1) res+=kindcnt(mid+1,r,rs[id]);
		return res;
	}
	int kthsmall(int l,int r,int id,int pre,int k)
	{
		if(l==r) return l;
		int mid=(l+r)>>1;
		int temp=sum[ls[id]]-sum[ls[pre]];
		if(temp>=k) return kthsmall(l,mid,ls[id],ls[pre],k);
		else return kthsmall(mid+1,r,rs[id],rs[pre],k-temp);
	}
	int kthbig(int l,int r,int id,int pre,int k)
	{
		if(l==r) return l;
		int mid=(l+r)>>1;
		int temp=sum[rs[id]]-sum[rs[pre]];
		if(temp>=k) return kthbig(mid+1,r,rs[id],rs[pre],k);
		else return kthbig(l,mid,ls[id],ls[pre],k-temp);
	}
	void set(int l,int r,type v=0){ql=l;qr=r;qv=v;}
}pt;







//笛卡尔树          by:sjf
/*
O(n)构造笛卡尔树 返回根 
性质:
1.树中的元素满足二叉搜索树性质，要求按照中序遍历得到的序列为原数组序列
2.树中节点满足堆性质，节点的key值要大于其左右子节点的key值
*/
namespace Cartesian_Tree
{
	int l[MAX],r[MAX],vis[MAX],stk[MAX];
	int build(int *a,int n)
	{
		int i,top=0;
		for(i=1;i<=n;i++) l[i]=0,r[i]=0,vis[i]=0;
		for(i=1;i<=n;i++)
		{
			int k=top;
			while(k>0&&a[stk[k-1]]>a[i]) k--;
			if(k) r[stk[k-1]]=i;
			if(k<top) l[i]=stk[k];
			stk[k++]=i;
			top=k;
		}
		for(i=1;i<=n;i++) vis[l[i]]=vis[r[i]]=1;
		for(i=1;i<=n;i++)
		{
			if(!vis[i]) return i;
		}
	}
}







//KD-Tree
namespace kd_tree
{
	const double alpha=0.75;
	const int dim=2;
	#define type int
	const type NONE=INF;   //初始值 
	struct kdtnode
	{
		bool exist;
		int l,r,sz,fa,dep,x[dim],mx[dim],mn[dim];
		type v,tag;
		kdtnode(){}
		void initval()
		{
			sz=exist;tag=v;
			if(exist) for(int i=0;i<dim;i++) mn[i]=mx[i]=x[i];
		}
		void null()
		{
			exist=sz=0;
			v=tag=NONE;
			for(int i=0;i<dim;i++)
			{
				mx[i]=-INF;
				mn[i]=INF;
			}
		}
		void newnode(int x0,int x1,type val=NONE)
		{
			x[0]=x0;
			x[1]=x1;
			l=r=fa=0;
			exist=1;
			v=val;
			initval();
		}
		kdtnode(int a,int b,type d=NONE){newnode(a,b,d);}
	};
	struct KDT
	{
		#define ls t[id].l
		#define rs t[id].r
		kdtnode t[MAX];
		int tot,idx,root;
		inline void pushup(int id)
		{
			t[id].initval();
			t[id].sz+=t[ls].sz+t[rs].sz;
			t[id].tag=min({t[ls].tag,t[rs].tag,t[id].tag});
			for(int i=0;i<dim;i++)
			{
				if(ls)
				{
					t[id].mx[i]=max(t[id].mx[i],t[ls].mx[i]);
					t[id].mn[i]=min(t[id].mn[i],t[ls].mn[i]);
				}
				if(rs)
				{
					t[id].mx[i]=max(t[id].mx[i],t[rs].mx[i]);
					t[id].mn[i]=min(t[id].mn[i],t[rs].mn[i]);
				}
			}
		}
		bool isbad(int id){return t[id].sz*alpha+3<max(t[ls].sz,t[rs].sz);}
		int st[MAX],top;
		void build(int &id,int l,int r,int fa,int dep=0)
		{
			id=0;if(l>r) return;
			int m=(l+r)>>1; idx=dep;
			nth_element(st+l,st+m,st+r+1,[&](int x,int y){return t[x].x[idx]<t[y].x[idx];});
			id=st[m];
			build(ls,l,m-1,id,(dep+1)%dim);
			build(rs,m+1,r,id,(dep+1)%dim);
			pushup(id);
			t[id].dep=dep;
			t[id].fa=fa;
		}
		inline void init(int n=0)
		{
			root=0;
			t[0].null();
			for(int i=1;i<=n;i++) st[i]=i;
			if(n) build(root,1,n,0);
			tot=n;
		}
		void travel(int id)
		{
			if(!id) return;
			if(t[id].exist) st[++top]=id;
			travel(ls);
			travel(rs);
		}
		void rebuild(int &id,int dep)
		{
			top=0;travel(id);
			build(id,1,top,t[id].fa,dep);
		}
		void insert(int &id,int now,int fa,int dep=0)
		{
			if(!id)
			{
				id=now;
				t[id].dep=dep;
				t[id].fa=fa;
				return;
			}
			if(t[now].x[dep]<t[id].x[dep]) insert(ls,now,id,(dep+1)%dim);
			else insert(rs,now,id,(dep+1)%dim);
			pushup(id);
			if(isbad(id)) rebuild(id,t[id].dep);
			t[id].dep=dep;
			t[id].fa=fa;
		}
		inline void insert(kdtnode x){t[++tot]=x;insert(root,tot,0,0);}
		inline void del(int id)
		{
			if(!id) return;
			t[id].null();
			int x=id;
			while(x)
			{
				pushup(x);
				x=t[x].fa;
			}
			if(isbad(id))
			{
				x=t[id].fa;
				rebuild(root==id?root:(t[x].l==id?t[x].l:t[x].r),t[id].dep);
			}
		}
		kdtnode q;
		ll dist(ll x,ll y){return x*x+y*y;}
		ll getdist(int id)//点q离区域t[id]最短距离 
		{
			if(!id) return LLINF;
			ll res=0;
			if(q.x[0]<t[id].mn[0]) res+=dist(q.x[0]-t[id].mn[0],0);
			if(q.x[1]<t[id].mn[1]) res+=dist(q.x[1]-t[id].mn[1],0);
			if(q.x[0]>t[id].mx[0]) res+=dist(q.x[0]-t[id].mx[0],0);
			if(q.x[1]>t[id].mx[1]) res+=dist(q.x[1]-t[id].mx[1],0);
			return res;
		}
		kdtnode a,b;
		inline int check(kdtnode &x)//x在矩形(a,b)内 
		{
			int ok=1;
			for(int i=0;i<dim;i++)
			{
				ok&=(x.x[i]>=a.x[i]);
				ok&=(x.x[i]<=b.x[i]);
			}
			return ok;
		}
		inline int allin(kdtnode &x)//x的子树全在矩形(a,b)内 
		{
			int ok=1;
			for(int i=0;i<dim;i++)
			{
				ok&=(x.mn[i]>=a.x[i]);
				ok&=(x.mx[i]<=b.x[i]);
			}
			return ok;
		}
		inline int allout(kdtnode &x)//x的子树全不在矩形(a,b)内 
		{
			int ok=0;
			for(int i=0;i<dim;i++)
			{
				ok|=(x.mx[i]<a.x[i]);
				ok|=(x.mn[i]>b.x[i]);
			}
			return ok;
		}
		type res;
		void query(int id)
		{
			if(!id) return;
			if(allout(t[id])||t[id].sz==0) return;
			if(allin(t[id]))
			{
				res=min(res,t[id].tag);
				return;
			}
			if(check(t[id])&&t[id].exist) res=min(res,t[id].v);
			int l,r;
			l=ls;
			r=rs;
			if(t[l].tag>t[r].tag) swap(l,r);
			if(t[l].tag<res) query(l);
			if(t[r].tag<res) query(r);
		}
		inline type query(kdtnode _a,kdtnode _b)
		{
			a=_a;b=_b;
			res=INF;
			query(root);
			return res;
		}
	}kd;
	#undef type
	#undef ls
	#undef rs
}
using namespace kd_tree;







//LCA 最近公共祖先
struct LCA
{
	#define type int
	struct node{int to;type w;node(){}node(int _to,type _w):to(_to),w(_w){}};
	type dis[MAX];
	int path[2*MAX],deep[2*MAX],first[MAX],len[MAX],tot,n;
	int dp[2*MAX][22];
	vector<node> mp[MAX];
	void dfs(int x,int pre,int h)
	{
		int i;
		path[++tot]=x;
		first[x]=tot;
		deep[tot]=h;
		for(i=0;i<mp[x].size();i++)
		{
			int to=mp[x][i].to;
			if(to==pre) continue;
			dis[to]=dis[x]+mp[x][i].w;
			len[to]=len[x]+1;
			dfs(to,x,h+1);
			path[++tot]=x;
			deep[tot]=h;
		}
	}
	void ST(int n)
	{
		int i,j,x,y;
		for(i=1;i<=n;i++) dp[i][0]=i;
		for(j=1;(1<<j)<=n;j++)
		{
			for(i=1;i+(1<<j)-1<=n;i++)
			{
				x=dp[i][j-1];
				y=dp[i+(1<<(j-1))][j-1];
				dp[i][j]=deep[x]<deep[y]?x:y;
			}
		}
	}
	int query(int l,int r)
	{
		int len,x,y;
		len=(int)log2(r-l+1); 
		x=dp[l][len];
		y=dp[r-(1<<len)+1][len];
		return deep[x]<deep[y]?x:y;
	}
	int lca(int x,int y)
	{
		int l,r,pos;
		l=first[x];
		r=first[y];
		if(l>r) swap(l,r);
		pos=query(l,r);
		return path[pos];
	} 
	type get_dis(int a,int b){return dis[a]+dis[b]-2*dis[lca(a,b)];}
	int get_len(int a,int b){return len[a]+len[b]-2*len[lca(a,b)];}
	void init(int _n)                //初始化
	{
		n=_n;
		for(int i=0;i<=n;i++)
		{
			dis[i]=0;
			len[i]=0;
			mp[i].clear();
		}
	}
	void add_edge(int a,int b,type w=1)            //加入无向边
	{
		mp[a].pb(node(b,w));
		mp[b].pb(node(a,w));
	}
	void work(int rt)
	{
		tot=0;
		dfs(rt,0,0);
		ST(2*n-1);
	}
	int lca_root(int rt,int a,int b)
	{
		int fa,fb;
		fa=lca(a,rt);
		fb=lca(b,rt);
		if(fa==fb) return lca(a,b);
		else
		{
			if(get_dis(fa,rt)<get_dis(fb,rt)) return fa;
			else return fb;
		}
	}
	#undef type
}lca;
/*
lca.init(n);
lca.add_edge(a,b,w) undirected edge.
lca.work(rt);
*/













/*
	图论板子
*/


//dijkstra算法求最短路 堆优化 O(nlogn)   性能较高但各大oj都ce??
const int INF=0x3f3f3f3f;
int n,m;
struct qnode
{
    int v,c;
    qnode(int v,int c):v(v),c(c){}
    friend bool operator<(qnode q1,qnode q2)
    {
        return q1.c>q2.c;          //定义优先级相反，此为小值更优先
    }
};
struct edge
{
    int v,w;
    edge(int v,int w):v(v),w(w){}
};
vector<edge> E[maxn];
int vis[maxn];
int dist[maxn];
inline void add(int u,int v,int w)
{
    E[u].emplace_back(v,w);
    E[v].emplace_back(u,w);        //emplace不适用于poj -_-!
}
void dijkstra(int x)       //起点
{
    //memset(vis,0,sizeof(vis));              //这个地方注意如果test Case大于2e5的话用memset可能会被卡常
   // memset(dist,INF,sizeof(dist));          //必要时用for(int i=1;i<=n;i++)初始化  memset慎用
    for(int i=1;i<=n;i++)
        vis[i]=0,dist[i]=INF;
    priority_queue<qnode> q;
    dist[x]=0;
    q.emplace(x,0);
    while(!q.empty())
    {
        qnode temp=q.top();
        q.pop();
        int point=temp.v;       //选取松弛点
        if(vis[point])
            continue;
        vis[point]=1;
        for(edge i:E[point])
        {
            int v=i.v,w=i.w;
            if(!vis[v]&&dist[v]>dist[point]+w)   //松弛操作
            {
                dist[v]=dist[point]+w;
                q.emplace(v,dist[v]);
            }
        }
    }
}

/*
ps: poj上不让用emplace 万能头文件 和包括push({u,v}) foreach等一切c++11的操作
*/



//dijkstra算法求最短路 堆优化 O(nlogn) 有点改变但影响不大 可以在各大oj上编译
int n,m;
struct Dijkstra
{
    struct qnode
    {
        int v,c;
        qnode(int v,int c):v(v),c(c) {}
        friend bool operator<(qnode q1,qnode q2)
        {
            return q1.c>q2.c;          //定义优先级相反，此为小值更优先
        }
    };
    struct edge
    {
        int v,w;
        edge(int v,int w):v(v),w(w) {}
    };
    vector<edge> E[maxn];
    int vis[maxn];
    int dist[maxn];
    inline void add(int u,int v,int w)
    {
        E[u].push_back(edge(v,w));
        E[v].push_back(edge(u,w));
    }
    void solve(int x)       //起点
    {
        //memset(vis,0,sizeof(vis));              //这个地方注意如果test Case大于2e5的话用memset可能会被卡常
        //memset(dist,INF,sizeof(dist));          //必要时用for(int i=1;i<=n;i++)初始化  memset慎用
        for(int i=1; i<=n; i++)
            vis[i]=0,dist[i]=INF;
        priority_queue<qnode> q;
        dist[x]=0;
        q.push(qnode(x,0));
        while(!q.empty())
        {
            qnode temp=q.top();
            q.pop();
            int point=temp.v;       //选取松弛点
            if(vis[point])
                continue;
            vis[point]=1;
            int len=E[point].size();
            for(int i=0; i<len; i++)
            {
                int v=E[point][i].v,w=E[point][i].w;
                if(!vis[v]&&dist[v]>dist[point]+w)   //松弛操作
                {
                    dist[v]=dist[point]+w;
                    q.push(qnode(v,dist[v]));
                }
            }
        }
    }
    int query(int t)
    {
        return dist[t];
    }
}ans;





#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<queue>
#include<stack>
#include<map>
#include<sstream>
using namespace std;
typedef long long ll;
const int maxn = 1e3 + 10;
const int INF = 1 << 30;
int T, n, m;
struct edge
{
    int from, to, dist;
    edge(int u, int v, int d):from(u), to(v), dist(d){}
    edge(){}
};
struct Heapnode
{
    int d, u;//d为距离，u为起点
    Heapnode(){}
    Heapnode(int d, int u):d(d), u(u){}
    bool operator <(const Heapnode & a)const
    {
        return d > a.d;//这样优先队列先取出d小的
    }
};
struct Dijkstra
{
    int n, m;
    vector<edge>edges;//存边的信息
    vector<int>G[maxn];//G[i]表示起点为i的边的序号集
    bool v[maxn];//标记点是否加入集合
    int d[maxn];//起点s到各个点的最短路
    int p[maxn];//倒叙记录路径
    Dijkstra(){}
    void init(int n)
    {
        this -> n = n;
        for(int i = 0; i < n; i++)G[i].clear();
        edges.clear();
    }
    void addedge(int from, int to, int dist)
    {
        edges.push_back(edge(from, to, dist));
        m = edges.size();
        G[from].push_back(m - 1);//存以from为起点的下一条边
    }
    void dijkstra(int s)//以s为起点
    {
        priority_queue<Heapnode>q;
        for(int i = 0; i < n; i++)d[i] = INF;
        d[s] = 0;
        memset(v, 0, sizeof(v));
        memset(p, -1, sizeof(p));
        q.push(Heapnode(0, s));
        while(!q.empty())
        {
            Heapnode now = q.top();
            q.pop();
            int u = now.u;//当前起点
            if(v[u])continue;//如果已经加入集合，continue
            v[u] = 1;
            for(int i = 0; i < G[u].size(); i++)
            {
                edge& e = edges[G[u][i]];//引用节省代码
                if(d[e.to] > d[u] + e.dist)
                {
                    d[e.to] = d[u] + e.dist;
                    p[e.to] = G[u][i];//记录e.to前的边的编号，p存的是边的下标,这样可以通过边找出之前的点以及每条路的路径，如果用邻接矩阵存储的话这里可以直接存节点u
                    q.push(Heapnode(d[e.to], e.to));
                }
            }
        }
    }
    void output(int u)
    {
        for(int i = 0; i < n; i++)
        {
            if(i == u)continue;
            printf("从%d到%d距离是：%2d   ", u, i, d[i]);
            stack<int>q;//存的是边的编号
            int x = i;//x就是路径上所有的点
            while(p[x] != -1)
            {
                q.push(x);
                x = edges[p[x]].from;//x变成这条边的起点
            }
            cout<<u;
            while(!q.empty())
            {
                cout<<"->"<<q.top();
                q.pop();
            }
            cout<<endl;
        }
    }
};
Dijkstra ans;
int main()
{
    cin >> n >> m;
    ans.init(n);
    for(int i = 0; i < m; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        ans.addedge(u, v, w);
    }
    int u = 0;
    ans.dijkstra(u);
    ans.output(u);
}




//拓扑排序
vector<int> g[maxn];
int du[maxn], n, m=0, l[maxn];
bool toposort()
{
    memset(du, 0, sizeof du);
    for(int i = 1; i <= n; ++i)
        for(int j = 0; j < g[i].size(); ++j)
            ++du[g[i][j]];
    int tot = 0;
    priority_queue<int, vector<int>, greater<int> > q;//按字典序最小的排序时
    //queue<int> q;
    for(int i = 1; i <= n; ++i)
        if(!du[i])
            q.push(i);
    while(!q.empty())
    {
        int x = q.top(); q.pop();
        l[tot++] = x;
        for(int j = 0; j < g[x].size(); ++j)
        {
            int t = g[x][j];
            --du[t];
            if(!du[t])q.push(t);
        }
    }
    if(tot == n)return 1;
    else        return 0;
}















//组合数前缀和(1e9版本)
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace std;
const int N = 700005;
typedef long long ll;
const ll mod = 998244353ll;
ll fac[N], rfac[N];
ll ksm(ll x, ll k)
{
    ll s = 1;
    while (k)
    {
        if (k & 1)
            s = s * x % mod;
        x = x * x % mod;
        k >>= 1;
    }
    return s;
}
ll G = 3;
struct NTT
{
    ll ft[N];
    int rev[N];
    void init(int n)
    {
        int k;
        for (k = 0; (1 << k) < n; k++)
            ;
        for (int i = 0; i < n; i++) rev[i] = rev[i >> 1] >> 1 | ((i & 1) << (k - 1));
    }
    void trans(ll *a, int n, int ty)
    {
        for (int i = 0; i < n; i++)
            if (i < rev[i])
                swap(a[i], a[rev[i]]);
        ft[0] = 1;
        for (int m = 1; m < n; m <<= 1)
        {
            ll t0 = ksm(G, mod - 1 + ty * (mod - 1) / (m << 1));
            for (int i = 1; i < m; i++) ft[i] = ft[i - 1] * t0 % mod;
            for (int k = 0; k < n; k += (m << 1))
                for (int i = k; i < k + m; i++)
                {
                    ll t0 = a[i], t1 = a[i + m] * ft[i - k] % mod;
                    a[i] = (t0 + t1) % mod;
                    a[i + m] = (t0 - t1 + mod) % mod;
                }
        }
    }
    void dft(ll *a, int n)
    {
        trans(a, n, 1);
    }
    void idft(ll *a, int n)
    {
        trans(a, n, -1);
        ll t0 = ksm(n, mod - 2);
        for (int i = 0; i < n; i++) a[i] = a[i] * t0 % mod;
    }
} ntt;
ll A[N], B[N], C[N], ff[N];
ll inv(ll x)
{
    return ksm(x, mod - 2);
}
void calc(ll *st, ll *ed, int d, int k)
{
    ff[0] = 1;
    for (int i = 0; i <= d; i++) ff[0] = ff[0] * (k - i) % mod;
    for (int i = 1; i <= d; i++) ff[i] = ff[i - 1] * (i + k) % mod * inv(i + k - d - 1) % mod;

    int len;
    for (len = 1; len <= 3 * d; len <<= 1)
        ;
    ntt.init(len);
    for (int i = 0; i <= d; i++)
    {
        A[i] = st[i] * rfac[i] % mod * rfac[d - i] % mod;
        if ((d - i) & 1)
            A[i] = mod - A[i];
    }
    for (int i = 0; i <= 2 * d; i++) B[i] = inv(i - d + k);
    ntt.dft(A, len);
    ntt.dft(B, len);
    for (int i = 0; i < len; i++) C[i] = A[i] * B[i] % mod;
    ntt.idft(C, len);
    for (int i = 0; i <= d; i++)
    {
        ed[i] = C[i + d] * ff[i] % mod;
    }
    for (int i = 0; i < len; i++) A[i] = B[i] = C[i] = 0;
}
ll qz[N], hz[N], V, rV;
ll ag[N], revg[N], dg[N];
ll af[N], df[N];
int n;
void work(int r, vector<ll> &g, vector<ll> &f)
{
    if (!r)
    {
        g.push_back(1);
        f.push_back(0);
        return;
    }
    if (r & 1)
    {
        work(r - 1, g, f);
        for (int i = 0; i < r; i++) ag[i] = g[i];
        calc(ag, revg, r - 1, (-n - 1) * rV % mod);
        for (int i = 0; i < r; i++) g[i] = g[i] * (i * V % mod + r) % mod;
        ll p = 1;
        for (int i = 1; i <= r; i++) p = p * (r * V + i) % mod;
        g.push_back(p);
        for (int i = 0; i < r; i++)
        {
            if ((r - 1) & 1)
                f[i] = (f[i] - revg[i]) % mod;
            else
                f[i] = (f[i] + revg[i]) % mod;
            f[i] = f[i] * (i * V + r) % mod;
        }
        hz[r + 1] = 1;
        for (int i = r; i >= 0; i--) hz[i] = hz[i + 1] * (i + r * V) % mod;
        qz[-1] = 1;
        for (int i = 0; i <= r; i++) qz[i] = qz[i - 1] * (n - i - r * V) % mod;
        ll sum = 0;
        for (int i = 0; i < r; i++) sum = (sum + hz[i + 1] * qz[i - 1] % mod) % mod;
        f.push_back(sum);
        return;
    }

    int d = r >> 1;
    work(d, g, f);
    for (int i = 0; i <= d; i++) ag[i] = g[i];
    calc(ag, ag + d + 1, d, d + 1);
    calc(ag, dg, 2 * d, d * rV % mod);
    calc(ag, revg, 2 * d, (-n - 1) * rV % mod);
    for (int i = 0; i <= d; i++) af[i] = f[i];
    calc(af, af + d + 1, d, d + 1);
    calc(af, df, 2 * d, d * rV % mod);
    f.resize(r + 1);
    for (int i = 0; i <= r; i++)
    {
        ll s = df[i] * revg[i] % mod;
        if (d & 1)
            s = mod - s;
        f[i] = (s + af[i] * dg[i] % mod) % mod;
    }
    g.resize(r + 1);
    for (int i = 0; i <= r; i++) g[i] = ag[i] * dg[i] % mod;
}
void sol()
{
    int m;
    scanf("%d%d", &n, &m);
    V = sqrt(n);
    rV = inv(V);
    vector<ll> g, f;
    work(V, g, f);
    ll fac = 1;
    for (int i = 0; i < V; i++) fac = fac * g[i] % mod;
    for (int i = V * V + 1; i <= n; i++) fac = fac * i % mod;
    for (int i = 0; i <= V; i++) ag[i] = g[i];
    calc(ag, revg, V, (-n - 1) * rV % mod);
    qz[-1] = 1;
    for (int i = 0; i <= V; i++) qz[i] = qz[i - 1] * revg[i] % mod;
    hz[V] = 1;
    for (int i = V * V + 1; i <= n; i++) hz[V] = hz[V] * i % mod;
    for (int i = V - 1; i >= 0; i--) hz[i] = hz[i + 1] * g[i] % mod;
    ll sum = 0;
    int cr = 0;
    for (int i = 0; i < V && (i + 1) * V - 1 <= m; i++)
    {
        ll t = hz[i + 1] * qz[i - 1] % mod * f[i] % mod;
        if ((i * V) & 1)
            t = mod - t;
        sum = (sum + t) % mod;
        cr = i + 1;
    }
    sum = sum * inv(fac) % mod;
    ll C = fac, G = 1;
    for (int j = 0; j < cr; j++) G = G * g[j] % mod;
    int s = n - cr * V, cp = 0;
    for (int i = 0; i < V && (i + 1) * V <= s; i++) G = G * g[i] % mod, cp = i + 1;
    for (int i = cp * V + 1; i <= s; i++) G = G * i % mod;
    C = C * inv(G) % mod;
    for (int i = cr * V; i <= m; i++)
    {
        if (i != cr * V)
        {
            C = C * inv(i) % mod * (n - i + 1) % mod;
        }
        sum = (sum + C) % mod;
    }
    sum = (sum % mod + mod) % mod;
    printf("%lld\n", sum);
}
int main()
{
    fac[0] = 1;
    for (int i = 1; i < N; i++) fac[i] = fac[i - 1] * i % mod;
    rfac[N - 1] = inv(fac[N - 1]);
    for (int i = N - 2; i >= 0; i--) rfac[i] = rfac[i + 1] * (i + 1) % mod;
    int T;
    scanf("%d", &T);
    while (T--) sol();

    return 0;
}












//.bat对拍      把data.exe,biaoda.exe,mycode.exe和.bat文件放在一个文件夹后执行.bat文件

:again
data > input.txt
biaoda < input.txt > biaoda_output.txt
mycode < input.txt > mycode_output.txt
fc biaoda_output.txt test_output.txt
if not errorlevel 1 goto again
pause

//.cpp对拍

#include<iostream>
#include<windows.h>
using namespace std;
int main()
{
    //int t=200;
    while(1)
    {
//      t--;
        system("data.exe > data.txt");
        system("biaoda.exe < data.txt > biaoda.txt");
         system("test.exe < data.txt > test.txt");
        if(system("fc test.txt biaoda.txt"))   break;
    }
    if(t==0) cout<<"no error"<<endl;
    else cout<<"error"<<endl;
    //system("pause");
    return 0;
}
//https://blog.csdn.net/code12hour/article/details/51252457






//快排注意：重载<时返回true是不交换，返回false是交换
/*
战术研究:
●读新题的优先级高于一切
●读完题之后必须看一 遍clarification，交题之前必须看一遍clarification
●可能有SPJ的题目提交前也应该尽量做到与样例输出完全一 致 WA时需要检查INF是否设小
●构造题不可开场做
●每道题需至少有两个人确认题意。上机之前做法需得到队友确认
●带有猜想性质的算法应放后面写
●当发现题目不会做但是过了一片时应冲一发暴力
●将待写的题按所需时间放入小根堆中，每次选堆顶的题目写
●交完题目后立马打印随后让出机器
●写题超过半小时应考虑是否弃题
●细节、公式等在上机前应在草稿纸上准备好，防止上机后越写越乱
●提交题目之前应检查solve(n, m)是否等于solve(m, n)
●检查是否所有东西都已经清空
●对于中后期题应该考虑一人写题，另一人在一旁辅助， 及时发现手误
●最后半小时不能慌张
●对于取模的题，在输出之前一定要再取模一次进行保险
●对于舍入输出，若abs不超过eps,需要强行设置为0来防止-0.0000的出现。
*/