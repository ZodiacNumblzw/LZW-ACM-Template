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