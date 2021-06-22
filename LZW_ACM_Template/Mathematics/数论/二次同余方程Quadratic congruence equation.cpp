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