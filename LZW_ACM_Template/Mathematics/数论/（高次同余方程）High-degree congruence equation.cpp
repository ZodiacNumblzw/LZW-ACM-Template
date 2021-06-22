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