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