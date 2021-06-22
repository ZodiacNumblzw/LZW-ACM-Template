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