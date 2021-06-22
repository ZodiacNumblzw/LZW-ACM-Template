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