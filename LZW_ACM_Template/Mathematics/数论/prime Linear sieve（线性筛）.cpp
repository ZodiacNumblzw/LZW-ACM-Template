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