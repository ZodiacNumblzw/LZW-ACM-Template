//ÏßÐÔÉ¸Eularº¯Êý
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