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