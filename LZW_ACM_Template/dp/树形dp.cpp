//����dp   (HDU1561�ı�)
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
    //dp[i][j]����������iΪ���ڵ������ѡȡj����������ֵ
    //����ط���ö��x���ӽڵ�i ����ǰi��������ö�ٵ�i�������Ľ������k
    //��ʱȡ���ڶ���ǰi-1���ӽڵ�ȡj-k���ڵ��״̬
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