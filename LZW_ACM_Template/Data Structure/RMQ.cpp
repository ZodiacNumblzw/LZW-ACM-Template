//RMQ算法            求解区间最值，预处理O(nlogn) 查询O(1)
int a[maxn];
int maxsum[maxn][20],minsum[maxn][20];     //dp[i][j] 指a[i]-a[i+2^j-1]这长度为2^j串的最小（最大）值
void RMQ(int num) //预处理 O(nlogn)       一共num个数 且num<2^20（倍增）
{
    for(int i=1;i<=num;i++)
        maxsum[i][0]=minsum[i][0]=a[i];
    for(int j = 1; j < 20; j++)
    {
        for(int i = 1; i <= num; i++)
        {
            if(i + (1 << j) - 1 <= num)
            {
                maxsum[i][j] = max(maxsum[i][j - 1], maxsum[i + (1 << (j - 1))][j - 1]);
                minsum[i][j] = min(minsum[i][j - 1], minsum[i + (1 << (j - 1))][j - 1]);
            }
        }
    }
}
int maxl,minl;
void Query(int l,int r)
{
    int k=(int)((log(r-l+1))/log(2.0));
    maxl=max(maxsum[l][k],maxsum[r-(1<<k)+1][k]);
    minl=min(minsum[l][k],minsum[r-(1<<k)+1][k]);
}