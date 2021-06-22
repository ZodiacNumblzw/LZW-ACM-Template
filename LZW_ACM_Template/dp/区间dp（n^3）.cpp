//区间dp板子 未用四边形优化   O(n^3)          P1880 石子合并
for(int len=1;len<=n-1;len++)             //枚举区间长度
{
    for(int i=1;i<n;i++)           //枚举区间左端点
    {
        int j=i+len;                      //区间右端点
        dp1[i][j]=INF;
        for(int k=i;k<j;k++)              //枚举断点
        {
            dp1[i][j]=min(dp1[i][j],dp1[i][k]+dp1[k+1][j]+a[j]-a[i-1]);           //合并[i,k],[k+1,j]所需的花费
            dp2[i][j]=max(dp2[i][j],dp2[i][k]+dp2[k+1][j]+a[j]-a[i-1]);
        }
    }
}