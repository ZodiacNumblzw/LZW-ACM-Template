//RMQ�㷨            ���������ֵ��Ԥ����O(nlogn) ��ѯO(1)
int a[maxn];
int maxsum[maxn][20],minsum[maxn][20];     //dp[i][j] ָa[i]-a[i+2^j-1]�ⳤ��Ϊ2^j������С�����ֵ
void RMQ(int num) //Ԥ���� O(nlogn)       һ��num���� ��num<2^20��������
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