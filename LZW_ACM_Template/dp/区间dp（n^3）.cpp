//����dp���� δ���ı����Ż�   O(n^3)          P1880 ʯ�Ӻϲ�
for(int len=1;len<=n-1;len++)             //ö�����䳤��
{
    for(int i=1;i<n;i++)           //ö��������˵�
    {
        int j=i+len;                      //�����Ҷ˵�
        dp1[i][j]=INF;
        for(int k=i;k<j;k++)              //ö�ٶϵ�
        {
            dp1[i][j]=min(dp1[i][j],dp1[i][k]+dp1[k+1][j]+a[j]-a[i-1]);           //�ϲ�[i,k],[k+1,j]����Ļ���
            dp2[i][j]=max(dp2[i][j],dp2[i][k]+dp2[k+1][j]+a[j]-a[i-1]);
        }
    }
}