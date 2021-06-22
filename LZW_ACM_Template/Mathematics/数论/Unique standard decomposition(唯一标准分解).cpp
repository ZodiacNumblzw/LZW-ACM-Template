//num的一维存素因子，一维存对应素因子的次数次数
int num[2][maxn],cnt;
void fenjie_num(ll x)                //唯一标准分解
{
    cnt=0;
    for(int i=0; i<tol&&prime[i]<=x/prime[i]; i++)
    {
        if(x%prime[i]==0)
        {
            int temp=0;
            while(x%prime[i]==0)
            {
                x/=prime[i];
                temp++;
            }
            num[0][cnt]=prime[i];
            num[1][cnt++]=temp;
        }
    }
    if(x!=1)
    {
        num[0][cnt]=x;
        num[1][cnt++]=1;
    }
}
