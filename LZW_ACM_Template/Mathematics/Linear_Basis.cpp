//线性基
ll p[maxn];
ll a[maxn];
void get_linear_ji(ll n)            //线性基O(62*62*n)
{
    for(ll i=0;i<n;i++)
    {
        for(ll j=62;j>=0;j--)
        {
            if((a[i]>>j))
            {
               if(p[j])
               {
                   a[i]^=p[j];
               }
               else
               {
                   p[j]=a[i];
                   break;
               }
            }
        }
    }
    /* int r=0;
    for (int j=0;j<=62;j++) if (P[j]) r++;             //返回秩 线性基的个数
    return r; */
}





//网友写的很好的线性基模板
struct Linear_Basis
{
    LL b[63],nb[63],tot;

    Linear_Basis()
    {
        tot=0;
        memset(b,0,sizeof(b));
        memset(nb,0,sizeof(nb));
    }
    void clear()
    {
        tot=0;
        memset(b,0,sizeof(b));
        memset(nb,0,sizeof(nb));
    }
    bool ins(LL x)                  //插入一个数
    {
        for(int i=62;i>=0;i--)
            if (x&(1LL<<i))
            {
                if (!b[i]) {b[i]=x;break;}
                x^=b[i];
            }
        return x>0;
    }

    LL Max(LL x)                  //求异或最大值
    {
        LL res=x;
        for(int i=62;i>=0;i--)
            res=max(res,res^b[i]);
        return res;
    }

    LL Min(LL x)                 //求异或最小值
    {
        LL res=x;
        for(int i=0;i<=62;i++)
            if (b[i]) res^=b[i];
        return res;
    }

    void rebuild()                //重构化为对角矩阵 为求Kth_MAX做铺垫
    {
        for(int i=62;i>=0;i--)
            for(int j=i-1;j>=0;j--)
                if (b[i]&(1LL<<j)) b[i]^=b[j];
        for(int i=0;i<=62;i++)
            if (b[i]) nb[tot++]=b[i];
    }

    LL Kth_Min(LL k)             //第K小   在调用之前必须rebuild
    {
    	rebuild();
        LL res=0;
        for(int i=62;i>=0;i--)
            if (k&(1LL<<i)) res^=nb[i];
        return res;
    }

} LB;

Linear_Basis merge(const Linear_Basis &n1,const Linear_Basis &n2)   //暴力合并两个线性基
{
    Linear_Basis ret=n1;
    for (int i=62; i>=0; i--)
        if (n2.b[i])
            ret.ins(n2.b[i]);
    return ret;
}