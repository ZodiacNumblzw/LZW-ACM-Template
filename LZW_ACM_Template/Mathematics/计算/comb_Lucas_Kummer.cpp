//组合数取模
ll comb(ll a,ll b,ll m)                                    //组合数取模
{
    if(a<b) return 0;
    if(a==b) return 1;
    else b=min(b,a-b);
    ll ans=1,ca=1,cb=1;
    for(int i=0;i<b;i++)
    {
        ca=(ca*(a-i))%m;
        cb=(cb*(b-i))%m;
    }
    ans=(ca*inverse(cb,m))%m;
    return ans;
}


//预处理阶乘和阶乘逆元的O(1)组合数求法
int fac[maxn];
int ifac[maxn];
int comb(int a,int b)
{
    return 1LL*fac[a]*ifac[b]%mod*ifac[a-b]%mod;
}

void init(int n)
{
    fac[0]=1;
    for(int i=1;i<=n;i++)
        fac[i]=1LL*fac[i-1]*i%mod;
    ifac[n]=quick(fac[n],mod-2,mod)%mod;
    for(int i=n-1;i>=0;i--)
        ifac[i]=1LL*ifac[i+1]*(i+1)%mod;
}




//Lucas定理
ll lucas(ll a,ll b,ll p)                   //Lucas定理
{
	if(a<b)
		return 0;
    ll ans=1;
    while(a&&b)
    {
        ans=(ans*(comp(a%p,b%p,p)))%p;
        a/=p; b/=p;
    }
    return ans;
}

//Kummer定理
//设m，n为正整数，p为素数，则C(m+n,m)含p的幂次等于m+n在p进制下的进位次数。
//设m，n为正整数，p为素数，则C(n,m)含p的幂次等于n-m在p进制下的借位次数。
//https://www.luogu.org/blog/i207M/kummer-ding-li-shuo-lun-xue-xi-bi-ji

