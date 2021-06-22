//米勒 罗宾算法模版 ==============//
LL prime[6] = {2, 3, 5, 233, 331};           //这个地方不是随机的
LL qu(LL x, LL y, LL mod) {
    LL ret = 0;
    while(y) {
        if(y & 1)
            ret = (ret + x) % mod;
        x = x * 2 % mod;
        y >>= 1;
    }
    return ret;
}
LL qpow(LL a, LL n, LL mod) {
    LL ret = 1;
    while(n) {
        if(n & 1) ret = qu(ret, a, mod);
        a = qu(a, a, mod);
        n >>= 1;
    }
    return ret;
}
bool MR(LL p) {
    if(p < 2) return 0;
    if(p != 2 && p % 2 == 0) return 0;
    LL s = p - 1;
    while(! (s & 1)) s >>= 1;
    for(int i = 0; i < 5; ++i) {
        if(p == prime[i]) return 1;
        LL t = s, m = qpow(prime[i], s, p);
        while(t != p - 1 && m != 1 && m != p - 1) {
            m = qu(m, m, p);
            t <<= 1;
        }
        if(m != p - 1 && !(t & 1)) return 0;
    }
    return 1;
}

//这个板子有点快 2018宁夏B





//板子2 by:sjf

//MR素数测试
// Miller_Rabin + Pollard_rho

const int S=20;
mt19937 rd(time(0));
ll mul2(ll a,ll b,ll p)
{
	ll res=0;
	while(b)
	{
		if(b&1) res=(res+a)%p;
		a=(a+a)%p;
		b>>=1;
	}
	return res;
}
ll pow2(ll a,ll b,ll p)
{
	ll res=1;
	while(b)
	{
		if(b&1) res=mul2(res,a,p);
		a=mul2(a,a,p);
		b>>=1;
	}
	return res;
}
int check(ll a,ll n,ll x,ll t)//一定是合数返回1,不一定返回0
{
	ll now,nex,i;
	now=nex=pow2(a,x,n);
	for(i=1;i<=t;i++)
	{
		now=mul2(now,now,n);
		if(now==1&&nex!=1&&nex!=n-1) return 1;
		nex=now;
	}
	if(now!=1) return 1;
	return 0;
}
int Miller_Rabin(ll n)
{
	if(n<2) return 0;
	if(n==2) return 1;
	if((n&1)==0) return 0;
	ll x,t,i;
	x=n-1;
	t=0;
	while((x&1)==0) x>>=1,t++;
	for(i=0;i<S;i++)
	{
		if(check(rd()%(n-1)+1,n,x,t)) return 0;
	}
	return 1;
}
ll Pollard_rho(ll x,ll c)
{
	ll i,k,g,t,y;
	i=1;
	k=2;
	y=t=rd()%x;
	while(1)
	{
		i++;
		t=(mul2(t,t,x)+c)%x;
		g=__gcd(y-t+x,x);
		if(g!=1&&g!=x) return g;
		if(y==t) return x;
		if(i==k)
		{
			y=t;
			k+=k;
		}
	}
}
vector<ll> fac;
void findfac(ll n)
{
	if(Miller_Rabin(n))
	{
		fac.pb(n);
		return;
	}
	ll t=n;
	while(t>=n) t=Pollard_rho(t,rd()%(n-1)+1);
	findfac(t);
	findfac(n/t);
}
void work(ll x)
{
	fac.clear();
	findfac(x);
}