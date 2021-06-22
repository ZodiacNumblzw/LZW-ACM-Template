//单点求值 Eular函数
ll Euler(ll x)
{
    ll ans = x, m = (ll)sqrt(x*1.0)+1;
    for(int i = 2; i < m; ++i)
    {
        if(x%i == 0)
        {
            ans = ans / i * (i-1);
            while(x%i == 0) x /= i;
        }
    }
    if(x > 1) ans = ans / x * (x-1);
    return ans;
}