//行列式求值O(n^3logn)
ll det(int n)
{
    ll ans=1;
    for(int i=1; i<=n; ++i)
    {
        for(int k=i+1; k<=n; ++k)
        {
            while(a[k][i])
            {
                ll d=a[i][i]/a[k][i];
                for(int j=i; j<=n; ++j) a[i][j]=(a[i][j]-1LL*d*a[k][j]);
                std::swap(a[i],a[k]),ans=-ans;
            }
        }
        ans*=a[i][i];
    }
    return ans;
}



//取模行列式
ll det(int n)
{
    ll ans=1;
    for(int i=1; i<=n; ++i)
    {
        for(int k=i+1; k<=n; ++k)
        {
            while(a[k][i])
            {
                ll d=a[i][i]/a[k][i];
                for(int j=i; j<=n; ++j) a[i][j]=(a[i][j]-1LL*d*a[k][j])%mod;
                std::swap(a[i],a[k]),ans=-ans;
            }
        }
        ans=ans*a[i][i]%mod;
    }
    return ans;
}