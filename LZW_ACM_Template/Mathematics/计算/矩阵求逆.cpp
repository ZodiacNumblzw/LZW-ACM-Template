//矩阵求逆 高性能
#include<bits/stdc++.h>
#define int64 long long
using namespace std;
const int64 mod=1e9+7;
int64 a[410][410];
int n,is[410],js[410];
void exgcd(int a,int b,int &x,int &y)
{
    if(!b)return x=1,y=0,void();
    exgcd(b,a%b,y,x);
    y-=x*(a/b);
}
int inv(int p)
{
    int x,y;
    exgcd(p,mod,x,y);
    return (x+mod)%mod;
}
void inv()
{
    for(int k=1; k<=n; k++)
    {
        for(int i=k; i<=n; i++) // 1 找到当前方阵（假设是第k个）的主元
            for(int j=k; j<=n; j++)if(a[i][j])
                {
                    is[k]=i,js[k]=j;
                    break;
                }
        for(int i=1; i<=n; i++) // 2 将主元所在行、列与第k行、第k列交换，并且记录下交换的行和列
            swap(a[k][i],a[is[k]][i]);
        for(int i=1; i<=n; i++)
            swap(a[i][k],a[i][js[k]]);
        if(!a[k][k])
        {
            puts("No Solution");
            exit(0);
        }
        a[k][k]=inv(a[k][k]); // 3 把第k行第k个元素变成它的逆元（同时判无解：逆元不存在）
        for(int j=1; j<=n; j++)if(j!=k) // 4 更改当前行（乘上刚才那个逆元）
                (a[k][j]*=a[k][k])%=mod;
        for(int i=1; i<=n; i++)if(i!=k) // 5 对矩阵中的其他行做和高斯消元几乎完全一样的操作，只有每行的第k列不一样
                for(int j=1; j<=n; j++)if(j!=k)
                        (a[i][j]+=mod-a[i][k]*a[k][j]%mod)%=mod;
        for(int i=1; i<=n; i++)if(i!=k) // 就是这里不同
                a[i][k]=(mod-a[i][k]*a[k][k]%mod)%mod;
    }
    for(int k=n; k; k--) // 6 最后把第②步中的交换逆序地交换回去
    {
        for(int i=1; i<=n; i++)
            swap(a[js[k]][i],a[k][i]);
        for(int i=1; i<=n; i++)
            swap(a[i][is[k]],a[i][k]);
    }
}
int main()
{
    scanf("%d",&n);
    for(int i=1; i<=n; i++)
        for(int j=1; j<=n; j++)
            scanf("%lld",a[i]+j);
    inv();
    for(int i=1; i<=n; i++)
        for(int j=1; j<=n; j++)
            printf("%lld%c",a[i][j],j==n?'\n':' ');
    return 0;
}









#include<bits/stdc++.h>
#define ll long long
using namespace std;
const ll mod = 1e9 + 7;
ll inv(ll a){
    if(a == 1) return 1;
    else return (mod - mod/a)*inv(mod%a)%mod;
}
ll invw;
ll p[550][550];
void inv_mt(vector< vector<ll> > &a, int n){//矩阵求逆,求得的逆矩阵存在a[0~n-1][n~2n-1]
    for(int i = 0; i < n; ++i){
        int p = i;
        while(p < n && a[p][i] == 0) p++;
        if(p < n && a[p][i] != 0 && p!=i){//需要换行
            for(int j = i; j < (n<<1); ++j) swap(a[i][j], a[p][j]);
        }
        assert(a[i][i] != 0);//检测换完是否不为0

        ll t = inv(a[i][i]);
        for(int j = i; j < n + n; ++j) a[i][j] = a[i][j]*t%mod;//整行乘a[i][i]的逆元，使a[i][i] = 1
        for(int k = 0; k < n; ++k){//消掉其他行的第i列
            if(k == i) continue;
            t = a[k][i];
            for(int j = 0; j < (n<<1); ++j) a[k][j] = (a[k][j] - t*a[i][j])%mod;
        }
    }
}
int n, m;
void init()
{
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n+m; ++j){
            scanf("%lld", &p[i][j]);
            p[i][j] = p[i][j] * invw % mod;
        }
    }
}
ll ans[550];
void sol()
{

    vector< vector<ll> > a( n, vector<ll>(2*n) );
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            a[i][j] = ((i==j)-p[i][j]+mod)%mod;
            a[i][n+j] = 0;
        }
        a[i][n+i] = 1;
    }
    inv_mt(a, n);
    for(int j = 0; j < m; ++j){
        ans[j] = 0;
        for(int i = 0; i < n; ++i){
            ans[j] = (ans[j] + a[0][i+n]*p[i][n+j]%mod)%mod;
        }
    }
    for(int i = 0; i < m; ++i){
        if(i > 0) printf(" ");
        printf("%lld", (ans[i]+mod)%mod);
    }printf("\n");
}
int main()
{
//    freopen("1.in", "r", stdin);
//    freopen("1.txt","w", stdout);
    invw = inv(10000);
    while(~scanf("%d%d", &n, &m)){
        init();sol();
    }
}
