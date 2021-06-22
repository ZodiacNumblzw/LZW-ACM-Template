//组合数前缀和(1e9版本)
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
using namespace std;
const int N = 700005;
typedef long long ll;
const ll mod = 998244353ll;
ll fac[N], rfac[N];
ll ksm(ll x, ll k)
{
    ll s = 1;
    while (k)
    {
        if (k & 1)
            s = s * x % mod;
        x = x * x % mod;
        k >>= 1;
    }
    return s;
}
ll G = 3;
struct NTT
{
    ll ft[N];
    int rev[N];
    void init(int n)
    {
        int k;
        for (k = 0; (1 << k) < n; k++)
            ;
        for (int i = 0; i < n; i++) rev[i] = rev[i >> 1] >> 1 | ((i & 1) << (k - 1));
    }
    void trans(ll *a, int n, int ty)
    {
        for (int i = 0; i < n; i++)
            if (i < rev[i])
                swap(a[i], a[rev[i]]);
        ft[0] = 1;
        for (int m = 1; m < n; m <<= 1)
        {
            ll t0 = ksm(G, mod - 1 + ty * (mod - 1) / (m << 1));
            for (int i = 1; i < m; i++) ft[i] = ft[i - 1] * t0 % mod;
            for (int k = 0; k < n; k += (m << 1))
                for (int i = k; i < k + m; i++)
                {
                    ll t0 = a[i], t1 = a[i + m] * ft[i - k] % mod;
                    a[i] = (t0 + t1) % mod;
                    a[i + m] = (t0 - t1 + mod) % mod;
                }
        }
    }
    void dft(ll *a, int n)
    {
        trans(a, n, 1);
    }
    void idft(ll *a, int n)
    {
        trans(a, n, -1);
        ll t0 = ksm(n, mod - 2);
        for (int i = 0; i < n; i++) a[i] = a[i] * t0 % mod;
    }
} ntt;
ll A[N], B[N], C[N], ff[N];
ll inv(ll x)
{
    return ksm(x, mod - 2);
}
void calc(ll *st, ll *ed, int d, int k)
{
    ff[0] = 1;
    for (int i = 0; i <= d; i++) ff[0] = ff[0] * (k - i) % mod;
    for (int i = 1; i <= d; i++) ff[i] = ff[i - 1] * (i + k) % mod * inv(i + k - d - 1) % mod;

    int len;
    for (len = 1; len <= 3 * d; len <<= 1)
        ;
    ntt.init(len);
    for (int i = 0; i <= d; i++)
    {
        A[i] = st[i] * rfac[i] % mod * rfac[d - i] % mod;
        if ((d - i) & 1)
            A[i] = mod - A[i];
    }
    for (int i = 0; i <= 2 * d; i++) B[i] = inv(i - d + k);
    ntt.dft(A, len);
    ntt.dft(B, len);
    for (int i = 0; i < len; i++) C[i] = A[i] * B[i] % mod;
    ntt.idft(C, len);
    for (int i = 0; i <= d; i++)
    {
        ed[i] = C[i + d] * ff[i] % mod;
    }
    for (int i = 0; i < len; i++) A[i] = B[i] = C[i] = 0;
}
ll qz[N], hz[N], V, rV;
ll ag[N], revg[N], dg[N];
ll af[N], df[N];
int n;
void work(int r, vector<ll> &g, vector<ll> &f)
{
    if (!r)
    {
        g.push_back(1);
        f.push_back(0);
        return;
    }
    if (r & 1)
    {
        work(r - 1, g, f);
        for (int i = 0; i < r; i++) ag[i] = g[i];
        calc(ag, revg, r - 1, (-n - 1) * rV % mod);
        for (int i = 0; i < r; i++) g[i] = g[i] * (i * V % mod + r) % mod;
        ll p = 1;
        for (int i = 1; i <= r; i++) p = p * (r * V + i) % mod;
        g.push_back(p);
        for (int i = 0; i < r; i++)
        {
            if ((r - 1) & 1)
                f[i] = (f[i] - revg[i]) % mod;
            else
                f[i] = (f[i] + revg[i]) % mod;
            f[i] = f[i] * (i * V + r) % mod;
        }
        hz[r + 1] = 1;
        for (int i = r; i >= 0; i--) hz[i] = hz[i + 1] * (i + r * V) % mod;
        qz[-1] = 1;
        for (int i = 0; i <= r; i++) qz[i] = qz[i - 1] * (n - i - r * V) % mod;
        ll sum = 0;
        for (int i = 0; i < r; i++) sum = (sum + hz[i + 1] * qz[i - 1] % mod) % mod;
        f.push_back(sum);
        return;
    }

    int d = r >> 1;
    work(d, g, f);
    for (int i = 0; i <= d; i++) ag[i] = g[i];
    calc(ag, ag + d + 1, d, d + 1);
    calc(ag, dg, 2 * d, d * rV % mod);
    calc(ag, revg, 2 * d, (-n - 1) * rV % mod);
    for (int i = 0; i <= d; i++) af[i] = f[i];
    calc(af, af + d + 1, d, d + 1);
    calc(af, df, 2 * d, d * rV % mod);
    f.resize(r + 1);
    for (int i = 0; i <= r; i++)
    {
        ll s = df[i] * revg[i] % mod;
        if (d & 1)
            s = mod - s;
        f[i] = (s + af[i] * dg[i] % mod) % mod;
    }
    g.resize(r + 1);
    for (int i = 0; i <= r; i++) g[i] = ag[i] * dg[i] % mod;
}
void sol()
{
    int m;
    scanf("%d%d", &n, &m);
    V = sqrt(n);
    rV = inv(V);
    vector<ll> g, f;
    work(V, g, f);
    ll fac = 1;
    for (int i = 0; i < V; i++) fac = fac * g[i] % mod;
    for (int i = V * V + 1; i <= n; i++) fac = fac * i % mod;
    for (int i = 0; i <= V; i++) ag[i] = g[i];
    calc(ag, revg, V, (-n - 1) * rV % mod);
    qz[-1] = 1;
    for (int i = 0; i <= V; i++) qz[i] = qz[i - 1] * revg[i] % mod;
    hz[V] = 1;
    for (int i = V * V + 1; i <= n; i++) hz[V] = hz[V] * i % mod;
    for (int i = V - 1; i >= 0; i--) hz[i] = hz[i + 1] * g[i] % mod;
    ll sum = 0;
    int cr = 0;
    for (int i = 0; i < V && (i + 1) * V - 1 <= m; i++)
    {
        ll t = hz[i + 1] * qz[i - 1] % mod * f[i] % mod;
        if ((i * V) & 1)
            t = mod - t;
        sum = (sum + t) % mod;
        cr = i + 1;
    }
    sum = sum * inv(fac) % mod;
    ll C = fac, G = 1;
    for (int j = 0; j < cr; j++) G = G * g[j] % mod;
    int s = n - cr * V, cp = 0;
    for (int i = 0; i < V && (i + 1) * V <= s; i++) G = G * g[i] % mod, cp = i + 1;
    for (int i = cp * V + 1; i <= s; i++) G = G * i % mod;
    C = C * inv(G) % mod;
    for (int i = cr * V; i <= m; i++)
    {
        if (i != cr * V)
        {
            C = C * inv(i) % mod * (n - i + 1) % mod;
        }
        sum = (sum + C) % mod;
    }
    sum = (sum % mod + mod) % mod;
    printf("%lld\n", sum);
}
int main()
{
    fac[0] = 1;
    for (int i = 1; i < N; i++) fac[i] = fac[i - 1] * i % mod;
    rfac[N - 1] = inv(fac[N - 1]);
    for (int i = N - 2; i >= 0; i--) rfac[i] = rfac[i + 1] * (i + 1) % mod;
    int T;
    scanf("%d", &T);
    while (T--) sol();

    return 0;
}