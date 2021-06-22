//暴力求解原根
int G(int s)
{
    int q[1010]={0};
    for(int i=2;i<=s-2;i++) if ((s-1)%i==0) q[++q[0]]=i;
    for (int i=2;;i++)
    {
        bool B=1;
        for (int j=1;j<=q[0]&&B;j++) if (quick(i,q[j],s)==1) B=0;
        if (B) return i;
    }
    return -1;
}










//牛逼网友的NTT模板
#include<bits/stdc++.h>
#define fo(i, x, y) for(int i = x, B = y; i <= B; i ++)
#define ff(i, x, y) for(int i = x, B = y; i <  B; i ++)
#define fd(i, x, y) for(int i = x, B = y; i >= B; i --)
#define ll long long
#define pp printf
#define hh pp("\n")
using namespace std;
  
const int mo = 998244353;
  
ll ksm(ll x, ll y) {
    ll s = 1;
    for(; y; y /= 2, x = x * x % mo)
        if(y & 1) s = s * x % mo;
    return s;
}
  
typedef vector<ll> V;
#define pb push_back
#define si size()
#define re resize
  
namespace ntt {
    const int nm = 1 << 18;
    ll a[nm], b[nm], w[nm]; int r[nm];
    void build() {
        for(int i = 1; i < nm; i *= 2) ff(j, 0, i)
            w[i + j] = ksm(3, (mo - 1) / 2 / i * j);
    }
    void dft(ll *a, int n, int f) {
        ff(i, 0, n) {
            r[i] = r[i / 2] / 2 + (i & 1) * (n / 2);
            if(i < r[i]) swap(a[i], a[r[i]]);
        } ll b;
        for(int i = 1; i < n; i *= 2) for(int j = 0; j < n; j += 2 * i)
            ff(k, 0, i) b = a[i + j + k] * w[i + k], a[i + j + k] = (a[j + k] - b) % mo, a[j + k] = (a[j + k] + b) % mo;
        if(f == -1) {
            reverse(a + 1, a + n);
            b = ksm(n, mo - 2);
            ff(i, 0, n) a[i] = (a[i] + mo) * b % mo;
        }
    }
    void fft(V &p, V &q) {
        int p0 = p.si + q.si - 1;
        int n = 1; while(n <= p0) n *= 2;
        ff(i, 0, n) a[i] = b[i] = 0;
        ff(i, 0, p.si) a[i] = p[i];
        ff(i, 0, q.si) b[i] = q[i];
        dft(a, n, 1); dft(b, n, 1);
        ff(i, 0, n) a[i] = a[i] * b[i] % mo;
        dft(a, n, -1);
        p.resize(p0);
        ff(i, 0, p0) p[i] = a[i];
    }
}
  
V operator * (V a, V b) {
    ntt :: fft(a, b);
    return a;
}
  
const int N = 2e5 + 5;
  
int n;
struct P {
    int x, y;
} a[N];
  
int cmp(P a, P b) { return a.x < b.x;}
  
  
ll fac[N], nf[N];
  
void build(int n) {
    fac[0] = 1;
    fo(i, 1, n) fac[i] = fac[i - 1]  * i % mo;
    nf[n] = ksm(fac[n], mo - 2);
    fd(i, n, 1) nf[i - 1] = nf[i] * (i) % mo;
}
  
ll C(int n, int m) {
    return fac[n] * nf[m] % mo * nf[n - m] % mo;
}
  
V dg(int x, int y) {
    if(x > y) {
        V b; b.re(1); b[0] = 1; return b;
    }
    if(x == y) {
        V b; b.re(a[x].x + 1);
        b[0] = ksm(2, a[x].y) - 1;
        ff(i, 1, b.si) b[i] = C(a[x].x, i);
        return b;
    }
    int m = x+y >>1;
    return dg(x, m) * dg(m + 1, y);
}
  
V b;
      
int main() {
//  freopen("a.in", "r", stdin);
    ntt :: build();
    build(2e5);
    scanf("%d", &n);
    fo(i, 1, n) scanf("%d", &a[i].x);
    fo(i, 1, n) scanf("%d", &a[i].y);
    b = dg(1, n);
    ff(i, 0, b.si) pp("%lld ", b[i]);
}







//Tourist
//NTT算法
namespace ntt
{
struct num
{
    double x, y;
    num()
    {
        x = y = 0;
    }
    num(double x, double y) : x(x), y(y) {}
};
inline num operator+(num a, num b)
{
    return num(a.x + b.x, a.y + b.y);
}
inline num operator-(num a, num b)
{
    return num(a.x - b.x, a.y - b.y);
}
inline num operator*(num a, num b)
{
    return num(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline num conj(num a)
{
    return num(a.x, -a.y);
}

int base = 1;
vector<num> roots = { { 0, 0 }, { 1, 0 } };
vector<int> rev = { 0, 1 };
const double PI = acosl(-1.0);

void ensure_base(int nbase)
{
    if (nbase <= base)
        return;
    rev.resize(1 << nbase);
    for (int i = 0; i < (1 << nbase); i++) rev[i] = (rev[i >> 1] >> 1) + ((i & 1) << (nbase - 1));
    roots.resize(1 << nbase);
    while (base < nbase)
    {
        double angle = 2 * PI / (1 << (base + 1));
        for (int i = 1 << (base - 1); i < (1 << base); i++)
        {
            roots[i << 1] = roots[i];
            double angle_i = angle * (2 * i + 1 - (1 << base));
            roots[(i << 1) + 1] = num(cos(angle_i), sin(angle_i));
        }
        base++;
    }
}

void fft(vector<num> &a, int n = -1)
{
    if (n == -1)
        n = a.size();
    assert((n & (n - 1)) == 0);
    int zeros = __builtin_ctz(n);
    ensure_base(zeros);
    int shift = base - zeros;
    for (int i = 0; i < n; i++)
        if (i < (rev[i] >> shift))
            swap(a[i], a[rev[i] >> shift]);
    for (int k = 1; k < n; k <<= 1)
    {
        for (int i = 0; i < n; i += 2 * k)
        {
            for (int j = 0; j < k; j++)
            {
                num z = a[i + j + k] * roots[j + k];
                a[i + j + k] = a[i + j] - z;
                a[i + j] = a[i + j] + z;
            }
        }
    }
}

vector<num> fa, fb;

vector<int> multiply(vector<int> &a, vector<int> &b)
{
    int need = a.size() + b.size() - 1;
    int nbase = 0;
    while ((1 << nbase) < need) nbase++;
    ensure_base(nbase);
    int sz = 1 << nbase;
    if (sz > (int)fa.size())
        fa.resize(sz);
    for (int i = 0; i < sz; i++)
    {
        int x = (i < (int)a.size() ? a[i] : 0);
        int y = (i < (int)b.size() ? b[i] : 0);
        fa[i] = num(x, y);
    }
    fft(fa, sz);
    num r(0, -0.25 / sz);
    for (int i = 0; i <= (sz >> 1); i++)
    {
        int j = (sz - i) & (sz - 1);
        num z = (fa[j] * fa[j] - conj(fa[i] * fa[i])) * r;
        if (i != j)
            fa[j] = (fa[i] * fa[i] - conj(fa[j] * fa[j])) * r;
        fa[i] = z;
    }
    fft(fa, sz);
    vector<int> res(need);
    for (int i = 0; i < need; i++) res[i] = fa[i].x + 0.5;
    return res;
}

vector<int> multiply(vector<int> &a, vector<int> &b, int m, int eq = 0)
{
    int need = a.size() + b.size() - 1;
    int nbase = 0;
    while ((1 << nbase) < need) nbase++;
    ensure_base(nbase);
    int sz = 1 << nbase;
    if (sz > (int)fa.size())
        fa.resize(sz);
    for (int i = 0; i < (int)a.size(); i++)
    {
        int x = (a[i] % m + m) % m;
        fa[i] = num(x & ((1 << 15) - 1), x >> 15);
    }
    fill(fa.begin() + a.size(), fa.begin() + sz, num{ 0, 0 });
    fft(fa, sz);
    if (sz > (int)fb.size())
        fb.resize(sz);
    if (eq)
        copy(fa.begin(), fa.begin() + sz, fb.begin());
    else
    {
        for (int i = 0; i < (int)b.size(); i++)
        {
            int x = (b[i] % m + m) % m;
            fb[i] = num(x & ((1 << 15) - 1), x >> 15);
        }
        fill(fb.begin() + b.size(), fb.begin() + sz, num{ 0, 0 });
        fft(fb, sz);
    }
    double ratio = 0.25 / sz;
    num r2(0, -1), r3(ratio, 0), r4(0, -ratio), r5(0, 1);
    for (int i = 0; i <= (sz >> 1); i++)
    {
        int j = (sz - i) & (sz - 1);
        num a1 = (fa[i] + conj(fa[j]));
        num a2 = (fa[i] - conj(fa[j])) * r2;
        num b1 = (fb[i] + conj(fb[j])) * r3;
        num b2 = (fb[i] - conj(fb[j])) * r4;
        if (i != j)
        {
            num c1 = (fa[j] + conj(fa[i]));
            num c2 = (fa[j] - conj(fa[i])) * r2;
            num d1 = (fb[j] + conj(fb[i])) * r3;
            num d2 = (fb[j] - conj(fb[i])) * r4;
            fa[i] = c1 * d1 + c2 * d2 * r5;
            fb[i] = c1 * d2 + c2 * d1;
        }
        fa[j] = a1 * b1 + a2 * b2 * r5;
        fb[j] = a1 * b2 + a2 * b1;
    }
    fft(fa, sz);
    fft(fb, sz);
    vector<int> res(need);
    for (int i = 0; i < need; i++)
    {
        ll aa = fa[i].x + 0.5;
        ll bb = fb[i].x + 0.5;
        ll cc = fa[i].y + 0.5;
        res[i] = (aa + ((bb % m) << 15) + ((cc % m) << 30)) % m;
    }
    return res;
}
vector<int> square(vector<int> &a, int m)
{
    return multiply(a, a, m, 1);
}
};