//普通FFT 自己写的递归形式FFT，可能不太好用（极其不好用）
//SPOJ TSUM因不明原因运行错误（段错误/爆内存）
const ld pi=acos(-1);

struct Complex
{
    ld r,i;
    Complex(ld rr=0,ld ii=0):r(rr),i(ii){}
    friend Complex operator+(Complex a,Complex b){return Complex(a.r+b.r,a.i+b.i);}
    friend Complex operator-(Complex a,Complex b){return Complex(a.r-b.r,a.i-b.i);}
    friend Complex operator*(Complex a,Complex b){return Complex(a.r*b.r-a.i*b.i,a.r*b.i+a.i*b.r);}
};


void FFT(int n,Complex *c,int type)
{
    if(n==1)return;
    Complex a1[n>>1],a2[n>>1];
    for(int i=0;i<=n;i+=2)
    {
        a1[i>>1]=c[i];
        a2[i>>1]=c[i+1];
    }
    FFT(n>>1,a1,type);
    FFT(n>>1,a2,type);
    Complex Wn=Complex(cos(2.0*pi/n),type*sin(2.0*pi/n)),w=Complex(1,0);
    for(int i=0;i<(n>>1);i++,w=w*Wn)
    {
        c[i]=a1[i]+w*a2[i];
        c[i+(n>>1)]=a1[i]-w*a2[i];
    }
}








//由kuangbin的FFT板子改编的板子


//复数结构体
struct Complex
{
    double x,y;//实部和虚部 x+yi
    Complex(double _x = 0.0,double _y = 0.0){x = _x;y = _y;}
    Complex operator -(const Complex &b)const{return Complex(x-b.x,y-b.y);}
    Complex operator +(const Complex &b)const{return Complex(x+b.x,y+b.y);}
    Complex operator *(const Complex &b)const{return Complex(x*b.x-y*b.y,x*b.y+y*b.x);}
};
/*
* 进行FFT和IFFT前的反转变换。
* 位置i和 （i二进制反转后位置）互换
* len必须取2的幂
*/
void change(Complex y[],int len)
{
    int i,j,k;
    for(i = 1, j = len/2; i <len-1; i++)
    {
        if(i < j)swap(y[i],y[j]);
//交换互为小标反转的元素，i<j保证交换一次
//i做正常的+1，j左反转类型的+1,始终保持i和j是反转的
        k = len/2;
        while(j >= k)
        {
            j -= k;
            k /= 2;
        }
        if(j < k)j += k;
    }
}
/*
* 做FFT
* len必须为2^k形式，
* on==1时是DFT，on==-1时是IDFT
*/
void fft(Complex y[],int len,int on)
{
    change(y,len);
    for(int h = 2; h <= len; h <<= 1)
    {
        Complex wn(cos(-on*2*PI/h),sin(-on*2*PI/h));
        for(int j = 0; j < len; j+=h)
        {
            Complex w(1,0);
            for(int k = j; k < j+h/2; k++)
            {
                Complex u = y[k];
                Complex t = w*y[k+h/2];
                y[k] = u+t;
                y[k+h/2] = u-t;
                w = w*wn;
            }
        }
    }
    if(on == -1)
        for(int i = 0; i < len; i++)
            y[i].x /= len;
}
Complex x1[maxn],x2[maxn];
int Multiply(int *a,int len1,const int *b,int len2)   //这里的len1和len2为数组的长度
{                                                    //在传参的时候记得len+1
    int len = 1;
    while(len < len1*2 || len < len2*2)len<<=1;
    for(int i=0; i<len1; i++)
        x1[i]=Complex(a[i],0);
    for(int i=len1; i<len; i++)
        x1[i]=Complex(0,0);
    for(int i=0; i<len2; i++)
        x2[i]=Complex(b[i],0);
    for(int i=len2; i<len; i++)
        x2[i]=Complex(0,0);
    fft(x1,len,1);
    fft(x2,len,1);
    for(int i = 0; i < len; i++)
        x1[i] = x1[i]*x2[i];
    fft(x1,len,-1);
    for(int i = 0; i < len; i++)
        a[i] = (int)(x1[i].x+0.5);
    len=len1+len2-1;
    while(a[len] <= 0 && len > 0)len--;
    return len;         //返回的len为数组的最后一个数的下标，是数组的长度-1
}
int a1[maxn],a11[maxn],a2[maxn],a3[maxn];
//玄学开maxn










//还是FFT的板子，代码比较简洁，非递归写法，不会炸
namespace FFT
{
    struct Complex
    {
        double r,i;
        Complex(double real=0.0,double image=0.0)
        {
            r=real; i=image;
        }
        Complex operator + (const Complex o)
        {
            return Complex(r+o.r,i+o.i);
        }
        Complex operator - (const Complex o)
        {
            return Complex(r-o.r,i-o.i);
        }
        Complex operator * (const Complex o)
        {
            return Complex(r*o.r-i*o.i,r*o.i+i*o.r);
        }
    };

    void brc(Complex *y, int l)
    {
        register int i,j,k;
        for( i = 1, j = l / 2; i < l - 1; i++)
        {
            if (i < j) swap(y[i], y[j]);
            k = l / 2; while ( j >= k) j -= k,k /= 2;
            if (j < k) j += k;
        }
    }

    void FFT(Complex *y, int len, double on)
    {
        register int h, j, k;
        Complex u, t; brc(y, len);
        for(h = 2; h <= len; h <<= 1)
        {
            Complex wn(cos(on * 2 * PI / h), sin(on * 2 * PI / h));
            for(j = 0; j < len; j += h)
            {
                Complex w(1, 0);
                for(k = j; k < j + h / 2; k++)
                {
                    u = y[k]; t = w * y[k + h / 2];
                    y[k] = u + t; y[k + h / 2] = u - t;
                    w = w * wn;
                }
            }
        }
        if (on<0) for (int i = 0; i < len; i++) y[i].r/=len;
    }

}

FFT::Complex A[N],B[N],C[N],ans[N];







//FFT Tourist神仙的板子
#pragma GCC optimize(3)
#include<bits/stdc++.h>
#define MAXN 100005
#define INF 1000000000
#define MOD 998244353
#define F first
#define S second
using namespace std;
typedef long long ll;
typedef pair<int,int> P;
int n,k;
const double PI=acos(-1.0);
//tourist
namespace fft
{
    struct num
    {
        double x,y;
        num() {x=y=0;}
        num(double x,double y):x(x),y(y){}
    };
    inline num operator+(num a,num b) {return num(a.x+b.x,a.y+b.y);}
    inline num operator-(num a,num b) {return num(a.x-b.x,a.y-b.y);}
    inline num operator*(num a,num b) {return num(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
    inline num conj(num a) {return num(a.x,-a.y);}

    int base=1;
    vector<num> roots={{0,0},{1,0}};
    vector<int> rev={0,1};
    const double PI=acosl(-1.0);

    void ensure_base(int nbase)
    {
        if(nbase<=base) return;
        rev.resize(1<<nbase);
        for(int i=0;i<(1<<nbase);i++)
            rev[i]=(rev[i>>1]>>1)+((i&1)<<(nbase-1));
        roots.resize(1<<nbase);
        while(base<nbase)
        {
            double angle=2*PI/(1<<(base+1));
            for(int i=1<<(base-1);i<(1<<base);i++)
            {
                roots[i<<1]=roots[i];
                double angle_i=angle*(2*i+1-(1<<base));
                roots[(i<<1)+1]=num(cos(angle_i),sin(angle_i));
            }
            base++;
        }
    }

    void fft(vector<num> &a,int n=-1)
    {
        if(n==-1) n=a.size();
        assert((n&(n-1))==0);
        int zeros=__builtin_ctz(n);
        ensure_base(zeros);
        int shift=base-zeros;
        for(int i=0;i<n;i++)
            if(i<(rev[i]>>shift))
                swap(a[i],a[rev[i]>>shift]);
        for(int k=1;k<n;k<<=1)
        {
            for(int i=0;i<n;i+=2*k)
            {
                for(int j=0;j<k;j++)
                {
                    num z=a[i+j+k]*roots[j+k];
                    a[i+j+k]=a[i+j]-z;
                    a[i+j]=a[i+j]+z;
                }
            }
        }
    }

    vector<num> fa,fb;

    vector<int> multiply(vector<int> &a, vector<int> &b)
    {
        int need=a.size()+b.size()-1;
        int nbase=0;
        while((1<<nbase)<need) nbase++;
        ensure_base(nbase);
        int sz=1<<nbase;
        if(sz>(int)fa.size()) fa.resize(sz);
        for(int i=0;i<sz;i++)
        {
            int x=(i<(int)a.size()?a[i]:0);
            int y=(i<(int)b.size()?b[i]:0);
            fa[i]=num(x,y);
        }
        fft(fa,sz);
        num r(0,-0.25/sz);
        for(int i=0;i<=(sz>>1);i++)
        {
            int j=(sz-i)&(sz-1);
            num z=(fa[j]*fa[j]-conj(fa[i]*fa[i]))*r;
            if(i!=j) fa[j]=(fa[i]*fa[i]-conj(fa[j]*fa[j]))*r;
            fa[i]=z;
        }
        fft(fa,sz);
        vector<int> res(need);
        for(int i=0;i<need;i++) res[i]=fa[i].x+0.5;
        return res;
    }

    vector<int> multiply_mod(vector<int> &a,vector<int> &b,int m,int eq=0)
    {
        int need=a.size()+b.size()-1;
        int nbase=0;
        while((1<<nbase)<need) nbase++;
        ensure_base(nbase);
        int sz=1<<nbase;
        if(sz>(int)fa.size()) fa.resize(sz);
        for(int i=0;i<(int)a.size();i++)
        {
            int x=(a[i]%m+m)%m;
            fa[i]=num(x&((1<<15)-1),x>>15);
        }
        fill(fa.begin()+a.size(),fa.begin()+sz,num{0,0});
        fft(fa,sz);
        if(sz>(int)fb.size()) fb.resize(sz);
        if(eq) copy(fa.begin(),fa.begin()+sz,fb.begin());
        else
        {
            for(int i=0;i<(int)b.size();i++)
            {
                int x=(b[i]%m+m)%m;
                fb[i]=num(x&((1<<15)-1),x>>15);
            }
            fill(fb.begin()+b.size(),fb.begin()+sz,num{0,0});
            fft(fb,sz);
        }
        double ratio=0.25/sz;
        num r2(0,-1),r3(ratio,0),r4(0,-ratio),r5(0,1);
        for(int i=0;i<=(sz>>1);i++)
        {
            int j=(sz-i)&(sz-1);
            num a1=(fa[i]+conj(fa[j]));
            num a2=(fa[i]-conj(fa[j]))*r2;
            num b1=(fb[i]+conj(fb[j]))*r3;
            num b2=(fb[i]-conj(fb[j]))*r4;
            if(i!=j)
            {
                num c1=(fa[j]+conj(fa[i]));
                num c2=(fa[j]-conj(fa[i]))*r2;
                num d1=(fb[j]+conj(fb[i]))*r3;
                num d2=(fb[j]-conj(fb[i]))*r4;
                fa[i]=c1*d1+c2*d2*r5;
                fb[i]=c1*d2+c2*d1;
            }
            fa[j]=a1*b1+a2*b2*r5;
            fb[j]=a1*b2+a2*b1;
        }
        fft(fa,sz);fft(fb,sz);
        vector<int> res(need);
        for(int i=0;i<need;i++)
        {
            ll aa=fa[i].x+0.5;
            ll bb=fb[i].x+0.5;
            ll cc=fa[i].y+0.5;
            res[i]=(aa+((bb%m)<<15)+((cc%m)<<30))%m;
        }
        return res;
    }
    vector<int> square_mod(vector<int> &a,int m)
    {
        return multiply_mod(a,a,m,1);
    }
};
vector<int> v;
int main()
{
    scanf("%d%d",&n,&k);
    v.resize(10,0);
    for(int i=0;i<k;i++)
    {
        int x;
        scanf("%d",&x);
        v[x]=1;
    }
    vector<int> ans;
    ans.push_back(1);
    int p=n/2;
    while(p)
    {
        if(p&1) ans=fft::multiply_mod(ans,v,MOD);
        v=fft::square_mod(v,MOD);
        p>>=1;
    }
    int res=0;
    for(auto t:ans) res=(res+1LL*t*t)%MOD;
    printf("%d\n",res);
    return 0;
}