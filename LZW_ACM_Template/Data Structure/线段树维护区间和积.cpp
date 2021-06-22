//Ïß¶ÎÊ÷°å×Ó2 SegmentTree
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
bool Finish_read;
template<class T>inline void read(T &x)
{
    Finish_read=0;
    x=0;
    int f=1;
    char ch=getchar();
    while(!isdigit(ch))
    {
        if(ch=='-')f=-1;
        if(ch==EOF)return;
        ch=getchar();
    }
    while(isdigit(ch))x=x*10+ch-'0',ch=getchar();
    x*=f;
    Finish_read=1;
}
template<class T>inline void print(T x)
{
    if(x/10!=0)print(x/10);
    putchar(x%10+'0');
}
template<class T>inline void writeln(T x)
{
    if(x<0)putchar('-');
    x=abs(x);
    print(x);
    putchar('\n');
}
template<class T>inline void write(T x)
{
    if(x<0)putchar('-');
    x=abs(x);
    print(x);
}
/*================Header Template==============*/
const int maxn=2e5+5;
#define ls(o) o<<1
#define rs(o) (o<<1|1)
int n,m;
ll p;
int a[maxn];
struct node
{
    int l,r,sz;
    ll val;
    ll addtag,multag;
} t[maxn<<2];
/*==================Define Area================*/
namespace SegmentTree
{
void update(int o)
{
    t[o].val=(t[ls(o)].val+t[rs(o)].val)%p;
}

void pushdown(int o)
{
    if(t[o].multag!=1)
    {
        t[ls(o)].multag*=t[o].multag;
        t[ls(o)].multag%=p;
        t[rs(o)].multag*=t[o].multag;
        t[rs(o)].multag%=p;
        t[ls(o)].addtag*=t[o].multag;
        t[ls(o)].addtag%=p;
        t[rs(o)].addtag*=t[o].multag;
        t[rs(o)].addtag%=p;
        t[ls(o)].val*=t[o].multag;
        t[ls(o)].val%=p;
        t[rs(o)].val*=t[o].multag;
        t[rs(o)].val%=p;
        t[o].multag=1;
    }
    if(t[o].addtag)
    {
        t[ls(o)].addtag+=t[o].addtag;
        t[ls(o)].addtag%=p;
        t[rs(o)].addtag+=t[o].addtag;
        t[rs(o)].addtag%=p;
        t[ls(o)].val+=t[o].addtag*t[ls(o)].sz;
        t[ls(o)].val%=p;
        t[rs(o)].val+=t[o].addtag*t[rs(o)].sz;
        t[rs(o)].val%=p;
        t[o].addtag=0;
    }
    return ;
}

void Build(int o,int l,int r)
{
    t[o].l=l,t[o].r=r;
    t[o].sz=r-l+1;
    t[o].multag=1;
    if(t[o].l==t[o].r)
    {
        t[o].val=a[l];
        return ;
    }
    int mid=(l+r)>>1;
    Build(ls(o),l,mid);
    Build(rs(o),mid+1,r);
    update(o);
}

ll IntervalSum(int o,int l,int r)
{
    if(t[o].l>r||t[o].r<l) return 0;
    if(t[o].l>=l&&t[o].r<=r)
    {
        pushdown(o);
        return t[o].val;
    }
    pushdown(o);
    ll ans=0;
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) ans+=IntervalSum(ls(o),l,r),ans%=p;
    if(mid<=r) ans+=IntervalSum(rs(o),l,r),ans%=p;
    return ans;
}

void IntervalAdd(int o,int l,int r,int v)
{
    if(t[o].l>r||t[o].r<l) return ;
    if(t[o].l>=l&&t[o].r<=r)
    {
        t[o].val+=t[o].sz*v;
        t[o].addtag+=v;
        t[o].addtag%=p;
        t[o].val%=p;
        return ;
    }
    pushdown(o);
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) IntervalAdd(ls(o),l,r,v);
    if(mid<=r) IntervalAdd(rs(o),l,r,v);
    update(o);
    return ;
}

void IntervalMul(int o,int l,int r,int v)
{
    if(t[o].l>r||t[o].r<l) return ;
    if(t[o].l>=l&&t[o].r<=r)
    {
        pushdown(o);
        t[o].val*=v;
        t[o].val%=p;
        t[o].addtag*=v;
        t[o].addtag%=p;
        t[o].multag*=v;
        t[o].multag%=p;
        return ;
    }
    pushdown(o);
    int mid=(t[o].l+t[o].r)>>1;
    if(mid>=l) IntervalMul(ls(o),l,r,v);
    if(mid<=r) IntervalMul(rs(o),l,r,v);
    update(o);
    return ;
}
}
using namespace SegmentTree;

int main()
{
    read(n);
    read(m);
    read(p);
    for(int i=1; i<=n; i++)
    {
        read(a[i]);
    }
    Build(1,1,n);
    for(int i=1; i<=m; i++)
    {
        int opt;
        read(opt);
        if(opt==1)
        {
            int x,y,k;
            read(x);
            read(y);
            read(k);
            IntervalMul(1,x,y,k);
        }
        if(opt==2)
        {
            int x,y,k;
            read(x);
            read(y);
            read(k);
            IntervalAdd(1,x,y,k);
        }
        if(opt==3)
        {
            int x,y;
            read(x);
            read(y);
            ll ans=IntervalSum(1,x,y);
            printf("%lld\n",ans);
        }
    }
    return 0;
}
