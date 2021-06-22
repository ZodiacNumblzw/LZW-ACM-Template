#include <iostream>
#include <cstdio>
#include <cstring>
#include <stack>
#include <queue>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;
const double eps = 1e-6;
const double pi = acos(-1.0);
const int INF = 0x3f3f3f3f;
const int MOD = 1000000007;
#define ll long long
#define CL(a,b) memset(a,b,sizeof(a))
#define MAXN 100010
 
struct node
{
    int l,r;
    ll s,add;//add为每次加的数
}t[MAXN<<2];
int hh[MAXN];
int n,q;
ll ans;
 
void build(int l, int r, int i)
{
    t[i].l = l;
    t[i].r = r;
    t[i].add = 0;
    if(l == r) return ;
    int mid = (l+r)>>1;
    build(l, mid, i<<1);
    build(mid+1, r, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
void update(int l, int r, int add, int i)
{
    if(t[i].l>r || t[i].r<l) return ;
    if(t[i].l>=l && t[i].r<=r)
    {
        t[i].s += (t[i].r-t[i].l+1)*add;
        t[i].add += add;
        return ;
    }
    if(t[i].add)
    {
        t[i<<1].s += (t[i<<1].r-t[i<<1].l+1)*t[i].add;
        t[i<<1].add += t[i].add;
        t[i<<1|1].s += (t[i<<1|1].r-t[i<<1|1].l+1)*t[i].add;
        t[i<<1|1].add += t[i].add;
        t[i].add = 0;
    }
    update(l, r, add, i<<1);
    update(l, r, add, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
void query(int l, int r, int i)
{
    if(t[i].l>r || t[i].r<l) return ;
    if(t[i].l>=l && t[i].r<=r)
    {
        ans += t[i].s;
        return ;
    }
    if(t[i].add)
    {
        t[i<<1].s += (t[i<<1].r-t[i<<1].l+1)*t[i].add;
        t[i<<1].add += t[i].add;
        t[i<<1|1].s += (t[i<<1|1].r-t[i<<1|1].l+1)*t[i].add;
        t[i<<1|1].add += t[i].add;
        t[i].add = 0;
    }
    query(l, r, i<<1);
    query(l, r, i<<1|1);
    t[i].s = t[i<<1].s+t[i<<1|1].s;
}
 
int main()
{
    int a,b,c;
    ll k;
    char ch;
    while(scanf("%d%d",&n,&q)==2)
    {
        for(int i=1; i<=n; i++)
            scanf("%d",&hh[i]);
        build(1, n, 1);
        for(int i=1; i<=n; i++)
            update(i, i, hh[i], 1);
        while(q--)
        {
            getchar();
            scanf("%c",&ch);
            if(ch == 'C')
            {
                scanf("%d%d%d",&a,&b,&c);
                update(a, b, c, 1);
            }
            if(ch == 'Q')
            {
                ans = 0;
                scanf("%d%d",&a,&b);
                query(a, b, 1);
                printf("%lld\n",ans);
            }
        }
    }
    return 0;
}

//还是线段树板子 维护区间和，区间平方和，支持区间修改
#include <bits/stdc++.h>
using namespace std;
typedef long long int ll;
struct node
{
    ll sum;//当前节点所表示的区间的和
    ll asign;//加法延迟标记
    ll msign;//乘法延迟标记
    ll sq;
};
ll a[10009];//以此数组建树
ll n,m;//数组的大小,取模,询问次数
node t[4*10009];
void build(int root,int l,int r)//build(1,1,n)进行建树
{
    t[root].msign=1;
    if(l==r)
    {t[root].sum=a[l];
    t[root].sq=a[l]*a[l];
    return ;}
    int mid=(l+r)>>1;
    build(root<<1,l,mid);
    build(root<<1|1,mid+1,r);
    t[root].sum=t[root<<1].sum+t[root<<1|1].sum;
    t[root].sq=t[root<<1].sq+t[root<<1|1].sq;
}
void push_down(int rt,int l,int r)
{
    if(t[rt].msign!=1)
    {
        t[rt<<1].msign*=t[rt].msign;
        t[rt<<1].asign*=t[rt].msign;
        t[rt<<1|1].msign*=t[rt].msign;
        t[rt<<1|1].asign*=t[rt].msign;
        t[rt<<1].sum*=t[rt].msign;
        t[rt<<1].sq*=t[rt].msign*t[rt].msign;
        t[rt<<1|1].sum*=t[rt].msign;
        t[rt<<1|1].sq*=t[rt].msign*t[rt].msign;
        t[rt].msign=1;
    }
    if(t[rt].asign)
    {
        int mid=(l+r)>>1;
        t[rt<<1].sq+=(2*t[rt].asign*t[rt<<1].sum+t[rt].asign*t[rt].asign*(mid-l+1));
        t[rt<<1].sum+=(t[rt].asign*(mid-l+1));
        t[rt<<1|1].sq+=(2*t[rt].asign*t[rt<<1|1].sum+t[rt].asign*t[rt].asign*(r-mid));
        t[rt<<1|1].sum+=(t[rt].asign*(r-mid));
        t[rt<<1].asign+=t[rt].asign;
        t[rt<<1|1].asign+=t[rt].asign;
        t[rt].asign=0;
    }
}
void range_add(int rt,int l,int r,int x,int y,ll val)//[x,y]区间加上val
{
    if(x<=l&&y>=r)
    {
        t[rt].sq=t[rt].sq+2*val*t[rt].sum+(r-l+1)*val*val;
        t[rt].sum=t[rt].sum+(r-l+1)*val;
        t[rt].asign=t[rt].asign+val;
        return ;
    }
    if(t[rt].asign!=0||t[rt].msign!=1)
        push_down(rt,l,r);
    int mid=(l+r)>>1;
    if(x<=mid)
        range_add(rt<<1,l,mid,x,y,val);
    if(y>mid)
        range_add(rt<<1|1,mid+1,r,x,y,val);
    t[rt].sum=(t[rt<<1].sum+t[rt<<1|1].sum);
    t[rt].sq=t[rt<<1].sq+t[rt<<1|1].sq;
}
void range_mul(int rt,int l,int r,int x,int y,ll val)//[x,y]区间乘上val
{
    if(x<=l&&y>=r)
    {
        t[rt].sq=val*val*t[rt].sq;
        t[rt].sum=val*t[rt].sum;
        t[rt].asign=(t[rt].asign*val);
        t[rt].msign=(t[rt].msign*val);
        return ;
    }
    if(t[rt].asign!=0||t[rt].msign!=1)
        push_down(rt,l,r);
    int mid=(l+r)>>1;
    if(x<=mid)
        range_mul(rt<<1,l,mid,x,y,val);
    if(y>mid)
        range_mul(rt<<1|1,mid+1,r,x,y,val);
    t[rt].sum=(t[rt<<1].sum+t[rt<<1|1].sum);
    t[rt].sq=t[rt<<1].sq+t[rt<<1|1].sq;
}
ll query_sum(int rt,int l,int r,int x,int y)//查询[x,y]的和
{
    if(x<=l&&y>=r)
        return t[rt].sum;
    if(t[rt].asign!=0||t[rt].msign!=1)
    push_down(rt,l,r);
    int mid=(l+r)>>1;
    ll sum=0;
    if(x<=mid)
        sum+=query_sum(rt<<1,l,mid,x,y);
    if(y>mid)
        sum+=query_sum(rt<<1|1,mid+1,r,x,y);
    return sum;
}
ll query_sq(int rt,int l,int r,int x,int y)
{
    if(x<=l&&y>=r)
        return t[rt].sq;
    if(t[rt].asign!=0||t[rt].msign!=1)
    push_down(rt,l,r);
    int mid=(l+r)>>1;
    ll sum=0;
    if(x<=mid)
        sum+=query_sq(rt<<1,l,mid,x,y);
    if(y>mid)
        sum+=query_sq(rt<<1|1,mid+1,r,x,y);
    return sum;
}
int main()
{
    cin.sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int opt,l,r;
    ll x;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
        {
            cin>>a[i];
        }
    build(1,1,n);
    for(int i=1;i<=m;i++)
    {
        cin>>opt;
        if(opt==1)
        {
            cin>>l>>r;
            cout<<query_sum(1,1,n,l,r)<<'\n';
        }
        if(opt==2)
        {
            cin>>l>>r;
            cout<<query_sq(1,1,n,l,r)<<'\n';
        }
        if(opt==3)
        {
            cin>>l>>r>>x;
            range_mul(1,1,n,l,r,x);
        }
        if(opt==4)
        {
            cin>>l>>r>>x;
            range_add(1,1,n,l,r,x);
        }
    }
    return 0;
}