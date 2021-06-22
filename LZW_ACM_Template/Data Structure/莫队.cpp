//莫队算法  O(nsqrt(n))
ll block;
ll nowAns = 0;
ll a[maxn];
inline ll calc(ll x)
{
    return x*(x-1)/2;
}
ll ans[maxn];
struct Query
{
    int l, r, id;
    void input()
    {
        scanf("%d%d",&l,&r);
    }
    bool operator < (const Query &rhs) const
    {
        if(l/block == rhs.l/block) return r > rhs.r;
        return l > rhs.l;
    }
} q[maxn];

void add(int p)
{
    nowAns +=mp[a[p]];          //此处注意修改
    mp[a[p]]++;
}

void del(int p)
{
    mp[a[p]]--;                 //此处注意修改
    nowAns -= mp[a[p]];
}

void Mos(int n,int m)
{
    for(int i=0;i<m;i++)              //初始化询问的编号
        q[i].id=i;
    block = (int)sqrt(n);
    sort(q, q + m);
    int l = 1, r = 1;              //当前窗口
    mp[a[1]]++;                     //别忘记初始化
    for(int i = 0; i < m; i++)
    {
        const Query &c = q[i];
        if(c.l==c.r){                           //特判
            ans[c.id].fz = 0,ans[c.id].fm =1;
            continue;
        }
        while(l > c.l) add(--l);      //四种转移 O(1)
        while(r < c.r) add(++r);
        while(l < c.l) del(l++);
        while(r > c.r) del(r--);
        ans[c.id].fz = nowAns;
        ans[c.id].fm = calc(r+1-l);
    }
    for(int i = 0; i < m; i++)                //按序号输出
    {
        ll g=gcd(ans[i].fz,ans[i].fm);
        printf("%lld/%lld\n",ans[i].fz/g,ans[i].fm/g);
    }
}