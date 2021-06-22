//Ī���㷨  O(nsqrt(n))
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
    nowAns +=mp[a[p]];          //�˴�ע���޸�
    mp[a[p]]++;
}

void del(int p)
{
    mp[a[p]]--;                 //�˴�ע���޸�
    nowAns -= mp[a[p]];
}

void Mos(int n,int m)
{
    for(int i=0;i<m;i++)              //��ʼ��ѯ�ʵı��
        q[i].id=i;
    block = (int)sqrt(n);
    sort(q, q + m);
    int l = 1, r = 1;              //��ǰ����
    mp[a[1]]++;                     //�����ǳ�ʼ��
    for(int i = 0; i < m; i++)
    {
        const Query &c = q[i];
        if(c.l==c.r){                           //����
            ans[c.id].fz = 0,ans[c.id].fm =1;
            continue;
        }
        while(l > c.l) add(--l);      //����ת�� O(1)
        while(r < c.r) add(++r);
        while(l < c.l) del(l++);
        while(r > c.r) del(r--);
        ans[c.id].fz = nowAns;
        ans[c.id].fm = calc(r+1-l);
    }
    for(int i = 0; i < m; i++)                //��������
    {
        ll g=gcd(ans[i].fz,ans[i].fm);
        printf("%lld/%lld\n",ans[i].fz/g,ans[i].fm/g);
    }
}