//并查集
int fa[maxn];
int Find(int x)
{
    return x==fa[x]?x:fa[x]=Find(fa[x]);
}
void unit(int u,int v)
{
    int xx=Find(u);
    int yy=Find(v);
    if(xx!=yy)
    {
        fa[xx]=yy;
    }
}




//DSU 并查集 by:sjf
struct DSU
{
    int pre[MAX];
    void init(int n)
    {
        for(int i=0; i<=n; i++) pre[i]=i;
    }
    int find(int x)
    {
        if(pre[x]!=x) pre[x]=find(pre[x]);
        return pre[x];
    }
    bool merge(int a,int b)
    {
        int ra,rb;
        ra=find(a);
        rb=find(b);
        if(ra!=rb)
        {
            pre[ra]=rb;
            return 1;
        }
        return 0;
    }
} dsu;