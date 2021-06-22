//Kruskal  O(mlogm)   适用于边数较少的稀疏图  by:LZW
struct Kruskal
{
#define MAX 105
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
        // 返回值1为merge成功，否则两个点在一个集合中，merge失败
        bool merge(int a,int b)
        {
            int fa=find(a),fb=find(b);
            if(fa!=fb)
                return pre[fa]=fb,1;
            return 0;
        }
    } dsu;
    struct node
    {
        int u,v,w;
        node() {}
        node(int u,int v,int w):u(u),v(v),w(w) {}
        friend bool operator<(node a,node b)
        {
            return a.w<b.w;
        }
    };
    vector<node> edge;
    Kruskal()
    {
        edge.clear();
    }
    void addEdge(int u,int v,int w)
    {
        edge.push_back({u,v,w});
    }
    int kruskal(int n)
    {
        int res=0;
        dsu.init(n);
        sort(edge.begin(),edge.end());
        int len=edge.size();
        for(int i=0; i<len; i++)
        {
            if(dsu.merge(edge[i].u,edge[i].v)) res+=edge[i].w;
        }
        return res;
    }
#undef MAX
}K;
