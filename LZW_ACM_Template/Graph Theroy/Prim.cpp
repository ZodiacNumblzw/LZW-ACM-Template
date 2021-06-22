//Prim算法 O(VlogV+E) 适用于稠密图  by:LZW
struct Prim
{
    //Prim算法中点的编号必须是1~n
    #define MAX 510
    struct node
    {
        int u,w;
        node() {}
        node(int u,int w) :u(u),w(w) {}
        friend bool operator <(node a,node b)
        {
            return a.w>b.w;
        }
    };
    //邻接表
    vector<node> mp[MAX];
    //是否被选入点集
    bool vis[MAX];
    //到已选点集中的最小距离
    int dis[MAX];
    Prim()
    {
        for(int i=0;i<MAX;i++) mp[i].clear();
        memset(vis,0,sizeof(vis));
        memset(dis,0x3f3f3f3f,sizeof(dis));
    }
    void addEdge(int u,int v,int w)
    {
        mp[u].push_back({v,w});
        mp[v].push_back({u,w});
    }
    int prim()
    {
        int res=0;
        node to;
        priority_queue<node> q;
        dis[1]=0;
        q.push(node(1,dis[1]));
        while(!q.empty())
        {
            node temp=q.top();
            q.pop();
            if(vis[temp.u]) continue;
            vis[temp.u]=1;
            res+=dis[temp.u];
            int len=mp[temp.u].size();
            for(int i=0; i<len; i++)
            {
                to=mp[temp.u][i];
                if(!vis[to.u]&&dis[to.u]>to.w)
                {
                    dis[to.u]=to.w;
                    q.push(node(to.u,dis[to.u]));
                }
            }
        }
        return res;
    }
    #undef MAX
}P;
