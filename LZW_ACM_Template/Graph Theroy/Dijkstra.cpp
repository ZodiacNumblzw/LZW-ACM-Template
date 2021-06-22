//dijkstra算法求最短路 堆优化 O(nlogn)   性能较高但各大oj都ce??
const int INF=0x3f3f3f3f;
int n,m;
struct qnode
{
    int v,c;
    qnode(int v,int c):v(v),c(c){}
    friend bool operator<(qnode q1,qnode q2)
    {
        return q1.c>q2.c;          //定义优先级相反，此为小值更优先
    }
};
struct edge
{
    int v,w;
    edge(int v,int w):v(v),w(w){}
};
vector<edge> E[maxn];
int vis[maxn];
int dist[maxn];
inline void add(int u,int v,int w)
{
    E[u].emplace_back(v,w);
    E[v].emplace_back(u,w);        //emplace不适用于poj -_-!
}
void dijkstra(int x)       //起点
{
    //memset(vis,0,sizeof(vis));              //这个地方注意如果test Case大于2e5的话用memset可能会被卡常
   // memset(dist,INF,sizeof(dist));          //必要时用for(int i=1;i<=n;i++)初始化  memset慎用
    for(int i=1;i<=n;i++)
        vis[i]=0,dist[i]=INF;
    priority_queue<qnode> q;
    dist[x]=0;
    q.emplace(x,0);
    while(!q.empty())
    {
        qnode temp=q.top();
        q.pop();
        int point=temp.v;       //选取松弛点
        if(vis[point])
            continue;
        vis[point]=1;
        for(edge i:E[point])
        {
            int v=i.v,w=i.w;
            if(!vis[v]&&dist[v]>dist[point]+w)   //松弛操作
            {
                dist[v]=dist[point]+w;
                q.emplace(v,dist[v]);
            }
        }
    }
}

/*
ps: poj上不让用emplace 万能头文件 和包括push({u,v}) foreach等一切c++11的操作
*/



//dijkstra算法求最短路 堆优化 O(nlogn) 有点改变但影响不大 可以在各大oj上编译
int n,m;
struct Dijkstra
{
    struct qnode
    {
        int v,c;
        qnode(int v,int c):v(v),c(c) {}
        friend bool operator<(qnode q1,qnode q2)
        {
            return q1.c>q2.c;          //定义优先级相反，此为小值更优先
        }
    };
    struct edge
    {
        int v,w;
        edge(int v,int w):v(v),w(w) {}
    };
    vector<edge> E[maxn];
    int vis[maxn];
    int dist[maxn];
    inline void add(int u,int v,int w)
    {
        E[u].push_back(edge(v,w));
        E[v].push_back(edge(u,w));
    }
    void solve(int x)       //起点
    {
        //memset(vis,0,sizeof(vis));              //这个地方注意如果test Case大于2e5的话用memset可能会被卡常
        //memset(dist,INF,sizeof(dist));          //必要时用for(int i=1;i<=n;i++)初始化  memset慎用
        for(int i=1; i<=n; i++)
            vis[i]=0,dist[i]=INF;
        priority_queue<qnode> q;
        dist[x]=0;
        q.push(qnode(x,0));
        while(!q.empty())
        {
            qnode temp=q.top();
            q.pop();
            int point=temp.v;       //选取松弛点
            if(vis[point])
                continue;
            vis[point]=1;
            int len=E[point].size();
            for(int i=0; i<len; i++)
            {
                int v=E[point][i].v,w=E[point][i].w;
                if(!vis[v]&&dist[v]>dist[point]+w)   //松弛操作
                {
                    dist[v]=dist[point]+w;
                    q.push(qnode(v,dist[v]));
                }
            }
        }
    }
    int query(int t)
    {
        return dist[t];
    }
}ans;





#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<queue>
#include<stack>
#include<map>
#include<sstream>
using namespace std;
typedef long long ll;
const int maxn = 1e3 + 10;
const int INF = 1 << 30;
int T, n, m;
struct edge
{
    int from, to, dist;
    edge(int u, int v, int d):from(u), to(v), dist(d){}
    edge(){}
};
struct Heapnode
{
    int d, u;//d为距离，u为起点
    Heapnode(){}
    Heapnode(int d, int u):d(d), u(u){}
    bool operator <(const Heapnode & a)const
    {
        return d > a.d;//这样优先队列先取出d小的
    }
};
struct Dijkstra
{
    int n, m;
    vector<edge>edges;//存边的信息
    vector<int>G[maxn];//G[i]表示起点为i的边的序号集
    bool v[maxn];//标记点是否加入集合
    int d[maxn];//起点s到各个点的最短路
    int p[maxn];//倒叙记录路径
    Dijkstra(){}
    void init(int n)
    {
        this -> n = n;
        for(int i = 0; i < n; i++)G[i].clear();
        edges.clear();
    }
    void addedge(int from, int to, int dist)
    {
        edges.push_back(edge(from, to, dist));
        m = edges.size();
        G[from].push_back(m - 1);//存以from为起点的下一条边
    }
    void dijkstra(int s)//以s为起点
    {
        priority_queue<Heapnode>q;
        for(int i = 0; i < n; i++)d[i] = INF;
        d[s] = 0;
        memset(v, 0, sizeof(v));
        memset(p, -1, sizeof(p));
        q.push(Heapnode(0, s));
        while(!q.empty())
        {
            Heapnode now = q.top();
            q.pop();
            int u = now.u;//当前起点
            if(v[u])continue;//如果已经加入集合，continue
            v[u] = 1;
            for(int i = 0; i < G[u].size(); i++)
            {
                edge& e = edges[G[u][i]];//引用节省代码
                if(d[e.to] > d[u] + e.dist)
                {
                    d[e.to] = d[u] + e.dist;
                    p[e.to] = G[u][i];//记录e.to前的边的编号，p存的是边的下标,这样可以通过边找出之前的点以及每条路的路径，如果用邻接矩阵存储的话这里可以直接存节点u
                    q.push(Heapnode(d[e.to], e.to));
                }
            }
        }
    }
    void output(int u)
    {
        for(int i = 0; i < n; i++)
        {
            if(i == u)continue;
            printf("从%d到%d距离是：%2d   ", u, i, d[i]);
            stack<int>q;//存的是边的编号
            int x = i;//x就是路径上所有的点
            while(p[x] != -1)
            {
                q.push(x);
                x = edges[p[x]].from;//x变成这条边的起点
            }
            cout<<u;
            while(!q.empty())
            {
                cout<<"->"<<q.top();
                q.pop();
            }
            cout<<endl;
        }
    }
};
Dijkstra ans;
int main()
{
    cin >> n >> m;
    ans.init(n);
    for(int i = 0; i < m; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        ans.addedge(u, v, w);
    }
    int u = 0;
    ans.dijkstra(u);
    ans.output(u);
}