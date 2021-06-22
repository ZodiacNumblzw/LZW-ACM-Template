//dijkstra�㷨�����· ���Ż� O(nlogn)   ���ܽϸߵ�����oj��ce??
const int INF=0x3f3f3f3f;
int n,m;
struct qnode
{
    int v,c;
    qnode(int v,int c):v(v),c(c){}
    friend bool operator<(qnode q1,qnode q2)
    {
        return q1.c>q2.c;          //�������ȼ��෴����ΪСֵ������
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
    E[v].emplace_back(u,w);        //emplace��������poj -_-!
}
void dijkstra(int x)       //���
{
    //memset(vis,0,sizeof(vis));              //����ط�ע�����test Case����2e5�Ļ���memset���ܻᱻ����
   // memset(dist,INF,sizeof(dist));          //��Ҫʱ��for(int i=1;i<=n;i++)��ʼ��  memset����
    for(int i=1;i<=n;i++)
        vis[i]=0,dist[i]=INF;
    priority_queue<qnode> q;
    dist[x]=0;
    q.emplace(x,0);
    while(!q.empty())
    {
        qnode temp=q.top();
        q.pop();
        int point=temp.v;       //ѡȡ�ɳڵ�
        if(vis[point])
            continue;
        vis[point]=1;
        for(edge i:E[point])
        {
            int v=i.v,w=i.w;
            if(!vis[v]&&dist[v]>dist[point]+w)   //�ɳڲ���
            {
                dist[v]=dist[point]+w;
                q.emplace(v,dist[v]);
            }
        }
    }
}

/*
ps: poj�ϲ�����emplace ����ͷ�ļ� �Ͱ���push({u,v}) foreach��һ��c++11�Ĳ���
*/



//dijkstra�㷨�����· ���Ż� O(nlogn) �е�ı䵫Ӱ�첻�� �����ڸ���oj�ϱ���
int n,m;
struct Dijkstra
{
    struct qnode
    {
        int v,c;
        qnode(int v,int c):v(v),c(c) {}
        friend bool operator<(qnode q1,qnode q2)
        {
            return q1.c>q2.c;          //�������ȼ��෴����ΪСֵ������
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
    void solve(int x)       //���
    {
        //memset(vis,0,sizeof(vis));              //����ط�ע�����test Case����2e5�Ļ���memset���ܻᱻ����
        //memset(dist,INF,sizeof(dist));          //��Ҫʱ��for(int i=1;i<=n;i++)��ʼ��  memset����
        for(int i=1; i<=n; i++)
            vis[i]=0,dist[i]=INF;
        priority_queue<qnode> q;
        dist[x]=0;
        q.push(qnode(x,0));
        while(!q.empty())
        {
            qnode temp=q.top();
            q.pop();
            int point=temp.v;       //ѡȡ�ɳڵ�
            if(vis[point])
                continue;
            vis[point]=1;
            int len=E[point].size();
            for(int i=0; i<len; i++)
            {
                int v=E[point][i].v,w=E[point][i].w;
                if(!vis[v]&&dist[v]>dist[point]+w)   //�ɳڲ���
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
    int d, u;//dΪ���룬uΪ���
    Heapnode(){}
    Heapnode(int d, int u):d(d), u(u){}
    bool operator <(const Heapnode & a)const
    {
        return d > a.d;//�������ȶ�����ȡ��dС��
    }
};
struct Dijkstra
{
    int n, m;
    vector<edge>edges;//��ߵ���Ϣ
    vector<int>G[maxn];//G[i]��ʾ���Ϊi�ıߵ���ż�
    bool v[maxn];//��ǵ��Ƿ���뼯��
    int d[maxn];//���s������������·
    int p[maxn];//�����¼·��
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
        G[from].push_back(m - 1);//����fromΪ������һ����
    }
    void dijkstra(int s)//��sΪ���
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
            int u = now.u;//��ǰ���
            if(v[u])continue;//����Ѿ����뼯�ϣ�continue
            v[u] = 1;
            for(int i = 0; i < G[u].size(); i++)
            {
                edge& e = edges[G[u][i]];//���ý�ʡ����
                if(d[e.to] > d[u] + e.dist)
                {
                    d[e.to] = d[u] + e.dist;
                    p[e.to] = G[u][i];//��¼e.toǰ�ıߵı�ţ�p����Ǳߵ��±�,��������ͨ�����ҳ�֮ǰ�ĵ��Լ�ÿ��·��·����������ڽӾ���洢�Ļ��������ֱ�Ӵ�ڵ�u
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
            printf("��%d��%d�����ǣ�%2d   ", u, i, d[i]);
            stack<int>q;//����Ǳߵı��
            int x = i;//x����·�������еĵ�
            while(p[x] != -1)
            {
                q.push(x);
                x = edges[p[x]].from;//x��������ߵ����
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