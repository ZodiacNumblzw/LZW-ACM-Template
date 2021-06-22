//拓扑排序
vector<int> g[maxn];
int du[maxn], n, m=0, l[maxn];
bool toposort()
{
    memset(du, 0, sizeof du);
    for(int i = 1; i <= n; ++i)
        for(int j = 0; j < g[i].size(); ++j)
            ++du[g[i][j]];
    int tot = 0;
    priority_queue<int, vector<int>, greater<int> > q;//按字典序最小的排序时
    //queue<int> q;
    for(int i = 1; i <= n; ++i)
        if(!du[i])
            q.push(i);
    while(!q.empty())
    {
        int x = q.top(); q.pop();
        l[tot++] = x;
        for(int j = 0; j < g[x].size(); ++j)
        {
            int t = g[x][j];
            --du[t];
            if(!du[t])q.push(t);
        }
    }
    if(tot == n)return 1;
    else        return 0;
}