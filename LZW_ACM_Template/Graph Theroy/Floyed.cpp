//Floyed O(n^3)           by:LZW
struct Floyed
{
#define type double
#define MAX 105
    type Graph[MAX][MAX];
    void initInt()
    {
        memset(Graph,0x3f3f3f3f,sizeof(Graph));
    }
    void initDouble()
    {
        memset(Graph,127,sizeof(Graph));
    }
    void addEdge(int u,int v,type w)
    {
        Graph[u][v]=min(Graph[u][v],w);
    }
    void floyd(int n)
    {
        for(int i=1;i<=n;i++)
            Graph[i][i]=0;
        for(int k=1; k<=n; k++)
        {
            for(int i=1; i<=n; i++)
            {
                for(int j=1; j<=n; j++)
                {
                    if(Graph[i][k] + Graph[k][j] < Graph[i][j])
                    {
                        Graph[i][j]=Graph[i][k] + Graph[k][j];
                    }
                }
            }
        }
    }
    type query(int u,int v)
    {
        return Graph[u][v];
    }
#undef type
#undef MAX
} F;