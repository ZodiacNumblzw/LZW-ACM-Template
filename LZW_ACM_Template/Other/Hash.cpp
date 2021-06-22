const int HashMod=100007;
struct HashTable
{
    struct Line{int u,v,next;}e[1000000];
    int h[HashMod],cnt;
    void Hash(int u,int v,int w){e[++cnt]=(Line){w,v,h[u]};h[u]=cnt;}
    void Clear(){memset(h,0,sizeof(h));cnt=0;}
    void Add(int x,int k)
    {
        int s=x%HashMod;
        Hash(s,k,x);
    }
    int Query(int x)
    {
        int s=x%HashMod;
        for(int i=h[s];i;i=e[i].next)
            if(e[i].u==x)return e[i].v;
        return -1;
    }
}Hash;