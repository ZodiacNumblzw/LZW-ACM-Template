//扩展BSGS (yyb巨佬)
namespace exBSGS
{
const int HashMod=100007;
struct HashTable
{
    struct Line
    {
        int u,v,next;
    } e[1000000];
    int h[HashMod],cnt;
    void Add(int u,int v,int w)
    {
        e[++cnt]=(Line)
        {
            w,v,h[u]
        };
        h[u]=cnt;
    }
    void Clear()
    {
        memset(h,0,sizeof(h));
        cnt=0;
    }
    void Hash(int x,int k)
    {
        int s=x%HashMod;
        Add(s,k,x);
    }
    int Query(int x)
    {
        int s=x%HashMod;
        for(int i=h[s]; i; i=e[i].next)
            if(e[i].u==x)return e[i].v;
        return -1;
    }
} Hash;
int fpow(int a,int b,int MOD)
{
    int s=1;
    while(b){if(b&1)s=1ll*s*a%MOD;a=1ll*a*a%MOD;b>>=1;}
    return s;
}
void NoAnswer(){puts("No Solution");}
//exBSGS(yyb巨佬)
void solve(int y,int z,int p)         //y^x=z(modp) p不一定为素数
{
    if(z==1){puts("0");return;}
    int k=0,a=1;
    while(233)
    {
        int d=__gcd(y,p);if(d==1)break;
        if(z%d){NoAnswer();return;}
        z/=d;p/=d;++k;a=1ll*a*y/d%p;
        if(z==a){printf("%d\n",k);return;}
    }
    Hash.Clear();
    int m=sqrt(p)+1;
    for(int i=0,t=z;i<m;++i,t=1ll*t*y%p)Hash.Hash(t,i);
    for(int i=1,tt=fpow(y,m,p),t=1ll*a*tt%p;i<=m;++i,t=1ll*t*tt%p)
    {
        int B=Hash.Query(t);if(B==-1)continue;
        printf("%d\n",i*m-B+k);return;
    }
    NoAnswer();
}
}