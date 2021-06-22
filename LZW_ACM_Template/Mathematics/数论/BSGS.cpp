//BSGS         求y^x=z(mod p)的最小非负 x    其中p为素数
void BSGS(int y,int z,int p)     //令x=am-b   其中m=sqrt(p)时复杂度最优
{
    //y^(am)=z*y^b(modp)
    if(y%p==0)                 //特判gcd(y,p)!=1时的情况
    {
        cout<<"Orz, I cannot find x!"<<endl;
        return ;
    }
    y%=p,z%=p;
    if(z==1)
    {
        cout<<0<<endl;
        return;
    }
    map<ll,ll> mp;                 //用来存z*y^b(modp)的值 也可以自己写hash
    mp.clear();
    int m=sqrt(p)+1;
    ll temp=quick(y,m,p),b=z;
    for(int i=0; i<m; i++,b=b*y%p)           //存z*y^b(modp)
    {
        mp[b]=i;
    }
    ll sum=temp;
    for(int a=1; (a-1)*(m)<=p; a++)   //用来验证左式是否存在和右式相等的值
    {
        if(mp.count(sum))
        {
            ll ans=a*m-mp[sum];
            cout<<ans<<endl;
            return;
        }
        sum=(sum*temp)%p;
    }
    cout<<"Orz, I cannot find x!"<<endl;
}
//PS:洛谷上可以过 poj上T飞了





//手写Hash的方法
//BSGS  yyb巨佬 poj2417
namespace BSGS
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
//BSGS(yyb巨佬)
void solve(int y,int z,int p)
{
    if(y%p==0)
    {
        printf("no solution\n");
        return;
    }
    y%=p;
    z%=p;
    if(z==1)
    {
        puts("0");
        return;
    }
    int m=sqrt(p)+1;
    Hash.Clear();
    for(register int i=0,t=z; i<m; ++i,t=1ll*t*y%p)Hash.Hash(t,i);
    for(register int i=1,tt=fpow(y,m,p),t=tt; i<=m+1; ++i,t=1ll*t*tt%p)
    {
        int k=Hash.Query(t);
        if(k==-1)continue;
        printf("%d\n",i*m-k);
        return;
    }
    printf("no solution\n");
}
}